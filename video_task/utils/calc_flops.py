import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from numpy import prod
import numpy as np
from fvcore.nn import FlopCountAnalysis
from torch.cuda.amp import autocast
from models.mamba_simple import Mamba
from models import RMeeTo_Merge


@torch.no_grad()
def throughput(images, model):
    model.eval()

    images = images.cuda(non_blocking=True)
    batch_size = images.shape[0]

    print(images.dtype)

    if images.dtype == torch.float32:
        images = images.half()
        print(images.dtype)

    model = model.half()

    for i in range(100):
        with autocast():
            model(images)
    torch.cuda.synchronize()
    print(f"throughput averaged with 30 times")
    tic1 = time.time()
    for i in range(30):
        with autocast():
            model(images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    MB = 1024.0 * 1024.0
    print('memory:', torch.cuda.max_memory_allocated() / MB)

    print(f"Single image inference time: {(tic2 - tic1) / (30 * batch_size) * 1000:.3f} ms")


def get_flops(model, img_size=224, show_detail=False):
    conv_flops = []

    def conv_hook(self, input, output, path=None):
        if show_detail:
            print(f'{path} is called')

        input_tensor = input[0]
        output_tensor = output[0]
        batch_size, input_channels, input_depth, input_height, input_width = input_tensor.size()
        output_channels, output_depth, output_height, output_width = output_tensor.size()

        assert self.in_channels % self.groups == 0

        # 修改kernel_ops来包括3D卷积核的深度维度
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * (self.in_channels // self.groups)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_depth * output_height * output_width

        conv_flops.append(flops)
        if show_detail:
            print(f'Conv3d flops: {flops}')

    linear_flops = []

    def linear_hook(self, input, output, path=None):
        if show_detail:
            print(f'{path} is called')

        input_tensor = input[0]
        in_feature, out_feature = self.weight.size()
        batch_size = input_tensor.numel() // input_tensor.size(-1)  # .numel:返回张量元素的个数
        flops = batch_size * in_feature * out_feature

        linear_flops.append(flops)
        if show_detail:
            print(f'Linear flops: {flops}')

    mamba_flops = []

    def mamba_hook(self, input, output, path=None):
        if show_detail:
            print(f'{path} is called')

        """
        Glossary:
        b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
        l: sequence length                  (`L` in [1] Algorithm 2)
        d or d_model: hidden dim
        n or d_state: latent state dim      (`N` in [1] Algorithm 2)
        expand: expansion factor            (`E` in [1] Section 3.4)
        d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
        A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
        Δ or delta: input-dependent step size
        dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")
        """

        input_tensor = input[0]
        b, l, _ = input_tensor.size()

        flops = 0

        # 1. in_proj
        flops += b * l * self.d_model * self.d_inner * 2
        # 2.1 causual conv1d
        flops += b * (l + self.d_conv - 1) * self.d_inner * self.d_conv
        # 2.2 x_proj
        flops += b * l * self.d_inner * (self.dt_rank + self.d_state * 2)
        # 2.3 dt_proj
        flops += b * l * self.dt_rank * self.d_inner
        # 3 selective scan

        flops += 9 * b * l * self.d_inner * self.d_state + 2 * b * l * self.d_inner
        # 4 out_proj
        flops += b * l * self.d_inner * self.d_model

        mamba_flops.append(flops)
        if show_detail:
            print(f'Mamba flops: {flops}')

    merge_flops = []

    def merge_hook(self, input, output, path=None):
        """
        Calculate the FLOPs for the bipartite_soft_matching function, ignoring the batch size effect.
        """
        _, tokens, channels = input[0].size()

        if self.metric == "X":
            flops_norm = 6 * tokens * channels  # mean:1*tokens*channels, std:3*tokens*channels, norm:2*tokens*channels
            flops_dot_product = (channels * 2) * (tokens / 2) * (tokens / 2)  # match between two group
        elif self.metric == "B" or self.metric == "C":
            flops_norm = 6 * tokens * self.dstate
            flops_dot_product = (self.dstate * 2) * (tokens / 2) * (tokens / 2)
        elif self.metric == "delta":
            flops_norm = 6 * tokens * (2 * channels)
            flops_dot_product = (2 * channels * 2) * (tokens / 2) * (tokens / 2)

        flops_merge = self.num_prune * channels  # merge the tokens

        total_flops = flops_norm + flops_dot_product + flops_merge

        merge_flops.append(total_flops)
        if show_detail:
            print(f'RMeeTo_Merge flops: {total_flops}')

    def register_module_hook(net, hook_handle, prefix='', path=''):
        for name, module in net.named_children():
            registerd = False
            if isinstance(module, torch.nn.Conv3d):
                hook_handle.append(module.register_forward_hook(
                    lambda *args, path=path + '/' + name + ':' + module.__class__.__name__: conv_hook(*args,
                                                                                                      path=path)))
                registerd = True
            if isinstance(module, torch.nn.Linear):
                hook_handle.append(module.register_forward_hook(
                    lambda *args, path=path + '/' + name + ':' + module.__class__.__name__: linear_hook(*args,
                                                                                                        path=path)))
                registerd = True
            if isinstance(module, Mamba):
                hook_handle.append(module.register_forward_hook(
                    lambda *args, path=path + '/' + name + ':' + module.__class__.__name__: mamba_hook(*args,
                                                                                                       path=path)))
                registerd = True
            if isinstance(module, RMeeTo_Merge):
                hook_handle.append(module.register_forward_hook(
                    lambda *args, path=path + '/' + name + ':' + module.__class__.__name__: merge_hook(*args,
                                                                                                       path=path)))
                registerd = True

            if registerd:
                if show_detail:
                    print(f"{prefix}{name}: {module.__class__.__name__} (registerd)")
            else:
                if show_detail:
                    print(f"{prefix}{name}: {module.__class__.__name__}")
                register_module_hook(module, hook_handle, prefix + '  ',
                                     path + '/' + name + ':' + module.__class__.__name__)

    hooks = []
    register_module_hook(model, hooks)

    # input_shape = (3, img_size, img_size)
    input_shape = (3, 8, 224, 224)
    # input = torch.rand(*input_shape).unsqueeze(0).to(next(model.parameters()).device)
    # 用全1的tensor代替随机tensor
    input = torch.ones(*input_shape).unsqueeze(0).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        out = model(input)
    for handle in hooks:
        handle.remove()

    # total_flops = sum(sum(i) for i in [conv_flops, linear_flops, mamba_flops])
    print('total flops (G):', sum(sum(i) for i in [conv_flops, linear_flops, mamba_flops, merge_flops]) / 1e9)
