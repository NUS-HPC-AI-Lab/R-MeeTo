import torch
from torch import nn, Tensor
from models.videomamba import Block, VisionMamba
from typing import Tuple, Optional

from .merge import RMeeTo_Merge
from functools import partial
from models.videomamba import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import random


class RMeeToBlock(Block):
    def __init__(
            self, dim, mixer_cls, num_prune, distance='cosine', metric='X',
            norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0., if_merge=False,
            class_token=False, if_pruning=False, if_order=True
    ):
        super().__init__(dim, mixer_cls, norm_cls, fused_add_norm, residual_in_fp32, drop_path)
        self.merge = RMeeTo_Merge(class_token, num_prune, if_pruning, if_order, distance)
        self.if_merge = if_merge
        self.metric = metric

    def forward(
            self, hidden_states: Tensor,
            residual: Optional[Tensor] = None,
            size: Optional[Tensor] = None,
            B_mamba: Optional[Tensor] = None,
            C_mamba: Optional[Tensor] = None,
            delta: Optional[Tensor] = None,
            inference_params=None, token_position: int = 0,
    ):
        # supper forward
        if size == None:
            B, L, C = hidden_states.shape
            size = torch.ones(B, L, 1, device=hidden_states.device)

        if self.if_merge:  # metric
            if self.metric == 'X':
                metric = hidden_states
            elif self.metric == 'B':
                metric = B_mamba
            elif self.metric == 'C':
                metric = C_mamba
            elif self.metric == 'delta':
                metric = delta

            merge_fn = self.merge(metric, token_position)
            hidden_states, new_size, new_token_position = self.merge.merge_wavg(merge_fn, hidden_states, size)
            residual, _, _ = self.merge.merge_wavg(merge_fn, residual, size)

        hidden_states, residual, B_mamba, C_mamba, delta = super().forward(hidden_states, residual,
                                                                           inference_params)  # metric

        if self.if_merge:
            return hidden_states, residual, new_size, new_token_position, B_mamba, C_mamba, delta
        else:
            return hidden_states, residual, size, token_position, B_mamba, C_mamba, delta


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        layer_idx=None,
        bimamba=True,
        device=None,
        dtype=None,
        # merge
        num_prune=5,
        distance='cosine',
        metric='X',
        merge_interval: int = 2,
        class_token=False,
        if_pruning=False,
        if_order=True,
        if_merge_odd=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, if_teacher=False,
                        **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    if not if_merge_odd:
        if layer_idx > 0 and layer_idx % merge_interval == 0:
            if_merge = True
        else:
            if_merge = False
    else:
        if layer_idx > 0 and (layer_idx + 1) % merge_interval == 0:
            if_merge = True
        else:
            if_merge = False

    block = RMeeToBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        num_prune=num_prune,
        distance=distance,
        metric=metric,
        if_merge=if_merge,
        class_token=class_token,
        if_pruning=if_pruning,
        if_order=if_order
    )

    return block


class RMeeTo_Mamba_Video(VisionMamba):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            stride=16,
            depth=24,
            embed_dim=192,
            channels=3,
            num_classes=1000,
            d_state=16,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True,
            residual_in_fp32=True,
            bimamba=True,
            device=None,
            dtype=None,
            if_cls_token=True,
            # merge
            merge_method="ToMe",
            if_prune=False,
            num_prune: int = 5,
            if_order=True,
            metric="X",
            if_random=False,
            distance="cosine",
            merge_interval: int = 2,
            if_merge=True,
            if_merge_odd=False,
            class_token=True,
            **kwargs):

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            depth=depth,
            embed_dim=embed_dim,
            channels=channels,
            num_classes=num_classes,
            ssm_cfg=ssm_cfg,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            device=device,
            bimamba=bimamba,
            dtype=dtype,
            **kwargs
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.dstate = d_state
        self.num_prune = num_prune
        self.merge_method = merge_method
        self.merge_interval = merge_interval
        self.metric = metric

        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    num_prune=num_prune,
                    distance=distance,
                    metric=metric,
                    merge_interval=merge_interval,
                    class_token=if_cls_token,
                    if_pruning=if_prune,
                    if_order=if_order,
                    if_merge_odd=if_merge_odd
                )
                for i in range(depth)
            ]
        )

    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # mamba impl
        residual = None
        B_mamba = None
        C_mamba = None
        delta = None

        hidden_states = x
        B, L, C = hidden_states.shape
        size = torch.ones(B, L, 1, device=x.device)
        token_position = 0
        for layer in self.layers:
            hidden_states, residual, size, token_position, B_mamba, C_mamba, delta = layer(
                hidden_states, residual, size,
                inference_params=inference_params, token_position=token_position,
                B_mamba=B_mamba, C_mamba=C_mamba, delta=delta
            )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token
        return hidden_states[:, 0, :]

    def forward(self, x, inference_params=None, return_flop=True):
        x = self.forward_features(x, inference_params)
        x = self.head(x)
        if return_flop:
            if self.training:
                return x
            else:
                flops, layer_flops = self.calculate_flop_inference()
                return x, flops, layer_flops

    # FLOPs
    def flops_selective_scan_fn(self, B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True,
                                with_complex=False):
        """
        u: r(B D L)
        delta: r(B D L)
        A: r(D N)
        B: r(B N L)
        C: r(B N L)
        D: r(D)
        z: r(B D L)
        delta_bias: r(D), fp32

        ignores:
            [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
        """
        d_rank = 24
        assert not with_complex
        # https://github.com/state-spaces/mamba/issues/110

        flops = 9 * B * L * D * N

        if with_D:
            flops += B * D * L
        if with_Z:
            flops += B * D * L
        return flops

    def selective_scan_flop_jit(self, B, L, D, dstate):
        N = dstate
        flops = self.flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=True, with_Group=True,
                                             with_complex=False)
        return flops

    def MambaInnerFnNoOutProj_flop_jit(self, b, l, layer):
        flops = 0
        # 2.1 causual conv1d
        flops += b * (l + layer.mixer.d_conv - 1) * layer.mixer.d_inner * layer.mixer.d_conv
        # 2.2 x_proj
        flops += b * l * layer.mixer.d_inner * (layer.mixer.dt_rank + layer.mixer.d_state * 2)
        # 2.3 dt_proj
        flops += b * l * layer.mixer.dt_rank * layer.mixer.d_inner

        return flops

    def calculate_flops_designer_ToMe(self, tokens, channels, num_prune):
        """
        Calculate the FLOPs for the bipartite_soft_matching function, ignoring the batch size effect.
        """
        if self.metric == "X":
            flops_norm = 6 * tokens * channels
            flops_dot_product = (channels * 2) * (tokens / 2) * (tokens / 2)
        elif self.metric == "B" or self.metric == "C":
            flops_norm = 6 * tokens * self.dstate
            flops_dot_product = (self.dstate * 2) * (tokens / 2) * (tokens / 2)
        elif self.metric == "delta":
            flops_norm = 6 * tokens * (2 * channels)
            flops_dot_product = (2 * channels * 2) * (tokens / 2) * (tokens / 2)

        flops_merge = num_prune * channels

        total_flops = flops_norm + flops_dot_product + flops_merge

        return total_flops

    def calculate_flop_inference(self):
        C = self.embed_dim
        patch_number = float(self.patch_embed.num_patches)
        N = patch_number + 1
        last_kept_number = N
        flops = 0
        layer_flops = 0
        patch_embedding_flops = N * C * (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1] * 3)
        classifier_flops = C * self.num_classes
        num_pruned = self.num_prune
        with torch.cuda.amp.autocast(enabled=False):
            for i, block in enumerate(self.layers):
                if i == 0:
                    mamba_flops = N * C * (4 * C) + N * C * (2 * C)
                    mamba_flops += self.MambaInnerFnNoOutProj_flop_jit(b=1, l=N, layer=block)
                    mamba_flops += self.selective_scan_flop_jit(B=1, L=N, D=2 * C, dstate=self.dstate)
                    flops += mamba_flops
                    layer_flops += mamba_flops
                else:
                    if i % self.merge_interval == 0:
                        num_pruned = self.num_prune
                        last_kept_number = last_kept_number - num_pruned
                        if self.merge_method == "ToMe":
                            flops += self.calculate_flops_designer_ToMe(tokens=N, channels=C, num_prune=num_pruned)
                        elif self.merge_method == "No_merge":
                            flops += 0
                        else:
                            flops += self.calculate_flops_designer(tokens=N, channels=C, num_prune=num_pruned)

                    N = last_kept_number
                    mamba_flops = N * C * (4 * C) + N * C * (2 * C)
                    mamba_flops += self.MambaInnerFnNoOutProj_flop_jit(b=1, l=N, layer=block)
                    mamba_flops += self.selective_scan_flop_jit(B=1, L=N, D=2 * C, dstate=self.dstate)
                    layer_flops += mamba_flops
                    flops += mamba_flops
        flops += patch_embedding_flops
        flops += classifier_flops
        layer_flops += patch_embedding_flops
        layer_flops += classifier_flops
        return flops, layer_flops


@register_model
def RMeeTo_video_tiny(pretrained=False, model_pth=None, merge_para={}, **kwargs):
    if merge_para == {}:
        merge_para = {"num_prune": 11, "distance": "cosine", "metric": "X", "merge_interval": 2, "if_prune": False,
                      "if_order": True, "cls_token": True, "if_merge": True, "if_merge_odd": False}
    num_prune, distance, metric, merge_interval, if_prune, if_order, if_cls_token, if_merge, if_merge_odd = merge_para[
        'num_prune'], merge_para['distance'], merge_para['metric'], merge_para['merge_interval'], merge_para[
        'if_prune'], merge_para['if_order'], merge_para['class_token'], merge_para['if_merge'], merge_para[
        'if_merge_odd']
    model = RMeeTo_Mamba_Video(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        num_prune=num_prune, distance=distance, metric=metric, merge_interval=merge_interval, if_prune=if_prune,
        if_order=if_order, if_merge=if_merge, if_merge_odd=if_merge_odd, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(f'{model_pth}/videomamba_t16_in1k_res224.pth', map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def RMeeTo_video_small(pretrained=False, model_pth=None, merge_para={}, **kwargs):
    if merge_para == {}:
        merge_para = {"num_prune": 11, "distance": "cosine", "metric": "X", "merge_interval": 2, "if_prune": False,
                      "if_order": True, "cls_token": True, "if_merge": True, "if_merge_odd": False}
    num_prune, distance, metric, merge_interval, if_prune, if_order, if_cls_token, if_merge, if_merge_odd = merge_para[
        'num_prune'], merge_para['distance'], merge_para['metric'], merge_para['merge_interval'], merge_para[
        'if_prune'], merge_para['if_order'], merge_para['class_token'], merge_para['if_merge'], merge_para[
        'if_merge_odd']
    model = RMeeTo_Mamba_Video(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        num_prune=num_prune, distance=distance, metric=metric, merge_interval=merge_interval, if_prune=if_prune,
        if_order=if_order, if_merge=if_merge, if_merge_odd=if_merge_odd, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(f'{model_pth}/videomamba_s16_in1k_res224.pth', map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def RMeeTo_video_middle(pretrained=False, model_pth=None, merge_para={}, **kwargs):
    if merge_para == {}:
        merge_para = {"num_prune": 11, "distance": "cosine", "metric": "X", "merge_interval": 2, "if_prune": False,
                      "if_order": True, "cls_token": True, "if_merge": True, "if_merge_odd": False}
    num_prune, distance, metric, merge_interval, if_prune, if_order, if_cls_token, if_merge, if_merge_odd = merge_para[
        'num_prune'], merge_para['distance'], merge_para['metric'], merge_para['merge_interval'], merge_para[
        'if_prune'], merge_para['if_order'], merge_para['class_token'], merge_para['if_merge'], merge_para[
        'if_merge_odd']
    model = RMeeTo_Mamba_Video(
        patch_size=16, embed_dim=576, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        num_prune=num_prune, distance=distance, metric=metric, merge_interval=merge_interval, if_prune=if_prune,
        if_order=if_order, if_merge=if_merge, if_merge_odd=if_merge_odd, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(f'{model_pth}/videomamba_s16_in1k_res224.pth', map_location='cpu')
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def RMeeTo_video_base(pretrained=False, model_pth=None, merge_para={}, **kwargs):
    if merge_para == {}:
        merge_para = {"num_prune": 11, "distance": "cosine", "metric": "X", "merge_interval": 2, "if_prune": False,
                      "if_order": True, "cls_token": True, "if_merge": True, "if_merge_odd": False}
    num_prune, distance, metric, merge_interval, if_prune, if_order, if_cls_token, if_merge, if_merge_odd = merge_para[
        'num_prune'], merge_para['distance'], merge_para['metric'], merge_para['merge_interval'], merge_para[
        'if_prune'], merge_para['if_order'], merge_para['class_token'], merge_para['if_merge'], merge_para[
        'if_merge_odd']
    model = RMeeTo_Mamba_Video(
        patch_size=16, embed_dim=768, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        num_prune=num_prune, distance=distance, metric=metric, merge_interval=merge_interval, if_prune=if_prune,
        if_order=if_order, if_merge=if_merge, if_merge_odd=if_merge_odd, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(f'{model_pth}/videomamba_b16_in1k_res224.pth', map_location='cpu')
        model.load_state_dict(checkpoint["model"], strict=False)
    return model
