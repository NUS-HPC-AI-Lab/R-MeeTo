import torch
from torch import nn, Tensor
from models.deit import VisionTransformer, Block
from typing import Tuple, Optional

from .merge import RMeeTo_Merge_ViT
from functools import partial

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import random


class RMeeToBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_prune=5, if_prune=False, if_order=True, distance='cosine', metric=None, class_token=True,
                 if_merge=False
                 ):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer,
                         norm_layer)
        self.merge = RMeeTo_Merge_ViT(class_token=class_token, num_prune=num_prune, if_prune=if_prune,
                                      if_order=if_order, distance=distance)
        self.metric = metric
        self.if_merge = if_merge

    def forward(self, x: Tensor, size: Tensor = None):
        if size == None:
            size = torch.ones(x.shape[0], 1).to(x.device)

        attn, q, k, v = self.drop_path(self.attn(self.norm1(x), size=size))
        x = x + attn

        if self.if_merge:
            if self.metric == 'X':
                metric = x
            elif self.metric == 'Q':
                metric = q
            elif self.metric == 'K':
                metric = k
            elif self.metric == 'V':
                metric = v

        merge_fn = self.merge(metric)
        x, new_size = self.merge.merge_wavg_vit(merge_fn, x, size)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.if_merge:
            return x, new_size
        else:
            return x, size


class RMeeToVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 if_prune=True, distance='cosine', metric='X', class_token=True, if_merge=False, num_prune=5,
                 if_order=False):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth,
                         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         representation_size=representation_size,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                         hybrid_backbone=hybrid_backbone, norm_layer=norm_layer)

        self.blocks = nn.ModuleList([
            RMeeToBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer, num_prune=num_prune,
                if_prune=if_prune, if_order=if_order, distance=distance, metric=metric, class_token=class_token,
                if_merge=if_merge
            )
            for i in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        b, l, c = x.shape
        size = torch.ones(b, l, 1).to(x.device)

        for i, blk in enumerate(self.blocks):
            x, size = blk(x, size)

        feature = self.norm(x)
        cls = feature[:, 0]

        cls = self.pre_logits(cls)
        cls = self.head(cls)
        return cls


# tiny
@register_model
def RMeeTo_deit_tiny(pretrained=False, model_pth=None, merge_para={}, **kwargs):
    if merge_para == {}:
        merge_para = {'if_prune': False, 'distance': 'cosine', 'metric': 'X', 'class_token': True, 'if_merge': True,
                      'num_prune': 5, 'if_order': False}
    num_prune, if_prune, if_order, distance, metric, class_token, if_merge = merge_para['num_prune'], merge_para[
        'if_prune'], merge_para['if_order'], merge_para['distance'], merge_para['metric'], merge_para['class_token'], \
    merge_para['if_merge']

    model = RMeeToVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0,
        attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=torch.nn.LayerNorm,
        if_prune=if_prune, distance=distance, metric=metric, class_token=class_token, if_merge=if_merge,
        num_prune=num_prune, if_order=if_order
    )
    if pretrained:
        checkpoint = torch.load(f'{model_pth}/deit_tiny_patch16_224-a1311bcf.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    return model


@register_model
def RMeeTo_deit_small(pretrained=False, model_pth=None, merge_para={}, **kwargs):
    if merge_para == {}:
        merge_para = {'if_prune': False, 'distance': 'cosine', 'metric': 'X', 'class_token': True, 'if_merge': True,
                      'num_prune': 11, 'if_order': False}
    num_prune, if_prune, if_order, distance, metric, class_token, if_merge = merge_para['num_prune'], merge_para[
        'if_prune'], merge_para['if_order'], merge_para['distance'], merge_para['metric'], merge_para['class_token'], \
    merge_para['if_merge']

    model = RMeeToVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0,
        attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=torch.nn.LayerNorm,
        if_prune=if_prune, distance=distance, metric=metric, class_token=class_token, if_merge=if_merge,
        num_prune=num_prune, if_order=if_order
    )
    if pretrained:
        checkpoint = torch.load(f'{model_pth}/deit_small_patch16_224-cd65a155.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    return model


@register_model
def RMeeTo_deit_base(pretrained=False, model_pth=None, merge_para={}, **kwargs):
    if merge_para == {}:
        merge_para = {'if_prune': False, 'distance': 'cosine', 'metric': 'X', 'class_token': True, 'if_merge': True,
                      'num_prune': 17, 'if_order': False}
    num_prune, if_prune, if_order, distance, metric, class_token, if_merge = merge_para['num_prune'], merge_para[
        'if_prune'], merge_para['if_order'], merge_para['distance'], merge_para['metric'], merge_para['class_token'], \
    merge_para['if_merge']

    model = RMeeToVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0,
        attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=torch.nn.LayerNorm,
        if_prune=if_prune, distance=distance, metric=metric, class_token=class_token, if_merge=if_merge,
        num_prune=num_prune, if_order=if_order
    )
    if pretrained:
        checkpoint = torch.load(f'{model_pth}/deit_base_patch16_224-b5f2ef4d.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    return model
