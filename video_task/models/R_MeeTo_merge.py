import torch
import math
from typing import Callable, Tuple
from torch import nn


class RMeeTo_Merge(nn.Module):
    # merge function for Mamba
    def __init__(self, class_token: bool = False, num_prune: int = 5,
                 if_prune: bool = False, if_order: bool = True, distance: str = 'cosine', metric='X'):
        """
        Initializes the RMeeTo_Merge class with the given parameters.

        Args:
            class_token (bool): Whether or not there's a class token.
            num_prune (int): The number of tokens to prune.
            if_prune (bool): Whether to perform pruning.
            if_order (bool): Whether to apply ordered merge.
            distance (str): Distance metric to use ('cosine', 'l1', or 'l2').
        """
        super().__init__()
        self.class_token = class_token
        self.num_prune = num_prune
        self.if_prune = if_prune
        self.if_order = if_order
        self.distance = distance
        self.metric = metric

    def forward(self, metric: torch.Tensor, token_position: int) -> Callable:
        """w
        Applies the merge logic to the given metric tensor and token position.

        Args:
            metric (torch.Tensor): The input metric tensor.
            token_position (int): The position of the token.

        Returns:
            Callable: A merge function based on the initialized parameters.
        """
        protected = 1 if self.class_token else 0
        # Calculate the number of tokens to prune
        r = min(self.num_prune, (metric.shape[1] - protected) // 2)

        def do_noting(x: torch.Tensor, mode=None) -> Tuple[torch.Tensor, int]:
            return x, token_position

        if r <= 0:
            return do_noting

        metric = torch.cat([metric[..., :token_position, :], metric[..., token_position + 1:, :]], dim=1)
        metric = metric / metric.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            if self.distance == 'cosine':
                scores = a @ b.transpose(-1, -2)
            elif self.distance == 'l1':
                scores = -torch.cdist(a, b, p=1)
            elif self.distance == 'l2':
                scores = -torch.cdist(a, b, p=2)

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        def merge(x: torch.Tensor, mode="mean") -> Tuple[torch.Tensor, int]:
            cls_token = x[:, token_position, :]
            x = torch.cat([x[..., :token_position, :], x[..., token_position + 1:, :]], dim=1)
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape

            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            if not self.if_prune:
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            new_token_position = int(token_position * (t1 - r) // t1)
            unm = torch.cat([unm[:, :new_token_position], cls_token.unsqueeze(1), unm[:, new_token_position:]], dim=1)
            return torch.cat([unm, dst], dim=1), new_token_position

        def ordered_merge(x: torch.Tensor, mode="mean") -> Tuple[torch.Tensor, int]:
            n, t, c = x.shape
            cls_token = x[:, token_position, :]
            x = torch.cat([x[..., :token_position, :], x[..., token_position + 1:, :]], dim=1)
            src, dst = x[..., ::2, :], x[..., 1::2, :]

            idx_origin = torch.arange(t - 1, device=x.device).unsqueeze(0).expand(n, t - 1).unsqueeze(-1)
            src_idx_original, dst_idx_original = idx_origin[..., ::2, :], idx_origin[..., 1::2, :]

            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))

            src_idx_original = src_idx_original.gather(dim=-2, index=unm_idx)

            if not self.if_prune:
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            original_idx = torch.cat([src_idx_original, dst_idx_original], dim=1)
            sorted_idx, idx = original_idx.sort(dim=1)

            seq = torch.cat([unm, dst], dim=1)
            seq = seq.gather(dim=-2, index=idx.expand(n, seq.shape[1], c))

            new_token_position = int(token_position * (t - r) // t)
            seq = torch.cat([seq[:, :new_token_position], cls_token.unsqueeze(1), seq[:, new_token_position:]], dim=1)

            return seq, new_token_position

        return ordered_merge if self.if_order else merge

    def merge_wavg(self, merge: Callable, x: torch.Tensor, size: torch.Tensor = None):
        """
        Applies the merge function by taking a weighted average based on token size.
        Returns the merged tensor and the new token sizes.
        """
        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x, token_position = merge(x * size, mode="sum")
        size, _ = merge(size, mode="sum")

        x = x / size
        return x, size, token_position
