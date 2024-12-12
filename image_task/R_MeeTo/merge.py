import torch
import math
from typing import Callable, Tuple
from torch import nn


class RMeeTo_Merge(nn.Module):
    # merge function for Mamba
    def __init__(self, class_token: bool = False, num_prune: int = 5,
                 if_prune: bool = False, if_order: bool = True, distance: str = 'cosine', metric='X', merge_mode='sum',
                 choose='max'
                 ):
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
        self.merge_mode = merge_mode
        self.choose = choose

    def change_num_prune(self, num_prune: int):
        self.num_prune = num_prune

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

            if self.choose == 'max':
                node_max, node_idx = scores.max(dim=-1)
            elif self.choose == '3c':
                sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)  # descending=True 表示降序排列
                node_max = sorted_scores[..., 2]
                node_idx = sorted_indices[..., 2]
            elif self.choose == '5c':
                sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
                node_max = sorted_scores[..., 4]
                node_idx = sorted_indices[..., 4]
            elif self.choose == '7c':
                sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
                node_max = sorted_scores[..., 6]
                node_idx = sorted_indices[..., 6]
            elif self.choose == '14c':
                sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
                node_max = sorted_scores[..., 13]
                node_idx = sorted_indices[..., 13]
            elif self.choose == '21c':
                sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
                node_max = sorted_scores[..., 20]
                node_idx = sorted_indices[..., 20]

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

        x, token_position = merge(x * size, mode=self.merge_mode)
        size, _ = merge(size, mode=self.merge_mode)

        x = x / size
        return x, size, token_position

    def merge_source(
            self, merge: Callable, x: torch.Tensor, source: torch.Tensor = None
    ) -> torch.Tensor:
        """
        For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
        x is used to find out how many tokens there are in case the source is None.
        """
        if source is None:
            n, t, _ = x.shape
            source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

        source, _ = merge(source, mode="amax")
        return source


class RMeeTo_Merge_ViT:  # merge function for ViT
    def __init__(self, class_token: bool = False, num_prune: int = 5,
                 if_prune: bool = False, if_order: bool = True, distance: str = 'cosine'):
        """
        Initializes the RMeeTo_Merge class with the given parameters.

        Args:
            class_token (bool): Whether or not there's a class token.
            num_prune (int): The number of tokens to prune.
            if_prune (bool): Whether to perform pruning.
            if_order (bool): Whether to apply ordered merge.
            distance (str): Distance metric to use ('cosine', 'l1', or 'l2').
        """
        self.class_token = class_token
        self.num_prune = num_prune
        self.if_prune = if_prune
        self.if_order = if_order
        self.distance = distance

    def forward(self, metric: torch.Tensor) -> Callable:
        """
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
            return x

        if r <= 0:
            return do_noting

        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]

            if self.distance == 'cosine':
                scores = a @ b.transpose(-1, -2)
            elif self.distance == 'l1':
                scores = -torch.cdist(a, b, p=1)
            elif self.distance == 'l2':
                scores = -torch.cdist(a, b, p=2)

            if self.class_token:
                scores[..., 0, :] = -math.inf

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

            if self.class_token:
                unm_idx = unm_idx.sort(dim=1)[0]

        def merge(x: torch.Tensor, mode="mean") -> Tuple[torch.Tensor, int]:
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            if not self.if_prune:
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            return torch.cat([unm, dst], dim=1)

        def ordered_merge(x: torch.Tensor, mode="mean") -> Tuple[torch.Tensor, int]:
            n, t, c = x.shape
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            idx_origin = torch.arange(t, device=x.device).unsqueeze(0).expand(n, t).unsqueeze(-1)

            src_idx_original, dst_idx_original = idx_origin[..., ::2, :], idx_origin[..., 1::2, :]

            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))

            src_idx_original = src_idx_original.gather(dim=-2, index=unm_idx)

            if self.if_prune is False:
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            original_idx = torch.cat([src_idx_original, dst_idx_original], dim=1)

            sorted_idx, idx = original_idx.sort(dim=1)
            seq = torch.cat([unm, dst], dim=1)

            seq = seq.gather(dim=-2, index=idx.expand(n, seq.shape[1], c))
            return seq

        return ordered_merge if self.if_order else merge

    def merge_wavg_vit(
            self, merge: Callable, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the merge function by taking a weighted average based on token size.
        Returns the merged tensor and the new token sizes.
        """
        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x = merge(x * size, mode="sum")
        size = merge(size, mode="sum")

        x = x / size
        return x, size
