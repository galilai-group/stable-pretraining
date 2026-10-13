"""Reusable point-cloud tokenization, ordering, and JEPA masking."""

from dataclasses import dataclass

import torch
from torch import nn


@torch.no_grad()
def farthest_point_indices(points: torch.Tensor, num_points: int) -> torch.Tensor:
    """Sample point indices using random-start batched FPS.

    Args:
        points: XYZ clouds [B,N,3].
        num_points: Number of points to select.

    Returns:
        Selected indices [B,num_points].
    """
    batch, count, _ = points.shape
    distances = torch.full((batch, count), float("inf"), device=points.device)
    farthest = torch.randint(count, (batch,), device=points.device)
    indices = torch.empty(batch, num_points, dtype=torch.long, device=points.device)
    rows = torch.arange(batch, device=points.device)
    for index in range(num_points):
        indices[:, index] = farthest
        center = points[rows, farthest]
        distances = torch.minimum(
            distances, (points - center[:, None]).square().sum(-1)
        )
        farthest = distances.argmax(-1)
    return indices


class PointTokenizer(nn.Module):
    """Group XYZ points with FPS/KNN and embed each local patch with PointNet.

    The mini PointNet is the point encoder in Point-JEPA Section 3.1, Figure 2:
    a shared MLP (3 -> 128 -> 256), patch max pooling, concatenation of
    pooled and per-point features, a shared MLP (512 -> 512 -> embed_dim),
    and final patch max pooling. Kernel-size-one convolutions apply the same
    MLP to every point; pooling makes each patch embedding invariant to its
    point order. Unlike the full PointNet classifier, this local encoder has
    no learned alignment transforms or classification head.

    Args:
        num_groups: Number of farthest-point centers.
        group_size: Number of nearest neighbors per center.
        embed_dim: Output token width.
    """

    def __init__(
        self, num_groups: int = 64, group_size: int = 32, embed_dim: int = 384
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.pointnet_local_mlp = nn.Sequential(
            nn.Conv1d(3, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
        )
        self.pointnet_fusion_mlp = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, embed_dim, 1),
        )

    @torch.no_grad()
    def group(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Group points around farthest-point centers.

        Args:
            points: XYZ point cloud [B,N,3].

        Returns:
            Relative groups [B,G,K,3] and centers [B,G,3].
        """
        batch = points.shape[0]
        centers_idx = farthest_point_indices(points.float(), self.num_groups)
        rows = torch.arange(batch, device=points.device)
        centers = points[rows[:, None], centers_idx]
        nearest = (
            torch.cdist(centers, points).topk(self.group_size, largest=False).indices
        )
        groups = points[rows[:, None, None], nearest] - centers[:, :, None]
        return groups, centers

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed local point groups.

        Args:
            points: XYZ point cloud [B,N,3].

        Returns:
            Patch tokens [B,G,D] and centers [B,G,3].
        """
        groups, centers = self.group(points)
        batch, count, neighbors, _ = groups.shape
        x = groups.reshape(batch * count, neighbors, 3).transpose(1, 2)
        local_features = self.pointnet_local_mlp(x)
        patch_features = local_features.max(-1, keepdim=True).values
        fused_features = torch.cat(
            [patch_features.expand_as(local_features), local_features], dim=1
        )
        point_features = self.pointnet_fusion_mlp(fused_features)
        patch_tokens = point_features.max(-1).values
        return patch_tokens.reshape(batch, count, -1), centers


class PointSequencer(nn.Module):
    """Order point-cloud patches along a greedy spatial nearest-neighbor walk.

    Starts at the center with minimum XYZ sum and reorders tokens and centers
    together, making contiguous token intervals spatially local for JEPA masking.
    """

    def forward(
        self, tokens: torch.Tensor, centers: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reorder tokens and centers along the same spatial walk.

        Args:
            tokens: Point-patch tokens [B,G,D].
            centers: Patch centers [B,G,3].

        Returns:
            Ordered tokens and ordered centers.
        """
        batch, count, _ = centers.shape
        with torch.no_grad():
            distances = torch.cdist(centers, centers)
            visited = torch.zeros(batch, count, device=centers.device, dtype=torch.bool)
            current = centers.sum(-1).argmin(-1)
            rows = torch.arange(batch, device=centers.device)
            order = torch.empty(batch, count, device=centers.device, dtype=torch.long)
            for index in range(count):
                order[:, index] = current
                visited[rows, current] = True
                current = (
                    distances[rows, current]
                    .masked_fill(visited, float("inf"))
                    .argmin(-1)
                )
        return tokens.gather(1, order[..., None].expand_as(tokens)), centers.gather(
            1, order[..., None].expand_as(centers)
        )


@dataclass
class PointJEPAMaskOutput:
    """Point-JEPA patch indices shared across a batch.

    ``context_idx`` has shape [C]; ``target_idx`` has shape [num_targets,T],
    preserving the boundaries of independently predicted target blocks.
    """

    context_idx: torch.Tensor
    target_idx: torch.Tensor


class PointJEPAMasking(nn.Module):
    """Sample overlapping contiguous targets and target-free filtered context.

    Operates on spatially sequenced point-cloud patches. Target blocks share a
    sampled length; context excludes every target patch before sampling an interval.

    Args:
        num_targets: Number of target intervals.
        target_scale: Interval length ratio relative to all tokens.
        context_scale: Context length ratio relative to remaining tokens.
    """

    def __init__(
        self,
        num_targets: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.20),
        context_scale: tuple[float, float] = (0.40, 0.75),
    ) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.context_scale = context_scale

    def forward(self, num_tokens: int, device: torch.device) -> PointJEPAMaskOutput:
        """Sample context and target token indices shared by a batch.

        Args:
            num_tokens: Number of ordered patches.
            device: Device for index tensors.

        Returns:
            Context indices [C] and target indices [num_targets,T].
        """
        ratio = torch.empty(()).uniform_(*self.target_scale).item()
        length = max(1, int(num_tokens * ratio))
        starts = torch.randint(
            num_tokens - length + 1, (self.num_targets,), device=device
        )
        targets = starts[:, None] + torch.arange(length, device=device)
        available = torch.ones(num_tokens, dtype=torch.bool, device=device)
        available[targets.flatten()] = False
        valid = torch.arange(num_tokens, device=device)[available]
        context_ratio = torch.empty(()).uniform_(*self.context_scale).item()
        context_length = max(1, int(len(valid) * context_ratio))
        start = torch.randint(len(valid) - context_length + 1, (), device=device)
        return PointJEPAMaskOutput(valid[start : start + context_length], targets)


class PointTokenTransformer(nn.Module):
    """Point-cloud patch transformer with XYZ position embeddings at every block.

    Accepts patch tokens and externally encoded XYZ positions of matching width.
    Uses the same transformer blocks as ViT without assuming a regular patch grid.

    Args:
        embed_dim: Token width.
        depth: Number of blocks.
        num_heads: Attention heads.
        drop_path_rate: Maximum stochastic depth probability.
        dropout: Projection dropout probability.
        attn_dropout: Attention dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        drop_path_rate: float = 0.25,
        dropout: float = 0.1,
        attn_dropout: float = 0.05,
    ) -> None:
        # Preprocessing imports FPS here; vit imports data through patch_masking.
        from .vit import TransformerBlock

        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4,
                drop_path=drop_path_rate * index / max(depth - 1, 1),
                proj_drop=dropout,
                attn_drop=attn_dropout,
            )
            for index in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, tokens: torch.Tensor, positions: torch.Tensor, *, normalize: bool = True
    ) -> torch.Tensor:
        """Encode point-patch tokens with XYZ positional embeddings.

        Args:
            tokens: Point-patch tokens [B,G,D].
            positions: Shared XYZ position embeddings [B,G,D].
            normalize: Apply learned output LayerNorm; disable for teacher targets.

        Returns:
            Encoded patch tokens [B,G,D].
        """
        for block in self.blocks:
            tokens = block(tokens + positions)
        return self.norm(tokens) if normalize else tokens
