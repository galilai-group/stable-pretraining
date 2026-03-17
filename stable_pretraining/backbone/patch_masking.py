"""Patch masking strategies for masked image modeling."""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from stable_pretraining.data.masking import multi_block_mask

__all__ = [
    "PatchMasking",
    "MaskingOutput",
    "IJEPAMasking",
    "IJEPAMaskOutput",
    "MultiBlockMasking",
]


@dataclass
class MaskingOutput:
    """Output from patch masking operation.

    :ivar visible: Visible patch embeddings (B, N_keep, D)
    :ivar mask: Binary mask where 1 = masked, 0 = visible (B, N)
    :ivar ids_restore: Indices to restore original order (B, N)
    :ivar ids_keep: Indices of kept (visible) patches (B, N_keep)
    """

    visible: torch.Tensor
    mask: torch.Tensor
    ids_restore: torch.Tensor
    ids_keep: torch.Tensor


class PatchMasking(nn.Module):
    """Flexible patch masking module for masked image modeling.

    Supports three masking strategies that are selected stochastically:

    - **Random**: Uniformly random patch selection (when block_size=1)
    - **Block**: Square blocks of adjacent patches (when block_size > 1)
    - **Crop**: Rectangular crop region, remaining patches masked (when crop_ratio > 0)

    Strategy selection per sample:

    1. With probability ``crop_ratio``, use crop masking
    2. Otherwise, if ``block_size > 1``, use block masking
    3. Otherwise, use random masking

    :param mask_ratio: Fraction of patches to mask, in [0, 1)
    :param block_size: Size of square blocks for block masking (1 = random masking)
    :param crop_ratio: Probability of using crop masking vs block/random
    :param crop_aspect_ratio: (min, max) aspect ratio range for crop regions

    Example::

        masking = PatchMasking(mask_ratio=0.75, block_size=4)
        output = masking(patch_embeddings, grid_h=14, grid_w=14)

        visible_patches = output.visible  # (B, N_keep, D)
        mask = output.mask  # (B, N), 1=masked, 0=visible
        ids_keep = output.ids_keep  # (B, N_keep)
    """

    def __init__(
        self,
        mask_ratio: float = 0.75,
        block_size: int = 1,
        crop_ratio: float = 0.0,
        crop_aspect_ratio: tuple[float, float] = (0.75, 1.33),
    ):
        super().__init__()

        # Validation
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if not 0.0 <= crop_ratio <= 1.0:
            raise ValueError(f"crop_ratio must be in [0, 1], got {crop_ratio}")
        if len(crop_aspect_ratio) != 2:
            raise ValueError(
                f"crop_aspect_ratio must be a tuple of 2 floats, got {crop_aspect_ratio}"
            )
        if crop_aspect_ratio[0] <= 0 or crop_aspect_ratio[1] <= 0:
            raise ValueError(
                f"crop_aspect_ratio values must be positive, got {crop_aspect_ratio}"
            )
        if crop_aspect_ratio[0] > crop_aspect_ratio[1]:
            raise ValueError(
                f"crop_aspect_ratio[0] must be <= crop_aspect_ratio[1], "
                f"got {crop_aspect_ratio}"
            )

        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.crop_ratio = crop_ratio
        self.crop_aspect_ratio = crop_aspect_ratio

    def forward(
        self,
        x: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> MaskingOutput:
        """Apply masking to patch embeddings.

        :param x: Patch embeddings of shape (B, N, D) where N = grid_h * grid_w
        :param grid_h: Height of the patch grid
        :param grid_w: Width of the patch grid
        :return: MaskingOutput containing visible patches and mask information
        :raises ValueError: If x.shape[1] != grid_h * grid_w
        :raises ValueError: If input tensor has wrong number of dimensions
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (B, N, D), got {x.dim()}D tensor with shape {x.shape}"
            )

        B, N, D = x.shape

        if N != grid_h * grid_w:
            raise ValueError(
                f"Number of patches {N} doesn't match grid size "
                f"{grid_h} x {grid_w} = {grid_h * grid_w}"
            )

        if self.mask_ratio == 0 or not self.training:
            # No masking - return all patches as visible
            return MaskingOutput(
                visible=x,
                mask=torch.zeros(B, N, device=x.device),
                ids_restore=torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1),
                ids_keep=torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1),
            )

        num_mask = int(N * self.mask_ratio)
        num_keep = N - num_mask
        device = x.device

        # Determine which strategy to use per sample
        use_crop = torch.rand(B, device=device) < self.crop_ratio
        noise = torch.rand(B, N, device=device)

        # Apply crop masking where selected
        if use_crop.any():
            crop_noise = self._generate_crop_noise(B, grid_h, grid_w, num_keep, device)
            noise = torch.where(use_crop.view(B, 1), crop_noise, noise)

        # Apply block masking where selected (and not using crop)
        if self.block_size > 1 and (~use_crop).any():
            block_noise = self._generate_block_noise(
                B, grid_h, grid_w, num_mask, device
            )
            noise = torch.where((~use_crop).view(B, 1), block_noise, noise)

        # Convert noise to indices via sorting (lower noise = keep)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]

        # Gather visible patches
        visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Create binary mask (1 = masked, 0 = visible)
        mask = torch.ones(B, N, device=device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return MaskingOutput(
            visible=visible,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
        )

    def _generate_block_noise(
        self, B: int, grid_h: int, grid_w: int, num_mask: int, device: torch.device
    ) -> torch.Tensor:
        """Generate noise that induces block-structured masking."""
        N = grid_h * grid_w
        mask = torch.zeros(B, grid_h, grid_w, device=device)
        half = self.block_size // 2
        patches_per_block = self.block_size * self.block_size
        num_blocks_needed = (num_mask // patches_per_block) + 5

        centers_y = torch.randint(0, grid_h, (B, num_blocks_needed), device=device)
        centers_x = torch.randint(0, grid_w, (B, num_blocks_needed), device=device)

        rows = torch.arange(grid_h, device=device).view(1, 1, grid_h, 1)
        cols = torch.arange(grid_w, device=device).view(1, 1, 1, grid_w)

        for i in range(num_blocks_needed):
            cy = centers_y[:, i].view(B, 1, 1)
            cx = centers_x[:, i].view(B, 1, 1)

            y_start = (cy - half).clamp(min=0)
            y_end = (cy - half + self.block_size).clamp(max=grid_h)
            x_start = (cx - half).clamp(min=0)
            x_end = (cx - half + self.block_size).clamp(max=grid_w)

            in_block = (
                (rows >= y_start.unsqueeze(-1))
                & (rows < y_end.unsqueeze(-1))
                & (cols >= x_start.unsqueeze(-1))
                & (cols < x_end.unsqueeze(-1))
            ).squeeze(1)
            mask = torch.maximum(mask, in_block.float())

            if (mask.view(B, -1).sum(dim=1) >= num_mask).all():
                break

        mask_flat = self._adjust_mask_count(mask.view(B, N), num_mask, device)
        return torch.rand(B, N, device=device) * 0.5 + mask_flat * 0.5

    def _generate_crop_noise(
        self, B: int, grid_h: int, grid_w: int, num_keep: int, device: torch.device
    ) -> torch.Tensor:
        """Generate noise that induces crop-style masking."""
        N = grid_h * grid_w
        target_area = float(num_keep)

        log_ratio_min = math.log(self.crop_aspect_ratio[0])
        log_ratio_max = math.log(self.crop_aspect_ratio[1])
        log_ratios = torch.empty(B, device=device).uniform_(
            log_ratio_min, log_ratio_max
        )
        aspect_ratios = log_ratios.exp()

        crop_h = (target_area / aspect_ratios).sqrt().round().clamp(1, grid_h).long()
        crop_w = (target_area * aspect_ratios).sqrt().round().clamp(1, grid_w).long()

        max_y = (grid_h - crop_h).clamp(min=0)
        max_x = (grid_w - crop_w).clamp(min=0)
        top = (
            (torch.rand(B, device=device) * (max_y.float() + 1)).long().clamp(max=max_y)
        )
        left = (
            (torch.rand(B, device=device) * (max_x.float() + 1)).long().clamp(max=max_x)
        )

        rows = torch.arange(grid_h, device=device).view(1, grid_h, 1)
        cols = torch.arange(grid_w, device=device).view(1, 1, grid_w)

        in_crop = (
            (rows >= top.view(B, 1, 1))
            & (rows < (top + crop_h).view(B, 1, 1))
            & (cols >= left.view(B, 1, 1))
            & (cols < (left + crop_w).view(B, 1, 1))
        )
        crop_mask = (~in_crop).float().view(B, N)
        crop_mask = self._adjust_crop_to_target(
            crop_mask, num_keep, grid_h, grid_w, device
        )

        return torch.rand(B, N, device=device) * 0.5 + crop_mask * 0.5

    def _adjust_mask_count(
        self, mask_flat: torch.Tensor, target_masked: int, device: torch.device
    ) -> torch.Tensor:
        """Adjust mask to have exactly target_masked patches masked per sample."""
        B, N = mask_flat.shape
        mask_flat = mask_flat.clone()
        current_masked = mask_flat.sum(dim=1)

        excess = (current_masked - target_masked).clamp(min=0).long()
        if excess.any():
            noise = torch.rand(B, N, device=device) + (1 - mask_flat) * 2
            sorted_idx = noise.argsort(dim=1)
            position_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            unmask_positions = position_idx < excess.unsqueeze(1)
            unmask_idx = torch.gather(sorted_idx, 1, position_idx)
            mask_flat.scatter_(
                1,
                unmask_idx,
                mask_flat.gather(1, unmask_idx) * (~unmask_positions).float(),
            )

        current_masked = mask_flat.sum(dim=1)
        deficit = (target_masked - current_masked).clamp(min=0).long()
        if deficit.any():
            noise = torch.rand(B, N, device=device) + mask_flat * 2
            sorted_idx = noise.argsort(dim=1)
            position_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            mask_positions = position_idx < deficit.unsqueeze(1)
            mask_idx = torch.gather(sorted_idx, 1, position_idx)
            mask_flat.scatter_(
                1, mask_idx, mask_flat.gather(1, mask_idx) + mask_positions.float()
            )

        return mask_flat.clamp(0, 1)

    def _adjust_crop_to_target(
        self,
        crop_mask: torch.Tensor,
        num_keep: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Adjust crop mask using morphological operations to hit target visible count."""
        B, N = crop_mask.shape
        crop_mask = crop_mask.clone()

        max_iterations = 20
        for _ in range(max_iterations):
            num_visible = (crop_mask == 0).sum(dim=1)
            diff = num_visible - num_keep

            if (diff == 0).all():
                break

            mask_2d = crop_mask.view(B, 1, grid_h, grid_w)

            need_erode = diff > 0
            if need_erode.any():
                visible = (mask_2d == 0).float()
                padded = F.pad(1 - visible, (1, 1, 1, 1), value=1)
                neighbor_masked = F.max_pool2d(padded, 3, stride=1, padding=0)
                boundary = (visible.squeeze(1) == 1) & (neighbor_masked.squeeze(1) > 0)

                boundary_noise = (
                    torch.rand(B, grid_h, grid_w, device=device) * boundary.float()
                )
                boundary_noise[~need_erode] = -1

                flat_noise = boundary_noise.view(B, N)
                boundary_count = boundary.view(B, -1).sum(dim=1)
                to_remove = torch.minimum(diff.clamp(min=0), boundary_count)
                max_k = int(to_remove.max().item())

                if max_k > 0:
                    _, top_idx = flat_noise.topk(max_k, dim=1)
                    position_idx = torch.arange(max_k, device=device).unsqueeze(0)
                    valid = position_idx < to_remove.unsqueeze(1)
                    crop_mask.scatter_(
                        1, top_idx, crop_mask.gather(1, top_idx) + valid.float()
                    )

            need_dilate = diff < 0
            if need_dilate.any():
                mask_2d = crop_mask.view(B, 1, grid_h, grid_w)
                visible = (mask_2d == 0).float()
                padded = F.pad(visible, (1, 1, 1, 1), value=0)
                neighbor_visible = F.max_pool2d(padded, 3, stride=1, padding=0)
                boundary = (mask_2d.squeeze(1) == 1) & (neighbor_visible.squeeze(1) > 0)

                boundary_noise = (
                    torch.rand(B, grid_h, grid_w, device=device) * boundary.float()
                )
                boundary_noise[~need_dilate] = -1

                flat_noise = boundary_noise.view(B, N)
                boundary_count = boundary.view(B, -1).sum(dim=1)
                to_add = torch.minimum((-diff).clamp(min=0), boundary_count)
                max_k = int(to_add.max().item())

                if max_k > 0:
                    _, top_idx = flat_noise.topk(max_k, dim=1)
                    position_idx = torch.arange(max_k, device=device).unsqueeze(0)
                    valid = position_idx < to_add.unsqueeze(1)
                    crop_mask.scatter_(
                        1, top_idx, crop_mask.gather(1, top_idx) * (~valid).float()
                    )

        return crop_mask.clamp(0, 1)

    def extra_repr(self) -> str:
        return (
            f"mask_ratio={self.mask_ratio}, block_size={self.block_size}, "
            f"crop_ratio={self.crop_ratio}, crop_aspect_ratio={self.crop_aspect_ratio}"
        )


@dataclass
class IJEPAMaskOutput:
    """Output from I-JEPA masking operation.

    :ivar context_idx: Indices of context (visible) patches [B, N_ctx]
    :ivar target_idx: Combined indices of all target patches [B, N_tgt]
    :ivar target_block_masks: Per-block boolean masks [M x [B, N]], True = in this block
    :ivar mask: Full mask where 1 = target, 0 = context [B, N]
    """

    context_idx: torch.Tensor
    target_idx: torch.Tensor
    target_block_masks: List[torch.Tensor]
    mask: torch.Tensor


class IJEPAMasking(nn.Module):
    """I-JEPA multi-block masking for joint-embedding predictive architecture.

    Samples M non-overlapping target blocks and a context region that excludes
    all targets. This is the key masking strategy from I-JEPA [1]_.
    Strategy:
        1. Sample M target blocks with specified scale and aspect ratio
        2. Context = all patches NOT in any target block
        3. Optionally subsample context to specified ratio
    :param num_targets: Number of target blocks to sample (default: 4)
    :param target_scale: (min, max) fraction of patches per target block
    :param target_aspect_ratio: (min, max) aspect ratio of target blocks
    :param context_scale: (min, max) fraction of non-target patches to keep as context
    :param allow_target_overlap: Allow target blocks to overlap (default: False)
    Example::
        masking = IJEPAMasking(
            num_targets=4,
            target_scale=(0.15, 0.2),
            target_aspect_ratio=(0.75, 1.5),
            context_scale=(0.85, 1.0),
        )

        # x: patch embeddings [B, N, D]
        output = masking(x, grid_h=14, grid_w=14)

        context_patches = x.gather(
            1, output.context_idx.unsqueeze(-1).expand(-1, -1, D)
        )
        target_patches = x.gather(1, output.target_idx.unsqueeze(-1).expand(-1, -1, D))

    References:
        .. [1] Assran et al. "Self-Supervised Learning from Images with a
               Joint-Embedding Predictive Architecture." CVPR 2023.
    """

    def __init__(
        self,
        num_targets: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        context_scale: Tuple[float, float] = (0.85, 1.0),
        allow_target_overlap: bool = False,
    ):
        super().__init__()

        if num_targets < 1:
            raise ValueError(f"num_targets must be >= 1, got {num_targets}")
        if not (0 < target_scale[0] <= target_scale[1] < 1):
            raise ValueError(f"target_scale must be in (0, 1), got {target_scale}")
        if not (0 < target_aspect_ratio[0] <= target_aspect_ratio[1]):
            raise ValueError("target_aspect_ratio values must be positive")
        if not (0 < context_scale[0] <= context_scale[1] <= 1):
            raise ValueError(f"context_scale must be in (0, 1], got {context_scale}")
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.context_scale = context_scale
        self.allow_target_overlap = allow_target_overlap

    def _sample_block_params(
        self,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> Tuple[int, int, int, int]:
        """Sample parameters for a single target block.

        :return: (top, left, height, width) of the block
        """
        num_patches = grid_h * grid_w

        # Sample scale and aspect ratio
        scale = torch.empty(1, device=device).uniform_(*self.target_scale).item()
        log_ar = (
            torch.empty(1, device=device)
            .uniform_(
                torch.tensor(self.target_aspect_ratio[0]).log().item(),
                torch.tensor(self.target_aspect_ratio[1]).log().item(),
            )
            .item()
        )
        aspect_ratio = torch.tensor(log_ar).exp().item()

        # Compute block dimensions
        block_area = num_patches * scale
        block_h = int(round((block_area / aspect_ratio) ** 0.5))
        block_w = int(round((block_area * aspect_ratio) ** 0.5))

        # Clamp to grid bounds
        block_h = max(1, min(block_h, grid_h))
        block_w = max(1, min(block_w, grid_w))

        # Sample position
        top = torch.randint(0, max(1, grid_h - block_h + 1), (1,), device=device).item()
        left = torch.randint(
            0, max(1, grid_w - block_w + 1), (1,), device=device
        ).item()

        return top, left, block_h, block_w

    def _create_block_mask(
        self,
        top: int,
        left: int,
        block_h: int,
        block_w: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a 2D boolean mask for a block.

        :return: Boolean mask [grid_h, grid_w], True = in block
        """
        mask = torch.zeros(grid_h, grid_w, dtype=torch.bool, device=device)
        mask[top : top + block_h, left : left + block_w] = True
        return mask

    def forward(
        self,
        x: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> IJEPAMaskOutput:
        """Apply I-JEPA masking.

        :param x: Patch embeddings [B, N, D] where N = grid_h * grid_w
        :param grid_h: Height of patch grid
        :param grid_w: Width of patch grid
        :return: IJEPAMaskOutput with context/target information

        Note:
            Always returns exactly `num_targets` block masks. If overlap prevention
            makes it impossible to fit all blocks, some masks will be empty (all False).
            The combined `target_idx` only includes patches from non-empty blocks.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, N, D), got {x.dim()}D")

        B, N, D = x.shape
        device = x.device

        if N != grid_h * grid_w:
            raise ValueError(
                f"N={N} doesn't match grid {grid_h}x{grid_w}={grid_h * grid_w}"
            )

        # Eval mode: no masking, everything is context
        if not self.training:
            all_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            empty_block_masks = [
                torch.zeros(B, N, dtype=torch.bool, device=device)
                for _ in range(self.num_targets)
            ]
            return IJEPAMaskOutput(
                context_idx=all_idx,
                target_idx=torch.empty(B, 0, dtype=torch.long, device=device),
                target_block_masks=empty_block_masks,
                mask=torch.zeros(B, N, device=device),
            )

        # Sample target blocks (shared across batch for efficiency)
        target_masks_2d = []
        combined_target = torch.zeros(grid_h, grid_w, dtype=torch.bool, device=device)

        max_attempts_per_block = 100

        for _ in range(self.num_targets):
            block_mask = None

            # Try to find a valid (non-overlapping if required) block
            for _ in range(max_attempts_per_block):
                top, left, bh, bw = self._sample_block_params(grid_h, grid_w, device)
                candidate = self._create_block_mask(
                    top, left, bh, bw, grid_h, grid_w, device
                )

                # Accept if overlap allowed OR no overlap with existing targets
                if self.allow_target_overlap or not (candidate & combined_target).any():
                    block_mask = candidate
                    break

            if block_mask is not None:
                # Found a valid block
                target_masks_2d.append(block_mask)
                combined_target = combined_target | block_mask
            else:
                # Couldn't find non-overlapping block, append empty mask
                empty_mask = torch.zeros(
                    grid_h, grid_w, dtype=torch.bool, device=device
                )
                target_masks_2d.append(empty_mask)

        # Guarantee: len(target_masks_2d) == self.num_targets
        assert len(target_masks_2d) == self.num_targets

        # Flatten masks: List of [B, N] tensors
        target_block_masks_flat = [
            m.flatten().unsqueeze(0).expand(B, -1) for m in target_masks_2d
        ]

        # Combined target indices (only from non-empty blocks)
        combined_target_flat = combined_target.flatten()  # [N]
        target_idx = combined_target_flat.nonzero(as_tuple=True)[0]  # [N_tgt]
        target_idx = target_idx.unsqueeze(0).expand(B, -1)  # [B, N_tgt]

        # Context = non-target patches, subsampled according to context_scale
        context_available = ~combined_target_flat  # [N]
        available_idx = context_available.nonzero(as_tuple=True)[0]
        n_available = len(available_idx)

        # Handle edge case: all patches are targets
        if n_available == 0:
            # Fallback: use all patches as context (degenerate case)
            context_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        else:
            # Subsample context
            context_ratio = (
                torch.empty(1, device=device).uniform_(*self.context_scale).item()
            )
            n_context = max(1, int(n_available * context_ratio))

            # Per-sample random subsampling
            context_idx_list = []
            for _ in range(B):
                perm = torch.randperm(n_available, device=device)[:n_context]
                ctx_idx = available_idx[perm].sort().values
                context_idx_list.append(ctx_idx)

            context_idx = torch.stack(context_idx_list)  # [B, N_ctx]

        # Full mask: 1 = target, 0 = context/available
        mask = combined_target_flat.float().unsqueeze(0).expand(B, -1)  # [B, N]

        return IJEPAMaskOutput(
            context_idx=context_idx,
            target_idx=target_idx,
            target_block_masks=target_block_masks_flat,
            mask=mask,
        )

    def extra_repr(self) -> str:
        return (
            f"num_targets={self.num_targets}, "
            f"target_scale={self.target_scale}, "
            f"target_aspect_ratio={self.target_aspect_ratio}, "
            f"context_scale={self.context_scale}"
        )


class MultiBlockMasking(nn.Module):
    """Multi-block masking for SALT Stage 1 (VPixel).

    Generates one large context block and M target blocks using
    :func:`multi_block_mask`, then makes context disjoint from targets.
    Returns :class:`MaskingOutput` compatible with :class:`MaskedEncoder`.

    :param num_targets: Number of target blocks (default: 4)
    :param context_scale: (min, max) scale for context block (default: (0.85, 1.0))
    :param target_scale: (min, max) scale for each target block (default: (0.15, 0.2))
    :param context_aspect_ratio: (min, max) aspect ratio for context (default: (1.0, 1.0))
    :param target_aspect_ratio: (min, max) aspect ratio for targets (default: (0.75, 1.5))

    Example::

        masking = MultiBlockMasking(num_targets=4)
        output = masking(patch_embeddings, grid_h=14, grid_w=14)

        visible_patches = output.visible  # (B, N_keep, D)
        mask = output.mask  # (B, N), 1=masked, 0=visible
    """

    def __init__(
        self,
        num_targets: int = 4,
        context_scale: Tuple[float, float] = (0.85, 1.0),
        target_scale: Tuple[float, float] = (0.15, 0.2),
        context_aspect_ratio: Tuple[float, float] = (1.0, 1.0),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
    ):
        super().__init__()

        if num_targets < 1:
            raise ValueError(f"num_targets must be >= 1, got {num_targets}")

        self.num_targets = num_targets
        self.context_scale = context_scale
        self.target_scale = target_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.target_aspect_ratio = target_aspect_ratio

    def forward(
        self,
        x: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> MaskingOutput:
        """Apply multi-block masking to patch embeddings.

        :param x: Patch embeddings [B, N, D] where N = grid_h * grid_w
        :param grid_h: Height of patch grid
        :param grid_w: Width of patch grid
        :return: MaskingOutput with visible patches and mask info
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (B, N, D), got {x.dim()}D tensor with shape {x.shape}"
            )

        B, N, D = x.shape
        device = x.device

        if N != grid_h * grid_w:
            raise ValueError(
                f"Number of patches {N} doesn't match grid size "
                f"{grid_h} x {grid_w} = {grid_h * grid_w}"
            )

        if not self.training:
            return MaskingOutput(
                visible=x,
                mask=torch.zeros(B, N, device=device),
                ids_restore=torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
                ids_keep=torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
            )

        # Generate masks: 1 context block + M target blocks
        block_scales = [self.context_scale] + [self.target_scale] * self.num_targets
        aspect_ratios = [self.context_aspect_ratio] + [
            self.target_aspect_ratio
        ] * self.num_targets

        masks = multi_block_mask(
            grid_h,
            grid_w,
            block_scales=block_scales,
            aspect_ratios=aspect_ratios,
        )

        context_mask = masks[0]  # [H, W], 1=in block
        target_masks = masks[1:]

        # Make context disjoint from targets
        for t in target_masks:
            context_mask = context_mask * (1 - t)

        # Context = visible, everything else = masked
        context_flat = context_mask.flatten().bool().to(device)  # [N]

        # mask: 1 = masked (not in context), 0 = visible (in context)
        mask = (~context_flat).float()  # [N]

        # Get sorted context indices
        ids_keep = context_flat.nonzero(as_tuple=True)[0]  # [N_keep]
        num_keep = ids_keep.shape[0]

        # Build ids_restore: assign low noise to context, high to masked
        noise = torch.zeros(N, device=device)
        noise[ids_keep] = torch.arange(num_keep, device=device, dtype=torch.float)
        noise[~context_flat] = torch.arange(
            num_keep, N, device=device, dtype=torch.float
        )
        ids_shuffle = noise.long()
        ids_restore = torch.argsort(ids_shuffle)

        # Expand to batch
        ids_keep = ids_keep.unsqueeze(0).expand(B, -1)
        ids_restore = ids_restore.unsqueeze(0).expand(B, -1)
        mask = mask.unsqueeze(0).expand(B, -1)

        # Gather visible patches
        visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return MaskingOutput(
            visible=visible,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
        )

    def extra_repr(self) -> str:
        return (
            f"num_targets={self.num_targets}, "
            f"context_scale={self.context_scale}, "
            f"target_scale={self.target_scale}, "
            f"context_aspect_ratio={self.context_aspect_ratio}, "
            f"target_aspect_ratio={self.target_aspect_ratio}"
        )
