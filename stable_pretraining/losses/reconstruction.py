"""Reconstruction-based SSL losses.

This module contains reconstruction-based self-supervised learning losses
such as Masked Autoencoder (MAE).
"""

from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.patch_masking import patchify


def mae(target, pred, mask, norm_pix_loss=False):
    """Compute masked autoencoder loss.

    Args:
        target: [N, L, p*p*3] target images
        pred: [N, L, p*p*3] predicted images
        mask: [N, L], 0 is keep, 1 is remove
        norm_pix_loss: whether to normalize pixels

    Returns:
        loss: mean loss value
    """
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


class MAELoss(nn.Module):
    """Modular MAE reconstruction loss with configurable loss functions.

    Supports MSE, cosine similarity, and custom loss functions with optional
    per-patch normalization.

    :param patch_size: Size of each square patch (default: 16)
    :param loss_type: Loss function type - 'mse', 'cosine', or 'smooth_l1' (default: 'mse')
    :param mask_only: If True, compute loss only on masked patches (default: True)
    :param patch_normalize: If True, normalize each target patch to zero mean/unit var (default: True)
    :param reduction: How to reduce patch losses - 'mean' or 'sum' (default: 'mean')

    Examples::

        >>> loss_fn = MAELoss(patch_size=16, loss_type='mse')
        >>> loss = loss_fn(pred, imgs, mask)

        >>> # Cosine similarity loss
        >>> loss_fn = MAELoss(patch_size=16, loss_type='cosine')
        >>> loss = loss_fn(pred, imgs, mask)

        >>> # Custom loss function
        >>> loss_fn = MAELoss(patch_size=16, loss_type='custom')
        >>> loss_fn.register_custom_loss(lambda p, t: (p - t).abs().mean(dim=-1))
        >>> loss = loss_fn(pred, imgs, mask)
    """

    LOSS_TYPES = Literal["mse", "cosine", "smooth_l1", "custom"]

    def __init__(
        self,
        patch_size: int = 16,
        loss_type: LOSS_TYPES = "mse",
        mask_only: bool = True,
        patch_normalize: bool = True,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.loss_type = loss_type
        self.mask_only = mask_only
        self.patch_normalize = patch_normalize
        self.reduction = reduction
        self._custom_loss_fn: Optional[Callable] = None

    def register_custom_loss(
        self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        """Register a custom loss function.

        :param fn: Callable taking (pred, target) both of shape (N, T, P) and
                   returning per-patch losses of shape (N, T).
        """
        self._custom_loss_fn = fn

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-patch loss based on loss_type.

        :param pred: Predictions, shape (N, T, P)
        :param target: Targets, shape (N, T, P)
        :return: Per-patch losses, shape (N, T)
        """
        if self.loss_type == "mse":
            return (pred - target).pow(2).mean(dim=-1)

        elif self.loss_type == "cosine":
            # Cosine similarity: 1 = identical, -1 = opposite
            # Loss: 1 - similarity (so 0 = perfect, 2 = worst)
            similarity = F.cosine_similarity(pred, target, dim=-1)
            return 1 - similarity

        elif self.loss_type == "smooth_l1":
            # Huber loss, less sensitive to outliers than MSE
            return F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)

        elif self.loss_type == "custom":
            if self._custom_loss_fn is None:
                raise ValueError(
                    "loss_type='custom' but no custom loss registered. "
                    "Call register_custom_loss() first."
                )
            return self._custom_loss_fn(pred, target)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _validate_inputs(
        self, pred: torch.Tensor, imgs: torch.Tensor, mask: torch.Tensor
    ):
        """Validate input tensors for correctness."""
        p = self.patch_size

        # NaN/Inf checks
        assert not torch.isnan(imgs).any(), "imgs contains NaN values"
        assert not torch.isinf(imgs).any(), "imgs contains Inf values"
        assert not torch.isnan(pred).any(), "pred contains NaN values"
        assert not torch.isinf(pred).any(), "pred contains Inf values"

        # Shape checks
        assert imgs.ndim == 4, f"imgs must be 4D (N, C, H, W), got {imgs.shape}"
        N, C, H, W = imgs.shape

        assert H % p == 0, f"Height {H} must be divisible by patch_size {p}"
        assert W % p == 0, f"Width {W} must be divisible by patch_size {p}"

        T_expected = (H // p) * (W // p)
        pixels_per_patch = p * p * C

        assert pred.ndim == 3, f"pred must be 3D (N, T, D), got {pred.shape}"
        assert pred.shape == (
            N,
            T_expected,
            pixels_per_patch,
        ), (
            f"pred shape {pred.shape} != expected ({N}, {T_expected}, {pixels_per_patch})"
        )

        assert mask.ndim == 2, f"mask must be 2D (N, T), got {mask.shape}"
        assert mask.shape == (
            N,
            T_expected,
        ), f"mask shape {mask.shape} != expected ({N}, {T_expected})"

        if self.mask_only:
            assert mask.sum() > 0, "mask has no masked patches"

        # Device/dtype consistency
        assert pred.device == imgs.device and mask.device == imgs.device
        assert pred.dtype == imgs.dtype

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches.

        :param imgs: Images of shape (N, C, H, W)
        :return: Patches of shape (N, T, P) where T = num_patches, P = pixels_per_patch
        """
        return patchify(imgs, (imgs.size(1), self.patch_size, self.patch_size))

    def forward(
        self,
        pred: torch.Tensor,
        imgs: torch.Tensor,
        mask: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        """Compute MAE reconstruction loss.

        :param pred: Decoder predictions, shape (N, T, patch_size² × C)
        :param imgs: Original images, shape (N, C, H, W)
        :param mask: Binary mask, shape (N, T), 1 = masked (compute loss)
        :param debug: If True, print debug statistics
        :return: Scalar loss value
        """
        self._validate_inputs(pred, imgs, mask)

        # Patchify target images
        target = self.patchify(imgs)

        if debug:
            self._print_debug(pred, target, mask)

        # Per-patch normalization (optional)
        if self.patch_normalize:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # Compute per-patch loss
        loss = self._compute_loss(pred, target)  # (N, T)

        # Apply mask and reduce
        if self.mask_only:
            if self.reduction == "mean":
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = (loss * mask).sum()
        else:
            if self.reduction == "mean":
                loss = loss.mean()
            else:
                loss = loss.sum()

        assert not torch.isnan(loss), "Loss is NaN"
        return loss

    def _print_debug(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ):
        """Print debug statistics."""
        print("=" * 60)
        print(f"MAE Loss Debug | loss_type={self.loss_type}")
        print("=" * 60)
        print(
            f"pred:   shape={tuple(pred.shape)}, "
            f"min={pred.min():.4f}, max={pred.max():.4f}, "
            f"mean={pred.mean():.4f}, std={pred.std():.4f}"
        )
        print(
            f"target: shape={tuple(target.shape)}, "
            f"min={target.min():.4f}, max={target.max():.4f}, "
            f"mean={target.mean():.4f}, std={target.std():.4f}"
        )
        print(
            f"mask:   {mask.sum().item()}/{mask.numel()} patches masked "
            f"({mask.float().mean().item() * 100:.1f}%)"
        )
        print("=" * 60)
