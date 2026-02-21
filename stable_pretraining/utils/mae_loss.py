import torch
from math import prod
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Callable


def patchify(x, patch_size):
    """Convert tensor to patches along the last len(patch_size) dimensions.

    Splits the last k spatial dimensions into non-overlapping patches and
    flattens them into a sequence of patch tokens. This is the standard
    patchification used in Vision Transformers (ViT), MAE, etc.

    :param x: Input tensor of shape (..., S_0, S_1, ..., S_{k-1}) where the
              last k dimensions are spatial and will be patchified.
              Leading dimensions are preserved (e.g., batch, channels).
    :param patch_size: Tuple/list of k patch sizes (p_0, p_1, ..., p_{k-1}).
                       Each spatial dim S_i must be divisible by p_i.
    :return: Patches of shape (..., T, P) where:
             - T = prod(S_i // p_i) is the number of patches
             - P = prod(p_i) is the number of elements per patch

    Examples::

        >>> import torch

        # =================================================================
        # 2D Images: (N, C, H, W) -> (N, C, num_patches, patch_elements)
        # =================================================================
        >>> images = torch.randn(8, 3, 224, 224)
        >>> patches = patchify(images, patch_size=(16, 16))
        >>> patches.shape
        torch.Size([8, 3, 196, 256])  # 196 = 14*14 patches, 256 = 16*16 elements

        # Non-square patches
        >>> patches = patchify(images, patch_size=(14, 16))
        >>> patches.shape
        torch.Size([8, 3, 224, 224])  # 16*14=224 patches, 14*16=224 elements

        # =================================================================
        # 3D Volumes: (N, C, D, H, W) -> (N, C, num_patches, patch_elements)
        # =================================================================
        >>> volumes = torch.randn(4, 1, 64, 128, 128)
        >>> patches = patchify(volumes, patch_size=(8, 16, 16))
        >>> patches.shape
        torch.Size([4, 1, 512, 2048])  # 8*8*8=512 patches, 8*16*16=2048 elements

        # =================================================================
        # 1D Signals: (N, C, L) -> (N, C, num_patches, patch_elements)
        # =================================================================
        >>> signals = torch.randn(16, 2, 1024)
        >>> patches = patchify(signals, patch_size=(64,))
        >>> patches.shape
        torch.Size([16, 2, 16, 64])  # 16 patches of 64 elements each

        # =================================================================
        # Flexible batch dimensions
        # =================================================================
        # No batch dims: (H, W) -> (T, P)
        >>> image = torch.randn(224, 224)
        >>> patches = patchify(image, patch_size=(16, 16))
        >>> patches.shape
        torch.Size([196, 256])

        # Multiple batch dims: (B1, B2, C, H, W) -> (B1, B2, C, T, P)
        >>> x = torch.randn(2, 4, 3, 224, 224)
        >>> patches = patchify(x, patch_size=(16, 16))
        >>> patches.shape
        torch.Size([2, 4, 3, 196, 256])

        # =================================================================
        # Typical ViT usage (channels folded into patches)
        # =================================================================
        >>> images = torch.randn(8, 3, 224, 224)
        >>> # Reshape to (N, H, W, C) then patchify spatial dims
        >>> x = images.permute(0, 2, 3, 1)  # (8, 224, 224, 3)
        >>> patches = patchify(x, patch_size=(16, 16))  # (8, 196, 768)
        >>> patches.shape  # 768 = 16 * 16 * 3
        torch.Size([8, 196, 768])

    See Also:
        :func:`unpatchify`: Inverse operation to reconstruct the original tensor.
    """
    patch_size = tuple(patch_size)
    k = len(patch_size)
    batch_shape = x.shape[:-k]
    spatial_shape = x.shape[-k:]

    # Validate divisibility
    for i, (s, p) in enumerate(zip(spatial_shape, patch_size)):
        if s % p != 0:
            raise ValueError(
                f"Spatial dim {i} (size {s}) must be divisible by patch_size[{i}]={p}"
            )

    # Compute grid size (number of patches per spatial dim)
    grid_size = tuple(s // p for s, p in zip(spatial_shape, patch_size))

    # (..., S_0, S_1, ...) -> (..., n_0, p_0, n_1, p_1, ...)
    interleaved = sum(zip(grid_size, patch_size), ())
    x = x.reshape(*batch_shape, *interleaved)

    # (..., n_0, p_0, n_1, p_1, ...) -> (..., n_0, n_1, ..., p_0, p_1, ...)
    b = len(batch_shape)
    perm = (*range(b), *range(b, b + 2 * k, 2), *range(b + 1, b + 2 * k, 2))
    x = x.permute(perm)

    # (..., n_0, n_1, ..., p_0, p_1, ...) -> (..., T, P)
    return x.reshape(*batch_shape, prod(grid_size), prod(patch_size))


def unpatchify(patches, patch_size, grid_size=None):
    """Reconstruct tensor from patches (inverse of patchify).

    Reverses the patchification process, reconstructing the original spatial
    dimensions from a sequence of flattened patches.

    :param patches: Patch tensor of shape (..., T, P) where:
                    - T is the number of patches
                    - P is the number of elements per patch (must equal prod(patch_size))
    :param patch_size: Tuple/list of k patch sizes (p_0, p_1, ..., p_{k-1}).
    :param grid_size: Tuple/list of k grid sizes (n_0, n_1, ..., n_{k-1}) where
                      n_i is the number of patches along spatial dimension i.
                      If None, assumes a uniform grid (T must be a perfect k-th power).
    :return: Reconstructed tensor of shape (..., S_0, S_1, ..., S_{k-1})
             where S_i = n_i * p_i.

    Examples::

        >>> import torch

        # =================================================================
        # 2D Images: Roundtrip
        # =================================================================
        >>> images = torch.randn(8, 3, 224, 224)
        >>> patches = patchify(images, patch_size=(16, 16))
        >>> reconstructed = unpatchify(patches, patch_size=(16, 16))
        >>> torch.allclose(images, reconstructed)
        True

        # =================================================================
        # 3D Volumes: Roundtrip
        # =================================================================
        >>> volumes = torch.randn(4, 1, 64, 128, 128)
        >>> patches = patchify(volumes, patch_size=(8, 16, 16))
        >>> reconstructed = unpatchify(patches, patch_size=(8, 16, 16))
        >>> torch.allclose(volumes, reconstructed)
        True

        # =================================================================
        # 1D Signals: Roundtrip
        # =================================================================
        >>> signals = torch.randn(16, 2, 1024)
        >>> patches = patchify(signals, patch_size=(64,))
        >>> reconstructed = unpatchify(patches, patch_size=(64,))
        >>> torch.allclose(signals, reconstructed)
        True

        # =================================================================
        # Non-square grid (must specify grid_size)
        # =================================================================
        >>> images = torch.randn(8, 3, 224, 256)  # Non-square image
        >>> patches = patchify(images, patch_size=(16, 16))
        >>> patches.shape
        torch.Size([8, 3, 224, 256])  # 14*16=224 patches
        >>> reconstructed = unpatchify(patches, patch_size=(16, 16), grid_size=(14, 16))
        >>> torch.allclose(images, reconstructed)
        True

        # =================================================================
        # MAE-style reconstruction (predict pixels from patch embeddings)
        # =================================================================
        >>> # Decoder outputs: (N, num_patches, patch_pixels)
        >>> predictions = torch.randn(8, 196, 768)  # 768 = 16*16*3
        >>> # Reconstruct to (N, num_patches, H, W, C) then permute
        >>> images = unpatchify(predictions, patch_size=(16, 16))  # (8, 224, 224)
        >>> # For RGB: reshape last dim and permute
        >>> predictions = torch.randn(8, 196, 768)
        >>> images = unpatchify(predictions.reshape(8, 196, 16, 16, 3), patch_size=(16, 16))
        >>> images = images.permute(0, 3, 1, 2)  # (8, 3, 224, 224)

        # =================================================================
        # Explicit grid_size for non-uniform grids
        # =================================================================
        >>> patches = torch.randn(4, 168, 256)  # 168 = 12 * 14 patches
        >>> images = unpatchify(patches, patch_size=(16, 16), grid_size=(12, 14))
        >>> images.shape
        torch.Size([4, 192, 224])  # 12*16=192, 14*16=224

        # =================================================================
        # Error case: Cannot infer non-uniform grid
        # =================================================================
        >>> patches = torch.randn(4, 168, 256)  # 168 is not a perfect square
        >>> unpatchify(patches, patch_size=(16, 16))  # Raises ValueError
        ValueError: Cannot infer grid: T=168 is not a perfect 2-th power

    See Also:
        :func:`patchify`: Forward operation to convert tensors to patches.
    """
    patch_size = tuple(patch_size)
    k = len(patch_size)
    batch_shape = patches.shape[:-2]
    T, patch_elements = patches.shape[-2:]

    if patch_elements != prod(patch_size):
        raise ValueError(
            f"patches last dim {patch_elements} != prod(patch_size)={prod(patch_size)}"
        )

    # Infer or validate grid_size
    if grid_size is None:
        n = round(T ** (1.0 / k))
        if n**k != T:
            raise ValueError(
                f"Cannot infer grid: T={T} is not a perfect {k}-th power. "
                f"Please provide grid_size explicitly."
            )
        grid_size = (n,) * k
    else:
        grid_size = tuple(grid_size)
        if len(grid_size) != k:
            raise ValueError(
                f"grid_size has {len(grid_size)} dims but patch_size has {k} dims"
            )
        if prod(grid_size) != T:
            raise ValueError(f"prod(grid_size)={prod(grid_size)} != num_patches T={T}")

    # (..., T, P) -> (..., n_0, n_1, ..., p_0, p_1, ...)
    x = patches.reshape(*batch_shape, *grid_size, *patch_size)

    # (..., n_0, n_1, ..., p_0, p_1, ...) -> (..., n_0, p_0, n_1, p_1, ...)
    b = len(batch_shape)
    perm = (*range(b), *sum(zip(range(b, b + k), range(b + k, b + 2 * k)), ()))
    x = x.permute(perm)

    # (..., n_0, p_0, n_1, p_1, ...) -> (..., S_0, S_1, ...)
    spatial_shape = tuple(n * p for n, p in zip(grid_size, patch_size))
    return x.reshape(*batch_shape, *spatial_shape)


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
        imgs = imgs.to(pred.dtype)
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
