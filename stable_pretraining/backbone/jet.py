"""Jet invertible image encoder with exact affine-coupling log determinants.

Adapted from btrude/jet-pytorch at d71d0dcfdc8b190f9b01795fd188742ce669130d
(Apache-2.0). See licenses/JET_LICENSE and THIRD_PARTY_NOTICES.md. This port
uses BCHW images, indexed permutations, configurable scale parameterization,
FP32-or-higher affine arithmetic, and explicit checkpoint scale metadata.
"""

import math
import warnings
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def parameterize_scale(
    raw_scale: Tensor,
    eps: float = 1e-4,
    parameterization: str = "exp_floor",
) -> tuple[Tensor, Tensor]:
    """Return an affine scale and its analytic logarithm without clipping.

    Args:
        raw_scale: Floating-point conditioner output. Half precision is promoted.
        eps: Positive lower floor, strictly between zero and one.
        parameterization: ``exp_floor`` for eps + (1-eps)*exp(raw), or
            ``jet_sigmoid`` for the original bounded 2*sigmoid(raw) ablation.

    Returns:
        Scale and log scale in float32, or float64 for float64 input. Both
        parameterizations give identity at raw=0 up to floating-point precision.

    Raises:
        ValueError: Invalid configuration or non-floating-point input.
        FloatingPointError: Nonfinite input, overflow, or scale underflow to zero.

    Note:
        The lower floor controls the determinant factors, not every singular
        value of the complete coupling Jacobian. Unbounded expansion can overflow
        finite-precision arithmetic; this raises rather than silently clipping.
    """
    if not 0 < eps < 1:
        raise ValueError("scale_eps must be strictly between zero and one")
    if not raw_scale.is_floating_point():
        raise ValueError("raw_scale must be floating point")
    raw = raw_scale if raw_scale.dtype == torch.float64 else raw_scale.float()
    _require_finite("raw_scale", raw)
    if parameterization == "exp_floor":
        term = raw + math.log1p(-eps)
        log_scale = torch.logaddexp(torch.full_like(raw, math.log(eps)), term)
        # exp(log(eps)) can round below eps; direct addition preserves the floor.
        scale = eps + term.exp()
    elif parameterization == "jet_sigmoid":
        scale = 2 * raw.sigmoid()
        log_scale = F.logsigmoid(raw) + math.log(2)
    else:
        raise ValueError(f"Unknown scale_parameterization: {parameterization!r}")
    _require_finite("scale", scale)
    _require_finite("log_scale", log_scale)
    if not (scale > 0).all():
        raise FloatingPointError("Jet scale underflowed to zero")
    if raw.numel() and log_scale.detach().abs().max() > 10:
        warnings.warn(
            "Jet |log_scale| exceeded 10; no clamping applied",
            RuntimeWarning,
            stacklevel=2,
        )
    return scale, log_scale


def _require_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Jet {name} contains NaN or Inf")


def _stats(value: Tensor, prefix: str) -> dict[str, Tensor]:
    value = value.detach()
    return {
        f"{prefix}_mean": value.mean(),
        f"{prefix}_std": value.std(correction=0),
        f"{prefix}_min": value.min(),
        f"{prefix}_max": value.max(),
    }


class _TransformerBlock(nn.Module):
    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm1(x)
        y, _ = self.attention(y, y, y, need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))


class _Conditioner(nn.Module):
    def __init__(
        self, patch_dim: int, n_patches: int, width: int, depth: int, num_heads: int
    ):
        super().__init__()
        self.input_proj = nn.Linear(patch_dim // 2, width)
        self.position = nn.Parameter(torch.empty(1, n_patches, width))
        nn.init.normal_(self.position, std=width**-0.5)
        self.blocks = nn.Sequential(
            *[_TransformerBlock(width, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)
        self.final_proj = nn.Linear(width, patch_dim)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.input_proj(x) + self.position
        return self.final_proj(self.norm(self.blocks(x))).chunk(2, dim=-1)


class _Coupling(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        n_patches: int,
        width: int,
        depth: int,
        num_heads: int,
        kind: str,
        order: Tensor,
        mode: str,
        eps: float,
        checkpoint_conditioner: bool,
    ):
        super().__init__()
        self.kind = kind
        self.mode = mode
        self.eps = eps
        self.checkpoint_conditioner = checkpoint_conditioner
        self.register_buffer("order", order)
        self.register_buffer("undo", order.argsort())
        self.conditioner = _Conditioner(patch_dim, n_patches, width, depth, num_heads)
        self.diagnostics: dict[str, Tensor] = {}

    def forward(
        self, x: Tensor, inverse: bool = False, capture_stats: bool = False
    ) -> tuple[Tensor, Tensor]:
        axis = -1 if self.kind == "channel" else -2
        a, b = x.index_select(axis, self.order).chunk(2, dim=axis)
        if self.kind == "spatial":
            a = a.reshape(x.shape[0], x.shape[1], x.shape[2] // 2)
            b = b.reshape_as(a)
        if self.checkpoint_conditioner and self.training and torch.is_grad_enabled():
            bias, raw = checkpoint(self.conditioner, a, use_reentrant=False)
        else:
            bias, raw = self.conditioner(a)
        bias = bias.to(dtype=x.dtype)
        raw = raw.to(dtype=x.dtype)
        _require_finite("bias", bias)
        scale, log_scale = parameterize_scale(raw, self.eps, self.mode)
        # Keep Jet's pre-scale bias convention: effective shift is bias * scale.
        b = b / scale - bias if inverse else (b + bias) * scale
        logdet = log_scale.flatten(1).sum(dim=1)
        self.diagnostics = {}
        if capture_stats:
            self.diagnostics = {
                **_stats(log_scale, "log_scale"),
                **_stats(scale, "scale"),
            }
        if self.kind == "spatial":
            a = a.reshape(x.shape[0], x.shape[1] // 2, x.shape[2])
            b = b.reshape_as(a)
        result = torch.cat((a, b), dim=axis).index_select(axis, self.undo)
        _require_finite("coupling output", result)
        _require_finite("coupling logdet", logdet)
        return result, -logdet if inverse else logdet


class Jet(nn.Module):
    """Invertible transformer image encoder with per-sample exact logdet.

    Args:
        image_size: Image height/width, or a square side length.
        patch_size: Square patch side; must divide both image dimensions.
        in_channels: Image channels. patch_size**2 * in_channels must be even.
        coupling_layers: Number of affine coupling layers.
        hidden_dim: Transformer conditioner width, divisible by num_heads.
        depth: Transformer blocks in each conditioner.
        num_heads: Attention heads per conditioner.
        coupling_types: Repeated sequence of ``channel`` and/or ``spatial``.
            Spatial coupling requires an even number of patches.
        scale_parameterization: ``exp_floor`` (default) or ``jet_sigmoid``.
        scale_eps: Lower floor for exp_floor, strictly between zero and one.
        checkpoint_conditioner: Recompute conditioner activations during backward.
        capture_stats: Populate detached ``diagnostics`` after each forward pass.

    Note:
        Forward returns tokens and logdet, not a pooled embedding. Pooling tokens
        is lossy: the reported determinant belongs to the full token output only.
        Images and affine arithmetic use float32 minimum even under autocast.
        Reconstruction is accurate to floating-point precision, not bitwise
        guaranteed after training. Use the same precision context for inverse.
        Scale semantics are stored in state_dict extra state and checked on load.
        This port does not load upstream or deepstats state_dicts directly.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int] = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        coupling_layers: int = 8,
        hidden_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        coupling_types: tuple[str, ...] = ("channel", "spatial"),
        scale_parameterization: str = "exp_floor",
        scale_eps: float = 1e-4,
        checkpoint_conditioner: bool = False,
        capture_stats: bool = False,
    ):
        super().__init__()
        size = (
            (image_size, image_size)
            if isinstance(image_size, int)
            else tuple(image_size)
        )
        if len(size) != 2 or any(
            not isinstance(v, int) or v <= 0
            for v in (
                *size,
                patch_size,
                in_channels,
                coupling_layers,
                hidden_dim,
                depth,
                num_heads,
            )
        ):
            raise ValueError(
                "Jet dimensions, depth, and coupling_layers must be positive integers"
            )
        if any(v % patch_size for v in size):
            raise ValueError("patch_size must divide both image dimensions")
        patch_dim = in_channels * patch_size**2
        n_patches = math.prod(size) // patch_size**2
        if patch_dim % 2 or hidden_dim % num_heads:
            raise ValueError(
                "patch dimension must be even and hidden_dim divisible by num_heads"
            )
        if not coupling_types or set(coupling_types) - {"channel", "spatial"}:
            raise ValueError("coupling_types must contain channel and/or spatial")
        if "spatial" in coupling_types and n_patches % 2:
            raise ValueError("spatial coupling requires an even number of patches")
        if (
            scale_parameterization not in {"exp_floor", "jet_sigmoid"}
            or not 0 < scale_eps < 1
        ):
            raise ValueError("Invalid scale_parameterization or scale_eps")
        self.image_size = size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = patch_dim
        self.n_patches = n_patches
        self.capture_stats = capture_stats
        self.diagnostics: dict[str, Tensor] = {}
        self._scale_config = {
            "version": 1,
            "scale_parameterization": scale_parameterization,
            "scale_eps": scale_eps,
        }
        generator = torch.Generator().manual_seed(0)
        rows, cols = size[0] // patch_size, size[1] // patch_size
        parity = (
            torch.arange(rows)[:, None] + torch.arange(cols)[None, :]
        ).flatten() % 2
        self.blocks = nn.ModuleList()
        spatial_index = 0
        for index in range(coupling_layers):
            kind = coupling_types[index % len(coupling_types)]
            if kind == "channel":
                order = torch.randperm(patch_dim, generator=generator)
            else:
                first = spatial_index % 2
                order = torch.cat(
                    (torch.where(parity == first)[0], torch.where(parity != first)[0])
                )
                spatial_index += 1
            self.blocks.append(
                _Coupling(
                    patch_dim,
                    n_patches,
                    hidden_dim,
                    depth,
                    num_heads,
                    kind,
                    order,
                    scale_parameterization,
                    scale_eps,
                    checkpoint_conditioner,
                )
            )

    def get_extra_state(self) -> dict[str, Any]:
        """Return checkpoint metadata protecting affine scale semantics.

        Returns:
            Scale configuration plus image geometry and coupling layout.
        """
        return {
            **self._scale_config,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "coupling_types": tuple(block.kind for block in self.blocks),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Validate checkpoint metadata before accepting its scale semantics.

        Args:
            state: Metadata produced by get_extra_state.

        Raises:
            RuntimeError: The checkpoint describes a different transform.
        """
        if state != self.get_extra_state():
            raise RuntimeError(
                "Jet checkpoint configuration differs; construct Jet with the saved scale and geometry settings"
            )

    def patchify(self, images: Tensor) -> Tensor:
        """Rearrange BCHW images into tokens without changing volume.

        Args:
            images: Batch matching the configured image geometry.

        Returns:
            Tensor shaped (batch, patches, patch_dim).
        """
        expected = (self.in_channels, *self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"Jet expects BCHW images with C,H,W={expected}; got {tuple(images.shape)}"
            )
        b, c, h, w = images.shape
        p = self.patch_size
        return (
            images.reshape(b, c, h // p, p, w // p, p)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b, self.n_patches, self.patch_dim)
        )

    def unpatchify(self, tokens: Tensor) -> Tensor:
        """Invert patchify into BCHW images.

        Args:
            tokens: Tensor shaped (batch, patches, patch_dim).

        Returns:
            Image tensor with the configured geometry.
        """
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.n_patches,
            self.patch_dim,
        ):
            raise ValueError("Jet token shape must match (batch, n_patches, patch_dim)")
        h, w = self.image_size
        p = self.patch_size
        return (
            tokens.reshape(tokens.shape[0], h // p, w // p, p, p, self.in_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(tokens.shape[0], self.in_channels, h, w)
        )

    def _transform(self, tokens: Tensor, inverse: bool) -> tuple[Tensor, Tensor]:
        if not tokens.is_floating_point() or tokens.shape[0] == 0:
            raise ValueError("Jet requires a nonempty floating-point batch")
        tokens = tokens if tokens.dtype == torch.float64 else tokens.float()
        _require_finite("input", tokens)
        logdet = tokens.new_zeros(tokens.shape[0])
        self.diagnostics = {}
        indices = (
            range(len(self.blocks) - 1, -1, -1) if inverse else range(len(self.blocks))
        )
        for i in indices:
            tokens, ld = self.blocks[i](
                tokens, inverse=inverse, capture_stats=self.capture_stats
            )
            logdet = logdet + ld
            if self.capture_stats:
                self.diagnostics.update(
                    {
                        f"flow/layer_{i}/{key}": value
                        for key, value in self.blocks[i].diagnostics.items()
                    }
                )
        _require_finite("encoder logdet", logdet)
        if self.capture_stats:
            self.diagnostics.update(
                _stats(
                    logdet / (self.n_patches * self.patch_dim), "flow/logdet_per_dim"
                )
            )
        return tokens, logdet

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Encode images and compute the full-output log absolute determinant.

        Args:
            images: Floating-point BCHW images.

        Returns:
            Tokens (B, N, D) and per-sample logdet (B,).
        """
        return self._transform(self.patchify(images), inverse=False)

    def inverse(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Decode tokens with the identical affine scale and opposite logdet.

        Args:
            tokens: Full encoder output (B, N, D), without pooling or projection.

        Returns:
            Reconstructed BCHW images and per-sample inverse logdet (B,).
        """
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.n_patches,
            self.patch_dim,
        ):
            raise ValueError("Jet token shape must match (batch, n_patches, patch_dim)")
        x, logdet = self._transform(tokens, inverse=True)
        return self.unpatchify(x), logdet
