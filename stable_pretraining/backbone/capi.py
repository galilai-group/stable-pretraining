# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from facebookresearch/capi at 98b4fa17ee8eec8810c17022df9a27a44845368b.
# Licensed under Apache-2.0; see licenses/CAPI-Apache-2.0.txt.
"""Coordinate-preserving CAPI encoder and cross-attention predictor."""

import logging
import math
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    """Mlp component adapted from the pinned CAPI reference."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        mlp_ratio: int | float | None = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_features is None:
            assert mlp_ratio is not None
            hidden_features = int(in_features * mlp_ratio)
        else:
            assert mlp_ratio is None
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def rotate_half(x: Tensor) -> Tensor:
    """Rotate paired RoPE channels following the reference layout.

    Args:
        x: Query or key channels.

    Returns:
        Tensor with negated second half followed by the first half.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rope(nn.Module):
    """Rope component adapted from the pinned CAPI reference."""

    def __init__(
        self, dim: int, max_freq: float | int = 7, min_freq: float | int = 0.0007
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def _device_weight_init(self) -> None:
        freqs_1d = self.max_freq * (self.max_freq / self.min_freq) ** torch.linspace(
            0, -1, self.dim // 4
        )
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()


class Attention(nn.Module):
    """Attention component adapted from the pinned CAPI reference."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
        rope_kwargs: Mapping = {},
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        context_dim = context_dim or dim
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = Rope(dim=head_dim, **rope_kwargs)

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        context: Tensor | None = None,
        context_coords: Tensor | None = None,
    ) -> Tensor:
        if context is None or context_coords is None:
            context = x
            context_coords = coords
        b, n_q, d = x.shape
        b, n_k, _ = context.shape
        h = self.num_heads
        q = self.q_proj(x).reshape(b, n_q, h, d // h).transpose(1, 2)
        k = self.k_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)
        v = self.v_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)
        q = self.rope(q, coords[:, None, :, :])
        k = self.rope(k, context_coords[:, None, :, :])
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape([b, n_q, d])
        x = self.proj(x)
        return x


class NaiveResidual(nn.Module):
    """NaiveResidual component adapted from the pinned CAPI reference."""

    def __init__(self, drop_prob: float | int, norm: nn.Module, fn: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.keep_prob = 1 - drop_prob

    def forward(self, x: Tensor, **kwargs: Tensor | None) -> Tensor:
        fn_out = self.fn(self.norm(x), **kwargs)
        if self.keep_prob == 1.0 or not self.training:
            return x + fn_out
        mask = fn_out.new_empty(x.shape[0]).bernoulli_(self.keep_prob)[:, None, None]
        return x + fn_out * mask / self.keep_prob


class EfficientResidual(NaiveResidual):
    """EfficientResidual component adapted from the pinned CAPI reference."""

    def forward(self, x: Tensor, **kwargs: Tensor | None) -> Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x + self.fn(self.norm(x), **kwargs)
        b, _, _ = x.shape
        n_keep = max(int(b * self.keep_prob), 1)
        indices = torch.randperm(b, device=x.device)[:n_keep]
        for k, v in kwargs.items():
            if v is not None:
                kwargs[k] = v[indices]
        return torch.index_add(
            x,
            dim=0,
            # Autocast leaves master registers in fp32; index_add requires matching dtypes.
            source=self.fn(self.norm(x[indices]), **kwargs).to(x.dtype),
            index=indices,
            alpha=b / n_keep,
        )


class Block(nn.Module):
    """Block component adapted from the pinned CAPI reference."""

    def __init__(
        self,
        dim: int,
        drop_path: float | int,
        norm_layer: Callable[[int], nn.Module],
        context_dim: int | None,
        drop_path_type: str = "efficient",
        attn_kwargs: Mapping = {},
    ) -> None:
        super().__init__()
        residual_module = {"naive": NaiveResidual, "efficient": EfficientResidual}[
            drop_path_type
        ]
        self.residual1 = residual_module(
            drop_path,
            norm_layer(dim),
            Attention(dim, context_dim=context_dim, **attn_kwargs),
        )
        self.residual2 = residual_module(
            drop_path, norm_layer(dim), Mlp(in_features=dim)
        )

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        coords: Tensor | None = None,
        context_coords: Tensor | None = None,
    ) -> Tensor:
        x = self.residual1(
            x, context=context, coords=coords, context_coords=context_coords
        )
        x = self.residual2(x)
        return x


class Transformer(nn.Module):
    """Transformer component adapted from the pinned CAPI reference."""

    def __init__(
        self,
        embed_dim: int,
        norm_layer: Callable[[int], nn.Module],
        depth: int,
        drop_path_rate: float | int,
        context_dim: int | None = None,
        block_kwargs: Mapping[str, Any] = {},
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    context_dim=context_dim,
                    **block_kwargs,
                )
                for i in range(depth)
            ]
        )

    def forward(
        self,
        x: Tensor,
        return_layers: set[int],
        contexts: list[Tensor] | None = None,
        coords: Tensor | None = None,
        context_coords: Tensor | None = None,
    ) -> dict[int, Tensor]:
        outputs = {}
        if 0 in return_layers:
            outputs[0] = x
        for blk_idx, blk in enumerate(self.blocks):
            context = contexts[blk_idx] if contexts is not None else None
            x = blk(x, context=context, coords=coords, context_coords=context_coords)
            if blk_idx + 1 in return_layers:
                outputs[blk_idx + 1] = x
        return outputs


class CAPIBackbone(nn.Module):
    """CAPI image encoder and separate cross-attention predictor.

    Use :func:`capi_backbone` to build an initialized model. Register tokens are
    prepended to context, never used as patch targets. Patch removal preserves
    original normalized image coordinates for both self- and cross-attention.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_chans: int = 3,
        norm_layer_type: str = "RMSNorm",
        n_registers: int = 16,
        register_coordinates: str = "edge",
        scale_reg_to_visible: bool = True,
        transformers_kwargs: Mapping[str, Any] = {},
        encoder_kwargs: Mapping[str, Any] = {},
        decoder_kwargs: Mapping[str, Any] = {},
        norm_layer_kwargs: Mapping[str, Any] = {"eps": 1e-05},
        final_norm_kwargs: Mapping[str, Any] = {"elementwise_affine": False},
        out_layer: int = -1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_prefix_tokens = self.n_registers = n_registers
        self.register_coordinates = register_coordinates
        self.scale_reg_to_visible = scale_reg_to_visible
        norm_layer: Callable[[int], nn.Module] = partial(
            getattr(torch.nn, norm_layer_type), **norm_layer_kwargs
        )
        self.encoder = Transformer(
            **transformers_kwargs, **encoder_kwargs, norm_layer=norm_layer
        )
        self.decoder = Transformer(
            **transformers_kwargs,
            **decoder_kwargs,
            context_dim=self.encoder.embed_dim,
            norm_layer=norm_layer,
        )
        self.embed_dim = self.encoder.embed_dim
        self.pred_dim = self.decoder.embed_dim
        self.n_blocks = len(self.encoder.blocks)
        self.out_layer = out_layer % (len(self.encoder.blocks) + 1)
        self.mask_token = nn.Parameter(torch.empty(1, self.decoder.embed_dim))
        self.registers = nn.Parameter(
            torch.empty(1, n_registers, self.encoder.embed_dim)
        )
        self.patch_embed = nn.Conv2d(
            in_chans, self.encoder.embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.enc_norm = norm_layer(self.embed_dim, **final_norm_kwargs)
        self.dec_norm = norm_layer(self.decoder.embed_dim, **final_norm_kwargs)
        logger.debug(f"Created {type(self).__name__}: {self}")

    def train(self, mode: bool = True) -> "CAPIBackbone":
        return super().train(False if getattr(self, "_capi_frozen", False) else mode)

    def init_weights(self) -> "CAPIBackbone":
        """Initialize Xavier projections, reference RoPE and Gaussian registers.

        Returns:
            This initialized backbone.
        """
        self.patch_embed.reset_parameters()
        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.registers, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)
        return self

    def prepare_tokens_and_drop(
        self, x: Tensor, visible_indices: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, _, h, w = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        coord_x = torch.linspace(
            0, 1, h // self.patch_size, device=x.device, dtype=x.dtype
        )
        coord_y = torch.linspace(
            0, 1, w // self.patch_size, device=x.device, dtype=x.dtype
        )
        coords_all = torch.cartesian_prod(coord_x, coord_y)
        coords_all = coords_all[None].expand(b, -1, -1)
        if visible_indices is not None:
            coords = coords_all.flatten(0, 1)[visible_indices].reshape(b, -1, 2)
            x = x.flatten(0, 1)[visible_indices].reshape(b, -1, self.embed_dim)
        else:
            coords = coords_all
        if self.register_coordinates == "zeros":
            reg_coords = torch.zeros(
                b, self.n_registers, 2, device=x.device, dtype=x.dtype
            )
        elif self.register_coordinates == "edge":
            reg_coords = self.get_edge_coordinates(
                self.n_registers, x.dtype, x.device
            ).expand(b, -1, -1)
        else:
            raise ValueError(self.register_coordinates)
        if self.scale_reg_to_visible:
            mi, ma = (
                coords.min(dim=1, keepdim=True).values,
                coords.max(dim=1, keepdim=True).values,
            )
            reg_coords = mi + reg_coords * (ma - mi)
        x = torch.cat([self.registers.expand(b, -1, -1), x], dim=1)
        coords = torch.cat([reg_coords, coords], dim=1)
        return (x, coords, coords_all)

    def get_edge_coordinates(
        self, n: int, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        side = n // 4
        assert n == 4 * side
        reg_coords = torch.zeros(1, n, 2, dtype=dtype, device=device)
        c = torch.arange(side, dtype=dtype, device=device) / side
        reg_coords[:, 0 * side : 1 * side, 0] = c
        reg_coords[:, 0 * side : 1 * side, 1] = 0
        reg_coords[:, 1 * side : 2 * side, 0] = 1
        reg_coords[:, 1 * side : 2 * side, 1] = c
        reg_coords[:, 2 * side : 3 * side, 0] = 1 - c
        reg_coords[:, 2 * side : 3 * side, 1] = 1
        reg_coords[:, 3 * side : 4 * side, 0] = 0
        reg_coords[:, 3 * side : 4 * side, 1] = 1 - c
        return reg_coords

    def forward_features(
        self,
        x: Tensor,
        visible_indices: Tensor | None,
        predict_indices: Tensor | None,
        enc_layer: int,
        dec_layer: int | None,
    ) -> tuple[Tensor, Tensor | None]:
        b, _, _, _ = x.shape
        x, coords_enc, coords_all = self.prepare_tokens_and_drop(x, visible_indices)
        enc_layers = {enc_layer}
        if dec_layer is not None:
            enc_layers.add(len(self.encoder.blocks))
        encoder_outputs = self.encoder(x, coords=coords_enc, return_layers=enc_layers)
        encoder_outputs = {k: self.enc_norm(v) for k, v in encoder_outputs.items()}
        if dec_layer is not None:
            coords_dec = coords_all.flatten(0, 1)[predict_indices].reshape(b, -1, 2)
            decoder_outputs = self.decoder(
                self.mask_token[None].expand(*coords_dec.shape[:2], -1),
                contexts=[encoder_outputs[len(self.encoder.blocks)]]
                * self.decoder.n_blocks,
                coords=coords_dec,
                context_coords=coords_enc,
                return_layers={dec_layer},
            )
            dec_out = self.dec_norm(decoder_outputs[dec_layer])
        else:
            dec_out = None
        logger.debug(f"Taking layer {enc_layer}")
        enc_out = encoder_outputs[enc_layer]
        return (enc_out, dec_out)

    def forward_pretrain(
        self,
        x: Tensor,
        visible_indices: Tensor | None = None,
        predict_indices: Tensor | None = None,
        do_prediction: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Encode visible patches and optionally predict sampled dropped positions."""
        encoder_output, decoder_output = self.forward_features(
            x,
            visible_indices,
            predict_indices,
            enc_layer=self.out_layer,
            dec_layer=len(self.decoder.blocks) if do_prediction else None,
        )
        if decoder_output is not None:
            decoder_output = decoder_output.flatten(0, 1)
        return (encoder_output[:, self.num_prefix_tokens :], decoder_output)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return mean patch features, registers, and the dense feature map."""
        b, _, h, w = x.shape
        enc_out, dec_out = self.forward_features(
            x, None, None, enc_layer=self.out_layer, dec_layer=None
        )
        global_repr = enc_out[:, self.num_prefix_tokens :].mean(dim=1)
        registers = enc_out[:, : self.num_prefix_tokens]
        feature_map = enc_out[:, self.num_prefix_tokens :].reshape(
            b, h // self.patch_size, w // self.patch_size, self.embed_dim
        )
        return (global_repr, registers, feature_map)


def _init_weights(m: nn.Module, xavier_gain=1) -> None:
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=xavier_gain)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm | nn.RMSNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if hasattr(m, "_device_weight_init"):
        m._device_weight_init()


def capi_backbone(
    embed_dim: int = 1024,
    depth: int = 24,
    num_heads: int = 16,
    predictor_depth: int = 12,
    patch_size: int = 14,
    n_registers: int = 16,
    drop_path_rate: float = 0.2,
) -> CAPIBackbone:
    """Build the reference CAPI architecture, optionally at reduced scale.

    Args:
        embed_dim: Encoder and predictor width.
        depth: Encoder block count.
        num_heads: Attention heads; head width must be divisible by four.
        predictor_depth: Cross-attention predictor block count.
        patch_size: Patch side length (14 in the author configuration).
        n_registers: Register count, divisible by four for edge coordinates.
        drop_path_rate: Constant stochastic depth probability.

    Returns:
        Initialized encoder and predictor with explicit patch coordinates.
    """
    if embed_dim % num_heads or (embed_dim // num_heads) % 4:
        raise ValueError("CAPI head width must be divisible by four")
    if n_registers < 4 or n_registers % 4:
        raise ValueError("CAPI requires a positive multiple of four registers")
    if depth < 1 or predictor_depth < 1 or patch_size < 1:
        raise ValueError("Depths and patch size must be positive")
    if not 0 <= drop_path_rate < 1:
        raise ValueError("drop_path_rate must be in [0, 1)")
    return CAPIBackbone(
        patch_size=patch_size,
        n_registers=n_registers,
        transformers_kwargs={
            "embed_dim": embed_dim,
            "drop_path_rate": drop_path_rate,
            "block_kwargs": {"attn_kwargs": {"num_heads": num_heads}},
        },
        encoder_kwargs={"depth": depth},
        decoder_kwargs={"depth": predictor_depth},
    ).init_weights()


def capi_mask(
    batch_size: int,
    grid: tuple[int, int],
    device: torch.device,
    mask_ratio: float = 0.65,
    prediction_subsampling: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample visible blocks and dropped prediction queries.

    Args:
        batch_size: Number of images.
        grid: Patch grid height and width.
        device: Output device.
        mask_ratio: Fraction of dropped patches.
        prediction_subsampling: Fraction of dropped patches to predict.

    Returns:
        Visible and predicted flattened batch-position indices. Counts follow
        the reference integer truncation; tiny grids retain at least one query.
    """
    if not 0 < mask_ratio < 1 or not 0 < prediction_subsampling <= 1:
        raise ValueError("Mask ratio must be in (0,1), subsampling in (0,1]")
    height, width = grid
    n = height * width
    n_masked = int(n * mask_ratio)
    n_visible = n - n_masked
    if n_masked < 1 or n_visible < 1:
        raise ValueError("Grid must support both visible and masked patches")
    n_predict = max(1, int(n_masked * prediction_subsampling))
    visible, predicted = [], []
    # CPU scalar sampling avoids a device synchronization for every rectangle.
    for b in range(batch_size):
        low = max(math.log(0.5), math.log(n_visible / width**2))
        high = min(math.log(2), math.log(height**2 / (n_visible + 1e-5)))
        if low > high:
            # Narrow grids may require aspect ratios outside the reference range.
            low = math.log(n_visible / width**2)
            high = math.log(height**2 / n_visible)
        aspect = math.exp(float(torch.empty(()).uniform_(low, high)))
        h = min(height, math.ceil(math.sqrt(n_visible * aspect)))
        w = min(width, math.ceil(math.sqrt(n_visible / aspect)))
        top = int(torch.randint(height - h + 1, ()))
        left = int(torch.randint(width - w + 1, ()))
        block = torch.zeros(grid, dtype=torch.bool)
        block[top : top + h, left : left + w] = True
        ids = block.flatten().nonzero().flatten()[:n_visible]
        block.zero_().flatten()[ids] = True
        block = torch.roll(
            block,
            (int(torch.randint(height, ())), int(torch.randint(width, ()))),
            (0, 1),
        )
        visible.append(block.flatten().nonzero().flatten() + b * n)
        dropped = (~block).flatten().nonzero().flatten()
        predicted.append(dropped[torch.randperm(n_masked)[:n_predict]] + b * n)
    return torch.cat(visible).to(device), torch.cat(predicted).to(device)


__all__ = ["CAPIBackbone", "capi_backbone"]
