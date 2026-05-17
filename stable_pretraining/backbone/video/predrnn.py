"""PredRNN-v2 recurrent video encoder.

Reference
---------
Wang et al., "PredRNN: A Recurrent Neural Network for Spatiotemporal
Predictive Learning", TPAMI 2022 (https://arxiv.org/abs/2103.09504).
Extends the original PredRNN (NeurIPS 2017) and PredRNN++ (ICML 2018)
with a *memory-decoupling* auxiliary loss that pushes the two memory
tracks to capture different aspects of the dynamics.

Design
------
- Per layer: a :class:`STLSTMCell` with two memory tracks:

    * ``C`` — the standard LSTM cell memory, flowing **temporally** within
      each layer (layer ``l`` at time ``t`` reads ``C_{l, t-1}``).
    * ``M`` — the spatiotemporal memory, flowing **vertically** through
      layers within a time step then **zigzagging** back to the bottom
      layer at the next time step (layer ``l`` reads ``M_{l-1, t}``,
      and layer 0 at time ``t`` reads ``M_{L-1, t-1}``).

- Optional :class:`GHU` (Gradient Highway Unit, from PredRNN++) between
  the first and second ST-LSTM layers. Helps gradient flow for long
  sequences.
- Memory decoupling loss (PredRNN-v2 contribution): cosine-similarity
  penalty between the input-gate increments ``i_t * g_t`` and
  ``i'_t * g'_t`` of ``C`` and ``M``. Exposed via
  ``return_decouple_loss=True``.

The model is **naturally causal** in time — output frame ``t`` only
depends on input frames ``[0, t]`` because of the recurrence. No padding
tricks needed (unlike causal 3D convolution).

Compilation
-----------
The model loops over time in Python: ``for t in range(num_frames): ...``.
With a static ``num_frames`` (the recommended path), ``torch.compile``
traces the loop fully and emits one graph. Dynamic ``num_frames`` works
but triggers recompilation on shape change.

Scaling
-------
Recurrent video models have **no published reference past ~50M params**.
The ``huge`` preset (~235M) is honest scaling territory rather than a
reproduction; we do not provide a ``giant`` or ``gigantic`` preset because
training backpropagation-through-time at that scale is an open research
problem.

Example::

    enc = predrnn_v2_base(num_frames=16)
    out = enc(torch.randn(2, 3, 16, 64, 64))
    out.feature_map.shape  # (2, 128, 16, 64, 64)
    out.pooled.shape  # (2, 128)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from transformers.utils import ModelOutput


@dataclass
class PredRNNv2Output(ModelOutput):
    """Structured output of :class:`PredRNNv2`.

    :param feature_map: ``(B, hidden_channels, T, H', W')`` — per-timestep
        hidden states from the top layer.
    :param pooled: ``(B, hidden_channels)`` global average over ``T, H', W'``
        when ``global_pool='avg'``, else ``None``.
    :param decouple_loss: Scalar memory-decoupling loss when
        ``return_decouple_loss=True``, else ``None``. Backprop-friendly.
    """

    feature_map: torch.Tensor = None
    pooled: Optional[torch.Tensor] = None
    decouple_loss: Optional[torch.Tensor] = None


class STLSTMCell(nn.Module):
    """Spatiotemporal LSTM cell (PredRNN family).

    Maintains the two memory tracks ``C`` (temporal) and ``M`` (spatio-
    temporal). All "linear" projections are convolutional so the cell
    operates on full 2D feature maps. To minimize Python-level overhead
    inside the time loop, the per-input gate projections are fused into
    three large convs:

    - ``conv_x``: input ``X`` → 7 gate streams ``[g, i, f, g', i', f', o]``.
    - ``conv_h``: hidden ``H_prev`` → 4 streams ``[g, i, f, o]``.
    - ``conv_m``: memory ``M_prev`` → 3 streams ``[g', i', f']``.

    :param in_channels: Input channels.
    :param hidden_channels: Output / state channels.
    :param kernel_size: Spatial kernel size (odd).
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        p = kernel_size // 2

        self.conv_x = nn.Conv2d(
            in_channels, 7 * hidden_channels, kernel_size, padding=p
        )
        self.conv_h = nn.Conv2d(
            hidden_channels, 4 * hidden_channels, kernel_size, padding=p
        )
        self.conv_m = nn.Conv2d(
            hidden_channels, 3 * hidden_channels, kernel_size, padding=p
        )
        # Extra projection from cat(C_new, M_new) into the output gate's logit.
        self.conv_o = nn.Conv2d(
            2 * hidden_channels, hidden_channels, kernel_size, padding=p
        )
        # 1x1 fuse cat(C_new, M_new) -> H_new.
        self.conv_last = nn.Conv2d(2 * hidden_channels, hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        m_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the cell one step.

        :param x: ``(B, in_channels, H, W)``.
        :param h_prev: previous hidden state, ``(B, hidden_channels, H, W)``.
        :param c_prev: previous temporal memory, same shape as ``h_prev``.
        :param m_prev: previous spatiotemporal memory, same shape as ``h_prev``.
        :return: ``(h_new, c_new, m_new, delta_c, delta_m)``. ``delta_c`` and
            ``delta_m`` are ``i * g`` and ``i' * g'`` respectively — the
            increments used by the memory-decoupling loss.
        """
        x_proj = self.conv_x(x)
        h_proj = self.conv_h(h_prev)
        m_proj = self.conv_m(m_prev)

        d = self.hidden_channels
        x_g, x_i, x_f, x_g2, x_i2, x_f2, x_o = torch.split(x_proj, d, dim=1)
        h_g, h_i, h_f, h_o = torch.split(h_proj, d, dim=1)
        m_g, m_i, m_f = torch.split(m_proj, d, dim=1)

        # Temporal memory track (C)
        g_t = torch.tanh(x_g + h_g)
        i_t = torch.sigmoid(x_i + h_i)
        f_t = torch.sigmoid(x_f + h_f)
        delta_c = i_t * g_t
        c_new = f_t * c_prev + delta_c

        # Spatiotemporal memory track (M)
        g_t2 = torch.tanh(x_g2 + m_g)
        i_t2 = torch.sigmoid(x_i2 + m_i)
        f_t2 = torch.sigmoid(x_f2 + m_f)
        delta_m = i_t2 * g_t2
        m_new = f_t2 * m_prev + delta_m

        cm = torch.cat([c_new, m_new], dim=1)
        o_t = torch.sigmoid(x_o + h_o + self.conv_o(cm))
        h_new = o_t * torch.tanh(self.conv_last(cm))

        return h_new, c_new, m_new, delta_c, delta_m


class GHU(nn.Module):
    """Gradient Highway Unit (PredRNN++).

    Z_t = s * tanh(W_x X + W_z Z_{t-1}) + (1 - s) * Z_{t-1}, with
    s = sigmoid(W_x' X + W_z' Z_{t-1}). Sits between the first and
    second ST-LSTM layers to shortcut gradient back through the time
    loop. Same spatial conv flavor as the cell.

    :param channels: Channel count of ``X`` and ``Z`` (must match).
    :param kernel_size: Odd spatial kernel size.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.channels = channels
        p = kernel_size // 2
        self.conv_x = nn.Conv2d(channels, 2 * channels, kernel_size, padding=p)
        self.conv_z = nn.Conv2d(channels, 2 * channels, kernel_size, padding=p)

    def forward(self, x: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        x_p, x_s = torch.split(self.conv_x(x), self.channels, dim=1)
        z_p, z_s = torch.split(self.conv_z(z_prev), self.channels, dim=1)
        p = torch.tanh(x_p + z_p)
        s = torch.sigmoid(x_s + z_s)
        return s * p + (1.0 - s) * z_prev


def _decouple_term(delta_c: torch.Tensor, delta_m: torch.Tensor) -> torch.Tensor:
    """Cosine similarity penalty between ``delta_c`` and ``delta_m``.

    Used by the memory-decoupling loss. Computed per batch element on the
    flattened tensors then averaged.
    """
    b = delta_c.size(0)
    a = delta_c.reshape(b, -1)
    c = delta_m.reshape(b, -1)
    num = (a * c).sum(dim=1).abs()
    denom = a.norm(dim=1) * c.norm(dim=1) + 1e-7
    return (num / denom).mean()


class PredRNNv2(nn.Module):
    """Stacked PredRNN-v2 encoder.

    :param in_channels: Input channel count (3 for RGB).
    :param hidden_channels: Hidden / output channel width for all layers.
    :param num_layers: Number of stacked ST-LSTM layers.
    :param kernel_size: Spatial kernel size in cell + GHU (odd).
    :param num_frames: Expected number of input frames ``T``. Pinning ``T``
        as an attribute keeps the time loop graph static under
        ``torch.compile``. The forward still works for shorter or longer
        clips at runtime (with recompilation under ``torch.compile``).
    :param patch_size: If >1, apply a ``Conv2d(patch, stride=patch)`` patch
        embed before the recurrent stack. Halves H, W per stride.
    :param use_ghu: Insert a :class:`GHU` between layer 0 and layer 1.
        Requires ``num_layers >= 2``.
    :param return_decouple_loss: If True, accumulate the memory-decoupling
        loss across all (layer, time) and return it in the output dataclass.
    :param global_pool: ``'avg'`` (pool feature map to ``(B, hidden)``) or
        ``''`` (return only the feature map).
    :param use_checkpoint: If True, wrap each per-timestep computation in
        ``torch.utils.checkpoint``. Off by default — backprop-through-time
        is already memory-heavy and recompute makes small models slower.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        num_frames: int = 16,
        patch_size: int = 1,
        use_ghu: bool = True,
        return_decouple_loss: bool = False,
        global_pool: str = "avg",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if use_ghu and num_layers < 2:
            raise ValueError("use_ghu=True requires num_layers >= 2")
        if global_pool not in ("avg", ""):
            raise ValueError(f"global_pool must be 'avg' or '', got {global_pool!r}")
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.return_decouple_loss = return_decouple_loss
        self.global_pool = global_pool
        self.use_checkpoint = use_checkpoint

        if patch_size > 1:
            self.patch_embed = nn.Conv2d(
                in_channels, hidden_channels, patch_size, stride=patch_size
            )
        else:
            self.patch_embed = nn.Conv2d(in_channels, hidden_channels, 1)

        self.cells = nn.ModuleList(
            [
                STLSTMCell(hidden_channels, hidden_channels, kernel_size)
                for _ in range(num_layers)
            ]
        )
        self.ghu = GHU(hidden_channels, kernel_size) if use_ghu else None

    def _step(
        self,
        x_t: torch.Tensor,
        h: List[torch.Tensor],
        c: List[torch.Tensor],
        m: torch.Tensor,
        z: Optional[torch.Tensor],
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """Advance all layers one time step.

        Returns updated ``(h, c, m, z, decouple_acc)``. ``decouple_acc`` is
        the *summed* decoupling term for this step (caller accumulates
        across time only if it wants the loss).
        """
        new_h = [None] * self.num_layers
        new_c = [None] * self.num_layers
        decouple_acc = x_t.new_zeros(())

        # Layer 0
        h0, c0, m, dc, dm = self.cells[0](x_t, h[0], c[0], m)
        new_h[0] = h0
        new_c[0] = c0
        if self.return_decouple_loss:
            decouple_acc = decouple_acc + _decouple_term(dc, dm)

        layer_input = h0
        if self.ghu is not None:
            z = self.ghu(h0, z)
            layer_input = z

        for layer_idx in range(1, self.num_layers):
            h_l, c_l, m, dc, dm = self.cells[layer_idx](
                layer_input, h[layer_idx], c[layer_idx], m
            )
            new_h[layer_idx] = h_l
            new_c[layer_idx] = c_l
            if self.return_decouple_loss:
                decouple_acc = decouple_acc + _decouple_term(dc, dm)
            layer_input = h_l

        return new_h, new_c, m, z, decouple_acc

    def forward(self, x: torch.Tensor) -> PredRNNv2Output:
        """Encode a video clip.

        :param x: ``(B, C, T, H, W)``.
        :return: :class:`PredRNNv2Output`.
        """
        b, _, t, _, _ = x.shape

        # Patch-embed each frame independently for efficiency.
        # (B, C, T, H, W) -> (B*T, C, H, W) -> conv -> (B, hidden, T, H', W')
        x_flat = x.transpose(1, 2).reshape(b * t, x.size(1), x.size(3), x.size(4))
        x_emb = self.patch_embed(x_flat)
        hp, wp = x_emb.shape[-2:]
        x_emb = x_emb.reshape(b, t, self.hidden_channels, hp, wp).transpose(1, 2)
        # x_emb: (B, hidden, T, H', W')

        device = x_emb.device
        dtype = x_emb.dtype
        h = [
            torch.zeros(b, self.hidden_channels, hp, wp, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(b, self.hidden_channels, hp, wp, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        m = torch.zeros(b, self.hidden_channels, hp, wp, device=device, dtype=dtype)
        z = (
            torch.zeros(b, self.hidden_channels, hp, wp, device=device, dtype=dtype)
            if self.ghu is not None
            else None
        )

        decouple_loss = (
            torch.zeros((), device=device, dtype=dtype)
            if self.return_decouple_loss
            else None
        )
        outputs = []
        for ti in range(t):
            x_t = x_emb[:, :, ti]
            if self.use_checkpoint and self.training:
                h, c, m, z, dl = ckpt.checkpoint(
                    self._step, x_t, h, c, m, z, use_reentrant=False
                )
            else:
                h, c, m, z, dl = self._step(x_t, h, c, m, z)
            if self.return_decouple_loss:
                decouple_loss = decouple_loss + dl
            outputs.append(h[-1])

        feature_map = torch.stack(outputs, dim=2)  # (B, hidden, T, H', W')

        pooled = feature_map.mean(dim=(2, 3, 4)) if self.global_pool == "avg" else None

        return PredRNNv2Output(
            feature_map=feature_map, pooled=pooled, decouple_loss=decouple_loss
        )


# -----------------------------------------------------------------------------
# Scaling presets — width via ``hidden_channels``, depth via ``num_layers``.
# Recurrent video models have no published precedent past ~50M params; the
# huge preset is honest scaling territory, not a reproduction. No giant /
# gigantic presets — training BPTT at that scale is an open problem.
# -----------------------------------------------------------------------------


def predrnn_v2_tiny(**kwargs) -> PredRNNv2:
    """PredRNN-v2 Tiny. ``hidden=32, layers=3, k=3`` (~0.5M params)."""
    return PredRNNv2(hidden_channels=32, num_layers=3, kernel_size=3, **kwargs)


def predrnn_v2_small(**kwargs) -> PredRNNv2:
    """PredRNN-v2 Small. ``hidden=64, layers=4, k=3`` (~2.5M params)."""
    return PredRNNv2(hidden_channels=64, num_layers=4, kernel_size=3, **kwargs)


def predrnn_v2_base(**kwargs) -> PredRNNv2:
    """PredRNN-v2 Base. ``hidden=128, layers=4, k=3`` (~10M params)."""
    return PredRNNv2(hidden_channels=128, num_layers=4, kernel_size=3, **kwargs)


def predrnn_v2_large(**kwargs) -> PredRNNv2:
    """PredRNN-v2 Large. ``hidden=192, layers=6, k=5`` (~100M params)."""
    return PredRNNv2(hidden_channels=192, num_layers=6, kernel_size=5, **kwargs)


def predrnn_v2_huge(**kwargs) -> PredRNNv2:
    """PredRNN-v2 Huge. ``hidden=256, layers=8, k=5`` (~235M params).

    Scaling experiment territory — no published reference past ~50M.
    """
    return PredRNNv2(hidden_channels=256, num_layers=8, kernel_size=5, **kwargs)
