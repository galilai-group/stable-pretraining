"""v2's antialiased squash to 224 x 224, computed only in a K x K window (squash coordinates) around each kept patch.

data.decode resizes every whole stored frame to 224 x 224 (antialiased bilinear) before ViewBuilder crops it, but a
kept patch's roi_align only reads a ~18 x 18 (global) or up to ~30 x 30 (local) area of that 224 frame. Each Triton
program here builds one patch's window for one channel straight from the uint8 frame canvas:

    window = round(Wy @ X @ Wx^T)

X is the stored-pixel tile under the window (one coalesced load), Wy / Wx are the rows of the antialias filter
matrices for the window's 224 rows / columns, evaluated with PyTorch's formula (upsample_aa: triangle filter of
support max(scale, 1), normalised per output pixel). The products run on tensor cores in 3xTF32 (near-fp32). Matches F.interpolate(..., antialias=True) on the
whole frame up to fp32 summation order (rounding ties).
"""

import os

import torch
import triton
import triton.language as tl

from .order import K, TILES


@triton.jit
def _filter_rows(
    o0, out_size: tl.constexpr, in_size, n0, N: tl.constexpr, R: tl.constexpr
):
    """(R, N) weights of input pixels n0 + j for output pixels o0 + i, as upsample_aa (bilinear, antialias)."""
    scale = in_size.to(tl.float32) / out_size
    support = tl.maximum(scale, 1.0)
    inv = tl.where(scale >= 1.0, 1.0 / scale, 1.0)
    o = (o0 + tl.arange(0, R)).to(tl.float32)
    center = scale * (o + 0.5)
    lo = tl.maximum((center - support + 0.5).to(tl.int32), 0)
    hi = tl.minimum((center + support + 0.5).to(tl.int32), in_size)
    j = n0 + tl.arange(0, N)
    w = 1.0 - tl.abs((j[None, :].to(tl.float32) - center[:, None] + 0.5) * inv)
    w = tl.where(
        (j[None, :] >= lo[:, None]) & (j[None, :] < hi[:, None]),
        tl.maximum(w, 0.0),
        0.0,
    )
    return w / tl.sum(w, axis=1)[:, None]


@triton.jit
def _first_input(o0, out_size: tl.constexpr, in_size):
    scale = in_size.to(tl.float32) / out_size
    support = tl.maximum(scale, 1.0)
    return tl.maximum((scale * (o0 + 0.5) - support + 0.5).to(tl.int32), 0)


@triton.jit
def _windows_kernel(
    canvas,
    out,
    frame,
    r0,
    c0,
    hs,
    ws,
    sf,
    sc,
    sh,
    n_out,
    KK: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    OUT: tl.constexpr,
    PREC: tl.constexpr,
):
    p = tl.program_id(0)
    ch = tl.program_id(1)
    f = tl.load(frame + p)
    R0 = tl.load(r0 + p)
    C0 = tl.load(c0 + p)
    H = tl.load(hs + f)
    W = tl.load(ws + f)
    y0 = _first_input(R0, OUT, H)
    x0 = _first_input(C0, OUT, W)
    wy = _filter_rows(R0, OUT, H, y0, SH, KK)  # (K, SH)
    wx = _filter_rows(C0, OUT, W, x0, SW, KK)  # (K, SW)
    ys = y0 + tl.arange(0, SH)
    xs = x0 + tl.arange(0, SW)
    m = (ys[:, None] < H) & (xs[None, :] < W)
    x = tl.load(
        canvas + f.to(tl.int64) * sf + ch * sc + ys[:, None] * sh + xs[None, :],
        mask=m,
        other=0,
    ).to(tl.float32)
    h = tl.dot(x, tl.trans(wx), input_precision=PREC)  # (SH, K)
    y = tl.dot(wy, h, input_precision=PREC)  # (K, K)
    y = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(y), 0.0), 255.0) * (
        1.0 / 255.0
    )
    i = tl.arange(0, KK)
    tl.store(
        out + (p.to(tl.int64) * 3 + ch) * KK * KK + i[:, None] * KK + i[None, :], y
    )


def tile_need(h, w, r0, c0, out_size):
    """Stored rows / columns a window reads (to pick the tile size)."""

    def span(o0, n):
        scale = n / out_size
        sup = torch.clamp(scale, min=1.0)
        first = torch.clamp((scale * (o0 + 0.5) - sup + 0.5).floor(), min=0)
        last = torch.minimum((scale * (o0 + K - 1 + 0.5) + sup + 0.5).floor(), n)
        return last - first

    return span(r0.float(), h.float()), span(c0.float(), w.float())


PREC = os.environ.get("SQUASH_PREC", "tf32x3")
WARPS = int(os.environ.get("SQUASH_WARPS", "2"))


def windows(canvas, hw, frame, r0, c0, out_size=224):
    """Squash each frame inside a K x K window around each patch.

    Canvas (F, 3, HC, WC) uint8; hw (F, 2) int32 stored sizes; per patch frame, window origin r0, c0 (squash px).
    Returns (N, 3, K, K) float32 in [0, 1]: the out_size x out_size squash of each frame inside each window.
    """
    N = len(frame)
    out = torch.empty(N, 3, K, K, device=canvas.device)
    hs, ws = hw[:, 0].contiguous(), hw[:, 1].contiguous()
    nh, nw = tile_need(hs[frame], ws[frame], r0, c0, out_size)
    sizes = torch.tensor(TILES, device=nh.device, dtype=nh.dtype)
    bh, bw = torch.bucketize(nh, sizes), torch.bucketize(nw, sizes)
    assert int(torch.maximum(bh, bw).max()) < len(TILES), (
        f"a window needs more than {TILES[-1]} stored pixels; frame too large"
    )
    key = bh * len(TILES) + bw
    for k in torch.unique(
        key
    ).tolist():  # tile (rows x cols) per bucket, one of TILES each
        SH, SW = TILES[k // len(TILES)], TILES[k % len(TILES)]
        idx = torch.nonzero(key == k).flatten()
        sub = torch.empty(len(idx), 3, K, K, device=canvas.device)
        _windows_kernel[(len(idx), 3)](
            canvas,
            sub,
            frame[idx].int().contiguous(),
            r0[idx].int().contiguous(),
            c0[idx].int().contiguous(),
            hs,
            ws,
            canvas.stride(0),
            canvas.stride(1),
            canvas.stride(2),
            out_size,
            KK=K,
            SH=SH,
            SW=SW,
            OUT=out_size,
            PREC=PREC,
            num_warps=WARPS,
        )
        out[idx] = sub
    return out
