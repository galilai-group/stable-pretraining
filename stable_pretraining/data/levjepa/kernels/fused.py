"""The token loader's GPU stage after decoding, fused: kept patches straight from the decoded frame canvas.

TokenViews (views.py) builds each kept patch in ~30 launches per batch: window squash (token_squash), patch
coordinates, roi_align, flip, jitter, normalize, a copy into the trainer's ring. Every step is linear per axis, so
one Triton program per patch does all of it in registers:

    window = round(Wy @ X @ Wx^T)        v2's antialiased squash, in a K x K window around the patch (squash.py)
    patch  = Ry @ window @ Rx^T          roi_align(aligned=True, sampling_ratio=SR) of the patch's sub-box, as
                                         separable sampling matrices built with roi_align's own rules (window
                                         bounds, clamping); a flip reverses Rx's rows

Globals are normalized and written in PatchEmbed's (c p1 p2) layout. Locals get brightness and add their sum to
their view's total; a second, elementwise kernel then applies contrast around that view mean (C2), saturation,
grayscale and normalization. The CPU workers sort each batch's patches by stored-pixel tile size (one tile size per
clip, from its frame size), so launching needs no device -> host sync.
"""

import os

import torch
import triton
import triton.language as tl

from .order import K, TILES
from .squash import _filter_rows, _first_input

# precision of the squash's column pass (P1: "split" is exact for integer pixels), its row pass (P2) and the roi_align
# passes (PR); warps per program. Defaults from the H100 sweep (probe_fused_kernel.py).
P1 = os.environ.get("FUSED_P1", "split")
P2 = os.environ.get("FUSED_P2", "tf32x3")
PR = os.environ.get("FUSED_PR", "tf32x3")
WARPS = int(os.environ.get("FUSED_WARPS", "2"))
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@triton.jit
def _roi_rows(start, bin_size, src, KK: tl.constexpr, SR: tl.constexpr):
    """roi_align's weights of window pixels, as a (16, KK) matrix.

    (16, KK) roi_align weights of window pixels for output rows src (16,) float: SR samples per bin, each bilinear
    with torchvision's border rules (zero beyond [-1, KK], clamped to the last pixel), averaged.
    """
    j = tl.arange(0, KK)[None, :]
    r = tl.zeros((16, KK), tl.float32)
    for s in tl.static_range(SR):
        y = start + src[:, None] * bin_size + (s + 0.5) * bin_size / SR
        valid = (y >= -1.0) & (y <= KK)
        y = tl.maximum(y, 0.0)
        lo = y.to(tl.int32)
        top = lo >= KK - 1
        lo = tl.where(top, KK - 1, lo)
        hi = tl.where(top, KK - 1, lo + 1)
        y = tl.where(top, lo.to(tl.float32), y)
        ly = y - lo.to(tl.float32)
        w = tl.where(j == lo, 1.0 - ly, 0.0) + tl.where(j == hi, ly, 0.0)
        r += tl.where(valid, w, 0.0)
    return r / SR


@triton.jit
def _patch_kernel(
    canvas,
    sf,
    sc,
    sh,
    hs,
    ws,
    order,
    box,
    flip,
    idx,
    bri,
    vsum,
    out,
    B,
    S,
    mean0,
    mean1,
    mean2,
    std0,
    std1,
    std2,
    T: tl.constexpr,
    G: tl.constexpr,
    SIZE: tl.constexpr,
    SR: tl.constexpr,
    LOCAL: tl.constexpr,
    KK: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    PR: tl.constexpr,
    OUT: tl.constexpr,
):
    q = tl.load(order + tl.program_id(0))  # patch id: view row * S + k
    ch = tl.program_id(1)
    row = q // S
    i = tl.load(idx + q).to(tl.int32)
    t = i // (G * G)
    n = i % (G * G)
    prow = n // G
    pcol = n % G
    fl = tl.load(flip + row) != 0
    pcol = tl.where(fl, G - 1 - pcol, pcol)
    x1 = tl.load(box + row * 4)
    y1 = tl.load(box + row * 4 + 1)
    bw = (tl.load(box + row * 4 + 2) - x1) / SIZE * 16
    bh = (tl.load(box + row * 4 + 3) - y1) / SIZE * 16
    px = x1 + pcol * bw
    py = y1 + prow * bh
    f = (row % B) * T + t
    # window origin as TokenViews.patches: clamped to the squashed frame, past every bilinear tap away from its edges
    r0 = tl.minimum(
        tl.maximum(tl.extra.cuda.libdevice.floor(py - 0.5) - 1, 0.0), (OUT - KK) * 1.0
    ).to(tl.int32)
    c0 = tl.minimum(
        tl.maximum(tl.extra.cuda.libdevice.floor(px - 0.5) - 1, 0.0), (OUT - KK) * 1.0
    ).to(tl.int32)
    H = tl.load(hs + f)
    W = tl.load(ws + f)
    y0 = _first_input(r0, OUT, H)
    x0 = _first_input(c0, OUT, W)
    ys = y0 + tl.arange(0, SH)
    xs = x0 + tl.arange(0, SW)
    m = (ys[:, None] < H) & (xs[None, :] < W)
    x = tl.load(
        canvas + f.to(tl.int64) * sf + ch * sc + ys[:, None] * sh + xs[None, :],
        mask=m,
        other=0,
    ).to(tl.float32)
    wx = _filter_rows(c0, OUT, W, x0, SW, KK)  # (KK, SW)
    if (
        P1 == "split"
    ):  # x is integer-valued (exact in tf32): split only the weights, 2 tf32 passes instead of 3
        hi = (wx.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
        h = tl.dot(x, tl.trans(hi), input_precision="tf32")
        h = tl.dot(x, tl.trans(wx - hi), h, input_precision="tf32")  # (SH, KK)
    else:
        h = tl.dot(x, tl.trans(wx), input_precision=P1)
    wy = _filter_rows(r0, OUT, H, y0, SH, KK)  # (KK, SH)
    win = tl.dot(wy, h, input_precision=P2)  # (KK, KK)
    win = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(win), 0.0), 255.0) * (
        1.0 / 255.0
    )
    a = tl.arange(0, 16).to(tl.float32)
    ry = _roi_rows(py - r0.to(tl.float32) - 0.5, bh / 16, a, KK, SR)  # (16, KK)
    z = tl.dot(ry, win, input_precision=PR)  # (16, KK)
    rx = _roi_rows(
        px - c0.to(tl.float32) - 0.5, bw / 16, tl.where(fl, 15.0 - a, a), KK, SR
    )  # (16, KK)
    p = tl.dot(z, tl.trans(rx), input_precision=PR)  # (16, 16)
    if LOCAL:
        p = tl.minimum(tl.maximum(p * tl.load(bri + row), 0.0), 1.0)
        tl.atomic_add(vsum + row, tl.sum(p))
    else:
        mu = tl.where(ch == 0, mean0, tl.where(ch == 1, mean1, mean2))
        sd = tl.where(ch == 0, std0, tl.where(ch == 1, std1, std2))
        p = (p - mu) / sd
    tl.store(
        out
        + q.to(tl.int64) * 768
        + ch * 256
        + tl.arange(0, 16)[:, None] * 16
        + tl.arange(0, 16)[None, :],
        p,
    )


@triton.jit
def _scatter_kernel(
    base,
    offs,
    prows,
    mcu,
    mdim,
    canvas,
    sf,
    sc,
    sh,
    T: tl.constexpr,
    NCOL: tl.constexpr,
):
    """Copy one decoded packed MCU to its place in the frame canvas.

    Decoded packed MCU i -> its place in the frame canvas. mcu[i] = (frame, MCU row, MCU column, slot k in its
    clip's packed image); mdim[clip] = MCU width, height (8 or 16); offs: byte offset of each clip's decoded image
    (3, rows * mh, NCOL * mw) from base.
    """
    i = tl.program_id(0).to(tl.int64)
    f = tl.load(mcu + i * 4).to(tl.int64)
    my = tl.load(mcu + i * 4 + 1)
    mx = tl.load(mcu + i * 4 + 2)
    k = tl.load(mcu + i * 4 + 3)
    clip = f // T
    mw = tl.load(mdim + clip * 2)
    mh = tl.load(mdim + clip * 2 + 1)
    W = NCOL * mw
    plane = tl.load(prows + clip) * mh * W
    r = tl.arange(0, 16)[:, None]
    c = tl.arange(0, 16)[None, :]
    inside = (r < mh) & (c < mw)
    src = (
        base
        + tl.load(offs + clip)
        + ((k // NCOL) * mh + r).to(tl.int64) * W
        + (k % NCOL) * mw
        + c
    )
    dst = canvas + f * sf + (my * mh + r).to(tl.int64) * sh + mx * mw + c
    for ch in tl.static_range(3):
        tl.store(dst + ch * sc, tl.load(src + ch * plane, mask=inside), mask=inside)


def scatter_mcus(imgs, prow, mcu, canvas, T, ncol=64, mdim=None):
    """Scatter the decoded packed MCUs into the frame canvas.

    imgs: the B decoded packed images (3, rows * mh, ncol * mw) uint8; prow (B,) rows; mcu (M, 4) int32; mdim
    (B, 2) int32 MCU width, height per clip (default 16 x 16); all CUDA.
    """
    base = imgs[
        0
    ]  # every image addressed from one tensor: a byte offset each (may be negative)
    offs = torch.tensor(
        [im.data_ptr() - base.data_ptr() for im in imgs], dtype=torch.int64
    ).to(canvas.device, non_blocking=True)
    if mdim is None:
        mdim = torch.full((len(imgs), 2), 16, dtype=torch.int32, device=canvas.device)
    _scatter_kernel[(len(mcu),)](
        base,
        offs,
        prow,
        mcu,
        mdim,
        canvas,
        canvas.stride(0),
        canvas.stride(1),
        canvas.stride(2),
        T=T,
        NCOL=ncol,
        num_warps=1,
    )


@triton.jit
def _jitter_kernel(out, vsum, jit, S, mean0, mean1, mean2, std0, std1, std2):
    """ViewBuilder.jitter after brightness (done in _patch_kernel), contrast pivoting on the view's kept patches."""
    q = tl.program_id(0)
    row = q // S
    mu = tl.load(vsum + row) / (S * 768)
    con = tl.load(jit + row * 4 + 1)
    sat = tl.load(jit + row * 4 + 2)
    gray = tl.load(jit + row * 4 + 3) != 0
    o = out + q.to(tl.int64) * 768 + tl.arange(0, 256)
    r = tl.load(o)
    g = tl.load(o + 256)
    b = tl.load(o + 512)
    r = tl.minimum(tl.maximum((r - mu) * con + mu, 0.0), 1.0)
    g = tl.minimum(tl.maximum((g - mu) * con + mu, 0.0), 1.0)
    b = tl.minimum(tl.maximum((b - mu) * con + mu, 0.0), 1.0)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    r = tl.minimum(tl.maximum((r - lum) * sat + lum, 0.0), 1.0)
    g = tl.minimum(tl.maximum((g - lum) * sat + lum, 0.0), 1.0)
    b = tl.minimum(tl.maximum((b - lum) * sat + lum, 0.0), 1.0)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    r = tl.where(gray, lum, r)
    g = tl.where(gray, lum, g)
    b = tl.where(gray, lum, b)
    tl.store(o, (r - mean0) / std0)
    tl.store(o + 256, (g - mean1) / std1)
    tl.store(o + 512, (b - mean2) / std2)


def kept_patches(
    canvas,
    hw,
    B,
    T,
    box,
    flip,
    idx,
    order,
    count,
    size,
    sr,
    local,
    out,
    bri=None,
    jit=None,
    squash=224,
    mean=MEAN,
    std=STD,
):
    """Build one view type's kept patches, normalized.

    Kept patches of one view type into out (rows, S, 768) float32, normalized. canvas (F, 3, HC, WC) uint8;
    hw (F, 2) int32 stored sizes; box (rows, 4), flip (rows,) bool, idx (rows, S) int16 (CUDA); order (rows * S,)
    int32 (CUDA) and count (9,) (host) from patch_order; bri / jit (rows,) / (rows, 4) for locals.
    """
    rows, S = idx.shape
    dev = canvas.device
    hs, ws = hw[:, 0].contiguous(), hw[:, 1].contiguous()
    vsum = torch.zeros(rows, device=dev) if local else out
    fl = flip.view(torch.uint8) if flip.dtype == torch.bool else flip
    start = 0
    for k, c in enumerate(count.tolist()):
        if c:
            _patch_kernel[(c, 3)](
                canvas,
                canvas.stride(0),
                canvas.stride(1),
                canvas.stride(2),
                hs,
                ws,
                order[start:],
                box,
                fl,
                idx,
                bri if local else box,
                vsum,
                out,
                B,
                S,
                *mean,
                *std,
                T=T,
                G=size // 16,
                SIZE=size,
                SR=sr,
                LOCAL=local,
                KK=K,
                SH=TILES[k // len(TILES)],
                SW=TILES[k % len(TILES)],
                P1=P1,
                P2=P2,
                PR=PR,
                OUT=squash,
                num_warps=WARPS,
            )
        start += c
    if local:
        _jitter_kernel[(rows * S,)](out, vsum, jit, S, *mean, *std, num_warps=4)
    return out
