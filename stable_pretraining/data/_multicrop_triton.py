"""Fused crop + colour kernel behind :class:`GPUMultiCrop` (Triton, CUDA only).

One program writes a block of output pixels of one (sample, view, frame): it
resamples the crop from the uint8 source exactly like
``torchvision.ops.roi_align(aligned=True, sampling_ratio=-1)``, applies the
colour ops in registers, and stores uint8. The unfused path converts every
source frame to float and makes ~30 passes over the float views; this one
reads the source once per sample point and writes each output byte once.

Contrast blends towards the mean luma of the frame as it is when contrast
runs, which depends on the ops before it, so :func:`_contrast_frame_kernel`
does each (sample, view, frame) in one program: it resamples the whole view and
applies the ops before contrast while summing that luma, parks the float32
colours in a scratch buffer it reads straight back (still in L2), then applies
the rest. For 96 clips x 10 local views this takes L40S 6.6 -> 3.4 ms and
H100 4.1 -> 3.4 ms against a separate reduction pass, and the mean no longer
depends on atomic ordering.
"""

from typing import Optional, Sequence

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BRIGHTNESS, CONTRAST, SATURATION, HUE = 0, 1, 2, 3
_NONE = -1


@triton.jit
def _clamp01(x):
    return tl.minimum(tl.maximum(x, 0.0), 1.0)


@triton.jit
def _fmod_floor(x, m):
    return x - m * tl.floor(x / m)


@triton.jit
def _luma(r, g, b):
    return 0.2989 * r + 0.587 * g + 0.114 * b


@triton.jit
def _hue_shift(r, g, b, fh):
    maxc = tl.maximum(r, tl.maximum(g, b))
    delta = maxc - tl.minimum(r, tl.minimum(g, b))
    safe = tl.where(delta > 0, delta, 1.0)
    is_r = (r >= g) & (r >= b)
    is_g = (~is_r) & (g >= b)
    h = tl.where(
        is_r,
        _fmod_floor((g - b) / safe, 6.0),
        tl.where(is_g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    h = tl.where(delta > 0, h / 6.0, 0.0)
    s = tl.where(maxc > 0, delta / tl.where(maxc > 0, maxc, 1.0), 0.0)
    h = _fmod_floor(h + fh, 1.0)
    k = _fmod_floor(5.0 + h * 6.0, 6.0)
    r2 = maxc - maxc * s * _clamp01(tl.minimum(k, 4.0 - k))
    k = _fmod_floor(3.0 + h * 6.0, 6.0)
    g2 = maxc - maxc * s * _clamp01(tl.minimum(k, 4.0 - k))
    k = _fmod_floor(1.0 + h * 6.0, 6.0)
    b2 = maxc - maxc * s * _clamp01(tl.minimum(k, 4.0 - k))
    return _clamp01(r2), _clamp01(g2), _clamp01(b2)


@triton.jit
def _apply(OP, r, g, b, fb, fc, fs, fh, mean):
    if OP == 0:
        r, g, b = _clamp01(r * fb), _clamp01(g * fb), _clamp01(b * fb)
    elif OP == 1:
        m = (1.0 - fc) * mean
        r, g, b = _clamp01(fc * r + m), _clamp01(fc * g + m), _clamp01(fc * b + m)
    elif OP == 2:
        m = (1.0 - fs) * _luma(r, g, b)
        r, g, b = _clamp01(fs * r + m), _clamp01(fs * g + m), _clamp01(fs * b + m)
    elif OP == 3:
        r, g, b = _hue_shift(r, g, b, fh)
    return r, g, b


@triton.jit
def _resample(
    src,
    boxes,
    flip_ptr,
    bv,
    frame,
    offs,
    mask,
    H,
    W,
    stride_f,
    stride_c,
    stride_h,
    OH: tl.constexpr,
    OW: tl.constexpr,
    FLIP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Crop ``bv`` of ``frame`` at output pixels ``offs`` as RGB in [0, 1]."""
    py = offs // OW
    px = offs % OW
    if FLIP:
        px = tl.where(tl.load(flip_ptr + bv) != 0, OW - 1 - px, px)

    x1 = tl.load(boxes + bv * 4)
    y1 = tl.load(boxes + bv * 4 + 1)
    x2 = tl.load(boxes + bv * 4 + 2)
    y2 = tl.load(boxes + bv * 4 + 3)
    bin_w = (x2 - x1) / OW
    bin_h = (y2 - y1) / OH
    gw = tl.ceil((x2 - x1) / OW).to(tl.int32)
    gh = tl.ceil((y2 - y1) / OH).to(tl.int32)
    count = tl.maximum(gw * gh, 1).to(tl.float32)
    base = src + frame.to(tl.int64) * stride_f
    Hf = H.to(tl.float32)
    Wf = W.to(tl.float32)

    r = tl.zeros([BLOCK], dtype=tl.float32)
    g = tl.zeros([BLOCK], dtype=tl.float32)
    b = tl.zeros([BLOCK], dtype=tl.float32)
    for iy in range(0, gh):
        y = (y1 - 0.5) + py * bin_h + (iy + 0.5) * bin_h / gh
        for ix in range(0, gw):
            x = (x1 - 0.5) + px * bin_w + (ix + 0.5) * bin_w / gw
            valid = mask & (y >= -1.0) & (y <= Hf) & (x >= -1.0) & (x <= Wf)
            yc = tl.maximum(y, 0.0)
            xc = tl.maximum(x, 0.0)
            yl = yc.to(tl.int32)
            xl = xc.to(tl.int32)
            ytop = yl >= H - 1
            xtop = xl >= W - 1
            yl = tl.where(ytop, H - 1, yl)
            xl = tl.where(xtop, W - 1, xl)
            yh = tl.where(ytop, H - 1, yl + 1)
            xh = tl.where(xtop, W - 1, xl + 1)
            ly = tl.where(ytop, 0.0, yc - yl)
            lx = tl.where(xtop, 0.0, xc - xl)
            hy = 1.0 - ly
            hx = 1.0 - lx
            w1 = hy * hx
            w2 = hy * lx
            w3 = ly * hx
            w4 = ly * lx
            o1 = yl * stride_h + xl
            o2 = yl * stride_h + xh
            o3 = yh * stride_h + xl
            o4 = yh * stride_h + xh
            for c in tl.static_range(3):
                p = base + c * stride_c
                v = (
                    w1 * tl.load(p + o1, mask=valid, other=0).to(tl.float32)
                    + w2 * tl.load(p + o2, mask=valid, other=0).to(tl.float32)
                    + w3 * tl.load(p + o3, mask=valid, other=0).to(tl.float32)
                    + w4 * tl.load(p + o4, mask=valid, other=0).to(tl.float32)
                )
                v = tl.where(valid, v, 0.0)
                if c == 0:
                    r += v
                elif c == 1:
                    g += v
                else:
                    b += v
    scale = 1.0 / (count * 255.0)
    r *= scale
    g *= scale
    b *= scale
    return r, g, b


@triton.jit
def _store_rgb(
    dst, offs, mask, r, g, b, gray_ptr, bv, NPIX: tl.constexpr, GRAY: tl.constexpr
):
    if GRAY:
        lum = _luma(r, g, b)
        on = tl.load(gray_ptr + bv) != 0
        r = tl.where(on, lum, r)
        g = tl.where(on, lum, g)
        b = tl.where(on, lum, b)
    tl.store(dst + offs, libdevice.rint(r * 255.0).to(tl.uint8), mask=mask)
    tl.store(dst + NPIX + offs, libdevice.rint(g * 255.0).to(tl.uint8), mask=mask)
    tl.store(dst + 2 * NPIX + offs, libdevice.rint(b * 255.0).to(tl.uint8), mask=mask)


# Triton specializes int arguments equal to 1 or divisible by 16; op codes,
# frame sizes (padded batch maxima) and strides vary between calls, and each
# new specialization is a ~0.7 s compile.
_DYNAMIC = [
    "T",
    "H",
    "W",
    "n_views",
    "stride_f",
    "stride_c",
    "stride_h",
    "O0",
    "O1",
    "O2",
    "O3",
]


@triton.jit(do_not_specialize=_DYNAMIC)
def _crop_color_kernel(
    src,
    out,
    boxes,
    fb_ptr,
    fc_ptr,
    fs_ptr,
    fh_ptr,
    gray_ptr,
    flip_ptr,
    T,
    H,
    W,
    n_views,
    stride_f,
    stride_c,
    stride_h,
    O0,
    O1,
    O2,
    O3,
    OH: tl.constexpr,
    OW: tl.constexpr,
    JITTER: tl.constexpr,
    GRAY: tl.constexpr,
    FLIP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """No contrast: one block of output pixels of one (sample, view, frame)."""
    vf = tl.program_id(0)  # ((b * n_views + v) * T + t)
    bv = vf // T
    frame = (bv // n_views) * T + vf % T
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < OH * OW
    r, g, b = _resample(
        src,
        boxes,
        flip_ptr,
        bv,
        frame,
        offs,
        mask,
        H,
        W,
        stride_f,
        stride_c,
        stride_h,
        OH,
        OW,
        FLIP,
        BLOCK,
    )
    if JITTER:
        fb = tl.load(fb_ptr + bv)
        fc = tl.load(fc_ptr + bv)
        fs = tl.load(fs_ptr + bv)
        fh = tl.load(fh_ptr + bv)
        # The op order is a runtime argument: one compiled kernel for all orders.
        r, g, b = _apply(O0, r, g, b, fb, fc, fs, fh, 0.0)
        r, g, b = _apply(O1, r, g, b, fb, fc, fs, fh, 0.0)
        r, g, b = _apply(O2, r, g, b, fb, fc, fs, fh, 0.0)
        r, g, b = _apply(O3, r, g, b, fb, fc, fs, fh, 0.0)
    dst = out + vf.to(tl.int64) * (3 * OH * OW)
    _store_rgb(dst, offs, mask, r, g, b, gray_ptr, bv, OH * OW, GRAY)


@triton.jit(do_not_specialize=_DYNAMIC + ["N", "CPOS"])
def _contrast_frame_kernel(
    src,
    out,
    scratch,
    boxes,
    fb_ptr,
    fc_ptr,
    fs_ptr,
    fh_ptr,
    gray_ptr,
    flip_ptr,
    N,
    T,
    H,
    W,
    n_views,
    stride_f,
    stride_c,
    stride_h,
    O0,
    O1,
    O2,
    O3,
    CPOS,
    OH: tl.constexpr,
    OW: tl.constexpr,
    GRAY: tl.constexpr,
    FLIP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """With contrast (at position ``CPOS``): all of one (sample, view, frame)."""
    sp = scratch + tl.program_id(0).to(tl.int64) * (3 * OH * OW)
    for vf in range(tl.program_id(0), N, tl.num_programs(0)):
        _contrast_frame(
            src,
            out,
            sp,
            boxes,
            fb_ptr,
            fc_ptr,
            fs_ptr,
            fh_ptr,
            gray_ptr,
            flip_ptr,
            vf,
            T,
            H,
            W,
            n_views,
            stride_f,
            stride_c,
            stride_h,
            O0,
            O1,
            O2,
            O3,
            CPOS,
            OH,
            OW,
            GRAY,
            FLIP,
            BLOCK,
        )


@triton.jit
def _contrast_frame(
    src,
    out,
    sp,
    boxes,
    fb_ptr,
    fc_ptr,
    fs_ptr,
    fh_ptr,
    gray_ptr,
    flip_ptr,
    vf,
    T,
    H,
    W,
    n_views,
    stride_f,
    stride_c,
    stride_h,
    O0,
    O1,
    O2,
    O3,
    CPOS,
    OH: tl.constexpr,
    OW: tl.constexpr,
    GRAY: tl.constexpr,
    FLIP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bv = vf // T
    frame = (bv // n_views) * T + vf % T
    fb = tl.load(fb_ptr + bv)
    fc = tl.load(fc_ptr + bv)
    fs = tl.load(fs_ptr + bv)
    fh = tl.load(fh_ptr + bv)
    total = 0.0
    for start in range(0, OH * OW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < OH * OW
        r, g, b = _resample(
            src,
            boxes,
            flip_ptr,
            bv,
            frame,
            offs,
            mask,
            H,
            W,
            stride_f,
            stride_c,
            stride_h,
            OH,
            OW,
            FLIP,
            BLOCK,
        )
        if CPOS > 0:
            r, g, b = _apply(O0, r, g, b, fb, fc, fs, fh, 0.0)
        if CPOS > 1:
            r, g, b = _apply(O1, r, g, b, fb, fc, fs, fh, 0.0)
        if CPOS > 2:
            r, g, b = _apply(O2, r, g, b, fb, fc, fs, fh, 0.0)
        total += tl.sum(tl.where(mask, _luma(r, g, b), 0.0), axis=0)
        tl.store(sp + offs, r, mask=mask)
        tl.store(sp + OH * OW + offs, g, mask=mask)
        tl.store(sp + 2 * OH * OW + offs, b, mask=mask)
    mean = total / (OH * OW)
    dst = out + vf.to(tl.int64) * (3 * OH * OW)
    for start in range(0, OH * OW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < OH * OW
        r = tl.load(sp + offs, mask=mask, other=0.0)
        g = tl.load(sp + OH * OW + offs, mask=mask, other=0.0)
        b = tl.load(sp + 2 * OH * OW + offs, mask=mask, other=0.0)
        if CPOS <= 0:
            r, g, b = _apply(O0, r, g, b, fb, fc, fs, fh, mean)
        if CPOS <= 1:
            r, g, b = _apply(O1, r, g, b, fb, fc, fs, fh, mean)
        if CPOS <= 2:
            r, g, b = _apply(O2, r, g, b, fb, fc, fs, fh, mean)
        r, g, b = _apply(O3, r, g, b, fb, fc, fs, fh, mean)
        _store_rgb(dst, offs, mask, r, g, b, gray_ptr, bv, OH * OW, GRAY)


def crop_color(
    x: torch.Tensor,
    boxes: torch.Tensor,
    size: tuple[int, int],
    jitter: Optional[dict] = None,
    order: Sequence[int] = (),
    gray: Optional[torch.Tensor] = None,
    flip: Optional[torch.Tensor] = None,
    block: int = 256,
) -> torch.Tensor:
    """Crop, resize, colour-jitter, grayscale and flip all views in one kernel.

    Args:
        x: Source clips ``(B, T, 3, H, W)`` uint8 on a CUDA device (rows and
            columns contiguous).
        boxes: Crops ``(B, n, 4)`` as ``(x1, y1, x2, y2)`` in pixels.
        size: Output ``(h, w)``.
        jitter: ``(B, n)`` float factors under ``"brightness"``,
            ``"contrast"``, ``"saturation"`` and ``"hue"``; ``None`` skips
            colour jitter.
        order: Op codes (``BRIGHTNESS``, ``CONTRAST``, ``SATURATION``,
            ``HUE``) to apply, in order.
        gray: ``(B, n)`` bool, convert the view to grayscale.
        flip: ``(B, n)`` bool, mirror the view horizontally.
        block: Output pixels per program without contrast.

    Returns:
        Views ``(B, n, T, 3, h, w)`` uint8.
    """
    B, T, C, H, W = x.shape
    if C != 3 or x.dtype != torch.uint8 or x.stride(-1) != 1:
        raise ValueError("expected uint8 (B, T, 3, H, W) with contiguous rows")
    n = boxes.shape[1]
    oh, ow = size
    dev = x.device
    frames = x.reshape(B * T, C, H, W)
    out = torch.empty(B, n, T, 3, oh, ow, dtype=torch.uint8, device=dev)
    boxes = boxes.to(device=dev, dtype=torch.float32).contiguous()
    use_jitter = jitter is not None and len(order) > 0
    ops = list(order) if use_jitter else []
    codes = ops + [_NONE] * (4 - len(ops))
    cpos = ops.index(CONTRAST) if CONTRAST in ops else -1
    dummy = boxes
    f = (
        [
            jitter[k].to(dev, torch.float32).contiguous()
            for k in ("brightness", "contrast", "saturation", "hue")
        ]
        if use_jitter
        else [dummy] * 4
    )
    gray_t = gray.to(dev, torch.int8).contiguous() if gray is not None else dummy
    flip_t = flip.to(dev, torch.int8).contiguous() if flip is not None else dummy
    strides = (frames.stride(0), frames.stride(1), frames.stride(2))
    with torch.cuda.device(dev):
        if cpos < 0:
            _crop_color_kernel[(B * n * T, triton.cdiv(oh * ow, block))](
                frames,
                out,
                boxes,
                *f,
                gray_t,
                flip_t,
                T,
                H,
                W,
                n,
                *strides,
                *codes,
                OH=oh,
                OW=ow,
                JITTER=use_jitter,
                GRAY=gray is not None,
                FLIP=flip is not None,
                BLOCK=block,
                num_warps=4,
            )
        else:
            # Each program handles whole (sample, view, frame)s and reuses its
            # scratch slot. Measured: H100 is fastest with a program per frame
            # (512 px blocks, 4 warps); L40S with 8 programs per SM, each
            # 1024 px / 8 warps, which also keeps the scratch small.
            hopper = torch.cuda.get_device_capability(dev) >= (9, 0)
            progs = B * n * T
            if not hopper:
                sms = torch.cuda.get_device_properties(dev).multi_processor_count
                progs = min(progs, 8 * sms)
            scratch = torch.empty(progs, 3, oh, ow, dtype=torch.float32, device=dev)
            _contrast_frame_kernel[(progs,)](
                frames,
                out,
                scratch,
                boxes,
                *f,
                gray_t,
                flip_t,
                B * n * T,
                T,
                H,
                W,
                n,
                *strides,
                *codes,
                cpos,
                OH=oh,
                OW=ow,
                GRAY=gray is not None,
                FLIP=flip is not None,
                BLOCK=512 if hopper else 1024,
                num_warps=4 if hopper else 8,
            )
    return out
