"""View parameters and kept-patch footprints (CPU, numpy).

View parameters (CPU, numpy), drawn as ViewBuilder.crop/jitter and the encoder's iid PatchEmbed draw them, and the
region of the stored frame each kept patch reads.
"""

import math

import numpy as np


def sample_boxes(rng, n, scale, r):
    area = r.squash * r.squash * rng.uniform(*scale, n)
    ratio = np.exp(rng.uniform(math.log(r.ratio[0]), math.log(r.ratio[1]), n))
    w = np.minimum(np.sqrt(area * ratio), r.squash)
    h = np.minimum(np.sqrt(area / ratio), r.squash)
    x1 = rng.random(n) * (r.squash - w)
    y1 = rng.random(n) * (r.squash - h)
    return np.stack([x1, y1, x1 + w, y1 + h], 1).astype(np.float32)


def sample_idx(rng, n, grid, keep, frames):
    """Kept tokens, drawn iid: frames * keep of the frames * grid^2 of each view, sorted (frame-major)."""
    N = frames * grid * grid
    return np.sort(
        np.argpartition(rng.random((n, N)), frames * keep, axis=1)[:, : frames * keep],
        axis=1,
    ).astype(np.int16)


def sample_params(rng, B, r):
    T = r.frames
    p = {
        "gbox": sample_boxes(rng, B, r.global_scale, r),
        "gflip": rng.random(B) < r.flip_p,
        "gidx": sample_idx(rng, B, r.global_size // r.patch, r.keep_global, T),
    }
    n = r.num_local * B  # row v * B + b, as ViewBuilder.crop
    p.update(
        lbox=sample_boxes(rng, n, r.local_scale, r),
        lflip=rng.random(n) < r.flip_p,
        lidx=sample_idx(rng, n, r.local_size // r.patch, r.keep_local, T),
    )
    apply = rng.random(n) < r.jitter_p
    jit = np.ones((n, 4), np.float32)
    for j, s in enumerate(r.jitter):
        jit[:, j] = np.where(apply, rng.uniform(1 - s, 1 + s, n), 1.0)
    jit[:, 3] = rng.random(n) < r.gray_p
    p["ljit"] = jit
    return p


def token_footprints(p, B, H, W, r):
    """Every kept patch's region of the stored frame.

    Every kept patch's region of the stored frame: the roi_align sample points of its patch x patch sub-box (+-1
    for bilinear taps) in squash space, widened by the antialiased squash filter's support. Returns per patch: frame
    index (clip-major, b * T + t) and box xa, ya, xb, yb in stored pixels. H, W: (B,) stored frame size per clip.
    """
    T, PS, S = r.frames, r.patch, r.squash
    fr, xa, ya, xb, yb = [], [], [], [], []
    for box, flip, idx, size, nview in (
        (p["gbox"], p["gflip"], p["gidx"], r.global_size, 1),
        (p["lbox"], p["lflip"], p["lidx"], r.local_size, r.num_local),
    ):
        g = size // PS
        idx = idx.astype(np.int64)
        t, n = idx // (g * g), idx % (g * g)
        row, col = n // g, n % g
        col = np.where(flip[:, None], g - 1 - col, col)
        bw = ((box[:, 2] - box[:, 0]) / size * PS)[:, None]
        bh = ((box[:, 3] - box[:, 1]) / size * PS)[:, None]
        px, py = box[:, :1] + col * bw, box[:, 1:2] + row * bh
        clip = np.tile(np.arange(B), nview)[:, None]  # local row v * B + b -> clip b
        sx, sy = (W / S)[clip], (H / S)[clip]
        rx, ry = np.maximum(sx, 1.0), np.maximum(sy, 1.0)
        # aligned roi_align samples px - 0.5 .. px + bw - 0.5; bilinear reads one more pixel each side; squash pixel i
        # is centred on stored (i + 0.5) * s - 0.5 with support +-max(s, 1)
        fr.append(np.broadcast_to(clip * T + t, t.shape).ravel())
        xa.append(((px - 2.0 + 0.5) * sx - 0.5 - rx).ravel())
        xb.append(((px + bw + 1.0 + 0.5) * sx - 0.5 + rx).ravel())
        ya.append(((py - 2.0 + 0.5) * sy - 0.5 - ry).ravel())
        yb.append(((py + bh + 1.0 + 0.5) * sy - 0.5 + ry).ravel())
    return tuple(np.concatenate(a) for a in (fr, xa, ya, xb, yb))
