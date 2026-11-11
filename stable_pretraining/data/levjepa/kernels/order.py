"""Launch order for the fused GPU stage (CPU, numpy): patches sorted by the stored-pixel tile their window reads."""

import numpy as np

K = 32  # window side in squash pixels; covers a 16 x 16 patch sub-box of up to 27.3 px plus bilinear taps
# 512: the long side of a wide frame (aspect > ~4.5 at a 384 short edge)
TILES = (64, 128, 256, 512)


def tile_bucket(n, out=224):
    """Smallest tile (index into TILES) holding every stored row (or column) a K-window of an n -> out squash reads."""
    s = n / out
    need = np.floor(s * (K - 1) + 2 * np.maximum(s, 1.0) + 1) + 1
    b = np.searchsorted(np.asarray(TILES), need)
    assert (b < len(TILES)).all(), (
        f"a window needs more than {TILES[-1]} stored pixels; frame too large ({int(np.max(n))} -> {out})"
    )
    return b


def patch_order(H, W, B, rows_per_clip, S, out=224):
    """Per view type: patch ids (row * S + k) sorted by their clip's tile bucket, and the count per bucket (len(TILES) ** 2)."""
    key = tile_bucket(H, out) * len(TILES) + tile_bucket(W, out)  # (B,) per clip
    rows = (
        np.arange(rows_per_clip * B) % B
    )  # view row r belongs to clip r % B (ViewBuilder.crop's v * B + b)
    pk = np.repeat(key[rows], S)
    return np.argsort(pk, kind="stable").astype(np.int32), np.bincount(
        pk, minlength=len(TILES) ** 2
    ).astype(np.int64)
