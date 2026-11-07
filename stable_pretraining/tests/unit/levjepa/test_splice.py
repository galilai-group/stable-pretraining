"""The token loader on the CPU: splicing and the view recipe.

stable_pretraining.data.levjepa on the CPU: the packed JPEGs a batch splices hold exactly the stored frames' MCUs under the kept
patches, for every chroma layout the splicer takes, and the view recipe refuses what the GPU stage cannot build.

Decoded with per-MCU chroma upsampling (as nvJPEG decodes, simplejpeg fastupsample=True), every packed MCU equals the
same MCU of its whole frame, bit for bit.

    pytest tests/levjepa_tokens/test_splice.py
"""

# ruff: noqa: E402  (the imports follow the importorskip checks)

import numpy as np
import pytest

pytestmark = pytest.mark.unit
pytest.importorskip("lance")
pytest.importorskip("numba")
simplejpeg = pytest.importorskip("simplejpeg")

from stable_datasets.framestore.convert import convert_store
from stable_datasets.framestore.store import FrameStore
from stable_pretraining.data.levjepa.recipe import V2, ViewRecipe
from stable_pretraining.data.levjepa.sampling import sample_idx, token_footprints
from stable_pretraining.data.levjepa.splice import MCU_COLS, PACK_COLS, make_batch, needed_mcus
from ._stores import make_store


def decode(b):
    return simplejpeg.decode_jpeg(
        np.asarray(b), colorspace="RGB", fastdct=False, fastupsample=True
    )


@pytest.fixture(scope="module", params=[2, 1, 0], ids=["4:2:0", "4:2:2", "4:4:4"])
def store(request, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("splice") / "frames.lance")
    make_store(
        path, episodes=4, frames=40, size=(70, 101), subsampling=request.param
    )  # partial edge MCUs
    convert_store(path, gpu="none", batch=64, memory="2GB", nice=0)
    return FrameStore(path)


@pytest.mark.parametrize("seed", [0, 1])
def test_packed_mcus_are_the_frames_mcus(store, seed):
    B, T = 4, V2.frames
    jpeg, mcu = np.empty(16 << 20, np.uint8), np.empty((1 << 18, MCU_COLS), np.int32)
    p = make_batch(store, np.random.default_rng(seed), B, jpeg, mcu, V2, stride=2)
    fid, pos, size, h, w = store.descriptors(p["rows"])
    full = [decode(store.frame(f, o, n)) for f, o, n in zip(fid, pos, size)]
    assert (
        p["moff"][-1] > B * T
    )  # the batch packs MCUs of every clip (several per frame)
    mw, mh = (int(x) for x in p["mdim"][0])
    for b in range(B):
        packed = decode(jpeg[p["joff"][b] : p["joff"][b + 1]])
        assert packed.shape == (p["prow"][b] * mh, PACK_COLS * mw, 3)
        for f, my, mx, k in mcu[p["moff"][b] : p["moff"][b + 1]]:
            assert f // T == b
            src = packed[
                (k // PACK_COLS) * mh : (k // PACK_COLS + 1) * mh,
                (k % PACK_COLS) * mw : (k % PACK_COLS + 1) * mw,
            ]
            dst = full[f][my * mh : (my + 1) * mh, mx * mw : (mx + 1) * mw]
            assert np.array_equal(src[: dst.shape[0], : dst.shape[1]], dst), (
                b,
                f,
                my,
                mx,
            )


def test_every_kept_patch_is_covered(store):
    """The MCUs packed for a batch include every MCU any kept patch's footprint touches."""
    B = 4
    jpeg, mcu = np.empty(16 << 20, np.uint8), np.empty((1 << 18, MCU_COLS), np.int32)
    p = make_batch(store, np.random.default_rng(5), B, jpeg, mcu, V2, stride=2)
    H, W = p["size"][:, 0].astype(np.float64), p["size"][:, 1].astype(np.float64)
    fr, xa, ya, xb, yb = token_footprints(p, B, H, W, V2)
    mw, mh = (int(x) for x in p["mdim"][0])
    rows, cols = -(-int(H[0]) // mh), -(-int(W[0]) // mw)
    geo = np.tile([mw, mh, rows, cols], (B * V2.frames, 1))
    want = set()  # every MCU a footprint (widened 1 px) overlaps, clipped to the frame
    for f, x0, y0, x1, y1 in zip(fr, xa, ya, xb, yb):
        c0, c1 = (
            min(max(int(np.floor(v / mw)), 0), cols - 1) for v in (x0 - 1, x1 + 1)
        )
        r0, r1 = (
            min(max(int(np.floor(v / mh)), 0), rows - 1) for v in (y0 - 1, y1 + 1)
        )
        want |= {
            (int(f), r * cols + c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
        }
    mframe, mid = needed_mcus(fr, xa, ya, xb, yb, geo)
    # sorted by frame then id, unique
    assert np.all(np.diff(mframe * (rows * cols) + mid) > 0)
    assert set(zip(mframe.tolist(), mid.tolist())) == want
    got = {(int(f), int(r) * cols + int(c)) for f, r, c, _ in mcu[: p["moff"][-1]]}
    assert want == got


def test_kept_tokens_are_uniform():
    """sample_idx: k distinct sorted tokens per view, every token kept with probability k / N."""
    n, grid, keep, frames = 20000, 6, 2, 16
    idx = sample_idx(np.random.default_rng(0), n, grid, keep, frames)
    N, k = frames * grid * grid, frames * keep
    assert idx.shape == (n, k) and idx.min() >= 0 and idx.max() < N
    assert np.all(np.diff(idx.astype(np.int64), axis=1) > 0)
    freq = np.bincount(idx.ravel(), minlength=N) / n
    sd = np.sqrt(k / N * (1 - k / N) / n)
    assert np.abs(freq - k / N).max() < 5 * sd
    # a pair of tokens is kept together with probability k (k - 1) / (N (N - 1))
    both = np.mean((idx == 0).any(1) & (idx == N - 1).any(1))
    q = k * (k - 1) / (N * (N - 1))
    assert abs(both - q) < 5 * np.sqrt(q / n)


def test_recipe_limits():
    assert (V2.keep_global, V2.keep_local) == (10, 2)
    with pytest.raises(ValueError, match="16 x 16"):
        ViewRecipe(patch=8)
    with pytest.raises(ValueError, match="window"):
        ViewRecipe(local_scale=(0.02, 0.9))
    with pytest.raises(ValueError, match="multiple"):
        ViewRecipe(local_size=100)
