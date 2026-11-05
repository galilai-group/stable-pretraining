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

pytest.importorskip("lance")
pytest.importorskip("numba")
simplejpeg = pytest.importorskip("simplejpeg")

from stable_datasets.framestore.convert import convert_store
from stable_datasets.framestore.store import FrameStore
from stable_pretraining.data.levjepa.recipe import V2, ViewRecipe
from stable_pretraining.data.levjepa.sampling import token_footprints
from stable_pretraining.data.levjepa.splice import (
    MCU_COLS,
    PACK_COLS,
    make_batch,
    needed_mcus,
)
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
    want = set(zip(*needed_mcus(fr, xa, ya, xb, yb, geo)))
    got = {(int(f), int(r) * cols + int(c)) for f, r, c, _ in mcu[: p["moff"][-1]]}
    assert want == got


def test_recipe_limits():
    assert (V2.keep_global, V2.keep_local) == (10, 2)
    with pytest.raises(ValueError, match="16 x 16"):
        ViewRecipe(patch=8)
    with pytest.raises(ValueError, match="window"):
        ViewRecipe(local_scale=(0.02, 0.9))
    with pytest.raises(ValueError, match="multiple"):
        ViewRecipe(local_size=100)
