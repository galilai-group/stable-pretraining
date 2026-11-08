"""The token loader's GPU stages (skipped without CUDA).

The GPU stages (skipped without CUDA): the spliced frame canvas equals nvJPEG's decode of the whole frames at
every packed MCU (both scatters; and libjpeg-turbo's, decoded on the CPU or by stable_datasets.framestore.gpu_jpeg), the fused Triton
stage agrees with the unfused one, both shared loaders hand out batches of the right shapes, and batches decoded by
CPU workers equal the GPU's.

    pytest tests/levjepa_tokens/test_gpu.py
"""

# ruff: noqa: E402  (the imports follow the importorskip checks)

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

from stable_datasets.framestore.convert import convert_store
from stable_datasets.framestore.decoders import Decoder
from stable_datasets.framestore.store import FrameStore
from stable_pretraining.data.levjepa.recipe import V2
from stable_pretraining.data.levjepa.splice import MCU_COLS, make_batch
from stable_pretraining.data.levjepa.views import TokenViews
from ._stores import make_store

B = 8


@pytest.fixture(scope="module", params=[2, 0], ids=["4:2:0", "4:4:4"])
def path(request, tmp_path_factory):
    p = str(tmp_path_factory.mktemp("gpu") / "frames.lance")
    make_store(p, episodes=6, frames=40, size=(144, 256), subsampling=request.param)
    convert_store(p, batch=64, memory="4GB", nice=0)
    return p


def slots_of(store, seed, decode="partial"):
    jpeg, mcu = np.empty(64 << 20, np.uint8), np.empty((1 << 20, MCU_COLS), np.int32)
    p = make_batch(
        store, np.random.default_rng(seed), B, jpeg, mcu, V2, stride=2, decode=decode
    )
    slots = {
        k: torch.as_tensor(np.asarray(v))[None] for k, v in p.items() if k != "rows"
    }
    slots["jpeg"], slots["mcu"] = (
        torch.from_numpy(jpeg)[None],
        torch.from_numpy(mcu)[None],
    )
    return p, slots


@pytest.mark.parametrize(
    "fused, decoder",
    [
        (False, "nvjpeg"),
        (True, "nvjpeg"),
        (True, "cpu"),
        (True, "triton"),
    ],
    ids=[
        "torch scatter",
        "triton scatter",
        "triton scatter, CPU-decoded",
        "triton scatter, triton-decoded",
    ],
)
def test_canvas_equals_full_decode(path, fused, decoder):
    store = FrameStore(path)
    p, slots = slots_of(store, 0)
    fid, pos, size, _, _ = store.descriptors(p["rows"])
    # the reference: whole frames by the same decoder (nvJPEG), or by libjpeg-turbo for the two exact ones (the CPU,
    # and cuda); nvJPEG's own pixels differ from libjpeg's by up to 3/255 where it decodes on the CUDA cores (A40)
    frames = [np.array(store.frame(f, o, n)) for f, o, n in zip(fid, pos, size)]
    full = (
        Decoder("nvjpeg")(frames)
        if decoder == "nvjpeg"
        else [x.cuda() for x in Decoder("cpu", exact=True)(frames)]
    )
    v = TokenViews(B, fused=fused, decoder="nvjpeg" if decoder == "cpu" else decoder)
    imgs = None
    if (
        decoder == "cpu"
    ):  # as a CPU worker decodes (libjpeg-turbo, chroma per MCU), uploaded
        joff = slots["joff"][0].tolist()
        cpu = Decoder("cpu", exact=True)
        imgs = [
            cpu([np.asarray(slots["jpeg"][0][joff[b] : joff[b + 1]])])[0].cuda()
            for b in range(B)
        ]
    v.fill_canvas(slots, 0, imgs)
    m = slots["mcu"][0][: int(p["moff"][-1])].long()
    mw, mh = (int(x) for x in p["mdim"][0])
    for f in torch.unique(m[:, 0]).tolist():
        rc = m[m[:, 0] == f]
        h, w = full[f].shape[1:]
        need = torch.zeros(-(-h // mh), -(-w // mw), dtype=torch.bool)
        need[rc[:, 1], rc[:, 2]] = True
        need = need.repeat_interleave(mh, 0).repeat_interleave(mw, 1)[:h, :w].cuda()
        d = (v.canvas[f, :, :h, :w].int() - full[f].int()).abs().amax(0)[need]
        assert int(d.max()) == 0, f


def test_fused_matches_unfused(path):
    store = FrameStore(path)
    _, slots = slots_of(store, 3)
    ref, fus = TokenViews(B), TokenViews(B, fused=True)
    r, f = ref(slots, 0), fus(slots, 0)
    assert torch.equal(r[1], f[1]) and torch.equal(r[3], f[3])
    for x, y in (
        (r[0], f[0]),
        (r[2], f[2]),
    ):  # the squash rounds to 8 bits: a rounding tie can flip one step,
        d = (
            (x - y).abs() * 255 * 0.225
        )  # which local jitter's contrast enlarges (measured: max 1.41, 0.04%)
        assert float(d.max()) < 2.0 and float((d > 0.5).float().mean()) < 1e-3


def test_shared_token_loader(path):
    from stable_pretraining.data.levjepa.loader import SharedTokenLoader

    loader = SharedTokenLoader(path, B, workers=2, servers=1, ring=2, slots=4)
    try:
        for _ in range(3):
            gp, gi, lp, li = next(loader)
        T = V2.frames
        assert gp.shape == (B, T * V2.keep_global, 768) and gi.shape == (
            B,
            T * V2.keep_global,
        )
        assert lp.shape == (V2.num_local * B, T * V2.keep_local, 768)
        assert torch.isfinite(gp).all() and torch.isfinite(lp).all()
    finally:
        loader.close()


@pytest.mark.parametrize("spec", ["cpu", "triton:1,cpu"])
def test_cpu_decoded_batches_reach_the_gpu(path, spec):
    """CPU workers' decoded images reach the GPU stage (shared pixel slots, not per-process copies).

    One worker and one server see the same batches in the same order, so CPU decoding agrees batch for batch with the
    triton decoder (both give libjpeg-turbo's pixels; nvJPEG's differ slightly on GPUs without JPEG engines).
    """
    from stable_pretraining.data.levjepa.loader import SharedTokenLoader

    def batches(decoders):
        loader = SharedTokenLoader(
            path, B, workers=1, servers=1, ring=2, slots=4, seed=3, decoders=decoders
        )
        try:
            return [tuple(x.clone() for x in next(loader)) for _ in range(6)]
        finally:
            loader.close()

    got, want = batches(spec), batches("triton:1")
    for g, w in zip(got, want):
        assert torch.equal(g[1], w[1]) and torch.equal(g[3], w[3])
        assert torch.allclose(g[0], w[0], atol=1e-5) and torch.allclose(
            g[2], w[2], atol=1e-5
        )


def test_shared_clip_loader(path):
    from stable_datasets.framestore.pixel_loader import SharedClipLoader

    loader = SharedClipLoader(
        path, B, frames=16, stride=2, cpu_workers=1, byte_readers=1, gpu_servers=1
    )
    try:
        for _ in range(3):
            clip = next(loader)
        assert clip.shape == (B, 16, 224, 224, 3) and clip.dtype == torch.uint8
    finally:
        loader.close()


def test_shared_clip_loader_cheapest_first(path):
    """A consumer slower than the CPU decoders is fed by them alone.

    The GPU servers only fill the ring at the start; decoders="auto" on a converted store runs CPU, nvjpeg and triton.
    """
    import time

    from stable_datasets.framestore.pixel_loader import SharedClipLoader

    loader = SharedClipLoader(
        path, B, frames=16, stride=2, decoders="auto", clip_slots=2
    )
    try:
        assert loader.spec.split(",")[0] == "cpu" and "nvjpeg" in loader.spec
        for _ in range(4):  # the ring's first fill, by whichever decoder is ready
            next(loader)
        before = loader._stats.clone()
        for _ in range(4):
            time.sleep(1.0)  # far slower than one CPU decoder's batch of B clips
            next(loader)
        gpu = int((loader._stats - before)[:, 1].sum()) // B  # clips -> batches
        assert gpu <= 1, (
            f"GPU servers decoded {gpu} batches for a consumer the CPU decoders keep up with"
        )
        st = loader.stats()
        assert set(st) == {"gpu_starved", "decoded_by"} and 0 <= st["gpu_starved"] < 1
    finally:
        loader.close()
