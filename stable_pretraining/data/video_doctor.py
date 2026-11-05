"""Environment checks for the GPU video data pipeline (``spt video-pipeline doctor``).

The pipeline is: DataLoader workers fetch JPEG frame bytes from a Lance store
built with ``build_lance_video_dataset(..., encoding="jpeg")``
(:class:`LanceVideoSegments` with ``decode=False``), then
:class:`GPUDecodeJPEG` and :class:`GPUMultiCrop` decode and augment the batch
on the GPU, optionally ahead of time through :class:`GPUPrefetcher`.

:func:`run_doctor` reports, for the current machine, every piece that is
missing or that caps throughput: the torch build vs the GPU architecture, the
CUDA ops the pipeline needs, the hardware JPEG decoder, CPU quota, shared
memory, and (given a store) whether its frames can take the fast path. With
``bench=True`` it also measures each stage and names the bottleneck.
"""

import ctypes
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
_MARK = {OK: "✓", WARN: "!", FAIL: "✗", SKIP: "-"}

# LeVJEPA's recipe: 1 global 224 view + 10 local 96 views of 16-frame clips.
LEVJEPA_GROUPS = [
    dict(n_views=1, size=224, scale=(0.8, 1.0)),
    dict(
        n_views=10,
        size=96,
        scale=(0.02, 0.4),
        hflip_p=0.5,
        color_jitter=(0.4, 0.4, 0.2, 0.1),
        color_jitter_p=0.8,
        grayscale_p=0.2,
    ),
]


@dataclass
class Check:
    """One doctor finding.

    Args:
        name: Short label of what was checked.
        status: ``"ok"``, ``"warn"``, ``"fail"`` or ``"skip"``.
        detail: What was found.
        fix: What to do about a ``warn`` / ``fail``.
    """

    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass
class Report:
    """All findings plus optional stage throughputs (clips/s).

    Args:
        checks: Individual findings, in run order.
        throughput: Measured clips/s per pipeline stage (``bench=True``).
        bottleneck: Name of the slowest measured stage.
    """

    checks: list[Check] = field(default_factory=list)
    throughput: dict[str, float] = field(default_factory=dict)
    bottleneck: Optional[str] = None

    @property
    def ok(self) -> bool:
        """``True`` if no check failed."""
        return all(c.status != FAIL for c in self.checks)

    def add(self, name: str, status: str, detail: str, fix: str = "") -> Check:
        """Append a finding and return it."""
        c = Check(name, status, detail, fix)
        self.checks.append(c)
        return c

    def to_json(self) -> str:
        """Serialize the report."""
        return json.dumps(
            {
                "ok": self.ok,
                "checks": [asdict(c) for c in self.checks],
                "throughput_clips_per_s": self.throughput,
                "bottleneck": self.bottleneck,
            },
            indent=2,
        )

    def format(self) -> str:
        """Render the report as aligned plain text."""
        w = max((len(c.name) for c in self.checks), default=0)
        lines = []
        for c in self.checks:
            lines.append(f" {_MARK[c.status]} {c.name:<{w}}  {c.detail}")
            if c.fix and c.status in (WARN, FAIL):
                lines.append(f"   {'':<{w}}  → {c.fix}")
        if self.throughput:
            lines.append("")
            lines.append(" Throughput (clips/s, 16-frame clips):")
            tw = max(len(k) for k in self.throughput)
            for k, v in self.throughput.items():
                tag = "  ← bottleneck" if k == self.bottleneck else ""
                lines.append(f"   {k:<{tw}}  {v:8.0f}{tag}")
        n_fail = sum(c.status == FAIL for c in self.checks)
        n_warn = sum(c.status == WARN for c in self.checks)
        lines.append("")
        lines.append(
            " Ready."
            if not (n_fail or n_warn)
            else f" {n_fail} failed, {n_warn} warnings."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def usable_cpus() -> float:
    """CPUs this process may use: affinity mask capped by the cgroup CPU quota.

    Containers (Kubernetes, RunPod, SLURM with cgroups) often report every
    host core in ``os.cpu_count()`` while the quota allows a fraction of them.

    Returns:
        Number of usable CPUs (fractional if the quota is fractional).
    """
    n = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    quota = None
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            quota = int(q) / int(p)
    except (OSError, ValueError):
        try:
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                quota = q / p
        except (OSError, ValueError):
            pass
    return min(float(n), quota) if quota else float(n)


def _tiny_jpeg(h: int = 64, w: int = 96) -> bytes:
    from torchvision.io import encode_jpeg

    g = torch.Generator().manual_seed(0)
    img = (torch.rand(3, h, w, generator=g) * 255).to(torch.uint8)
    return encode_jpeg(img, quality=90).numpy().tobytes()


def _torchvision_nvjpeg() -> Optional[ctypes.CDLL]:
    import torchvision

    root = Path(torchvision.__file__).parent
    for d in (root.with_name(root.name + ".libs"), root):
        for lib in sorted(d.glob("libnvjpeg*.so*")):
            return ctypes.CDLL(str(lib))
    for name in ("libnvjpeg.so.12", "libnvjpeg.so.13", "libnvjpeg.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    return None


def hardware_jpeg_decoder(device: int = 0) -> tuple[Optional[bool], str]:
    """Whether nvJPEG's hardware backend (the GPU's NVJPG engines) is available.

    torchvision's batched GPU ``decode_jpeg`` tries this backend first and
    falls back to CUDA cores, so this tells which of the two it uses.

    Args:
        device: CUDA device index.

    Returns:
        ``(available, detail)``; ``available`` is ``None`` if nvJPEG could not
        be probed.
    """
    try:
        lib = _torchvision_nvjpeg()
    except OSError as e:
        return None, f"cannot load nvJPEG: {e}"
    if lib is None:
        return None, "nvJPEG library not found next to torchvision"
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # nvJPEG binds to the current context
        handle = ctypes.c_void_p()
        hardware = ctypes.c_int(3)  # NVJPEG_BACKEND_HARDWARE
        status = lib.nvjpegCreateEx(
            hardware, None, None, ctypes.c_uint(0), ctypes.byref(handle)
        )
        if status == 0:
            lib.nvjpegDestroy(handle)
    return status == 0, f"nvjpegCreateEx(HARDWARE) status {status}"


def _arch_supported(
    cap: tuple[int, int], archs: Optional[list[str]] = None
) -> tuple[bool, str]:
    """Whether a torch build with ``archs`` can run kernels on compute capability ``cap``.

    SASS runs on GPUs of the same major version and an equal or higher minor
    one (``sm_86`` code runs on an ``sm_89`` L40S); arch-specific targets
    (``sm_90a``) only on their exact GPU. PTX JIT-compiles for any newer GPU.
    """
    archs = torch.cuda.get_arch_list() if archs is None else archs
    sm = f"sm_{cap[0]}{cap[1]}"
    if sm in archs:
        return True, sm
    target = cap[0] * 10 + cap[1]
    for a in archs:
        kind, _, num = a.partition("_")
        if not num.isdigit():
            continue
        n = int(num)
        if kind == "sm" and n // 10 == cap[0] and n <= target:
            return True, f"{sm} via {a} kernels"
        if kind == "compute" and n <= target:
            return True, f"{sm} via PTX {a}"
    return False, sm


def _check_torch_gpu(r: Report, device: int) -> bool:
    r.add(
        "python / torch",
        OK,
        f"Python {platform.python_version()}, torch {torch.__version__}"
        f" (CUDA {torch.version.cuda or 'none'})",
    )
    if not torch.cuda.is_available():
        r.add(
            "GPU",
            FAIL,
            "torch sees no CUDA device",
            "install a CUDA build of torch and check `nvidia-smi`",
        )
        return False
    cap = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    ok, how = _arch_supported(cap)
    if not ok:
        need = "12.8" if cap[0] >= 10 else "11.8"
        r.add(
            "GPU",
            FAIL,
            f"{name} ({how}) is not in this torch build's arch list "
            f"{torch.cuda.get_arch_list()}",
            f"install torch built with CUDA >= {need}, e.g. "
            "`pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu128`",
        )
        return False
    try:
        x = torch.randn(256, 256, device=device)
        (x @ x).sum().item()
    except Exception as e:  # noqa: BLE001
        r.add("GPU", FAIL, f"{name}: kernel launch failed ({e})", "check the driver")
        return False
    family = (
        "Ada"
        if cap == (8, 9)
        else {
            7: "Volta/Turing",
            8: "Ampere",
            9: "Hopper",
            10: "Blackwell",
            12: "Blackwell",
        }.get(cap[0], "arch")
    )
    r.add("GPU", OK, f"{name}, {family} {how}")
    return True


def _check_cuda_ops(r: Report, device: int) -> bool:
    good = True
    try:
        from torchvision.ops import roi_align

        x = torch.rand(1, 3, 32, 32, device=device)
        boxes = torch.tensor([[0, 0, 0, 31, 31]], dtype=torch.float, device=device)
        roi_align(x, boxes, (8, 8), 1.0, -1, True)
        r.add("torchvision CUDA ops", OK, "roi_align (GPUMultiCrop) runs on GPU")
    except Exception as e:  # noqa: BLE001
        good = False
        r.add(
            "torchvision CUDA ops",
            FAIL,
            f"roi_align on GPU failed: {e}",
            "install the torchvision wheel matching your torch build",
        )
    try:
        from torchvision.io import ImageReadMode, decode_jpeg

        data = torch.frombuffer(bytearray(_tiny_jpeg()), dtype=torch.uint8)
        decode_jpeg([data], mode=ImageReadMode.RGB, device=f"cuda:{device}")
        r.add("nvJPEG (torchvision)", OK, "batched GPU decode_jpeg works")
    except Exception as e:  # noqa: BLE001
        good = False
        r.add(
            "nvJPEG (torchvision)",
            FAIL,
            f"decode_jpeg(device='cuda') failed: {e}",
            "install a CUDA torchvision wheel (pip, not a CPU-only build)",
        )
    return good


def _check_hw_decoder(r: Report, device: int) -> bool:
    ok, detail = hardware_jpeg_decoder(device)
    if ok:
        r.add(
            "hardware JPEG decoder",
            OK,
            "NVJPG engines present; GPUDecodeJPEG decodes on them, off the SMs",
        )
    elif ok is None:
        r.add("hardware JPEG decoder", SKIP, detail)
    else:
        r.add(
            "hardware JPEG decoder",
            WARN,
            f"none on this GPU ({detail}): decode runs on CUDA cores and "
            "competes with training for SMs",
            "GPUs with NVJPG engines (A100, H100, B200, ...) decode off the SMs",
        )
    return bool(ok)


def _check_packages(r: Report) -> None:
    for mod, why, pip in (
        ("lance", "Lance video store (LanceVideoSegments)", "pylance"),
        (
            "cv2",
            "building the store (build_lance_video_dataset)",
            "opencv-python-headless",
        ),
        ("kornia", "GPUVideoSequential (GPUMultiCrop does not need it)", "kornia"),
    ):
        try:
            m = __import__(mod)
            r.add(mod, OK, f"{getattr(m, '__version__', '')} — {why}")
        except ImportError:
            r.add(
                mod,
                WARN if mod != "lance" else FAIL,
                f"missing — {why}",
                f"pip install {pip}",
            )


def _check_host(r: Report, workers: Optional[int]) -> None:
    cpus = usable_cpus()
    host = os.cpu_count() or 1
    detail = f"{cpus:g} usable" + (
        f" of {host} visible (cgroup quota)" if cpus < host else ""
    )
    if workers is not None and workers > cpus:
        r.add(
            "CPUs",
            WARN,
            f"{detail}; {workers} workers oversubscribe them",
            f"use num_workers <= {int(cpus)}",
        )
    elif cpus < 8:
        r.add(
            "CPUs",
            WARN,
            f"{detail}; byte-fetching workers may cap throughput",
            "request more CPUs per GPU (run with --bench --store to measure)",
        )
    else:
        r.add("CPUs", OK, f"{detail}; suggested num_workers ≈ {max(1, int(cpus) - 2)}")
    if sys.platform.startswith("linux"):
        try:
            st = os.statvfs("/dev/shm")
            gb = st.f_frsize * st.f_blocks / 1e9
            if gb < 2:
                r.add(
                    "/dev/shm",
                    WARN,
                    f"{gb:.1f} GB: DataLoader workers pass batches through it",
                    "raise it (docker --shm-size=16g) or use fewer workers",
                )
            else:
                r.add("/dev/shm", OK, f"{gb:.0f} GB")
        except OSError:
            r.add("/dev/shm", WARN, "not mounted", "mount a tmpfs at /dev/shm")
    try:
        torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)
        r.add("pinned memory", OK, "pin_memory=True works")
    except Exception as e:  # noqa: BLE001
        r.add(
            "pinned memory",
            WARN,
            f"cannot pin host memory ({e}); H2D copies will be synchronous",
            "raise the locked-memory limit (ulimit -l)",
        )


def _sample_store(store: str, n_clips: int, window: int):
    from .video import LanceVideoSegments

    ds = LanceVideoSegments(store, window_length=window, hop_size=window, decode=False)
    idx = torch.randperm(len(ds))[:n_clips].tolist()
    return ds, [ds[i] for i in idx]


def _check_store(r: Report, store: str, hw_ok: bool, device: int):
    p = Path(store)
    sidecar = p.with_name(p.name + ".videos.json")
    if not p.exists():
        r.add("store", FAIL, f"{store} does not exist")
        return None
    if not sidecar.exists():
        r.add(
            "store",
            FAIL,
            f"sidecar {sidecar.name} missing",
            "build the store with build_lance_video_dataset",
        )
        return None
    meta = json.loads(sidecar.read_text())
    enc = meta.get("encoding", "webp")
    n_videos = len(meta.get("videos", []))
    if enc != "jpeg":
        r.add(
            "store encoding",
            FAIL,
            f"{enc}: frames must be decoded on the CPU, which caps throughput "
            "at a few hundred clips/s",
            "rebuild with build_lance_video_dataset(..., encoding='jpeg')",
        )
        return None
    r.add("store encoding", OK, f"jpeg, {n_videos} videos")
    try:
        ds, items = _sample_store(store, 32, 16)
    except Exception as e:  # noqa: BLE001
        r.add("store read", FAIL, f"LanceVideoSegments failed: {e}")
        return None
    frames = [bytes(f) for it in items for f in it["video_bytes"]]
    kb = sum(len(f) for f in frames) / len(frames) / 1e3
    headers = [jpeg_header(f) for f in frames]
    if any(h is None for h in headers):
        bad = sum(h is None for h in headers)
        r.add("store frames", FAIL, f"{bad}/{len(frames)} sampled frames are not JPEG")
        return None
    sizes = {(h["height"], h["width"]) for h in headers}
    subs = sorted({h["chroma"] for h in headers})
    progressive = sum(h["progressive"] for h in headers)
    hs = sorted({h for h, _ in sizes})
    detail = (
        f"{len(ds)} clips; frames {kb:.1f} kB, chroma {'/'.join(subs)}, "
        f"{len(sizes)} sizes (height {hs[0]}–{hs[-1]})"
    )
    if progressive and hw_ok:
        r.add(
            "store frames",
            WARN,
            f"{detail}; {progressive}/{len(frames)} progressive JPEGs: the "
            "hardware decoder only takes baseline",
            "re-encode as baseline JPEG",
        )
    elif len(sizes) > 8:
        r.add(
            "store frames",
            WARN,
            f"{detail}: many sizes make padding costlier",
            "build with a fixed resize, e.g. resize=(256, 256)",
        )
    else:
        r.add("store frames", OK, detail)
    return ds


_CHROMA = {
    (1, 1): "4:4:4",
    (2, 1): "4:2:2",
    (2, 2): "4:2:0",
    (1, 2): "4:4:0",
    (4, 1): "4:1:1",
}


def jpeg_header(b: bytes) -> Optional[dict]:
    """Parse the frame header of a JPEG bitstream.

    Args:
        b: Encoded JPEG.

    Returns:
        ``{"height", "width", "chroma", "progressive"}``, or ``None`` if ``b``
        is not a JPEG with a frame header.
    """
    if b[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(b):
        if b[i] != 0xFF:
            return None
        marker = b[i + 1]
        length = int.from_bytes(b[i + 2 : i + 4], "big")
        # SOF0-SOF15 except DHT (C4), JPG (C8) and DAC (CC)
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h = int.from_bytes(b[i + 5 : i + 7], "big")
            w = int.from_bytes(b[i + 7 : i + 9], "big")
            ncomp = b[i + 9]
            if ncomp == 1:
                chroma = "gray"
            else:
                y = b[i + 11]
                chroma = _CHROMA.get((y >> 4, y & 15), f"{y >> 4}x{y & 15}")
            return {
                "height": h,
                "width": w,
                "chroma": chroma,
                "progressive": marker in (0xC2, 0xC6, 0xCA, 0xCE),
            }
        if marker == 0xDA:
            return None
        i += 2 + length
    return None


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def _timed(fn: Callable[[], object], n: int) -> float:
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def _synthetic_batch(batch: int, window: int, hw: tuple[int, int]) -> dict:
    from torchvision.io import encode_jpeg

    g = torch.Generator().manual_seed(0)
    base = (torch.rand(3, *hw, generator=g) * 255).to(torch.uint8)
    frames = [
        encode_jpeg(base.roll(i, dims=2), quality=90).numpy().tobytes()
        for i in range(32)
    ]
    return {
        "video_bytes": [
            [frames[(b + t) % 32] for b in range(batch)] for t in range(window)
        ]
    }


def _gpu_ms(fn: Callable[[], object]) -> float:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type.name == "CUDA"]
    return sum(e.device_time for e in kernels) / 1e3


def _bench(
    r: Report,
    ds,
    batch: int,
    window: int,
    workers: int,
    device: int,
) -> None:
    from torch.utils.data import DataLoader, RandomSampler

    from .gpu_transforms import GPUCompose, GPUDecodeJPEG, GPUMultiCrop, GPUPrefetcher

    if ds is not None:
        from .video import LanceVideoSegments

        raw = LanceVideoSegments.collate(
            [ds[i] for i in torch.randperm(len(ds))[:batch].tolist()]
        )
        src = "store"
    else:
        raw = _synthetic_batch(batch, window, (256, 340))
        src = "synthetic 256x340 frames"
    dev = torch.device("cuda", device)
    dec = GPUDecodeJPEG(source="video_bytes", target="video", device=dev)
    crop = GPUMultiCrop(
        LEVJEPA_GROUPS,
        source="video",
        size_key="video_size",
        output="groups",
        record_params=False,
    ).to(dev)
    decoded = dec(dict(raw))
    t_dec = _timed(lambda: dec(dict(raw)), 5)
    t_crop = _timed(lambda: crop(dict(decoded)), 5)
    r.throughput["GPU decode (GPUDecodeJPEG)"] = batch / t_dec
    r.throughput["GPU multi-crop (LeVJEPA recipe)"] = batch / t_crop
    sm_ms = _gpu_ms(lambda: crop(dec(dict(raw))))
    r.add(
        "bench input",
        OK,
        f"{batch} clips x {window} frames ({src}); GPU kernel time "
        f"{sm_ms:.0f} ms/batch competes with training",
    )
    if ds is None or workers <= 0:
        r.bottleneck = min(r.throughput, key=r.throughput.get)
        return

    # Workers fill their queues before the clock starts; time a window long
    # enough that the burst does not count. Sampling with replacement keeps
    # small stores from hitting epoch restarts (each stalls every worker).
    warm, n = 2 * workers + 2, max(8, 2 * workers)
    dl = DataLoader(
        ds,
        batch_size=batch,
        sampler=RandomSampler(ds, replacement=True, num_samples=(warm + n + 1) * batch),
        num_workers=workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=LanceVideoSegments.worker_init,
        collate_fn=LanceVideoSegments.collate,
        multiprocessing_context="spawn",
    )
    prefetched = GPUPrefetcher(
        dl, GPUCompose([dec, crop], compile=False), device=dev, depth=2
    )

    for name, loader in (
        (f"CPU byte loader ({workers} workers)", dl),
        ("full pipeline (GPUPrefetcher)", prefetched),
    ):
        it = iter(loader)
        for _ in range(warm):
            next(it)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            next(it)
        torch.cuda.synchronize()
        r.throughput[name] = n * batch / (time.perf_counter() - t0)
        del it
    stages = {k: v for k, v in r.throughput.items() if not k.startswith("full")}
    r.bottleneck = min(stages, key=stages.get)


def run_doctor(
    store: Optional[str] = None,
    bench: bool = False,
    workers: Optional[int] = None,
    batch: int = 96,
    window: int = 16,
    device: int = 0,
) -> Report:
    """Check this machine for the GPU video pipeline and optionally benchmark it.

    Args:
        store: Optional Lance video store to validate (and benchmark from).
        bench: Measure decode, multi-crop and (with ``store``) CPU loader
            throughput.
        workers: DataLoader workers to check / benchmark; default derived
            from the usable CPUs.
        batch: Clips per batch for the benchmark.
        window: Frames per clip.
        device: CUDA device index.

    Returns:
        The :class:`Report`.
    """
    r = Report()
    gpu_ok = _check_torch_gpu(r, device)
    hw_ok = False
    if gpu_ok:
        ops_ok = _check_cuda_ops(r, device)
        hw_ok = _check_hw_decoder(r, device)
    else:
        ops_ok = False
    _check_packages(r)
    _check_host(r, workers)
    ds = _check_store(r, store, hw_ok, device) if store else None
    if bench:
        if gpu_ok and ops_ok:
            w = workers if workers is not None else max(1, int(usable_cpus()) - 2)
            _bench(r, ds, batch, window, w, device)
        else:
            r.add("bench", SKIP, "GPU checks failed")
    return r
