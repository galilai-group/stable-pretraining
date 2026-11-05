"""Token loader for LeVJEPA: hands the training step the kept patches of every view instead of decoded clips.

v2's pipeline decodes whole frames, squashes them to 224 x 224, crops 1 global + 4 local views, jitters the locals,
and then its PatchEmbed keeps ~5% of the patches. Here the crops, flips, jitter factors and kept tokens are drawn
first (in the CPU workers, with v2's distributions: sampling.py), so only the JPEG MCUs under kept patches are
decoded:

  workers (CPU, xW)   sample a batch's clips and view parameters; map every kept patch back to the stored frame;
                      splice the needed MCUs of each clip, straight out of the store's memory-mapped files, into one
                      packed JPEG per clip (splice.py); write all of it into a shared-memory slot
  servers (GPU, xS)   MPS clients: nvJPEG-decode the packed JPEGs, scatter the MCUs back into a frame canvas, build
                      every kept patch (squash, roi_align, flip, jitter, normalize: views.py, kernels/) into a ring
                      of patch slots on the GPU shared with the trainer over CUDA IPC

    loader = SharedTokenLoader("/vol/frames.lance", batch=64, index="shared")  # workers: from the node
    gp, gi, lp, li = next(loader)   # (64, 160, 768), (64, 160), (256, 32, 768), (256, 32) for v2's recipe
"""

import math
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from stable_datasets.framestore.index import FrameIndex
from stable_datasets.framestore.resources import (
    Budget,
    Resources,
    Stage,
    gpu_free,
    process_bytes,
)
from stable_datasets.framestore.store import FrameStore

from .recipe import V2
from .splice import MCU_COLS, PACK_COLS, make_batch
from .views import TokenViews, ring_slot

# The loader's default share of the node: it runs beside the training process it feeds, which needs the rest.
LOADER_RESOURCES = Resources(cpu=1.0, memory=0.5, gpu_memory=0.5)
# Per process, for the memory budget: the first worker and server are measured once they have produced a batch
# (private memory, x MEASURE_MARGIN); these stand in where /proc is missing. A server's GPU memory is at least its
# CUDA context and nvJPEG (SERVER_GPU_BYTES) plus twice its canvas at the store's largest frame, which the first
# batch may not have reached.
WORKER_BYTES, SERVER_BYTES, SERVER_GPU_BYTES, MEASURE_MARGIN = 0.8e9, 2e9, 0.6e9, 1.25
PIX_BYTES_PER_CLIP = (
    6 << 20
)  # a CPU-decoded packed clip: ~3 MB in v2's recipe on 384p frames; room for twice that


def wait_for(q, procs, what, timeout=900):
    """The next item of q, raising if one of procs has died first."""
    t0 = time.time()
    while True:
        try:
            return q.get(timeout=5)
        except queue.Empty:
            dead = [p.exitcode for p in procs if not p.is_alive()]
            if dead or time.time() - t0 > timeout:
                raise RuntimeError(
                    f"{what} exited ({dead})"
                    if dead
                    else f"{what}: nothing after {timeout} s"
                )


def slot_spec(B, r, jpeg_bytes=96 << 20, mcus=1 << 20):
    T, V, KG, KL = r.frames, r.num_local * B, r.keep_global, r.keep_local
    return {
        "jpeg": ((jpeg_bytes,), torch.uint8),
        "mcu": ((mcus, MCU_COLS), torch.int32),
        "joff": ((B * T + 1,), torch.int64),
        "moff": ((B + 1,), torch.int64),
        "prow": ((B,), torch.int64),
        "size": ((B, 2), torch.int32),
        "mdim": ((B, 2), torch.int32),
        "gbox": ((B, 4), torch.float32),
        "gflip": ((B,), torch.bool),
        "gidx": ((B, T * KG), torch.int16),
        "lbox": ((V, 4), torch.float32),
        "lflip": ((V,), torch.bool),
        "lidx": ((V, T * KL), torch.int16),
        "ljit": ((V, 4), torch.float32),
        "gorder": ((B * T * KG,), torch.int32),
        "gcount": ((9,), torch.int64),
        "lorder": ((V * T * KL,), torch.int32),
        "lcount": ((9,), torch.int64),
        "pix": (
            (1,),
            torch.int32,
        ),  # the pixel slot a CPU decoder decoded this batch into, or -1
        "poff": (
            (B + 1,),
            torch.int64,
        ),  # each clip's decoded packed image in it: byte offsets, (h, w)
        "phw": ((B, 2), torch.int32),
    }


def slot_bytes(B, r):
    return sum(
        math.prod(s) * torch.empty(0, dtype=dt).element_size()
        for s, dt in slot_spec(B, r).values()
    )


def shared_empty(shape, dtype):
    """A tensor in untouched shared memory.

    A tensor in shared memory whose pages are not touched (share_memory_() would copy into it), so slots never
    handed out cost nothing.
    """
    st = torch.UntypedStorage._new_shared(
        max(1, math.prod(shape) * torch.empty(0, dtype=dtype).element_size())
    )
    return torch.empty(0, dtype=dtype).set_(st, 0, shape)


def slot_tensors(n, B, r):
    return {k: shared_empty((n,) + s, dt) for k, (s, dt) in slot_spec(B, r).items()}


def worker(
    store_path, B, r, stride, slots, free, ready, seed, decode, index, read, cpu=None
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    store = FrameStore(store_path, index=index, read=read)
    rng = np.random.default_rng(seed)
    view = {k: v.numpy() for k, v in slots.items()}
    drop = os.environ.get("FRAMESTORE_DROP_PAGES") == "1"
    while (s := free.get()) is not None:
        p = make_batch(
            store, rng, B, view["jpeg"][s], view["mcu"][s], r, stride, decode
        )
        for k, v in p.items():
            if k in view:
                view[k][s][: len(v)] = v
        view["pix"][s][0] = -1
        if cpu is not None and decode == "partial" and backlog(ready) >= cpu[2]:
            cpu_decode(view, s, p, B, *cpu[:2])
        if drop:
            store.drop_pages()
        ready.put(s)


def backlog(q):
    """Batches waiting for the GPU servers (0 where the platform cannot count a queue)."""
    try:
        return q.qsize()
    except NotImplementedError:
        return 0


def cpu_decode(view, s, p, B, pix, pix_free):
    """Decode slot s's packed JPEGs on this CPU into a free pixel slot.

    libjpeg-turbo, chroma upsampled per MCU as nvJPEG does; the GPU server then only uploads the images. Workers
    decode only while the GPU servers have a backlog (a CPU decodes a batch ~10x slower than it splices one, so
    decoding while the GPU keeps up would starve it of spliced batches), and only into a free pixel slot the images
    fit: CPU decoding only ever adds capacity.
    """
    import simplejpeg

    try:
        q = pix_free.get_nowait()
    except queue.Empty:
        return
    buf, o = pix[q], 0
    poff, phw = view["poff"][s], view["phw"][s]
    poff[0] = 0
    for b in range(B):
        mw, mh = (int(x) for x in p["mdim"][b])
        h, w = int(p["prow"][b]) * mh, PACK_COLS * mw
        n = h * w * 3
        if o + n > len(buf):
            pix_free.put(q)
            return
        simplejpeg.decode_jpeg(
            view["jpeg"][s][p["joff"][b] : p["joff"][b + 1]],
            colorspace="RGB",
            fastdct=False,
            fastupsample=True,
            buffer=buf[o : o + n],
        )
        o += n
        poff[b + 1], phw[b] = o, (h, w)
    view["pix"][s][0] = q


class TokenLoader:
    """CPU workers filling shared-memory slots with spliced batches.

    Iterator of slots ready on the host (self.slots[k][slot]); TokenViews turns one into patches on the GPU.
    Clips are `recipe.frames` rows, every `stride`-th, of one episode of `store` (a converted frame store; `index`,
    `read`: stable_datasets.framestore.FrameStore's). Workers (default: the CPUs the budget allows, 2 kept free, less one for the
    trainer and one per GPU-stage server) and
    slots are what the memory budget allows (stable_datasets.framestore.resources; logged when fewer than asked): one worker
    starts first and is measured once it has filled a batch, then the rest start. plan=False leaves the rest to the caller (SharedTokenLoader sizes workers, slots and its servers
    together).
    """

    def __init__(
        self,
        store,
        batch,
        workers=None,
        slots=None,
        seed=0,
        decode="partial",
        index=False,
        read=None,
        recipe=V2,
        stride=2,
        resources=None,
        budget=None,
        plan=True,
        servers=0,
        cpu_decoders=0,
        pix_slots=None,
    ):
        self.ctx = mp.get_context("spawn")
        self.budget = budget or Budget("loader", resources or LOADER_RESOURCES)
        workers = workers or max(1, self.budget.cpus - 1 - servers)
        self.B, self.r, self.want = batch, recipe, (workers, slots or 2 * workers)
        self.args = (store, recipe, stride, decode, index, read, seed)
        self.servers = servers
        # CPU decoders: the first cpu_decoders workers also decode their batches into pixel slots (shared memory; a
        # packed clip decodes to ~3 MB)
        self.cpu_decoders = (
            workers // 2 if cpu_decoders == "auto" else min(cpu_decoders, workers)
        )
        self.pix_bytes = PIX_BYTES_PER_CLIP * batch
        n_pix = (pix_slots or 2 * self.cpu_decoders) if self.cpu_decoders else 0
        self.pix = (
            shared_empty((max(n_pix, 1), self.pix_bytes), torch.uint8)
            if n_pix
            else None
        )
        self.pix_free, self.n_pix = (
            self.ctx.Queue(),
            n_pix,
        )  # handed out by add_pix once the budget allows
        self.slots = slot_tensors(
            self.want[1], batch, recipe
        )  # shared memory is touched only by the slots handed out
        self.free, self.ready = self.ctx.Queue(), self.ctx.Queue()
        self.procs, self.n = [], 0
        self.add(workers=1, slots=2)
        if plan:
            nw, ns, *npix = self.budget.allocate(self.stages())
            self.add(nw - 1, ns - 2)
            self.add_pix(*npix)

    def add_pix(self, n=0):
        """Hand out n pixel slots to the CPU decoders."""
        for q in range(n):
            self.pix_free.put(q)

    def add(self, workers=0, slots=0):
        """Start more workers and hand out more slots."""
        store, recipe, stride, decode, index, read, seed = self.args
        # CPU decoders decode while at least one batch per GPU server waits
        pix = (
            (self.pix.numpy(), self.pix_free, max(1, self.servers))
            if self.pix is not None
            else None
        )
        for s in range(self.n, self.n + slots):
            self.free.put(s)
        self.n += slots
        for _ in range(workers):
            p = self.ctx.Process(
                target=worker,
                args=(
                    store,
                    self.B,
                    recipe,
                    stride,
                    self.slots,
                    self.free,
                    self.ready,
                    seed * 1000 + len(self.procs),
                    decode,
                    index,
                    read,
                    pix if len(self.procs) < self.cpu_decoders else None,
                ),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

    def stages(self):
        """The loader's stages, for Budget.allocate.

        Workers (measured: the first one's private memory once it has filled a batch) and slots, for
        Budget.allocate.
        """
        s = wait_for(self.ready, self.procs, "loader worker")
        self.ready.put(s)
        b = process_bytes(self.procs[0].pid)
        per = WORKER_BYTES if b is None else MEASURE_MARGIN * b
        return [
            Stage("CPU worker processes", self.want[0], per),
            Stage(
                "batch slots (shared memory, page-locked by the GPU stage)",
                self.want[1],
                slot_bytes(self.B, self.r),
                least=2,
                shm=True,
            ),
        ] + (
            [
                Stage(
                    "CPU-decoded pixel slots (shared memory)",
                    self.n_pix,
                    self.pix_bytes,
                    least=0,
                    shm=True,
                )
            ]
            if self.pix is not None
            else []
        )

    def __iter__(self):
        return self

    def __next__(self):
        return self.ready.get()

    def release(self, slot):
        self.free.put(slot)

    def close(self):
        for _ in self.procs:
            self.free.put(None)
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


# ---------------------------------------------------------------------------------------------------------------
# GPU: packed JPEGs -> patches
# ---------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------
# GPU stage in separate processes: nvJPEG cannot overlap a busy training step from inside the training process
# (its host side waits on per-image events), so TokenViews runs in server processes (MPS clients) that write
# finished patches into a CUDA-IPC ring shared with the trainer, as stable_datasets.framestore.pixel_loader's decode servers do.
# ---------------------------------------------------------------------------------------------------------------
def token_server(
    slots,
    host_free,
    host_ready,
    ring,
    ring_free,
    ring_ready,
    B,
    r,
    stop,
    decode,
    fused,
    decoder,
    pix=(None, None),
    stats=None,
):
    try:
        _token_server(
            slots,
            host_free,
            host_ready,
            ring,
            ring_free,
            ring_ready,
            B,
            r,
            stop,
            decode,
            fused,
            decoder,
            pix,
            stats,
        )
    except torch.OutOfMemoryError:
        print("token server: out of GPU memory", file=sys.stderr, flush=True)
        os._exit(OOM_EXIT)


OOM_EXIT = 3


def _token_server(
    slots,
    host_free,
    host_ready,
    ring,
    ring_free,
    ring_ready,
    B,
    r,
    stop,
    decode,
    fused,
    decoder,
    pix=(None, None),
    stats=None,
):
    import warnings

    warnings.filterwarnings("ignore")
    torch.cuda.set_device(0)

    def lock(t):  # page-lock, so copies to the GPU are asynchronous
        err = torch.cuda.cudart().cudaHostRegister(
            t.data_ptr(), t.numel() * t.element_size(), 0
        )
        assert int(err) == 0, err

    # the large per-slot buffers (page-aligned, MBs each) are locked as each slot is first used, so slots never handed
    # out stay untouched; the small ones (several slots per page: CUDA refuses overlapping ranges) whole, at once
    big = [t for t in slots.values() if t[0].numel() * t.element_size() >= 1 << 20]
    for t in slots.values():
        if not any(t is b for b in big):
            lock(t)
    locked = set()
    views, stream = (
        TokenViews(B, r, decode=decode, fused=fused, decoder=decoder),
        torch.cuda.Stream(),
    )
    while not stop.is_set():
        try:
            s = host_ready.get(timeout=0.5)
        except queue.Empty:
            continue
        if s not in locked:
            for t in big:
                lock(t[s])
            locked.add(s)
        v = ring_free.get()
        q = int(slots["pix"][s][0])
        with torch.cuda.stream(stream):
            imgs = None
            if q >= 0:  # decoded by a CPU worker: upload
                if ("pix", q) not in locked:
                    lock(pix[0][q])
                    locked.add(("pix", q))
                off, hw = slots["poff"][s].tolist(), slots["phw"][s].tolist()
                imgs = [
                    pix[0][q][off[b] : off[b + 1]]
                    .view(h, w, 3)
                    .to(views.dev, non_blocking=True)
                    .permute(2, 0, 1)
                    .contiguous()
                    for b, (h, w) in enumerate(hw[:B])
                ]
            if fused:  # straight into the ring slot
                views.fused_call(slots, s, out=tuple(x[v] for x in ring), imgs=imgs)
            else:
                for dst, src in zip(ring, views(slots, s, imgs=imgs)):
                    dst[v].copy_(src)
        stream.synchronize()
        if q >= 0:
            pix[1].put(q)
        if (
            stats is not None
        ):  # batches this server decoded itself, and batches a CPU worker decoded
            stats[int(q >= 0)] += 1
        host_free.put(s)
        ring_ready.put(v)
    os._exit(0)


def server_decoders(decoders, servers, decoder):
    """The GPU-stage servers' decoders, and the number of CPU decoders.

    From a spec ("nvjpeg:2,cpu:4"; a kind with no count: nvjpeg 2, cpu "auto", half the workers), else
    `servers` servers of `decoder` and no CPU decoding.
    """
    if decoders is None:
        return [decoder] * servers, 0
    from stable_datasets.framestore.decoders import decoders as parse

    d = parse(decoders)
    kinds = ["nvjpeg"] * (2 if d.get("nvjpeg", 0) is None else d.get("nvjpeg", 0))
    if not kinds:
        raise ValueError(
            f"{decoders!r}: the GPU stage needs a server (nvjpeg), even if the CPU decodes"
        )
    return kinds, "auto" if d.get("cpu", 0) is None else d.get("cpu", 0)


class SharedTokenLoader:
    """Every view's kept patches on the GPU, from CPU workers and GPU-stage servers.

    next(loader) -> (global patches, global idx, local patches, local idx) on the GPU, valid until the next call,
    from a ring written by `servers` GPU-stage processes fed by `workers` CPU workers (TokenLoader: store, recipe,
    stride, index, read; workers default to the CPUs the budget allows (2 kept free), less one for the trainer and
    one per server). Servers, workers and slots are cut to what the host and GPU memory budget allows
    (stable_datasets.framestore.resources; said when they are). fused=True: the GPU stage's Triton kernels (kernels/fused.py).
    decoder: the GPU stage's JPEG decoder, "nvjpeg" (stable_datasets.framestore.decoders; see TokenViews).
    """

    def __init__(
        self,
        store,
        batch,
        workers=None,
        servers=2,
        ring=None,
        seed=0,
        decode="partial",
        index=False,
        read=None,
        slots=None,
        fused=True,
        decoder="nvjpeg",
        recipe=V2,
        stride=2,
        resources=None,
        budget=None,
        decoders=None,
    ):
        ctx = mp.get_context("spawn")
        r = recipe
        kinds, cpu = server_decoders(decoders, servers, decoder)
        if (
            index == "shared"
        ):  # build the node's index here, before any worker needs it (and before the budget
            idx = FrameIndex.shared(store)  # reads the free shared memory)
            h, w = int(idx.h.max(initial=0)), int(idx.w.max(initial=0))
        else:
            import lance

            t = lance.dataset(store).to_table(columns=["h", "w"])
            h, w = (int(np.max(t.column(k).to_numpy(), initial=0)) for k in ("h", "w"))
        budget = budget or Budget("loader", resources or LOADER_RESOURCES)
        with budget.measure("patch ring and CUDA context", gpu=True):
            ring = ring or max(
                4, 2 * len(kinds)
            )  # two patch slots per server, so none waits on another
            self.ring = ring_slot(batch, r, n=(ring,))
            torch.cuda.synchronize()
        self.host = TokenLoader(
            store,
            batch,
            workers=workers,
            seed=seed,
            decode=decode,
            index=index,
            read=read,
            slots=slots,
            recipe=r,
            stride=stride,
            budget=budget,
            plan=False,
            servers=len(kinds),
            cpu_decoders=cpu,
        )
        self.kinds = kinds
        host_stages = self.host.stages()  # measures the first worker
        self.free, self.ready, self.stop = ctx.Queue(), ctx.Queue(), ctx.Event()
        for v in range(ring):
            self.free.put(v)
        args = (
            self.host.slots,
            self.host.free,
            self.host.ready,
            self.ring,
            self.free,
            self.ready,
            batch,
            r,
            self.stop,
            decode,
            fused,
        )
        pix = (self.host.pix, self.host.pix_free)
        self.stats = torch.zeros(len(kinds), 2, dtype=torch.int64).share_memory_()
        self.servers = []

        def start(k):  # server i decodes with kinds[i]
            for _ in range(k):
                kind = kinds[len(self.servers)]
                self.servers.append(
                    ctx.Process(
                        target=token_server,
                        args=(*args, kind, pix, self.stats[len(self.servers)]),
                    )
                )
                self.servers[-1].start()

        g0 = gpu_free()
        start(1)  # measured once it has produced a batch
        v = wait_for(self.ready, self.servers, "token server")
        self.ready.put(v)
        hb, g1 = process_bytes(self.servers[0].pid), gpu_free()
        canvas = (
            batch * r.frames * 3 * -(-h // 16) * 16 * -(-w // 16) * 16
        )  # bytes: the frames at their largest
        per_host = SERVER_BYTES if hb is None else MEASURE_MARGIN * hb
        per_gpu = max(MEASURE_MARGIN * (g0 - g1), SERVER_GPU_BYTES + 2 * canvas)
        nw, ns, *npix, nh = budget.allocate(
            host_stages
            + [Stage("GPU server processes (host memory)", len(kinds), per_host)]
        )
        (ng,) = budget.allocate(
            [Stage("GPU server processes (GPU memory)", len(kinds), per_gpu, gpu=True)]
        )
        n = min(nh, ng)
        budget.release((nh - n) * per_host)
        budget.release((ng - n) * per_gpu, gpu=True)
        self.host.add(nw - 1, ns - 2)
        self.host.add_pix(*npix)
        start(n - 1)
        self.prev = None
        self._release = queue.Queue()
        threading.Thread(target=self._releaser, daemon=True).start()

    def _releaser(self):
        while True:
            ev, v = self._release.get()
            ev.synchronize()
            self.free.put(v)

    def __iter__(self):
        return self

    def __next__(self):
        if (
            self.prev is not None
        ):  # the previous step has queued its reads of that slot; free it once they ran
            ev = torch.cuda.Event()
            ev.record()
            self._release.put((ev, self.prev))
        while True:
            try:
                v = self.ready.get(timeout=5)
                break
            except queue.Empty:
                dead = [p.exitcode for p in self.servers if not p.is_alive()]
                if dead:
                    raise RuntimeError(
                        f"token server exited ({dead})"
                        + (
                            "; out of GPU memory: run fewer servers"
                            if OOM_EXIT in dead
                            else ""
                        )
                    )
        self.prev = v
        return tuple(x[v] for x in self.ring)

    def close(self):
        self.stop.set()
        for p in self.servers:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
        self.host.close()
