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
    lower_priority,
    process_bytes,
)
from stable_datasets.framestore.store import FrameStore, store_format

from .kernels.order import TILES
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


# A loader restarts processes that die (each crew of them, over a run) up to RESTARTS times, then fails. When no batch
# comes for STALL seconds with every process alive, it restarts all of them with new queues (a process killed inside
# a queue operation can leave that queue unusable for the rest: its lock held, or a message cut in two), up to RESETS
# times, and fails if it stalls again before a batch. It checks its processes every CHECK seconds.
RESTARTS, RESETS, STALL, CHECK = 10, 3, 120, 1.0


class Stalled(RuntimeError):
    """No batch for STALL seconds with every loader process alive."""


def take(q, parent, stop=None, wait=1.0):
    """q.get(), or None once `parent` (the loader's process) has gone, or `stop` is set: no orphan outlives it.

    It waits on the queue's pipe and takes items without waiting: a get that waits holds the queue's lock, shared by
    every process reading it, and a process killed there (say, by the kernel's OOM killer) would leave it held.
    """
    while True:
        try:
            return q.get_nowait()
        except queue.Empty:
            if os.getppid() != parent or (stop is not None and stop.is_set()):
                return None
            q._reader.poll(wait)


class Crew:
    """A loader's processes of one kind, started again when one dies.

    Each process keeps its own shared row: the queue items it has taken and not passed on (-1: none), and last, the
    batches it has passed on. It clears an item before passing it on, so a process that dies between the two leaks
    that item rather than handing it out twice. reclaim(row) hands a dead process's items back, then the process is
    started again (spawn(i, row, generation) -> an unstarted Process). One that dies again without passing a batch on
    since its restart (a crash loop), or a death past `restarts`, raises instead.
    """

    def __init__(self, what, spawn, reclaim, restarts=RESTARTS, hints=None):
        self.what, self.spawn, self.reclaim, self.left = what, spawn, reclaim, restarts
        self.hints = hints or {}
        self.procs, self.rows, self.gens = [], [], []

    def start(self, n=1):
        for _ in range(n):
            self.procs.append(None), self.rows.append(None), self.gens.append(-1)
            self._start(len(self.procs) - 1)

    def _start(self, i):
        row = torch.tensor([-1, -1, 0], dtype=torch.int64).share_memory_()
        self.gens[i] += 1
        p = self.spawn(i, row, self.gens[i])
        p.start()
        self.procs[i], self.rows[i] = p, row

    def check(self):
        for i, p in enumerate(self.procs):
            if p.is_alive():
                continue
            row = self.rows[i].tolist()
            self.reclaim(row)
            self.left -= 1
            loop = self.gens[i] > 0 and row[-1] == 0
            msg = f"{self.what} {i} exited ({p.exitcode})" + (
                f"; {self.hints[p.exitcode]}" if p.exitcode in self.hints else ""
            )
            if loop or self.left < 0:
                raise RuntimeError(
                    msg
                    + (
                        "; again before passing a batch on since its restart"
                        if loop
                        else "; no restarts left"
                    )
                )
            print(
                f"{msg}; restarted ({self.left} restarts left)",
                file=sys.stderr,
                flush=True,
            )
            self._start(i)

    def stop(self, deadline):
        for p in self.procs:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        for p in self.procs:
            if p.is_alive():
                p.kill()

    def kill(self):
        for p in self.procs:
            p.kill()
        for p in self.procs:
            p.join()

    def restart(self):
        """Start every (killed) process again."""
        for i in range(len(self.procs)):
            self._start(i)


def supervised(q, crews, last, stall=None):
    """The next item of q, checking `crews` every CHECK seconds.

    last: [time of the last check]. Raises Stalled after `stall` (default STALL) seconds without an item.
    """
    stall, t0 = stall or STALL, time.monotonic()
    while True:
        if time.monotonic() - last[0] > CHECK:
            for c in crews:
                c.check()
            last[0] = time.monotonic()
        try:
            return q.get(timeout=CHECK)
        except queue.Empty:
            if time.monotonic() - t0 > stall:
                raise Stalled(
                    f"no batch for {stall} s with every loader process alive"
                ) from None


def unwedged(next_batch, reset, resets):
    """next_batch(), resetting the loader (reset()) when it stalls.

    resets: [resets left, whether one came since the last batch]. Raises when out of resets, or stalled again since
    the last reset.
    """
    while True:
        try:
            x = next_batch()
            resets[1] = False
            return x
        except Stalled as e:
            if resets[0] <= 0 or resets[1]:
                raise
            resets[0] -= 1
            resets[1] = True
            print(
                f"{e}: restarting every loader process with new queues ({resets[0]} resets left)",
                file=sys.stderr,
                flush=True,
            )
            reset()


def discard(*qs):
    """Queues replaced after a stall: nothing this process put on them is waited for at exit."""
    for q in qs:
        q.cancel_join_thread()
        q.close()


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
        "gcount": ((len(TILES) ** 2,), torch.int64),
        "lorder": ((V * T * KL,), torch.int32),
        "lcount": ((len(TILES) ** 2,), torch.int64),
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
    store_path,
    B,
    r,
    stride,
    slots,
    free,
    ready,
    seed,
    decode,
    index,
    read,
    cpu=None,
    nice=10,
    held=None,
    parent=None,
):
    """A loader worker.

    held: this worker's Crew row (the slot it fills, the pixel slot it decodes into, batches passed on). parent: the
    loader's process id (passed in: one that dies while this one starts would leave os.getppid() at init's).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    lower_priority(
        nice
    )  # the consumer, and the GPU servers, win any contention for the CPU
    torch.set_num_threads(1)
    store = FrameStore(store_path, index=index, read=read)
    rng = np.random.default_rng(seed)
    view = {k: v.numpy() for k, v in slots.items()}
    if cpu is not None:
        pix, pix_free, least, stalled = cpu[0].numpy(), cpu[1], cpu[2], cpu[3].numpy()
    drop = os.environ.get("FRAMESTORE_DROP_PAGES") == "1"
    while (s := take(free, parent)) is not None:
        held[0] = s
        p = make_batch(
            store, rng, B, view["jpeg"][s], view["mcu"][s], r, stride, decode
        )
        for k, v in p.items():
            if k in view:
                view[k][s][: len(v)] = v
        view["pix"][s][0] = -1
        # decode here only while the GPU servers are the bottleneck: batches wait for them, and they are not
        # themselves waiting for the consumer (stalled; decoding then would only take CPU from the consumer)
        if (
            cpu is not None
            and decode == "partial"
            and backlog(ready) >= least
            and not (least and stalled[0])
        ):
            cpu_decode(view, s, p, B, pix, pix_free, held, parent, wait=least == 0)
        if drop:
            store.drop_pages()
        held[0], held[1] = (
            -1,
            -1,
        )  # the pixel slot, if any, goes with the batch slot (view["pix"][s])
        held[2] += 1
        ready.put(s)


def backlog(q):
    """Batches waiting for the GPU servers (0 where the platform cannot count a queue)."""
    try:
        return q.qsize()
    except NotImplementedError:
        return 0


def cpu_decode(view, s, p, B, pix, pix_free, held, parent, wait=False):
    """Decode slot s's packed JPEGs on this CPU into a free pixel slot.

    libjpeg-turbo, chroma upsampled per MCU as nvJPEG does; the GPU server then only uploads the images. Workers
    decode only while the GPU servers have a backlog (a CPU decodes a batch ~10x slower than it splices one, so
    decoding while the GPU keeps up would starve it of spliced batches), and only into a free pixel slot the images
    fit: CPU decoding only ever adds capacity. wait=True (CPU-only decoding): every batch, waiting for a pixel slot.
    """
    import simplejpeg

    try:
        q = take(pix_free, parent) if wait else pix_free.get_nowait()
    except queue.Empty:
        return
    if q is None:
        return
    held[1] = q
    buf, o = pix[q], 0
    poff, phw = view["poff"][s], view["phw"][s]
    poff[0] = 0
    for b in range(B):
        mw, mh = (int(x) for x in p["mdim"][b])
        h, w = int(p["prow"][b]) * mh, PACK_COLS * mw
        n = h * w * 3
        if o + n > len(buf):
            held[1] = -1
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
    `read`: stable_datasets.framestore.FrameStore's). Workers (default: the CPUs the budget allows, 2 kept free; they run below the
    trainer and the GPU-stage servers in priority, so no core is held back for those) and
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
        cpu_only=False,
        nice=10,
    ):
        if decode == "partial":  # here rather than in each worker, which would only die
            import lance

            if store_format(lance.dataset(store).schema) is None:
                raise ValueError(
                    f"{store}: not a converted frame store (no seg_len); run python -m stable_datasets.framestore.convert on it"
                )
        self.ctx = mp.get_context("spawn")
        self.budget = budget or Budget("loader", resources or LOADER_RESOURCES)
        # every CPU the budget allows: workers are niced, so the trainer and the servers win any contention, and a
        # core held back for them would sit idle whenever they do not need it
        workers = workers or self.budget.cpus
        self.B, self.r, self.want = batch, recipe, (workers, slots or 2 * workers)
        self.args = (store, recipe, stride, decode, index, read, seed)
        self.servers = servers
        # CPU decoders: the first cpu_decoders workers also decode their batches into pixel slots (shared memory; a
        # packed clip decodes to ~3 MB)
        self.cpu_only, self.nice = cpu_only, nice
        # set by the GPU servers while they wait for the consumer (a free ring slot): CPU decoding is then no help
        self.stalled = torch.zeros(1, dtype=torch.int32).share_memory_()
        self.cpu_decoders = (
            workers
            if cpu_only
            else workers // 2
            if cpu_decoders == "auto"
            else min(cpu_decoders, workers)
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
        self.slots["pix"][
            :, 0
        ] = -1  # no slot holds a pixel slot (Crew reclaims read it)
        self.free, self.ready = self.ctx.Queue(), self.ctx.Queue()
        self.n, self.pix_given, self.last, self.resets = (
            0,
            0,
            [time.monotonic()],
            [RESETS, False],
        )
        self.workers = Crew("loader worker", self._spawn, self._reclaim)
        self.procs = self.workers.procs
        if (
            cpu_only and n_pix
        ):  # the first worker, measured, waits for a pixel slot for every batch
            self.add_pix(1)
        self.add(workers=1, slots=2)
        if plan:
            nw, ns, *npix = self.budget.allocate(self.stages())
            self.add(nw - 1, ns - 2)
            self.add_pix(*npix)

    def add_pix(self, n=0):
        """Hand out pixel slots to the CPU decoders, n in all."""
        for q in range(self.pix_given, n):
            self.pix_free.put(q)
        self.pix_given = max(self.pix_given, n)

    def add(self, workers=0, slots=0):
        """Start more workers and hand out more slots."""
        for s in range(self.n, self.n + slots):
            self.free.put(s)
        self.n += slots
        self.workers.start(workers)

    def _spawn(self, i, held, gen):
        store, recipe, stride, decode, index, read, seed = self.args
        # CPU decoders decode while at least one batch per GPU server waits (CPU-only: always)
        pix = (
            self.pix,  # the shared tensor: a numpy view would be pickled as a private copy
            self.pix_free,
            0 if self.cpu_only else max(1, self.servers),
            self.stalled,
        )
        return self.ctx.Process(
            target=worker,
            args=(
                store,
                self.B,
                recipe,
                stride,
                self.slots,
                self.free,
                self.ready,
                seed * 1000 + i if gen == 0 else [seed * 1000 + i, gen],
                decode,
                index,
                read,
                pix if self.pix is not None and i < self.cpu_decoders else None,
                self.nice,
                held,
                os.getpid(),
            ),
            daemon=True,
        )

    def _reclaim(self, row):
        """A dead worker's slot back to the free ones, and its pixel slot (held, or already in the slot's batch)."""
        s, q = row[0], row[1]
        for x in {q, int(self.slots["pix"][s][0]) if s >= 0 else -1} - {-1}:
            self.pix_free.put(x)
        if s >= 0:
            self.slots["pix"][s][0] = -1
            self.free.put(s)

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

    def renew(self):
        """New queues, every slot free (its workers killed first)."""
        discard(self.free, self.ready, self.pix_free)
        self.free, self.ready, self.pix_free = (self.ctx.Queue() for _ in range(3))
        self.slots["pix"][:, 0] = -1
        for s in range(self.n):
            self.free.put(s)
        for q in range(self.pix_given):
            self.pix_free.put(q)
        self.stalled[0] = 0
        self.last[0] = time.monotonic()

    def reset(self):
        self.workers.kill()
        self.renew()
        self.workers.restart()

    def __next__(self):
        return unwedged(
            lambda: supervised(self.ready, [self.workers], self.last),
            self.reset,
            self.resets,
        )

    def release(self, slot):
        self.free.put(slot)

    def close(self):
        if self.n < 0:
            return
        self.n = -1
        for _ in self.procs:
            self.free.put(None)
        self.workers.stop(time.monotonic() + 5)


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
    stalled=None,
    defer=0,
    held=None,
    parent=None,
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
            stalled,
            defer,
            held,
            parent,
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
    stalled=None,
    defer=0,
    held=None,
    parent=None,
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
    # held (this server's Crew row): the host slot and ring slot it works on, batches passed on
    while not stop.is_set():
        try:
            if defer:  # a reserve: take a batch only while `defer` or more wait (the first servers are behind)
                if backlog(host_ready) < defer:
                    if os.getppid() != parent:
                        break
                    time.sleep(0.0005)
                    continue
                s = host_ready.get_nowait()
            elif (s := take(host_ready, parent, stop, wait=0.5)) is None:
                break  # stopped, or the trainer has gone: no orphan holds the GPU
        except queue.Empty:
            if os.getppid() != parent:
                break
            continue
        held[0] = s
        if s not in locked:
            for t in big:
                lock(t[s])
            locked.add(s)
        try:
            v = ring_free.get_nowait()
            if stalled is not None:
                stalled[0] = 0
        except queue.Empty:  # every ring slot is with the consumer: it, not this server, is the bottleneck
            if stalled is not None:
                stalled[0] = 1
            v = take(ring_free, parent, stop)
            if v is None:
                host_ready.put(s)  # stopping: the batch back where it was
                break
        held[1] = v
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
        if (
            q >= 0
        ):  # the pixel slot leaves the batch slot before it is freed (a Crew reclaim reads it there)
            slots["pix"][s][0] = -1
            pix[1].put(q)
        if (
            stats is not None
        ):  # batches this server decoded itself, and batches a CPU worker decoded
            stats[int(q >= 0)] += 1
        held[0] = -1
        host_free.put(s)
        held[1] = -1
        held[2] += 1
        ring_ready.put(v)
    for q in (
        host_ready,
        host_free,
        ring_ready,
        pix[1],
    ):  # flush what this server put, then exit at once
        if q is not None:
            q.close()
            q.join_thread()
    os._exit(0)


def server_decoders(decoders, servers, decoder):
    """The GPU-stage servers' decoders, and the number of CPU decoders.

    From a spec ("nvjpeg:2,cpu:4", "triton:2"; a kind with no count: nvjpeg 2, triton 1, cpu "auto", half the workers;
    with both nvjpeg and triton, the triton servers are a reserve that decodes only while batches wait for the nvjpeg
    ones, so the JPEG engines take what they can and the CUDA cores only the rest), else
    `servers` servers of `decoder` and no CPU decoding. A spec with only cpu ("cpu"): every worker decodes every
    batch, and one server only uploads the images and builds the views (third value True).
    """
    if decoders is None:
        return [decoder] * servers, 0, False
    from stable_datasets.framestore.decoders import decoders as parse

    d = parse(decoders)
    kinds = ["nvjpeg"] * (2 if d.get("nvjpeg", 0) is None else d.get("nvjpeg", 0))
    kinds += ["triton"] * (1 if d.get("triton", 0) is None else d.get("triton", 0))
    if not kinds:
        if "cpu" not in d:
            raise ValueError(f"{decoders!r}: no decoder")
        return (
            [decoder],
            "auto",
            True,
        )  # one server: it only uploads and builds the views
    return kinds, "auto" if d.get("cpu", 0) is None else d.get("cpu", 0), False


class SharedTokenLoader:
    """Every view's kept patches on the GPU, from CPU workers and GPU-stage servers.

    next(loader) -> (global patches, global idx, local patches, local idx) on the GPU, valid until the next call,
    from a ring written by `servers` GPU-stage processes fed by `workers` CPU workers (TokenLoader: store, recipe,
    stride, index, read; workers default to the CPUs the budget allows, 2 kept free). With decoders="cpu" one
    server, which only uploads the images and builds the views. Servers, workers and slots are cut to what the host and GPU memory budget allows
    (stable_datasets.framestore.resources; said when they are). fused=True: the GPU stage's Triton kernels (kernels/fused.py).
    decoder: the GPU stage's JPEG decoder, "nvjpeg" (stable_datasets.framestore.decoders; see TokenViews). nice: the CPU workers'
    priority below the consumer's (and I/O priority, stable_datasets.framestore.resources.lower_priority); 0 to run level with it.

    A worker or server that dies is started again and the slots it held handed back (Crew; said on stderr); the loader
    raises instead on a crash loop, or past RESTARTS deaths of either kind. After STALL seconds without a batch it
    restarts every process with new queues (up to RESETS times; it raises if it stalls again before a batch). Its
    processes exit with the trainer, however it ends.
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
        nice=10,
    ):
        ctx = mp.get_context("spawn")
        r = recipe
        kinds, cpu, cpu_only = server_decoders(decoders, servers, decoder)
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
            cpu_only=cpu_only,
            nice=nice,
        )
        self.kinds = kinds
        host_stages = self.host.stages()  # measures the first worker
        self.free, self.ready, self.stop = ctx.Queue(), ctx.Queue(), ctx.Event()
        for v in range(ring):
            self.free.put(v)
        self.stats = torch.zeros(len(kinds), 2, dtype=torch.int64).share_memory_()
        self.ctx, self.resets = ctx, [RESETS, False]

        def spawn(
            i, held, gen
        ):  # server i decodes with kinds[i]; the queues as they are now (reset renews them)
            h = self.host
            return ctx.Process(
                target=token_server,
                args=(
                    h.slots,
                    h.free,
                    h.ready,
                    self.ring,
                    self.free,
                    self.ready,
                    batch,
                    r,
                    self.stop,
                    decode,
                    fused,
                    kinds[i],
                    (h.pix, h.pix_free),
                    self.stats[i],
                    self.host.stalled,
                    # triton servers beside nvjpeg ones decode only what the JPEG engines fall behind on
                    kinds.count("nvjpeg") if kinds[i] == "triton" else 0,
                    held,
                    os.getpid(),
                ),
                daemon=True,  # so an exiting trainer stops them rather than waiting for them
            )

        self.crew = Crew(
            "token server",
            spawn,
            self._reclaim,
            hints={OOM_EXIT: "out of GPU memory: run fewer servers"},
        )
        self.servers, start = self.crew.procs, self.crew.start
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
        self.prev, self.closed = None, False
        self._release = queue.Queue()
        threading.Thread(target=self._releaser, daemon=True).start()

    def _reclaim(self, row):
        """A dead server's host slot and ring slot back to the free ones.

        Its batch is dropped: it may be what killed the server.
        """
        s, v = row[0], row[1]
        if s >= 0:
            self.host._reclaim([s, -1, 0])
        if v >= 0:
            self.free.put(v)
        self.host.stalled[0] = 0

    def _releaser(self):
        while True:
            ev, v = self._release.get()
            ev.synchronize()
            self.free.put(v)
            self._release.task_done()

    def reset(self):
        """Every process killed and started again with new queues, every slot free.

        The consumer holds no slot while it waits: __next__ hands its last one to the releaser first.
        """
        self.crew.kill()
        self.host.workers.kill()
        self.host.renew()
        self._release.join()  # ring slots the consumer has finished with, into the old queue
        discard(self.free, self.ready)
        self.free, self.ready = self.ctx.Queue(), self.ctx.Queue()
        for v in range(len(self.ring[0])):
            self.free.put(v)
        self.crew.restart()
        self.host.workers.restart()

    def __iter__(self):
        return self

    def __next__(self):
        if (
            self.prev is not None
        ):  # the previous step has queued its reads of that slot; free it once they ran
            ev = torch.cuda.Event()
            ev.record()
            self._release.put((ev, self.prev))
        v = unwedged(
            lambda: supervised(
                self.ready, [self.crew, self.host.workers], self.host.last
            ),
            self.reset,
            self.resets,
        )
        self.prev = v
        return tuple(x[v] for x in self.ring)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.crew.stop(time.monotonic() + 10)
        self.host.close()
