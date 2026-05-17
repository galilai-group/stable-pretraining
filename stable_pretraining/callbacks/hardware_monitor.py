"""Background hardware-metrics monitor.

:class:`HardwareMonitor` samples CPU / RAM / disk / network / GPU usage in
a daemon thread and emits the latest readings through Lightning's normal
:meth:`pl_module.log_dict` path on every train batch. Sampling is
decoupled from training: the thread polls on a fixed wall-clock interval,
so per-step overhead is just a dict-merge (microseconds), even if the
sampler is reading expensive NVML counters.

Compatible with any Lightning logger (CSV, WandB, Trackio, SwanLab, …) —
metrics show up under the ``hardware/`` prefix by default.

Only rank-0 logs (otherwise multi-GPU runs would emit the same numbers
once per rank). For per-rank host stats on a multi-node job, run with
``rank_zero_only=False``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from .utils import log_header


_BYTES_PER_GB = 1024**3
_BYTES_PER_MB = 1024**2


class HardwareMonitor(Callback):
    """Background sampler for CPU / RAM / disk / network / GPU metrics.

    The sampler runs in a daemon thread at ``interval_seconds`` cadence and
    stores the latest reading in a lock-protected dict. On every train
    batch (and optionally validation batch), the most recent values are
    flushed to ``pl_module.log_dict``. This decouples sampling cost from
    training step cost — the forward/backward path only pays a single
    dict-copy + ``log_dict`` call.

    Parameters
    ----------
    interval_seconds
        Wall-clock sampling cadence. Default 10 s (matches WandB).
    prefix
        Key prefix for all emitted metrics. Default ``"hardware/"``.
    log_cpu, log_ram, log_disk, log_net, log_gpu
        Toggle individual metric families.
    log_per_gpu
        When True (default), emit ``gpu0_*``, ``gpu1_*``, … per device in
        addition to the aggregate ``gpu_avg_*`` / ``gpu_total_*``.
    rank_zero_only
        When True (default), only rank 0 logs. Set False to log per-rank
        (useful on multi-node where each node has its own hardware).
    log_on_validation
        Also flush latest sample at the end of each validation batch.
        Default False — validation batches are usually short enough that
        the train-batch hook covers it.
    """

    # Names of NVML counters that we expose. Listed here so a missing /
    # mis-permissioned counter on one card doesn't kill the whole loop.
    _NVML_OPTIONAL = ("temp_c", "power_w")

    def __init__(
        self,
        interval_seconds: float = 10.0,
        prefix: str = "hardware/",
        log_cpu: bool = True,
        log_ram: bool = True,
        log_disk: bool = True,
        log_net: bool = True,
        log_gpu: bool = True,
        log_per_gpu: bool = True,
        rank_zero_only: bool = True,
        log_on_validation: bool = False,
    ):
        super().__init__()
        if interval_seconds <= 0:
            raise ValueError(
                f"HardwareMonitor: interval_seconds must be > 0, got {interval_seconds}"
            )
        self.interval_seconds = float(interval_seconds)
        self.prefix = prefix
        self.log_cpu = log_cpu
        self.log_ram = log_ram
        self.log_disk = log_disk
        self.log_net = log_net
        self.log_gpu = log_gpu
        self.log_per_gpu = log_per_gpu
        self.rank_zero_only = rank_zero_only
        self.log_on_validation = log_on_validation

        self._lock = threading.Lock()
        self._latest: Dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Capabilities resolved lazily inside the polling thread (so import
        # errors don't block construction or hang the main process).
        self._psutil = None
        self._nvml = None
        self._nvml_handles: list = []

        # Prev-reading state for delta metrics (disk / net).
        self._prev_disk: Optional[Any] = None
        self._prev_net: Optional[Any] = None
        self._prev_ts: Optional[float] = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.rank_zero_only and trainer.global_rank != 0:
            return
        if self._thread is not None and self._thread.is_alive():
            return  # idempotent (Lightning calls setup more than once)

        log_header("HardwareMonitor")
        logging.info(f"  interval: {self.interval_seconds:.1f}s")
        logging.info(f"  prefix:   {self.prefix!r}")
        logging.info(
            f"  metrics:  cpu={self.log_cpu} ram={self.log_ram} "
            f"disk={self.log_disk} net={self.log_net} gpu={self.log_gpu}"
        )
        logging.info(
            f"  per-rank: {'rank 0 only' if self.rank_zero_only else 'every rank'}"
        )

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="spt-hardware-monitor",
            daemon=True,
        )
        self._thread.start()

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        # NVML must be shut down on the thread that initialised it; we do it
        # inside _poll_loop's finally clause.

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        if self.rank_zero_only and trainer.global_rank != 0:
            return
        self._flush(pl_module)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.log_on_validation:
            return
        if self.rank_zero_only and trainer.global_rank != 0:
            return
        self._flush(pl_module)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush(self, pl_module: LightningModule) -> None:
        with self._lock:
            if not self._latest:
                return
            metrics = dict(self._latest)
        pl_module.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            prog_bar=False,
        )

    def _init_capabilities(self) -> None:
        try:
            import psutil

            self._psutil = psutil
            # Prime CPU percent — the first call always returns 0.0.
            psutil.cpu_percent(interval=None)
            if self.log_disk:
                self._prev_disk = psutil.disk_io_counters()
            if self.log_net:
                self._prev_net = psutil.net_io_counters()
            self._prev_ts = time.time()
        except Exception as e:
            logging.warning(
                f"HardwareMonitor: psutil unavailable ({e!r}); CPU/RAM/disk/net "
                "metrics disabled."
            )
            self._psutil = None

        if not self.log_gpu:
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            self._nvml_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)
            ]
            self._nvml = pynvml
            logging.info(f"HardwareMonitor: pynvml found {n} GPU(s).")
        except Exception as e:
            logging.warning(
                f"HardwareMonitor: pynvml unavailable ({e!r}); GPU metrics disabled."
            )
            self._nvml = None
            self._nvml_handles = []

    def _poll_loop(self) -> None:
        self._init_capabilities()
        try:
            while not self._stop.is_set():
                try:
                    sample = self._sample()
                except Exception as e:
                    logging.warning(f"HardwareMonitor: sample failed: {e!r}")
                    sample = {}
                if sample:
                    with self._lock:
                        self._latest = sample
                # interruptible sleep
                self._stop.wait(self.interval_seconds)
        finally:
            if self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                except Exception:
                    pass

    def _sample(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        now = time.time()
        dt = (now - self._prev_ts) if self._prev_ts else None
        self._prev_ts = now

        if self._psutil is not None:
            ps = self._psutil
            if self.log_cpu:
                out[f"{self.prefix}cpu_percent"] = float(ps.cpu_percent(interval=None))
            if self.log_ram:
                vm = ps.virtual_memory()
                out[f"{self.prefix}ram_used_pct"] = float(vm.percent)
                out[f"{self.prefix}ram_used_gb"] = vm.used / _BYTES_PER_GB
            if self.log_disk and dt and dt > 0:
                cur = ps.disk_io_counters()
                if cur is not None and self._prev_disk is not None:
                    rd = (cur.read_bytes - self._prev_disk.read_bytes) / dt
                    wr = (cur.write_bytes - self._prev_disk.write_bytes) / dt
                    out[f"{self.prefix}disk_read_mb_s"] = rd / _BYTES_PER_MB
                    out[f"{self.prefix}disk_write_mb_s"] = wr / _BYTES_PER_MB
                self._prev_disk = cur
            if self.log_net and dt and dt > 0:
                cur = ps.net_io_counters()
                if cur is not None and self._prev_net is not None:
                    rx = (cur.bytes_recv - self._prev_net.bytes_recv) / dt
                    tx = (cur.bytes_sent - self._prev_net.bytes_sent) / dt
                    out[f"{self.prefix}net_recv_mb_s"] = rx / _BYTES_PER_MB
                    out[f"{self.prefix}net_sent_mb_s"] = tx / _BYTES_PER_MB
                self._prev_net = cur

        if self._nvml is not None and self._nvml_handles:
            nvml = self._nvml
            utils, used, total, temps, powers = [], [], [], [], []
            for i, h in enumerate(self._nvml_handles):
                try:
                    u = nvml.nvmlDeviceGetUtilizationRates(h)
                    m = nvml.nvmlDeviceGetMemoryInfo(h)
                except Exception as e:
                    logging.warning(f"HardwareMonitor: GPU{i} core query failed: {e!r}")
                    continue
                utils.append(float(u.gpu))
                used.append(float(m.used))
                total.append(float(m.total))
                if self.log_per_gpu:
                    out[f"{self.prefix}gpu{i}_util_pct"] = float(u.gpu)
                    out[f"{self.prefix}gpu{i}_mem_used_gb"] = m.used / _BYTES_PER_GB
                    out[f"{self.prefix}gpu{i}_mem_pct"] = (
                        100.0 * m.used / max(m.total, 1)
                    )
                # Optional counters — silently skip on permission / unsupported.
                try:
                    t = nvml.nvmlDeviceGetTemperature(h, 0)  # 0 = NVML_TEMPERATURE_GPU
                    temps.append(float(t))
                    if self.log_per_gpu:
                        out[f"{self.prefix}gpu{i}_temp_c"] = float(t)
                except Exception:
                    pass
                try:
                    p = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW -> W
                    powers.append(p)
                    if self.log_per_gpu:
                        out[f"{self.prefix}gpu{i}_power_w"] = p
                except Exception:
                    pass

            # Aggregates — useful single-number summaries for dashboards.
            if utils:
                out[f"{self.prefix}gpu_avg_util_pct"] = sum(utils) / len(utils)
            if used and total:
                out[f"{self.prefix}gpu_total_mem_used_gb"] = sum(used) / _BYTES_PER_GB
                out[f"{self.prefix}gpu_avg_mem_pct"] = (
                    100.0 * sum(used) / max(sum(total), 1)
                )
            if temps:
                out[f"{self.prefix}gpu_avg_temp_c"] = sum(temps) / len(temps)
            if powers:
                out[f"{self.prefix}gpu_total_power_w"] = sum(powers)

        return out
