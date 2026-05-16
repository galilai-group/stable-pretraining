"""Unit tests for the HardwareMonitor callback.

Coverage:
1. The polling thread starts, populates ``_latest`` with sensible keys,
   and exits cleanly on teardown.
2. ``on_train_batch_end`` actually emits the latest sample through
   ``pl_module.log_dict`` (verified by intercepting the call).
3. Graceful degradation when ``pynvml`` is unavailable — no GPU keys,
   but psutil-backed keys still emit.
4. Rank-zero guard suppresses both the polling thread and the log flush
   on non-zero ranks.
5. Invalid interval is rejected at construction.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from stable_pretraining.callbacks.hardware_monitor import HardwareMonitor


@pytest.mark.unit
class TestSampling:
    """Verify the polling thread populates _latest with expected metrics."""

    def test_sampler_populates_latest(self):
        # Short interval so the test runs quickly.
        cb = HardwareMonitor(interval_seconds=0.05, log_gpu=False)
        # Run init + first sample synchronously instead of starting the
        # thread, so we don't have to sleep.
        cb._init_capabilities()
        # First sample after _init only has cpu/ram (disk/net need a delta).
        # Take two samples to exercise the delta path too.
        cb._sample()  # prime delta counters
        # disk_io_counters resolution can be slow on some platforms; just
        # wait a moment so the deltas are non-zero.
        import time as _t

        _t.sleep(0.05)
        s2 = cb._sample()

        # CPU/RAM keys are unconditional (psutil installed in the test env)
        assert "hardware/cpu_percent" in s2
        assert "hardware/ram_used_pct" in s2
        assert "hardware/ram_used_gb" in s2
        # Delta-based metrics show up on the second sample
        # (allowed to be absent if disk_io_counters returned None — rare in CI)
        if "hardware/disk_read_mb_s" in s2:
            assert s2["hardware/disk_read_mb_s"] >= 0
        if "hardware/net_recv_mb_s" in s2:
            assert s2["hardware/net_recv_mb_s"] >= 0

    def test_thread_lifecycle(self):
        cb = HardwareMonitor(interval_seconds=0.05, log_gpu=False)
        trainer = MagicMock()
        trainer.global_rank = 0
        pl_module = MagicMock()

        cb.setup(trainer, pl_module, "fit")
        assert cb._thread is not None
        assert cb._thread.is_alive()

        # Wait until at least one sample lands.
        deadline = threading.Event()
        for _ in range(20):
            with cb._lock:
                if cb._latest:
                    break
            deadline.wait(0.05)
        with cb._lock:
            assert cb._latest, "polling thread never populated _latest"

        cb.teardown(trainer, pl_module, "fit")
        assert cb._thread is None


@pytest.mark.unit
class TestLogging:
    """Verify on_train_batch_end calls pl_module.log_dict with the latest sample."""

    def test_flush_emits_through_pl_module(self):
        cb = HardwareMonitor(interval_seconds=10.0)
        # Inject a fake latest sample so we don't have to start the thread.
        with cb._lock:
            cb._latest = {
                "hardware/cpu_percent": 12.3,
                "hardware/gpu0_util_pct": 88.0,
            }

        trainer = MagicMock()
        trainer.global_rank = 0
        pl_module = MagicMock()

        cb.on_train_batch_end(trainer, pl_module, None, None, 0)
        assert pl_module.log_dict.called
        args, kwargs = pl_module.log_dict.call_args
        emitted = args[0] if args else kwargs.get("dictionary", {})
        assert emitted["hardware/cpu_percent"] == 12.3
        assert emitted["hardware/gpu0_util_pct"] == 88.0
        # Don't sync across ranks (hardware is per-host).
        assert kwargs.get("sync_dist") is False

    def test_flush_no_op_when_latest_empty(self):
        cb = HardwareMonitor(interval_seconds=10.0)
        trainer = MagicMock()
        trainer.global_rank = 0
        pl_module = MagicMock()
        cb.on_train_batch_end(trainer, pl_module, None, None, 0)
        pl_module.log_dict.assert_not_called()


@pytest.mark.unit
class TestRankZeroGuard:
    """Non-zero ranks must not start the thread or flush metrics."""

    def test_setup_skipped_on_nonzero_rank(self):
        cb = HardwareMonitor(interval_seconds=0.05)
        trainer = MagicMock()
        trainer.global_rank = 1
        pl_module = MagicMock()
        cb.setup(trainer, pl_module, "fit")
        assert cb._thread is None  # never started

    def test_flush_skipped_on_nonzero_rank(self):
        cb = HardwareMonitor(interval_seconds=10.0)
        with cb._lock:
            cb._latest = {"hardware/cpu_percent": 1.0}
        trainer = MagicMock()
        trainer.global_rank = 1
        pl_module = MagicMock()
        cb.on_train_batch_end(trainer, pl_module, None, None, 0)
        pl_module.log_dict.assert_not_called()


@pytest.mark.unit
class TestGracefulDegradation:
    """Sampler must keep working if pynvml is missing or broken."""

    def test_no_gpu_keys_when_nvml_unavailable(self):
        cb = HardwareMonitor(interval_seconds=10.0, log_gpu=True)
        # Force the import to fail mid-init.
        with patch.dict("sys.modules", {"pynvml": None}):
            cb._init_capabilities()
        assert cb._nvml is None
        sample = cb._sample()
        # No gpu_* keys at all
        assert not any(k.startswith("hardware/gpu") for k in sample)
        # But cpu/ram keys still present
        assert "hardware/cpu_percent" in sample

    def test_log_gpu_false_skips_nvml(self):
        cb = HardwareMonitor(interval_seconds=10.0, log_gpu=False)
        cb._init_capabilities()
        assert cb._nvml is None  # never initialised


@pytest.mark.unit
class TestConstruction:
    """Constructor argument validation."""

    def test_rejects_zero_interval(self):
        with pytest.raises(ValueError, match="interval_seconds must be > 0"):
            HardwareMonitor(interval_seconds=0)

    def test_rejects_negative_interval(self):
        with pytest.raises(ValueError, match="interval_seconds must be > 0"):
            HardwareMonitor(interval_seconds=-1.0)
