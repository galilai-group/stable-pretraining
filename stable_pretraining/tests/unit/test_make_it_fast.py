"""Unit tests for the opt-in ``spt.make_it_fast()`` performance helper.

These assert the flags actually land and that the determinism guard suppresses
``cudnn.benchmark``. CUDA-specific assertions are guarded so the test passes on
CPU-only machines (where those settings are deliberate no-ops).
"""

import pytest
import torch

import stable_pretraining as spt
from stable_pretraining import _fast


@pytest.fixture(autouse=True)
def _restore_fast_flag():
    """Isolate the process-global fast tag so tests don't leak into each other."""
    prev = _fast.enabled()
    try:
        yield
    finally:
        _fast.set_enabled(prev)


@pytest.mark.unit
class TestMakeItFast:
    """Unit tests for :func:`stable_pretraining.make_it_fast`."""

    def test_returns_dict_and_sets_matmul_precision(self):
        applied = spt.make_it_fast(verbose=False)
        assert isinstance(applied, dict)
        assert applied["matmul_precision"] == "high"
        assert torch.get_float32_matmul_precision() == "high"

    def test_sets_fast_mode_tag(self):
        _fast.set_enabled(False)
        spt.make_it_fast(verbose=False)
        assert _fast.enabled() is True

    def test_inductor_cache_env_set(self):
        spt.make_it_fast(verbose=False)
        import os

        assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"

    def test_matmul_precision_none_skips(self):
        applied = spt.make_it_fast(matmul_precision=None, verbose=False)
        assert "matmul_precision" not in applied

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_tf32_enabled_on_cuda(self):
        spt.make_it_fast(verbose=False)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True

    def test_tf32_skipped_without_cuda(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA present")
        applied = spt.make_it_fast(verbose=False)
        assert applied["tf32"] == "skipped: no CUDA"

    def test_cudnn_benchmark_skipped_under_determinism(self):
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            applied = spt.make_it_fast(cudnn_benchmark=True, verbose=False)
            assert applied["cudnn_benchmark"] == ("skipped: deterministic mode enabled")
        finally:
            torch.use_deterministic_algorithms(was_deterministic, warn_only=True)

    def test_cudnn_benchmark_enabled_when_not_deterministic(self):
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            applied = spt.make_it_fast(cudnn_benchmark=True, verbose=False)
            assert applied["cudnn_benchmark"] is True
            assert torch.backends.cudnn.benchmark is True
        finally:
            torch.use_deterministic_algorithms(was_deterministic, warn_only=True)

    def test_exported(self):
        assert "make_it_fast" in spt.__all__
        assert callable(spt.make_it_fast)


@pytest.mark.unit
class TestFusedOptimizer:
    """Fast-mode fused-kernel wiring in :func:`create_optimizer`."""

    def _make_model(self):
        return torch.nn.Linear(4, 4)

    def test_no_fused_when_fast_disabled(self):
        from stable_pretraining.optim import create_optimizer

        _fast.set_enabled(False)
        model = self._make_model()
        opt = create_optimizer(model.parameters(), {"type": "AdamW", "lr": 1e-3})
        assert opt.defaults.get("fused") in (None, False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_fused_added_on_cuda(self):
        from stable_pretraining.optim import create_optimizer

        _fast.set_enabled(True)
        model = self._make_model().cuda()
        opt = create_optimizer(model.parameters(), {"type": "AdamW", "lr": 1e-3})
        assert opt.defaults.get("fused") is True

    def test_fused_fallback_on_construction_error(self, monkeypatch):
        # When fused construction raises (e.g. optimizer/torch combo that
        # rejects fused for the given params), create_optimizer must retry
        # without fused rather than crash. Forced deterministically here.
        import torch.optim as to

        from stable_pretraining.optim import create_optimizer

        real_init = to.AdamW.__init__
        calls = {"n": 0}

        def fake_init(self, params, fused=False, **kw):
            calls["n"] += 1
            if fused:
                raise RuntimeError("fused requires CUDA")
            return real_init(self, params, **kw)

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(to.AdamW, "__init__", fake_init)
        _fast.set_enabled(True)
        model = self._make_model()
        opt = create_optimizer(model.parameters(), {"type": "AdamW", "lr": 1e-3})
        assert isinstance(opt, to.AdamW)
        assert calls["n"] == 2  # first (fused) raised, retried without fused

    def test_explicit_fused_false_respected(self, monkeypatch):
        from stable_pretraining.optim import create_optimizer

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        _fast.set_enabled(True)
        model = self._make_model()
        opt = create_optimizer(
            model.parameters(), {"type": "AdamW", "lr": 1e-3, "fused": False}
        )
        assert opt.defaults.get("fused") in (None, False)


@pytest.mark.unit
class TestManagerFastDefaults:
    """Fast-mode Trainer-config injection in :class:`Manager`.

    Built via ``__new__`` so we exercise the pure config logic without the
    heavy ``Manager.__init__`` (these helpers only read ``self.trainer``).
    """

    def _manager(self, trainer):
        from stable_pretraining import Manager

        m = Manager.__new__(Manager)
        m.trainer = trainer
        return m

    def test_no_op_when_fast_disabled(self):
        _fast.set_enabled(False)
        m = self._manager({"max_epochs": 1})
        m._apply_fast_trainer_defaults()
        assert "precision" not in m.trainer

    def test_precision_defaulted_when_supported(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        _fast.set_enabled(True)
        m = self._manager({"max_epochs": 1})
        m._apply_fast_trainer_defaults()
        assert m.trainer["precision"] == "bf16-mixed"

    def test_explicit_precision_respected(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        _fast.set_enabled(True)
        m = self._manager({"precision": "32-true"})
        m._apply_fast_trainer_defaults()
        assert m.trainer["precision"] == "32-true"

    def test_ddp_strategy_tuned_multi_gpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
        _fast.set_enabled(True)
        m = self._manager({"accelerator": "gpu", "devices": 4, "strategy": "ddp"})
        m._apply_fast_trainer_defaults()
        s = m.trainer["strategy"]
        assert isinstance(s, dict)
        assert s["_target_"].endswith("DDPStrategy")
        assert s["gradient_as_bucket_view"] is True
        assert s["find_unused_parameters"] is False

    def test_ddp_keeps_find_unused_when_requested(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        _fast.set_enabled(True)
        m = self._manager({"devices": 2, "strategy": "ddp_find_unused_parameters_true"})
        m._apply_fast_trainer_defaults()
        assert m.trainer["strategy"]["find_unused_parameters"] is True

    def test_single_gpu_strategy_untouched(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        _fast.set_enabled(True)
        m = self._manager({"devices": 1, "strategy": "ddp"})
        m._apply_fast_trainer_defaults()
        assert m.trainer["strategy"] == "ddp"

    def test_custom_strategy_object_untouched(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
        _fast.set_enabled(True)
        custom = {"_target_": "lightning.pytorch.strategies.FSDPStrategy"}
        m = self._manager({"devices": 4, "strategy": custom})
        m._apply_fast_trainer_defaults()
        assert m.trainer["strategy"] is custom

    def test_non_ddp_strategy_untouched(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
        _fast.set_enabled(True)
        m = self._manager({"devices": 4, "strategy": "fsdp"})
        m._apply_fast_trainer_defaults()
        assert m.trainer["strategy"] == "fsdp"

    def test_prebuilt_trainer_warns_and_skips(self):
        import lightning as pl

        _fast.set_enabled(True)
        trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=False)
        m = self._manager(trainer)
        m._apply_fast_trainer_defaults()  # must not raise
        assert m.trainer is trainer
