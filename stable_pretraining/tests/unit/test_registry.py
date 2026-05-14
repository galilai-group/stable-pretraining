# test_registry.py
import warnings

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from stable_pretraining.callbacks.registry import (
    ModuleRegistryCallback,
    get_module,
    _MODULE_REGISTRY,
    _METRIC_BUFFER,
    _DICT_BUFFER,
    _IN_STEP,
)
from stable_pretraining import log, log_dict


class DummyModel(pl.LightningModule):
    """Minimal model for testing."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
        self.logged_values = []

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        # Test global log function
        log("test_metric", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.fixture
def dummy_dataloader():
    """Create a minimal dataloader for testing."""
    x = torch.randn(20, 10)
    y = torch.randn(20, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before and after each test."""
    _MODULE_REGISTRY.clear()
    _METRIC_BUFFER.clear()
    _DICT_BUFFER.clear()
    _IN_STEP.clear()
    yield
    _MODULE_REGISTRY.clear()
    _METRIC_BUFFER.clear()
    _DICT_BUFFER.clear()
    _IN_STEP.clear()


class TestModuleRegistryCallback:
    """Tests for ModuleRegistryCallback lifecycle and logging."""

    def test_module_registered_on_setup(self, dummy_model, dummy_dataloader):
        """Test that module is registered when training starts."""
        callback = ModuleRegistryCallback()
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[callback],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )

        assert get_module() is None
        trainer.fit(dummy_model, dummy_dataloader)
        # After teardown, should be cleaned up
        assert get_module() is None

    def test_module_accessible_during_training(self, dummy_model, dummy_dataloader):
        """Test that module is accessible via get_module during training."""
        accessed_during_training = []

        class CheckAccessCallback(pl.Callback):
            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                accessed_during_training.append(get_module() is not None)

        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[ModuleRegistryCallback(), CheckAccessCallback()],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        trainer.fit(dummy_model, dummy_dataloader)

        assert all(accessed_during_training)

    def test_log_function_works(self, dummy_model, dummy_dataloader):
        """Test that global log() function works during training."""
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[ModuleRegistryCallback()],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        # Should not raise
        trainer.fit(dummy_model, dummy_dataloader)

    def test_custom_registry_name(self, dummy_model, dummy_dataloader):
        """Test registration with custom name."""
        accessed_modules = {}

        class CheckNameCallback(pl.Callback):
            def on_train_start(self, trainer, pl_module):
                accessed_modules["default"] = get_module("default")
                accessed_modules["custom"] = get_module("custom")

        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[ModuleRegistryCallback("custom"), CheckNameCallback()],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        trainer.fit(dummy_model, dummy_dataloader)

        assert accessed_modules["default"] is None
        assert accessed_modules["custom"] is not None


class TestLogWarnings:
    """Tests for warning behavior when logging outside valid contexts."""

    def test_log_warns_when_no_module(self):
        """log() should warn when no module is registered."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log("my_metric", 1.0)
        assert len(w) == 1
        assert "no module registered" in str(w[0].message)

    def test_log_dict_warns_when_no_module(self):
        """log_dict() should warn when no module is registered."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_dict({"a": 1.0})
        assert len(w) == 1
        assert "no module registered" in str(w[0].message)

    def test_log_buffers_outside_step(self, dummy_model):
        """log() should buffer metrics when called outside a step."""
        _MODULE_REGISTRY["default"] = dummy_model
        _IN_STEP["default"] = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log("buffered_metric", 42.0)

        assert len(w) == 1
        assert "buffered" in str(w[0].message)
        assert "default" in _METRIC_BUFFER
        assert len(_METRIC_BUFFER["default"]) == 1
        assert _METRIC_BUFFER["default"][0][0] == "buffered_metric"

    def test_log_dict_buffers_outside_step(self, dummy_model):
        """log_dict() should buffer metrics when called outside a step."""
        _MODULE_REGISTRY["default"] = dummy_model
        _IN_STEP["default"] = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log_dict({"a": 1.0, "b": 2.0})

        assert len(w) == 1
        assert "buffered" in str(w[0].message)
        assert "default" in _DICT_BUFFER
        assert len(_DICT_BUFFER["default"]) == 1


class TestLogDict:
    """Tests for log_dict routing to module.log_dict (not module.log)."""

    def test_log_dict_calls_log_dict(self, dummy_model, dummy_dataloader):
        """log_dict() should call module.log_dict(), not module.log()."""
        called_log_dict = []

        original_log_dict = dummy_model.log_dict

        def patched_log_dict(*args, **kwargs):
            called_log_dict.append(args)
            return original_log_dict(*args, **kwargs)

        dummy_model.log_dict = patched_log_dict

        class LogDictCallback(pl.Callback):
            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                if batch_idx == 0:
                    log_dict({"a": torch.tensor(1.0)})

        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[ModuleRegistryCallback(), LogDictCallback()],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        trainer.fit(dummy_model, dummy_dataloader)

        assert len(called_log_dict) > 0


class TestBufferFlush:
    """Tests for buffer flushing during training."""

    def test_buffered_metrics_flushed_during_training(
        self, dummy_model, dummy_dataloader
    ):
        """Metrics buffered before training should be flushed at first batch."""
        callback = ModuleRegistryCallback()

        class BufferBeforeTrainCallback(pl.Callback):
            def on_train_start(self, trainer, pl_module):
                # This is outside a step — should buffer
                log("pre_train_metric", torch.tensor(1.0))

        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[callback, BufferBeforeTrainCallback()],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            trainer.fit(dummy_model, dummy_dataloader)

        # Buffer should be empty after training (flushed or cleaned up)
        assert "default" not in _METRIC_BUFFER

    def test_teardown_warns_on_dropped_metrics(self, dummy_model):
        """Teardown should warn if buffered metrics are dropped."""
        callback = ModuleRegistryCallback()
        _MODULE_REGISTRY["default"] = dummy_model
        _METRIC_BUFFER["default"] = [("dropped_metric", 1.0, {})]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            callback.teardown(None, dummy_model, "fit")

        assert len(w) == 1
        assert "dropped" in str(w[0].message)


@pytest.mark.unit
class TestInStepInBatchEnd:
    """Regression: ``_IN_STEP`` must stay True through ``on_train_batch_end``.

    Other callbacks (e.g. TeacherStudentCallback) that log inside that hook
    must get inline-logged with the current ``trainer.global_step`` instead
    of being buffered and replayed at the next batch's step. This guards
    against the EMA-coefficient step-stamp drift seen in the OceanCurrent
    v2 sweeps.
    """

    def test_in_step_true_during_on_train_batch_end(
        self, dummy_model, dummy_dataloader
    ):
        """Late on_train_batch_end callbacks should still see _IN_STEP=True.

        A callback firing AFTER ``ModuleRegistryCallback.on_train_batch_end``
        must observe ``_IN_STEP=True`` so its ``log()`` goes inline (not
        buffered).
        """
        registry_cb = ModuleRegistryCallback()
        seen_in_step = []

        class SnoopCallback(pl.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                # By the time this fires, the ModuleRegistryCallback has already
                # processed its own on_train_batch_end. _IN_STEP must remain True.
                seen_in_step.append(_IN_STEP.get("default", False))

        # Order matters: registry callback first so its hooks fire first.
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[registry_cb, SnoopCallback()],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        trainer.fit(dummy_model, dummy_dataloader)

        assert len(seen_in_step) > 0, "snoop callback never fired"
        assert all(seen_in_step), (
            f"expected _IN_STEP=True at every on_train_batch_end, got {seen_in_step}"
        )

    def test_log_in_batch_end_is_inline_not_buffered(
        self, dummy_model, dummy_dataloader
    ):
        """log() inside on_train_batch_end must be inline, not buffered.

        Before the fix, it landed in ``_METRIC_BUFFER`` because ``_IN_STEP``
        had already been flipped False by the time the user callback ran.
        """
        registry_cb = ModuleRegistryCallback()
        n_buffered_at_call = []

        class LoggerCallback(pl.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                pre = len(_METRIC_BUFFER.get("default", []))
                log("ema_in_batch_end", torch.tensor(0.5))
                post = len(_METRIC_BUFFER.get("default", []))
                # If buffer length grew → metric was buffered (BUG).
                # If it stayed the same → metric was inline-logged (FIX).
                n_buffered_at_call.append(post - pre)

        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[registry_cb, LoggerCallback()],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        trainer.fit(dummy_model, dummy_dataloader)

        assert len(n_buffered_at_call) > 0
        assert all(d == 0 for d in n_buffered_at_call), (
            f"log() in on_train_batch_end should be inline, but {sum(d for d in n_buffered_at_call if d > 0)} "
            f"of {len(n_buffered_at_call)} calls were buffered"
        )

    def test_in_step_false_after_train_epoch_end(self, dummy_model, dummy_dataloader):
        """_IN_STEP must exit cleanly at the train_epoch_end boundary.

        The exit moved from ``on_train_batch_end`` to ``on_train_epoch_end``;
        verify it actually flips False at the epoch boundary so we don't
        leak a True value into pre-fit / post-fit log calls.
        """
        registry_cb = ModuleRegistryCallback()
        seen_after_epoch_end = []

        class SnoopAtFitEnd(pl.Callback):
            def on_train_end(self, trainer, pl_module):
                # on_train_end fires AFTER on_train_epoch_end.
                seen_after_epoch_end.append(_IN_STEP.get("default", False))

        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[registry_cb, SnoopAtFitEnd()],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        trainer.fit(dummy_model, dummy_dataloader)

        assert seen_after_epoch_end == [False], (
            f"expected _IN_STEP=False after train epoch end, got {seen_after_epoch_end}"
        )

    def test_teardown_clean_when_no_buffer(self, dummy_model):
        """Teardown should not warn if no buffered metrics."""
        callback = ModuleRegistryCallback()
        _MODULE_REGISTRY["default"] = dummy_model

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            callback.teardown(None, dummy_model, "fit")

        assert len(w) == 0
