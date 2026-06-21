"""Behavior lock for manual-opt modules with Trainer-level clip / accumulation.

Manual-optimization modules must accept Trainer-level ``gradient_clip_val`` /
``accumulate_grad_batches`` when launched via ``Manager``.

A vanilla Lightning ``Trainer`` *rejects* these two arguments when the module
uses manual optimization (``automatic_optimization = False``) — its
``_verify_manual_optimization_support`` validator raises before training
starts. Every ``spt.Module`` is manual-opt, so something has to defuse that
validation while still honoring the requested clip value and accumulation.

Historically that was a global monkey-patch on Lightning's validator. The
clean replacement strips-and-stashes the two values inside ``Manager`` right
before ``trainer.fit()``. **This test is deliberately mechanism-agnostic**: it
exercises the supported ``Manager`` entry point end-to-end and asserts the
observable outcome, so it passes identically whether the stashing is done by
the old patch or by ``Manager`` itself. Do not rewrite it to call a specific
implementation — its value is that it spans the change.

The internal contract it locks:

* ``trainer.gradient_clip_val`` is cleared to ``None`` (so Lightning's
  validator does not raise) and the original value moves to
  ``trainer.gradient_clip_val_``.
* ``trainer.accumulate_grad_batches`` is reset to ``1`` and the original moves
  to ``trainer.accumulate_grad_batches_``.
* ``Module.on_train_start`` reads those ``*_`` attributes back, so the single
  optimizer ends up with the requested clip value and step frequency.
* Training actually runs (``global_step`` advances) — i.e. ``fit`` did not
  crash on the manual-opt validation.
"""

from types import SimpleNamespace

import lightning as pl
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining.manager import Manager


class _DictDataset(Dataset):
    """Tiny dict-batch dataset (``spt.Module`` requires dict batches)."""

    def __init__(self, n: int = 16):
        self.x = torch.randn(n, 4)
        self.y = torch.randint(0, 2, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"image": self.x[i], "label": self.y[i]}


class _DictDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(_DictDataset(), batch_size=self.batch_size, shuffle=False)


def _forward(self, batch, stage):
    emb = self.backbone(batch["image"])
    out = {"embedding": emb, "label": batch["label"]}
    if stage == "fit":
        out["loss"] = emb.pow(2).mean()
    return out


def _make_module():
    return spt.Module(
        forward=_forward,
        hparams={},
        backbone=nn.Linear(4, 8),
        optim={
            "optimizer": {"type": "SGD", "lr": 0.1},
            "scheduler": {"type": "ConstantLR"},
            "interval": "step",
        },
    )


def _stub_manager_io(manager, monkeypatch):
    """Silence the side-effecting bits of ``Manager.__call__`` so the test is hermetic.

    Stubs wandb / loggers / signal-printing. We intentionally do NOT stub
    ``trainer.fit`` — the real fit is what triggers Lightning's manual-opt
    validation, which is the whole point of the test.
    """
    monkeypatch.setattr(manager, "init_and_sync_wandb", lambda: None)
    monkeypatch.setattr(manager, "_inject_registry_logger", lambda: None)
    monkeypatch.setattr(manager, "_configure_checkpointing", lambda: None)
    monkeypatch.setattr(
        "stable_pretraining.manager.print_logger_info", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "stable_pretraining.manager.print_signal_info", lambda *a, **kw: None
    )


@pytest.mark.unit
class TestManualOptTrainerArgs:
    """Trainer-level clip/accumulation must survive a manual-opt ``Manager`` run."""

    def test_clip_and_accumulate_are_stashed_and_honored(self, tmp_path, monkeypatch):
        clip_val = 1.0
        accumulate = 2

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm="norm",
            accumulate_grad_batches=accumulate,
            default_root_dir=str(tmp_path),
        )
        module = _make_module()
        data = _DictDataModule(batch_size=4)

        manager = spt.Manager(trainer=trainer, module=module, data=data, seed=0)
        _stub_manager_io(manager, monkeypatch)

        # Must NOT raise Lightning's manual-optimization validation error.
        manager()

        # --- the stash/clear contract that Module.on_train_start relies on ---
        assert trainer.gradient_clip_val is None
        assert getattr(trainer, "gradient_clip_val_", None) == clip_val
        assert getattr(trainer, "gradient_clip_algorithm_", None) is not None
        assert trainer.accumulate_grad_batches == 1
        assert getattr(trainer, "accumulate_grad_batches_", None) == accumulate

        # --- the values were actually resolved onto the single optimizer ---
        resolved_module = manager.instantiated_module
        names = list(resolved_module._optimizer_index_to_name.values())
        assert len(names) == 1, names
        name = names[0]
        assert resolved_module._optimizer_gradient_clip_val[name] == clip_val
        assert resolved_module._optimizer_frequencies[name] == accumulate

        # --- training actually happened (fit didn't crash on validation) ---
        assert trainer.global_step >= 1

    # ------------------------------------------------------------------
    # Fast, direct unit tests of the strip/stash logic itself (no fit).
    # ------------------------------------------------------------------

    @staticmethod
    def _run_prepare(automatic_optimization, **trainer_attrs):
        attrs = {
            "gradient_clip_val": None,
            "gradient_clip_algorithm": "norm",
            "accumulate_grad_batches": 1,
        }
        attrs.update(trainer_attrs)
        trainer = SimpleNamespace(**attrs)
        fake = SimpleNamespace(
            instantiated_module=SimpleNamespace(
                automatic_optimization=automatic_optimization
            ),
            _trainer=trainer,
        )
        Manager._prepare_manual_optimization(fake)
        return trainer

    def test_unit_strips_and_stashes_for_manual_opt(self):
        t = self._run_prepare(
            automatic_optimization=False,
            gradient_clip_val=1.5,
            gradient_clip_algorithm="value",
            accumulate_grad_batches=4,
        )
        assert t.gradient_clip_val is None
        assert t.gradient_clip_val_ == 1.5
        assert t.gradient_clip_algorithm_ == "value"
        assert t.accumulate_grad_batches == 1
        assert t.accumulate_grad_batches_ == 4

    def test_unit_noop_for_automatic_opt(self):
        """Automatic-optimization modules must be left untouched.

        Lightning applies clip/accumulation itself for them.
        """
        t = self._run_prepare(
            automatic_optimization=True,
            gradient_clip_val=1.5,
            accumulate_grad_batches=4,
        )
        assert t.gradient_clip_val == 1.5  # untouched
        assert t.accumulate_grad_batches == 4  # untouched
        assert not hasattr(t, "gradient_clip_val_")
        assert not hasattr(t, "accumulate_grad_batches_")

    def test_unit_noop_when_nothing_set(self):
        t = self._run_prepare(automatic_optimization=False)
        assert t.gradient_clip_val is None
        assert t.accumulate_grad_batches == 1
        assert not hasattr(t, "gradient_clip_val_")
        assert not hasattr(t, "accumulate_grad_batches_")

    def test_unit_clip_val_zero_is_not_stripped(self):
        """``gradient_clip_val == 0`` means "no clipping" and must not be stashed.

        Lightning does not reject it, so we leave it (mirrors the ``> 0`` guard).
        """
        t = self._run_prepare(
            automatic_optimization=False,
            gradient_clip_val=0,
            accumulate_grad_batches=2,
        )
        assert t.gradient_clip_val == 0  # untouched
        assert not hasattr(t, "gradient_clip_val_")
        # accumulation still handled independently
        assert t.accumulate_grad_batches == 1
        assert t.accumulate_grad_batches_ == 2

    def test_defaults_untouched_when_not_set(self, tmp_path, monkeypatch):
        """A manual-opt run with nothing requested is unaffected.

        The stash attributes are never created.
        """
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path),
        )
        module = _make_module()
        data = _DictDataModule(batch_size=4)

        manager = spt.Manager(trainer=trainer, module=module, data=data, seed=0)
        _stub_manager_io(manager, monkeypatch)
        manager()

        # Lightning's defaults: no clipping, accumulate every step.
        assert trainer.gradient_clip_val is None
        assert trainer.accumulate_grad_batches == 1
        assert trainer.global_step >= 1
