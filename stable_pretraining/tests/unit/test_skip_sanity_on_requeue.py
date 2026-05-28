"""Regression: skip Lightning's validation sanity-check on SLURM requeue.

Lightning re-runs ``num_sanity_val_steps`` validation batches at every
``trainer.fit()`` call, including on a preempt-resume. That's wasteful
and pollutes the resume logs — the original launch already validated
the pipeline. ``Manager._maybe_skip_sanity_on_requeue`` forces
``trainer.num_sanity_val_steps = 0`` when ``SLURM_RESTART_COUNT >= 1``.

These tests exercise the helper directly (without invoking ``fit()``)
so they're fast and don't need a real SLURM job.
"""

import lightning as pl
import pytest


@pytest.fixture
def fake_manager_with_trainer():
    """Bare manager-like object wrapping a real Trainer.

    We bypass ``Manager.__init__`` (it has many dependencies — Hydra
    config, run-dir resolution, signal handlers) and just attach a
    real ``pl.Trainer`` so the helper can flip ``num_sanity_val_steps``.
    """
    from stable_pretraining.manager import Manager

    trainer = pl.Trainer(
        max_steps=1,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        accelerator="cpu",
        devices=1,
    )
    m = Manager.__new__(Manager)
    m._trainer = trainer
    return m, trainer


@pytest.mark.unit
class TestSkipSanityOnRequeue:
    """``_maybe_skip_sanity_on_requeue`` flips the flag iff requeue is detected."""

    def test_no_op_when_not_requeue(self, fake_manager_with_trainer, monkeypatch):
        m, trainer = fake_manager_with_trainer
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        m._maybe_skip_sanity_on_requeue()
        # Fresh launch — sanity check stays as configured.
        assert trainer.num_sanity_val_steps == 2

    def test_no_op_when_restart_count_zero(
        self, fake_manager_with_trainer, monkeypatch
    ):
        m, trainer = fake_manager_with_trainer
        monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
        m._maybe_skip_sanity_on_requeue()
        assert trainer.num_sanity_val_steps == 2

    def test_skips_when_requeue(self, fake_manager_with_trainer, monkeypatch):
        m, trainer = fake_manager_with_trainer
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        m._maybe_skip_sanity_on_requeue()
        assert trainer.num_sanity_val_steps == 0

    def test_skips_on_higher_restart_count(
        self, fake_manager_with_trainer, monkeypatch
    ):
        m, trainer = fake_manager_with_trainer
        monkeypatch.setenv("SLURM_RESTART_COUNT", "7")
        m._maybe_skip_sanity_on_requeue()
        assert trainer.num_sanity_val_steps == 0

    def test_no_op_when_already_zero(self, monkeypatch):
        """User who already set num_sanity_val_steps=0 shouldn't see anything change."""
        from stable_pretraining.manager import Manager

        trainer = pl.Trainer(
            max_steps=1,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu",
            devices=1,
        )
        m = Manager.__new__(Manager)
        m._trainer = trainer
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        m._maybe_skip_sanity_on_requeue()
        assert trainer.num_sanity_val_steps == 0

    def test_malformed_restart_count(self, fake_manager_with_trainer, monkeypatch):
        """A non-int env value should fall back to ``not requeue`` (no skip)."""
        m, trainer = fake_manager_with_trainer
        monkeypatch.setenv("SLURM_RESTART_COUNT", "not-an-int")
        m._maybe_skip_sanity_on_requeue()
        assert trainer.num_sanity_val_steps == 2
