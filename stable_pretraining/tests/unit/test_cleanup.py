"""Unit tests for CleanUpCallback."""

import os

import pytest
from unittest.mock import Mock, patch

from stable_pretraining.callbacks.cleanup import CleanUpCallback

pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_artifacts(tmp_path):
    """Create a set of fake training artifacts in a temp dir."""
    # SLURM logs
    (tmp_path / "slurm-12345.out").write_text("slurm stdout")
    (tmp_path / "slurm-12345.err").write_text("slurm stderr")

    # Hydra
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text("key: value")
    (hydra_dir / "overrides.yaml").write_text("[]")
    (tmp_path / "hydra.log").write_text("hydra log content")

    # Checkpoints
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "epoch=0.ckpt").write_bytes(b"\x00" * 1024)
    (ckpt_dir / "epoch=1.ckpt").write_bytes(b"\x00" * 2048)

    # Logger dir (CSV)
    log_dir = tmp_path / "csv_logs"
    log_dir.mkdir()
    (log_dir / "metrics.csv").write_text("step,loss\n1,0.5\n")

    return tmp_path


@pytest.fixture
def mock_trainer(tmp_artifacts):
    """Mock trainer pointing at tmp_artifacts for checkpoints and loggers."""
    trainer = Mock()
    trainer.global_rank = 0
    trainer.default_root_dir = str(tmp_artifacts)

    ckpt_cb = Mock()
    ckpt_cb.dirpath = str(tmp_artifacts / "checkpoints")
    trainer.checkpoint_callbacks = [ckpt_cb]

    lg = Mock()
    lg.log_dir = str(tmp_artifacts / "csv_logs")
    lg.save_dir = None
    trainer.loggers = [lg]

    trainer.callbacks = []

    return trainer


# ============================================================================
# Defaults / init
# ============================================================================


def test_default_keeps():
    cb = CleanUpCallback()
    assert cb.keep_checkpoints is True
    assert cb.keep_logs is True
    assert cb.keep_hydra is False
    assert cb.keep_slurm is False
    assert cb.dry_run is False


def test_custom_init():
    cb = CleanUpCallback(
        keep_checkpoints=False,
        keep_logs=False,
        keep_hydra=True,
        keep_slurm=True,
        slurm_patterns=["job-*.log"],
        extra_patterns=["*.tmp"],
        dry_run=True,
    )
    assert cb.keep_checkpoints is False
    assert cb.keep_logs is False
    assert cb.keep_hydra is True
    assert cb.keep_slurm is True
    assert cb.slurm_patterns == ["job-*.log"]
    assert cb.extra_patterns == ["*.tmp"]
    assert cb.dry_run is True


# ============================================================================
# _collect_targets
# ============================================================================


def test_collect_slurm(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_slurm=False)
    with patch("os.getcwd", return_value=str(tmp_artifacts)):
        targets = cb._collect_targets(mock_trainer)
    slurm_targets = [t for t in targets if t[0] == "slurm"]
    assert len(slurm_targets) == 2
    paths = {os.path.basename(t[1]) for t in slurm_targets}
    assert paths == {"slurm-12345.out", "slurm-12345.err"}


def test_collect_slurm_kept(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_slurm=True)
    with patch("os.getcwd", return_value=str(tmp_artifacts)):
        targets = cb._collect_targets(mock_trainer)
    slurm_targets = [t for t in targets if t[0] == "slurm"]
    assert len(slurm_targets) == 0


def test_collect_hydra(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_hydra=False)
    with patch(
        "stable_pretraining.callbacks.cleanup._resolve_hydra_output_dir",
        return_value=str(tmp_artifacts),
    ):
        targets = cb._collect_targets(mock_trainer)
    hydra_targets = [t for t in targets if t[0] == "hydra"]
    assert len(hydra_targets) == 2  # hydra.log + .hydra/


def test_collect_hydra_kept(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_hydra=True)
    with patch(
        "stable_pretraining.callbacks.cleanup._resolve_hydra_output_dir",
        return_value=str(tmp_artifacts),
    ):
        targets = cb._collect_targets(mock_trainer)
    hydra_targets = [t for t in targets if t[0] == "hydra"]
    assert len(hydra_targets) == 0


def test_collect_checkpoints(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_checkpoints=False)
    targets = cb._collect_targets(mock_trainer)
    ckpt_targets = [t for t in targets if t[0] == "checkpoint"]
    assert len(ckpt_targets) == 1
    assert ckpt_targets[0][1] == str(tmp_artifacts / "checkpoints")


def test_collect_checkpoints_kept(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_checkpoints=True)
    targets = cb._collect_targets(mock_trainer)
    ckpt_targets = [t for t in targets if t[0] == "checkpoint"]
    assert len(ckpt_targets) == 0


def test_collect_logs(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_logs=False)
    targets = cb._collect_targets(mock_trainer)
    log_targets = [t for t in targets if t[0] == "logs"]
    assert len(log_targets) == 1
    assert log_targets[0][1] == str(tmp_artifacts / "csv_logs")


def test_collect_logs_kept(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_logs=True)
    targets = cb._collect_targets(mock_trainer)
    log_targets = [t for t in targets if t[0] == "logs"]
    assert len(log_targets) == 0


def test_collect_extra_patterns(tmp_artifacts, mock_trainer):
    (tmp_artifacts / "debug.tmp").write_text("tmp")
    pattern = str(tmp_artifacts / "*.tmp")
    cb = CleanUpCallback(extra_patterns=[pattern])
    targets = cb._collect_targets(mock_trainer)
    extra_targets = [t for t in targets if t[0] == "extra"]
    assert len(extra_targets) == 1


# ============================================================================
# on_fit_end — actual deletion
# ============================================================================


def test_on_fit_end_deletes_slurm(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_slurm=False)
    with patch("os.getcwd", return_value=str(tmp_artifacts)):
        cb.on_fit_end(mock_trainer, Mock())

    assert not (tmp_artifacts / "slurm-12345.out").exists()
    assert not (tmp_artifacts / "slurm-12345.err").exists()


def test_on_fit_end_deletes_checkpoints(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_checkpoints=False)
    cb.on_fit_end(mock_trainer, Mock())
    assert not (tmp_artifacts / "checkpoints").exists()


def test_on_fit_end_deletes_logs(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_logs=False)
    cb.on_fit_end(mock_trainer, Mock())
    assert not (tmp_artifacts / "csv_logs").exists()


def test_on_fit_end_deletes_all(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(
        keep_checkpoints=False,
        keep_logs=False,
        keep_hydra=False,
        keep_slurm=False,
    )
    with (
        patch("os.getcwd", return_value=str(tmp_artifacts)),
        patch(
            "stable_pretraining.callbacks.cleanup._resolve_hydra_output_dir",
            return_value=str(tmp_artifacts),
        ),
    ):
        cb.on_fit_end(mock_trainer, Mock())

    assert not (tmp_artifacts / "slurm-12345.out").exists()
    assert not (tmp_artifacts / "slurm-12345.err").exists()
    assert not (tmp_artifacts / ".hydra").exists()
    assert not (tmp_artifacts / "hydra.log").exists()
    assert not (tmp_artifacts / "checkpoints").exists()
    assert not (tmp_artifacts / "csv_logs").exists()


def test_on_fit_end_keeps_defaults(tmp_artifacts, mock_trainer):
    """Default settings keep checkpoints and logs, delete slurm and hydra."""
    cb = CleanUpCallback()
    with (
        patch("os.getcwd", return_value=str(tmp_artifacts)),
        patch(
            "stable_pretraining.callbacks.cleanup._resolve_hydra_output_dir",
            return_value=str(tmp_artifacts),
        ),
    ):
        cb.on_fit_end(mock_trainer, Mock())

    # Deleted by default
    assert not (tmp_artifacts / "slurm-12345.out").exists()
    assert not (tmp_artifacts / ".hydra").exists()
    # Kept by default
    assert (tmp_artifacts / "checkpoints").exists()
    assert (tmp_artifacts / "csv_logs").exists()


# ============================================================================
# Environment dump
# ============================================================================


def test_collect_env_dump(tmp_artifacts, mock_trainer):
    (tmp_artifacts / "environment.json").write_text("{}")
    (tmp_artifacts / "requirements_frozen.txt").write_text("torch==2.0")
    mock_trainer.default_root_dir = str(tmp_artifacts)

    cb = CleanUpCallback(keep_env_dump=False)
    targets = cb._collect_targets(mock_trainer)
    env_targets = [t for t in targets if t[0] == "env_dump"]
    assert len(env_targets) == 2
    names = {os.path.basename(t[1]) for t in env_targets}
    assert names == {"environment.json", "requirements_frozen.txt"}


def test_collect_env_dump_kept(tmp_artifacts, mock_trainer):
    (tmp_artifacts / "environment.json").write_text("{}")
    mock_trainer.default_root_dir = str(tmp_artifacts)

    cb = CleanUpCallback(keep_env_dump=True)
    targets = cb._collect_targets(mock_trainer)
    env_targets = [t for t in targets if t[0] == "env_dump"]
    assert len(env_targets) == 0


def test_on_fit_end_deletes_env_dump(tmp_artifacts, mock_trainer):
    (tmp_artifacts / "environment.json").write_text("{}")
    (tmp_artifacts / "requirements_frozen.txt").write_text("torch==2.0")
    mock_trainer.default_root_dir = str(tmp_artifacts)

    cb = CleanUpCallback(keep_env_dump=False)
    cb.on_fit_end(mock_trainer, Mock())

    assert not (tmp_artifacts / "environment.json").exists()
    assert not (tmp_artifacts / "requirements_frozen.txt").exists()


# ============================================================================
# Callback artifacts
# ============================================================================


def test_collect_callback_artifacts(tmp_artifacts, mock_trainer):
    # Simulate a LatentViz callback with a save_dir
    viz_dir = tmp_artifacts / "latent_viz_test"
    viz_dir.mkdir()
    (viz_dir / "epoch_0000.npz").write_bytes(b"\x00" * 100)

    viz_cb = Mock()
    viz_cb.save_dir = str(viz_dir)
    viz_cb.name = "test"
    mock_trainer.callbacks = [viz_cb]

    cb = CleanUpCallback(keep_callback_artifacts=False)
    targets = cb._collect_targets(mock_trainer)
    cb_targets = [t for t in targets if t[0] == "callback"]
    assert len(cb_targets) == 1
    assert cb_targets[0][1] == str(viz_dir)


def test_collect_callback_artifacts_kept(tmp_artifacts, mock_trainer):
    viz_dir = tmp_artifacts / "latent_viz_test"
    viz_dir.mkdir()

    viz_cb = Mock()
    viz_cb.save_dir = str(viz_dir)
    viz_cb.name = "test"
    mock_trainer.callbacks = [viz_cb]

    cb = CleanUpCallback(keep_callback_artifacts=True)
    targets = cb._collect_targets(mock_trainer)
    cb_targets = [t for t in targets if t[0] == "callback"]
    assert len(cb_targets) == 0


def test_on_fit_end_deletes_callback_artifacts(tmp_artifacts, mock_trainer):
    viz_dir = tmp_artifacts / "latent_viz_test"
    viz_dir.mkdir()
    (viz_dir / "epoch_0000.npz").write_bytes(b"\x00")

    viz_cb = Mock()
    viz_cb.save_dir = str(viz_dir)
    viz_cb.name = "test"
    mock_trainer.callbacks = [viz_cb]

    cb = CleanUpCallback(keep_callback_artifacts=False)
    cb.on_fit_end(mock_trainer, Mock())

    assert not viz_dir.exists()


# ============================================================================
# Dry run
# ============================================================================


def test_dry_run_does_not_delete(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(
        keep_checkpoints=False,
        keep_logs=False,
        keep_slurm=False,
        dry_run=True,
    )
    with patch("os.getcwd", return_value=str(tmp_artifacts)):
        cb.on_fit_end(mock_trainer, Mock())

    # Everything should still exist
    assert (tmp_artifacts / "slurm-12345.out").exists()
    assert (tmp_artifacts / "checkpoints").exists()
    assert (tmp_artifacts / "csv_logs").exists()


# ============================================================================
# Exception safety
# ============================================================================


def test_exception_skips_cleanup(tmp_artifacts, mock_trainer):
    cb = CleanUpCallback(keep_slurm=False, keep_checkpoints=False)

    # Simulate exception during training
    cb.on_exception(mock_trainer, Mock(), RuntimeError("boom"))

    with patch("os.getcwd", return_value=str(tmp_artifacts)):
        cb.on_fit_end(mock_trainer, Mock())

    # Nothing should be deleted
    assert (tmp_artifacts / "slurm-12345.out").exists()
    assert (tmp_artifacts / "checkpoints").exists()


# ============================================================================
# Edge cases
# ============================================================================


def test_no_artifacts_to_clean(mock_trainer):
    """No error when there are nothing to clean."""
    mock_trainer.checkpoint_callbacks = []
    mock_trainer.loggers = []
    cb = CleanUpCallback(keep_slurm=True, keep_hydra=True)
    # Should not raise
    cb.on_fit_end(mock_trainer, Mock())


def test_missing_checkpoint_dir(mock_trainer):
    """No error when checkpoint dir doesn't exist."""
    mock_trainer.checkpoint_callbacks[0].dirpath = "/nonexistent/path"
    cb = CleanUpCallback(keep_checkpoints=False)
    targets = cb._collect_targets(mock_trainer)
    ckpt_targets = [t for t in targets if t[0] == "checkpoint"]
    assert len(ckpt_targets) == 0


def test_logger_with_save_dir_fallback(tmp_artifacts, mock_trainer):
    """Falls back to save_dir when log_dir is None."""
    mock_trainer.loggers[0].log_dir = None
    mock_trainer.loggers[0].save_dir = str(tmp_artifacts / "csv_logs")
    cb = CleanUpCallback(keep_logs=False)
    targets = cb._collect_targets(mock_trainer)
    log_targets = [t for t in targets if t[0] == "logs"]
    assert len(log_targets) == 1
