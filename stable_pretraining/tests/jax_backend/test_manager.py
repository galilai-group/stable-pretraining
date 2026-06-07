"""Tests for the JAX Manager: cache-dir/run-dir resolution, SLURM resume, preemption."""

import os
import signal

import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining as spt  # noqa: E402
import stable_pretraining.jax as spj  # noqa: E402
from stable_pretraining.jax.forward import supervised  # noqa: E402


def _module(seed=0):
    rngs = nnx.Rngs(seed)
    return spj.Module(
        forward=supervised,
        optim="adamw",
        backbone=spj.backbone.MLP(8, [16], rngs=rngs),
        classifier=nnx.Linear(16, 3, rngs=rngs),
    )


def _data(n=3, b=8, seed=0):
    rng = np.random.RandomState(seed)
    return [
        {"image": rng.randn(b, 8).astype("float32"), "label": rng.randint(0, 3, b)}
        for _ in range(n)
    ]


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the global config's cache_dir at a temp dir; restore afterward.

    Uses the property setter (not ``spt.set(cache_dir=None)``, where ``None`` is
    the "unset" sentinel) and clears SLURM env so tests are isolated.
    """
    for k in ("SLURM_JOB_ID", "SLURM_RESTART_COUNT", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(k, raising=False)
    cfg = spt.get_config()
    old = cfg.cache_dir
    cfg.cache_dir = str(tmp_path)
    yield tmp_path
    cfg.cache_dir = old


def test_run_dir_layout(cache_dir):
    mgr = spj.Manager(spj.Trainer(max_epochs=1), _module(), _data())
    # {cache_dir}/runs/{YYYYMMDD}/{HHMMSS}/{run_id}
    assert mgr.run_dir is not None
    rel = mgr.run_dir.relative_to(cache_dir)
    assert rel.parts[0] == "runs" and len(rel.parts) == 4
    assert mgr.ckpt_path.name == "last.msgpack"


def test_no_cache_dir_disables_checkpointing(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    cfg = spt.get_config()
    old = cfg.cache_dir
    cfg.cache_dir = None
    try:
        mgr = spj.Manager(spj.Trainer(max_epochs=1), _module(), _data())
        assert mgr.run_dir is None and mgr.ckpt_path is None
        mgr()  # still trains fine without checkpointing
    finally:
        cfg.cache_dir = old


def test_manager_writes_checkpoint_and_autoresumes(cache_dir):
    mgr = spj.Manager(spj.Trainer(max_epochs=2), _module(), _data())
    mgr()
    assert mgr.ckpt_path.exists()

    # A second Manager pointed at the same run_dir auto-resumes from it.
    mgr2 = spj.Manager(
        spj.Trainer(max_epochs=1), _module(seed=9), _data(), run_dir=str(mgr.run_dir)
    )
    trainer = mgr2()
    assert trainer.global_step >= 2 * len(_data())  # resumed step carried over


def test_slurm_requeue_reuses_run_dir(cache_dir, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)

    # First run writes the slurm index.
    mgr = spj.Manager(spj.Trainer(max_epochs=1), _module(), _data())
    mgr()
    first_run_dir = mgr.run_dir
    assert (cache_dir / ".slurm_index" / "123456").read_text().strip() == str(
        first_run_dir
    )

    # Requeue: same job id, RESTART_COUNT>=1 -> reuse the same run_dir + resume.
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    mgr2 = spj.Manager(spj.Trainer(max_epochs=1), _module(seed=5), _data())
    assert mgr2.run_dir == first_run_dir
    trainer = mgr2()
    assert trainer.global_step >= 2 * len(_data())


def test_preemption_checkpoints_and_exits(cache_dir, monkeypatch):
    """A flagged SIGTERM must checkpoint then sys.exit (no requeue off-SLURM)."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    mgr = spj.Manager(spj.Trainer(max_epochs=5), _module(), _data())
    # Simulate SLURM delivering SIGTERM mid-run.
    os.kill(os.getpid(), signal.SIGTERM)
    with pytest.raises(SystemExit):
        mgr()
    assert mgr.ckpt_path.exists()  # checkpoint written before exit
    step = spj.checkpoint.load(str(mgr.ckpt_path), module=_module(seed=3))
    assert step >= 0
