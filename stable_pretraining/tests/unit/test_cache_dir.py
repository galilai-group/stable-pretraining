"""Unit tests for the cache_dir / run directory feature."""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from stable_pretraining._config import get_config, set as spt_set
from stable_pretraining.manager import (
    Manager,
    _generate_run_id,
    _RunDirCallback,
    _RUN_META_FILENAME,
)
from stable_pretraining.tests.utils import BoringTrainer, BoringModule, BoringDataModule

pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global config before and after every test."""
    cfg = get_config()
    cfg.reset()
    yield
    cfg.reset()


@pytest.fixture()
def cache_dir(tmp_path):
    """Provide a temporary cache_dir and configure it globally."""
    d = tmp_path / "spt_cache"
    spt_set(cache_dir=str(d))
    return d


# ============================================================================
# _config.py — cache_dir property
# ============================================================================


class TestCacheDirConfig:
    """Tests for cache_dir config defaults and env overrides."""

    def test_default_is_set(self, monkeypatch):
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        get_config().reset()
        assert get_config().cache_dir is not None
        assert "stable-pretraining" in get_config().cache_dir

    def test_set_via_spt_set(self, tmp_path):
        spt_set(cache_dir=str(tmp_path))
        assert get_config().cache_dir == str(tmp_path)

    def test_set_via_property(self, tmp_path):
        cfg = get_config()
        cfg.cache_dir = str(tmp_path)
        assert cfg.cache_dir == str(tmp_path)

    def test_set_to_none(self, tmp_path):
        spt_set(cache_dir=str(tmp_path))
        cfg = get_config()
        cfg.cache_dir = None
        assert cfg.cache_dir is None

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="must not be empty"):
            spt_set(cache_dir="")

    def test_rejects_whitespace_only(self):
        with pytest.raises(ValueError, match="must not be empty"):
            spt_set(cache_dir="   ")

    def test_rejects_non_string(self):
        with pytest.raises(TypeError, match="must be a str"):
            get_config().cache_dir = 123

    def test_reset_restores_default_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        spt_set(cache_dir=str(tmp_path))
        get_config().reset()
        # reset() restores the default (~/.cache/stable-pretraining), not None
        assert get_config().cache_dir is not None
        assert "stable-pretraining" in get_config().cache_dir

    def test_repr_includes_cache_dir(self, tmp_path):
        spt_set(cache_dir=str(tmp_path))
        assert "cache_dir=" in repr(get_config())

    def test_env_var_sets_default(self, monkeypatch, tmp_path):
        """SPT_CACHE_DIR env var should be picked up on init."""
        monkeypatch.setenv("SPT_CACHE_DIR", str(tmp_path))
        cfg = get_config()
        cfg.reset()  # re-reads env var in _init_defaults
        assert cfg.cache_dir == str(tmp_path)

    def test_tilde_expansion(self):
        """cache_dir with ~ should be stored as-is (expanded in _resolve_run_dir)."""
        spt_set(cache_dir="~/spt_cache")
        assert get_config().cache_dir == "~/spt_cache"

    def test_set_no_args_does_not_affect_cache_dir(self, tmp_path):
        spt_set(cache_dir=str(tmp_path))
        spt_set()  # no-op
        assert get_config().cache_dir == str(tmp_path)


# ============================================================================
# _config.py — requeue_checkpoint property
# ============================================================================


class TestRequeueCheckpointConfig:
    """Tests for requeue checkpoint config resolution."""

    def test_default_is_true(self):
        assert get_config().requeue_checkpoint is True

    def test_set_via_spt_set(self):
        spt_set(requeue_checkpoint=False)
        assert get_config().requeue_checkpoint is False

    def test_set_via_property(self):
        cfg = get_config()
        cfg.requeue_checkpoint = False
        assert cfg.requeue_checkpoint is False

    def test_set_back_to_true(self):
        spt_set(requeue_checkpoint=False)
        spt_set(requeue_checkpoint=True)
        assert get_config().requeue_checkpoint is True

    def test_rejects_non_bool(self):
        with pytest.raises(TypeError, match="must be a bool"):
            get_config().requeue_checkpoint = "yes"

    def test_rejects_int(self):
        with pytest.raises(TypeError, match="must be a bool"):
            spt_set(requeue_checkpoint=1)

    def test_reset_restores_default(self):
        spt_set(requeue_checkpoint=False)
        get_config().reset()
        assert get_config().requeue_checkpoint is True

    def test_repr_includes_requeue_checkpoint(self):
        assert "requeue_checkpoint=" in repr(get_config())

    def test_set_no_args_does_not_affect(self):
        spt_set(requeue_checkpoint=False)
        spt_set()  # no-op
        assert get_config().requeue_checkpoint is False


# ============================================================================
# _generate_run_id
# ============================================================================


class TestGenerateRunId:
    """``_generate_run_id`` always returns a fresh uuid4 hex (12 chars).

    SLURM/torchrun env-var awareness moved out of run_id generation entirely;
    requeue resume is handled by the SLURM-index lookup in
    ``_resolve_run_dir`` instead. See ``TestSlurmRequeueIndex`` below.
    """

    def test_returns_12_hex_chars(self):
        run_id = _generate_run_id()
        assert len(run_id) == 12
        assert run_id.isalnum()

    def test_two_calls_are_different(self):
        assert _generate_run_id() != _generate_run_id()

    def test_unaffected_by_slurm_env_vars(self, monkeypatch):
        """SLURM env vars must not leak into run_id.

        SLURM_JOB_ID etc. used to be baked into run_id, but the new design
        keeps run_id always-uuid and uses SLURM env vars only for the index
        lookup.
        """
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "abc123")
        run_id = _generate_run_id()
        assert "12345" not in run_id and "abc123" not in run_id
        assert len(run_id) == 12

    def test_slurm_array_tasks_differ(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "100")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
        id0 = _generate_run_id()
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
        id1 = _generate_run_id()
        assert id0 != id1


# ============================================================================
# Manager._resolve_run_dir
# ============================================================================


class TestResolveRunDir:
    """Tests for run_dir resolution logic."""

    def _make_manager(self, ckpt_path=None):
        return Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
            ckpt_path=str(ckpt_path) if ckpt_path else None,
        )

    def test_returns_none_when_cache_dir_none(self):
        get_config()._cache_dir = None
        manager = self._make_manager()
        assert manager._resolve_run_dir() is None

    def test_creates_run_dir_under_cache_dir(self, cache_dir):
        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        assert run_dir is not None
        assert str(run_dir).startswith(str(cache_dir))
        assert run_dir.is_dir()

    def test_run_dir_has_date_time_id_structure(self, cache_dir, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        parts = run_dir.relative_to(cache_dir).parts
        assert parts[0] == "runs"
        assert len(parts[1]) == 8 and parts[1].isdigit()  # YYYYMMDD
        assert len(parts[2]) == 6 and parts[2].isdigit()  # HHMMSS
        assert len(parts[3]) == 12  # uuid hex

    def test_writes_run_meta_json(self, cache_dir):
        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        meta_path = run_dir / _RUN_META_FILENAME
        assert meta_path.is_file()
        meta = json.loads(meta_path.read_text())
        assert meta["run_dir"] == str(run_dir)
        assert "run_id" in meta

    def test_restores_via_slurm_index_on_requeue(self, cache_dir, monkeypatch):
        """SLURM requeue reuses the indexed run_dir.

        Same SLURM_JOB_ID *and* SLURM_RESTART_COUNT≥1 *and* a recorded
        ``.slurm_index/<key>`` entry → reuse the original run_dir.
        """
        monkeypatch.setenv("SLURM_JOB_ID", "99999")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        # Pre-existing run from the original (pre-requeue) invocation.
        prev_run_dir = cache_dir / "runs" / "20260101" / "120000" / "previd123456"
        prev_run_dir.mkdir(parents=True)
        idx_dir = cache_dir / ".slurm_index"
        idx_dir.mkdir()
        (idx_dir / "99999").write_text(str(prev_run_dir))

        manager = self._make_manager(ckpt_path=None)
        run_dir = manager._resolve_run_dir()
        assert run_dir == prev_run_dir

    def test_fresh_run_when_sidecar_missing(self, cache_dir, monkeypatch):
        # Don't trigger the requeue path while testing sidecar fallback.
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        ckpt = cache_dir / "stale.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.touch()
        manager = self._make_manager(ckpt_path=ckpt)
        run_dir = manager._resolve_run_dir()
        assert run_dir is not None
        assert run_dir.is_dir()

    def test_fresh_run_when_sidecar_corrupt(self, cache_dir, monkeypatch):
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        ckpt = cache_dir / "corrupt.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.touch()
        (cache_dir / _RUN_META_FILENAME).write_text("NOT JSON!!")
        manager = self._make_manager(ckpt_path=ckpt)
        run_dir = manager._resolve_run_dir()
        assert run_dir is not None
        assert run_dir.is_dir()

    def test_run_id_is_uuid_even_with_slurm_job_id(self, cache_dir, monkeypatch):
        """run_id stays a 12-char uuid even under SLURM.

        The SLURM_JOB_ID is recorded in the side index, not embedded in
        run_id.
        """
        monkeypatch.setenv("SLURM_JOB_ID", "42")
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        assert run_dir.name != "42"
        assert len(run_dir.name) == 12 and run_dir.name.isalnum()

    def test_fresh_run_writes_slurm_index(self, cache_dir, monkeypatch):
        """Fresh SLURM run records its run_dir in the index.

        Writing ``cache_dir/.slurm_index/<key>`` is what lets a future
        requeue (Strategy 2) find the original directory.
        """
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        idx = cache_dir / ".slurm_index" / "12345"
        assert idx.is_file()
        assert idx.read_text().strip() == str(run_dir)

    def test_array_task_index_key_includes_task_id(self, cache_dir, monkeypatch):
        """SLURM_ARRAY_TASK_ID disambiguates tasks within the same array job."""
        monkeypatch.setenv("SLURM_JOB_ID", "100")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        manager = self._make_manager()
        manager._resolve_run_dir()
        assert (cache_dir / ".slurm_index" / "100_3").is_file()
        assert not (cache_dir / ".slurm_index" / "100").exists()

    def test_interactive_rerun_gets_fresh_dir(self, cache_dir, monkeypatch):
        """Interactive SLURM reruns get distinct run dirs.

        Two consecutive Manager calls inside the same SLURM allocation (same
        SLURM_JOB_ID, RESTART_COUNT=0) must produce DIFFERENT run dirs —
        this was the whole point of the redesign.
        """
        monkeypatch.setenv("SLURM_JOB_ID", "55555")
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        m1 = self._make_manager()
        d1 = m1._resolve_run_dir()
        m2 = self._make_manager()
        d2 = m2._resolve_run_dir()
        assert d1 != d2, "interactive re-runs must not share a run_dir"

    def test_requeue_without_index_falls_through_to_fresh(self, cache_dir, monkeypatch):
        """Requeue + no index + no orphan run_dir = early-preempt fallback.

        Cluster-wide eviction during submitit pickle load can produce
        ``RESTART_COUNT≥1`` with no artefact on disk — the prior task
        died before reaching ``Manager.__init__``. There's nothing to
        recover; treat as a fresh run. The strict raise is reserved for
        the orphan scenario (see ``TestEarlyPreemptFallback`` in
        ``test_slurm_index.py``).
        """
        monkeypatch.setenv("SLURM_JOB_ID", "77777")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        # No .slurm_index/77777 and no run_dir stamped with this JOB_ID.
        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        assert run_dir is not None and run_dir.is_dir()
        assert manager._early_preempt_fallback is True
        # Index now written so a future requeue will find this attempt.
        assert (cache_dir / ".slurm_index" / "77777").is_file()

    def test_requeue_with_stale_index_raises(self, cache_dir, monkeypatch):
        """If the index points at a directory that's been deleted, error."""
        monkeypatch.setenv("SLURM_JOB_ID", "88888")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        idx_dir = cache_dir / ".slurm_index"
        idx_dir.mkdir(parents=True, exist_ok=True)
        (idx_dir / "88888").write_text("/nonexistent/path/that/does/not/exist")
        manager = self._make_manager()
        with pytest.raises(RuntimeError, match="no longer exists"):
            manager._resolve_run_dir()

    def test_tilde_expanded(self, monkeypatch):
        spt_set(cache_dir="~/spt_test_cache")
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        run_dir = manager._resolve_run_dir()
        assert "~" not in str(run_dir)
        assert str(run_dir).startswith(os.path.expanduser("~"))
        import shutil

        if run_dir.exists():
            shutil.rmtree(run_dir.parent.parent.parent.parent)

    def test_sets_run_id_attribute(self, cache_dir, monkeypatch):
        """run_id mirrors the run_dir name (uuid), regardless of SLURM env."""
        monkeypatch.setenv("SLURM_JOB_ID", "555")
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._resolve_run_dir()
        assert manager._run_id == manager._run_dir.name
        assert len(manager._run_id) == 12 and manager._run_id.isalnum()


# ============================================================================
# DDP rank coordination — rank-0 publishes run_dir, rank-N waits for it
# ============================================================================


class TestDDPRankHandoff:
    """Rank-0 publishes run_dir; rank-N blocks on the handoff and adopts it.

    The bug being guarded against: every rank used to call ``_generate_run_id``
    and ``datetime.now()`` independently, so each rank created its own
    ``runs/<datetime>/<uuid>/`` and last-writer-wins for ``.slurm_index/<key>``.
    On preempt+requeue, the published index entry could point at a non-rank-0
    directory that was never written to → silent loss of training history.

    The fix: rank 0 picks the dir, atomically publishes it under
    ``cache_dir/.rank_handoff/<launch_key>``; ranks 1..N block on that file
    and adopt its value.
    """

    def _make_manager(self):
        from stable_pretraining.tests.utils import (
            BoringTrainer,
            BoringModule,
            BoringDataModule,
        )

        return Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )

    @staticmethod
    def _set_rank(monkeypatch, rank: int) -> None:
        """Override Lightning's cached rank for the duration of one test.

        ``rank_zero_only.rank`` is set at module-import time from env vars and
        cached, so monkeypatching env vars after import has no effect on it.
        We patch the attribute directly — this is exactly what Lightning's own
        Strategy does at setup time when it learns the real rank from the
        process group.
        """
        from lightning.pytorch.utilities.rank_zero import rank_zero_only

        monkeypatch.setattr(rank_zero_only, "rank", rank, raising=False)

    # -- launch-key uniqueness ------------------------------------------------

    def test_launch_key_slurm_batch(self, monkeypatch):
        from stable_pretraining.manager import _ddp_launch_key

        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
        monkeypatch.delenv("MASTER_ADDR", raising=False)
        monkeypatch.setenv("SLURM_JOB_ID", "98341")
        assert _ddp_launch_key() == "slurm-98341"

    def test_launch_key_slurm_array(self, monkeypatch):
        from stable_pretraining.manager import _ddp_launch_key

        monkeypatch.setenv("SLURM_JOB_ID", "98341")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "5")
        assert _ddp_launch_key() == "slurm-98341_5"

    def test_launch_key_torchelastic(self, monkeypatch):
        from stable_pretraining.manager import _ddp_launch_key

        for v in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "MASTER_ADDR", "MASTER_PORT"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "abc123")
        assert _ddp_launch_key() == "elastic-abc123"

    def test_launch_key_local_ddp(self, monkeypatch):
        from stable_pretraining.manager import _ddp_launch_key

        for v in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "TORCHELASTIC_RUN_ID"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        key = _ddp_launch_key()
        assert key.startswith("local-127.0.0.1-29500-")

    def test_launch_key_none_for_single_process(self, monkeypatch):
        from stable_pretraining.manager import _ddp_launch_key

        for v in (
            "SLURM_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "TORCHELASTIC_RUN_ID",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            monkeypatch.delenv(v, raising=False)
        assert _ddp_launch_key() is None

    # -- end-to-end: rank-0 publishes, rank-N adopts -------------------------

    def test_rank_zero_publishes_handoff_after_fresh_dir(self, cache_dir, monkeypatch):
        # Force a known launch_key + rank 0.
        monkeypatch.setenv("SLURM_JOB_ID", "777")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        self._set_rank(monkeypatch, 0)

        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        assert run_dir is not None and run_dir.is_dir()

        handoff = cache_dir / ".rank_handoff" / "slurm-777"
        assert handoff.is_file(), "rank-0 must publish handoff"
        assert handoff.read_text().strip() == str(run_dir)

    def test_rank_n_adopts_handoff(self, cache_dir, monkeypatch):
        """Rank 1 reads rank-0's published path instead of computing its own."""
        # Pre-populate the handoff file as if rank 0 already wrote it.
        published = cache_dir / "runs" / "20260101" / "120000" / "rank0uuid1234"
        published.mkdir(parents=True)
        ho_dir = cache_dir / ".rank_handoff"
        ho_dir.mkdir(parents=True)
        (ho_dir / "slurm-777").write_text(str(published))

        # Pretend we're rank 1 of the same launch.
        monkeypatch.setenv("SLURM_JOB_ID", "777")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        self._set_rank(monkeypatch, 1)

        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        assert run_dir == published

    def test_rank_n_falls_back_on_timeout(self, cache_dir, monkeypatch):
        """Rank-N falls back to local resolution if rank-0 never publishes.

        If rank-0 crashed before publishing, rank-N must fall back rather than
        block forever. Data integrity is preserved because only rank-0 writes
        via ``@rank_zero_only`` loggers.
        """
        import stable_pretraining.manager as mgr_mod

        monkeypatch.setattr(mgr_mod, "_RANK_HANDOFF_TIMEOUT_S", 0.2)
        monkeypatch.setenv("SLURM_JOB_ID", "777")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        self._set_rank(monkeypatch, 1)

        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        # Returned a real dir despite no handoff (graceful fallback).
        assert run_dir is not None and run_dir.is_dir()

    def test_rank_n_ignores_dangling_handoff(self, cache_dir, monkeypatch):
        """Stale handoff (pointing at a vanished dir) is ignored.

        A handoff file from a crashed prior launch may point at a non-existent
        directory; rank-N keeps polling until a valid pointer arrives or the
        timeout elapses, then falls back to local resolution.
        """
        import stable_pretraining.manager as mgr_mod

        monkeypatch.setattr(mgr_mod, "_RANK_HANDOFF_TIMEOUT_S", 0.3)
        # Stale pointer to a directory that doesn't exist.
        ho_dir = cache_dir / ".rank_handoff"
        ho_dir.mkdir(parents=True)
        (ho_dir / "slurm-777").write_text(str(cache_dir / "vanished"))

        monkeypatch.setenv("SLURM_JOB_ID", "777")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        self._set_rank(monkeypatch, 1)

        manager = self._make_manager()
        # Falls back to local resolution rather than chasing the dangling ptr.
        run_dir = manager._resolve_run_dir()
        assert run_dir is not None and run_dir.is_dir()
        assert "vanished" not in str(run_dir)

    def test_polling_picks_up_handoff_written_after_wait_starts(self, cache_dir):
        """Rank-N's poll picks up a handoff that appears mid-wait.

        Rank-N starts polling on an empty dir; rank-0 publishes shortly after;
        rank-N must return the published path before timeout.

        This bypasses ``Manager.__call__`` entirely — pure-function semantics
        of the helpers — so it avoids the cross-thread ``os.environ`` mess
        that real DDP doesn't have (each rank is a separate process).
        """
        import threading

        manager = self._make_manager()
        target = cache_dir / "runs" / "20260101" / "120000" / "publishedID"
        target.mkdir(parents=True)

        # Run wait in a background thread; publish from main after a short delay.
        result = {}

        def poll():
            result["adopted"] = manager._wait_for_rank_zero_handoff(
                cache_dir, "slurm-test"
            )

        t = threading.Thread(target=poll)
        t.start()
        time.sleep(0.1)  # rank-N is now polling
        manager._publish_rank_zero_handoff(cache_dir, "slurm-test", target)
        t.join(timeout=5)

        assert result.get("adopted") == target

    def test_publish_is_atomic(self, cache_dir):
        """Handoff is written via atomic temp+replace.

        Rank-0 writes the handoff file via temp+``replace`` so a rank-N reader
        never sees a partially-written file.
        """
        manager = self._make_manager()
        target = cache_dir / "runs" / "20260101" / "120000" / "atomicID"
        target.mkdir(parents=True)
        manager._publish_rank_zero_handoff(cache_dir, "slurm-test", target)
        handoff = cache_dir / ".rank_handoff" / "slurm-test"
        assert handoff.is_file()
        # No leftover .tmp file from the rename trick.
        assert not handoff.with_name(handoff.name + ".tmp").exists()
        assert handoff.read_text() == str(target)

    def test_single_process_skips_handoff(self, cache_dir, monkeypatch):
        """No DDP env vars → ``_ddp_launch_key`` is None → no handoff file."""
        for v in (
            "SLURM_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "TORCHELASTIC_RUN_ID",
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "LOCAL_RANK",
            "SLURM_PROCID",
        ):
            monkeypatch.delenv(v, raising=False)

        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        assert run_dir is not None and run_dir.is_dir()
        assert not (cache_dir / ".rank_handoff").exists()


# ============================================================================
# Manager._inject_run_dir_into_trainer_config
# ============================================================================


class TestInjectRunDir:
    """Tests for injecting run_dir into the config tree."""

    def test_injects_into_dictconfig(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "logger": False,
            }
        )
        manager = Manager(
            trainer=cfg,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        manager._inject_run_dir_into_trainer_config(run_dir)
        assert manager.trainer.default_root_dir == str(run_dir)

    def test_overrides_existing_default_root_dir(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "default_root_dir": "/old/dir",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "logger": False,
            }
        )
        manager = Manager(
            trainer=cfg,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        manager._inject_run_dir_into_trainer_config(run_dir)
        assert manager.trainer.default_root_dir == str(run_dir)

    def test_warns_for_prebuilt_trainer(self, tmp_path):
        trainer = BoringTrainer(enable_checkpointing=False, logger=False)
        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        # Should not crash — just warn
        manager._inject_run_dir_into_trainer_config(tmp_path / "run")


# ============================================================================
# Manager._resolve_load_path — decoupled load logic
# ============================================================================


# ``Manager._resolve_load_path`` is now covered in test_slurm_index.py
# (see ``TestResolveLoadPath`` there) — the new behavior matrix is:
# fresh run + user ckpt_path → load user path with user weights_only;
# SLURM requeue → forced load of <run_dir>/checkpoints/last.ckpt with
# weights_only=False; user ckpt_path under requeue is ignored with a
# warning.


# ============================================================================
# Manager._configure_cache_dir_checkpointing
# ============================================================================


class TestConfigureCacheDirCheckpointing:
    """Tests for configuring cache_dir-based checkpointing."""

    def _make_manager_with_trainer(self, tmp_path):
        """Create a Manager with an instantiated trainer and run_dir."""
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer
        manager._run_dir = tmp_path / "run"
        manager._run_dir.mkdir()
        return manager

    def test_adds_model_checkpoint(self, tmp_path):
        manager = self._make_manager_with_trainer(tmp_path)
        initial_count = len(manager._trainer.callbacks)
        manager._configure_cache_dir_checkpointing()

        mc_callbacks = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        assert len(mc_callbacks) >= 1
        assert len(manager._trainer.callbacks) == initial_count + 1

    def test_saves_to_run_dir_checkpoints(self, tmp_path):
        manager = self._make_manager_with_trainer(tmp_path)
        manager._configure_cache_dir_checkpointing()

        mc = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ][-1]
        assert mc.dirpath == str(manager._run_dir / "checkpoints")

    def test_no_requeue_checkpoint_when_disabled(self, tmp_path):
        """spt.set(requeue_checkpoint=False) prevents the requeue checkpoint."""
        spt_set(requeue_checkpoint=False)
        manager = self._make_manager_with_trainer(tmp_path)
        manager._configure_cache_dir_checkpointing()

        mc_callbacks = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        # No requeue "last" should be added
        assert not any(cb.filename == "last" for cb in mc_callbacks)

    def test_always_adds_requeue_checkpoint(self, tmp_path):
        """Even with user's ModelCheckpoint, a requeue 'last' checkpoint is always added."""
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        run_dir = tmp_path / "run"
        save_dir = run_dir / "checkpoints"
        save_dir.mkdir(parents=True)

        user_mc = ModelCheckpoint(
            dirpath=str(save_dir), filename="best", monitor="val_loss"
        )
        trainer.callbacks.append(user_mc)

        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer
        manager._run_dir = run_dir

        manager._configure_cache_dir_checkpointing()

        mc_callbacks = [
            cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        # User's "best" + our requeue "last"
        assert len(mc_callbacks) == 2
        filenames = {cb.filename for cb in mc_callbacks}
        assert "best" in filenames
        assert "last" in filenames

    def test_redirects_user_checkpoint_to_run_dir(self, tmp_path):
        """User's ModelCheckpoint with custom dirpath gets redirected."""
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        run_dir = tmp_path / "run"
        (run_dir / "checkpoints").mkdir(parents=True)

        user_mc = ModelCheckpoint(
            dirpath="/some/other/path",
            filename="best-{epoch}",
            monitor="val_loss",
            mode="min",
        )
        trainer.callbacks.append(user_mc)

        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer
        manager._run_dir = run_dir

        manager._configure_cache_dir_checkpointing()

        # dirpath redirected, but filename/monitor/mode preserved
        assert user_mc.dirpath == str(run_dir / "checkpoints")
        assert user_mc.filename == "best-{epoch}"
        assert user_mc.monitor == "val_loss"

    def test_redirects_multiple_checkpoints(self, tmp_path):
        """All ModelCheckpoint callbacks get redirected, plus requeue is added."""
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        run_dir = tmp_path / "run"
        (run_dir / "checkpoints").mkdir(parents=True)

        mc1 = ModelCheckpoint(dirpath="/path/a", filename="every-epoch")
        mc2 = ModelCheckpoint(dirpath="/path/b", filename="best", monitor="val_loss")
        trainer.callbacks.extend([mc1, mc2])

        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer
        manager._run_dir = run_dir

        manager._configure_cache_dir_checkpointing()

        expected = str(run_dir / "checkpoints")
        assert mc1.dirpath == expected
        assert mc2.dirpath == expected
        # 2 user + 1 requeue
        mc_count = sum(isinstance(cb, ModelCheckpoint) for cb in trainer.callbacks)
        assert mc_count == 3
        filenames = {
            cb.filename for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)
        }
        assert filenames == {"every-epoch", "best", "last"}


# ============================================================================
# _RunDirCallback
# ============================================================================


class TestRunDirCallback:
    """Tests for the run_dir callback wiring."""

    def test_persists_run_dir_in_checkpoint(self):
        cb = _RunDirCallback("/some/path")
        checkpoint = {}
        cb.on_save_checkpoint(None, None, checkpoint)
        assert checkpoint["spt_run_dir"] == "/some/path"

    def test_stores_as_string(self):
        cb = _RunDirCallback(str(Path("/a/b/c")))
        checkpoint = {}
        cb.on_save_checkpoint(None, None, checkpoint)
        assert isinstance(checkpoint["spt_run_dir"], str)


# ============================================================================
# Manager._warn_hydra_conflicts (static method)
# ============================================================================


class TestHydraConflictWarnings:
    """Tests for Hydra config conflict warnings."""

    def test_no_crash_without_hydra(self):
        Manager._warn_hydra_conflicts()


# ============================================================================
# Manager.save_checkpoint with run_dir
# ============================================================================


class TestSaveCheckpointWithRunDir:
    """Tests for saving checkpoints under run_dir."""

    def test_default_path_uses_run_dir(self, tmp_path):
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer
        manager._run_dir = tmp_path / "my_run"
        (manager._run_dir / "checkpoints").mkdir(parents=True)

        saved_path = None

        def mock_save(path):
            nonlocal saved_path
            saved_path = path

        trainer.save_checkpoint = mock_save
        manager.save_checkpoint(verbose=False)
        assert saved_path is not None
        assert "my_run" in saved_path
        assert "checkpoints" in saved_path

    def test_default_path_without_run_dir(self, tmp_path):
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer

        saved_path = None

        def mock_save(path):
            nonlocal saved_path
            saved_path = path

        trainer.save_checkpoint = mock_save
        manager.save_checkpoint(verbose=False)
        assert saved_path is not None
        assert "checkpoint.ckpt" in saved_path


# ============================================================================
# Integration: Manager.__call__ with cache_dir
# ============================================================================


class TestManagerCallWithCacheDir:
    """Tests for Manager invocation with cache_dir wiring."""

    def test_full_flow_with_config_trainer(self, cache_dir, monkeypatch):
        """Manager.__call__() creates run_dir, injects into trainer, sets up checkpointing."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        def mock_fit(self_trainer, module, **kwargs):
            pass

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        # run_dir exists
        assert hasattr(manager, "_run_dir")
        assert manager._run_dir.is_dir()
        assert str(manager._run_dir).startswith(str(cache_dir))

        # ckpt_path is untouched (user didn't pass one)
        assert manager.ckpt_path is None

        # _RunDirCallback was added
        assert any(isinstance(cb, _RunDirCallback) for cb in manager._trainer.callbacks)

        # ModelCheckpoint saving to run_dir/checkpoints/ was added
        mc_callbacks = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        assert any(
            cb.dirpath == str(manager._run_dir / "checkpoints") for cb in mc_callbacks
        )

        # trainer.default_root_dir points to run_dir
        assert manager._trainer.default_root_dir == str(manager._run_dir)

    def test_cache_dir_none_preserves_old_behavior(self, tmp_path, monkeypatch):
        """When cache_dir is explicitly None, no run_dir is created."""
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        get_config()._cache_dir = None
        assert get_config().cache_dir is None

        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
        )
        monkeypatch.setattr(manager, "init_and_sync_wandb", lambda: None)
        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        def mock_fit(self_trainer, module, **kwargs):
            pass

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        assert not hasattr(manager, "_run_dir")
        assert manager.ckpt_path is None

    def test_user_ckpt_path_not_overridden_by_cache_dir(self, cache_dir, monkeypatch):
        """User's ckpt_path stays untouched — used for load only."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

        user_ckpt = cache_dir / "custom" / "my_model.ckpt"
        # ckpt_path validation requires the file to exist at __init__.
        user_ckpt.parent.mkdir(parents=True, exist_ok=True)
        user_ckpt.touch()

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
            ckpt_path=str(user_ckpt),
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        def mock_fit(self_trainer, module, **kwargs):
            pass

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        # ckpt_path is exactly what user passed (resolved)
        assert manager.ckpt_path == user_ckpt.resolve()
        # But checkpoints are saved to run_dir, NOT to user_ckpt.parent
        mc_callbacks = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        assert any(
            cb.dirpath == str(manager._run_dir / "checkpoints") for cb in mc_callbacks
        )

    def test_run_meta_written_for_requeue(self, cache_dir, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        def mock_fit(self_trainer, module, **kwargs):
            pass

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        meta_path = manager._run_dir / _RUN_META_FILENAME
        assert meta_path.is_file()
        meta = json.loads(meta_path.read_text())
        assert Path(meta["run_dir"]) == manager._run_dir

    def test_user_ckpt_path_forwarded_with_fresh_run_dir(self, cache_dir, monkeypatch):
        """User ckpt_path is forwarded to ``fit`` AND a FRESH run_dir is used.

        Strategy-1-style sidecar restoration is gone: outside of a SLURM
        requeue (RESTART_COUNT≥1) every invocation creates a new uuid'd
        run_dir, regardless of where the user's ckpt_path lives.
        """
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

        prev_run_dir = cache_dir / "runs" / "20260101" / "120000" / "prev12345678"
        ckpt_dir = prev_run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "last.ckpt").touch()
        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
            ckpt_path=str(ckpt_dir / "last.ckpt"),
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        captured_kwargs = {}

        def mock_fit(self_trainer, module, **kwargs):
            captured_kwargs.update(kwargs)

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        # User ckpt_path forwarded to fit verbatim.
        assert captured_kwargs["ckpt_path"] == str((ckpt_dir / "last.ckpt").resolve())
        # But the run_dir is FRESH — never the prev_run_dir.
        assert manager._run_dir != prev_run_dir
        assert manager._run_dir.is_dir()

    def test_slurm_requeue_no_ckpt_path_auto_loads(self, cache_dir, monkeypatch):
        """SLURM requeue auto-loads ``last.ckpt`` from the indexed dir.

        With RESTART_COUNT≥1 and a ``.slurm_index/<key>`` entry, the
        index-recorded run_dir is reused and its ``last.ckpt`` is picked up.
        """
        monkeypatch.setenv("SLURM_JOB_ID", "88888")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)

        # Simulate a previous run for this SLURM job
        prev_run_dir = cache_dir / "runs" / "20260101" / "100000" / "previd123456"
        ckpt_dir = prev_run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "last.ckpt").touch()
        (prev_run_dir / _RUN_META_FILENAME).write_text(
            json.dumps({"run_dir": str(prev_run_dir), "run_id": "previd123456"})
        )
        # The new mechanism: index file points at the prev run.
        idx = cache_dir / ".slurm_index"
        idx.mkdir()
        (idx / "88888").write_text(str(prev_run_dir))

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        # NO ckpt_path — this is the requeue scenario
        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        captured_kwargs = {}

        def mock_fit(self_trainer, module, **kwargs):
            captured_kwargs.update(kwargs)

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        # run_dir should be the restored prev_run_dir
        assert manager._run_dir == prev_run_dir
        # fit should have been called with the auto-detected checkpoint
        assert captured_kwargs["ckpt_path"] == str(ckpt_dir / "last.ckpt")
        # ckpt_path on manager stays None (user didn't set it)
        assert manager.ckpt_path is None

    def test_requeue_checkpoint_disabled_no_last_ckpt(self, cache_dir, monkeypatch):
        """With requeue_checkpoint=False, no 'last' ModelCheckpoint is added."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
        spt_set(requeue_checkpoint=False)

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        def mock_fit(self_trainer, module, **kwargs):
            pass

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        mc_callbacks = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        assert not any(cb.filename == "last" for cb in mc_callbacks)


# ============================================================================
# Callback path resolution tests
# ============================================================================


class TestCallbackPathResolution:
    """Tests for callback path resolution."""

    def test_hf_checkpoint_resolves_relative_save_dir(self):
        try:
            from stable_pretraining.callbacks.hf_models import (
                HuggingFaceCheckpointCallback,
            )
        except ImportError:
            pytest.skip("transformers not installed")

        cb = HuggingFaceCheckpointCallback(save_dir="hf_exports")
        assert not cb.save_dir.is_absolute()

    def test_online_writer_resolves_relative_path(self, tmp_path):
        from stable_pretraining.callbacks.writer import OnlineWriter

        run_dir = tmp_path / "run_dir"
        run_dir.mkdir()

        cb = OnlineWriter(names="test", path="outputs", during="train")
        assert not cb.path.is_absolute()

        mock_trainer = MagicMock()
        mock_trainer.default_root_dir = str(run_dir)
        cb.setup(mock_trainer, MagicMock(), stage="fit")

        assert cb.path.is_absolute()
        assert str(cb.path).startswith(str(run_dir))
        assert cb.path.is_dir()

    def test_online_writer_absolute_path_unchanged(self, tmp_path):
        from stable_pretraining.callbacks.writer import OnlineWriter

        abs_path = tmp_path / "my_abs_outputs"
        abs_path.mkdir()

        cb = OnlineWriter(names="test", path=str(abs_path), during="train")
        assert cb.path.is_absolute()

        mock_trainer = MagicMock()
        mock_trainer.default_root_dir = str(tmp_path / "different_dir")
        cb.setup(mock_trainer, MagicMock(), stage="fit")

        assert cb.path == abs_path


# ============================================================================
# Collision resistance
# ============================================================================


class TestNoCollisions:
    """Tests that independent runs do not collide on paths."""

    def test_two_managers_get_different_run_dirs(self, cache_dir, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

        m1 = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        m2 = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )

        d1 = m1._resolve_run_dir()
        d2 = m2._resolve_run_dir()
        assert d1 != d2
        assert d1.is_dir()
        assert d2.is_dir()
