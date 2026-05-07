"""Unit tests for the atomic SLURM index write in Manager.

Background: ``Manager`` records ``SLURM_JOB_ID[_TASK_ID] → run_dir`` in
``cache_dir/.slurm_index/<key>`` so a SLURM-requeued process (signalled
via ``SLURM_RESTART_COUNT >= 1``) can find its original run directory.

The previous implementation was a plain ``Path.write_text`` which could
produce a partial file on NFS hiccup or be lost entirely if the process
was killed mid-write — the next requeue would then fail with::

    RuntimeError: SLURM reports this is a requeue ... but no index file
    exists at ...

These tests validate the atomic write recipe (sibling temp +
:func:`os.fsync` + :func:`os.replace`) actually leaves a valid file on
disk, even when the rename step is interrupted.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


def _atomic_write_index(idx_dir: Path, slurm_key: str, run_dir: Path) -> Path:
    """Convenience wrapper that calls Manager's real helper.

    The cache_dir we pass is ``idx_dir.parent`` because Manager's helper
    derives the index dir as ``cache_dir / ".slurm_index"`` internally.
    """
    from stable_pretraining.manager import Manager

    cache_dir = idx_dir.parent
    # Make sure the helper sees the SLURM_JOB_ID we expect by
    # round-tripping through ``slurm_key``.  The helper reads
    # ``SLURM_JOB_ID[_TASK_ID]`` from the env, so set it for the call.
    job, _, task = slurm_key.partition("_")
    saved = (os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID"))
    os.environ["SLURM_JOB_ID"] = job
    if task:
        os.environ["SLURM_ARRAY_TASK_ID"] = task
    else:
        os.environ.pop("SLURM_ARRAY_TASK_ID", None)
    try:
        Manager._write_slurm_index(cache_dir, run_dir)
    finally:
        for k, v in zip(("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID"), saved):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return idx_dir / slurm_key


@pytest.mark.unit
class TestSlurmIndexWrite:
    """Atomic-write recipe used by Manager for the SLURM index file."""

    def test_basic_write_round_trips(self, tmp_path):
        """Plain happy path — file ends up where Manager will look for it."""
        cache_dir = tmp_path / "cache"
        run_dir = tmp_path / "runs" / "20260501" / "abc123"
        run_dir.mkdir(parents=True)
        slurm_key = "352145_75"

        idx_path = _atomic_write_index(cache_dir / ".slurm_index", slurm_key, run_dir)

        # Exact path the Manager lookup uses (cache_dir/.slurm_index/<key>).
        assert idx_path == cache_dir / ".slurm_index" / slurm_key
        assert idx_path.is_file()
        assert idx_path.read_text() == str(run_dir)

    def test_array_task_keys_are_distinct_files(self, tmp_path):
        """Sanity: 352145_7 and 352145_75 don't collide on disk."""
        idx_dir = tmp_path / ".slurm_index"
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        run_a.mkdir()
        run_b.mkdir()

        _atomic_write_index(idx_dir, "352145_7", run_a)
        _atomic_write_index(idx_dir, "352145_75", run_b)

        assert (idx_dir / "352145_7").read_text() == str(run_a)
        assert (idx_dir / "352145_75").read_text() == str(run_b)

    def test_replace_failure_during_first_write_cleans_up_temp(self, tmp_path):
        """OSError during ``os.replace`` on a *fresh* write (no prior entry).

        We can only hit the failure path on the first write — subsequent
        calls short-circuit because of the write-iff-missing rule. The
        helper swallows the OSError, logs a warning, and removes the
        sibling temp file so nothing leaks.
        """
        idx_dir = tmp_path / ".slurm_index"
        run = tmp_path / "run"
        run.mkdir()
        idx_dir.mkdir(parents=True)
        # Sanity: no prior entry, so the helper will run the full write.
        assert not (idx_dir / "352145_75").exists()

        with patch("os.replace", side_effect=OSError("EIO")):
            _atomic_write_index(idx_dir, "352145_75", run)  # no raise

        # Target was never created (rename failed).
        assert not (idx_dir / "352145_75").exists()
        # No leaked temp.
        leaked = [p.name for p in idx_dir.iterdir() if p.name.startswith(".")]
        assert leaked == [], f"temp files leaked: {leaked}"

    def test_no_index_dir_yet_creates_it(self, tmp_path):
        """First-ever write must auto-create ``.slurm_index/`` itself."""
        cache_dir = tmp_path / "cache"
        # cache_dir doesn't exist yet — helper should mkdir -p.
        run = tmp_path / "run"
        run.mkdir()
        from stable_pretraining.manager import Manager

        os.environ["SLURM_JOB_ID"] = "777"
        os.environ.pop("SLURM_ARRAY_TASK_ID", None)
        try:
            Manager._write_slurm_index(cache_dir, run)
        finally:
            os.environ.pop("SLURM_JOB_ID", None)

        assert (cache_dir / ".slurm_index" / "777").read_text() == str(run)

    def test_existing_entry_is_preserved(self, tmp_path):
        """Same-SLURM_JOB_ID requeue: the existing entry is authoritative.

        ``_write_slurm_index`` is "write iff missing" — calling it again
        with a different ``run_dir`` is a no-op. This protects requeue
        semantics: every invocation under the same SLURM key resolves
        to the same ``run_dir`` (the one written by the first call).
        """
        idx_dir = tmp_path / ".slurm_index"
        original = tmp_path / "original"
        attempted = tmp_path / "attempted_overwrite"
        original.mkdir()
        attempted.mkdir()

        _atomic_write_index(idx_dir, "999_0", original)
        _atomic_write_index(idx_dir, "999_0", attempted)  # no-op

        assert (idx_dir / "999_0").read_text() == str(original)


@pytest.mark.unit
class TestRunDirCallbackGuard:
    """Defensive on_train_start guard verifies / self-heals the SLURM index."""

    def _make(self, run_dir, cache_dir):
        from stable_pretraining.manager import _RunDirCallback

        return _RunDirCallback(str(run_dir), cache_dir=str(cache_dir))

    def test_guard_raises_when_index_missing(self, tmp_path, monkeypatch):
        """If the index is missing at on_train_start, the guard must KILL the run.

        Self-healing here would let a real regression in
        :meth:`Manager._resolve_run_dir` (failure to call
        :meth:`Manager._write_slurm_index` on some new branch) persist
        silently. The guard raises a ``RuntimeError`` with full
        diagnostic context instead, so the bug surfaces immediately.
        """
        monkeypatch.setenv("SLURM_JOB_ID", "352145")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "75")
        cache_dir = tmp_path / "cache"
        run_dir = tmp_path / "runs" / "abc"
        run_dir.mkdir(parents=True)
        idx_path = cache_dir / ".slurm_index" / "352145_75"
        assert not idx_path.exists()

        cb = self._make(run_dir, cache_dir)
        with pytest.raises(RuntimeError, match="SLURM-index guard FAILED"):
            cb.on_train_start(trainer=None, pl_module=None)

        # Guard does NOT write — the failure is preserved for diagnosis.
        assert not idx_path.exists()

    def test_guard_does_not_touch_existing_entry(self, tmp_path, monkeypatch):
        """Existing entry is authoritative — guard does NOT overwrite it.

        Within a single SLURM_JOB_ID the index is written once on the
        first invocation; subsequent requeues simply read it. Touching
        an existing entry would risk clobbering another session's
        record if SLURM ever recycled job IDs (it doesn't, but this
        keeps the invariant simple).
        """
        monkeypatch.setenv("SLURM_JOB_ID", "999")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        cache_dir = tmp_path / "cache"
        idx_path = cache_dir / ".slurm_index" / "999"
        idx_path.parent.mkdir(parents=True)
        idx_path.write_text("/some/recorded/dir")

        # Even if the current run_dir looks different from the recorded
        # one, the guard leaves the existing entry alone.
        diff_run = tmp_path / "runs" / "different"
        diff_run.mkdir(parents=True)
        cb = self._make(diff_run, cache_dir)
        cb.on_train_start(trainer=None, pl_module=None)
        assert idx_path.read_text() == "/some/recorded/dir"

    def test_guard_no_op_when_index_correct(self, tmp_path, monkeypatch):
        """Steady state: index already matches — guard reads, exits."""
        monkeypatch.setenv("SLURM_JOB_ID", "777")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        cache_dir = tmp_path / "cache"
        run_dir = tmp_path / "runs" / "x"
        run_dir.mkdir(parents=True)
        idx_path = cache_dir / ".slurm_index" / "777"
        idx_path.parent.mkdir(parents=True)
        idx_path.write_text(str(run_dir))
        mtime_before = idx_path.stat().st_mtime

        cb = self._make(run_dir, cache_dir)
        cb.on_train_start(trainer=None, pl_module=None)

        # File untouched (no rewrite — happy path).
        assert idx_path.stat().st_mtime == mtime_before
        assert idx_path.read_text() == str(run_dir)

    def test_guard_skips_outside_slurm(self, tmp_path, monkeypatch):
        """No SLURM env vars → no index file expected → no-op."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        cache_dir = tmp_path / "cache"
        run_dir = tmp_path / "runs" / "y"
        run_dir.mkdir(parents=True)
        cb = self._make(run_dir, cache_dir)
        cb.on_train_start(trainer=None, pl_module=None)
        # No index dir created at all.
        assert not (cache_dir / ".slurm_index").exists()

    def test_guard_skips_when_no_cache_dir(self, tmp_path, monkeypatch):
        """If Manager wasn't configured with a cache_dir, guard is a no-op."""
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        from stable_pretraining.manager import _RunDirCallback

        cb = _RunDirCallback(str(tmp_path / "run"), cache_dir=None)
        # Doesn't raise.
        cb.on_train_start(trainer=None, pl_module=None)


@pytest.mark.unit
class TestCkptPathValidation:
    """``Manager`` rejects relative or missing ``ckpt_path`` at __init__.

    The validation runs BEFORE any heavy setup (trainer/module/data
    instantiation), so we can pass placeholder configs — the failure
    fires immediately.
    """

    def test_relative_ckpt_path_raises(self):
        from stable_pretraining.manager import Manager

        with pytest.raises(ValueError, match="absolute"):
            Manager(
                trainer={},
                module={},
                data={},
                ckpt_path="relative/path.ckpt",
            )

    def test_missing_ckpt_path_raises(self, tmp_path):
        from stable_pretraining.manager import Manager

        bogus = tmp_path / "does-not-exist.ckpt"
        with pytest.raises(FileNotFoundError, match="no such file"):
            Manager(
                trainer={},
                module={},
                data={},
                ckpt_path=str(bogus),
            )

    def test_absolute_existing_ckpt_path_accepted(self, tmp_path):
        """Happy path: an absolute, existing checkpoint passes validation."""
        from stable_pretraining.manager import Manager

        ckpt = tmp_path / "real.ckpt"
        ckpt.write_text("dummy")
        # We expect Manager.__init__ to get past validation and then
        # error on the dummy trainer/module/data configs (which is fine —
        # the validation we care about happened first).
        try:
            Manager(trainer={}, module={}, data={}, ckpt_path=str(ckpt))
        except (ValueError, FileNotFoundError):
            pytest.fail("ckpt_path validation should accept abs+existing path")
        except Exception:
            pass  # downstream registration failure: not what we're testing


@pytest.mark.unit
class TestResolveLoadPath:
    """Behaviour matrix for ``Manager._resolve_load_path``.

    We test the resolver in isolation by constructing a Manager-like stub
    with just the attributes the method reads (``ckpt_path``,
    ``weights_only``). Avoids spinning up a full Trainer + Module.
    """

    class _Stub:
        ckpt_path = None
        weights_only = True
        _early_preempt_fallback = False

        # Bind the real methods so tests exercise production code, not
        # a re-implementation.
        from stable_pretraining.manager import Manager

        _resolve_load_path = Manager._resolve_load_path

    def test_fresh_no_ckpt(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        s = self._Stub()
        s.ckpt_path = None
        path, wo = s._resolve_load_path(tmp_path)
        assert path is None and wo is None

    def test_fresh_with_user_ckpt_uses_user_flag(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        ckpt = tmp_path / "user.ckpt"
        ckpt.write_text("dummy")
        s = self._Stub()
        s.ckpt_path = ckpt
        s.weights_only = True
        path, wo = s._resolve_load_path(tmp_path)
        assert path == str(ckpt)
        assert wo is True

    def test_fresh_user_can_disable_weights_only(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        ckpt = tmp_path / "user.ckpt"
        ckpt.write_text("dummy")
        s = self._Stub()
        s.ckpt_path = ckpt
        s.weights_only = False
        path, wo = s._resolve_load_path(tmp_path)
        assert path == str(ckpt) and wo is False

    def test_requeue_loads_last_ckpt_full_state(self, tmp_path, monkeypatch):
        """REQUEUE always loads ``last.ckpt`` with ``weights_only=False``.

        Forced regardless of the user-supplied flag.
        """
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        run_dir = tmp_path / "run"
        ckpts = run_dir / "checkpoints"
        ckpts.mkdir(parents=True)
        last = ckpts / "last.ckpt"
        last.write_text("dummy")

        s = self._Stub()
        s.ckpt_path = None
        s.weights_only = True  # user said weights-only…
        path, wo = s._resolve_load_path(run_dir)
        # …but on requeue we override to False.
        assert path == str(last)
        assert wo is False

    def test_requeue_overrides_user_ckpt_path(self, tmp_path, monkeypatch):
        """User-given ``ckpt_path`` is *ignored* on requeue."""
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        run_dir = tmp_path / "run"
        ckpts = run_dir / "checkpoints"
        ckpts.mkdir(parents=True)
        (ckpts / "last.ckpt").write_text("real")
        user_ckpt = tmp_path / "user_pretrain.ckpt"
        user_ckpt.write_text("pretrain")

        s = self._Stub()
        s.ckpt_path = user_ckpt
        path, wo = s._resolve_load_path(run_dir)
        # We pick last.ckpt, NOT user_ckpt.
        assert path == str(ckpts / "last.ckpt")
        assert wo is False

    def test_requeue_without_last_ckpt_raises(self, tmp_path, monkeypatch):
        """REQUEUE but no last.ckpt → loud RuntimeError, not silent restart."""
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        s = self._Stub()
        with pytest.raises(RuntimeError, match="REQUEUE but no last.ckpt"):
            s._resolve_load_path(run_dir)


@pytest.mark.unit
class TestEarlyPreemptFallback:
    """Requeue + missing index: only fall through if NO orphan run_dir exists.

    A SLURM ``RESTART_COUNT >= 1`` with the index file absent has two
    possible causes:

    * **Early preempt** — prior task was killed before reaching
      ``Manager.__init__`` (typical: cluster-wide eviction during the
      submitit pickle load). No artefact was written. Correct response:
      treat as fresh run.
    * **Partial write** — a prior attempt got far enough to mkdir+stamp
      a ``run_dir`` but died before writing the index. There IS an orphan
      on disk. Correct response: raise.

    Tests below pin this down by stamping (or not stamping) a matching
    ``slurm_job_id`` into ``run_meta.json``.
    """

    def _stamp_run_meta(self, run_dir, job_id, task_id=None):
        import json as _json

        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {"run_dir": str(run_dir), "run_id": run_dir.name}
        meta["slurm_job_id"] = str(job_id)
        if task_id is not None:
            meta["slurm_array_task_id"] = str(task_id)
        (run_dir / "run_meta.json").write_text(_json.dumps(meta))

    def test_find_orphans_returns_empty_when_runs_dir_missing(self, tmp_path):
        from stable_pretraining.manager import Manager

        assert Manager._find_orphans_for_slurm_key(tmp_path, "352145") == []

    def test_find_orphans_picks_up_matching_job_id(self, tmp_path):
        from stable_pretraining.manager import Manager

        cache_dir = tmp_path / "cache"
        orphan = cache_dir / "runs" / "20260101" / "120000" / "abc1"
        self._stamp_run_meta(orphan, "352145")
        # Unrelated run from a different job — must NOT be returned.
        other = cache_dir / "runs" / "20260101" / "120000" / "abc2"
        self._stamp_run_meta(other, "999999")

        hits = Manager._find_orphans_for_slurm_key(cache_dir, "352145")
        assert hits == [orphan]

    def test_find_orphans_distinguishes_array_task_ids(self, tmp_path):
        """Same JOB_ID but different ARRAY_TASK_ID is a *different* session."""
        from stable_pretraining.manager import Manager

        cache_dir = tmp_path / "cache"
        ours = cache_dir / "runs" / "20260101" / "120000" / "abc1"
        self._stamp_run_meta(ours, "352145", task_id="3")
        sibling = cache_dir / "runs" / "20260101" / "120000" / "abc2"
        self._stamp_run_meta(sibling, "352145", task_id="4")

        hits = Manager._find_orphans_for_slurm_key(cache_dir, "352145_3")
        assert hits == [ours]

    def test_find_orphans_skips_corrupt_run_meta(self, tmp_path):
        """A truncated/invalid run_meta.json must not crash the scan."""
        from stable_pretraining.manager import Manager

        cache_dir = tmp_path / "cache"
        broken = cache_dir / "runs" / "20260101" / "120000" / "broken"
        broken.mkdir(parents=True)
        (broken / "run_meta.json").write_text("not valid json {")
        good = cache_dir / "runs" / "20260101" / "120000" / "good"
        self._stamp_run_meta(good, "352145")

        hits = Manager._find_orphans_for_slurm_key(cache_dir, "352145")
        assert hits == [good]

    def test_resolve_run_dir_falls_through_when_no_orphan(self, tmp_path, monkeypatch):
        """True early-preempt: requeue env, no index, no orphans → fresh run."""
        from stable_pretraining._config import set as spt_set
        from stable_pretraining.manager import Manager
        from stable_pretraining.tests.utils import (
            BoringDataModule,
            BoringModule,
            BoringTrainer,
        )

        cache_dir = tmp_path / "cache"
        spt_set(cache_dir=str(cache_dir))
        monkeypatch.setenv("SLURM_JOB_ID", "352145")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)

        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        run_dir = manager._resolve_run_dir()
        # Fell through to fresh-run: brand new uuid'd dir, index now written.
        assert run_dir is not None and run_dir.is_dir()
        assert (cache_dir / ".slurm_index" / "352145").is_file()
        assert manager._early_preempt_fallback is True

    def test_resolve_run_dir_raises_when_orphan_present(self, tmp_path, monkeypatch):
        """Partial-write scenario: orphan exists → must raise, not fall through."""
        from stable_pretraining._config import set as spt_set
        from stable_pretraining.manager import Manager
        from stable_pretraining.tests.utils import (
            BoringDataModule,
            BoringModule,
            BoringTrainer,
        )

        cache_dir = tmp_path / "cache"
        spt_set(cache_dir=str(cache_dir))
        monkeypatch.setenv("SLURM_JOB_ID", "352145")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)

        # Stamp an orphan run_dir for this SLURM_JOB_ID.
        orphan = cache_dir / "runs" / "20260101" / "120000" / "abc1"
        self._stamp_run_meta(orphan, "352145")

        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        with pytest.raises(
            RuntimeError, match="already stamped with this SLURM_JOB_ID"
        ):
            manager._resolve_run_dir()
        assert manager._early_preempt_fallback is False

    def test_resolve_load_path_after_early_preempt_treats_as_fresh(
        self, tmp_path, monkeypatch
    ):
        """After the early-preempt fallback, ``_resolve_load_path`` skips the
        requeue last.ckpt requirement and returns the user's ckpt_path
        (or None).
        """
        from stable_pretraining.manager import Manager
        from stable_pretraining.tests.utils import (
            BoringDataModule,
            BoringModule,
            BoringTrainer,
        )

        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        run_dir = tmp_path / "fresh_run"
        run_dir.mkdir()  # No checkpoints/last.ckpt under here.

        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._early_preempt_fallback = True  # simulate post-fallback state.
        # Must not raise — under fallback we behave like a fresh run.
        path, weights_only = manager._resolve_load_path(run_dir)
        assert path is None and weights_only is None


@pytest.mark.unit
class TestManagerCodepath:
    """Smoke test for ``manager._slurm_session_key`` after the atomic patch.

    Imports Manager to confirm our edit didn't introduce a SyntaxError /
    NameError, and verifies the SLURM-key derivation still yields the
    exact string the index lookup uses.
    """

    def test_slurm_session_key_array(self, monkeypatch):
        from stable_pretraining.manager import _slurm_session_key

        monkeypatch.setenv("SLURM_JOB_ID", "352145")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "75")
        assert _slurm_session_key() == "352145_75"

    def test_slurm_session_key_no_array(self, monkeypatch):
        from stable_pretraining.manager import _slurm_session_key

        monkeypatch.setenv("SLURM_JOB_ID", "999")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        assert _slurm_session_key() == "999"

    def test_slurm_session_key_outside_slurm(self, monkeypatch):
        from stable_pretraining.manager import _slurm_session_key

        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        assert _slurm_session_key() is None
