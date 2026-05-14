"""Unit tests for the atomic checkpoint save helper."""

from unittest.mock import patch

import pytest
import torch

from stable_pretraining.utils.atomic_checkpoint import (
    _PATCHED_FLAG,
    atomic_torch_save,
    install_atomic_checkpoint_save,
)


@pytest.mark.unit
class TestAtomicTorchSave:
    """Regression tests for :func:`atomic_torch_save`."""

    def test_writes_round_trips(self, tmp_path):
        target = tmp_path / "ckpt.pt"
        payload = {"a": torch.arange(10), "b": "hello"}
        atomic_torch_save(payload, target)
        assert target.is_file()
        loaded = torch.load(target, weights_only=False)
        assert torch.equal(loaded["a"], payload["a"])
        assert loaded["b"] == "hello"

    def test_temp_file_is_in_target_dir(self, tmp_path):
        """The temp file must be a sibling of the target, not in /tmp.

        This is the whole point of the helper: same directory ⇒ same
        filesystem ⇒ ``os.replace`` is atomic. Putting the temp anywhere
        else falls back to a non-atomic cross-device copy.
        """
        target = tmp_path / "deep" / "nested" / "ckpt.pt"
        seen_dirs = []
        real_mkstemp = __import__("tempfile").mkstemp

        def spy(*args, **kwargs):
            seen_dirs.append(kwargs.get("dir"))
            return real_mkstemp(*args, **kwargs)

        with patch("tempfile.mkstemp", side_effect=spy):
            atomic_torch_save({"x": 1}, target)

        assert seen_dirs and all(d == str(target.parent) for d in seen_dirs), (
            f"temp file dirs were {seen_dirs}, expected only {target.parent}"
        )

    def test_kill_mid_rename_preserves_old_content(self, tmp_path):
        """If the rename step is killed, the previous content survives.

        We can't truly SIGKILL the process from a unit test, but we can
        simulate the equivalent by making :func:`os.replace` raise — the
        same code path that runs when a kill arrives between the temp
        write and the rename is the same code path that handles an
        exception there.
        """
        target = tmp_path / "ckpt.pt"
        atomic_torch_save({"version": "OLD"}, target)
        assert torch.load(target, weights_only=False) == {"version": "OLD"}

        boom = RuntimeError("simulated kill before rename")
        with patch("os.replace", side_effect=boom):
            with pytest.raises(RuntimeError, match="simulated"):
                atomic_torch_save({"version": "NEW"}, target)

        # Target still holds the OLD content.
        assert torch.load(target, weights_only=False) == {"version": "OLD"}
        # No orphaned temp files left in the target dir.
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
        assert leftovers == [], f"temp files leaked: {leftovers}"

    def test_creates_missing_parent_dir(self, tmp_path):
        target = tmp_path / "does" / "not" / "exist" / "ckpt.pt"
        atomic_torch_save({"x": 1}, target)
        assert target.is_file()


@pytest.mark.unit
class TestInstallAtomicCheckpointSave:
    """Make sure the Lightning class-level patch is wired correctly."""

    def test_patch_is_idempotent(self):
        from lightning.pytorch.plugins.io.torch_plugin import TorchCheckpointIO

        install_atomic_checkpoint_save()
        first = TorchCheckpointIO.save_checkpoint
        install_atomic_checkpoint_save()
        install_atomic_checkpoint_save()
        second = TorchCheckpointIO.save_checkpoint
        assert first is second, "subsequent installs must not replace the patch"
        assert getattr(second, _PATCHED_FLAG, False)

    def test_patched_save_round_trips_via_lightning(self, tmp_path):
        from lightning.pytorch.plugins.io.torch_plugin import TorchCheckpointIO

        install_atomic_checkpoint_save()
        target = tmp_path / "ckpt.pt"
        TorchCheckpointIO().save_checkpoint({"hello": "world"}, str(target))
        assert torch.load(target, weights_only=False) == {"hello": "world"}

    def test_patched_save_preserves_storage_options_error(self, tmp_path):
        """Lightning's API contract: ``storage_options`` must raise TypeError."""
        from lightning.pytorch.plugins.io.torch_plugin import TorchCheckpointIO

        install_atomic_checkpoint_save()
        with pytest.raises(TypeError, match="storage_options"):
            TorchCheckpointIO().save_checkpoint(
                {"x": 1}, str(tmp_path / "x.pt"), storage_options={"foo": 1}
            )

    def test_async_inner_inherits_patch(self, tmp_path):
        """Default ``AsyncCheckpointIO`` inner is ``TorchCheckpointIO``.

        When that class is patched, the async path picks up the atomic
        save automatically — no extra wiring needed. We invoke the inner
        directly here instead of going through the async executor (which
        would require a Lightning Trainer + a thread join in the test).
        """
        from lightning.pytorch.plugins.io.async_plugin import AsyncCheckpointIO
        from lightning.pytorch.plugins.io.torch_plugin import TorchCheckpointIO

        install_atomic_checkpoint_save()
        async_io = AsyncCheckpointIO()
        # Lightning sets the inner lazily — match its default if absent.
        if async_io.checkpoint_io is None:
            async_io.checkpoint_io = TorchCheckpointIO()

        target = tmp_path / "async_ckpt.pt"
        async_io.checkpoint_io.save_checkpoint({"k": "v"}, str(target))
        assert target.is_file()
        assert torch.load(target, weights_only=False) == {"k": "v"}
