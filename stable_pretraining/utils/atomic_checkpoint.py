"""Crash-safe checkpoint saving for Lightning.

Symptom we're solving: a checkpoint write killed by SIGTERM/SIGKILL/SLURM-preempt
leaves ``last.ckpt`` corrupted. Reading it later fails with::

    RuntimeError: PytorchStreamReader failed reading zip archive: failed
    finding central directory. ... your checkpoint file is corrupted ...

Why Lightning's built-in path doesn't protect against this on NFS:

* :func:`lightning.fabric.utilities.cloud_io._atomic_save` uses an
  ``fsspec`` transaction.
* ``fsspec``'s ``LocalFileOpener`` creates the temp file via
  :func:`tempfile.mkstemp` (no ``dir=`` argument), so the temp lands in
  the system temp dir (``$TMPDIR`` or ``/tmp``) — a *local* filesystem.
* The target lives on NFS. ``commit()`` calls :func:`shutil.move`, which
  falls back to **non-atomic** ``copy2 + unlink`` across filesystems.
* If killed during the cross-device copy, the target is left half-written.

Fix: a small helper :func:`atomic_torch_save` that puts the temp file in
the *same directory* as the target so :func:`os.replace` is atomic, plus
a class-level monkey-patch on :class:`TorchCheckpointIO.save_checkpoint`
installed from spt's deferred init. The patch covers every save path
that ultimately resolves to ``TorchCheckpointIO``:

* Default sync (``trainer.save_checkpoint(...)``,
  :class:`~lightning.pytorch.callbacks.ModelCheckpoint`).
* :class:`~lightning.pytorch.plugins.io.async_plugin.AsyncCheckpointIO`,
  whose default inner is :class:`TorchCheckpointIO` — patched via
  inheritance / instance dispatch.
* Any user subclass of :class:`TorchCheckpointIO` that doesn't itself
  override ``save_checkpoint``.

Subclasses that *do* override ``save_checkpoint`` (e.g. a custom S3
plugin) keep their behaviour — those paths have their own atomicity
story we shouldn't second-guess.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Union

import torch
from loguru import logger as logging


def atomic_torch_save(checkpoint: Any, path: Union[str, os.PathLike]) -> None:
    """Serialise ``checkpoint`` to ``path`` atomically.

    Four-step protocol:

    1. ``torch.save`` to a RAM buffer. No file I/O on the target yet — a
       kill here doesn't touch ``path``.
    2. Write the bytes to a sibling temp file in the *same* directory as
       ``path``. Same directory ⇒ same filesystem ⇒ atomic rename.
    3. ``fsync`` the temp file so the bytes are durable on disk before the
       rename — protects against a power loss / kernel crash window.
    4. :func:`os.replace` the temp into the target. Atomic on every
       filesystem we care about, including NFS: the final ``path`` either
       points at the previous good content or the new content, never a
       partial write.

    A process killed (e.g. SLURM ``SIGTERM`` followed by ``os._exit(1)``)
    between steps 2 and 4 leaves a stale ``.<name>.<random>.tmp`` file
    behind but the target still points at the previous good checkpoint.

    Args:
        checkpoint: Any object accepted by :func:`torch.save`.
        path: Destination file path. Parent directory is created if missing.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"[atomic_save] → {target}")

    # Step 1: serialise to RAM. No I/O on the target yet.
    t0 = time.perf_counter()
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    n_bytes = buf.getbuffer().nbytes
    t_ser = time.perf_counter() - t0
    logging.debug(
        f"[atomic_save]  serialised {n_bytes / 1024**2:.1f} MiB in {t_ser:.2f}s"
    )

    # Step 2 & 3: temp file in the SAME directory + fsync.
    t1 = time.perf_counter()
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    logging.debug(f"[atomic_save]  temp = {tmp_path}")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(buf.getvalue())
            f.flush()
            os.fsync(f.fileno())
        t_write = time.perf_counter() - t1
        logging.debug(
            f"[atomic_save]  wrote + fsynced {n_bytes / 1024**2:.1f} MiB in {t_write:.2f}s"
        )

        # Step 4: atomic rename. If a kill happens BEFORE this line, the
        # target still holds the previous good checkpoint.
        t2 = time.perf_counter()
        os.replace(tmp_path, str(target))
        t_rename = time.perf_counter() - t2
        logging.info(
            f"[atomic_save] ✓ {target.name} saved "
            f"({n_bytes / 1024**2:.1f} MiB, "
            f"serialise={t_ser:.2f}s write={t_write:.2f}s rename={t_rename * 1000:.1f}ms)"
        )
    except BaseException as e:
        # Clean up the orphaned temp; leave the original target alone.
        logging.error(
            f"[atomic_save] ✗ {target.name} FAILED ({type(e).__name__}: {e}) "
            f"— removing temp {tmp_path}; previous checkpoint at {target} is intact"
        )
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_save_checkpoint(self, checkpoint, path, storage_options=None):
    """Patched body for :class:`TorchCheckpointIO.save_checkpoint`.

    ``self`` is the :class:`TorchCheckpointIO` instance; we ignore it
    because :func:`atomic_torch_save` doesn't need any plugin state.
    """
    if storage_options is not None:
        # Match Lightning's API contract — TorchCheckpointIO never accepted
        # ``storage_options`` and signals so explicitly.
        raise TypeError(
            "`Trainer.save_checkpoint(..., storage_options=...)` with "
            "`storage_options` arg is not supported for "
            f"`{type(self).__name__}`. Please implement your custom "
            "`CheckpointIO` to define how you'd like to use `storage_options`."
        )
    atomic_torch_save(checkpoint, path)


_PATCHED_FLAG = "_spt_atomic_patched"


def install_atomic_checkpoint_save() -> None:
    """Replace :class:`TorchCheckpointIO.save_checkpoint` with the atomic version.

    Called from spt's deferred init so the patch is in place by the time
    any user code runs. Idempotent — safe to call multiple times.
    """
    # Imported here (not at module top) so that ``import
    # stable_pretraining.utils.atomic_checkpoint`` doesn't drag Lightning
    # in for users who only want :func:`atomic_torch_save`.
    from lightning.pytorch.plugins.io.torch_plugin import (  # noqa: PLC0415
        TorchCheckpointIO,
    )

    if getattr(TorchCheckpointIO.save_checkpoint, _PATCHED_FLAG, False):
        return
    setattr(_atomic_save_checkpoint, _PATCHED_FLAG, True)
    TorchCheckpointIO.save_checkpoint = _atomic_save_checkpoint
    logging.info(
        "[atomic_save] installed crash-safe checkpoint plugin "
        "(write to sibling .tmp + fsync + atomic rename)"
    )
