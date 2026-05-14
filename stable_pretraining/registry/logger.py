# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning logger for the filesystem-backed run registry.

:class:`RegistryLogger` is a thin subclass of Lightning's
:class:`~lightning.pytorch.loggers.CSVLogger`.  It writes the standard
CSV + hparams artifacts **and** an indexable ``sidecar.json``, a
fast-readable ``summary.json`` (per-metric stats), and a ``heartbeat``
file in the run directory.

Nothing in the training path touches SQLite or a network server:

* ``log_hyperparams`` → CSV hparams + sidecar snapshot.
* ``log_metrics``     → CSV metrics row + per-metric stats accumulator
                        (last / min / max / count) + heartbeat touch.
* ``save``            → CSV flush + sidecar rewrite + summary.json
                        rewrite (both atomic). Lightning calls this at
                        epoch boundaries and on the flush cadence.
* ``finalize``        → terminal status in sidecar + final summary flush.
* ``after_save_checkpoint`` → ``checkpoint_path`` in sidecar.

A separate scanner (see :mod:`stable_pretraining.registry._scanner`)
turns sidecars into a fast-queryable SQLite cache.  Deleting that cache
is harmless — rerun ``spt registry scan --full`` to rebuild.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import csv as _csv

from loguru import logger as logging
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.csv_logs import ExperimentWriter as _LightningCSVWriter
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from . import _sidecar


class _AppendingExperimentWriter(_LightningCSVWriter):
    """CSV writer that preserves any existing ``metrics.csv`` on init.

    Lightning's stock ``_ExperimentWriter`` deletes the file in
    ``_check_log_dir_exists`` whenever the log dir is non-empty. That makes
    SLURM-preempt-and-requeue runs lose all prior training history because
    the resumed process re-creates the writer, which truncates the existing
    file before the first append.

    This subclass skips the deletion. To avoid the parent's ``new_keys``
    detection from rewriting the file with a header collision on first save,
    we also bootstrap ``metrics_keys`` from the existing CSV header — so
    the parent only triggers a header-rewrite when the schema *actually*
    changes (e.g. a brand-new metric appears mid-run), not just because
    its in-memory ``metrics_keys`` is empty after a fresh process start.
    """

    def _check_log_dir_exists(self) -> None:  # type: ignore[override]
        # Intentional no-op: do not delete prior metrics.csv on resume.
        return

    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir=log_dir)
        # Bootstrap metrics_keys from existing header (if any) so the parent's
        # `new_keys = current_keys - metrics_keys` doesn't mistake a fresh
        # process start for a schema change.
        try:
            if self._fs.isfile(self.metrics_file_path):
                with self._fs.open(self.metrics_file_path, "r", newline="") as f:
                    reader = _csv.reader(f)
                    header = next(reader, None)
                if header:
                    self.metrics_keys = list(header)
        except Exception:
            # Bootstrap is best-effort; if it fails the parent's rewrite path
            # will still preserve old rows via _rewrite_with_new_header.
            pass

    def _record_new_keys(self) -> set:  # type: ignore[override]
        """Append new keys to ``metrics_keys`` *without sorting*.

        Lightning's parent calls ``self.metrics_keys.sort()`` after each
        update, which silently reorders columns relative to the existing
        on-disk CSV header. When a resumed process appends rows in the
        sorted order while the file's header retains insertion order, the
        column values get scrambled. Preserving insertion order keeps
        appended rows aligned with the original header.
        """
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        # Append in a stable order (sorted among the new keys only) so two
        # appends in the same process are deterministic, but DON'T touch
        # the existing prefix.
        for k in sorted(new_keys):
            self.metrics_keys.append(k)
        return new_keys


class RegistryLogger(CSVLogger):
    """CSV logger with a filesystem-indexable sidecar.

    The sidecar is an atomically-rewritten JSON file that captures the
    run's hparams, latest metric values (``summary``), status, and
    checkpoint path.  It is the source of truth for the registry
    scanner.

    Args:
        run_dir: Directory this run writes to.  CSV logs,
            ``sidecar.json`` and ``heartbeat`` all live here.
        run_id: Unique identifier for this run (typically the SLURM job
            id or a deterministic hash).  Used as the primary key in
            the registry cache and as the CSV version component.
        tags: Free-form string tags for grouping runs (e.g. model
            architecture, experiment name, sweep id).  Any
            ``SLURM_ARRAY_JOB_ID`` env var is auto-appended as
            ``"sweep:<id>"`` for array-job convenience.
        notes: Optional free-text description.
        flush_logs_every_n_steps: How often the CSV is flushed; the
            sidecar is rewritten on the same cadence.  The heartbeat
            is touched on every ``log_metrics`` call (cheap).
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        run_id: str,
        *,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        flush_logs_every_n_steps: int = 50,
    ) -> None:
        run_dir = Path(run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # save_dir + name="" + version="" ⇒ CSVLogger.log_dir == run_dir.
        # Matches the existing Manager-auto-CSV layout.
        super().__init__(
            save_dir=str(run_dir),
            name="",
            version="",
            flush_logs_every_n_steps=flush_logs_every_n_steps,
        )

        self._run_dir = run_dir
        self._run_id = str(run_id)

        self._tags: list[str] = list(tags or [])
        array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
        if array_job and f"sweep:{array_job}" not in self._tags:
            self._tags.append(f"sweep:{array_job}")

        self._notes = notes or ""
        self._hparams: dict[str, Any] = {}
        self._summary: dict[str, Any] = {}
        # Per-metric extended stats for summary.json: each entry is a dict
        # with last / min / max / count. Updated incrementally on every
        # log_metrics.
        self._metric_stats: dict[str, dict[str, Any]] = {}
        self._checkpoint_path: Optional[str] = None
        self._status = "running"
        # Last step / epoch seen via log_metrics — step is used to attach a
        # step to media events when log_image / log_video doesn't supply one
        # (matches Lightning's WandbLogger behaviour). Epoch is read from
        # the metrics dict (Lightning auto-injects an "epoch" key) and is
        # surfaced as a top-level field in summary.json.
        self._last_step: int = 0
        self._last_epoch: int = 0
        # Replace Lightning's truncate-on-init writer with our appending one
        # so SLURM preempt/requeue cycles don't erase prior training history.
        self._experiment = _AppendingExperimentWriter(log_dir=str(run_dir))
        # Preserve the first-write timestamp across sidecar rewrites so
        # the registry can order runs chronologically regardless of how
        # often we flush.
        self._created_at: Optional[float] = None
        # First-write flag for summary.json — used to log a one-shot info
        # line on creation, then debug lines on subsequent rewrites so we
        # don't spam every flush.
        self._summary_written: bool = False
        logging.info(
            f"[RegistryLogger] run_id={self._run_id} "
            f"run_dir={self._run_dir} — sidecar.json + summary.json + "
            "metrics.csv will live here"
        )

    # -- identity ---------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    # -- lightning hooks --------------------------------------------------------

    @rank_zero_only
    def log_hyperparams(
        self, params: Union[dict[str, Any], Any], *args: Any, **kw: Any
    ) -> None:
        # Persist to CSVLogger's hparams.yaml.
        super().log_hyperparams(params, *args, **kw)
        self._hparams = _flatten_params(params)
        self._write_sidecar()

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        # CSV-side: write the raw per-step row.
        super().log_metrics(metrics, step)
        if step is not None:
            self._last_step = int(step)
        # Lightning auto-injects "epoch" into the metrics dict; surface it
        # as a top-level field in summary.json.
        if "epoch" in metrics:
            ep = _to_scalar(metrics["epoch"])
            if ep is not None:
                self._last_epoch = int(ep)

        for k, v in metrics.items():
            scalar = _to_scalar(v)
            if scalar is None:
                continue
            # Sidecar-side: accumulate last-value-per-key summary (kept for
            # backward compatibility with the scanner / SQLite cache).
            self._summary[k] = scalar
            # summary.json-side: extended stats with last/min/max/count.
            stats = self._metric_stats.get(k)
            if stats is None:
                self._metric_stats[k] = {
                    "last": scalar,
                    "min": scalar,
                    "max": scalar,
                    "count": 1,
                }
            else:
                stats["last"] = scalar
                stats["count"] += 1
                if scalar < stats["min"]:
                    stats["min"] = scalar
                if scalar > stats["max"]:
                    stats["max"] = scalar

        # Heartbeat: cheap, fire-and-forget; used by the scanner to
        # distinguish running / stalled / dead without contacting SLURM.
        _sidecar.touch_heartbeat(self._run_dir)

    @rank_zero_only
    def save(self) -> None:
        super().save()
        self._write_sidecar()
        # Lightning calls save() at flush cadence and at epoch boundaries —
        # piggyback the summary flush on the same path so summary.json is
        # always fresh after each epoch.
        self._write_summary_safe()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # Map Lightning status strings to our canonical vocabulary.
        self._status = {"success": "completed", "failed": "failed"}.get(status, status)
        # Parent writes CSVs.  We don't call super().finalize first
        # because _experiment may be None on rank-zero callers that
        # never logged — super() handles that no-op correctly.
        super().finalize(status)
        self._write_sidecar()
        self._write_summary_safe()

    def after_save_checkpoint(self, checkpoint_callback: Any) -> None:
        # This callback fires on every rank; we gate on rank_zero via
        # the helper write (which is rank-zero-only upstream).
        path = (
            getattr(checkpoint_callback, "best_model_path", None)
            or getattr(checkpoint_callback, "last_model_path", None)
            or None
        )
        if path:
            self._checkpoint_path = str(path)
            self._write_sidecar_safe()

    # -- media (images / videos) -----------------------------------------------

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list,
        step: Optional[int] = None,
        caption: Optional[list] = None,
        **_: Any,
    ) -> None:
        """Save images under ``{run_dir}/media/<safe_tag>/<step>_<i>.png``.

        Compatible with Lightning's :class:`WandbLogger.log_image` signature,
        so existing callbacks that gate on ``hasattr(logger, "log_image")``
        will start writing media to disk without code changes.

        Accepts numpy arrays (HWC or CHW, uint8 or float[0,1]), PIL images,
        torch tensors, or paths to existing files. Each entry is also
        appended to ``media.jsonl`` so the registry / web viewer can index
        events without walking the filesystem.
        """
        s = self._resolve_step(step)
        media_dir = self._media_dir(key)
        media_dir.mkdir(parents=True, exist_ok=True)
        cap = list(caption) if caption else []
        for i, img in enumerate(images):
            dst = media_dir / f"{s:08d}_{i}.png"
            try:
                _save_image_to(img, dst)
            except Exception as e:
                # Don't kill training on a media-save error.
                print(f"[RegistryLogger.log_image] failed to save {key}[{i}]: {e}")
                continue
            self._append_media_event(
                {
                    "step": s,
                    "tag": key,
                    "type": "image",
                    "path": str(dst.relative_to(self._run_dir)),
                    "caption": cap[i] if i < len(cap) else None,
                }
            )

    @rank_zero_only
    def log_video(
        self,
        key: str,
        videos: list,
        step: Optional[int] = None,
        caption: Optional[list] = None,
        fps: Optional[int] = None,
        format: Optional[str] = None,
        **_: Any,
    ) -> None:
        """Save videos under ``{run_dir}/media/<safe_tag>/<step>_<i>.<ext>``.

        Inputs may be filesystem paths to already-encoded files (preferred —
        zero re-encoding cost) or raw bytes. The ``fps`` and detected
        ``format`` are recorded in ``media.jsonl`` so a viewer can play them
        back at the right rate.
        """
        s = self._resolve_step(step)
        media_dir = self._media_dir(key)
        media_dir.mkdir(parents=True, exist_ok=True)
        cap = list(caption) if caption else []
        for i, vid in enumerate(videos):
            ext = (format or "mp4").lstrip(".")
            if isinstance(vid, (str, Path)):
                src_ext = Path(vid).suffix.lstrip(".")
                if src_ext:
                    ext = src_ext
            dst = media_dir / f"{s:08d}_{i}.{ext}"
            try:
                _save_video_to(vid, dst)
            except Exception as e:
                print(f"[RegistryLogger.log_video] failed to save {key}[{i}]: {e}")
                continue
            self._append_media_event(
                {
                    "step": s,
                    "tag": key,
                    "type": "video",
                    "path": str(dst.relative_to(self._run_dir)),
                    "caption": cap[i] if i < len(cap) else None,
                    "fps": fps,
                    "format": ext,
                }
            )

    def _resolve_step(self, step: Optional[int]) -> int:
        if step is not None:
            return int(step)
        return int(self._last_step)

    def _media_dir(self, key: str) -> Path:
        # Replace path separators so the tag becomes a single safe directory.
        safe = key.replace("/", "__").replace("\\", "__")
        return self._run_dir / "media" / safe

    @rank_zero_only
    def _append_media_event(self, event: dict[str, Any]) -> None:
        """Append a JSONL line to ``{run_dir}/media.jsonl``.

        JSONL (one event per line) keeps writes O(1) — no read-merge-write —
        and is robust to crashes (a partially-written line is just discarded
        on the next read).
        """
        manifest = self._run_dir / "media.jsonl"
        try:
            with manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except OSError as e:
            print(f"[RegistryLogger] media.jsonl write failed: {e}")

    # -- sidecar ----------------------------------------------------------------

    @rank_zero_only
    def _write_sidecar(self) -> None:
        """Atomically (re)write the sidecar.  Let exceptions propagate."""
        data = _sidecar.make_sidecar(
            run_id=self._run_id,
            run_dir=str(self._run_dir),
            status=self._status,
            created_at=self._created_at,
            hparams=self._hparams,
            summary=self._summary,
            tags=self._tags,
            notes=self._notes,
            checkpoint_path=self._checkpoint_path,
        )
        _sidecar.write_sidecar(self._run_dir, data)
        if self._created_at is None:
            self._created_at = data["created_at"]

    @rank_zero_only
    def _write_sidecar_safe(self) -> None:
        """Same as :meth:`_write_sidecar` but swallows I/O errors.

        Used from callback hooks where a failed write should never take
        down a training run.
        """
        try:
            self._write_sidecar()
        except OSError:
            pass

    # -- summary.json -----------------------------------------------------------

    @rank_zero_only
    def _write_summary(self) -> None:
        """Atomically (re)write ``summary.json`` with per-metric stats.

        Format is intentionally tight (a flat ``{metric: stats_dict}`` map)
        so a downstream reader can parse it with a single ``json.load`` and
        reach any metric's last/min/max in O(1).
        """
        data = {
            "schema_version": 1,
            "run_id": self._run_id,
            "run_dir": str(self._run_dir),
            "updated_at": time.time(),
            "step": self._last_step,
            "epoch": self._last_epoch,
            "metrics": dict(self._metric_stats),
        }
        path = self._run_dir / "summary.json"
        _sidecar.atomic_json_write(path, data)
        msg = (
            f"[RegistryLogger] summary.json flushed "
            f"({len(self._metric_stats)} metrics, step={self._last_step}, "
            f"epoch={self._last_epoch}) → {path}"
        )
        if not self._summary_written:
            logging.info(msg)
            self._summary_written = True
        else:
            logging.debug(msg)

    @rank_zero_only
    def _write_summary_safe(self) -> None:
        """Same as :meth:`_write_summary` but swallows I/O errors.

        ``summary.json`` is auxiliary — a failed write must never take down
        a training run.
        """
        try:
            self._write_summary()
        except OSError:
            pass


# --------------------------------------------------------------------- helpers


def _save_image_to(img: Any, path: Path) -> None:
    """Persist an image-like value to ``path`` as PNG.

    Accepts: file path (bytes-copy), ``PIL.Image``, ``numpy.ndarray``
    (HWC or CHW, uint8 or float in [0, 1]), or ``torch.Tensor`` (treated as
    numpy after detach/cpu).
    """
    if isinstance(img, (str, Path)):
        Path(path).write_bytes(Path(img).read_bytes())
        return

    # PIL.Image — duck-type to avoid a hard dependency on PIL at import time.
    if hasattr(img, "save") and hasattr(img, "mode") and hasattr(img, "size"):
        img.save(path, format="PNG")
        return

    # torch.Tensor → numpy
    try:  # pragma: no cover — torch is optional at logger-import time
        import torch  # type: ignore

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
    except ImportError:
        pass

    try:
        import numpy as np  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "log_image requires numpy or PIL to save non-path inputs"
        ) from e

    if not isinstance(img, np.ndarray):
        raise TypeError(f"unsupported image type for log_image: {type(img)}")

    arr = img
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        # CHW → HWC heuristic (only flips when the last axis isn't already a channel count).
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)

    from PIL import Image  # imported lazily so logger import doesn't drag PIL.

    Image.fromarray(arr).save(path, format="PNG")


def _save_video_to(vid: Any, path: Path) -> None:
    """Persist a video-like value to ``path``.

    Inputs are expected to be already-encoded media — either a filesystem
    path or a ``bytes`` blob. We don't re-encode here: callbacks that build
    frames in memory should write them out (e.g. via imageio / opencv) and
    pass us the resulting path.
    """
    if isinstance(vid, (str, Path)):
        Path(path).write_bytes(Path(vid).read_bytes())
        return
    if isinstance(vid, (bytes, bytearray)):
        Path(path).write_bytes(bytes(vid))
        return
    raise TypeError(
        f"unsupported video type for log_video: {type(vid)} (pass a path or raw bytes)"
    )


def _flatten_params(params: Any) -> dict[str, Any]:
    """Flatten a (possibly nested) hparams object to a flat JSON-safe dict.

    Accepts ``DictConfig``, ``Namespace``, dicts, lists, scalars, or
    anything.  Non-serializable values are stringified so the sidecar
    stays round-trippable.
    """
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)
    except ImportError:
        pass

    if not isinstance(params, dict):
        params = (
            vars(params) if hasattr(params, "__dict__") else {"params": str(params)}
        )

    out: dict[str, Any] = {}
    _flatten(params, "", out)
    return out


def _flatten(obj: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}{k}." if prefix else f"{k}.", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(v, f"{prefix}{i}.", out)
    else:
        key = prefix.rstrip(".")
        try:
            json.dumps(obj)
            out[key] = obj
        except (TypeError, ValueError):
            out[key] = str(obj)


def _to_scalar(v: Any) -> Optional[float]:
    """Coerce metric value to a float scalar, or ``None`` if not scalar.

    Handles torch Tensors, numpy scalars, int, float, bool.  Anything
    else (strings, multi-element tensors, etc.) is skipped — we
    deliberately keep the summary numeric so downstream tools can
    always ``float()`` it.
    """
    # Common path: plain float/int.
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    # Tensor-like with an ``item()`` method and 0-dim shape.
    item = getattr(v, "item", None)
    if callable(item):
        try:
            numel = getattr(v, "numel", None)
            if callable(numel) and numel() != 1:
                return None
            return float(item())
        except (RuntimeError, ValueError, TypeError):
            return None
    return None
