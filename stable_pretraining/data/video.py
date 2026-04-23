"""Random-access video segment dataset backed by Lance.

This module provides two things:

    - :func:`build_lance_video_dataset` converts a folder of video files into a
      Lance dataset of per-frame WebP-encoded rows, optimised for fast random
      contiguous-segment reads on slow filesystems (e.g. NFS).

    - :class:`LanceVideoSegments` is a PyTorch :class:`~torch.utils.data.Dataset`
      that deterministically enumerates every valid ``(window_length,
      frame_skip, hop_size)`` segment across all videos. ``len(dataset)`` is
      the total number of segments; :attr:`LanceVideoSegments.segment_filenames`
      returns the source filename for every segment index.

Benchmark (40 videos / 43 k frames / 224x224 / NFS; see ``_video_bench/``):

    ================================  =====  =======
    reader                            disk   fps
    ================================  =====  =======
    LanceVideoSegments (8 workers)    55 M   13 500
    mp4 H.265 + decord (8 workers)    4.4 M   5 190
    ================================  =====  =======

"""

import json
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Union

import cv2
import lance
import numpy as np
import pyarrow as pa
import torch
from loguru import logger as logging

from .datasets import Dataset


# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------

_SCHEMA = pa.schema(
    [
        ("video_id", pa.int32()),
        ("frame_idx", pa.int32()),
        ("bytes", pa.binary()),
    ]
)

# One-time per-process cache of opened Lance datasets.
# IMPORTANT: must be reset in every DataLoader worker (see ``worker_init``)
# because the Rust/tokio runtime inside a Lance handle is not fork-safe.
_LANCE_CACHE: dict = {}


def _open_dataset(path: str):
    ds = _LANCE_CACHE.get(path)
    if ds is None:
        ds = lance.dataset(path)
        _LANCE_CACHE[path] = ds
    return ds


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _encode_one_video(task):
    """Worker payload: decode → maybe resize → WebP-encode a single video.

    Runs in a child process (``spawn``). Decodes the source video with
    :class:`cv2.VideoCapture` (frames come back in BGR uint8), which avoids a
    dedicated video-decode dependency. Returns either a success payload with
    the list of WebP-encoded frames, or an error tuple to be logged and
    skipped.
    """
    video_id, path_str, quality, resize = task
    try:
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            return ("error", video_id, path_str, "cv2.VideoCapture failed to open")
        enc_params = [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]
        blobs: list[bytes] = []
        out_H = out_W = None
        do_resize = None  # decided after we see the first frame's shape
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if do_resize is None:
                H0, W0 = bgr.shape[:2]
                do_resize = resize is not None and max(H0, W0) > int(resize)
                out_H = int(resize) if do_resize else H0
                out_W = int(resize) if do_resize else W0
            if do_resize:
                bgr = cv2.resize(bgr, (out_W, out_H), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".webp", bgr, enc_params)
            if not ok:
                cap.release()
                return (
                    "error",
                    video_id,
                    path_str,
                    f"webp encode failed at t={len(blobs)}",
                )
            blobs.append(buf.tobytes())
        cap.release()
        T = len(blobs)
        if T == 0:
            return ("error", video_id, path_str, "empty video")
        return ("ok", video_id, path_str, T, out_H, out_W, blobs)
    except Exception as e:  # decode error, codec missing, corrupt file, ...
        return ("error", video_id, path_str, f"{type(e).__name__}: {e}")


def _batch_stream(
    pool_iter,
    video_records: list,
    skip_corrupt: bool,
    progress,
    task_id,
):
    """Stream RecordBatches into Lance and record per-video metadata.

    Generator of ``pa.RecordBatch`` fed to Lance. Also populates
    ``video_records`` with per-video metadata for the sidecar.
    """
    row = 0
    for result in pool_iter:
        tag = result[0]
        if tag == "error":
            _, video_id, path_str, msg = result
            logging.warning(f"skipping {path_str}: {msg}")
            if not skip_corrupt:
                raise RuntimeError(f"failed on {path_str}: {msg}")
            progress.update(task_id, advance=1)
            continue
        _, video_id, path_str, T, H, W, blobs = result
        video_records.append(
            {
                "id": int(video_id),
                "path": path_str,
                "T": int(T),
                "H": int(H),
                "W": int(W),
                "start_row": int(row),
            }
        )
        # Emit one RecordBatch per video (natural chunk boundary in Lance).
        vid_col = pa.array([video_id] * T, type=pa.int32())
        idx_col = pa.array(np.arange(T, dtype=np.int32))
        byt_col = pa.array(blobs, type=pa.binary())
        yield pa.record_batch([vid_col, idx_col, byt_col], schema=_SCHEMA)
        row += T
        progress.update(task_id, advance=1, frames=row)


def build_lance_video_dataset(
    source_dir: Union[str, Path],
    output_path: Union[str, Path],
    *,
    quality: int = 65,
    resize: Optional[int] = None,
    workers: Optional[int] = None,
    glob: str = "*.mp4",
    recursive: bool = True,
    overwrite: bool = False,
    skip_corrupt: bool = True,
) -> Path:
    """Build a random-access Lance video dataset from a folder of video files.

    Each frame is decoded once, optionally downsampled, then WebP-encoded and
    stored as one row ``(video_id, frame_idx, bytes)``. A side-car file at
    ``<output_path>.videos.json`` stores per-video metadata (``path``, ``T``,
    ``H``, ``W``, ``start_row``) so the matching :class:`LanceVideoSegments`
    reader can map ``(video, frame) → row`` without rescanning files.

    The build streams frames from worker processes directly into Lance, so
    memory usage is bounded by a single video's worth of encoded frames
    (plus per-worker prefetch buffer).

    Args:
        source_dir: Folder containing the video files.
        output_path: Target path for the ``.lance`` dataset directory.
        quality: WebP quality in ``[1, 100]``. The benchmark sweet-spot for
            random segment reads at 224x224 on NFS is 65; raise to 80 if you
            care about fine texture preservation.
        resize: If set and a video's ``max(H, W) > resize``, each frame is
            resized to ``(resize, resize)`` with :func:`cv2.resize` (area
            interpolation). Videos whose native resolution is already at or
            below ``resize`` are stored without resampling. ``None`` (default)
            keeps every video at its native resolution — in which case you
            must provide a custom collate function if resolutions differ.
        workers: Number of decode processes. Defaults to ``os.cpu_count()``.
        glob: File pattern for discovery (default ``"*.mp4"``).
        recursive: If ``True`` (default) walk ``source_dir`` recursively.
        overwrite: If ``True``, remove any existing dataset at
            ``output_path``. Otherwise raises if the target already exists.
        skip_corrupt: If ``True`` (default), videos that fail to decode are
            logged and skipped. If ``False``, the first failure aborts the
            build.

    Returns:
        Path to the ``.lance`` dataset directory that was written.
    """
    source_dir = Path(source_dir)
    output_path = Path(output_path)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Pass overwrite=True to replace it."
            )
        import shutil

        shutil.rmtree(output_path)

    if recursive:
        files = sorted(source_dir.rglob(glob))
    else:
        files = sorted(source_dir.glob(glob))
    if not files:
        raise FileNotFoundError(
            f"no files matching '{glob}' under {source_dir} (recursive={recursive})"
        )
    logging.info(
        f"building Lance video dataset from {len(files)} files "
        f"under {source_dir} → {output_path}"
    )

    n_workers = int(workers) if workers else max(1, os.cpu_count() or 1)
    # cv2 + fork can deadlock when the parent has already touched OpenCV (its
    # internal thread pool ends up in a bad state in the child). Always spawn.
    ctx = mp.get_context("spawn")

    tasks = [(i, str(p), int(quality), resize) for i, p in enumerate(files)]
    video_records: list[dict] = []

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    # Custom column to display running frame count.
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("frames={task.fields[frames]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    task_id = progress.add_task("encoding", total=len(tasks), frames=0)

    t0 = time.time()
    with progress:
        with ctx.Pool(n_workers) as pool:
            pool_iter = pool.imap(_encode_one_video, tasks, chunksize=1)
            reader = pa.RecordBatchReader.from_batches(
                _SCHEMA,
                _batch_stream(
                    pool_iter, video_records, skip_corrupt, progress, task_id
                ),
            )
            lance.write_dataset(reader, str(output_path), mode="create")

    elapsed = time.time() - t0
    total_frames = sum(v["T"] for v in video_records)
    n_ok = len(video_records)
    n_skipped = len(files) - n_ok
    size_b = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())

    sidecar_path = output_path.with_name(output_path.name + ".videos.json")
    sidecar_path.write_text(
        json.dumps(
            {
                "quality": int(quality),
                "resize": int(resize) if resize is not None else None,
                "encoding": "webp",
                "total_rows": int(total_frames),
                "n_videos": n_ok,
                "n_skipped": n_skipped,
                "videos": video_records,
            },
            indent=2,
        )
    )
    logging.info(
        f"done in {elapsed:.1f}s. {n_ok}/{len(files)} videos "
        f"({n_skipped} skipped), {total_frames} frames, "
        f"{size_b / 1024**2:.1f} MiB, {total_frames / max(elapsed, 1e-9):.0f} fps. "
        f"sidecar={sidecar_path}"
    )
    return output_path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _compute_segment_plan(
    videos: list[dict],
    window_length: int,
    frame_skip: int,
    hop_size: int,
    min_video_frames: Optional[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Enumerate all segments.

    Returns three parallel arrays/lists of length ``n_segments``:
        - ``seg_vid[i]``    : index into ``videos`` for segment ``i``
        - ``seg_start[i]``  : start frame (in the source video) for segment ``i``
        - ``per_video``     : segments produced by each included video, in order
    """
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")
    if frame_skip < 0:
        raise ValueError(f"frame_skip must be >= 0, got {frame_skip}")
    if hop_size < 1:
        raise ValueError(f"hop_size must be >= 1, got {hop_size}")

    stride = frame_skip + 1
    span = (window_length - 1) * stride + 1
    min_t = max(span, int(min_video_frames) if min_video_frames is not None else span)

    seg_vid: list[int] = []
    seg_start: list[int] = []
    per_video: list[int] = []
    for vi, v in enumerate(videos):
        T = int(v["T"])
        if T < min_t:
            per_video.append(0)
            continue
        last_start = T - span
        n = last_start // hop_size + 1
        per_video.append(int(n))
        starts = range(0, last_start + 1, hop_size)
        seg_vid.extend([vi] * n)
        seg_start.extend(starts)
    return (
        np.asarray(seg_vid, dtype=np.int64),
        np.asarray(seg_start, dtype=np.int64),
        per_video,
    )


class LanceVideoSegments(Dataset):
    """Deterministic random-access video-segment dataset.

    Enumerates every valid ``(window_length, frame_skip, hop_size)`` segment
    across all videos in a Lance dataset built by
    :func:`build_lance_video_dataset`. ``len(dataset)`` is the total number
    of segments, computed at init time.

    For segment ``i`` the returned frames are those at positions::

        [s, s + stride, s + 2 * stride, ..., s + (L - 1) * stride]

    where ``L = window_length``, ``stride = frame_skip + 1``, and ``s`` is
    a valid start frame ``0, hop_size, 2*hop_size, ...``

    For unbiased random sampling during training, wrap the dataset with
    :class:`torch.utils.data.RandomSampler` or an equivalent. The dataset
    itself is deterministic so that you can also iterate it in order (e.g.
    for validation).

    Args:
        lance_path: Path to a ``.lance`` directory produced by
            :func:`build_lance_video_dataset`. The sidecar
            ``<lance_path>.videos.json`` must sit next to it.
        window_length: Number of frames per returned segment.
        frame_skip: Frames to skip between consecutive frames in the window.
            ``0`` (default) returns consecutive frames.
        hop_size: Stride in source-frame units between consecutive segments
            of the same video. ``1`` (default) maximises the segment count.
        min_video_frames: Skip any video shorter than this. Defaults to the
            window span ``(L - 1) * (frame_skip + 1) + 1``.
        transform: Optional callable applied to each sample dict (the library
            standard). Receives a dict, returns a dict.

    Returns per item (before optional ``transform``):
        ``video``          : uint8 tensor ``(L, H, W, 3)`` RGB
        ``video_idx``      : index into :attr:`video_paths`
        ``filename``       : source video path for this segment
        ``start_frame``    : starting frame index in the source video
        ``frame_indices``  : list of ``L`` frame indices read
        ``sample_idx``     : segment index (same as ``i``)
    """

    def __init__(
        self,
        lance_path: Union[str, Path],
        window_length: int,
        *,
        frame_skip: int = 0,
        hop_size: int = 1,
        min_video_frames: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(transform=transform)

        self.lance_path = str(lance_path)
        self.window_length = int(window_length)
        self.frame_skip = int(frame_skip)
        self.hop_size = int(hop_size)
        self._stride = self.frame_skip + 1
        self._span = (self.window_length - 1) * self._stride + 1

        sidecar = Path(self.lance_path + ".videos.json")
        if not sidecar.exists():
            # Accept also `name.videos.json` next to an `name/` directory.
            alt = Path(
                str(Path(self.lance_path).parent / Path(self.lance_path).name)
                + ".videos.json"
            )
            sidecar = alt if alt.exists() else sidecar
        if not sidecar.exists():
            raise FileNotFoundError(
                f"sidecar not found at {sidecar}. "
                f"Was the dataset built with build_lance_video_dataset?"
            )
        meta = json.loads(sidecar.read_text())
        self._videos: list[dict] = list(meta["videos"])
        if not self._videos:
            raise ValueError(f"no videos recorded in {sidecar}")
        # The dataset assumes all videos share H, W (otherwise batching fails
        # without a user-provided collate_fn). Warn if resolutions vary.
        hs = {v["H"] for v in self._videos}
        ws = {v["W"] for v in self._videos}
        if len(hs) > 1 or len(ws) > 1:
            logging.warning(
                f"video resolutions differ ({len(hs)} Hs, {len(ws)} Ws). "
                "Default collate will fail — supply a custom collate_fn."
            )
        self._H = self._videos[0]["H"]
        self._W = self._videos[0]["W"]

        self._seg_vid, self._seg_start, self._per_video = _compute_segment_plan(
            self._videos,
            self.window_length,
            self.frame_skip,
            self.hop_size,
            min_video_frames,
        )
        if len(self._seg_vid) == 0:
            raise ValueError(
                f"no valid segments: every video has fewer than "
                f"span={self._span} frames (window_length={self.window_length}, "
                f"frame_skip={self.frame_skip})."
            )

    # --- inspection utilities -----------------------------------------------

    def __len__(self) -> int:
        return int(self._seg_vid.shape[0])

    @property
    def video_paths(self) -> list[str]:
        """Source file paths, indexed by ``video_id``."""
        return [v["path"] for v in self._videos]

    @property
    def segment_filenames(self) -> list[str]:
        """Same length as :meth:`__len__` — source path for every segment."""
        paths = self.video_paths
        return [paths[int(v)] for v in self._seg_vid]

    def segment_filename(self, i: int) -> str:
        """Source filename for segment ``i`` (O(1))."""
        return self._videos[int(self._seg_vid[i])]["path"]

    def segment_info(self, i: int) -> dict:
        """Source video + start frame + frame indices for segment ``i``."""
        vi = int(self._seg_vid[i])
        start = int(self._seg_start[i])
        return {
            "video_idx": vi,
            "filename": self._videos[vi]["path"],
            "start_frame": start,
            "frame_indices": [
                start + j * self._stride for j in range(self.window_length)
            ],
        }

    # --- frame access --------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        if not (0 <= idx < len(self)):
            raise IndexError(idx)
        vi = int(self._seg_vid[idx])
        start = int(self._seg_start[idx])
        v = self._videos[vi]
        stride = self._stride
        L = self.window_length

        # Video-local frame indices that the user sees.
        frame_indices = [start + j * stride for j in range(L)]
        # Translate to global Lance row indices for the take().
        row0 = int(v["start_row"])
        rows = [row0 + f for f in frame_indices]

        ds = _open_dataset(self.lance_path)
        blobs = ds.take(rows, columns=["bytes"]).column("bytes").to_pylist()

        out = np.empty((L, v["H"], v["W"], 3), dtype=np.uint8)
        for j, b in enumerate(blobs):
            bgr = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
            out[j] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        sample = {
            "video": torch.from_numpy(out),
            "video_idx": vi,
            "filename": v["path"],
            "start_frame": start,
            "frame_indices": frame_indices,
            "sample_idx": int(idx),
        }
        return self.process_sample(sample)

    # --- DataLoader helper ---------------------------------------------------

    @staticmethod
    def worker_init(worker_id: int) -> None:
        """Pass as ``worker_init_fn=LanceVideoSegments.worker_init``.

        Lance is not fork-safe (its Rust/tokio runtime is inherited with a
        dead state on ``fork``), so every worker must reset the module-level
        handle cache. Also pins cv2 to a single thread so N DataLoader
        workers × M OpenCV threads does not oversubscribe the CPU.
        """
        global _LANCE_CACHE
        _LANCE_CACHE = {}
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
