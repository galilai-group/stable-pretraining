"""Unit tests for the Lance-backed video segment dataset.

Tests are marked ``unit`` and are fully self-cleaning: they only write under
``tmp_path`` (pytest-managed scratch) and don't touch any user directories.
Synthetic .mp4 files are written with ``cv2.VideoWriter`` (no extra dep
needed — cv2 is already a core dependency of the library).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from stable_pretraining.data.video import (
    LanceVideoSegments,
    build_lance_video_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_mp4(
    path: Path, n_frames: int, h: int = 48, w: int = 48, fps: int = 10
) -> None:
    """Write a tiny, codec-friendly synthetic .mp4 file.

    Uses OpenCV's ``mp4v`` writer (works out of the box with the standard
    opencv-python wheel). Decord can read these files back without issues.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert vw.isOpened(), f"cv2.VideoWriter failed to open {path}"
    for t in range(n_frames):
        # A gradient that shifts with t so consecutive frames differ.
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = (t * 7) % 256
        img[:, :, 1] = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, 0)
        img[:, :, 2] = np.linspace(0, 255, h, dtype=np.uint8)[:, None].repeat(w, 1)
        vw.write(img)
    vw.release()


@pytest.fixture
def mp4_dir(tmp_path):
    """Three synthetic .mp4 files under tmp_path/videos/.

    Frame counts picked so different (window_length, frame_skip, hop_size)
    combinations exercise the enumeration math — e.g. one video shorter than
    the default window span.
    """
    d = tmp_path / "videos"
    _write_mp4(d / "a.mp4", n_frames=32)
    _write_mp4(d / "b.mp4", n_frames=20)
    _write_mp4(d / "c.mp4", n_frames=8)  # intentionally short
    return d


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildLanceVideoDataset:
    """Unit tests for :func:`build_lance_video_dataset`."""

    def test_build_creates_dataset_and_sidecar(self, mp4_dir, tmp_path):
        out = tmp_path / "ds.lance"
        returned = build_lance_video_dataset(
            mp4_dir, out, quality=70, resize=None, workers=1
        )
        assert returned == out
        assert out.is_dir(), "lance dataset directory not created"

        sidecar = tmp_path / "ds.lance.videos.json"
        assert sidecar.exists(), "sidecar videos.json missing"
        meta = json.loads(sidecar.read_text())
        assert meta["encoding"] == "webp"
        assert meta["quality"] == 70
        assert meta["resize"] is None
        assert meta["n_videos"] == 3
        assert meta["n_skipped"] == 0
        # total_rows == sum of per-video T; start_row is cumulative.
        assert meta["total_rows"] == sum(v["T"] for v in meta["videos"])
        starts = [v["start_row"] for v in meta["videos"]]
        assert starts == sorted(starts), "start_row must be non-decreasing"

    def test_resize_shrinks_only_when_bigger(self, tmp_path):
        src = tmp_path / "src"
        # Pick sizes that are (a) clearly distinct from the resize threshold
        # so the test is meaningful, and (b) large enough to avoid ultra-small
        # mp4 codec quirks in cv2.VideoWriter (<32px can break the readback).
        _write_mp4(src / "big.mp4", n_frames=6, h=96, w=96)
        _write_mp4(src / "small.mp4", n_frames=6, h=48, w=48)
        out = tmp_path / "ds.lance"
        build_lance_video_dataset(src, out, quality=60, resize=64, workers=1)
        meta = json.loads((tmp_path / "ds.lance.videos.json").read_text())
        by_path = {Path(v["path"]).name: v for v in meta["videos"]}
        # 96x96 > 64 → downsized to 64x64
        assert by_path["big.mp4"]["H"] == 64 and by_path["big.mp4"]["W"] == 64
        # 48x48 <= 64 → kept native
        assert by_path["small.mp4"]["H"] == 48 and by_path["small.mp4"]["W"] == 48

    def test_overwrite_false_raises_on_existing(self, mp4_dir, tmp_path):
        out = tmp_path / "ds.lance"
        build_lance_video_dataset(mp4_dir, out, quality=60, workers=1)
        with pytest.raises(FileExistsError):
            build_lance_video_dataset(mp4_dir, out, quality=60, workers=1)

    def test_overwrite_true_replaces(self, mp4_dir, tmp_path):
        out = tmp_path / "ds.lance"
        build_lance_video_dataset(mp4_dir, out, quality=60, workers=1)
        # Second build with a different setting should succeed with overwrite.
        build_lance_video_dataset(mp4_dir, out, quality=75, workers=1, overwrite=True)
        meta = json.loads((tmp_path / "ds.lance.videos.json").read_text())
        assert meta["quality"] == 75

    def test_skip_corrupt(self, mp4_dir, tmp_path):
        # Add a 0-byte file that looks like an mp4 but isn't decodable.
        corrupt = mp4_dir / "bad.mp4"
        corrupt.write_bytes(b"")
        out = tmp_path / "ds.lance"
        build_lance_video_dataset(
            mp4_dir, out, quality=60, workers=1, skip_corrupt=True
        )
        meta = json.loads((tmp_path / "ds.lance.videos.json").read_text())
        assert meta["n_skipped"] == 1
        assert meta["n_videos"] == 3
        # corrupt video must not appear in the kept list
        assert all(Path(v["path"]).name != "bad.mp4" for v in meta["videos"])

    def test_no_videos_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            build_lance_video_dataset(empty, tmp_path / "x.lance", workers=1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@pytest.fixture
def built_dataset(mp4_dir, tmp_path):
    """Build the dataset once and return the lance path + its metadata."""
    out = tmp_path / "ds.lance"
    build_lance_video_dataset(
        mp4_dir, out, quality=65, resize=32, workers=1, overwrite=True
    )
    meta = json.loads((tmp_path / "ds.lance.videos.json").read_text())
    return out, meta


@pytest.mark.unit
class TestLanceVideoSegments:
    """Unit tests for :class:`LanceVideoSegments` enumeration + reads."""

    # --- enumeration ------------------------------------------------------

    def test_len_matches_manual_count(self, built_dataset):
        out, meta = built_dataset
        # (L=4, skip=0, hop=1) → segments per video = T - 3
        L, skip, hop = 4, 0, 1
        stride = skip + 1
        span = (L - 1) * stride + 1
        expected = sum(max(0, v["T"] - span + 1) for v in meta["videos"])
        ds = LanceVideoSegments(out, window_length=L, frame_skip=skip, hop_size=hop)
        assert len(ds) == expected

    def test_frame_skip_changes_stride(self, built_dataset):
        out, _ = built_dataset
        ds = LanceVideoSegments(out, window_length=4, frame_skip=1)
        info = ds.segment_info(0)
        # stride = frame_skip + 1 = 2
        assert info["frame_indices"] == [0, 2, 4, 6]

    def test_hop_size_changes_stride_between_segments(self, built_dataset):
        out, _ = built_dataset
        ds = LanceVideoSegments(out, window_length=4, frame_skip=0, hop_size=3)
        # Segment 0 starts at frame 0, segment 1 at frame 3 (same video).
        i0 = ds.segment_info(0)
        i1 = ds.segment_info(1)
        assert i0["video_idx"] == 0 and i1["video_idx"] == 0
        assert i0["start_frame"] == 0 and i1["start_frame"] == 3

    def test_short_videos_skipped(self, built_dataset):
        out, meta = built_dataset
        # c.mp4 has 8 frames; span=10 (L=10, skip=0) exceeds → no segments from it.
        ds = LanceVideoSegments(out, window_length=10, frame_skip=0, hop_size=1)
        video_ids_used = {int(v) for v in ds._seg_vid.tolist()}
        # There are 3 videos; c.mp4 should be the one not used.
        names_used = {Path(meta["videos"][i]["path"]).name for i in video_ids_used}
        assert "c.mp4" not in names_used

    def test_no_valid_segments_raises(self, built_dataset):
        out, _ = built_dataset
        with pytest.raises(ValueError, match="no valid segments"):
            LanceVideoSegments(out, window_length=10_000)

    def test_invalid_args_raise(self, built_dataset):
        out, _ = built_dataset
        with pytest.raises(ValueError):
            LanceVideoSegments(out, window_length=0)
        with pytest.raises(ValueError):
            LanceVideoSegments(out, window_length=4, frame_skip=-1)
        with pytest.raises(ValueError):
            LanceVideoSegments(out, window_length=4, hop_size=0)

    # --- segment → filename mapping ---------------------------------------

    def test_segment_filenames_length_matches_len(self, built_dataset):
        out, _ = built_dataset
        ds = LanceVideoSegments(out, window_length=4)
        names = ds.segment_filenames
        assert len(names) == len(ds)

    def test_segment_filename_matches_segment_info(self, built_dataset):
        out, _ = built_dataset
        ds = LanceVideoSegments(out, window_length=4, hop_size=1)
        for i in [0, len(ds) // 2, len(ds) - 1]:
            assert ds.segment_filename(i) == ds.segment_info(i)["filename"]

    # --- __getitem__ ------------------------------------------------------

    def test_getitem_shape_and_keys(self, built_dataset):
        out, _ = built_dataset
        ds = LanceVideoSegments(out, window_length=4)
        sample = ds[0]
        assert set(sample.keys()) == {
            "video",
            "video_idx",
            "filename",
            "start_frame",
            "frame_indices",
            "sample_idx",
        }
        assert sample["video"].dtype == torch.uint8
        # H, W come from the sidecar (resize=32 applied).
        assert tuple(sample["video"].shape) == (4, 32, 32, 3)
        assert sample["sample_idx"] == 0
        assert len(sample["frame_indices"]) == 4

    def test_frame_indices_are_video_local(self, built_dataset):
        out, meta = built_dataset
        # Find a segment from a video that is NOT the first one (so its
        # start_row in the sidecar > 0) to make sure we return video-local
        # frame indices, not global Lance row indices.
        ds = LanceVideoSegments(out, window_length=4, hop_size=1)
        # First segment that belongs to video index 1.
        idx = next(i for i, v in enumerate(ds._seg_vid.tolist()) if v == 1)
        info = ds.segment_info(idx)
        sample = ds[idx]
        assert sample["frame_indices"] == info["frame_indices"]
        # Sanity: the smallest frame index must be < video's T, and strictly
        # below the number of rows for that video.
        assert max(sample["frame_indices"]) < meta["videos"][1]["T"]

    def test_out_of_range_raises(self, built_dataset):
        out, _ = built_dataset
        ds = LanceVideoSegments(out, window_length=4)
        with pytest.raises(IndexError):
            _ = ds[len(ds)]
        with pytest.raises(IndexError):
            _ = ds[-1]

    # --- DataLoader smoke test --------------------------------------------

    def test_dataloader_single_process(self, built_dataset):
        """Run through DataLoader with ``num_workers=0`` (no fork concerns)."""
        from torch.utils.data import DataLoader

        out, _ = built_dataset
        ds = LanceVideoSegments(out, window_length=4, hop_size=1)

        def collate(xs):
            return {
                k: (
                    torch.stack([x[k] for x in xs])
                    if isinstance(xs[0][k], torch.Tensor)
                    else [x[k] for x in xs]
                )
                for k in xs[0]
            }

        loader = DataLoader(
            ds, batch_size=3, num_workers=0, collate_fn=collate, shuffle=False
        )
        batch = next(iter(loader))
        assert batch["video"].shape[0] == 3
        assert batch["video"].shape[1] == 4  # window_length
        assert len(batch["filename"]) == 3
