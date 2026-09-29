"""Unit tests for the ReconViz reconstruction-visualisation callback.

Covers three layers:

1. The pure-torch helpers (denorm, stitch-with-separator, grid, uint8
   conversion) and constructor validation — shapes, ranges, and the
   defining behaviours (identity denorm without stats; a separator column
   between target and reconstruction).
2. The caching + emission pipeline in isolation, driven by a fake trainer and
   fake logger: grid routes to ``log_image``, video routes to ``log_video``,
   the ``{name}_preds`` convention resolves, and a missing/absent logger is a
   silent no-op rather than a crash.
3. ``ReconViz`` wired end-to-end with ``OnlineImageDecoder`` through a real
   Lightning Trainer + Manager: a ``target | recon`` grid must reach the media
   logger every validation epoch with zero model coupling.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.callbacks.recon_viz import ReconViz


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class _FakeImageLogger:
    def __init__(self):
        self.calls = []

    def log_image(self, key, images, step=None, caption=None):
        self.calls.append((key, images, step, caption))


class _FakeVideoLogger:
    def __init__(self):
        self.calls = []

    def log_video(self, key, videos, step=None, caption=None, fps=None):
        # snapshot the path's existence now (ReconViz deletes it afterwards)
        exists = [isinstance(v, str) and os.path.getsize(v) > 0 for v in videos]
        self.calls.append((key, videos, step, caption, fps, exists))


def _trainer(loggers, epoch=0, step=5, rank=0, sanity=False):
    return SimpleNamespace(
        loggers=list(loggers),
        current_epoch=epoch,
        global_step=step,
        global_rank=rank,
        sanity_checking=sanity,
    )


# --------------------------------------------------------------------------- #
# 1. helpers + validation
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestReconVizHelpers:
    """Pure-torch helpers and constructor validation."""

    def test_denorm_identity_without_stats_only_clamps(self):
        viz = ReconViz([("recon", "image", None)])
        x = torch.tensor([[-0.5, 0.5, 2.0]]).view(1, 1, 1, 3)
        out = viz._denormalize(x)
        # no mean/std -> values pass through but clamp to [0, 1]
        assert torch.equal(out, torch.tensor([[0.0, 0.5, 1.0]]).view(1, 1, 1, 3))

    def test_denorm_with_stats_inverts_then_clamps(self):
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        viz = ReconViz([("recon", "image", None)], pixel_mean=mean, pixel_std=std)
        x = torch.zeros(2, 3, 4, 4)  # normalised zeros -> 0.5 grey
        out = viz._denormalize(x)
        assert torch.allclose(out, torch.full_like(out, 0.5))
        assert out.min() >= 0 and out.max() <= 1

    def test_stitch_inserts_separator_column(self):
        viz = ReconViz([("recon", "image", None)])
        C, H, W = 3, 8, 6
        target = torch.zeros(C, H, W)
        pred = torch.ones(C, H, W)
        out = viz._stitch(target, pred)
        assert out.shape == (C, H, 2 * W + ReconViz._SEPARATOR_WIDTH)
        # the separator column carries the separator value, not target/pred
        sep = out[:, :, W : W + ReconViz._SEPARATOR_WIDTH]
        assert torch.all(sep == ReconViz._SEPARATOR_VALUE)
        assert torch.all(out[:, :, :W] == 0.0)
        assert torch.all(out[:, :, W + ReconViz._SEPARATOR_WIDTH :] == 1.0)

    def test_make_grid_shape_single_column(self):
        viz = ReconViz([("recon", "image", None)])
        tiles = [torch.rand(3, 8, 12) for _ in range(3)]
        grid = viz._make_grid(tiles, ncol=1)
        pad = ReconViz._PADDING
        assert grid.shape == (3, 3 * (8 + pad) - pad, 12)

    def test_to_hwc_uint8_expands_grayscale(self):
        viz = ReconViz([("recon", "image", None)])
        arr = viz._to_hwc_uint8(torch.rand(1, 8, 8))
        assert arr.dtype == np.uint8
        assert arr.shape == (8, 8, 3)  # single channel expanded to RGB

    def test_length_c_stats_are_reshaped_to_broadcast(self):
        viz = ReconViz(
            [("recon", "image", None)], pixel_mean=[0.1, 0.2, 0.3], pixel_std=[1, 1, 1]
        )
        assert viz.pixel_mean.shape == (3, 1, 1)

    def test_two_tuple_spec_defaults_caption_to_none(self):
        viz = ReconViz([("recon", "image")])
        assert viz.specs == [("recon", "image", None)]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            ReconViz([("recon", "image", None)], mode="bogus")

    def test_video_without_seq_len_raises(self):
        with pytest.raises(ValueError, match="requires seq_len"):
            ReconViz([("recon", "image", None)], mode="video")

    def test_partial_pixel_stats_raises(self):
        with pytest.raises(ValueError, match="both be given or both"):
            ReconViz([("recon", "image", None)], pixel_mean=[0.5, 0.5, 0.5])

    def test_empty_specs_raises(self):
        with pytest.raises(ValueError, match="specs must not be empty"):
            ReconViz([])

    @pytest.mark.parametrize("name", ["n_items", "seq_len", "fps"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_nonpositive_sizes_rejected(self, name, value):
        kwargs = {"mode": "video", "seq_len": 2, name: value}
        with pytest.raises(ValueError, match=f"{name} must be positive"):
            ReconViz([("recon", "image")], **kwargs)

    def test_stats_are_detached_snapshots(self):
        mean = torch.full((3,), 0.5, requires_grad=True)
        viz = ReconViz([("recon", "image")], pixel_mean=mean, pixel_std=[1] * 3)
        with torch.no_grad():
            mean.zero_()
        assert not viz.pixel_mean.requires_grad
        assert torch.all(viz.pixel_mean == 0.5)


# --------------------------------------------------------------------------- #
# 2. caching + emission in isolation
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestReconVizEmission:
    """Caching + emission pipeline driven by fake trainer/logger."""

    def test_grid_routes_to_log_image(self):
        lgr = _FakeImageLogger()
        viz = ReconViz([("recon", "image", "my recon")], mode="grid", n_items=4)
        B, C, H, W = 6, 3, 8, 8
        outputs = {"recon_preds": torch.rand(B, C, H, W)}
        batch = {"image": torch.rand(B, C, H, W)}

        viz.on_validation_batch_end(_trainer([lgr]), None, outputs, batch, 0)
        viz.on_validation_epoch_end(_trainer([lgr], epoch=3, step=9), None)

        assert len(lgr.calls) == 1
        key, images, step, caption = lgr.calls[0]
        assert key == "val/recon"
        assert step == 9
        assert caption == ["my recon (epoch 3)"]
        # one grid image, HWC uint8, 4 stitched rows tall (n_items caps at 4)
        img = images[0]
        assert img.dtype == np.uint8 and img.shape[2] == 3
        assert img.shape[1] == 2 * W + ReconViz._SEPARATOR_WIDTH

    def test_only_first_batch_is_cached(self):
        lgr = _FakeImageLogger()
        viz = ReconViz([("recon", "image", None)], mode="grid", n_items=2)
        B, C, H, W = 2, 3, 8, 8
        first = {"recon_preds": torch.zeros(B, C, H, W)}
        second = {"recon_preds": torch.ones(B, C, H, W)}
        batch = {"image": torch.rand(B, C, H, W)}
        viz.on_validation_batch_end(_trainer([lgr]), None, first, batch, 0)
        # batch_idx != 0 must be ignored
        viz.on_validation_batch_end(_trainer([lgr]), None, second, batch, 1)
        assert torch.all(viz._cache["recon"]["preds"] == 0.0)

    def test_preds_name_convention_and_verbatim_key(self):
        # spec "recon" resolves "recon_preds"; spec "recon_preds" resolves itself.
        for spec_key in ("recon", "recon_preds"):
            lgr = _FakeImageLogger()
            viz = ReconViz([(spec_key, "image", None)], mode="grid", n_items=2)
            B, C, H, W = 2, 3, 8, 8
            outputs = {"recon_preds": torch.rand(B, C, H, W)}
            batch = {"image": torch.rand(B, C, H, W)}
            viz.on_validation_batch_end(_trainer([lgr]), None, outputs, batch, 0)
            viz.on_validation_epoch_end(_trainer([lgr]), None)
            assert len(lgr.calls) == 1

    def test_cache_cleared_after_epoch(self):
        lgr = _FakeImageLogger()
        viz = ReconViz([("recon", "image", None)], mode="grid", n_items=2)
        B, C, H, W = 2, 3, 8, 8
        outputs = {"recon_preds": torch.rand(B, C, H, W)}
        batch = {"image": torch.rand(B, C, H, W)}
        viz.on_validation_batch_end(_trainer([lgr]), None, outputs, batch, 0)
        assert viz._cache
        viz.on_validation_epoch_end(_trainer([lgr]), None)
        assert viz._cache == {}

    def test_missing_key_is_silent_skip(self):
        lgr = _FakeImageLogger()
        viz = ReconViz([("does_not_exist", "image", None)], mode="grid")
        outputs = {"recon_preds": torch.rand(2, 3, 8, 8)}
        batch = {"image": torch.rand(2, 3, 8, 8)}
        # must not raise, and nothing to emit
        viz.on_validation_batch_end(_trainer([lgr]), None, outputs, batch, 0)
        viz.on_validation_epoch_end(_trainer([lgr]), None)
        assert lgr.calls == []

    @pytest.mark.parametrize("target", [None, torch.empty(1, 3, 4, 4)])
    def test_missing_or_mismatched_target_is_skipped(self, target):
        lgr = _FakeImageLogger()
        tr = _trainer([lgr])
        viz = ReconViz([("recon", "image")])
        batch = {} if target is None else {"image": target}
        viz.on_validation_batch_end(
            tr, None, {"recon_preds": torch.rand(2, 3, 8, 8)}, batch, 0
        )
        viz.on_validation_epoch_end(tr, None)
        assert not lgr.calls and not viz._cache

    def test_multiple_specs_continue_after_logger_failure(self):
        def fail(**kwargs):
            raise RuntimeError("unavailable logger")

        lgr = _FakeImageLogger()
        tr = _trainer([SimpleNamespace(log_image=fail), lgr])
        viz = ReconViz([("a", "image"), ("b", "image")])
        x = torch.rand(2, 3, 8, 8)
        viz.on_validation_batch_end(tr, None, None, {"a": x, "b": x, "image": x}, 0)
        viz.on_validation_epoch_end(tr, None)
        assert [call[0] for call in lgr.calls] == ["val/a", "val/b"]
        assert not viz._cache

    def test_video_with_incomplete_trajectory_is_skipped(self):
        lgr = _FakeVideoLogger()
        tr = _trainer([lgr])
        viz = ReconViz([("recon", "image")], mode="video", seq_len=3)
        x = torch.rand(2, 3, 8, 8)
        viz.on_validation_batch_end(tr, None, {"recon_preds": x}, {"image": x}, 0)
        viz.on_validation_epoch_end(tr, None)
        assert not lgr.calls and not viz._cache

    def test_grid_without_image_logger_is_noop(self):
        # a logger that only speaks video must not be used for a grid.
        vlgr = _FakeVideoLogger()
        viz = ReconViz([("recon", "image", None)], mode="grid", n_items=2)
        outputs = {"recon_preds": torch.rand(2, 3, 8, 8)}
        batch = {"image": torch.rand(2, 3, 8, 8)}
        viz.on_validation_batch_end(_trainer([vlgr]), None, outputs, batch, 0)
        viz.on_validation_epoch_end(_trainer([vlgr]), None)  # must not raise
        assert vlgr.calls == []

    def test_sanity_check_epoch_is_skipped(self):
        lgr = _FakeImageLogger()
        viz = ReconViz([("recon", "image", None)], mode="grid", n_items=2)
        outputs = {"recon_preds": torch.rand(2, 3, 8, 8)}
        batch = {"image": torch.rand(2, 3, 8, 8)}
        tr = _trainer([lgr], sanity=True)
        viz.on_validation_batch_end(tr, None, outputs, batch, 0)
        viz.on_validation_epoch_end(tr, None)
        assert lgr.calls == [] and viz._cache == {}

    def test_video_routes_to_log_video(self):
        pytest.importorskip("imageio_ffmpeg")
        lgr = _FakeVideoLogger()
        n_traj, seq_len, C, H, W = 2, 3, 3, 16, 16
        viz = ReconViz(
            [("recon", "image", "traj")],
            mode="video",
            n_items=n_traj,
            seq_len=seq_len,
            fps=4,
        )
        rows = n_traj * seq_len
        outputs = {"recon_preds": torch.rand(rows, C, H, W)}
        batch = {"image": torch.rand(rows, C, H, W)}
        viz.on_validation_batch_end(_trainer([lgr]), None, outputs, batch, 0)
        viz.on_validation_epoch_end(_trainer([lgr], epoch=2, step=7), None)

        assert len(lgr.calls) == 1
        key, videos, step, caption, fps, exists = lgr.calls[0]
        assert key == "val/recon"
        assert step == 7 and fps == 4
        assert caption == ["traj (epoch 2)"]
        # a real, non-empty mp4 existed at log time
        assert exists == [True]
        assert all(not Path(path).exists() for path in videos)

    def test_missing_imageio_does_not_break_validation(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "imageio", None)
        lgr = _FakeVideoLogger()
        viz = ReconViz([("recon", "image")], mode="video", seq_len=2)
        x = torch.rand(2, 3, 8, 8)
        tr = _trainer([lgr])
        viz.on_validation_batch_end(tr, None, {"recon_preds": x}, {"image": x}, 0)
        viz.on_validation_epoch_end(tr, None)
        assert not lgr.calls and not viz._cache

    @pytest.mark.parametrize("failure", ["encoder", "logger"])
    def test_video_failure_cleans_temporary_file(self, monkeypatch, failure):
        paths = []

        def encode(path, frames, fps):
            paths.append(path)
            Path(path).write_bytes(b"partial video")
            if failure == "encoder":
                raise RuntimeError("encoder failed")

        def log_video(**kwargs):
            assert Path(kwargs["videos"][0]).exists()
            raise RuntimeError("logger failed")

        monkeypatch.setitem(sys.modules, "imageio", SimpleNamespace(mimsave=encode))
        tr = _trainer([SimpleNamespace(log_video=log_video)])
        viz = ReconViz([("recon", "image")], mode="video", seq_len=2)
        x = torch.rand(2, 3, 8, 8)
        viz.on_validation_batch_end(tr, None, {"recon_preds": x}, {"image": x}, 0)
        viz.on_validation_epoch_end(tr, None)
        assert len(paths) == 1 and not Path(paths[0]).exists()
        assert not viz._cache

    @pytest.mark.parametrize("shape", [(0, 3, 8, 8), (2, 3, 8), (2, 3, 0, 8)])
    def test_invalid_images_skipped(self, shape):
        lgr = _FakeImageLogger()
        tr = _trainer([lgr])
        viz = ReconViz([("recon", "image")])
        x = torch.empty(shape)
        viz.on_validation_batch_end(tr, None, {"recon_preds": x}, {"image": x}, 0)
        viz.on_validation_epoch_end(tr, None)
        assert not lgr.calls and not viz._cache

    def test_cache_is_snapshot_and_second_loader_cannot_overwrite(self):
        viz = ReconViz([("recon", "image")])
        tr = _trainer([])
        x = torch.zeros(2, 3, 8, 8)
        viz.on_validation_batch_end(tr, None, {"recon_preds": x}, {"image": x}, 0)
        x.fill_(1)
        viz.on_validation_batch_end(
            tr, None, {"recon_preds": x}, {"image": x}, 0, dataloader_idx=1
        )
        assert torch.count_nonzero(viz._cache["recon"]["preds"]) == 0
        assert torch.count_nonzero(viz._cache["recon"]["target"]) == 0

    def test_video_caches_n_traj_times_seq_len_rows(self):
        viz = ReconViz([("recon", "image", None)], mode="video", n_items=3, seq_len=4)
        assert viz._rows_to_cache() == 12

    def test_video_without_video_logger_is_noop(self):
        ilgr = _FakeImageLogger()
        viz = ReconViz([("recon", "image", None)], mode="video", n_items=1, seq_len=2)
        outputs = {"recon_preds": torch.rand(2, 3, 8, 8)}
        batch = {"image": torch.rand(2, 3, 8, 8)}
        viz.on_validation_batch_end(_trainer([ilgr]), None, outputs, batch, 0)
        viz.on_validation_epoch_end(_trainer([ilgr]), None)  # must not raise
        assert ilgr.calls == []

    def test_non_rank_zero_does_not_log(self):
        lgr = _FakeImageLogger()
        viz = ReconViz([("recon", "image", None)], mode="grid", n_items=2)
        outputs = {"recon_preds": torch.rand(2, 3, 8, 8)}
        batch = {"image": torch.rand(2, 3, 8, 8)}
        viz.on_validation_batch_end(_trainer([lgr]), None, outputs, batch, 0)
        viz.on_validation_epoch_end(_trainer([lgr], rank=1), None)
        assert lgr.calls == [] and viz._cache == {}  # cache still cleared


# --------------------------------------------------------------------------- #
# 3. frame alignment (exact, codec-free)
# --------------------------------------------------------------------------- #
def _fill_fingerprints(n, C, H, W):
    """Two (n, C, H, W) batches whose row i is a constant image.

    ``target[i]`` is filled with ``vt(i)`` and ``preds[i]`` with a distinct
    ``vp(i)``, so decoding either half of a rendered cell recovers *which* row
    (and whether it came from target or preds) exactly — no motion, centroids,
    or lossy codec involved.
    """

    def vt(i):
        return (i + 1) / (n + 2)

    def vp(i):
        return 1.0 - (i + 1) / (n + 2)

    target = torch.zeros(n, C, H, W)
    preds = torch.zeros(n, C, H, W)
    for i in range(n):
        target[i] = vt(i)
        preds[i] = vp(i)
    return target, preds, vt, vp


@pytest.mark.unit
class TestReconVizFrameAlignment:
    """Target and reconstruction must line up per cell — exact, no MP4 decode.

    These guard the stitch order and the trajectory-contiguous reshape: if a
    later change transposes the reshape, swaps target/pred, or shifts the
    timestep index, the recovered fingerprint stops matching and the test fails.
    """

    def test_grid_row_i_is_target_i_then_pred_i(self):
        n, C, H, W = 5, 3, 8, 8
        target, preds, vt, vp = _fill_fingerprints(n, C, H, W)
        viz = ReconViz([("recon", "image", None)], mode="grid", n_items=n)
        img = viz._build_grid_image(target, preds).astype(np.float32) / 255.0

        sep, pad = ReconViz._SEPARATOR_WIDTH, ReconViz._PADDING
        for i in range(n):
            y0 = i * (H + pad)
            row = img[y0 : y0 + H]
            left = row[:, :W].mean()
            right = row[:, W + sep : W + sep + W].mean()
            assert abs(left - vt(i)) <= 1 / 255 + 1e-6, f"grid row {i} target half"
            assert abs(right - vp(i)) <= 1 / 255 + 1e-6, f"grid row {i} recon half"

    def test_video_frame_t_row_j_is_traj_j_at_step_t(self):
        # Trajectory-contiguous flat layout: flat = j * seq_len + t.
        n_traj, seq_len, C, H, W = 3, 5, 3, 8, 8
        total = n_traj * seq_len
        target, preds, vt, vp = _fill_fingerprints(total, C, H, W)
        viz = ReconViz(
            [("recon", "image", None)], mode="video", n_items=n_traj, seq_len=seq_len
        )
        frames = viz._build_video_frames(target, preds, n_traj)
        assert len(frames) == seq_len

        sep, pad = ReconViz._SEPARATOR_WIDTH, ReconViz._PADDING
        for t in range(seq_len):
            fr = frames[t].astype(np.float32) / 255.0
            for j in range(n_traj):
                flat = j * seq_len + t
                y0 = j * (H + pad)
                row = fr[y0 : y0 + H]
                left = row[:, :W].mean()
                right = row[:, W + sep : W + sep + W].mean()
                assert abs(left - vt(flat)) <= 1 / 255 + 1e-6, (
                    f"frame t={t} traj j={j}: target half maps to wrong row"
                )
                assert abs(right - vp(flat)) <= 1 / 255 + 1e-6, (
                    f"frame t={t} traj j={j}: recon half maps to wrong row"
                )

    def test_swapping_target_and_preds_is_detectable(self):
        # Sanity: the fingerprint check is not vacuous — swapping the two inputs
        # must flip which value each half recovers.
        n, C, H, W = 4, 3, 8, 8
        target, preds, vt, vp = _fill_fingerprints(n, C, H, W)
        viz = ReconViz([("recon", "image", None)], mode="grid", n_items=n)
        img = viz._build_grid_image(preds, target).astype(np.float32) / 255.0  # swapped
        left = img[:H, :W].mean()
        # left half now carries preds' fingerprint for row 0, not target's
        assert abs(left - vp(0)) <= 1 / 255 + 1e-6
        assert abs(left - vt(0)) > 0.05


# --------------------------------------------------------------------------- #
# 4. end-to-end wiring with OnlineImageDecoder
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestReconVizEndToEnd:
    """ReconViz wired into a real Lightning Trainer + Manager."""

    def test_pairs_with_online_image_decoder(self):
        import lightning as pl
        import torch.nn as nn

        embed_dim, img_size, C, B = 64, 32, 3, 4
        torch.manual_seed(0)
        encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(C * img_size * img_size, embed_dim)
        )

        def forward(self, batch, stage):
            batch["embedding"] = self.encoder(batch["image"])
            return batch

        module = spt.Module(encoder=encoder, forward=forward, optim=None)
        x = torch.randn(B, C, img_size, img_size)

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return B

            def __getitem__(self, idx):
                return {"image": x[idx]}

        dl = torch.utils.data.DataLoader(_DS(), batch_size=B)
        data = spt.data.DataModule(train=dl, val=dl)

        logged = []

        class _MediaLogger(pl.pytorch.loggers.logger.Logger):
            @property
            def name(self):
                return "media"

            @property
            def version(self):
                return "0"

            def log_hyperparams(self, *a, **k):
                pass

            def log_metrics(self, *a, **k):
                pass

            def log_image(self, key, images, step=None, caption=None):
                logged.append((key, tuple(images[0].shape)))

        dec = spt.callbacks.OnlineImageDecoder(
            module=module,
            name="recon",
            input="embedding",
            target="image",
            image_shape=(C, img_size, img_size),
            embed_dim=embed_dim,
            decoder_kwargs=dict(base_channels=32),
        )
        viz = spt.callbacks.ReconViz([("recon", "image", "recon")], mode="grid")

        trainer = pl.Trainer(
            max_epochs=1,
            num_sanity_val_steps=0,
            limit_train_batches=1,
            limit_val_batches=1,
            callbacks=[dec, viz],
            logger=_MediaLogger(),
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()

        assert any(key == "val/recon" for key, _ in logged), (
            f"ReconViz did not log a grid; logged keys: {logged}"
        )
        # the logged image is a target | recon grid: width is 2*img_size + sep
        _, shape = next((k, s) for k, s in logged if k == "val/recon")
        assert shape[1] == 2 * img_size + ReconViz._SEPARATOR_WIDTH
        events = [
            json.loads(line)
            for line in (manager._run_dir / "media.jsonl").read_text().splitlines()
        ]
        event = next(event for event in events if event["tag"] == "val/recon")
        assert event["type"] == "image"
        from PIL import Image

        with Image.open(manager._run_dir / event["path"]) as image:
            assert image.size == (shape[1], shape[0])
