"""Unit tests for the DINO-style visualisation callbacks.

Covers three layers:

1. The pure-torch maths in ``_viz_utils`` (PCA→RGB, CLS-attention thresholding,
   grid inference, robust scaling, denormalisation) — shapes, ranges, and the
   defining behaviours (mask is binary and lands on the attended patch;
   foreground trick paints the background).
2. Figure rendering + logger/disk emission in isolation, with a fake logger.
3. ``PCATokenVisualizer`` and ``AttentionVisualizer`` wired end-to-end into a
   Lightning Trainer: the visualisation tensors must appear in the batch dict
   under the expected keys, and a media logger must receive a figure.
"""

import numpy as np
import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.callbacks import _viz_utils as V


# --------------------------------------------------------------------------- #
# 1. maths
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestVizMaths:
    """Pure-torch building blocks: grid inference, scaling, PCA, attention."""

    def test_infer_grid_square(self):
        assert V.infer_grid_size(196) == (14, 14)
        assert V.infer_grid_size(64) == (8, 8)

    def test_infer_grid_explicit_and_aspect(self):
        assert V.infer_grid_size(200, grid_size=(10, 20)) == (10, 20)
        # 2:1 image aspect, 200 tokens -> 20x10 grid (gh:gw = 2:1)
        assert V.infer_grid_size(200, image_hw=(256, 128)) == (20, 10)

    def test_infer_grid_bad(self):
        with pytest.raises(ValueError):
            V.infer_grid_size(15)  # not square, no hint

    def test_robust_minmax_range(self):
        x = torch.randn(1000, 3) * 5
        y = V.robust_minmax(x, quantile=0.02, dim=0)
        assert y.min() >= 0.0 and y.max() <= 1.0
        assert y.shape == x.shape

    def test_denormalize_with_stats(self):
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        x = torch.zeros(2, 3, 8, 8)  # normalised zeros -> 0.5 grey
        out = V.denormalize_images(x, mean, std)
        assert torch.allclose(out, torch.full_like(out, 0.5))
        assert out.min() >= 0 and out.max() <= 1

    def test_denormalize_without_stats_is_per_image_minmax(self):
        x = torch.randn(3, 3, 8, 8) * 10 + 4
        out = V.denormalize_images(x)
        assert out.min() >= 0 and out.max() <= 1
        # each image should hit both extremes
        for i in range(3):
            assert out[i].min() == pytest.approx(0.0, abs=1e-5)
            assert out[i].max() == pytest.approx(1.0, abs=1e-5)

    def test_pca_rgb_shape_and_range(self):
        tokens = torch.randn(4, 64, 48)
        rgb = V.pca_tokens_to_rgb(tokens, grid_hw=(8, 8))
        assert rgb.shape == (4, 3, 8, 8)
        assert rgb.min() >= 0 and rgb.max() <= 1

    def test_pca_rgb_is_deterministic_basis(self):
        # A rank-1 structured signal must dominate PC1 -> stable coloring.
        torch.manual_seed(0)
        base = torch.randn(64, 16)
        tokens = base.unsqueeze(0) + 0.001 * torch.randn(2, 64, 16)
        rgb = V.pca_tokens_to_rgb(tokens, grid_hw=(8, 8), normalize_features=False)
        # two near-identical images -> near-identical PCA images
        assert (rgb[0] - rgb[1]).abs().mean() < 0.05

    def test_pca_per_image_basis_is_independent(self):
        # Two images with different dominant directions. Per-image PCA colours
        # each by its own basis; per-batch shares one. The two modes must
        # therefore disagree, and per-image must still be shape/range-valid.
        torch.manual_seed(0)
        d0 = torch.randn(16)
        d1 = torch.randn(16)
        img0 = torch.linspace(-3, 3, 64)[:, None] * d0[None, :]
        img1 = torch.linspace(-3, 3, 64)[:, None] * d1[None, :]
        tokens = torch.stack([img0, img1], dim=0) + 0.01 * torch.randn(2, 64, 16)
        rgb_batch = V.pca_tokens_to_rgb(
            tokens, grid_hw=(8, 8), normalize_features=False, per_image=False
        )
        rgb_image = V.pca_tokens_to_rgb(
            tokens, grid_hw=(8, 8), normalize_features=False, per_image=True
        )
        assert rgb_image.shape == (2, 3, 8, 8)
        assert rgb_image.min() >= 0 and rgb_image.max() <= 1
        # The two strategies should not produce identical colourings.
        assert (rgb_batch - rgb_image).abs().mean() > 1e-3

    def test_pca_foreground_trick_paints_background(self):
        # Build tokens whose first PCA component clearly splits two groups.
        torch.manual_seed(0)
        D = 16
        direction = torch.randn(D)
        direction = direction / direction.norm()
        # half patches strongly +direction (fg), half -direction (bg)
        signs = torch.cat([torch.ones(32), -torch.ones(32)])
        tokens = (signs[:, None] * direction[None, :] * 5).unsqueeze(0)
        tokens = tokens + 0.01 * torch.randn(1, 64, D)
        rgb = V.pca_tokens_to_rgb(
            tokens,
            grid_hw=(8, 8),
            normalize_features=False,
            foreground_threshold=0.5,
            background_color=(0.0, 0.0, 0.0),
        )
        flat = rgb.reshape(3, -1).T  # (64, 3)
        n_black = (flat.sum(dim=1) < 1e-4).sum().item()
        # roughly half the patches should be painted background-black
        assert 20 <= n_black <= 44

    def test_process_cls_attention_shapes(self):
        B, Hh, T = 2, 6, 65  # 64 patches + 1 CLS
        attn = torch.rand(B, Hh, T, T)
        maps, mask = V.process_cls_attention(
            attn, grid_hw=(8, 8), num_prefix_tokens=1, threshold=0.6
        )
        assert maps.shape == (B, Hh, 8, 8)
        assert mask.shape == (B, Hh, 8, 8)
        assert maps.min() >= 0 and maps.max() <= 1
        assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}

    def test_process_cls_attention_mask_hits_peak(self):
        # One patch gets almost all the CLS attention -> mask must include it.
        B, Hh, T = 1, 1, 17  # 16 patches + CLS
        attn = torch.zeros(B, Hh, T, T)
        attn[0, 0, 0, :] = 0.001
        attn[0, 0, 0, 5] = 10.0  # CLS attends hard to token 5 (patch index 4)
        maps, mask = V.process_cls_attention(
            attn, grid_hw=(4, 4), num_prefix_tokens=1, threshold=0.6
        )
        peak_patch = 4  # token 5 minus 1 prefix
        assert mask.reshape(-1)[peak_patch] == 1.0
        assert maps.reshape(-1)[peak_patch] == pytest.approx(1.0)

    def test_process_cls_attention_row_input(self):
        # Already-reduced CLS row (B, heads, patches) with no prefix.
        maps, mask = V.process_cls_attention(
            torch.rand(2, 3, 64), grid_hw=(8, 8), num_prefix_tokens=0, threshold=None
        )
        assert maps.shape == (2, 3, 8, 8)
        assert mask is None

    def test_process_cls_attention_bad_shape(self):
        with pytest.raises(ValueError):
            V.process_cls_attention(torch.rand(4, 10), grid_hw=(3, 3))


# --------------------------------------------------------------------------- #
# 2. rendering + emission
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestRenderAndEmit:
    """Figure rendering and logger/disk emission in isolation."""

    def test_render_grid_figure_returns_uint8_hwc(self):
        cells = [[torch.rand(3, 16, 16), torch.rand(3, 16, 16)] for _ in range(2)]
        arr = V.render_grid_figure(
            cells, col_titles=["a", "b"], title="t", dpi=60, cell_size=1.0
        )
        assert arr.dtype == torch.uint8
        assert arr.dim() == 3 and arr.shape[2] == 3
        assert arr.shape[0] > 0 and arr.shape[1] > 0

    def test_apply_colormap_shape(self):
        rgb = V.apply_colormap(torch.rand(8, 8), cmap="inferno")
        assert rgb.shape == (3, 8, 8)
        assert rgb.min() >= 0 and rgb.max() <= 1

    def test_emit_figure_routes_to_logger(self):
        from types import SimpleNamespace

        calls = []

        class _FakeLogger:
            def log_image(self, key, images, step=None, caption=None):
                calls.append((key, len(images), step))

        trainer = SimpleNamespace(
            loggers=[_FakeLogger()], default_root_dir=".", global_step=7
        )
        arr = torch.zeros(4, 4, 3, dtype=torch.uint8)
        ok = V.emit_figure(trainer, "val/pca", arr, step=7, caption="e0")
        assert ok is True
        assert calls == [("val/pca", 1, 7)]

    def test_emit_figure_no_logger_is_noop(self):
        from types import SimpleNamespace

        # No media logger -> nothing logged, no manual disk I/O, no crash.
        trainer = SimpleNamespace(loggers=[])
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        ok = V.emit_figure(trainer, "val/pca", arr)
        assert ok is False


# --------------------------------------------------------------------------- #
# 3. end-to-end wiring
# --------------------------------------------------------------------------- #
def _make_vit_stub_module(B, N, D, heads, img_size, C):
    """Tiny module whose forward emits ViT-like tokens + attention keys."""
    import torch.nn as nn

    torch.manual_seed(0)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(C * img_size * img_size, D))

    def forward(self, batch, stage):
        B_ = batch["image"].shape[0]
        pooled = self.encoder(batch["image"])  # (B, D)
        # Fake token grid + prefix (CLS) token: (B, N+1, D)
        tokens = pooled[:, None, :].expand(B_, N + 1, D).contiguous()
        tokens = tokens + 0.01 * torch.randn_like(tokens)
        batch["tokens"] = tokens
        # Fake attention (B, heads, T, T), T = N + 1
        T = N + 1
        batch["attn"] = torch.softmax(
            torch.randn(B_, heads, T, T, device=pooled.device), dim=-1
        )
        return batch

    return spt.Module(encoder=encoder, forward=forward, optim=None)


def _make_data(B, C, img_size):
    x = torch.randn(B, C, img_size, img_size)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return B

        def __getitem__(self, idx):
            return {"image": x[idx]}

    dl = torch.utils.data.DataLoader(_DS(), batch_size=B)
    return spt.data.DataModule(train=dl, val=dl)


@pytest.mark.unit
class TestVisualizersEndToEnd:
    """Both callbacks wired into a real Lightning Trainer + Manager."""

    def test_both_callbacks_populate_batch_and_log(self, tmp_path):
        import lightning as pl

        B, N, D, heads, img_size, C = 4, 16, 32, 3, 32, 3
        module = _make_vit_stub_module(B, N, D, heads, img_size, C)
        data = _make_data(B, C, img_size)

        captured = {}

        class _Capture(pl.pytorch.callbacks.Callback):
            def on_validation_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
            ):
                if batch_idx == 0 and isinstance(batch, dict):
                    captured["keys"] = set(batch.keys())
                    captured["pca_shape"] = tuple(batch["pca"].shape)
                    captured["attn_shape"] = tuple(batch["cls_attn"].shape)

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
                logged.append(key)

        pca = spt.callbacks.PCATokenVisualizer(
            name="pca",
            features="tokens",
            image="image",
            num_prefix_tokens=1,
            grid_size=(4, 4),
            foreground_threshold=None,
        )
        attn = spt.callbacks.AttentionVisualizer(
            name="cls_attn",
            attention="attn",
            image="image",
            num_prefix_tokens=1,
            grid_size=(4, 4),
            threshold=0.6,
            head_reduction="all",
        )

        trainer = pl.Trainer(
            max_epochs=1,
            num_sanity_val_steps=0,
            limit_train_batches=1,
            limit_val_batches=1,
            callbacks=[pca, attn, _Capture()],
            logger=_MediaLogger(),
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()

        # Dict keys populated with image-resolution tensors
        assert "pca" in captured["keys"]
        assert "cls_attn" in captured["keys"]
        assert "cls_attn_mask" in captured["keys"]
        assert captured["pca_shape"] == (B, 3, img_size, img_size)
        assert captured["attn_shape"] == (B, 1, img_size, img_size)
        # Both figures reached the media logger
        assert "val/pca" in logged
        assert "val/cls_attn" in logged

    def test_missing_key_does_not_crash(self, tmp_path):
        import lightning as pl

        B, C, img_size = 4, 3, 32
        module = _make_vit_stub_module(B, 16, 32, 3, img_size, C)
        data = _make_data(B, C, img_size)

        # Point at a key the forward never produces -> should warn + skip.
        pca = spt.callbacks.PCATokenVisualizer(
            name="pca", features="does_not_exist", image="image", grid_size=(4, 4)
        )
        trainer = pl.Trainer(
            max_epochs=1,
            num_sanity_val_steps=0,
            limit_train_batches=1,
            limit_val_batches=1,
            callbacks=[pca],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()  # must not raise
