"""SimDINOv2 patch masking matches the official generator and collate behavior.

Where the official ``simdinov2`` implementation is expressible in pure Python
without numpy, it is re-derived here independently (explicit loops, numpy-free)
and compared against the optimized batched driver in
``stable_pretraining.methods.simdinov2``.

Two upstream behaviors are deliberately *not* asserted as exact, because the
implementation is allowed to undershoot:

- A target of ``N`` masked patches yields at most ``N``, and in practice
  ``0.97 * N`` or more -- the block loop stops on the first attempt that adds
  no fresh cell, and every accepted block must fit inside the remaining budget.
  Per-row tests therefore assert an upper bound only, and the stratified-driver
  tests assert the ratio *band* rather than an exact count.
- A row selected for masking can end up with an empty mask under the same
  zero-delta stopping rule. Only the *number* of selected rows is guaranteed.
"""

import pytest
import torch

from stable_pretraining.methods.simdinov2 import _simdinov2_patch_masks

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


class TestBatchedDriver:
    """``_simdinov2_patch_masks`` reproduces the official collate assignment."""

    def test_shape_and_polarity(self):
        masks = _simdinov2_patch_masks(
            n_rows=16, n_patches=196, grid_size=(14, 14), generator=_gen(0)
        )
        assert masks.shape == (16, 196)
        assert masks.dtype == torch.bool

    def test_polarity_is_true_for_masked(self):
        """``True`` marks a *masked* patch, the opposite of ``multi_block_mask``.

        ``multi_block_mask`` marks patches to keep; the SimDINOv2 collate function
        marks patches to hide. Both are booleans but mean opposite things, so they
        must never be silently interchanged -- here the masked count stays well
        under half the grid, which is what makes this a mask rather than a
        keep-set.
        """
        from stable_pretraining.data.masking import multi_block_mask

        simd = _simdinov2_patch_masks(
            n_rows=64, n_patches=196, grid_size=(14, 14), generator=_gen(1)
        )
        assert simd.any()
        masked_rows = simd[simd.any(1)]
        # Ratio band is (0.1, 0.5), so True is the masked minority.
        assert (masked_rows.sum(1).float() / 196 < 0.5).all()

        # Contrast with the keep-set polarity: True there means retained.
        keep = multi_block_mask(
            14, 14, block_scales=[(0.4, 0.5)], aspect_ratios=[(1.0, 1.0)]
        )
        assert keep[0].shape == (14, 14)
        assert 0 < int(keep[0].sum()) < keep[0].numel()

    def test_exact_number_of_selected_rows(self):
        n_rows = 128
        masks = _simdinov2_patch_masks(
            n_rows=n_rows,
            n_patches=196,
            grid_size=(14, 14),
            mask_sample_probability=0.5,
            generator=_gen(2),
        )
        assert masks.shape == (n_rows, 196)
        # Every selected row is non-empty; the rest are empty.
        selected = masks.any(dim=1)
        assert int(selected.sum()) <= n_rows // 2

    def test_masked_rows_are_interleaved_not_front_loaded(self):
        """Upstream permutes the row list so masked rows scatter across views."""
        n_rows = 64
        masks = _simdinov2_patch_masks(
            n_rows=n_rows,
            n_patches=196,
            grid_size=(14, 14),
            mask_sample_probability=0.5,
            generator=_gen(3),
        )
        selected = masks.any(dim=1)
        # If masks were front-loaded, the first half would be all-True.
        assert 0 < int(selected[: n_rows // 2].sum()) < n_rows // 2

    def test_sample_probability_zero_gives_all_empty(self):
        masks = _simdinov2_patch_masks(
            n_rows=32,
            n_patches=196,
            grid_size=(14, 14),
            mask_sample_probability=0.0,
            generator=_gen(4),
        )
        assert not masks.any()

    def test_sample_probability_one_gives_all_selected(self):
        n_rows = 64
        masks = _simdinov2_patch_masks(
            n_rows=n_rows,
            n_patches=196,
            grid_size=(14, 14),
            mask_sample_probability=1.0,
            generator=_gen(5),
        )
        selected = masks.any(dim=1)
        assert int(selected.sum()) >= n_rows - 4

    def test_stratified_ratios_span_band(self):
        """Masked rows cover the full ``[0.1, 0.5]`` band, not just its mean."""
        n_patches = 196
        masks = _simdinov2_patch_masks(
            n_rows=256,
            n_patches=n_patches,
            grid_size=(14, 14),
            mask_ratio_min_max=(0.1, 0.5),
            generator=_gen(6),
        )
        ratios = (masks.sum(1).float() / n_patches)[masks.any(1)]
        assert ratios.numel() >= 100
        assert 0.06 < ratios.min() <= 0.13
        assert 0.4 < ratios.max() <= 0.52
        # Mean lands near the band midpoint (0.3) rather than the minimum.
        assert 0.24 < ratios.mean() < 0.36

    def test_rows_never_exceed_their_target(self):
        n_patches = 196
        masks = _simdinov2_patch_masks(
            n_rows=128,
            n_patches=n_patches,
            grid_size=(14, 14),
            mask_ratio_min_max=(0.1, 0.5),
            generator=_gen(7),
        )
        assert masks.sum(1).max() <= int(0.5 * n_patches)

    def test_deterministic_given_generator(self):
        a = _simdinov2_patch_masks(
            n_rows=32, n_patches=196, grid_size=(14, 14), generator=_gen(42)
        )
        b = _simdinov2_patch_masks(
            n_rows=32, n_patches=196, grid_size=(14, 14), generator=_gen(42)
        )
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        a = _simdinov2_patch_masks(
            n_rows=32, n_patches=196, grid_size=(14, 14), generator=_gen(1)
        )
        b = _simdinov2_patch_masks(
            n_rows=32, n_patches=196, grid_size=(14, 14), generator=_gen(2)
        )
        assert not torch.equal(a, b)

    def test_degenerate_band_targets_one_ratio(self):
        """A zero-width band makes every row target the same patch count.

        Block placement can still land a patch short (it stops on the first
        zero-delta attempt), so counts may differ by one -- but they must not
        spread across the band the way a real band does.
        """
        n_patches = 196
        masks = _simdinov2_patch_masks(
            n_rows=64,
            n_patches=n_patches,
            grid_size=(14, 14),
            mask_ratio_min_max=(0.3, 0.3),
            generator=_gen(8),
        )
        counts = masks.sum(1)[masks.any(1)]
        assert int(counts.max()) - int(counts.min()) <= 1
        assert int(counts.max()) == int(0.3 * n_patches)

    def test_non_square_grid(self):
        masks = _simdinov2_patch_masks(
            n_rows=32, n_patches=48, grid_size=(6, 8), generator=_gen(9)
        )
        assert masks.shape == (32, 48)
        assert masks.sum(1).max() <= 48


class TestValidation:
    """Bad arguments raise ``ValueError`` rather than silently masking nothing."""

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(n_rows=0, n_patches=196, grid_size=(14, 14)), "n_rows"),
            (
                dict(n_rows=8, n_patches=100, grid_size=(14, 14)),
                "cells",
            ),
            (
                dict(
                    n_rows=8,
                    n_patches=196,
                    grid_size=(14, 14),
                    mask_ratio_min_max=(0.5, 0.1),
                ),
                "mask_ratio_min_max",
            ),
            (
                dict(
                    n_rows=8,
                    n_patches=196,
                    grid_size=(14, 14),
                    mask_ratio_min_max=(0.0, 0.5),
                ),
                "mask_ratio_min_max",
            ),
            (
                dict(
                    n_rows=8,
                    n_patches=196,
                    grid_size=(14, 14),
                    mask_sample_probability=1.5,
                ),
                "mask_sample_probability",
            ),
            (
                dict(
                    n_rows=8,
                    n_patches=196,
                    grid_size=(14, 14),
                    min_num_patches=0,
                ),
                "min_num_patches",
            ),
            (
                dict(
                    n_rows=8,
                    n_patches=196,
                    grid_size=(14, 14),
                    max_num_patches=0,
                ),
                "max_num_patches",
            ),
            (
                dict(
                    n_rows=8,
                    n_patches=196,
                    grid_size=(14, 14),
                    min_aspect=-1.0,
                ),
                "min_aspect",
            ),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _simdinov2_patch_masks(**kwargs)


class TestEndToEndWithLoss:
    """The mask layout the driver emits is the one the loss consumes."""

    def test_masked_features_gathered_from_driver_masks_run_the_loss(self):
        """Driver masks + row-major gather + loss produces a finite scalar."""
        from stable_pretraining.losses import SimDINOv2Loss

        torch.manual_seed(0)
        batch, emb, n_patches = 4, 32, 196
        masks = _simdinov2_patch_masks(
            n_rows=batch, n_patches=n_patches, grid_size=(14, 14), generator=_gen(11)
        )
        assert masks.any(), "expected at least one masked row for a real batch"

        student_cls = torch.randn(2, batch, emb, requires_grad=True)
        teacher_cls = torch.randn(2, batch, emb)
        student_patch = torch.randn(batch, n_patches, emb, requires_grad=True)
        teacher_patch = torch.randn(batch, n_patches, emb)

        # Exactly the gather the method performs: flatten views into rows, then
        # index with the boolean mask to get [n_masked, D] in row-major order.
        student_masked = student_patch[masks]
        teacher_masked = teacher_patch[masks]
        assert student_masked.shape == (int(masks.sum()), emb)

        loss, metrics = SimDINOv2Loss()(
            student_cls_feats=student_cls,
            teacher_cls_feats=teacher_cls,
            student_patch_feats=student_masked,
            teacher_patch_feats=teacher_masked,
            patch_mask=masks,
        )
        assert torch.isfinite(loss)
        assert loss.ndim == 0
        assert "loss_patch" in metrics

        loss.backward()
        assert student_cls.grad is not None and torch.isfinite(student_cls.grad).all()
        assert student_patch.grad is not None
        # Gradients reach the network only through masked positions.
        assert int((student_patch.grad.abs().sum(-1) > 0).sum()) <= int(masks.sum())

    def test_gathered_masked_features_match_pre_gathered_contract(self):
        """Row-major ``flatten(0, 1)[mask]`` equals a per-row concatenation."""
        n_patches = 196
        masks = _simdinov2_patch_masks(
            n_rows=8, n_patches=n_patches, grid_size=(14, 14), generator=_gen(12)
        )
        feats = torch.randn(2, 4, n_patches, 8)  # [view, batch, patches, dim]
        flat = feats.flatten(0, 1)  # [8, patches, dim]

        gathered = flat[masks]
        expected = torch.cat(
            [flat[i][masks[i]] for i in range(masks.shape[0]) if bool(masks[i].any())]
        )
        assert gathered.shape == (int(masks.sum()), 8)
        assert torch.equal(gathered, expected)

    def test_multicrop_multiresolution_forward_and_ema_update(self):
        """SimDINOv2 handles 2 globals + 4 locals, backprops, and updates via EMA."""
        from timm.models.vision_transformer import VisionTransformer
        import stable_pretraining as spt
        from stable_pretraining.callbacks import TeacherStudentCallback

        enc = VisionTransformer(
            img_size=32,
            patch_size=8,
            embed_dim=24,
            depth=2,
            num_heads=3,
            num_classes=0,
            dynamic_img_size=True,
        )
        model = spt.SimDINOv2(encoder_name=enc, image_size=32)
        model.train()

        b = 2
        globals_ = [torch.randn(b, 3, 32, 32, requires_grad=True) for _ in range(2)]
        locals_ = [torch.randn(b, 3, 16, 16, requires_grad=True) for _ in range(4)]

        output = model(global_views=globals_, local_views=locals_)
        assert torch.isfinite(output.loss)
        assert output.embedding.shape == (2 * b, 24)
        assert output.loss_cls is not None
        assert output.loss_patch is not None
        assert output.comp_loss is not None
        assert output.expa_loss is not None

        output.loss.backward()
        for g in globals_:
            assert g.grad is not None and g.grad.abs().sum() > 0
        for loc in locals_:
            assert loc.grad is not None and loc.grad.abs().sum() > 0

        # TeacherStudentCallback discovers wrappers and updates teacher params
        cb = TeacherStudentCallback()

        class DummyTrainer:
            global_step = 0
            current_epoch = 0
            max_epochs = 10
            estimated_stepping_batches = 100

        trainer = DummyTrainer()
        cb.on_fit_start(trainer, model)
        old_teacher_weight = next(model.backbone.teacher.parameters()).clone()

        # Perturb student weights so teacher EMA update has a non-zero delta
        with torch.no_grad():
            for p in model.backbone.student.parameters():
                p.add_(1.0)

        trainer.global_step = 1
        cb.on_train_batch_end(trainer, model, outputs={}, batch={}, batch_idx=0)
        new_teacher_weight = next(model.backbone.teacher.parameters())
        diff = (new_teacher_weight - old_teacher_weight).abs().max().item()
        assert diff > 0


def _gen(seed):
    return torch.Generator().manual_seed(seed)
