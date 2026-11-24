"""JEM: loss terms against references, view construction, training and resume."""

import lightning as pl
import pytest
import timm
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence

import stable_pretraining as spt
from stable_pretraining.data.transforms import AlignedMultiCrop
from stable_pretraining.forward import _jem_mask
from stable_pretraining.losses import JEMLoss

pytestmark = pytest.mark.unit

SIZES = {"teacher": 32, "global": 32, "local_0": 16, "local_1": 16}


def _eva(depth: int = 3) -> torch.nn.Module:
    """A small DINOv3-style encoder: registers and axial RoPE, no class token."""
    return timm.models.eva.Eva(
        img_size=32,
        patch_size=8,
        embed_dim=32,
        depth=depth,
        num_heads=2,
        class_token=False,
        num_reg_tokens=1,
        global_pool="",
        num_classes=0,
        dynamic_img_size=True,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type="dinov3",
    )


def _vit(depth: int = 3) -> torch.nn.Module:
    return timm.models.VisionTransformer(
        img_size=32,
        patch_size=8,
        embed_dim=32,
        depth=depth,
        num_heads=2,
        class_token=False,
        reg_tokens=1,
        global_pool="avg",
        num_classes=0,
        dynamic_img_size=True,
    )


def _model(**kwargs) -> spt.JEM:
    kwargs = {"encoder_name": _eva(), "num_categories": 16, **kwargs}
    return spt.JEM(
        head_hidden_dim=32, head_bottleneck_dim=8, structure_dim=16, **kwargs
    )


class _Images(torch.utils.data.Dataset):
    """Eight random images, each given as the views JEM trains on."""

    def __init__(self):
        self.views = AlignedMultiCrop(SIZES, patch_size=8)

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict:
        image = torch.rand(3, 50, 70, generator=torch.Generator().manual_seed(index))
        return self.views({"image": image, "label": index % 2})


def _batch(size: int = 4) -> dict:
    images = _Images()
    return torch.utils.data.default_collate([images[i] for i in range(size)])


def _logits(seed: int, batch: int = 6, patches: int = 5, categories: int = 7):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, patches, categories, generator=generator)


def _log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return JEMLoss().log_probabilities(logits)


class TestJEMLoss:
    """Each term against a value computed another way."""

    def test_alignment_matches_kl_divergence(self):
        loss = JEMLoss(temperature=0.1, masked_weight=0.8, visible_weight=0.2)
        student, teacher = _logits(0), _logits(1)
        masked = torch.rand(6, 5, generator=torch.Generator().manual_seed(2)) < 0.5
        overlap = torch.rand(6, 5, generator=torch.Generator().manual_seed(3)) < 0.7
        divergence = kl_divergence(
            Categorical(logits=teacher / 0.1), Categorical(logits=student / 0.1)
        )
        expected = (
            0.8 * divergence[overlap & masked].mean()
            + 0.2 * divergence[overlap & ~masked].mean()
        )
        value = loss.alignment(
            _log_probabilities(student), _log_probabilities(teacher), masked, overlap
        )
        torch.testing.assert_close(value, expected)

    def test_alignment_ignores_patches_outside_the_overlap(self):
        loss = JEMLoss()
        student, teacher = (
            _log_probabilities(_logits(0)),
            _log_probabilities(_logits(1)),
        )
        masked = torch.zeros(6, 5, dtype=torch.bool)
        overlap = torch.zeros(6, 5, dtype=torch.bool)
        assert loss.alignment(student, teacher, masked, overlap) == 0

    def test_global_regularizer_matches_entropies(self):
        logits = _logits(0)
        distribution = Categorical(logits=logits / 0.1)
        marginal = Categorical(probs=distribution.probs.mean(0))
        expected = distribution.entropy().mean() - marginal.entropy().mean()
        torch.testing.assert_close(
            JEMLoss().global_regularizer(_log_probabilities(logits)), expected
        )

    def test_global_regularizer_of_a_batch_spread_over_all_categories(self):
        logits = 1e4 * torch.eye(7).repeat_interleave(5, dim=0).view(7, 5, 7)
        torch.testing.assert_close(
            JEMLoss().global_regularizer(_log_probabilities(logits)),
            -torch.tensor(7.0).log(),
        )

    def test_global_regularizer_of_a_collapsed_batch_is_zero(self):
        logits = _logits(0)[:1].expand(6, -1, -1)
        torch.testing.assert_close(
            JEMLoss().global_regularizer(_log_probabilities(logits)),
            torch.tensor(0.0),
            rtol=0,
            atol=1e-6,
        )

    def test_structural_regularizer_matches_pairwise_cosines(self):
        generator = torch.Generator().manual_seed(0)
        student = torch.randn(2, 4, 3, generator=generator)
        teacher = torch.randn(2, 4, 6, generator=generator)
        overlap = torch.tensor([[True, True, False, True], [True, True, True, True]])
        normed = F.layer_norm(teacher, (6,))
        expected = []
        for b in range(2):
            pairs = [
                (i, j)
                for i in range(4)
                for j in range(4)
                if i != j and overlap[b, i] and overlap[b, j]
            ]
            expected.append(
                sum(
                    (
                        F.cosine_similarity(student[b, i], student[b, j], dim=0)
                        - F.cosine_similarity(normed[b, i], normed[b, j], dim=0)
                    )
                    ** 2
                    for i, j in pairs
                )
                / len(pairs)
            )
        torch.testing.assert_close(
            JEMLoss().structural_regularizer(student, teacher, overlap),
            torch.stack(expected).mean(),
        )

    def test_structural_regularizer_is_zero_for_matching_similarities(self):
        teacher = torch.randn(2, 4, 6, generator=torch.Generator().manual_seed(0))
        overlap = torch.ones(2, 4, dtype=torch.bool)
        value = JEMLoss().structural_regularizer(
            3 * F.layer_norm(teacher, (6,)), teacher, overlap
        )
        torch.testing.assert_close(value, torch.tensor(0.0), rtol=0, atol=1e-6)

    @pytest.mark.parametrize(
        "logits",
        [1e4 * _logits(0), torch.zeros(6, 5, 7), _logits(0)[:1].expand(6, -1, -1)],
        ids=["huge", "zero", "collapsed"],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_categorical_terms_stay_finite_on_extreme_logits(self, logits, dtype):
        loss = JEMLoss()
        logits = logits.to(dtype).clone().requires_grad_()
        student = loss.log_probabilities(logits)
        teacher = loss.log_probabilities(logits.detach().flip(-1))
        masked = torch.rand(6, 5, generator=torch.Generator().manual_seed(1)) < 0.5
        overlap = torch.ones(6, 5, dtype=torch.bool)
        value = loss.alignment(
            student, teacher, masked, overlap
        ) + loss.global_regularizer(student)
        value.backward()
        assert value.dtype == torch.float32
        assert torch.isfinite(value) and torch.isfinite(logits.grad).all()

    @pytest.mark.parametrize("scale", [0.0, 1e-20, 1e10], ids=["zero", "tiny", "huge"])
    def test_structural_regularizer_stays_finite_on_extreme_features(self, scale):
        generator = torch.Generator().manual_seed(0)
        student = (scale * torch.randn(2, 4, 3, generator=generator)).requires_grad_()
        teacher = scale * torch.randn(2, 4, 6, generator=generator)
        overlap = torch.ones(2, 4, dtype=torch.bool)
        value = JEMLoss().structural_regularizer(student, teacher, overlap)
        value.backward()
        assert torch.isfinite(value) and torch.isfinite(student.grad).all()


class TestAlignedMultiCrop:
    """Views of one image on a shared patch grid."""

    @pytest.mark.parametrize("seed", range(20))
    def test_overlapping_patches_hold_the_same_pixels(self, seed):
        torch.manual_seed(seed)
        views = AlignedMultiCrop(SIZES, patch_size=8)({"image": torch.rand(3, 50, 70)})
        teacher = views.pop("teacher")
        for view in views.values():
            top, left = ((view["offset"] - teacher["offset"]) * 8).tolist()
            size = view["image"].shape[-1]
            rows = slice(max(top, 0), min(top + size, 32))
            columns = slice(max(left, 0), min(left + size, 32))
            shared = teacher["image"][:, rows, columns]
            assert shared.numel() >= 0.1 * view["image"].numel()
            torch.testing.assert_close(
                view["image"][
                    :,
                    rows.start - top : rows.stop - top,
                    columns.start - left : columns.stop - left,
                ],
                shared,
            )

    def test_views_keep_the_sample_and_pass_through_the_transform(self):
        def double(view: dict) -> dict:
            return {**view, "image": 2 * view["image"]}

        views = AlignedMultiCrop(SIZES, transform=double, patch_size=8)(
            {"image": torch.ones(3, 40, 40), "label": 3}
        )
        assert list(views) == list(SIZES)
        for name, view in views.items():
            assert view["label"] == 3
            assert view["image"].shape == (3, SIZES[name], SIZES[name])
            torch.testing.assert_close(view["image"], torch.full_like(view["image"], 2))

    @pytest.mark.parametrize("sizes", [{"global": 32}, {"teacher": 32, "local": 12}])
    def test_refuses_sizes_without_teacher_or_off_the_grid(self, sizes):
        with pytest.raises(ValueError, match="teacher"):
            AlignedMultiCrop(sizes, patch_size=8)


class TestJEMMask:
    """The patches masked in the student views."""

    def test_random_masks_cover_exactly_the_ratio(self):
        masks = _jem_mask(8, 6, 6, 0.5, blocks=False)
        assert masks.shape == (8, 36) and (masks.sum(1) == 18).all()

    def test_block_masks_cover_about_the_ratio(self):
        masks = _jem_mask(64, 14, 14, 0.65, blocks=True)
        share = masks.float().mean(1)
        assert (share >= 0.65 - 1 / 196).all() and share.mean() < 0.75

    def test_masks_differ_between_calls_and_views(self):
        first, second = (
            _jem_mask(4, 14, 14, 0.65, True),
            _jem_mask(4, 14, 14, 0.65, True),
        )
        assert not torch.equal(first, second) and not torch.equal(first[0], first[1])

    def test_block_masks_are_made_of_large_connected_regions(self):
        torch.manual_seed(0)
        masks = _jem_mask(32, 14, 14, 0.65, blocks=True).view(32, 14, 14)
        torus = masks.roll(1, 1) & masks.roll(-1, 1) | masks.roll(1, 2) & masks.roll(
            -1, 2
        )
        assert (masks & torus).sum() / masks.sum() > 0.5


def _patchwise_vit() -> torch.nn.Module:
    """A ViT cut down to its patch embedding, so a patch token sees its pixels only."""
    vit = _vit(depth=1)
    vit.blocks = torch.nn.ModuleList([torch.nn.Identity()])
    vit.norm = torch.nn.Identity()
    torch.nn.init.zeros_(vit.pos_embed)
    return vit


class TestJEMForward:
    """The forward function on real views."""

    @pytest.mark.parametrize("encoder", [_eva, _vit], ids=["rope", "absolute"])
    def test_training_returns_the_loss_terms_and_trains_the_student_only(self, encoder):
        model = _model(encoder_name=encoder())
        out = model(_batch(), stage="fit")
        out["loss"].backward()
        assert out["embedding"].shape == (4, 32) and out["label"].shape == (4,)
        for name in ("loss", "alignment_loss", "global_loss", "structure_loss"):
            assert out[name].ndim == 0 and torch.isfinite(out[name])
        for wrapper in (model.backbone, model.projector):
            assert all(p.grad is None for p in wrapper.teacher.parameters())
            assert any(p.grad.abs().sum() > 0 for p in wrapper.student.parameters())
        assert model.mask_token.grad.abs().sum() > 0

    def test_single_image_batch_returns_the_embedding_without_loss(self):
        model = _model().eval()
        out = model(
            {"image": torch.rand(2, 3, 32, 32), "label": torch.tensor([0, 1])},
            stage="validate",
        )
        assert out["embedding"].shape == (2, 32)
        assert "loss" not in out and out["label"].tolist() == [0, 1]

    def test_unmasked_shifted_views_align_exactly_on_shared_patches(self):
        model = _model(
            encoder_name=_patchwise_vit(),
            target_layers=(1,),
            structure_layers=(0,),
            mask_probability=0.0,
        )
        out = model(_batch(), stage="fit")
        torch.testing.assert_close(
            out["alignment_loss"], torch.tensor(0.0), rtol=0, atol=1e-7
        )

    def test_views_matched_at_the_wrong_patches_do_not_align(self):
        model = _model(
            encoder_name=_patchwise_vit(),
            target_layers=(1,),
            structure_layers=(0,),
            mask_probability=0.0,
        )
        batch = _batch()
        batch["teacher"]["offset"] = batch["teacher"]["offset"] + 1
        assert model(batch, stage="fit")["alignment_loss"] > 1e-5

    def test_builds_the_dinov3_encoder_without_class_token_by_default(self):
        model = spt.JEM(num_categories=16)
        encoder = model.backbone.student
        assert isinstance(encoder, timm.models.eva.Eva) and encoder.rope is not None
        assert encoder.cls_token is None and encoder.num_prefix_tokens == 4
        assert model.target_layers == (4, 8, 12)
        assert model.structure_layers == (0, 4, 8)
        assert model.layer_weights == (0.25, 0.25, 1.0)

    def test_refuses_target_layers_that_do_not_end_with_the_last_block(self):
        with pytest.raises(ValueError, match="last block"):
            _model(target_layers=(1, 2))


class _TeacherAtStart(pl.Callback):
    """Keeps the teacher's weights as they are when training starts."""

    def on_train_start(self, trainer, pl_module):
        self.state = {
            name: value.clone()
            for name, value in pl_module.state_dict().items()
            if ".teacher." in name
        }


def _fit(model: spt.JEM, max_epochs: int, **manager) -> pl.Trainer:
    trainer = pl.Trainer(
        callbacks=[_TeacherAtStart()],
        accelerator="cpu",
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(_Images(), batch_size=4)
    )
    spt.Manager(trainer=trainer, module=model, data=data, seed=0, **manager)()
    return trainer


def test_manager_trains_and_resumes_with_the_ema_teacher(tmp_path):
    spt.set(cache_dir=str(tmp_path / "runs"), default_callbacks={"env_dump": False})
    model = _model(ema_decay_start=0.5)
    trainer = _fit(model, max_epochs=1)
    assert trainer.global_step == 2
    start = trainer.callbacks[0].state
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    moved = [name for name in start if not torch.equal(start[name], saved[name])]
    assert any(name.startswith("backbone.") for name in moved)
    assert any(name.startswith("projector.") for name in moved)
    assert not torch.equal(
        saved["backbone.teacher.patch_embed.proj.weight"],
        saved["backbone.student.patch_embed.proj.weight"],
    )
    trainer.save_checkpoint(tmp_path / "last.ckpt")

    resumed = _model(ema_decay_start=0.5)
    trainer = _fit(
        resumed,
        max_epochs=2,
        ckpt_path=str((tmp_path / "last.ckpt").resolve()),
        weights_only=False,
    )
    assert trainer.global_step == 4
    for name, value in trainer.callbacks[0].state.items():
        torch.testing.assert_close(value, saved[name])
    final = resumed.state_dict()
    assert all(torch.isfinite(value).all() for value in final.values())
    assert not torch.equal(
        final["backbone.teacher.patch_embed.proj.weight"],
        saved["backbone.teacher.patch_embed.proj.weight"],
    )
