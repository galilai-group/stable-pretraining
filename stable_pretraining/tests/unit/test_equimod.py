"""Unit tests for EquiMod (equivariance module on top of SimCLR)."""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from stable_pretraining.data import transforms
from stable_pretraining.losses import NTXEntLoss
from stable_pretraining.methods.equimod import EquiMod

pytestmark = pytest.mark.unit

B = 8
EMBED_DIM = 16


def _tiny_encoder() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, EMBED_DIM, kernel_size=3, stride=2),
        nn.BatchNorm2d(EMBED_DIM),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


def _model(**kwargs) -> EquiMod:
    defaults = dict(
        encoder_name=_tiny_encoder(),
        projector_dims=(32, 8),
        equi_projector_dims=(32, 8),
        param_projector_dims=(8,),
    )
    defaults.update(kwargs)
    return EquiMod(**defaults)


def _inputs(param_dim: int = 15):
    g = torch.Generator().manual_seed(0)
    images = [torch.randn(B, 3, 32, 32, generator=g) for _ in range(3)]
    params = [torch.randn(B, param_dim, generator=g) for _ in range(2)]
    return images, params


def _augmented(blur=None):
    steps = [
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5
        ),
        transforms.RandomGrayscale(p=0.2),
    ]
    if blur is not None:
        steps.append(blur)
    return transforms.Compose(*steps, transforms.ToImage())


def _collated_views(blur=None, n: int = 32):
    """Three-view batch as the dataloader delivers it: [original, aug, aug]."""
    multi_view = transforms.MultiViewTransform(
        [
            transforms.Compose(transforms.RGB(), transforms.ToImage()),
            _augmented(blur),
            _augmented(blur),
        ]
    )
    rng = np.random.default_rng(0)
    samples = [
        multi_view(
            {
                "image": Image.fromarray(
                    rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
                ),
                "label": i % 10,
            }
        )
        for i in range(n)
    ]
    return torch.utils.data.default_collate(samples)["views"]


# --- augmentation-parameter assembly ---------------------------------------


def test_augmentation_params_from_transform_pipeline():
    torch.manual_seed(0)
    views = _collated_views()
    assert "ColorJitter" not in views[0], "un-augmented view must carry no params"

    p1 = EquiMod.augmentation_params(views[1])
    p2 = EquiMod.augmentation_params(views[2])
    assert p1.shape == p2.shape == (32, 15)
    assert p1.dtype == torch.float32
    assert not torch.equal(p1, p2), "each view must keep its own parameters"

    crop, flip, applied, order, factors, gray = (
        p1[:, :4],
        p1[:, 4],
        p1[:, 5],
        p1[:, 6:10],
        p1[:, 10:14],
        p1[:, 14],
    )
    assert torch.equal(crop, views[1]["RandomResizedCrop"])
    assert torch.equal(flip.bool(), views[1]["RandomHorizontalFlip"])
    assert torch.equal(gray.bool(), views[1]["RandomGrayscale"])
    assert set(applied.tolist()) == {0.0, 1.0}, "p=0.5 should give both cases"
    # the order is always a permutation of 0..3, applied or not
    assert torch.equal(order.sort(dim=1).values, torch.arange(4.0).repeat(32, 1))
    on = applied.bool()
    assert torch.equal(factors[on], views[1]["ColorJitter"][on][:, :4])


def test_augmentation_params_skipped_jitter_is_identity():
    view = {
        "ColorJitter": torch.tensor(
            [[0.0] * 8, [1.2, 0.8, 1.1, -0.05, 2.0, 0.0, 3.0, 1.0]]
        )
    }
    p = EquiMod.augmentation_params(view, crop=None, flip=None, grayscale=None)
    assert p.shape == (2, 9)
    skipped = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 0.0])
    assert torch.equal(p[0], skipped)
    assert torch.allclose(
        p[1], torch.tensor([1.0, 2.0, 0.0, 3.0, 1.0, 1.2, 0.8, 1.1, -0.05])
    )


def test_augmentation_params_with_blur():
    torch.manual_seed(0)
    views = _collated_views(blur=transforms.PILGaussianBlur(p=0.5))
    p = EquiMod.augmentation_params(views[1], blur="PILGaussianBlur")
    assert p.shape == (32, 17)
    blurred, sigma = p[:, 15], p[:, 16]
    assert torch.equal(sigma, views[1]["PILGaussianBlur"].reshape(-1))
    assert torch.equal(blurred.bool(), sigma > 0)
    assert set(blurred.tolist()) == {0.0, 1.0}
    assert torch.equal(p[:, :15], EquiMod.augmentation_params(views[1]))


def test_augmentation_params_with_two_sigma_blur_record():
    torch.manual_seed(0)
    blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5)
    views = _collated_views(blur=blur)
    assert views[1]["GaussianBlur"].shape == (32, 2)
    p = EquiMod.augmentation_params(views[1], blur="GaussianBlur")
    assert p.shape == (32, 17)
    assert torch.equal(p[:, 16], views[1]["GaussianBlur"][:, 0])


def test_augmentation_params_missing_record_raises():
    with pytest.raises(KeyError, match="RandomHorizontalFlip.*RandomResizedCrop"):
        EquiMod.augmentation_params({"RandomResizedCrop": torch.zeros(2, 4)})


def test_param_keys_reach_the_equivariance_module():
    model = _model(param_dim=17, param_keys={"blur": "PILGaussianBlur"})
    torch.manual_seed(0)
    views = _collated_views(blur=transforms.PILGaussianBlur(p=0.5))
    assert model.equi.params_from_view(views[1]).shape == (32, 17)


def test_equivariance_module_rejects_non_view_major_rows():
    model = _model()
    with pytest.raises(ValueError, match="multiple"):
        model.equi(
            torch.randn(3, EMBED_DIM), torch.randn(5, EMBED_DIM), torch.randn(5, 15)
        )


# --- model ------------------------------------------------------------------


def _hooks(model: EquiMod) -> dict:
    """Record what the equivariance module sees and produces."""
    seen = {}
    model.equi.projector.register_forward_hook(
        lambda m, i, o: seen.update(y=o.detach())
    )
    model.equi.param_projector.register_forward_pre_hook(
        lambda m, i: seen.update(param_in=i[0].detach())
    )
    model.equi.param_projector.register_forward_hook(
        lambda m, i, o: seen.update(param_out=o.detach())
    )
    model.equi.predictor.register_forward_pre_hook(
        lambda m, i: seen.update(predictor_in=i[0].detach())
    )
    model.equi.predictor.register_forward_hook(
        lambda m, i, o: seen.update(y_hat=o.detach())
    )
    return seen


def test_forward_shapes_and_loss_composition():
    model = _model(equi_weight=0.5).train()
    (original, v1, v2), (p1, p2) = _inputs()
    out = model(v1, v2, original, p1, p2)
    assert out.embedding.shape == (2 * B, EMBED_DIM)
    assert out.projection.shape == (2 * B, 8)
    assert torch.allclose(
        out.loss, out.invariance_loss + 0.5 * out.equivariance_loss, atol=1e-6
    )


def test_predictor_gets_original_embedding_and_each_views_own_params():
    mean, std = torch.full((15,), 2.0), torch.full((15,), 4.0)
    model = _model(param_stats=(mean, std)).train()
    seen = _hooks(model)
    (original, v1, v2), (p1, p2) = _inputs()
    model(v1, v2, original, p1, p2)
    # view k's prediction is conditioned on view k's standardised parameters
    assert torch.allclose(seen["param_in"], (torch.cat([p1, p2]) - 2.0) / 4.0)
    # ...and on the embedding of the un-augmented image, not of the view itself
    y_original = seen["y"][:B]
    assert torch.equal(seen["predictor_in"][:, :8], y_original.repeat(2, 1))
    assert torch.equal(seen["predictor_in"][:, 8:], seen["param_out"])


def test_losses_and_gain_match_independent_reference():
    model = _model().train()
    seen = _hooks(model)
    (original, v1, v2), (p1, p2) = _inputs()
    out = model(v1, v2, original, p1, p2)

    # one backbone pass over [original, view1, view2]: batch-norm statistics
    # are shared by the three, so separate passes would give other features
    h = model.backbone(torch.cat([original, v1, v2]))
    z = model.projector(h[B:])
    assert torch.allclose(out.embedding, h[B:], atol=1e-6)
    assert torch.allclose(out.projection, z, atol=1e-5)
    assert torch.allclose(out.invariance_loss, NTXEntLoss(0.5)(z[:B], z[B:]), atol=1e-5)

    y_views, y_original, y_hat = seen["y"][B:], seen["y"][:B], seen["y_hat"]
    assert torch.allclose(
        out.equivariance_loss, NTXEntLoss(0.2)(y_views, y_hat), atol=1e-5
    )
    gain = (
        F.cosine_similarity(y_views, y_hat)
        - F.cosine_similarity(y_views, y_original.repeat(2, 1))
    ).mean()
    assert torch.allclose(out.equivariance_gain, gain, atol=1e-6)


def test_equivariance_branch_can_fit_a_small_batch():
    torch.manual_seed(0)
    model = _model().train()
    (original, v1, v2), (p1, p2) = _inputs()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        out = model(v1, v2, original, p1, p2)
        first = out.equivariance_loss if first is None else first
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    assert out.equivariance_loss < 0.5 * first
    assert out.equivariance_gain > 0.3
    # the fit depends on the parameters: swapping them between views breaks it
    swapped = model(v1, v2, original, p2, p1)
    assert swapped.equivariance_loss > out.equivariance_loss + 0.5


def test_end_to_end_from_transform_pipeline():
    torch.manual_seed(0)
    views = _collated_views()
    model = _model().train()
    out = model(
        views[1]["image"],
        views[2]["image"],
        views[0]["image"],
        EquiMod.augmentation_params(views[1]),
        EquiMod.augmentation_params(views[2]),
    )
    assert torch.isfinite(out.loss)


def test_equivariance_loss_trains_every_new_module():
    model = _model().train()
    (original, v1, v2), (p1, p2) = _inputs()
    model(v1, v2, original, p1, p2).loss.backward()
    for name in (
        "equi.projector",
        "equi.param_projector",
        "equi.predictor",
        "backbone",
    ):
        grads = [
            p.grad for p in model.get_submodule(name).parameters() if p.requires_grad
        ]
        assert any(g is not None and g.abs().sum() > 1e-4 for g in grads), name


def test_equivariance_loss_reaches_the_backbone():
    with_equi = _model().train()
    without = copy.deepcopy(with_equi)
    without.equi_weight = 0.0
    (original, v1, v2), (p1, p2) = _inputs()
    grads = []
    for model in (with_equi, without):
        model(v1, v2, original, p1, p2).loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.backbone.parameters()]))
    assert not torch.allclose(grads[0], grads[1])


def test_zero_weight_removes_equivariance_gradient():
    model = _model(equi_weight=0.0).train()
    (original, v1, v2), (p1, p2) = _inputs()
    out = model(v1, v2, original, p1, p2)
    assert torch.allclose(out.loss, out.invariance_loss)
    out.loss.backward()
    for p in model.equi.predictor.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0


def test_training_requires_original_and_params():
    model = _model().train()
    (original, v1, v2), (p1, p2) = _inputs()
    with pytest.raises(ValueError, match="un-augmented image"):
        model(v1, v2)
    with pytest.raises(ValueError, match="un-augmented image"):
        model(v1, v2, original, p1)


def test_eval_single_view_returns_embedding_only():
    model = _model().eval()
    (_, v1, _), _ = _inputs()
    with torch.no_grad():
        out = model(v1)
    assert out.embedding.shape == (B, EMBED_DIM)
    assert out.loss == 0 and out.projection is None


def test_fixed_param_stats_standardise_parameters():
    mean, std = torch.full((15,), 2.0), torch.full((15,), 4.0)
    model = _model(param_stats=(mean, std))
    assert model.equi.param_norm is None
    assert "equi.param_mean" in dict(model.named_buffers())
    params = torch.full((3, 15), 6.0)
    assert torch.allclose(model.equi.normalize_params(params), torch.ones(3, 15))


def test_default_param_norm_is_non_affine_batch_norm():
    model = _model()
    assert isinstance(model.equi.param_norm, nn.BatchNorm1d)
    assert not model.equi.param_norm.affine


def test_param_stats_shape_mismatch_raises():
    with pytest.raises(ValueError, match="param_stats"):
        _model(param_stats=(torch.zeros(4), torch.ones(4)))


def test_param_stats_zero_std_raises():
    std = torch.ones(15)
    std[5] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        _model(param_stats=(torch.zeros(15), std))


@pytest.mark.parametrize("param_stats", [None, ([0.0] * 15, [1.0] * 15)])
def test_wrong_parameter_width_raises(param_stats):
    model = _model(param_stats=param_stats).train()
    (original, v1, v2), _ = _inputs()
    wrong = torch.randn(B, 17)
    with pytest.raises(ValueError, match="param_dim"):
        model(v1, v2, original, wrong, wrong)
