"""Unit tests for Rectified LpJEPA and distribution matching."""

import math
from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from stable_pretraining import Module
from stable_pretraining.forward import lpjepa
from stable_pretraining.methods.lpjepa import (
    LpJEPA,
    RDMReg,
    SlicedWassersteinDistance,
)

pytestmark = pytest.mark.unit


def _latent_views() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(10)
    return tuple(
        torch.rand(16, 4, generator=generator, requires_grad=True) for _ in range(2)
    )


def _rdmreg_loss_and_grads(
    regularizer: RDMReg, views: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss = regularizer(*views)
    grad1, grad2 = torch.autograd.grad(loss, views)
    return loss.detach(), grad1, grad2


def test_swd_matches_hand_computed_distance_and_gradients():
    samples = torch.tensor(
        [[0.0, 3.0], [2.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    targets = torch.tensor([[3.0, 2.0], [1.0, 4.0]], dtype=torch.float64)
    diagonal = 1 / math.sqrt(2)
    directions = torch.tensor(
        [[1.0, 0.0, diagonal], [0.0, 1.0, diagonal]], dtype=torch.float64
    )

    loss = SlicedWassersteinDistance()(samples, targets, directions)

    # The two axis slices contribute 7 in total; the diagonal contributes 13/2.
    torch.testing.assert_close(loss, loss.new_tensor(9 / 4), rtol=0, atol=1e-12)
    loss.backward()
    expected_grad = samples.new_tensor([[-2 / 3, -2 / 3], [-5 / 6, -7 / 6]])
    torch.testing.assert_close(samples.grad, expected_grad, rtol=0, atol=1e-12)


def test_swd_is_invariant_to_sample_order_and_zero_for_same_distribution():
    samples = torch.tensor([[0.0, 2.0], [3.0, 1.0], [1.0, 4.0]])
    targets = torch.tensor([[2.0, 1.0], [4.0, 0.0], [0.0, 3.0]])
    directions = torch.eye(2)
    distance = SlicedWassersteinDistance()

    expected = distance(samples, targets, directions)
    actual = distance(samples[[2, 0, 1]], targets[[1, 2, 0]], directions)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert distance(samples, samples[[1, 2, 0]], directions).item() == 0


@pytest.mark.parametrize(
    "p,expected",
    [(0.5, math.sqrt(2 / 15)), (1.0, 1 / math.sqrt(2)), (2.0, 1.0)],
)
def test_rdmreg_automatic_sigma_matches_unit_variance_cases(p, expected):
    assert RDMReg(p=p).sigma == pytest.approx(expected, rel=1e-12, abs=0)


def test_rdmreg_preserves_explicit_sigma():
    assert RDMReg(p=0.75, sigma=1.7).sigma == 1.7


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rdmreg_samples_unit_directions_with_matching_dtype(dtype):
    regularizer = RDMReg(num_slices=17)
    reference = torch.empty(8, 4, dtype=dtype)
    directions = regularizer._sample_directions(
        reference, torch.Generator().manual_seed(42)
    )

    assert directions.shape == (4, 17)
    assert directions.dtype == reference.dtype
    assert directions.device == reference.device
    torch.testing.assert_close(
        directions.norm(dim=0), torch.ones(17, dtype=dtype), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("p", [0.75, 1.0, 2.0, 3.0])
def test_rdmreg_target_samples_match_rectified_distribution(p):
    regularizer = RDMReg(p=p)
    reference = torch.empty(8192, 4, dtype=torch.float64, requires_grad=True)
    targets = regularizer._sample_targets(reference, torch.Generator().manual_seed(42))
    scaled_thresholds = reference.new_tensor([0.0, 0.5, 1.0, 2.0])
    empirical_cdf = (
        (targets.flatten()[:, None] <= regularizer.sigma * scaled_thresholds)
        .double()
        .mean(dim=0)
    )
    expected_cdf = 0.5 + 0.5 * torch.special.gammainc(
        reference.new_tensor(1 / p), scaled_thresholds.pow(p) / p
    )

    assert targets.shape == reference.shape
    assert targets.dtype == reference.dtype
    assert targets.device == reference.device
    assert not targets.requires_grad
    assert torch.isfinite(targets).all()
    assert (targets >= 0).all()
    # With 32768 samples, DKW bounds the chance of an error above .02 by 8.3e-12.
    torch.testing.assert_close(empirical_cdf, expected_cdf, rtol=0, atol=0.02)


def test_rdmreg_applies_location_and_scale_before_rectification(monkeypatch):
    regularizer = RDMReg(p=1, mu=1, sigma=2)
    samples = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    monkeypatch.setattr(
        regularizer, "_sample_product_laplace", lambda x, generator: samples
    )

    targets = regularizer._sample_targets(samples, torch.Generator().manual_seed(0))

    torch.testing.assert_close(
        targets, torch.tensor([[0.0, 0.0, 1.0, 3.0, 5.0]]), rtol=0, atol=0
    )


def test_rdmreg_averages_two_view_distances_and_gradients(monkeypatch):
    regularizer = RDMReg(num_slices=2)
    z1 = torch.tensor([[0.0], [2.0]], dtype=torch.float64, requires_grad=True)
    z2 = torch.tensor([[4.0], [1.0]], dtype=torch.float64, requires_grad=True)
    directions = z1.new_tensor([[1.0, -1.0]])
    targets = iter([z1.new_tensor([[3.0], [1.0]]), z2.new_tensor([[0.0], [2.0]])])
    monkeypatch.setattr(
        regularizer, "_sample_directions", lambda x, generator: directions
    )
    monkeypatch.setattr(
        regularizer, "_sample_targets", lambda x, generator: next(targets)
    )

    loss, grad1, grad2 = _rdmreg_loss_and_grads(regularizer, (z1, z2))

    torch.testing.assert_close(loss, z1.new_tensor(7 / 4), rtol=0, atol=1e-12)
    torch.testing.assert_close(
        grad1, z1.new_tensor([[-0.5], [-0.5]]), rtol=0, atol=1e-12
    )
    torch.testing.assert_close(grad2, z2.new_tensor([[1.0], [0.5]]), rtol=0, atol=1e-12)


def test_rdmreg_shares_directions_and_samples_separate_targets(monkeypatch):
    regularizer = RDMReg(num_slices=8)
    calls = []
    swd_forward = regularizer.swd.forward

    def record_swd(samples, targets, directions):
        calls.append((targets.clone(), directions.clone()))
        return swd_forward(samples, targets, directions)

    monkeypatch.setattr(regularizer.swd, "forward", record_swd)
    regularizer(*_latent_views())

    assert len(calls) == 2
    targets1, directions1 = calls[0]
    targets2, directions2 = calls[1]
    torch.testing.assert_close(directions1, directions2, rtol=0, atol=0)
    assert not torch.equal(targets1, targets2)


@pytest.mark.parametrize("p", [0.75, 1.0, 2.0])
def test_rdmreg_reproducibility_is_independent_of_global_rng(p):
    first = RDMReg(num_slices=8, p=p)
    second = RDMReg(num_slices=8, p=p)
    views = _latent_views()

    with torch.random.fork_rng(devices=[]):
        before_first = torch.random.get_rng_state().clone()
        expected = _rdmreg_loss_and_grads(first, views)
        assert torch.equal(torch.random.get_rng_state(), before_first)

        torch.rand(127)
        before_second = torch.random.get_rng_state().clone()
        actual = _rdmreg_loss_and_grads(second, views)
        assert torch.equal(torch.random.get_rng_state(), before_second)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_rdmreg_advances_sampling_once_per_forward():
    regularizer = RDMReg(num_slices=8)
    views = _latent_views()

    assert regularizer.global_step.item() == 0
    first = regularizer(*views)
    assert regularizer.global_step.item() == 1
    second = regularizer(*views)
    assert regularizer.global_step.item() == 2
    assert not torch.equal(first, second)


@pytest.mark.parametrize("p", [0.75, 1.0, 2.0])
def test_rdmreg_checkpoint_restores_next_loss_and_gradients(p):
    regularizer = RDMReg(num_slices=8, p=p)
    views = _latent_views()
    regularizer(*views)
    checkpoint = deepcopy(regularizer.state_dict())
    expected = _rdmreg_loss_and_grads(regularizer, views)

    restored = RDMReg(num_slices=8, p=p)
    restored.load_state_dict(checkpoint)
    actual = _rdmreg_loss_and_grads(restored, views)

    assert restored.global_step.item() == regularizer.global_step.item() == 2
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rdmreg_promotes_half_precision_and_preserves_gradients(dtype):
    views = tuple(view.detach().to(dtype).requires_grad_() for view in _latent_views())
    reference_views = tuple(view.detach().float().requires_grad_() for view in views)

    loss, *grads = _rdmreg_loss_and_grads(RDMReg(num_slices=8), views)
    expected_loss, *expected_grads = _rdmreg_loss_and_grads(
        RDMReg(num_slices=8), reference_views
    )

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    for grad, expected_grad in zip(grads, expected_grads):
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
        torch.testing.assert_close(grad, expected_grad.to(dtype), rtol=0, atol=0)


def test_rdmreg_preserves_double_precision_and_backpropagates():
    views = tuple(view.detach().double().requires_grad_() for view in _latent_views())

    loss, *grads = _rdmreg_loss_and_grads(RDMReg(num_slices=8), views)

    assert loss.dtype == torch.float64
    assert torch.isfinite(loss)
    for grad in grads:
        assert grad.dtype == torch.float64
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


@pytest.fixture
def make_model(monkeypatch):
    def create_encoder(*args, **kwargs):
        return VisionTransformer(
            img_size=32,
            patch_size=8,
            embed_dim=24,
            depth=2,
            num_heads=3,
            num_classes=0,
        )

    monkeypatch.setattr("timm.create_model", create_encoder)

    def create_model(**kwargs):
        config = dict(
            inv_weight=25.0,
            rdmreg_weight=125.0,
            encoder_name="tiny",
            projector=nn.Linear(24, 8),
            n_slices=16,
            pretrained=False,
            drop_path_rate=0.0,
        )
        config.update(kwargs)
        model = LpJEPA(**config)
        monkeypatch.setattr(model, "log_dict", Mock())
        return model

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield create_model


def _image_batch():
    generator = torch.Generator().manual_seed(11)
    return {
        "views": [
            {"image": torch.randn(2, 3, 32, 32, generator=generator)} for _ in range(2)
        ]
    }


def test_lpjepa_training_output_and_gradients(make_model):
    model = make_model().train()
    batch = _image_batch()
    output = model(batch, "fit")

    assert isinstance(output, dict)
    assert set(output) == {"embedding", "projection", "loss", "inv_loss", "rdmreg_loss"}
    for loss in (output["loss"], output["inv_loss"], output["rdmreg_loss"]):
        assert loss.ndim == 0
        assert torch.isfinite(loss)
    assert output["embedding"].shape == (4, 24)
    assert output["projection"].shape == (4, 8)
    assert torch.isfinite(output["projection"]).all()
    assert (output["projection"] >= 0).all()
    with torch.no_grad():
        expected = torch.cat([model.backbone(view["image"]) for view in batch["views"]])
    torch.testing.assert_close(output["embedding"], expected, rtol=1e-5, atol=1e-6)

    output["loss"].backward()
    for module in (model.backbone, model.projector):
        grads = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        assert grads and all(grad is not None for grad in grads)
        assert all(torch.isfinite(grad).all() for grad in grads)
        assert any(grad.abs().sum() > 0 for grad in grads)


@pytest.mark.parametrize(
    "inv_weight,rdmreg_weight", [(2.0, 3.0), (0.0, 3.0), (2.0, 0.0)]
)
def test_lpjepa_forward_uses_configured_loss_weights(
    make_model, inv_weight, rdmreg_weight
):
    model = make_model(inv_weight=inv_weight, rdmreg_weight=rdmreg_weight)
    output = model(_image_batch(), "fit")
    expected = inv_weight * output["inv_loss"] + rdmreg_weight * output["rdmreg_loss"]
    torch.testing.assert_close(output["loss"], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("view_format", ["list", "dict", "named"])
def test_lpjepa_shared_forward_matches_direct_mse_and_preserves_order(
    monkeypatch, view_format
):
    model = Module(
        forward=lpjepa,
        backbone=nn.Identity(),
        projector=nn.Identity(),
        rdmreg=RDMReg(num_slices=2),
        inv_weight=2.0,
        rdmreg_weight=3.0,
    )
    z1 = torch.tensor([[0.0, 2.0], [4.0, 6.0]])
    z2 = torch.tensor([[1.0, 0.0], [2.0, 3.0]])
    views = {
        "second": {"image": z1, "label": torch.tensor([0, 1])},
        "first": {"image": z2, "label": torch.tensor([2, 3])},
    }
    if view_format == "list":
        batch = {"views": list(views.values())}
    elif view_format == "dict":
        batch = {"views": views}
    else:
        batch = views
    regularizer = Mock(return_value=torch.tensor(2.0))
    monkeypatch.setattr(model.rdmreg, "forward", regularizer)
    log_dict = Mock()
    monkeypatch.setattr(model, "log_dict", log_dict)

    output = model(batch, "fit")

    regularizer.assert_called_once()
    for actual, expected in zip(regularizer.call_args.args, (z1, z2)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert output["inv_loss"].item() == pytest.approx(4.5, abs=1e-6)
    assert output["rdmreg_loss"].item() == pytest.approx(2.0, abs=1e-6)
    assert output["loss"].item() == pytest.approx(15.0, abs=1e-6)
    torch.testing.assert_close(output["embedding"], torch.cat([z1, z2]))
    torch.testing.assert_close(output["projection"], torch.cat([z1, z2]))
    assert output["label"].tolist() == [0, 1, 2, 3]
    log_dict.assert_called_once_with(
        {f"fit/{key}": output[key] for key in ("loss", "inv_loss", "rdmreg_loss")},
        on_step=True,
        on_epoch=True,
        sync_dist=True,
        batch_size=2,
    )


def test_lpjepa_default_projector_outputs_2048_features(make_model):
    model = make_model(projector=None)
    shapes = []
    handle = model.projector.register_forward_hook(
        lambda module, inputs, output: shapes.append(output.shape)
    )
    try:
        with torch.no_grad():
            output = model(_image_batch(), "fit")
    finally:
        handle.remove()

    assert shapes == [(2, 2048), (2, 2048)]
    assert output["projection"].shape == (4, 2048)
    assert torch.isfinite(output["loss"])


def test_lpjepa_custom_projector_is_rectified_before_both_losses(
    make_model, monkeypatch
):
    projector = nn.Linear(24, 8)
    bias = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    with torch.no_grad():
        projector.weight.zero_()
        projector.weight[:, 0].fill_(1)
        projector.bias.copy_(bias)
    model = make_model(projector=projector)
    model.backbone = nn.Identity()
    assert model.projector is projector
    regularizer = Mock(wraps=model.rdmreg.forward)
    monkeypatch.setattr(model.rdmreg, "forward", regularizer)
    view1, view2 = torch.zeros(2, 24), torch.zeros(2, 24)
    view1[:, 0] = -1
    view2[:, 0] = 1

    output = model({"views": [{"image": view1}, {"image": view2}]}, "fit")

    regularizer.assert_called_once()
    z1, z2 = regularizer.call_args.args
    expected1 = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]).expand(2, -1)
    expected2 = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expand(2, -1)
    torch.testing.assert_close(z1, expected1, rtol=0, atol=0)
    torch.testing.assert_close(z2, expected2, rtol=0, atol=0)
    torch.testing.assert_close(output["projection"], torch.cat([expected1, expected2]))
    assert output["inv_loss"].item() == pytest.approx(21 / 8, abs=1e-6)


@pytest.mark.parametrize("supply_second_view", [False, True])
def test_lpjepa_evaluation_returns_features_and_projections_without_losses(
    make_model, monkeypatch, supply_second_view
):
    model = make_model().eval()
    views = _image_batch()["views"][: 2 if supply_second_view else 1]
    for index, view in enumerate(views):
        view["label"] = torch.tensor([2 * index, 2 * index + 1])
    batch = {"views": views} if supply_second_view else views[0]
    with torch.no_grad():
        embeddings = [model.backbone(view["image"]) for view in views]
        projections = [model.projector(embedding).relu() for embedding in embeddings]
    projector = Mock(wraps=model.projector.forward)
    regularizer = Mock(side_effect=AssertionError("RDMReg called in evaluation"))
    monkeypatch.setattr(model.projector, "forward", projector)
    monkeypatch.setattr(model.rdmreg, "forward", regularizer)
    model.rdmreg.global_step.fill_(7)

    with torch.no_grad():
        output = model(batch, "validate")

    assert isinstance(output, dict)
    assert set(output) == {"embedding", "projection", "label"}
    assert output["embedding"].shape == (2 * len(views), 24)
    assert output["projection"].shape == (2 * len(views), 8)
    torch.testing.assert_close(
        output["embedding"], torch.cat(embeddings), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        output["projection"], torch.cat(projections), rtol=1e-5, atol=1e-6
    )
    assert output["label"].tolist() == list(range(2 * len(views)))
    assert projector.call_count == len(views)
    regularizer.assert_not_called()
    model.log_dict.assert_not_called()
    assert model.rdmreg.global_step.item() == 7


def test_lpjepa_training_requires_second_view(make_model):
    model = make_model().train()
    batch = _image_batch()["views"][0]
    with pytest.raises(ValueError, match="exactly 2 views"):
        model(batch, "fit")
