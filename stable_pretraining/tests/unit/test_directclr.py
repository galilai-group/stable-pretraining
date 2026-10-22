"""Numerical, gradient, and public-API contracts for DirectCLR."""

import copy
from importlib import import_module
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import stable_pretraining as spt
from stable_pretraining.losses import DirectCLRLoss, NTXEntLoss

pytestmark = pytest.mark.unit


def reference(a, b, temperature=0.1):
    """Independent dense formula, without the library's InfoNCE helper."""
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scores = a @ b.T / temperature
    return (
        (scores.logsumexp(dim=1) - scores.diagonal()).mean()
        + (scores.logsumexp(dim=0) - scores.diagonal()).mean()
    ) / 2


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
def test_reference_values_and_gradients(temperature):
    generator = torch.Generator().manual_seed(17)
    a = torch.randn(5, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    b = torch.randn(5, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    actual = DirectCLRLoss(temperature)(a, b)
    expected = reference(a, b, temperature)
    torch.testing.assert_close(actual, expected)
    for grad, ref_grad in zip(
        torch.autograd.grad(actual, (a, b)), torch.autograd.grad(expected, (a, b))
    ):
        torch.testing.assert_close(grad, ref_grad)


def test_gradcheck():
    torch.manual_seed(11)
    a = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(DirectCLRLoss(0.3), (a, b))


def test_closed_form_and_same_view_negatives_are_absent():
    z = torch.eye(4, dtype=torch.float64)
    expected = torch.log1p(3 * torch.exp(torch.tensor(-10.0, dtype=z.dtype)))
    torch.testing.assert_close(DirectCLRLoss()(z, z), expected)
    assert not torch.isclose(DirectCLRLoss()(z, z), NTXEntLoss(0.1)(z, z))
    collapsed = torch.ones(4, 3)
    torch.testing.assert_close(
        DirectCLRLoss()(collapsed, collapsed), torch.tensor(4.0).log()
    )


def test_view_symmetry_and_joint_sample_permutation():
    torch.manual_seed(13)
    a, b = torch.randn(5, 6), torch.randn(5, 6)
    permutation = torch.tensor([2, 4, 1, 0, 3])
    loss = DirectCLRLoss()
    torch.testing.assert_close(loss(a, b), loss(b, a))
    torch.testing.assert_close(loss(a, b), loss(a[permutation], b[permutation]))


@pytest.mark.parametrize("temperature", [0, -1, float("inf"), float("nan")])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        DirectCLRLoss(temperature)


@pytest.mark.parametrize(
    "shapes",
    [((0, 3), (0, 3)), ((2, 0), (2, 0)), ((2, 3), (3, 3)), ((2, 3, 1), (2, 3, 1))],
)
def test_invalid_shapes(shapes):
    with pytest.raises(ValueError, match="shapes"):
        DirectCLRLoss()(torch.randn(shapes[0]), torch.randn(shapes[1]))


def test_invalid_dtype():
    with pytest.raises(ValueError, match="floating point"):
        DirectCLRLoss()(
            torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.long)
        )
    with pytest.raises(ValueError, match="dtypes"):
        DirectCLRLoss()(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_is_promoted_and_zero_features_are_finite(dtype):
    a = torch.zeros(3, 4, dtype=dtype, requires_grad=True)
    b = torch.ones(3, 4, dtype=dtype, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = DirectCLRLoss()(a, b)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, reference(a.float(), b.float()))
    # Zero-vector normalization has a very large derivative in half precision;
    # test the loss here, and gradients on nondegenerate half inputs below.
    a = torch.randn(3, 4, dtype=dtype, requires_grad=True)
    b = torch.randn(3, 4, dtype=dtype, requires_grad=True)
    DirectCLRLoss()(a, b).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def make_model(dim=3):
    model = spt.Module(
        forward=spt.forward.directclr,
        backbone=nn.Linear(4, 6, bias=False),
        subspace_dim=dim,
        directclr_loss=DirectCLRLoss(),
    )
    model.log = Mock()
    return model


def paired_batch():
    torch.manual_seed(21)
    labels = torch.tensor([1, 0, 1, 2])
    return {
        "views": [
            {"image": torch.randn(4, 4), "label": labels.clone()} for _ in range(2)
        ]
    }


def test_full_features_subspace_gradients_and_no_projector():
    model, batch = make_model(), paired_batch()
    out = model(batch, stage="fit")
    assert out["embedding"].shape == (8, 6)
    torch.testing.assert_close(out["projection"], out["embedding"][:, :3])
    torch.testing.assert_close(out["label"], batch["views"][0]["label"].repeat(2))
    assert not hasattr(model, "projector")
    out["loss"].backward()
    grad = model.backbone.weight.grad
    assert grad[:3].abs().sum() > 0
    assert torch.count_nonzero(grad[3:]) == 0
    assert spt.methods.DirectCLR is spt.DirectCLR


def test_ssl_loss_is_independent_of_labels():
    model, batch = make_model(), paired_batch()
    before = model(batch, stage="fit")["loss"]
    for view in batch["views"]:
        view.pop("label")
    after = model(batch, stage="fit")["loss"]
    torch.testing.assert_close(before, after)


def test_evaluation_returns_unsliced_features_without_loss():
    model = make_model().eval()
    image = torch.randn(4, 4)
    with torch.no_grad():
        out = model({"image": image}, stage="validate")
        torch.testing.assert_close(out["embedding"], model.backbone(image))
        assert "loss" not in out and "projection" not in out
        assert not out["embedding"].requires_grad


@pytest.mark.parametrize("dim", [0, -1, 1.5, True])
def test_invalid_subspace_dimension(dim):
    with pytest.raises(ValueError, match="subspace_dim"):
        make_model(dim)(paired_batch(), stage="fit")


def test_training_rejects_single_view_and_invalid_width():
    with pytest.raises(ValueError, match="two views"):
        make_model()({"image": torch.randn(4, 4)}, stage="fit")
    with pytest.raises(ValueError, match="subspace_dim"):
        make_model(7)(paired_batch(), stage="fit")
    batch = paired_batch()
    batch["views"].append(batch["views"][0])
    with pytest.raises(ValueError, match="two views"):
        make_model()(batch, stage="fit")


def test_misaligned_labels_are_rejected():
    batch = paired_batch()
    batch["views"][1]["label"] = torch.ones(3)
    with pytest.raises(ValueError, match="labels"):
        make_model()(batch, stage="fit")


def image_encoder():
    return nn.Sequential(
        nn.Conv2d(3, 6, 1, bias=False),
        nn.BatchNorm2d(6),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


def test_composable_forward_matches_method_class():
    model = spt.DirectCLR(encoder_name=image_encoder(), subspace_dim=3)
    module = spt.Module(
        forward=spt.forward.directclr,
        backbone=copy.deepcopy(model.backbone),
        subspace_dim=3,
        directclr_loss=DirectCLRLoss(),
    )
    module.log = Mock()
    first, second = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
    batch = {"views": [{"image": first}, {"image": second}]}
    tensor_output, dict_output = model(first, second), module(batch, stage="fit")
    assert isinstance(
        tensor_output,
        import_module("stable_pretraining.methods.directclr").DirectCLROutput,
    )
    assert model.embed_dim == 6
    for key in ("loss", "embedding", "projection"):
        torch.testing.assert_close(getattr(tensor_output, key), dict_output[key])
    assert tensor_output.embedding.shape == (8, 6)
    assert tensor_output.projection.shape == (8, 3)
    for actual, expected in zip(
        torch.autograd.grad(tensor_output.loss, model.backbone.parameters()),
        torch.autograd.grad(dict_output["loss"], module.backbone.parameters()),
    ):
        torch.testing.assert_close(actual, expected)


def test_method_class_single_view_returns_full_features_and_zero_loss():
    model = spt.DirectCLR(image_encoder().double(), subspace_dim=2).eval()
    images = torch.randn(4, 3, 8, 8, dtype=torch.float64)
    with torch.no_grad():
        output = model(images)
        torch.testing.assert_close(output.embedding, model.backbone(images))
    assert output.embedding.shape == (4, 6)
    assert output.loss.shape == () and output.loss.item() == 0
    assert output.loss.dtype == output.embedding.dtype
    assert output.loss.device == output.embedding.device
    assert output.projection is None and not output.embedding.requires_grad


@pytest.mark.parametrize("dimension", [0, -1, 1.5, True, 7])
def test_method_class_rejects_invalid_subspace_at_construction(dimension):
    with pytest.raises(ValueError, match="subspace_dim"):
        spt.DirectCLR(image_encoder(), subspace_dim=dimension)


def test_shape_probe_preserves_bn_buffers_mixed_modes_and_rng():
    encoder = image_encoder().train()
    encoder[1].eval()
    before = copy.deepcopy(encoder.state_dict())
    modes = [m.training for m in encoder.modules()]
    rng = torch.random.get_rng_state().clone()
    model = spt.DirectCLR(encoder, subspace_dim=6)
    assert model.backbone is encoder and model.embed_dim == 6
    assert [m.training for m in encoder.modules()] == modes
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name, tensor in encoder.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_method_class_forwards_timm_encoder_options(monkeypatch):
    requests = []

    def build(name, **kwargs):
        requests.append((name, kwargs))
        return image_encoder()

    monkeypatch.setattr(
        import_module("stable_pretraining.methods.directclr"), "from_timm", build
    )
    model = spt.DirectCLR("tiny", subspace_dim=3, low_resolution=True, pretrained=True)
    assert model.embed_dim == 6
    assert requests == [
        ("tiny", {"num_classes": 0, "low_resolution": True, "pretrained": True})
    ]


def test_method_class_rejects_nonflat_encoder_features():
    with pytest.raises(ValueError, match="flat"):
        spt.DirectCLR(nn.Identity(), subspace_dim=2)


@pytest.mark.gpu
def test_cuda_autocast_matches_float32():
    torch.manual_seed(8)
    a = torch.randn(8, 16, device="cuda", requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = DirectCLRLoss()(a, b)
    torch.testing.assert_close(actual, reference(a, b))
    actual.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
