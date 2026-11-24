"""DCL objective values, gradients, stability, and forward contract."""

import importlib
import math
from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

import stable_pretraining as spt
from stable_pretraining.forward import dcl
from stable_pretraining.losses import DCLLoss

pytestmark = pytest.mark.unit


def _reference(z_i, z_j, temperature):
    features = F.normalize(torch.cat([z_i, z_j]), dim=-1)
    n = z_i.size(0)
    losses = []
    for i in range(2 * n):
        positive = (i + n) % (2 * n)
        negative_indices = [j for j in range(2 * n) if j not in (i, positive)]
        negative_logits = features[i] @ features[negative_indices].T / temperature
        positive_logit = features[i] @ features[positive] / temperature
        losses.append(torch.logsumexp(negative_logits, dim=0) - positive_logit)
    return torch.stack(losses).mean()


def _backbone():
    return torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(3, 8),
    )


def _batch():
    return {
        "views": [
            {"image": torch.randn(4, 3, 8, 8), "label": torch.arange(4)}
            for _ in range(2)
        ]
    }


@pytest.mark.parametrize("temperature", [0.01, 0.1, 0.5])
def test_loss_matches_reference_values_and_gradients(temperature):
    torch.manual_seed(0)
    z_i = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
    z_j = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
    actual = DCLLoss(temperature)(z_i, z_j)
    expected = _reference(z_i, z_j, temperature)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, (z_i, z_j))
    expected_grads = torch.autograd.grad(expected, (z_i, z_j))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_loss_excludes_positive_and_self_and_can_be_negative():
    features = torch.eye(4)
    actual = DCLLoss(0.1)(features, features)
    torch.testing.assert_close(actual, torch.tensor(math.log(6) - 10))
    assert actual < 0


def test_loss_passes_finite_difference_gradcheck():
    torch.manual_seed(1)
    inputs = tuple(
        torch.randn(3, 4, dtype=torch.double, requires_grad=True) for _ in range(2)
    )
    assert torch.autograd.gradcheck(DCLLoss(temperature=0.2), inputs)


@pytest.mark.parametrize("temperature", [1e-4, 0.1])
@pytest.mark.parametrize("scale", [0.0, 1e-8, 1.0, 1e8])
def test_loss_and_gradients_remain_finite(temperature, scale):
    torch.manual_seed(2)
    inputs = tuple((scale * torch.randn(4, 6)).requires_grad_() for _ in range(2))
    loss = DCLLoss(temperature)(*inputs)
    assert torch.isfinite(loss)
    for gradient in torch.autograd.grad(loss, inputs):
        assert torch.isfinite(gradient).all()
    expected = _reference(*(x.double() for x in inputs), temperature)
    torch.testing.assert_close(loss.double(), expected, atol=2e-3, rtol=2e-5)


def test_identical_features_have_expected_loss_and_zero_gradients():
    inputs = tuple(torch.ones(3, 4, requires_grad=True) for _ in range(2))
    loss = DCLLoss()(*inputs)
    torch.testing.assert_close(loss, torch.tensor(math.log(4)))
    for gradient in torch.autograd.grad(loss, inputs):
        torch.testing.assert_close(
            gradient, torch.zeros_like(gradient), atol=1e-6, rtol=0
        )


def test_loss_is_symmetric_and_invariant_to_feature_scale():
    z_i, z_j = torch.randn(4, 6), torch.randn(4, 6)
    loss = DCLLoss()
    torch.testing.assert_close(loss(z_i, z_j), loss(z_j, z_i))
    torch.testing.assert_close(loss(z_i, z_j), loss(10 * z_i, 2 * z_j))


@pytest.mark.parametrize(
    "shape_i,shape_j",
    [
        ((4, 6), (3, 6)),
        ((4, 6), (4, 5)),
        ((4, 6, 1), (4, 6, 1)),
        ((0, 6), (0, 6)),
        ((4, 0), (4, 0)),
    ],
)
def test_loss_rejects_invalid_features(shape_i, shape_j):
    with pytest.raises(ValueError, match="shape"):
        DCLLoss()(torch.randn(shape_i), torch.randn(shape_j))


def test_loss_requires_negatives():
    with pytest.raises(ValueError, match="negative"):
        DCLLoss()(torch.randn(1, 6), torch.randn(1, 6))


@pytest.mark.parametrize("rank", [0, 1])
def test_loss_uses_global_negatives_and_rank_offsets(monkeypatch, rank):
    loss_module = importlib.import_module("stable_pretraining.losses.joint_embedding")
    z_i = torch.randn(4, 6, requires_grad=True)
    z_j = torch.randn(4, 6, requires_grad=True)
    blocks = [
        F.normalize(torch.cat([z_i[i : i + 2], z_j[i : i + 2]]), dim=-1) for i in (0, 2)
    ]
    monkeypatch.setattr(loss_module, "all_gather", lambda _: blocks)
    monkeypatch.setattr(loss_module, "get_rank", lambda: rank)
    actual = DCLLoss()(z_i[rank * 2 : rank * 2 + 2], z_j[rank * 2 : rank * 2 + 2])

    features = F.normalize(torch.cat([z_i, z_j]), dim=-1)
    losses = []
    for i in list(range(rank * 2, rank * 2 + 2)) + list(
        range(4 + rank * 2, 6 + rank * 2)
    ):
        positive = (i + 4) % 8
        negatives = [j for j in range(8) if j not in (i, positive)]
        losses.append(
            torch.logsumexp(features[i] @ features[negatives].T / 0.1, 0)
            - features[i] @ features[positive] / 0.1
        )
    expected = torch.stack(losses).mean()
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, (z_i, z_j), retain_graph=True),
        torch.autograd.grad(expected, (z_i, z_j)),
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("named", [False, True])
def test_forward_preserves_labels_and_backpropagates(named):
    module = spt.Module(
        forward=dcl,
        backbone=_backbone(),
        projector=torch.nn.Linear(8, 4),
        dcl_loss=DCLLoss(),
    )
    module.log = Mock()
    batch = _batch()
    if named:
        batch = dict(zip(("first", "second"), batch["views"]))
        batch["batch_idx"] = 0
    output = module(batch, stage="fit")
    assert output["embedding"].shape == (8, 8)
    assert output["projection"].shape == (8, 4)
    torch.testing.assert_close(output["label"], torch.arange(4).repeat(2))
    output["loss"].backward()
    for component in (module.backbone, module.projector):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in component.parameters()
        )
    assert module.log.call_args.args[0] == "fit/loss"


@pytest.mark.parametrize("multi_view", [False, True])
def test_evaluation_skips_projector_and_loss(multi_view):
    module = spt.Module(forward=dcl, backbone=_backbone())
    module.eval()
    batch = _batch() if multi_view else _batch()["views"][0]
    output = module(batch, stage="validate")
    assert output["embedding"].shape == (8 if multi_view else 4, 8)
    assert output["label"].shape == (8 if multi_view else 4,)
    assert "loss" not in output and "projection" not in output


@pytest.mark.parametrize("count", [0, 1, 3])
def test_forward_requires_exactly_two_views(count):
    module = spt.Module(forward=dcl, backbone=_backbone())
    view = _batch()["views"][0]
    with pytest.raises(ValueError, match="exactly 2 views"):
        module({"views": [view] * count}, stage="fit")


def test_training_rejects_single_view_batch():
    module = spt.Module(forward=dcl, backbone=_backbone())
    with pytest.raises(ValueError, match="exactly 2 views"):
        module(_batch()["views"][0], stage="fit")


def test_tensor_pair_matches_batch_call_and_backpropagates():
    model = spt.DCL(encoder_name=_backbone(), projector_dims=(16, 4))
    model.log = Mock()
    batch = _batch()
    view1, view2 = (view["image"] for view in batch["views"])
    actual = model(view1, view2)
    expected = model(batch, stage="fit")
    for key in ("loss", "embedding", "projection"):
        torch.testing.assert_close(actual[key], expected[key])
    actual.loss.backward()
    for component in (model.backbone, model.projector):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in component.parameters()
        )
    assert model.log.call_args.args[0] == "fit/loss"


@pytest.mark.parametrize("training", [False, True])
def test_single_tensor_defaults_to_prediction(training, monkeypatch):
    model = spt.DCL(encoder_name=_backbone(), projector_dims=(16, 4))
    model.train(training)
    monkeypatch.setattr(model.projector, "forward", Mock(side_effect=AssertionError))
    images = torch.randn(4, 3, 8, 8)
    output = model(images)
    torch.testing.assert_close(output.embedding, model.backbone(images))
    assert output.loss is None and output.projection is None
    assert model.training is training


@pytest.mark.parametrize("stage", ["val", "validate", "test", "predict"])
def test_explicit_evaluation_stage_skips_loss_while_model_is_training(stage):
    module = spt.Module(forward=dcl, backbone=_backbone())
    output = module(_batch(), stage=stage)
    assert output["embedding"].shape == (8, 8)
    assert "loss" not in output and "projection" not in output


def test_dictionary_accepts_positional_stage():
    model = spt.DCL(encoder_name=_backbone(), projector_dims=(16, 4))
    model.log = Mock()
    batch = _batch()
    actual = model(batch, "fit")
    expected = model(batch, stage="fit")
    torch.testing.assert_close(actual.loss, expected.loss)
    torch.testing.assert_close(actual.label, expected.label)


@pytest.mark.parametrize("single_tensor", [False, True])
def test_adapter_rejects_single_view_training(single_tensor):
    model = spt.DCL(encoder_name=_backbone(), projector_dims=(16, 4))
    batch = _batch()["views"][0]
    with pytest.raises(ValueError, match="exactly 2 views"):
        model(batch["image"] if single_tensor else batch, stage="fit")


@pytest.mark.parametrize(
    "view1,view2,stage,error",
    [
        ({}, "fit", "fit", ValueError),
        ({}, torch.zeros(2, 3, 8, 8), None, ValueError),
        (torch.zeros(2, 3, 8, 8), "fit", None, TypeError),
        ([], None, None, TypeError),
    ],
)
def test_adapter_rejects_ambiguous_or_invalid_inputs(view1, view2, stage, error):
    model = spt.DCL(encoder_name=_backbone(), projector_dims=(16, 4))
    with pytest.raises(error):
        model(view1, view2, stage=stage)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("named", [False, True])
def test_method_matches_composable_forward_outputs_and_gradients(training, named):
    model = spt.DCL(encoder_name=_backbone(), projector_dims=(16, 4))
    module = spt.Module(
        forward=dcl,
        backbone=deepcopy(model.backbone),
        projector=deepcopy(model.projector),
        dcl_loss=DCLLoss(),
    )
    for current in (model, module):
        current.train(training)
        current.log = Mock()
    batch = _batch()
    if named:
        batch = dict(zip(("first", "second"), batch["views"]))
    batch["batch_idx"] = 0
    stage = "fit" if training else "validate"
    actual = model(batch, stage=stage)
    expected = module(batch, stage=stage)
    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    if training:
        actual["loss"].backward()
        expected["loss"].backward()
        for model_param, module_param in zip(model.parameters(), module.parameters()):
            torch.testing.assert_close(model_param.grad, module_param.grad)
        assert model.log.call_args.args[0] == module.log.call_args.args[0]
