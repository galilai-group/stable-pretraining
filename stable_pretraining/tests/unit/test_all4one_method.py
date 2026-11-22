"""All4One dict-forward, optimizer, EMA, and checkpoint integration contracts."""

import copy
import importlib
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import stable_pretraining as spt
from stable_pretraining.forward import all4one
from stable_pretraining.methods import All4One
from stable_pretraining.tests._all4one_helpers import tiny_all4one

pytestmark = pytest.mark.unit


def _model(**kwargs):
    model = tiny_all4one(**kwargs)
    model.log = Mock()
    return model


def _batch(format="list", labels=True):
    views = [{"image": torch.randn(3, 6)} for _ in range(2)]
    if labels:
        for view in views:
            view["label"] = torch.tensor([1, 2, 3])
    if format == "list":
        return {"views": views}
    named = {"first": views[0], "second": views[1]}
    return {"views": named} if format == "nested" else named


@pytest.mark.parametrize(
    "format,stage", [("list", "fit"), ("nested", "fit"), ("named", "train")]
)
def test_training_routes_both_views_before_enqueue_and_trains_all_online_parts(
    format, stage
):
    model = _model()
    batch = _batch(format)
    support = model.queue.queue.get().clone()
    teacher_parameters = [p.clone() for p in model.backbone.teacher.parameters()]
    result = model(batch, stage)
    assert result["embedding"].shape == (6, 8)
    assert not result["embedding"].requires_grad
    assert result["label"].tolist() == [1, 2, 3, 1, 2, 3]
    assert result["neighbours"].shape == (6, 5, 8)
    normalized = F.normalize(result["momentum_projection"], dim=-1)
    expected_indices = (normalized @ support.T).topk(5, dim=1).indices
    torch.testing.assert_close(result["nn_index"], expected_indices)
    torch.testing.assert_close(result["neighbours"], support[expected_indices])
    torch.testing.assert_close(model.queue.queue.out[:3], normalized[:3])
    torch.testing.assert_close(model.queue.queue.out[3:], support[3:])
    assert model.queue.queue.pointer.item() == 3
    assert not result["momentum_projection"].requires_grad
    assert not result["neighbours"].requires_grad
    assert result["loss"].ndim == 0 and torch.isfinite(result["loss"])
    torch.testing.assert_close(
        result["loss"],
        0.5 * result["neighbour_loss"]
        + 0.5 * result["centroid_loss"]
        + result["feature_loss"],
    )
    result["loss"].backward()
    for module in (
        model.backbone.student,
        model.projector.student,
        model.predictor,
        model.centroid_predictor,
        model.attention,
    ):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
    for wrapper in (model.backbone, model.projector):
        assert not wrapper.update_buffers and wrapper.ema_schedule == "step"
        assert all(p.grad is None for p in wrapper.teacher.parameters())
    for before, after in zip(teacher_parameters, model.backbone.teacher.parameters()):
        torch.testing.assert_close(before, after)
    assert model.backbone.teacher[1].num_batches_tracked.item() == 2
    assert model.log.call_args_list[0].args[0] == f"{stage}/loss"


def test_composable_module_matches_class_outputs_and_gradients():
    model = _model()
    composed = spt.Module(
        forward=all4one,
        **{
            name: copy.deepcopy(getattr(model, name))
            for name in (
                "backbone",
                "projector",
                "predictor",
                "centroid_predictor",
                "attention",
                "queue",
                "all4one_loss",
            )
        },
    )
    composed.log = Mock()
    batch = _batch(labels=False)
    actual = model(batch)
    expected = composed(batch, "fit")
    assert "label" not in actual
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name])
    actual["loss"].backward()
    expected["loss"].backward()
    for name, parameter in model.named_parameters():
        other = dict(composed.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad)


@pytest.mark.parametrize(
    "stage,multiview",
    [("validate", False), ("test", True), ("predict", False), ("fit", True)],
)
def test_eval_returns_student_embeddings_without_mutating_state(stage, multiview):
    model = _model().eval()
    batch = (
        _batch()
        if multiview
        else {"image": torch.randn(3, 6), "label": torch.arange(3)}
    )
    state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        images = [v["image"] for v in batch["views"]] if multiview else [batch["image"]]
        expected = torch.cat([model.backbone.student(image) for image in images])
        result = model(batch, stage)
    assert set(result) == {"embedding", "label"}
    torch.testing.assert_close(result["embedding"], expected)
    assert result["label"].shape[0] == expected.shape[0]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name])
    model.log.assert_not_called()


@pytest.mark.parametrize(
    "batch",
    [
        {"image": torch.ones(3, 6)},
        {"views": []},
        {"views": [{"image": torch.ones(3, 6)}]},
        {"views": [{"image": torch.ones(3, 6)}] * 3},
        {"views": [{"image": torch.ones(3, 6)}, {}]},
        {"views": [{"image": torch.ones(3, 6)}, {"image": torch.ones(2, 6)}]},
        {"views": [{"image": torch.ones(1, 6)}] * 2},
    ],
)
def test_invalid_training_batches_fail_before_state_changes(batch):
    model = _model()
    state = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="All4One"):
        model(batch)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name])


def test_cpu_autocast_forward_backward_is_finite():
    model = _model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(_batch())
    assert result["loss"].dtype == torch.float32
    result["loss"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_public_exports_and_backbone_dimension_inference(monkeypatch):
    assert spt.All4One is All4One
    assert "All4One" in spt.__all__ and "All4One" in dir(spt)
    assert spt.forward.all4one is all4one
    assert spt.losses.All4OneLoss is type(_model().all4one_loss)
    backbone = nn.Linear(6, 8)
    backbone.num_features = 8
    factory = Mock(return_value=backbone)
    monkeypatch.setattr(
        importlib.import_module("stable_pretraining.methods.all4one"),
        "from_torchvision",
        factory,
    )
    model = _model(backbone="resnet18", feature_dim=None, low_resolution=False)
    factory.assert_called_once_with(
        "resnet18", low_resolution=False, weights=None, zero_init_residual=False
    )
    assert model.feature_dim == 8
    assert model.backbone.student is backbone


def test_combined_forward_matches_explicit_cross_view_objectives():
    model = _model()
    output = model(_batch())
    target = torch.arange(3)

    def contrast(a, b, temperature):
        logits = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T / temperature
        return F.cross_entropy(logits, target)

    n1, n2 = output["neighbours"][:, 0].chunk(2)
    p1, p2 = output["prediction"].chunk(2)
    c1, c2 = output["centroid"].chunk(2)
    s1, s2 = output["shifted_centroid"].chunk(2)
    z1, z2 = output["projection"].chunk(2)
    m1, m2 = F.normalize(output["momentum_projection"], dim=-1).chunk(2)
    neighbour = (contrast(n1, p2, 0.2) + contrast(n2, p1, 0.2)) / 2
    centroid = (contrast(c1, s2, 0.1) + contrast(c2, s1, 0.1)) / 2
    correlations = [
        F.normalize(m1, dim=0).T @ F.normalize(z2, dim=0),
        F.normalize(m2, dim=0).T @ F.normalize(z1, dim=0),
    ]
    off_mask = ~torch.eye(8, dtype=torch.bool)
    diagonal = (
        torch.stack([(c.diagonal() - 1).square().mean() for c in correlations])
        .mean()
        .sqrt()
    )
    off_diagonal = (
        torch.stack([c[off_mask].square().mean() for c in correlations]).mean().sqrt()
    )
    expected = 0.5 * neighbour + 0.5 * centroid + 2.5 * (diagonal + off_diagonal)
    torch.testing.assert_close(output["loss"], expected)
    parameters = [p for p in model.parameters() if p.requires_grad]
    actual_gradients = torch.autograd.grad(
        output["loss"], parameters, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected, parameters)
    for actual, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
