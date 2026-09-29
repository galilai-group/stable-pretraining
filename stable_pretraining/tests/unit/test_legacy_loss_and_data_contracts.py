"""Public compatibility APIs preserve reference losses and nested batch structure."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from stable_pretraining.backbone.nn_modules import EMA
from stable_pretraining.callbacks.queues import OrderedQueue, UnsortedQueue
from stable_pretraining.data.collate import Collator, _collapse_nested_dict
from stable_pretraining.losses.joint_embedding import SwAVLoss
from stable_pretraining.losses.reconstruction import MAELoss, mae

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("component", [EMA, OrderedQueue, UnsortedQueue, Collator])
def test_embedded_reference_contracts(component):
    assert component._test() is True


@pytest.mark.parametrize("normalized", [False, True])
def test_functional_mae_matches_masked_mse_reference(normalized):
    target = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    pred = torch.ones_like(target, requires_grad=True)
    mask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    expected_target = target
    if normalized:
        expected_target = (target - target.mean(-1, keepdim=True)) / (
            target.var(-1, keepdim=True) + 1e-6
        ).sqrt()
    expected = (pred - expected_target).square().mean(-1)[mask.bool()].mean()
    actual = mae(target, pred, mask, normalized)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.all(pred.grad[~mask.bool()] == 0)


def test_mae_sum_and_debug_do_not_change_loss(capsys):
    images = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    loss = MAELoss(
        patch_size=2, patch_normalize=False, mask_only=False, reduction="sum"
    )
    target = loss.patchify(images)
    pred = torch.ones_like(target, requires_grad=True)
    mask = torch.ones(2, 4)
    actual = loss(pred, images, mask, debug=True)
    torch.testing.assert_close(actual, (pred - target).square().mean(-1).sum())
    assert "MAE Loss Debug" in capsys.readouterr().out
    actual.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("use_queue", [False, True])
def test_legacy_swav_loss_normalizes_prototypes_and_detaches_assignments(use_queue):
    torch.manual_seed(1)
    x = torch.randn(4, 3, requires_grad=True)
    y = torch.randn(4, 3, requires_grad=True)
    queue = torch.randn(5, 3, requires_grad=True) if use_queue else None
    prototypes = nn.Linear(3, 2, bias=False)
    loss = SwAVLoss(epsilon=0.5, sinkhorn_iterations=10)
    actual = loss(x, y, prototypes, queue)
    xn, yn = nn.functional.normalize(x), nn.functional.normalize(y)
    with torch.no_grad():
        scores = torch.cat(
            [prototypes(xn), prototypes(yn)]
            + ([prototypes(nn.functional.normalize(queue))] if use_queue else [])
        )
        assignments = loss.sinkhorn(scores)
    expected = -0.5 * (
        (
            assignments[4:8]
            * nn.functional.log_softmax(prototypes(xn) / loss.temperature, dim=1)
        )
        .sum(1)
        .mean()
        + (
            assignments[:4]
            * nn.functional.log_softmax(prototypes(yn) / loss.temperature, dim=1)
        )
        .sum(1)
        .mean()
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(prototypes.weight.norm(dim=1), torch.ones(2))
    actual.backward()
    assert x.grad is not None and y.grad is not None
    assert prototypes.weight.grad is not None
    if queue is not None:
        assert queue.grad is None


def test_nested_collation_and_continuous_affinity():
    samples = [
        {
            "image": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
            "metadata": [{"index": 5}, {"index": 6}],
            "feature": [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])],
        }
    ]
    result = Collator(G_from="feature")(samples)
    torch.testing.assert_close(result["G"], torch.tensor([[1.0, 1.0], [1.0, 2.0]]))
    assert result["metadata"]["index"].tolist() == [5, 6]
    base = {"views": [torch.ones(1, 2), torch.zeros(1, 2)]}
    other = {"views": [torch.full((1, 2), 3.0), torch.full((1, 2), 4.0)]}
    joined = _collapse_nested_dict(base, other)
    assert joined["views"][0].tolist() == [[1.0, 1.0], [3.0, 3.0]]
    assert joined["views"][1].tolist() == [[0.0, 0.0], [4.0, 4.0]]


@pytest.mark.parametrize("callable_manager", [False, True])
def test_hydra_run_entrypoint_dispatches_and_prints_config(
    monkeypatch, capsys, callable_manager
):
    from stable_pretraining import run

    manager = Mock() if callable_manager else SimpleNamespace(value=1)
    factory = Mock(return_value=manager)
    monkeypatch.setattr(run, "instantiate_from_config", factory)
    cfg = OmegaConf.create({"seed": 19})
    result = run.main.__wrapped__(cfg)
    factory.assert_called_once_with(cfg)
    assert "seed: 19" in capsys.readouterr().out
    if callable_manager:
        manager.assert_called_once_with()
    else:
        assert result is manager
