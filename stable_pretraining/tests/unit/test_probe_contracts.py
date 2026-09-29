"""Probe pooling, loss aggregation, and gradients against explicit references."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from stable_pretraining.backbone.probe import (
    AutoLinearClassifier,
    LinearProbe,
    MultiHeadAttentiveProbe,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "pooling,shape",
    [("cls", (2, 3, 4)), ("mean", (2, 3, 4)), (None, (2, 2, 2)), ("cls", (2, 4))],
)
@pytest.mark.parametrize("normalization", [None, nn.LayerNorm])
def test_linear_probe_matches_pooling_reference(pooling, shape, normalization):
    model = LinearProbe(4, 3, pooling=pooling, norm_layer=normalization)
    x = (
        torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
        .reshape(shape)
        .requires_grad_()
    )
    pooled = (
        x
        if len(shape) == 2
        else (
            x[:, 0]
            if pooling == "cls"
            else x.mean(1)
            if pooling == "mean"
            else x.flatten(1)
        )
    )
    normalized = pooled if normalization is None else model.norm(pooled)
    expected = model.fc(normalized)
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_attention_probe_is_permutation_invariant_and_differentiable():
    model = MultiHeadAttentiveProbe(4, 3, num_heads=2)
    x = torch.randn(2, 5, 4, requires_grad=True)
    output = model(x)
    torch.testing.assert_close(output, model(x[:, [3, 1, 4, 0, 2]]))
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert model.attn_vectors.grad is not None


@pytest.mark.parametrize(
    "pooling,shape",
    [
        ("cls", (3, 2, 4)),
        ("mean", (3, 2, 4)),
        ("mean", (3, 4, 2, 2)),
        (None, (3, 2, 2)),
        (None, (3, 4)),
    ],
)
def test_auto_probe_losses_equal_sum_of_individual_classifiers(pooling, shape):
    model = AutoLinearClassifier(
        "probe",
        4,
        3,
        pooling=pooling,
        normalization=["none", "norm", "bn"],
        dropout=[0],
        label_smoothing=[0],
    )
    x = torch.randn(*shape, requires_grad=True)
    y = torch.tensor([0, 1, 2])
    module = Mock()
    loss = model(x, y, module)
    logged = module.log_dict.call_args.args[0]
    torch.testing.assert_close(loss, sum(logged.values()))
    assert len(logged) == 3
    loss.backward()
    assert torch.isfinite(x.grad).all()
    for classifier in model.fc.values():
        assert classifier[-1].weight.grad is not None
    model.eval()
    module.reset_mock()
    assert torch.isfinite(model(x.detach(), y, module))
    assert module.log_dict.call_args.args[0] is model.metrics
    assert all(0 <= value <= 1 for value in model.metrics.compute().values())
