"""Value and gradient checks for BayesNCL gate and KL primitives."""

import math

import pytest
import torch
from torch.distributions import Bernoulli, kl_divergence

from stable_pretraining.losses.bayesncl import bernoulli_kl, straight_through_gate

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_gate_is_binary_with_sigmoid_gradients(temperature):
    logits = torch.tensor([[-2.0, 0.0, 2.0]], requires_grad=True)
    gates, probabilities = straight_through_gate(logits, temperature=temperature)

    torch.testing.assert_close(gates, torch.tensor([[0.0, 0.0, 1.0]]))
    torch.testing.assert_close(probabilities, torch.sigmoid(logits / temperature))

    gates.sum().backward()
    expected_gradient = probabilities.detach() * (1 - probabilities.detach())
    torch.testing.assert_close(logits.grad, expected_gradient / temperature)


def test_gate_supports_a_custom_threshold():
    gates, _ = straight_through_gate(
        torch.tensor([[0.0, 1.0, 2.0]]), temperature=2.0, threshold=0.7
    )
    torch.testing.assert_close(gates, torch.tensor([[0.0, 0.0, 1.0]]))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_gate_rejects_invalid_temperatures(temperature):
    with pytest.raises(ValueError, match="temperature"):
        straight_through_gate(torch.zeros(1, 2), temperature=temperature)


def test_kl_value_and_gradient_match_torch_distributions():
    probabilities = torch.tensor(
        [[0.2, 0.5, 0.8], [0.3, 0.6, 0.9]], dtype=torch.float64, requires_grad=True
    )
    actual = bernoulli_kl(probabilities, prior_probability=0.8)
    expected = kl_divergence(Bernoulli(probabilities), Bernoulli(0.8)).sum(1).mean()

    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, probabilities, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, probabilities)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_kl_is_zero_at_prior_and_sums_features_not_samples():
    torch.testing.assert_close(
        bernoulli_kl(torch.full((2, 3), 0.8), 0.8), torch.tensor(0.0)
    )
    expected = torch.tensor(3 * 0.5 * math.log(4 / 3))
    for batch_size in (1, 4):
        actual = bernoulli_kl(torch.full((batch_size, 3), 0.5), 0.25)
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("prior", [0.0, 0.8, 1.0])
def test_kl_clamps_saturated_probabilities_like_reference(prior):
    probabilities = torch.tensor([[0.0, 1.0]], requires_grad=True)
    actual = bernoulli_kl(probabilities, prior)
    expected = kl_divergence(
        Bernoulli(torch.tensor([[0.01, 0.99]])),
        Bernoulli(torch.tensor(prior).clamp(0.01, 0.99)),
    ).sum()

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(probabilities.grad).all()


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_probability_settings_raise(value):
    with pytest.raises(ValueError, match="threshold"):
        straight_through_gate(torch.zeros(1, 2), threshold=value)
    with pytest.raises(ValueError, match="prior_probability"):
        bernoulli_kl(torch.full((1, 2), 0.5), value)


def test_kl_rejects_an_input_without_batch_and_feature_axes():
    with pytest.raises(ValueError, match="shape"):
        bernoulli_kl(torch.tensor([0.2, 0.8]), 0.8)
