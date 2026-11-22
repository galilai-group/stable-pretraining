"""Value and gradient checks for the BayesNCL loss and gate primitives."""

import math

import pytest
import torch
from torch.distributions import Bernoulli, kl_divergence

from stable_pretraining.losses import BayesNCLLoss, NTXEntLoss
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


def test_kl_value_and_gradient_match_torch_distributions():
    probabilities = torch.tensor(
        [[0.2, 0.5, 0.8], [0.3, 0.6, 0.9]], dtype=torch.float64, requires_grad=True
    )
    actual = bernoulli_kl(probabilities, prior_probability=0.8)
    prior = Bernoulli(probabilities.new_tensor(0.8))
    expected = kl_divergence(Bernoulli(probabilities), prior).sum(1).mean()

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


@pytest.mark.parametrize("kl_weight", [0.0, 0.3])
def test_loss_matches_weighted_objective_and_backpropagates(kl_weight):
    z_i = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    z_j = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0]], requires_grad=True)
    probabilities_i = torch.tensor(
        [[0.2, 0.5, 0.8], [0.3, 0.6, 0.9]], requires_grad=True
    )
    probabilities_j = torch.tensor(
        [[0.4, 0.7, 0.6], [0.5, 0.8, 0.2]], requires_grad=True
    )

    actual = BayesNCLLoss(temperature=0.2, kl_weight=kl_weight)(
        z_i, z_j, probabilities_i, probabilities_j
    )
    posterior = Bernoulli(torch.cat([probabilities_i, probabilities_j], dim=0))
    expected_kl = kl_divergence(posterior, Bernoulli(0.8)).sum(1).mean()
    expected = NTXEntLoss(temperature=0.2)(z_i, z_j) + kl_weight * expected_kl
    assert actual.ndim == 0
    torch.testing.assert_close(actual, expected)

    actual.backward()
    for representation in (z_i, z_j):
        assert torch.isfinite(representation.grad).all()
        assert representation.grad.abs().sum() > 0
    for probabilities in (probabilities_i, probabilities_j):
        assert torch.isfinite(probabilities.grad).all()
        if kl_weight == 0:
            torch.testing.assert_close(
                probabilities.grad, torch.zeros_like(probabilities)
            )
        else:
            assert probabilities.grad.abs().sum() > 0


def test_loss_and_gradients_are_finite_when_all_gates_are_closed():
    logits_i = torch.full((2, 3), -1.0, requires_grad=True)
    logits_j = torch.full((2, 3), -2.0, requires_grad=True)
    gates_i, probabilities_i = straight_through_gate(logits_i)
    gates_j, probabilities_j = straight_through_gate(logits_j)
    assert torch.count_nonzero(gates_i) == torch.count_nonzero(gates_j) == 0

    loss = BayesNCLLoss()(gates_i, gates_j, probabilities_i, probabilities_j)
    assert torch.isfinite(loss)
    loss.backward()
    for logits in (logits_i, logits_j):
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"temperature": 0.0}, "temperature"),
        ({"kl_weight": -0.1}, "kl_weight"),
        ({"prior_probability": 1.1}, "prior_probability"),
    ],
)
def test_loss_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BayesNCLLoss(**kwargs)
