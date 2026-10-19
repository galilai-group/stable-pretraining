"""Unit tests for LeHami: the guarantees of its flow and stencils, and the method."""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.methods.lehami import (
    LeHami,
    SeparableHamiltonianFlow,
    causal_momentum,
    centered_momentum,
)

pytestmark = pytest.mark.unit

D = 4


def _flow(action_dim: int = 0, even_kinetic: bool = False) -> SeparableHamiltonianFlow:
    torch.manual_seed(0)
    flow = SeparableHamiltonianFlow(
        D, action_dim, hidden_dim=32, dt=0.7, even_kinetic=even_kinetic
    ).double()
    with torch.no_grad():
        flow.log_scale.normal_(0, 0.3)
    return flow


def _step_jacobian(flow, x: torch.Tensor, u=None) -> torch.Tensor:
    def step(x):
        q, p = flow(x[:D][None], x[D:][None], u)
        return torch.cat([q[0], p[0]])

    return torch.autograd.functional.jacobian(step, x)


def _quadratic(t: torch.Tensor) -> torch.Tensor:
    a, b, c = (
        torch.tensor([1.0, -2.0]),
        torch.tensor([0.5, 3.0]),
        torch.tensor([-0.25, 0.1]),
    )
    return a + b * t[:, None] + c * t[:, None] ** 2


def test_stencils_are_exact_on_quadratics():
    t = torch.arange(10, dtype=torch.float64) * 0.5
    q = _quadratic(t).double()
    slope = torch.tensor([0.5, 3.0]) + 2 * torch.tensor([-0.25, 0.1]) * t[:, None]
    assert torch.allclose(centered_momentum(q, dt=0.5), slope[2:-2].double())
    assert torch.allclose(causal_momentum(q[:6], dt=0.5), slope[5].double())


def test_causal_stencil_ignores_future_and_older_frames():
    q = torch.randn(3, 8, D)
    changed = q.clone()
    changed[:, :3] = torch.randn(3, 3, D)
    assert torch.equal(causal_momentum(q), causal_momentum(changed))
    assert not torch.equal(causal_momentum(q[:, :6]), causal_momentum(q[:, :7]))


@pytest.mark.parametrize("action_dim", [0, 3])
def test_leapfrog_preserves_volume(action_dim):
    flow = _flow(action_dim)
    u = torch.randn(1, action_dim, dtype=torch.float64) if action_dim else None
    for _ in range(3):
        jac = _step_jacobian(flow, torch.randn(2 * D, dtype=torch.float64), u)
        assert torch.linalg.det(jac).item() == pytest.approx(1.0, abs=1e-10)


def test_leapfrog_is_symplectic():
    flow = _flow(2)
    u = torch.randn(1, 2, dtype=torch.float64)
    jac = _step_jacobian(flow, torch.randn(2 * D, dtype=torch.float64), u)
    eye, zero = (
        torch.eye(D, dtype=torch.float64),
        torch.zeros(D, D, dtype=torch.float64),
    )
    omega = torch.cat([torch.cat([zero, eye], 1), torch.cat([-eye, zero], 1)])
    assert torch.allclose(jac.T @ omega @ jac, omega, atol=1e-10)


def test_even_kinetic_makes_the_flow_time_reversible():
    def round_trip(flow):
        q0, p0 = (
            torch.randn(5, D, dtype=torch.float64),
            torch.randn(5, D, dtype=torch.float64),
        )
        q, p = q0, p0
        with torch.no_grad():
            for _ in range(10):
                q, p = flow(q, p)
            p = -p
            for _ in range(10):
                q, p = flow(q, p)
        return (q - q0).abs().max().item(), (-p - p0).abs().max().item()

    assert max(round_trip(_flow(even_kinetic=True))) < 1e-10
    assert max(round_trip(_flow(even_kinetic=False))) > 1e-4


def test_flow_outputs_are_detached_without_grad():
    flow = _flow()
    q = torch.randn(2, D, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        q_next, p_next = flow(q, torch.randn(2, D, dtype=torch.float64))
    assert not q_next.requires_grad and not p_next.requires_grad


def _model(**kwargs) -> LeHami:
    torch.manual_seed(0)
    encoder = nn.Sequential(
        nn.Conv2d(3, 16, 4, 4),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    defaults = dict(
        encoder=encoder,
        latent_dim=8,
        image_size=32,
        projector_hidden_dim=32,
        flow_hidden_dim=32,
        n_slices=16,
    )
    return LeHami(**{**defaults, **kwargs})


@pytest.mark.parametrize("sigreg_on", ["q", "joint"])
def test_lehami_trains_with_actions(sigreg_on):
    model = _model(action_dim=2, sigreg_on=sigreg_on).train()
    video, actions = torch.randn(4, 9, 3, 32, 32), torch.randn(4, 8, 2)
    output = model(video, actions)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.flow.potential_net[0].weight.grad.abs().sum() > 0
    assert model.action_encoder[0].weight.grad.abs().sum() > 0
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert output.embedding.shape == (4, 9, 8)


def test_lehami_without_actions_rolls_out():
    model = _model().eval()
    with torch.no_grad():
        output = model(torch.randn(3, 7, 3, 32, 32))
        q, p = model.rollout(torch.randn(3, 5, 3, 32, 32), steps=6)
    assert torch.isfinite(output.loss)
    assert q.shape == p.shape == (3, 6, 8)


def test_rollout_matches_training_unroll():
    """``rollout`` starts from the same anchor and steps as the training loss."""
    model = _model(action_dim=1).eval()
    video, actions = torch.randn(2, 8, 3, 32, 32), torch.randn(2, 7, 1)
    with torch.no_grad():
        q_all = model.encode(video)
        q, _ = model.rollout(video[:, :4], actions[:, 3:7])
        output = model(video, actions)
    expected = ((q - q_all[:, 4:8]) ** 2).mean(dim=(0, 2))
    assert output.pos_loss.item() == pytest.approx(expected[:2].mean().item(), rel=1e-4)


def test_lehami_rejects_short_clips_and_missing_actions():
    model = _model(action_dim=1)
    with pytest.raises(ValueError, match="at least 7 frames"):
        model(torch.randn(2, 6, 3, 32, 32), torch.randn(2, 5, 1))
    with pytest.raises(ValueError, match="takes actions"):
        model(torch.randn(2, 8, 3, 32, 32))
