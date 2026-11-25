"""Unit tests for LeHami: the guarantees of its flow and stencils, and the method."""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.losses import LeHamiLoss
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


class _HalfSquare(nn.Module):
    """``K(p) = |p|^2 / 2``, standing in for the kinetic MLP."""

    def forward(self, x):
        return 0.5 * (x**2).sum(-1, keepdim=True)


class _ForcedSpring(nn.Module):
    """``V(q, u) = |q|^2 / 2 - q . u`` on ``x = [q, u]``, for the potential MLP."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, x):
        q, u = x[:, : self.d], x[:, self.d :]
        return (0.5 * q**2 - q * u).sum(-1, keepdim=True)


def _analytic(flow: SeparableHamiltonianFlow, d: int) -> SeparableHamiltonianFlow:
    flow.kinetic_net = _HalfSquare()
    flow.potential_net = _ForcedSpring(d)
    return flow


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


def test_leapfrog_step_matches_reference():
    """One step of ``|p / s|^2 / 2 + |q|^2 / 2 - q . u`` against hand-computed values.

    With ``s = (1, 2)``, ``dt = 0.5``, ``q = (1, -0.5)``, ``p = (0.25, 2)`` and
    ``u = (0.5, 0)``: ``p_half = (0.125, 2.125)``, then ``q_next`` and
    ``p_next`` below. Flipping the sign of either momentum half-step changes
    both outputs.
    """
    flow = _analytic(SeparableHamiltonianFlow(2, 2, dt=0.5), 2).double()
    with torch.no_grad():
        flow.log_scale.copy_(torch.tensor([1.0, 2.0], dtype=torch.float64).log())
    q, p = flow(
        torch.tensor([[1.0, -0.5]], dtype=torch.float64),
        torch.tensor([[0.25, 2.0]], dtype=torch.float64),
        torch.tensor([[0.5, 0.0]], dtype=torch.float64),
    )
    expected_q = torch.tensor([[1.0625, -0.234375]], dtype=torch.float64)
    expected_p = torch.tensor([[-0.015625, 2.18359375]], dtype=torch.float64)
    assert torch.allclose(q, expected_q, rtol=0, atol=1e-12)
    assert torch.allclose(p, expected_p, rtol=0, atol=1e-12)


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


@pytest.mark.parametrize("scale", [1e-30, 1e6])
def test_flow_is_finite_on_extreme_states(scale):
    flow = SeparableHamiltonianFlow(D, 2, hidden_dim=32)
    q = (torch.randn(3, D) * scale).requires_grad_(True)
    p = (torch.randn(3, D) * scale).requires_grad_(True)
    q_next, p_next = flow(q, p, torch.randn(3, 2) * scale)
    assert torch.isfinite(q_next).all() and torch.isfinite(p_next).all()
    (q_next.sum() + p_next.sum()).backward()
    grads = [q.grad, p.grad] + [w.grad for w in flow.parameters() if w.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)


def test_flow_runs_in_fp32_on_bf16_inputs():
    flow = SeparableHamiltonianFlow(D, hidden_dim=32)
    q, p = torch.randn(3, D).bfloat16(), torch.randn(3, D).bfloat16()
    q_next, p_next = flow(q, p)
    assert q_next.dtype == p_next.dtype == torch.bfloat16
    assert torch.isfinite(q_next.float()).all() and torch.isfinite(p_next.float()).all()


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


def _batch(video: torch.Tensor, actions=None) -> dict:
    return {"video": video} if actions is None else {"video": video, "action": actions}


def _objective(model, video: torch.Tensor, actions=None, stage: str = "fit"):
    """The training objective of ``model`` on a batch of clips."""
    return model(_batch(video, actions), stage)


@pytest.mark.parametrize("sigreg_on", ["q", "joint"])
def test_lehami_trains_with_actions(sigreg_on):
    model = _model(action_dim=2, sigreg_on=sigreg_on).train()
    video, actions = torch.randn(4, 9, 3, 32, 32), torch.randn(4, 8, 2)
    output = _objective(model, video, actions)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.flow.potential_net[0].weight.grad.abs().sum() > 0
    assert model.action_encoder[0].weight.grad.abs().sum() > 0
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert output["embedding"].shape == (4, 9, 8)
    assert not output["embedding"].requires_grad


def test_lehami_without_actions_rolls_out():
    model = _model().eval()
    with torch.no_grad():
        output = _objective(model, torch.randn(3, 7, 3, 32, 32), stage="validate")
        q, p = model.rollout(torch.randn(3, 5, 3, 32, 32), steps=6)
    assert torch.isfinite(output["loss"])
    assert q.shape == p.shape == (3, 6, 8)


def test_rollout_matches_training_unroll():
    """``rollout`` starts from the same anchor and steps as the training loss."""
    model = _model(action_dim=1).eval()
    video, actions = torch.randn(2, 8, 3, 32, 32), torch.randn(2, 7, 1)
    with torch.no_grad():
        q_all = model.encode(video)
        q, _ = model.rollout(video[:, :4], actions[:, 3:7])
        output = _objective(model, video, actions)
    expected = ((q - q_all[:, 4:8]) ** 2).mean(dim=(0, 2))
    assert output["pos_loss"].item() == pytest.approx(
        expected[:2].mean().item(), rel=1e-4
    )


def test_lehami_rejects_short_clips_and_missing_actions():
    model = _model(action_dim=1)
    with pytest.raises(ValueError, match="at least 7 frames"):
        _objective(model, torch.randn(2, 6, 3, 32, 32), torch.randn(2, 5, 1))
    with pytest.raises(ValueError, match="takes actions"):
        _objective(model, torch.randn(2, 8, 3, 32, 32))


def test_objective_matches_reference():
    """The full objective against an independent numpy computation.

    The encoder and projector are replaced so that ``q`` is the input itself,
    the flow uses ``|p|^2 / 2 + q^2 / 2 - q u`` and the latent is 1-D, where
    every SIGReg slice is ``+-1`` and SIGReg reduces to the Epps-Pulley
    statistic of ``q``. Clips follow ``q_bt = a_b sin(0.4 t + c_b)`` with
    ``a = (1, 0.5, -0.8)``, ``c = (0, 1, 2)`` and actions ``0.2 b - 0.1 t``;
    ``dt = 0.5``, ``momentum_weight = 0.5``, ``lamb = 0.25``. The constants
    come from a numpy reimplementation of the stencils, leapfrog and
    Epps-Pulley test.
    """
    model = _model(
        action_dim=1, latent_dim=1, dt=0.5, momentum_weight=0.5, lamb=0.25
    ).eval()
    model.encoder, model.projector = nn.Flatten(), nn.Identity()
    model.action_encoder = nn.Identity()
    _analytic(model.flow, 1)
    t = torch.arange(8.0)
    amp, phase = torch.tensor([1.0, 0.5, -0.8]), torch.tensor([0.0, 1.0, 2.0])
    q = amp[:, None] * torch.sin(0.4 * t + phase[:, None])
    video = q[:, :, None, None, None]
    actions = (0.2 * torch.arange(3.0)[:, None] - 0.1 * t[:7])[..., None]
    with torch.no_grad():
        output = _objective(model, video, actions)
    assert output["pos_loss"].item() == pytest.approx(0.015085318168826339, rel=1e-5)
    assert output["mom_loss"].item() == pytest.approx(0.0763349263855668, rel=1e-5)
    assert output["sigreg_loss"].item() == pytest.approx(1.217197717144903, rel=1e-5)
    assert output["loss"].item() == pytest.approx(0.3575522106478355, rel=1e-5)


@pytest.mark.parametrize("sigreg_on", ["q", "joint"])
@pytest.mark.parametrize("scale", [0.0, 1e-8, 1e6])
def test_objective_and_gradients_are_finite_on_extreme_inputs(scale, sigreg_on):
    """Constant, near-zero and huge frames and actions keep the loss usable."""
    model = _model(action_dim=2, sigreg_on=sigreg_on).train()
    torch.manual_seed(1)
    video = torch.randn(4, 9, 3, 32, 32) * scale
    actions = torch.randn(4, 8, 2) * max(scale, 1.0)
    output = _objective(model, video, actions)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    for name, w in model.named_parameters():
        if w.grad is not None:
            assert torch.isfinite(w.grad).all(), name


def test_lehami_loss_weights_and_joint_sigreg():
    loss_fn = LeHamiLoss(momentum_weight=2.0, lamb=0.0, n_slices=8)
    zeros = torch.zeros(2, 3, 4)
    loss, pos, mom, _ = loss_fn(
        zeros + 1, zeros + 2, zeros, zeros, torch.randn(5, 3, 4)
    )
    assert pos.item() == pytest.approx(1.0) and mom.item() == pytest.approx(4.0)
    assert loss.item() == pytest.approx(9.0)

    joint = LeHamiLoss(sigreg_on="joint", n_slices=8)
    with pytest.raises(ValueError, match="needs p_reg"):
        joint(zeros, zeros, zeros, zeros, torch.randn(5, 3, 4))
    with pytest.raises(ValueError, match="sigreg_on"):
        LeHamiLoss(sigreg_on="p")


def test_predict_returns_embeddings_only():
    model = _model(action_dim=1).eval()
    with torch.no_grad():
        output = _objective(model, torch.randn(2, 3, 3, 32, 32), stage="predict")
    assert set(output) == {"embedding"}
    assert output["embedding"].shape == (2, 3, 8)


def test_lehami_matches_module_with_forward_function():
    """``LeHami`` is ``spt.Module`` with ``forward.lehami`` and its submodules."""
    import stable_pretraining as spt

    model = _model(action_dim=2).eval()
    module = spt.Module(
        forward=spt.forward.lehami,
        encoder=model.encoder,
        projector=model.projector,
        flow=model.flow,
        action_encoder=model.action_encoder,
        lehami_loss=model.lehami_loss,
    ).eval()
    assert spt.LeHami is LeHami
    video, actions = torch.randn(4, 8, 3, 32, 32), torch.randn(4, 7, 2)
    with torch.no_grad():
        expected = _objective(model, video, actions)
        model.lehami_loss.sigreg.global_step.zero_()
        got = _objective(module, video, actions)
    for key in ("loss", "pos_loss", "mom_loss", "sigreg_loss", "embedding"):
        assert torch.allclose(got[key], expected[key]), key
