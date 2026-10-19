"""LeHami: symplectic Hamiltonian latent dynamics for JEPA world models.

The encoder's latent is read as a position ``q``; the momentum ``p`` is its
time derivative, estimated with fixed least-squares (Savitzky-Golay) stencils
over neighbouring frames. The predictor is an action-conditioned separable
Hamiltonian ``H(q, p, u) = K(p / s) + V(q, u)`` integrated with a leapfrog
step, which is exactly symplectic: every step preserves phase-space volume,
whatever the learned weights or actions, so rollouts cannot contract the latent
distribution, settle into attracting fixed points or merge distinct states.
With an even kinetic energy (``K(-p) = K(p)``) the step is also exactly
time-reversible.

Training rolls the flow out on its own predictions from a causal anchor and
scores every step against the encoder's positions and the centered-stencil
momenta, with SIGReg keeping the encoder from collapsing.

References:
    Yalovetzky, Balestriero & LeCun. "LeHami: Symplectic Latent Dynamics for
    World Models." 2026. https://github.com/rominay/LeHami

Example::

    import torch
    from stable_pretraining.methods import LeHami

    model = LeHami(encoder="resnet18", action_dim=1, latent_dim=32)
    video = torch.randn(4, 14, 3, 64, 64)  # [B, T, C, H, W]
    actions = torch.randn(4, 14, 1)  # action t leads from frame t to t + 1

    model.train()
    output = model(video, actions)
    output.loss.backward()

    model.eval()
    q, p = model.rollout(video[:, :4], actions[:, 3:13])  # 10 steps ahead
"""

from dataclasses import dataclass
from typing import Optional, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.methods.lejepa import SlicedEppsPulley

CENTERED_HALFWIDTH = 2
CAUSAL_WINDOW = 4


def _derivative_weights(offsets: list[int], order: int = 2) -> torch.Tensor:
    """Weights ``w`` with ``sum_m w[m] f(t + offsets[m])`` the LSQ-polynomial slope at ``t``."""
    x = torch.tensor(offsets, dtype=torch.float64)
    vander = torch.stack([x**k for k in range(order + 1)], dim=1)
    return torch.linalg.pinv(vander)[1]


_CENTERED_WEIGHTS = _derivative_weights(
    list(range(-CENTERED_HALFWIDTH, CENTERED_HALFWIDTH + 1))
)
_CAUSAL_WEIGHTS = _derivative_weights(list(range(-(CAUSAL_WINDOW - 1), 1)))


def centered_momentum(q: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Time derivative of a latent trajectory from a centered 5-frame stencil.

    Fits a quadratic by least squares to frames ``t - 2 .. t + 2`` and
    differentiates it at ``t``, so the estimate has no half-step lag and is
    exact on quadratic trajectories.

    Args:
        q: Positions ``[..., T, d]`` with ``T >= 5``.
        dt: Time between frames.

    Returns:
        Momenta ``[..., T - 4, d]`` at frames ``2 .. T - 3``.
    """
    n = q.shape[-2] - 2 * CENTERED_HALFWIDTH
    w = _CENTERED_WEIGHTS.to(dtype=q.dtype, device=q.device)
    return sum(w[m] * q[..., m : m + n, :] for m in range(len(w))) / dt


def causal_momentum(q: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Time derivative at the last frame from a causal 4-frame stencil.

    Uses only frames ``t - 3 .. t``, so it can start a rollout without seeing
    the frames the rollout is scored on.

    Args:
        q: Positions ``[..., T, d]``; only the last ``4`` frames are used.
        dt: Time between frames.

    Returns:
        Momentum ``[..., d]`` at the last frame.
    """
    w = _CAUSAL_WEIGHTS.to(dtype=q.dtype, device=q.device)
    return torch.einsum("...wd,w->...d", q[..., -CAUSAL_WINDOW:, :], w) / dt


def _mlp(in_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
    dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [1]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        last = i == len(dims) - 2
        # A constant energy offset has no gradient, so the output bias would
        # never train.
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=not last))
        if not last:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _requiring_grad(t: torch.Tensor) -> torch.Tensor:
    if t.requires_grad:
        return t
    return t.detach().requires_grad_(True)


class SeparableHamiltonianFlow(nn.Module):
    """Leapfrog step of a learned separable Hamiltonian ``K(p / s) + V(q, u)``.

    ``K`` sees only ``p`` and ``V`` only ``(q, u)``, so each leapfrog sub-step
    is a shear with unit Jacobian and the step is exactly symplectic::

        p_half = p - dt / 2 * dV / dq(q, u)
        q_next = q + dt * dK / dp(p_half)
        p_next = p_half - dt / 2 * dV / dq(q_next, u)

    Args:
        latent_dim: Size ``d`` of ``q`` and of ``p``.
        action_dim: Size of the action embedding ``u`` (0 for no actions).
        hidden_dim: Width of the ``K`` and ``V`` MLPs.
        n_layers: Linear layers per MLP.
        dt: Integrator step.
        learnable_scale: Learn a per-dimension momentum scale ``s`` (a
            diagonal mass), which absorbs the relative scale of ``q`` and ``p``
            that self-supervised training leaves free.
        even_kinetic: Use ``(K(p) + K(-p)) / 2``, which makes the step exactly
            time-reversible: flipping ``p`` and stepping retraces the path.

    Note:
        The step differentiates ``H`` with ``torch.autograd.grad``, so it runs
        in at least fp32 with autocast off and must not be wrapped in
        ``torch.compile``.
        Under ``torch.no_grad`` it still builds a local graph for the
        gradients of ``H`` and returns detached outputs.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 0,
        hidden_dim: int = 1024,
        n_layers: int = 3,
        dt: float = 1.0,
        learnable_scale: bool = True,
        even_kinetic: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dt = dt
        self.even_kinetic = even_kinetic
        self.kinetic_net = _mlp(latent_dim, hidden_dim, n_layers)
        self.potential_net = _mlp(latent_dim + action_dim, hidden_dim, n_layers)
        self.log_scale = nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learnable_scale
        )

    def kinetic(self, p: torch.Tensor) -> torch.Tensor:
        """Kinetic energy ``K(p / s)``, shape ``[B]``."""
        p = p / self.log_scale.exp()
        if self.even_kinetic:
            k = self.kinetic_net(torch.cat([p, -p]))
            return 0.5 * (k[: len(p)] + k[len(p) :]).squeeze(-1)
        return self.kinetic_net(p).squeeze(-1)

    def potential(self, q: torch.Tensor, u: Optional[torch.Tensor]) -> torch.Tensor:
        """Potential energy ``V(q, u)``, shape ``[B]``."""
        x = q if u is None else torch.cat([q, u], dim=-1)
        return self.potential_net(x).squeeze(-1)

    def energy(
        self, q: torch.Tensor, p: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Hamiltonian ``H(q, p, u)``, shape ``[B]``."""
        return self.kinetic(p) + self.potential(q, u)

    def forward(
        self, q: torch.Tensor, p: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One leapfrog step.

        Args:
            q: Positions ``[B, d]``.
            p: Momenta ``[B, d]``.
            u: Action embedding ``[B, action_dim]``, or ``None``.

        Returns:
            ``(q_next, p_next)``, each ``[B, d]``.
        """
        train = torch.is_grad_enabled()
        dt = self.dt
        # At least fp32: the gradients of H must not run in reduced precision.
        dtype = torch.promote_types(q.dtype, torch.float32)
        device_type = "cuda" if q.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            with torch.enable_grad():
                q0 = _requiring_grad(q.to(dtype))
                p0 = _requiring_grad(p.to(dtype))
                u = None if u is None else u.to(dtype)

                (dv,) = torch.autograd.grad(
                    self.potential(q0, u).sum(), q0, create_graph=train
                )
                p_half = p0 - 0.5 * dt * dv
                if not train:
                    p_half = p_half.detach().requires_grad_(True)
                (dk,) = torch.autograd.grad(
                    self.kinetic(p_half).sum(), p_half, create_graph=train
                )
                q_next = q0 + dt * dk
                if not train:
                    q_next = q_next.detach().requires_grad_(True)
                (dv,) = torch.autograd.grad(
                    self.potential(q_next, u).sum(), q_next, create_graph=train
                )
                p_next = p_half - 0.5 * dt * dv
        if not train:
            q_next, p_next = q_next.detach(), p_next.detach()
        return q_next.to(q.dtype), p_next.to(p.dtype)


@dataclass
class LeHamiOutput(ModelOutput):
    """Output of a LeHami forward pass.

    :ivar loss: ``pos_loss + momentum_weight * mom_loss + lamb * sigreg_loss``.
    :ivar embedding: Encoder positions ``q`` for every frame ``[B, T, d]``.
    :ivar pos_loss: Rollout error on positions, averaged over steps.
    :ivar mom_loss: Rollout error on momenta, averaged over steps.
    :ivar sigreg_loss: SIGReg on the encoder's latents.
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    pos_loss: torch.Tensor = None
    mom_loss: torch.Tensor = None
    sigreg_loss: torch.Tensor = None


class LeHami(Module):
    """LeHami: JEPA world model with a symplectic Hamiltonian predictor.

    Architecture:
        - **Encoder**: timm model or any module mapping frames ``[N, C, H, W]``
          to features ``[N, D]``
        - **Projector**: BatchNorm MLP ``D -> latent_dim``, giving positions ``q``
        - **Momentum**: Savitzky-Golay stencils on the ``q`` trajectory
        - **Predictor**: :class:`SeparableHamiltonianFlow`, conditioned on an
          MLP embedding of the action through ``V(q, u)``
        - **Loss**: rollout MSE on ``q`` and ``p`` plus ``lamb`` times SIGReg

    A training clip of ``T`` frames anchors the rollout at frame 3 (``q`` from
    the encoder, ``p`` from the causal stencil over frames ``0 .. 3``) and
    unrolls ``T - 6`` steps on the flow's own predictions; step ``k`` is scored
    against frame ``4 + k``, whose momentum target comes from the centered
    stencil and therefore needs two frames on each side. ``T = 14`` gives the
    8-step unroll of the paper. SIGReg is applied per frame across the batch,
    on ``q`` (``sigreg_on="q"``, the paper's main model) or on ``[q, p]``
    (``"joint"``).

    Args:
        encoder: timm model name or an ``nn.Module`` returning ``[N, D]``.
        action_dim: Action size per frame; 0 for passive dynamics.
        latent_dim: Size ``d`` of ``q`` and ``p``.
        encoder_dim: Encoder output size ``D``; inferred from timm models or
            from a dummy forward at ``image_size``.
        image_size: Frame size used to infer ``encoder_dim``.
        projector_hidden_dim: Hidden width of the projector.
        flow_hidden_dim: Hidden width of the ``K`` and ``V`` MLPs.
        flow_layers: Linear layers per ``K`` / ``V`` MLP.
        dt: Integrator step, in frames.
        even_kinetic: Make the flow exactly time-reversible.
        learnable_scale: Learn a per-dimension momentum scale.
        momentum_weight: Weight of the momentum rollout loss.
        lamb: SIGReg weight.
        sigreg_on: ``"q"`` or ``"joint"`` (``[q, p]``).
        n_slices: SIGReg random projections.
        pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder: Union[str, nn.Module] = "vit_tiny_patch16_224",
        action_dim: int = 0,
        latent_dim: int = 32,
        encoder_dim: Optional[int] = None,
        image_size: int = 224,
        projector_hidden_dim: int = 2048,
        flow_hidden_dim: int = 1024,
        flow_layers: int = 3,
        dt: float = 1.0,
        even_kinetic: bool = False,
        learnable_scale: bool = True,
        momentum_weight: float = 1.0,
        lamb: float = 0.25,
        sigreg_on: str = "q",
        n_slices: int = 1024,
        pretrained: bool = False,
    ):
        super().__init__()
        if sigreg_on not in ("q", "joint"):
            raise ValueError(f"sigreg_on must be 'q' or 'joint', got {sigreg_on!r}")
        if isinstance(encoder, str):
            encoder = timm.create_model(encoder, pretrained=pretrained, num_classes=0)
        self.encoder = encoder
        if encoder_dim is None:
            encoder_dim = getattr(encoder, "num_features", None) or self._infer_dim(
                image_size
            )
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.GELU(),
            nn.Linear(projector_hidden_dim, latent_dim),
        )
        action_embed_dim = 2 * latent_dim if action_dim else 0
        self.action_encoder = (
            nn.Sequential(
                nn.Linear(action_dim, 4 * action_embed_dim),
                nn.SiLU(),
                nn.Linear(4 * action_embed_dim, action_embed_dim),
            )
            if action_dim
            else None
        )
        self.flow = SeparableHamiltonianFlow(
            latent_dim,
            action_embed_dim,
            flow_hidden_dim,
            flow_layers,
            dt,
            learnable_scale=learnable_scale,
            even_kinetic=even_kinetic,
        )
        self.sigreg = SlicedEppsPulley(num_slices=n_slices)
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.dt = dt
        self.momentum_weight = momentum_weight
        self.lamb = lamb
        self.sigreg_on = sigreg_on

    @torch.no_grad()
    def _infer_dim(self, image_size: int) -> int:
        was_training = self.encoder.training
        self.encoder.eval()
        out = self.encoder(torch.zeros(1, 3, image_size, image_size))
        self.encoder.train(was_training)
        return out.shape[-1]

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Positions ``q`` ``[B, T, d]`` of frames ``[B, T, C, H, W]``."""
        b, t = video.shape[:2]
        features = self.encoder(video.flatten(0, 1))
        return self.projector(features).view(b, t, self.latent_dim)

    def _embed_actions(self, actions: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.action_encoder is None:
            return None
        if actions is None:
            raise ValueError(f"this model takes actions of size {self.action_dim}")
        return self.action_encoder(actions.float())

    def _sigreg(self, x: torch.Tensor) -> torch.Tensor:
        """SIGReg across the batch at each frame of ``x`` ``[B, T, D]``."""
        return torch.stack([self.sigreg(x[:, i]) for i in range(x.shape[1])]).mean()

    def forward(
        self, video: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) -> LeHamiOutput:
        """Rollout loss on a batch of clips.

        Args:
            video: Frames ``[B, T, C, H, W]`` with ``T >= 7``.
            actions: ``[B, T - 1, action_dim]`` or longer, where action ``t``
                leads from frame ``t`` to frame ``t + 1``; ``None`` when
                ``action_dim`` is 0.

        Returns:
            :class:`LeHamiOutput`. The losses are computed in eval mode too,
            for validation.
        """
        t = video.shape[1]
        anchor = CAUSAL_WINDOW - 1
        n_unroll = t - 1 - CENTERED_HALFWIDTH - anchor
        if n_unroll < 1:
            raise ValueError(f"clips need at least 7 frames, got {t}")
        q = self.encode(video)
        p_ref = centered_momentum(q, self.dt)  # frames 2 .. T - 3
        u = self._embed_actions(actions)

        q_pred = q[:, anchor]
        p_pred = causal_momentum(q[:, : anchor + 1], self.dt)
        pos_loss = mom_loss = 0.0
        for k in range(n_unroll):
            step_u = None if u is None else u[:, anchor + k]
            q_pred, p_pred = self.flow(q_pred, p_pred, step_u)
            frame = anchor + 1 + k
            pos_loss = pos_loss + F.mse_loss(q_pred, q[:, frame])
            mom_loss = mom_loss + F.mse_loss(
                p_pred, p_ref[:, frame - CENTERED_HALFWIDTH]
            )
        pos_loss = pos_loss / n_unroll
        mom_loss = mom_loss / n_unroll

        q_win = q[:, CENTERED_HALFWIDTH : t - CENTERED_HALFWIDTH]
        if self.sigreg_on == "joint":
            sigreg_loss = self._sigreg(torch.cat([q_win, p_ref], dim=-1))
        else:
            sigreg_loss = self._sigreg(q_win)
        loss = pos_loss + self.momentum_weight * mom_loss + self.lamb * sigreg_loss
        return LeHamiOutput(
            loss=loss,
            embedding=q.detach(),
            pos_loss=pos_loss,
            mom_loss=mom_loss,
            sigreg_loss=sigreg_loss,
        )

    def rollout(
        self,
        context: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict future phase-space states from the last four frames.

        Args:
            context: Frames ``[B, T, C, H, W]`` with ``T >= 4``; the rollout
                starts at the last frame.
            actions: ``[B, K, action_dim]``, applied from the last context
                frame on; ``None`` when ``action_dim`` is 0.
            steps: Number of steps ``K`` when there are no actions.

        Returns:
            ``(q, p)``, each ``[B, K, d]``: the state after each step.
        """
        if context.shape[1] < CAUSAL_WINDOW:
            raise ValueError(f"rollout needs {CAUSAL_WINDOW} context frames")
        q = self.encode(context[:, -CAUSAL_WINDOW:])
        q_t, p_t = q[:, -1], causal_momentum(q, self.dt)
        u = self._embed_actions(actions)
        steps = u.shape[1] if u is not None else steps
        if steps is None:
            raise ValueError("pass actions or steps")
        qs, ps = [], []
        for k in range(steps):
            q_t, p_t = self.flow(q_t, p_t, None if u is None else u[:, k])
            qs.append(q_t)
            ps.append(p_t)
        return torch.stack(qs, 1), torch.stack(ps, 1)
