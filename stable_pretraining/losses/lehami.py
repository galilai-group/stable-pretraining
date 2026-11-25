"""LeHami rollout objective: phase-space prediction error plus SIGReg."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeHamiLoss(nn.Module):
    """Rollout loss of LeHami (Yalovetzky, Balestriero & LeCun, 2026).

    ``pos_loss + momentum_weight * mom_loss + lamb * sigreg_loss``, where
    ``pos_loss`` and ``mom_loss`` are the mean squared errors of the predicted
    positions and momenta over every rollout step, and ``sigreg_loss`` is
    SIGReg applied across the batch at each frame and averaged over frames.

    Args:
        momentum_weight: Weight of the momentum rollout error.
        lamb: SIGReg weight.
        sigreg_on: ``"q"`` to regularize positions only, or ``"joint"`` for
            the concatenated ``[q, p]``.
        n_slices: SIGReg random projections.

    Note:
        Under DDP, the SIGReg statistics are **per GPU** when training through
        Lightning or :class:`~stable_pretraining.Manager`: each rank tests only
        its local batch, and the gradients are then averaged as usual. This is
        because :class:`~stable_pretraining.methods.lejepa.SlicedEppsPulley`
        decides whether to all-reduce when it is constructed, and the model is
        built before Lightning starts the process group. A loss constructed
        after ``torch.distributed.init_process_group`` instead averages the
        statistics across ranks, i.e. tests the global batch. Either way all
        ranks draw the same random projections.
    """

    def __init__(
        self,
        momentum_weight: float = 1.0,
        lamb: float = 0.25,
        sigreg_on: str = "q",
        n_slices: int = 1024,
    ):
        super().__init__()
        if sigreg_on not in ("q", "joint"):
            raise ValueError(f"sigreg_on must be 'q' or 'joint', got {sigreg_on!r}")
        # Imported here: ``stable_pretraining.methods`` imports ``losses``.
        from stable_pretraining.methods.lejepa import SlicedEppsPulley

        self.momentum_weight = momentum_weight
        self.lamb = lamb
        self.sigreg_on = sigreg_on
        self.sigreg = SlicedEppsPulley(num_slices=n_slices)

    def _sigreg(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.sigreg(x[:, i]) for i in range(x.shape[1])]).mean()

    def forward(
        self,
        q_pred: torch.Tensor,
        p_pred: torch.Tensor,
        q_target: torch.Tensor,
        p_target: torch.Tensor,
        q_reg: torch.Tensor,
        p_reg: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the LeHami objective.

        Args:
            q_pred: Predicted positions ``[B, K, d]`` after each of ``K`` steps.
            p_pred: Predicted momenta ``[B, K, d]``.
            q_target: Encoder positions at the same frames ``[B, K, d]``.
            p_target: Reference momenta at the same frames ``[B, K, d]``.
            q_reg: Positions ``[B, F, d]`` of the frames SIGReg is applied to.
            p_reg: Momenta ``[B, F, d]`` of the same frames; required when
                ``sigreg_on="joint"``.

        Returns:
            ``(loss, pos_loss, mom_loss, sigreg_loss)``, each a scalar.
        """
        pos_loss = F.mse_loss(q_pred, q_target)
        mom_loss = F.mse_loss(p_pred, p_target)
        if self.sigreg_on == "joint":
            if p_reg is None:
                raise ValueError("sigreg_on='joint' needs p_reg")
            sigreg_loss = self._sigreg(torch.cat([q_reg, p_reg], dim=-1))
        else:
            sigreg_loss = self._sigreg(q_reg)
        loss = pos_loss + self.momentum_weight * mom_loss + self.lamb * sigreg_loss
        return loss, pos_loss, mom_loss, sigreg_loss
