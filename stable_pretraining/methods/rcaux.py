"""RC-aux training through the composable world-model forward function."""

from typing import Any

import torch
from torch import nn

from stable_pretraining.forward import rcaux
from stable_pretraining.losses import RCauxLoss
from stable_pretraining.module import Module


class RCaux(Module):
    """Train an action-conditioned world model with a reachability auxiliary loss.

    Args:
        world_model: Model accepting a trajectory batch dictionary and returning
            ``embedding`` of shape ``(B, T, D)`` and, in training mode,
            ``prediction`` of shape ``(B, K, D)``. It may also return a scalar
            ``regularization_loss``. Architecture and rollout are user supplied.
        embedding_dim: Latent dimension ``D`` shared by the model and loss.
        max_horizon: Largest reachability budget to supervise.
        context_index: Index of the first encoded target for the rollout.
            Defaults to ``T - K``.
        loss_kwargs: Additional :class:`~stable_pretraining.losses.RCauxLoss`
            options, such as ``reachability_weight`` and ``horizon_weights``.
        optim: Module optimizer configuration. Defaults to AdamW at 3e-4.

    Note:
        Cross-trajectory negatives are sampled from the local batch on each
        DDP rank (per GPU), without gathering states from other ranks. A local
        batch of one has no cross-trajectory negatives. DDP averages parameter
        gradients; it does not turn this objective into a global-batch loss.
        Recreate the same architecture and loss options when resuming with
        :class:`~stable_pretraining.Manager` and ``weights_only=False``.
    """

    def __init__(
        self,
        world_model: nn.Module,
        embedding_dim: int,
        max_horizon: int = 5,
        context_index: int | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        optim: dict[str, Any] | None = None,
    ) -> None:
        options = dict(loss_kwargs or {})
        super().__init__(
            world_model=world_model,
            rcaux_loss=RCauxLoss(
                embedding_dim=embedding_dim, max_horizon=max_horizon, **options
            ),
            context_index=context_index,
            optim=optim
            if optim is not None
            else {"optimizer": {"type": "AdamW", "lr": 3e-4}},
            hparams={
                "embedding_dim": embedding_dim,
                "max_horizon": max_horizon,
                "context_index": context_index,
                "loss_kwargs": options,
            },
        )

    def forward(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
        """Delegate trajectory encoding and loss computation to ``rcaux``.

        Args:
            batch: Dictionary consumed by the supplied world model.
            stage: Lightning stage used as the metric prefix.

        Returns:
            World-model outputs and, during training, the RC-aux loss terms.
        """
        return rcaux(self, batch, stage)
