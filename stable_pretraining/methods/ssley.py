"""Composable SSL-EY method using the standard dict-based training contract."""

from typing import Any

from torch import nn

from stable_pretraining.module import Module
from stable_pretraining.forward import ssley
from stable_pretraining.losses import SSLEYLoss


class SSLEY(Module):
    """SSL-EY: Unconstrained Stochastic CCA (ICLR 2024).

    Args:
        backbone: Encoder returning a (batch, features) tensor.
        projector: Projection head mapping encoder features to SSL features.
        optim: Standard Module optimizer/scheduler configuration. Defaults to
            AdamW with learning rate 1e-3; tune for the chosen architecture.
        **kwargs: Other Module options, including hparams and parallelize_fn.

    Note:
        The explicit backbone/projector interface avoids dummy image forwards
        and supports non-image inputs. Use ``spt.forward.ssley`` with
        ``spt.Module`` for equivalent custom wiring.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: nn.Module,
        optim: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            forward=ssley,
            backbone=backbone,
            projector=projector,
            ssley_loss=SSLEYLoss(),
            optim=optim
            if optim is not None
            else {"optimizer": {"type": "AdamW", "lr": 1e-3}},
            **kwargs,
        )
