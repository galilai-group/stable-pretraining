"""Two-view SFRIK with user-supplied encoder and projection head."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ..module import Module
from ..forward import sfrik
from ..losses import SFRIKLoss


class SFRIK(Module):
    """Self-supervised learning with rotation-invariant kernels (ICLR 2023).

    This class wires :func:`stable_pretraining.forward.sfrik` into ``Module``.
    Pass two view dictionaries for training and one image dictionary for
    evaluation. Backbone embeddings and labels are exposed to online probes.

    Args:
        backbone: Encoder returning a ``(batch, features)`` tensor.
        projector: Head mapping backbone features to at least two dimensions.
        sim_coeff: Normalized MSE weight; see :class:`SFRIKLoss` for scaling.
        mmd_coeff: Uniformity weight.
        kernel_weights: Weights for the three spherical kernel polynomials.
        gather_distributed: Compute the loss over the global distributed batch.
        eps: L2 normalization denominator floor.
        **kwargs: ``Module`` options, including ``optim`` and ``hparams``.
            The forward callable is provided by this class.

    Example:
        ::

            import torch
            import stable_pretraining as spt

            model = spt.SFRIK(
                backbone=torch.nn.Linear(32, 64),
                projector=torch.nn.Linear(64, 128),
                sim_coeff=64.0,
                optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
            )

    Note:
        Zheng et al., https://arxiv.org/abs/2208.00789. The class does not
        infer encoder dimensions with a dummy forward or download weights.
        Loss settings are saved in checkpoint hyperparameters. When using
        ``load_from_checkpoint``, supply the backbone, projector and optimizer
        configuration again; their architecture is not inferred from weights.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: nn.Module,
        sim_coeff: float = 4000.0,
        mmd_coeff: float = 1.0,
        kernel_weights: Sequence[float] = (1.0, 40.0, 40.0),
        gather_distributed: bool = True,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            forward=SFRIK.forward,
            backbone=backbone,
            projector=projector,
            sfrik_loss=SFRIKLoss(
                sim_coeff=sim_coeff,
                mmd_coeff=mmd_coeff,
                kernel_weights=kernel_weights,
                gather_distributed=gather_distributed,
                eps=eps,
            ),
            **kwargs,
        )
        self.save_hyperparameters(
            {
                "sim_coeff": self.sfrik_loss.sim_coeff,
                "mmd_coeff": self.sfrik_loss.mmd_coeff,
                "kernel_weights": self.sfrik_loss.kernel_weights,
                "gather_distributed": self.sfrik_loss.gather_distributed,
                "eps": self.sfrik_loss.eps,
            }
        )

    def forward(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
        """Encode views using the shared SFRIK forward function.

        Args:
            batch: Two training views or one evaluation image dictionary.
            stage: Stage name used for logging.

        Returns:
            Features, optional labels, and the training projection and loss.

        Note:
            A named ``forward`` method keeps the bound callable pickleable
            when Lightning launches distributed workers with spawn.
        """
        return sfrik(self, batch, stage)
