"""MMCR with the standard Module/Manager dictionary training interface.

Example:
    Compose a model without downloading pretrained weights::

        import torch
        import stable_pretraining as spt

        model = spt.MMCR(
            backbone=spt.backbone.from_torchvision("resnet18", low_resolution=True),
            projector=torch.nn.Linear(512, 128),
        )
        output = model(
            {
                "views": [
                    {"image": torch.randn(4, 3, 32, 32)},
                    {"image": torch.randn(4, 3, 32, 32)},
                ]
            },
            "fit",
        )
        output["loss"].backward()

Reference: https://arxiv.org/abs/2303.03307
"""

from typing import Any

import torch
from torch import nn

from ..forward import mmcr
from ..losses import MMCRLoss
from ..module import Module


class MMCR(Module):
    """Maximum Manifold Capacity Representations with aligned multi-view batches.

    Args:
        backbone: Encoder returning a matrix of shape ``[B, D]`` per view.
        projector: Projection head mapping encoder embeddings to ``[B, P]``.
        lmbda: Nonnegative local nuclear-norm weight; default is zero.
        eps: Positive L2-normalization denominator floor.
        gather_distributed: Use the global batch through differentiable gather.
        optim: Module optimizer/scheduler configuration. Defaults to AdamW
            with learning rate 1e-3 and weight decay 1e-4.

    Note:
        This class uses the same dict-based API as ``Module(forward=mmcr)``.
        Use ``eval()`` and a single image dict to extract backbone features.
        No backbone shape probing, pretrained download, or implicit projector
        construction occurs. Restore full checkpoints through Manager with
        the same explicitly constructed backbone and projector.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: nn.Module,
        lmbda: float = 0.0,
        eps: float = 1e-12,
        gather_distributed: bool = True,
        optim: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            forward=type(self).forward,
            backbone=backbone,
            projector=projector,
            mmcr_loss=MMCRLoss(lmbda, eps, gather_distributed),
            optim=optim
            if optim is not None
            else {"optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-4}},
            hparams={
                "method": "mmcr",
                "lmbda": lmbda,
                "eps": eps,
                "gather_distributed": gather_distributed,
            },
        )

    def forward(
        self, batch: dict[str, Any], stage: str = "fit"
    ) -> dict[str, torch.Tensor]:
        """Return embeddings and, during training, MMCR loss.

        Args:
            batch: Aligned multi-view batch, or a single image dict in eval mode.
            stage: Runtime stage used as the loss logging prefix.

        Returns:
            Dict containing embeddings, optional labels, and training loss.
        """
        return mmcr(self, batch, stage)
