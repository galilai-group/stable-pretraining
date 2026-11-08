"""VIbCReg: Variance-Invariance-better-Covariance Regularization.

A VICReg variant with two changes to the projected embeddings of the two
augmented views:
- **IterNorm** whitening at the end of the projector
- **Feature decorrelation** on the normalized (correlation) matrix in place
  of VICReg's covariance penalty, so the term is independent of feature scale

The defaults (2048-d GELU projector, IterNorm with 64 groups and 5 iterations,
loss weights 25/25/200) follow the solo-learn recipe behind the reference
CIFAR-10 results.

Example::

    import lightning as pl
    import stable_pretraining as spt
    from stable_pretraining.methods import VIbCReg

    model = VIbCReg("resnet18", low_resolution=True)
    model.optim = {
        "optimizer": {"type": "LARS", "lr": 0.3, "weight_decay": 1e-4},
        "scheduler": {"type": "LinearWarmupCosineAnnealing"},
        "interval": "step",
    }
    manager = spt.Manager(
        trainer=pl.Trainer(max_epochs=1000), module=model, data=datamodule
    )
    manager()

References:
    Lee, Aune. "VIbCReg: Variance-Invariance-better-Covariance Regularization
    for Self-Supervised Learning on Time Series." arXiv 2021.
    https://arxiv.org/abs/2109.00783
"""

from typing import Any, Sequence, Union

import torch
import torch.nn as nn

from stable_pretraining import Module
from stable_pretraining.backbone import IterNorm, from_timm
from stable_pretraining.forward import vibcreg as vibcreg_forward
from stable_pretraining.losses import VIbCRegLoss


def _build_vibcreg_projector(
    in_dim: int,
    hidden_dims: Sequence[int],
    iternorm: bool,
    iternorm_groups: int,
    iternorm_iters: int,
) -> nn.Module:
    """Linear-BN-GELU layers ending in a Linear, optionally followed by IterNorm."""
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain at least one entry")
    layers = []
    prev = in_dim
    for dim in hidden_dims[:-1]:
        layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.GELU()]
        prev = dim
    layers.append(nn.Linear(prev, hidden_dims[-1]))
    if iternorm:
        layers.append(IterNorm(hidden_dims[-1], iternorm_groups, T=iternorm_iters))
    return nn.Sequential(*layers)


class VIbCReg(Module):
    """VIbCReg: variance-invariance-better-covariance self-supervised learning.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: Hidden + output dims for the projector.
        Default ``(2048, 2048, 2048)`` matches the solo-learn CIFAR recipe.
    :param sim_coeff: Invariance term weight (default 25.0).
    :param std_coeff: Variance term weight (default 25.0).
    :param cov_coeff: Feature decorrelation term weight (default 200.0).
    :param iternorm: Append :class:`IterNorm` to the projector (default ``True``).
    :param iternorm_groups: Number of IterNorm groups; must divide the
        projector output dim (default 64).
    :param iternorm_iters: Number of Newton-Schulz iterations (default 5).
    :param low_resolution: Adapt first conv for 32x32 inputs (CIFAR-style).
    :param pretrained: Load pretrained timm weights for the encoder.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 2048),
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 200.0,
        iternorm: bool = True,
        iternorm_groups: int = 64,
        iternorm_iters: int = 5,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            self.backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            self.backbone = encoder_name

        with torch.no_grad():
            embed_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embed_dim

        self.projector = _build_vibcreg_projector(
            embed_dim,
            list(projector_dims),
            iternorm=iternorm,
            iternorm_groups=iternorm_groups,
            iternorm_iters=iternorm_iters,
        )
        self.vibcreg_loss = VIbCRegLoss(
            sim_coeff=sim_coeff, std_coeff=std_coeff, cov_coeff=cov_coeff
        )

    def forward(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
        """VIbCReg forward on a dataloader batch.

        Delegates to :func:`stable_pretraining.forward.vibcreg`, so the method
        class and the forward-function form share one implementation.

        Expected batch keys:

            - **Training**: two views, either as ``{"views": [view_0, view_1]}``
              or as a dict of named views (the output of
              :class:`~stable_pretraining.data.transforms.MultiViewTransform`).
              Each view is a dict with ``"image"`` ``[B, C, H, W]`` and
              optionally ``"label"`` ``[B]``.
            - **Eval**: ``"image"`` ``[B, C, H, W]`` and optionally ``"label"`` ``[B]``.

        :param batch: Batch dictionary from the dataloader.
        :param stage: Lightning stage string (``"fit"``, ``"validate"``,
            ``"test"``, or ``"predict"``).
        :return: Dictionary with:

            - ``"loss"``: VIbCReg loss (training only)
            - ``"embedding"``: Backbone features, ``[2B, D]`` in training
              (both views) and ``[B, D]`` in eval
            - ``"label"``: Labels if present, repeated to match ``"embedding"``
        """
        return vibcreg_forward(self, batch, stage)
