"""TWIST: Twin Class Distribution Estimation.

A siamese network terminated by a softmax classifies two augmented views of
an image into ``C`` latent classes, end-to-end and without labels. The loss
makes the two class distributions agree (consistency), makes each of them
confident (sharpness), and spreads the predictions evenly over the classes
(diversity). The last two terms maximise the mutual information between an
image and its predicted class, which rules out collapsed solutions without a
stop-gradient, a momentum encoder, negative pairs, or a clustering step.

References:
    Wang, Kong, Zhang, Liu, Li. "Self-Supervised Learning by Estimating Twin
    Class Distributions." arXiv 2021. https://arxiv.org/abs/2110.07402

Example::

    import lightning as pl
    from stable_pretraining.methods import TWIST

    model = TWIST(encoder_name="vit_small_patch16_224")
    model.optim = {"optimizer": {"type": "AdamW", "lr": 3e-4}}

    # ``dataloader`` yields ``{"views": [view_1, view_2]}`` batches, e.g.
    # built with ``stable_pretraining.data.transforms.MultiViewTransform``.
    trainer = pl.Trainer(max_epochs=300)
    trainer.fit(model, dataloader)
"""

from typing import Sequence, Union

import torch
import torch.nn as nn

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.forward import twist
from stable_pretraining.losses import TWISTLoss


def _build_twist_projector(
    in_dim: int,
    hidden_dims: Sequence[int],
    norm_before_softmax: bool = True,
) -> nn.Module:
    """TWIST head: (Linear -> BN -> ReLU) stacked, then Linear -> BN(no affine).

    Args:
        in_dim: Backbone output dimension.
        hidden_dims: Hidden dimensions followed by the number of classes
            ``C``, e.g. ``(4096, 4096, 4096)``.
        norm_before_softmax: Append the batch normalisation without affine
            parameters that the paper places right before the softmax ("NBS").

    Returns:
        The head as an ``nn.Sequential``.

    Note:
        Every normalisation layer is a plain ``nn.BatchNorm1d``. Under DDP
        their statistics, including those of the layer before the softmax,
        are therefore computed per process (per GPU), unless the Trainer
        converts them to ``SyncBatchNorm`` with ``sync_batchnorm=True``.
    """
    if len(hidden_dims) < 1:
        raise ValueError(
            "hidden_dims must contain at least one entry (the number of classes)"
        )
    layers = []
    prev = in_dim
    for i, dim in enumerate(hidden_dims):
        is_last = i == len(hidden_dims) - 1
        # A bias before a BatchNorm is redundant, but the last layer keeps one
        # when no normalisation follows (the ``norm_before_softmax=False``
        # ablation), as in the official head.
        bias = is_last and not norm_before_softmax
        layers.append(nn.Linear(prev, dim, bias=bias))
        if not is_last:
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
        elif norm_before_softmax:
            layers.append(nn.BatchNorm1d(dim, affine=False))
        prev = dim
    return nn.Sequential(*layers)


class TWIST(Module):
    """TWIST: self-supervised learning by estimating twin class distributions.

    The class builds the backbone, the classification head and the loss, and
    uses :func:`~stable_pretraining.forward.twist` as its forward. It is called
    with the batch dict and the stage, as :class:`~stable_pretraining.Module`
    does in its training and evaluation steps, and returns a dict.

    Architecture:
        - **Backbone**: any feature extractor producing a flat [B, D] embedding
          (timm ViT/ResNet with the head removed)
        - **Projector**: MLP mapping features to ``C`` class logits, ending with
          a batch normalisation without affine parameters
        - **Loss**: :class:`~stable_pretraining.losses.TWISTLoss` on the softmax
          of the two views' logits

    As in the official implementation, the two views are concatenated and go
    through the backbone and the projector in a single pass, so every batch
    normalisation layer sees both views jointly.

    This class implements the symmetric two-view objective used for CNNs in
    the paper. The multi-crop strategy, the self-labeling stage and the
    momentum encoder used for the ViT results are not implemented.

    Args:
        encoder_name: timm model name (e.g. ``"vit_small_patch16_224"``,
            ``"resnet50"``) or a pre-instantiated ``nn.Module`` whose
            ``forward`` returns a ``[B, D]`` tensor.
        projector_dims: Hidden dimensions of the MLP projector followed by
            the number of classes ``C``. ``(4096, 4096, 4096)`` is the paper's
            representation-learning setting; use a last entry equal to the
            number of semantic classes for unsupervised classification.
        sharpness_weight: Weight ``alpha`` of the sharpness term (1.0 for
            ResNets in the paper, 0.4 for ViTs). The default is the ResNet
            value for any encoder; the paper's ViT setting also relies on the
            momentum encoder, which this class does not implement.
        diversity_weight: Weight ``beta`` of the diversity term.
        norm_before_softmax: Keep the batch normalisation before the softmax.
            Disabling it reproduces the paper's ablation and is expected to
            hurt.
        low_resolution: Adapt first conv for 32x32 inputs (CIFAR-style).
        pretrained: Load pretrained timm weights for the encoder.

    Note:
        - Training batches are ``{"views": [view_1, view_2]}`` with an
          ``"image"`` and optionally a ``"label"`` in each view; evaluation
          batches hold a single ``"image"``. The returned dict has
          ``"embedding"``, ``"label"`` when the batch has one and, in
          training, ``"loss"``. The loss and its three terms are logged as
          ``{stage}/loss`` and ``{stage}/twist_*``.
        - Set ``model.optim`` to choose the optimizer; without it ``Module``
          falls back to AdamW with a cosine schedule.
        - Under DDP every batch normalisation of the head, including the one
          before the softmax, uses the statistics of its own process (per
          GPU), not of the global batch, because they are plain
          ``nn.BatchNorm1d`` layers. Create the Trainer with
          ``sync_batchnorm=True`` to convert them to ``SyncBatchNorm`` and
          use global statistics. The diversity term of the loss uses the
          global batch in both cases. The official implementation uses
          ``SyncBatchNorm`` throughout: over all processes, except for the
          layer before the softmax, which it synchronises over groups of
          processes that hold ``bunch_size`` (256 by default) images.

    Example::

        model = TWIST(
            encoder_name="vit_small_patch16_224",
            projector_dims=(4096, 4096, 4096),
        )

        views = [
            {"image": torch.randn(64, 3, 224, 224)},
            {"image": torch.randn(64, 3, 224, 224)},
        ]
        out = model({"views": views}, stage="fit")
        out["loss"].backward()

        # eval: single view, no loss
        model.eval()
        out = model({"image": views[0]["image"]}, stage="validate")
        features = out["embedding"]  # [64, embed_dim]
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (4096, 4096, 4096),
        sharpness_weight: float = 1.0,
        diversity_weight: float = 1.0,
        norm_before_softmax: bool = True,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        if isinstance(encoder_name, str):
            backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            backbone = encoder_name

        with torch.no_grad():
            embed_dim = backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        projector_dims = list(projector_dims)

        super().__init__(
            forward=twist,
            hparams={
                "encoder_name": encoder_name
                if isinstance(encoder_name, str)
                else type(encoder_name).__name__,
                "projector_dims": projector_dims,
                "sharpness_weight": sharpness_weight,
                "diversity_weight": diversity_weight,
                "norm_before_softmax": norm_before_softmax,
                "low_resolution": low_resolution,
                "pretrained": pretrained,
            },
            backbone=backbone,
            projector=_build_twist_projector(
                embed_dim, projector_dims, norm_before_softmax=norm_before_softmax
            ),
            twist_loss=TWISTLoss(
                sharpness_weight=sharpness_weight, diversity_weight=diversity_weight
            ),
        )
        self.embed_dim = embed_dim
        self.n_classes = projector_dims[-1]
