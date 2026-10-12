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

    from stable_pretraining.methods import TWIST
    import lightning as pl

    model = TWIST(encoder_name="vit_small_patch16_224")

    trainer = pl.Trainer(max_epochs=300)
    trainer.fit(model, dataloader)
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.losses import TWISTLoss


@dataclass
class TWISTOutput(ModelOutput):
    """Output from TWIST forward pass.

    :ivar loss: TWIST loss (0 in eval mode)
    :ivar embedding: Backbone features [B, D] in eval mode, [2B, D] in train mode
    :ivar logits: Class logits [2B, C], view 1 then view 2 (None in eval mode)
    :ivar consistency: Unweighted, detached consistency term (None in eval mode)
    :ivar sharpness: Unweighted, detached sharpness term (None in eval mode)
    :ivar diversity: Unweighted, detached diversity term (None in eval mode)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    logits: Optional[torch.Tensor] = None
    consistency: Optional[torch.Tensor] = None
    sharpness: Optional[torch.Tensor] = None
    diversity: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None


def _build_twist_projector(
    in_dim: int,
    hidden_dims: Sequence[int],
    norm_before_softmax: bool = True,
) -> nn.Module:
    """TWIST head: (Linear -> BN -> ReLU) stacked, then Linear -> BN(no affine).

    :param in_dim: Backbone output dimension
    :param hidden_dims: Sequence of hidden dimensions followed by the number of
        classes ``C``, e.g. (4096, 4096, 4096)
    :param norm_before_softmax: Append the batch normalisation without affine
        parameters that the paper places right before the softmax ("NBS")
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

    :param encoder_name: timm model name (e.g. ``"vit_small_patch16_224"``,
        ``"resnet50"``) or a pre-instantiated ``nn.Module`` whose ``forward``
        returns a ``[B, D]`` tensor.
    :param projector_dims: Hidden dimensions of the MLP projector followed by
        the number of classes ``C``. ``(4096, 4096, 4096)`` is the paper's
        representation-learning setting; use a last entry equal to the number
        of semantic classes for unsupervised classification.
    :param sharpness_weight: Weight ``alpha`` of the sharpness term (1.0 for
        ResNets in the paper, 0.4 for ViTs). The default is the ResNet value
        for any encoder; the paper's ViT setting also relies on the momentum
        encoder, which this class does not implement.
    :param diversity_weight: Weight ``beta`` of the diversity term (default 1.0).
    :param norm_before_softmax: Keep the batch normalisation before the softmax
        (default True). Disabling it reproduces the paper's ablation and is
        expected to hurt.
    :param low_resolution: Adapt first conv for 32x32 inputs (CIFAR-style).
    :param pretrained: Load pretrained timm weights for the encoder.

    Example::

        model = TWIST(
            encoder_name="vit_small_patch16_224",
            projector_dims=(4096, 4096, 4096),
        )

        v1 = torch.randn(64, 3, 224, 224)
        v2 = torch.randn(64, 3, 224, 224)
        out = model(v1, v2)
        out.loss.backward()

        # eval: single view, no loss
        model.eval()
        out = model(v1)
        features = out.embedding  # [64, embed_dim]
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

        self.projector = _build_twist_projector(
            embed_dim, list(projector_dims), norm_before_softmax=norm_before_softmax
        )
        self.n_classes = list(projector_dims)[-1]
        self.twist_loss = TWISTLoss(
            sharpness_weight=sharpness_weight, diversity_weight=diversity_weight
        )

    def forward(
        self,
        view1: Union[torch.Tensor, dict],
        view2: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ) -> TWISTOutput:
        """Forward pass.

        Accepts either two image tensors or the batch dict (plus ``stage``)
        that :class:`~stable_pretraining.Module` passes from its training and
        evaluation steps, so the class can be handed directly to a Trainer.

        :param view1: First augmented view ``[B, C, H, W]``, or a batch dict
            with ``"views"`` (two view dicts with ``"image"`` and optionally
            ``"label"``) or a single ``"image"``. A tensor without ``view2``
            is treated as a single eval batch.
        :param view2: Second augmented view ``[B, C, H, W]`` (training only).
        :param stage: Lightning stage, accepted for compatibility with
            ``Module``. The loss is computed whenever two views are given.
        :return: :class:`TWISTOutput` with loss, embedding, class logits, the
            three loss terms and, for batch dicts, the labels for the probes.
        """
        if isinstance(view1, dict):
            return self._forward_batch(view1)

        if view2 is None:
            embedding = self.backbone(view1)
            return TWISTOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
            )

        embedding = self.backbone(torch.cat([view1, view2], dim=0))
        logits = self.projector(embedding)
        logits_1, logits_2 = logits.chunk(2, dim=0)
        terms = self.twist_loss.terms(logits_1, logits_2)
        return TWISTOutput(
            loss=terms["loss"],
            embedding=embedding,
            logits=logits,
            consistency=terms["consistency"].detach(),
            sharpness=terms["sharpness"].detach(),
            diversity=terms["diversity"].detach(),
        )

    def _forward_batch(self, batch: dict) -> TWISTOutput:
        """Unpack a dataloader batch, run the forward and attach the labels."""
        if "views" in batch:
            views = batch["views"]
        elif "image" in batch:
            output = self.forward(batch["image"])
            if "label" in batch:
                output["label"] = batch["label"]
            return output
        else:
            # Dict of named views (``batch_idx`` and other scalars are skipped).
            views = [v for v in batch.values() if isinstance(v, dict) and "image" in v]
        if len(views) != 2:
            raise ValueError(f"TWIST requires exactly 2 views, got {len(views)}.")
        output = self.forward(views[0]["image"], views[1]["image"])
        if "label" in views[0]:
            output["label"] = torch.cat([view["label"] for view in views], dim=0)
        return output
