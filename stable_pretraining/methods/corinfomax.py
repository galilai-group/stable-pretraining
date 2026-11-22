"""CorInfoMax: self-supervised learning with a covariance log-determinant.

Reference: Ozsoy et al., https://arxiv.org/abs/2209.07999.

Example::

    import stable_pretraining as spt

    model = spt.CorInfoMax(encoder_name="resnet50", projector_dims=(2048, 128))
"""

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.forward import corinfomax
from stable_pretraining.losses import CorInfoMaxLoss


@dataclass
class CorInfoMaxOutput(ModelOutput):
    """Dict-compatible features and objective returned by CorInfoMax.

    Args:
        loss: Scalar objective, or zero on the embedding-only path.
        embedding: Encoder features in view-major order.
        projection: Raw projector outputs before loss normalization, if training.
        label: Optional labels in the same order as the embeddings.
    """

    loss: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    projection: torch.Tensor | None = None
    label: torch.Tensor | None = None


class CorInfoMax(Module):
    """Configurable encoder/projector with the CorInfoMax objective.

    Args:
        encoder_name: timm model name or module returning ``[N, embed_dim]``.
        projector_dims: Hidden and output widths; hidden layers use ReLU, no BN.
        embed_dim: Encoder output width. Defaults to encoder ``num_features``;
            required for custom encoders without this metadata.
        sim_coeff: Weight of the MSE agreement term.
        cov_coeff: Weight of the covariance log-determinant term.
        mean_momentum: Previous-mean weight, in ``[0, 1)``.
        covariance_momentum: Previous-covariance weight, in ``[0, 1)``.
        initial_covariance: Initial covariance diagonal.
        epsilon: Diagonal regularizer inside each log-determinant.
        normalize: Normalize projections inside the loss.
        statistics_dtype: Precision of persistent mean/covariance buffers.
        low_resolution: Request a low-resolution stem for a named encoder.
        pretrained: Request pretrained weights for a named encoder.
        optim: Module optimizer configuration, optimizer name, or None to disable.

    Note:
        Dataset transforms and class counts are external to the method. Custom
        spatial or token encoders need an adapter returning flat feature vectors.
        See ``examples/corinfomax_cifar10.py`` for the reference CIFAR recipe.
    """

    def __init__(
        self,
        encoder_name: str | nn.Module = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 64),
        *,
        embed_dim: int | None = None,
        sim_coeff: float = 250.0,
        cov_coeff: float = 1.0,
        mean_momentum: float = 0.01,
        covariance_momentum: float = 0.01,
        initial_covariance: float = 1.0,
        epsilon: float = 1e-8,
        normalize: bool = True,
        statistics_dtype: torch.dtype = torch.float64,
        low_resolution: bool = False,
        pretrained: bool = False,
        optim: Any = "AdamW",
    ) -> None:
        backbone = (
            from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
            if isinstance(encoder_name, str)
            else encoder_name
        )
        if not isinstance(backbone, nn.Module):
            raise TypeError("encoder_name must be a timm name or torch.nn.Module")
        if embed_dim is None:
            embed_dim = getattr(backbone, "num_features", None)
        if not isinstance(embed_dim, int) or embed_dim < 1:
            raise ValueError(
                "Provide a positive embed_dim for encoders without num_features"
            )
        dims = tuple(projector_dims)
        if not dims or any(not isinstance(dim, int) or dim < 1 for dim in dims):
            raise ValueError("projector_dims must contain positive integer widths")
        layers = []
        previous = embed_dim
        for index, width in enumerate(dims):
            final = index == len(dims) - 1
            layers.append(nn.Linear(previous, width, bias=not final))
            if not final:
                layers.append(nn.ReLU())
            previous = width
        loss_kwargs = dict(
            sim_coeff=sim_coeff,
            cov_coeff=cov_coeff,
            mean_momentum=mean_momentum,
            covariance_momentum=covariance_momentum,
            initial_covariance=initial_covariance,
            epsilon=epsilon,
            normalize=normalize,
            statistics_dtype=statistics_dtype,
        )
        super().__init__(
            backbone=backbone,
            projector=nn.Sequential(*layers),
            corinfomax_loss=CorInfoMaxLoss(dims[-1], **loss_kwargs),
            optim={"optimizer": optim} if isinstance(optim, str) else optim,
            hparams={
                "method": "CorInfoMax",
                "encoder_name": (
                    encoder_name
                    if isinstance(encoder_name, str)
                    else type(backbone).__name__
                ),
                "embed_dim": embed_dim,
                "projector_dims": dims,
                "low_resolution": low_resolution,
                "pretrained": pretrained,
                **{
                    key: value
                    for key, value in loss_kwargs.items()
                    if key != "statistics_dtype"
                },
                "statistics_dtype": str(statistics_dtype),
            },
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        view1: torch.Tensor | dict[str, Any],
        view2: torch.Tensor | str | None = None,
        *,
        stage: str | None = None,
    ) -> CorInfoMaxOutput:
        """Handle direct tensor calls and the Module/OnlineProbe batch contract.

        Args:
            view1: First view tensor or batch dictionary.
            view2: Second tensor, or positional stage for dictionary calls.
            stage: Explicit runtime stage. Single-tensor calls default to predict.

        Returns:
            Dict-compatible loss, embeddings, and optional projections/labels.
        """
        if isinstance(view1, dict):
            if view2 is not None:
                if not isinstance(view2, str) or stage is not None:
                    raise ValueError("Dictionary calls accept exactly one stage")
                stage = view2
            batch = view1
            stage = stage or ("fit" if self.training else "validate")
        elif torch.is_tensor(view1):
            if view2 is None:
                batch = {"image": view1}
                stage = stage or "predict"
            elif torch.is_tensor(view2):
                batch = {"views": [{"image": view1}, {"image": view2}]}
                stage = stage or ("fit" if self.training else "validate")
            else:
                raise TypeError("The second view must be a tensor")
        else:
            raise TypeError("Expected an image tensor or batch dictionary")
        outputs = corinfomax(self, batch, stage)
        if "loss" not in outputs:
            outputs["loss"] = outputs["embedding"].new_zeros(())
        return CorInfoMaxOutput(**outputs)
