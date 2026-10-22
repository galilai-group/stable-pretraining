"""DCL: Decoupled Contrastive Learning.

Uses two augmented views and a contrastive loss whose denominator contains
only negative pairs.

References:
    Yeh et al. "Decoupled Contrastive Learning." ECCV 2022.
    https://arxiv.org/abs/2110.06848

Example::

    import torch
    import stable_pretraining as spt

    model = spt.DCL(encoder_name="resnet18", low_resolution=True)
    view1 = torch.randn(4, 3, 32, 32)
    view2 = torch.randn(4, 3, 32, 32)
    output = model(view1, view2)
    batch = {
        "views": [
            {"image": view1},
            {"image": view2},
        ]
    }
    output = model(batch, stage="fit")
    output.loss.backward()
    model.eval()
    features = model(view1).embedding
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import BatchNorm1dNoBias, from_timm
from stable_pretraining.forward import dcl
from stable_pretraining.losses import DCLLoss


@dataclass
class DCLOutput(ModelOutput):
    """Output from DCL forward pass.

    Attributes:
        loss: Decoupled contrastive loss, or None during evaluation.
        embedding: Backbone features [B, D] for a single view or [2B, D]
            for two views.
        projection: Projector outputs [2B, P], or None during evaluation.
        label: Labels aligned with the embeddings, if present in the batch.
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None


def _build_projector(in_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    """Build a projection MLP with a final bias-free batch normalization.

    Args:
        in_dim: Backbone output dimension.
        hidden_dims: Hidden and output dimensions of the projection MLP.

    Returns:
        Projection network mapping backbone features to the contrastive space.

    Raises:
        ValueError: If hidden_dims is empty.
    """
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain at least one entry (the output dim)")
    layers = []
    prev = in_dim
    for i, dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev, dim, bias=False))
        if i == len(hidden_dims) - 1:
            layers.append(BatchNorm1dNoBias(dim))
        else:
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
        prev = dim
    return nn.Sequential(*layers)


class DCL(Module):
    """DCL: decoupled contrastive learning with a shared encoder and projector.

    Args:
        encoder_name: timm model name or a pre-instantiated nn.Module whose
            forward returns a [B, D] tensor.
        projector_dims: Hidden and output dimensions of the projection MLP.
            Default is (2048, 2048, 256).
        temperature: Temperature for the DCL loss. Default is 0.1.
        low_resolution: Adapt the encoder for small images (CIFAR-style).
        pretrained: Load pretrained timm weights for a named encoder.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 256),
        temperature: float = 0.1,
        low_resolution: bool = False,
        pretrained: bool = False,
    ) -> None:
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
            dummy = torch.zeros(1, 3, 224, 224)
            self.embed_dim = self.backbone(dummy).shape[-1]

        self.projector = _build_projector(self.embed_dim, list(projector_dims))
        self.dcl_loss = DCLLoss(temperature=temperature)

    def forward(
        self,
        view1: torch.Tensor | dict[str, Any],
        view2: torch.Tensor | str | None = None,
        *,
        stage: str | None = None,
    ) -> DCLOutput:
        """Accept direct tensor calls or the Module batch-dictionary interface.

        Args:
            view1: First image tensor or a batch dictionary.
            view2: Second image tensor, or a positional stage for dictionary calls.
            stage: Runtime stage. Single-tensor calls default to "predict";
                other calls default to "fit" in training mode and "validate"
                in evaluation mode.

        Returns:
            Dictionary-compatible output with embeddings, optional labels, and
            training loss and projections.

        Raises:
            TypeError: If the input is neither a tensor nor a dictionary, or
                the second view of a tensor call is not a tensor.
            ValueError: If dictionary arguments specify conflicting stages, or
                a training batch does not contain exactly two views.
        """
        if isinstance(view1, dict):
            if view2 is not None:
                if not isinstance(view2, str) or stage is not None:
                    raise ValueError("Dictionary calls accept exactly one stage")
                stage = view2
            batch = view1
        elif torch.is_tensor(view1):
            if view2 is None:
                batch = {"image": view1}
                if stage is None:
                    stage = "predict"
            elif torch.is_tensor(view2):
                batch = {"views": [{"image": view1}, {"image": view2}]}
            else:
                raise TypeError("The second view must be a tensor")
        else:
            raise TypeError("Expected an image tensor or batch dictionary")

        if stage is None:
            stage = "fit" if self.training else "validate"
        return DCLOutput(**dcl(self, batch, stage))
