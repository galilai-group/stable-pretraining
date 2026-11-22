"""BayesNCL: Bayesian Gated Non-Negative Contrastive Learning.

Reference: https://arxiv.org/abs/2605.28441
Code: https://github.com/Peng-Cui-2003/BayesNCL

Example::

    import torch
    import stable_pretraining as spt

    model = spt.BayesNCL(encoder_name="resnet18", low_resolution=True)
    view1, view2 = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    output = model(view1, view2)
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
from stable_pretraining.backbone import from_timm
from stable_pretraining.forward import bayesncl
from stable_pretraining.losses import BayesNCLLoss


@dataclass
class BayesNCLOutput(ModelOutput):
    """Output from a BayesNCL forward pass.

    Attributes:
        loss: Scalar objective, or zero for embedding-only evaluation.
        embedding: Encoder features, shaped [2B, D] for paired views or [B, D].
        projection: Normalized gated features [2B, P], or None for evaluation.
        gate_probabilities: Sigmoid gate probabilities [2B, P], or None for evaluation.
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None
    gate_probabilities: Optional[torch.Tensor] = None


class BayesNCL(Module):
    """Contrastive learning with nonnegative projections and learned binary gates.

    Args:
        encoder_name: A timm model name or an encoder returning [B, D] features.
        projector_dims: Hidden and output dimensions for both two-layer MLPs.
        temperature: Temperature for the contrastive loss.
        kl_weight: Weight of the Bernoulli KL penalty.
        prior_probability: Prior probability of an open gate.
        gate_temperature: Temperature of the sigmoid gate probabilities.
        gate_threshold: Probability threshold above which a gate is open.
        low_resolution: Adapt a timm ResNet's first convolution for CIFAR images.
        pretrained: Load pretrained timm encoder weights.
        embed_dim: Encoder output dimension. If omitted, use num_features or
            infer it with a dummy image, following the other two-view methods.

    Note:
        Implements the reference's default STE gate and rep_relu recipe:
        ReLU projection values with GELU gradients, followed by normalization,
        gating from detached encoder features, and normalization again.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "resnet18",
        projector_dims: Sequence[int] = (2048, 256),
        temperature: float = 0.2,
        kl_weight: float = 3e-5,
        prior_probability: float = 0.8,
        gate_temperature: float = 1.0,
        gate_threshold: float = 0.5,
        low_resolution: bool = False,
        pretrained: bool = False,
        embed_dim: Optional[int] = None,
    ):
        super().__init__()
        if len(projector_dims) != 2:
            raise ValueError(
                "projector_dims must contain a hidden and an output dimension"
            )

        if isinstance(encoder_name, str):
            self.backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            self.backbone = encoder_name

        if embed_dim is None:
            embed_dim = getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            with torch.no_grad():
                embed_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embed_dim

        hidden_dim, output_dim = projector_dims
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gating_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gate_temperature = gate_temperature
        self.gate_threshold = gate_threshold
        self.bayesncl_loss = BayesNCLLoss(
            temperature=temperature,
            kl_weight=kl_weight,
            prior_probability=prior_probability,
        )

    def forward(
        self,
        view1: Union[torch.Tensor, dict[str, Any]],
        view2: Optional[torch.Tensor] = None,
        stage: str = "fit",
    ) -> Union[BayesNCLOutput, dict[str, torch.Tensor]]:
        """Encode one view or compute the two-view objective.

        Args:
            view1: First image tensor [B, C, H, W], or the batch dictionary
                supplied by Module's training and evaluation steps.
            view2: Second image tensor. Omit it for embedding-only inference.
            stage: Stage passed to the composable forward function for logging.

        Returns:
            A BayesNCLOutput for tensor inputs, or a dictionary for a batch input.
            Single-view inputs return encoder features without projection or gating.
        """
        if isinstance(view1, dict):
            return bayesncl(self, view1, stage)

        batch = (
            {"image": view1}
            if view2 is None
            else {"views": [{"image": view1}, {"image": view2}]}
        )
        output = bayesncl(self, batch, stage)
        return BayesNCLOutput(
            loss=output.get("loss", output["embedding"].new_zeros(())),
            embedding=output["embedding"],
            projection=output.get("projection"),
            gate_probabilities=output.get("gate_probabilities"),
        )
