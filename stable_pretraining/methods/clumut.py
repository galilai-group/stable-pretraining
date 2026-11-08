"""CluMuT: Clustering-based Mutual Targeting for self-supervised learning.

CluMuT learns cluster assignments from two augmented views and combines
stop-gradient mutual targeting with a cluster-space cross-correlation loss.
It uses no negative pairs, cluster balancing, memory bank, or predictor.

References:
    Mei, Wang, Yu. "Self-supervised learning from images: No negative pairs,
    no cluster-balancing." Pattern Recognition, 2025.
    https://doi.org/10.1016/j.patcog.2024.111081
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.losses import CluMuTLoss


@dataclass
class CluMuTOutput(ModelOutput):
    """Outputs and loss terms from a CluMuT forward pass."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None
    h_a: Optional[torch.Tensor] = None
    h_b: Optional[torch.Tensor] = None
    q_a: Optional[torch.Tensor] = None
    q_b: Optional[torch.Tensor] = None
    mutual_targeting_loss: Optional[torch.Tensor] = None
    identitization_loss: Optional[torch.Tensor] = None
    correlation: Optional[torch.Tensor] = None


def _build_projector(in_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    """Build a Barlow Twins style MLP projection head."""
    if not hidden_dims:
        raise ValueError("projector_dims must contain at least one dimension")
    layers = []
    previous_dim = in_dim
    for index, dim in enumerate(hidden_dims):
        layers.append(nn.Linear(previous_dim, dim, bias=False))
        layers.extend((nn.BatchNorm1d(dim), nn.ReLU(inplace=True)))
        previous_dim = dim
    return nn.Sequential(*layers)


class CluMuT(Module):
    """CluMuT clustering-based mutual targeting.

    Args:
        encoder_name: timm encoder name or a pre-built feature-extractor module.
        projector_dims: Hidden and output dimensions for the projector.
        n_prototypes: Number of learnable cluster prototypes.
        temperature: Softmax temperature used for prototype assignments.
        lambd: Off-diagonal correlation scale in the identitization loss.
        low_resolution: Adapt the encoder stem for low-resolution inputs.
        pretrained: Load pretrained timm encoder weights.
        optim: Optional optimizer configuration accepted by
            :class:`stable_pretraining.Module`.

    Note:
        Training batches can provide two tensors under ``batch["views"]``;
        each view may also be a dictionary containing an ``"image"`` tensor.
        Correlation estimation requires a batch size of at least two.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "resnet18",
        projector_dims: Sequence[int] = (1024, 1024),
        n_prototypes: int = 1024,
        temperature: float = 0.1,
        lambd: float = 1.0,
        low_resolution: bool = True,
        pretrained: bool = False,
        optim: Optional[dict] = None,
    ):
        if n_prototypes < 1:
            raise ValueError("n_prototypes must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if optim is None:
            super().__init__()
        else:
            super().__init__(optim=optim)

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
            embedding_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embedding_dim
        self.projector = _build_projector(embedding_dim, projector_dims)
        self.projection_dim = projector_dims[-1]
        self.prototypes = nn.Parameter(torch.empty(n_prototypes, self.projection_dim))
        nn.init.normal_(self.prototypes, std=0.02)
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(self.prototypes, dim=-1))

        self.temperature = temperature
        self.clumut_loss = CluMuTLoss(lambd=lambd)

    def _assign(self, projection: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_projection = F.normalize(projection, dim=-1)
        normalized_prototypes = F.normalize(self.prototypes, dim=-1)
        h = normalized_projection @ normalized_prototypes.T
        q = F.softmax(h / self.temperature, dim=-1)
        return h, q

    @staticmethod
    def _extract_views(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Read two augmented image tensors from framework batch formats."""
        views = batch.get("views")
        if isinstance(views, dict):
            views = list(views.values())
        if not isinstance(views, (list, tuple)) or len(views) != 2:
            raise ValueError('CluMuT expects exactly two views in batch["views"]')
        images = [view["image"] if isinstance(view, dict) else view for view in views]
        if not all(isinstance(image, torch.Tensor) for image in images):
            raise ValueError("each CluMuT view must be an image tensor")
        return images[0], images[1]

    def forward(
        self,
        view1: Union[torch.Tensor, dict],
        view2: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ) -> Union[CluMuTOutput, dict]:
        """Run two-view CluMuT, or handle the framework's batch-dict contract."""
        batch = view1 if isinstance(view1, dict) else None
        if batch is not None:
            view1, view2 = self._extract_views(batch)

        if view2 is not None and view1.shape[0] < 2:
            raise ValueError("CluMuT requires a batch size of at least 2")

        if view2 is None:
            embedding = self.backbone(view1)
            output = CluMuTOutput(
                loss=embedding.new_zeros(()),
                embedding=embedding,
            )
        else:
            embedding_a = self.backbone(view1)
            embedding_b = self.backbone(view2)
            projection_a = self.projector(embedding_a)
            projection_b = self.projector(embedding_b)
            h_a, q_a = self._assign(projection_a)
            h_b, q_b = self._assign(projection_b)
            correlation = self.clumut_loss.correlation(h_a, h_b)
            mutual_loss = self.clumut_loss.mutual_targeting_loss(q_a, q_b)
            identitization_loss = self.clumut_loss.identitization_loss(
                h_a, h_b, correlation=correlation
            )
            loss = mutual_loss + identitization_loss
            output = CluMuTOutput(
                loss=loss,
                embedding=torch.cat((embedding_a, embedding_b), dim=0),
                projection=torch.cat((projection_a, projection_b), dim=0),
                h_a=h_a,
                h_b=h_b,
                q_a=q_a,
                q_b=q_b,
                mutual_targeting_loss=mutual_loss,
                identitization_loss=identitization_loss,
                correlation=correlation,
            )

        if batch is None:
            return output

        if stage == "fit":
            self.log("train/loss", output.loss, on_step=True, on_epoch=True)
            self.log(
                "train/mutual_targeting_loss",
                output.mutual_targeting_loss,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/identitization_loss",
                output.identitization_loss,
                on_step=True,
                on_epoch=True,
            )
        return {
            **batch,
            "loss": output.loss,
            "embedding": output.embedding,
            "projection": output.projection,
            "h_a": output.h_a,
            "h_b": output.h_b,
            "q_a": output.q_a,
            "q_b": output.q_b,
            "mutual_targeting_loss": output.mutual_targeting_loss,
            "identitization_loss": output.identitization_loss,
            "correlation": output.correlation,
        }
