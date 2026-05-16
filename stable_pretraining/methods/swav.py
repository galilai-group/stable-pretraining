"""SwAV: Swapping Assignments between Views.

Online clustering with prototypes — projects features onto a set of
trainable prototypes and uses Sinkhorn-Knopp to compute soft cluster
assignments. The student of one view predicts the assignment of the
other, encouraging consistent clustering across augmentations.

References:
    Caron et al. "Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments." NeurIPS 2020. https://arxiv.org/abs/2006.09882
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import L2Norm, from_timm
from stable_pretraining.losses import SwAVLoss


@dataclass
class SwAVOutput(ModelOutput):
    """Structured output of the :class:`SwAV` SSL method."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
        L2Norm(),
    )


class SwAV(Module):
    """SwAV: prototype-based online clustering for SSL.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: ``(hidden, output)`` for the projector
        (default ``(2048, 128)``).
    :param n_prototypes: Number of prototypes (default 3000).
    :param temperature: Temperature for the swapped-prediction softmax
        (default 0.1).
    :param sinkhorn_iterations: Sinkhorn iterations (default 3).
    :param epsilon: Sinkhorn entropy coefficient (default 0.05).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 128),
        n_prototypes: int = 3000,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        epsilon: float = 0.05,
        low_resolution: bool = False,
        pretrained: bool = False,
        dynamic_img_size: bool = True,
    ):
        super().__init__()
        if isinstance(encoder_name, str):
            self.backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
                dynamic_img_size=dynamic_img_size,
            )
        else:
            self.backbone = encoder_name

        with torch.no_grad():
            embed_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embed_dim

        proj_hidden, proj_out = projector_dims
        self.projector = _projector(embed_dim, proj_hidden, proj_out)
        self.prototypes = nn.Linear(proj_out, n_prototypes, bias=False)
        self.swav_loss = SwAVLoss(
            temperature=temperature,
            sinkhorn_iterations=sinkhorn_iterations,
            epsilon=epsilon,
        )

    def forward(
        self,
        view1: Optional[torch.Tensor] = None,
        view2: Optional[torch.Tensor] = None,
        global_views: Optional[Sequence[torch.Tensor]] = None,
        local_views: Optional[Sequence[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> SwAVOutput:
        """SwAV forward.

        Three calling conventions:

        * ``forward(view1, view2)`` — original 2-view (no multi-crop).
        * ``forward(global_views=[...], local_views=[...])`` — full multi-crop.
        * ``forward(images=...)`` — eval / single-image embedding extraction.
        """
        # Eval / single-image
        if images is not None or (
            view1 is not None and view2 is None and global_views is None
        ):
            single = images if images is not None else view1
            embedding = self.backbone(single)
            return SwAVOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
            )

        # Collect views
        if global_views is not None:
            views = list(global_views) + list(local_views or [])
            n_global = len(global_views)
        elif view1 is not None and view2 is not None:
            views = [view1, view2]
            n_global = 2
        else:
            raise ValueError(
                "SwAV.forward needs either (view1, view2), (global_views, ...), or images"
            )

        # Encode + project every view.
        hs = [self.backbone(v) for v in views]
        zs = [self.projector(h) for h in hs]

        # Renormalise prototypes ONCE (in-place is fine when done before any
        # forward through them — only one version bump per training step).
        with torch.no_grad():
            w = F.normalize(self.prototypes.weight.data, dim=1, p=2)
            self.prototypes.weight.data.copy_(w)

        # Compute scores for every view through the (now unit-norm) prototypes.
        scores = [self.prototypes(F.normalize(z, dim=1, p=2)) for z in zs]

        # Sinkhorn assignments (no_grad) for every view independently.
        with torch.no_grad():
            qs = [self.swav_loss.sinkhorn(s) for s in scores]

        # Swapped-prediction loss across all unordered (global, other) pairs.
        loss = 0.0
        n_pairs = 0
        for i in range(n_global):
            for j in range(i + 1, len(views)):
                loss = loss + self.swav_loss.swapped_prediction(scores[i], qs[j])
                loss = loss + self.swav_loss.swapped_prediction(scores[j], qs[i])
                n_pairs += 2
        loss = loss / max(n_pairs, 1)

        return SwAVOutput(
            loss=loss,
            embedding=torch.cat([hs[i] for i in range(n_global)], dim=0).detach(),
            projection=torch.cat(zs, dim=0),
        )
