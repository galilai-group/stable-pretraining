"""VICRegL: VICReg + local feature matching.

Adds a local term to VICReg by matching the *most similar* spatial
location across two views (or by L2-distance on coordinate maps). The
global term is the same as VICReg (variance / invariance / covariance);
the local term applies the same VICReg objective on patch tokens after a
nearest-neighbour assignment.

References:
    Bardes, Ponce, LeCun. "VICRegL: Self-Supervised Learning of Local
    Visual Features." NeurIPS 2022. https://arxiv.org/abs/2210.01571
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.losses import VICRegLoss


@dataclass
class VICRegLOutput(ModelOutput):
    """Structured output of the :class:`VICRegL` SSL method."""

    loss: torch.Tensor = None
    loss_global: torch.Tensor = None
    loss_local: torch.Tensor = None
    embedding: torch.Tensor = None


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim, bias=False),
    )


class VICRegL(Module):
    """VICRegL: VICReg with an extra local-feature term.

    :param encoder_name: timm ViT name (default ``"vit_small_patch16_224"``).
    :param projector_dim: Output dim of both global and local projectors
        (default 2048).
    :param sim_coeff: Invariance weight (default 25.0).
    :param std_coeff: Variance weight (default 25.0).
    :param cov_coeff: Covariance weight (default 1.0).
    :param alpha: Mixing weight between global and local terms (default 0.75
        means global gets 75%).
    :param image_size: Input size (default 224).
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dim: int = 2048,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        alpha: float = 0.75,
        image_size: int = 224,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            import timm

            self.encoder = timm.create_model(
                encoder_name, num_classes=0, pretrained=pretrained
            )
        else:
            self.encoder = encoder_name

        with torch.no_grad():
            seq = self.encoder.forward_features(
                torch.zeros(1, 3, image_size, image_size)
            )
        self._has_cls = (
            hasattr(self.encoder, "cls_token")
            and self.encoder.cls_token is not None
            and seq.shape[1] > 1
        )
        embed_dim = seq.shape[-1]
        self.embed_dim = embed_dim
        self.alpha = alpha

        self.projector_global = _projector(embed_dim, projector_dim, projector_dim)
        self.projector_local = _projector(embed_dim, projector_dim, projector_dim)
        self.vicreg_loss = VICRegLoss(
            sim_coeff=sim_coeff, std_coeff=std_coeff, cov_coeff=cov_coeff
        )

    def _split(self, features: torch.Tensor):
        """Return (cls, patches) from the encoder's forward_features."""
        if self._has_cls:
            return features[:, 0], features[:, 1:]
        return features.mean(dim=1), features

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> VICRegLOutput:
        if view2 is None:
            features = self.encoder.forward_features(view1)
            cls, _ = self._split(features)
            return VICRegLOutput(
                loss=torch.zeros((), device=view1.device, dtype=view1.dtype),
                embedding=cls,
            )

        # Global features (CLS) and patch tokens
        f1 = self.encoder.forward_features(view1)
        f2 = self.encoder.forward_features(view2)
        cls1, p1 = self._split(f1)
        cls2, p2 = self._split(f2)

        z1 = self.projector_global(cls1)
        z2 = self.projector_global(cls2)
        loss_global = self.vicreg_loss(z1, z2)

        # Local: project patch tokens, then match each patch in view1 to its
        # nearest neighbour in view2 (cosine similarity) and apply VICReg on
        # the matched pairs.
        B, N, D = p1.shape
        zl1 = self.projector_local(p1.reshape(B * N, D)).view(B, N, -1)
        zl2 = self.projector_local(p2.reshape(B * N, D)).view(B, N, -1)
        # Normalise, find nearest neighbour per (image, patch).
        nzl1 = F.normalize(zl1, dim=-1)
        nzl2 = F.normalize(zl2, dim=-1)
        sim = nzl1 @ nzl2.transpose(1, 2)  # [B, N, N]
        nn_idx = sim.argmax(dim=-1)  # [B, N]
        idx = nn_idx.unsqueeze(-1).expand(-1, -1, zl2.shape[-1])
        zl2_aligned = torch.gather(zl2, dim=1, index=idx)
        loss_local = self.vicreg_loss(
            zl1.reshape(B * N, -1), zl2_aligned.reshape(B * N, -1)
        )

        loss = self.alpha * loss_global + (1 - self.alpha) * loss_local
        return VICRegLOutput(
            loss=loss,
            loss_global=loss_global.detach(),
            loss_local=loss_local.detach(),
            embedding=torch.cat([cls1, cls2], dim=0),
        )
