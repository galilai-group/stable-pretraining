"""CMAE: Contrastive Masked AutoEncoders.

Combines MAE-style pixel reconstruction with a SimSiam-/BYOL-style
contrastive loss between two views. The student encodes a masked view; an
EMA target encoder encodes a different (un-masked) view; the loss is
``MAE_recon + lambda * contrastive``.

References:
    Huang et al. "Contrastive Masked Autoencoders are Stronger Vision
    Learners." TPAMI 2023. https://arxiv.org/abs/2207.13532
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import TeacherStudentWrapper, patchify


@dataclass
class CMAEOutput(ModelOutput):
    loss: torch.Tensor = None
    loss_recon: torch.Tensor = None
    loss_contrast: torch.Tensor = None
    embedding: torch.Tensor = None


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


def _predictor(in_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, in_dim),
    )


class CMAE(Module):
    """CMAE: MAE pixel loss + EMA contrastive loss.

    :param encoder_name: timm ViT name (default ``"vit_small_patch16_224"``).
    :param patch_size: Patch size (default 16).
    :param mask_ratio: Mask ratio (default 0.75, as in MAE).
    :param projector_dim: Contrastive projector hidden/out dim (default 256).
    :param contrast_weight: Weight on the contrastive term (default 1.0).
    :param ema_decay_start: Initial EMA (default 0.99).
    :param ema_decay_end: Final EMA (default 1.0).
    :param image_size: Input size (default 224).
    :param in_channels: Channels (default 3).
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        projector_dim: int = 256,
        contrast_weight: float = 1.0,
        ema_decay_start: float = 0.99,
        ema_decay_end: float = 1.0,
        image_size: int = 224,
        in_channels: int = 3,
        pretrained: bool = False,
    ):
        super().__init__()
        if isinstance(encoder_name, str):
            import timm

            base = timm.create_model(encoder_name, num_classes=0, pretrained=pretrained)
        else:
            base = encoder_name

        with torch.no_grad():
            seq = base.forward_features(torch.zeros(1, in_channels, image_size, image_size))
        self._has_cls = (
            hasattr(base, "cls_token") and base.cls_token is not None and seq.shape[1] > 1
        )
        embed_dim = seq.shape[-1]
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.contrast_weight = contrast_weight
        self.image_size = image_size
        self.in_channels = in_channels

        self.backbone = TeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.projector = TeacherStudentWrapper(
            _projector(embed_dim, embed_dim, projector_dim),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.predictor = _predictor(projector_dim, projector_dim * 2)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.recon_head = nn.Linear(embed_dim, in_channels * patch_size * patch_size)

    def _encode(self, vit, images, mask=None):
        x = vit.patch_embed(images)
        if mask is not None:
            m = mask.unsqueeze(-1)
            x = x * (1 - m) + self.mask_token.expand_as(x) * m
        if self._has_cls:
            cls = vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + vit.pos_embed
        x = vit.pos_drop(x)
        x = vit.blocks(x)
        return vit.norm(x)

    def _split(self, features):
        if features.ndim == 2:
            return features, None
        if self._has_cls:
            return features[:, 0], features[:, 1:]
        return features.mean(dim=1), features

    def _random_mask(self, B, N, device):
        n_mask = int(round(N * self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        order = noise.argsort(dim=1)
        mask = torch.zeros(B, N, device=device)
        mask.scatter_(1, order[:, :n_mask], 1.0)
        return mask

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> CMAEOutput:
        if view2 is None:
            with torch.no_grad():
                feats = self.backbone.forward_teacher(view1)
                cls, _ = self._split(feats)
            return CMAEOutput(
                loss=torch.zeros((), device=cls.device, dtype=cls.dtype),
                embedding=cls.detach(),
            )

        # Student: masked view1 → reconstruct + contrast.
        B = view1.shape[0]
        with torch.no_grad():
            n_patches = self.backbone.student.patch_embed(view1[:1]).shape[1]
        mask = self._random_mask(B, n_patches, device=view1.device)

        s_feats = self._encode(self.backbone.student, view1, mask=mask)
        s_cls, s_patches = self._split(s_feats)
        zs = self.projector.forward_student(s_cls)
        ps = self.predictor(zs)

        # Reconstruction (only on masked positions, using normalised pixel target)
        target = patchify(view1, (self.in_channels, self.patch_size, self.patch_size))
        m = target.mean(dim=-1, keepdim=True)
        v = target.var(dim=-1, keepdim=True)
        target = (target - m) / (v + 1e-6).sqrt()
        recon = self.recon_head(s_patches)
        loss_per = F.mse_loss(recon, target, reduction="none").mean(dim=-1)
        loss_recon = (loss_per * mask).sum() / mask.sum().clamp(min=1.0)

        # Teacher: unmasked view2 → contrastive target.
        with torch.no_grad():
            t_feats = self._encode(self.backbone.teacher, view2, mask=None)
            t_cls, _ = self._split(t_feats)
            zt = self.projector.forward_teacher(t_cls)

        # Negative cosine similarity (BYOL-style)
        ps_n = F.normalize(ps, dim=-1)
        zt_n = F.normalize(zt, dim=-1)
        loss_contrast = -(ps_n * zt_n).sum(dim=-1).mean()

        loss = loss_recon + self.contrast_weight * loss_contrast

        return CMAEOutput(
            loss=loss,
            loss_recon=loss_recon.detach(),
            loss_contrast=loss_contrast.detach(),
            embedding=t_cls.detach(),
        )
