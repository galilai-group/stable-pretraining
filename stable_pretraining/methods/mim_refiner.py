"""MIM-Refiner: refining MIM-pretrained features with self-distillation.

A thin convenience wrapper around :class:`iBOT` that takes a *pretrained*
MIM encoder (MAE, SimMIM, data2vec, ...) and refines its features with
DINO+iBOT-style self-distillation. The hypothesis is that MIM gives
strong patch-level features but weak global ones; a short DINO+iBOT
phase aligns the global head while preserving the patch features.

References:
    Lehner et al. "MIM-Refiner: A Contrastive Learning Boost from
    Pre-Trained Vision Models." arXiv 2024.
    https://arxiv.org/abs/2402.10093
"""

from typing import Optional, Union

import torch.nn as nn

from .ibot import iBOT, iBOTOutput


__all__ = ["MIMRefiner", "MIMRefinerOutput"]

MIMRefinerOutput = iBOTOutput


class MIMRefiner(iBOT):
    """Refine a pretrained MIM encoder with iBOT-style self-distillation.

    :param pretrained_encoder: A pre-trained ``nn.Module`` (e.g. the encoder
        of a trained ``MAE`` / ``SimMIM`` / ``Data2Vec`` instance, or a timm
        ViT loaded with ``pretrained=True``). Required.
    :param freeze_lower_blocks: Number of leading transformer blocks to
        freeze on the student (default 0). The teacher's EMA already holds
        the MIM features regardless.
    :param **ibot_kwargs: Forwarded to :class:`iBOT` (projector dims,
        prototypes, mask ratio, etc.).
    """

    def __init__(
        self,
        pretrained_encoder: nn.Module,
        freeze_lower_blocks: int = 0,
        **ibot_kwargs,
    ):
        # iBOT accepts a pre-built encoder via the ``encoder_name`` arg.
        super().__init__(encoder_name=pretrained_encoder, **ibot_kwargs)

        if freeze_lower_blocks > 0:
            student_vit = self.backbone.student
            for i, block in enumerate(student_vit.blocks):
                if i < freeze_lower_blocks:
                    for p in block.parameters():
                        p.requires_grad_(False)
