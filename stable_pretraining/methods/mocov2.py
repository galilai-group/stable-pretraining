"""MoCo v2: Momentum Contrast (CNN-style with queue).

Memory-bank contrastive: a momentum encoder produces "key" features that
are pushed onto a FIFO queue; the student "query" is contrasted (InfoNCE)
against the latest queue. Distinct from MoCo v3 (no queue, ViT-tuned).

References:
    Chen et al. "Improved Baselines with Momentum Contrastive Learning."
    arXiv 2020. https://arxiv.org/abs/2003.04297
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import TeacherStudentWrapper, from_timm
from stable_pretraining.callbacks.queues import UnsortedQueue


@dataclass
class MoCov2Output(ModelOutput):
    """Structured output of the :class:`MoCov2` SSL method."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    queries: Optional[torch.Tensor] = None
    keys: Optional[torch.Tensor] = None


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class MoCov2(Module):
    """MoCo v2 with a fixed-size FIFO queue of momentum-encoder keys.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: ``(hidden, output)`` for the 2-layer head
        (default ``(2048, 128)``; matches MoCo v2's ResNet50 recipe).
    :param queue_length: FIFO key queue size (default 65536).
    :param temperature: InfoNCE temperature (default 0.2).
    :param ema_decay_start: Initial momentum (default 0.999, paper).
    :param ema_decay_end: Final momentum (default 0.999).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 128),
        queue_length: int = 65536,
        temperature: float = 0.2,
        ema_decay_start: float = 0.999,
        ema_decay_end: float = 0.999,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()
        if isinstance(encoder_name, str):
            base = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            base = encoder_name

        with torch.no_grad():
            embed_dim = base(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embed_dim
        self.temperature = temperature

        proj_hidden, proj_out = projector_dims
        self.backbone = TeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.projector = TeacherStudentWrapper(
            _projector(embed_dim, proj_hidden, proj_out),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.queue = UnsortedQueue(
            max_length=queue_length, shape=(proj_out,), dtype=torch.float32
        )

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> MoCov2Output:
        if view2 is None:
            with torch.no_grad():
                emb = self.backbone.forward_teacher(view1)
            return MoCov2Output(
                loss=torch.zeros((), device=emb.device, dtype=emb.dtype),
                embedding=emb.detach(),
            )

        # Query (student) on view1
        q = self.projector.forward_student(self.backbone.forward_student(view1))
        q = F.normalize(q, dim=-1)

        # Key (teacher / momentum) on view2 — no grad
        with torch.no_grad():
            k = self.projector.forward_teacher(self.backbone.forward_teacher(view2))
            k = F.normalize(k, dim=-1)
            queue_keys = self.queue.append(k.detach().to(torch.float32)).to(q.dtype)

        # InfoNCE: positive = matched key, negatives = queue
        logits_pos = (q * k).sum(dim=-1, keepdim=True)  # [B, 1]
        logits_neg = q @ queue_keys.T  # [B, K]
        logits = torch.cat([logits_pos, logits_neg], dim=1) / self.temperature
        targets = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, targets)

        with torch.no_grad():
            embedding = self.backbone.forward_teacher(view1).detach()

        return MoCov2Output(
            loss=loss,
            embedding=embedding,
            queries=q,
            keys=k,
        )
