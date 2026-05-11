"""NNCLR: Nearest-Neighbor Contrastive Learning of Visual Representations.

Replaces a SimCLR positive with the nearest neighbour of the anchor's
projection in a queue of past projections. Acts as a soft sampler that
brings semantically similar but different instances together.

References:
    Dwibedi et al. "With a Little Help from My Friends: Nearest-Neighbor
    Contrastive Learning of Visual Representations." ICCV 2021.
    https://arxiv.org/abs/2104.14548
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.callbacks.queues import UnsortedQueue
from stable_pretraining.losses import NTXEntLoss


@dataclass
class NNCLROutput(ModelOutput):
    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None
    nn_index: Optional[torch.Tensor] = None


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim, bias=False),
        nn.BatchNorm1d(out_dim),
    )


def _predictor(in_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, in_dim),
    )


def _nearest_neighbour(query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """For each row in ``query``, return the closest row in ``support`` (cosine)."""
    q = F.normalize(query, dim=-1)
    s = F.normalize(support, dim=-1)
    sim = q @ s.T
    idx = sim.argmax(dim=1)
    return support[idx]


class NNCLR(Module):
    """NNCLR: SimCLR with a nearest-neighbour queue.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: ``(hidden, output)`` for the projector
        (default ``(2048, 256)``).
    :param predictor_hidden_dim: Predictor hidden dim (default 4096).
    :param queue_length: Number of past projections to keep for the NN lookup
        (default 16384).
    :param temperature: NT-Xent temperature (default 0.1).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 256),
        predictor_hidden_dim: int = 4096,
        queue_length: int = 16384,
        temperature: float = 0.1,
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

        proj_hidden, proj_out = projector_dims
        self.projector = _projector(embed_dim, proj_hidden, proj_out)
        self.predictor = _predictor(proj_out, predictor_hidden_dim)
        self.queue = UnsortedQueue(max_length=queue_length, shape=(proj_out,), dtype=torch.float32)
        self.nnclr_loss = NTXEntLoss(temperature=temperature)

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> NNCLROutput:
        if view2 is None:
            embedding = self.backbone(view1)
            return NNCLROutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Maintain queue with current projections (detached)
        with torch.no_grad():
            support = self.queue.append(z1.detach().to(torch.float32))
        # Need at least a couple of items in the queue before NN lookup is meaningful;
        # fall back to z2/z1 directly during warmup.
        if support.shape[0] < 2:
            target1, target2 = z2.detach(), z1.detach()
        else:
            target1 = _nearest_neighbour(z1.detach(), support).to(z1.dtype)
            target2 = _nearest_neighbour(z2.detach(), support).to(z2.dtype)

        loss = (
            self.nnclr_loss(p1, target2) + self.nnclr_loss(p2, target1)
        ) / 2

        return NNCLROutput(
            loss=loss,
            embedding=torch.cat([h1, h2], dim=0).detach(),
            projection=torch.cat([z1, z2], dim=0),
        )
