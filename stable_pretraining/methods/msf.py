"""MSF: Mean Shift for Self-Supervised Learning.

The student predicts the EMA teacher embedding of the target view, but instead
of matching only that one target it is pulled toward the mean of the target's k
nearest neighbours in a memory bank of past teacher embeddings. As in the paper,
the loss is asymmetric: the first view is the query, the second view is the
target, and only the target is added to the bank before the neighbour search.
Under DDP the memory bank is per GPU and is not gathered across GPUs.

References:
    Koohpayegani et al. "Mean Shift for Self-Supervised Learning." ICCV 2021.
    https://arxiv.org/abs/2105.07269
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import TeacherStudentWrapper, from_timm
from stable_pretraining.callbacks.queues import UnsortedQueue
from stable_pretraining.forward import msf
from stable_pretraining.losses import MSFLoss


@dataclass
class MSFOutput(ModelOutput):
    """Output from MSF forward pass.

    :ivar loss: Asymmetric MSF loss (0 in eval mode)
    :ivar embedding: Teacher backbone features (always detached)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class MSF(Module):
    """MSF: BYOL with a k-nearest-neighbour target from a memory bank.

    The loss is asymmetric: ``view1`` is the query and ``view2`` is the target.
    Under DDP the memory bank is per GPU: each rank stores only its own teacher
    embeddings, which are not gathered across GPUs.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: ``(hidden, output)`` for the projector (default ``(4096, 512)``).
    :param predictor_hidden_dim: Predictor hidden dim (default 4096).
    :param queue_length: Memory bank size (default 16384).
    :param topk: Number of nearest neighbours (default 5).
    :param ema_decay_start: Initial EMA decay (default 0.99).
    :param ema_decay_end: Final EMA decay (default 1.0).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (4096, 512),
        predictor_hidden_dim: int = 4096,
        queue_length: int = 16384,
        topk: int = 5,
        ema_decay_start: float = 0.99,
        ema_decay_end: float = 1.0,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()
        if isinstance(encoder_name, str):
            encoder_name = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        with torch.no_grad():
            embed_dim = encoder_name(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embed_dim
        proj_hidden, proj_out = projector_dims

        self.backbone = TeacherStudentWrapper(
            encoder_name,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.projector = TeacherStudentWrapper(
            _mlp(embed_dim, proj_hidden, proj_out),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.predictor = _mlp(proj_out, predictor_hidden_dim, proj_out)
        self.queue = UnsortedQueue(
            max_length=queue_length, shape=(proj_out,), dtype=torch.float32
        )
        self.msf_loss = MSFLoss(topk=topk)

    def forward(
        self,
        view1: Union[torch.Tensor, dict],
        view2: Optional[torch.Tensor] = None,
        stage: str = "fit",
    ) -> Union[MSFOutput, dict]:
        # Manager / Module.training_step pass the whole batch dict
        if isinstance(view1, dict):
            return msf(self, view1, stage)

        if view2 is None:
            batch = {"image": view1}
        else:
            batch = {"views": [{"image": view1}, {"image": view2}]}
        out = msf(self, batch, stage)
        loss = out.get("loss", torch.zeros((), device=view1.device))
        return MSFOutput(loss=loss, embedding=out["embedding"])
