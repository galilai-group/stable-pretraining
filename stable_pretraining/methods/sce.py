from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import TeacherStudentWrapper, from_timm
from stable_pretraining.callbacks.queues import UnsortedQueue
from stable_pretraining.losses.sce import SCELoss


@dataclass
class SCEOutput(ModelOutput):
    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    queries: Optional[torch.Tensor] = None
    keys: Optional[torch.Tensor] = None


#maps backbone embessing into contrastive representation space
def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class SCE(Module):
    """Similarity Contrastive Estimation SSL method.

    Reference:
        https://arxiv.org/abs/2111.14585
    """
    def __init__(
    self,
    encoder_name: Union[str, nn.Module] = "resnet18",
    projector_dims: Sequence[int] = (2048, 128),
    queue_length: int = 65536,
    temperature: float = 0.2,
    target_temperature: float = 0.05,
    lambda_weight: float = 0.5,
    ema_decay_start: float = 0.999,
    ema_decay_end: float = 0.999,
    low_resolution: bool = False,
    pretrained: bool = False,
    optim: Optional[dict] = None,
    ):
        super().__init__(optim=optim)

        # Create the backbone encoder
        if isinstance(encoder_name, str):
            base = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            base = encoder_name

        # Determine the dimension of backbone embeddings
        with torch.no_grad():
            embed_dim = base(torch.zeros(1, 3, 224, 224)).shape[-1]

        self.embed_dim = embed_dim

        # Create online and momentum backbone encoders
        self.backbone = TeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        # Create online and momentum projection heads
        proj_hidden, proj_out = projector_dims

        self.projector = TeacherStudentWrapper(
            _projector(embed_dim, proj_hidden, proj_out),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        # Create the memory queue
        self.queue = UnsortedQueue(
            max_length=queue_length,
            shape=(proj_out,),
            dtype=torch.float32,
        )

        # Connect SCE loss function
        self.sce_loss = SCELoss(
            temperature=temperature,
            target_temperature=target_temperature,
            lambda_weight=lambda_weight,
        )

    def forward(
        self,
        view1: Union[torch.Tensor, dict],
        view2: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ) -> SCEOutput:

        if isinstance(view1, dict):
            batch = view1

            if "views" in batch:
                views = batch["views"]

                if len(views) != 2:
                    raise ValueError("SCE expects exactly two augmented views.")

                view1 = views[0]["image"]
                view2 = views[1]["image"]

            else:
                view1 = batch["view1"]
                view2 = batch.get("view2")

        # Evaluation: extract a representation without training
        if view2 is None:
            with torch.no_grad():
                embedding = self.backbone.forward_teacher(view1)

            return SCEOutput(
                loss=torch.zeros(
                    (), device=embedding.device, dtype=embedding.dtype
                ),
                embedding=embedding.detach(),
            )

        #1. online/student branch
        q = self.projector.forward_student(
            self.backbone.forward_student(view1)
        )
        q = F.normalize(q, dim=-1)

        #2. momentum/teacher branch
        with torch.no_grad():
            k = self.projector.forward_teacher(
                self.backbone.forward_teacher(view2)
            )
            k = F.normalize(k, dim=-1)

        #3. retrieve existing representations before inserting new keys
        queue_keys = self.queue.get().detach().clone().to(q.dtype)

        #4. calculate SCE loss
        loss = self.sce_loss(q, k, queue_keys)

        #5. update queue with current keys during training
        # Queue insertion is handled separately after the training update.

        #6. return features and loss
        with torch.no_grad():
            embedding = self.backbone.forward_teacher(view1).detach()

        return SCEOutput(
            loss=loss,
            embedding=embedding,
            queries=q,
            keys=k,
        )

    @torch.no_grad()
    def enqueue_keys(self, keys: torch.Tensor) -> None:
        """Store normalized momentum keys in the feature queue."""
        keys = F.normalize(keys.detach(), dim=-1)
        self.queue.append(keys.to(torch.float32))