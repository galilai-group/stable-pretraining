"""ReSSL: Relational Self-Supervised Learning.

Student-teacher self-supervised learning with relational distributions over
a memory queue of teacher projections.

Reference:
    Zheng et al., "ReSSL: Relational Self-Supervised Learning with
    Weak Augmentation," NeurIPS 2021.
"""

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone.ressl import ReSSLTeacherStudentWrapper
from stable_pretraining.callbacks.queues import UnsortedQueue
from stable_pretraining.losses import ReSSLLoss
from stable_pretraining.utils import gather_teacher_keys


@dataclass
class ReSSLOutput(ModelOutput):
    """Output of the ReSSL forward pass.

    Attributes:
        loss: Scalar relational loss during training, or None during
            evaluation.
        student_embedding: Student backbone features of shape (N, D).
        teacher_embedding: Teacher backbone features of shape (N, D),
            or None during evaluation.
    """

    loss: torch.Tensor = None
    student_embedding: torch.Tensor = None
    teacher_embedding: torch.Tensor = None


class ReSSL(Module):
    """Relational Self-Supervised Learning with an EMA teacher and memory queue.

    The student learns to match the teacher's similarity distribution over
    stored teacher projections. The teacher processes weak augmentations,
    while the student processes strong augmentations.

    Teacher parameters are updated separately through EMA, for example
    by TeacherStudentCallback.

    Args:
        encoder_name: Timm backbone name or instantiated PyTorch backbone.
        encoder_dim: Backbone feature dimension. Inferred from num_features
            when available.
        temperature_teacher: Teacher similarity-distribution temperature.
        temperature_student: Student similarity-distribution temperature.
        queue_size: Number of teacher projections retained in the queue.
        ema_decay_start: Initial teacher EMA momentum.
        ema_decay_end: Final teacher EMA momentum.
        pretrained: Whether to use pretrained weights for a named backbone.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "resnet18",
        encoder_dim: int | None = None,
        temperature_teacher: float = 0.05,
        temperature_student: float = 0.1,
        queue_size: int = 256,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        pretrained: bool = False,
    ):
        super().__init__()

        if queue_size <= 0:
            raise ValueError("queue_size must be positive")

        if isinstance(encoder_name, str):
            import timm

            base = timm.create_model(
                encoder_name,
                num_classes=0,
                pretrained=pretrained,
            )
        else:
            base = encoder_name

        if encoder_dim is None:
            encoder_dim = getattr(base, "num_features", None)

        if not isinstance(encoder_dim, int) or encoder_dim <= 0:
            raise ValueError(
                "encoder_dim must be a positive integer. Provide it "
                "explicitly if the backbone has no num_features attribute."
            )

        self.temperature_teacher = temperature_teacher
        self.temperature_student = temperature_student

        self.backbone = ReSSLTeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        self.projection_head = ReSSLTeacherStudentWrapper(
            nn.Sequential(
                nn.Linear(encoder_dim, 4096),
                nn.ReLU(),
                nn.Linear(4096, 512),
            ),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        # Initialize the queue with normalized random teacher keys.
        self.queue = UnsortedQueue(max_length=queue_size, shape=512)
        with torch.no_grad():
            initial_keys = F.normalize(
                torch.randn(queue_size, 512),
                dim=-1,
            )
            self.queue.append(initial_keys)

        self.ressl_loss = ReSSLLoss(
            temperature_student=temperature_student,
            temperature_teacher=temperature_teacher,
        )

    def forward(
        self,
        weak_augmentation: torch.Tensor,
        contrastive_augmentation: torch.Tensor,
    ) -> ReSSLOutput:
        """Compute ReSSL relational cross-entropy.

        The teacher encodes weakly augmented images and the student encodes
        strongly augmented images. Their normalized projections are
        compared against the current memory queue.

        During training, teacher projections are gathered across
        distributed processes and appended after loss computation.

        Args:
            weak_augmentation: Teacher images of shape (N, C, H, W).
            contrastive_augmentation: Student images of shape (N, C, H, W).

        Returns:
            ReSSLOutput containing the loss and backbone embeddings.
            During evaluation, only student embeddings are returned.

        Notes:
            Teacher features are computed without gradients. The memory
            queue is updated only during training, and EMA teacher
            parameter updates are handled externally.
        """
        if not self.training:
            with torch.no_grad():
                student_h = self.backbone.forward_student(contrastive_augmentation)
            return ReSSLOutput(
                loss=None,
                student_embedding=student_h,
                teacher_embedding=None,
            )

        student_h = self.backbone.forward_student(contrastive_augmentation)

        with torch.no_grad():
            shuffle = torch.randperm(
                weak_augmentation.shape[0],
                device=weak_augmentation.device,
            )
            unshuffle = torch.argsort(shuffle)

            teacher_h = self.backbone.forward_teacher(weak_augmentation[shuffle])
            teacher_z = self.projection_head.forward_teacher(teacher_h)

            teacher_h = teacher_h[unshuffle]
            teacher_z = F.normalize(teacher_z, dim=-1)[unshuffle]

        student_z = self.projection_head.forward_student(student_h)
        student_z = F.normalize(student_z, dim=-1)

        # Clone to protect autograd from the subsequent in-place queue update.
        queue_snapshot = self.queue.get().detach().clone()

        loss = self.ressl_loss(
            student_z,
            teacher_z,
            queue_snapshot,
        )

        with torch.no_grad():
            keys = gather_teacher_keys(teacher_z.detach())
            self.queue.append(keys)

        return ReSSLOutput(
            loss=loss,
            student_embedding=student_h,
            teacher_embedding=teacher_h,
        )
