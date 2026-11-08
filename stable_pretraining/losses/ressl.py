"""Relational self-supervised learning losses.

This module contains the ReSSL loss, which learns representations by
matching student and teacher similarity distributions over a memory queue.
"""

import torch
import torch.nn.functional as F


class ReSSLLoss(torch.nn.Module):
    """Relational Self-Supervised Learning (ReSSL) loss.

    Computes cross-entropy between the student and teacher similarity
    distributions over a shared memory queue of teacher projections.

    The teacher distribution serves as a fixed target, while the student
    distribution is optimized to match it. Separate temperatures control
    the sharpness of the two distributions.

    Args:
        temperature_student: Temperature for the student distribution.
        temperature_teacher: Temperature for the teacher distribution.

    Raises:
        ValueError: If either temperature is not strictly positive.
    """

    def __init__(
        self,
        temperature_student: float = 0.1,
        temperature_teacher: float = 0.05,
    ):
        super().__init__()

        if temperature_student <= 0 or temperature_teacher <= 0:
            raise ValueError("ReSSL temperatures must be strictly positive")

        self.temperature_student = temperature_student
        self.temperature_teacher = temperature_teacher

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        queue: torch.Tensor,
    ) -> torch.Tensor:
        """Compute relational cross-entropy between student and teacher.

        Args:
            student_embeddings: Student projections of shape (B, D).
            teacher_embeddings: Teacher projections of shape (B, D).
            queue: Memory queue of teacher projections of shape (K, D).

        Returns:
            Scalar cross-entropy loss averaged over the batch.

        Shape:
            - student_embeddings: (B, D)
            - teacher_embeddings: (B, D)
            - queue: (K, D)
            - output: scalar

        Notes:
            B is the batch size, D is the projection dimension, and K
            is the queue size. Embeddings and queue entries are expected
            to be L2-normalized before calling this function.

            Gradients flow through the student distribution only.
        """
        queue = queue.detach()
        student_logits = student_embeddings @ queue.T
        student_log_probs = F.log_softmax(
            student_logits.float() / self.temperature_student,
            dim=-1,
        )

        teacher_logits = teacher_embeddings.detach() @ queue.T
        teacher_probs = F.softmax(
            teacher_logits.float() / self.temperature_teacher,
            dim=-1,
        )

        return -(teacher_probs * student_log_probs).sum(dim=-1).mean()
