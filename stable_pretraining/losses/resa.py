"""ReSA loss.

Contrastive learning with Sinkhorn self-assignment targets.

Reference:
    Weng et al. "Clustering Properties of Self-Supervised Learning." ICML 2025.
    https://arxiv.org/abs/2501.18452
    Official code: https://github.com/winci-ai/resa (ImageNet) and
    https://github.com/winci-ai/resa_cifar (CIFAR).
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ..utils import all_gather, get_world_size
from .utils import sinkhorn_knopp


class ReSALoss(torch.nn.Module):
    """Representation Self-Assignment (ReSA) loss.

    This loss computes the cross-entropy between student and teacher
    projections, using a self-assignment matrix as soft targets. The
    self-assignment matrix is computed with the Sinkhorn-Knopp algorithm on
    the cosine similarities between the student and teacher embeddings of one
    view (the weakly augmented view in the paper). The student projection of
    each view is compared with the teacher projections of every other view.
    With an identity assignment the loss reduces to InfoNCE.

    Introduced in :cite:`weng2025clustering`.

    Args:
        temperature: Temperature of the prediction softmax. Default is 0.4.
        sinkhorn_temperature: Temperature of the Sinkhorn-Knopp assignment;
            lower values give sharper targets. Default is 0.05.
        sinkhorn_iterations: Number of Sinkhorn-Knopp iterations. Default is 3.

    Note:
        Under DDP, the teacher embeddings and projections are gathered from
        all ranks, so each rank's targets and predictions span the global
        batch, while the student side stays local and DDP's gradient averaging
        gives the global mean. Every rank must use the same local batch size.

        The loss is computed in float32 with autocast disabled.
    """

    def __init__(
        self,
        temperature: float = 0.4,
        sinkhorn_temperature: float = 0.05,
        sinkhorn_iterations: int = 3,
    ):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_temperature = sinkhorn_temperature
        self.sinkhorn_iterations = sinkhorn_iterations

    @torch.no_grad()
    def assignment(
        self, student_embedding: torch.Tensor, teacher_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Sinkhorn self-assignment targets.

        Args:
            student_embedding: Student encoder features of the assignment view,
                of shape ``[N, D]``.
            teacher_embedding: Teacher encoder features of the same view, of
                shape ``[N, D]``.

        Returns:
            torch.Tensor: Float32 soft targets of shape ``[N, N * world_size]``.
            Each row sums to 1.
        """
        if student_embedding.shape != teacher_embedding.shape:
            raise ValueError(
                "student_embedding and teacher_embedding must have the same shape, got "
                f"{tuple(student_embedding.shape)} and {tuple(teacher_embedding.shape)}."
            )
        # exp(S / sinkhorn_temperature) amplifies rounding error, so stay in float32.
        with torch.amp.autocast(
            device_type=student_embedding.device.type, enabled=False
        ):
            student = F.normalize(student_embedding.float(), dim=1)
            teacher = F.normalize(
                torch.cat(all_gather(teacher_embedding), dim=0).float(), dim=1
            )
            return sinkhorn_knopp(
                student @ teacher.T,
                teacher_temp=self.sinkhorn_temperature,
                num_samples=student.shape[0] * get_world_size(),
                n_iterations=self.sinkhorn_iterations,
            )

    def forward(
        self,
        student_embedding: torch.Tensor,
        teacher_embedding: torch.Tensor,
        student_projections: Sequence[torch.Tensor],
        teacher_projections: Sequence[torch.Tensor],
        assignment: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the ReSA loss.

        Args:
            student_embedding: Student encoder features of the assignment view,
                of shape ``[N, D]``.
            teacher_embedding: Teacher encoder features of the same view, of
                shape ``[N, D]``.
            student_projections: Student projector outputs, one ``[N, d]``
                tensor per view. View ``i`` is compared with every teacher
                view ``j != i``.
            teacher_projections: Teacher projector outputs, one ``[N, d]``
                tensor per view, in the same view order as
                ``student_projections``. Usually only the first two views.
            assignment: Targets precomputed with :meth:`assignment`, for
                example to log diagnostics without running Sinkhorn twice.
                Computed from the embeddings when omitted.

        Returns:
            torch.Tensor: Scalar loss, the mean soft-target cross-entropy over
            every (student view, teacher view) pair of different views.
        """
        pairs = [
            (i, j)
            for i in range(len(student_projections))
            for j in range(len(teacher_projections))
            if i != j
        ]
        if not pairs:
            raise ValueError(
                "ReSALoss needs at least one student view and one different teacher "
                f"view, got {len(student_projections)} student and "
                f"{len(teacher_projections)} teacher projection(s)."
            )
        if assignment is None:
            assignment = self.assignment(student_embedding, teacher_embedding)

        with torch.amp.autocast(
            device_type=student_embedding.device.type, enabled=False
        ):
            with torch.no_grad():
                targets = [
                    F.normalize(torch.cat(all_gather(t), dim=0).float(), dim=1)
                    for t in teacher_projections
                ]
            predictions = [F.normalize(z.float(), dim=1) for z in student_projections]
            losses = [
                F.cross_entropy(
                    predictions[i] @ targets[j].T / self.temperature, assignment
                )
                for i, j in pairs
            ]
        return torch.stack(losses).mean()
