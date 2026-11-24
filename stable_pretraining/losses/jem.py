"""JEM losses: patch-wise alignment with two regularizers against collapse.

References:
    Seitzer et al. "Scalable Patch-Level Self-Supervised Learning." 2026.
    https://arxiv.org/abs/2610.10013
"""

import torch
import torch.nn.functional as F

from ..utils import all_reduce


class JEMLoss(torch.nn.Module):
    """The three terms of the JEM objective (Seitzer et al., 2026).

    Student and teacher heads give every patch a categorical distribution, a
    softmax at one shared temperature (``log_probabilities``). ``alignment`` pulls the student's
    distribution to the teacher's on the patches both views see,
    ``global_regularizer`` keeps the categories informative over the batch and
    ``structural_regularizer`` keeps the patches of one image apart from each
    other. The forward function weighs and sums them.

    Args:
        temperature: Softmax temperature of student and teacher.
        masked_weight: Weight of the alignment on patches masked for the student.
        visible_weight: Weight of the alignment on patches the student sees.

    Note:
        All terms are computed in float32, whatever the autocast precision.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        masked_weight: float = 0.8,
        visible_weight: float = 0.2,
    ):
        super().__init__()
        self.temperature = temperature
        self.masked_weight = masked_weight
        self.visible_weight = visible_weight

    def log_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Log of the categorical distributions the head outputs stand for.

        Args:
            logits: Head outputs, ``[B, N, K]``.

        Returns:
            Log-softmax of the logits at the loss temperature, in float32.
        """
        return F.log_softmax(logits.float() / self.temperature, dim=-1)

    def alignment(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        masked: torch.Tensor,
        overlap: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence from the teacher's to the student's patch distributions (Eq. 4).

        Args:
            student: Student log-probabilities, ``[B, N, K]``.
            teacher: Teacher log-probabilities at the matching patches, ``[B, N, K]``.
            masked: Patches masked in the student's input, bool ``[B, N]``.
            overlap: Patches that also lie in the teacher view, bool ``[B, N]``.

        Returns:
            Weighted sum of the mean divergence over the masked and over the
            visible overlapping patches. A set without patches counts as zero.

        Note:
            The means are over the local batch: under DDP each rank aligns its
            own samples.
        """
        divergence = F.kl_div(student, teacher, reduction="none", log_target=True)
        divergence = divergence.sum(-1)

        def mean(patches: torch.Tensor) -> torch.Tensor:
            return (divergence * patches).sum() / patches.sum().clamp(min=1)

        return self.masked_weight * mean(overlap & masked) + self.visible_weight * mean(
            overlap & ~masked
        )

    def global_regularizer(self, student: torch.Tensor) -> torch.Tensor:
        """Negative mutual information between inputs and patch categories (Eqs. 5-6).

        For each patch position, the entropy of the batch-mean distribution
        minus the mean entropy of the single distributions: high when the
        batch spreads over the categories and each patch picks one.

        Args:
            student: Student log-probabilities, ``[B, N, K]``.

        Returns:
            Mean over the patch positions of conditional minus marginal entropy.

        Note:
            The batch-mean distribution is global under DDP: the ranks sum their
            probabilities (``all_reduce``), so the term and its gradients equal
            those of one process holding the whole batch. The conditional
            entropy is a mean over the local batch.
        """
        q = student.exp()
        conditional = -(q * student).sum(-1).mean()
        samples = all_reduce(torch.tensor(float(len(q)), device=q.device))
        marginal = all_reduce(q.sum(0)) / samples
        marginal_entropy = -(marginal * marginal.clamp(min=1e-12).log()).sum(-1).mean()
        return conditional - marginal_entropy

    def structural_regularizer(
        self,
        student_projection: torch.Tensor,
        teacher_features: torch.Tensor,
        overlap: torch.Tensor,
    ) -> torch.Tensor:
        """Match patch-to-patch cosine similarities to a teacher layer's (Eqs. 7-8).

        Args:
            student_projection: Projected final student patches, ``[B, N, E]``.
            teacher_features: Teacher patches of one layer at the matching
                positions, ``[B, N, D]``.
            overlap: Patches that also lie in the teacher view, bool ``[B, N]``.

        Returns:
            Squared difference of the two similarity matrices, averaged over the
            pairs of distinct overlapping patches of each sample, then over the
            samples.

        Note:
            Each sample is compared with itself only, so the term is the same
            per GPU and global under DDP.
        """
        student = F.normalize(student_projection.float(), dim=-1)
        teacher = teacher_features.float()
        teacher = F.normalize(F.layer_norm(teacher, teacher.shape[-1:]), dim=-1)
        difference = student @ student.mT - teacher @ teacher.mT
        pairs = overlap[:, :, None] & overlap[:, None, :]
        pairs = pairs & ~torch.eye(
            pairs.shape[-1], dtype=torch.bool, device=pairs.device
        )
        per_sample = (difference.square() * pairs).sum((1, 2)) / pairs.sum(
            (1, 2)
        ).clamp(min=1)
        return per_sample.mean()
