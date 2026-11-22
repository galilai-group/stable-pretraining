"""SimDINOv2 losses: cosine compression + coding-rate expansion.

SimDINOv2 simplifies DINOv2 by deleting both prototype heads and every softmax,
centering and Sinkhorn-Knopp operation, replacing them with:

- a **compression** term: mean cosine similarity between student and teacher CLS
  features over all (teacher view, student view) pairs, excluding same-index
  pairs;
- an **expansion** term: the Gaussian coding rate of the student CLS features,
  estimated from the eigenvalues of their uncentered Gram matrix;
- a **masked patch** term: plain cosine similarity between student and teacher
  patch features at the masked positions only.

Because the loss operates on cosine similarities, ``d_l2(a, b) = 2 - 2 cos(a, b)``
(up to a constant offset that does not affect gradients), which is where the
``0.5`` factor of the paper's formulation comes from -- there is no separate
coefficient to configure.

Reference:
    Zhang et al. "Simplifying DINO via Coding Rate Regularization."
    arXiv:2502.10385. https://github.com/RobinWu218/SimDINO

Note:
    Ported from the official SimDINOv2 implementation
    (``simdinov2/loss/dino_clstoken_loss.py`` and
    ``simdinov2/loss/ibot_patch_loss.py``), which is the ground truth whenever it
    disagrees with the paper:

    - the coding-rate denominator is linear in ``1/eps`` (not ``1/eps^2``);
    - the balancing factor is ``gamma = (p + N * m) / (p * N * m)``;
    - student views are ordered **locals first**, so ``student[:2]`` (used by both
      ``no_diag`` and ``expa_type=1``) refers to local crops, matching upstream.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..utils import all_reduce, get_world_size


class SimDINOv2Loss(torch.nn.Module):
    """SimDINOv2 loss: cosine compression, coding-rate expansion, masked patches.

    Args:
        eps (float): Coding-rate regularizer scale. Smaller values penalize
            feature collapse harder. Default is 0.05 (official SimDINOv2 value).
        coeff (float): Weight on the compression (cosine similarity) term.
            Default is 1.0.
        expa_type (int): How the expansion features are built. ``0`` uses
            ``student[:2]`` alone, ``1`` averages those with the teacher
            features. Default is 1 (official SimDINOv2 value), which smooths
            early-training spikes in the coding rate. Note that
            ``student[:2]`` are *local* crops under upstream's locals-first
            ordering, so neither branch uses global crops here.
        reduce_cov (bool): Sum the covariance matrices across distributed ranks.
            The official implementation defaults this to ``False`` because
            per-GPU batches of 64 samples are already large enough. Default is
            ``False``.
        cls_loss_weight (float): Weight on the whole CLS term (compression +
            expansion). Default is 1.0.
        patch_loss_weight (float): Weight on the masked patch term. Default is
            1.0.

    Shape:
        - ``student_cls_feats``: ``[n_student, B, D]``, locals ordered first
        - ``teacher_cls_feats``: ``[n_teacher, B, D]``
        - ``student_patch_feats`` / ``teacher_patch_feats``: ``[n_masked, D]``,
          flattened masked positions only
        - ``patch_mask``: ``[n_rows, n_patches]``, 1/True where a patch is masked

    Example:
        >>> loss_fn = SimDINOv2Loss()
        >>> n_rows, n_patches, per_row = 16, 196, 2
        >>> patch_mask = torch.zeros(n_rows, n_patches).bool()
        >>> patch_mask[:, :per_row] = True
        >>> n_masked = n_rows * per_row
        >>> loss, metrics = loss_fn(
        ...     student_cls_feats=torch.randn(12, 8, 256),
        ...     teacher_cls_feats=torch.randn(2, 8, 256),
        ...     student_patch_feats=torch.randn(n_masked, 256),
        ...     teacher_patch_feats=torch.randn(n_masked, 256),
        ...     patch_mask=patch_mask,
        ... )
        >>> sorted(metrics)
        ['comp_loss', 'expa_loss', 'global_comp_loss', 'loss', 'loss_cls', 'loss_patch']
    """

    def __init__(
        self,
        eps: float = 0.05,
        coeff: float = 1.0,
        expa_type: int = 1,
        reduce_cov: bool = False,
        cls_loss_weight: float = 1.0,
        patch_loss_weight: float = 1.0,
    ):
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if expa_type not in (0, 1):
            raise ValueError(f"expa_type must be 0 or 1, got {expa_type}")
        self.eps = eps
        self.coeff = coeff
        self.expa_type = expa_type
        self.reduce_cov = reduce_cov
        self.cls_loss_weight = cls_loss_weight
        self.patch_loss_weight = patch_loss_weight

    @property
    def world_size(self) -> int:
        """Number of distributed ranks, or 1 when not running distributed."""
        return get_world_size()

    def expansion_loss(self, feats: torch.Tensor) -> torch.Tensor:
        """Estimate the Gaussian coding rate of ``feats`` from its Gram matrix.

        Computes ``0.5 * mean_v logdet(I + (p / (m * N * eps)) * C_v) * gamma``
        where ``C_v`` is the uncentered covariance (second moment) matrix of view
        ``v``, ``p`` the feature dimension, ``m`` the batch size, ``N`` the world
        size and ``gamma = (p + N * m) / (p * N * m)``. Minimizing it spreads
        features apart; the logdet is evaluated in float32 for stability.

        Args:
            feats: Feature tensor ``[n_views, m, p]``.

        Returns:
            Scalar coding-rate estimate.
        """
        _, m, p = feats.shape
        cov = torch.einsum("nbc,nbd->ncd", feats, feats)
        if self.reduce_cov and self.world_size > 1:
            cov = all_reduce(cov)

        scalar = p / (m * self.world_size * self.eps)
        eye = torch.eye(p, device=cov.device, dtype=torch.float32)
        # 0.5 * logdet(A) = sum(log(diag(L))) via the Cholesky factor A = L L^T.
        # Vectorized across all views and evaluated in float32 for stability.
        chol, _ = torch.linalg.cholesky_ex(eye + scalar * cov.float())
        loss = chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1).mean()
        return loss * (p + self.world_size * m) / (p * self.world_size * m)

    def compression_loss(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
        no_diag: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean cosine similarity over student/teacher view pairs.

        Same-index pairs (teacher view ``t`` vs student view ``t``) are excluded
        when ``no_diag`` is set, following the official implementation. Note that
        with the upstream locals-first ordering those excluded pairs involve the
        first two *local* student crops.

        Upstream scores a raw dot product, which equals this cosine because the
        head L2-normalizes its output. Normalizing here costs nothing and keeps the
        term correct for unnormalized inputs.

        Args:
            student_feats: Student CLS features ``[n_student, B, D]``.
            teacher_feats: Teacher CLS features ``[n_teacher, B, D]``.
            no_diag: Exclude same-index view pairs. Default is ``True``.

        Returns:
            Tuple of the scalar mean similarity and the detached diagnostic mean
            similarity over the first ``n_teacher`` student views.
        """
        sim = F.cosine_similarity(
            teacher_feats.unsqueeze(1), student_feats.unsqueeze(0), dim=-1
        ).mean(-1)
        # Read before the diagonal is zeroed, like upstream, which is why this
        # stays usable as a run-quality diagnostic.
        global_sim = sim[:, : teacher_feats.shape[0]].detach().clone()

        n_diag = min(sim.shape) if no_diag else 0
        if n_diag > 0:
            diag = torch.arange(n_diag, device=sim.device)
            sim[diag, diag] = 0.0
        n_terms = sim.numel() - n_diag
        return sim.sum() / n_terms, global_sim.mean()

    def masked_patch_loss(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
        patch_mask: torch.Tensor,
        masks_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Negative cosine similarity at masked patch positions.

        Each row of ``patch_mask`` is one ``(view, sample)`` pair, and the row
        contributes the mean similarity over *its own* masked positions. The total
        is normalized by the number of **rows** (not the number of masked
        patches), so rows without any masked position contribute zero -- which is
        what makes ``mask_sample_probability`` act as a scale factor on this term.
        Both behaviours match the official implementation and are easy to get
        wrong.

        Args:
            student_feats: Student features at every patch position, row-major
                flattened over ``(rows, patches)`` as
                ``[n_rows * n_patches, D]``. This is what gathering out of a
                ``flatten(0, 1)``-ed token tensor yields, which is how masked
                patches reach the loss.
            teacher_feats: Teacher features, same shape as ``student_feats``.
            patch_mask: Boolean mask ``[n_rows, n_patches]``, 1/True where masked.
            masks_weight: Optional precomputed per-masked-patch weight, usually
                ``1 / patches_masked_per_row``.

        Returns:
            Scalar masked patch loss (negative cosine similarity).
        """
        sim = F.cosine_similarity(teacher_feats, student_feats, dim=-1)
        if masks_weight is None:
            masks_weight = (
                (1.0 / patch_mask.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(patch_mask)[patch_mask]
            )
        # Pinning the weight to sim's shape stops a [n_masked] tensor from
        # broadcasting against anything but the masked positions themselves.
        masks_weight = masks_weight.reshape(sim.shape)
        return -(sim * masks_weight).sum() / patch_mask.shape[0]

    def forward(
        self,
        student_cls_feats: torch.Tensor,
        teacher_cls_feats: torch.Tensor,
        student_patch_feats: Optional[torch.Tensor] = None,
        teacher_patch_feats: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        masks_weight: Optional[torch.Tensor] = None,
        no_diag: bool = True,
    ):
        """Compute the SimDINOv2 loss.

        Args:
            student_cls_feats: Student CLS features ``[n_student, B, D]`` with
                locals ordered first (see the module docstring).
            teacher_cls_feats: Teacher CLS features ``[n_teacher, B, D]``.
            student_patch_feats: Student features at masked positions
                ``[n_masked, D]``, or ``None`` to skip the patch term.
            teacher_patch_feats: Teacher features at masked positions
                ``[n_masked, D]``.
            patch_mask: Boolean mask ``[n_rows, n_patches]`` marking which
                patches were masked in the student views.
            masks_weight: Optional per-masked-patch weights ``[n_masked]``.
            no_diag (bool): Exclude same-index view pairs from the compression
                term. Default is ``True``.

        Returns:
            Tuple of the scalar total loss and a dict of detached metrics with
            keys ``loss``, ``loss_cls``, ``comp_loss``, ``expa_loss``,
            ``global_comp_loss`` and (when patches are given) ``loss_patch``.

        Raises:
            ValueError: If only one of the patch feature tensors is provided, or
                if ``patch_mask`` is missing while patch features are given.
        """
        student_cls_feats = F.normalize(student_cls_feats, dim=-1)
        teacher_cls_feats = F.normalize(teacher_cls_feats, dim=-1)

        comp_loss, global_comp_loss = self.compression_loss(
            student_cls_feats, teacher_cls_feats, no_diag=no_diag
        )
        if self.expa_type == 1:
            expansion_feats = (
                student_cls_feats[: teacher_cls_feats.shape[0]] + teacher_cls_feats
            ) / 2
        else:
            expansion_feats = student_cls_feats[: teacher_cls_feats.shape[0]]
        expa_loss = self.expansion_loss(expansion_feats)

        loss_cls = -self.coeff * comp_loss - expa_loss
        loss = self.cls_loss_weight * loss_cls
        metrics = {
            "loss_cls": loss_cls.detach(),
            "comp_loss": comp_loss.detach(),
            "expa_loss": expa_loss.detach(),
            "global_comp_loss": global_comp_loss,
        }

        if student_patch_feats is not None or teacher_patch_feats is not None:
            if student_patch_feats is None or teacher_patch_feats is None:
                raise ValueError(
                    "student_patch_feats and teacher_patch_feats must be provided "
                    f"together, got {type(student_patch_feats).__name__} and "
                    f"{type(teacher_patch_feats).__name__}"
                )
            if patch_mask is None:
                raise ValueError(
                    "patch_mask is required to normalize the masked patch loss, "
                    "got None"
                )
            loss_patch = self.masked_patch_loss(
                student_patch_feats, teacher_patch_feats, patch_mask, masks_weight
            )
            loss = loss + self.patch_loss_weight * loss_patch
            metrics["loss_patch"] = loss_patch.detach()

        metrics["loss"] = loss.detach()
        return loss, metrics
