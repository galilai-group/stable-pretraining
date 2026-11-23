"""DAE-Net reconstruction and regularization objective.

Source and license: THIRD_PARTY_NOTICES.md (DAE-Net).
"""

import math

import torch
from torch import Tensor, nn

from ..utils.distributed import all_reduce, get_world_size


class DAENetLoss(nn.Module):
    """Compute the reconstruction and regularization terms of DAE-Net.

    Args:
        occupancy_weight: Multiplier of max-occupancy reconstruction.
        sparse_weight: Existence sparsity multiplier (.002 for Chair, .02 for Mug).
        deform_weight: Multiplier of occupancy-weighted squared displacements.
        affine_weight: Residual affine penalty multiplier, disabled by default.

    Note:
        Sparsity is negative. The occupancy floor is applied before squaring,
        and displacement weighting uses detached raw occupancy before squaring.
        The caller supplies the era-dependent regularization weight.
    """

    def __init__(
        self,
        occupancy_weight: float = 0.1,
        sparse_weight: float = 0.002,
        deform_weight: float = 100.0,
        affine_weight: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in dict(
            occupancy_weight=occupancy_weight,
            sparse_weight=sparse_weight,
            deform_weight=deform_weight,
            affine_weight=affine_weight,
        ).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, value)

    def forward(
        self,
        output: dict[str, Tensor],
        target: Tensor,
        regularization_weight: float = 1.0,
        *,
        distributed: bool = False,
    ) -> dict[str, Tensor]:
        """Return the total scalar loss and weighted scalar components.

        Args:
            output: The result of ``DAENetBackbone.decode``.
            target: Occupancy targets ``[B,Q,1]``.
            regularization_weight: Era warmup multiplier, normally .001 to 1.
            distributed: Enable collectives only when all training ranks call
                this loss together. Offline and rank-zero evaluation must leave
                this disabled.

        Returns:
            Dictionary with ``loss``, ``reconstruction_sum``,
            ``reconstruction_max``, ``sparsity``, ``deformation``, and ``affine``.
            With multiple ranks, components and ``loss_report`` are detached
            global metrics; ``loss`` carries the local DDP backward contribution.
            ``global_batch_size`` supplies the epoch logging weight.

        Raises:
            ValueError: If targets could broadcast silently or the weight is invalid.
        """
        summed, maximum = output["occupancy_sum"], output["occupancy_max"]
        if (
            target.ndim != 3
            or target.shape[-1] != 1
            or target.numel() == 0
            or target.shape != summed.shape
            or target.shape != maximum.shape
        ):
            raise ValueError(
                "target and sum/max occupancies must have the same [B,Q,1] shape"
            )
        if not math.isfinite(regularization_weight) or regularization_weight < 0:
            raise ValueError("regularization_weight must be finite and nonnegative")
        raw = output["raw_occupancy"]
        batch, queries, _ = target.shape
        if raw.ndim != 3 or raw.shape[:2] != (batch, queries) or raw.shape[2] == 0:
            raise ValueError("raw_occupancy must have shape [B,Q,K] with K > 0")
        branches = raw.shape[2]
        for key, shape in {
            "existence": (batch, branches, 1),
            "affine": (batch, branches, 12),
            "deformation": (batch, branches, 3, queries),
        }.items():
            if output[key].shape != shape:
                raise ValueError(f"{key} must have shape {shape}")
        world_size = get_world_size() if distributed else 1
        existence_mean = output["existence"].mean(dim=0)
        if world_size > 1:
            local_counts = summed.new_tensor(
                [
                    batch,
                    target.numel(),
                    output["affine"].numel(),
                    output["deformation"].numel(),
                ]
            )
            (
                global_batch_size,
                global_query_elements,
                global_affine_elements,
                global_deformation_elements,
            ) = all_reduce(local_counts).unbind()
            if self.sparse_weight > 0:
                # The mean must be global before squaring. The differentiable
                # SUM backward cancels DDP's averaging of parameter gradients.
                existence_mean = (
                    all_reduce(output["existence"].sum(dim=0)) / global_batch_size
                )
        terms = {
            "reconstruction_sum": torch.where(
                target > 0.5, 1 - summed, summed.clamp_min(0.1)
            )
            .square()
            .mean(),
            "reconstruction_max": torch.where(
                target > 0.5, 1 - maximum, maximum.clamp_min(0.1)
            )
            .square()
            .mean()
            * self.occupancy_weight,
        }
        zero = summed.new_zeros(())
        terms["sparsity"] = (
            -(existence_mean - 1).square().mean()
            * (self.sparse_weight * regularization_weight)
            if self.sparse_weight > 0
            else zero
        )
        terms["affine"] = (
            output["affine"].square().mean()
            * (self.affine_weight * regularization_weight)
            if self.affine_weight > 0
            else zero
        )
        terms["deformation"] = (
            (
                output["deformation"]
                * output["raw_occupancy"].detach().transpose(1, 2).unsqueeze(2)
            )
            .square()
            .mean()
            * (self.deform_weight * regularization_weight)
            if self.deform_weight > 0
            else zero
        )
        if world_size > 1:
            # DDP averages gradients across ranks, so local mean terms need
            # world_size * local_count / global_count, including uneven shards.
            for name, local_count, global_count in (
                ("reconstruction_sum", target.numel(), global_query_elements),
                ("reconstruction_max", target.numel(), global_query_elements),
                ("affine", output["affine"].numel(), global_affine_elements),
                (
                    "deformation",
                    output["deformation"].numel(),
                    global_deformation_elements,
                ),
            ):
                terms[name] = terms[name] * (world_size * local_count / global_count)
        # Match upstream's addition order for numerical comparisons.
        loss = (
            terms["reconstruction_sum"]
            + terms["reconstruction_max"]
            + terms["sparsity"]
            + terms["affine"]
            + terms["deformation"]
        )
        if world_size > 1:
            decomposable_names = (
                "reconstruction_sum",
                "reconstruction_max",
                "affine",
                "deformation",
            )
            metrics = (
                all_reduce(
                    torch.stack([terms[name].detach() for name in decomposable_names])
                )
                / world_size
            )
            reported_terms = dict(zip(decomposable_names, metrics))
            reported_terms["sparsity"] = terms["sparsity"].detach()
            report = (
                reported_terms["reconstruction_sum"]
                + reported_terms["reconstruction_max"]
                + reported_terms["sparsity"]
                + reported_terms["affine"]
                + reported_terms["deformation"]
            )
            return {
                "loss": loss,
                "loss_report": report,
                "global_batch_size": global_batch_size.detach(),
                **reported_terms,
            }
        return {"loss": loss, **terms}
