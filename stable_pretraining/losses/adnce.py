"""Adjusted InfoNCE with detached Gaussian weighting of negative similarities."""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..utils import all_gather, get_rank, get_world_size


class ADNCELoss(torch.nn.Module):
    """Compute the image formulation of Adjusted InfoNCE (ADNCE).

    Negative cosine similarities receive Gaussian weights centered at ``mu``
    with width ``sigma``. Weights have mean one per anchor and are detached
    before entering the contrastive denominator. The denominator includes the
    positive, matching the authors' image implementation. Both views act as
    anchors; an anchor's own copy and its positive are excluded from negatives.

    Args:
        temperature: Positive contrastive temperature. Default is the authors'
            CIFAR-10 setting, 0.4.
        mu: Finite center of the Gaussian on unscaled cosine similarity.
            Default is 0.7.
        sigma: Positive width of the Gaussian. Default is 0.5.

    Note:
        Distributed training uses local anchors and differentiably gathered
        global candidates. Each process must supply the same nonempty shape
        and dtype; the global batch must contain at least two image pairs.
        The returned scalar is a local mean, intended for DDP gradient averaging.
        Low-precision inputs are evaluated in float32, including under autocast;
        float64 inputs retain their precision. The loss has no trainable state.

        Reference: Wu et al., "Understanding Contrastive Learning via
        Distributionally Robust Optimization", NeurIPS 2023, arXiv:2310.11048.
        Image implementation: ``junkangwu/ADNCE`` commit
        ``cdd110f85402641d52762ba63aa2a17ba907a6f9``, ``image/model.py``.
    """

    def __init__(
        self, temperature: float = 0.4, mu: float = 0.7, sigma: float = 0.5
    ) -> None:
        super().__init__()
        for name, value in (("temperature", temperature), ("mu", mu), ("sigma", sigma)):
            if not math.isfinite(value) or (name != "mu" and value <= 0):
                requirement = "finite" if name == "mu" else "finite and positive"
                raise ValueError(f"{name} must be {requirement}, got {value}.")
        self.temperature = float(temperature)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Return the mean loss over the local batch's two views.

        Args:
            z_i: Floating-point first-view projections, shape ``[B, D]``.
            z_j: Second-view projections in the same pair order, shape ``[B, D]``.

        Returns:
            Scalar loss with gradients to both sets of projections.

        Raises:
            ValueError: Shapes, dtypes, devices, or batch sizes are incompatible.
            TypeError: Inputs are not floating-point tensors.
        """
        if not isinstance(z_i, torch.Tensor) or not isinstance(z_j, torch.Tensor):
            raise TypeError("ADNCE inputs must be floating-point tensors.")
        if not z_i.is_floating_point() or not z_j.is_floating_point():
            raise TypeError("ADNCE inputs must be floating-point tensors.")
        if z_i.ndim != 2 or z_i.shape != z_j.shape or min(z_i.shape) == 0:
            raise ValueError(
                "ADNCE requires matching nonempty [batch, features] inputs."
            )
        if z_i.device != z_j.device or z_i.dtype != z_j.dtype:
            raise ValueError("ADNCE inputs must have the same device and dtype.")

        batch = z_i.shape[0]
        world_size = get_world_size()
        if world_size > 1:
            # Check sizes collectively before a mismatched feature gather can hang.
            shape = torch.tensor(
                (*z_i.shape, int(z_i.dtype == torch.float64)),
                device=z_i.device,
                dtype=torch.long,
            )
            shapes = [torch.empty_like(shape) for _ in range(world_size)]
            dist.all_gather(shapes, shape)
            if any(not torch.equal(shape, other) for other in shapes):
                raise ValueError(
                    "ADNCE requires equal batch and feature sizes and compatible "
                    "computation dtypes on all ranks."
                )
        if batch * world_size < 2:
            raise ValueError(
                "ADNCE needs at least two image pairs in the global batch."
            )

        # Autocast must not turn the similarity matmul back into half precision.
        with torch.autocast(device_type=z_i.device.type, enabled=False):
            dtype = torch.float64 if z_i.dtype == torch.float64 else torch.float32
            anchors = F.normalize(torch.cat((z_i, z_j)).to(dtype), dim=-1)
            candidates = torch.cat(all_gather(anchors), dim=0)
            similarities = anchors @ candidates.T

            rows = torch.arange(2 * batch, device=z_i.device)
            offset = get_rank() * 2 * batch if world_size > 1 else 0
            positives = offset + (rows + batch) % (2 * batch)
            positive = similarities[rows, positives].unsqueeze(1)
            mask = torch.ones_like(similarities, dtype=torch.bool)
            mask[rows, offset + rows] = False
            mask[rows, positives] = False
            negative = similarities[mask].view(2 * batch, -1)

            # The Gaussian prefactor cancels in mean normalization. Log-softmax
            # avoids underflow when every unnormalized Gaussian weight is tiny.
            gaussian_logits = (
                -0.5 * ((negative.detach() - self.mu) / self.sigma).square()
            )
            log_weights = F.log_softmax(gaussian_logits, dim=1) + math.log(
                negative.size(1)
            )
            relative_logits = (negative - positive) / self.temperature + log_weights
            logits = torch.cat((torch.zeros_like(positive), relative_logits), dim=1)
            return torch.logsumexp(logits, dim=1).mean()
