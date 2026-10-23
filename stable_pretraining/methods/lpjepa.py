"""Rectified LpJEPA: invariance and rectified distribution matching.

Self-supervised learning by aligning the rectified projections of two
augmented views. Rectified Distribution Matching Regularization (RDMReg)
matches each view's distribution to a product of rectified generalized
Gaussian distributions using the squared sliced 2-Wasserstein distance.

References:
    Kuang et al. "Rectified LpJEPA: Joint-Embedding Predictive Architectures
    with Sparse and Maximum-Entropy Representations." 2026.
    https://arxiv.org/abs/2602.01456
"""

import math
from typing import Any, Optional

import timm
import torch
import torch.nn as nn
from torch.distributed.nn import all_gather

from stable_pretraining import Module
from stable_pretraining.forward import lpjepa as lpjepa_forward


class SlicedWassersteinDistance(nn.Module):
    """Squared sliced 2-Wasserstein distance for equally weighted samples.

    Projects two sample sets onto supplied unit directions, sorts each
    projected batch, and averages the squared differences. Target samples
    and directions are supplied by the caller.
    """

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """Compare empirical distributions along the supplied directions.

        All inputs must share a floating-point dtype and device. Samples
        are matched by their sorted projections, not by their row indices.

        :param x: Samples [N, D].
        :param targets: Reference samples [N, D], with the same shape as x.
        :param directions: Unit projection vectors stored as columns [D, S].
        :return: Scalar squared distance averaged over samples and directions,
            differentiable through the projections and sorting.
        """
        projected = (x @ directions).sort(dim=0).values
        projected_targets = (targets @ directions).sort(dim=0).values
        return (projected - projected_targets).square().mean()


class RDMReg(nn.Module):
    """Rectified Distribution Matching Regularization (RDMReg).

    Matches two views' rectified projections to a product of rectified
    generalized Gaussian distributions. The views share random projection
    directions and use independently sampled targets. Their squared sliced
    2-Wasserstein distances are averaged.

    Under DDP, each view's batch is gathered with gradients; local batch
    sizes must be equal across ranks. A checkpointed, synchronised counter
    seeds a private generator so all ranks sample identical directions and
    targets. The counter advances once per forward call.

    :param num_slices: Number of random 1-D projections (default: 1024).
    :param p: Positive shape parameter. ``1`` gives Laplace and ``2`` gives
        Gaussian samples before rectification (default: 1.0).
    :param mu: Location of the target distribution before rectification
        (default: 0.0).
    :param sigma: Positive scale before rectification. When ``None``, choose
        the scale that gives the unrectified distribution unit variance.
    """

    def __init__(
        self,
        num_slices: int = 1024,
        p: float = 1.0,
        mu: float = 0.0,
        sigma: Optional[float] = None,
    ) -> None:
        super().__init__()
        self._is_ddp = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        if sigma is None:
            sigma = math.exp(
                0.5 * (math.lgamma(1 / p) - math.lgamma(3 / p)) - math.log(p) / p
            )

        self.num_slices = num_slices
        self.p = float(p)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.swd = SlicedWassersteinDistance()
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    @staticmethod
    @torch.no_grad()
    def _sample_product_laplace(
        x: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """Sample independent Laplace(0, 1) coordinates matching x.

        :param x: Reference tensor supplying the shape, dtype, and device.
            Its values are unused.
        :param generator: Private generator for magnitudes and signs.
        :return: Unrectified standard Laplace samples with the same shape as x.
        """
        # A fair sign times an Exp(1) sample is Laplace(0, 1). These tensor
        # samplers accept a generator, unlike torch.distributions.Laplace.sample.
        magnitudes = torch.empty_like(x).exponential_(1.0, generator=generator)
        signs = torch.empty_like(x).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        return signs * magnitudes

    @torch.no_grad()
    def _sample_targets(
        self, x: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """Sample the product of rectified generalized Gaussian distributions.

        :param x: Reference tensor supplying the shape, dtype, and device.
            Its values are unused.
        :param generator: Private generator for target sampling.
        :return: Rectified target samples with the same shape as x.
        """
        if self.p == 1.0:
            samples = self._sample_product_laplace(x, generator)
        elif self.p == 2.0:
            samples = torch.randn(
                x.shape, device=x.device, dtype=x.dtype, generator=generator
            )
        else:
            concentration = x.new_tensor(1 / self.p).expand_as(x)
            # The underlying Gamma kernel permits explicit RNG control
            # even on older PyTorch versions without Gamma.sample(generator=...).
            gamma = torch._standard_gamma(concentration, generator=generator)
            gamma.clamp_min_(torch.finfo(x.dtype).tiny)
            magnitudes = (self.p * gamma).pow(1 / self.p)
            signs = (
                torch.empty_like(x).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
            )
            samples = signs * magnitudes
        return (self.mu + self.sigma * samples).relu()

    @torch.no_grad()
    def _sample_directions(
        self, x: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """Sample random unit directions shared by both views.

        :param x: Embeddings [B, P], supplying the feature dimension, dtype,
            and device.
        :param generator: Private generator for direction sampling.
        :return: Unit projection vectors stored as columns [P, num_slices].
        """
        directions = torch.randn(
            x.shape[1],
            self.num_slices,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        norms = directions.norm(dim=0, keepdim=True)
        return directions / norms.clamp_min(torch.finfo(x.dtype).tiny)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute the mean distribution-matching loss for two views.

        Inputs are projections after ReLU. Half-precision inputs are promoted
        to float32 for sampling and loss computation.

        :param z1: First view's rectified projections [B, P].
        :param z2: Second view's rectified projections, with the same shape,
            floating-point dtype, and device as z1.
        :return: Scalar mean squared sliced 2-Wasserstein distance, with
            gradients to both views.
        """
        # Lightning may initialize the process group after this module is created.
        self._is_ddp = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        # Sampling and squaring can overflow when AMP supplies half precision.
        with torch.autocast(device_type=z1.device.type, enabled=False):
            if z1.dtype in (torch.float16, torch.bfloat16):
                z1, z2 = z1.float(), z2.float()

            with torch.no_grad():
                step = self.global_step.to(device=z1.device, copy=True)
                if self._is_ddp:
                    torch.distributed.broadcast(step, src=0)
                generator = torch.Generator(device=z1.device).manual_seed(step.item())
                directions = self._sample_directions(z1, generator)
                self.global_step.copy_(step + 1)

            if self._is_ddp:
                # Sorting needs the global batch, unlike a reducible mean statistic.
                z1 = torch.cat(all_gather(z1.contiguous()), dim=0)
                z2 = torch.cat(all_gather(z2.contiguous()), dim=0)

            targets1 = self._sample_targets(z1, generator)
            targets2 = self._sample_targets(z2, generator)
            loss1 = self.swd(z1, targets1, directions)
            loss2 = self.swd(z2, targets2, directions)
            return (loss1 + loss2) / 2


class LpJEPA(Module):
    """Rectified LpJEPA: two-view invariance + RDMReg.

    Architecture:
        - **Encoder**: timm backbone producing pooled features [B, D].
        - **Projector**: MLP followed by ReLU, producing projections [B, P].
        - **Loss**: ``inv_weight * MSE + rdmreg_weight * RDMReg``.

    Both losses operate on rectified projections of two augmented views.
    The invariance term compares corresponding images across views, while
    RDMReg matches each view's distribution to sampled targets. The forward
    delegates to :func:`stable_pretraining.forward.lpjepa`. Evaluation returns
    encoder features and rectified projections, without loss computation.

    :param inv_weight: Weight of the invariance MSE term.
    :param rdmreg_weight: Weight of the distribution-matching term.
    :param encoder_name: timm model name (default: ``"vit_base_patch16_224"``).
    :param projector: Optional projection head mapping [B, D] to [B, P].
        When ``None``, create a 3-layer MLP with dimensions
        ``D -> 2048 -> 2048 -> 2048`` and BN+ReLU after the first two layers,
        matching the paper's ImageNet-100 setup. A final ReLU is applied in
        ``forward`` for both default and custom projectors.
    :param n_slices: Random projection directions for RDMReg (default: 1024).
    :param p: Positive target shape parameter. ``1`` gives Laplace and ``2``
        gives Gaussian samples before rectification (default: 1.0).
    :param mu: Target location before rectification (default: 0.0).
    :param sigma: Target scale before rectification. When ``None``, choose
        the scale that gives the unrectified distribution unit variance.
    :param pretrained: Load pretrained timm encoder weights (default: False).
    :param drop_path_rate: Encoder stochastic depth rate (default: 0.1).

    Example:
        Create a small ViT model and exercise the dictionary API with synthetic
        image batches. In training, the two views must be augmentations of the
        same images in matching row order.

        .. code-block:: python

            import torch
            import stable_pretraining as spt

            model = spt.LpJEPA(
                inv_weight=25.0,
                rdmreg_weight=125.0,
                encoder_name="vit_tiny_patch16_224",
                pretrained=False,
            )
            views = [{"image": torch.randn(2, 3, 224, 224)} for _ in range(2)]

            model.train()
            output = model({"views": views}, stage="fit")
            output["loss"].backward()

            model.eval()
            with torch.no_grad():
                output = model(views[0], stage="validate")
            embedding = output["embedding"]  # [2, 192] backbone features
            projection = output["projection"]  # [2, 2048] rectified projections

        Both entry points use :func:`stable_pretraining.forward.lpjepa`, which
        documents the batch format, output keys, and training/evaluation behavior.
    """

    def __init__(
        self,
        inv_weight: float,
        rdmreg_weight: float,
        encoder_name: str = "vit_base_patch16_224",
        projector: Optional[nn.Module] = None,
        n_slices: int = 1024,
        p: float = 1.0,
        mu: float = 0.0,
        sigma: Optional[float] = None,
        pretrained: bool = False,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.inv_weight = inv_weight
        self.rdmreg_weight = rdmreg_weight
        self.backbone = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            **({"dynamic_img_size": True} if "vit" in encoder_name else {}),
            drop_path_rate=drop_path_rate,
        )
        if projector is None:
            projector = nn.Sequential(
                nn.Linear(self.backbone.num_features, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
            )

        self.projector = projector
        self.rdmreg = RDMReg(num_slices=n_slices, p=p, mu=mu, sigma=sigma)

    def forward(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
        """Run the shared LpJEPA forward function.

        Args:
            batch: Two-view training batch or evaluation batch, following
                :func:`stable_pretraining.forward.lpjepa`.
            stage: Logging prefix, such as ``"fit"`` or ``"validate"``.

        Returns:
            Dictionary of encoder features, rectified projections, optional
            labels, and training losses from the shared forward function.
        """
        return lpjepa_forward(self, batch, stage)
