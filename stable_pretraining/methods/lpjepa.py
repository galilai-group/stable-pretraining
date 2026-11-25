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

from typing import Any, Optional

import timm
import torch
import torch.nn as nn

from stable_pretraining import Module
from stable_pretraining.forward import lpjepa as lpjepa_forward
from stable_pretraining.losses import RDMReg, SlicedWassersteinDistance

# Existing Hydra configs and pickled modules can refer to the original loss paths.
__all__ = ["LpJEPA", "RDMReg", "SlicedWassersteinDistance"]


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
