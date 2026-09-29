"""VISReg: multi-view invariance, centering, variance, and sliced wasserstein distance.

Self-supervised learning via multi-view invariance combined with VISReg, a
sliced goodness-of-fit regularizer that pushes embeddings toward an isotropic
standard Gaussian. VISReg is a drop-in alternative to the Epps-Pulley SIGReg
term used by :class:`~stable_pretraining.methods.LeJEPA`: instead of matching
the empirical characteristic function, it matches the sorted 1-D projections of
the embeddings to the theoretical Gaussian quantiles (a sliced Wasserstein
distance to the standard normal), plus explicit mean-centering and unit-scale
terms. It provides a strong embedding collapse prevention signal, and the
flexibility of regularization terms. It is very useful for low-quality datasets.

References:
    Wu et al. "VISReg." 2026.
    https://arxiv.org/abs/2606.02572

Example::

    from stable_pretraining.methods import VISReg

    model = VISReg("vit_small_patch16_224")

    global_images = [torch.randn(4, 3, 224, 224)] * 2
    all_images = [torch.randn(4, 3, 224, 224)] * 6
    model.train()
    output = model(global_images, all_images)
    output.loss.backward()

    model.eval()
    output = model(images=torch.randn(4, 3, 224, 224))
    features = output.embedding  # [N, D]
"""

import math
from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import MLP


class VISRegLoss(nn.Module):
    """VISReg sliced Gaussianity regularizer.

    For each view (leading dimension) the batch of embeddings is regularized
    toward an isotropic standard Gaussian via three terms:

    - **center**: penalizes non-zero feature means.
    - **scale**: penalizes per-feature standard deviations away from one.
    - **shape**: projects the standardized embeddings onto ``K`` random 1-D
      directions, sorts each projection over the batch, and matches it to the
      theoretical standard-normal quantiles (a sliced Wasserstein-2 distance
      to the standard normal).

    :param num_projections: Number of random 1-D projections ``K`` (default: 256).
    :param lambda_scale: Weight on the scale (unit-variance) term (default: 1.0).
    :param lambda_shape: Weight on the shape (sliced-quantile) term (default: 1.0).
    :param lambda_center: Weight on the center (zero-mean) term (default: 1.0).

    Note:
        Statistics are computed over the batch dimension of each view
        independently, and the random projections are drawn fresh on every
        forward pass. Under DDP each rank regularizes its own local batch;
        gradients are then averaged across ranks by the framework.
    """

    def __init__(
        self,
        num_projections: int = 256,
        lambda_scale: float = 1.0,
        lambda_shape: float = 1.0,
        lambda_center: float = 1.0,
    ):
        super().__init__()
        self.K = num_projections
        self.lambda_scale = lambda_scale
        self.lambda_shape = lambda_shape
        self.lambda_center = lambda_center
        self._cached_B = -1
        self._cached_target = None

    def _get_target(self, B: int, device) -> torch.Tensor:
        """Theoretical standard-normal quantiles for ``B`` sorted samples."""
        if self._cached_B != B:
            q = torch.linspace(1, B, B, device=device, dtype=torch.float32) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """:param z: Embeddings [V, B, D] (views, batch, dim).

        :return: Scalar VISReg loss (center + scale + shape).
        """
        _, B, D = z.shape

        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()

        z_centered = z - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)).clamp(min=1e-6)
        scale_loss = (std - 1.0).pow(2).mean()

        z_norm = z_centered / std.detach().unsqueeze(1)
        W = F.normalize(torch.randn(D, self.K, device=z.device, dtype=z.dtype), dim=0)
        p_sorted = (z_norm @ W).sort(dim=1).values
        target = self._get_target(B, z.device).view(1, B, 1)
        shape_loss = (p_sorted - target).pow(2).mean()

        return (
            self.lambda_scale * scale_loss
            + self.lambda_shape * shape_loss
            + self.lambda_center * center_loss
        )


@dataclass
class VISRegOutput(ModelOutput):
    """Output from VISReg forward pass.

    :ivar loss: Combined invariance + VISReg loss (0 in eval mode).
    :ivar embedding: Backbone embeddings [V*N, D] (train) or [N, D] (eval).
    :ivar inv_loss: Invariance component.
    :ivar visreg_loss: Sliced Gaussianity component.
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    inv_loss: torch.Tensor = None
    visreg_loss: torch.Tensor = None


class VISReg(Module):
    """VISReg: multi-view invariance + centering + variance + sliced wasserstein distance.

    Architecture:
        - **Backbone**: timm ViT (CLS-pooled, ``num_classes=0``)
        - **Projector**: MLP projection head
        - **Loss**: ``(1 - λ) * invariance + (λ * VISReg)``

    Centers are computed from global-view projections only. The invariance
    term penalizes the MSE between each view's projection and the center.
    The VISReg term is a sliced goodness-of-fit regularizer that pushes
    projected embeddings toward an isotropic standard Gaussian, applied
    per view over the batch dimension.

    :param encoder_name: timm model name (e.g., ``"vit_base_patch16_224"``)
    :param projector: Optional projection head. When ``None``, a 3-layer
        BN+ReLU MLP (``embed_dim → 2048 → 2048 → 512``) is created.
    :param num_projections: Random projection directions for the shape term (default: 256)
    :param lambda_scale: Weight on the VISReg scale (unit-variance) term (default: 1.0)
    :param lambda_shape: Weight on the VISReg shape (sliced-quantile) term (default: 1.0)
    :param lambda_center: Weight on the VISReg center (zero-mean) term (default: 1.0)
    :param lamb: Convex mixing weight λ ∈ [0, 1] between the invariance and
        VISReg terms: ``(1 - λ) * inv + λ * visreg`` (default: 0.02)
    :param pretrained: Load pretrained timm weights
    :param drop_path_rate: Stochastic depth rate for the backbone (default: 0.1)

    Example::

        model = VISReg("vit_base_patch16_224")
        images = torch.randn(4, 3, 224, 224)

        model.train()
        output = model(
            global_views=[images, images],
            local_views=[images, images],
        )
        output.loss.backward()

        model.eval()
        output = model(images=images)
        features = output.embedding  # [4, 768]

    Example with Lightning::

        import lightning as pl
        from stable_pretraining.methods import VISReg


        class VISRegLightning(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = VISReg("vit_base_patch16_224")

            def training_step(self, batch, batch_idx):
                views = [v["image"] for v in batch["views"]]
                output = self.model(global_views=views, local_views=views)
                self.log("loss", output.loss)
                return output.loss

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1e-3)
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        projector: Optional[nn.Module] = None,
        num_projections: int = 256,
        lambda_scale: float = 1.0,
        lambda_shape: float = 1.0,
        lambda_center: float = 1.0,
        lamb: float = 0.9,
        pretrained: bool = False,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            **({"dynamic_img_size": True} if "vit" in encoder_name else {}),
            drop_path_rate=drop_path_rate,
        )

        embed_dim = self.backbone.embed_dim

        if projector is None:
            projector = nn.Sequential(
                nn.Linear(embed_dim, 512, bias=True),
                MLP(
                    in_channels=512,
                    hidden_channels=[2048, 2048, 512],
                    norm_layer="batch_norm",
                    activation_layer=nn.ReLU,
                    inplace=True,
                    dropout=0.0,
                ),
            )

        self.projector = projector

        self.visreg = VISRegLoss(
            num_projections=num_projections,
            lambda_scale=lambda_scale,
            lambda_shape=lambda_shape,
            lambda_center=lambda_center,
        )
        self.lamb = lamb
        self.embed_dim = embed_dim

    @staticmethod
    def _compute_loss(
        all_projected: torch.Tensor,
        n_global: int,
        visreg: VISRegLoss,
        lamb: float,
    ):
        """Compute the VISReg loss.

        :param all_projected: All view projections [V, N, K].
        :param n_global: Number of global views.
        :param visreg: VISRegLoss module.
        :param lamb: Convex mixing weight λ: ``(1 - λ) * inv + λ * visreg``.
        :return: Tuple of (total_loss, inv_loss, visreg_loss).
        """
        centers = all_projected[:n_global].mean(0)  # [N, K]
        inv_loss = (centers.unsqueeze(0) - all_projected).square().mean()

        visreg_loss = visreg(all_projected)

        loss = (1 - lamb) * inv_loss + lamb * visreg_loss
        return loss, inv_loss, visreg_loss

    def forward(
        self,
        global_views: Optional[list[torch.Tensor]] = None,
        local_views: Optional[list[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> VISRegOutput:
        if self.training:
            assert global_views is not None and local_views is not None, (
                "global_views and local_views must be provided in training mode"
            )

            g_features = self.backbone(torch.cat(global_views))
            l_features = self.backbone(torch.cat(local_views))

            all_features = torch.cat([g_features, l_features])
            all_projected = self.projector(all_features)

            bs = global_views[0].shape[0]
            n_views = len(global_views) + len(local_views)
            all_projected = all_projected.view(n_views, bs, -1)

            loss, inv_loss, visreg_loss = self._compute_loss(
                all_projected, len(global_views), self.visreg, self.lamb
            )

            embedding = g_features.detach()
            return VISRegOutput(
                loss=loss,
                embedding=embedding,
                inv_loss=inv_loss,
                visreg_loss=visreg_loss,
            )
        else:
            assert images is not None, "images must be provided in eval mode"
            embedding = self.backbone(images)
            zero = torch.tensor(0.0, device=images.device)
            return VISRegOutput(
                loss=zero,
                embedding=embedding,
                inv_loss=zero,
                visreg_loss=zero,
            )
