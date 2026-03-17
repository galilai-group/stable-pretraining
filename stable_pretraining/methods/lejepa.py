"""LeJEPA: Latent Embedding Joint-Embedding Predictive Architecture.

Self-supervised learning via multi-view invariance combined with a
sliced goodness-of-fit test (SIGReg) that pushes embeddings toward
an isotropic Gaussian.

References:
    Balestriero & LeCun. "LeJEPA: Provable and Scalable
    Self-Supervised Learning Without the Heuristics." 2025.
    https://arxiv.org/abs/2511.08544

Example::

    from stable_pretraining.methods.lejepa import LeJEPA

    model = LeJEPA("vit_small_patch16_224")

    global_images = [torch.randn(4, 3, 224, 224)] * 2
    all_images = [torch.randn(4, 3, 224, 224)] * 6
    model.train()
    output = model(global_images, all_images)
    output.loss.backward()

    model.eval()
    output = model(images=torch.randn(4, 3, 224, 224))
    features = output.embedding  # [N, D]
"""

from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn
from torch.distributed.nn import all_reduce

from stable_pretraining import Module
from stable_pretraining.backbone import MLP


class EppsPulley(nn.Module):
    """Epps-Pulley goodness-of-fit test for univariate normality.

    Projects data onto a grid of points and computes the Epps-Pulley statistic.

    :param t_max: Integration upper bound.
    :param n_points: Number of integration points.
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1

        self._is_ddp = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.world_size = torch.distributed.get_world_size() if self._is_ddp else 1

        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        self.register_buffer("t", t)

        phi = (-0.5 * t**2).exp()
        self.register_buffer("phi", phi)

        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Samples [N, S] (N samples, S slices).

        :return: Per-slice statistic [S].
        """
        N = x.size(0)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = x_t.cos().mean(0)
        sin_mean = x_t.sin().mean(0)

        if self._is_ddp:
            all_reduce(cos_mean, op=torch.distributed.ReduceOp.AVG)
            all_reduce(sin_mean, op=torch.distributed.ReduceOp.AVG)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N * self.world_size


class SlicedEppsPulley(nn.Module):
    """Sliced Epps-Pulley goodness-of-fit test for multivariate normality.

    Projects data onto random 1-D directions and averages the univariate
    Epps-Pulley statistics.  A synchronised step counter seeds the random
    projections so all DDP ranks sample identical directions.

    :param num_slices: Number of random 1-D projections.
    :param t_max: EP integration upper bound.
    :param n_points: EP quadrature nodes.
    """

    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        self._is_ddp = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.num_slices = num_slices
        self.ep = EppsPulley(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Embeddings [N, D].

        :return: Scalar mean EP statistic.
        """
        with torch.no_grad():
            step = self.global_step.clone()

            if self._is_ddp:
                # All ranks increment global_step in lockstep, so this
                # broadcast is redundant under normal synchronous training.
                # It is kept as a safety net against step drift from
                # uneven batches (e.g. drop_last=False).
                torch.distributed.broadcast(step, src=0)

            g = torch.Generator(device=x.device).manual_seed(step.item())
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0)
            self.global_step.add_(1)

        proj = x @ A
        return self.ep(proj).mean()


@dataclass
class LeJEPAOutput:
    """Output from LeJEPA forward pass.

    :ivar loss: Combined invariance + SIGReg loss (0 in eval mode).
    :ivar embedding: Backbone embeddings [V*N, D] (train) or [N, D] (eval).
    :ivar inv_loss: Invariance component.
    :ivar sigreg_loss: Epps-Pulley goodness-of-fit component.
    """

    loss: torch.Tensor
    embedding: torch.Tensor
    inv_loss: torch.Tensor
    sigreg_loss: torch.Tensor


class LeJEPA(Module):
    """LeJEPA: multi-view invariance + sliced Epps-Pulley SIGReg.

    Architecture:
        - **Backbone**: timm ViT (CLS-pooled, ``num_classes=0``)
        - **Projector**: MLP projection head
        - **Loss**: ``invariance + (λ * SIGReg)``

    Centers are computed from global-view projections only.  The invariance
    term penalises the MSE between each view's projection and the center.
    The SIGReg term is a sliced goodness-of-fit test that pushes
    projected embeddings toward an isotropic Gaussian, averaged over views.

    :param encoder_name: timm model name (e.g., ``"vit_base_patch16_224"``)
    :param projector: Optional projection head.  When ``None``, a 3-layer
        BN+ReLU MLP (``embed_dim → 2048 → 2048 → 512``) is created.
    :param n_slices: Random projection directions for the goodness-of-fit test (default: 1024)
    :param t_max: EP integration upper bound (default: 3.0)
    :param n_points: EP quadrature nodes (default: 17)
    :param lamb: SIGReg weight λ (default: 0.02)
    :param pretrained: Load pretrained timm weights

    Example::

        model = LeJEPA("vit_base_patch16_224")
        images = torch.randn(4, 3, 224, 224)

        model.train()
        output = model(
            global_views=[images, images],
            all_views=[images, images, images, images],
        )
        output.loss.backward()

        model.eval()
        output = model(images=images)
        features = output.embedding  # [4, 768]

    Example with Lightning::

        import lightning as pl
        from stable_pretraining.methods.lejepa import LeJEPA


        class LeJEPALightning(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = LeJEPA("vit_base_patch16_224")

            def training_step(self, batch, batch_idx):
                views = [v["image"] for v in batch["views"]]
                output = self.model(global_views=views, all_views=views)
                self.log("loss", output.loss)
                return output.loss

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1e-3)
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        projector: Optional[nn.Module] = None,
        n_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
        lamb: float = 0.02,
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

        self.sigreg = SlicedEppsPulley(
            num_slices=n_slices, t_max=t_max, n_points=n_points
        )
        self.lamb = lamb
        self.embed_dim = embed_dim

    @staticmethod
    def _compute_loss(
        all_projected: torch.Tensor,
        n_global: int,
        sigreg: SlicedEppsPulley,
        lamb: float,
    ):
        """Compute the LeJEPA loss.

        :param all_projected: All view projections [V, N, K].
        :param n_global: Number of global views.
        :param sigreg: SlicedEppsPulley module.
        :param lamb: SIGReg weight λ.
        :return: Tuple of (total_loss, inv_loss, sigreg_loss).
        """
        centers = all_projected[:n_global].mean(0)  # [N, K]
        inv_loss = (centers.unsqueeze(0) - all_projected).square().mean()

        sigreg_loss = sigreg(all_projected.reshape(-1, all_projected.size(-1)))

        loss = inv_loss + lamb * sigreg_loss
        return loss, inv_loss, sigreg_loss

    def forward(
        self,
        global_views: Optional[list[torch.Tensor]] = None,
        local_views: Optional[list[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> LeJEPAOutput:
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

            loss, inv_loss, sigreg_loss = self._compute_loss(
                all_projected, len(global_views), self.sigreg, self.lamb
            )

            embedding = g_features.detach()
            return LeJEPAOutput(
                loss=loss,
                embedding=embedding,
                inv_loss=inv_loss,
                sigreg_loss=sigreg_loss,
            )
        else:
            assert images is not None, "images must be provided in eval mode"
            embedding = self.backbone(images)
            zero = torch.tensor(0.0, device=images.device)
            return LeJEPAOutput(
                loss=zero,
                embedding=embedding,
                inv_loss=zero,
                sigreg_loss=zero,
            )
