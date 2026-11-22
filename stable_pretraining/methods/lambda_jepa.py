"""λ-JEPA: Spectral Anti-Collapse Regularization for Self-Supervised Learning.

Self-supervised learning via multi-view invariance combined with SACReg, a
sliced Gaussian moment regularizer that pushes both the projected embeddings
and the backbone representations toward an isotropic Gaussian.

References:
    Demirel et al. "λ-JEPA: Spectral Anti-Collapse Regularization for
    Self-Supervised Learning." 2026.
    https://arxiv.org/abs/2609.35288

Example::

    from stable_pretraining.methods import LambdaJEPA

    model = LambdaJEPA("vit_small_patch16_224")

    global_images = [torch.randn(4, 3, 224, 224)] * 4
    model.train()
    output = model(global_images)
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
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import MLP, TeacherStudentWrapper
from stable_pretraining.callbacks.queues import UnsortedQueue
from stable_pretraining.utils import all_gather, is_dist_avail_and_initialized


class SACRegLoss(nn.Module):
    """SACReg sliced Gaussian moment regularizer.

    The batch of embeddings is projected onto a random orthonormal slice
    of width ``d'`` and its mean ``μ`` and covariance ``Σ`` (normalized by
    ``B - 1``) are regularized toward a standard Gaussian via::

        SACReg = ½ (‖μ‖² + tr(Σ + εI) - log det(Σ + εI) - d') / d'

    :param slice_dim: Slice width ``d'`` (default: 128).
    :param eps: Ridge added to the slice covariance (default: 1e-4).
    :param queue_length: Number of previous (detached) inputs kept in a FIFO
        queue and concatenated with the current batch (default: 0, no queue).

    Note:
        Statistics are computed in float32 over the batch gathered from all
        ranks. Pass the same ``generator`` on every rank so that all ranks
        draw the same slice.
    """

    def __init__(self, slice_dim: int = 128, eps: float = 1e-4, queue_length: int = 0):
        super().__init__()
        self.slice_dim = slice_dim
        self.eps = eps
        self.queue = UnsortedQueue(max_length=queue_length) if queue_length else None

    def forward(
        self, x: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """:param x: Embeddings [B, D].

        :param generator: Random generator for the slice (default: global RNG).
        :return: Scalar SACReg loss.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = torch.cat(all_gather(x.float()))

            samples = x
            if self.training and self.queue is not None:
                queued = self.queue.get()
                if len(queued):
                    samples = torch.cat([x, queued])
                self.queue.append(x.detach())

            gauss = torch.randn(
                samples.size(1),
                self.slice_dim,
                device=samples.device,
                generator=generator,
            )
            q, _ = torch.linalg.qr(gauss)
            p = samples @ q
            d = p.size(1)
            mu = p.mean(0)
            centered = p - mu
            cov = centered.T @ centered / (p.size(0) - 1)
            cov = cov + self.eps * torch.eye(d, device=p.device)
            logdet = 2 * torch.linalg.cholesky(cov).diagonal().log().sum()
            return 0.5 * (mu.square().sum() + cov.diagonal().sum() - logdet - d) / d


@dataclass
class LambdaJEPAOutput(ModelOutput):
    """Output from LambdaJEPA forward pass.

    :ivar loss: Combined invariance + SACReg loss (0 in eval mode).
    :ivar embedding: Backbone embeddings [V*N, D] (train) or [N, D] (eval).
    :ivar inv_loss: Invariance component.
    :ivar sacreg_z_loss: Projector SACReg component.
    :ivar sacreg_h_loss: Backbone SACReg component.
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    inv_loss: torch.Tensor = None
    sacreg_z_loss: torch.Tensor = None
    sacreg_h_loss: torch.Tensor = None


class LambdaJEPA(Module):
    """λ-JEPA: multi-view invariance + SACReg on the projector and backbone.

    Architecture:
        - **Backbone**: timm ViT (CLS-pooled, ``num_classes=0``)
        - **Projector**: MLP projection head
        - **Loss**: ``w_inv * invariance + β_z * SACReg(z̄) + β_h * SACReg(h̄)``

    The invariance term penalizes the MSE between each view's projection and
    the per-image center of an EMA copy of the backbone and projector
    (``ema_target=True``), or between all pairs of view projections
    (``ema_target=False``). The SACReg terms are applied to the per-image
    centers ``z̄`` and ``h̄`` of the projections and of the backbone
    embeddings.

    :param encoder_name: timm model name (e.g., ``"vit_base_patch16_224"``)
    :param projector: Optional projection head. When ``None``, a 3-layer
        BN+ReLU MLP (``embed_dim → 2048 → 2048 → 256``) is created.
    :param w_inv: Weight on the invariance term (default: 22.81)
    :param beta_z: Weight on the projector SACReg term (default: 222.26)
    :param beta_h: Weight on the backbone SACReg term (default: 4.054)
    :param z_slice_dim: Slice width of the projector SACReg term (default: 128)
    :param h_slice_dim: Slice width of the backbone SACReg term (default: 256)
    :param eps: Covariance ridge of the SACReg terms (default: 1e-4)
    :param z_queue_length: Queue length of the projector SACReg term
        (default: 384, i.e. 3 steps of batch 128)
    :param h_queue_length: Queue length of the backbone SACReg term
        (default: 896, i.e. 7 steps of batch 128)
    :param ema_target: Use the EMA-target invariance instead of the pairwise
        one (default: True)
    :param ema_decay_start: Initial EMA decay when ``ema_target`` (default: 0.996)
    :param ema_decay_end: Final EMA decay when ``ema_target`` (default: 1.0)
    :param pretrained: Load pretrained timm weights
    :param drop_path_rate: Stochastic depth rate for the backbone (default: 0.1)

    Note:
        - The defaults are the paper's ImageNet-1k ViT-B/16 setting (6 global
          views, batch 128). For ViT-S/16 the paper uses ``w_inv=31.07``,
          ``beta_z=225.50``, ``beta_h=1.671``, ``h_slice_dim=128`` and
          ``h_queue_length=384``; on ImageNet-100 it uses the pairwise
          invariance (``ema_target=False``).
        - Use :class:`~stable_pretraining.callbacks.TeacherStudentCallback`
          to drive teacher EMA updates during training.

    Example::

        model = LambdaJEPA("vit_base_patch16_224")
        images = torch.randn(4, 3, 224, 224)

        model.train()
        output = model(global_views=[images] * 4)
        output.loss.backward()

        model.eval()
        output = model(images=images)
        features = output.embedding  # [4, 768]

    Example with Lightning::

        import lightning as pl
        from stable_pretraining.methods import LambdaJEPA


        class LambdaJEPALightning(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = LambdaJEPA("vit_base_patch16_224")

            def training_step(self, batch, batch_idx):
                views = [v["image"] for v in batch["views"]]
                output = self.model(global_views=views)
                self.log("loss", output.loss)
                return output.loss

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1e-3)
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        projector: Optional[nn.Module] = None,
        w_inv: float = 22.81,
        beta_z: float = 222.26,
        beta_h: float = 4.054,
        z_slice_dim: int = 128,
        h_slice_dim: int = 256,
        eps: float = 1e-4,
        z_queue_length: int = 384,
        h_queue_length: int = 896,
        ema_target: bool = True,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
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
            projector = MLP(
                in_channels=embed_dim,
                hidden_channels=[2048, 2048, 256],
                norm_layer="batch_norm",
                activation_layer=nn.ReLU,
                inplace=True,
                dropout=0.0,
            )

        self.projector = projector
        if ema_target:
            self.backbone = TeacherStudentWrapper(
                self.backbone,
                warm_init=True,
                base_ema_coefficient=ema_decay_start,
                final_ema_coefficient=ema_decay_end,
            )
            self.projector = TeacherStudentWrapper(
                self.projector,
                warm_init=True,
                base_ema_coefficient=ema_decay_start,
                final_ema_coefficient=ema_decay_end,
            )
        self.ema_target = ema_target

        self.sacreg_z = SACRegLoss(
            slice_dim=z_slice_dim, eps=eps, queue_length=z_queue_length
        )
        self.sacreg_h = SACRegLoss(
            slice_dim=h_slice_dim, eps=eps, queue_length=h_queue_length
        )
        self.register_buffer("slice_step", torch.zeros((), dtype=torch.long))
        self.w_inv = w_inv
        self.beta_z = beta_z
        self.beta_h = beta_h
        self.embed_dim = embed_dim

    @staticmethod
    def _invariance(
        z: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the invariance loss.

        The pairwise MSE over views equals ``2V / (V - 1)`` times the MSE to
        the view center; with ``target`` the center is replaced by it.

        :param z: View projections [V, N, K].
        :param target: Optional per-image target [N, K].
        :return: Scalar invariance loss.
        """
        V = z.size(0)
        center = z.mean(0, keepdim=True) if target is None else target.unsqueeze(0)
        return (z - center).square().mean() * (2 * V / (V - 1))

    def _slice_generator(self, device: torch.device) -> torch.Generator:
        """Return a generator seeded with the training step for the SACReg slices.

        :param device: Device of the embeddings.
        :return: Generator shared by all ranks.
        """
        with torch.no_grad():
            step = self.slice_step.clone()
            if is_dist_avail_and_initialized():
                torch.distributed.broadcast(step, src=0)
            self.slice_step.add_(1)
        return torch.Generator(device=device).manual_seed(step.item())

    def forward(
        self,
        global_views: Optional[list[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> LambdaJEPAOutput:
        if self.training:
            assert global_views is not None and len(global_views) >= 2, (
                "at least two global_views must be provided in training mode"
            )

            bs = global_views[0].shape[0]
            n_views = len(global_views)

            images = torch.cat(global_views)
            if self.ema_target:
                features = self.backbone.forward_student(images)
                projected = self.projector.forward_student(features)
                teacher_features = self.backbone.forward_teacher(images)
                target = self.projector.forward_teacher(teacher_features)
                target = target.view(n_views, bs, -1).mean(0)
            else:
                features = self.backbone(images)
                projected = self.projector(features)
                target = None

            h = features.view(n_views, bs, -1)
            z = projected.view(n_views, bs, -1)

            inv_loss = self._invariance(z, target)
            generator = self._slice_generator(z.device)
            sacreg_z_loss = self.sacreg_z(z.mean(0), generator)
            sacreg_h_loss = self.sacreg_h(h.mean(0), generator)

            loss = (
                self.w_inv * inv_loss
                + self.beta_z * sacreg_z_loss
                + self.beta_h * sacreg_h_loss
            )
            return LambdaJEPAOutput(
                loss=loss,
                embedding=features.detach(),
                inv_loss=inv_loss,
                sacreg_z_loss=sacreg_z_loss,
                sacreg_h_loss=sacreg_h_loss,
            )
        else:
            assert images is not None, "images must be provided in eval mode"
            if self.ema_target:
                embedding = self.backbone.forward_student(images)
            else:
                embedding = self.backbone(images)
            zero = torch.tensor(0.0, device=images.device)
            return LambdaJEPAOutput(
                loss=zero,
                embedding=embedding,
                inv_loss=zero,
                sacreg_z_loss=zero,
                sacreg_h_loss=zero,
            )
