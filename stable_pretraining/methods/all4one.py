"""All4One: Symbiotic Neighbour Contrastive Learning via Self-Attention and Redundancy Reduction.

Estepa et al., ICCV 2023: https://arxiv.org/abs/2303.09417
Reference implementation: https://github.com/ImaGonEs/all4one
"""

from typing import Any

import torch
from torch import nn

from stable_pretraining import Module
from stable_pretraining.backbone import (
    TeacherStudentWrapper,
    from_timm,
    from_torchvision,
)
from stable_pretraining.callbacks import TeacherStudentCallback
from stable_pretraining.forward import all4one
from stable_pretraining.losses import All4OneLoss
from stable_pretraining.utils.all4one import (
    All4OneAttention,
    All4OneQueue,
    build_all4one_predictor,
    build_all4one_projector,
)


class All4One(Module):
    """All4One with momentum teachers and neighbour attention :cite:`estepa2023all4one`.

    Args:
        backbone: A model name or module returning ``(batch, feature_dim)``.
            ResNet-18/50 use torchvision (solo-learn initialization); other
            names use timm.
        feature_dim: Backbone output width. If omitted, use ``num_features`` or ResNet's ``inplanes``.
            Custom backbones without that attribute must supply this argument.
        projection_dim: Width of projections and attention tokens. Default 256.
        projector_hidden_dim: Width of both projector hidden layers. Default 2048.
        predictor_hidden_dim: Hidden width of each predictor. Default 4096.
        queue_size: Number of momentum projections stored. Default 98304.
        num_neighbours: Neighbours per query. Default 5.
        num_heads: Attention heads per layer. Default 8.
        num_layers: Transformer encoder depth. Default 3.
        dropout: Transformer dropout. Default 0.1.
        ema_decay_start: Initial parameter-only EMA coefficient. Default 0.99.
        ema_decay_end: Final cosine-scheduled coefficient. Default 1.0.
        temperature: Neighbour contrast temperature. Default 0.2.
        centroid_temperature: Centroid contrast temperature. Default 0.1.
        neighbour_weight: Weight of neighbour contrast. Default 0.5.
        centroid_weight: Weight of centroid contrast. Default 0.5.
        feature_diagonal_weight: Diagonal feature RMS weight. Default 2.5 (released code).
        feature_off_diagonal_weight: Off-diagonal feature RMS weight. Default 2.5.
        low_resolution: Adapt a named backbone for small images.
        pretrained: Load pretrained weights for a named backbone.
        optim: Module optimizer/scheduler configuration. None uses Module's
            default AdamW/cosine setup; supply a dataset-specific recipe for
            reproduction, including the attention learning rate.
        **kwargs: Additional Module options such as ``hparams`` or ``parallelize_fn``.

    Note:
        Projector and loss defaults follow the released implementation; see
        the All4One guide for differences from the paper. Teacher
        BatchNorm buffers evolve through teacher forwards, independently of the
        student. The method uses Module's training loop and accepts dict batches.
        Manager supplies the EMA callback; direct Trainer use is also supported
        via ``configure_callbacks`` without duplicating an existing callback.

        Checkpoints store weights, queue contents/pointer, and EMA coefficients.
        Reconstruct with the same architecture and configuration when loading.
        Online classification/evaluation can be attached using existing callbacks.
    """

    def __init__(
        self,
        backbone: str | nn.Module = "resnet18",
        *,
        feature_dim: int | None = None,
        projection_dim: int = 256,
        projector_hidden_dim: int = 2048,
        predictor_hidden_dim: int = 4096,
        queue_size: int = 98304,
        num_neighbours: int = 5,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        ema_decay_start: float = 0.99,
        ema_decay_end: float = 1.0,
        temperature: float = 0.2,
        centroid_temperature: float = 0.1,
        neighbour_weight: float = 0.5,
        centroid_weight: float = 0.5,
        feature_diagonal_weight: float = 2.5,
        feature_off_diagonal_weight: float = 2.5,
        low_resolution: bool = False,
        pretrained: bool = False,
        optim: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(backbone, str):
            if backbone in ("resnet18", "resnet50"):
                backbone = from_torchvision(
                    backbone,
                    low_resolution=low_resolution,
                    weights="DEFAULT" if pretrained else None,
                    zero_init_residual=False,
                )
                if low_resolution:
                    backbone.conv1.padding = (2, 2)
            else:
                backbone = from_timm(
                    backbone,
                    num_classes=0,
                    low_resolution=low_resolution,
                    pretrained=pretrained,
                )
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a module or timm model name")
        if feature_dim is None:
            feature_dim = getattr(
                backbone, "num_features", getattr(backbone, "inplanes", None)
            )
        if feature_dim is None:
            raise ValueError("Supply feature_dim for a backbone without num_features")
        if projection_dim < 2:
            raise ValueError("projection_dim must be at least two for feature contrast")
        projector = build_all4one_projector(
            feature_dim, projector_hidden_dim, projection_dim
        )
        ema = dict(
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
            update_buffers=False,
            ema_schedule="step",
        )
        self.backbone = TeacherStudentWrapper(backbone, **ema)
        self.projector = TeacherStudentWrapper(projector, **ema)
        self.predictor = build_all4one_predictor(projection_dim, predictor_hidden_dim)
        self.centroid_predictor = build_all4one_predictor(
            projection_dim, predictor_hidden_dim
        )
        self.attention = All4OneAttention(
            projection_dim, num_neighbours, num_heads, num_layers, dropout
        )
        self.queue = All4OneQueue(projection_dim, queue_size, num_neighbours)
        self.all4one_loss = All4OneLoss(
            temperature,
            centroid_temperature,
            neighbour_weight,
            centroid_weight,
            feature_diagonal_weight,
            feature_off_diagonal_weight,
        )
        self.feature_dim = feature_dim
        if optim is not None:
            self.optim = optim

    def forward(
        self, batch: dict[str, Any], stage: str = "fit"
    ) -> dict[str, torch.Tensor]:
        """Delegate dict-based training and evaluation to the composable forward.

        Args:
            batch: Two augmented views for training, or images for evaluation.
            stage: Module stage, usually ``"fit"`` or ``"validate"``.

        Returns:
            Embeddings, optional labels, and training loss/representation tensors.
        """
        return all4one(self, batch, stage)

    def configure_callbacks(self) -> list[TeacherStudentCallback]:
        """Supply EMA updates if Manager or the caller has not already done so.

        Returns:
            One EMA callback, or an empty list if the trainer already has one.
        """
        if self._trainer is not None and any(
            isinstance(callback, TeacherStudentCallback)
            for callback in self.trainer.callbacks
        ):
            return []
        return [TeacherStudentCallback()]
