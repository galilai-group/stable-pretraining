"""DirectCLR: contrastive learning on a fixed encoder subspace.

Reference: Jing et al., ICLR 2022. https://arxiv.org/abs/2110.09348
"""

from itertools import chain
from typing import Any

import torch
from torch import nn

from ..backbone import from_timm
from ..forward import directclr
from ..losses import CLIPLoss
from ..module import Module


class DirectCLR(Module):
    """Apply symmetric cross-view InfoNCE to fixed leading feature coordinates.

    Args:
        encoder_name: timm model name or an image encoder returning flat features.
        subspace_dim: Positive number of leading coordinates used by the loss.
        temperature: Positive finite contrastive temperature.
        low_resolution: Adapt timm encoders for low-resolution images.
        pretrained: Load pretrained timm encoder weights.

    Note:
        Training uses two views under ``batch["views"]``. Single-view evaluation
        returns the full embedding for downstream callbacks.
    """

    def __init__(
        self,
        encoder_name: str | nn.Module = "resnet50",
        subspace_dim: int = 360,
        temperature: float = 0.1,
        low_resolution: bool = False,
        pretrained: bool = False,
    ) -> None:
        if isinstance(subspace_dim, bool) or not isinstance(subspace_dim, int):
            raise ValueError("subspace_dim must be a positive integer")
        if subspace_dim < 1:
            raise ValueError("subspace_dim must be a positive integer")
        if not isinstance(encoder_name, (str, nn.Module)):
            raise TypeError("encoder_name must be a timm name or torch.nn.Module")
        super().__init__()
        self.directclr_loss = CLIPLoss(temperature)
        self.backbone = (
            from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
            if isinstance(encoder_name, str)
            else encoder_name
        )
        tensor = next(
            (
                t
                for t in chain(self.backbone.parameters(), self.backbone.buffers())
                if t.is_floating_point()
            ),
            torch.empty(0),
        )
        image_size = getattr(
            getattr(self.backbone, "patch_embed", None), "img_size", (224, 224)
        )
        if low_resolution:
            image_size = (32, 32)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        states = [(module, module.training) for module in self.backbone.modules()]
        # Shape discovery must not update BN buffers or consume the caller's RNG.
        devices = [tensor.device.index] if tensor.device.type == "cuda" else []
        try:
            self.backbone.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                features = self.backbone(
                    torch.zeros(
                        1, 3, *image_size, device=tensor.device, dtype=tensor.dtype
                    )
                )
        finally:
            for module, training in states:
                module.training = training
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("encoder must return flat [B, D] features")
        self.embed_dim = features.shape[1]
        if not subspace_dim <= self.embed_dim:
            raise ValueError("subspace_dim must not exceed the encoder feature width")
        self.subspace_dim = subspace_dim

    def forward(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
        """Run the shared DirectCLR forward for a training or evaluation batch.

        Args:
            batch: Two training views or a single evaluation image batch.
            stage: Stage name supplied by the training manager.

        Returns:
            Full embeddings and, during training, the loss and selected subspace.
        """
        return directclr(self, batch, stage)
