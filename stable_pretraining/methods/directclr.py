"""DirectCLR: contrastive learning on a fixed encoder subspace.

Reference: Jing et al., ICLR 2022. https://arxiv.org/abs/2110.09348
"""

from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn
from transformers.utils import ModelOutput

from ..backbone import from_timm
from ..losses import DirectCLRLoss
from ..module import Module


@dataclass
class DirectCLROutput(ModelOutput):
    """Outputs of the tensor-based DirectCLR interface.

    Attributes:
        loss: Scalar contrastive loss, or zero for a single view.
        embedding: Full encoder features, ``[2B, D]`` for two views or ``[B, D]``.
        projection: Leading feature coordinates ``[2B, subspace_dim]``, or None.
    """

    loss: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    projection: torch.Tensor | None = None


class DirectCLR(Module):
    """Apply symmetric cross-view InfoNCE to fixed leading feature coordinates.

    Args:
        encoder_name: timm model name or an image encoder returning flat features.
        subspace_dim: Positive number of leading coordinates used by the loss.
        temperature: Positive finite contrastive temperature.
        low_resolution: Adapt timm encoders for low-resolution images.
        pretrained: Load pretrained timm encoder weights.

    Note:
        Single-view calls return complete features and zero loss. Call ``eval()``
        and use ``torch.no_grad()`` for frozen evaluation. Training through the
        library's dict-based Manager uses ``Module(forward=forward.directclr)``.
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
        self.directclr_loss = DirectCLRLoss(temperature)
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

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor | None = None
    ) -> DirectCLROutput:
        """Return full features and, for two views, a fixed-subspace loss.

        Args:
            view1: First image view, or the only view during evaluation.
            view2: Optional aligned second image view.

        Returns:
            Full embeddings, selected coordinates and scalar loss. Single-view
            output has ``projection=None`` and a zero loss on the feature device.
        """
        first = self.backbone(view1)
        if view2 is None:
            return DirectCLROutput(
                loss=first.new_zeros(()), embedding=first, projection=None
            )
        second = self.backbone(view2)
        z1, z2 = first[:, : self.subspace_dim], second[:, : self.subspace_dim]
        return DirectCLROutput(
            loss=self.directclr_loss(z1, z2),
            embedding=torch.cat((first, second)),
            projection=torch.cat((z1, z2)),
        )
