"""FlatNCE with SimCLR's backbone, projector, and two-view interfaces."""

from typing import Any, Optional, Sequence, Union

import torch
from torch import nn

from stable_pretraining.forward import flatnce
from stable_pretraining.losses import FlatNCELoss

from .simclr import SimCLR, SimCLROutput


class FlatNCE(SimCLR):
    """Train SimCLR representations with negative-only FlatNCE normalization.

    Args:
        encoder_name: A timm model name or an encoder returning ``[B, D]``.
        projector_dims: Hidden and output dimensions of the SimCLR projector.
        temperature: Finite, strictly positive contrastive temperature.
        low_resolution: Adapt a timm encoder for CIFAR-style images.
        pretrained: Load pretrained timm encoder weights.
        optim: Optional Module optimizer/scheduler configuration. When omitted,
            uses Module's default AdamW configuration.

    Note:
        Dict batches use the composable forward function and support Manager,
        probes, and Lightning's inherited training/validation steps. Direct
        ``model(view1, view2)`` calls retain SimCLR's structured output; a
        single tensor returns embeddings without a contrastive loss.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 256),
        temperature: float = 0.5,
        low_resolution: bool = False,
        pretrained: bool = False,
        optim: Optional[dict[str, Any]] = None,
    ) -> None:
        loss = FlatNCELoss(temperature=temperature)
        super().__init__(
            encoder_name=encoder_name,
            projector_dims=projector_dims,
            temperature=temperature,
            low_resolution=low_resolution,
            pretrained=pretrained,
        )
        self.simclr_loss = loss
        if optim is not None:
            self.optim = optim

    @property
    def flatnce_loss(self) -> FlatNCELoss:
        """Expose the shared SimCLR loss slot under the FlatNCE forward contract."""
        return self.simclr_loss

    def forward(
        self,
        view1: Union[torch.Tensor, dict[str, Any]],
        view2: Optional[Union[torch.Tensor, str]] = None,
        *,
        stage: str = "fit",
    ) -> Union[SimCLROutput, dict[str, torch.Tensor]]:
        """Encode tensor views directly or process a Module training batch.

        Args:
            view1: First image tensor, or a dict batch from the data module.
            view2: Paired image tensor for direct two-view calls, or a
                positional stage string when a callback wraps dict forwarding.
            stage: Logging stage for dict batches.

        Returns:
            SimCLR's structured output for tensor inputs, or the FlatNCE
            embedding/label/loss dict for Module batch inputs.

        Raises:
            ValueError: If a second tensor is supplied alongside a dict batch.
        """
        if isinstance(view1, dict):
            # OnlineProbe wraps forwards as fn(batch, stage), passing the stage
            # positionally even when training_step supplied it by keyword.
            if isinstance(view2, str):
                stage = view2
            elif view2 is not None:
                raise ValueError("view2 must be omitted when passing a dict batch")
            return flatnce(self, view1, stage)
        return super().forward(view1, view2)
