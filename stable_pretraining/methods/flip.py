"""FLIP: fast language-image pre-training through image-token masking.

References:
    Li et al. "Scaling Language-Image Pre-training via Masking." CVPR 2023.
    https://arxiv.org/abs/2212.00794

Example::

    import stable_pretraining as spt

    model = spt.FLIP(text_encoder=text_encoder, text_dim=512, mask_ratio=0.5)
    outputs = model({"image": images, "input_ids": tokens}, stage="fit")
    outputs["loss"].backward()

See ``examples/flip.py`` for a complete COCO training and retrieval example.
"""

import math
from typing import Any

import torch
from torch import nn

from stable_pretraining import Module
from stable_pretraining.backbone import MaskedEncoder, PatchMasking
from stable_pretraining.forward import _flip_image, _flip_text, flip
from stable_pretraining.losses import CLIPLoss


class FLIP(Module):
    """Train paired image and text encoders with FLIP :cite:`li2023scaling`.

    The image encoder processes randomly selected patches during training and
    all patches in evaluation. Its output tokens, including CLS, are averaged
    before a linear projection. Text encoding and tokenization are independent
    of FLIP: supply an encoder returning one pooled feature vector per caption.

    Args:
        text_encoder: Module accepting ``input_ids`` and, when provided,
            ``attention_mask`` as keyword arguments; returns a ``[B, text_dim]``
            tensor. The reference uses the first token of a bidirectional
            transformer, without an extra BERT pooler.
        text_dim: Width of the text encoder's pooled features.
        encoder_name: Standard timm ViT name or prebuilt timm ViT with absolute
            positional embeddings. A supplied model's classifier is removed.
        projection_dim: Shared image/text embedding width.
        mask_ratio: Fraction of image patches removed during training, in
            ``[0, 1)``. Zero supports unmasked training with gradients enabled.
        temperature: Initial contrastive temperature, in ``[0.01, 1]``.
        pretrained: Load pretrained image weights when using a model name.
        **kwargs: Passed to :class:`Module`, including ``optim`` and ``hparams``.

    Note:
        Call with a batch dictionary containing ``image`` and ``input_ids``.
        In evaluation either modality may be omitted. Use ``model.eval()`` for
        full-image retrieval; a stage name alone does not change module mode.
        Distributed training requires equal local batch sizes and paired rows
        in the same order on both modalities.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        text_dim: int,
        encoder_name: str | nn.Module = "vit_base_patch16_224",
        projection_dim: int = 512,
        mask_ratio: float = 0.5,
        temperature: float = 0.07,
        pretrained: bool = False,
        **kwargs: Any,
    ) -> None:
        if text_dim <= 0 or projection_dim <= 0:
            raise ValueError("text_dim and projection_dim must be positive")
        if not 0.01 <= temperature <= 1.0:
            raise ValueError("temperature must be in [0.01, 1]")
        masking = PatchMasking(mask_ratio=mask_ratio)
        super().__init__(forward=flip, **kwargs)
        self.image_encoder = MaskedEncoder(
            encoder_name, masking=masking, pretrained=pretrained
        )
        self.text_encoder = text_encoder
        self.image_projector = nn.Linear(
            self.image_encoder.embed_dim, projection_dim, bias=False
        )
        self.text_projector = nn.Linear(text_dim, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.clip_loss = CLIPLoss()
        self.save_hyperparameters(
            dict(
                text_dim=text_dim,
                projection_dim=projection_dim,
                mask_ratio=mask_ratio,
                temperature=temperature,
                pretrained=pretrained,
            )
        )
        if isinstance(encoder_name, str):
            self.save_hyperparameters({"encoder_name": encoder_name})

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return normalized projected image features, preserving gradient mode.

        Args:
            images: Images of shape ``[B, C, H, W]``. Masking follows the
                encoder's train/eval mode, just as in the paired forward pass.

        Returns:
            Image embeddings of shape ``[B, projection_dim]``.
        """
        return _flip_image(self, images)[1]

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return normalized projected text features, preserving gradient mode.

        Args:
            input_ids: Token IDs of shape ``[B, L]``.
            attention_mask: Optional padding mask forwarded to the text encoder.

        Returns:
            Text embeddings of shape ``[B, projection_dim]``.
        """
        return _flip_text(self, input_ids, attention_mask)
