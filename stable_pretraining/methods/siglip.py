"""SigLIP: Sigmoid Loss for Language-Image Pre-training.

Image-text pretraining where every image-text pair in the batch is scored as
an independent binary classification (match / no match) with a sigmoid loss,
instead of CLIP's batch-wide softmax.

References:
    Zhai et al. "Sigmoid Loss for Language Image Pre-Training." ICCV 2023.
    https://arxiv.org/abs/2303.15343

Example::

    from stable_pretraining.methods import SigLIP

    model = SigLIP()  # random-init ViT-B/16 image + text towers

    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(0, 32000, (8, 64))
    out = model(images=images, input_ids=input_ids)
    out.loss.backward()
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.forward import _pooled_embedding
from stable_pretraining.losses import SigLIPLoss


@dataclass
class SigLIPOutput(ModelOutput):
    """Output from SigLIP forward pass.

    :ivar loss: Pairwise sigmoid loss (0 unless both images and text are given)
    :ivar image_embeds: L2-normalized image embeddings [B, D] (None if no images)
    :ivar text_embeds: L2-normalized text embeddings [B, D] (None if no text)
    """

    loss: torch.Tensor = None
    image_embeds: Optional[torch.Tensor] = None
    text_embeds: Optional[torch.Tensor] = None


def _build_siglip_towers(model_name: Optional[str], pretrained: bool):
    """Return (vision_model, text_model, logit_scale, logit_bias) from Hugging Face.

    ``logit_scale`` / ``logit_bias`` are the checkpoint's learned values when
    ``pretrained`` is True, else ``None``.
    """
    from transformers import SiglipConfig, SiglipModel

    if pretrained:
        if model_name is None:
            raise ValueError("pretrained=True requires a model_name.")
        model = SiglipModel.from_pretrained(model_name)
        return (
            model.vision_model,
            model.text_model,
            float(model.logit_scale.detach().reshape(())),
            float(model.logit_bias.detach().reshape(())),
        )

    config = SiglipConfig.from_pretrained(model_name) if model_name else SiglipConfig()
    model = SiglipModel(config)
    return model.vision_model, model.text_model, None, None


class SigLIP(Module):
    """SigLIP: image-text pretraining with a pairwise sigmoid loss.

    Architecture:
        - **Vision tower**: image encoder producing a pooled [B, D] embedding
        - **Text tower**: text encoder producing a pooled [B, D] embedding
        - **Loss**: :class:`~stable_pretraining.losses.SigLIPLoss` with a
          learnable logit scale and bias

    :param model_name: Hugging Face SigLIP checkpoint (e.g.
        ``"google/siglip-base-patch16-224"``). With ``pretrained=False`` only
        its config is used (random init). If ``None``, uses the default
        ``SiglipConfig`` (ViT-B/16 at 224px, 64-token text), which needs no
        download.
    :param pretrained: Load the checkpoint's weights, including its learned
        logit scale and bias.
    :param vision_encoder: Custom image encoder. Must be passed together with
        ``text_encoder``; overrides ``model_name``. Called as
        ``vision_encoder(images)``.
    :param text_encoder: Custom text encoder, called as
        ``text_encoder(input_ids, attention_mask=...)``.
    :param init_logit_scale: Initial log-space logit scale (ignored when the
        values come from a pretrained checkpoint). Paper default ``log(10)``.
    :param init_logit_bias: Initial logit bias (ignored when loaded from a
        pretrained checkpoint). Paper default ``-10``.
    :param gather_distributed: Use negatives from all ranks under DDP.

    Example::

        model = SigLIP(model_name="google/siglip-base-patch16-224", pretrained=True)

        out = model(images=images, input_ids=input_ids)
        out.loss.backward()

        # eval: embed one modality, no loss
        model.eval()
        image_embeds = model(images=images).image_embeds
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        pretrained: bool = False,
        vision_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        init_logit_scale: float = math.log(10.0),
        init_logit_bias: float = -10.0,
        gather_distributed: bool = True,
    ):
        super().__init__()

        if (vision_encoder is None) != (text_encoder is None):
            raise ValueError("Pass both vision_encoder and text_encoder, or neither.")
        if pretrained and vision_encoder is not None:
            raise ValueError(
                "pretrained=True loads Hugging Face towers; it cannot be combined "
                "with custom vision_encoder/text_encoder."
            )

        if vision_encoder is not None:
            self.vision_model = vision_encoder
            self.text_model = text_encoder
        else:
            vision, text, scale, bias = _build_siglip_towers(model_name, pretrained)
            self.vision_model = vision
            self.text_model = text
            if scale is not None:
                init_logit_scale, init_logit_bias = scale, bias

        # Embeddings are normalized in forward(), so the loss skips it.
        self.siglip_loss = SigLIPLoss(
            init_logit_scale=init_logit_scale,
            init_logit_bias=init_logit_bias,
            normalize=False,
            gather_distributed=gather_distributed,
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """L2-normalized image embeddings [B, D]."""
        return F.normalize(
            _pooled_embedding(self.vision_model(images), "image_embeds"), dim=-1
        )

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """L2-normalized text embeddings [B, D]."""
        kwargs = {} if attention_mask is None else {"attention_mask": attention_mask}
        outputs = self.text_model(input_ids, **kwargs)
        return F.normalize(_pooled_embedding(outputs, "text_embeds"), dim=-1)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> SigLIPOutput:
        """Forward pass.

        :param images: Images [B, C, H, W].
        :param input_ids: Token ids [B, L], paired row-wise with ``images``.
        :param attention_mask: Optional text attention mask [B, L]. The original
            SigLIP checkpoints were trained with ``padding="max_length"`` and
            no mask.
        :return: :class:`SigLIPOutput`. ``loss`` is computed only when both
            ``images`` and ``input_ids`` are given.
        """
        if images is None and input_ids is None:
            raise ValueError("Pass images, input_ids, or both.")

        image_embeds = self.encode_image(images) if images is not None else None
        text_embeds = (
            self.encode_text(input_ids, attention_mask)
            if input_ids is not None
            else None
        )

        if image_embeds is not None and text_embeds is not None:
            loss = self.siglip_loss(image_embeds, text_embeds)
        else:
            ref = image_embeds if image_embeds is not None else text_embeds
            loss = torch.zeros((), device=ref.device, dtype=ref.dtype)

        return SigLIPOutput(
            loss=loss, image_embeds=image_embeds, text_embeds=text_embeds
        )
