"""PIRL: Pretext-Invariant Representation Learning.

NCE-based memory-bank method that trains the encoder so that an image and
its *jigsaw-shuffled* version share the same representation. The original
image embedding goes into the memory bank; the shuffled-patch embedding is
the query and matched against the bank as positive.

References:
    Misra, van der Maaten. "Self-Supervised Learning of Pretext-Invariant
    Representations." CVPR 2020. https://arxiv.org/abs/1912.01991
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.callbacks.queues import UnsortedQueue


@dataclass
class PIRLOutput(ModelOutput):
    loss: torch.Tensor = None
    embedding: torch.Tensor = None


def _shuffle_patches(images: torch.Tensor, grid: int = 4) -> torch.Tensor:
    """Crop each image into a ``grid x grid`` jigsaw and randomly permute the
    patches. The image is resized so its side length is divisible by ``grid``
    and then resized back, so the encoder always sees its expected size.
    """
    B, C, H, W = images.shape
    if H != W:
        raise ValueError("PIRL jigsaw assumes square images")
    target_h = (H // grid) * grid
    if target_h != H:
        images_resized = F.interpolate(images, size=target_h, mode="bilinear", align_corners=False)
    else:
        images_resized = images
    p = target_h // grid
    x = images_resized.unfold(2, p, p).unfold(3, p, p)  # [B, C, grid, grid, p, p]
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    x = x.view(B, grid * grid, C, p, p)
    perm = torch.stack([torch.randperm(grid * grid, device=images.device) for _ in range(B)])
    x = torch.gather(x, 1, perm[..., None, None, None].expand_as(x))
    x = x.view(B, grid, grid, C, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
    out = x.view(B, C, target_h, target_h)
    if target_h != H:
        out = F.interpolate(out, size=H, mode="bilinear", align_corners=False)
    return out


class PIRL(Module):
    """PIRL: jigsaw-invariant memory-bank SSL.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dim: Output projection dim (default 128).
    :param queue_length: Memory bank size (default 16384; paper used full
        dataset, but a queue works as an approximation).
    :param temperature: NCE temperature (default 0.07).
    :param lambda_pirl: Weight on the (jigsaw, original) loss vs (jigsaw,
        shuffled-elsewhere) (default 0.5).
    :param jigsaw_grid: Grid size for the jigsaw transform (default 3).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dim: int = 128,
        queue_length: int = 16384,
        temperature: float = 0.07,
        lambda_pirl: float = 0.5,
        jigsaw_grid: int = 4,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()
        if isinstance(encoder_name, str):
            self.backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            self.backbone = encoder_name

        with torch.no_grad():
            embed_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.lambda_pirl = lambda_pirl
        self.jigsaw_grid = jigsaw_grid

        # Two projection heads: one for original images, one for the jigsaw.
        self.proj_image = nn.Linear(embed_dim, projector_dim, bias=False)
        self.proj_jigsaw = nn.Linear(embed_dim, projector_dim, bias=False)

        self.queue = UnsortedQueue(
            max_length=queue_length, shape=(projector_dim,), dtype=torch.float32
        )

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> PIRLOutput:
        # Eval / single-image
        if view2 is None:
            embedding = self.backbone(view1)
            return PIRLOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
            )

        # ``view1`` is treated as the original image; ``view2`` is ignored
        # and replaced by a jigsaw of view1 (PIRL is single-augmentation).
        jigsaw = _shuffle_patches(view1, grid=self.jigsaw_grid)

        h_img = self.backbone(view1)
        h_jig = self.backbone(jigsaw)
        z_img = F.normalize(self.proj_image(h_img), dim=-1)
        z_jig = F.normalize(self.proj_jigsaw(h_jig), dim=-1)

        with torch.no_grad():
            queue_keys = self.queue.append(z_img.detach().to(torch.float32)).to(z_img.dtype)

        # NCE 1: jigsaw vs (positive=image, negatives=queue)
        pos1 = (z_jig * z_img.detach()).sum(dim=-1, keepdim=True)
        neg1 = z_jig @ queue_keys.T
        logits1 = torch.cat([pos1, neg1], dim=1) / self.temperature
        tgt = torch.zeros(z_jig.shape[0], dtype=torch.long, device=z_jig.device)
        loss1 = F.cross_entropy(logits1, tgt)

        # NCE 2: image vs (positive=image-itself in queue, negatives=other queue)
        pos2 = (z_img * z_img.detach()).sum(dim=-1, keepdim=True)
        neg2 = z_img @ queue_keys.T
        logits2 = torch.cat([pos2, neg2], dim=1) / self.temperature
        loss2 = F.cross_entropy(logits2, tgt)

        loss = self.lambda_pirl * loss1 + (1 - self.lambda_pirl) * loss2

        return PIRLOutput(loss=loss, embedding=h_img)
