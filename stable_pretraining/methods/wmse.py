"""W-MSE: Whitening Mean-Squared Error.

Joint-embedding SSL where projections are batch-whitened (Cholesky) and
the loss is the MSE between the whitened projections of the two views.
Whitening removes second-order redundancy without negatives or stop-gradient.

References:
    Ermolov et al. "Whitening for Self-Supervised Representation Learning."
    ICML 2021. https://arxiv.org/abs/2007.06346
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm


@dataclass
class WMSEOutput(ModelOutput):
    """Structured output of the :class:`WMSE` SSL method."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


def _eigen_whiten(z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """ZCA-style whiten ``[B, D]``.

    Uses the symmetric eigendecomposition of the (B-centred) covariance.
    Forced into fp32 (``eigh`` has no fp16 kernel).
    """
    orig_dtype = z.dtype
    # Disable autocast so the float() cast actually sticks.
    with torch.amp.autocast(device_type=z.device.type, enabled=False):
        z32 = z.float()
        z32 = z32 - z32.mean(dim=0, keepdim=True)
        cov = (z32.T @ z32) / max(z32.shape[0] - 1, 1)
        cov = (cov + cov.T) * 0.5
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp(min=eps)
        inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
        out = z32 @ inv_sqrt
    return out.to(orig_dtype)


# Backward-compat alias
_cholesky_whiten = _eigen_whiten


class WMSE(Module):
    """W-MSE: whitening + MSE between paired views.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: ``(hidden, output)`` for the projector
        (default ``(1024, 64)``; a small whitening dim helps stability).
    :param eps: Cholesky regularisation (default ``1e-3``).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (1024, 64),
        eps: float = 1e-3,
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

        proj_hidden, proj_out = projector_dims
        self.projector = _projector(embed_dim, proj_hidden, proj_out)
        self.eps = eps

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> WMSEOutput:
        if view2 is None:
            embedding = self.backbone(view1)
            return WMSEOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # Whiten the *concatenated* batch jointly so both views see the same
        # statistics (matches the paper).
        z = torch.cat([z1, z2], dim=0)
        zw = _cholesky_whiten(z, eps=self.eps)
        zw = F.normalize(zw, dim=-1)
        zw1, zw2 = zw.chunk(2, dim=0)

        loss = F.mse_loss(zw1, zw2)

        return WMSEOutput(
            loss=loss,
            embedding=torch.cat([h1, h2], dim=0),
            projection=z,
        )
