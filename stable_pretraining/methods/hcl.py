"""HCL: Contrastive Learning with Hard Negative Samples.

SimCLR-style contrastive learning in which the negatives of each anchor are
reweighted toward *hard* negatives (those most similar to the anchor) with a
concentration parameter ``beta``, combined with the debiasing correction
``tau_plus`` of Debiased Contrastive Learning. No labels are used: hardness is
estimated from the embedding similarities alone.

Special cases:

- ``beta=0, tau_plus=0`` recovers NT-Xent (SimCLR) exactly.
- ``beta=0, tau_plus>0`` recovers Debiased Contrastive Learning.

References:
    Robinson et al. "Contrastive Learning with Hard Negative Samples."
    ICLR 2021. https://arxiv.org/abs/2010.04592
    (official code: https://github.com/joshr17/HCL)

    Chuang et al. "Debiased Contrastive Learning." NeurIPS 2020.
    https://arxiv.org/abs/2007.00224

Example::

    from stable_pretraining.methods import HCL
    import lightning as pl

    model = HCL(encoder_name="resnet18", low_resolution=True)

    trainer = pl.Trainer(max_epochs=200)
    trainer.fit(model, dataloader)

``HCLLoss`` is also a drop-in replacement for ``NTXEntLoss`` in the
composable ``forward.simclr`` pipeline::

    import stable_pretraining as spt
    from stable_pretraining.forward import simclr
    from stable_pretraining.methods.hcl import HCLLoss

    module = spt.Module(
        forward=simclr,
        backbone=backbone,
        projector=projector,
        simclr_loss=HCLLoss(temperature=0.5, beta=1.0, tau_plus=0.1),
    )
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.utils import all_gather, get_rank


class HCLLoss(nn.Module):
    r"""Hard negative contrastive loss (Robinson et al., ICLR 2021).

    For anchor :math:`i` with positive :math:`p(i)`, similarities
    :math:`s_{ik} = z_i^\top z_k / t` of L2-normalised projections and
    :math:`N` negatives :math:`k`:

    .. math::

        R_i = N \frac{\sum_k e^{(1+\beta) s_{ik}}}{\sum_k e^{\beta s_{ik}}},
        \qquad
        N_g = \max\left(\frac{R_i - \tau^+ N e^{s_{ip(i)}}}{1 - \tau^+},\;
        N e^{-1/t}\right),

        \ell_i = -\log \frac{e^{s_{ip(i)}}}{e^{s_{ip(i)}} + N_g}.

    This matches ``criterion(..., estimator="hard")`` of the official
    implementation, but is evaluated with ``logsumexp`` and a per-row shift so
    that it cannot overflow at low temperatures (the official code exponentiates
    ``s / t`` directly).

    Under DDP the negatives span all ranks, exactly as in
    :class:`~stable_pretraining.losses.NTXEntLoss`.

    :param temperature: Softmax temperature :math:`t` (0.5 in the paper).
    :param beta: Hardness concentration :math:`\beta \ge 0`; larger values
        focus on harder negatives, 0 weights all negatives uniformly.
    :param tau_plus: Class prior :math:`\tau^+ \in [0, 1)`: the assumed
        probability that a random negative shares the anchor's (unknown) class.
        0 disables debiasing.
    """

    def __init__(
        self, temperature: float = 0.5, beta: float = 1.0, tau_plus: float = 0.1
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if not 0 <= tau_plus < 1:
            raise ValueError(f"tau_plus must be in [0, 1), got {tau_plus}")
        self.temperature = temperature
        self.beta = beta
        self.tau_plus = tau_plus

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute the HCL loss for two batches of paired views.

        :param z_i: Projections of the first view, shape ``[N, D]``.
        :param z_j: Projections of the second view, shape ``[N, D]``.
        :return: Scalar loss averaged over all ``2N`` anchors.
        """
        if z_i.ndim != 2 or z_i.shape != z_j.shape:
            raise ValueError(
                "z_i and z_j must both have shape [N, D], got "
                f"{tuple(z_i.shape)} and {tuple(z_j.shape)}"
            )

        anchors = F.normalize(torch.cat([z_i, z_j], dim=0), dim=-1)
        candidates = torch.cat(all_gather(anchors), dim=0)

        two_n = anchors.size(0)
        n = z_i.size(0)
        num_neg = candidates.size(0) - 2
        if num_neg < 1:
            raise ValueError("HCLLoss needs at least 2 samples to form negatives")

        offset = get_rank() * two_n
        local_idx = torch.arange(two_n, device=anchors.device)
        self_idx = offset + local_idx
        pos_idx = offset + (local_idx + n) % two_n

        sim = anchors @ candidates.T / self.temperature
        pos = sim[local_idx, pos_idx]

        not_neg = torch.zeros_like(sim, dtype=torch.bool)
        not_neg[local_idx, self_idx] = True
        not_neg[local_idx, pos_idx] = True

        # Masking must happen after scaling: with beta=0, 0 * -inf would be NaN.
        log_reweighted_neg = (
            math.log(num_neg)
            + torch.logsumexp(
                ((1 + self.beta) * sim).masked_fill(not_neg, -torch.inf), dim=-1
            )
            - torch.logsumexp(
                (self.beta * sim).masked_fill(not_neg, -torch.inf), dim=-1
            )
        )

        # The loss is invariant to the shift, so it can be treated as a constant.
        shift = torch.maximum(pos, log_reweighted_neg).detach()
        pos_exp = torch.exp(pos - shift)
        neg_exp = torch.exp(log_reweighted_neg - shift)
        n_g = (neg_exp - self.tau_plus * num_neg * pos_exp) / (1 - self.tau_plus)
        n_g = torch.maximum(n_g, num_neg * torch.exp(-1 / self.temperature - shift))

        return (torch.log(pos_exp + n_g) + shift - pos).mean()


@dataclass
class HCLOutput(ModelOutput):
    """Output from HCL forward pass.

    :ivar loss: HCL contrastive loss (0 in eval mode)
    :ivar embedding: Backbone features [B, D] in eval mode, [2B, D] in train mode
    :ivar projection: Projector outputs [2B, P] (None in eval mode)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


def _build_projector(in_dim: int, dims: Sequence[int]) -> nn.Module:
    """Projector from the official HCL code: [Linear(no bias) -> BN -> ReLU]* -> Linear.

    :param in_dim: Backbone output dimension
    :param dims: Hidden + output dimensions, e.g. ``(512, 128)`` (official)
    """
    if len(dims) < 1:
        raise ValueError(
            "projector_dims must contain at least one entry (the output dim)"
        )
    layers = []
    prev = in_dim
    for dim in dims[:-1]:
        layers += [
            nn.Linear(prev, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        ]
        prev = dim
    layers.append(nn.Linear(prev, dims[-1], bias=True))
    return nn.Sequential(*layers)


class HCL(Module):
    """HCL: contrastive learning with hard negative sampling.

    Architecture:
        - **Backbone**: any feature extractor producing a flat [B, D] embedding
          (timm ViT/ResNet with the head removed)
        - **Projector**: MLP ``Linear -> BN -> ReLU -> Linear`` (official HCL head)
        - **Loss**: :class:`HCLLoss` (hardness-reweighted, debiased NT-Xent)

    :param encoder_name: timm model name (e.g. ``"resnet50"``,
        ``"vit_small_patch16_224"``) or a pre-instantiated ``nn.Module`` whose
        ``forward`` returns a ``[B, D]`` tensor.
    :param projector_dims: Hidden + output dimensions of the projector.
        ``(512, 128)`` matches the official implementation.
    :param temperature: Softmax temperature (0.5 in the paper).
    :param beta: Hardness concentration (1.0 is the official default).
    :param tau_plus: Class prior for debiasing (0.1 is the official default).
    :param low_resolution: Adapt first conv for 32x32 inputs (CIFAR-style).
    :param pretrained: Load pretrained timm weights for the encoder.

    Example::

        model = HCL(encoder_name="resnet18", low_resolution=True)

        v1 = torch.randn(64, 3, 32, 32)
        v2 = torch.randn(64, 3, 32, 32)
        out = model(v1, v2)
        out.loss.backward()

        # eval: single view, no loss
        model.eval()
        out = model(v1)
        features = out.embedding  # [64, embed_dim]
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "resnet50",
        projector_dims: Sequence[int] = (512, 128),
        temperature: float = 0.5,
        beta: float = 1.0,
        tau_plus: float = 0.1,
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
            dummy = torch.zeros(1, 3, 224, 224)
            embed_dim = self.backbone(dummy).shape[-1]
        self.embed_dim = embed_dim

        self.projector = _build_projector(embed_dim, list(projector_dims))
        self.hcl_loss = HCLLoss(temperature=temperature, beta=beta, tau_plus=tau_plus)

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> HCLOutput:
        """Forward pass.

        :param view1: First augmented view [B, C, H, W] (or single view at eval).
        :param view2: Second augmented view [B, C, H, W]. If ``None``, returns
            only the backbone embedding (eval mode).
        :return: :class:`HCLOutput`.
        """
        if view2 is None:
            embedding = self.backbone(view1)
            return HCLOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
                projection=None,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        loss = self.hcl_loss(z1, z2)
        return HCLOutput(
            loss=loss,
            embedding=torch.cat([h1, h2], dim=0),
            projection=torch.cat([z1, z2], dim=0),
        )
