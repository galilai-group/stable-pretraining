"""Non-contrastive vision-language prediction with per-modality SIGReg.

Reference: https://github.com/MLO-lab/LeVLJEPA (``forwards.py`` and
``utils/sigreg.py``). The distributed implementation uses differentiable
collectives and shared directions; no negatives or momentum teacher are used.
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.nn.functional import all_reduce


class _SlicedGaussianRegularizer(nn.Module):
    """Epps-Pulley statistic on synchronized random one-dimensional slices."""

    def __init__(
        self, num_projections: int, num_knots: int, distributed: bool, seed: int
    ) -> None:
        super().__init__()
        if num_projections < 1:
            raise ValueError("num_projections must be positive")
        if num_knots < 3 or num_knots % 2 != 1:
            raise ValueError("num_knots must be odd and at least 3")
        self.num_projections = num_projections
        self.distributed = distributed
        self.seed = seed
        t = torch.linspace(0, 3, num_knots)
        phi = torch.exp(-t.square() / 2)
        weights = torch.full_like(t, 6 / (num_knots - 1))
        weights[[0, -1]] /= 2
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        distributed = self.distributed and dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        # Rank-local RNG histories may differ; every rank must test the same slices.
        step = self.step.clone()
        if distributed:
            dist.broadcast(step, src=0)
        generator = torch.Generator(device=embedding.device)
        generator.manual_seed(self.seed + int(step))
        directions = torch.randn(
            embedding.shape[-1],
            self.num_projections,
            generator=generator,
            device=embedding.device,
            dtype=torch.float32,
        )
        directions = F.normalize(directions, dim=0)
        if self.training:
            self.step.copy_(step + 1)

        # Characteristic functions need fp32 even under bf16/fp16 autocast.
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            t, phi, weights = self.t.float(), self.phi.float(), self.weights.float()
            if self.t.dtype != torch.float32:
                # .half()/.bfloat16() round buffers before autocast is disabled.
                # Reconstruct quadrature constants rather than upcast rounded ones.
                t = torch.linspace(0, 3, self.t.numel(), device=embedding.device)
                phi = torch.exp(-t.square() / 2)
                weights = torch.full_like(t, 6 / (t.numel() - 1))
                weights[[0, -1]] /= 2
                weights *= phi
            projected = embedding.float() @ directions
            phase = projected.unsqueeze(-1) * t
            moments = torch.stack((phase.cos().sum(0), phase.sin().sum(0)))
            count = embedding.new_tensor(embedding.shape[0], dtype=torch.float32)
            if distributed:
                moments = all_reduce(moments, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            moments = moments / count
            error = (moments[0] - phi).square() + moments[1].square()
            # Match the reference's LOCAL-batch scaling for equal DDP batches.
            return (error @ weights).mean() * (count / world_size)


class LeVLJEPALoss(nn.Module):
    """Bidirectional stop-gradient prediction plus marginal Gaussian regularity.

    Args:
        lambda_vision: Vision SIGReg weight (default 0.02).
        lambda_text: Text SIGReg weight (default 0.02). Weights must be finite,
            nonnegative, and sum to less than one.
        alignment: ``"mse"`` (reference default) or ``"cosine"``.
        num_projections: Number of random SIGReg slices.
        num_knots: Odd number of quadrature knots on [0, 3].
        distributed: Estimate each marginal on the global distributed batch.
            With equal local batch sizes, scale by the local batch size as in
            the official implementation. DDP gradients average correctly.
        seed: Seed for checkpointed, reproducible slice sequences.

    Note:
        Prediction targets are detached here. Regularization targets are not:
        SIGReg must train both encoders. The estimator is evaluated in fp32.
    """

    def __init__(
        self,
        lambda_vision: float = 0.02,
        lambda_text: float = 0.02,
        alignment: str = "mse",
        num_projections: int = 4096,
        num_knots: int = 17,
        distributed: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if (
            not math.isfinite(lambda_vision)
            or not math.isfinite(lambda_text)
            or min(lambda_vision, lambda_text) < 0
            or lambda_vision + lambda_text >= 1
        ):
            raise ValueError(
                "SIGReg weights must be finite, nonnegative and sum to < 1"
            )
        if alignment not in {"mse", "cosine"}:
            raise ValueError("alignment must be 'mse' or 'cosine'")
        self.lambda_vision = lambda_vision
        self.lambda_text = lambda_text
        self.alignment = alignment
        self.vision_regularizer = _SlicedGaussianRegularizer(
            num_projections, num_knots, distributed, seed
        )
        self.text_regularizer = _SlicedGaussianRegularizer(
            num_projections, num_knots, distributed, seed + 1
        )

    def forward(
        self,
        image_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        image_prediction: torch.Tensor,
        text_prediction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the reference objective and its unweighted components.

        Args:
            image_embedding: Vision marginal, shape [B, Dv].
            text_embedding: Text marginal, shape [B, Dt].
            image_prediction: Prediction FROM vision TO text, shape [B, Dt].
            text_prediction: Prediction FROM text TO vision, shape [B, Dv].

        Returns:
            Dict with ``loss``, ``prediction_loss``, both directional losses,
            ``sigreg_vision`` and ``sigreg_text``.
        """
        tensors = (image_embedding, text_embedding, image_prediction, text_prediction)
        if any(t.ndim != 2 or t.shape[0] < 1 or t.shape[1] < 1 for t in tensors):
            raise ValueError(
                "embeddings and predictions must be nonempty [B, D] tensors"
            )
        if (
            image_embedding.shape != text_prediction.shape
            or text_embedding.shape != image_prediction.shape
            or image_embedding.shape[0] != text_embedding.shape[0]
        ):
            raise ValueError("predictions must match opposite-modality target shapes")
        with torch.autocast(device_type=image_embedding.device.type, enabled=False):
            image, text, image_pred, text_pred = (t.float() for t in tensors)
            if self.alignment == "mse":
                to_text = F.mse_loss(image_pred, text.detach())
                to_image = F.mse_loss(text_pred, image.detach())
            else:
                to_text = (
                    2
                    - 2
                    * (
                        F.normalize(image_pred, dim=-1)
                        * F.normalize(text.detach(), dim=-1)
                    ).sum(-1)
                ).mean()
                to_image = (
                    2
                    - 2
                    * (
                        F.normalize(text_pred, dim=-1)
                        * F.normalize(image.detach(), dim=-1)
                    ).sum(-1)
                ).mean()
            prediction = (to_text + to_image) / 2
            vision_reg = self.vision_regularizer(image)
            text_reg = self.text_regularizer(text)
            loss = (
                (1 - self.lambda_vision - self.lambda_text) * prediction
                + self.lambda_vision * vision_reg
                + self.lambda_text * text_reg
            )
        return {
            "loss": loss,
            "prediction_loss": prediction,
            "image_to_text_loss": to_text,
            "text_to_image_loss": to_image,
            "sigreg_vision": vision_reg,
            "sigreg_text": text_reg,
        }
