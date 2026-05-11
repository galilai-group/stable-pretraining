"""Reusable neural network layers used across backbones and methods."""

import numpy as np
import torch
import torch.nn.functional as F


class BatchNorm1dNoBias(torch.nn.BatchNorm1d):
    """BatchNorm1d with learnable scale but no learnable bias (center=False).

    This is used in contrastive learning methods like SimCLR where the final
    projection layer uses batch normalization with scale (gamma) but without
    bias (beta). This follows the original SimCLR implementation where the
    bias term is removed from the final BatchNorm layer.

    The bias is frozen at 0 and set to non-trainable, while the weight (scale)
    parameter remains learnable.

    Example:
        ```python
        # SimCLR-style projector
        projector = nn.Sequential(
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128, bias=False),
            BatchNorm1dNoBias(128),  # Final layer: no bias
        )
        ```

    Note:
        This is equivalent to TensorFlow's BatchNorm with center=False, scale=True.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False
        with torch.no_grad():
            self.bias.zero_()


class L2Norm(torch.nn.Module):
    """L2 normalization layer that normalizes input to unit length.

    Normalizes the input tensor along the last dimension to have unit L2 norm.
    Commonly used in DINO before the prototypes layer.

    Example:
        ```python
        projector = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 256),
            L2Norm(),  # Normalize to unit length
            nn.Linear(256, 4096, bias=False),  # Prototypes
        )
        ```
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to unit L2 norm.

        Args:
            x: Input tensor [..., D]

        Returns:
            L2-normalized tensor [..., D] where each D-dimensional vector has unit length
        """
        return F.normalize(x, dim=-1, p=2)


class Normalize(torch.nn.Module):
    """Normalize tensor and scale by square root of number of elements."""

    def forward(self, x):
        return F.normalize(x, dim=(0, 1, 2)) * np.sqrt(x.numel())


class ImageToVideoEncoder(torch.nn.Module):
    """Wrapper to apply an image encoder to video data by processing each frame independently.

    This module takes video data with shape (batch, time, channel, height, width) and applies
    an image encoder to each frame, returning the encoded features.

    Args:
        encoder (torch.nn.Module): The image encoder module to apply to each frame.
    """

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, video):
        # we expect something of the shape
        # (batch, time, channel, height, width)
        batch_size, num_timesteps = video.shape[:2]
        assert video.ndim == 5
        # (BxT)xCxHxW
        video = video.contiguous().flatten(0, 1)
        # (BxT)xF
        features = self.encoder(video)
        # BxTxF
        features = features.contiguous().view(
            batch_size, num_timesteps, features.size(1)
        )
        return features


class EMA(torch.nn.Module):
    """Exponential Moving Average module.

    Maintains an exponential moving average of input tensors.

    Args:
        alpha: Smoothing factor between 0 and 1.
               0 = no update (always return first value)
               1 = no smoothing (always return current value)
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.item = torch.nn.UninitializedBuffer()

    def forward(self, item):
        """Update EMA and return smoothed value.

        Args:
            item: New tensor to incorporate into the average

        Returns:
            Exponentially smoothed tensor
        """
        if self.alpha < 1 and isinstance(self.item, torch.nn.UninitializedBuffer):
            with torch.no_grad():
                self.item.materialize(
                    shape=item.shape, dtype=item.dtype, device=item.device
                )
                self.item.copy_(item, non_blocking=True)
            return item
        elif self.alpha == 1:
            return item
        with torch.no_grad():
            self.item.mul_(1 - self.alpha)
        output = item.mul(self.alpha).add(self.item)
        with torch.no_grad():
            self.item.copy_(output)
        return output

    @staticmethod
    def _test():
        q = EMA(0)
        R = torch.randn(10, 10)
        q(R)
        for i in range(10):
            v = q(torch.randn(10, 10))
            assert torch.allclose(v, R)
        q = EMA(1)
        R = torch.randn(10, 10)
        q(R)
        for i in range(10):
            R = torch.randn(10, 10)
            v = q(R)
            assert torch.allclose(v, R)

        q = EMA(0.5)
        R = torch.randn(10, 10)
        ground = R.detach()
        v = q(R)
        assert torch.allclose(ground, v)
        for i in range(10):
            R = torch.randn(10, 10)
            v = q(R)
            ground = R * 0.5 + ground * 0.5
            assert torch.allclose(v, ground)
        return True
