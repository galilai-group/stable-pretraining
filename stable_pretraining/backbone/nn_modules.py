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


class IterNorm(torch.nn.Module):
    """Group-wise iterative whitening layer :cite:`huang2019iterative`.

    Splits the ``D`` features into ``num_groups`` contiguous groups and whitens
    each group over the batch. The inverse square root of each group covariance
    is approximated with ``T`` Newton-Schulz iterations instead of an
    eigendecomposition, so the layer stays cheap and differentiable. VIbCReg
    appends it to the end of the projector.

    Running estimates of the mean and whitening matrix are tracked during
    training and used in eval mode, mirroring :class:`torch.nn.BatchNorm1d`.

    Args:
        num_features: Feature dimension ``D``.
        num_groups: Number of groups. Must divide ``num_features``. Default 64.
        T: Number of Newton-Schulz iterations. Default 5.
        eps: Value added to the covariance diagonal. Default 1e-5.
        momentum: Momentum of the running statistics. Default 0.1.
        affine: If ``True``, apply a learnable per-feature scale and shift
            after whitening. Default ``True``.

    Example:
        ```python
        projector = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            IterNorm(2048, num_groups=64, T=5),
        )
        ```

    Note:
        Matches the solo-learn implementation used for the VIbCReg reference
        results, except that gradients come from autograd rather than a
        hand-written backward pass.
    """

    def __init__(
        self,
        num_features: int,
        num_groups: int = 64,
        T: int = 5,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        if num_features % num_groups != 0:
            raise ValueError(
                f"num_features ({num_features}) must be divisible by "
                f"num_groups ({num_groups})"
            )
        self.num_features = num_features
        self.num_groups = num_groups
        self.num_channels = num_features // num_groups
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer(
            "running_mean", torch.zeros(num_groups, self.num_channels, 1)
        )
        self.register_buffer(
            "running_wm",
            torch.eye(self.num_channels).repeat(num_groups, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Whiten a batch of features.

        Args:
            x: Input tensor [N, D]

        Returns:
            Whitened tensor [N, D], followed by the affine transform if enabled
        """
        if x.ndim != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"IterNorm expects input of shape [N, {self.num_features}], "
                f"got {tuple(x.shape)}"
            )
        # Newton-Schulz iterations diverge in half precision.
        dtype = torch.promote_types(x.dtype, torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self._whiten(x.to(dtype))
        if self.affine:
            out = out * self.weight + self.bias
        return out

    def _whiten(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        g, c = self.num_groups, self.num_channels
        xg = x.T.reshape(g, c, n)

        if self.training:
            mean = xg.mean(dim=-1, keepdim=True)
            xc = xg - mean
            eye = torch.eye(c, device=x.device, dtype=x.dtype).expand(g, c, c)
            sigma = torch.baddbmm(
                eye, xc, xc.transpose(1, 2), beta=self.eps, alpha=1.0 / n
            )
            # Dividing by the trace puts the spectrum in (0, 1], the region
            # where Newton-Schulz converges to Sigma^{-1/2}.
            r_tr = sigma.diagonal(dim1=1, dim2=2).sum(-1).reciprocal()[:, None, None]
            sigma_n = sigma * r_tr
            p = eye
            for _ in range(self.T):
                p = torch.baddbmm(
                    p,
                    torch.linalg.matrix_power(p, 3),
                    sigma_n,
                    beta=1.5,
                    alpha=-0.5,
                )
            wm = p * r_tr.sqrt()
            with torch.no_grad():
                self.running_mean.lerp_(mean.to(self.running_mean.dtype), self.momentum)
                self.running_wm.lerp_(wm.to(self.running_wm.dtype), self.momentum)
        else:
            xc = xg - self.running_mean.to(x.dtype)
            wm = self.running_wm.to(x.dtype)

        return (wm @ xc).reshape(g * c, n).T

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, num_groups={self.num_groups}, T={self.T}, "
            f"eps={self.eps}, momentum={self.momentum}, affine={self.affine}"
        )


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
