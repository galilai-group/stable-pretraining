"""TS2Vec time-series representation encoder."""

import torch
from torch import nn
from torch.nn import functional as F


class _SamePadConv1d(nn.Module):
    """A dilated 1D convolution that preserves sequence length."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        receptive_field = 2 * dilation + 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=receptive_field // 2,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the padded convolution."""
        return self.conv(x)


class _ResidualBlock(nn.Module):
    """Two-layer dilated convolution block with a residual projection."""

    def __init__(
        self, in_channels: int, out_channels: int, dilation: int, final: bool
    ) -> None:
        super().__init__()
        self.conv1 = _SamePadConv1d(in_channels, out_channels, dilation)
        self.conv2 = _SamePadConv1d(out_channels, out_channels, dilation)
        self.projector = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block."""
        residual = x if self.projector is None else self.projector(x)
        out = self.conv1(F.gelu(x))
        out = self.conv2(F.gelu(out))
        return out + residual


class TS2VecEncoder(nn.Module):
    """Residual dilated-convolution encoder for multivariate time series.

    Args:
        input_dims: Number of input features per timestamp.
        output_dims: Number of output representation features.
        hidden_dims: Number of channels in hidden residual blocks.
        depth: Number of hidden residual blocks before the output block.

    The forward method accepts ``[batch, time, features]`` input and returns
    ``[batch, time, output_dims]`` representations. Missing timestamps marked
    by NaNs are zero-filled and excluded from the timestamp mask.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_dims", input_dims),
            ("output_dims", output_dims),
            ("hidden_dims", hidden_dims),
            ("depth", depth),
        ):
            minimum = 0 if name == "depth" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        channels = [hidden_dims] * depth + [output_dims]
        blocks = []
        in_channels = hidden_dims
        for index, out_channels in enumerate(channels):
            blocks.append(
                _ResidualBlock(
                    in_channels,
                    out_channels,
                    dilation=2**index,
                    final=index == len(channels) - 1,
                )
            )
            in_channels = out_channels
        self.feature_extractor = nn.Sequential(*blocks)
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        """Encode a batch of time series.

        Args:
            x: Input tensor with shape ``[batch, time, input_dims]``.
            mask: Optional boolean tensor with shape ``[batch, time]``, or one
                of ``"all_true"``, ``"all_false"``, and ``"binomial"``.
                When omitted, binomial masking is used in training and all
                timestamps are retained during evaluation.

        Returns:
            Tensor of shape ``[batch, time, output_dims]``.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, features].")
        batch, length, features = x.shape
        if batch < 1 or length < 1 or features != self.input_dims:
            raise ValueError(
                f"x must have non-empty batch/time dimensions and {self.input_dims} features."
            )
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype.")

        nan_valid = ~torch.isnan(x).any(dim=-1)
        clean_x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        hidden = self.input_fc(clean_x)

        if mask is None:
            mask = "binomial" if self.training else "all_true"
        if isinstance(mask, str):
            if mask == "all_true":
                timestamp_mask = torch.ones(
                    (batch, length), dtype=torch.bool, device=x.device
                )
            elif mask == "all_false":
                timestamp_mask = torch.zeros(
                    (batch, length), dtype=torch.bool, device=x.device
                )
            elif mask == "binomial":
                timestamp_mask = torch.rand((batch, length), device=x.device) < 0.5
            else:
                raise ValueError(
                    "mask must be a boolean tensor or 'all_true', 'all_false', or 'binomial'."
                )
        elif isinstance(mask, torch.Tensor):
            if mask.dtype != torch.bool or tuple(mask.shape) != (batch, length):
                raise ValueError(
                    "Tensor mask must be boolean with shape [batch, time]."
                )
            timestamp_mask = mask.to(device=x.device)
        else:
            raise TypeError("mask must be a string, boolean tensor, or None.")

        timestamp_mask = timestamp_mask & nan_valid
        hidden = hidden.masked_fill(~timestamp_mask.unsqueeze(-1), 0)
        hidden = hidden.transpose(1, 2)
        encoded = self.repr_dropout(self.feature_extractor(hidden))
        return encoded.transpose(1, 2)
