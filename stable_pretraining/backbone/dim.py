"""Critic networks for Deep InfoMax (DIM) :cite:`hjelm2019learningdeeprepresentationsmutual`."""

import torch
import torch.nn as nn


class DIMEmbedding(nn.Module):
    """Residual 1x1-convolution embedding used by the DIM encode-and-dot-product critic.

    Maps every location of a ``[B, C, H, W]`` feature map to an
    ``out_channels``-dimensional vector with a non-linear branch
    (``1x1 conv -> BN -> ReLU -> 1x1 conv``) plus a linear 1x1 shortcut,
    followed by LayerNorm over channels. A global feature vector is embedded
    by treating it as a ``[B, C, 1, 1]`` map.

    Args:
        in_channels: Number of input channels.
        out_channels: Embedding dimension.

    Note:
        Mirrors ``MI1x1ConvNet`` from the reference implementation
        (https://github.com/rdevon/DIM). When ``out_channels >= in_channels``
        the shortcut is initialized close to an identity map, so the
        embedding initially preserves the input features.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block_nonlinear = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        self.linear_shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.layer_norm = nn.LayerNorm(out_channels)

        if out_channels >= in_channels:
            with torch.no_grad():
                weight = self.linear_shortcut.weight
                weight.uniform_(-0.01, 0.01)
                idx = torch.arange(in_channels)
                weight[idx, idx, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed every location of a feature map.

        Args:
            x: Feature map of shape ``[B, in_channels, H, W]``.

        Returns:
            torch.Tensor: Embeddings of shape ``[B, out_channels, H, W]``.
        """
        h = self.block_nonlinear(x) + self.linear_shortcut(x)
        return self.layer_norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LocalGlobalCritic(nn.Module):
    """Encode-and-dot-product critic scoring global/local feature pairs for local DIM.

    The global feature of each image and every location of each local
    feature map are embedded into a shared space by two :class:`DIMEmbedding`
    networks; the score of a pair is the dot product of their embeddings.
    All ``B x B x (H * W)`` pairs of a batch are scored at once, which is the
    input expected by :class:`~stable_pretraining.losses.LocalDIMLoss`.

    Args:
        local_channels: Channels ``C`` of the local feature map.
        global_dim: Dimension ``G`` of the global feature vector.
        embed_dim: Dimension of the shared embedding space. Defaults to 2048,
            as in the reference implementation. When it equals ``global_dim``
            the global feature is used as is, as in the reference
            implementation.

    Example:
        ::

            critic = LocalGlobalCritic(local_channels=128, global_dim=64)
            scores = critic(feature_map, global_features)  # [B, B, H * W]
    """

    def __init__(self, local_channels: int, global_dim: int, embed_dim: int = 2048):
        super().__init__()
        self.local_net = DIMEmbedding(local_channels, embed_dim)
        if global_dim == embed_dim:
            self.global_net = nn.Identity()
        else:
            self.global_net = DIMEmbedding(global_dim, embed_dim)

    def forward(
        self, local_features: torch.Tensor, global_features: torch.Tensor
    ) -> torch.Tensor:
        """Score every (global, local location) pair in the batch.

        Args:
            local_features: Local feature map of shape ``[B, C, H, W]``.
            global_features: Global features of shape ``[B, G]``.

        Returns:
            torch.Tensor: Scores of shape ``[B, B, H * W]`` where
            ``scores[a, b, i]`` compares the global feature of image ``a``
            with location ``i`` of the local feature map of image ``b``.

        Raises:
            ValueError: If the inputs are not ``[B, C, H, W]`` and ``[B, G]``
                with the same batch size.
        """
        if local_features.ndim != 4 or global_features.ndim != 2:
            raise ValueError(
                "expected local_features [B, C, H, W] and global_features [B, G], "
                f"got {tuple(local_features.shape)} and {tuple(global_features.shape)}"
            )
        if local_features.size(0) != global_features.size(0):
            raise ValueError(
                "local_features and global_features must share the batch size, "
                f"got {local_features.size(0)} and {global_features.size(0)}"
            )
        local_emb = self.local_net(local_features).flatten(2)
        global_emb = self.global_net(global_features[:, :, None, None]).flatten(1)
        return torch.einsum("ad,bdi->abi", global_emb, local_emb)
