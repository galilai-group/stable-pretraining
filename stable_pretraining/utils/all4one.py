"""Components for All4One, composed from the library's existing building blocks.

Reference: https://github.com/ImaGonEs/all4one/blob/main/solo/methods/all4one.py
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..backbone.mlp import MLP
from ..backbone.pos_embed import get_1d_sincos_pos_embed
from ..callbacks.queues import UnsortedQueue
from .distributed import all_gather


def _positive_dimension(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def build_all4one_projector(
    input_dim: int, hidden_dim: int = 2048, projection_dim: int = 256
) -> nn.Sequential:
    """Build the released three-layer projector, including final batch normalization.

    Args:
        input_dim: Backbone feature dimension.
        hidden_dim: Width of both hidden layers.
        projection_dim: Output dimension.

    Returns:
        Projector with biased linear layers, batch normalization, and ReLU.
    """
    for name, value in (
        ("input_dim", input_dim),
        ("hidden_dim", hidden_dim),
        ("projection_dim", projection_dim),
    ):
        _positive_dimension(name, value)
    return nn.Sequential(
        MLP(
            input_dim,
            [hidden_dim, hidden_dim, projection_dim],
            norm_layer="batch_norm",
        ),
        nn.BatchNorm1d(projection_dim),
    )


def build_all4one_predictor(projection_dim: int = 256, hidden_dim: int = 4096) -> MLP:
    """Build a predictor for neighbour contrast or shifted attention inputs.

    Args:
        projection_dim: Input and output dimension.
        hidden_dim: Width of the batch-normalized hidden layer.

    Returns:
        Two-layer MLP with biased linear layers and no output normalization.
    """
    _positive_dimension("projection_dim", projection_dim)
    _positive_dimension("hidden_dim", hidden_dim)
    return MLP(projection_dim, [hidden_dim, projection_dim], norm_layer="batch_norm")


class All4OneQueue(nn.Module):
    """Retrieve sorted neighbours from a randomly initialized support queue.

    Args:
        projection_dim: Dimension of each stored momentum projection.
        queue_size: Number of stored projections. Default is 98304.
        num_neighbours: Number of neighbours retrieved per query. Default is 5.

    Note:
        Retrieval never mutates the queue. Retrieve both views before calling
        ``enqueue`` with the first view's momentum projections. The queue starts
        full of normalized random vectors, matching the released implementation.
        Storage and checkpointing use ``UnsortedQueue``, including its support
        for wraparound and batches larger than the capacity.

        Enqueue gathers equal-sized local batches in rank order. Distributed
        training must synchronize initial buffers (as Lightning DDP does).
    """

    def __init__(
        self,
        projection_dim: int = 256,
        queue_size: int = 98304,
        num_neighbours: int = 5,
    ) -> None:
        super().__init__()
        for name, value in (
            ("projection_dim", projection_dim),
            ("queue_size", queue_size),
            ("num_neighbours", num_neighbours),
        ):
            _positive_dimension(name, value)
        if num_neighbours > queue_size:
            raise ValueError("num_neighbours must not exceed queue_size")
        self.projection_dim = projection_dim
        self.num_neighbours = num_neighbours
        self.queue = UnsortedQueue(
            queue_size, shape=projection_dim, dtype=torch.float32
        )
        self.queue.append(F.normalize(torch.randn(queue_size, projection_dim), dim=-1))

    def _validate(self, projections: torch.Tensor) -> None:
        if projections.ndim != 2 or projections.shape[1] != self.projection_dim:
            raise ValueError(
                f"projections must have shape (batch, {self.projection_dim})"
            )
        if not projections.is_floating_point():
            raise ValueError("projections must be floating-point tensors")

    @torch.no_grad()
    def forward(self, projections: torch.Tensor) -> dict[str, torch.Tensor]:
        """Find neighbours without inserting the current batch.

        Args:
            projections: Query projections of shape ``(batch, projection_dim)``.

        Returns:
            Dict with detached ``neighbours`` of shape ``(batch, k, dim)`` in
            the query dtype and ``indices`` of shape ``(batch, k)`` identifying
            queue slots, ordered by decreasing cosine similarity. Returned
            values remain valid if the queue is updated before backward.
        """
        self._validate(projections)
        support = self.queue.get()
        # Keep neighbour selection out of autocast to avoid reduced-precision ties.
        with torch.autocast(device_type=projections.device.type, enabled=False):
            dtype = torch.float64 if support.dtype == torch.float64 else torch.float32
            query = F.normalize(projections.to(dtype), dim=-1)
            indices = (
                (query @ support.to(dtype).T).topk(self.num_neighbours, dim=1).indices
            )
        return {
            "neighbours": support[indices].to(projections.dtype),
            "indices": indices,
        }

    @torch.no_grad()
    def enqueue(self, projections: torch.Tensor) -> None:
        """Gather, normalize, and store momentum projections without gradients.

        Args:
            projections: First-view momentum projections of shape ``(batch, dim)``.
                All ranks must call with equal local batch sizes.
        """
        self._validate(projections)
        dtype = (
            torch.float64 if self.queue.out.dtype == torch.float64 else torch.float32
        )
        projections = F.normalize(projections.detach().to(dtype), dim=-1)
        gathered = torch.cat(all_gather(projections), dim=0)
        self.queue.append(gathered.to(self.queue.out.dtype))


class All4OneAttention(nn.Module):
    """Encode a neighbour sequence or its prediction-shifted counterpart.

    Args:
        projection_dim: Token dimension; must be divisible by ``num_heads``.
        num_neighbours: Sequence length. Default is 5.
        num_heads: Attention heads per layer. Default is 8.
        num_layers: Number of transformer encoder layers. Default is 3.
        dropout: Dropout probability throughout the transformer. Default is 0.1.

    Note:
        Matches the released ``PositionalEncodingPermute1D(k)`` convention:
        feature coordinates are positions and neighbour slots are channels.
        For odd ``k``, the frequency denominator is rounded up to the next even
        integer. The existing sinusoidal helper supplies the encoding without
        requiring the external ``positional_encodings`` package.

        Both centroid branches train this shared transformer. Retrieved
        neighbours are already detached by ``All4OneQueue``; predictions retain
        their gradients through the shift operation.
    """

    def __init__(
        self,
        projection_dim: int = 256,
        num_neighbours: int = 5,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("projection_dim", projection_dim),
            ("num_neighbours", num_neighbours),
            ("num_heads", num_heads),
            ("num_layers", num_layers),
        ):
            _positive_dimension(name, value)
        if projection_dim % num_heads:
            raise ValueError("projection_dim must be divisible by num_heads")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        self.projection_dim = projection_dim
        self.num_neighbours = num_neighbours
        even_channels = num_neighbours + num_neighbours % 2
        position = get_1d_sincos_pos_embed(even_channels, projection_dim)
        self.register_buffer(
            "position",
            position[:, :num_neighbours].T.unsqueeze(0).contiguous(),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=num_heads,
            dim_feedforward=2 * projection_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self, neighbours: torch.Tensor, prediction: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the first output token as the centroid representation.

        Args:
            neighbours: Sorted neighbours of shape ``(batch, k, dim)``.
            prediction: Optional ``(batch, dim)`` prediction. When provided,
                prepend it and drop the last neighbour before adding positions.

        Returns:
            Centroid of shape ``(batch, dim)``, with gradients through the
            transformer and any supplied prediction.
        """
        if neighbours.ndim != 3 or neighbours.shape[1:] != (
            self.num_neighbours,
            self.projection_dim,
        ):
            raise ValueError(
                f"neighbours must have shape (batch, {self.num_neighbours}, "
                f"{self.projection_dim})"
            )
        if prediction is not None:
            if prediction.shape != (neighbours.shape[0], self.projection_dim):
                raise ValueError("prediction must have shape (batch, projection_dim)")
            neighbours = torch.cat((prediction.unsqueeze(1), neighbours[:, :-1]), dim=1)
        return self.encoder(neighbours + self.position.to(neighbours.dtype))[:, 0]
