"""Multi-layer perceptron backbone/projector for the JAX backend.

Flax-NNX port of :class:`stable_pretraining.backbone.mlp.MLP`. The layer
ordering matches the torch version (``Linear`` then activation for every hidden
layer, a final bare ``Linear``) so :func:`stable_pretraining.jax.utils.copy_torch_params_`
can transfer weights one-to-one for parity testing.
"""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    """Stack of ``Linear`` layers with an activation between hidden layers.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden + output dimensions, e.g. ``[2048, 128]``. The
            last entry is the output dimension.
        rngs: NNX RNG collection used to initialize parameters.
        activation: Activation applied after every layer except the last.
            Defaults to ``jax.nn.relu``.
        use_bias: Whether linear layers carry a bias. Default ``True``.

    Note:
        Norm layers and dropout from the torch ``MLP`` are intentionally
        omitted here (the parity path uses ``norm_layer=None, dropout=0``);
        they are added alongside the conv/BN backbones.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        *,
        rngs: nnx.Rngs,
        activation: Callable = jax.nn.relu,
        use_bias: bool = True,
        dtype=None,
    ):
        if len(hidden_channels) < 1:
            raise ValueError("hidden_channels must contain at least the output dim.")
        self.activation = activation
        dims = [in_channels, *hidden_channels]
        # ``dtype`` is the computation dtype (e.g. bfloat16); params stay f32.
        # nnx.List registers the contained modules as data (a plain list would
        # be treated as a static attribute and rejected by the pytree check).
        self.layers = nnx.List(
            [
                nnx.Linear(
                    dims[i], dims[i + 1], use_bias=use_bias, dtype=dtype, rngs=rngs
                )
                for i in range(len(hidden_channels))
            ]
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run the MLP.

        Args:
            x: Input array of shape ``[..., in_channels]``.

        Returns:
            jnp.ndarray: Output of shape ``[..., hidden_channels[-1]]``.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


__all__ = ["MLP"]
