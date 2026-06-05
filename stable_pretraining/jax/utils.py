"""Interop helpers for the JAX backend.

The headline utility is :func:`copy_torch_params_`, which copies parameters
from a torch module into a structurally matching Flax-NNX module. It is the
backbone of the torch/JAX *parity* regression tests (same weights + same input
must give the same output on both backends) and is also useful for loading
torch checkpoints into the JAX path.
"""

from typing import Any

import jax.numpy as jnp
from flax import nnx


def to_jax(x: Any) -> jnp.ndarray:
    """Convert a torch tensor (or array-like) to a JAX array on host.

    Args:
        x: A ``torch.Tensor``, NumPy array, or anything ``jnp.asarray`` accepts.

    Returns:
        jnp.ndarray: The value as a JAX array.
    """
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return jnp.asarray(x)


def copy_linear_(dst: nnx.Linear, src: Any) -> None:
    """Copy a torch ``nn.Linear`` into an :class:`flax.nnx.Linear` in place.

    torch stores the weight as ``[out, in]`` while NNX stores the kernel as
    ``[in, out]``, so the weight is transposed on copy.

    Args:
        dst: Destination NNX linear layer.
        src: Source torch ``nn.Linear`` (read-only).
    """
    dst.kernel = nnx.Param(to_jax(src.weight).T)
    if getattr(src, "bias", None) is not None and dst.bias is not None:
        dst.bias = nnx.Param(to_jax(src.bias))


def copy_torch_params_(dst: nnx.Module, src: Any) -> None:
    """Copy params from a torch ``nn.Sequential``-style stack into an NNX module.

    Walks the linear layers of ``src`` (a torch module exposing ``modules()``)
    in order and copies each into the correspondingly ordered NNX ``Linear``
    layer found on ``dst`` via :func:`flax.nnx.iter_children`-style traversal.
    Only ``Linear`` layers are handled — this targets the MLP/projector parity
    path; richer backbones (conv/BN) get their own mappers.

    Args:
        dst: Destination NNX module (e.g. :class:`stable_pretraining.jax.backbone.MLP`).
        src: Source torch module with the same ordered sequence of linear layers.

    Raises:
        ValueError: If the number of linear layers differs between the two.
    """
    import torch

    torch_linears = [m for m in src.modules() if isinstance(m, torch.nn.Linear)]
    nnx_linears = [m for _, m in _iter_linears(dst)]
    if len(torch_linears) != len(nnx_linears):
        raise ValueError(
            f"Linear-layer count mismatch: torch has {len(torch_linears)}, "
            f"nnx has {len(nnx_linears)}. Structures must align for parity copy."
        )
    for d, s in zip(nnx_linears, torch_linears):
        copy_linear_(d, s)


def _iter_linears(module: nnx.Module):
    """Yield ``(path, nnx.Linear)`` pairs in deterministic insertion order."""
    for path, value in _walk(module, ()):
        if isinstance(value, nnx.Linear):
            yield path, value


def _walk(obj: Any, path: tuple):
    """Depth-first walk over NNX submodules in attribute-definition order."""
    if isinstance(obj, nnx.Linear):
        yield path, obj
        return
    if isinstance(obj, nnx.Module):
        for name, child in vars(obj).items():
            # Check sequence containers first: ``nnx.List`` subclasses
            # ``nnx.Module``, so the Module branch would otherwise swallow it
            # and miss the contained layers.
            if isinstance(child, (list, tuple, nnx.List)):
                for i, item in enumerate(child):
                    if isinstance(item, nnx.Module):
                        yield from _walk(item, path + (name, i))
            elif isinstance(child, nnx.Module):
                yield from _walk(child, path + (name,))
