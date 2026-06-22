"""Distributed training utilities.

This module also hosts the in-house rank-zero / seeding helpers that used to be
imported from PyTorch Lightning (``lightning.pytorch.utilities.rank_zero`` and
``lightning.pytorch.seed_everything``). They are reimplemented here so the
library does not depend on Lightning's internal utility surface for these basic
primitives. The implementations are behavior-compatible with Lightning for the
inputs this codebase exercises; see ``tests/unit/test_distributed_rank_seed.py``
for the parity assertions.
"""

import functools
import logging
import os
import random
import warnings
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.nn

log = logging.getLogger(__name__)

# Same precedence Lightning uses (see ``lightning.fabric.utilities.rank_zero``):
# LOCAL_RANK before SLURM_PROCID because SLURM_PROCID can be set even when SLURM
# is not the process launcher.
_RANK_ENV_KEYS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized.

    Returns:
        bool: True if distributed is available and initialized, False otherwise
    """
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the global rank of the current process.

    Resolution order (a superset of Lightning's, and at least as correct in
    every launch mode):

    1. ``torch.distributed.get_rank()`` when a process group is initialized —
       the authoritative rank during training.
    2. The first set environment variable among ``RANK``, ``LOCAL_RANK``,
       ``SLURM_PROCID``, ``JSM_NAMESPACE_RANK`` (covers torchrun, SLURM, and
       Lightning's subprocess launcher *before* the process group is built).
    3. ``0`` otherwise (single-process / pre-launch).

    Returns:
        int: The resolved global rank (``0`` on a single process).
    """
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    for key in _RANK_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                # A malformed env var should not crash logging gating; treat
                # it as "unknown" and fall through to the next candidate.
                continue
    return 0


def get_world_size() -> int:
    """Return the number of processes in the distributed group (``1`` if none).

    Returns:
        int: ``torch.distributed.get_world_size()`` when initialized, else ``1``.
    """
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def rank_zero_only(fn: Callable, default=None) -> Callable:
    """Decorator that runs ``fn`` only on global rank 0.

    On non-zero ranks the wrapped call is skipped and ``default`` is returned
    (``None`` if not given). The rank is resolved via :func:`get_rank` **at call
    time**, so a rank assigned after import (the normal DDP/FSDP case, where the
    launcher sets ``RANK``/``LOCAL_RANK`` or the process group is initialized
    before the call) is correctly honored.

    Drop-in replacement for ``lightning.pytorch.utilities.rank_zero_only``.

    Args:
        fn: The function to guard.
        default: Value returned on non-zero ranks. Defaults to ``None``.

    Returns:
        Callable: The rank-gated wrapper.
    """

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if get_rank() == 0:
            return fn(*args, **kwargs)
        return default

    return wrapped_fn


@rank_zero_only
def rank_zero_warn(message: Union[str, Warning], stacklevel: int = 4, **kwargs) -> None:
    """Emit a warning only on global rank 0.

    Drop-in replacement for ``lightning.pytorch.utilities.rank_zero_warn``.
    """
    warnings.warn(message, stacklevel=stacklevel, **kwargs)


@rank_zero_only
def rank_zero_info(*args, stacklevel: int = 4, **kwargs) -> None:
    """Emit an info-level log message only on global rank 0."""
    kwargs["stacklevel"] = stacklevel
    log.info(*args, **kwargs)


class _DummyExperiment:
    """No-op stand-in for a logger experiment on non-zero ranks.

    Mirrors Lightning's ``_DummyExperiment``: every attribute access returns a
    no-op callable, indexing returns ``self`` (so ``experiment[0].add_image(...)``
    works), and item assignment is ignored.
    """

    def nop(self, *args, **kwargs):
        """Swallow any call."""
        return None

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx: int) -> "_DummyExperiment":
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Return the real logger experiment on rank 0, a :class:`_DummyExperiment` otherwise.

    Drop-in replacement for
    ``lightning.pytorch.loggers.logger.rank_zero_experiment``; decorates a
    logger's ``experiment`` property so non-zero ranks never touch the real
    backend.
    """

    @functools.wraps(fn)
    def experiment(self):
        if get_rank() > 0:
            return _DummyExperiment()
        return fn(self)

    return experiment


# uint32 range, matching numpy's accepted seed bounds (and Lightning's).
_MAX_SEED_VALUE = 4294967295  # 2**32 - 1
_MIN_SEED_VALUE = 0


def seed_everything(
    seed: Optional[int] = None, workers: bool = False, verbose: bool = True
) -> int:
    """Seed Python ``random``, NumPy, and PyTorch RNGs for reproducibility.

    Behavior-compatible with ``lightning.pytorch.seed_everything``. In
    particular it sets the same environment variables Lightning's data
    connector reads:

    - ``PL_GLOBAL_SEED``: forwarded to spawned subprocesses (``ddp_spawn``) and
      used by ``seed=None`` to recover a previously set seed.
    - ``PL_SEED_WORKERS``: set to ``1`` when ``workers=True`` so Lightning still
      installs its dataloader ``worker_init_fn`` and seeds worker processes.

    Keeping that env contract is intentional: it preserves ``workers=True``
    worker-seeding without re-depending on Lightning's seeding code.

    Args:
        seed: Integer seed. If ``None``, read from ``PL_GLOBAL_SEED`` (or ``0``
            if unset). Must be in ``[0, 2**32 - 1]``.
        workers: If ``True``, configure dataloader worker seeding via the
            ``PL_SEED_WORKERS`` env contract.
        verbose: If ``True``, log the seed being set.

    Returns:
        int: The seed that was set.
    """
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = 0
            if verbose:
                rank_zero_warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                raise ValueError(
                    f"Invalid seed specified via PL_GLOBAL_SEED: {env_seed!r}"
                )
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (_MIN_SEED_VALUE <= seed <= _MAX_SEED_VALUE):
        raise ValueError(
            f"{seed} is not in bounds, numpy accepts from "
            f"{_MIN_SEED_VALUE} to {_MAX_SEED_VALUE}"
        )

    if verbose:
        log.info(f"Seed set to {seed}")

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        pass
    torch.manual_seed(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


def all_gather(tensor, *args, **kwargs):
    """Gather tensors from all processes (autograd-aware).

    Uses ``torch.distributed.nn.functional.all_gather`` so gradients flow back
    to each rank's source tensor — the behavior SSL contrastive losses need
    when gathering negatives across GPUs.

    Note:
        The list returned **must** be used (e.g. ``torch.cat(all_gather(x), 0)``)
        — the gathered values are the return value, not an in-place mutation of
        ``tensor``.

    Args:
        tensor: The tensor to gather.
        *args: Forwarded to the underlying collective (e.g. ``group``).
        **kwargs: Forwarded to the underlying collective.

    Returns:
        list[torch.Tensor]: One tensor per rank, in rank order. A single-element
        ``[tensor]`` when distributed is not initialized.
    """
    if is_dist_avail_and_initialized():
        return list(torch.distributed.nn.functional.all_gather(tensor, *args, **kwargs))
    return [tensor]


def all_reduce(tensor, *args, **kwargs):
    """Reduce a tensor across all processes (autograd-aware, SUM by default).

    Uses ``torch.distributed.nn.functional.all_reduce``, which returns a new
    tensor rather than mutating in place.

    Note:
        The result **must** be used (``x = all_reduce(x)``) — the previous
        implementation discarded the return and was a silent no-op under DDP.

    Args:
        tensor: The tensor to reduce.
        *args: Forwarded to the underlying collective (e.g. ``op``, ``group``).
        **kwargs: Forwarded to the underlying collective.

    Returns:
        torch.Tensor: The reduced tensor (the input unchanged when distributed
        is not initialized).
    """
    if is_dist_avail_and_initialized():
        return torch.distributed.nn.functional.all_reduce(tensor, *args, **kwargs)
    return tensor


class FullGatherLayer(torch.autograd.Function):
    """Gather tensors from all process and support backward propagation.

    Supports backward propagation for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if not torch.distributed.is_initialized():
            return x.unsqueeze(0)
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return torch.stack(output)

    @staticmethod
    def backward(ctx, grad):
        if not torch.distributed.is_initialized():
            return grad.squeeze(0)
        torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.AVG)
        return grad[torch.distributed.get_rank()]
