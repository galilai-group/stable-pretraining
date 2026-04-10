"""Distributed training utilities."""

import torch
import torch.distributed as dist
import torch.distributed.nn


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized.

    Returns:
        bool: True if distributed is available and initialized, False otherwise
    """
    return dist.is_available() and dist.is_initialized()


def all_gather(tensor, *args, **kwargs):
    """Gather tensors from all processes.

    Args:
        tensor: The tensor to gather
        *args: Additional arguments for all_gather
        **kwargs: Additional keyword arguments for all_gather

    Returns:
        Tuple containing the gathered tensors
    """
    if is_dist_avail_and_initialized():
        return torch.distributed.nn.functional.all_gather(tensor, *args, **kwargs)
    return (tensor,)


def all_reduce(tensor, *args, **kwargs):
    """Reduce tensors across all processes.

    Args:
        tensor: The tensor to reduce
        *args: Additional arguments for all_reduce
        **kwargs: Additional keyword arguments for all_reduce

    Returns:
        The reduced tensor
    """
    if is_dist_avail_and_initialized():
        torch.distributed.nn.functional.all_reduce(tensor, *args, **kwargs)
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
