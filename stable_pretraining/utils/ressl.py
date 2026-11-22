import torch
import torch.distributed as dist


@torch.no_grad()
def gather_teacher_keys(keys: torch.Tensor) -> torch.Tensor:
    """Gather teacher projections across distributed processes.

    Concatenates teacher projections from all ranks so that each process
    appends the same keys in the same order to its memory queue.

    Args:
        keys: Local teacher projections of shape (N, D).

    Returns:
        Gathered projections of shape (N * world_size, D), or the
        original tensor when distributed training is inactive.

    Notes:
        All ranks must provide tensors of the same shape.
    """
    if not dist.is_available() or not dist.is_initialized():
        return keys

    world_size = dist.get_world_size()
    if world_size == 1:
        return keys

    gathered = [torch.empty_like(keys) for _ in range(world_size)]
    dist.all_gather(gathered, keys.contiguous())

    return torch.cat(gathered, dim=0)
