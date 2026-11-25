import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_reduce as functional_all_reduce


def mean_var(
    x: Tensor,
    dim: int = 0,
    keepdim: bool = False,
    unbiased: bool = True,
    sync: bool = True,
    same_shape_across_devices: bool = True,
) -> tuple[Tensor, Tensor, int]:
    """Compute mean and variance synchronized across DDP ranks.

    Numerically stable for bf16/fp16 by using fp32 accumulation internally.
    Supports variable batch sizes across ranks.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int
        Dimension to reduce.
    keepdim : bool
        Retain reduced dimension.
    unbiased : bool
        Apply Bessel's correction (N-1 denominator).

    Returns:
    -------
    mean : Tensor
        Global mean across all ranks.
    var : Tensor
        Global variance across all ranks.
    n_global : int
        Total sample count across all ranks.
    """
    n_local = x.size(dim)
    if not sync or not dist.is_initialized():
        var, mean = torch.var_mean(x, dim=dim, keepdims=keepdim, unbiased=unbiased)
        return mean, var, n_local
    elif same_shape_across_devices:
        n_global = n_local * dist.get_world_size()
        acc_x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        # E[X²] = Var + Mean²  →  single all_reduce
        mean = acc_x.mean(dim, keepdim=keepdim)
        ssq = acc_x.square().mean(dim, keepdim=keepdim)
        stats = torch.stack([mean, ssq])
        stats = functional_all_reduce(stats, op=dist.ReduceOp.AVG)
        global_mean = stats[0]
        global_var = stats[1] - global_mean.square()
        if unbiased:
            global_var = global_var * n_global / (n_global - 1)
        return global_mean.to(x.dtype), global_var.to(x.dtype), n_global

    input_dtype = x.dtype
    device = x.device
    use_fp32 = input_dtype in (torch.float16, torch.bfloat16)
    acc_x = x.float() if use_fp32 else x

    # Local statistics (fp32 accumulator via dtype arg)
    local_mean = acc_x.mean(dim, keepdim=True)
    local_sq_mean = acc_x.square().mean(dim, keepdim=True)

    # Scale by count (single multiply, avoids accumulated sum)
    n_mean = n_local * local_mean
    n_sq_mean = n_local * local_sq_mean

    # Fused all-reduce
    stacked = torch.cat([n_mean, n_sq_mean], dim=dim)
    stacked = functional_all_reduce(stacked, op=dist.ReduceOp.SUM)

    n_local_t = torch.tensor([n_local], device=device, dtype=torch.float32)
    n_global_t = functional_all_reduce(n_local_t, op=dist.ReduceOp.SUM)

    # Recover statistics
    n_mean, n_sq_mean = stacked.chunk(2, dim=dim)

    global_mean = n_mean / n_global_t
    global_sq_mean = n_sq_mean / n_global_t
    global_var = global_sq_mean - global_mean * global_mean

    n_global = int(n_global_t.item())
    if unbiased:
        global_var = global_var * n_global / (n_global - 1)

    if use_fp32:
        global_mean = global_mean.to(input_dtype)
        global_var = global_var.to(input_dtype)

    if not keepdim:
        global_mean = global_mean.squeeze(dim)
        global_var = global_var.squeeze(dim)

    return global_mean, global_var, n_global


def mean_std(
    x: Tensor,
    dim: int = 0,
    keepdim: bool = False,
    unbiased: bool = True,
    eps: float = 1e-8,
    sync: bool = True,
    same_shape_across_devices: bool = True,
) -> tuple[Tensor, Tensor, int]:
    mean, var, bs = mean_var(x, dim, keepdim, unbiased, sync, same_shape_across_devices)
    std = var.add(eps).sqrt()
    return mean, std, bs
