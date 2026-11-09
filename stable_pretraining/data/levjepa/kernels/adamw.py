"""The end of a training step in one pass: gradient clipping, AdamW, an EMA teacher and a low-precision weight copy.

After backward, v2's step runs clip_grad_norm_ (a norm per tensor, their norm, a scale per tensor), fused AdamW, a
lerp of the teacher towards the weights and, in bf16 training, a copy of the weights into the bf16 model: about ten
multi-tensor kernels, each reading the weights or their state again. Here the weights, gradients, AdamW moments,
teacher and bf16 copy each live in one flat buffer (the parameters become views into it), and two kernels do it all:
one sums the squared gradients in fixed partials, the next reads them, clips and updates every element once.

    step = FusedStep(params, decay, teacher=teacher_params, copies=bf16_params, lr=4e-4, betas=(0.9, 0.95),
                     weight_decay=0.04, max_norm=1.0, ema=0.9998)
    torch._foreach_copy_(step.grads, [p.grad for p in bf16_params])   # or the fp32 params' grads
    step.step()

teacher and copies pair with the first len(teacher) / len(copies) params. Same arithmetic as clip_grad_norm_ +
AdamW(fused=True) + torch._foreach_lerp_ + torch._foreach_copy_, CUDA graph capturable (the step count lives on the
GPU).
"""

import math

import torch
import triton
import triton.language as tl


PARTIALS = 1024  # programs (and partial sums) of the squared-gradient pass
BLOCK = 2048


@triton.jit
def _sumsq(g, partial, n, BLOCK: tl.constexpr, P: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(pid * BLOCK, n, P * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(g + offs, mask=offs < n, other=0.0)
        acc += x * x
    tl.store(partial + pid, tl.sum(acc))


@triton.jit
def _update(p, g, m, v, decay, t, low, partial, count, n, n_teacher, n_low, lr, b1, b2, eps, wd, max_norm, w,
            BLOCK: tl.constexpr, P: tl.constexpr):  # fmt: skip
    total = tl.sqrt(tl.sum(tl.load(partial + tl.arange(0, P))))
    coef = tl.minimum(max_norm / (total + 1e-6), 1.0)
    step = tl.load(count)
    bc1 = 1 - tl.exp2(step * tl.log2(b1))
    bc2 = tl.sqrt(1 - tl.exp2(step * tl.log2(b2)))
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offs < n
    gr = tl.load(g + offs, mask=live, other=0.0) * coef
    pv = tl.load(p + offs, mask=live, other=0.0)
    pv = pv * (1 - lr * wd * tl.load(decay + offs, mask=live, other=0).to(tl.float32))
    mv = tl.load(m + offs, mask=live, other=0.0)
    vv = tl.load(v + offs, mask=live, other=0.0)
    mv += (1 - b1) * (gr - mv)  # exp_avg.lerp_(grad, 1 - beta1), as AdamW
    vv = b2 * vv + (1 - b2) * gr * gr
    pv -= (lr / bc1) * mv / (tl.sqrt(vv) / bc2 + eps)
    tl.store(p + offs, pv, mask=live)
    tl.store(m + offs, mv, mask=live)
    tl.store(v + offs, vv, mask=live)
    tm = offs < n_teacher
    tv = tl.load(t + offs, mask=tm, other=0.0)
    tl.store(t + offs, tv + w * (pv - tv), mask=tm)  # torch.lerp for weights < 0.5
    tl.store(low + offs, pv.to(low.dtype.element_ty), mask=offs < n_low)


def _flatten(tensors, dtype, device):
    """One buffer holding tensors (copied in); each tensor's data becomes a view into it."""
    flat = torch.zeros(
        max(sum(x.numel() for x in tensors), 1), dtype=dtype, device=device
    )
    o = 0
    for x in tensors:
        k = x.numel()
        flat[o : o + k].copy_(x.detach().reshape(-1))
        x.data = flat[o : o + k].view_as(x)
        o += k
    return flat


class FusedStep:
    """AdamW (decay[i]: params[i] takes weight decay) over params, which become views into one flat buffer.

    Each step clips self.grads (views into one flat gradient buffer, filled by the caller) to max_norm, updates the
    params, lerps the teacher towards them by 1 - ema and writes them into copies.
    """

    def __init__(
        self,
        params,
        decay,
        teacher=(),
        copies=(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        max_norm=1.0,
        ema=0.999,
    ):
        params, teacher, copies = list(params), list(teacher), list(copies)
        dev = params[0].device
        for paired in (teacher, copies):
            assert all(a.shape == b.shape for a, b in zip(params, paired)), (
                "pairs with the first params"
            )
        sizes = [p.numel() for p in params]
        self.n = sum(sizes)
        self.p = _flatten(params, torch.float32, dev)
        self.t = _flatten(teacher, torch.float32, dev)
        self.low = _flatten(copies, copies[0].dtype if copies else torch.bfloat16, dev)
        self.n_teacher, self.n_low = (
            sum(x.numel() for x in teacher),
            sum(x.numel() for x in copies),
        )
        self.g = torch.zeros(self.n, device=dev)
        self.grads, o = [], 0
        for p, k in zip(params, sizes):
            self.grads.append(self.g[o : o + k].view_as(p))
            o += k
        self.m, self.v = torch.zeros_like(self.p), torch.zeros_like(self.p)
        self.decay = torch.cat(
            [
                torch.full((k,), int(d), dtype=torch.uint8, device=dev)
                for k, d in zip(sizes, decay)
            ]
        )
        self.count = torch.zeros((), device=dev)
        self.partial = torch.empty(PARTIALS, device=dev)
        self.hyper = (lr, *betas, eps, weight_decay, max_norm, 1 - ema)

    def step(self):
        """Clip the gradients in self.grads to max_norm, AdamW, then the teacher and the low-precision copies."""
        self.count += 1
        _sumsq[(PARTIALS,)](
            self.g, self.partial, self.n, BLOCK=BLOCK, P=PARTIALS, num_warps=8
        )
        _update[(math.ceil(self.n / BLOCK),)](
            self.p, self.g, self.m, self.v, self.decay, self.t, self.low, self.partial, self.count, self.n,
            self.n_teacher, self.n_low, *self.hyper, BLOCK=BLOCK, P=PARTIALS, num_warps=8,
        )  # fmt: skip
