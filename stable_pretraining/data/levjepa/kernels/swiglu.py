"""v2's SwiGLU MLP with its elementwise work inside the gate/up and down GEMMs.

SwiGLU.forward runs down(silu(gate(x)) * up(x)): two GEMMs write g and u, an elementwise pass reads them and writes
h, and the down GEMM reads h. Backward has the mirror image: the down GEMM's dgrad writes dh, an elementwise pass
reads dh, g and u and writes dg and du. With the model's small inner dims (K = 384 or 768) these GEMMs move about as
many bytes as they compute, so the elementwise passes cost ~10% of a ViT-S step on the H100. Here:

- forward: one Triton GEMM computes the gate and up tiles of the same rows from one load of x, stores g and u (for
  backward) and h = silu(g) * u from its epilogue; the down projection stays on cuBLAS;
- backward: one Triton GEMM computes dh = dy @ W_down and turns it into dg and du in its epilogue, written side by
  side as (N, 2H) so dx is one cuBLAS GEMM over both.

    y = swiglu(x, gate_weight, up_weight, down_weight)       # (..., D) -> (..., D), nn.Linear weights, no bias

bf16 or fp16 inputs (under autocast, cast to its dtype); silu in float32 from g and u rounded to the input dtype, as
the separate ops compute it.
"""

import torch
import triton
import triton.language as tl


def _configs():
    """GEMM tiles to autotune (those that don't fit a GPU's shared memory are skipped)."""
    return [
        triton.Config(
            {"BM": bm, "BN": bn, "BK": bk, "GROUP": 8},
            num_warps=warps,
            num_stages=stages,
        )
        for bm, bn, bk, warps, stages in (
            (128, 128, 64, 8, 3),
            (128, 128, 64, 8, 4),
            (128, 64, 64, 4, 3),
            (128, 64, 64, 8, 4),
            (128, 64, 128, 8, 3),
            (64, 128, 64, 4, 3),
            (128, 128, 32, 8, 4),
            (256, 64, 64, 8, 3),
            (64, 64, 64, 4, 4),
        )
    ]


@triton.jit
def _tile(N, H, BM: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr):
    """(rows, cols) of this program's output tile, in grouped order (GROUP row tiles share their column tiles in L2)."""
    pid = tl.program_id(0)
    m_tiles, n_tiles = tl.cdiv(N, BM), H // BN
    per_group = GROUP * n_tiles
    first = (pid // per_group) * GROUP
    size = min(m_tiles - first, GROUP)
    pm = first + (pid % per_group) % size
    pn = (pid % per_group) // size
    return pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN)


@triton.autotune(configs=_configs(), key=["N", "K", "H"])
@triton.jit
def _fwd(x, wg, wu, gu, h, N, K: tl.constexpr, H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr, GROUP: tl.constexpr):  # fmt: skip
    rows, cols = _tile(N, H, BM, BN, GROUP)
    ks = tl.arange(0, BK)
    live = rows[:, None] < N
    xp = x + rows[:, None].to(tl.int64) * K + ks[None, :]
    # (BK, BN) tiles of the (H, K) weights: their transposes
    gp = wg + cols[None, :] * K + ks[:, None]
    up = wu + cols[None, :] * K + ks[:, None]
    acc_g = tl.zeros((BM, BN), tl.float32)
    acc_u = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(xp, mask=live, other=0.0)
        acc_g = tl.dot(a, tl.load(gp), acc_g)
        acc_u = tl.dot(a, tl.load(up), acc_u)
        xp += BK
        gp += BK
        up += BK
    g = acc_g.to(gu.dtype.element_ty)
    u = acc_u.to(gu.dtype.element_ty)
    out = rows[:, None].to(tl.int64) * (2 * H) + cols[None, :]
    tl.store(gu + out, g, mask=live)
    tl.store(gu + out + H, u, mask=live)
    g = g.to(tl.float32)
    hv = g * tl.sigmoid(g) * u.to(tl.float32)
    tl.store(
        h + rows[:, None].to(tl.int64) * H + cols[None, :],
        hv.to(h.dtype.element_ty),
        mask=live,
    )


@triton.autotune(configs=_configs(), key=["N", "K", "H"])
@triton.jit
def _bwd(dy, wd, gu, dgu, N, K: tl.constexpr, H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr, GROUP: tl.constexpr):  # fmt: skip
    rows, cols = _tile(N, H, BM, BN, GROUP)
    ks = tl.arange(0, BK)
    live = rows[:, None] < N
    ap = dy + rows[:, None].to(tl.int64) * K + ks[None, :]
    bp = wd + ks[:, None] * H + cols[None, :]  # (BK, BN) of the (K, H) down weight
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(ap, mask=live, other=0.0), tl.load(bp), acc)
        ap += BK
        bp += BK * H
    dh = acc.to(dgu.dtype.element_ty).to(tl.float32)  # the separate ops round dh too
    out = rows[:, None].to(tl.int64) * (2 * H) + cols[None, :]
    g = tl.load(gu + out, mask=live, other=0.0).to(tl.float32)
    u = tl.load(gu + out + H, mask=live, other=0.0).to(tl.float32)
    s = tl.sigmoid(g)
    tl.store(
        dgu + out, (dh * u * s * (1 + g * (1 - s))).to(dgu.dtype.element_ty), mask=live
    )
    tl.store(dgu + out + H, (dh * g * s).to(dgu.dtype.element_ty), mask=live)


def _grid(N, H):
    return lambda c: (triton.cdiv(N, c["BM"]) * (H // c["BN"]),)


@torch.library.custom_op("levjepa::swiglu_fwd", mutates_args=())
def _swiglu_fwd(
    x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    N, K = x.shape
    H = wg.shape[0]
    gu = torch.empty(N, 2 * H, device=x.device, dtype=x.dtype)
    h = torch.empty(N, H, device=x.device, dtype=x.dtype)
    _fwd[_grid(N, H)](x, wg, wu, gu, h, N, K, H)
    return h, gu


@_swiglu_fwd.register_fake
def _(x, wg, wu):
    return x.new_empty(x.shape[0], wg.shape[0]), x.new_empty(
        x.shape[0], 2 * wg.shape[0]
    )


@torch.library.custom_op("levjepa::swiglu_mlp", mutates_args=())
def _swiglu_mlp(
    x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, gu = _swiglu_fwd(x, wg, wu)
    return h @ wd.t(), h, gu


@_swiglu_mlp.register_fake
def _(x, wg, wu, wd):
    H = wg.shape[0]
    return (
        x.new_empty(x.shape[0], wd.shape[0]),
        x.new_empty(x.shape[0], H),
        x.new_empty(x.shape[0], 2 * H),
    )


@torch.library.custom_op("levjepa::swiglu_bwd", mutates_args=())
def _swiglu_bwd(
    dy: torch.Tensor, x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor, h: torch.Tensor,
    gu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    N, K = dy.shape
    H = wg.shape[0]
    dy = dy.contiguous()
    dgu = torch.empty_like(gu)
    _bwd[_grid(N, H)](dy, wd, gu, dgu, N, K, H)
    dx = dgu @ torch.cat([wg, wu])
    return dx, dgu[:, :H].t() @ x, dgu[:, H:].t() @ x, dy.t() @ h


@_swiglu_bwd.register_fake
def _(dy, x, wg, wu, wd, h, gu):
    return (
        torch.empty_like(x),
        torch.empty_like(wg),
        torch.empty_like(wu),
        torch.empty_like(wd),
    )


def _setup(ctx, inputs, output):
    x, wg, wu, wd = inputs
    _, h, gu = output
    ctx.save_for_backward(x, wg, wu, wd, h, gu)
    ctx.set_materialize_grads(
        False
    )  # h and gu are saved, never used: no zero-filled gradients for them


def _backward(ctx, dy, dh, dgu):
    return _swiglu_bwd(dy, *ctx.saved_tensors)


_swiglu_mlp.register_autograd(_backward, setup_context=_setup)


def supported(dim, hidden):
    return dim % 64 == 0 and hidden % 128 == 0


def swiglu(x, gate_weight, up_weight, down_weight):
    """SwiGLU.forward: down(silu(gate(x)) * up(x)) for x (..., D) and nn.Linear weights (H, D), (H, D), (D, H).

    D a multiple of 64, H of 128 (supported() checks).
    """
    ws = (gate_weight, up_weight, down_weight)
    if torch.is_autocast_enabled("cuda"):
        dt = torch.get_autocast_dtype("cuda")
        x, ws = x.to(dt), [w.to(dt) for w in ws]
    shape = x.shape
    y = _swiglu_mlp(x.reshape(-1, shape[-1]).contiguous(), *ws)[0]
    return y.view(shape)
