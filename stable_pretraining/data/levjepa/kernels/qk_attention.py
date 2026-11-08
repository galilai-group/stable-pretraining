"""v2's whole attention core in one Triton kernel per pass: QK-RMSNorm, RoPE and prefix attention (attention.py).

Attention.forward runs qkv -> split heads -> RMSNorm(q), RMSNorm(k) -> RoPE -> masked attention -> merge heads, and
autograd runs each step back: about ten elementwise and reduction passes over (B, H, L, D) per layer around the
attention itself. Here the kernels read q, k and v straight from the qkv projection's output (B, L, 3 * H * D),
normalize and rotate q and k in registers (recomputed in the backward, never stored), and write the output as
(B, L, H * D) for the output projection. The backward writes d(qkv) in the projection's layout and the two norm
weights' gradients.

    out = qk_attention(qkv, q_weight, k_weight, cos, sin, kv_len, heads, eps)

cos, sin: Attention.rope_tables (1 or B, 1, L, D) float32; kv_len: attention.prefix_lengths; eps: the RMSNorms' (None
in v2: the input dtype's machine epsilon, as nn.RMSNorm). Computed in float32 from the bf16 inputs.
"""

import math
import os

import torch
import triton
import triton.language as tl

from .attention import LOG2E


BLOCK = 64
STAGES = int(os.environ.get("QK_STAGES", 1))
BLOCK_B = int(os.environ.get("QK_BLOCK_B", 32))  # backward: 64 spills registers
WARPS_B = int(os.environ.get("QK_WARPS_B", 4))


@triton.jit
def _rot(x, BM: tl.constexpr, D: tl.constexpr):
    """RoPE's pair rotation of the last dim: (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)."""
    a, b = tl.split(tl.reshape(x, (BM, D // 2, 2)))
    return tl.reshape(tl.join(-b, a), (BM, D))


@triton.jit
def _rot_t(g, BM: tl.constexpr, D: tl.constexpr):
    """The transpose of _rot: (g0, g1, g2, g3, ...) -> (g1, -g0, g3, -g2, ...)."""
    a, b = tl.split(tl.reshape(g, (BM, D // 2, 2)))
    return tl.reshape(tl.join(b, -a), (BM, D))


@triton.jit
def _qk(
    X, C, S, xoff, coff, rows, rd, L, xl, w, eps, BM: tl.constexpr, D: tl.constexpr
):
    """Rows of q or k, normalized and rotated (float32), from the raw projection; also the raw rows and their rstd."""
    m = rows[:, None] < L
    x = tl.load(X + xoff + rows[:, None] * xl + rd[None], mask=m, other=0.0).to(
        tl.float32
    )
    r = tl.rsqrt(tl.sum(x * x, 1) / D + eps)
    xn = x * r[:, None] * w[None]
    c = tl.load(C + coff + rows[:, None] * D + rd[None], mask=m, other=0.0)
    s = tl.load(S + coff + rows[:, None] * D + rd[None], mask=m, other=0.0)
    return xn * c + _rot(xn, BM, D) * s, x, r, c, s


@triton.jit
def _qk_back(g, x, r, c, s, w, BM: tl.constexpr, D: tl.constexpr):
    """d(raw rows) and this block's d(norm weight) from g = d(normalized, rotated rows)."""
    gn = g * c + _rot_t(g * s, BM, D)  # through RoPE
    gw = gn * w[None]
    dx = r[:, None] * gw - x * (r * r * r)[:, None] * (tl.sum(gw * x, 1) / D)[:, None]
    return dx, tl.sum(gn * x * r[:, None], 0)


@triton.jit
def _fwd(
    QKV,
    WQ,
    WK,
    C,
    S,
    KVL,
    Out,
    LSE,
    H,
    L,
    csb,
    eps,
    qk_scale,
    BM: tl.constexpr,
    D: tl.constexpr,
):
    m0 = tl.program_id(0) * BM
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    rd = tl.arange(0, D)
    xl = 3 * H * D
    base = b * L * xl + h * D
    coff = b * csb
    wq, wk = tl.load(WQ + rd).to(tl.float32), tl.load(WK + rd).to(tl.float32)
    rm = m0 + tl.arange(0, BM)
    q, _, _, _, _ = _qk(QKV, C, S, base, coff, rm, rd, L, xl, wq, eps, BM, D)
    q = q.to(tl.bfloat16)
    kvl = tl.load(KVL + b * L + rm, mask=rm < L, other=0)
    hi = tl.max(kvl, 0)
    m_i = tl.full([BM], -1e30, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(0, hi, BM):
        rn = n0 + tl.arange(0, BM)
        k, _, _, _, _ = _qk(
            QKV, C, S, base + H * D, coff, rn, rd, L, xl, wk, eps, BM, D
        )
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * qk_scale
        s = tl.where(rn[None] < kvl[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            QKV + base + 2 * H * D + rn[:, None] * xl + rd[None],
            mask=rn[:, None] < L,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o = (acc / l_safe[:, None]).to(Out.dtype.element_ty)
    tl.store(
        Out + b * L * H * D + rm[:, None] * H * D + h * D + rd[None],
        o,
        mask=rm[:, None] < L,
    )
    tl.store(LSE + bh * L + rm, m_i + tl.log2(l_safe), mask=rm < L)


@triton.jit
def _delta(Out, DO, DELTA, H, L, BM: tl.constexpr, D: tl.constexpr):
    m0 = tl.program_id(0) * BM
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    rm, rd = m0 + tl.arange(0, BM), tl.arange(0, D)
    off = b * L * H * D + rm[:, None] * H * D + h * D + rd[None]
    o = tl.load(Out + off, mask=rm[:, None] < L, other=0.0).to(tl.float32)
    do = tl.load(DO + off, mask=rm[:, None] < L, other=0.0).to(tl.float32)
    tl.store(DELTA + bh * L + rm, tl.sum(o * do, 1), mask=rm < L)


@triton.jit
def _bwd_dkdv(
    QKV,
    WQ,
    WK,
    C,
    S,
    KVL,
    DO,
    LSE,
    DELTA,
    DQKV,
    DWK,
    H,
    L,
    csb,
    eps,
    qk_scale,
    sm_scale,
    BM: tl.constexpr,
    D: tl.constexpr,
):
    n0 = tl.program_id(0) * BM
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    rd = tl.arange(0, D)
    xl = 3 * H * D
    base = b * L * xl + h * D
    coff = b * csb
    wq, wk = tl.load(WQ + rd).to(tl.float32), tl.load(WK + rd).to(tl.float32)
    rn = n0 + tl.arange(0, BM)
    k, _, _, _, _ = _qk(QKV, C, S, base + H * D, coff, rn, rd, L, xl, wk, eps, BM, D)
    kb = k.to(tl.bfloat16)
    v = tl.load(
        QKV + base + 2 * H * D + rn[:, None] * xl + rd[None],
        mask=rn[:, None] < L,
        other=0.0,
    )
    dk = tl.zeros([BM, D], tl.float32)
    dv = tl.zeros([BM, D], tl.float32)
    for m0 in range(0, L, BM):
        rm = m0 + tl.arange(0, BM)
        kvl = tl.load(KVL + b * L + rm, mask=rm < L, other=0)
        if tl.max(kvl, 0) > n0:
            q, _, _, _, _ = _qk(QKV, C, S, base, coff, rm, rd, L, xl, wq, eps, BM, D)
            q = q.to(tl.bfloat16)
            do = tl.load(
                DO + b * L * H * D + rm[:, None] * H * D + h * D + rd[None],
                mask=rm[:, None] < L,
                other=0.0,
            )
            lse = tl.load(LSE + bh * L + rm, mask=rm < L, other=0.0)
            delta = tl.load(DELTA + bh * L + rm, mask=rm < L, other=0.0)
            sT = (
                tl.dot(kb, tl.trans(q)) * qk_scale
            )  # keys x queries: only loaded-tile transposes
            pT = tl.where(rn[:, None] < kvl[None], tl.exp2(sT - lse[None]), 0.0)
            dv += tl.dot(pT.to(do.dtype), do)
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None])
            dk += tl.dot(dsT.to(q.dtype), q)
    _, kx, kr, kc, ks = _qk(
        QKV, C, S, base + H * D, coff, rn, rd, L, xl, wk, eps, BM, D
    )  # reloaded, not kept live
    dkx, dwk = _qk_back(dk * sm_scale, kx, kr, kc, ks, wk, BM, D)
    out = DQKV + base + rn[:, None] * xl + rd[None]
    tl.store(out + H * D, dkx.to(DQKV.dtype.element_ty), mask=rn[:, None] < L)
    tl.store(out + 2 * H * D, dv.to(DQKV.dtype.element_ty), mask=rn[:, None] < L)
    tl.store(
        DWK + (bh * tl.num_programs(0) + tl.program_id(0)) * D + rd, dwk
    )  # a partial per program, summed after


@triton.jit
def _bwd_dq(
    QKV,
    WQ,
    WK,
    C,
    S,
    KVL,
    DO,
    LSE,
    DELTA,
    DQKV,
    DWQ,
    H,
    L,
    csb,
    eps,
    qk_scale,
    sm_scale,
    BM: tl.constexpr,
    D: tl.constexpr,
):
    m0 = tl.program_id(0) * BM
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    rd = tl.arange(0, D)
    xl = 3 * H * D
    base = b * L * xl + h * D
    coff = b * csb
    wq, wk = tl.load(WQ + rd).to(tl.float32), tl.load(WK + rd).to(tl.float32)
    rm = m0 + tl.arange(0, BM)
    q, _, _, _, _ = _qk(QKV, C, S, base, coff, rm, rd, L, xl, wq, eps, BM, D)
    qb = q.to(tl.bfloat16)
    do = tl.load(
        DO + b * L * H * D + rm[:, None] * H * D + h * D + rd[None],
        mask=rm[:, None] < L,
        other=0.0,
    )
    lse = tl.load(LSE + bh * L + rm, mask=rm < L, other=0.0)
    delta = tl.load(DELTA + bh * L + rm, mask=rm < L, other=0.0)
    kvl = tl.load(KVL + b * L + rm, mask=rm < L, other=0)
    hi = tl.max(kvl, 0)
    dq = tl.zeros([BM, D], tl.float32)
    for n0 in range(0, hi, BM):
        rn = n0 + tl.arange(0, BM)
        k, _, _, _, _ = _qk(
            QKV, C, S, base + H * D, coff, rn, rd, L, xl, wk, eps, BM, D
        )
        kb = k.to(tl.bfloat16)
        v = tl.load(
            QKV + base + 2 * H * D + rn[:, None] * xl + rd[None],
            mask=rn[:, None] < L,
            other=0.0,
        )
        s = tl.dot(qb, tl.trans(kb)) * qk_scale
        p = tl.where(rn[None] < kvl[:, None], tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.bfloat16), kb)
    _, qx, qr, qc, qs = _qk(
        QKV, C, S, base, coff, rm, rd, L, xl, wq, eps, BM, D
    )  # reloaded, not kept live
    dqx, dwq = _qk_back(dq * sm_scale, qx, qr, qc, qs, wq, BM, D)
    tl.store(
        DQKV + base + rm[:, None] * xl + rd[None],
        dqx.to(DQKV.dtype.element_ty),
        mask=rm[:, None] < L,
    )
    tl.store(
        DWQ + (bh * tl.num_programs(0) + tl.program_id(0)) * D + rd, dwq
    )  # a partial per program, summed after


def _geometry(qkv, cos, heads):
    B, L, three_hd = qkv.shape
    D = three_hd // (3 * heads)
    assert qkv.is_contiguous() and cos.is_contiguous() and cos.shape[-2:] == (L, D)
    return B, L, D, (L * D if cos.shape[0] > 1 else 0)


@torch.library.custom_op("levjepa::qk_attention_fwd", mutates_args=())
def _qk_fwd(
    qkv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_len: torch.Tensor,
    heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, L, D, csb = _geometry(qkv, cos, heads)
    out = torch.empty(B, L, heads * D, device=qkv.device, dtype=qkv.dtype)
    lse = torch.empty(B, heads, L, device=qkv.device, dtype=torch.float32)
    _fwd[(triton.cdiv(L, BLOCK), B * heads)](
        qkv,
        wq,
        wk,
        cos,
        sin,
        kv_len,
        out,
        lse,
        heads,
        L,
        csb,
        eps,
        LOG2E / math.sqrt(D),
        BM=BLOCK,
        D=D,
        num_stages=STAGES,
        num_warps=4,
    )
    return out, lse


@_qk_fwd.register_fake
def _(qkv, wq, wk, cos, sin, kv_len, heads, eps):
    B, L, three_hd = qkv.shape
    return qkv.new_empty(B, L, three_hd // 3), qkv.new_empty(
        B, heads, L, dtype=torch.float32
    )


@torch.library.custom_op("levjepa::qk_attention_bwd", mutates_args=())
def _qk_bwd(
    do: torch.Tensor,
    qkv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_len: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L, D, csb = _geometry(qkv, cos, heads)
    do = do.contiguous()
    delta = torch.empty(B, heads, L, device=qkv.device, dtype=torch.float32)
    grid = (triton.cdiv(L, BLOCK_B), B * heads)
    _delta[grid](
        out, do, delta, heads, L, BM=BLOCK_B, D=D, num_warps=4, num_stages=STAGES
    )
    dqkv = torch.empty_like(qkv)
    dw = torch.empty(2, grid[0] * grid[1], D, device=qkv.device, dtype=torch.float32)
    args = (heads, L, csb, eps, LOG2E / math.sqrt(D), 1 / math.sqrt(D))
    _bwd_dkdv[grid](
        qkv,
        wq,
        wk,
        cos,
        sin,
        kv_len,
        do,
        lse,
        delta,
        dqkv,
        dw[1],
        *args,
        BM=BLOCK_B,
        D=D,
        num_warps=WARPS_B,
        num_stages=STAGES,
    )
    _bwd_dq[grid](
        qkv,
        wq,
        wk,
        cos,
        sin,
        kv_len,
        do,
        lse,
        delta,
        dqkv,
        dw[0],
        *args,
        BM=BLOCK_B,
        D=D,
        num_warps=WARPS_B,
        num_stages=STAGES,
    )
    dw = dw.sum(1)
    return dqkv, dw[0].to(wq.dtype), dw[1].to(wk.dtype)


@_qk_bwd.register_fake
def _(do, qkv, wq, wk, cos, sin, kv_len, out, lse, heads, eps):
    return torch.empty_like(qkv), torch.empty_like(wq), torch.empty_like(wk)


def _setup(ctx, inputs, output):
    qkv, wq, wk, cos, sin, kv_len, heads, eps = inputs
    out, lse = output
    ctx.save_for_backward(qkv, wq, wk, cos, sin, kv_len, out, lse)
    ctx.heads, ctx.eps = heads, eps


def _backward(ctx, dout, dlse):
    qkv, wq, wk, cos, sin, kv_len, out, lse = ctx.saved_tensors
    dqkv, dwq, dwk = _qk_bwd(
        dout, qkv, wq, wk, cos, sin, kv_len, out, lse, ctx.heads, ctx.eps
    )
    return dqkv, dwq, dwk, None, None, None, None, None


_qk_fwd.register_autograd(_backward, setup_context=_setup)


def qk_attention(qkv, q_weight, k_weight, cos, sin, kv_len, heads, eps=None):
    """Attention.forward's core: (B, L, 3 * H * D) qkv -> (B, L, H * D).

    RMSNorm(q), RMSNorm(k), RoPE and attention to keys j < kv_len[b, i]. D a power of two >= 16; cos, sin (1 or B, 1,
    L, D) float32.
    """
    eps = torch.finfo(qkv.dtype).eps if eps is None else eps
    return _qk_fwd(
        qkv, q_weight, k_weight, cos.contiguous(), sin.contiguous(), kv_len, heads, eps
    )[0]
