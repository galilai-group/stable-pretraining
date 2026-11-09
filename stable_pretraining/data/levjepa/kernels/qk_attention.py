"""v2's attention core around prefix attention (attention.py): QK-RMSNorm, RoPE, attention and their backward.

Attention.forward runs qkv -> split heads -> RMSNorm(q), RMSNorm(k) -> RoPE -> masked attention -> merge heads, and
autograd runs each step back. Here:

- forward: one pass normalizes and rotates q and k from the qkv projection's output (as inductor's fused kernel does)
  and stores them per head; the attention reads them and v (straight from qkv) and writes (tokens, H * D) for the
  output projection;
- backward: the dq and dk/dv kernels run the prefix kernels' loops on the stored q and k, then apply RoPE's and the
  RMSNorm's backward once per row in their epilogue and write d(qkv) in the projection's layout, plus partial sums
  of the norm weights' gradients. No inner loop recomputes a norm (an earlier version did, and lost on the H100).

Sequences are packed: groups of equal-length sequences, one after another in a flat token stream (B sequences of L
tokens are the special case), so one call can run views of different sizes together. Each kernel launches one
program per real block of rows (a table of (sequence, first row)), so short sequences waste no programs.

    out = qk_attention(qkv, q_weight, k_weight, cos, sin, kv_len, heads, eps)              # (B, L, 3HD) -> (B, L, HD)
    out = qk_attention_packed(qkv, q_weight, k_weight, cos, sin, kv_len, ((count, length), ...), heads, eps)

cos, sin: Attention.rope_tables, per token (float32); kv_len: attention.prefix_lengths, per token, counted from its
sequence's start; eps: the RMSNorms' (None in v2: the input dtype's machine epsilon, as nn.RMSNorm). Normalization in
float32 from the bf16 inputs; q and k stored in bf16, as the separate ops store them.
"""

import math
import os

import torch
import triton
import triton.language as tl

from .attention import LOG2E


BLOCK = int(os.environ.get("QK_BLOCK", 64))  # forward and prep: rows
BLOCK_N = int(os.environ.get("QK_BLOCK_N", 32))  # forward: keys


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
def _raw(X, C, S, tok, m, rd, xl, col, w, eps, D: tl.constexpr):
    """Raw rows of q or k (columns col..col + D of qkv), their rstd, and the RoPE tables at those tokens."""
    x = tl.load(X + tok[:, None] * xl + col + rd[None], mask=m, other=0.0).to(
        tl.float32
    )
    r = tl.rsqrt(tl.sum(x * x, 1) / D + eps)
    c = tl.load(C + tok[:, None] * D + rd[None], mask=m, other=0.0)
    s = tl.load(S + tok[:, None] * D + rd[None], mask=m, other=0.0)
    return x, r, c, s


@triton.jit
def _prep(
    QKV,
    WQ,
    WK,
    C,
    S,
    STARTS,
    LENS,
    TAB,
    QN,
    KN,
    H,
    N,
    eps,
    BM: tl.constexpr,
    D: tl.constexpr,
):
    pid, h = (
        tl.program_id(0),
        tl.program_id(1),
    )  # a block of one sequence (TAB: sequence, first row), a head
    sq, m0 = tl.load(TAB + 2 * pid), tl.load(TAB + 2 * pid + 1)
    t0, n = tl.load(STARTS + sq), tl.load(LENS + sq)
    if m0 < n:
        rd = tl.arange(0, D)
        rows = m0 + tl.arange(0, BM)
        m = rows[:, None] < n
        tok = t0 + rows
        xl = 3 * H * D
        for which in tl.static_range(2):
            w = tl.load((WQ if which == 0 else WK) + rd).to(tl.float32)
            x, r, c, s = _raw(
                QKV, C, S, tok, m, rd, xl, which * H * D + h * D, w, eps, D
            )
            xn = x * r[:, None] * w[None]
            y = xn * c + _rot(xn, BM, D) * s
            tl.store(
                (QN if which == 0 else KN) + (h * N + tok[:, None]) * D + rd[None],
                y.to(tl.bfloat16),
                mask=m,
            )


@triton.jit
def _fwd(
    QKV,
    QN,
    KN,
    KVL,
    STARTS,
    LENS,
    TAB,
    Out,
    LSE,
    H,
    N,
    qk_scale,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    pid, h = (
        tl.program_id(0),
        tl.program_id(1),
    )  # a block of one sequence (TAB: sequence, first row), a head
    sq, m0 = tl.load(TAB + 2 * pid), tl.load(TAB + 2 * pid + 1)
    t0, n = tl.load(STARTS + sq), tl.load(LENS + sq)
    if m0 < n:
        rd = tl.arange(0, D)
        rm = m0 + tl.arange(0, BM)
        mq = rm < n
        q = tl.load(
            QN + (h * N + t0 + rm[:, None]) * D + rd[None], mask=mq[:, None], other=0.0
        )
        kvl = tl.load(KVL + t0 + rm, mask=mq, other=0)
        hi = tl.max(kvl, 0)
        m_i = tl.full([BM], -1e30, tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, D], tl.float32)
        for k0 in range(
            0, hi, BN
        ):  # keys in blocks of BN: no row reaches the readout, so 32 for a local view
            rn = k0 + tl.arange(0, BN)
            mk = (rn < n)[:, None]
            k = tl.load(
                KN + (h * N + t0 + rn[:, None]) * D + rd[None], mask=mk, other=0.0
            )
            s = tl.dot(q, tl.trans(k)) * qk_scale
            s = tl.where(rn[None] < kvl[:, None], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp2(s - m_new[:, None])
            alpha = tl.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(
                QKV + (t0 + rn[:, None]) * 3 * H * D + 2 * H * D + h * D + rd[None],
                mask=mk,
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        tl.store(
            Out + (t0 + rm[:, None]) * H * D + h * D + rd[None],
            (acc / l_safe[:, None]).to(Out.dtype.element_ty),
            mask=mq[:, None],
        )
        tl.store(LSE + h * N + t0 + rm, m_i + tl.log2(l_safe), mask=mq)


@triton.jit
def _delta(Out, DO, STARTS, LENS, TAB, DELTA, H, N, BM: tl.constexpr, D: tl.constexpr):
    pid, h = (
        tl.program_id(0),
        tl.program_id(1),
    )  # a block of one sequence (TAB: sequence, first row), a head
    sq, m0 = tl.load(TAB + 2 * pid), tl.load(TAB + 2 * pid + 1)
    t0, n = tl.load(STARTS + sq), tl.load(LENS + sq)
    if m0 < n:
        rd = tl.arange(0, D)
        rm = m0 + tl.arange(0, BM)
        m = rm < n
        off = (t0 + rm[:, None]) * H * D + h * D + rd[None]
        o = tl.load(Out + off, mask=m[:, None], other=0.0).to(tl.float32)
        do = tl.load(DO + off, mask=m[:, None], other=0.0).to(tl.float32)
        tl.store(DELTA + h * N + t0 + rm, tl.sum(o * do, 1), mask=m)


@triton.jit
def _epilogue(
    g,
    QKV,
    WX,
    C,
    S,
    DQKV,
    DW,
    tok,
    m,
    rd,
    col,
    H,
    eps,
    part,
    BM: tl.constexpr,
    D: tl.constexpr,
):
    """Backward through RoPE and the RMSNorm, once per row.

    From g = d(normalized, rotated rows): d(raw rows) into DQKV's columns col..col + D, and this block's d(norm
    weight) into its partial row.
    """
    xl = 3 * H * D
    w = tl.load(WX + rd).to(tl.float32)
    x, r, c, s = _raw(QKV, C, S, tok, m, rd, xl, col, w, eps, D)
    gn = g * c + _rot_t(g * s, BM, D)  # through RoPE
    gw = gn * w[None]
    dx = r[:, None] * gw - x * (r * r * r)[:, None] * (tl.sum(gw * x, 1) / D)[:, None]
    tl.store(
        DQKV + tok[:, None] * xl + col + rd[None], dx.to(DQKV.dtype.element_ty), mask=m
    )
    # partials stored (D, programs): summed along the contiguous axis after (the other way round was 3.5% of an H100
    # step)
    tl.store(DW + rd * (tl.num_programs(0) * H) + part, tl.sum(gn * x * r[:, None], 0))


@triton.jit
def _bwd_dkdv(
    QKV,
    QN,
    KN,
    WK,
    C,
    S,
    KVL,
    STARTS,
    LENS,
    TAB,
    DO,
    LSE,
    DELTA,
    DQKV,
    DWK,
    H,
    N,
    eps,
    qk_scale,
    sm_scale,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    pid, h = (
        tl.program_id(0),
        tl.program_id(1),
    )  # a block of BN keys of one sequence (TAB: sequence, first key), a head; queries in blocks of BM
    sq, k0 = tl.load(TAB + 2 * pid), tl.load(TAB + 2 * pid + 1)
    t0, n = tl.load(STARTS + sq), tl.load(LENS + sq)
    rd = tl.arange(0, D)
    part = pid * H + h
    if k0 >= n:
        tl.store(DWK + rd * (tl.num_programs(0) * H) + part, tl.zeros([D], tl.float32))
    else:
        rn = k0 + tl.arange(0, BN)
        mk = (rn < n)[:, None]
        k = tl.load(KN + (h * N + t0 + rn[:, None]) * D + rd[None], mask=mk, other=0.0)
        v = tl.load(
            QKV + (t0 + rn[:, None]) * 3 * H * D + 2 * H * D + h * D + rd[None],
            mask=mk,
            other=0.0,
        )
        dk = tl.zeros([BN, D], tl.float32)
        dv = tl.zeros([BN, D], tl.float32)
        for m0 in range(0, n, BM):
            rm = m0 + tl.arange(0, BM)
            mq = rm < n
            kvl = tl.load(KVL + t0 + rm, mask=mq, other=0)
            if (
                tl.max(kvl, 0) > k0
            ):  # some row of this query block reaches this key block
                q = tl.load(
                    QN + (h * N + t0 + rm[:, None]) * D + rd[None],
                    mask=mq[:, None],
                    other=0.0,
                )
                do = tl.load(
                    DO + (t0 + rm[:, None]) * H * D + h * D + rd[None],
                    mask=mq[:, None],
                    other=0.0,
                )
                lse = tl.load(LSE + h * N + t0 + rm, mask=mq, other=0.0)
                delta = tl.load(DELTA + h * N + t0 + rm, mask=mq, other=0.0)
                sT = (
                    tl.dot(k, tl.trans(q)) * qk_scale
                )  # keys x queries: only loaded tiles are transposed
                pT = tl.where(rn[:, None] < kvl[None], tl.exp2(sT - lse[None]), 0.0)
                dv += tl.dot(pT.to(do.dtype), do)
                dpT = tl.dot(v, tl.trans(do))
                dsT = pT * (dpT - delta[None])
                dk += tl.dot(dsT.to(q.dtype), q)
        tok = t0 + rn
        xl = 3 * H * D
        tl.store(
            DQKV + tok[:, None] * xl + 2 * H * D + h * D + rd[None],
            dv.to(DQKV.dtype.element_ty),
            mask=mk,
        )
        _epilogue(
            dk * sm_scale,
            QKV,
            WK,
            C,
            S,
            DQKV,
            DWK,
            tok,
            mk,
            rd,
            H * D + h * D,
            H,
            eps,
            part,
            BN,
            D,
        )


@triton.jit
def _bwd_dq(
    QKV,
    QN,
    KN,
    WQ,
    C,
    S,
    KVL,
    STARTS,
    LENS,
    TAB,
    DO,
    LSE,
    DELTA,
    DQKV,
    DWQ,
    H,
    N,
    eps,
    qk_scale,
    sm_scale,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    pid, h = (
        tl.program_id(0),
        tl.program_id(1),
    )  # a block of one sequence (TAB: sequence, first row), a head; keys in blocks of BN
    sq, m0 = tl.load(TAB + 2 * pid), tl.load(TAB + 2 * pid + 1)
    t0, n = tl.load(STARTS + sq), tl.load(LENS + sq)
    rd = tl.arange(0, D)
    part = pid * H + h
    if m0 >= n:
        tl.store(DWQ + rd * (tl.num_programs(0) * H) + part, tl.zeros([D], tl.float32))
    else:
        rm = m0 + tl.arange(0, BM)
        mq = rm < n
        q = tl.load(
            QN + (h * N + t0 + rm[:, None]) * D + rd[None], mask=mq[:, None], other=0.0
        )
        do = tl.load(
            DO + (t0 + rm[:, None]) * H * D + h * D + rd[None],
            mask=mq[:, None],
            other=0.0,
        )
        lse = tl.load(LSE + h * N + t0 + rm, mask=mq, other=0.0)
        delta = tl.load(DELTA + h * N + t0 + rm, mask=mq, other=0.0)
        kvl = tl.load(KVL + t0 + rm, mask=mq, other=0)
        hi = tl.max(kvl, 0)
        dq = tl.zeros([BM, D], tl.float32)
        for k0 in range(0, hi, BN):
            rn = k0 + tl.arange(0, BN)
            mk = (rn < n)[:, None]
            k = tl.load(
                KN + (h * N + t0 + rn[:, None]) * D + rd[None], mask=mk, other=0.0
            )
            v = tl.load(
                QKV + (t0 + rn[:, None]) * 3 * H * D + 2 * H * D + h * D + rd[None],
                mask=mk,
                other=0.0,
            )
            s = tl.dot(q, tl.trans(k)) * qk_scale
            p = tl.where(rn[None] < kvl[:, None], tl.exp2(s - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k)
        _epilogue(
            dq * sm_scale,
            QKV,
            WQ,
            C,
            S,
            DQKV,
            DWQ,
            t0 + rm,
            mq[:, None],
            rd,
            h * D,
            H,
            eps,
            part,
            BM,
            D,
        )


def _tables(lengths, block, device):
    """Per sequence its start and length; per block of `block` rows, (sequence, first row).

    One program per real block, however the lengths differ. lengths: ((count, length), ...), Python ints, so no host
    sync; plain tensor ops, so capturable.
    """
    starts, lens, tab, at, seq = [], [], [], 0, 0
    for count, length in lengths:
        nb = triton.cdiv(length, block)
        ids = torch.arange(count, device=device, dtype=torch.int32)
        starts.append(at + ids * length)
        lens.append(torch.full((count,), length, device=device, dtype=torch.int32))
        rows = torch.arange(nb, device=device, dtype=torch.int32) * block
        tab.append(
            torch.stack([(seq + ids).repeat_interleave(nb), rows.repeat(count)], dim=1)
        )
        at, seq = at + count * length, seq + count
    return torch.cat(starts), torch.cat(lens), torch.cat(tab).contiguous()


def _tiles():
    """(backward rows, backward keys, stages) for this GPU.

    Bigger tiles and deeper pipelines where shared memory allows (sm90: 228 KB); 64-row backward tiles spill registers
    on Ampere and Ada (measured on A40 and L40S). The dK/dV kernel's key tiles stay as tall as its query tiles: 32-key
    tiles cost the H100 0.4 ms per ViT-S step (too short for its wgmma), though the forward gains from them.
    """
    big = torch.cuda.get_device_capability()[0] >= 9
    env = os.environ.get
    rows = int(env("QK_BLOCK_B", 64 if big else 32))
    return rows, int(env("QK_BLOCK_BN", rows)), int(env("QK_STAGES", 3 if big else 2))


@torch.library.custom_op("levjepa::qk_attention_fwd", mutates_args=())
def _qk_fwd(
    qkv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_len: torch.Tensor,
    starts: torch.Tensor,
    lens: torch.Tensor,
    tab: torch.Tensor,
    tab_b: torch.Tensor,
    tab_k: torch.Tensor,
    heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N, three_hd = qkv.shape
    D = three_hd // (3 * heads)
    qn = torch.empty(heads, N, D, device=qkv.device, dtype=qkv.dtype)
    kn = torch.empty_like(qn)
    out = torch.empty(N, heads * D, device=qkv.device, dtype=qkv.dtype)
    lse = torch.empty(heads, N, device=qkv.device, dtype=torch.float32)
    grid = (tab.shape[0], heads)
    _prep[grid](
        qkv,
        wq,
        wk,
        cos,
        sin,
        starts,
        lens,
        tab,
        qn,
        kn,
        heads,
        N,
        eps,
        BM=BLOCK,
        D=D,
        num_warps=4,
    )
    _fwd[grid](
        qkv, qn, kn, kv_len, starts, lens, tab, out, lse, heads, N, LOG2E / math.sqrt(D), BM=BLOCK, BN=BLOCK_N,
        D=D, num_warps=4, num_stages=_tiles()[2],
    )  # fmt: skip
    return out, lse, qn, kn


@_qk_fwd.register_fake
def _(qkv, wq, wk, cos, sin, kv_len, starts, lens, tab, tab_b, tab_k, heads, eps):
    N, three_hd = qkv.shape
    D = three_hd // (3 * heads)
    return (
        qkv.new_empty(N, heads * D),
        qkv.new_empty(heads, N, dtype=torch.float32),
        qkv.new_empty(heads, N, D),
        qkv.new_empty(heads, N, D),
    )


@torch.library.custom_op("levjepa::qk_attention_bwd", mutates_args=())
def _qk_bwd(
    do: torch.Tensor,
    qkv: torch.Tensor,
    qn: torch.Tensor,
    kn: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_len: torch.Tensor,
    starts: torch.Tensor,
    lens: torch.Tensor,
    tab: torch.Tensor,
    tab_k: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N, three_hd = qkv.shape
    D = three_hd // (3 * heads)
    bm, bn, stages = _tiles()
    do = do.contiguous()
    grid, grid_k = (tab.shape[0], heads), (tab_k.shape[0], heads)
    delta = torch.empty(heads, N, device=qkv.device, dtype=torch.float32)
    _delta[grid](out, do, starts, lens, tab, delta, heads, N, BM=bm, D=D, num_warps=4)
    dqkv = torch.empty_like(qkv)
    dwq = torch.empty(D, grid[0] * heads, device=qkv.device, dtype=torch.float32)
    dwk = torch.empty(D, grid_k[0] * heads, device=qkv.device, dtype=torch.float32)
    common = (kv_len, starts, lens)
    rest = (do, lse, delta, dqkv)
    scales = (heads, N, eps, LOG2E / math.sqrt(D), 1 / math.sqrt(D))
    _bwd_dkdv[grid_k](qkv, qn, kn, wk, cos, sin, *common, tab_k, *rest, dwk, *scales, BM=bm, BN=bn, D=D,
                      num_warps=4, num_stages=stages)  # fmt: skip
    _bwd_dq[grid](qkv, qn, kn, wq, cos, sin, *common, tab, *rest, dwq, *scales, BM=bm, BN=bn, D=D,
                  num_warps=4, num_stages=stages)  # fmt: skip
    return dqkv, dwq.sum(1).to(wq.dtype), dwk.sum(1).to(wk.dtype)


@_qk_bwd.register_fake
def _(
    do,
    qkv,
    qn,
    kn,
    wq,
    wk,
    cos,
    sin,
    kv_len,
    starts,
    lens,
    tab,
    tab_k,
    out,
    lse,
    heads,
    eps,
):
    return torch.empty_like(qkv), torch.empty_like(wq), torch.empty_like(wk)


def _setup(ctx, inputs, output):
    qkv, wq, wk, cos, sin, kv_len, starts, lens, tab, tab_b, tab_k, heads, eps = inputs
    out, lse, qn, kn = output
    ctx.save_for_backward(
        qkv, qn, kn, wq, wk, cos, sin, kv_len, starts, lens, tab_b, tab_k, out, lse
    )
    ctx.args = (heads, eps)


def _backward(ctx, dout, dlse, dqn, dkn):
    dqkv, dwq, dwk = _qk_bwd(dout, *ctx.saved_tensors, *ctx.args)
    return dqkv, dwq, dwk, None, None, None, None, None, None, None, None, None, None


_qk_fwd.register_autograd(_backward, setup_context=_setup)


def qk_attention_packed(
    qkv, q_weight, k_weight, cos, sin, kv_len, lengths, heads, eps=None
):
    """Attention.forward's core over packed sequences: qkv (N, 3 * H * D) -> (N, H * D).

    RMSNorm(q), RMSNorm(k), RoPE and, per sequence, attention to keys j < kv_len[i] (counted from the sequence's
    start). The sequences lie one after another: lengths ((count, length), ...) in order, Python ints (e.g.
    ((64, 161), (256, 33)) for 64 global and 256 local views). cos, sin (N, D) float32; kv_len (N,) int32 >= 1. D a
    power of two >= 16.
    """
    eps = torch.finfo(qkv.dtype).eps if eps is None else eps
    starts, lens, tab = _tables(lengths, BLOCK, qkv.device)
    bm, bn, _ = _tiles()
    tab_b = _tables(lengths, bm, qkv.device)[2]
    tab_k = _tables(lengths, bn, qkv.device)[2]
    args = (
        qkv.contiguous(),
        q_weight,
        k_weight,
        cos.contiguous(),
        sin.contiguous(),
        kv_len.contiguous(),
    )
    return _qk_fwd(*args, starts, lens, tab, tab_b, tab_k, heads, eps)[0]


def qk_attention(qkv, q_weight, k_weight, cos, sin, kv_len, heads, eps=None):
    """Attention.forward's core: (B, L, 3 * H * D) qkv -> (B, L, H * D).

    cos, sin (1 or B, 1, L, D) float32; kv_len (B, L) int32.
    """
    B, L, _ = qkv.shape
    D = qkv.shape[-1] // (3 * heads)
    cos, sin = (t.expand(B, 1, L, D).reshape(B * L, D) for t in (cos, sin))
    out = qk_attention_packed(
        qkv.reshape(B * L, -1),
        q_weight,
        k_weight,
        cos,
        sin,
        kv_len.reshape(-1),
        ((B, L),),
        heads,
        eps,
    )
    return out.view(B, L, heads * D)
