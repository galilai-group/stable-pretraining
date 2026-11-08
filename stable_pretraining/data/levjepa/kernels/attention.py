"""Attention under LeVJEPA v2's training mask, as one Triton kernel per pass instead of masked SDPA.

v2's encoder lets token i attend to token j when frame(j) <= frame(i) and frame(j) > frame(i) - window, never to the
readout (Encoder.build_mask). Training views keep few tokens per frame at iid positions, so the mask differs per
sample and SDPA falls back to its memory-efficient kernel with a dense (B, 1, L, L) mask. The kept tokens are sorted
by frame and the readout comes last, so while the window covers the clip (window >= T, v2's 16 frames) every row's
allowed keys are a prefix: j < kv_len[i], kv_len[i] the number of patch tokens in frames <= frame(i). These kernels
take that one count per row (FlashAttention-2 forward and backward, with the bound per row instead of causal), skip
the key blocks past a query block's longest prefix and never read a mask.

    kv_len = prefix_lengths(frame_id, S)     # (B, L) int32, frame_id (B, L) sorted with the readout last
    out = prefix_attention(q, k, v, kv_len)  # q, k, v (B, H, L, D) as for scaled_dot_product_attention

A torch custom op (opaque to torch.compile, capturable in a CUDA graph), with its backward.
"""

import math
import os

import torch
import triton
import triton.language as tl


LOG2E = 1.4426950408889634


def prefix_lengths(frame_id, S):
    """kv_len (B, L) int32 for build_mask(frame_id, is_readout, window >= T).

    Per token, the patch tokens (the first S, sorted by frame) in frames up to its own. frame_id (1 or B, L).
    """
    f = frame_id.to(torch.int32)
    return torch.searchsorted(f[:, :S].contiguous(), f.contiguous(), right=True).to(
        torch.int32
    )


def _block(L):
    return 64


@triton.jit
def _fwd(
    Q,
    K,
    V,
    Out,
    LSE,
    KVL,
    sqb,
    sqh,
    sql,
    skb,
    skh,
    skl,
    svb,
    svh,
    svl,
    sob,
    soh,
    sol,
    H,
    L,
    qk_scale,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    m0 = tl.program_id(0) * BM
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    rm, rd = m0 + tl.arange(0, BM), tl.arange(0, D)
    q_off = b * sqb + h * sqh
    k_off = b * skb + h * skh
    v_off = b * svb + h * svh
    q = tl.load(
        Q + q_off + rm[:, None] * sql + rd[None], mask=rm[:, None] < L, other=0.0
    )
    kvl = tl.load(KVL + b * L + rm, mask=rm < L, other=0)
    hi = tl.max(kvl, 0)
    m_i = tl.full([BM], -1e30, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in range(0, hi, BN):
        rn = n0 + tl.arange(0, BN)
        k = tl.load(
            K + k_off + rn[:, None] * skl + rd[None], mask=rn[:, None] < L, other=0.0
        )
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(rn[None] < kvl[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            V + v_off + rn[:, None] * svl + rd[None], mask=rn[:, None] < L, other=0.0
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o_off = b * sob + h * soh
    tl.store(
        Out + o_off + rm[:, None] * sol + rd[None],
        (acc / l_safe[:, None]).to(Out.dtype.element_ty),
        mask=rm[:, None] < L,
    )
    tl.store(
        LSE + bh * L + rm, m_i + tl.log2(l_safe), mask=rm < L
    )  # base 2, of the scaled scores


@triton.jit
def _bwd_dkdv(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    KVL,
    DK,
    DV,
    sqb,
    sqh,
    sql,
    skb,
    skh,
    skl,
    svb,
    svh,
    svl,
    sdob,
    sdoh,
    sdol,
    H,
    L,
    qk_scale,
    sm_scale,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    n0 = tl.program_id(0) * BN
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    rn, rd = n0 + tl.arange(0, BN), tl.arange(0, D)
    k_off = b * skb + h * skh
    v_off = b * svb + h * svh
    k = tl.load(
        K + k_off + rn[:, None] * skl + rd[None], mask=rn[:, None] < L, other=0.0
    )
    v = tl.load(
        V + v_off + rn[:, None] * svl + rd[None], mask=rn[:, None] < L, other=0.0
    )
    q_off = b * sqb + h * sqh
    do_off = b * sdob + h * sdoh
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    for m0 in range(0, L, BM):
        rm = m0 + tl.arange(0, BM)
        kvl = tl.load(KVL + b * L + rm, mask=rm < L, other=0)
        if tl.max(kvl, 0) > n0:  # some row of this query block reaches this key block
            q = tl.load(
                Q + q_off + rm[:, None] * sql + rd[None],
                mask=rm[:, None] < L,
                other=0.0,
            )
            do = tl.load(
                DO + do_off + rm[:, None] * sdol + rd[None],
                mask=rm[:, None] < L,
                other=0.0,
            )
            lse = tl.load(LSE + bh * L + rm, mask=rm < L, other=0.0)
            delta = tl.load(DELTA + bh * L + rm, mask=rm < L, other=0.0)
            # transposed scores (keys x queries): only the loaded q and do tiles are transposed, not p and ds
            sT = tl.dot(k, tl.trans(q)) * qk_scale
            pT = tl.where(rn[:, None] < kvl[None], tl.exp2(sT - lse[None]), 0.0)
            dv += tl.dot(pT.to(do.dtype), do)
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None])
            dk += tl.dot(dsT.to(q.dtype), q)
    st = bh * L * D  # dk, dv are (B, H, L, D) contiguous
    tl.store(
        DK + st + rn[:, None] * D + rd[None],
        (dk * sm_scale).to(DK.dtype.element_ty),
        mask=rn[:, None] < L,
    )
    tl.store(
        DV + st + rn[:, None] * D + rd[None],
        dv.to(DV.dtype.element_ty),
        mask=rn[:, None] < L,
    )


@triton.jit
def _bwd_dq(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    KVL,
    DQ,
    sqb,
    sqh,
    sql,
    skb,
    skh,
    skl,
    svb,
    svh,
    svl,
    sdob,
    sdoh,
    sdol,
    H,
    L,
    qk_scale,
    sm_scale,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    m0 = tl.program_id(0) * BM
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    rm, rd = m0 + tl.arange(0, BM), tl.arange(0, D)
    q_off = b * sqb + h * sqh
    do_off = b * sdob + h * sdoh
    q = tl.load(
        Q + q_off + rm[:, None] * sql + rd[None], mask=rm[:, None] < L, other=0.0
    )
    do = tl.load(
        DO + do_off + rm[:, None] * sdol + rd[None], mask=rm[:, None] < L, other=0.0
    )
    lse = tl.load(LSE + bh * L + rm, mask=rm < L, other=0.0)
    delta = tl.load(DELTA + bh * L + rm, mask=rm < L, other=0.0)
    kvl = tl.load(KVL + b * L + rm, mask=rm < L, other=0)
    hi = tl.max(kvl, 0)
    k_off = b * skb + h * skh
    v_off = b * svb + h * svh
    dq = tl.zeros([BM, D], tl.float32)
    for n0 in range(0, hi, BN):
        rn = n0 + tl.arange(0, BN)
        k = tl.load(
            K + k_off + rn[:, None] * skl + rd[None], mask=rn[:, None] < L, other=0.0
        )
        v = tl.load(
            V + v_off + rn[:, None] * svl + rd[None], mask=rn[:, None] < L, other=0.0
        )
        s = tl.dot(q, tl.trans(k)) * qk_scale
        p = tl.where(rn[None] < kvl[:, None], tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(
        DQ + bh * L * D + rm[:, None] * D + rd[None],
        (dq * sm_scale).to(DQ.dtype.element_ty),
        mask=rm[:, None] < L,
    )


@triton.jit
def _delta(
    Out,
    DO,
    DELTA,
    sob,
    soh,
    sol,
    sdob,
    sdoh,
    sdol,
    H,
    L,
    BM: tl.constexpr,
    D: tl.constexpr,
):
    m0 = tl.program_id(0) * BM
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    rm, rd = m0 + tl.arange(0, BM), tl.arange(0, D)
    o = tl.load(
        Out + b * sob + h * soh + rm[:, None] * sol + rd[None],
        mask=rm[:, None] < L,
        other=0.0,
    )
    do = tl.load(
        DO + b * sdob + h * sdoh + rm[:, None] * sdol + rd[None],
        mask=rm[:, None] < L,
        other=0.0,
    )
    tl.store(
        DELTA + bh * L + rm,
        tl.sum(o.to(tl.float32) * do.to(tl.float32), 1),
        mask=rm < L,
    )


def _strides(*xs):
    """(batch, head, seq) strides of each (B, H, L, D) tensor, flat."""
    return [s for x in xs for s in x.stride()[:3]]


def _last_contiguous(x):
    return x if x.stride(-1) == 1 else x.contiguous()


@torch.library.custom_op("levjepa::prefix_attention_fwd", mutates_args=())
def _prefix_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_len: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v = map(_last_contiguous, (q, k, v))
    B, H, L, D = q.shape
    o = torch.empty(B, H, L, D, device=q.device, dtype=q.dtype)
    lse = torch.empty(B, H, L, device=q.device, dtype=torch.float32)
    st = _strides(q, k, v, o)
    bm = _block(L)
    _fwd[(triton.cdiv(L, bm), B * H)](
        q,
        k,
        v,
        o,
        lse,
        kv_len,
        *st,
        H,
        L,
        LOG2E / math.sqrt(D),
        BM=bm,
        BN=bm,
        D=D,
        num_warps=4,
    )
    return o, lse


@_prefix_fwd.register_fake
def _(q, k, v, kv_len):
    B, H, L, D = q.shape
    return q.new_empty(B, H, L, D), q.new_empty(B, H, L, dtype=torch.float32)


@torch.library.custom_op("levjepa::prefix_attention_bwd", mutates_args=())
def _prefix_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    kv_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, do = map(_last_contiguous, (q, k, v, do))
    B, H, L, D = q.shape
    dq, dk, dv = (
        torch.empty(B, H, L, D, device=q.device, dtype=q.dtype) for _ in range(3)
    )
    delta = torch.empty(B, H, L, device=q.device, dtype=torch.float32)
    bm = int(os.environ.get("PA_BLOCK_B", 32))  # 64 spills registers
    _delta[(triton.cdiv(L, bm), B * H)](
        o, do, delta, *_strides(o, do), H, L, BM=bm, D=D, num_warps=4
    )
    st = _strides(q, k, v, do)
    args = (H, L, LOG2E / math.sqrt(D), 1 / math.sqrt(D))
    _bwd_dkdv[(triton.cdiv(L, bm), B * H)](
        q,
        k,
        v,
        do,
        lse,
        delta,
        kv_len,
        dk,
        dv,
        *st,
        *args,
        BM=bm,
        BN=bm,
        D=D,
        num_warps=int(os.environ.get("NW_KV", 4)),
        num_stages=int(os.environ.get("NS_KV", 3)),
    )
    _bwd_dq[(triton.cdiv(L, bm), B * H)](
        q,
        k,
        v,
        do,
        lse,
        delta,
        kv_len,
        dq,
        *st,
        *args,
        BM=bm,
        BN=bm,
        D=D,
        num_warps=int(os.environ.get("NW_Q", 4)),
        num_stages=int(os.environ.get("NS_Q", 3)),
    )
    return dq, dk, dv


@_prefix_bwd.register_fake
def _(do, q, k, v, o, lse, kv_len):
    return q.new_empty(q.shape), q.new_empty(q.shape), q.new_empty(q.shape)


def _setup(ctx, inputs, output):
    q, k, v, kv_len = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, kv_len)


def _backward(ctx, do, dlse):
    q, k, v, o, lse, kv_len = ctx.saved_tensors
    dq, dk, dv = _prefix_bwd(do, q, k, v, o, lse, kv_len)
    return dq, dk, dv, None


_prefix_fwd.register_autograd(_backward, setup_context=_setup)


def prefix_attention(q, k, v, kv_len):
    """softmax(q k^T / sqrt(D)) v with token i attending to keys j < kv_len[b, i].

    q, k, v (B, H, L, D), last dim contiguous, D a power of two >= 16; kv_len (B, L) int32, >= 1 per row.
    """
    return _prefix_fwd(q, k, v, kv_len)[0]
