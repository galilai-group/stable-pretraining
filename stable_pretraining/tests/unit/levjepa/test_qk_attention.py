"""stable_pretraining.data.levjepa.kernels.qk_attention against v2's Attention.forward ops.

RMSNorm(q), RMSNorm(k), RoPE, masked SDPA; forward and backward (qkv and both norm weights).

    pytest tests/levjepa_tokens/test_qk_attention.py      (needs a GPU)
"""

# ruff: noqa: E402  (the imports follow the importorskip checks)

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

import torch.nn.functional as F

from stable_pretraining.data.levjepa.kernels.attention import prefix_lengths
from stable_pretraining.data.levjepa.kernels.qk_attention import (
    qk_attention,
    qk_attention_packed,
)


T, H, D = 16, 6, 64


def frames(B, per_frame, kept, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    idx = (
        torch.rand(B, T * per_frame, device="cuda", generator=g)
        .topk(kept, dim=-1)
        .indices.sort(dim=-1)
        .values
    )
    return torch.cat(
        [idx // per_frame, torch.full((B, 1), T - 1, device="cuda")], dim=1
    )


def rope_tables(pos):
    """Attention.rope_tables (head_dim 64: sections 32, 16, 16 for time, row, column)."""
    cos, sin = [], []
    for axis, d in enumerate([D // 2, D // 4, D // 4]):
        omega = 1.0 / 10000 ** (
            torch.arange(d // 2, device="cuda", dtype=torch.float32) / (d // 2)
        )
        freq = pos[..., axis].float()[:, None, :, None] * omega
        cos.append(freq.cos().repeat_interleave(2, dim=-1))
        sin.append(freq.sin().repeat_interleave(2, dim=-1))
    return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)


def rope(x, cos, sin):
    y = x.unflatten(-1, (-1, 2))
    y = torch.stack((-y[..., 1], y[..., 0]), dim=-1).flatten(-2)
    return x * cos + y * sin


def reference(qkv, wq, wk, cos, sin, frame_id, eps):
    B, L, _ = qkv.shape
    q, k, v = qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
    q = rope(F.rms_norm(q, (D,), wq, eps), cos, sin)
    k = rope(F.rms_norm(k, (D,), wk, eps), cos, sin)
    is_readout = torch.zeros(L, dtype=torch.bool, device="cuda")
    is_readout[-1] = True
    mask = ((frame_id[:, None, :] <= frame_id[:, :, None]) & ~is_readout)[:, None]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).reshape(B, L, H * D)


@pytest.mark.parametrize("B,per_frame,kept", [(4, 196, 160), (8, 36, 32), (2, 64, 300)])
def test_matches_v2_attention(B, per_frame, kept):
    frame_id = frames(B, per_frame, kept, seed=kept)
    L = frame_id.shape[1]
    g = torch.Generator(device="cuda").manual_seed(0)
    pos = torch.stack(
        [
            frame_id,
            torch.randint(0, 15, (B, L), device="cuda", generator=g),
            torch.randint(0, 14, (B, L), device="cuda", generator=g),
        ],
        dim=-1,
    )
    cos, sin = rope_tables(pos)
    qkv = (
        (2 * torch.randn(B, L, 3 * H * D, device="cuda", generator=g))
        .to(torch.bfloat16)
        .requires_grad_()
    )
    wq = (
        (1 + 0.1 * torch.randn(D, device="cuda", generator=g))
        .to(torch.bfloat16)
        .requires_grad_()
    )
    wk = (
        (1 + 0.1 * torch.randn(D, device="cuda", generator=g))
        .to(torch.bfloat16)
        .requires_grad_()
    )
    go = torch.randn(B, L, H * D, device="cuda", generator=g).to(torch.bfloat16)
    eps = torch.finfo(torch.bfloat16).eps
    out = qk_attention(qkv, wq, wk, cos, sin, prefix_lengths(frame_id, L - 1), H)
    out.backward(go)
    leaves = [x.detach().float().requires_grad_() for x in (qkv, wq, wk)]
    ref = reference(*leaves, cos, sin, frame_id, eps)
    ref.backward(go.float())
    assert (out.float() - ref).abs().max() < 3e-2
    for got, want in [
        (qkv.grad, leaves[0].grad),
        (wq.grad, leaves[1].grad),
        (wk.grad, leaves[2].grad),
    ]:
        err = (got.float() - want).abs().max().item()
        assert err < 3e-2 * max(1.0, want.abs().max().item()), err


def test_packed_sequences_match_separate_calls():
    """Global (161 tokens) and local (33 tokens) sequences in one packed call, as each run alone through v2's ops."""
    g = torch.Generator(device="cuda").manual_seed(1)
    groups = [frames(2, 196, 160, seed=3), frames(3, 36, 32, seed=4)]
    eps = torch.finfo(torch.bfloat16).eps
    wq = (
        (1 + 0.1 * torch.randn(D, device="cuda", generator=g))
        .to(torch.bfloat16)
        .requires_grad_()
    )
    wk = (
        (1 + 0.1 * torch.randn(D, device="cuda", generator=g))
        .to(torch.bfloat16)
        .requires_grad_()
    )
    qkvs, tables, outs = [], [], []
    for frame_id in groups:
        B, L = frame_id.shape
        pos = torch.stack(
            [frame_id, torch.randint(0, 15, (B, L), device="cuda", generator=g),
             torch.randint(0, 14, (B, L), device="cuda", generator=g)], dim=-1,
        )  # fmt: skip
        tables.append(rope_tables(pos))
        qkvs.append(
            (2 * torch.randn(B, L, 3 * H * D, device="cuda", generator=g)).to(
                torch.bfloat16
            )
        )
    flat = torch.cat([x.reshape(-1, 3 * H * D) for x in qkvs]).requires_grad_()
    cos = torch.cat([c.reshape(-1, D) for c, _ in tables])
    sin = torch.cat([s.reshape(-1, D) for _, s in tables])
    kv_len = torch.cat([prefix_lengths(f, f.shape[1] - 1).reshape(-1) for f in groups])
    out = qk_attention_packed(
        flat, wq, wk, cos, sin, kv_len, tuple(tuple(f.shape) for f in groups), H
    )
    go = torch.randn_like(out)
    out.backward(go)
    leaves = [flat.detach().float().requires_grad_(), wq.detach().float().requires_grad_(),
              wk.detach().float().requires_grad_()]  # fmt: skip
    at = 0
    for frame_id, (c, s) in zip(groups, tables):
        B, L = frame_id.shape
        outs.append(
            reference(
                leaves[0][at : at + B * L].view(B, L, -1),
                leaves[1],
                leaves[2],
                c,
                s,
                frame_id,
                eps,
            )
        )
        at += B * L
    ref = torch.cat([o.reshape(-1, H * D) for o in outs])
    ref.backward(go.float())
    assert (out.float() - ref).abs().max() < 3e-2
    for got, want in [
        (flat.grad, leaves[0].grad),
        (wq.grad, leaves[1].grad),
        (wk.grad, leaves[2].grad),
    ]:
        err = (got.float() - want).abs().max().item()
        assert err < 3e-2 * max(1.0, want.abs().max().item()), err
