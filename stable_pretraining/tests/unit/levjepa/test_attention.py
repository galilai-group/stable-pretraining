"""stable_pretraining.data.levjepa.kernels.attention against scaled_dot_product_attention with v2's build_mask, forward and backward.

pytest tests/levjepa_tokens/test_attention.py      (needs a GPU)
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

from stable_pretraining.data.levjepa.kernels.attention import (
    prefix_attention,
    prefix_lengths,
)


T = 16


def frames(B, per_frame, kept, seed):
    """frame_id (B, L) of iid kept tokens sorted by frame, then the readout (frame T - 1), as Encoder.assemble."""
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


def build_mask(frame_id, window=16):
    """Encoder.build_mask."""
    is_readout = torch.zeros(frame_id.shape[1], dtype=torch.bool, device="cuda")
    is_readout[-1] = True
    ti, tj = frame_id[:, :, None], frame_id[:, None, :]
    return ((tj <= ti) & (tj > ti - window) & ~is_readout)[:, None]


@pytest.mark.parametrize(
    "B,per_frame,kept", [(4, 196, 160), (8, 36, 32), (3, 196, 40), (2, 64, 300)]
)
def test_matches_masked_sdpa(B, per_frame, kept):
    frame_id = frames(B, per_frame, kept, seed=kept)
    L = frame_id.shape[1]
    g = torch.Generator(device="cuda").manual_seed(0)
    # (B, L, 3, H, D) -> q, k, v (B, H, L, D) strided views, as Attention.forward's rearrange gives
    qkv = (
        torch.randn(B, L, 3, 6, 64, device="cuda", generator=g)
        .to(torch.bfloat16)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = (x.detach().requires_grad_() for x in qkv)
    go = torch.randn(B, 6, L, 64, device="cuda", generator=g).to(torch.bfloat16)
    out = prefix_attention(q, k, v, prefix_lengths(frame_id, L - 1))
    out.backward(go)
    q32, k32, v32 = (x.detach().float().requires_grad_() for x in (q, k, v))
    ref = F.scaled_dot_product_attention(q32, k32, v32, attn_mask=build_mask(frame_id))
    ref.backward(go.float())
    assert (out.float() - ref).abs().max() < 2e-2
    for got, want in [(q.grad, q32.grad), (k.grad, k32.grad), (v.grad, v32.grad)]:
        assert (got.float() - want).abs().max() < 3e-2 * max(
            1.0, want.abs().max().item()
        )


def test_prefix_lengths():
    frame_id = torch.tensor([[0, 0, 2, 3, 3, 15]], device="cuda")
    assert prefix_lengths(frame_id, 5).tolist() == [[2, 2, 3, 5, 5, 5]]


def test_compiles_and_captures():
    frame_id = frames(4, 36, 32, seed=1)
    L = frame_id.shape[1]
    kv_len = prefix_lengths(frame_id, L - 1)
    x = torch.randn(
        4, 6, L, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    @torch.compile(fullgraph=True)
    def f(x):
        return prefix_attention(x * 2, x, x, kv_len).float().square().sum()

    f(x).backward()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        f(x).backward()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loss = f(x)
        loss.backward()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(x.grad).all()
