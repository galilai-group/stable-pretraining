"""stable_pretraining.data.levjepa.kernels.adamw against clip_grad_norm_ + AdamW(fused=True) + an EMA lerp + a bf16 copy.

pytest tests/levjepa_tokens/test_adamw.py      (needs a GPU)
"""

# ruff: noqa: E402  (the imports follow the importorskip checks)

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

from stable_pretraining.data.levjepa.kernels.adamw import FusedStep


SHAPES = [(384, 1152), (1152,), (384,), (2048, 384), (7,), (256, 2048)]
HYPER = {"lr": 4e-4, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.04}


def make(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(*s, device="cuda", generator=g) for s in SHAPES]


# gradients under and over the clipping norm
@pytest.mark.parametrize("scale", [1e-3, 10.0])
def test_matches_torch(scale):
    params = [p.requires_grad_() for p in make(0)]
    decay = [p.ndim > 1 for p in params]
    ref = [p.detach().clone().requires_grad_() for p in params]
    k = 4  # the teacher and the bf16 copy cover the first k params, as the encoder does in train_v2
    teacher, ref_teacher = make(1)[:k], [t.clone() for t in make(1)[:k]]
    low = [p.detach().to(torch.bfloat16) for p in params[:k]]
    fused = FusedStep(
        params, decay, teacher=teacher, copies=low, max_norm=1.0, ema=0.99, **HYPER
    )
    opt = torch.optim.AdamW(
        [
            {"params": [p for p, d in zip(ref, decay) if d]},
            {"params": [p for p, d in zip(ref, decay) if not d], "weight_decay": 0.0},
        ],
        fused=True,
        **HYPER,
    )
    for step in range(5):
        grads = [x * scale for x in make(10 + step)]
        torch._foreach_copy_(fused.grads, grads)
        fused.step()
        for p, x in zip(ref, grads):
            p.grad = x.clone()
        torch.nn.utils.clip_grad_norm_(ref, 1.0, foreach=True)
        opt.step()
        torch._foreach_lerp_(ref_teacher, ref[:k], 1 - 0.99)
    for a, b in zip(params, ref):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=1e-5, atol=1e-6)
    for a, b in zip(teacher, ref_teacher):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    for a, b in zip(low, ref[:k]):
        assert torch.equal(a, b.detach().to(torch.bfloat16)) or torch.allclose(
            a.float(), b.detach(), rtol=1e-2
        )
