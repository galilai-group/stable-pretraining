"""stable_pretraining.data.levjepa.kernels.swiglu against v2's SwiGLU ops: forward and backward (x and the three weights).

pytest tests/levjepa_tokens/test_swiglu.py      (needs a GPU)
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

from stable_pretraining.data.levjepa.kernels.swiglu import swiglu


def reference(x, wg, wu, wd):
    return F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd)


@pytest.mark.parametrize("N, D", [(1000, 384), (4133, 768)])
def test_matches_separate_ops(N, D):
    torch.manual_seed(0)
    H = int(8 / 3 * D)
    x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    ws = [
        torch.randn(*s, device="cuda", dtype=torch.bfloat16) / s[1] ** 0.5
        for s in ((H, D), (H, D), (D, H))
    ]
    dy = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)

    def run(fn):
        leaves = [t.detach().requires_grad_() for t in (x, *ws)]
        y = fn(*leaves)
        y.backward(dy)
        return y, *(t.grad for t in leaves)

    # the reference in float32 from the same bf16 inputs; ours and the bf16 ops both within bf16 rounding of it
    exact = run(lambda *a: reference(*(t.float() for t in a)))
    ours, ops = run(swiglu), run(reference)
    for name, e, a, b in zip(("y", "dx", "dwg", "dwu", "dwd"), exact, ours, ops):
        err_ours = (a.float() - e).norm() / e.norm()
        err_ops = (b.float() - e).norm() / e.norm()
        assert err_ours <= 1.5 * err_ops + 1e-3, (name, err_ours.item(), err_ops.item())


def test_under_autocast_and_leading_dims():
    torch.manual_seed(0)
    x = torch.randn(4, 50, 384, device="cuda", requires_grad=True)
    ws = [
        (torch.randn(*s, device="cuda") / s[1] ** 0.5).requires_grad_()
        for s in ((1024, 384),) * 2 + ((384, 1024),)
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, ref = swiglu(x, *ws), reference(x, *ws)
    assert y.shape == x.shape and y.dtype == ref.dtype
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)
    y.float().sum().backward()
    assert all(t.grad is not None and t.grad.dtype == torch.float32 for t in (x, *ws))
