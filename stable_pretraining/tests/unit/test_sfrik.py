"""Numerical and public-interface checks for two-view SFRIK."""

import pickle
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import stable_pretraining as spt
from stable_pretraining.losses import SFRIKLoss

pytestmark = pytest.mark.unit


def test_method_survives_pickle_round_trip():
    model = spt.SFRIK(backbone=nn.Linear(4, 6), projector=nn.Linear(6, 3), sim_coeff=3)
    restored = pickle.loads(pickle.dumps(model))
    model.eval()
    restored.eval()
    batch = {"image": torch.randn(5, 4)}
    torch.testing.assert_close(
        restored(batch, stage="validate")["embedding"],
        model(batch, stage="validate")["embedding"],
    )
    assert restored.sfrik_loss.sim_coeff == 3


def _reference(x, y, alpha=4.0, beta=1.0, weights=(1.0, 40.0, 40.0)):
    # Evaluate via moments of tensor products rather than a pairwise kernel.
    x, y = F.normalize(x, dim=1), F.normalize(y, dim=1)
    d = x.shape[1]
    uniformity = []
    for z in (x, y):
        first = z.mean(0).square().sum()
        second = torch.einsum("bi,bj->ij", z, z).div(len(z)).square().sum()
        third = torch.einsum("bi,bj,bk->ijk", z, z, z).div(len(z)).square().sum()
        uniformity.append(
            weights[0] * first
            + weights[1] * (d * second - 1) / (d - 1)
            + weights[2] * ((d + 2) * third - 3 * first) / (d - 1)
        )
    return alpha * (x - y).square().mean() + beta * sum(uniformity) / 2


@pytest.mark.parametrize("shape", [(1, 2), (7, 3), (4, 8)])
@pytest.mark.parametrize(
    "weights", [(1.0, 40.0, 40.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)]
)
def test_reference_value_and_gradients(shape, weights):
    torch.manual_seed(7)
    x, y = [
        torch.randn(shape, dtype=torch.double, requires_grad=True) for _ in range(2)
    ]
    actual = SFRIKLoss(sim_coeff=4, kernel_weights=weights)(x, y)
    expected = _reference(x, y, weights=weights)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual, (x, y)), torch.autograd.grad(expected, (x, y))
    ):
        torch.testing.assert_close(a, b)


def test_gradcheck():
    torch.manual_seed(8)
    inputs = tuple(
        torch.randn(3, 4, dtype=torch.double, requires_grad=True) for _ in range(2)
    )
    assert torch.autograd.gradcheck(SFRIKLoss(sim_coeff=2), inputs)


def test_geometry_and_diagonal_convention():
    fn = SFRIKLoss(sim_coeff=2)
    collapsed = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
    # Four equally spaced circle points annihilate degrees one through three.
    square = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    torch.testing.assert_close(fn(collapsed, collapsed), torch.tensor(81.0))
    torch.testing.assert_close(fn(square, square), torch.tensor(0.0))
    torch.testing.assert_close(fn(collapsed[:1], collapsed[:1]), torch.tensor(81.0))
    torch.testing.assert_close(fn(square, -square), torch.tensor(4.0))


def test_invariance_to_rotation_permutation_and_positive_scale():
    torch.manual_seed(3)
    x, y = torch.randn(7, 5, dtype=torch.double), torch.randn(7, 5, dtype=torch.double)
    q = torch.linalg.qr(torch.randn(5, 5, dtype=torch.double)).Q
    perm = torch.randperm(7)
    fn = SFRIKLoss(sim_coeff=2)
    torch.testing.assert_close(fn(x, y), fn(x @ q, y @ q))
    torch.testing.assert_close(fn(x, y), fn(x[perm], y[perm]))
    torch.testing.assert_close(fn(x, y), fn(x * 3, y * 0.2))
    torch.testing.assert_close(fn(x, y), fn(y, x))
    torch.testing.assert_close(
        fn(x, y), SFRIKLoss(sim_coeff=2, gather_distributed=False)(x, y)
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_zero_inputs_are_finite(dtype):
    x = torch.zeros(4, 3, dtype=dtype, requires_grad=True)
    loss = SFRIKLoss()(x, x)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_autocast_uses_float32_loss():
    torch.manual_seed(5)
    x, y = [torch.randn(8, 6).bfloat16().requires_grad_() for _ in range(2)]
    fn = SFRIKLoss(sim_coeff=3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = fn(x, y)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, fn(x.float(), y.float()), rtol=0, atol=0)
    loss.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sim_coeff": -1},
        {"mmd_coeff": float("nan")},
        {"sim_coeff": float("inf")},
        {"sim_coeff": 0, "mmd_coeff": 0},
        {"kernel_weights": (1, 2)},
        {"kernel_weights": (0, 0, 0)},
        {"kernel_weights": (1, -1, 2)},
        {"kernel_weights": (1, float("inf"), 2)},
        {"eps": 0},
        {"eps": float("nan")},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        SFRIKLoss(**kwargs)


@pytest.mark.parametrize(
    "x,y,error",
    [
        (torch.ones(2, 3), torch.ones(3, 3), ValueError),
        (torch.ones(2, 3, 1), torch.ones(2, 3, 1), ValueError),
        (torch.ones(0, 3), torch.ones(0, 3), ValueError),
        (torch.ones(2, 1), torch.ones(2, 1), ValueError),
        (torch.ones(2, 3), torch.ones(2, 3).double(), ValueError),
        (torch.ones(2, 3).long(), torch.ones(2, 3).long(), TypeError),
    ],
)
def test_invalid_inputs(x, y, error):
    with pytest.raises(error):
        SFRIKLoss()(x, y)


def _model(composable=False):
    kwargs = dict(
        backbone=nn.Linear(4, 6), projector=nn.Linear(6, 3), optim={"optimizer": "Adam"}
    )
    if composable:
        model = spt.Module(
            forward=spt.forward.sfrik, sfrik_loss=SFRIKLoss(sim_coeff=3), **kwargs
        )
    else:
        model = spt.SFRIK(sim_coeff=3, **kwargs)
    model.log = Mock()
    return model


@pytest.mark.parametrize("composable", [False, True])
@pytest.mark.parametrize("layout", ["list", "named", "direct"])
def test_forward_and_callback_contract(composable, layout):
    model = _model(composable)
    views = [{"image": torch.randn(5, 4), "label": torch.arange(5)} for _ in range(2)]
    batch = {"views": views}
    if layout == "named":
        batch = {"views": dict(zip(("a", "b"), views))}
    elif layout == "direct":
        batch = dict(zip(("a", "b"), views))
    result = model(batch, stage="fit")
    assert result["embedding"].shape == (10, 6)
    assert result["projection"].shape == (10, 3)
    torch.testing.assert_close(result["label"], torch.arange(5).repeat(2))
    torch.testing.assert_close(
        result["embedding"], torch.cat([model.backbone(v["image"]) for v in views])
    )
    result["loss"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    with torch.no_grad(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            model.projector,
            "forward",
            Mock(side_effect=AssertionError("projector called")),
        )
        assert "loss" not in model(batch, stage="validate")
        single = model(views[0], stage="test")
    assert single["embedding"].shape == (5, 6)
    torch.testing.assert_close(single["label"], views[0]["label"])


def test_unlabelled_and_bad_view_counts():
    model = _model()
    v = {"image": torch.randn(3, 4)}
    assert "label" not in model({"views": [v, v]}, stage="fit")
    for batch in (v, {"views": [v]}, {"views": [v, v, v]}):
        with pytest.raises(ValueError, match="two views"):
            model(batch, stage="fit")
    model.eval()
    assert set(model(v, stage="predict")) == {"embedding"}


@pytest.mark.parametrize("device", ["mps", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_accelerator_matches_cpu(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(20)
    cpu = [torch.randn(16, 32, requires_grad=True) for _ in range(2)]
    accelerated = [x.detach().to(device).requires_grad_() for x in cpu]
    fn = SFRIKLoss(sim_coeff=16)
    a, b = fn(*cpu), fn(*accelerated)
    torch.testing.assert_close(a, b.cpu(), atol=1e-5, rtol=1e-5)
    for x, y in zip(torch.autograd.grad(a, cpu), torch.autograd.grad(b, accelerated)):
        torch.testing.assert_close(x, y.cpu(), atol=1e-5, rtol=1e-4)
