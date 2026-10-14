"""Numerical and framework contracts for MMCR."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import stable_pretraining as spt
from stable_pretraining.forward import mmcr
from stable_pretraining.losses import MMCRLoss

pytestmark = pytest.mark.unit


def _reference(z, lmbda):
    # The author's flattened input is sample-major, then rearranged to B,D,V.
    b, v, d = z.shape
    normalized = F.normalize(z.reshape(b * v, d), dim=-1)
    manifolds = normalized.reshape(b, v, d).transpose(1, 2)
    centroids = manifolds.mean(-1)
    local = sum(torch.linalg.matrix_norm(m, ord="nuc") for m in manifolds)
    return lmbda * local / b - torch.linalg.matrix_norm(centroids, ord="nuc")


@pytest.mark.parametrize("shape", [(5, 2, 7), (3, 4, 2), (1, 2, 5), (4, 3, 1)])
@pytest.mark.parametrize("lmbda", [0.0, 0.2, 1.0])
def test_reference_values_and_gradients(shape, lmbda):
    torch.manual_seed(71)
    x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    loss = MMCRLoss(lmbda)(x)
    reference = _reference(y, lmbda)
    torch.testing.assert_close(loss, reference, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        torch.autograd.grad(loss, x)[0],
        torch.autograd.grad(reference, y)[0],
        rtol=1e-8,
        atol=1e-8,
    )


def test_gradcheck_and_invariances():
    torch.manual_seed(12)
    x = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    loss = MMCRLoss(0.3)
    assert torch.autograd.gradcheck(loss, (x,))
    expected = loss(x)
    for altered in (x[[2, 0, 1]], x[:, [1, 0]], x * 7.0):
        torch.testing.assert_close(loss(altered), expected)
    q = torch.linalg.qr(torch.randn(4, 4, dtype=x.dtype)).Q
    torch.testing.assert_close(loss(x @ q), expected)


def test_analytic_scale_and_unnormalized_centroids():
    z = torch.eye(3, dtype=torch.float64)[:, None].repeat(1, 2, 1)
    torch.testing.assert_close(
        MMCRLoss(0.5)(z), torch.tensor(0.5 * 2**0.5 - 3, dtype=z.dtype)
    )
    opposite = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    torch.testing.assert_close(MMCRLoss()(opposite), torch.tensor(0.0))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_precision_and_finite_backward(dtype):
    x = torch.randn(4, 3, 6).to(dtype).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = MMCRLoss(0.1)(x)
    assert result.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    result.backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("kind", ["zero", "constant", "duplicate"])
def test_rank_deficient_inputs(kind):
    z = torch.zeros(3, 2, 5) if kind == "zero" else torch.ones(3, 2, 5)
    if kind == "duplicate":
        z = torch.randn(3, 1, 5).repeat(1, 2, 1)
    z.requires_grad_()
    result = MMCRLoss(0.2)(z)
    result.backward()
    assert torch.isfinite(result) and torch.isfinite(z.grad).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lmbda": -1},
        {"lmbda": float("nan")},
        {"lmbda": float("inf")},
        {"eps": 0},
        {"eps": -1},
        {"eps": float("nan")},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        MMCRLoss(**kwargs)


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(2, 4),
        torch.randn(0, 2, 4),
        torch.randn(2, 1, 4),
        torch.randn(2, 2, 0),
        torch.full((2, 2, 4), float("nan")),
        torch.full((2, 2, 4), float("inf")),
    ],
)
def test_invalid_shapes_and_values(x):
    with pytest.raises(ValueError):
        MMCRLoss()(x)


@pytest.mark.parametrize(
    "x",
    [
        None,
        torch.ones(2, 2, 4, dtype=torch.int64),
        torch.ones(2, 2, 4, dtype=torch.complex64),
    ],
)
def test_invalid_dtype(x):
    with pytest.raises(TypeError):
        MMCRLoss()(x)


def test_zero_lambda_skips_local_svd(monkeypatch):
    real = torch.linalg.svdvals
    calls = []

    def capture(x):
        calls.append(x.shape)
        return real(x)

    monkeypatch.setattr(torch.linalg, "svdvals", capture)
    MMCRLoss()(torch.randn(4, 3, 5))
    assert calls == [torch.Size([4, 5])]


def _module():
    return SimpleNamespace(
        training=True,
        backbone=nn.Identity(),
        projector=nn.Identity(),
        mmcr_loss=MMCRLoss(0.2),
        log=Mock(),
    )


@pytest.mark.parametrize("layout", ["list", "named", "nested"])
def test_view_grouping_and_labels(layout):
    views = [{"image": torch.randn(4, 5), "label": torch.arange(4)} for _ in range(3)]
    named = {f"view{i}": view for i, view in enumerate(views)}
    batch = (
        {"views": views}
        if layout == "list"
        else (named if layout == "named" else {"views": named})
    )
    model = _module()
    output = mmcr(model, batch, "fit")
    expected = torch.stack([v["image"] for v in views], dim=1)
    torch.testing.assert_close(output["loss"], _reference(expected, 0.2))
    torch.testing.assert_close(
        output["projection"], torch.cat([v["image"] for v in views])
    )
    torch.testing.assert_close(output["label"], torch.arange(4).repeat(3))
    model.log.assert_called_once()


def test_eval_does_not_project_and_unlabelled_training():
    model = _module()
    views = [{"image": torch.randn(4, 5)} for _ in range(2)]
    assert "label" not in mmcr(model, {"views": views}, "fit")
    model.training = False
    model.projector = Mock(side_effect=AssertionError("eval must skip projector"))
    output = mmcr(
        model, {"image": views[0]["image"], "label": torch.arange(4)}, "validate"
    )
    assert set(output) == {"embedding", "label"}
    assert set(mmcr(model, {"image": views[0]["image"]}, "predict")) == {"embedding"}
    assert set(mmcr(model, {"views": views}, "test")) == {"embedding"}


@pytest.mark.parametrize(
    "batch",
    [
        {"image": torch.ones(3, 4)},
        {"views": []},
        {"views": [{"image": torch.ones(3, 4)}]},
        {"views": [{"image": torch.ones(3, 4)}, {"image": torch.ones(2, 4)}]},
        {"views": [{"image": torch.ones(0, 4)}, {"image": torch.ones(0, 4)}]},
        {
            "views": [
                {"image": torch.ones(3, 4), "label": torch.arange(3)},
                {"image": torch.ones(3, 4)},
            ]
        },
        {"views": [{"image": torch.ones(3, 4), "label": torch.arange(2)}] * 2},
    ],
)
def test_invalid_view_batches(batch):
    with pytest.raises(ValueError):
        mmcr(_module(), batch, "fit")


@pytest.mark.parametrize("component", ["backbone", "projector"])
def test_invalid_encoder_or_projector_shape(component):
    model = _module()
    setattr(model, component, nn.Unflatten(1, (2, 2)))
    with pytest.raises(ValueError):
        mmcr(model, {"views": [{"image": torch.ones(3, 4)}] * 2}, "fit")


def test_method_exports_composable_equivalence_and_state_dict():
    from stable_pretraining.methods import MMCR

    assert spt.MMCR is MMCR
    torch.manual_seed(8)
    backbone, projector = nn.Linear(5, 4), nn.Linear(4, 3)
    model = spt.MMCR(deepcopy(backbone), deepcopy(projector), lmbda=0.2)
    composed = spt.Module(
        forward=mmcr, backbone=backbone, projector=projector, mmcr_loss=MMCRLoss(0.2)
    )
    batch = {"views": [{"image": torch.randn(4, 5)} for _ in range(3)]}
    actual, expected = model(batch), composed(batch, "fit")
    torch.testing.assert_close(actual["loss"], expected["loss"])
    actual["loss"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    restored = spt.MMCR(nn.Linear(5, 4), nn.Linear(4, 3), optim={"optimizer": "SGD"})
    restored.load_state_dict(model.state_dict())
    model.eval()
    restored.eval()
    batch = {"image": torch.randn(2, 5)}
    torch.testing.assert_close(model(batch)["embedding"], restored(batch)["embedding"])


def test_distributed_shape_guard_and_opt_out(monkeypatch):
    import importlib

    module = importlib.import_module("stable_pretraining.losses.mmcr")
    monkeypatch.setattr(module, "get_world_size", lambda: 2)
    monkeypatch.setattr(module, "all_gather", lambda shape: [shape, shape + 1])
    z = torch.randn(3, 2, 4)
    with pytest.raises(ValueError, match="equal"):
        MMCRLoss()(z)
    torch.testing.assert_close(
        MMCRLoss(gather_distributed=False)(z), _reference(z, 0.0)
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_autocast_value_and_gradient(dtype):
    torch.manual_seed(91)
    x = torch.randn(5, 3, 7, device="cuda", dtype=dtype, requires_grad=True)
    reference = x.detach().cpu().float().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16):
        actual = MMCRLoss(0.2)(x)
    expected = _reference(reference, 0.2)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        x.grad.cpu().float(), reference.grad, rtol=2e-2, atol=2e-3
    )


def test_global_gather_preserves_values_and_input_gradients(monkeypatch):
    import importlib

    module = importlib.import_module("stable_pretraining.losses.mmcr")
    torch.manual_seed(19)
    x = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    remote = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    expected_remote = remote.detach().clone().requires_grad_()

    def gather(tensor):
        if tensor.ndim == 1:
            return [tensor, tensor.clone()]
        return [tensor, F.normalize(remote, dim=-1)]

    monkeypatch.setattr(module, "get_world_size", lambda: 2)
    monkeypatch.setattr(module, "all_gather", gather)
    loss = MMCRLoss(0.3)(x)
    expected = _reference(torch.cat([expected_x, expected_remote]), 0.3)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(x.grad, expected_x.grad)
    torch.testing.assert_close(remote.grad, expected_remote.grad)
