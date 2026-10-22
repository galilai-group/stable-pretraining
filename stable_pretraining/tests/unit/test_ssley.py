"""SSL-EY mathematical and framework contract tests."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning as pl

import stable_pretraining as spt

pytestmark = pytest.mark.unit


def test_reference_value_gradients_and_invariances():
    torch.manual_seed(7)
    x, y = [torch.randn(5, 7, dtype=torch.double, requires_grad=True) for _ in range(2)]
    loss = spt.losses.SSLEYLoss()
    xc, yc = x - x.mean(0), y - y.mean(0)
    c = 2 * xc.T @ yc / 4
    v = (xc.T @ xc + yc.T @ yc) / 4
    reference = -2 * torch.trace(c) + torch.trace(v @ v)
    actual = loss(x, y)
    torch.testing.assert_close(actual, reference)
    for a, b in zip(
        torch.autograd.grad(actual, (x, y)), torch.autograd.grad(reference, (x, y))
    ):
        torch.testing.assert_close(a, b)
    assert torch.autograd.gradcheck(loss, (x, y))
    torch.testing.assert_close(loss(x, y), loss(y, x))
    torch.testing.assert_close(loss(x + 10, y - 5), loss(x, y))
    # One whitened component shared by both views achieves -1.
    z = torch.tensor([[-0.5], [0.5]], dtype=torch.double)
    torch.testing.assert_close(loss(z, z), torch.tensor(-1.0, dtype=torch.double))


@pytest.mark.parametrize(
    "shape1,shape2",
    [((1, 3), (1, 3)), ((4, 3), (3, 3)), ((4, 0), (4, 0)), ((4, 3, 1), (4, 3, 1))],
)
def test_invalid_shapes(shape1, shape2):
    with pytest.raises(ValueError):
        spt.losses.SSLEYLoss()(torch.randn(shape1), torch.randn(shape2))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_and_autocast(dtype):
    x, y = [torch.randn(8, 4).to(dtype).requires_grad_() for _ in range(2)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = spt.losses.SSLEYLoss()(x, y)
    reference = spt.losses.SSLEYLoss()(x.float(), y.float())
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference)
    actual.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


def _model():
    return spt.SSLEY(backbone=nn.Linear(3, 4), projector=nn.Linear(4, 2))


@pytest.mark.parametrize("layout", ["list", "named", "direct"])
def test_forward_contract(layout):
    model = _model()
    model.log = Mock()
    views = [{"image": torch.randn(4, 3), "label": torch.arange(4)} for _ in range(2)]
    batch = {"views": views}
    if layout != "list":
        named = dict(zip(["first", "second"], views))
        batch = {"views": named} if layout == "named" else named
    out = model(batch, stage="fit")
    assert out["embedding"].shape == (8, 4)
    torch.testing.assert_close(out["label"], torch.arange(4).repeat(2))
    out["loss"].backward()
    assert model.backbone.weight.grad is not None
    model.eval()
    hook = model.projector.register_forward_hook(
        lambda *args: pytest.fail("projector used in evaluation")
    )
    with torch.no_grad():
        result = model(views[0], stage="validate")
    assert "loss" not in result and result["embedding"].shape == (4, 4)
    hook.remove()
    model.train()
    with pytest.raises(ValueError, match="two views"):
        model(views[0], stage="fit")
    unlabeled = {"views": [{"image": view["image"]} for view in views]}
    assert "label" not in model(unlabeled, stage="fit")


def test_manager_training_and_checkpoint(tmp_path):
    pl.seed_everything(7)
    model = _model()
    initial = model.backbone.weight.detach().clone()
    samples = [
        {"views": [{"image": torch.randn(3)}, {"image": torch.randn(3)}]}
        for _ in range(8)
    ]
    data = spt.data.DataModule(train=DataLoader(samples, batch_size=4))
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    # Manager writes its registry into the configured cache directory.
    previous = spt.get_config().cache_dir
    spt.set(cache_dir=str(tmp_path))
    try:
        spt.Manager(trainer=trainer, module=model, data=data)()
    finally:
        spt.set(cache_dir=previous)
    assert trainer.global_step == 2
    assert not torch.equal(initial, model.backbone.weight)
    path = tmp_path / "ssley.ckpt"
    trainer.save_checkpoint(path)
    restored = _model()
    restored.load_state_dict(torch.load(path, weights_only=False)["state_dict"])
    model.eval()
    restored.eval()
    batch = {"image": torch.randn(3, 3)}
    torch.testing.assert_close(
        model(batch, stage="test")["embedding"],
        restored(batch, stage="test")["embedding"],
    )
