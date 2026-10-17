"""Forward and public API contracts for multiview FroSSL."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.forward import frossl
from stable_pretraining.losses import FroSSLLoss


pytestmark = pytest.mark.unit


def _module(training: bool) -> SimpleNamespace:
    return SimpleNamespace(
        training=training,
        backbone=torch.nn.Identity(),
        projector=torch.nn.Identity(),
        frossl_loss=FroSSLLoss(),
        log=Mock(),
    )


@pytest.mark.parametrize("container", ["list", "named", "nested_named"])
def test_training_views_keep_embedding_label_alignment(container):
    first = {"image": torch.eye(3), "label": torch.tensor([0, 1, 2])}
    second = {
        "image": torch.eye(3).roll(1, dims=0),
        "label": torch.tensor([0, 1, 2]),
    }
    if container == "list":
        batch = {"views": [first, second], "batch_idx": 0}
    elif container == "named":
        batch = {"first": first, "second": second, "batch_idx": 0}
    else:
        batch = {"views": {"first": first, "second": second}, "batch_idx": 0}
    module = _module(training=True)

    out = frossl(module, batch, "fit")

    torch.testing.assert_close(
        out["embedding"], torch.cat([first["image"], second["image"]])
    )
    torch.testing.assert_close(
        out["label"], torch.cat([first["label"], second["label"]])
    )
    torch.testing.assert_close(
        out["loss"], FroSSLLoss()([first["image"], second["image"]])
    )
    module.log.assert_called_once_with(
        "fit/loss", out["loss"], on_step=True, on_epoch=True, sync_dist=True
    )


def test_single_view_evaluation_returns_backbone_features_only():
    module = _module(training=False)
    image = torch.randn(4, 3)
    label = torch.arange(4)

    out = frossl(module, {"image": image, "label": label}, "validate")

    assert set(out) == {"embedding", "label"}
    torch.testing.assert_close(out["embedding"], image)
    torch.testing.assert_close(out["label"], label)
    module.log.assert_not_called()


def test_rejects_missing_or_misaligned_training_views():
    module = _module(training=True)
    image = torch.randn(3, 2)
    with pytest.raises(ValueError, match="at least two"):
        frossl(module, {"image": image}, "fit")
    with pytest.raises(ValueError, match="all contain label"):
        frossl(
            module,
            {"views": [{"image": image, "label": torch.arange(3)}, {"image": image}]},
            "fit",
        )
    with pytest.raises(ValueError, match="aligned sample_idx"):
        frossl(
            module,
            {
                "views": [
                    {"image": image, "sample_idx": torch.arange(3)},
                    {"image": image, "sample_idx": torch.arange(3).flip(0)},
                ]
            },
            "fit",
        )


def test_public_method_wires_forward_loss_and_hparams():
    model = spt.FroSSL(
        backbone=torch.nn.Linear(3, 4),
        projector=torch.nn.Linear(4, 2),
        optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
        invariance_weight=1.4,
    )

    assert isinstance(model, spt.Module)
    assert isinstance(model.frossl_loss, FroSSLLoss)
    assert (
        model.frossl_loss.invariance_weight == model.hparams["invariance_weight"] == 1.4
    )
    assert model.forward.__func__.__name__ == "frossl"


def test_public_method_backpropagates_through_backbone_and_projector():
    model = spt.FroSSL(
        backbone=torch.nn.Linear(3, 4),
        projector=torch.nn.Linear(4, 2),
        optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
    )
    batch = {"views": [{"image": torch.randn(4, 3)} for _ in range(3)]}
    model.log = Mock()

    model(batch, "fit")["loss"].backward()

    assert model.backbone.weight.grad is not None
    assert model.projector.weight.grad is not None
    assert torch.isfinite(model.backbone.weight.grad).all()
    assert torch.isfinite(model.projector.weight.grad).all()
