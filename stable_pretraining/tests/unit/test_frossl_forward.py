"""Forward and public API contracts for multiview FroSSL."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

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


def test_manager_training_resumes_from_checkpoint() -> None:
    """Check that resuming FroSSL reproduces uninterrupted CPU training."""
    generator = torch.Generator().manual_seed(23)
    images = torch.randn(8, 3, generator=generator)
    other_view = images + 0.25 * torch.randn(8, 3, generator=generator)
    samples = [
        {"views": [{"image": first}, {"image": second}]}
        for first, second in zip(images, other_view, strict=True)
    ]

    def fit(epochs: int, ckpt_path: str | None = None) -> tuple[pl.Trainer, spt.FroSSL]:
        torch.manual_seed(31 if ckpt_path else 29)
        model = spt.FroSSL(
            backbone=torch.nn.Linear(3, 4),
            projector=torch.nn.Linear(4, 2),
            optim={
                "optimizer": {"type": "SGD", "lr": 0.05, "momentum": 0.9},
                "scheduler": {"type": "CosineAnnealingLR", "T_max": 4},
                "interval": "step",
            },
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=epochs,
            limit_val_batches=0,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            log_every_n_steps=1,
            callbacks=[pl.callbacks.ModelCheckpoint(save_last=True, save_top_k=0)],
        )
        data = spt.data.DataModule(train=DataLoader(samples, batch_size=4))
        spt.Manager(
            trainer=trainer,
            module=model,
            data=data,
            ckpt_path=ckpt_path,
            weights_only=False,
            seed=29,
        )()
        return trainer, model

    first_trainer, _ = fit(epochs=1)
    assert first_trainer.global_step == 2
    checkpoint = first_trainer.checkpoint_callback.last_model_path
    assert Path(checkpoint).is_file()

    resumed_trainer, resumed = fit(epochs=2, ckpt_path=checkpoint)
    full_trainer, full = fit(epochs=2)

    assert resumed_trainer.global_step == full_trainer.global_step == 4
    torch.testing.assert_close(resumed.state_dict(), full.state_dict())
    torch.testing.assert_close(
        resumed_trainer.optimizers[0].state_dict()["state"],
        full_trainer.optimizers[0].state_dict()["state"],
    )
