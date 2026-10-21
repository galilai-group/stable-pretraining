"""FlatNCE completes real optimizer updates, evaluation, and checkpoint reloads."""

from copy import deepcopy

import lightning as pl
import pytest
import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader

import stable_pretraining as spt

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    from stable_pretraining.callbacks.factories import _DEFAULT_CALLBACK_REGISTRY

    config = spt.get_config()
    monkeypatch.setattr(
        config,
        "_default_callbacks",
        {name: name == "registry" for name in _DEFAULT_CALLBACK_REGISTRY},
    )
    monkeypatch.setattr(config, "_requeue_checkpoint", False)


@pytest.mark.parametrize("use_method_class", [False, True])
def test_flatnce_manager_trains_evaluates_and_reloads(tmp_path, use_method_class):
    pl.seed_everything(7, workers=True)
    backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 6))
    optim = {"optimizer": {"type": "SGD", "lr": 0.03}}
    if use_method_class:
        module = spt.FlatNCE(encoder_name=backbone, projector_dims=(8, 4), optim=optim)
    else:
        module = spt.Module(
            backbone=backbone,
            projector=nn.Linear(6, 4),
            forward=spt.forward.flatnce,
            flatnce_loss=spt.losses.FlatNCELoss(),
            optim=optim,
        )
    initial = deepcopy(module.state_dict())
    train = []
    val = []
    for i in range(12):
        image = torch.randn(3, 8, 8)
        label = torch.tensor(i % 2)
        train.append(
            {
                "views": [
                    {"image": image, "label": label},
                    {"image": image + 0.1 * torch.randn_like(image), "label": label},
                ]
            }
        )
        val.append({"image": image, "label": label})
    data = spt.data.DataModule(
        train=DataLoader(train, batch_size=4),
        val=DataLoader(val, batch_size=4),
    )
    probe = spt.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(6, 2),
        loss=nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(2)},
    )
    knn = spt.OnlineKNN(
        name="knn",
        input="embedding",
        target="label",
        queue_length=32,
        input_dim=6,
        k=2,
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(2)},
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[probe, knn],
        default_root_dir=str(tmp_path),
    )
    spt.Manager(trainer=trainer, module=module, data=data)()

    assert trainer.global_step == 3
    assert trainer.callback_metrics["fit/loss_epoch"].item() == 1.0
    assert any(
        "linear_probe" in key and "accuracy" in key for key in trainer.callback_metrics
    )
    assert any("knn" in key and "accuracy" in key for key in trainer.callback_metrics)
    parameter_names = dict(module.named_parameters())
    for prefix in ("backbone.", "projector."):
        assert any(
            name.startswith(prefix)
            and not torch.equal(value, module.state_dict()[name])
            for name, value in initial.items()
            if name in parameter_names
        )

    module.eval()
    images = val[0]["image"].unsqueeze(0)
    expected = module.backbone(images).detach()
    checkpoint = tmp_path / "flatnce.ckpt"
    trainer.save_checkpoint(checkpoint)
    with torch.no_grad():
        for parameter in module.backbone.parameters():
            parameter.add_(1)
    module.load_state_dict(torch.load(checkpoint, weights_only=False)["state_dict"])
    actual = module.backbone(images).detach()
    torch.testing.assert_close(actual, expected)


class _StopAfterEpoch(pl.Callback):
    """Interrupt a run without changing its planned optimization horizon."""

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: spt.Module) -> None:
        trainer.should_stop = True


def test_flatnce_full_state_resume_matches_uninterrupted_training(tmp_path):
    generator = torch.Generator().manual_seed(37)
    first = torch.randn(12, 4, generator=generator)
    second = first + 0.1 * torch.randn(12, 4, generator=generator)
    samples = [{"views": [{"image": x}, {"image": y}]} for x, y in zip(first, second)]

    def run(stop=False, checkpoint=None):
        pl.seed_everything(44, workers=True)
        module = spt.Module(
            backbone=nn.Linear(4, 6),
            projector=nn.Linear(6, 3),
            forward=spt.forward.flatnce,
            flatnce_loss=spt.losses.FlatNCELoss(),
            optim={
                "optimizer": {"type": "SGD", "lr": 0.03, "momentum": 0.9},
                "scheduler": {"type": "CosineAnnealingLR", "T_max": 9},
            },
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=3,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[_StopAfterEpoch()] if stop else [],
        )
        data = spt.data.DataModule(train=DataLoader(samples, batch_size=4))
        spt.Manager(
            trainer=trainer,
            module=module,
            data=data,
            seed=44,
            ckpt_path=str(checkpoint) if checkpoint else None,
            weights_only=False,
        )()
        return trainer, module

    expected_trainer, expected_module = run()
    interrupted, _ = run(stop=True)
    assert interrupted.global_step == 3
    checkpoint = tmp_path / "interrupted.ckpt"
    interrupted.save_checkpoint(checkpoint)
    resumed, actual_module = run(checkpoint=checkpoint)
    assert resumed.global_step == expected_trainer.global_step == 9
    torch.testing.assert_close(actual_module.state_dict(), expected_module.state_dict())
    torch.testing.assert_close(
        resumed.optimizers[0].state_dict(), expected_trainer.optimizers[0].state_dict()
    )
    assert (
        resumed.lr_scheduler_configs[0].scheduler.state_dict()
        == expected_trainer.lr_scheduler_configs[0].scheduler.state_dict()
    )
