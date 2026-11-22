"""All4One training, optimizer, EMA, and full-state resume regressions."""

import os
from unittest.mock import Mock

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import stable_pretraining as spt
from stable_pretraining.callbacks import TeacherStudentCallback
from stable_pretraining.data.module import DataModule
from stable_pretraining.tests._all4one_helpers import tiny_all4one

from ._interaction_helpers import _Trace, _assert_tree_close, _snapshot

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]


@pytest.fixture(autouse=True)
def standalone_runtime(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SLURM_"):
            monkeypatch.delenv(key)


@pytest.mark.parametrize("explicit_callback", [False, True])
def test_manager_trains_both_optimizers_and_updates_ema_once_per_step(
    monkeypatch, explicit_callback
):
    optimizer = {"type": "SGD", "lr": 0.01, "momentum": 0.9}
    scheduler = {"type": "ConstantLR", "factor": 1.0}
    model = tiny_all4one(
        optim={
            "representation": {
                "modules": r"^(backbone\.student|projector\.student|predictor|centroid_predictor)$",
                "optimizer": optimizer,
                "scheduler": scheduler,
            },
            "attention": {
                "modules": r"^attention$",
                "optimizer": {**optimizer, "lr": 0.02},
                "scheduler": scheduler,
            },
        }
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    update = Mock(wraps=model.backbone.update_teacher)
    monkeypatch.setattr(model.backbone, "update_teacher", update)
    samples = [
        {"views": [{"image": torch.randn(6)}, {"image": torch.randn(6)}]}
        for _ in range(6)
    ]
    train = DataLoader(samples, batch_size=3)
    val = DataLoader([{"image": torch.randn(6)} for _ in range(3)], batch_size=3)
    manager = spt.Manager(
        trainer={
            "_target_": "lightning.Trainer",
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
            "callbacks": (
                [{"_target_": "stable_pretraining.callbacks.TeacherStudentCallback"}]
                if explicit_callback
                else []
            ),
        },
        module=model,
        data=DataModule(train=train, val=val),
        seed=17,
    )
    manager()
    trainer = manager._trainer
    assert trainer.global_step == 2
    assert len(trainer.optimizers) == 2
    assert sum(isinstance(cb, TeacherStudentCallback) for cb in trainer.callbacks) == 1
    assert update.call_count == 2
    assert model.backbone.ema_coefficient.item() == 1.0
    assert model.queue.queue.pointer.item() == 6
    for prefix in (
        "backbone.student",
        "projector.student",
        "predictor",
        "centroid_predictor",
        "attention",
    ):
        assert any(
            not torch.equal(p, before[name])
            for name, p in model.named_parameters()
            if name.startswith(prefix)
        )
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assigned = [
        id(p)
        for opt in trainer.optimizers
        for group in opt.param_groups
        for p in group["params"]
    ]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == trainable


class _StopAfterFirstEpoch(pl.Callback):
    """Interrupt at an epoch boundary without shortening the EMA horizon."""

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def _resume_components():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        optimizer = {"type": "SGD", "lr": 0.01, "momentum": 0.9}
        model = tiny_all4one(
            optim={
                "representation": {
                    "modules": r"^(backbone\.student|projector\.student|predictor|centroid_predictor)$",
                    "optimizer": optimizer,
                    "scheduler": {"type": "ExponentialLR", "gamma": 0.8},
                    "interval": "step",
                },
                "attention": {
                    "modules": r"^attention$",
                    "optimizer": {**optimizer, "lr": 0.02},
                    "scheduler": {"type": "ExponentialLR", "gamma": 0.9},
                    "interval": "step",
                },
            }
        )
        probe = spt.OnlineProbe(
            model,
            name="linear",
            input="embedding",
            target="label",
            probe=nn.Linear(8, 4),
            loss=nn.CrossEntropyLoss(),
            optimizer={**optimizer, "lr": 0.03},
            scheduler={"type": "ExponentialLR", "gamma": 0.85},
            metrics={"accuracy": MulticlassAccuracy(4, average="micro")},
            verbose=False,
        )
    trace = _Trace()
    return model, [probe, TeacherStudentCallback(verbose=False), trace], trace


def _run_resume(model, callbacks, checkpoint=None):
    generator = torch.Generator().manual_seed(23)
    samples = [
        {
            "views": [
                {"image": torch.randn(6, generator=generator), "label": i % 4}
                for _ in range(2)
            ]
        }
        for i in range(6)
    ]
    data = DataModule(
        train=DataLoader(samples, batch_size=3, shuffle=False),
        val=DataLoader([sample["views"][0] for sample in samples[:3]], batch_size=3),
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    spt.Manager(
        trainer=trainer,
        module=model,
        data=data,
        seed=17,
        ckpt_path=str(checkpoint) if checkpoint else None,
        weights_only=False,
    )()
    return trainer


def test_full_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    reference, callbacks, full_trace = _resume_components()
    full_trainer = _run_resume(reference, callbacks)

    interrupted, callbacks, first_trace = _resume_components()
    first_trainer = _run_resume(interrupted, [*callbacks, _StopAfterFirstEpoch()])
    assert first_trainer.global_step == 2
    _assert_tree_close(first_trace.epochs[0], full_trace.epochs[0])
    checkpoint = tmp_path / "all4one.ckpt"
    first_trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert len(saved["optimizer_states"]) == len(saved["lr_schedulers"]) == 3
    assert all(state["state"] for state in saved["optimizer_states"])

    resumed, callbacks, resumed_trace = _resume_components()
    with torch.no_grad():
        for parameter in resumed.parameters():
            parameter.add_(1)
        resumed.queue.queue.out.zero_()
        resumed.backbone.ema_coefficient.zero_()
    resumed_trainer = _run_resume(resumed, callbacks, checkpoint)

    _assert_tree_close(resumed_trace.start, first_trace.epochs[0])
    assert resumed_trace.restored_batches == 2
    assert resumed_trace.batches_seen == full_trace.batches_seen == 6
    assert len(resumed_trace.batches) == 4
    for actual, expected in zip(resumed_trace.batches, full_trace.batches[2:]):
        _assert_tree_close(actual, expected)
    _assert_tree_close(
        _snapshot(resumed_trainer, resumed), _snapshot(full_trainer, reference)
    )
    assert resumed_trainer.global_step == full_trainer.global_step == 6
    assert resumed_trainer.current_epoch == full_trainer.current_epoch == 3
    assert resumed.queue.queue.pointer.item() == 18 % 11
    assert resumed.backbone.ema_coefficient.item() == 1.0
    assert resumed.backbone.teacher[1].num_batches_tracked.item() == 12
    for metric in ("fit/loss_epoch", "eval/linear_accuracy"):
        _assert_tree_close(
            resumed_trainer.callback_metrics[metric],
            full_trainer.callback_metrics[metric],
        )
