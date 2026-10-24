"""DirectPred trained with the Manager: the method class, and resuming a run."""

from copy import deepcopy

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import stable_pretraining as spt
from stable_pretraining import forward

from ._interaction_helpers import _assert_tree_close, _snapshot

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]

# 16 two-view samples in batches of 4: 4 training steps per epoch.
STEPS_PER_EPOCH = 4
OPTIM = {
    "optimizer": {"type": "SGD", "lr": 0.1, "momentum": 0.9},
    "scheduler": {"type": "ConstantLR", "factor": 1.0, "total_iters": 0},
}


def _samples():
    g = torch.Generator().manual_seed(23)
    images = torch.randn(16, 3, 8, 8, generator=g)
    noise = 0.1 * torch.randn(2, 16, 3, 8, 8, generator=g)
    return [
        {
            "views": [
                {"image": images[i] + noise[v, i], "label": torch.tensor(i % 2)}
                for v in range(2)
            ]
        }
        for i in range(16)
    ]


class _StopAfterEpoch(pl.Callback):
    """Keep the planned training horizon unchanged while stopping the first leg."""

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def _fit(module, *, max_epochs, ckpt_path=None, stop_after_first_epoch=False):
    callbacks = [spt.callbacks.TeacherStudentCallback(verbose=False)]
    if stop_after_first_epoch:
        callbacks.append(_StopAfterEpoch())
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    val = [sample["views"][0] for sample in _samples()[:8]]
    data = spt.data.DataModule(
        train=DataLoader(_samples(), batch_size=4, shuffle=False),
        val=DataLoader(val, batch_size=4, shuffle=False),
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        seed=59,
        ckpt_path=ckpt_path,
        weights_only=False,
    )()
    return trainer


def _module():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 8))
        projector = nn.Sequential(nn.ReLU(), nn.Linear(8, 4))
    return spt.Module(
        backbone=spt.backbone.TeacherStudentWrapper(
            backbone, base_ema_coefficient=0.9, final_ema_coefficient=0.9
        ),
        projector=spt.backbone.TeacherStudentWrapper(
            projector, base_ema_coefficient=0.9, final_ema_coefficient=0.9
        ),
        predictor=spt.backbone.DirectPredictor(4),
        forward=forward.directpred,
        byol_loss=spt.losses.BYOLLoss(),
        optim=OPTIM,
    )


def test_method_class_trains_with_the_manager():
    """The Manager's steps call ``model(batch, stage)`` on the method class."""
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )
    model = spt.DirectPred(encoder_name=backbone, projector_dims=(16, 4))
    model.optim = OPTIM
    teacher_before = deepcopy(model.backbone.teacher.state_dict())

    trainer = _fit(model, max_epochs=2)

    assert trainer.global_step == 2 * STEPS_PER_EPOCH
    assert model.predictor.num_updates.item() == trainer.global_step
    assert torch.isfinite(trainer.callback_metrics["fit/loss_epoch"])
    teacher_after = model.backbone.teacher.state_dict()
    assert any(not torch.equal(v, teacher_after[k]) for k, v in teacher_before.items())


def test_resumed_run_matches_uninterrupted_training(tmp_path):
    """The predictor's statistics are restored, so training continues unchanged."""
    reference = _module()
    full = _fit(reference, max_epochs=3)

    interrupted = _module()
    first = _fit(interrupted, max_epochs=3, stop_after_first_epoch=True)
    assert first.global_step == STEPS_PER_EPOCH
    checkpoint = tmp_path / "restart.ckpt"
    first.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)["state_dict"]
    assert saved["predictor.num_updates"].item() == STEPS_PER_EPOCH
    torch.testing.assert_close(saved["predictor.corr"], interrupted.predictor.corr)

    resumed = _module()  # fresh predictor: identity weight, empty statistics
    resumed_trainer = _fit(resumed, max_epochs=3, ckpt_path=checkpoint)

    assert resumed.predictor.num_updates.item() == 3 * STEPS_PER_EPOCH
    _assert_tree_close(_snapshot(resumed_trainer, resumed), _snapshot(full, reference))
