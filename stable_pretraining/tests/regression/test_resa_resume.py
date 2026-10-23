"""ReSA restarted from a full checkpoint matches uninterrupted training."""

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import stable_pretraining as spt
from stable_pretraining.forward import resa

from ._interaction_helpers import _assert_tree_close

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]

STEPS_PER_EPOCH = 4


class _StopAfterEpoch(pl.Callback):
    """Keep the planned training horizon unchanged while stopping the first leg."""

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def _module():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        backbone = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        projector = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8, affine=False),
        )
    ema = dict(base_ema_coefficient=0.9, final_ema_coefficient=1.0)
    return spt.Module(
        backbone=spt.TeacherStudentWrapper(backbone, **ema),
        projector=spt.TeacherStudentWrapper(projector, **ema),
        forward=resa,
        resa_loss=spt.losses.ReSALoss(),
        optim={
            "optimizer": {
                "type": "SGD",
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": 2,
                "max_steps": 3 * STEPS_PER_EPOCH,
                "warmup_start_lr": 0.0,
                "eta_min": 0.01,
            },
            "interval": "step",
        },
    )


def _data():
    g = torch.Generator().manual_seed(17)
    images = torch.randn(16, 3, 8, 8, generator=g)
    noise = 0.1 * torch.randn(2, 16, 3, 8, 8, generator=g)
    train = [
        {
            "views": [
                {"image": images[i] + noise[v, i], "label": torch.tensor(i % 2)}
                for v in range(2)
            ]
        }
        for i in range(16)
    ]
    val = [{"image": images[i], "label": torch.tensor(i % 2)} for i in range(8)]
    return spt.data.DataModule(
        train=DataLoader(train, batch_size=4, shuffle=False),
        val=DataLoader(val, batch_size=4, shuffle=False),
    )


def _train(module, callbacks=(), ckpt_path=None):
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=list(callbacks),
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=_data(),
        seed=59,
        ckpt_path=ckpt_path,
        weights_only=False,
    )()
    return trainer


def test_resa_restart_matches_uninterrupted_training(tmp_path):
    reference = _module()
    full = _train(reference)

    interrupted = _module()
    first = _train(interrupted, callbacks=[_StopAfterEpoch()])
    assert first.global_step == STEPS_PER_EPOCH
    checkpoint = tmp_path / "resa.ckpt"
    first.save_checkpoint(checkpoint)

    resumed = _module()
    with torch.no_grad():
        for parameter in resumed.parameters():
            parameter.add_(10)
    second = _train(resumed, ckpt_path=str(checkpoint))

    assert second.global_step == full.global_step == 3 * STEPS_PER_EPOCH
    # Student and teacher weights, BatchNorm statistics and the EMA
    # coefficient buffers all follow the uninterrupted run.
    _assert_tree_close(resumed.state_dict(), reference.state_dict())
    assert resumed.backbone.ema_coefficient.item() < 1.0
    assert not torch.equal(
        resumed.backbone.teacher[0].weight, resumed.backbone.student[0].weight
    )
