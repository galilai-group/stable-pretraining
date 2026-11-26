"""MEC trains through ``spt.Manager`` and its EMA teacher survives a resume."""

import lightning as pl
import pytest
import torch
from lightning.pytorch.callbacks import Callback
from torch import nn
from torch.utils.data import DataLoader

import stable_pretraining as spt
from stable_pretraining.methods import MEC

pytestmark = pytest.mark.unit


class _TwoViewDataset(torch.utils.data.Dataset):
    def __init__(self, n=32):
        g = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 2, 3, 16, 16, generator=g)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return {
            "views": [{"image": self.images[i, v], "label": i % 2} for v in range(2)]
        }


def _data():
    return spt.data.DataModule(
        train=DataLoader(_TwoViewDataset(), batch_size=8, drop_last=True)
    )


def _mec(seed, ema_decay_start=0.9):
    torch.manual_seed(seed)
    encoder = nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2), nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )
    model = MEC(
        encoder_name=encoder,
        projector_dim=16,
        predictor_hidden_dim=8,
        warmup_epochs=2,
        ema_decay_start=ema_decay_start,
    )
    model.optim = {"optimizer": {"type": "SGD", "lr": 0.1}}
    return model


class _CaptureTeacherAtTrainStart(Callback):
    """Snapshot the teacher right after the checkpoint is restored."""

    def on_train_start(self, trainer, pl_module):
        self.teacher = _teacher_state(pl_module)
        self.ema_coefficient = pl_module.backbone.ema_coefficient.clone()
        self.global_step = trainer.global_step


def _teacher_state(model):
    return {
        f"{name}.{k}": v.detach().clone()
        for name in ("backbone", "projector")
        for k, v in getattr(model, name).teacher.state_dict().items()
    }


def _student_state(model):
    return {
        f"{name}.{k}": v.detach().clone()
        for name in ("backbone", "projector")
        for k, v in getattr(model, name).student.state_dict().items()
    }


def _trainer(max_steps, *callbacks):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=max_steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[spt.callbacks.TeacherStudentCallback(), *callbacks],
    )


def test_mec_trains_and_resumes_with_ema_teacher(tmp_path):
    # --- train a few steps and save a checkpoint --------------------------
    model = _mec(seed=0)
    initial_teacher = _teacher_state(model)
    trainer = _trainer(max_steps=3)
    spt.Manager(trainer=trainer, module=model, data=_data(), seed=0)()

    assert trainer.global_step == 3
    assert torch.isfinite(trainer.callback_metrics["fit/loss"])
    assert "fit/mec_entropy" in trainer.callback_metrics

    saved_teacher = _teacher_state(model)
    saved_student = _student_state(model)
    saved_ema = model.backbone.ema_coefficient.clone()
    weights = [k for k in saved_teacher if k.endswith("weight")]
    # The EMA moved the teacher, and it lags the student rather than copying it.
    assert any(not torch.equal(saved_teacher[k], initial_teacher[k]) for k in weights)
    assert any(not torch.equal(saved_teacher[k], saved_student[k]) for k in weights)

    ckpt = tmp_path / "mec.ckpt"
    trainer.save_checkpoint(ckpt)

    # --- resume into a freshly initialized model -------------------------
    # Different seed and EMA schedule, so anything not restored would show.
    resumed = _mec(seed=1, ema_decay_start=0.5)
    assert any(
        not torch.equal(v, saved_teacher[k]) for k, v in _teacher_state(resumed).items()
    ), "a fresh model must start from different weights for this test to mean anything"

    capture = _CaptureTeacherAtTrainStart()
    resumed_trainer = _trainer(5, capture)
    spt.Manager(
        trainer=resumed_trainer,
        module=resumed,
        data=_data(),
        seed=0,
        ckpt_path=str(ckpt),
        weights_only=False,
    )()

    # The teacher came back exactly as saved, not re-copied from the student.
    assert capture.global_step == 3
    assert capture.teacher.keys() == saved_teacher.keys()
    for k, v in saved_teacher.items():
        torch.testing.assert_close(capture.teacher[k], v, rtol=0, atol=0, msg=k)
    torch.testing.assert_close(capture.ema_coefficient, saved_ema)

    # Training continued from step 3 and kept updating the restored teacher.
    assert resumed_trainer.global_step == 5
    assert torch.isfinite(resumed_trainer.callback_metrics["fit/loss"])
    assert any(
        not torch.equal(_teacher_state(resumed)[k], saved_teacher[k]) for k in weights
    )
