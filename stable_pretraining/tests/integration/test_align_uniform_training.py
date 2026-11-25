"""Integration test for AlignUniform training and checkpoint restoration."""

import copy
import sys

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stable_pretraining.data import DataModule
from stable_pretraining.manager import Manager
from stable_pretraining.methods import AlignUniform

pytestmark = pytest.mark.integration


class TwoViewDataset(Dataset):
    """Provide deterministic image pairs for a small training run."""

    def __init__(self):
        generator = torch.Generator().manual_seed(7)
        self.images = torch.randn(8, 3, 8, 8, generator=generator)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        return {"views": [{"image": image}, {"image": image.flip(-1)}]}


class CaptureRestoredState(pl.Callback):
    """Capture model and optimizer state before resumed training."""

    def __init__(self):
        self.step = None
        self.model = None
        self.optimizer = None

    def on_train_start(self, trainer, pl_module):
        self.step = trainer.global_step
        self.model = copy.deepcopy(pl_module.state_dict())
        self.optimizer = copy.deepcopy(trainer.optimizers[0].state_dict())


def _make_model():
    backbone = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    model = AlignUniform(encoder_name=backbone, projector_dims=(4,))
    model.optim = {"optimizer": {"type": "Adam", "lr": 0.01}}
    return model


def _make_trainer(tmp_path, max_epochs, callbacks=None):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        default_root_dir=str(tmp_path),
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_manager_trains_and_resumes_align_uniform(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    if sys.platform == "win32":
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda *_: None
        )
    torch.manual_seed(42)
    data = DataModule(train=DataLoader(TwoViewDataset(), batch_size=4))
    model = _make_model()
    initial_weight = model.backbone[0].weight.detach().clone()
    trainer = _make_trainer(tmp_path / "first", max_epochs=1)
    Manager(trainer=trainer, module=model, data=data, seed=42)()

    assert trainer.global_step == 2
    assert not torch.equal(model.backbone[0].weight, initial_weight)
    checkpoint_path = tmp_path / "align_uniform.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 2
    assert checkpoint["optimizer_states"][0]["state"]

    capture = CaptureRestoredState()
    resumed_model = _make_model()
    resumed_trainer = _make_trainer(tmp_path / "resumed", 2, [capture])
    Manager(
        trainer=resumed_trainer,
        module=resumed_model,
        data=data,
        seed=42,
        ckpt_path=str(checkpoint_path.resolve()),
        weights_only=False,
    )()

    assert capture.step == checkpoint["global_step"]
    _assert_nested_equal(capture.model, checkpoint["state_dict"])
    _assert_nested_equal(capture.optimizer, checkpoint["optimizer_states"][0])
    assert resumed_trainer.global_step == 4
    assert not torch.equal(
        resumed_model.backbone[0].weight, checkpoint["state_dict"]["backbone.0.weight"]
    )
