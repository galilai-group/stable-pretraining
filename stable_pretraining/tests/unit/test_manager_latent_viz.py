"""Manager wires Hydra partial callbacks and preserves the EMA callback exactly once."""

import json
import os

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from stable_pretraining import Module
from stable_pretraining.backbone.utils import TeacherStudentWrapper
from stable_pretraining.callbacks.latent_viz import LatentViz
from stable_pretraining.callbacks.teacher_student import TeacherStudentCallback
from stable_pretraining.data.module import DataModule
from stable_pretraining.manager import Manager

pytestmark = pytest.mark.unit


def _forward(self, batch, stage):
    embeddings = self.backbone.forward_student(batch["image"])
    return {"embedding": embeddings, "loss": embeddings.square().mean()}


@pytest.mark.parametrize("explicit_teacher_callback", [False, True])
def test_manager_instantiates_latent_projection_and_trains_with_ema(
    tmp_path, monkeypatch, explicit_teacher_callback
):
    for key in list(os.environ):
        if key.startswith("SLURM_"):
            monkeypatch.delenv(key)
    model = Module(
        forward=_forward,
        backbone=TeacherStudentWrapper(nn.Linear(3, 4)),
        optim={
            "optimizer": {"type": "SGD", "lr": 0.01},
            "scheduler": {"type": "ConstantLR", "factor": 1.0},
        },
    )
    callbacks = [
        {
            "_target_": "stable_pretraining.callbacks.LatentViz",
            "_partial_": True,
            "name": "latent",
            "input": "embedding",
            "target": None,
            "projection": {
                "_target_": "torch.nn.Linear",
                "in_features": 4,
                "out_features": 2,
            },
            "input_dim": 4,
            "queue_length": 8,
            "update_interval": 1,
        }
    ]
    if explicit_teacher_callback:
        callbacks.append(
            {"_target_": "stable_pretraining.callbacks.TeacherStudentCallback"}
        )
    trainer_config = {
        "_target_": "lightning.Trainer",
        "accelerator": "cpu",
        "devices": 1,
        "max_epochs": 1,
        "callbacks": callbacks,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
    }
    loader = DataLoader(
        [{"image": torch.tensor([1.0, i, 2.0])} for i in range(8)], batch_size=2
    )
    manager = Manager(
        trainer=trainer_config,
        module=model,
        data=DataModule(train=loader, val=loader),
        seed=17,
    )
    manager()
    trainer = manager._trainer
    assert trainer.global_step == 4
    assert sum(isinstance(cb, TeacherStudentCallback) for cb in trainer.callbacks) == 1
    viz = next(cb for cb in trainer.callbacks if isinstance(cb, LatentViz))
    assert "latent" not in model.callbacks_modules
    assert len(trainer.optimizers) == 1
    assert viz._optimizer.state[viz.module.weight]
    with np.load(manager._run_dir / "latent_viz_latent" / "epoch_0000.npz") as archive:
        assert archive["coordinates"].shape == (8, 2)
        assert "labels" not in archive.files
    assert (
        json.loads((manager._run_dir / "sidecar.json").read_text())["status"]
        == "completed"
    )
