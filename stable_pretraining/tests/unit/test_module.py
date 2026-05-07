import pickle
from functools import partial

import pytest
from stable_pretraining import Module, forward
from stable_pretraining.losses import NTXEntLoss
from stable_pretraining.data import DataModule
import torch.nn as nn
import torch
from lightning.pytorch import Trainer


def _partial_forward(self, batch, stage, cfg):
    return {"loss": batch["image"].mean() * cfg["scale"]}


@pytest.mark.unit
def test_module_accepts_partial_forward():
    module = Module(
        backbone=nn.Linear(1, 1),
        forward=partial(_partial_forward, cfg={"scale": 2.0}),
        optim={"opt": {"modules": "backbone", "optimizer": {"type": "AdamW"}}},
    )
    pickle.dumps(module.forward)


@pytest.mark.unit
def test_module_initialization():
    """Test initialization of the Module object with a specific configuration."""
    backbone = nn.Linear(1, 1)  # Simple nn.Module with a single parameter
    projector = nn.Linear(1, 1)  # Simple nn.Module with a single parameter

    module = Module(
        backbone=backbone,
        projector=projector,
        forward=forward.simclr_forward,  # Or byol_forward, vicreg_forward, etc.
        simclr_loss=NTXEntLoss(temperature=0.5),
        optim={
            "optimizer": {"type": "Adam", "lr": 0.001},
            "scheduler": {"type": "CosineAnnealing"},
            "interval": "epoch",
        },
    )

    assert module is not None


@pytest.mark.integration
def test_module_integration():
    """Integration test for the Module class with multiple optimizers.

    trainer.fit() is called to ensure configure_optimizers work as expected.
    """
    # Define simple backbone and projector
    backbone = nn.Linear(1, 1)  # Simple nn.Module with a single parameter
    projector = nn.Linear(1, 1)  # Simple nn.Module with a single parameter

    # Define the module with multiple optimizers
    module = Module(
        backbone=backbone,
        projector=projector,
        forward=forward.simclr_forward,  # Or byol_forward, vicreg_forward, etc.
        simclr_loss=NTXEntLoss(temperature=0.5),
        optim={
            "backbone_opt": {
                "modules": "backbone",
                "optimizer": {"type": "AdamW", "lr": 1e-3},
            },
            "projector_opt": {
                "modules": "projector",
                "optimizer": {"type": "AdamW", "lr": 1e-3},
            },
        },
    )

    # Define dummy data loaders
    train_loader = torch.utils.data.DataLoader(
        [{"image": torch.tensor([1.0]), "label": torch.tensor([0])}], batch_size=1
    )

    val_loader = torch.utils.data.DataLoader(
        [{"image": torch.tensor([1.0]), "label": torch.tensor([0])}], batch_size=1
    )

    data = DataModule(train=train_loader, val=val_loader)

    # Define the trainer
    trainer = Trainer(
        max_epochs=0,
        num_sanity_val_steps=0,
        callbacks=[],
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(module, datamodule=data, ckpt_path=None)
