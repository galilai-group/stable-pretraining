"""End-to-end Lightning DDP integration test for LeWM."""

import lightning as pl
import torch
from torch.utils.data import DataLoader

from stable_pretraining.tests.unit.test_lewm_training import (
    SyntheticSequences,
    make_method,
)


def test_lewm_lightning_ddp(tmp_path):
    pl.seed_everything(42)

    model = make_method()
    loader = DataLoader(
        SyntheticSequences(),
        batch_size=2,
        num_workers=0,
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp_find_unused_parameters_true",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(model, train_dataloaders=loader)

    assert trainer.global_step == 2

    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
