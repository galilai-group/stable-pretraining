"""Run the Lightning DDP smoke test in a separate process."""

import sys
from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import DataLoader

from stable_pretraining.tests.distributed.test_ressl_ddp import (
    _CheckQueueCallback,
    _make_model,
    _TwoViews,
)


def main():
    pl.seed_everything(42)
    model = _make_model(queue_size=7)
    loader = DataLoader(_TwoViews(), batch_size=2, num_workers=0)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp",
        max_steps=3,
        callbacks=[_CheckQueueCallback()],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=Path(sys.argv[1]),
    )
    trainer.fit(model, train_dataloaders=loader)

    assert trainer.global_step == 3
    assert bool(torch.isfinite(model.queue.out).all())


if __name__ == "__main__":
    main()
