"""Dict-native DirectCLR keeps full optimizer/BN state across a real restart."""

from pathlib import Path

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import stable_pretraining as spt

from ._interaction_helpers import _assert_tree_close, _probe, _snapshot

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]


class _StopAfterEpoch(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def _module():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        backbone = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
    options = dict(
        backbone=backbone,
        subspace_dim=2,
        optim={
            "optimizer": {"type": "SGD", "lr": 0.03, "momentum": 0.9},
            "scheduler": {"type": "ExponentialLR", "gamma": 0.9},
            "interval": "step",
        },
    )
    return spt.Module(
        forward=spt.forward.directclr,
        directclr_loss=spt.losses.DirectCLRLoss(0.2),
        **options,
    )


def _run(module, directory, stop=False, checkpoint=None):
    generator = torch.Generator().manual_seed(19)
    images = torch.randn(12, 4, generator=generator)
    paired = [
        {
            "views": [
                {"image": image, "label": torch.tensor(i % 2)},
                {"image": image + 0.1, "label": torch.tensor(i % 2)},
            ]
        }
        for i, image in enumerate(images)
    ]
    validation = [
        {"image": image, "label": torch.tensor(i % 2)}
        for i, image in enumerate(images[:4])
    ]
    callbacks = [_probe(module)]
    if stop:
        callbacks.append(_StopAfterEpoch())
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=directory,
        callbacks=callbacks,
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(
            train=DataLoader(paired, batch_size=4, shuffle=False),
            val=DataLoader(validation, batch_size=4),
        ),
        seed=59,
        ckpt_path=checkpoint,
        weights_only=False,
    )()
    return trainer


def test_directclr_manager_resume_matches_uninterrupted_training(tmp_path):
    """Check model, BN, online probe, momentum and scheduler at a fixed horizon."""
    reference = _module()
    full = _run(reference, tmp_path / "full")
    interrupted = _module()
    first = _run(interrupted, tmp_path / "first", stop=True)
    assert first.global_step == 3
    checkpoint = tmp_path / "restart.ckpt"
    first.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert len(saved["optimizer_states"]) == len(saved["lr_schedulers"]) == 2
    assert all(state["state"] for state in saved["optimizer_states"])
    resumed = _module()
    second = _run(resumed, tmp_path / "second", checkpoint=Path(checkpoint))
    assert second.global_step == full.global_step == 6
    _assert_tree_close(_snapshot(second, resumed), _snapshot(full, reference))
    _assert_tree_close(
        second.callback_metrics["eval/probe_accuracy"],
        full.callback_metrics["eval/probe_accuracy"],
    )
