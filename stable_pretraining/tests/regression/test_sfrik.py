"""Actual training, checkpoint continuation, and distributed SFRIK gradients."""

from datetime import timedelta
from pathlib import Path

import lightning as pl
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import stable_pretraining as spt
from stable_pretraining.losses import SFRIKLoss
from ._interaction_helpers import _assert_tree_close

pytestmark = pytest.mark.regression


def _model():
    torch.manual_seed(17)
    return spt.SFRIK(
        backbone=nn.Linear(4, 6),
        projector=nn.Linear(6, 3),
        sim_coeff=3,
        optim={
            "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
            "scheduler": {"type": "ExponentialLR", "gamma": 0.9},
            "interval": "epoch",
        },
    )


def _data():
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(12, 4, generator=generator)
    train = [
        {
            "views": [
                {"image": row, "label": torch.tensor(i % 2)},
                {"image": row + 0.05, "label": torch.tensor(i % 2)},
            ]
        }
        for i, row in enumerate(x)
    ]
    val = [{"image": row, "label": torch.tensor(i % 2)} for i, row in enumerate(x)]
    return spt.data.DataModule(
        train=DataLoader(train, batch_size=4), val=DataLoader(val, batch_size=4)
    )


def _fit(model, path, epochs, checkpoint=None, callbacks=(), devices=1):
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=devices,
        strategy="ddp_spawn" if devices > 1 else "auto",
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=path,
        callbacks=list(callbacks),
        log_every_n_steps=1,
    )
    spt.Manager(
        trainer=trainer,
        module=model,
        data=_data(),
        seed=19,
        ckpt_path=checkpoint,
        weights_only=False,
    )()
    return trainer


@pytest.mark.usefixtures("interaction_runtime")
def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    full = _model()
    initial = full.backbone.weight.detach().clone()
    full_trainer = _fit(full, tmp_path / "full", 2)
    split = _model()
    first = _fit(split, tmp_path / "split", 1)
    checkpoint = tmp_path / "sfrik.ckpt"
    first.save_checkpoint(checkpoint)
    restored = spt.SFRIK.load_from_checkpoint(
        checkpoint,
        backbone=nn.Linear(4, 6),
        projector=nn.Linear(6, 3),
        optim=split.optim,
    )
    assert restored.sfrik_loss.sim_coeff == 3
    assert restored.sfrik_loss.kernel_weights == (1.0, 40.0, 40.0)
    _assert_tree_close(restored.state_dict(), split.state_dict())
    second = _fit(restored, tmp_path / "restored", 2, checkpoint)
    assert full_trainer.global_step == second.global_step == 6
    assert not torch.equal(initial, full.backbone.weight)
    _assert_tree_close(restored.state_dict(), full.state_dict())
    _assert_tree_close(
        second.optimizers[0].state_dict(), full_trainer.optimizers[0].state_dict()
    )
    _assert_tree_close(
        second.lr_scheduler_configs[0].scheduler.state_dict(),
        full_trainer.lr_scheduler_configs[0].scheduler.state_dict(),
    )


@pytest.mark.usefixtures("interaction_runtime")
def test_online_probe_receives_matching_labels(tmp_path):
    model = _model()
    probe = spt.OnlineProbe(
        model,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(6, 2),
        loss=nn.CrossEntropyLoss(),
        metrics={"accuracy": MulticlassAccuracy(2)},
        verbose=False,
    )
    trainer = _fit(model, tmp_path, 1, callbacks=[probe])
    assert trainer.global_step == 3
    assert "eval/linear_probe_accuracy" in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics["eval/linear_probe_accuracy"])


class _TrainingSnapshot(pl.Callback):
    """Return evidence from the spawned trainer without relying on parent state."""

    def __init__(self, path):
        self.path = path

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            torch.save(
                {
                    "step": trainer.global_step,
                    "weight": pl_module.backbone.weight.detach().cpu(),
                },
                self.path,
            )


@pytest.mark.usefixtures("interaction_runtime")
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_manager_two_process_spawn_training(tmp_path):
    model = _model()
    initial = model.backbone.weight.detach().clone()
    snapshot = tmp_path / "ddp.pt"
    _fit(model, tmp_path, 1, callbacks=[_TrainingSnapshot(snapshot)], devices=2)
    result = torch.load(snapshot, weights_only=True)
    assert result["step"] == 2
    assert torch.isfinite(result["weight"]).all()
    assert not torch.equal(initial, result["weight"])


class _DistributedObjective(nn.Module):
    """Exercise a real DDP parameter gradient through the differentiable gather."""

    def __init__(self, gather):
        super().__init__()
        self.encoder = nn.Linear(4, 3, bias=False).double()
        self.loss = SFRIKLoss(sim_coeff=3, gather_distributed=gather)

    def forward(self, a, b):
        return self.loss(self.encoder(a), self.encoder(b))


def _distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        torch.manual_seed(11)
        reference = _DistributedObjective(False)
        actual = _DistributedObjective(True)
        actual.load_state_dict(reference.state_dict())
        actual = DistributedDataParallel(actual)
        a, b = (
            torch.randn(8, 4, dtype=torch.double),
            torch.randn(8, 4, dtype=torch.double),
        )
        expected_loss = reference(a, b)
        expected_loss.backward()
        selection = slice(rank * 4, (rank + 1) * 4)
        actual_loss = actual(a[selection], b[selection])
        actual_loss.backward()
        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(
            actual.module.encoder.weight.grad, reference.encoder.weight.grad
        )
        torch.optim.SGD(actual.parameters(), lr=0.01).step()
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        torch.testing.assert_close(
            actual.module.encoder.weight, reference.encoder.weight
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_two_process_ddp_matches_global_batch(tmp_path):
    # Not marked ddp: that repository marker skips tests without two CUDA GPUs.
    mp.spawn(
        _distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True
    )
