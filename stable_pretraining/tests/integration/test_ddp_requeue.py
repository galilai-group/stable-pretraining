"""Integration test: DDP training with simulated SLURM preemption and requeue.

Launches a real multi-GPU DDP run on fake data, lets it complete 1 epoch,
then simulates requeue (same SLURM_JOB_ID) and verifies the second run
picks up the checkpoint and continues from epoch 1 instead of restarting.

Run with:
    srun --partition=main --gpus=2 --mem=64G -c 10 --pty \
        pytest stable_pretraining/tests/integration/test_ddp_requeue.py -v -s -m ddp
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.manager import Manager

pytestmark = pytest.mark.ddp


# ---------------------------------------------------------------------------
# Fake dataset — deterministic, no downloads
# ---------------------------------------------------------------------------


class FakeDataset(Dataset):
    """Tiny image dataset for fast GPU testing."""

    def __init__(self, num_samples=64, image_size=32, num_classes=10):
        g = torch.Generator().manual_seed(0)
        self.images = torch.randn(num_samples, 3, image_size, image_size, generator=g)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 64
NUM_CLASSES = 10


def _make_backbone():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, EMBED_DIM),
    )


def _make_module():
    return spt.Module(
        backbone=_make_backbone(),
        classifier=nn.Linear(EMBED_DIM, NUM_CLASSES),
        forward=forward.supervised_forward,
        supervised_loss=nn.CrossEntropyLoss(),
        optim={"optimizer": {"type": "Adam", "lr": 1e-3}},
    )


def _make_data():
    ds = FakeDataset(num_samples=64, image_size=32, num_classes=NUM_CLASSES)
    train_dl = DataLoader(ds, batch_size=8, drop_last=True, num_workers=0)
    return spt.data.DataModule(train=train_dl)


def _make_trainer_cfg(max_epochs, num_devices):
    return OmegaConf.create(
        {
            "_target_": "lightning.Trainer",
            "max_epochs": max_epochs,
            "accelerator": "gpu",
            "devices": num_devices,
            "strategy": "ddp",
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
            "logger": False,
        }
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestDDPRequeue:
    """Train 1 epoch with DDP, simulate preemption, verify requeue resumes."""

    def test_requeue_resumes_from_checkpoint(self, tmp_path, monkeypatch):
        num_devices = min(torch.cuda.device_count(), 2)
        assert num_devices >= 2, "This test requires at least 2 GPUs"

        # Use a fixed SLURM_JOB_ID so both runs resolve to the same run_dir
        fake_job_id = "99999"
        monkeypatch.setenv("SLURM_JOB_ID", fake_job_id)
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

        cache_dir = tmp_path / "spt_cache"
        spt.set(cache_dir=str(cache_dir))

        # ── Run 1: train for 1 epoch ──────────────────────────────────
        manager1 = Manager(
            trainer=_make_trainer_cfg(max_epochs=1, num_devices=num_devices),
            module=_make_module(),
            data=_make_data(),
            seed=42,
        )
        manager1()

        # Verify run_dir was created and checkpoint exists
        run_dir_1 = manager1._run_dir
        assert run_dir_1.is_dir(), "run_dir was not created"
        assert str(run_dir_1).startswith(str(cache_dir))

        last_ckpt = run_dir_1 / "checkpoints" / "last.ckpt"
        assert last_ckpt.is_file(), "last.ckpt not saved after epoch 1"

        # Verify run_meta.json sidecar was written
        run_meta = run_dir_1 / "run_meta.json"
        assert run_meta.is_file(), "run_meta.json not written"

        # Inspect the checkpoint: should be at epoch 0 (0-indexed), global_step > 0
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 1, f"Expected epoch=1, got {ckpt['epoch']}"
        assert ckpt["global_step"] > 0, "global_step should be > 0"
        step_after_epoch1 = ckpt["global_step"]

        # ── Run 2: simulate requeue (same SLURM_JOB_ID) ──────────────
        manager2 = Manager(
            trainer=_make_trainer_cfg(max_epochs=2, num_devices=num_devices),
            module=_make_module(),
            data=_make_data(),
            seed=42,
        )
        manager2()

        # Same run_dir should be reused (not a fresh one)
        run_dir_2 = manager2._run_dir
        assert run_dir_2 == run_dir_1, (
            f"Requeue created a new run_dir!\n"
            f"  expected: {run_dir_1}\n"
            f"  got:      {run_dir_2}"
        )

        # The checkpoint should now reflect 2 epochs of training
        ckpt2 = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        assert ckpt2["epoch"] == 2, (
            f"Expected epoch=2 after requeue, got {ckpt2['epoch']} "
            "(training may have restarted from scratch)"
        )
        assert ckpt2["global_step"] > step_after_epoch1, (
            f"global_step did not advance: {ckpt2['global_step']} <= {step_after_epoch1}"
        )
