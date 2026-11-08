"""Integration tests: VIbCReg trained end-to-end through :class:`spt.Manager`.

The method class is used directly as the Lightning module through its
dict-based ``forward(batch, stage)`` contract, on fake data (no downloads).

The CPU test runs anywhere. The DDP test needs at least 2 GPUs and is skipped
otherwise; run it with, e.g.:
    srun --gres=gpu:2 --pty \
        pytest stable_pretraining/tests/integration/test_vibcreg.py -v -s -m ddp
"""

import lightning as pl
import pytest
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining.backbone import IterNorm

EMBED_DIM = 32
NUM_CLASSES = 4


class _TwoViewDataset(Dataset):
    """Deterministic fake images; two noisy views per sample in train mode."""

    def __init__(self, num_samples: int = 64, train: bool = True):
        g = torch.Generator().manual_seed(0)
        self.images = torch.randn(num_samples, 3, 16, 16, generator=g)
        self.labels = torch.randint(0, NUM_CLASSES, (num_samples,), generator=g)
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if not self.train:
            return {"image": image, "label": label}
        return {
            f"view_{v}": {
                "image": image + 0.1 * torch.randn_like(image),
                "label": label,
            }
            for v in range(2)
        }


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, EMBED_DIM),
        )

    def forward(self, x):
        return self.net(x)


def _make_module() -> spt.VIbCReg:
    module = spt.VIbCReg(
        encoder_name=_TinyEncoder(), projector_dims=(64, 64), iternorm_groups=8
    )
    module.optim = {"optimizer": {"type": "Adam", "lr": 1e-3}}
    return module


def _make_data(batch_size: int = 16) -> spt.data.DataModule:
    return spt.data.DataModule(
        train=DataLoader(
            _TwoViewDataset(train=True),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        ),
        val=DataLoader(_TwoViewDataset(train=False), batch_size=batch_size),
    )


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:`isinstance.treespec, LeafSpec.` is deprecated")
class TestVIbCRegManagerCPU:
    """VIbCReg trains, checkpoints and feeds callbacks through Manager on CPU."""

    def test_manager_trains_and_checkpoints(self):
        pl.seed_everything(0, workers=True)
        module = _make_module()
        iternorm = next(m for m in module.projector if isinstance(m, IterNorm))
        initial_wm = iternorm.running_wm.clone()

        probe = spt.callbacks.OnlineProbe(
            module,
            name="linear_probe",
            input="embedding",
            target="label",
            probe=nn.Linear(EMBED_DIM, NUM_CLASSES),
            loss=nn.CrossEntropyLoss(),
            metrics={"top1": torchmetrics.classification.MulticlassAccuracy(4)},
        )
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            callbacks=[probe],
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            logger=False,
        )

        manager = spt.Manager(trainer=trainer, module=module, data=_make_data())
        manager()

        metrics = trainer.callback_metrics
        assert torch.isfinite(metrics["fit/loss_epoch"])
        assert "eval/linear_probe_top1" in metrics
        assert not torch.equal(iternorm.running_wm, initial_wm), (
            "IterNorm running statistics were never updated"
        )

        last_ckpt = manager._run_dir / "checkpoints" / "last.ckpt"
        assert last_ckpt.is_file(), "last.ckpt not saved"
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        # 64 samples / batch 16 = 4 steps per epoch, 2 epochs
        assert ckpt["global_step"] == 8
        assert any("running_wm" in k for k in ckpt["state_dict"]), (
            "IterNorm buffers missing from the checkpoint"
        )


@pytest.mark.ddp
@pytest.mark.filterwarnings("ignore:.*does not have many workers")
class TestVIbCRegManagerDDP:
    """VIbCReg trains under multi-GPU DDP through Manager."""

    def test_ddp_training_completes_and_checkpoints(self, tmp_path):
        num_devices = min(torch.cuda.device_count(), 2)
        assert num_devices >= 2, "This test requires at least 2 GPUs"
        spt.set(cache_dir=str(tmp_path / "spt_cache"))

        trainer = OmegaConf.create(
            {
                "_target_": "lightning.Trainer",
                "max_epochs": 2,
                "accelerator": "gpu",
                "devices": num_devices,
                "strategy": "ddp",
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "num_sanity_val_steps": 0,
                "logger": False,
            }
        )
        manager = spt.Manager(
            trainer=trainer, module=_make_module(), data=_make_data(), seed=0
        )
        manager()

        last_ckpt = manager._run_dir / "checkpoints" / "last.ckpt"
        assert last_ckpt.is_file(), "last.ckpt not saved"
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        # DDP shards 64 samples over 2 ranks: 32 / batch 16 = 2 steps per epoch
        assert ckpt["global_step"] == 4
        state = ckpt["state_dict"]
        assert all(
            torch.isfinite(v).all() for v in state.values() if v.is_floating_point()
        )
