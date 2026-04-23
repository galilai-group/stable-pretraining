# test_log_unused_parameters_once.py

import pytest
import sys
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from contextlib import contextmanager

import lightning.pytorch as pl
import stable_pretraining as spt

from loguru import logger


@contextmanager
def capture_loguru():
    """Context manager to capture loguru logs."""
    messages = []

    def sink(message):
        messages.append(str(message))

    logger.remove()
    handler_id = logger.add(sink, format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler_id)
        logger.add(sys.stderr)


class RandomDataset(Dataset):
    """Minimal testing class that returns dict batches."""

    def __init__(self, length: int = 16, in_dim: int = 8, out_dim: int = 4):
        self.length = length
        self.in_dim = in_dim
        self.out_dim = out_dim

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "x": torch.randn(self.in_dim),
            "y": torch.randint(0, self.out_dim, (1,)).item(),
        }


# ---- Automatic Optimization Models (plain LightningModule) ----


class AutoModelAllUsed(pl.LightningModule):
    """Model with automatic optimization where all parameters are used."""

    def __init__(self, in_dim: int = 8, out_dim: int = 4):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        logits = self(batch["x"])
        loss = self.loss_fn(logits, batch["y"])
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class AutoModelWithUnusedParam(pl.LightningModule):
    """Model with automatic optimization that has unused parameters."""

    def __init__(self, in_dim: int = 8, out_dim: int = 4):
        super().__init__()
        self.used_layer = nn.Linear(in_dim, out_dim)
        self.unused_layer = nn.Linear(in_dim, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.used_layer(x)

    def training_step(self, batch, batch_idx):
        logits = self(batch["x"])
        loss = self.loss_fn(logits, batch["y"])
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


# ---- Manual Optimization Models (spt.Module) ----


class ManualModelAllUsed(spt.Module):
    """Model with manual optimization where all parameters are used."""

    def __init__(self, in_dim: int = 8, out_dim: int = 4):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch, stage):
        logits = self.layer(batch["x"])
        loss = self.loss_fn(logits, batch["y"])
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class ManualModelWithUnusedParam(spt.Module):
    """Model with manual optimization that has unused parameters."""

    def __init__(self, in_dim: int = 8, out_dim: int = 4):
        super().__init__()
        self.used_layer = nn.Linear(in_dim, out_dim)
        self.unused_layer = nn.Linear(in_dim, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch, stage):
        logits = self.used_layer(batch["x"])
        loss = self.loss_fn(logits, batch["y"])
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class ManualModelNoBackward(spt.Module):
    """Model with manual optimization that never calls backward."""

    def __init__(self, in_dim: int = 8, out_dim: int = 4):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch, stage):
        logits = self.layer(batch["x"])
        loss = self.loss_fn(logits, batch["y"])
        return {"loss": loss.detach()}  # Detached = no gradients flow

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


# ---- Trainer Utilities ----


def _make_trainer(limit_train_batches: int = 1):
    """Utility to create a tiny trainer for unit tests."""
    return pl.Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        log_every_n_steps=1,
        default_root_dir=tempfile.gettempdir(),
    )


# ---- Tests: Automatic Optimization ----


@pytest.mark.unit
def test_auto_all_parameters_used():
    """Automatic optimization: all parameters receive grads."""
    model = AutoModelAllUsed()
    dataset = RandomDataset()
    loader = DataLoader(dataset, batch_size=4)
    trainer = _make_trainer()

    with capture_loguru() as messages:
        trainer.fit(model, train_dataloaders=loader)

    text = "\n".join(messages)
    assert "all tracked parameters received gradients" in text
    assert "did NOT receive gradients" not in text


@pytest.mark.unit
def test_auto_unused_parameters_logged():
    """Automatic optimization: unused parameters are logged."""
    model = AutoModelWithUnusedParam()
    dataset = RandomDataset()
    loader = DataLoader(dataset, batch_size=4)
    trainer = _make_trainer()

    with capture_loguru() as messages:
        trainer.fit(model, train_dataloaders=loader)

    text = "\n".join(messages)
    assert "did NOT receive gradients on the first backward pass" in text

    for name, _ in model.unused_layer.named_parameters():
        full_name = f"unused_layer.{name}"
        assert full_name in text


@pytest.mark.unit
def test_auto_callback_runs_only_once():
    """Automatic optimization: callback only runs once across multiple batches."""
    model = AutoModelAllUsed()
    dataset = RandomDataset(length=32)
    loader = DataLoader(dataset, batch_size=4)
    trainer = _make_trainer(limit_train_batches=2)

    with capture_loguru() as messages:
        trainer.fit(model, train_dataloaders=loader)

    text = "\n".join(messages)
    assert text.count("hooks removed, callback disabled") == 1


# ---- Tests: Manual Optimization ----


@pytest.mark.unit
def test_manual_all_parameters_used():
    """Manual optimization: all parameters receive grads."""
    model = ManualModelAllUsed()
    dataset = RandomDataset()
    loader = DataLoader(dataset, batch_size=4)
    trainer = _make_trainer()

    with capture_loguru() as messages:
        trainer.fit(model, train_dataloaders=loader)

    text = "\n".join(messages)
    assert "all tracked parameters received gradients" in text
    assert "did NOT receive gradients" not in text


@pytest.mark.unit
def test_manual_unused_parameters_logged():
    """Manual optimization: unused parameters are logged."""
    model = ManualModelWithUnusedParam()
    dataset = RandomDataset()
    loader = DataLoader(dataset, batch_size=4)
    trainer = _make_trainer()

    with capture_loguru() as messages:
        trainer.fit(model, train_dataloaders=loader)

    text = "\n".join(messages)
    assert "did NOT receive gradients on the first backward pass" in text

    for name, _ in model.unused_layer.named_parameters():
        full_name = f"unused_layer.{name}"
        assert full_name in text


@pytest.mark.unit
def test_manual_callback_runs_only_once():
    """Manual optimization: callback only runs once across multiple batches."""
    model = ManualModelAllUsed()
    dataset = RandomDataset(length=32)
    loader = DataLoader(dataset, batch_size=4)
    trainer = _make_trainer(limit_train_batches=2)

    with capture_loguru() as messages:
        trainer.fit(model, train_dataloaders=loader)

    text = "\n".join(messages)
    assert text.count("hooks removed, callback disabled") == 1
