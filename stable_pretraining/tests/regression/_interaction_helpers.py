"""Small real training components shared by interaction regressions."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import stable_pretraining as spt


def _samples() -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(17)
    images = torch.randn(12, 4, generator=generator)
    targets = torch.randn(12, 3, generator=generator)
    return [
        {"image": image, "target": target, "label": torch.tensor(i % 2)}
        for i, (image, target) in enumerate(zip(images, targets))
    ]


def _forward(self, batch: dict, stage: str) -> dict[str, torch.Tensor]:
    if hasattr(self.backbone, "forward_student"):
        embedding = self.backbone.forward_student(batch["image"])
        teacher = self.backbone.forward_teacher(batch["image"])
        loss = (embedding - batch["target"]).square().mean()
        loss = loss + 0.1 * (embedding - teacher).square().mean()
    else:
        embedding = self.backbone(batch["image"])
        loss = (embedding - batch["target"]).square().mean()
    return {"embedding": embedding, "loss": loss}


def _model(frequency: int = 1, teacher: bool = False, adam: bool = False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        backbone = nn.Linear(4, 3)
    if teacher:
        backbone = spt.backbone.TeacherStudentWrapper(
            backbone, base_ema_coefficient=0.5, final_ema_coefficient=0.9
        )
    optimizer = (
        {"type": "Adam", "lr": 0.03}
        if adam
        else {"type": "SGD", "lr": 0.03, "momentum": 0.8}
    )
    return spt.Module(
        backbone=backbone,
        forward=_forward,
        optim={
            "optimizer": optimizer,
            "scheduler": {"type": "ExponentialLR", "gamma": 0.8},
            "interval": "step",
            "frequency": frequency,
        },
    )


def _probe(module, name: str = "probe", frequency: int = 1):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        head = nn.Linear(3, 2)
    return spt.OnlineProbe(
        module,
        name=name,
        input="embedding",
        target="label",
        probe=head,
        loss=nn.CrossEntropyLoss(),
        optimizer={"type": "Adam", "lr": 0.02},
        scheduler={"type": "ExponentialLR", "gamma": 0.7},
        accumulate_grad_batches=frequency,
        metrics={"accuracy": MulticlassAccuracy(2)},
        verbose=False,
    )


def _snapshot(trainer, module) -> dict[str, Any]:
    return deepcopy(
        {
            "model": module.state_dict(),
            "optimizers": [opt.state_dict() for opt in trainer.optimizers],
            "schedulers": [
                cfg.scheduler.state_dict() for cfg in trainer.lr_scheduler_configs
            ],
            "global_step": trainer.global_step,
        }
    )


class _Trace(pl.Callback):
    """Record trajectories and exercise Lightning's callback-state restoration."""

    def __init__(self):
        self.batches_seen = 0
        self.batches = []
        self.epochs = []
        self.start = None
        self.restored_batches = None

    def on_train_start(self, trainer, pl_module):
        self.start = _snapshot(trainer, pl_module)
        self.restored_batches = self.batches_seen

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batches_seen += 1
        self.batches.append(_snapshot(trainer, pl_module))

    def on_train_epoch_end(self, trainer, pl_module):
        self.epochs.append(_snapshot(trainer, pl_module))

    def state_dict(self):
        return {"batches_seen": self.batches_seen}

    def load_state_dict(self, state_dict):
        self.batches_seen = state_dict["batches_seen"]


def _run(
    module,
    callbacks: list,
    *,
    max_epochs: int = 2,
    max_steps: int = -1,
    ckpt_path: Path | None = None,
):
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=max_epochs,
        max_steps=max_steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    data = spt.data.DataModule(
        train=DataLoader(_samples(), batch_size=2, shuffle=False),
        val=DataLoader(_samples()[:4], batch_size=2, shuffle=False),
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        seed=59,
        ckpt_path=ckpt_path,
        weights_only=False,
    )()
    return trainer


def _assert_tree_close(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_close(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_tree_close(left, right)
    else:
        assert actual == expected
