"""Experiment reports reject incomplete, nonfinite, or unmatched evidence."""

import csv
import json
from pathlib import Path
import runpy

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

pytestmark = pytest.mark.unit


@pytest.fixture
def reporter():
    path = (
        Path(__file__).resolve().parents[3] / "benchmarks/flatnce/summarize_results.py"
    )
    return runpy.run_path(str(path))


def _csv(path, missing_validation=False, bad_gradient=False):
    fields = [
        "epoch",
        "diagnostics/gradient_norm",
        "diagnostics/embedding_std",
        "eval/linear_probe_top1_epoch",
        "eval/knn_probe_accuracy",
        "fit/loss_epoch",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in (0, 1):
            writer.writerow(
                {
                    "epoch": epoch,
                    "diagnostics/gradient_norm": "nan" if bad_gradient else 2,
                    "diagnostics/embedding_std": 0.5,
                }
            )
            if not (missing_validation and epoch == 1):
                writer.writerow(
                    {
                        "epoch": epoch,
                        "eval/linear_probe_top1_epoch": 0.2,
                        "eval/knn_probe_accuracy": 0.3,
                        "fit/loss_epoch": 1,
                    }
                )


def test_epoch_reporting_merges_step_and_validation_records(tmp_path, reporter):
    path = tmp_path / "metrics.csv"
    _csv(path)
    rows = reporter["_read_epochs"](path, 2)
    assert [row["epoch"] for row in rows] == [1, 2]
    assert all(row["updates_logged"] == 1 for row in rows)
    assert rows[-1]["linear_probe_macro_top1"] == 0.2
    assert rows[-1]["knn_macro_top1"] == 0.3


@pytest.mark.parametrize(
    "missing,bad,match",
    [(True, False, "Missing validation"), (False, True, "Nonfinite")],
)
def test_reporting_rejects_missing_or_nonfinite_results(
    tmp_path, reporter, missing, bad, match
):
    path = tmp_path / "metrics.csv"
    _csv(path, missing, bad)
    with pytest.raises(ValueError, match=match):
        reporter["_read_epochs"](path, 2)


def test_reporting_rejects_truncated_run_even_with_summary(tmp_path, reporter):
    summary = {
        "dataset": "CIFAR-10",
        "train_images": 4,
        "global_step": 1,
        "settings": {
            "method": "flatnce",
            "devices": 1,
            "batch_size": 4,
            "epochs": 2,
            "smoke_test": False,
        },
    }
    (tmp_path / "experiment_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="expected 2 updates"):
        reporter["_load_run"](tmp_path, "flatnce")


def test_reporting_rejects_changed_hyperparameters(reporter):
    settings = dict.fromkeys(reporter["_MATCHED_SETTINGS"], 0)
    first = {"settings": settings, "train_images": 45000, "evaluation_images": 5000}
    second = {**first, "settings": {**settings, "temperature": 0.5}}
    with pytest.raises(ValueError, match="temperature"):
        reporter["_validate_pair"](first, second)


def test_runner_saves_resumable_checkpoint_before_second_epoch(tmp_path):
    runner = runpy.run_path(
        str(
            Path(__file__).resolve().parents[3]
            / "benchmarks/flatnce/flatnce-resnet18.py"
        )
    )
    checkpoint_dir = tmp_path / "checkpoints"

    class TinyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 1)

        def training_step(self, batch, batch_idx):
            inputs, targets = batch
            return nn.functional.mse_loss(self.layer(inputs), targets)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    class CheckAtSecondEpoch(pl.Callback):
        observed = False

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch == 1:
                assert (checkpoint_dir / "last.ckpt").is_file()
                self.observed = True

    check = CheckAtSecondEpoch()
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=2,
        limit_train_batches=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[runner["_checkpoint_callback"](checkpoint_dir), check],
    )
    data = DataLoader(TensorDataset(torch.randn(4, 2), torch.randn(4, 1)))
    trainer.fit(TinyModel(), data)
    assert check.observed
