"""CPU wiring check for the actual CIFAR training entry point."""

import copy
import importlib
import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import lightning as pl
import pytest
import torch

import stable_pretraining as spt

pytestmark = pytest.mark.integration


def test_short_training_updates_encoder_validates_and_saves_probe(
    tmp_path, monkeypatch
):
    if os.name == "nt":
        # Upstream's signal diagnostics require POSIX; training math is unchanged.
        manager_module = importlib.import_module("stable_pretraining.manager")
        monkeypatch.setattr(manager_module, "print_signal_info", lambda *args: None)
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "examples" / "simco_cifar.py")
    )
    original_class = spt.SimCo
    captured = {}

    def capture_initial_encoder(*args, **kwargs):
        model = original_class(*args, **kwargs)
        captured["model"] = model
        captured["initial_weight"] = model.backbone.conv1.weight.detach().clone()
        return model

    monkeypatch.setattr(spt, "SimCo", capture_initial_encoder)
    args = example["_parser"]().parse_args(
        [
            "--synthetic",
            "--accelerator",
            "cpu",
            "--workers",
            "0",
            "--batch-size",
            "16",
            "--hidden-dim",
            "16",
            "--projection-dim",
            "8",
            "--warmup-epochs",
            "0",
            "--epochs",
            "1",
            "--max-steps",
            "2",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    example["_run"](args)
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["optimizer_steps"] == 2
    assert "eval/linear_probe_top1_accuracy" in summary["metrics"]
    assert Path(summary["best_checkpoint"]).is_file()
    checkpoint = torch.load(
        summary["last_checkpoint"], map_location="cpu", weights_only=False
    )
    assert len(checkpoint["optimizer_states"]) == 2
    assert checkpoint["global_step"] == summary["optimizer_steps"]
    assert Path(summary["last_checkpoint"]).name == "latest.ckpt"
    assert any(
        name.startswith("callbacks_modules.linear_probe.")
        for name in checkpoint["state_dict"]
    )
    assert not torch.equal(
        captured["initial_weight"], captured["model"].backbone.conv1.weight
    )
    record_paths = sorted(
        (args.output_dir / "validation_records").glob("validation_*.pt")
    )
    assert record_paths
    records = [
        torch.load(path, map_location="cpu", weights_only=True) for path in record_paths
    ]
    sample_indices = torch.cat([record["sample_idx"] for record in records])
    torch.testing.assert_close(sample_indices.sort().values, torch.arange(32))
    logits = torch.cat([record["linear_probe_preds"] for record in records])
    labels = torch.cat([record["label"] for record in records])
    assert logits.shape == (32, 10)
    measured_accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    assert summary["metrics"]["eval/linear_probe_top1_accuracy"] == pytest.approx(
        measured_accuracy
    )


def test_manager_resume_restores_full_training_state_before_continuing(
    tmp_path, monkeypatch
):
    """Resume an interrupted full schedule, including the online probe state."""
    if os.name == "nt":
        manager_module = importlib.import_module("stable_pretraining.manager")
        monkeypatch.setattr(manager_module, "print_signal_info", lambda *args: None)
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "examples" / "simco_cifar.py")
    )
    original_trainer = pl.Trainer
    trainers = []
    restored = {}
    saved_checkpoint = None

    def assert_state_equal(actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected)
            for actual_item, expected_item in zip(actual, expected):
                assert_state_equal(actual_item, expected_item)
        else:
            assert actual == expected

    class ContinuationCheck(pl.Callback):
        def __init__(self, stop_after_first_epoch):
            self.stop_after_first_epoch = stop_after_first_epoch

        def on_train_start(self, trainer, pl_module):
            if saved_checkpoint is None:
                assert trainer.global_step == 0
                return
            assert trainer.global_step == saved_checkpoint["global_step"] == 4
            # End-of-epoch checkpoints store the completed epoch's index;
            # continuation starts at the next epoch without replaying its data.
            assert trainer.current_epoch == saved_checkpoint["epoch"] + 1 == 1
            assert_state_equal(pl_module.state_dict(), saved_checkpoint["state_dict"])
            optimizer_states = [
                optimizer.state_dict() for optimizer in trainer.optimizers
            ]
            scheduler_states = [
                config.scheduler.state_dict() for config in trainer.lr_scheduler_configs
            ]
            assert_state_equal(optimizer_states, saved_checkpoint["optimizer_states"])
            assert_state_equal(scheduler_states, saved_checkpoint["lr_schedulers"])
            restored.update(
                global_step=trainer.global_step,
                epoch=trainer.current_epoch,
                optimizer_states=copy.deepcopy(optimizer_states),
                scheduler_states=copy.deepcopy(scheduler_states),
            )

        def on_train_epoch_end(self, trainer, pl_module):
            if self.stop_after_first_epoch and trainer.current_epoch == 0:
                # Preserve the two-epoch scheduler horizon when interrupting.
                trainer.should_stop = True

    def capture_trainer(*args, **kwargs):
        kwargs["callbacks"] = [
            *kwargs["callbacks"],
            ContinuationCheck(stop_after_first_epoch=not trainers),
        ]
        trainer = original_trainer(*args, **kwargs)
        trainers.append(trainer)
        return trainer

    # Restrict the factory to the example: Manager retains the actual Trainer
    # class for its isinstance checks and full-state checkpoint loading.
    monkeypatch.setitem(
        example["_run"].__globals__,
        "pl",
        SimpleNamespace(seed_everything=pl.seed_everything, Trainer=capture_trainer),
    )
    arguments = [
        "--synthetic",
        "--accelerator",
        "cpu",
        "--workers",
        "0",
        "--batch-size",
        "16",
        "--hidden-dim",
        "16",
        "--projection-dim",
        "8",
        "--warmup-epochs",
        "1",
        "--epochs",
        "2",
    ]
    initial_dir = tmp_path / "interrupted"
    initial_args = example["_parser"]().parse_args(
        [*arguments, "--output-dir", str(initial_dir)]
    )
    example["_run"](initial_args)
    initial_summary = json.loads((initial_dir / "summary.json").read_text())
    assert initial_summary["optimizer_steps"] == 4
    checkpoint_path = Path(initial_summary["last_checkpoint"])
    saved_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    assert saved_checkpoint["epoch"] == 0
    assert len(saved_checkpoint["optimizer_states"]) == 2
    assert len(saved_checkpoint["lr_schedulers"]) == 2
    assert saved_checkpoint["lr_schedulers"][0]["max_steps"] == 8
    assert saved_checkpoint["lr_schedulers"][0]["warmup_steps"] == 4
    for scheduler_state in saved_checkpoint["lr_schedulers"]:
        assert scheduler_state["last_epoch"] == 4
    for optimizer_state in saved_checkpoint["optimizer_states"]:
        assert optimizer_state["state"]
        momentum = [
            state["momentum_buffer"] for state in optimizer_state["state"].values()
        ]
        assert all(torch.isfinite(buffer).all() for buffer in momentum)
        assert any(buffer.norm() > 0 for buffer in momentum)
    probe_keys = [
        name
        for name in saved_checkpoint["state_dict"]
        if name.startswith("callbacks_modules.linear_probe.")
    ]
    assert probe_keys

    resumed_dir = tmp_path / "resumed"
    resumed_args = example["_parser"]().parse_args(
        [
            *arguments,
            "--resume",
            str(checkpoint_path),
            "--output-dir",
            str(resumed_dir),
        ]
    )
    example["_run"](resumed_args)
    assert restored["global_step"] == 4
    assert restored["epoch"] == 1
    resumed_summary = json.loads((resumed_dir / "summary.json").read_text())
    assert resumed_summary["optimizer_steps"] == 8
    assert resumed_summary["protocol_sha256"] == initial_summary["protocol_sha256"]
    final_checkpoint = torch.load(
        resumed_summary["last_checkpoint"], map_location="cpu", weights_only=False
    )
    assert final_checkpoint["global_step"] == 8
    assert final_checkpoint["epoch"] == 1
    for final_state, restored_state in zip(
        final_checkpoint["lr_schedulers"], restored["scheduler_states"]
    ):
        assert final_state["last_epoch"] == restored_state["last_epoch"] + 4 == 8
        assert final_state["_step_count"] == restored_state["_step_count"] + 4
    assert final_checkpoint["lr_schedulers"][0]["max_steps"] == 8
    assert final_checkpoint["optimizer_states"][0]["param_groups"][0][
        "lr"
    ] == pytest.approx(0.0, abs=1e-12)
    for prefix in ("backbone.", "projector.", "callbacks_modules.linear_probe."):
        assert any(
            not torch.equal(value, saved_checkpoint["state_dict"][name])
            for name, value in final_checkpoint["state_dict"].items()
            if name.startswith(prefix) and value.is_floating_point()
        )
    for final_state, restored_state in zip(
        final_checkpoint["optimizer_states"], restored["optimizer_states"]
    ):
        assert any(
            not torch.equal(
                state["momentum_buffer"],
                restored_state["state"][parameter_id]["momentum_buffer"],
            )
            for parameter_id, state in final_state["state"].items()
        )
    assert len(trainers) == 2
