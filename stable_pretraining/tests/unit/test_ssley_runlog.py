"""Regression tests for AMP overflow, failure detection and resume status."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def runlog():
    path = Path(__file__).resolve().parents[3] / "benchmarks/cifar10"
    sys.path.insert(0, str(path))
    try:
        spec = importlib.util.spec_from_file_location(
            "ssley_paper_runlog", path / "ssley-paper.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module.RunLog
    finally:
        sys.path.pop(0)


def _trainer(scaler):
    return SimpleNamespace(
        precision_plugin=SimpleNamespace(scaler=scaler),
        global_step=0,
        current_epoch=0,
        callback_metrics={},
    )


def test_amp_overflow_skips_and_recovers(runlog, tmp_path):
    model = torch.nn.Linear(2, 1)
    initial = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=16)
    trainer = _trainer(scaler)
    callback = runlog(tmp_path, None)
    callback.on_train_start(trainer, model)
    assert json.loads((tmp_path / "run-status.json").read_text())["state"] == "running"
    for invalid in [True, False]:
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(3, 2)).sum()
        if invalid:
            loss = loss * float("inf")
        scaled = scaler.scale(loss)
        callback.on_before_backward(trainer, model, scaled)
        scaled.backward()
        scaler.unscale_(optimizer)
        callback.on_before_optimizer_step(trainer, model, optimizer)
        scaler.step(optimizer)
        scaler.update()
        callback.on_train_batch_end(trainer, model, {"loss": loss}, {}, 0)
        if invalid:
            torch.testing.assert_close(model.weight, initial, rtol=0, atol=0)
            assert scaler.get_scale() == 8
        trainer.global_step += 1
    assert not torch.equal(model.weight, initial)
    assert callback.amp_skips == callback.nonfinite_batches == 1
    assert callback.consecutive_nonfinite == 0
    trainer.callback_metrics = {
        "fit/loss_epoch": torch.tensor(float("inf")),
        "val/top1": torch.tensor(0.5),
    }
    callback.on_train_epoch_end(trainer, model)
    row = json.loads((tmp_path / "history.jsonl").read_text())
    assert row["metrics"]["fit/loss_epoch"] is None
    assert row["nonfinite_metrics"] == ["fit/loss_epoch"]


def test_unscaled_failure_and_damaged_weights_still_stop(runlog, tmp_path):
    model = torch.nn.Linear(2, 1)
    callback = runlog(tmp_path, None)
    with pytest.raises(FloatingPointError, match="without AMP"):
        callback.on_before_backward(_trainer(None), model, torch.tensor(float("inf")))
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="model state"):
        callback.on_before_backward(
            _trainer(torch.amp.GradScaler("cpu")), model, torch.tensor(float("nan"))
        )


def test_persistent_invalid_loss_stops(runlog, tmp_path):
    model = torch.nn.Linear(2, 1)
    callback = runlog(tmp_path, None)
    trainer = _trainer(torch.amp.GradScaler("cpu"))
    for _ in range(31):
        callback.on_train_batch_end(
            trainer, model, {"loss": torch.tensor(float("inf"))}, {}, 0
        )
    with pytest.raises(FloatingPointError, match="32 consecutive"):
        callback.on_train_batch_end(
            trainer, model, {"loss": torch.tensor(float("inf"))}, {}, 0
        )
