"""Failed writes must preserve recoverable checkpoints and run metadata."""

import csv
import json

import pytest
import torch

from stable_pretraining.registry import _sidecar
from stable_pretraining.registry.logger import RegistryLogger
from stable_pretraining.utils import atomic_checkpoint

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("stage", ["serialize", "fsync", "replace"])
@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("existing", [True, False])
def test_checkpoint_failure_keeps_previous_file(
    tmp_path, monkeypatch, stage, error, existing
):
    path = tmp_path / "last.ckpt"
    if existing:
        atomic_checkpoint.atomic_torch_save(
            {"step": 12, "weights": torch.arange(3)}, path
        )
    before = path.read_bytes() if existing else None

    def fail(*args, **kwargs):
        raise error("injected write failure")

    target, name = (
        (atomic_checkpoint.torch, "save")
        if stage == "serialize"
        else (atomic_checkpoint.os, stage)
    )
    monkeypatch.setattr(target, name, fail)
    with pytest.raises(error, match="injected write failure"):
        atomic_checkpoint.atomic_torch_save({"step": 13}, path)
    assert (path.read_bytes() if path.exists() else None) == before
    assert not list(tmp_path.glob(".*.tmp"))
    if existing:
        recovered = torch.load(path, weights_only=False)
        assert recovered["step"] == 12
        torch.testing.assert_close(recovered["weights"], torch.arange(3))


@pytest.mark.parametrize("stage", ["fsync", "replace"])
@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("existing", [True, False])
def test_sidecar_failure_keeps_previous_metadata(
    tmp_path, monkeypatch, stage, error, existing
):
    path = tmp_path / "sidecar.json"
    previous = {"run_id": "r1", "summary": {"loss": 0.4}}
    if existing:
        _sidecar.atomic_json_write(path, previous)

    def fail(*args, **kwargs):
        raise error("injected write failure")

    monkeypatch.setattr(_sidecar.os, stage, fail)
    with pytest.raises(error, match="injected write failure"):
        _sidecar.atomic_json_write(path, {"run_id": "r1", "summary": {"loss": 0.3}})
    if existing:
        assert json.loads(path.read_text()) == previous
    else:
        assert not path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "first_key,second_key",
    [("z_loss", "a_acc"), ("train/loss", "val/acc"), ("loss", "epoch")],
)
def test_metrics_stay_aligned_across_multiple_resumes(tmp_path, first_key, second_key):
    expected = [
        {first_key: 1.25, second_key: 0.5},
        {second_key: 0.75, "new_metric": 3.5},
        {first_key: 0.25, "new_metric": 4.5},
    ]
    for step, metrics in enumerate(expected):
        logger = RegistryLogger(run_dir=tmp_path, run_id="r1")
        logger.log_metrics(metrics, step=step)
        logger.save()
    with (tmp_path / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(expected)
    for step, (row, metrics) in enumerate(zip(rows, expected)):
        assert int(row["step"]) == step
        for key in (first_key, second_key, "new_metric"):
            if key in metrics:
                assert float(row[key]) == metrics[key]
            else:
                assert row[key] == ""
