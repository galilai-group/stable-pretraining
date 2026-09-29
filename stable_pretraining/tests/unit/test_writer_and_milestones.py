"""Callback decisions must persist the right outputs and stop actual training."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import lightning as pl
import pytest
import torch
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader, TensorDataset

from stable_pretraining.callbacks.writer import OnlineWriter
from stable_pretraining.callbacks.earlystop import EpochMilestones, to_scalar
from stable_pretraining.callbacks.utils import EarlyStopping, format_metrics_as_dict

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("phase", ["train", "validation", "test", "predict"])
@pytest.mark.parametrize("gather,rank", [(False, 1), (True, 0), (True, 1)])
def test_writer_roundtrip_and_rank_ownership(tmp_path, phase, gather, rank):
    writer = OnlineWriter("embedding", tmp_path, during=phase, all_gather=gather)
    module = SimpleNamespace(
        current_epoch=2,
        local_rank=rank,
        trainer=SimpleNamespace(max_epochs=3),
        all_gather=lambda x: torch.stack([x, x]),
    )
    value = torch.randn(2, 3)
    getattr(writer, f"on_{phase}_batch_end")(
        module.trainer, module, {"embedding": value}, {}, 4
    )
    files = list(tmp_path.glob("*.pt"))
    if gather and rank != 0:
        assert not files
    else:
        assert len(files) == 1
        assert files[0].name.startswith(f"{phase}_epoch=2_batch=4")
        loaded = torch.load(files[0], weights_only=True)
        torch.testing.assert_close(
            loaded["embedding"], torch.stack([value, value]) if gather else value
        )


def test_writer_schedule_sanity_and_missing_key(tmp_path):
    module = SimpleNamespace(
        current_epoch=1, local_rank=0, trainer=SimpleNamespace(max_epochs=4)
    )
    writer = OnlineWriter(
        "x",
        tmp_path,
        during=["train"],
        every_k_epochs=2,
        save_last_epoch=True,
        all_gather=False,
    )
    writer.write_at_phase(module, "train", {"x": torch.ones(1)}, 0)
    assert not list(tmp_path.glob("*.pt"))
    module.current_epoch = 3
    writer.on_sanity_check_start(None, module)
    writer.write_at_phase(module, "train", {"x": torch.ones(1)}, 0)
    assert not list(tmp_path.glob("*.pt"))
    writer.on_sanity_check_end(None, module)
    with pytest.raises(ValueError, match="not present"):
        writer.write_at_phase(module, "train", {}, 0)
    writer.write_at_phase(module, "train", {"x": torch.ones(1)}, 0)
    assert len(list(tmp_path.glob("*.pt"))) == 1
    writer.every_k_epochs = 0
    module.current_epoch = 2
    assert not writer.is_writing_epoch(module)


def test_writer_relative_path_resolves_to_configured_cache(monkeypatch, tmp_path):
    from stable_pretraining._config import get_config

    monkeypatch.chdir(tmp_path)
    get_config().cache_dir = str(tmp_path / "cache")
    writer = OnlineWriter("x", "outputs", during="train")
    writer.setup(SimpleNamespace(default_root_dir=str(Path.cwd())), None)
    assert writer.path == tmp_path / "cache" / "outputs"
    assert writer.path.is_dir()


@pytest.mark.parametrize(
    "mode,value,expected",
    [("max", 0.4, True), ("max", 0.7, False), ("min", 0.7, True), ("min", 0.4, False)],
)
def test_milestone_thresholds_and_verbose_metrics(monkeypatch, mode, value, expected):
    import stable_pretraining.callbacks.earlystop as earlystop

    log = Mock()
    monkeypatch.setattr(earlystop, "_spt_log", log)
    trainer = SimpleNamespace(
        current_epoch=2,
        callback_metrics={"val/score": torch.tensor(value)},
        sanity_checking=False,
        should_stop=False,
    )
    callback = EpochMilestones({2: 0.5}, contains="score", direction=mode, verbose=True)
    callback.on_validation_epoch_end(trainer, None)
    assert trainer.should_stop is expected
    assert log.call_count == 2
    metric = MeanMetric()
    metric.update(torch.tensor(value))
    assert to_scalar(metric) == pytest.approx(value)


@pytest.mark.parametrize("strict", [False, True])
def test_milestone_sanity_missing_matches(strict):
    trainer = SimpleNamespace(
        current_epoch=0, callback_metrics={}, sanity_checking=True
    )
    callback = EpochMilestones({1: 0.5}, contains="score", strict=strict)
    if strict:
        with pytest.raises(RuntimeError, match="No metrics found"):
            callback.on_validation_epoch_end(trainer, None)
    else:
        callback.on_validation_epoch_end(trainer, None)


class TinyTraining(pl.LightningModule):
    """One-parameter model with a deliberately unmet score milestone."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        self.log("score", self.weight.detach() * 0, on_step=False, on_epoch=True)
        return self.weight.square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_training_milestone_stops_real_lightning_loop(tmp_path):
    callback = EpochMilestones({0: 0.5}, monitor="score", after_validation=False)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        callbacks=[callback],
    )
    trainer.fit(
        TinyTraining(), DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)
    )
    assert trainer.should_stop
    assert trainer.global_step == 2


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("mode", ["min", "max"])
def test_simple_stopper_checks_only_configured_milestones(named, mode):
    stop = EarlyStopping(
        mode=mode, milestones={2: 0.5}, metric_name="score" if named else None
    )
    metric = {"score": 0.4} if named else 0.4
    assert not stop.should_stop(metric, 1)
    assert stop.should_stop(metric, 2) is (mode == "max")


@pytest.mark.parametrize(
    "style", ["single", "list", "tuple", "dict", "split", "split_single", "none"]
)
def test_metric_normalization_retains_metric_instances_and_stage(style):
    train, val = MeanMetric(), MeanMetric()
    inputs = {
        "single": val,
        "list": [val],
        "tuple": (val,),
        "dict": {"mean": val},
        "split": {"train": [train], "val": [val]},
        "split_single": {"train": train, "val": val},
        "none": None,
    }
    result = format_metrics_as_dict(inputs[style])
    assert len(result["_train"]) == (1 if style.startswith("split") else 0)
    assert len(result["_val"]) == (0 if style == "none" else 1)
    if style != "none":
        assert next(iter(result["_val"].values())) is val
    if style.startswith("split"):
        assert next(iter(result["_train"].values())) is train


@pytest.mark.parametrize(
    "invalid", [42, [42], {"train": [42], "val": []}, {"train": [], "val": [42]}]
)
def test_metric_normalization_rejects_invalid_inputs(invalid):
    with pytest.raises(ValueError, match="metric"):
        format_metrics_as_dict(invalid)
