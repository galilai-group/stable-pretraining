"""Manager evaluation must consume the configured data and flush after completion."""

from unittest.mock import Mock

import lightning as pl
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from stable_pretraining.data.module import DataModule
from stable_pretraining.manager import Manager

pytestmark = pytest.mark.unit


class _EvaluationModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.seen = []

    def validation_step(self, batch, batch_idx):
        self.seen.append(("validate", batch["value"].tolist()))

    def test_step(self, batch, batch_idx):
        self.seen.append(("test", batch["value"].tolist()))

    def predict_step(self, batch, batch_idx):
        self.seen.append(("predict", batch["value"].tolist()))
        return batch["value"] * 2


@pytest.mark.parametrize(
    "stage,key", [("validate", "val"), ("test", "test"), ("predict", "predict")]
)
def test_evaluation_runs_all_batches_before_flushing(stage, key, tmp_path):
    model = _EvaluationModule()
    data = DataModule(
        **{key: DataLoader([{"value": i} for i in range(1, 5)], batch_size=2)}
    )
    manager = Manager.__new__(Manager)
    manager._register_module(model)
    manager._register_data(data)
    manager._trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    expected = [(stage, [1, 2]), (stage, [3, 4])]

    def flush():
        assert model.seen == expected

    manager._dump_wandb_data = Mock(side_effect=flush)
    getattr(manager, stage)()
    assert model.seen == expected
    assert manager._trainer.lightning_module is model
    manager._dump_wandb_data.assert_called_once_with()


@pytest.mark.parametrize("cuda_available", [False, True])
@pytest.mark.parametrize(
    "devices,expected",
    [
        (1, 1),
        ("3", 3),
        ([0, 2], 2),
        ((0, 1), 2),
        (OmegaConf.create([0, 2]), 2),
        (-1, 4),
        ("auto", "available"),
        ("-1", "available"),
        (None, "available"),
        (True, None),
        ("invalid", None),
        ({}, None),
    ],
)
def test_device_count_handles_lightning_config_forms(
    monkeypatch, cuda_available, devices, expected
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    manager = Manager.__new__(Manager)
    manager.trainer = {"accelerator": "gpu", "devices": devices}
    if expected == "available":
        expected = 4 if cuda_available else None
    assert manager._effective_device_count() == expected


@pytest.mark.parametrize("accelerator", ["cpu", "tpu", "mps"])
def test_non_cuda_accelerators_do_not_query_cuda(monkeypatch, accelerator):
    count = Mock(side_effect=AssertionError("CUDA must not be queried"))
    monkeypatch.setattr(torch.cuda, "device_count", count)
    manager = Manager.__new__(Manager)
    manager.trainer = {"accelerator": accelerator, "devices": "auto"}
    assert manager._effective_device_count() is None
    count.assert_not_called()


@pytest.mark.parametrize("component", ["module", "data", "trainer"])
def test_invalid_registration_preserves_existing_component(component):
    manager = Manager.__new__(Manager)
    original = object()
    setattr(manager, component, original)
    register = getattr(manager, f"_register_{component}")
    with pytest.raises(ValueError, match=f"`{component}` must be"):
        register(object())
    assert getattr(manager, component) is original


@pytest.mark.parametrize(
    "component,target",
    [
        ("module", __name__ + "._EvaluationModule"),
        ("data", "lightning.LightningDataModule"),
    ],
)
def test_component_identity_is_stable_until_explicitly_registered_again(
    component, target
):
    manager = Manager.__new__(Manager)
    register = getattr(manager, f"_register_{component}")
    register({"_target_": target})
    first = getattr(manager, f"instantiated_{component}")
    assert getattr(manager, f"instantiated_{component}") is first
    register({"_target_": target})
    second = getattr(manager, f"instantiated_{component}")
    assert second is not first
    assert getattr(manager, f"instantiated_{component}") is second
    register(first)
    assert getattr(manager, f"instantiated_{component}") is first
