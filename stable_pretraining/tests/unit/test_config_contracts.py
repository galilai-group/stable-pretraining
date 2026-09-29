"""Configuration round trips must preserve values and instantiate real objects."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from stable_pretraining import config
from stable_pretraining.utils import config as helpers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("as_omega", [False, True])
@pytest.mark.parametrize("separator", [".", "/"])
def test_flatten_config_handles_sequences_and_preserves_values(as_omega, separator):
    source = {"model": {"widths": [4, 8], "optional": None}, "enabled": True}
    cfg = OmegaConf.create(source) if as_omega else source
    assert config.collapse_nested_dict(cfg, separator) == {
        separator.join(["model", "widths", "0"]): 4,
        separator.join(["model", "widths", "1"]): 8,
        separator.join(["model", "optional"]): None,
        "enabled": True,
    }
    assert (OmegaConf.to_container(cfg) if as_omega else cfg) == source


@pytest.mark.parametrize("as_omega", [False, True])
def test_recursive_instantiation_resolves_forward_and_retains_plain_values(as_omega):
    source = {
        "module": {"_target_": "builtins.dict", "forward": "torch.neg"},
        "data": {
            "_target_": "builtins.dict",
            "nested": {"_target_": "torch.nn.Identity"},
        },
        "loss": {"_target_": "torch.nn.MSELoss"},
        "extra": {"_target_": "torch.nn.ReLU"},
        "seed": 7,
        "callbacks": [],
    }
    result = config.recursive_instantiate(
        OmegaConf.create(source) if as_omega else source
    )
    assert result["module"]["forward"] is torch.neg
    assert result["data"]["nested"]["_target_"] == "torch.nn.Identity"
    assert isinstance(result["loss"], nn.MSELoss)
    assert isinstance(result["extra"], nn.ReLU)
    assert result["seed"] == 7 and result["callbacks"] == []


def test_instantiation_resolves_root_interpolations():
    source = OmegaConf.create(
        {
            "width": 3,
            "module": {
                "_target_": "builtins.dict",
                "forward": "torch.neg",
                "width": "${width}",
            },
        }
    )
    assert config.recursive_instantiate(source)["module"]["width"] == 3


def test_failed_instantiation_warns_and_preserves_configuration(monkeypatch):
    warn = Mock()
    monkeypatch.setattr(config, "rank_zero_warn", warn)
    source = {
        "module": {"_target_": "not_a_module.Broken"},
        "extra": {"_target_": "not_a_module.Broken"},
    }
    assert config.recursive_instantiate(source) == source
    assert warn.call_count == 2
    assert config.recursive_instantiate(None) == {}


def test_config_entrypoint_sets_requested_precision_and_returns_components():
    previous = torch.get_float32_matmul_precision()
    try:
        result = config.instantiate_from_config(
            {"matmul_precision": "high", "loss": {"_target_": "torch.nn.L1Loss"}}
        )
        assert torch.get_float32_matmul_precision() == "high"
        assert isinstance(result["loss"], nn.L1Loss)
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("world_size,rank", [(0, 0), (1, 0), (2, 0), (2, 1)])
def test_checkpoint_hparams_match_on_all_distributed_paths(
    tmp_path, monkeypatch, world_size, rank
):
    path = tmp_path / "run.ckpt"
    expected = {"lr": 0.1, "width": 8}
    torch.save(
        {"hyper_parameters": expected, "state_dict": {"weight": torch.ones(3)}}, path
    )
    monkeypatch.setattr(helpers, "is_dist", lambda: world_size > 0)
    monkeypatch.setattr(helpers.dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(helpers.dist, "get_rank", lambda: rank)
    barrier = Mock()
    monkeypatch.setattr(helpers.dist, "barrier", barrier)

    def broadcast(objects, src):
        assert src == 0
        if rank == 0:
            assert objects == [expected]
        else:
            assert objects == [None]
            objects[0] = expected

    monkeypatch.setattr(helpers.dist, "broadcast_object_list", broadcast)
    assert helpers.load_hparams_from_ckpt(str(path)) == expected
    assert barrier.call_count == (1 if world_size > 1 else 0)


def test_recursive_attribute_access_and_replacement():
    obj = SimpleNamespace(data={"encoder": SimpleNamespace(width=4)})
    assert helpers.rgetattr(obj, "data.encoder.width") == 4
    helpers.rsetattr(obj, "data.encoder.width", 8)
    assert obj.data["encoder"].width == 8
    helpers.rsetattr(obj, "data.extra", 3)
    assert obj.data["extra"] == 3


def test_find_and_replace_nested_modules_preserves_unrelated_weights():
    model = nn.Sequential(nn.Linear(3, 3), nn.Sequential(nn.ReLU(), nn.Linear(3, 2)))
    names, layers = helpers.find_module(model, nn.Linear)
    assert names == ["0", "1.1"]
    assert layers == [model[0], model[1][1]]
    weights = [layer.weight.clone() for layer in layers]
    assert (
        helpers.replace_module(
            model,
            lambda name, layer: nn.Sigmoid() if isinstance(layer, nn.ReLU) else layer,
        )
        is model
    )
    assert isinstance(model[1][0], nn.Sigmoid)
    for layer, before in zip(layers, weights):
        torch.testing.assert_close(layer.weight, before)
    with pytest.raises(ValueError, match="Module expected"):
        helpers.replace_module({}, lambda *args: None)


def test_config_execution_locally_and_through_submitit(monkeypatch):
    manager = Mock(return_value="local")
    assert helpers.execute_from_config(manager, {}) == "local"
    executor = Mock()
    executor.submit.return_value.result.return_value = "remote-result"
    monkeypatch.setattr(helpers.hydra.utils, "instantiate", lambda *a, **kw: executor)
    monkeypatch.setattr(
        helpers.HydraConfig,
        "get",
        lambda: OmegaConf.create(
            {"job": {"name": "test"}, "sweep": {"dir": "sweep"}, "run": {"dir": "run"}}
        ),
    )
    cfg = OmegaConf.create(
        {"submitit": {"executor": {}, "update_parameters": {"timeout_min": 5}}}
    )
    assert helpers.execute_from_config(manager, cfg) == "remote-result"
    executor.update_parameters.assert_called_once_with(timeout_min=5)
    executor.submit.assert_called_once_with(manager)
    assert cfg.hydra.sweep.dir == "sweep"


def test_low_resolution_resnet_adaptation_preserves_later_layers():
    model = nn.Module()
    model.conv1 = nn.Conv2d(3, 64, 7, stride=2)
    model.maxpool = nn.MaxPool2d(3)
    model.layer1 = nn.Linear(2, 2)
    weight = model.layer1.weight
    assert helpers.adapt_resnet_for_lowres(model) is model
    assert model.conv1(torch.ones(1, 3, 8, 8)).shape == (1, 64, 8, 8)
    assert isinstance(model.maxpool, nn.Identity)
    assert model.layer1.weight is weight


def test_config_entrypoint_creates_manager_with_checkpoint_loading_policy(monkeypatch):
    import lightning as pl
    from stable_pretraining.manager import Manager
    from stable_pretraining.tests.utils import BoringDataModule, BoringModule

    components = {
        "trainer": pl.Trainer(logger=False, enable_checkpointing=False),
        "module": BoringModule(),
        "data": BoringDataModule(),
        "seed": 42,
        "weights_only": False,
    }
    monkeypatch.setattr(config, "recursive_instantiate", lambda _: components)
    manager = config.instantiate_from_config({})
    assert isinstance(manager, Manager)
    assert manager.instantiated_module is components["module"]
    assert manager.instantiated_data is components["data"]
    assert manager.weights_only is False
