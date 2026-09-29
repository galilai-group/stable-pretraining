"""Optimizer construction rejects invalid inputs without changing parameters."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from stable_pretraining import Module
from stable_pretraining.module import _gpu_transform_from_current_dataset
from stable_pretraining.optim import lr_scheduler, utils

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("config", ["Adam", 3, [1]])
def test_module_rejects_invalid_optimizer_container(config):
    model = Module(backbone=nn.Linear(2, 2), optim=config)
    with pytest.raises(ValueError, match="partial function or a dict"):
        model.configure_optimizers()


def test_invalid_multi_optimizer_values_and_unmatched_groups():
    model = Module(backbone=nn.Linear(2, 2), optim={"bad": "Adam"})
    with pytest.raises(ValueError, match="all config values must be dicts"):
        model.configure_optimizers()
    model.optim = {"unused": {"modules": "^missing$", "optimizer": "Adam"}}
    model._trainer = SimpleNamespace(estimated_stepping_batches=5)
    optimizers, schedulers = model.configure_optimizers()
    assert optimizers == schedulers == []


@pytest.mark.parametrize("kind", ["default", "partial"])
def test_module_default_and_partial_optimizer_update_real_parameters(kind):
    model = Module(backbone=nn.Linear(2, 2), optim=partial(torch.optim.SGD, lr=0.1))
    if kind == "default":
        del model.optim
    model._trainer = SimpleNamespace(estimated_stepping_batches=5)
    opts, scheds = model.configure_optimizers()
    before = model.backbone.weight.detach().clone()
    model.backbone(torch.ones(2, 2)).sum().backward()
    opts[0].step()
    assert not torch.equal(model.backbone.weight, before)
    assert scheds[0]["interval"] == "step"


@pytest.mark.parametrize(
    "configuration,monitor",
    [
        ("ReduceLROnPlateau", "val_loss"),
        ({"type": "ReduceLROnPlateau", "monitor": "accuracy"}, "accuracy"),
    ],
)
def test_plateau_scheduler_monitor_is_preserved(configuration, monitor):
    model = Module()
    opt = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    result = model._build_scheduler_config(sched, {"scheduler": configuration})
    assert result["monitor"] == monitor
    assert model._get_scheduler_name(None, None) == "Unknown"
    assert model._get_scheduler_name(None, sched) == "ReduceLROnPlateau"
    assert model._pick_global_step_ticker([]) == 0


@pytest.mark.parametrize(
    "configuration",
    [
        OmegaConf.create({"_target_": "torch.optim.SGD", "lr": 0.1}),
        lambda params: torch.optim.SGD(params, lr=0.1),
        {"type": torch.optim.SGD, "lr": 0.1},
    ],
)
def test_optimizer_factories_preserve_parameter_identity(configuration):
    weight = nn.Parameter(torch.ones(2))
    optimizer = utils.create_optimizer([weight], configuration)
    assert optimizer.param_groups[0]["params"][0] is weight
    weight.grad = torch.ones_like(weight)
    optimizer.step()
    torch.testing.assert_close(weight, torch.full_like(weight, 0.9))


def test_optimizer_reports_missing_required_arguments_and_empty_groups():
    weight = nn.Parameter(torch.ones(2))
    with pytest.raises(TypeError, match="Required parameters"):
        utils.create_optimizer([weight], {"type": "SGD", "not_an_argument": 2})
    with pytest.raises(ValueError, match="No parameters"):
        utils.create_optimizer(
            [], {"type": "SGD", "exclude_bias_norm": True}, named_params=[]
        )
    with pytest.raises(ValueError, match="not found"):
        utils.create_optimizer([weight], "NoSuchOptimizer")


def test_scheduler_defaults_and_failed_default_factory(monkeypatch):
    opt = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    assert lr_scheduler._build_default_params("NoSuchScheduler", None, opt) == {}
    monkeypatch.setattr(
        lr_scheduler,
        "_build_default_params",
        Mock(side_effect=RuntimeError("trainer unavailable")),
    )
    sched = lr_scheduler.create_scheduler(opt, "ConstantLR")
    assert isinstance(sched, torch.optim.lr_scheduler.ConstantLR)
    with pytest.raises(ValueError, match="not found"):
        lr_scheduler.create_scheduler(opt, "NoSuchScheduler")


def test_dataset_transform_resolution_handles_multi_loader_wrappers_and_errors():
    transform = nn.Identity()
    inner = SimpleNamespace(gpu_transform=transform)
    wrapped = SimpleNamespace(datasets=[inner])
    trainer = SimpleNamespace(val_dataloaders=[SimpleNamespace(dataset=wrapped)])
    assert _gpu_transform_from_current_dataset(trainer, "val", 0) is transform
    assert _gpu_transform_from_current_dataset(trainer, "val", 4) is None
    trainer.val_dataloaders[0].dataset.datasets = []
    assert _gpu_transform_from_current_dataset(trainer, "val", 0) is None
    assert Module._current_stage(None) is None
    trainer = SimpleNamespace(
        training=False,
        validating=False,
        sanity_checking=False,
        testing=False,
        predicting=True,
    )
    assert Module._current_stage(trainer) == "predict"
    trainer.predicting = False
    assert Module._current_stage(trainer) is None


def test_module_requires_keyword_arguments_and_dict_training_batches():
    with pytest.raises(ValueError, match="positional"):
        Module("unexpected")
    with pytest.raises(ValueError, match="dict"):
        Module().training_step(torch.ones(2), 0)
