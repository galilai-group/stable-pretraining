"""Learning-rate boundaries, configuration formats, and resume equivalence."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from stable_pretraining.optim import lr_scheduler as schedulers

pytestmark = pytest.mark.unit


def _optimizer():
    return torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)


def _step(optimizer, scheduler, count):
    values = [optimizer.param_groups[0]["lr"]]
    for _ in range(count):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    return values


@pytest.mark.parametrize(
    "config,step_size",
    [
        ("StepLR", 30),
        ({"type": "StepLR", "step_size": 2, "gamma": 0.1}, 2),
        (OmegaConf.create({"type": "StepLR", "step_size": 2}), 2),
        (
            OmegaConf.create(
                {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 2}
            ),
            2,
        ),
        (partial(torch.optim.lr_scheduler.StepLR, step_size=2), 2),
        (lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=2), 2),
        (
            lambda opt, module: torch.optim.lr_scheduler.StepLR(
                opt, step_size=module.period
            ),
            2,
        ),
        (torch.optim.lr_scheduler.StepLR, 30),
    ],
)
def test_factory_configs_produce_expected_schedule(config, step_size):
    optimizer = _optimizer()
    scheduler = schedulers.create_scheduler(
        optimizer, config, SimpleNamespace(period=2)
    )
    values = _step(optimizer, scheduler, step_size)
    assert values[:-1] == pytest.approx([1.0] * step_size)
    assert values[-1] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "config,exception,match",
    [
        ("UnknownScheduler", ValueError, "not found"),
        ({"type": 123}, ValueError, "must be a string"),
        (lambda a, b, c: None, NotImplementedError, "2 args"),
        (123, TypeError, "scheduler_config"),
    ],
)
def test_factory_rejects_invalid_configuration(config, exception, match):
    with pytest.raises(exception, match=match):
        schedulers.create_scheduler(_optimizer(), config)


def test_factory_uses_trainer_step_count_for_cosine_default():
    module = SimpleNamespace(trainer=SimpleNamespace(estimated_stepping_batches=8))
    optimizer = _optimizer()
    scheduler = schedulers.create_scheduler(optimizer, "CosineAnnealingLR", module)
    assert _step(optimizer, scheduler, 8)[-1] == pytest.approx(0.0)


@pytest.mark.parametrize("peak", [2, 0.25])
def test_linear_warmup_reaches_and_keeps_base_lr(peak):
    optimizer = _optimizer()
    scheduler = schedulers.LinearWarmup(
        optimizer, total_steps=8, start_factor=0.1, peak_step=peak
    )
    assert _step(optimizer, scheduler, 4) == pytest.approx([0.1, 0.55, 1, 1, 1])


@pytest.mark.parametrize(
    "factory",
    [
        partial(
            schedulers.LinearWarmupCosineAnnealing,
            total_steps=8,
            start_factor=0.1,
            peak_step=0.25,
            end_lr=0.05,
        ),
        partial(
            schedulers.LinearWarmupCosineAnnealingLR,
            warmup_steps=2,
            max_steps=8,
            warmup_start_lr=0.1,
            eta_min=0.05,
        ),
    ],
)
def test_cosine_warmup_boundaries_and_resume(factory):
    optimizer = _optimizer()
    scheduler = factory(optimizer)
    initial = _step(optimizer, scheduler, 3)
    assert initial[:3] == pytest.approx([0.1, 0.55, 1.0])
    resumed_optimizer = _optimizer()
    resumed_scheduler = factory(resumed_optimizer)
    resumed_scheduler.load_state_dict(scheduler.state_dict())
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    tail = _step(optimizer, scheduler, 5)
    assert _step(resumed_optimizer, resumed_scheduler, 5) == pytest.approx(tail)
    assert tail[-1] == pytest.approx(0.05)
    assert all(a >= b for a, b in zip(tail, tail[1:]))


def test_cyclic_warmup_reaches_zero_after_final_cycle():
    optimizer = _optimizer()
    scheduler = schedulers.LinearWarmupCyclicAnnealing(
        optimizer, total_steps=22, peak_step=0.1
    )
    values = _step(optimizer, scheduler, 22)
    assert values[0] == pytest.approx(0.01)
    assert values[2] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.0)
    assert all(value >= 0 for value in values)
    assert any(b > a for a, b in zip(values[2:], values[3:]))


def test_three_step_annealing_drops_at_each_milestone():
    optimizer = _optimizer()
    scheduler = schedulers.LinearWarmupThreeStepsAnnealing(
        optimizer, total_steps=12, peak_step=2, gamma=0.5
    )
    values = _step(optimizer, scheduler, 12)
    assert values[2:6] == pytest.approx([1] * 4)
    assert values[6:8] == pytest.approx([0.5] * 2)
    assert values[8:10] == pytest.approx([0.25] * 2)
    assert values[10:] == pytest.approx([0.125] * 3)
