"""Evaluation and EMA callbacks preserve the intended optimization trajectory."""

from copy import deepcopy
import math

import pytest
import torch

import stable_pretraining as spt

from ._interaction_helpers import (
    _Trace,
    _assert_tree_close,
    _model,
    _probe,
    _run,
    _samples,
)

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]


@pytest.mark.parametrize("frequency", [1, 2])
@pytest.mark.parametrize("evaluation", ["probe", "multiple_probes", "rankme", "both"])
def test_evaluation_callbacks_preserve_backbone_updates(frequency, evaluation):
    baseline = _model(frequency=frequency)
    expected = _Trace()
    _run(baseline, [expected], max_steps=6)

    module = _model(frequency=frequency)
    callbacks = []
    initial_probes = {}
    if evaluation in ("probe", "multiple_probes", "both"):
        count = 3 if evaluation == "multiple_probes" else 1
        for i in range(count):
            probe = _probe(module, name=f"probe_{i}", frequency=i + 1)
            initial_probes[probe.name] = deepcopy(probe._probe_config.state_dict())
            callbacks.append(probe)
    if evaluation in ("rankme", "both"):
        callbacks.append(spt.RankMe("rank", "embedding", 8, 3, verbose=False))
    actual = _Trace()
    trainer = _run(module, [*callbacks, actual], max_steps=6)

    assert trainer.global_step == 6
    assert len(actual.batches) == len(expected.batches) == 6 * frequency
    for left, right in zip(actual.batches, expected.batches):
        for name, value in right["model"].items():
            _assert_tree_close(left["model"][name], value)
        _assert_tree_close(left["optimizers"][0], right["optimizers"][0])
        _assert_tree_close(left["schedulers"][0], right["schedulers"][0])
        assert left["global_step"] == right["global_step"]
    for name, initial in initial_probes.items():
        trained = module.callbacks_modules[name].state_dict()
        assert any(
            not torch.equal(trained[key], value) for key, value in initial.items()
        )
        assert f"eval/{name}_accuracy" in trainer.callback_metrics
    if evaluation in ("rankme", "both"):
        assert 1 <= trainer.callback_metrics["rank"] <= 3.01


def _reference_updates(frequency, ema_frequency, before_step):
    wrapper = _model(teacher=True).backbone
    student = deepcopy(wrapper.student)
    teacher = deepcopy(wrapper.teacher)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.03, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    rows = _samples()
    records = []
    step = 0
    coefficient = 0.5
    for epoch in range(2):
        for batch_index in range(6):
            batch = rows[2 * batch_index : 2 * batch_index + 2]
            images = torch.stack([row["image"] for row in batch])
            targets = torch.stack([row["target"] for row in batch])
            prediction = student(images)
            with torch.no_grad():
                target_prediction = teacher(images)
            loss = (prediction - targets).square().mean()
            loss = loss + 0.1 * (prediction - target_prediction).square().mean()
            (loss / frequency).backward()
            if (batch_index + 1) % frequency == 0:
                step += 1
                update_teacher = step % ema_frequency == 0
                if update_teacher:
                    coefficient = 0.9 - 0.5 * (0.9 - 0.5) * (
                        1 + math.cos(epoch / 2 * math.pi)
                    )
                if update_teacher and before_step:
                    with torch.no_grad():
                        for target, source in zip(
                            teacher.parameters(), student.parameters()
                        ):
                            target.mul_(coefficient).add_(source, alpha=1 - coefficient)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if update_teacher and not before_step:
                    with torch.no_grad():
                        for target, source in zip(
                            teacher.parameters(), student.parameters()
                        ):
                            target.mul_(coefficient).add_(source, alpha=1 - coefficient)
            records.append(
                deepcopy(
                    {
                        "student": student.state_dict(),
                        "teacher": teacher.state_dict(),
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "coefficient": coefficient,
                    }
                )
            )
    return records


@pytest.mark.parametrize("frequency", [1, 2, 3])
@pytest.mark.parametrize("ema_frequency", [1, 2])
@pytest.mark.parametrize("before_step", [False, True])
def test_ema_and_scheduler_follow_optimizer_updates(
    frequency, ema_frequency, before_step
):
    expected = _reference_updates(frequency, ema_frequency, before_step)
    module = _model(frequency=frequency, teacher=True)
    ema = spt.callbacks.TeacherStudentCallback(
        update_frequency=ema_frequency, update_after_backward=before_step, verbose=False
    )
    trace = _Trace()
    _run(module, [_probe(module, frequency=1), ema, trace])

    assert len(trace.batches) == len(expected) == 12
    for actual, reference in zip(trace.batches, expected):
        for component in ("student", "teacher"):
            for name, value in reference[component].items():
                _assert_tree_close(
                    actual["model"][f"backbone.{component}.{name}"], value
                )
        assert actual["global_step"] == reference["step"]
        assert actual["optimizers"][0]["param_groups"][0]["lr"] == pytest.approx(
            reference["lr"]
        )
        assert actual["schedulers"][0]["last_epoch"] == reference["step"]
        assert actual["model"]["backbone.ema_coefficient"].item() == pytest.approx(
            reference["coefficient"]
        )
    assert all(
        parameter.grad is None for parameter in module.backbone.teacher.parameters()
    )
