"""Contracts for parameter-only EMA and optimizer-step scheduling."""

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_pretraining.backbone.utils import TeacherStudentWrapper
from stable_pretraining.callbacks.teacher_student import TeacherStudentCallback

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("coefficient", [0.0, 0.5, 1.0])
def test_parameter_only_ema_preserves_teacher_buffers(coefficient):
    wrapper = TeacherStudentWrapper(
        nn.BatchNorm1d(2),
        base_ema_coefficient=coefficient,
        final_ema_coefficient=coefficient,
        update_buffers=False,
    )
    assert wrapper.teacher is not wrapper.student
    with torch.no_grad():
        wrapper.student.weight.fill_(4)
        wrapper.teacher.weight.fill_(2)
        wrapper.student.running_mean.fill_(7)
        wrapper.teacher.running_mean.fill_(3)
        wrapper.student.num_batches_tracked.fill_(9)
        wrapper.teacher.num_batches_tracked.fill_(2)
    buffers = {k: v.clone() for k, v in wrapper.teacher.named_buffers()}
    wrapper.update_teacher()
    torch.testing.assert_close(
        wrapper.teacher.weight, torch.full((2,), 4 - 2 * coefficient)
    )
    for name, value in wrapper.teacher.named_buffers():
        torch.testing.assert_close(value, buffers[name])
    assert all(p.requires_grad for p in wrapper.student.parameters())
    assert not any(p.requires_grad for p in wrapper.teacher.parameters())


def test_teacher_batchnorm_learns_from_its_own_forward_only():
    wrapper = TeacherStudentWrapper(nn.BatchNorm1d(2), update_buffers=False)
    inputs = torch.tensor([[2.0, 4.0], [4.0, 8.0]], requires_grad=True)
    output = wrapper.forward_teacher(inputs)
    assert not output.requires_grad
    assert wrapper.teacher.training
    torch.testing.assert_close(wrapper.teacher.running_mean, torch.tensor([0.3, 0.6]))
    assert wrapper.teacher.num_batches_tracked.item() == 1
    assert wrapper.student.num_batches_tracked.item() == 0
    torch.testing.assert_close(wrapper.student.running_mean, torch.zeros(2))
    wrapper.forward_student(inputs).square().sum().backward()
    assert wrapper.student.weight.grad is not None
    assert wrapper.teacher.weight.grad is None
    wrapper.eval()
    state = copy.deepcopy(wrapper.teacher.state_dict())
    with torch.no_grad():
        wrapper.student.weight.fill_(9)
    wrapper.update_teacher()
    wrapper.forward_teacher(inputs)
    for name, value in wrapper.teacher.state_dict().items():
        torch.testing.assert_close(value, state[name])


def test_schedule_rejects_unknown_clock():
    with pytest.raises(ValueError, match="ema_schedule"):
        TeacherStudentWrapper(nn.Linear(1, 1), ema_schedule="batch")


def _step_wrapper():
    student = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(student.weight)
    return TeacherStudentWrapper(
        student, base_ema_coefficient=0.5, update_buffers=False, ema_schedule="step"
    )


def _trainer(model, global_step=0, total_steps=4):
    return SimpleNamespace(
        global_step=global_step,
        current_epoch=0,
        max_epochs=2,
        estimated_stepping_batches=total_steps,
        optimizers=[torch.optim.SGD(model.parameters(), lr=0.1)],
    )


@pytest.mark.parametrize("before_optimizer", [False, True])
@pytest.mark.parametrize("frequency", [1, 2])
def test_step_schedule_update_order_frequency_and_accumulation(
    before_optimizer, frequency
):
    wrapper = _step_wrapper()
    model = nn.Sequential(wrapper)
    trainer = _trainer(model)
    callback = TeacherStudentCallback(frequency, before_optimizer, verbose=False)
    callback.on_fit_start(trainer, model)
    callback.on_train_batch_end(trainer, model, {}, {}, 0)
    expected_weight, expected_coefficient = 0.0, 0.5
    for step in range(1, 7):
        with torch.no_grad():
            wrapper.student.weight.fill_(step)
        callback.on_before_optimizer_step(trainer, model, trainer.optimizers[0])
        callback.on_before_optimizer_step(trainer, model, trainer.optimizers[0])
        trainer.global_step = step
        callback.on_train_batch_end(trainer, model, {}, {}, step)
        callback.on_train_batch_end(trainer, model, {}, {}, step)
        if step % frequency == 0:
            expected_weight = (
                expected_coefficient * expected_weight
                + (1 - expected_coefficient) * step
            )
            expected_coefficient = 1 - 0.25 * (1 + math.cos(min(step, 4) * math.pi / 4))
        assert wrapper.teacher.weight.item() == pytest.approx(expected_weight)
        assert wrapper.ema_coefficient.item() == pytest.approx(expected_coefficient)


def test_step_and_default_epoch_schedules_can_coexist():
    step_wrapper = _step_wrapper()
    epoch_wrapper = TeacherStudentWrapper(
        nn.Linear(1, 1, bias=False), base_ema_coefficient=0.5
    )
    with torch.no_grad():
        epoch_wrapper.teacher.weight.zero_()
        epoch_wrapper.student.weight.fill_(2)
        step_wrapper.student.weight.fill_(2)
    model = nn.Sequential(step_wrapper, epoch_wrapper)
    trainer = _trainer(model)
    trainer.current_epoch = 1
    callback = TeacherStudentCallback(verbose=False)
    callback.on_fit_start(trainer, model)
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, model, {}, {}, 0)
    assert epoch_wrapper.ema_coefficient.item() == pytest.approx(0.75)
    assert epoch_wrapper.teacher.weight.item() == pytest.approx(0.5)
    assert step_wrapper.teacher.weight.item() == pytest.approx(1.0)
    assert step_wrapper.ema_coefficient.item() == pytest.approx(0.5732233)


def test_step_schedule_ignores_callback_optimizer():
    wrapper = _step_wrapper()
    model = nn.Sequential(wrapper)
    trainer = _trainer(model)
    main_optimizer = trainer.optimizers[0]
    extra_optimizer = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.1)
    trainer.optimizers.insert(0, extra_optimizer)
    model._pick_global_step_ticker = lambda optimizers: 1
    callback = TeacherStudentCallback(update_after_backward=True, verbose=False)
    callback.on_fit_start(trainer, model)
    with torch.no_grad():
        wrapper.student.weight.fill_(2)
    callback.on_before_optimizer_step(trainer, model, extra_optimizer)
    assert wrapper.teacher.weight.item() == 0
    assert wrapper.ema_coefficient.item() == 0.5
    callback.on_before_optimizer_step(trainer, model, main_optimizer)
    assert wrapper.teacher.weight.item() == 1


@pytest.mark.parametrize("total_steps", [0, -1, float("inf"), float("nan")])
def test_step_schedule_requires_finite_positive_duration(total_steps):
    wrapper = _step_wrapper()
    model = nn.Sequential(wrapper)
    trainer = _trainer(model, total_steps=total_steps)
    callback = TeacherStudentCallback(verbose=False)
    callback.on_fit_start(trainer, model)
    with torch.no_grad():
        wrapper.student.weight.fill_(2)
    trainer.global_step = 1
    with pytest.raises(ValueError, match="finite, positive"):
        callback.on_train_batch_end(trainer, model, {}, {}, 0)
    assert wrapper.teacher.weight.item() == 0
    assert wrapper.ema_coefficient.item() == 0.5


def test_step_schedule_checkpoint_resume_matches_uninterrupted_updates():
    wrapper = _step_wrapper()
    model = nn.Sequential(wrapper)
    trainer = _trainer(model)
    callback = TeacherStudentCallback(verbose=False)
    callback.on_fit_start(trainer, model)
    checkpoint = None
    for step in range(1, 5):
        with torch.no_grad():
            wrapper.student.weight.fill_(step)
        trainer.global_step = step
        callback.on_train_batch_end(trainer, model, {}, {}, step)
        if step == 2:
            checkpoint = copy.deepcopy(model.state_dict())

    resumed = nn.Sequential(_step_wrapper())
    resumed.load_state_dict(checkpoint, strict=True)
    resumed_trainer = _trainer(resumed, global_step=2)
    resumed_callback = TeacherStudentCallback(verbose=False)
    resumed_callback.on_fit_start(resumed_trainer, resumed)
    resumed_callback.on_train_batch_end(resumed_trainer, resumed, {}, {}, 2)
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, checkpoint[name])
    for step in range(3, 5):
        with torch.no_grad():
            resumed[0].student.weight.fill_(step)
        resumed_trainer.global_step = step
        resumed_callback.on_train_batch_end(resumed_trainer, resumed, {}, {}, step)
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)


def test_new_options_preserve_checkpoint_keys():
    legacy = TeacherStudentWrapper(nn.Linear(1, 1, bias=False))
    wrapper = _step_wrapper()
    assert wrapper.state_dict().keys() == legacy.state_dict().keys()
    wrapper.load_state_dict(legacy.state_dict(), strict=True)
