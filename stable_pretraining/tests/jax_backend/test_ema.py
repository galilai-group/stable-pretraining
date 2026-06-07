"""Tests for the EMA teacher-student path (TeacherStudentWrapper, BYOL)."""

import jax
import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining.jax as spj  # noqa: E402
from stable_pretraining.jax.backbone.teacher_student import EMAParam  # noqa: E402


def _two_view(n=4, b=16, d=32, seed=0):
    rng = np.random.RandomState(seed)
    return [
        {
            "views": [
                {"image": rng.randn(b, d).astype("float32")},
                {"image": rng.randn(b, d).astype("float32")},
            ]
        }
        for _ in range(n)
    ]


def _byol(d=32, seed=0, base_ema=0.99, final_ema=1.0):
    rngs = nnx.Rngs(seed)
    return spj.BYOL(
        backbone=spj.backbone.MLP(d, [64, d], rngs=rngs),
        embed_dim=d,
        rngs=rngs,
        projector_dims=(64, 32),
        predictor_hidden=64,
        base_ema=base_ema,
        final_ema=final_ema,
    )


def test_teacher_params_are_ema_not_param():
    """Teacher weights are EMAParam (excluded from grad/optimizer); student stays Param."""
    model = _byol()
    n_param = len(jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    n_ema = len(jax.tree_util.tree_leaves(nnx.state(model, EMAParam)))
    assert n_ema > 0  # teacher backbone + projector
    assert n_param > 0  # student + predictor
    # init: teacher == student
    t0 = float(model.backbone.teacher.layers[0].kernel[...].sum())
    s0 = float(model.backbone.student.layers[0].kernel[...].sum())
    assert abs(t0 - s0) < 1e-6


def test_optimizer_does_not_touch_teacher():
    """Without TeacherStudentCallback the teacher must stay frozen (only EMA moves it)."""
    model = _byol(seed=1)
    before = float(model.backbone.teacher.layers[0].kernel[...].sum())
    spj.Trainer(max_epochs=5).fit(model, _two_view())  # no EMA callback
    after = float(model.backbone.teacher.layers[0].kernel[...].sum())
    assert abs(before - after) < 1e-9


def test_byol_loss_decreases_with_ema_callback():
    model = _byol()
    data = _two_view()
    tr = spj.Trainer(max_epochs=1, callbacks=[spj.TeacherStudentCallback()])
    tr.fit(model, data)
    first = tr.callback_metrics["fit/loss"]
    tr.max_epochs = 25
    tr.fit(model, data)
    assert tr.callback_metrics["fit/loss"] < first


def test_ema_update_math():
    """One update with ema=c must give teacher = c*teacher0 + (1-c)*student."""
    from stable_pretraining.jax.backbone.teacher_student import (
        TeacherStudentWrapper,
        ema_update,
    )

    rngs = nnx.Rngs(0)
    student = spj.backbone.MLP(8, [8], rngs=rngs)
    wrapper = TeacherStudentWrapper(student, base_ema_coefficient=0.9)
    # make student differ from teacher
    wrapper.student.layers[0].kernel = nnx.Param(
        wrapper.student.layers[0].kernel[...] + 1.0
    )
    t0 = np.asarray(wrapper.teacher.layers[0].kernel[...])
    s = np.asarray(wrapper.student.layers[0].kernel[...])
    ema_update(wrapper, 0.9)
    t1 = np.asarray(wrapper.teacher.layers[0].kernel[...])
    np.testing.assert_allclose(t1, 0.9 * t0 + 0.1 * s, rtol=1e-5, atol=1e-6)


def test_ema_cosine_schedule_endpoints():
    from stable_pretraining.jax.backbone.teacher_student import TeacherStudentWrapper

    w = TeacherStudentWrapper(
        spj.backbone.MLP(4, [4], rngs=nnx.Rngs(0)),
        base_ema_coefficient=0.99,
        final_ema_coefficient=1.0,
    )
    w.update_ema_coefficient(0, 10)
    assert abs(w.ema - 0.99) < 1e-6  # start = base
    w.update_ema_coefficient(10, 10)
    assert abs(w.ema - 1.0) < 1e-6  # end = final
