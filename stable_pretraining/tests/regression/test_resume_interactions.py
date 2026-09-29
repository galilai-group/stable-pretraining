"""Full-state restart agrees with uninterrupted training at an epoch boundary."""

import lightning as pl
import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.callbacks.queue import OnlineQueue

from ._interaction_helpers import (
    _Trace,
    _assert_tree_close,
    _model,
    _probe,
    _run,
    _snapshot,
)

pytestmark = [pytest.mark.regression, pytest.mark.usefixtures("interaction_runtime")]


class _StopAfterEpoch(pl.Callback):
    """Keep the planned training horizon unchanged while stopping the first leg."""

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def _components(before_step):
    module = _model(frequency=2, teacher=True, adam=True)
    trace = _Trace()
    callbacks = [
        _probe(module, frequency=3),
        OnlineQueue("embedding", 5, dim=3, verbose=False),
        spt.callbacks.TeacherStudentCallback(
            update_frequency=2, update_after_backward=before_step, verbose=False
        ),
        trace,
    ]
    return module, callbacks, trace


@pytest.mark.parametrize("before_step", [False, True])
def test_restart_restores_model_optimizers_schedulers_teacher_and_callbacks(
    tmp_path, before_step
):
    reference, callbacks, expected = _components(before_step)
    full_trainer = _run(reference, callbacks, max_epochs=3)

    interrupted, callbacks, first = _components(before_step)
    first_trainer = _run(interrupted, [*callbacks, _StopAfterEpoch()], max_epochs=3)
    assert first_trainer.global_step == 3
    assert first.batches_seen == 6
    _assert_tree_close(first.epochs[0], expected.epochs[0])
    checkpoint = tmp_path / "restart.ckpt"
    first_trainer.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert len(saved["optimizer_states"]) == len(saved["lr_schedulers"]) == 2
    assert all(state["state"] for state in saved["optimizer_states"])
    assert saved["callbacks"][first.state_key]["batches_seen"] == 6
    assert any("ordered_queue_embedding" in key for key in saved["state_dict"])

    resumed, callbacks, second = _components(before_step)
    with torch.no_grad():
        for parameter in resumed.backbone.parameters():
            parameter.add_(10)
    resumed_trainer = _run(resumed, callbacks, max_epochs=3, ckpt_path=checkpoint)

    _assert_tree_close(second.start, first.epochs[0])
    assert second.restored_batches == 6
    assert second.batches_seen == expected.batches_seen == 18
    assert len(second.batches) == 12
    for actual, wanted in zip(second.batches, expected.batches[6:]):
        _assert_tree_close(actual, wanted)
    _assert_tree_close(
        _snapshot(resumed_trainer, resumed), _snapshot(full_trainer, reference)
    )
    assert resumed_trainer.current_epoch == full_trainer.current_epoch == 3
    assert resumed_trainer.global_step == full_trainer.global_step == 9
    for key in ("train/probe_loss_epoch", "eval/probe_accuracy"):
        _assert_tree_close(
            resumed_trainer.callback_metrics[key], full_trainer.callback_metrics[key]
        )
