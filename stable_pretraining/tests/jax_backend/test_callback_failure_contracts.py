"""JAX callback empty states and checkpoint failures preserve training state."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from stable_pretraining.jax import checkpoint, forward
from stable_pretraining.jax.callbacks import (
    EarlyStopping,
    OnlineKNN,
    OnlineQueue,
    RankMe,
    LiDAR,
    OnlineWriter,
)
from stable_pretraining.jax.losses.joint_embedding import InfoNCELoss
from stable_pretraining.jax.optim import create_optimizer

pytestmark = [pytest.mark.unit, pytest.mark.jax]


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_failed_atomic_save_preserves_previous_checkpoint_and_removes_temporary_file(
    tmp_path, monkeypatch, operation
):
    path = tmp_path / "state.msgpack"
    path.write_bytes(b"previous")
    monkeypatch.setattr(
        checkpoint.os, operation, Mock(side_effect=OSError("disk failure"))
    )
    with pytest.raises(OSError, match="disk failure"):
        checkpoint._atomic_write(path, b"new")
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]


def test_early_stopping_max_mode_ignores_absent_metric_and_resets_wait():
    cb = EarlyStopping("score", mode="max", patience=2, min_delta=0.1)
    trainer = SimpleNamespace(callback_metrics={}, should_stop=False)
    cb.on_validation_epoch_end(trainer, None)
    assert cb.wait == 0
    for value, expected_wait in [(1.0, 0), (1.05, 1), (1.2, 0), (1.21, 1), (1.22, 2)]:
        trainer.callback_metrics["score"] = value
        cb.on_validation_epoch_end(trainer, None)
        assert cb.wait == expected_wait
    assert trainer.should_stop


def test_empty_queue_and_missing_feature_do_not_create_partial_state():
    cb = OnlineQueue()
    cb.on_train_batch_end(None, None, {"label": jnp.array([0])}, {}, 0)
    assert cb.features is None and cb.labels is None and len(cb) == 0


def test_knn_empty_epochs_and_fifo_eviction():
    cb = OnlineKNN(num_classes=2, k=1, bank_size=2)
    trainer = SimpleNamespace(log=Mock())
    cb.on_train_batch_end(trainer, None, {}, {}, 0)
    cb.on_validation_epoch_start(trainer, None)
    cb.on_validation_batch_end(trainer, None, {}, {}, 0)
    cb.on_validation_epoch_end(trainer, None)
    trainer.log.assert_not_called()
    for label in (0, 1):
        outputs = {
            "embedding": jnp.ones((2, 2)) * label,
            "label": jnp.array([label, label]),
        }
        cb.on_train_batch_end(trainer, None, outputs, {}, label)
    cb.on_validation_epoch_start(trainer, None)
    np.testing.assert_array_equal(cb._bank_labels, [1, 1])
    cb.on_validation_batch_end(trainer, None, {}, {}, 0)
    cb.on_validation_batch_end(trainer, None, outputs, {}, 1)
    cb.on_validation_epoch_end(trainer, None)
    trainer.log.assert_called_once_with("eval/knn_acc", 1.0)


@pytest.mark.parametrize("callback,name", [(RankMe, "rankme"), (LiDAR, "lidar")])
def test_rank_callbacks_skip_empty_epoch_and_log_finite_diagnostics(callback, name):
    cb = callback(verbose=True)
    trainer = SimpleNamespace(log=Mock())
    cb.on_validation_epoch_start(trainer, None)
    cb.on_validation_epoch_end(trainer, None)
    trainer.log.assert_not_called()
    cb.on_validation_batch_end(
        trainer,
        None,
        {
            "embedding": jnp.array([[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.1]]),
            "label": jnp.array([0, 0, 1, 1]),
        },
        {},
        0,
    )
    cb.on_validation_epoch_end(trainer, None)
    logged = dict(call.args for call in trainer.log.call_args_list)
    assert name in logged and f"{name}/entropy" in logged
    assert all(np.isfinite(value) for value in logged.values())


def test_lidar_collapsed_between_class_scatter_has_unit_rank():
    from stable_pretraining.jax.callbacks.lidar import _lidar_components

    assert _lidar_components(np.ones((4, 2)), [0, 0, 1, 1]) == (1.0, 0.0, 0.0)


def test_writer_drops_skipped_epoch_buffer(tmp_path):
    cb = OnlineWriter("embedding", tmp_path, every_n_epochs=2)
    trainer = SimpleNamespace(current_epoch=0)
    cb.on_validation_batch_end(trainer, None, {"embedding": np.ones((2, 2))}, {}, 0)
    cb.on_validation_epoch_end(trainer, None)
    assert not cb._buffers and not list(tmp_path.iterdir())
    trainer.current_epoch = 1
    cb.on_validation_batch_end(
        trainer, None, {"embedding": np.full((1, 2), 3.0)}, {}, 0
    )
    cb.on_validation_epoch_end(trainer, None)
    with np.load(tmp_path / "val_epoch1.npz") as saved:
        np.testing.assert_array_equal(saved["embedding"], [[3.0, 3.0]])


@pytest.mark.parametrize("method", ["vicreg", "barlow_twins", "byol"])
def test_evaluation_forward_preserves_labels_and_skips_training_heads(method):
    backbone = (
        SimpleNamespace(forward_teacher=lambda x: x * 2, forward_student=lambda x: x)
        if method == "byol"
        else lambda x: x * 2
    )
    model = SimpleNamespace(backbone=backbone, training=False)
    batch = {"image": jnp.ones((2, 3)), "label": jnp.array([0, 1])}
    result = getattr(forward, method)(model, batch, "validate")
    np.testing.assert_array_equal(result["embedding"], np.full((2, 3), 2.0))
    np.testing.assert_array_equal(result["label"], batch["label"])
    assert "loss" not in result
    with pytest.raises(ValueError, match="exactly 2"):
        getattr(forward, method)(model, {"views": [batch]}, "fit")
    if method == "byol":
        result = forward.byol(model, {"views": [batch, batch]}, "validate")
        assert result["embedding"].shape == (4, 3)


def test_masked_infonce_excludes_negative_candidate():
    loss = InfoNCELoss()(
        jnp.eye(2), jnp.eye(2), jnp.arange(2), mask=~jnp.eye(2, dtype=bool)
    )
    assert float(loss) == pytest.approx(0.0)


def test_optimizer_factory_and_default_produce_updates():
    ready = optax.sgd(0.1)
    assert create_optimizer(lambda: ready) is ready
    optimizer = create_optimizer(None)
    parameters = {"weight": jnp.ones(2)}
    updates, _ = optimizer.update(parameters, optimizer.init(parameters), parameters)
    assert np.asarray(updates["weight"]).max() < 0
