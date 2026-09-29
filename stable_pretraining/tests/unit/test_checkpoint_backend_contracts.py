"""Checkpoint callbacks persist fitted estimators and logger identity without network calls."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from lightning.pytorch.loggers import WandbLogger

from stable_pretraining.callbacks import checkpoint_sklearn as checkpoints

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("nested", [False, True])
def test_fitted_sklearn_models_survive_checkpoint_round_trip(nested):
    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression().fit([[0], [1], [2]], [1, 3, 5])
    value = {"models": [regressor]} if nested else regressor
    model = SimpleNamespace(estimator=value, _private=regressor, other=42)
    cb = checkpoints.SklearnCheckpoint()
    cb.setup(None, model, "fit")
    checkpoint = {}
    cb.on_save_checkpoint(None, model, checkpoint)
    assert checkpoint == {"estimator": value}
    restored = SimpleNamespace()
    cb.on_load_checkpoint(None, restored, checkpoint)
    actual = restored.estimator["models"][0] if nested else restored.estimator
    assert actual.predict([[3]])[0] == pytest.approx(7)
    with pytest.raises(RuntimeError, match="already present"):
        cb.on_save_checkpoint(None, model, checkpoint)


def test_sklearn_unavailable_leaves_checkpoint_and_module_untouched(monkeypatch):
    monkeypatch.setattr(checkpoints, "SKLEARN_AVAILABLE", False)
    cb = checkpoints.SklearnCheckpoint()
    checkpoint = {"existing": 1}
    model = SimpleNamespace(value=[{"a": 2}])
    cb.on_save_checkpoint(None, model, checkpoint)
    cb.on_load_checkpoint(None, model, checkpoint)
    assert checkpoint == {"existing": 1}
    assert vars(model) == {"value": [{"a": 2}]}


@pytest.mark.parametrize("rank_zero", [False, True])
def test_wandb_checkpoint_identity_and_sidecar(tmp_path, rank_zero):
    logger = Mock(spec=WandbLogger)
    logger.version = "run-123"
    logger._wandb_init = {"id": "run-123", "project": "project", "entity": "team"}
    trainer = SimpleNamespace(
        loggers=[logger], is_global_zero=rank_zero, default_root_dir=str(tmp_path)
    )
    cb = checkpoints.WandbCheckpoint()
    cb.setup(trainer, None)
    state = {}
    cb.on_save_checkpoint(trainer, None, state)
    assert state["wandb"] == {"id": "run-123", "project": "project", "entity": "team"}
    sidecar = tmp_path / "wandb_resume.json"
    assert sidecar.exists() == rank_zero
    if rank_zero:
        assert json.loads(sidecar.read_text()) == state["wandb"]
    cb.on_load_checkpoint(trainer, None, state)
    logger._wandb_init["id"] = "wrong"
    cb.on_load_checkpoint(trainer, None, state)
    assert logger._wandb_init["id"] == "wrong"
    assert not logger.experiment.called


def test_wandb_absent_and_ambiguous_loggers():
    cb = checkpoints.WandbCheckpoint()
    empty = SimpleNamespace(loggers=[])
    state = {}
    cb.on_save_checkpoint(empty, None, state)
    cb.on_load_checkpoint(empty, None, state)
    cb.on_load_checkpoint(empty, None, {"wandb": {"id": "run"}})
    assert not state
    with pytest.raises(RuntimeError, match="Found 2"):
        checkpoints.find_wandb_logger(
            SimpleNamespace(loggers=[Mock(spec=WandbLogger), Mock(spec=WandbLogger)])
        )
