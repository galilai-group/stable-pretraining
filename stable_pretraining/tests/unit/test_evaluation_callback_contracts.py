"""Online evaluators consume queue snapshots and log numerically meaningful results."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from stable_pretraining.callbacks.knn import OnlineKNN
from stable_pretraining.callbacks.lidar import LiDAR
from stable_pretraining.callbacks.rankme import RankMe

pytestmark = pytest.mark.unit


def evaluator(kind, verbose=True):
    if kind == "rankme":
        return RankMe("quality", "embedding", 16, (2, 2), verbose=verbose)
    return LiDAR(
        "quality",
        "embedding",
        16,
        (4,),
        n_classes=4,
        samples_per_class=2,
        verbose=verbose,
    )


@pytest.mark.parametrize("kind", ["rankme", "lidar"])
def test_spectral_callbacks_discover_queue_once_and_skip_unavailable_data(
    monkeypatch, kind
):
    import importlib

    source = importlib.import_module("stable_pretraining.callbacks." + kind)
    callback = evaluator(kind)
    queue = SimpleNamespace(data=None)
    find = Mock(return_value=queue)
    monkeypatch.setattr(source, "find_or_create_queue_callback", find)
    trainer, module = SimpleNamespace(global_rank=0), SimpleNamespace(log=Mock())
    callback.setup(trainer, module, "fit")
    callback.setup(trainer, module, "validate")
    find.assert_called_once()
    assert "quality" in callback.state_key
    for data in [None, torch.empty(0, 4)]:
        queue.data = data
        callback.on_validation_batch_end(trainer, module, {}, {}, 0)
    module.log.assert_not_called()
    queue.data = torch.eye(4).repeat_interleave(2, 0)
    callback.on_validation_batch_end(trainer, module, {}, {}, 1)
    module.log.assert_not_called()
    extra_log = Mock()
    monkeypatch.setattr(source, "_spt_log", extra_log)
    callback.on_validation_batch_end(trainer, module, {}, {}, 0)
    score = module.log.call_args.args[1]
    assert 2.9 < score <= 4.01
    assert extra_log.call_count == (3 if kind == "rankme" else 2)


@pytest.mark.parametrize("label_location", ["batch", "outputs"])
def test_knn_callback_uses_available_targets_and_updates_real_metric(
    monkeypatch, label_location
):
    import stable_pretraining.callbacks.knn as source

    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    callback = OnlineKNN(
        "nearest",
        "embedding",
        "label",
        4,
        metrics={"accuracy": MulticlassAccuracy(2)},
        input_dim=(2,),
        k=1,
        num_classes=2,
    )
    queues = [SimpleNamespace(data=features), SimpleNamespace(data=labels)]
    find = Mock(side_effect=queues)
    monkeypatch.setattr(source, "find_or_create_queue_callback", find)
    module = SimpleNamespace(callbacks_metrics=nn.ModuleDict(), log_dict=Mock())
    callback.setup(None, module, "fit")
    callback.setup(None, module, "fit")
    assert find.call_count == 2
    assert callback.state_key == "OnlineKNN[name=nearest]"
    batch, outputs = {}, {"embedding": features}
    (batch if label_location == "batch" else outputs)["label"] = labels
    callback.on_validation_batch_end(None, module, outputs, batch, 0)
    torch.testing.assert_close(batch["nearest_preds"].argmax(1), labels)
    metric = module.callbacks_metrics["nearest"]["_val"]["accuracy"]
    assert metric.compute() == 1
    with pytest.raises(ValueError, match="already exists"):
        callback.on_validation_batch_end(None, module, outputs, batch, 0)


@pytest.mark.parametrize(
    "missing", ["input", "target", "missing_input", "missing_target", "queue", "empty"]
)
def test_knn_skips_incomplete_inputs_without_emitting_predictions(missing):
    callback = OnlineKNN("nearest", "embedding", "label", 4, metrics={})
    data = torch.eye(2)
    callback._input_queue = SimpleNamespace(
        data=None if missing == "queue" else data[:0] if missing == "empty" else data
    )
    callback._target_queue = SimpleNamespace(data=torch.tensor([0, 1]))
    batch = {"embedding": data, "label": torch.tensor([0, 1])}
    if missing.startswith("missing_"):
        del batch["embedding" if missing == "missing_input" else "label"]
        with pytest.raises(ValueError, match="not found"):
            callback.on_validation_batch_end(None, None, {}, batch, 0)
    else:
        if missing in ("input", "target"):
            batch["embedding" if missing == "input" else "label"] = None
        callback.on_validation_batch_end(None, None, {}, batch, 0)
    assert "nearest_preds" not in batch


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"temperature": 0}, {"chunk_size": 0}])
def test_knn_rejects_nonpositive_search_settings(kwargs):
    with pytest.raises(ValueError, match="positive"):
        OnlineKNN("nearest", "embedding", "label", 4, metrics={}, **kwargs)
