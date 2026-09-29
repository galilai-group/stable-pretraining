"""Embedding hooks attach atomically and leave the model reusable after teardown."""

import lightning as pl
import pytest
import torch
from torch import nn

from stable_pretraining.callbacks.embedding_cache import EmbeddingCache

pytestmark = pytest.mark.unit


class _Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 4), nn.ReLU())

    def forward(self, x):
        return {"output": self.encoder(x)}


@pytest.mark.parametrize("merge", [False, True])
def test_cache_hooks_preserve_gradients_and_are_removed_after_teardown(merge):
    model = _Model()
    callback = EmbeddingCache(["encoder.0"], add_to_forward_output=merge)
    callback.setup(None, model)
    x = torch.ones(2, 3, requires_grad=True)
    result = model(x)
    assert ("encoder.0" in result) is merge
    torch.testing.assert_close(model.embedding_cache["encoder.0"], model.encoder[0](x))
    model.embedding_cache["encoder.0"].sum().backward()
    assert x.grad is not None
    callback.teardown(None, model)
    assert not hasattr(model, "embedding_cache")
    assert not model._forward_hooks and not model.encoder[0]._forward_hooks
    assert model(x)["output"].shape == (2, 4)
    callback.teardown(None, model)
    callback.setup(None, model)
    assert model(x)["output"].shape == (2, 4)
    callback.teardown(None, model)


@pytest.mark.parametrize("stage", ["train", "validation", "test"])
def test_each_stage_clears_previous_batch_embeddings(stage):
    model = _Model()
    callback = EmbeddingCache(["encoder.0"])
    callback.setup(None, model)
    model(torch.ones(2, 3))
    getattr(callback, f"on_{stage}_batch_start")(None, model, {}, 0)
    assert model.embedding_cache == {}
    callback.teardown(None, model)


def test_invalid_layer_does_not_partially_mutate_model():
    model = _Model()
    callback = EmbeddingCache(["encoder.0", "missing.layer"])
    with pytest.raises(ValueError, match="not found"):
        callback.setup(None, model)
    assert not model.encoder[0]._forward_hooks
    assert not hasattr(model, "embedding_cache")


def test_existing_cache_is_not_overwritten():
    model = _Model()
    model.embedding_cache = {"existing": 1}
    with pytest.raises(RuntimeError, match="already present"):
        EmbeddingCache(["encoder.0"]).setup(None, model)
    assert model.embedding_cache == {"existing": 1}
