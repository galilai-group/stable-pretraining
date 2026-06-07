"""JAX-specific tests for the Lightning-style trainer + hook lifecycle."""

import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

from stable_pretraining.jax import Callback, Module, Trainer  # noqa: E402
from stable_pretraining.jax.backbone import MLP  # noqa: E402
from stable_pretraining.jax.forward import supervised  # noqa: E402


class _Recorder(Callback):
    """Records the order of hook invocations and inspects per-batch outputs."""

    def __init__(self):
        self.events = []
        self.seen_loss_key = False

    def on_fit_start(self, trainer, module):
        self.events.append("on_fit_start")

    def on_train_epoch_start(self, trainer, module):
        self.events.append("on_train_epoch_start")

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.events.append("on_train_batch_end")
        if "loss" in outputs and "embedding" in outputs:
            self.seen_loss_key = True

    def on_train_epoch_end(self, trainer, module):
        self.events.append("on_train_epoch_end")

    def on_validation_epoch_start(self, trainer, module):
        self.events.append("on_validation_epoch_start")

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.events.append("on_validation_batch_end")

    def on_validation_epoch_end(self, trainer, module):
        self.events.append("on_validation_epoch_end")

    def on_fit_end(self, trainer, module):
        self.events.append("on_fit_end")


def _supervised_module(d=8, c=3, seed=0):
    rngs = nnx.Rngs(seed)
    backbone = MLP(d, [16], rngs=rngs)
    classifier = nnx.Linear(16, c, rngs=rngs)
    return Module(
        forward=supervised, optim="adamw", backbone=backbone, classifier=classifier
    )


def _data(n_batches=3, b=8, d=8, c=3, seed=0):
    rng = np.random.RandomState(seed)
    return [
        {
            "image": rng.randn(b, d).astype("float32"),
            "label": rng.randint(0, c, size=b),
        }
        for _ in range(n_batches)
    ]


def test_hook_order_matches_lightning():
    rec = _Recorder()
    trainer = Trainer(max_epochs=2, callbacks=[rec])
    trainer.fit(_supervised_module(), _data(2), _data(1))
    # Lightning order: fit_start, then per epoch train_* then validation_*.
    assert rec.events[0] == "on_fit_start"
    assert rec.events[-1] == "on_fit_end"
    assert rec.events.count("on_train_epoch_start") == 2
    assert rec.events.count("on_train_batch_end") == 4  # 2 epochs * 2 batches
    assert rec.events.count("on_validation_epoch_end") == 2
    # train epoch fully precedes its validation epoch
    first_val = rec.events.index("on_validation_epoch_start")
    assert "on_train_epoch_end" in rec.events[:first_val]


def test_outputs_dict_reaches_callbacks():
    rec = _Recorder()
    Trainer(max_epochs=1, callbacks=[rec]).fit(_supervised_module(), _data(2))
    assert rec.seen_loss_key


def test_global_step_counts_training_batches():
    trainer = Trainer(max_epochs=3, callbacks=[])
    trainer.fit(_supervised_module(), _data(4))
    assert trainer.global_step == 12


def test_max_steps_caps_training():
    trainer = Trainer(max_epochs=10, max_steps=5, callbacks=[])
    trainer.fit(_supervised_module(), _data(4))
    assert trainer.global_step == 5


def test_supervised_loss_decreases():
    module = _supervised_module()
    data = _data(4)
    trainer = Trainer(max_epochs=1, callbacks=[])
    trainer.fit(module, data)
    first = trainer.callback_metrics["fit/loss"]
    trainer.max_epochs = 20
    trainer.fit(module, data)
    assert trainer.callback_metrics["fit/loss"] < first


def test_eval_only_module_runs_without_optimizer():
    """optim=None disables optimization; fit still flows batches to callbacks."""
    rngs = nnx.Rngs(0)
    backbone = MLP(8, [16], rngs=rngs)
    classifier = nnx.Linear(16, 3, rngs=rngs)
    module = Module(
        forward=supervised, optim=None, backbone=backbone, classifier=classifier
    )
    rec = _Recorder()
    trainer = Trainer(max_epochs=1, callbacks=[rec])
    trainer.fit(module, _data(2))
    assert trainer.optimizer is None
    assert rec.events.count("on_train_batch_end") == 2
