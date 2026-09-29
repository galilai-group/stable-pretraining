"""First-step checkpointing and full-state resume before the first epoch ends."""

import lightning as pl
import pytest
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader

from stable_pretraining._config import get_config
from stable_pretraining.data import DataModule
from stable_pretraining.manager import Manager, _RequeueCheckpoint

pytestmark = pytest.mark.unit


class _TrainModule(pl.LightningModule):
    """Small model with nonempty optimizer state and optional manual optimization."""

    def __init__(self, manual_steps=0):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.manual_steps = manual_steps
        self.automatic_optimization = manual_steps == 0
        self.saved_steps = []

    def training_step(self, batch, batch_idx):
        if self.automatic_optimization:
            return self.layer(batch["x"]).square().mean()
        optimizer = self.optimizers()
        for _ in range(self.manual_steps):
            optimizer.zero_grad()
            loss = self.layer(batch["x"]).square().mean()
            self.manual_backward(loss)
            optimizer.step()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint):
        self.saved_steps.append(checkpoint["global_step"])


def _loader(n=8):
    return DataLoader([{"x": torch.ones(2)} for _ in range(n)], batch_size=1)


def _checkpoint(path, interval=0):
    return _RequeueCheckpoint(
        dirpath=path,
        filename="last",
        save_last=False,
        every_n_train_steps=interval or None,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )


def _trainer(checkpoint, **kwargs):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        callbacks=[checkpoint] if checkpoint is not None else [],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        **kwargs,
    )


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize(
    "interval,expected",
    [(0, [1, 4]), (2, [1, 2, 4]), (1, [1, 2, 3, 4]), (3, [1, 3, 4])],
)
def test_first_step_then_regular_schedule(tmp_path, accumulation, interval, expected):
    checkpoint = _checkpoint(tmp_path, interval)
    module = _TrainModule()
    trainer = _trainer(checkpoint, accumulate_grad_batches=accumulation)
    trainer.fit(module, _loader(4 * accumulation))
    assert module.saved_steps == expected
    assert sorted(path.name for path in tmp_path.glob("*.ckpt")) == ["last.ckpt"]
    saved = torch.load(tmp_path / "last.ckpt", weights_only=False)
    assert saved["global_step"] == 4
    assert saved["optimizer_states"][0]["state"]
    assert saved["lr_schedulers"]


@pytest.mark.parametrize("manual_steps", [1, 2])
def test_manual_optimization_saves_after_first_batch_with_updates(
    tmp_path, manual_steps
):
    module = _TrainModule(manual_steps=manual_steps)
    _trainer(_checkpoint(tmp_path)).fit(module, _loader(4))
    assert module.saved_steps == [manual_steps, 4 * manual_steps]


def test_checkpoint_exists_before_epoch_end_and_resumes(tmp_path):
    class Interrupt(Callback):
        """Simulate preemption after the first completed training batch."""

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if trainer.global_step == 1:
                raise RuntimeError("simulated preemption")

    checkpoint = _checkpoint(tmp_path)
    module = _TrainModule()
    trainer = _trainer(checkpoint)
    trainer.callbacks.append(Interrupt())
    with pytest.raises(RuntimeError, match="simulated preemption"):
        trainer.fit(module, _loader())

    path = tmp_path / "last.ckpt"
    saved = torch.load(path, weights_only=False)
    assert saved["global_step"] == 1
    assert saved["optimizer_states"][0]["state"]
    for name, value in module.state_dict().items():
        torch.testing.assert_close(saved["state_dict"][name], value)

    resumed = _TrainModule()
    resumed_trainer = _trainer(_checkpoint(tmp_path), max_steps=3)
    resumed_trainer.fit(resumed, _loader(), ckpt_path=path, weights_only=False)
    assert resumed_trainer.global_step == 3
    assert resumed.saved_steps == [3]
    assert all(
        state["step"].item() == 3
        for state in resumed_trainer.optimizers[0].state.values()
    )
    assert resumed_trainer.lr_scheduler_configs[0].scheduler.last_epoch == 3


def test_fresh_directory_loaded_from_user_checkpoint_saves_first_new_step(tmp_path):
    source = tmp_path / "source"
    _trainer(_checkpoint(source), max_steps=1).fit(_TrainModule(), _loader())
    destination = tmp_path / "fresh"
    module = _TrainModule()
    trainer = _trainer(_checkpoint(destination), max_steps=3)
    trainer.fit(module, _loader(), ckpt_path=source / "last.ckpt", weights_only=False)
    assert module.saved_steps == [2, 3]
    assert (source / "last.ckpt").is_file()
    assert (destination / "last.ckpt").is_file()


def test_state_key_preserves_existing_checkpoint_compatibility(tmp_path):
    checkpoint = _checkpoint(tmp_path, interval=2)
    previous = ModelCheckpoint(every_n_train_steps=2)
    assert checkpoint.state_key == previous.state_key


def test_fast_dev_run_does_not_write_checkpoint(tmp_path):
    module = _TrainModule()
    _trainer(_checkpoint(tmp_path), fast_dev_run=True).fit(module, _loader())
    assert not module.saved_steps
    assert not (tmp_path / "last.ckpt").exists()


@pytest.mark.parametrize("user_checkpoint", [False, True])
def test_disabled_requeue_checkpoint_does_not_add_first_step_save(
    tmp_path, monkeypatch, user_checkpoint
):
    monkeypatch.setattr(get_config(), "requeue_checkpoint", False)
    checkpoint = (
        ModelCheckpoint(filename="user", every_n_train_steps=3)
        if user_checkpoint
        else None
    )
    module = _TrainModule()
    trainer = _trainer(checkpoint, enable_checkpointing=user_checkpoint)
    manager = Manager(
        trainer=trainer,
        module=module,
        data=DataModule(train=_loader(4)),
        seed=0,
    )
    manager()
    assert module.saved_steps == ([3] if user_checkpoint else [])
    assert not any(isinstance(cb, _RequeueCheckpoint) for cb in trainer.callbacks)
    assert not (manager._run_dir / "checkpoints" / "last.ckpt").exists()
