"""Callback for running evaluation functions on arbitrary datasets during training."""

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import TrainableCallback


class EvalOnDataset(Callback):
    """Run evaluation functions on a custom dataset every N epochs.

    Args:
        name: Unique name for this eval run (used as log prefix).
        data: Pre-built DataLoader for the eval dataset.
        evaluators: List of callables with signature
            fn(trainer, pl_module, dataloader) -> dict[str, float].
        every_n_epochs: Run evaluation every N epochs.
        start_epoch: First epoch to run evaluation.
    """

    def __init__(
        self,
        name: str,
        data: DataLoader,
        evaluators: list,
        every_n_epochs: int = 1,
        start_epoch: int = 0,
    ):
        super().__init__()
        self.name = name
        self._dataloader = data
        self.evaluators = evaluators
        self.every_n_epochs = every_n_epochs
        self.start_epoch = start_epoch

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Add DistributedSampler if needed."""
        if trainer.world_size > 1 and not isinstance(
            self._dataloader.sampler, DistributedSampler
        ):
            logging.info(f"{self.name}: Adding DistributedSampler for DDP")
            sampler = DistributedSampler(
                self._dataloader.dataset,
                num_replicas=trainer.world_size,
                rank=trainer.global_rank,
                shuffle=False,
            )
            self._dataloader = DataLoader(
                dataset=self._dataloader.dataset,
                batch_size=self._dataloader.batch_size,
                sampler=sampler,
                num_workers=self._dataloader.num_workers,
                pin_memory=self._dataloader.pin_memory,
                drop_last=self._dataloader.drop_last,
                collate_fn=self._dataloader.collate_fn,
            )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        if epoch < self.start_epoch:
            return
        if (epoch - self.start_epoch) % self.every_n_epochs != 0:
            return

        logging.info(f"[Epoch {epoch}] Running {self.name} evaluation")

        # Set sampler epoch for proper sharding
        if hasattr(self._dataloader.sampler, "set_epoch"):
            self._dataloader.sampler.set_epoch(epoch)

        # Run each evaluator
        all_metrics = {}
        for evaluator in self.evaluators:
            metrics = evaluator(trainer, pl_module, self._dataloader)
            if metrics:
                all_metrics.update(metrics)

        # Log on rank 0
        if trainer.is_global_zero and all_metrics:
            prefixed = {f"eval/{self.name}/{k}": v for k, v in all_metrics.items()}
            if trainer.logger is not None:
                trainer.logger.log_metrics(prefixed, step=trainer.global_step)
            logging.info(f"[Epoch {epoch}] {self.name}: {all_metrics}")

        # DDP barrier
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def callback_to_evaluator(callback):
    """Adapt a Lightning Callback to fn(trainer, pl_module, dataloader) -> dict.

    For callbacks using on_validation_batch_end (CLIPZeroShot, OnlineKNN, etc.):
        Calls setup, iterates batches calling the hook, collects metrics.

    For TrainableCallback subclasses (OnlineProbe):
        Calls forward with stage="eval_on_dataset" (probe computes predictions
        but skips logging), extracts predictions from outputs, computes metrics
        with cloned instances.

    Args:
        callback: A Lightning Callback instance. Must have a unique name to
            avoid conflicting with callbacks in the main training loop.

    Returns:
        Callable with signature fn(trainer, pl_module, dataloader) -> dict
    """
    _setup_done = False

    def eval_fn(trainer, pl_module, dataloader):
        nonlocal _setup_done
        device = pl_module.device

        # One-time setup
        if not _setup_done:
            callback.setup(trainer, pl_module, "validate")
            _setup_done = True

        if isinstance(callback, TrainableCallback):
            return _eval_trainable_callback(
                callback, trainer, pl_module, dataloader, device
            )
        else:
            return _eval_hook_callback(
                callback, trainer, pl_module, dataloader, device
            )

    return eval_fn


def _eval_hook_callback(callback, trainer, pl_module, dataloader, device):
    """Drive a callback that uses on_validation_batch_end."""
    # Reset metrics before eval
    if callback.name in pl_module.callbacks_metrics:
        for m in pl_module.callbacks_metrics[callback.name]["_val"].values():
            m.reset()

    for batch_idx, batch in enumerate(dataloader):
        batch = _move_to_device(batch, device)
        callback.on_validation_batch_end(
            trainer, pl_module, {}, batch, batch_idx
        )

    # Collect metrics
    metrics = {}
    if callback.name in pl_module.callbacks_metrics:
        for name, m in pl_module.callbacks_metrics[callback.name]["_val"].items():
            metrics[f"{callback.name}_{name}"] = m.compute().item()
            m.reset()
    return metrics


def _eval_trainable_callback(callback, trainer, pl_module, dataloader, device):
    """Drive a TrainableCallback (e.g. OnlineProbe) that wraps forward."""
    # Clone metrics to avoid interfering with main validation
    cloned_metrics = {
        name: m.clone().to(device)
        for name, m in pl_module.callbacks_metrics[callback.name]["_val"].items()
    }

    prediction_key = f"{callback.name}_preds"

    for batch_idx, batch in enumerate(dataloader):
        batch = _move_to_device(batch, device)
        with torch.no_grad():
            outputs = pl_module.forward(batch, stage="eval_on_dataset")

        if prediction_key in outputs:
            preds = outputs[prediction_key]
            target = batch.get(callback.target)
            if target is not None:
                for m in cloned_metrics.values():
                    m(preds, target)

    # Compute and return
    metrics = {}
    for name, m in cloned_metrics.items():
        metrics[f"{callback.name}_{name}"] = m.compute().item()
    return metrics


def _move_to_device(batch, device):
    """Move all tensors in a batch to the target device."""
    if isinstance(batch, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    elif isinstance(batch, (list, tuple)):
        return type(batch)(
            v.to(device) if isinstance(v, torch.Tensor) else v for v in batch
        )
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch
