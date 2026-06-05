"""A lightweight trainer that mirrors Lightning's hook lifecycle for JAX.

The trainer deliberately reuses Lightning's hook *names* and call order
(``on_fit_start`` → ``on_train_epoch_start`` → ``on_train_batch_end`` → …) so
that callbacks ported from the torch backend keep their wiring and the
producer/consumer ordering rules in ``AGENTS.md`` still hold. The engine
itself is JAX-native: a ``nnx.jit``-compiled functional ``train_step`` using
``optax`` instead of ``loss.backward()`` + ``torch.optim``.
"""

from typing import Any, Iterable, Optional

import jax
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from loguru import logger as logging

from .optim import create_optimizer
from .utils import to_jax


def _batch_to_jax(batch: Any) -> Any:
    """Recursively convert a (possibly nested) batch of array-likes to JAX arrays."""
    if isinstance(batch, dict):
        return {k: _batch_to_jax(v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_batch_to_jax(v) for v in batch)
    if hasattr(batch, "detach") or hasattr(batch, "shape") or hasattr(batch, "dtype"):
        return to_jax(batch)
    return batch  # leave scalars / non-arrays untouched


@nnx.jit
def _train_step(module: nnx.Module, optimizer: nnx.Optimizer, batch: dict):
    """One compiled optimization step. Returns ``(state, logs)`` from the forward."""

    def loss_fn(m):
        state, logs = m.compute(batch, "fit")
        return state["loss"], (state, logs)

    (_, (state, logs)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(module)
    optimizer.update(module, grads)
    return state, logs


@nnx.jit
def _eval_step(module: nnx.Module, batch: dict):
    """One compiled forward pass with no parameter update."""
    return module.compute(batch, "validate")


class Callback:
    """Base callback with Lightning-compatible no-op hooks.

    Subclasses override only the hooks they need. Every hook receives the
    :class:`Trainer` and the :class:`~stable_pretraining.jax.Module`; the
    per-batch hooks additionally receive the forward-output ``outputs`` dict
    (the same dict-flow contract as the torch backend), the input ``batch``,
    and ``batch_idx``.
    """

    def setup(self, trainer: "Trainer", module: nnx.Module, stage: str) -> None: ...
    def on_fit_start(self, trainer: "Trainer", module: nnx.Module) -> None: ...
    def on_train_epoch_start(self, trainer: "Trainer", module: nnx.Module) -> None: ...

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        module: nnx.Module,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None: ...

    def on_train_epoch_end(self, trainer: "Trainer", module: nnx.Module) -> None: ...

    def on_validation_epoch_start(
        self, trainer: "Trainer", module: nnx.Module
    ) -> None: ...

    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        module: nnx.Module,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None: ...

    def on_validation_epoch_end(
        self, trainer: "Trainer", module: nnx.Module
    ) -> None: ...

    def on_fit_end(self, trainer: "Trainer", module: nnx.Module) -> None: ...


class Trainer:
    """Minimal JAX training loop with a Lightning-style hook lifecycle.

    Args:
        max_epochs: Number of training epochs.
        callbacks: Sequence of :class:`Callback` instances, run in order.
        max_steps: Optional hard cap on training steps (``-1``/``None`` = no cap).
        enable_validation: Whether to run the validation loop each epoch.
        data_parallel: Enable single-program multiple-data (SPMD) data
            parallelism across all local devices. Model/optimizer state is
            replicated and each batch is sharded along its leading (batch) axis;
            XLA inserts the gradient all-reduce. The same code path runs on 1
            GPU, N GPUs, or N simulated CPU devices. Batch size must be
            divisible by the device count.

    Attributes:
        global_step: Number of completed training steps.
        current_epoch: Index of the epoch currently running.
        callback_metrics: Latest scalar metrics (floats) keyed by name —
            populated from forward ``self.log`` calls and by callbacks.
    """

    def __init__(
        self,
        max_epochs: int = 1,
        callbacks: Iterable[Callback] = (),
        max_steps: Optional[int] = None,
        enable_validation: bool = True,
        data_parallel: bool = False,
        logger=None,
    ):
        self.max_epochs = max_epochs
        self.callbacks = list(callbacks)
        self.max_steps = max_steps if (max_steps and max_steps > 0) else None
        self.enable_validation = enable_validation
        self.data_parallel = data_parallel
        # Loggers are the SAME classes as the torch path (RegistryLogger,
        # TrackioLogger, SwanLabLogger, WandbLogger) — duck-typed on
        # log_hyperparams / log_metrics / save / finalize. So logging looks
        # identical on either backend.
        if logger is None:
            self.loggers = []
        elif isinstance(logger, (list, tuple)):
            self.loggers = list(logger)
        else:
            self.loggers = [logger]
        self.global_step = 0
        self.current_epoch = 0
        self.should_stop = False  # callbacks (e.g. EarlyStopping) can set this
        self.callback_metrics: dict[str, float] = {}
        self._epoch_acc: dict[str, list] = {}  # {key: [sum, count]} for epoch means
        self.optimizer: Optional[nnx.Optimizer] = None
        self._batch_sharding: Optional[NamedSharding] = None
        self._replicated_sharding: Optional[NamedSharding] = None
        self._n_devices = 1

    def _setup_data_parallel(self, module: nnx.Module) -> None:
        """Replicate model/optimizer state across devices; shard batches by axis 0."""
        n = jax.device_count()
        if n <= 1:
            logging.info("  data_parallel requested but only 1 device — running plain.")
            return
        self._n_devices = n
        mesh = Mesh(np.array(jax.devices()), ("data",))
        self._replicated_sharding = NamedSharding(mesh, PartitionSpec())
        self._batch_sharding = NamedSharding(mesh, PartitionSpec("data"))
        # Replicate every parameter / optimizer-state leaf onto all devices.
        targets = (module, self.optimizer) if self.optimizer is not None else module
        state = nnx.state(targets)
        state = jax.device_put(state, self._replicated_sharding)
        nnx.update(targets, state)
        logging.info(f"  Data-parallel over {n} devices (batch sharded on axis 0).")

    def _shard(self, batch: Any) -> Any:
        if self._batch_sharding is None:
            return batch
        # A batch whose leading dim isn't divisible by the device count can't be
        # split evenly (e.g. a ragged final validation batch with drop_last=False).
        # Replicate it instead — correct, just not parallelized over that batch.
        leaves = jax.tree_util.tree_leaves(batch)
        leading = leaves[0].shape[0] if leaves and hasattr(leaves[0], "shape") else 0
        sharding = (
            self._batch_sharding
            if leading and leading % self._n_devices == 0
            else self._replicated_sharding
        )
        return jax.device_put(batch, sharding)

    def _run(self, hook: str, *args, **kwargs) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(self, *args, **kwargs)

    def _record_logs(self, logs: dict) -> None:
        for name, value in logs.items():
            v = float(value)
            self.callback_metrics[name] = v  # latest (callbacks read this)
            acc = self._epoch_acc.setdefault(name, [0.0, 0])
            acc[0] += v
            acc[1] += 1

    def log(self, name: str, value) -> None:
        """Log a scalar from a callback — same path/aggregation as ``Module.log``.

        Routes through the epoch accumulator so values flush to the loggers as
        epoch means (matching the torch path's ``on_epoch=True``), and updates
        ``callback_metrics`` so other callbacks (e.g. ``EarlyStopping``) can read
        the latest value.
        """
        self._record_logs({name: value})

    def _flush_loggers(self) -> None:
        """Push epoch metrics to all loggers (epoch-mean for forward logs)."""
        if not self.loggers:
            return
        metrics = dict(self.callback_metrics)
        # Forward-logged scalars (e.g. fit/loss) -> epoch mean, like on_epoch=True.
        for name, (total, count) in self._epoch_acc.items():
            if count:
                metrics[name] = total / count
        metrics["epoch"] = float(self.current_epoch)
        for lg in self.loggers:
            lg.log_metrics(metrics, step=self.global_step)

    def log_checkpoint_path(self, path) -> None:
        """Record a checkpoint path in any logger that tracks it (e.g. RegistryLogger)."""
        import types

        shim = types.SimpleNamespace(last_model_path=str(path), best_model_path=None)
        for lg in self.loggers:
            if hasattr(lg, "after_save_checkpoint"):
                lg.after_save_checkpoint(shim)

    def fit(
        self,
        module: nnx.Module,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        resume_from: Optional[str] = None,
    ) -> nnx.Module:
        """Train ``module`` over ``train_loader`` with the configured callbacks.

        Args:
            module: A :class:`~stable_pretraining.jax.Module` to optimize.
            train_loader: Iterable yielding batch dicts (arrays/torch tensors
                are converted to JAX arrays automatically).
            val_loader: Optional iterable for the validation loop.
            resume_from: Optional checkpoint path (written by
                :func:`stable_pretraining.jax.checkpoint.save`). Restores module +
                optimizer state and ``global_step`` for exact resume.

        Returns:
            nnx.Module: The trained module (updated in place).
        """
        if getattr(module, "optim", None) is not None:
            tx = create_optimizer(module.optim)
            self.optimizer = nnx.Optimizer(module, tx, wrt=nnx.Param)
        else:
            self.optimizer = None
            logging.info("  No optimizer configured — running in eval-only mode.")

        if resume_from is not None:
            # Restore before sharding so replication happens on the loaded state.
            from .checkpoint import load as _load

            objects = {"module": module}
            if self.optimizer is not None:
                objects["optimizer"] = self.optimizer
            self.global_step = _load(resume_from, **objects)
            logging.info(f"  Resumed from {resume_from} at step {self.global_step}.")

        if self.data_parallel:
            self._setup_data_parallel(module)

        # Log hyperparameters once at the start (same as the torch path).
        hparams = dict(getattr(module, "hparams", {}) or {})
        for lg in self.loggers:
            lg.log_hyperparams(hparams)

        self._run("setup", module, "fit")
        self._run("on_fit_start", module)

        status = "success"
        try:
            for epoch in range(self.max_epochs):
                self.current_epoch = epoch
                self._epoch_acc = {}
                self._train_epoch(module, train_loader)
                if self.enable_validation and val_loader is not None:
                    self._validate(module, val_loader)
                self._flush_loggers()  # emit metrics for this epoch
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    break
                if self.should_stop:  # set by e.g. EarlyStopping
                    logging.info(f"  Early stop requested at epoch {epoch}.")
                    break
        except BaseException:
            status = "failed"
            raise
        finally:
            for lg in self.loggers:
                lg.save()
                lg.finalize(status)

        self._run("on_fit_end", module)
        return module

    def _train_epoch(self, module: nnx.Module, train_loader: Iterable) -> None:
        module.training = True
        self._run("on_train_epoch_start", module)
        for batch_idx, raw in enumerate(train_loader):
            batch = self._shard(_batch_to_jax(raw))
            if self.optimizer is not None:
                state, logs = _train_step(module, self.optimizer, batch)
            else:
                state, logs = _eval_step(module, batch)
            self.global_step += 1
            self._record_logs(logs)
            self._run("on_train_batch_end", module, state, batch, batch_idx)
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break
        self._run("on_train_epoch_end", module)

    def _validate(self, module: nnx.Module, val_loader: Iterable) -> None:
        module.training = False
        self._run("on_validation_epoch_start", module)
        for batch_idx, raw in enumerate(val_loader):
            batch = self._shard(_batch_to_jax(raw))
            state, logs = _eval_step(module, batch)
            self._record_logs(logs)
            self._run("on_validation_batch_end", module, state, batch, batch_idx)
        self._run("on_validation_epoch_end", module)


__all__ = ["Trainer", "Callback"]
