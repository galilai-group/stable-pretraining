import re
import types
from functools import partial

import lightning as pl
import torch
import torchmetrics
from loguru import logger as logging
from omegaconf import DictConfig
from tabulate import tabulate
from pathlib import Path
from prettytable import PrettyTable
from lightning.pytorch.core.optimizer import LightningOptimizer
from .optim import create_optimizer, create_scheduler
from stable_pretraining.utils.error_handling import catch_errors_class
from stable_pretraining.callbacks.registry import log as _spt_log
from stable_pretraining.callbacks.utils import log_header


class _NamedForward:
    """Adapter giving a callable ``__name__`` so spawn-mode workers can pickle it."""

    __name__ = "forward"

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _ensure_named_callable(fn):
    return fn if hasattr(fn, "__name__") else _NamedForward(fn)


def _noop_hook() -> None:
    """Drop-in for ``LightningOptimizer._on_before_step`` / ``_on_after_step``.

    Used by :meth:`Module.training_step` to suppress the manual-opt
    progress counter for callback-owned optimizers without disturbing
    any of the strategy / precision-plugin logic inside
    ``LightningOptimizer.step()``.
    """
    return None


def _gpu_transform_from_current_dataset(trainer, stage, dataloader_idx):
    """Return ``dataset.gpu_transform`` for the active stage's DataLoader, or None.

    Lightning exposes the live dataloaders as ``trainer.train_dataloader``
    (singular) for training and ``trainer.val_dataloaders`` / similar
    (potentially a list) for the other stages. We peek into them, unwrap
    common dataset wrappers (``Subset``, ``ConcatDataset``), and read the
    ``gpu_transform`` attribute if present.

    Best-effort: any exception during lookup yields None, so the resolver
    falls through to the DataModule-level transform.
    """
    try:
        loader = None
        if stage == "train":
            loader = trainer.train_dataloader
        elif stage in ("val", "test", "predict"):
            attr = {
                "val": "val_dataloaders",
                "test": "test_dataloaders",
                "predict": "predict_dataloaders",
            }[stage]
            loaders = getattr(trainer, attr, None)
            if isinstance(loaders, (list, tuple)):
                if 0 <= dataloader_idx < len(loaders):
                    loader = loaders[dataloader_idx]
            else:
                loader = loaders
        if loader is None:
            return None
        dataset = getattr(loader, "dataset", None)
        while dataset is not None:
            if hasattr(dataset, "gpu_transform") and dataset.gpu_transform is not None:
                return dataset.gpu_transform
            # Walk through standard wrappers (torch ``Subset``,
            # ``ConcatDataset``) and our own attribute-proxying ``Subset``.
            inner = getattr(dataset, "dataset", None) or getattr(
                dataset, "datasets", None
            )
            if inner is dataset or inner is None:
                break
            if isinstance(inner, (list, tuple)):
                inner = inner[0]
            dataset = inner
    except Exception:
        return None
    return None


@catch_errors_class()
class Module(pl.LightningModule):
    """PyTorch Lightning module using manual optimization with multi-optimizer support.

    **Core usage**

    - Provide a custom ``forward(self, batch, stage)`` via the ``forward``
      argument at init.
    - During training, ``forward`` must return a dict with ``state["loss"]``
      (a single joint loss). When multiple optimizers are configured, this
      joint loss is used for all optimizers.

    **Optimizer configuration** (``self.optim``)

    - Single optimizer::

        {"optimizer": str|dict|partial|Class,
         "scheduler": <see below>,
         "interval": "step"|"epoch",
         "frequency": int}

      Optimizer accepted forms:

      * string name (e.g., ``"AdamW"``, ``"SGD"``) from ``torch.optim``
      * dict: ``{"type": "AdamW", "lr": 1e-3, ...}``
      * ``functools.partial``: ``partial(torch.optim.AdamW, lr=1e-3)``
      * optimizer class: ``torch.optim.AdamW``

    - Multiple optimizers::

        {
          name: {
            "modules": "regex",                  # assign params by module-name pattern (children inherit)
            "optimizer": str|dict|partial|Class, # optimizer factory (same accepted forms as above)
            "scheduler": str|dict|partial|Class, # flexible scheduler config (see below)
            "interval": "step"|"epoch",          # scheduler interval
            "frequency": int,                    # optimizer step frequency
            "monitor": str                       # (optional) for ReduceLROnPlateau
          }, ...
        }

    **Parameter assignment** (multi-optimizer)

    - Modules are matched by regex on their qualified name. Children
      inherit the parent's assignment unless they match a more specific
      pattern. Only direct parameters of each module are collected to
      avoid duplication.

    **Schedulers** (flexible)

    - Accepted forms: string name (e.g., ``"CosineAnnealingLR"``,
      ``"StepLR"``), dict with ``{"type": "...", ...}``,
      ``functools.partial``, or a scheduler class. Smart defaults are
      applied when params are omitted for common schedulers
      (``CosineAnnealingLR``, ``OneCycleLR``, ``StepLR``,
      ``ExponentialLR``, ``ReduceLROnPlateau``, ``LinearLR``,
      ``ConstantLR``). For ``ReduceLROnPlateau``, a ``monitor`` key is
      added (default: ``"val_loss"``). You may specify ``monitor`` either
      alongside the optimizer config (top level) or inside the scheduler
      dict itself.
    - The resulting Lightning scheduler dict includes ``interval`` and
      ``frequency`` (or ``scheduler_frequency``).

    **Training loop behavior**

    - Manual optimization (``automatic_optimization = False``).
    - Gradient accumulation: scales loss by ``1/N`` where
      ``N = Trainer.accumulate_grad_batches`` and steps on the boundary.
    - Per-optimizer step frequency: each optimizer steps only when its
      frequency boundary is met (in addition to accumulation boundary).
    - Gradient clipping: uses Trainer's ``gradient_clip_val`` and
      ``gradient_clip_algorithm`` before each step.
    - Returns the ``state`` dict from ``forward`` unchanged for
      logging/inspection.
    """

    _warned_named_parameters = False

    def __init__(
        self,
        *args,
        forward: callable = None,
        hparams: dict = None,
        parallelize_fn: callable = None,
        **kwargs,
    ):
        super().__init__()
        log_header("Module")

        # Manual optimization to support multiple optimizers and custom stepping
        self.automatic_optimization = False

        # FSDP2 sharding hook (used only under ``strategy="fsdp2"``). ``None``
        # means "use the library default" — resolved lazily in
        # ``configure_model`` so we don't import the FSDP utils on every Module.
        self._parallelize_fn = parallelize_fn
        self.callbacks_modules = torch.nn.ModuleDict()
        self.callbacks_metrics = torch.nn.ModuleDict()

        self._optimizer_index_to_name = {}
        self._optimizer_frequencies = {}
        self._optimizer_gradient_clip_val = {}
        self._optimizer_gradient_clip_algorithm = {}
        # Optimizer names registered by ``TrainableCallback`` (e.g. ``OnlineProbe``).
        # Their ``.step()`` is dispatched on the underlying ``torch.optim.Optimizer``
        # rather than the Lightning wrapper so they don't advance
        # ``trainer.global_step`` — see :meth:`training_step` and the
        # rationale below in ``training_step``'s docstring.
        self._callback_optimizer_names = set()

        if len(args) > 0:
            raise ValueError(
                "Module does not accept positional arguments (*args). Please use keyword arguments instead (e.g., Module(forward=my_forward, hparams=my_hparams))."
            )

        if hparams is None:
            logging.warning(
                "! No hyperparameters provided - hyperparameter logging is disabled."
            )
        else:
            logging.info("  Saving provided hyperparameters.")
            self.save_hyperparameters(hparams)
        self.save_hyperparameters(
            {**self.hparams, "system.working_dir": str(Path().resolve())}
        )

        logging.info("  Setting custom forward method.")
        if forward is None:
            logging.warning(
                "! You didn't pass a forward method. "
                "This will fail unless you implemented your own Module class."
            )
        elif not callable(forward):
            msg = "! You passed a `forward' object that is not callable!"
            logging.warning(msg)
            raise ValueError(msg)
        else:
            setattr(
                self, "forward", types.MethodType(_ensure_named_callable(forward), self)
            )

        for key, value in kwargs.items():
            logging.info(f"  Setting attribute: self.{key} = {type(value)}")
            setattr(self, key, value)

        headers = ["Stage", "Inputs", "Metric"]
        if hasattr(self, "metrics"):
            stats = []
            assert isinstance(self.metrics, torch.nn.ModuleDict)
            logging.info("  Metrics:")
            for stage, metrics in self.metrics.items():
                assert (
                    isinstance(metrics, torch.nn.ModuleDict)
                    or isinstance(metrics, torch.nn.ModuleList)
                    or isinstance(metrics, torchmetrics.Metric)
                )
                for name, metric in metrics.items():
                    stats.append([stage, name, str(metric)])
            logging.info(f"\n{tabulate(stats, headers, tablefmt='heavy_outline')}")
        else:
            self.metrics = dict(train={}, validate={}, test={}, predict={})
            logging.info(
                "  No metrics configuration provided - automatic metric tracking is disabled."
            )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("The forward() method must be implemented.")

    def setup(self, stage: str) -> None:
        """Lightning setup hook — reject the legacy FSDP1 strategy.

        FSDP1 (``Trainer(strategy="fsdp")``) uses a flat training-state machine
        that asserts a single forward/backward per step, which breaks the
        multi-forward methods that are this library's bread and butter (DINO,
        I-JEPA, LeJEPA, ...). FSDP2 (``strategy="fsdp2"``) has no such
        restriction, so we fail fast with a clear redirect.

        Args:
            stage: The Lightning stage (``"fit"``/``"validate"``/``"test"``/``"predict"``).
        """
        from lightning.pytorch.strategies import FSDPStrategy

        if isinstance(getattr(self.trainer, "strategy", None), FSDPStrategy):
            raise RuntimeError(
                "stable-pretraining does not support FSDP1 "
                "(Trainer(strategy='fsdp')). Use strategy='fsdp2' instead, which "
                "supports the multi-forward methods (DINO/I-JEPA/LeJEPA/...)."
            )

    def configure_model(self) -> None:
        """Lightning hook for FSDP2 — shard the model when a device mesh exists.

        Under ``Trainer(strategy="fsdp2")``, ``ModelParallelStrategy`` builds a
        device mesh and exposes it as ``self.device_mesh`` before calling this
        hook. We dispatch to the configured ``parallelize_fn`` (default:
        :func:`stable_pretraining.utils.fsdp2.default_parallelize_fn`) to apply
        ``fully_shard``. Under any other strategy (single-device / DDP) there is
        no mesh and this is a no-op.

        Runs before ``configure_optimizers``, so optimizers are built over the
        sharded ``DTensor`` parameters, as FSDP2 requires.
        """
        device_mesh = getattr(self, "device_mesh", None)
        if device_mesh is None:
            return  # not FSDP2 — nothing to shard

        from .utils.fsdp2 import default_parallelize_fn, describe_fsdp_strategy

        log_header("FSDP2 configure_model")
        # Verbose summary of the resolved sharding config — invaluable when
        # debugging "is it actually sharding, and over how many ranks?".
        logging.info(f"  fsdp2 strategy: {describe_fsdp_strategy(self.trainer)}")
        if self._parallelize_fn is None:
            # Default path: forward the strategy's mixed-precision policy.
            mp_policy = getattr(self.trainer.strategy, "_spt_mp_policy", None)
            default_parallelize_fn(self, device_mesh, mp_policy=mp_policy)
        else:
            logging.info("  fsdp2: using user-provided parallelize_fn")
            self._parallelize_fn(self, device_mesh)

    def named_parameters(
        self,
        with_callbacks=True,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        """Override to globally exclude callback-related parameters.

        Excludes parameters that belong to ``self.callbacks_modules`` or ``self.callbacks_metrics``.
        This prevents accidental optimization of callback/metric internals, even if external code
        calls ``self.parameters()`` or ``self.named_parameters()`` directly.

        Args:
            with_callbacks (bool, optional): If False, excludes callback parameters. Defaults to True.
            prefix (str, optional): Prefix to prepend to parameter names. Defaults to "".
            recurse (bool, optional): If True, yields parameters of this module and all submodules.
                If False, yields only direct parameters. Defaults to True.
            remove_duplicate (bool, optional): Whether to deduplicate shared parameters.
                Must be accepted and forwarded because PyTorch's ``fully_shard`` (FSDP2)
                wrap path calls ``named_parameters(remove_duplicate=False)``; without it
                this override would raise ``TypeError`` before sharding. Defaults to True.

        Yields:
            tuple[str, torch.nn.Parameter]: Name and parameter pairs.
        """
        if with_callbacks and not Module._warned_named_parameters:
            Module._warned_named_parameters = True
            logging.warning(
                "! You are calling self.parameters which also gives callbacks "
                "parameters, to remove them, pass `with_callbacks=False`"
            )
        for name, param in super().named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        ):
            is_callback = name.startswith("callbacks_")
            if is_callback and not with_callbacks:
                continue
            yield name, param

    def parameters(self, with_callbacks=True, recurse: bool = True):
        """Override to route through the filtered ``named_parameters`` implementation.

        Args:
            with_callbacks (bool, optional): If False, excludes callback parameters. Defaults to True.
            recurse (bool, optional): If True, yields parameters of this module and all submodules.
                If False, yields only direct parameters. Defaults to True.

        Yields:
            torch.nn.Parameter: Module parameters.
        """
        for _, param in self.named_parameters(with_callbacks, recurse=recurse):
            yield param

    def after_manual_backward(self):
        """Hook called immediately after ``manual_backward`` in ``training_step``.

        Override in a subclass to insert logic that must run after gradients are
        computed but before any optimizer step or ``zero_grad`` — for example,
        gradient norm logging, custom gradient clipping, or EMA teacher weight
        updates that depend on the current gradient. The default implementation
        does nothing.
        """
        pass

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply the active GPU-side batch transform after Lightning moves the batch.

        Resolution order (first match wins):

        1. ``dataset.gpu_transform`` — set on the dataset behind the active
           DataLoader. **Recommended:** pair augmentation with the dataset
           so train/val/test/predict each carry their own spec naturally.
        2. ``self.trainer.datamodule.gpu_transform`` — set on the DataModule.
           May be a callable or a ``{"train": ..., "val": ...}`` dict.

        Setting ``self.gpu_transform`` on the Module is **not** supported
        and is rejected at ``on_train_start``; attach it to the dataset
        (or the DataModule for third-party datasets) instead.

        Lazy device placement: when the resolved transform is an
        ``nn.Module`` (e.g. a :class:`GPUCompose`), it is moved to
        ``self.device`` on first use. Buffers (e.g. ``GPUNormalize``'s
        mean/std) therefore live on the correct GPU under DDP without
        manual wiring.

        When nothing resolves, this is a zero-cost passthrough.

        Args:
            batch: Batch dict already moved to ``self.device`` by Lightning.
            dataloader_idx: Index of the dataloader that produced this batch.

        Returns:
            The (possibly augmented) batch dict.
        """
        gpu_transform = self._resolve_gpu_transform(dataloader_idx)
        if gpu_transform is None:
            return batch
        return gpu_transform(batch)

    def _resolve_gpu_transform(self, dataloader_idx: int = 0):
        """Look up the active GPU transform from the dataset, then the DataModule."""
        # Access ``_trainer`` directly: the ``trainer`` property raises
        # when the Module is not attached, which is fine for direct
        # calls outside ``Trainer.fit`` (e.g. unit tests).
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return None
        stage = self._current_stage(trainer)
        # 1. Dataset-level — preferred placement.
        gt = _gpu_transform_from_current_dataset(trainer, stage, dataloader_idx)
        # 2. DataModule-level fallback (callable or stage-keyed dict).
        if gt is None:
            dm = getattr(trainer, "datamodule", None)
            if dm is not None:
                gt = getattr(dm, "gpu_transform", None)
                if isinstance(gt, dict):
                    gt = gt.get(stage)
        if gt is None:
            return None
        # Lazily move to the model's device so users don't have to .cuda() it.
        if isinstance(gt, torch.nn.Module):
            ref = next(gt.parameters(), None) or next(gt.buffers(), None)
            if ref is None or ref.device != self.device:
                gt.to(self.device)
        return gt

    @staticmethod
    def _current_stage(trainer):
        if trainer is None:
            return None
        if trainer.training:
            return "train"
        if trainer.validating or trainer.sanity_checking:
            return "val"
        if trainer.testing:
            return "test"
        if trainer.predicting:
            return "predict"
        return None

    def training_step(self, batch, batch_idx):
        """Run one training step with manual optimization across all configured optimizers.

        Calls ``forward(batch, stage="fit")`` to obtain a ``state`` dict, then performs
        a single ``manual_backward`` on ``state["loss"]``. Each optimizer steps only when
        its frequency boundary is met (``(batch_idx + 1) % frequency == 0``). Gradient
        clipping is applied per-optimizer using either the per-optimizer override or the
        Trainer's ``gradient_clip_val``. Learning rate is logged as ``hparams/lr_{name}``
        after each step. ``zero_grad`` is called only on optimizers that actually stepped
        this iteration.

        Args:
            batch: Input batch dict from the training dataloader. Must be a ``dict`` —
                a non-dict batch raises ``ValueError``.
            batch_idx: Index of the current batch within the epoch. Injected into the
                batch dict as ``batch["batch_idx"]`` before forwarding.

        Returns:
            dict: The ``state`` dict returned by ``forward``, passed unchanged to
                Lightning's callback hooks (``on_train_batch_end``).
        """
        if type(batch) is not dict:
            msg = f"! batch is expected to be a dict! Not as {type(batch)}"
            logging.warning(msg)
            raise ValueError(msg)
        batch["batch_idx"] = batch_idx
        state = self(batch, stage="fit")

        # Resolve optimizers and schedulers (can be single or list)
        optimizers = self.optimizers()
        # there are NO optimizers either from main or callbacks, no need to stay here!
        if isinstance(optimizers, pl.pytorch.core.optimizer._MockOptimizer):
            return state
        elif not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        schedulers = self.lr_schedulers()
        if schedulers is None:
            schedulers = []
        elif not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]

        if len(optimizers) > 1 and (len(optimizers) != len(schedulers)):
            raise ValueError(
                "When using more than one optimizer,"
                " we need as many schedulers as optimizers!"
                "if you don't want to use one, either use a "
                "ConstantLR, or return None"
            )
        elif len(optimizers) == 1 and len(schedulers) == 0:
            schedulers = [None]

        # Compute gradients once for the joint loss
        self.manual_backward(state["loss"])
        self.after_manual_backward()

        zero_grad_opts = []
        # Designate exactly ONE optimizer per batch as the
        # ``global_step`` ticker, so ``trainer.global_step`` always equals
        # the number of training batches — regardless of how many
        # optimizers are attached (main + callbacks). Preference:
        # first main optimizer; fallback: first callback optimizer
        # (covers ``Module(optim=None)`` setups where the only
        # optimizers come from ``TrainableCallback``s like
        # ``OnlineImageDecoder`` — without this fallback,
        # ``trainer.max_steps`` becomes unreachable and ``fit`` loops
        # forever).
        ticker_idx = self._pick_global_step_ticker(optimizers)
        # Stepping and gradient clipping at accumulation boundary
        for idx, opt in enumerate(optimizers):
            name = self._optimizer_index_to_name[idx]
            # Honor per-optimizer frequency if available
            freq = self._optimizer_frequencies[name]
            if (batch_idx + 1) % freq != 0:
                continue

            # Gradient accumulation: ``manual_backward`` was called every
            # micro-batch with no loss rescaling, so this optimizer's params
            # hold the SUM of ``freq`` mean-reduction gradients. Average them
            # by ``1/freq`` so the step matches a single full-(effective-)batch
            # update — i.e. (batch=B/freq, freq) == (batch=B, freq=1) with
            # seeded data. Done per-optimizer, BEFORE clipping, so optimizers
            # with different frequencies each average over their own window.
            # No-op when ``freq == 1`` (the common, non-accumulating case).
            if freq > 1:
                inner = opt.optimizer if isinstance(opt, LightningOptimizer) else opt
                inv = 1.0 / freq
                for group in inner.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            p.grad.mul_(inv)

            clip_val = self._optimizer_gradient_clip_val[name]
            clip_algo = self._optimizer_gradient_clip_algorithm[name]
            if clip_val is not None:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=clip_val,
                    gradient_clip_algorithm=clip_algo,
                )

            if not isinstance(opt, LightningOptimizer):
                msg = (
                    "We received an optimizer that is not wrapped"
                    "by lightning, make sure you define all your optimizers"
                    f"in the configure_optimizers method! {opt}"
                )
                logging.error(msg)
                raise ValueError(msg)
            # The ticker optimizer goes through ``LightningOptimizer.step()``
            # as normal so its single tick advances ``global_step``. Every
            # other optimizer (main or callback) keeps all of the strategy
            # + precision-plugin machinery intact (AMP GradScaler.step,
            # FP8 TransformerEngine calibration, DDP grad-sync, profiler
            # timing) but suppresses the manual-opt progress counter via
            # no-op hooks. The progress hooks are attached by
            # ``lightning.pytorch.loops.optimization.manual.ManualOptimization``
            # at batch start and reset at batch end; swapping them to
            # no-ops for one call is reversible and side-effect-free.
            if idx == ticker_idx:
                opt.step()
            else:
                saved_before, saved_after = opt._on_before_step, opt._on_after_step
                opt._on_before_step = _noop_hook
                opt._on_after_step = _noop_hook
                try:
                    opt.step()
                finally:
                    opt._on_before_step = saved_before
                    opt._on_after_step = saved_after
            zero_grad_opts.append(opt)
            # Step its scheduler if it exists
            if schedulers[idx] is not None:
                schedulers[idx].step()

            # Log learning rate for each optimizer
            lr = (
                opt.optimizer.param_groups[0]["lr"]
                if isinstance(opt, LightningOptimizer)
                else opt.param_groups[0]["lr"]
            )
            _spt_log(f"hparams/lr_{name}", lr, on_step=True, on_epoch=False)

        # zero grad what's needed
        for opt in zero_grad_opts:
            opt.zero_grad(set_to_none=True)
        return state

    def _pick_global_step_ticker(self, optimizers) -> int:
        """Return the index of the optimizer that should advance ``global_step``.

        Exactly one optimizer per batch ticks the manual-opt progress
        counter. Preference order:

        1. First main optimizer (i.e. not registered as a callback opt).
        2. First optimizer overall — used when ``Module(optim=None)`` and
           the only optimizers came from ``TrainableCallback`` callbacks.
           Without this fallback the counter never advances, so
           ``Trainer(max_steps=N)`` is unreachable and ``fit`` loops
           forever (the regression that motivated this method).

        Returns 0 when ``optimizers`` is empty — Lightning's mock-opt
        path handles the no-optimizer case earlier so this method
        isn't reached in that branch.
        """
        if not optimizers:
            return 0
        for idx in range(len(optimizers)):
            name = self._optimizer_index_to_name.get(idx)
            if name is not None and name not in self._callback_optimizer_names:
                return idx
        return 0  # only callback optimizers — promote the first

    def on_train_start(self):
        """Validate and log the optimizer configuration at the start of training.

        Runs once before the first training step. Fills in any missing per-optimizer
        metadata (gradient clip value, clip algorithm, step frequency) by falling back
        to the Trainer's global settings. Logs a summary table of optimizer index, name,
        class, clip value, and clip algorithm so misconfigured setups are caught early
        rather than silently misbehaving mid-run.
        """
        # Refuse Module-level ``gpu_transform``. Two-ways-to-do-one-thing
        # is a footgun: if a user sets it here AND on the dataset, the
        # dataset's silently loses. Make it loud.
        if getattr(self, "gpu_transform", None) is not None:
            raise RuntimeError(
                "Setting `gpu_transform` on the Module is not supported. "
                "Attach it to the dataset (recommended: pass "
                "`gpu_transform=...` to the dataset constructor) or to the "
                "DataModule via `gpu_transform=...` / `gpu_transform={'train': ..., 'val': ...}`."
            )

        log_header("Optimizers")
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        logging.info(f"  self.optimizers() gave us {len(optimizers)} optimizers")
        for i in range(len(optimizers)):
            # check if optimizer i is named and well setup
            if i not in self._optimizer_index_to_name:
                name = f"default_{i}"
                self._optimizer_index_to_name[i] = name
            name = self._optimizer_index_to_name[i]
            if name not in self._optimizer_gradient_clip_val:
                logging.warning(f"! No clip val found for optimizer {name}")
                clip_val = getattr(
                    self.trainer, "gradient_clip_val_", self.trainer.gradient_clip_val
                )
                logging.warning(f"! Will use the Trainer's value of {clip_val}")
                self._optimizer_gradient_clip_val[name] = clip_val
            if name not in self._optimizer_gradient_clip_algorithm:
                logging.warning(f"! No clip algorithm found for optimizer {name}")
                clip_algo = getattr(
                    self.trainer,
                    "gradient_clip_algorithm_",
                    self.trainer.gradient_clip_algorithm,
                )
                logging.warning(f"! Will use the Trainer's value of {clip_algo}")
                self._optimizer_gradient_clip_algorithm[name] = clip_algo
            if name not in self._optimizer_frequencies:
                freq = getattr(self.trainer, "accumulate_grad_batches", 1)
                freq = getattr(self.trainer, "accumulate_grad_batches_", freq)
                freq = max(int(freq), 1)
                # config priority — guard against ``self.optim is None``
                # (eval-only ``Module(optim=None)`` setup) and against
                # ``self.optim`` being a single per-optimizer config dict
                # (no ``.get("frequency", ...)`` semantics there).
                optim_cfg = getattr(self, "optim", None) or {}
                if isinstance(optim_cfg, dict):
                    freq = optim_cfg.get("frequency", freq)
                self._optimizer_frequencies[name] = int(freq)

        table = PrettyTable()
        # 2. Define the column headers.
        table.field_names = ["Opt. Index", "Opt. name", "opt", "clip val.", "clip alg."]
        for i in range(len(optimizers)):
            name = self._optimizer_index_to_name[i]
            row = [str(i), name, type(optimizers[i]).__name__]
            row.append(str(self._optimizer_gradient_clip_val[name]))
            row.append(str(self._optimizer_gradient_clip_algorithm[name]))
            table.add_row(row)
        logging.success("✓ Optimizer check complete:\n{}", table)

    def validation_step(self, batch, batch_idx):
        """Run the forward pass for a single validation batch.

        Calls ``forward(batch, stage="validate")`` with gradients disabled (Lightning
        handles ``torch.no_grad()``). The returned dict is passed to every registered
        callback via ``on_validation_batch_end``, making all keys — including
        ``"embedding"`` and ``"label"`` — available to ``OnlineProbe``, ``OnlineKNN``,
        ``RankMe``, and similar evaluation callbacks without any extra wiring.

        Args:
            batch: Input batch dict from the validation dataloader.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            dict: Output dict returned by ``forward``.
        """
        batch["batch_idx"] = batch_idx
        # Route through ``__call__`` (not ``.forward``) so FSDP2's all-gather
        # pre-forward hooks fire — a direct ``.forward()`` would leave sharded
        # DTensor params un-gathered. Harmless under DDP/single-device.
        return self(batch, stage="validate")

    def test_step(self, batch, batch_idx):
        """Run the forward pass for a single test batch.

        Mirrors ``validation_step`` but passes ``stage="test"`` to ``forward``, allowing
        forward functions to distinguish test-time behaviour if needed. The returned dict
        is forwarded to Lightning's ``on_test_batch_end`` callback hooks.

        Args:
            batch: Input batch dict from the test dataloader.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            dict: Output dict returned by ``forward``.
        """
        batch["batch_idx"] = batch_idx
        return self(batch, stage="test")

    def predict_step(self, batch, batch_idx):
        """Run the forward pass for a single prediction batch.

        Passes ``stage="predict"`` to ``forward`` so forward functions can omit loss
        computation and return only inference outputs (e.g., embeddings). Used by
        ``Trainer.predict()`` for large-scale feature extraction without a label set.

        Args:
            batch: Input batch dict from the prediction dataloader.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            dict: Output dict returned by ``forward``.
        """
        batch["batch_idx"] = batch_idx
        return self(batch, stage="predict")

    def _get_scheduler_name(self, scheduler_config, scheduler_instance=None):
        """Extract scheduler name from various config formats.

        Args:
            scheduler_config: Scheduler configuration (str, dict, partial, or class).
            scheduler_instance (optional): Instantiated scheduler instance. Defaults to None.

        Returns:
            str: Name of the scheduler.
        """
        if isinstance(scheduler_config, str):
            return scheduler_config
        elif isinstance(scheduler_config, dict):
            return scheduler_config.get("type", "CosineAnnealingLR")
        elif hasattr(scheduler_config, "func"):  # partial
            return scheduler_config.func.__name__
        elif scheduler_instance:
            return scheduler_instance.__class__.__name__
        else:
            return "Unknown"

    def _build_scheduler_config(self, scheduler, config, name=None):
        """Build scheduler config dict for Lightning.

        Args:
            scheduler: The instantiated scheduler.
            config (dict): Configuration dict containing interval, frequency, etc.
            name (str, optional): Name for the scheduler. Defaults to None.

        Returns:
            dict: Scheduler configuration dict compatible with Lightning.
        """
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": config.get("interval", "step"),
            "frequency": config.get("scheduler_frequency", config.get("frequency", 1)),
        }

        if name:
            scheduler_dict["name"] = name

        # Add monitor for ReduceLROnPlateau
        scheduler_cfg = config.get("scheduler", "CosineAnnealingLR")
        scheduler_name = self._get_scheduler_name(scheduler_cfg, scheduler)
        if scheduler_name == "ReduceLROnPlateau":
            # Prefer nested monitor inside scheduler dict, fallback to top-level
            nested_monitor = None
            if isinstance(scheduler_cfg, dict):
                nested_monitor = scheduler_cfg.get("monitor")
            scheduler_dict["monitor"] = nested_monitor or config.get(
                "monitor", "val_loss"
            )

        return scheduler_dict

    def _collect_parameters_by_optimizer_groups(self, optim_items):
        """Assign modules and collect parameters per optimizer group defined by regex.

        Args:
            optim_items: list of (name, config) where config contains a "modules" regex
                describing group membership.

        Returns:
            params_by_name: dict[name, List[nn.Parameter]]
            named_params_by_name: dict[name, List[Tuple[str, nn.Parameter]]]
            modules_by_name: dict[name, List[str]]
        """
        # Pre-compile regex with stable order from optim_items
        compiled = [
            (name, re.compile(config["modules"])) for name, config in optim_items
        ]

        # Initialize containers
        params_by_name = {name: [] for name, _ in compiled}
        named_params_by_name = {name: [] for name, _ in compiled}
        modules_by_name = {name: [] for name, _ in compiled}

        # Map module -> group index with inheritance
        module_to_group = {}
        for qual_name, module in self.named_modules():
            if "callbacks_modules" in qual_name or "callbacks_metrics" in qual_name:
                continue

            # inherit parent's group if any
            if "." in qual_name:
                parent_name = qual_name.rsplit(".", 1)[0]
                group_idx = module_to_group.get(parent_name)
            else:
                group_idx = None

            # override if explicit match
            for idx, (_, regex) in enumerate(compiled):
                if regex.match(qual_name):
                    group_idx = idx
                    break

            module_to_group[qual_name] = group_idx

            if group_idx is not None:
                group_name = compiled[group_idx][0]
                # record module name
                modules_by_name[group_name].append(qual_name)
                # collect direct parameters only to avoid duplication
                direct_params = list(module.parameters(recurse=False))
                if direct_params:
                    params_by_name[group_name].extend(direct_params)
                # Also collect named parameters for exclude_bias_norm support
                direct_named_params = list(module.named_parameters(recurse=False))
                if direct_named_params:
                    # Prefix with module's qualified name
                    prefixed = [
                        (f"{qual_name}.{pname}" if qual_name else pname, p)
                        for pname, p in direct_named_params
                    ]
                    named_params_by_name[group_name].extend(prefixed)

        # Logging summary
        rows = []
        for group_name, config in optim_items:
            pattern = config.get("modules", "")
            tensors = params_by_name[group_name]
            num_tensors = len(tensors)
            num_elements = sum(int(p.numel()) for p in tensors)
            num_requires_grad = sum(int(p.requires_grad) for p in tensors)
            rows.append(
                [
                    group_name,
                    pattern,
                    len(modules_by_name[group_name]),
                    num_tensors,
                    num_elements,
                    num_requires_grad,
                ]
            )

        if rows:
            headers = [
                "Optimizer",
                "Pattern",
                "Matched Modules",
                "Param Tensors",
                "Total Params",
                "RequiresGrad Tensors",
            ]
            logging.info(
                "\n" + tabulate(rows, headers=headers, tablefmt="heavy_outline")
            )

        return params_by_name, named_params_by_name, modules_by_name

    def configure_optimizers(self):
        """Configure optimizers and schedulers for manual optimization.

        Returns:
            dict or tuple: Optimizer configuration with optional learning rate scheduler.
            For single optimizer: Returns a dict with optimizer and lr_scheduler.
            For multiple optimizers: Returns a tuple of (optimizers, schedulers).

        Example:
            Multi-optimizer configuration with module pattern matching and schedulers:

            >>> # Simple single optimizer with scheduler
            >>> self.optim = {
            ...     "optimizer": partial(torch.optim.AdamW, lr=1e-3),
            ...     "scheduler": "CosineAnnealingLR",  # Uses smart defaults
            ...     "interval": "step",
            ...     "frequency": 1,
            ... }

            >>> # Multi-optimizer with custom scheduler configs
            >>> self.optim = {
            ...     "encoder_opt": {
            ...         "modules": "encoder",  # Matches 'encoder' and all children
            ...         "optimizer": {"type": "AdamW", "lr": 1e-3},
            ...         "scheduler": {
            ...             "type": "OneCycleLR",
            ...             "max_lr": 1e-3,
            ...             "total_steps": 10000,
            ...         },
            ...         "interval": "step",
            ...         "frequency": 1,
            ...     },
            ...     "head_opt": {
            ...         "modules": ".*head$",  # Matches modules ending with 'head'
            ...         "optimizer": "SGD",
            ...         "scheduler": {
            ...             "type": "ReduceLROnPlateau",
            ...             "mode": "max",
            ...             "patience": 5,
            ...             "factor": 0.5,
            ...         },
            ...         "monitor": "val_accuracy",  # Required for ReduceLROnPlateau
            ...         "interval": "epoch",
            ...         "frequency": 2,
            ...     },
            ... }

            With model structure:
            - encoder                 -> encoder_opt (matches "encoder")
            - encoder.layer1          -> encoder_opt (inherits from parent)
            - encoder.layer1.conv     -> encoder_opt (inherits from encoder.layer1)
            - classifier_head         -> head_opt (matches ".*head$")
            - classifier_head.linear  -> head_opt (inherits from parent)
            - decoder                 -> None (no match, no parameters collected)
        """
        log_header("Optimizers")

        # Early exit for disabled optimization
        if hasattr(self, "optim") and not self.optim:
            logging.info("  Optimization disabled - skipping optimizer configuration.")
            return None

        if not hasattr(self, "optim"):
            logging.info(
                "  Using default optimization setup: AdamW optimizer with CosineAnnealingLR scheduler."
            )
            self.optim = dict(optimizer=partial(torch.optim.AdamW))
        elif isinstance(self.optim, partial):
            logging.info("  Using user's partial optimizer.")
            self.optim = dict(optimizer=self.optim)

        # Single optimizer case
        optimizer_cfg = self.optim.get("optimizer")
        if isinstance(optimizer_cfg, (str, dict, DictConfig)) or hasattr(
            optimizer_cfg, "__call__"
        ):
            logging.info("  Configuring single optimizer.")

            # Direct parameter extraction - use globally filtered parameters
            params = list(self.parameters(with_callbacks=False))

            # Pass named_params for exclude_bias_norm support
            named_params = list(self.named_parameters(with_callbacks=False))
            opt = create_optimizer(params, optimizer_cfg, named_params=named_params)

            # Create scheduler
            default = dict(
                type="CosineAnnealingLR", T_max=self.trainer.estimated_stepping_batches
            )
            sched_config = self.optim.get("scheduler", default)
            sched = create_scheduler(opt, sched_config, module=self)
            sched_name = self._get_scheduler_name(sched_config, sched)

            logging.info(
                f"  Configured {opt.__class__.__name__} optimizer with {sched_name} scheduler."
            )

            # Build scheduler config dict for Lightning
            scheduler_dict = self._build_scheduler_config(sched, self.optim)

            # Return in list/dict style compatible with lr_schedulers() access
            return [opt], [scheduler_dict]

        # Multiple optimizers case - check once
        if not isinstance(self.optim, (dict, DictConfig)):
            raise ValueError(
                "Optimizer must be either a partial function or a dict of optimizer configs"
            )

        # Verify all values are dicts
        optim_items = list(self.optim.items())
        if not all(isinstance(v, (dict, DictConfig)) for _, v in optim_items):
            raise ValueError("For multiple optimizers, all config values must be dicts")

        logging.info(
            f"  Optimizer specified by Dict with keys {[k for k, _ in optim_items]}"
        )

        # DeepSpeed's ZeRO engine owns exactly one optimizer; multiple optimizers
        # (online probes, teacher-student EMA, ...) are unsupported. Fail early
        # with a clear redirect to FSDP2 rather than the opaque downstream
        # Lightning ``MisconfigurationException`` raised deep in DeepSpeed init.
        if len(optim_items) > 1:
            from .utils.fsdp2 import is_deepspeed_strategy

            if is_deepspeed_strategy(getattr(self, "_trainer", None)):
                raise RuntimeError(
                    f"DeepSpeed supports a single optimizer only, but this run "
                    f"configures {len(optim_items)} ({[k for k, _ in optim_items]}). "
                    "Online probes (OnlineProbe/OnlineKNN/...) and teacher-student "
                    "EMA each add their own optimizer and are incompatible with "
                    "DeepSpeed. Use strategy='fsdp2' instead — it supports "
                    "multi-optimizer / EMA / probe methods out of the box."
                )

        # Build grouping with detailed logging
        params_by_name, named_params_by_name, modules_by_name = (
            self._collect_parameters_by_optimizer_groups(optim_items)
        )

        # Build optimizers and schedulers
        optimizers = []
        schedulers = []

        for name, config in optim_items:
            params = params_by_name.get(name, [])
            if not params:
                logging.warning(f"! No parameters matched for optimizer {name}")
                # skip registration when there are no parameters
                continue

            # Pass named_params for exclude_bias_norm support
            named_params = named_params_by_name.get(name, [])
            opt = create_optimizer(
                params, config["optimizer"], named_params=named_params
            )
            optimizers.append(opt)

            sched_config = config.get("scheduler", "CosineAnnealingLR")
            scheduler = create_scheduler(opt, sched_config, module=self)
            sched_name = self._get_scheduler_name(sched_config, scheduler)

            # Build scheduler config dict for Lightning
            scheduler_dict = self._build_scheduler_config(scheduler, config, name)
            schedulers.append(scheduler_dict)

            logging.info(
                f"  Configured optimizer '{name}' (modules={len(modules_by_name.get(name, []))}, "
                f"param_tensors={len(params)}, total_params={sum(int(p.numel()) for p in params)}) "
                f"with {sched_name} scheduler."
            )

            # Track names and frequencies aligned to optimizer order. The
            # index→name mapping is essential: ``training_step`` looks up the
            # per-optimizer frequency (and clip settings) by
            # ``_optimizer_index_to_name[idx]``. Without it, ``on_train_start``
            # falls back to positional ``default_{i}`` names with frequency 1,
            # silently ignoring each optimizer's configured ``frequency`` (so
            # gradient accumulation never engaged for named/multi-optimizer
            # configs).
            self._optimizer_index_to_name[len(optimizers) - 1] = name
            self._optimizer_frequencies[name] = int(config.get("frequency", 1))

        return optimizers, schedulers
