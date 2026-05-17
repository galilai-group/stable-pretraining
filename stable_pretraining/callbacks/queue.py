"""Queue callback with unified size management and insertion order preservation.

This module provides a queue callback that uses OrderedQueue to maintain
insertion order and implements intelligent queue sharing when multiple callbacks
request the same data with different queue sizes.
"""

from typing import Dict, Optional, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from .registry import log as _spt_log

from .queues import OrderedQueue
from .utils import get_data_from_batch_or_outputs, log_header


class OnlineQueue(Callback):
    """Circular buffer callback with insertion order preservation and size unification.

    This callback maintains an OrderedQueue that accumulates data from specified batch
    keys during training while preserving insertion order. It implements intelligent
    queue sharing: when multiple callbacks request the same data with different sizes,
    it uses a single queue with the maximum size and serves appropriate subsets.

    Note:
        ``OnlineKNN``, ``RankMe``, and similar consumers auto-create their
        own ``OnlineQueue`` via ``find_or_create_queue_callback``, so users
        rarely need to register one explicitly. Add one manually only when
        you need a non-default queue length, a shared queue across multiple
        consumers, or specific ``gather_distributed`` semantics.

    Key features:
    - Maintains insertion order using OrderedQueue
    - Unified storage: one queue per key, shared across different size requests
    - Memory-efficient: no duplicate storage for same data
    - Size-based retrieval: each consumer gets exactly the amount they need

    Args:
        key: The batch key whose tensor values will be queued at every training step.
        queue_length: Number of elements this callback needs from the queue.
        dim: Pre-allocate buffer with this shape. Can be int or tuple.
        dtype: Pre-allocate buffer with this dtype.
        gather_distributed: If True, gather queue data across all processes.

    Attributes:
        data: Property returning the requested number of most recent samples.
        actual_queue_length: The actual size of the underlying shared queue.
    """

    # Class-level registry to track shared queues by key.
    #
    # Why class-level: multiple OnlineQueue instances created with the same key
    # share a single underlying OrderedQueue, so the registry must outlive any
    # individual instance. ``_owner_trainer_id`` records which trainer populated
    # the registry; if a different trainer's ``setup`` is observed, the stale
    # state is wiped before the new run begins (issue #378).
    _shared_queues: Dict[str, "OrderedQueue"] = {}
    _queue_info: Dict[str, dict] = {}  # Track max size and other info per key
    _owner_trainer_id: Optional[int] = None

    def __init__(
        self,
        key: str,
        queue_length: int,
        dim: Optional[Union[int, tuple]] = None,
        dtype: Optional[torch.dtype] = None,
        gather_distributed: bool = False,
        verbose: bool = None,
    ) -> None:
        super().__init__()

        self.key = key
        self.requested_length = queue_length  # What this callback wants
        self.dim = dim
        self.dtype = dtype
        self.gather_distributed = gather_distributed
        from .utils import resolve_verbose

        self.verbose = resolve_verbose(verbose)
        self._snapshot = None

        log_header("OnlineQueue")
        logging.info(f"  key: {key}")
        logging.info(f"  requested_length: {queue_length}")
        logging.info(f"  dim: {dim}")
        logging.info(f"  dtype: {dtype}")

    @property
    def actual_queue_length(self) -> int:
        """Get the actual size of the underlying shared queue."""
        if self.key in self._queue_info:
            return self._queue_info[self.key]["max_length"]
        return self.requested_length

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Initialize or connect to shared queue during setup phase."""
        # Wipe class-level state if it came from a different trainer (#378).
        OnlineQueue._maybe_wipe_stale_state(trainer)

        # Resolve target device once. ``pl_module.device`` is set by Lightning's
        # accelerator setup before ``setup()`` runs; we resolve it defensively
        # so plain ``LightningModule()`` instances (e.g. in unit tests with no
        # parameters) don't crash.
        target_device = self._resolve_module_device(pl_module)

        # Update the maximum queue length for this key
        if self.key not in self._queue_info:
            self._queue_info[self.key] = {
                "max_length": self.requested_length,
                "dim": self.dim,
                "dtype": self.dtype,
                "callbacks": [self],
            }
            logging.info(
                f"  new key '{self.key}' with initial size {self.requested_length}"
            )
        else:
            # Update max length if this callback needs more
            old_max = self._queue_info[self.key]["max_length"]
            if self.requested_length > old_max:
                self._queue_info[self.key]["max_length"] = self.requested_length
                logging.info(
                    f"  increased max size for key '{self.key}' "
                    f"from {old_max} to {self.requested_length}"
                )

            # Add this callback to the list
            if self not in self._queue_info[self.key]["callbacks"]:
                self._queue_info[self.key]["callbacks"].append(self)

        # Create or update the shared queue
        max_length = self._queue_info[self.key]["max_length"]

        if self.key not in self._shared_queues:
            # Create new shared queue on the target device (#379). Lightning's
            # ``pl_module.to(device)`` already ran before ``setup``; children
            # added afterwards are NOT moved, so we must place this one
            # explicitly or the buffer stays on CPU and every
            # ``OnlineKNN`` validation batch pays a CPU→GPU transfer.
            queue = OrderedQueue(max_length, self.dim, self.dtype)
            if target_device is not None:
                queue = queue.to(target_device)
            self._shared_queues[self.key] = queue
            # Register in callbacks_modules for consistency
            queue_key = f"ordered_queue_{self.key}"
            pl_module.callbacks_modules[queue_key] = queue
            logging.info(
                f"  created shared queue for '{self.key}' with size {max_length} "
                f"on device {queue.pointer.device}"
            )
        elif self._shared_queues[self.key].max_length < max_length:
            # Need to resize the existing queue
            old_queue = self._shared_queues[self.key]
            old_data = (
                old_queue.get() if old_queue.pointer > 0 or old_queue.filled else None
            )

            # Create new larger queue on the same device as the old one (or
            # ``target_device`` as fallback). Inheriting the old device keeps
            # any accumulated training data in place.
            new_queue = OrderedQueue(max_length, self.dim, self.dtype)
            resize_device = old_queue.pointer.device
            if target_device is not None and resize_device != target_device:
                # If module was moved between resizes, follow it.
                resize_device = target_device
            new_queue = new_queue.to(resize_device)

            # Copy old data if exists
            if old_data is not None and len(old_data) > 0:
                new_queue.append(old_data)
                logging.info(
                    f"  resized queue for '{self.key}' from "
                    f"{old_queue.max_length} to {max_length} "
                    f"on device {resize_device}, "
                    f"preserved {len(old_data)} items"
                )
            else:
                logging.info(
                    f"  resized empty queue for '{self.key}' from "
                    f"{old_queue.max_length} to {max_length} on device {resize_device}"
                )

            # Replace the queue
            self._shared_queues[self.key] = new_queue
            queue_key = f"ordered_queue_{self.key}"
            pl_module.callbacks_modules[queue_key] = new_queue

    @staticmethod
    def _maybe_wipe_stale_state(trainer: Trainer) -> None:
        """Clear class-level state if it belongs to a different trainer (#378).

        Called from both ``setup()`` and ``find_or_create_queue_callback`` so
        that whichever entry point fires first detects the new run and wipes
        cleanly. Idempotent within a single trainer's lifecycle.
        """
        trainer_id = id(trainer)
        if (
            OnlineQueue._owner_trainer_id is not None
            and OnlineQueue._owner_trainer_id != trainer_id
        ):
            n_stale = len(OnlineQueue._queue_info)
            stale_keys = sorted(OnlineQueue._queue_info.keys())
            OnlineQueue._shared_queues.clear()
            OnlineQueue._queue_info.clear()
            logging.info(
                f"OnlineQueue: detected new trainer, cleared {n_stale} stale "
                f"queue(s): {stale_keys}"
            )
        OnlineQueue._owner_trainer_id = trainer_id

    @staticmethod
    def _resolve_module_device(pl_module: LightningModule) -> Optional[torch.device]:
        """Return ``pl_module.device`` if it looks usable, else ``None``.

        ``LightningModule.device`` is a property that reads ``self._device``;
        for plain ``LightningModule()`` instances with no parameters this can
        be unset or absent. We fall back to ``None`` (caller leaves the queue
        on its default CPU placement) instead of crashing.
        """
        device = getattr(pl_module, "device", None)
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            try:
                return torch.device(device)
            except (RuntimeError, TypeError):
                return None
        return None

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Drop the snapshot reference so it can be garbage-collected.

        Class-level ``_shared_queues`` / ``_queue_info`` are NOT cleared here:
        ``Trainer.validate()`` or ``Trainer.test()`` after a ``fit()`` may
        legitimately consume the queue. The stale-state guard in ``setup()``
        handles cross-trainer cleanup instead (#378).
        """
        self._snapshot = None

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        """Append batch data to the shared queue."""
        # Only the first callback for each key should append data
        # Check if we're the first callback for this key
        if self._queue_info[self.key]["callbacks"][0] is not self:
            return  # Let the first callback handle appending

        with torch.no_grad():
            data = get_data_from_batch_or_outputs(
                self.key, batch, outputs, caller_name="OnlineQueue"
            )
            if data is None:
                return

            # If dim is specified as a single int and data is 1D, add a dimension
            if isinstance(self.dim, int) and data.dim() == 1:
                data = data.unsqueeze(1)

            # Device-mismatch guard (#379). If setup() placed the queue on the
            # wrong device — or the module was moved after setup — surface it
            # the first time it happens instead of silently copying every
            # batch. Move the queue to the data's device so subsequent appends
            # stay on-device.
            queue = self._shared_queues[self.key]
            if queue.pointer.device != data.device:
                logging.warning(
                    f"OnlineQueue[{self.key}]: device mismatch "
                    f"(queue on {queue.pointer.device}, data on {data.device}); "
                    f"moving queue to {data.device}. This typically indicates "
                    f"the module was moved after setup()."
                )
                queue = queue.to(data.device)
                self._shared_queues[self.key] = queue
                queue_key = f"ordered_queue_{self.key}"
                if hasattr(pl_module, "callbacks_modules"):
                    pl_module.callbacks_modules[queue_key] = queue

            # Append to the shared queue
            queue.append(data)

            if self.verbose:
                queue = self._shared_queues[self.key]
                n_items = (
                    queue.max_length if queue.filled else int(queue.pointer.item())
                )
                fill = n_items / queue.max_length if queue.max_length > 0 else 0.0
                _spt_log(
                    f"queue/{self.key}_fill_pct",
                    fill,
                    on_step=True,
                    on_epoch=False,
                )

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Create snapshot of the requested portion of queue contents."""
        queue = self._shared_queues[self.key]
        logging.debug(
            f"  creating snapshot for key '{self.key}' "
            f"(requesting {self.requested_length} from queue of size "
            f"{self.actual_queue_length} on device {queue.pointer.device})"
        )

        # Get the full ordered queue data
        full_queue_data = queue.get()

        # Take only the requested amount (most recent items)
        if len(full_queue_data) > self.requested_length:
            # Get the last N items (most recent)
            tensor = full_queue_data[-self.requested_length :]
            logging.debug(
                f"  extracted last {self.requested_length} items from {len(full_queue_data)} available"
            )
        else:
            tensor = full_queue_data
            if len(tensor) < self.requested_length:
                logging.debug(
                    f"  queue not full yet: {len(tensor)}/{self.requested_length} items"
                )

        if self.gather_distributed and trainer.world_size > 1:
            gathered = pl_module.all_gather(tensor).flatten(0, 1)
            self._snapshot = gathered
            logging.debug(
                f"  {self.key}: {tensor.shape} -> {gathered.shape} (gathered)"
            )
        else:
            self._snapshot = tensor
            logging.debug(f"  {self.key}: {tensor.shape}")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Clean up snapshot after validation."""
        self._snapshot = None

    @property
    def data(self) -> Optional[torch.Tensor]:
        """Get snapshot data during validation."""
        if self._snapshot is None:
            logging.warning("! no queue snapshot available, called outside validation?")
            return None
        return self._snapshot


def find_or_create_queue_callback(
    trainer: Trainer,
    key: str,
    queue_length: int,
    dim: Optional[Union[int, tuple]] = None,
    dtype: Optional[torch.dtype] = None,
    gather_distributed: bool = False,
    create_if_missing: bool = True,
) -> "OnlineQueue":
    """Find or create an OnlineQueue callback with unified size management.

    This function implements intelligent queue unification:
    - If a queue exists for the key with a different size, it reuses the same
      underlying queue and adjusts its size if needed
    - Each callback gets exactly the amount of data it requests
    - Memory is optimized by sharing the same storage

    Args:
        trainer: The Lightning trainer containing callbacks
        key: The batch key to look for
        queue_length: Number of samples this callback needs
        dim: Required dimension (None means any)
        dtype: Required dtype (None means any)
        gather_distributed: Whether to gather across distributed processes
        create_if_missing: If True, create queue when not found

    Returns:
        The matching or newly created OnlineQueue callback

    Raises:
        ValueError: If no matching queue is found and create_if_missing is False
    """
    # Wipe stale class-level state from a previous trainer before touching it
    # (#378). Idempotent within a trainer's lifecycle.
    OnlineQueue._maybe_wipe_stale_state(trainer)

    matching_queues = []

    for callback in trainer.callbacks:
        if isinstance(callback, OnlineQueue) and callback.key == key:
            # For unified queue management, we don't check queue_length equality
            # Just check dim and dtype compatibility

            # Check dim compatibility (None matches anything)
            if dim is not None and callback.dim is not None and callback.dim != dim:
                continue

            # Check dtype compatibility (None matches anything)
            if (
                dtype is not None
                and callback.dtype is not None
                and callback.dtype != dtype
            ):
                continue

            matching_queues.append(callback)

    if not matching_queues:
        if create_if_missing:
            # Create a new queue callback
            logging.info(
                f"  no queue found for key '{key}', creating new OnlineQueue with "
                f"length={queue_length}, dim={dim}, dtype={dtype}"
            )
            new_queue = OnlineQueue(
                key=key,
                queue_length=queue_length,
                dim=dim,
                dtype=dtype,
                gather_distributed=gather_distributed,
            )

            # Initialize queue info immediately for the first queue
            if key not in OnlineQueue._queue_info:
                OnlineQueue._queue_info[key] = {
                    "max_length": queue_length,
                    "dim": dim,
                    "dtype": dtype,
                    "callbacks": [new_queue],
                }

            # Add to trainer callbacks
            trainer.callbacks.append(new_queue)
            # Run setup if trainer is already set up
            if (
                hasattr(trainer, "lightning_module")
                and trainer.lightning_module is not None
            ):
                new_queue.setup(trainer, trainer.lightning_module, "fit")
            return new_queue
        else:
            # List all available queues for better error message
            available = [
                f"(key='{cb.key}', requested={cb.requested_length}, actual={cb.actual_queue_length}, "
                f"dim={cb.dim}, dtype={cb.dtype})"
                for cb in trainer.callbacks
                if isinstance(cb, OnlineQueue)
            ]
            raise ValueError(
                f"No OnlineQueue found for key '{key}'. Available queues: {available}"
            )

    # With unified management, we can have multiple callbacks for same key
    # Find the one with matching requested_length or create new one
    for callback in matching_queues:
        if callback.requested_length == queue_length:
            logging.info(
                f"  found existing OnlineQueue for key '{key}' with "
                f"requested_length={queue_length} (actual queue size: {callback.actual_queue_length})"
            )
            return callback

    # No exact match on requested_length, but we have queues for this key
    # Create a new callback that will share the underlying queue
    if create_if_missing:
        logging.info(
            f"  creating new OnlineQueue callback for key '{key}' with "
            f"requested_length={queue_length} (will share underlying queue)"
        )
        new_queue = OnlineQueue(
            key=key,
            queue_length=queue_length,
            dim=dim or matching_queues[0].dim,
            dtype=dtype or matching_queues[0].dtype,
            gather_distributed=gather_distributed,
        )

        # Update the queue info immediately if needed
        if key in OnlineQueue._queue_info:
            old_max = OnlineQueue._queue_info[key]["max_length"]
            if queue_length > old_max:
                OnlineQueue._queue_info[key]["max_length"] = queue_length
                logging.info(
                    f"  updated max size for key '{key}' "
                    f"from {old_max} to {queue_length}"
                )
            # Add the new callback to the list
            if new_queue not in OnlineQueue._queue_info[key]["callbacks"]:
                OnlineQueue._queue_info[key]["callbacks"].append(new_queue)

        trainer.callbacks.append(new_queue)
        if (
            hasattr(trainer, "lightning_module")
            and trainer.lightning_module is not None
        ):
            new_queue.setup(trainer, trainer.lightning_module, "fit")
        return new_queue

    # If we get here, we found queues but none with exact size and create_if_missing is False
    queue_details = [
        f"(requested={cb.requested_length}, actual={cb.actual_queue_length})"
        for cb in matching_queues
    ]
    logging.warning(
        f"! found OnlineQueue callbacks for key '{key}' but none with "
        f"requested_length={queue_length}. Existing queues: {queue_details}. "
        f"Using the first one."
    )
    return matching_queues[0]
