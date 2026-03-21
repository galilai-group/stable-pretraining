"""Queue callback with unified size management and insertion order preservation.

This module provides a queue callback that uses OrderedQueue to maintain
insertion order and implements intelligent queue sharing when multiple callbacks
request the same data with different queue sizes.
"""

from typing import Dict, Optional, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from ..utils import OrderedQueue, get_data_from_batch_or_outputs


class OnlineQueue(Callback):
    """Circular buffer callback with insertion order preservation and size unification.

    This callback maintains an OrderedQueue that accumulates data from specified batch
    keys during training while preserving insertion order. It implements intelligent
    queue sharing: when multiple callbacks request the same data with different sizes,
    it uses a single queue with the maximum size and serves appropriate subsets.

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

    # Class-level registry to track shared queues by key
    _shared_queues: Dict[str, "OrderedQueue"] = {}
    _queue_info: Dict[str, dict] = {}  # Track max size and other info per key

    def __init__(
        self,
        key: str,
        queue_length: int,
        dim: Optional[Union[int, tuple]] = None,
        dtype: Optional[torch.dtype] = None,
        gather_distributed: bool = False,
    ) -> None:
        super().__init__()

        self.key = key
        self.requested_length = queue_length  # What this callback wants
        self.dim = dim
        self.dtype = dtype
        self.gather_distributed = gather_distributed
        self._snapshot = None

        logging.info(f"OnlineQueue initialized for key '{key}'")
        logging.info(f"\t- requested_length: {queue_length}")
        logging.info(f"\t- dim: {dim}")
        logging.info(f"\t- dtype: {dtype}")

    @property
    def actual_queue_length(self) -> int:
        """Get the actual size of the underlying shared queue."""
        if self.key in self._queue_info:
            return self._queue_info[self.key]["max_length"]
        return self.requested_length

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Initialize or connect to shared queue during setup phase."""
        # Update the maximum queue length for this key
        if self.key not in self._queue_info:
            self._queue_info[self.key] = {
                "max_length": self.requested_length,
                "dim": self.dim,
                "dtype": self.dtype,
                "callbacks": [self],
            }
            logging.info(
                f"OnlineQueue: New key '{self.key}' with initial size {self.requested_length}"
            )
        else:
            # Update max length if this callback needs more
            old_max = self._queue_info[self.key]["max_length"]
            if self.requested_length > old_max:
                self._queue_info[self.key]["max_length"] = self.requested_length
                logging.info(
                    f"OnlineQueue: Increased max size for key '{self.key}' "
                    f"from {old_max} to {self.requested_length}"
                )

            # Add this callback to the list
            if self not in self._queue_info[self.key]["callbacks"]:
                self._queue_info[self.key]["callbacks"].append(self)

        # Create or update the shared queue
        max_length = self._queue_info[self.key]["max_length"]

        # Resolve target device from the module's parameters
        device = next(pl_module.parameters()).device

        if self.key not in self._shared_queues:
            # Create new shared queue with maximum requested size
            self._shared_queues[self.key] = OrderedQueue(
                max_length, self.dim, self.dtype
            ).to(device)
            # Register in callbacks_modules for consistency
            queue_key = f"ordered_queue_{self.key}"
            pl_module.callbacks_modules[queue_key] = self._shared_queues[self.key]
            logging.info(
                f"OnlineQueue: Created shared queue for '{self.key}' with size {max_length}"
                f" on {device}"
            )
        elif self._shared_queues[self.key].max_length < max_length:
            # Need to resize the existing queue
            old_queue = self._shared_queues[self.key]
            old_data = (
                old_queue.get() if old_queue.pointer > 0 or old_queue.filled else None
            )

            # Create new larger queue
            new_queue = OrderedQueue(max_length, self.dim, self.dtype).to(device)

            # Copy old data if exists
            if old_data is not None and len(old_data) > 0:
                new_queue.append(old_data)
                logging.info(
                    f"OnlineQueue: Resized queue for '{self.key}' from "
                    f"{old_queue.max_length} to {max_length}, preserved {len(old_data)} items"
                )

            # Replace the queue
            self._shared_queues[self.key] = new_queue
            queue_key = f"ordered_queue_{self.key}"
            pl_module.callbacks_modules[queue_key] = new_queue

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

            # Append to the shared queue
            self._shared_queues[self.key].append(data)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Create snapshot of the requested portion of queue contents."""
        logging.info(
            f"OnlineQueue: Creating snapshot for key '{self.key}' "
            f"(requesting {self.requested_length} from queue of size {self.actual_queue_length})"
        )

        # Get the full ordered queue data
        full_queue_data = self._shared_queues[self.key].get()

        # Take only the requested amount (most recent items)
        if len(full_queue_data) > self.requested_length:
            # Get the last N items (most recent)
            tensor = full_queue_data[-self.requested_length :]
            logging.info(
                f"\t- Extracted last {self.requested_length} items from {len(full_queue_data)} available"
            )
        else:
            tensor = full_queue_data
            if len(tensor) < self.requested_length:
                logging.info(
                    f"\t- Queue not full yet: {len(tensor)}/{self.requested_length} items"
                )

        if self.gather_distributed and trainer.world_size > 1:
            gathered = pl_module.all_gather(tensor).flatten(0, 1)
            self._snapshot = gathered
            logging.info(
                f"\t- {self.key}: {tensor.shape} -> {gathered.shape} (gathered)"
            )
        else:
            self._snapshot = tensor
            logging.info(f"\t- {self.key}: {tensor.shape}")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Clean up snapshot after validation."""
        self._snapshot = None

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str = None
    ) -> None:
        """Clean up class-level shared state when training ends."""
        if self.key in self._shared_queues:
            del self._shared_queues[self.key]
        if self.key in self._queue_info:
            del self._queue_info[self.key]

    @property
    def data(self) -> Optional[torch.Tensor]:
        """Get snapshot data during validation."""
        if self._snapshot is None:
            logging.warning("No queue snapshot available. Called outside validation?")
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
                f"No queue found for key '{key}', creating new OnlineQueue with "
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
                f"Found existing OnlineQueue for key '{key}' with "
                f"requested_length={queue_length} (actual queue size: {callback.actual_queue_length})"
            )
            return callback

    # No exact match on requested_length, but we have queues for this key
    # Create a new callback that will share the underlying queue
    if create_if_missing:
        logging.info(
            f"Creating new OnlineQueue callback for key '{key}' with "
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
                    f"OnlineQueue: Updated max size for key '{key}' "
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
        f"Found OnlineQueue callbacks for key '{key}' but none with "
        f"requested_length={queue_length}. Existing queues: {queue_details}. "
        f"Using the first one."
    )
    return matching_queues[0]
