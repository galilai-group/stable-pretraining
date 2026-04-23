from typing import Iterable, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from .registry import log as _spt_log

from .queue import find_or_create_queue_callback


class RankMe(Callback):
    """RankMe (effective rank) monitor using queue discovery.

    RankMe measures the effective rank of feature representations by computing
    the exponential of the entropy of normalized singular values. This metric
    helps detect dimensional collapse in self-supervised learning.

    Args:
        name: Unique name for this callback instance
        target: Key in batch dict containing the feature embeddings to monitor
        queue_length: Required queue length
        target_shape: Shape of the target embeddings (e.g., 768 for 768-dim features)
    """

    def __init__(
        self,
        name: str,
        target: str,
        queue_length: int,
        target_shape: Union[int, Iterable[int]],
        verbose: bool = None,
    ) -> None:
        super().__init__()

        if isinstance(target_shape, (list, tuple)):
            if len(target_shape) == 1:
                target_shape = target_shape[0]
            else:
                target_shape = int(torch.prod(torch.tensor(target_shape)))

        self.name = name
        self.target = target
        self.queue_length = queue_length
        self.target_shape = target_shape
        from .utils import resolve_verbose

        self.verbose = resolve_verbose(verbose)

        self._target_queue = None

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"RankMe[name={self.name}]"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Find or create the queue callback for target features."""
        if self._target_queue is None:
            self._target_queue = find_or_create_queue_callback(
                trainer,
                self.target,
                self.queue_length,
                self.target_shape,
                torch.float32,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"  target queue: {self.target}")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute RankMe metric on the first validation batch only."""
        if batch_idx > 0:
            return

        logging.info("  computing RankMe on first validation batch")

        embeddings = self._target_queue.data

        if embeddings is None:
            logging.warning(
                f"! {self.name}: queue data not available (not in validation?)"
            )
            return

        if embeddings.numel() == 0:
            logging.warning(
                f"! {self.name}: queue data is empty, skipping RankMe computation"
            )
            return

        if trainer.global_rank == 0:
            with torch.no_grad():
                s = torch.linalg.svdvals(embeddings)

                p = (s / torch.sum(s, axis=0)) + 1e-5

                entropy = -torch.sum(p * torch.log(p))

                rankme = torch.exp(entropy)

                pl_module.log(self.name, rankme.item())
                if self.verbose:
                    _spt_log(f"{self.name}/entropy", entropy.item())
                    _spt_log(f"{self.name}/top_singular_value", s[0].item())
                    _spt_log(
                        f"{self.name}/condition_number",
                        (s[0] / s[-1].clamp(min=1e-10)).item(),
                    )
