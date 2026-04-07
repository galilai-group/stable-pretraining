from pathlib import Path
from typing import Union

import torch
from lightning.pytorch import Callback, LightningModule
from loguru import logger as logging

from .utils import log_header


class OnlineWriter(Callback):
    """Writes specified batch data to disk during training and validation.

    This callback enables selective saving of batch data (e.g., features, predictions,
    embeddings) to disk at specified intervals during training. It's useful for
    debugging, visualization, and analysis of model behavior during training.

    Features:
    - Flexible saving schedule (every k epochs, last epoch, sanity check)
    - Support for distributed training with optional all_gather
    - Automatic directory creation
    - Configurable for different training phases (train, val, test)

    Args:
        names: Name(s) of the batch keys to save. Can be string or list of strings.
        path: Directory path where files will be saved.
        during: Training phase(s) when to save ('train', 'val', 'test', or list).
        every_k_epochs: Save every k epochs. -1 means every epoch.
        save_last_epoch: Whether to save on the last training epoch.
        save_sanity_check: Whether to save during sanity check phase.
        all_gather: Whether to gather data across all distributed processes.

    Files are saved with naming pattern: {phase}_{name}_epoch{epoch}_batch{batch}.pt
    """

    def __init__(
        self,
        names: str,
        path: Union[str, Path],
        during: Union[str, list[str]],
        every_k_epochs: int = -1,
        save_last_epoch: bool = False,
        save_sanity_check: bool = False,
        all_gather: bool = True,
    ) -> None:
        super().__init__()
        log_header("OnlineWriter")
        logging.info(f"  names: {names}")
        logging.info(f"  path: {path}")
        logging.info(f"  during: {during}")
        logging.info(f"  every_k_epochs: {every_k_epochs}")
        logging.info(f"  save_last_epoch: {save_last_epoch}")
        logging.info(f"  save_sanity_check: {save_sanity_check}")
        logging.info(f"  all_gather: {all_gather}")

        path = Path(path)
        if type(names) is str:
            names = [names]
        if type(during) is str:
            during = [during]
        self.names = names
        self.path = path
        self.during = during

        self.every_k_epochs = every_k_epochs
        self.save_last_epoch = save_last_epoch

        self.save_sanity_check = save_sanity_check
        self.is_sanity_check = False

        self.all_gather = all_gather

        if not path.is_dir():
            logging.warning(f"! path {path} does not exist, creating it")
            path.mkdir(parents=True, exist_ok=True)

    def on_sanity_check_start(self, trainer, pl_module):
        self.is_sanity_check = True

        if not self.save_sanity_check:
            logging.warning("! skipping sanity check writing")

    def on_sanity_check_end(self, trainer, pl_module):
        self.is_sanity_check = False

    def is_writing_epoch(self, pl_module):
        current_epoch = pl_module.current_epoch
        max_epochs = pl_module.trainer.max_epochs

        save_every_epoch = self.every_k_epochs == -1
        save_k_epoch = (
            current_epoch % self.every_k_epochs == 0
            if self.every_k_epochs != 0
            else False
        )

        is_last_epoch = current_epoch == max_epochs - 1
        save_last_epoch = self.save_last_epoch and is_last_epoch

        return any([save_every_epoch, save_k_epoch, save_last_epoch])

    def write_at_phase(
        self,
        pl_module,
        phase_name,
        outputs,
        batch_idx,
    ):
        if self.is_sanity_check and not self.save_sanity_check:
            return

        if not self.is_writing_epoch(pl_module) or phase_name not in self.during:
            return

        file_info = {
            "epoch": pl_module.current_epoch,
            "batch": batch_idx,
        }

        if not self.all_gather:
            file_info["device"] = pl_module.local_rank

        file_info = "_".join(f"{k}={v}" for k, v in file_info.items())
        filename = f"{phase_name}_{file_info}.pt"

        self.dump(pl_module, outputs, filename)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.write_at_phase(pl_module, "train", outputs, batch_idx)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.write_at_phase(pl_module, "predict", outputs, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.write_at_phase(pl_module, "test", outputs, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.write_at_phase(pl_module, "validation", outputs, batch_idx)

    def dump(self, pl_module: LightningModule, outputs: dict, filename: str):
        to_save = {}

        for name in self.names:
            if name not in outputs:
                msg = (
                    f"Asking to write {name} but not present "
                    f"in current batch {list(outputs.keys())}"
                )
                logging.error(msg)
                raise ValueError(msg)

            data = outputs[name]

            if self.all_gather:
                to_save[name] = pl_module.all_gather(data).cpu()
            else:
                to_save[name] = data.cpu()

        if not self.all_gather or pl_module.local_rank == 0:
            torch.save(to_save, self.path / filename)
