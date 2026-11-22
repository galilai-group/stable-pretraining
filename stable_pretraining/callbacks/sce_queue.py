import torch
from lightning.pytorch.callbacks import Callback


class SCEQueueCallback(Callback):
    """Store SCE momentum keys after each training batch."""

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if outputs is None or "keys" not in outputs:
            return

        keys = outputs["keys"]

        if keys is None:
            return

        with torch.no_grad():
            pl_module.enqueue_keys(keys)