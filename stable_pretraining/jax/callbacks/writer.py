"""Write forward-output values to disk during a run (JAX backend).

JAX port of :class:`stable_pretraining.callbacks.OnlineWriter`. Accumulates one
or more forward-output keys during a chosen phase and writes them — concatenated
across the epoch — to a ``.npz`` file. Useful for offline feature extraction.
"""

from pathlib import Path

import numpy as np

from ..trainer import Callback


class OnlineWriter(Callback):
    """Save forward-output arrays to ``{path}/{phase}_epoch{epoch}.npz``.

    Args:
        names: Forward-output key(s) to save (str or list of str).
        path: Output directory (created if missing).
        during: Phase(s) to write — ``"train"`` and/or ``"val"``. Default ``("val",)``.
        every_n_epochs: Write only on every n-th epoch. Default ``1``.
    """

    def __init__(self, names, path, during=("val",), every_n_epochs: int = 1):
        self.names = [names] if isinstance(names, str) else list(names)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.during = (during,) if isinstance(during, str) else tuple(during)
        self.every_n_epochs = every_n_epochs
        self._buffers: dict = {}

    def _collect(self, phase, outputs):
        if phase not in self.during:
            return
        for name in self.names:
            if name in outputs:
                self._buffers.setdefault(name, []).append(np.asarray(outputs[name]))

    def _flush(self, trainer, phase):
        if phase not in self.during or not self._buffers:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            self._buffers = {}
            return
        arrays = {k: np.concatenate(v, axis=0) for k, v in self._buffers.items()}
        out = self.path / f"{phase}_epoch{trainer.current_epoch}.npz"
        np.savez(out, **arrays)
        self._buffers = {}

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self._collect("train", outputs)

    def on_train_epoch_end(self, trainer, module):
        self._flush(trainer, "train")

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self._collect("val", outputs)

    def on_validation_epoch_end(self, trainer, module):
        self._flush(trainer, "val")
