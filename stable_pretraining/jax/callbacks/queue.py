"""Feature-queue callback for the JAX backend.

JAX port of :class:`stable_pretraining.callbacks.OnlineQueue`. Maintains a FIFO
bank of a chosen batch-dict key's values (and an optional companion label key)
across training batches — the building block for queue-based methods (NNCLR,
MoCo, SwAV-with-queue) and for queue-fed eval (KNN, LiDAR).

The torch version shares one queue across instances by key and gathers across
ranks; this is a clean single-instance FIFO (host-side numpy), sufficient for
single-process / SPMD-data-parallel runs.
"""

import jax.numpy as jnp
import numpy as np

from ..trainer import Callback


class OnlineQueue(Callback):
    """Maintain a FIFO bank of a forward-output key across training batches.

    Args:
        key: Forward-output key to enqueue (e.g. ``"embedding"`` or ``"projection"``).
        queue_length: Maximum number of rows retained (oldest dropped first).
        label_key: Optional companion key (e.g. ``"label"``) queued in lockstep.

    Attributes:
        features: Current queue contents as a ``[N, D]`` JAX array (or ``None``).
        labels: Current labels as ``[N]`` (or ``None`` if ``label_key`` absent).
    """

    def __init__(
        self, key: str = "embedding", queue_length: int = 8192, label_key=None
    ):
        if queue_length <= 0:
            raise ValueError("queue_length must be positive.")
        self.key = key
        self.queue_length = queue_length
        self.label_key = label_key
        self._feats: list = []
        self._labels: list = []

    def _trim(self):
        total = sum(f.shape[0] for f in self._feats)
        while total > self.queue_length and len(self._feats) > 1:
            total -= self._feats[0].shape[0]
            self._feats.pop(0)
            if self._labels:
                self._labels.pop(0)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.key not in outputs:
            return
        self._feats.append(np.asarray(outputs[self.key]))
        if self.label_key and self.label_key in outputs:
            self._labels.append(np.asarray(outputs[self.label_key]))
        self._trim()

    @property
    def features(self):
        if not self._feats:
            return None
        return jnp.asarray(np.concatenate(self._feats, axis=0))

    @property
    def labels(self):
        if not self._labels:
            return None
        return jnp.asarray(np.concatenate(self._labels, axis=0))

    def __len__(self):
        return sum(f.shape[0] for f in self._feats)
