"""Online weighted-KNN evaluation callback for the JAX backend.

JAX port of :class:`stable_pretraining.callbacks.OnlineKNN`. It maintains a
FIFO feature bank populated from training-batch embeddings and, each validation
epoch, classifies validation embeddings by distance-weighted k-NN vote against
the bank — a label-efficient, training-free probe of representation quality.

The weighting matches the torch version: ``weight = 1 / (distance + temperature)``
over the ``k`` nearest neighbours, summed per class.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..trainer import Callback, Trainer


def knn_predict(features, bank_feats, bank_labels, num_classes, k, temperature, metric):
    """Distance-weighted k-NN class scores for ``features`` against a bank.

    Args:
        features: Query embeddings ``[B, D]``.
        bank_feats: Bank embeddings ``[N, D]``.
        bank_labels: Bank integer labels ``[N]``.
        num_classes: Number of classes.
        k: Number of neighbours.
        temperature: Distance-weighting temperature.
        metric: ``"euclidean"`` or ``"cosine"``.

    Returns:
        jnp.ndarray: Per-class scores ``[B, num_classes]`` (argmax = prediction).
    """
    if metric == "cosine":
        f = features / (jnp.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
        b = bank_feats / (jnp.linalg.norm(bank_feats, axis=1, keepdims=True) + 1e-12)
        dist = 1.0 - f @ b.T  # cosine distance, [B, N]
    else:  # euclidean
        dist = jnp.linalg.norm(features[:, None, :] - bank_feats[None, :, :], axis=-1)

    k_actual = min(k, bank_feats.shape[0])
    neg_top, idx = jax.lax.top_k(-dist, k_actual)  # k smallest distances
    weights = 1.0 / (-neg_top + temperature)  # [B, k]
    neighbor_labels = bank_labels[idx]  # [B, k]
    onehot = jax.nn.one_hot(neighbor_labels, num_classes)  # [B, k, C]
    return jnp.sum(weights[..., None] * onehot, axis=1)  # [B, C]


class OnlineKNN(Callback):
    """Weighted k-NN validation accuracy over a FIFO bank of train embeddings.

    Args:
        name: Metric name (logged as ``val/<name>_acc``). Default ``"knn"``.
        num_classes: Number of classes.
        input: Forward-output key holding embeddings. Default ``"embedding"``.
        target: Forward-output key holding labels. Default ``"label"``.
        k: Number of neighbours. Default ``20``.
        temperature: Distance-weighting temperature. Default ``0.07``.
        metric: ``"euclidean"`` (default) or ``"cosine"``.
        bank_size: Max number of (feature, label) pairs retained. Default ``8192``.

    Attributes:
        accuracy: Top-1 k-NN accuracy from the most recent validation epoch.
    """

    def __init__(
        self,
        num_classes: int,
        name: str = "knn",
        input: str = "embedding",
        target: str = "label",
        k: int = 20,
        temperature: float = 0.07,
        metric: str = "euclidean",
        bank_size: int = 8192,
    ):
        if k <= 0 or temperature <= 0:
            raise ValueError("k and temperature must be positive.")
        self.num_classes = num_classes
        self.name = name
        self.input = input
        self.target = target
        self.k = k
        self.temperature = temperature
        self.metric = metric
        self.bank_size = bank_size
        self.accuracy: float = float("nan")
        self._feats: list = []
        self._labels: list = []
        self._correct = 0
        self._total = 0

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.input not in outputs or self.target not in outputs:
            return
        # Grow a FIFO bank from training embeddings (host-side numpy).
        self._feats.append(np.asarray(outputs[self.input]))
        self._labels.append(np.asarray(outputs[self.target]))
        total = sum(f.shape[0] for f in self._feats)
        while total > self.bank_size and len(self._feats) > 1:
            total -= self._feats[0].shape[0]
            self._feats.pop(0)
            self._labels.pop(0)

    def on_validation_epoch_start(self, trainer, module):
        self._correct = 0
        self._total = 0
        if self._feats:
            self._bank_feats = jnp.asarray(np.concatenate(self._feats, axis=0))
            self._bank_labels = jnp.asarray(np.concatenate(self._labels, axis=0))
        else:
            self._bank_feats = None

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if getattr(self, "_bank_feats", None) is None:
            return
        if self.input not in outputs or self.target not in outputs:
            return
        scores = knn_predict(
            outputs[self.input],
            self._bank_feats,
            self._bank_labels,
            self.num_classes,
            self.k,
            self.temperature,
            self.metric,
        )
        preds = jnp.argmax(scores, axis=-1)
        self._correct += int(jnp.sum(preds == outputs[self.target]))
        self._total += int(outputs[self.target].shape[0])

    def on_validation_epoch_end(self, trainer: Trainer, module):
        if self._total > 0:
            self.accuracy = self._correct / self._total
            # ``eval/`` prefix matches the torch OnlineKNN metric key.
            trainer.log(f"eval/{self.name}_acc", self.accuracy)
