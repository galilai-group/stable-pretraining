"""LiDAR representation-quality callback for the JAX backend.

JAX port of :class:`stable_pretraining.callbacks.LiDAR`
:cite:`thilak2023lidar`. LiDAR is the effective rank of the Linear Discriminant
Analysis eigenvalue spectrum: ``exp(entropy(normalized LDA eigenvalues))``,
between ``1`` and ``min(d, n_classes - 1)``.

This implementation groups validation embeddings by their **labels** (each label
is a surrogate class) rather than chunking a queue sequentially — a meaningful,
self-contained variant for monitoring class discriminability. The compute runs
host-side (NumPy) at validation-epoch end.
"""

import numpy as np

from ..trainer import Callback, Trainer


def lidar(embeddings, labels, delta: float = 1e-4) -> float:
    """Compute the LiDAR effective rank from labelled embeddings.

    Args:
        embeddings: ``[N, D]`` features.
        labels: ``[N]`` integer class labels (surrogate classes).
        delta: Ridge added to the within-class scatter for stability.

    Returns:
        float: ``exp(entropy)`` of the normalized LDA eigenvalue spectrum, or
        ``nan`` if fewer than 2 classes are present.
    """
    return _lidar_components(embeddings, labels, delta)[0]


def _lidar_components(embeddings, labels, delta: float = 1e-4):
    """Return ``(lidar, entropy, top_eigenvalue)``; ``(nan, nan, nan)`` if <2 classes."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n, d = embeddings.shape
    if len(classes) < 2:
        return float("nan"), float("nan"), float("nan")

    means = np.stack([embeddings[labels == c].mean(0) for c in classes])  # (C, d)
    grand = means.mean(0)
    centered = means - grand
    sb = (centered.T @ centered) / (len(classes) - 1)  # between-class scatter

    sw = np.zeros((d, d))
    for c in classes:
        x = embeddings[labels == c]
        xc = x - x.mean(0)
        sw += xc.T @ xc
    sw = sw / max(n - len(classes), 1) + delta * np.eye(d)

    # Whiten by Sw^{-1/2} (symmetric) then take eigenvalues of the result.
    w, v = np.linalg.eigh(sw)
    w = np.clip(w, 1e-12, None)
    sw_inv_half = v @ np.diag(1.0 / np.sqrt(w)) @ v.T
    m = sw_inv_half @ sb @ sw_inv_half
    eig = np.clip(np.linalg.eigvalsh(m), 0.0, None)
    total = eig.sum()
    if total <= 0:
        return 1.0, 0.0, float(eig[-1]) if eig.size else 0.0
    p = eig / total + 1e-12
    p = p / p.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy)), float(entropy), float(eig[-1])


class LiDAR(Callback):
    """Report the LiDAR effective rank of validation embeddings each epoch.

    Logs under the bare metric name ``<name>`` (matching the torch LiDAR key);
    with ``verbose=True`` also logs ``<name>/entropy`` and ``<name>/top_eigenvalue``.

    Args:
        name: Metric name (logged as ``<name>``). Default ``"lidar"``.
        input: Forward-output key holding embeddings. Default ``"embedding"``.
        target: Forward-output key holding labels. Default ``"label"``.
        delta: Within-class scatter ridge. Default ``1e-4``.
        verbose: Also log entropy + top eigenvalue. Default ``False``.

    Attributes:
        value: LiDAR value from the most recent validation epoch.
    """

    def __init__(
        self, name="lidar", input="embedding", target="label", delta=1e-4, verbose=False
    ):
        self.name = name
        self.input = input
        self.target = target
        self.delta = delta
        self.verbose = verbose
        self.value: float = float("nan")
        self._feats: list = []
        self._labels: list = []

    def on_validation_epoch_start(self, trainer, module):
        self._feats = []
        self._labels = []

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.input in outputs and self.target in outputs:
            self._feats.append(np.asarray(outputs[self.input]))
            self._labels.append(np.asarray(outputs[self.target]))

    def on_validation_epoch_end(self, trainer: Trainer, module):
        if not self._feats:
            return
        z = np.concatenate(self._feats, axis=0)
        y = np.concatenate(self._labels, axis=0)
        self.value, entropy, top_eig = _lidar_components(z, y, self.delta)
        if np.isfinite(self.value):
            trainer.log(self.name, self.value)  # bare key, matches torch
            if self.verbose:
                trainer.log(f"{self.name}/entropy", entropy)
                trainer.log(f"{self.name}/top_eigenvalue", top_eig)
