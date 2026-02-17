from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from torch import Tensor

from ..utils import get_data_from_batch_or_outputs
from ..utils.distance_metrics import compute_pairwise_distances_chunked

from .queue import find_or_create_queue_callback
from .utils import format_metrics_as_dict


class OnlineKNN(Callback):
    """Weighted K-Nearest Neighbors online evaluator using queue discovery.

    This callback implements a weighted KNN classifier that evaluates the quality of
    learned representations during training. It automatically discovers or creates
    OnlineQueue callbacks to maintain circular buffers of features and labels, then
    uses this cached data to compute KNN predictions during validation.

    The KNN evaluation is performed by:
    1. Finding k nearest neighbors in the feature space
    2. Weighting neighbors by inverse distance with temperature scaling
    3. Using weighted voting to produce class predictions
    4. Computing specified metrics on the predictions

    Args:
        name: Unique identifier for this callback instance. Used for logging and
            storing metrics.
        input: Key in batch dict containing input features to evaluate.
        target: Key in batch dict containing ground truth target labels.
        queue_length: Size of the circular buffer for caching features and labels.
            Larger values provide more representative samples but use more memory.
        metrics: Dictionary of metrics to compute during validation. Keys are metric
            names, values are metric instances (e.g., torchmetrics.Accuracy).
        num_classes: Explicit number of classes for prediction output. If None,
            inferred from metrics (via num_classes attribute) or from cached labels.
            Setting this explicitly avoids shape mismatches when the queue cache
            doesn't contain all classes (e.g., few-shot or imbalanced data).
        input_dim: Expected dimensionality of input features. Can be int, tuple/list
            (will be flattened to product), or None to accept any dimension.
        target_dim: Expected dimensionality of targets. None accepts any dimension.
        k: Number of nearest neighbors to consider for voting. Default is 5.
        temperature: Temperature parameter for distance weighting. Lower values give
            more weight to closer neighbors. Default is 0.07.
        chunk_size: Batch size for memory-efficient distance computation. Set to -1
            to compute all distances at once. Default is -1.
        distance_metric: Distance metric for finding nearest neighbors. Options are
            'euclidean', 'squared_euclidean', 'cosine', 'manhattan'. Default is 'euclidean'.

    Raises:
        ValueError: If k <= 0, temperature <= 0, or chunk_size is invalid.

    Note:
        - The callback automatically handles distributed training by gathering data
        - Mixed precision is supported through automatic dtype conversion
        - Predictions are stored in batch dict with key '{name}_preds'
        - Metrics are logged with prefix 'eval/{name}_'
    """

    def __init__(
        self,
        name: str,
        input: str,
        target: str,
        queue_length: int,
        metrics: Dict,
        num_classes: Optional[int] = None,
        input_dim: Optional[Union[Tuple[int, ...], List[int], int]] = None,
        target_dim: Optional[int] = None,
        k: int = 5,
        temperature: float = 0.07,
        chunk_size: int = -1,
        distance_metric: Literal[
            "euclidean", "squared_euclidean", "cosine", "manhattan"
        ] = "euclidean",
    ) -> None:
        super().__init__()

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if chunk_size == 0 or chunk_size < -1:
            raise ValueError(f"chunk_size must be positive or -1, got {chunk_size}")

        if input_dim is not None and isinstance(input_dim, (list, tuple)):
            input_dim = int(np.prod(input_dim))

        self.name = name
        self.input = input
        self.target = target
        self.queue_length = queue_length
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.k = k
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.distance_metric = distance_metric
        self.metrics = metrics

        self._input_queue = None
        self._target_queue = None

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"OnlineKNN[name={self.name}]"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Find or create queue callbacks and setup metrics."""
        logging.info(f"Setting up {self.state_key} callback!")
        if self._input_queue is None or self._target_queue is None:
            self._input_queue = find_or_create_queue_callback(
                trainer,
                self.input,
                self.queue_length,
                self.input_dim,
                torch.float32 if self.input_dim is not None else None,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"{self.name}: Using queue for input '{self.input}'")

            self._target_queue = find_or_create_queue_callback(
                trainer,
                self.target,
                self.queue_length,
                self.target_dim,
                torch.long if self.target_dim is not None else None,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"{self.name}: Using queue for target '{self.target}'")

            logging.info(f"{self.name}: Setting up metrics")
            pl_module.callbacks_metrics[self.name] = format_metrics_as_dict(
                self.metrics
            )

            # Infer num_classes from metrics if not explicitly provided
            if self.num_classes is None:
                self.num_classes = self._infer_num_classes_from_metrics()
                if self.num_classes is not None:
                    logging.info(
                        f"{self.name}: Inferred num_classes={self.num_classes} from metrics"
                    )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute KNN predictions during validation."""
        input_data = get_data_from_batch_or_outputs(
            self.input, batch, outputs, caller_name=self.name
        )
        if input_data is None:
            return

        target_data = get_data_from_batch_or_outputs(
            self.target, batch, outputs, caller_name=self.name
        )
        if target_data is None:
            return

        cached_features = self._input_queue.data
        cached_labels = self._target_queue.data

        if cached_features is None or cached_labels is None:
            logging.warning(
                f"{self.name}: Queue data not available (not in validation?)"
            )
            return

        if cached_features.numel() == 0 or cached_labels.numel() == 0:
            logging.warning(
                f"{self.name}: Queue data is empty, skipping KNN computation"
            )
            return

        predictions = self._compute_knn_predictions(
            input_data, cached_features, cached_labels
        )

        if predictions is not None:
            prediction_key = f"{self.name}_preds"
            if prediction_key in batch:
                raise ValueError(f"Key '{prediction_key}' already exists in batch")
            batch[prediction_key] = predictions

            self._log_metrics(pl_module, predictions, batch[self.target])

    def _infer_num_classes_from_metrics(self) -> Optional[int]:
        """Try to extract num_classes from the configured metrics."""
        metrics = self.metrics
        if isinstance(metrics, dict):
            items = metrics.values()
        elif isinstance(metrics, (list, tuple)):
            items = metrics
        else:
            items = [metrics]

        for metric in items:
            if hasattr(metric, "num_classes") and metric.num_classes is not None:
                return metric.num_classes
        return None

    @torch.no_grad()
    def _compute_knn_predictions(
        self,
        features: Tensor,
        cached_features: Tensor,
        cached_labels: Tensor,
    ) -> Optional[Tensor]:
        """Compute KNN predictions."""
        batch_size = features.size(0)

        if self.num_classes is not None:
            num_classes = self.num_classes
        else:
            num_classes = int(cached_labels.max().item()) + 1

        predictions = torch.zeros(
            batch_size, num_classes, device=features.device, dtype=torch.float32
        )

        k_actual = min(self.k, cached_features.size(0))

        if cached_features.dtype != features.dtype:
            cached_features = cached_features.float()
            features = features.float()

        chunk_size = batch_size if self.chunk_size == -1 else self.chunk_size
        dist_matrix = compute_pairwise_distances_chunked(
            cached_features,
            features,
            metric=self.distance_metric,
            chunk_size=chunk_size,
        )

        dist_weight, sim_indices = dist_matrix.topk(k=k_actual, dim=0, largest=False)

        dist_weight = 1 / dist_weight.add_(self.temperature)

        labels_1d = (
            cached_labels.squeeze(-1) if cached_labels.dim() > 1 else cached_labels
        )
        selected_labels = labels_1d[sim_indices].long()
        one_hot_labels = F.one_hot(selected_labels, num_classes=num_classes)

        predictions = (dist_weight.unsqueeze(-1) * one_hot_labels).sum(0)
        return predictions

    def _log_metrics(
        self, pl_module: LightningModule, predictions: Tensor, targets: Tensor
    ) -> None:
        """Compute and log validation metrics."""
        logs = {}
        for metric_name, metric in pl_module.callbacks_metrics[self.name][
            "_val"
        ].items():
            metric(predictions, targets)
            logs[f"eval/{self.name}_{metric_name}"] = metric

        pl_module.log_dict(logs, on_step=False, on_epoch=True)
