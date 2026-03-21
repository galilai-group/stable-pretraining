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
        multi_label: Whether the dataset uses multi-label classification. Default is False.
        num_classes: Number of total classes. Required if multi_label is True.

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
        input_dim: Optional[Union[Tuple[int, ...], List[int], int]] = None,
        target_dim: Optional[int] = None,
        k: int = 5,
        temperature: float = 0.07,
        chunk_size: int = -1,
        distance_metric: Literal[
            "euclidean", "squared_euclidean", "cosine", "manhattan"
        ] = "euclidean",
        multi_label: bool = False,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if chunk_size == 0 or chunk_size < -1:
            raise ValueError(f"chunk_size must be positive or -1, got {chunk_size}")

        if multi_label and num_classes is None:
            raise ValueError("You must provide `num_classes` when `multi_label=True`.")

        if input_dim is not None and isinstance(input_dim, (list, tuple)):
            input_dim = int(np.prod(input_dim))

        self.name = name
        self.input = input
        self.target = target
        self.queue_length = queue_length
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.k = k
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.distance_metric = distance_metric
        self.metrics = metrics
        self.multi_label = multi_label
        self.num_classes = num_classes

        self._input_queue = None
        self._target_queue = None

        for metric in self.metrics.values():
            if metric.__class__.__name__ in ["MultilabelAUROC", "MulticlassAUROC"]:
                logging.warning(
                    f"[{self.name}] AUROC metric detected. torchmetrics expects target labels to be Long integers. "
                    "If you are passing float targets, you can cast them to long in your dataset/forward pass. "
                )
                break

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

    @torch.no_grad()
    def _compute_knn_predictions(
        self,
        features: Tensor,
        cached_features: Tensor,
        cached_labels: Tensor,
    ) -> Optional[Tensor]:
        """Compute KNN predictions."""
        batch_size = features.size(0)

        if cached_features.device != features.device:
            cached_features = cached_features.to(features.device)
            cached_labels = cached_labels.to(features.device)

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

        if self.multi_label:
            selected_labels = cached_labels[sim_indices].float()
            weighted_votes = dist_weight.unsqueeze(-1) * selected_labels
            total_votes = weighted_votes.sum(dim=0)
            sum_of_weights = dist_weight.sum(dim=0).unsqueeze(-1)
            predictions = total_votes / sum_of_weights
        else:
            num_classes = int(cached_labels.max().item()) + 1
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
