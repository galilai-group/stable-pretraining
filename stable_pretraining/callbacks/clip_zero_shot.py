import torch
import torch.nn.functional as F
from typing import Optional, Callable

from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
import torchmetrics

from ..utils import get_data_from_batch_or_outputs
from .utils import format_metrics_as_dict


class CLIPZeroShot(Callback):
    """Zero-shot classification evaluator for CLIP-style models.

    This callback computes zero-shot predictions by computing the similarity between the image embeddings and the class embeddings.

    Args:
        name: Unique identifier for this callback instance (used as log prefix and registry key).
        image_key: Key in batch or outputs containing input images or precomputed image features.
        tokens_key: Key in batch containing tokenized text.
        class_key: Key in batch containing ground-truth class indices (0..C-1, aligned with class_names order).
        class_names: List of class names in index order.
        image_backbone: Module/callable to encode images into embeddings.
        text_backbone: Module/callable to encode tokenized text into embeddings.
        tokenizer_fn: Callable that maps str | list[str] -> tensor of shape (T,).
        metrics: Dict of torchmetrics to compute on validation (e.g., {"top1": MulticlassAccuracy(...)}).
    """

    def __init__(
        self,
        name: str,
        image_key: str,
        class_key: str,
        class_names: list[str],
        image_backbone: torch.nn.Module,
        text_backbone: torch.nn.Module,
        tokenizer_fn: Callable[[str | list[str]], torch.Tensor],
        metrics: Optional[dict | tuple | list | torchmetrics.Metric] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.image_key = image_key
        self.class_key = class_key
        self.class_names = class_names
        self.class_map = {i: c for i, c in enumerate(class_names)}
        self.image_backbone = image_backbone
        self.text_backbone = text_backbone
        self.tokenizer_fn = tokenizer_fn

        self._train_metrics = None
        self._val_metrics = None

        # Format metrics
        self.metrics_config = metrics

        logging.info(f"Initialized CLIPZeroShot callback: {name}")
        logging.info(f"  - Image key: {image_key}")
        logging.info(f"  - Number of classes: {len(class_names)}")
        logging.info(f"  - Class names: [{', '.join(class_names[:5])}...]")
        logging.info(f"  - Image backbone: {image_backbone.__class__.__name__}")

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Initialize optimizer, scheduler, and metrics."""
        # Call parent setup for module/optimizer/scheduler
        super().setup(trainer, pl_module, stage)

        # Setup metrics
        logging.info(f"{self.name}: Setting up metrics")
        pl_module.callbacks_metrics[self.name] = format_metrics_as_dict(
            self.metrics_config
        )

        self.image_backbone = self.image_backbone.to(device=pl_module.device)
        self.text_backbone = self.text_backbone.to(device=pl_module.device)

        self._train_metrics = pl_module.callbacks_metrics[self.name]["_train"]
        self._val_metrics = pl_module.callbacks_metrics[self.name]["_val"]
        self.class_tokens = self.tokenizer_fn(self.class_names).to(
            device=pl_module.device
        )
        self.class_embeds = self.text_backbone(input_ids=self.class_tokens).text_embeds
        self.class_embeds = F.normalize(self.class_embeds, dim=-1)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        image = get_data_from_batch_or_outputs(
            self.image_key, batch, outputs, caller_name=self.name
        )
        classes = get_data_from_batch_or_outputs(
            self.class_key, batch, outputs, caller_name=self.name
        )
        if image is None:
            return

        image = image.to(device=pl_module.device)

        with torch.no_grad():
            image_features = self.image_backbone(image).image_embeds
            image_features = F.normalize(image_features, dim=-1)
            logits = image_features @ self.class_embeds.T

        prediction_key = f"{self.name}_preds"
        if prediction_key not in batch:
            batch[prediction_key] = logits.detach()

        logs = {}
        for metric_name, metric in pl_module.callbacks_metrics[self.name][
            "_val"
        ].items():
            metric(
                logits.detach(),
                torch.tensor(classes) if isinstance(classes, list) else classes,
            )
            logs[f"val/{self.name}_{metric_name}"] = metric

        # torchmetrics handle distributed sync internally — do not double-sync
        pl_module.log_dict(logs, on_step=False, on_epoch=True, sync_dist=False)
