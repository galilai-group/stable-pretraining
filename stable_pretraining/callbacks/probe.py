from functools import partial
from typing import Optional, Union

import torch
import torchmetrics
from lightning.pytorch import LightningModule
from loguru import logger as logging
import types
from ..utils import get_data_from_batch_or_outputs, detach_tensors

from .utils import TrainableCallback


class OnlineProbe(TrainableCallback):
    """Online probe for evaluating learned representations during self-supervised training.

    This callback implements the standard linear evaluation protocol by training a probe
    (typically a linear classifier) on top of frozen features from the main model. The probe
    is trained simultaneously with the main model but maintains its own optimizer, scheduler,
    and training loop. This allows monitoring representation quality throughout training
    without modifying the base model.

    Key features:
    - Automatic gradient detachment to prevent probe gradients affecting the main model
    - Independent optimizer and scheduler management
    - Support for gradient accumulation
    - Mixed precision training compatibility through automatic dtype conversion
    - Metric tracking and logging

    Args:
        module: The spt.LightningModule to probe.
        name: Unique identifier for this probe instance. Used for logging and storing
            metrics/modules.
        input: Key in batch dict or outputs dict containing input features to probe.
        target: Key in batch dict containing ground truth target labels.
        probe: The probe module to train. Can be a nn.Module instance, callable that
            returns a module, or Hydra config to instantiate.
        loss: Loss function for probe training (e.g., nn.CrossEntropyLoss()).
        optimizer: Optimizer configuration for the probe. Can be:
            - str: optimizer name (e.g., "AdamW", "SGD", "LARS")
            - dict: {"type": "AdamW", "lr": 1e-3, ...}
            - partial: pre-configured optimizer factory
            - optimizer instance or callable
            - None: uses LARS(lr=0.1, clip_lr=True, eta=0.02, exclude_bias_n_norm=True,
              weight_decay=0), which is the standard for SSL linear probes (default)
        scheduler: Learning rate scheduler configuration. Can be:
            - str: scheduler name (e.g., "CosineAnnealingLR", "StepLR")
            - dict: {"type": "CosineAnnealingLR", "T_max": 1000, ...}
            - partial: pre-configured scheduler factory
            - scheduler instance or callable
            - None: uses ConstantLR(factor=1.0), maintaining constant learning rate (default)
        accumulate_grad_batches: Number of batches to accumulate gradients before
            optimizer step. Default is 1 (no accumulation).
        metrics: Metrics to track during training/validation. Can be dict, list, tuple,
            or single metric instance.

    Note:
        - The probe module is stored in pl_module.callbacks_modules[name]
        - Metrics are stored in pl_module.callbacks_metrics[name]
        - Predictions are stored in batch dict with key '{name}_preds'
        - Loss is logged as 'train/{name}_loss'
        - Metrics are logged with prefix 'train/{name}_' and 'eval/{name}_'
    """

    def __init__(
        self,
        module: LightningModule,
        name: str,
        input: str,
        target: str,
        probe: torch.nn.Module,
        loss: callable = None,
        optimizer: Optional[Union[str, dict, partial, torch.optim.Optimizer]] = None,
        scheduler: Optional[
            Union[str, dict, partial, torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = None,
        gradient_clip_algorithm: str = "norm",
        metrics: Optional[Union[dict, tuple, list, torchmetrics.Metric]] = None,
    ) -> None:
        # Initialize base class
        self.input = input
        self.target = target
        if loss is None:
            logging.warning(f"Not loss given to {name}, will use output of `probe`")
        self.loss = loss

        # Store probe configuration for later initialization
        self._probe_config = probe

        # Format metrics
        super().__init__(
            module=module,
            name=name,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

        logging.info(f"Initialized {self.name}")
        logging.info(f"  - Input: {input}")
        logging.info(f"  - Target: {target}")
        logging.info(f"  - Accumulate grad batches: {accumulate_grad_batches}")
        # Setup metrics
        self.metrics = metrics
        logging.info(f"{self.name}: We are wrapping up your `forward`!")
        self.wrap_forward(pl_module=module)

    def configure_model(self, pl_module: LightningModule) -> torch.nn.Module:
        """Initialize the probe module from configuration."""
        if isinstance(self._probe_config, torch.nn.Module):
            probe_module = self._probe_config
        elif callable(self._probe_config):
            probe_module = self._probe_config(pl_module)
        else:
            raise ValueError("the probe should be a module or a callable")
        return probe_module

    def wrap_forward(self, pl_module):
        fn = pl_module.forward

        def new_forward(self, batch, stage, callback=self, fn=fn):
            outputs = fn(batch, stage)
            if (
                callback.input is None
                or callback.target is None
                or callback.loss is None
            ):
                assert callback.target is None
                assert callback.input is None
                assert callback.loss is None
                return callback.module(batch, outputs, self)
            else:
                x = get_data_from_batch_or_outputs(
                    callback.input, batch, outputs, caller_name=callback.name
                )
                y = get_data_from_batch_or_outputs(
                    callback.target, batch, outputs, caller_name=callback.name
                )

                if x is None or y is None:
                    raise ValueError(
                        f"Callback {callback.name} missing {callback.input} or {callback.target}"
                    )

                preds = callback.module(detach_tensors(x))
                y = detach_tensors(y)

            prediction_key = f"{callback.name}_preds"
            assert prediction_key not in batch
            outputs[prediction_key] = preds

            logs = {}
            if stage == "fit":
                loss = callback.loss(preds, y)
                assert f"train/{callback.name}_loss" not in logs
                if "loss" not in outputs:
                    outputs["loss"] = 0
                outputs["loss"] = outputs["loss"] + loss
                logs[f"train/{callback.name}_loss"] = loss.item()

                my_metrics = self.callbacks_metrics[callback.name]["_train"]
                for metric_name, metric in my_metrics.items():
                    metric.update(preds, y)
                    assert f"train/{callback.name}_{metric_name}" not in logs
                    logs[f"train/{callback.name}_{metric_name}"] = metric
            elif stage == "validate":
                my_metrics = pl_module.callbacks_metrics[callback.name]["_val"]
                for metric_name, metric in my_metrics.items():
                    metric(preds, y)
                    logs[f"eval/{callback.name}_{metric_name}"] = metric

            self.log_dict(logs, on_step=True, on_epoch=True, sync_dist=True)
            return outputs

        # Bind the new method to the instance
        pl_module.forward = types.MethodType(new_forward, pl_module)
