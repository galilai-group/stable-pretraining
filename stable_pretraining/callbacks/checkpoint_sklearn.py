from typing import Optional

import numpy as np
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from tabulate import tabulate
from typing import Any, Dict
import lightning.pytorch as pl
from loguru import logger

from .. import SKLEARN_AVAILABLE

if SKLEARN_AVAILABLE:
    from sklearn.base import ClassifierMixin, RegressorMixin
else:
    ClassifierMixin = None
    RegressorMixin = None


class SklearnCheckpoint(Callback):
    """Callback for saving and loading sklearn models in PyTorch Lightning checkpoints.

    This callback automatically detects sklearn models (Regressors and Classifiers)
    attached to the Lightning module and handles their serialization/deserialization
    during checkpoint save/load operations. This is necessary because sklearn models
    are not natively supported by PyTorch's checkpoint system.

    The callback will:
    1. Automatically discover sklearn models attached to the Lightning module
    2. Save them to the checkpoint dictionary during checkpoint saving
    3. Restore them from the checkpoint during checkpoint loading
    4. Log information about discovered sklearn modules during setup

    Note:
        - Only attributes that are sklearn RegressorMixin or ClassifierMixin instances are saved
        - Private attributes (starting with '_') are ignored
        - The callback will raise an error if a sklearn model name conflicts with existing checkpoint keys
    """

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        sklearn_modules = _get_sklearn_modules(pl_module)
        stats = []
        for name, module in sklearn_modules.items():
            stats.append((name, module.__str__(), type(module)))
        headers = ["Module", "Name", "Type"]
        logging.info("Setting up SklearnCheckpoint callback!")
        logging.info("Sklearn Modules:")
        logging.info(f"\n{tabulate(stats, headers, tablefmt='heavy_outline')}")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for non PyTorch modules to save... 🔧")
        modules = _get_sklearn_modules(pl_module)
        for name, module in modules.items():
            if name in checkpoint:
                raise RuntimeError(
                    f"Can't pickle {name}, already present in checkpoint"
                )
            checkpoint[name] = module
            logging.info(f"Saving non PyTorch system: {name} 🔧")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for non PyTorch modules to load... 🔧")
        if not SKLEARN_AVAILABLE:
            return
        for name, item in checkpoint.items():
            if isinstance(item, RegressorMixin) or isinstance(item, ClassifierMixin):
                setattr(pl_module, name, item)
                logging.info(f"Loading non PyTorch system: {name} 🔧")


def _contains_sklearn_module(item):
    if not SKLEARN_AVAILABLE:
        return False
    if isinstance(item, RegressorMixin) or isinstance(item, ClassifierMixin):
        return True
    if isinstance(item, list):
        return np.any([_contains_sklearn_module(m) for m in item])
    if isinstance(item, dict):
        return np.any([_contains_sklearn_module(m) for m in item.values()])
    return False


def _get_sklearn_modules(module):
    modules = dict()
    for name, item in vars(module).items():
        if name[0] == "_":
            continue
        item = getattr(module, name)
        if _contains_sklearn_module(item):
            modules[name] = item
    return modules


class StrictCheckpointCallback(Callback):
    """A PyTorch Lightning callback that controls strict checkpoint loading behavior."""

    def __init__(self, strict: bool = True):
        super().__init__()
        self.strict = strict
        logger.info(f"StrictCheckpointCallback initialized with strict={self.strict}")

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Called when loading a checkpoint."""
        if self.strict:
            return

        logger.info("=" * 80)
        logger.info("Processing checkpoint with strict=False")
        logger.info("=" * 80)

        if "state_dict" not in checkpoint:
            logger.warning("No 'state_dict' found in checkpoint.")
            return

        checkpoint_state_dict = checkpoint["state_dict"]
        model_state_dict = pl_module.state_dict()

        # Track statistics
        matched_keys = []
        missing_in_checkpoint = []
        missing_in_model = []
        shape_mismatches = []

        # Build the new filtered state dict
        filtered_state_dict = {}

        # 1. Check all model keys
        for key in model_state_dict.keys():
            if key in checkpoint_state_dict:
                # Key exists in both
                if checkpoint_state_dict[key].shape == model_state_dict[key].shape:
                    # Shapes match - use checkpoint value
                    filtered_state_dict[key] = checkpoint_state_dict[key]
                    matched_keys.append(key)
                else:
                    # Shape mismatch - use model's current value
                    filtered_state_dict[key] = model_state_dict[key]
                    shape_mismatches.append(
                        {
                            "key": key,
                            "checkpoint_shape": checkpoint_state_dict[key].shape,
                            "model_shape": model_state_dict[key].shape,
                        }
                    )
                    logger.warning(
                        f"❌ Shape mismatch for '{key}': "
                        f"checkpoint={checkpoint_state_dict[key].shape}, "
                        f"model={model_state_dict[key].shape} - Using model's current value"
                    )
            else:
                # Key missing in checkpoint - use model's current value
                filtered_state_dict[key] = model_state_dict[key]
                missing_in_checkpoint.append(key)
                logger.warning(
                    f"⚠️  Parameter missing in checkpoint: '{key}' - Using model's current value"
                )

        # 2. Check for extra keys in checkpoint
        for key in checkpoint_state_dict.keys():
            if key not in model_state_dict:
                missing_in_model.append(key)
                logger.warning(
                    f"⚠️  Parameter in checkpoint but not in model: '{key}' - SKIPPING"
                )

        # Update checkpoint with filtered state dict
        checkpoint["state_dict"] = filtered_state_dict

        # Clear optimizer states if there were any mismatches
        if missing_in_model or shape_mismatches or missing_in_checkpoint:
            if "optimizer_states" in checkpoint:
                logger.warning(
                    "🗑️  Clearing optimizer states due to parameter mismatches."
                )
                checkpoint.pop("optimizer_states", None)

            if "lr_schedulers" in checkpoint:
                logger.warning("🗑️  Clearing learning rate scheduler states.")
                checkpoint.pop("lr_schedulers", None)

        # Print summary
        self._print_summary(
            matched_keys,
            missing_in_checkpoint,
            missing_in_model,
            shape_mismatches,
            len(model_state_dict),
        )

    def _print_summary(
        self,
        matched_keys,
        missing_in_checkpoint,
        missing_in_model,
        shape_mismatches,
        total_model_params,
    ):
        logger.info("-" * 80)
        logger.info(f"✅ Successfully matched parameters: {len(matched_keys)}")

        if missing_in_checkpoint:
            logger.warning(
                f"⚠️  Parameters missing in checkpoint: {len(missing_in_checkpoint)}"
            )

        if missing_in_model:
            logger.warning(
                f"⚠️  Extra parameters in checkpoint: {len(missing_in_model)}"
            )

        if shape_mismatches:
            logger.warning(f"❌ Shape mismatches found: {len(shape_mismatches)}")

        loaded_percentage = (
            (len(matched_keys) / total_model_params * 100)
            if total_model_params > 0
            else 0
        )
        logger.info(
            f"📊 Checkpoint loading coverage: {loaded_percentage:.2f}% ({len(matched_keys)}/{total_model_params})"
        )
        logger.info("=" * 80)
