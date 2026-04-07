import shutil
import inspect
from pathlib import Path
from typing import Dict, Any
from loguru import logger
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from transformers import PreTrainedModel
from .utils import log_header


class HuggingFaceCheckpointCallback(Callback):
    """Export HF-compatible checkpoints for PreTrainedModel submodules.

    Identifies submodules inheriting from Hugging Face's `PreTrainedModel`
    and exports them into standalone, "zero-knowledge" loadable HF directories.

    This callback automates the synchronization between Lightning training
    and the Hugging Face ecosystem, handling weight stripping (removing
    DDP/Lightning prefixes) and dependency copying.

    Args:
        save_dir (str): Root directory where HF models will be exported.
            Default is "hf_exports".
        verbose (bool): If True, logs a discovery table at the start of
            training. Default is True.

    Example:
        >>> # Setup your model with a HF submodule
        >>> class MySystem(pl.LightningModule):
        ...     def __init__(self, config):
        ...         super().__init__()
        ...         self.backbone = MyCustomHFModel(config)  # Inherits PreTrainedModel
        >>> # Add callback to trainer
        >>> hf_cb = HuggingFaceCheckpointCallback(save_dir="checkpoints/hf_models")
        >>> trainer = pl.Trainer(callbacks=[hf_cb])
        >>> trainer.fit(model, dataloader)

        >>> # Later, load without your source code library:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained(
        ...     "checkpoints/hf_models/step_5000/backbone", trust_remote_code=True
        ... )
    """

    def __init__(self, save_dir: str = "hf_exports", verbose: bool = None):
        super().__init__()
        from .utils import resolve_verbose

        self.save_dir = Path(save_dir)
        self.verbose = resolve_verbose(verbose)
        log_header("HuggingFaceCheckpoint")
        logger.info(f"  save_dir: <cyan>{self.save_dir}</cyan>")

    def _get_hf_submodules(
        self, pl_module: pl.LightningModule
    ) -> Dict[str, PreTrainedModel]:
        """Identifies top-level children that are instances of PreTrainedModel."""
        return {
            name: module
            for name, module in pl_module.named_children()
            if isinstance(module, PreTrainedModel)
        }

    def _log_discovery_table(self, submodules: Dict[str, PreTrainedModel]):
        """Renders a diagnostic table of discovered HF submodules using Loguru."""
        if not submodules:
            logger.warning(
                "! No Hugging Face (PreTrainedModel) submodules found in LightningModule."
            )
            return

        # Formatting a manual Markdown table for the console
        header = f"| {'Module Name':<18} | {'Class Type':<22} | {'Config Type':<22} |"
        sep = f"|{'-' * 20}|{'-' * 24}|{'-' * 24}|"

        logger.info("  HF Submodule Discovery Summary:")
        logger.info(sep)
        logger.info(header)
        logger.info(sep)
        for name, mod in submodules.items():
            logger.info(
                f"| {name:<18} | {mod.__class__.__name__:<22} | {mod.config.__class__.__name__:<22} |"
            )
        logger.info(sep)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Discovers modules and logs the status table once at start."""
        submodules = self._get_hf_submodules(pl_module)
        if self.verbose:
            self._log_discovery_table(submodules)

    def _copy_dependency_tree(self, model: PreTrainedModel, save_path: Path):
        """Copy source files so relative imports resolve in the export dir.

        Locates the source files for the model and its immediate neighbors
        to ensure relative imports (e.g. from .layers import X) resolve.
        """
        model_file = Path(inspect.getfile(model.__class__)).resolve()
        package_root = model_file.parent

        # Capture all python scripts in the model's directory
        # This handles siblings like 'pos_embed.py' or 'swiglu.py'
        for py_file in package_root.glob("*.py"):
            shutil.copy2(py_file, save_path / py_file.name)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ):
        """Create an atomic HF-compatible export for every found submodule.

        Triggered by Lightning's checkpointing logic.
        Only rank 0 performs the export to avoid filesystem race conditions.
        """
        if trainer.global_rank != 0:
            return
        step = trainer.global_step
        hf_step_dir = self.save_dir / f"step_{step}"

        # Ensure atomic overwrite: Clear directory if it exists
        if hf_step_dir.exists():
            logger.debug(f"Overwriting previous HF directory: {hf_step_dir}")
            shutil.rmtree(hf_step_dir)
        hf_step_dir.mkdir(parents=True, exist_ok=True)

        hf_submodules = self._get_hf_submodules(pl_module)

        for name, model in hf_submodules.items():
            model_save_path = hf_step_dir / name
            model_save_path.mkdir(parents=True, exist_ok=True)

            # Extract module/config filenames for the AutoModel map
            module_fn = Path(inspect.getfile(model.__class__)).stem
            config_fn = Path(inspect.getfile(model.config.__class__)).stem

            # Update auto_map so AutoModel knows which .py file contains the classes
            model.config.auto_map = {
                "AutoConfig": f"{config_fn}.{model.config.__class__.__name__}",
                "AutoModel": f"{module_fn}.{model.__class__.__name__}",
            }

            # 1. Save Weights & Config.json
            # Note: Using the model instance (not pl_module) strips all
            # lightning/DDP prefixes automatically.
            model.save_pretrained(model_save_path)

            # 2. Copy code dependencies
            self._copy_dependency_tree(model, model_save_path)

            logger.success(
                f"Exported HF submodule '<green>{name}</green>' at step {step} -> {model_save_path}"
            )
