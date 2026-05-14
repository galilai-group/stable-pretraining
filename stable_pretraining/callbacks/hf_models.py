import inspect
import os
import shutil
import time
import traceback
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
        per_step (bool): If True, save into a fresh ``step_{global_step}`` subdir
            on every checkpoint trigger (one snapshot per save). If False
            (default), save into a single ``last`` subdir that is overwritten
            on every trigger — only the most recent snapshot is kept on disk,
            no per-step accumulation, less I/O, simpler downstream load.
        verbose (bool): If True, logs a discovery table at the start of
            training. Default is True.
        raise_on_error (bool): If True, propagate export failures so training
            halts (legacy behavior). If False (default), log the full traceback
            via ``logger.exception`` and continue training — checkpoint export
            is auxiliary and should never crash a run.

    Example:
        >>> # Setup your model with a HF submodule
        >>> class MySystem(pl.LightningModule):
        ...     def __init__(self, config):
        ...         super().__init__()
        ...         self.backbone = MyCustomHFModel(config)  # Inherits PreTrainedModel
        >>> # Add callback to trainer (default: only "last" subdir, overwritten each save)
        >>> hf_cb = HuggingFaceCheckpointCallback(save_dir="checkpoints/hf_models")
        >>> # Or keep one folder per training step
        >>> hf_cb = HuggingFaceCheckpointCallback(save_dir="...", per_step=True)
        >>> trainer = pl.Trainer(callbacks=[hf_cb])
        >>> trainer.fit(model, dataloader)

        >>> # Later, load without your source code library:
        >>> from transformers import AutoModel
        >>> # per_step=False (default):
        >>> model = AutoModel.from_pretrained(
        ...     "checkpoints/hf_models/last/backbone", trust_remote_code=True
        ... )
        >>> # per_step=True:
        >>> model = AutoModel.from_pretrained(
        ...     "checkpoints/hf_models/step_5000/backbone", trust_remote_code=True
        ... )
    """

    def __init__(
        self,
        save_dir: str = "hf_exports",
        per_step: bool = False,
        verbose: bool = None,
        raise_on_error: bool = False,
    ):
        super().__init__()
        from .utils import resolve_verbose

        self.save_dir = Path(save_dir)
        self.per_step = bool(per_step)
        self.raise_on_error = bool(raise_on_error)
        self.verbose = resolve_verbose(verbose)
        log_header("HuggingFaceCheckpoint")
        logger.info(f"  save_dir: <cyan>{self.save_dir}</cyan>")
        logger.info(
            f"  per_step: <cyan>{self.per_step}</cyan>"
            f"  (False ⇒ overwrite single 'last/' subdir; True ⇒ keep step_N/)"
        )

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

        Triggered by Lightning's checkpointing logic. Only rank 0 performs
        the export to avoid filesystem race conditions. By default
        (``per_step=False``), the export overwrites a single ``last/``
        subdir each call. With ``per_step=True``, each call writes to a
        fresh ``step_{global_step}/`` subdir.

        Robustness: every filesystem op is retried with parents=True and
        wrapped in a guard. By default (``raise_on_error=False``) any
        failure is logged via ``logger.exception`` (with full traceback)
        and training continues — checkpoint export is auxiliary and must
        not kill the run. Set ``raise_on_error=True`` to restore the old
        crash-on-failure behavior.
        """
        if trainer.global_rank != 0:
            return
        try:
            self._do_export(trainer, pl_module)
        except Exception:
            # Loud, full-traceback log — visible in any reasonable log capture.
            logger.exception(
                "HuggingFaceCheckpointCallback export FAILED "
                "(auxiliary callback — training continues; "
                "set raise_on_error=True to halt instead)."
            )
            # Belt-and-suspenders for environments where loguru's exception
            # hook isn't routed to stdout/stderr (some submitit/Lightning
            # log captures): also dump to stderr.
            traceback.print_exc()
            if self.raise_on_error:
                raise

    def _do_export(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        step = trainer.global_step
        # Resolve relative save_dir against trainer.default_root_dir.
        # When running outside the Manager, default_root_dir may be CWD —
        # prefer cache_dir if configured so we never pollute CWD.
        from stable_pretraining._config import get_config

        root = trainer.default_root_dir
        cfg = get_config()
        if cfg.cache_dir is not None and root == str(Path().resolve()):
            root = cfg.cache_dir
        save_dir = self.save_dir
        if not save_dir.is_absolute():
            save_dir = Path(root) / save_dir
        # Always make sure the parent save_dir exists. mkdir with
        # parents=True + exist_ok=True is idempotent and safe.
        save_dir.mkdir(parents=True, exist_ok=True)

        # Per_step=False (default) keeps a single rolling "last/" snapshot;
        # per_step=True keeps one folder per save.
        subdir_name = f"step_{step}" if self.per_step else "last"
        hf_step_dir = save_dir / subdir_name

        # Race-safe write protocol — multiple processes may legitimately
        # write to the *same* ``save_dir`` (e.g. an array job sharing a
        # global HF export cache). The previous "rmtree-then-mkdir-then-
        # write" sequence had three races:
        #   1. Job A's ``rmtree`` runs while Job B is mid-``save_pretrained``
        #      → ``FileNotFoundError`` for B
        #   2. Both jobs see ``exists() == True`` and both ``rmtree``;
        #      second one fails with ``FileNotFoundError``
        #   3. Job A clears, Job B reads (e.g. AutoModel.from_pretrained)
        #      between A's rmtree and A's re-creation → missing files
        #
        # Fix: every process writes into a *unique* temp dir, then
        # atomically renames it into place at the end. The temp dir lives
        # alongside the target so the rename is same-FS atomic. Between
        # the move-aside and install steps a reader can still see "no
        # last/" briefly, but never a half-written one.
        tmp_uid = f"{os.getpid()}.{int(time.monotonic_ns()):x}"
        tmp_dir = save_dir / f".{subdir_name}.tmp.{tmp_uid}"
        # Make sure no stale temp from a previously-killed run blocks us.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[hf_export] writing to temp dir {tmp_dir}")

        hf_submodules = self._get_hf_submodules(pl_module)

        try:
            for name, model in hf_submodules.items():
                model_save_path = tmp_dir / name
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

            # Commit: move the existing target aside (if any), install the
            # new one, then drop the old. Both renames are atomic;
            # ``ignore_errors`` handles the corner case of a concurrent
            # process having already moved/deleted the dirs.
            old_aside = save_dir / f".{subdir_name}.old.{tmp_uid}"
            try:
                if hf_step_dir.exists():
                    os.rename(hf_step_dir, old_aside)
            except FileNotFoundError:
                # Lost the race to a concurrent commit — that's fine, the
                # target slot is empty when we install ours next.
                pass
            try:
                os.rename(tmp_dir, hf_step_dir)
            except FileNotFoundError as e:
                logger.warning(
                    f"[hf_export] commit rename failed for {hf_step_dir}: {e}"
                )
                raise
            if old_aside.exists():
                shutil.rmtree(old_aside, ignore_errors=True)
            logger.debug(f"[hf_export] ✓ committed {hf_step_dir}")
        except BaseException:
            # On any failure leave the previous good export intact and
            # clean up our temp.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

            logger.success(
                f"Exported HF submodule '<green>{name}</green>' at step {step} -> {model_save_path}"
            )
