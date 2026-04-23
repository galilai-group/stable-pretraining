from typing import Dict, List

import torch
from torch import nn
from lightning.pytorch.callbacks import Callback
from loguru import logger

from .utils import log_header


class LogUnusedParametersOnce(Callback):
    """Lightning callback that logs parameters which do NOT receive gradients.

    - Registers hooks on all leaf parameters (requires_grad=True).
    - After the first backward pass, logs unused parameters via loguru.
    - Removes all hooks and disables itself for the rest of training.

    Works with both automatic and manual optimization.
    """

    def __init__(self, verbose: bool = None):
        super().__init__()
        from .utils import resolve_verbose

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._used_flags: Dict[nn.Parameter, bool] = {}
        self._enabled: bool = True
        self._verbose = resolve_verbose(verbose)
        self._backward_called: bool = False

    def _register_hooks(self, model: nn.Module):
        """Attach hooks to all leaf parameters that require gradient."""
        assert not self._hooks, "Hooks already registered"
        self._backward_called = False

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if not p.is_leaf:
                continue

            self._used_flags[p] = False

            def make_hook(param):
                def hook(grad):
                    self._used_flags[param] = True
                    self._backward_called = True

                return hook

            h = p.register_hook(make_hook(p))
            self._hooks.append(h)

        if self._verbose:
            log_header("LogUnusedParametersOnce")
            logger.info(
                f"  registered hooks on {len(self._used_flags)} leaf parameters"
            )

    def _remove_hooks(self):
        """Remove all hooks and clear state."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._used_flags.clear()

    def _report_and_disable(self, pl_module: nn.Module):
        """Report unused parameters to loguru and disable further tracking."""
        name_by_param = {p: n for n, p in pl_module.named_parameters()}

        unused_names = [
            name_by_param[p] for p, used in self._used_flags.items() if not used
        ]

        if not unused_names:
            logger.success(
                "✓ all tracked parameters received gradients on the first backward pass"
            )
        else:
            logger.warning(
                "! the following parameters did NOT receive "
                "gradients on the first backward pass (potentially causing "
                "Lightning's 'unused parameters' error):"
            )
            for name in unused_names:
                logger.warning(f"  {name}")

        self._remove_hooks()
        self._enabled = False
        if self._verbose:
            logger.info("  hooks removed, callback disabled")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Register hooks right before the first training batch starts."""
        if not self._enabled:
            return

        if trainer.global_step == 0 and batch_idx == 0:
            self._remove_hooks()
            self._used_flags.clear()
            self._register_hooks(pl_module)

    def on_after_backward(self, trainer, pl_module):
        """After backward pass, report unused params (automatic optimization)."""
        if not self._enabled:
            return

        self._report_and_disable(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Fallback for manual optimization - check after first batch completes."""
        if not self._enabled:
            return

        # If hooks are still registered, on_after_backward wasn't called (manual optimization)
        if len(self._hooks) == 0:
            return

        if not self._backward_called:
            logger.warning(
                "! no gradient hooks fired during the first "
                "training step. This likely means backward() was never called. "
                "Cannot verify unused parameters."
            )
            self._remove_hooks()
            self._enabled = False
            if self._verbose:
                logger.info("  hooks removed, callback disabled")
        else:
            self._report_and_disable(pl_module)
