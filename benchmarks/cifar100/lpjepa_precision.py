"""FP16 precision for the LpJEPA benchmark's three-optimizer training loop.

Lightning normally updates the shared GradScaler after every optimizer. These
optimizers consume gradients from one backward, so the scale must stay fixed
until all three step attempts finish. Keep Lightning's optimizer implementation
and defer only its calls to GradScaler.update().
"""

from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.plugins.precision import MixedPrecision


class JointStepGradScaler(torch.amp.GradScaler):
    """Update the scale once per complete group of optimizer step attempts.

    Args:
        optimizers_per_step: Number of optimizers sharing one backward pass.
        **kwargs: Arguments passed to the CUDA gradient scaler.
    """

    def __init__(self, optimizers_per_step: int = 3, **kwargs: Any) -> None:
        if optimizers_per_step < 1:
            raise ValueError("optimizers_per_step must be positive")
        super().__init__("cuda", **kwargs)
        self.optimizers_per_step = optimizers_per_step
        self._pending_updates = 0
        self._awaiting_update = False
        self._optimizer_ids = None
        self._backward_active = False
        self.completed_cycles = 0

    def assert_boundary(self) -> None:
        """Require all optimizer steps from the preceding backward to be complete."""
        if self._backward_active or self._pending_updates or self._awaiting_update:
            raise RuntimeError("The joint AMP optimizer cycle is incomplete")

    def scale(self, outputs: Any) -> Any:
        """Start one backward pass with the current scale.

        Args:
            outputs: Loss tensor or tensors accepted by ``GradScaler.scale``.

        Returns:
            Scaled tensors with the same structure as the inputs.
        """
        self.assert_boundary()
        self._backward_active = True
        return super().scale(outputs)

    def bind_optimizers(self, optimizers: list[torch.optim.Optimizer]) -> None:
        """Record the disjoint optimizers in their expected stepping order.

        Args:
            optimizers: Ordinary non-fused optimizers sharing one backward pass.
        """
        self.assert_boundary()
        if len(optimizers) != self.optimizers_per_step:
            raise RuntimeError("Unexpected number of optimizers for joint AMP")
        ids = tuple(id(optimizer) for optimizer in optimizers)
        if len(set(ids)) != len(ids):
            raise RuntimeError("Joint AMP requires distinct optimizers")
        seen = set()
        for optimizer in optimizers:
            if isinstance(optimizer, torch.optim.LBFGS) or getattr(
                optimizer, "_step_supports_amp_scaling", False
            ):
                raise RuntimeError("Joint AMP requires ordinary non-fused optimizers")
            params = {
                id(param)
                for group in optimizer.param_groups
                for param in group["params"]
            }
            if not params or seen.intersection(params):
                raise RuntimeError("Joint AMP requires disjoint optimizer parameters")
            seen.update(params)
        self._optimizer_ids = ids

    def step(self, optimizer: torch.optim.Optimizer, *args: Any, **kwargs: Any) -> Any:
        """Step the next optimizer with native unscaling and overflow handling.

        Args:
            optimizer: Next optimizer in the registered order.
            *args: Positional arguments forwarded to the optimizer step.
            **kwargs: Keyword arguments forwarded to the optimizer step.

        Returns:
            The value returned by the native scaler's optimizer step.
        """
        if not self.is_enabled() or self._optimizer_ids is None:
            raise RuntimeError("Joint AMP requires an enabled, bound CUDA scaler")
        if not self._backward_active:
            raise RuntimeError("Optimizer steps require one scaled backward")
        if self._awaiting_update:
            raise RuntimeError("Expected scaler.update() after the optimizer step")
        if id(optimizer) != self._optimizer_ids[self._pending_updates]:
            raise RuntimeError("Unexpected optimizer order in the joint AMP cycle")
        result = super().step(optimizer, *args, **kwargs)
        self._awaiting_update = True
        return result

    def update(self, new_scale: float | torch.Tensor | None = None) -> None:
        """Defer the native scale update until every optimizer has stepped.

        Args:
            new_scale: Must be ``None``; explicit overrides are unsupported.
        """
        if new_scale is not None:
            raise RuntimeError("Explicit scale overrides are unsupported in this run")
        if not self._awaiting_update:
            raise RuntimeError("scaler.update() requires an optimizer step attempt")
        self._pending_updates += 1
        if self._pending_updates == self.optimizers_per_step:
            super().update()
            self._pending_updates = 0
            self._backward_active = False
            self.completed_cycles += 1
        self._awaiting_update = False

    def state_dict(self) -> dict[str, Any]:
        """Return native scaler state between complete optimizer cycles."""
        self.assert_boundary()
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore native scaler state between complete optimizer cycles.

        Args:
            state_dict: State saved by ``GradScaler.state_dict``.
        """
        self.assert_boundary()
        return super().load_state_dict(state_dict)


class JointStepMixedPrecision(MixedPrecision):
    """Standard Lightning 16-mixed precision with one update per joint step.

    Pass via Trainer(plugins=[JointStepMixedPrecision()]); omit precision=.
    Add JointStepGuard() for this SPT run's configuration and boundary checks.

    Args:
        optimizers_per_step: Number of optimizers sharing one backward pass.
        **scaler_kwargs: Arguments passed to the CUDA gradient scaler.
    """

    def __init__(self, optimizers_per_step: int = 3, **scaler_kwargs: Any) -> None:
        super().__init__(
            "16-mixed",
            "cuda",
            scaler=JointStepGradScaler(optimizers_per_step, **scaler_kwargs),
        )

    def pre_backward(
        self, tensor: torch.Tensor, module: LightningModule
    ) -> torch.Tensor:
        """Check the previous optimizer cycle before scaling the next loss.

        Args:
            tensor: Scalar loss for the shared backward pass.
            module: Lightning module using manual optimization.

        Returns:
            The loss after the precision plugin's backward preparation.
        """
        if module.automatic_optimization:
            raise RuntimeError("Joint AMP requires manual optimization")
        self.scaler.assert_boundary()
        return super().pre_backward(tensor, module)


class JointStepGuard(Callback):
    """Fail early if this run ceases to use its supported SPT optimizer loop."""

    def __init__(self) -> None:
        self._configuration_checked = False

    @staticmethod
    def _scaler(trainer: Trainer) -> JointStepGradScaler:
        plugin = trainer.precision_plugin
        if not isinstance(plugin, JointStepMixedPrecision):
            raise TypeError("JointStepGuard requires JointStepMixedPrecision")
        return plugin.scaler

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Bind the optimizers after Lightning restores any checkpoint state."""
        if pl_module.automatic_optimization:
            raise RuntimeError("Joint AMP requires manual optimization")
        scaler = self._scaler(trainer)
        if not scaler.is_enabled():
            raise RuntimeError("Joint AMP requires CUDA GradScaler")
        scaler.bind_optimizers(trainer.optimizers)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Check that this batch starts at a supported optimizer boundary."""
        scaler = self._scaler(trainer)
        scaler.assert_boundary()
        self._cycles_before_batch = scaler.completed_cycles
        if self._configuration_checked:
            return
        # SPT initializes these mappings in its on_train_start, after callbacks.
        names = [
            pl_module._optimizer_index_to_name[i]
            for i in range(len(trainer.optimizers))
        ]
        if any(pl_module._optimizer_frequencies[name] != 1 for name in names):
            raise RuntimeError("Joint AMP requires every optimizer frequency to be 1")
        if any(
            getattr(trainer, name, 1) != 1
            for name in ("accumulate_grad_batches", "accumulate_grad_batches_")
        ):
            raise RuntimeError("Joint AMP does not support gradient accumulation")
        if any(
            pl_module._optimizer_gradient_clip_val[name] not in (None, 0, 0.0)
            for name in names
        ):
            raise RuntimeError("This SPT joint AMP run requires clipping disabled")
        self._configuration_checked = True

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Require exactly one completed optimizer cycle for the batch."""
        scaler = self._scaler(trainer)
        scaler.assert_boundary()
        if scaler.completed_cycles != self._cycles_before_batch + 1:
            raise RuntimeError(
                "Expected exactly one joint AMP cycle per training batch"
            )

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Allow checkpoints only between complete optimizer cycles."""
        self._scaler(trainer).assert_boundary()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Require the final batch's optimizer cycle to be complete."""
        self._scaler(trainer).assert_boundary()
