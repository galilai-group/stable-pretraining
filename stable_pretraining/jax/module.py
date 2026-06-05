"""Flax-NNX analogue of :class:`stable_pretraining.Module`.

The JAX backend mirrors the torch/Lightning one: users define a stateless
``forward(self, batch, stage)`` function that returns a ``dict`` (containing
``"loss"`` during training), and :class:`Module` binds it to a Flax-NNX
module holding the backbone/projector/loss as attributes. Keeping the same
contract means the same forward-function *shape* and the same dict-flow
design carry over from the torch path — only the ops inside the loss and the
training loop differ (functional grads + optax instead of ``loss.backward()``).
"""

from typing import Any, Callable

from flax import nnx


class Module(nnx.Module):
    """Flax-NNX module that binds a user ``forward`` function and submodules.

    The forward function has the same signature as on the torch backend::

        def my_forward(self, batch: dict, stage: str) -> dict: ...

    ``self`` is this :class:`Module` instance (so ``self.backbone``,
    ``self.projector``, ``self.<loss>`` are all reachable), ``batch`` is a
    ``dict`` of arrays, and ``stage`` is ``"fit"``, ``"validate"``, ``"test"``
    or ``"predict"``. The function must return a ``dict``; during ``"fit"``
    that dict must contain ``"loss"`` (a scalar).

    Args:
        forward: The stateless forward callable, bound to this instance.
        optim: Optional optimizer spec consumed by
            :func:`stable_pretraining.jax.optim.create_optimizer`. Defaults to
            AdamW when omitted. Set to ``None`` to disable optimization
            (eval-only modules).
        transform: Optional on-device augmentation callable
            ``(module, batch) -> batch`` applied during ``"fit"`` *before* the
            forward function. Use it to generate augmentation realisations on
            the accelerator and populate ``batch["views"]`` (see
            :func:`stable_pretraining.jax.augment.two_view_transform`). Draws
            randomness from ``self.aug_rngs``.
        aug_seed: Seed for the augmentation RNG (``self.aug_rngs``) used by
            ``transform``. Only relevant when ``transform`` is set.
        **submodules: Named submodules / attributes stored on the instance
            (e.g. ``backbone=...``, ``projector=...``, ``simclr_loss=...``).

    Note:
        Mirrors :class:`stable_pretraining.Module`; see ``AGENTS.md`` for the
        shared design rationale (stateless forward, dict flow, callbacks read
        the returned dict).
    """

    def __init__(
        self,
        *,
        forward: Callable,
        optim: Any = "adamw",
        transform: Callable = None,
        aug_seed: int = 0,
        hparams: dict = None,
        **submodules: Any,
    ):
        if not callable(forward):
            raise ValueError("`forward` must be callable.")
        # Plain (non-Variable) attributes are treated as static by NNX.
        self._forward = forward
        self.optim = optim
        self._transform = transform
        # Hyperparameters logged via ``log_hyperparams`` at fit start (same as
        # the torch Module's ``hparams`` — flows to WandB/registry/etc.).
        self.hparams = dict(hparams) if hparams else {}
        self.training = True
        self._logs: dict[str, Any] = {}
        # RNG state for on-device augmentation; nnx threads it through jit and
        # the optimizer (wrt=nnx.Param) leaves it untouched.
        if transform is not None:
            self.aug_rngs = nnx.Rngs(aug_seed)
        for name, value in submodules.items():
            setattr(self, name, value)

    def compute(self, batch: dict, stage: str) -> tuple[dict, dict]:
        """Run the bound forward function and collect logged scalars.

        Args:
            batch: Input batch dict.
            stage: One of ``"fit"``, ``"validate"``, ``"test"``, ``"predict"``.

        Returns:
            tuple[dict, dict]: ``(state, logs)`` where ``state`` is the dict
            returned by ``forward`` and ``logs`` are the values recorded via
            :meth:`log`. Both are returned as JAX values so the trainer can use
            ``state`` as ``has_aux`` output of ``nnx.value_and_grad``.
        """
        self.training = stage == "fit"
        # nnx's train()/eval() propagate the BatchNorm running-average and
        # Dropout deterministic flags to every submodule — the analogue of
        # torch's ``self.training`` flowing through ``nn.Module`` children.
        if self.training:
            self.train()
        else:
            self.eval()
        # On-device augmentation: populate batch["views"] before the forward,
        # mirroring how the torch MultiViewTransform fills the batch dict.
        if self._transform is not None and self.training:
            batch = self._transform(self, batch)
        self._logs = {}
        state = self._forward(self, batch, stage)
        if not isinstance(state, dict):
            raise ValueError(
                f"forward must return a dict, got {type(state)}. "
                "Return a dict so callbacks can read intermediate values."
            )
        logs = self._logs
        # Reset so the attribute never carries traced arrays across jit traces.
        self._logs = {}
        return state, logs

    def log(self, name: str, value: Any, **kwargs: Any) -> None:
        """Record a scalar for the trainer to emit (mirrors ``LightningModule.log``).

        Extra keyword arguments (``on_step``, ``on_epoch``, ``sync_dist``, …)
        are accepted for signature compatibility with the torch forward
        functions and ignored on this backend.

        Args:
            name: Metric name (e.g. ``"train/loss"``).
            value: Scalar value to log.
            **kwargs: Ignored; present for torch-forward compatibility.
        """
        self._logs[name] = value
