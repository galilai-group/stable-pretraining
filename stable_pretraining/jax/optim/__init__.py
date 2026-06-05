"""Optimizer / scheduler factories for the JAX backend (thin optax wrappers).

Mirrors :mod:`stable_pretraining.optim` but returns ``optax`` gradient
transformations instead of ``torch.optim`` objects. The accepted spec forms
match the torch side as closely as optax allows: a string name, a dict with a
``type`` key, a ready-made ``optax.GradientTransformation``, or a zero-arg
factory/partial.
"""

from typing import Any, Callable, Union

import optax

_OPTIMIZERS: dict[str, Callable[..., optax.GradientTransformation]] = {
    "adamw": optax.adamw,
    "adam": optax.adam,
    "sgd": optax.sgd,
    "lars": optax.lars,
    "lamb": optax.lamb,
}


def create_optimizer(
    spec: Union[str, dict, optax.GradientTransformation, Callable, None],
    learning_rate: float = 1e-3,
) -> optax.GradientTransformation:
    """Build an ``optax`` optimizer from a flexible spec.

    Args:
        spec: One of:

            * ``optax.GradientTransformation`` — returned unchanged.
            * zero-arg callable / ``functools.partial`` — called and returned.
            * ``str`` — optimizer name (``"adamw"``, ``"sgd"``, ``"lars"``, …).
            * ``dict`` — ``{"type": "adamw", "learning_rate": 1e-3, ...}``.
        learning_rate: Default learning rate when the spec omits one.

    Returns:
        optax.GradientTransformation: The configured optimizer.

    Raises:
        ValueError: If a string/dict names an unknown optimizer type.
    """
    if isinstance(spec, optax.GradientTransformation):
        return spec
    if callable(spec) and not isinstance(spec, dict):
        return spec()
    if spec is None:
        spec = {}
    if isinstance(spec, str):
        spec = {"type": spec}

    cfg = dict(spec)
    name = str(cfg.pop("type", "adamw")).lower()
    if name not in _OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. Available: {sorted(_OPTIMIZERS)}."
        )
    cfg.setdefault("learning_rate", learning_rate)
    return _OPTIMIZERS[name](**cfg)


def warmup_cosine_schedule(
    base_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    end_lr: float = 0.0,
) -> optax.Schedule:
    """Linear warmup followed by cosine decay (the SSL default schedule).

    Args:
        base_lr: Peak learning rate reached at the end of warmup.
        total_steps: Total number of optimization steps.
        warmup_steps: Number of linear-warmup steps from ``0`` to ``base_lr``.
        end_lr: Final learning rate at ``total_steps``.

    Returns:
        optax.Schedule: A callable mapping step -> learning rate.
    """
    if warmup_steps <= 0:
        return optax.cosine_decay_schedule(
            base_lr, decay_steps=max(total_steps, 1), alpha=end_lr / max(base_lr, 1e-12)
        )
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, 1),
        end_value=end_lr,
    )


__all__: list[Any] = ["create_optimizer", "warmup_cosine_schedule"]
