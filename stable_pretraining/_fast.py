"""Process-global "fast mode" flag, toggled by :func:`stable_pretraining.make_it_fast`.

Kept in a tiny leaf module (no heavy imports) so that ``Manager`` and
``create_optimizer`` can read the flag without importing the package
``__init__`` — which would risk circular imports during package construction.

``make_it_fast()`` sets this. Readers consume it at the moment their decision is
actually made:

- :class:`stable_pretraining.Manager` reads it when it builds the Lightning
  ``Trainer`` (to default ``bf16-mixed`` precision + tuned DDP comm).
- :func:`stable_pretraining.optim.create_optimizer` reads it when it constructs
  the optimizer (to enable the fused CUDA kernel where supported).

In every case the user's explicit configuration wins; the flag only fills in
values the user left unset.
"""

_ENABLED = False


def set_enabled(value: bool = True) -> None:
    """Toggle process-global fast mode (called by ``make_it_fast``)."""
    global _ENABLED
    _ENABLED = bool(value)


def enabled() -> bool:
    """Return whether ``make_it_fast()`` has been called this process."""
    return _ENABLED
