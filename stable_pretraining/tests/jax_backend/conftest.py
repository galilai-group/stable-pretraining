"""Collection guard for the JAX backend test suite.

The entire ``tests/jax`` package is ignored at collection time when the JAX
stack is not installed, so the default torch CI run is unaffected. Install the
backend with ``pip install -e ".[jax]"`` to exercise these tests.
"""

import importlib.util

_HAS_JAX = all(
    importlib.util.find_spec(mod) is not None for mod in ("jax", "flax", "optax")
)

# When jax/flax/optax are missing, skip-collect every test module in this dir
# (avoids import errors that a module-level ``pytest.importorskip`` would still
# surface during collection of helper imports).
collect_ignore_glob = [] if _HAS_JAX else ["test_*.py"]
