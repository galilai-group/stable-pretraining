"""Behavior + parity tests for the in-house rank-zero / seed utilities.

These replace direct use of ``lightning.pytorch.utilities.rank_zero`` and
``lightning.pytorch.seed_everything``. The key guarantee is **parity**: for the
inputs this codebase actually exercises, the in-house functions must behave
identically to Lightning's, so swapping the import is a no-op at runtime.

Rank resolution is intentionally a superset of Lightning's: it consults
``torch.distributed.get_rank()`` first (authoritative once a process group is
initialized), then the same environment-variable chain Lightning uses, then
defaults to ``0``. The env-var and default cases are tested for exact parity;
the dist-aware case is tested as the documented improvement.
"""

import os
import random
import warnings

import numpy as np
import pytest
import torch

from stable_pretraining.utils.distributed import (
    _DummyExperiment,
    get_rank,
    get_world_size,
    rank_zero_experiment,
    rank_zero_only,
    rank_zero_warn,
    seed_everything,
)

# Lightning references for parity assertions.
import lightning as pl
from lightning.fabric.utilities.rank_zero import _get_rank as _lightning_get_rank


_RANK_ENV_KEYS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")


@pytest.fixture
def clean_rank_env(monkeypatch):
    """Remove every rank-related env var so each case starts from a known state."""
    for key in _RANK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield monkeypatch


@pytest.mark.unit
class TestGetRank:
    """Rank/world-size resolution: dist group, then env chain, then default."""

    def test_defaults_to_zero_without_env_or_dist(self, clean_rank_env):
        assert get_rank() == 0
        assert get_world_size() == 1

    @pytest.mark.parametrize("key", _RANK_ENV_KEYS)
    def test_reads_each_env_key(self, clean_rank_env, key):
        clean_rank_env.setenv(key, "3")
        assert get_rank() == 3

    def test_env_priority_matches_lightning(self, clean_rank_env):
        """RANK wins over LOCAL_RANK wins over SLURM_PROCID wins over JSM."""
        clean_rank_env.setenv("JSM_NAMESPACE_RANK", "9")
        clean_rank_env.setenv("SLURM_PROCID", "7")
        clean_rank_env.setenv("LOCAL_RANK", "5")
        clean_rank_env.setenv("RANK", "1")
        assert get_rank() == 1
        # Parity: Lightning's resolver agrees on the same env.
        assert get_rank() == _lightning_get_rank()

    def test_priority_after_dropping_rank(self, clean_rank_env):
        clean_rank_env.setenv("SLURM_PROCID", "7")
        clean_rank_env.setenv("LOCAL_RANK", "5")
        assert get_rank() == 5
        assert get_rank() == _lightning_get_rank()

    def test_dist_get_rank_takes_precedence(self, clean_rank_env, monkeypatch):
        """When a process group is initialized, the true rank wins over env."""
        clean_rank_env.setenv("RANK", "2")
        monkeypatch.setattr(
            "stable_pretraining.utils.distributed.is_dist_avail_and_initialized",
            lambda: True,
        )
        monkeypatch.setattr(
            "stable_pretraining.utils.distributed.dist.get_rank", lambda: 4
        )
        monkeypatch.setattr(
            "stable_pretraining.utils.distributed.dist.get_world_size", lambda: 8
        )
        assert get_rank() == 4
        assert get_world_size() == 8


@pytest.mark.unit
class TestRankZeroOnly:
    """The ``rank_zero_only`` decorator gates execution on global rank 0."""

    def test_runs_on_rank_zero(self, clean_rank_env):
        calls = []

        @rank_zero_only
        def f(x):
            calls.append(x)
            return x * 2

        assert f(21) == 42
        assert calls == [21]

    def test_skips_on_nonzero_rank_returns_default(self, clean_rank_env):
        clean_rank_env.setenv("RANK", "1")
        calls = []

        @rank_zero_only
        def f(x):
            calls.append(x)
            return x

        assert f(5) is None  # default when not provided
        assert calls == []

    def test_custom_default_on_nonzero_rank(self, clean_rank_env):
        clean_rank_env.setenv("RANK", "2")

        def f():
            return "real"

        wrapped = rank_zero_only(f, "fallback")
        assert wrapped() == "fallback"

    def test_preserves_wrapped_metadata(self, clean_rank_env):
        @rank_zero_only
        def my_func():
            """My docstring."""

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_rank_resolved_per_call_not_at_decoration(self, clean_rank_env):
        """Rank is read when the function is CALLED, not at decoration.

        So a rank assigned after import (the real DDP case) is still honored.
        """
        calls = []

        @rank_zero_only
        def f():
            calls.append(1)

        clean_rank_env.setenv("RANK", "3")  # becomes non-zero only now
        f()
        assert calls == []  # skipped because rank is read at call time


@pytest.mark.unit
class TestRankZeroWarn:
    """``rank_zero_warn`` emits only on rank 0."""

    def test_warns_on_rank_zero(self, clean_rank_env):
        with pytest.warns(UserWarning, match="hello"):
            rank_zero_warn("hello")

    def test_silent_on_nonzero_rank(self, clean_rank_env):
        clean_rank_env.setenv("RANK", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would raise
            rank_zero_warn("should not fire")  # no warning emitted → no raise


@pytest.mark.unit
class TestRankZeroExperiment:
    """``rank_zero_experiment`` returns the real experiment only on rank 0."""

    def test_returns_real_on_rank_zero(self, clean_rank_env):
        class Logger:
            @property
            @rank_zero_experiment
            def experiment(self):
                return "REAL"

        assert Logger().experiment == "REAL"

    def test_returns_dummy_on_nonzero_rank(self, clean_rank_env):
        clean_rank_env.setenv("RANK", "1")

        class Logger:
            @property
            @rank_zero_experiment
            def experiment(self):
                return "REAL"

        exp = Logger().experiment
        assert isinstance(exp, _DummyExperiment)

    def test_dummy_experiment_swallows_everything(self):
        d = _DummyExperiment()
        # Arbitrary attribute access returns a no-op callable that swallows any
        # call (this is exactly how Lightning's _DummyExperiment behaves — a
        # single level of attribute access, then a no-op call).
        assert d.log(1, 2, foo="bar") is None
        assert d.add_image("x", torch.zeros(1)) is None
        assert callable(d.whatever)
        # Indexing returns itself (enables experiment[0].add_image(...)).
        assert d[0] is d
        assert d[3].log() is None
        d["key"] = 123  # __setitem__ is a no-op, must not raise


@pytest.mark.unit
class TestSeedEverything:
    """``seed_everything`` reproducibility, Lightning parity, and env contract."""

    def _draws(self):
        return (
            torch.randn(5).tolist(),
            [random.random() for _ in range(3)],
            np.random.rand(3).tolist(),
        )

    def test_returns_seed(self, monkeypatch):
        assert seed_everything(123) == 123

    def test_reproducible(self):
        seed_everything(42)
        a = self._draws()
        seed_everything(42)
        b = self._draws()
        assert a == b

    def test_parity_with_lightning(self):
        """Same seed must produce byte-identical draws as Lightning's seed_everything.

        Covers torch / numpy / random — this is the whole point of the swap.
        """
        seed_everything(2024)
        ours = self._draws()
        pl.seed_everything(2024)
        theirs = self._draws()
        assert ours == theirs

    def test_sets_env_vars(self, monkeypatch):
        monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
        monkeypatch.delenv("PL_SEED_WORKERS", raising=False)
        seed_everything(7, workers=True)
        # The PL_* env contract is preserved so Lightning's data connector
        # still seeds dataloader workers.
        assert os.environ["PL_GLOBAL_SEED"] == "7"
        assert os.environ["PL_SEED_WORKERS"] == "1"

    def test_workers_false_sets_zero(self, monkeypatch):
        monkeypatch.delenv("PL_SEED_WORKERS", raising=False)
        seed_everything(7, workers=False)
        assert os.environ["PL_SEED_WORKERS"] == "0"

    def test_none_reads_env_then_defaults_to_zero(self, monkeypatch):
        monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
        assert seed_everything(None, verbose=False) == 0
        # Now PL_GLOBAL_SEED is set to "0"; a subsequent None reads it back.
        monkeypatch.setenv("PL_GLOBAL_SEED", "55")
        assert seed_everything(None, verbose=False) == 55

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            seed_everything(2**32 + 5)
        with pytest.raises(ValueError):
            seed_everything(-1)
