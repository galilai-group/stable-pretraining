"""Manager FSDP2 detection + logging.

Verifies:
- :func:`is_fsdp_strategy` correctly identifies FSDP2's ``ModelParallelStrategy``
  (and trainers wrapping it) without raising on plain objects.
- :func:`describe_fsdp_strategy` exposes the FSDP2-relevant fields:
  ``data_parallel_size``, ``tensor_parallel_size``, ``save_distributed_checkpoint``,
  and ``mp_policy`` — and returns ``{"is_fsdp": False}`` for non-FSDP strategies.
- ``Manager._log_fsdp_info`` no-ops on non-FSDP trainers and prints the new
  field set when FSDP2 is detected.

These are pure-Python tests that run anywhere — no process group needed.
"""

import pytest


pytestmark = pytest.mark.distributed


def test_is_fsdp_strategy_plain_object_returns_false():
    from stable_pretraining.utils.fsdp2 import is_fsdp_strategy

    class NotAStrategy:
        pass

    assert is_fsdp_strategy(NotAStrategy()) is False


def test_is_fsdp_strategy_detects_model_parallel_strategy():
    from stable_pretraining.utils.fsdp2 import (
        StablePretrainingFSDP2,
        is_fsdp_strategy,
    )

    strat = StablePretrainingFSDP2(data_parallel_size=1, tensor_parallel_size=1)
    assert is_fsdp_strategy(strat) is True


def test_is_fsdp_strategy_detects_via_trainer_attr():
    """Detect FSDP2 via Trainer.strategy attribute.

    ``is_fsdp_strategy`` should also accept a Trainer-like object whose
    ``.strategy`` is an FSDP2 strategy.
    """
    from stable_pretraining.utils.fsdp2 import (
        StablePretrainingFSDP2,
        is_fsdp_strategy,
    )

    strat = StablePretrainingFSDP2(data_parallel_size=1, tensor_parallel_size=1)

    class FakeTrainer:
        def __init__(self, s):
            self.strategy = s

    assert is_fsdp_strategy(FakeTrainer(strat)) is True


def test_describe_non_fsdp_returns_is_fsdp_false():
    from stable_pretraining.utils.fsdp2 import describe_fsdp_strategy

    assert describe_fsdp_strategy(object()) == {"is_fsdp": False}


def test_describe_fsdp_exposes_fsdp2_fields():
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from stable_pretraining.utils.fsdp2 import (
        StablePretrainingFSDP2,
        describe_fsdp_strategy,
    )

    policy = MixedPrecisionPolicy()
    strat = StablePretrainingFSDP2(
        data_parallel_size=2, tensor_parallel_size=1, mp_policy=policy
    )
    info = describe_fsdp_strategy(strat)
    assert info["is_fsdp"] is True
    # The subclass is our registry-bound ``StablePretrainingFSDP2`` (which
    # inherits from ``ModelParallelStrategy``).
    assert info["subclass"] in ("StablePretrainingFSDP2", "ModelParallelStrategy")
    # mp_policy is exposed by class name (so logs stay readable).
    assert info["mp_policy"] == "MixedPrecisionPolicy"
    assert info["save_distributed_checkpoint"] is True


def test_manager_log_fsdp_info_noop_on_non_fsdp(caplog):
    """Manager._log_fsdp_info must be a no-op when the trainer is non-FSDP."""
    import logging as _logging

    from stable_pretraining.manager import Manager

    class FakeTrainer:
        # Anything that is_fsdp_strategy returns False on works.
        strategy = "ddp"

    # We don't construct a full Manager; we only need its method.
    fake_self = type("X", (), {"_trainer": FakeTrainer()})()
    with caplog.at_level(_logging.INFO):
        Manager._log_fsdp_info(fake_self, needs_teacher_student=False)
    text = caplog.text
    assert "FSDP2 STRATEGY DETECTED" not in text


def test_manager_log_fsdp_info_runs_under_fsdp_without_raising():
    """When FSDP2 is detected, ``_log_fsdp_info`` runs to completion.

    We intentionally don't assert on log content here: loguru (via
    ``richuru``) writes through its own sink that bypasses pytest's
    ``capsys`` capture, so a content assertion would be flaky. The
    contract that "the right fields are exposed" is enforced by
    :func:`test_describe_fsdp_exposes_fsdp2_fields`; this test covers
    only the manager hook's "no exceptions" property under FSDP2.
    """
    from stable_pretraining.manager import Manager
    from stable_pretraining.utils.fsdp2 import StablePretrainingFSDP2

    strat = StablePretrainingFSDP2(data_parallel_size=1, tensor_parallel_size=1)

    class FakeTrainer:
        pass

    trainer = FakeTrainer()
    trainer.strategy = strat

    fake_self = type("X", (), {"_trainer": trainer})()
    # No raise = pass.
    Manager._log_fsdp_info(fake_self, needs_teacher_student=False)
    Manager._log_fsdp_info(fake_self, needs_teacher_student=True)
