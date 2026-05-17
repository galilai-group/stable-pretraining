"""Unit tests for the callback-order log helper.

Loguru does not write to stdlib ``logging`` by default, so ``caplog`` cannot
capture our log lines. We attach a temporary sink to ``loguru.logger`` instead.
"""

import pytest
from lightning.pytorch import Callback
from loguru import logger as loguru_logger

from stable_pretraining.callbacks.utils import (
    ORDER_SENSITIVE_CALLBACKS,
    log_callbacks_order,
)


@pytest.fixture
def captured_loguru_lines():
    """Capture every loguru message emitted during the test."""
    lines: list[str] = []
    sink_id = loguru_logger.add(lambda msg: lines.append(str(msg)), level="DEBUG")
    yield lines
    loguru_logger.remove(sink_id)


class _FakeOrderSensitive(Callback):
    pass


class _FakeRegular(Callback):
    pass


@pytest.mark.unit
class TestLogCallbacksOrder:
    """Smoke + content tests for the runtime callback-order log."""

    def test_empty_list_handled(self, captured_loguru_lines):
        log_callbacks_order([])
        # Doesn't crash; emits a "none registered" hint.
        assert any("none registered" in line for line in captured_loguru_lines), (
            captured_loguru_lines
        )

    def test_lists_classes_in_registration_order(self, captured_loguru_lines):
        callbacks = [_FakeRegular(), _FakeRegular(), _FakeRegular()]
        log_callbacks_order(callbacks)
        joined = "\n".join(captured_loguru_lines)
        # The index prefix lets us verify order
        idx_0 = joined.find("[0]")
        idx_1 = joined.find("[1]")
        idx_2 = joined.find("[2]")
        assert 0 <= idx_0 < idx_1 < idx_2, f"Indices not in order in output:\n{joined}"

    def test_order_sensitive_marked_with_glyph(
        self, captured_loguru_lines, monkeypatch
    ):
        """Callbacks in the registry get the ⚑ marker + their rule text."""
        # Temporarily teach the registry about our fake class so we exercise
        # the marker path without depending on real production names.
        monkeypatch.setitem(
            ORDER_SENSITIVE_CALLBACKS,
            "_FakeOrderSensitive",
            "test rule: must come last",
        )
        log_callbacks_order([_FakeRegular(), _FakeOrderSensitive()])
        joined = "\n".join(captured_loguru_lines)
        assert "⚑" in joined
        assert "_FakeOrderSensitive" in joined
        # And the rule text is surfaced.
        assert "must come last" in joined

    def test_non_sensitive_callback_has_no_rule_line(self, captured_loguru_lines):
        """Regular callbacks list ONE line; sensitive ones list two (name + rule)."""
        # Use the actual registry — _FakeRegular is not in it.
        log_callbacks_order([_FakeRegular()])
        joined = "\n".join(captured_loguru_lines)
        # ⚑ should NOT appear next to _FakeRegular itself (only in the trailing
        # legend line). And no "order rule:" prefix.
        assert "order rule" not in joined

    def test_real_registry_covers_expected_callbacks(self):
        """Sanity-check the actual registry has the entries the docs promise.

        Only callbacks that have a **same-hook** ordering constraint are
        listed — producer/consumer pairs split across different hooks
        (OnlineQueue → OnlineKNN, etc.) are intentionally NOT here because
        Lightning serializes hooks across all callbacks.
        """
        expected = {
            "TeacherStudentCallback",
            "OnlineProbe",
            "OnlineWriter",
            "CleanUpCallback",
        }
        actual = set(ORDER_SENSITIVE_CALLBACKS.keys())
        assert expected.issubset(actual), f"Missing from registry: {expected - actual}"

    def test_legend_line_present_when_any_sensitive(
        self, captured_loguru_lines, monkeypatch
    ):
        monkeypatch.setitem(ORDER_SENSITIVE_CALLBACKS, "_FakeOrderSensitive", "rule")
        log_callbacks_order([_FakeOrderSensitive()])
        joined = "\n".join(captured_loguru_lines)
        assert "AGENTS.md → Callback ordering" in joined
