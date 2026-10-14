"""Signal diagnostics must work on both Windows and Unix platforms."""

from enum import IntEnum
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import stable_pretraining.manager as manager


class DiagnosticSignal(IntEnum):
    """Provide Unix signal names without relying on the host operating system."""

    SIGUSR1 = 10
    SIGUSR2 = 12
    SIGCONT = 18
    SIGTERM = 15


@pytest.mark.unit
@pytest.mark.parametrize(
    "names",
    [
        ("SIGUSR1", "SIGUSR2", "SIGCONT", "SIGTERM"),
        ("SIGTERM",),
        (),
    ],
)
def test_signal_diagnostics_only_query_supported_signals(monkeypatch, names):
    getsignal = Mock(return_value=manager.signal.SIG_DFL)
    fake = SimpleNamespace(
        SIG_DFL=manager.signal.SIG_DFL,
        SIG_IGN=manager.signal.SIG_IGN,
        getsignal=getsignal,
        **{name: DiagnosticSignal[name] for name in names},
    )
    messages = []
    monkeypatch.setattr(manager, "signal", fake)
    monkeypatch.setattr(manager.logging, "info", messages.append)
    manager.print_signal_info("platform regression")
    assert [call.args[0] for call in getsignal.call_args_list] == [
        DiagnosticSignal[name] for name in names
    ]
    for name in DiagnosticSignal.__members__:
        line = next(message for message in messages if name in message)
        if name in names:
            assert "SIG_DFL" in line
        else:
            assert "unavailable on this platform" in line
