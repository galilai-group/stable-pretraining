import os
import signal
import pytest
from unittest.mock import MagicMock
from pathlib import Path
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from stable_pretraining.manager import (
    Manager,
    SIGTERMException,
    _describe_handler,
    _install_sigterm_preempt_handler,
    print_signal_info,
)
from stable_pretraining.tests.utils import BoringTrainer, BoringModule, BoringDataModule


@pytest.fixture
def manager_factory(tmp_path: Path) -> Manager:
    """Pytest fixture that returns a factory function for creating Manager instances.

    This allows each test to configure a Manager for its specific scenario by providing
    the necessary callbacks and checkpoint path, while abstracting away the boilerplate
    of creating the trainer, module, and datamodule.
    """

    def _create_manager(
        callbacks: list[pl.Callback],
        ckpt_path: Path | None,
        trainer_enable_checkpointing: bool,
    ):
        """Factory function to build a Manager with a specific test configuration."""
        trainer = BoringTrainer(
            callbacks=callbacks,
            default_root_dir=str(tmp_path),
            enable_checkpointing=trainer_enable_checkpointing,
        )

        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
            ckpt_path=str(ckpt_path) if ckpt_path else None,
        )
        # In the real code, `_trainer` is prepared inside `manager.__call__`.
        # For this unit test, we assign it manually to isolate the method under test.
        manager._trainer = trainer
        return manager

    return _create_manager


@pytest.mark.unit
class TestMatchesTemplate:
    """Directly tests the `_matches_template` helper function."""

    @pytest.mark.parametrize(
        "ckpt_name, callback, expected",
        [
            # --- Last Checkpoint Scenarios ---
            ("last.ckpt", ModelCheckpoint(save_last=True), True),
            ("last.ckpt", ModelCheckpoint(save_last=False), False),
            ("last-v1.ckpt", ModelCheckpoint(save_last=True), True),
            # --- Template Matching Scenarios ---
            ("epoch=1-step=100.ckpt", ModelCheckpoint(filename="{epoch}-{step}"), True),
            (
                "model-epoch=1-val_loss=0.5.ckpt",
                ModelCheckpoint(filename="model-{epoch}-{val_loss:.2f}"),
                True,
            ),
            (
                "model.ckpt",
                ModelCheckpoint(filename="{epoch}"),
                False,
            ),  # Fails: "epoch=" key is missing
            (
                "epoch=1.ckpt",
                ModelCheckpoint(filename="{epoch}-{step}"),
                False,
            ),  # Fails: "step=" key is missing
            (
                "model-epoch=1-lr=0.01.ckpt",
                ModelCheckpoint(filename="model-{epoch}"),
                False,
            ),  # Fails: lr in left, not in right
            (
                "model-epoch=1-lr=0.01.ckpt",
                ModelCheckpoint(filename="model-{epoch}-{lr}"),
                True,
            ),  # Succeeds: same metrics
            (
                "model-epoch=1.ckpt",
                ModelCheckpoint(filename="model-{epoch}-{lr}"),
                False,
            ),  # Fails: lr in right, not in left
            (
                "model.ckpt",
                ModelCheckpoint(filename="model"),
                True,
            ),  # Matches: template has no keys
        ],
    )
    def test_template_matching_logic(self, ckpt_name, callback, expected):
        """Tests various template matching scenarios."""
        assert Manager._matches_template(ckpt_name, callback) == expected


@pytest.mark.unit
class TestConfigureCheckpointing:
    """Tests the `configure_checkpointing` utility function across various user scenarios."""

    def test_case_1_intentional_ckpt_path_and_callback(
        self, manager_factory, tmp_path: Path
    ):
        """Tests Case 1: The user provides a `ckpt_path` and a matching `ModelCheckpoint` callback.

        This scenario represents a correctly configured setup where the user's intent to save/resume
        from a specific path is perfectly aligned with their callback configuration.

        Expectation: The function should recognize the valid setup and make no changes to the
                     trainer's callbacks.
        """
        ckpt_path = tmp_path / "checkpoints" / "last.ckpt"
        ckpt_path.parent.mkdir()
        # Manager.__init__ now validates that ckpt_path exists.
        ckpt_path.touch()
        callbacks = [ModelCheckpoint(dirpath=str(ckpt_path.parent), save_last=True)]
        manager = manager_factory(
            callbacks=callbacks, ckpt_path=ckpt_path, trainer_enable_checkpointing=True
        )

        initial_callback_count = len(manager._trainer.callbacks)

        manager._configure_checkpointing()

        assert len(manager._trainer.callbacks) == initial_callback_count
        assert 1 == sum(
            isinstance(cb, ModelCheckpoint) for cb in manager._trainer.callbacks
        )

    def test_case_2_intentional_ckpt_path_but_no_callback(
        self, manager_factory, tmp_path: Path
    ):
        """Tests Case 2: The user provides a `ckpt_path` but forgets the `ModelCheckpoint` callback.

        This is the critical "safety net" scenario. The user has signaled their intent to save a
        checkpoint by providing a path, but has not configured the means to do so.

        Expectation: The function should detect the mismatch and automatically add a new
                     `ModelCheckpoint` callback that saves to the specified path.
        """
        ckpt_path = tmp_path / "checkpoints" / "safety_net.ckpt"
        # Manager.__init__ now validates that ckpt_path exists.
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.touch()
        manager = manager_factory(
            callbacks=[], ckpt_path=ckpt_path, trainer_enable_checkpointing=True
        )

        initial_callback_count = len(manager._trainer.callbacks)

        manager._configure_checkpointing()

        assert len(manager._trainer.callbacks) == initial_callback_count + 1
        new_callback = manager._trainer.callbacks[-1]
        assert isinstance(new_callback, ModelCheckpoint)
        assert Path(new_callback.dirpath).resolve() == ckpt_path.parent.resolve()
        assert new_callback.filename == ckpt_path.with_suffix("").name

    def test_case_3_no_checkpointing_but_callback(
        self, manager_factory, tmp_path: Path
    ):
        """Tests Case 3: The user provides a `ModelCheckpoint` callback but no `ckpt_path`.

        In this scenario, the user is managing their own checkpointing (e.g., saving to a
        logger-defined directory) and has not asked the Manager to handle a specific resume path.

        Expectation: The function should respect the user's setup and make no changes.
        """
        user_dir = tmp_path / "user_checkpoints"
        callbacks = [ModelCheckpoint(dirpath=str(user_dir))]
        manager = manager_factory(
            callbacks=callbacks, ckpt_path=None, trainer_enable_checkpointing=True
        )

        initial_callback_count = len(manager._trainer.callbacks)

        # ckpt_path is None, simulating the user not providing it to the Manager
        manager._configure_checkpointing()

        assert len(manager._trainer.callbacks) == initial_callback_count
        assert (
            Path(manager._trainer.callbacks[-1].dirpath).resolve() == user_dir.resolve()
        )

    def test_case_4_no_checkpointing_no_callback(self, manager_factory, tmp_path: Path):
        """Tests Case 4: The user provides no `ckpt_path` and no `ModelCheckpoint` callback.

        This represents the user's intent to run a session without saving any model checkpoints.
        The trainer is configured with enable_checkpointing=False,
        so the trainer will not have a ModelCheckpoint callback.

        Expectation: The function should do nothing and the trainer should have no
                     `ModelCheckpoint` callbacks.
        """
        manager = manager_factory(
            callbacks=[], ckpt_path=None, trainer_enable_checkpointing=False
        )

        initial_callback_count = len(manager._trainer.callbacks)

        manager._configure_checkpointing()

        assert len(manager._trainer.callbacks) == initial_callback_count
        assert not any(
            isinstance(cb, ModelCheckpoint) for cb in manager._trainer.callbacks
        )


@pytest.mark.unit
class TestResumeWeightsOnly:
    """Tests resume weights_only forwarding behavior in Manager."""

    @pytest.mark.parametrize("weights_only", [False, True])
    def test_forward_weights_only_when_supported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        weights_only: bool,
    ):
        ckpt_path = tmp_path / "resume.ckpt"
        ckpt_path.touch()

        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )

        captured = {}

        def fit_with_weights_only(
            module,
            datamodule=None,
            ckpt_path=None,
            weights_only=None,
        ):
            captured["module"] = module
            captured["datamodule"] = datamodule
            captured["ckpt_path"] = ckpt_path
            captured["weights_only"] = weights_only

        trainer.fit = fit_with_weights_only

        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
            ckpt_path=str(ckpt_path),
            weights_only=weights_only,
        )

        monkeypatch.setattr(manager, "init_and_sync_wandb", lambda: None)
        monkeypatch.setattr(manager, "_configure_checkpointing", lambda: None)
        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda *a, **kw: None
        )

        manager()

        assert captured["module"] is manager.instantiated_module
        assert captured["datamodule"] is manager.instantiated_data
        assert captured["ckpt_path"] == str(ckpt_path.resolve())
        assert captured["weights_only"] is weights_only

    def test_ignore_weights_only_when_trainer_fit_does_not_support_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        ckpt_path = tmp_path / "resume.ckpt"
        ckpt_path.touch()

        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )

        fit_no_weights_only = MagicMock()
        trainer.fit = fit_no_weights_only

        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
            ckpt_path=str(ckpt_path),
            weights_only=True,
        )

        monkeypatch.setattr(manager, "init_and_sync_wandb", lambda: None)
        monkeypatch.setattr(manager, "_configure_checkpointing", lambda: None)
        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda *a, **kw: None
        )

        manager()

        fit_no_weights_only.assert_called_once()
        call_kwargs = fit_no_weights_only.call_args.kwargs
        assert call_kwargs["datamodule"] is manager.instantiated_data
        assert call_kwargs["ckpt_path"] == str(ckpt_path.resolve())
        assert "weights_only" not in call_kwargs


@pytest.fixture
def restore_sigterm():
    """Snapshot SIGTERM/SIGUSR2 handlers and restore them after the test.

    `_install_sigterm_preempt_handler` mutates process-global signal state;
    without this fixture an earlier test could leave a handler bound and
    poison later tests (or the pytest runner itself).
    """
    prior_term = signal.getsignal(signal.SIGTERM)
    prior_usr2 = signal.getsignal(signal.SIGUSR2)
    yield
    signal.signal(signal.SIGTERM, prior_term)
    signal.signal(signal.SIGUSR2, prior_usr2)


@pytest.mark.unit
class TestSIGTERMPreemptHandler:
    """Covers `_install_sigterm_preempt_handler` and the inner forwarder."""

    def test_no_op_outside_slurm(
        self, monkeypatch: pytest.MonkeyPatch, restore_sigterm
    ):
        # Ensure SLURM_JOB_ID is unset; install should leave SIGTERM untouched.
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        prior = signal.getsignal(signal.SIGTERM)
        _install_sigterm_preempt_handler()
        assert signal.getsignal(signal.SIGTERM) is prior, (
            "SIGTERM handler must not be replaced when SLURM_JOB_ID is unset"
        )

    def test_installs_handler_under_slurm(
        self, monkeypatch: pytest.MonkeyPatch, restore_sigterm
    ):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        prior = signal.getsignal(signal.SIGTERM)
        _install_sigterm_preempt_handler()
        new = signal.getsignal(signal.SIGTERM)
        assert new is not prior, "SIGTERM handler should be replaced under SLURM"
        assert callable(new)
        # The forwarder is a nested closure — its qualname should give it away.
        assert "_install_sigterm_preempt_handler" in getattr(new, "__qualname__", "")

    def test_handler_forwards_to_usr_signal(
        self, monkeypatch: pytest.MonkeyPatch, restore_sigterm
    ):
        """When SIGTERM fires, the handler must os.kill(self, USR_SIG)."""
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        kills: list[tuple[int, int]] = []
        monkeypatch.setattr(
            "stable_pretraining.manager.os.kill",
            lambda pid, sig: kills.append((pid, sig)),
        )
        _install_sigterm_preempt_handler()
        handler = signal.getsignal(signal.SIGTERM)
        # Invoke the handler directly with a synthetic frame; signal handlers
        # in Python are just plain callables when called this way.
        handler(signal.SIGTERM, None)
        assert kills == [(os.getpid(), int(signal.SIGUSR2))], (
            f"expected one os.kill(self, SIGUSR2) call; got {kills}"
        )

    def test_handler_raises_sigterm_exception_when_kill_fails(
        self, monkeypatch: pytest.MonkeyPatch, restore_sigterm
    ):
        """If os.kill fails, the handler surfaces SIGTERMException (typed fallback)."""
        monkeypatch.setenv("SLURM_JOB_ID", "12345")

        def _broken_kill(pid, sig):
            raise PermissionError("simulated EPERM")

        monkeypatch.setattr("stable_pretraining.manager.os.kill", _broken_kill)
        _install_sigterm_preempt_handler()
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SIGTERMException) as ei:
            handler(signal.SIGTERM, None)
        assert "SIGUSR2" in str(ei.value)
        # __cause__ should be the underlying PermissionError, not lost.
        assert isinstance(ei.value.__cause__, PermissionError)

    def test_handler_honors_submitit_preempt_signal_env(
        self, monkeypatch: pytest.MonkeyPatch, restore_sigterm
    ):
        """`$SUBMITIT_PREEMPT_SIGNAL=USR1` should make us forward to SIGUSR1, not SIGUSR2."""
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SUBMITIT_PREEMPT_SIGNAL", "USR1")
        # Force submitit's class attribute to re-read the env var. submitit
        # caches USR_SIG at class-definition time, so we patch the classmethod.
        import submitit

        monkeypatch.setattr(
            submitit.JobEnvironment,
            "_usr_sig",
            classmethod(lambda cls: signal.SIGUSR1),
        )

        kills: list[tuple[int, int]] = []
        monkeypatch.setattr(
            "stable_pretraining.manager.os.kill",
            lambda pid, sig: kills.append((pid, sig)),
        )
        _install_sigterm_preempt_handler()
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert kills == [(os.getpid(), int(signal.SIGUSR1))]

    def test_warns_when_usr_handler_is_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        restore_sigterm,
        caplog: pytest.LogCaptureFixture,
    ):
        """Install must warn when USR-sig has no handler bound (requeue would no-op)."""
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        # Force USR2 to SIG_DFL so the "no callable handler" branch fires.
        signal.signal(signal.SIGUSR2, signal.SIG_DFL)
        # loguru -> stdlib bridge: capture loguru warnings via propagate.
        import logging as stdlib_logging
        from loguru import logger as loguru_logger

        sink_id = loguru_logger.add(
            lambda msg: stdlib_logging.getLogger("loguru-bridge").warning(
                msg.record["message"]
            ),
            level="WARNING",
        )
        try:
            with caplog.at_level(stdlib_logging.WARNING, logger="loguru-bridge"):
                _install_sigterm_preempt_handler()
        finally:
            loguru_logger.remove(sink_id)
        joined = "\n".join(r.message for r in caplog.records)
        assert "no callable handler" in joined or "Requeue will likely NOT" in joined


@pytest.mark.unit
class TestDescribeHandler:
    """Covers `_describe_handler` — the renderer used by `print_signal_info`."""

    def test_default_action(self):
        assert "SIG_DFL" in _describe_handler(signal.SIG_DFL)

    def test_ignored(self):
        assert "SIG_IGN" in _describe_handler(signal.SIG_IGN)

    def test_none(self):
        assert _describe_handler(None) == "<None>"

    def test_callable_tagged_spt(self):
        # The forwarder closure lives in stable_pretraining.manager
        from stable_pretraining.manager import _install_sigterm_preempt_handler

        out = _describe_handler(_install_sigterm_preempt_handler)
        assert "[spt]" in out
        assert "stable_pretraining" in out

    def test_callable_tagged_submitit(self):
        import submitit

        # Pick any callable defined in submitit
        fn = submitit.JobEnvironment._handle_signals
        out = _describe_handler(fn)
        assert "[submitit]" in out

    def test_callable_tagged_lightning(self):
        from pytorch_lightning.trainer.connectors.signal_connector import (
            _SignalConnector,
        )

        out = _describe_handler(_SignalConnector._sigterm_notifier_fn)
        assert "[lightning]" in out


@pytest.mark.unit
class TestPrintSignalInfo:
    """Sanity-check that `print_signal_info` accepts both bare and labeled call forms."""

    def test_no_label(self):
        # Should not raise.
        print_signal_info()

    def test_with_label(self):
        # Should not raise; label is purely cosmetic.
        print_signal_info("post-fit")
