import pytest
from unittest.mock import MagicMock
from pathlib import Path
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from stable_pretraining.manager import Manager
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

    @pytest.mark.parametrize("resume_weights_only", [False, True])
    def test_forward_resume_weights_only_when_supported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        resume_weights_only: bool,
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
            resume_weights_only=resume_weights_only,
        )

        monkeypatch.setattr(manager, "init_and_sync_wandb", lambda: None)
        monkeypatch.setattr(manager, "_configure_checkpointing", lambda: None)
        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        manager()

        assert captured["module"] is manager.instantiated_module
        assert captured["datamodule"] is manager.instantiated_data
        assert captured["ckpt_path"] == str(ckpt_path.resolve())
        assert captured["weights_only"] is resume_weights_only

    def test_ignore_resume_weights_only_when_trainer_fit_does_not_support_it(
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
            resume_weights_only=True,
        )

        monkeypatch.setattr(manager, "init_and_sync_wandb", lambda: None)
        monkeypatch.setattr(manager, "_configure_checkpointing", lambda: None)
        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        manager()

        fit_no_weights_only.assert_called_once()
        call_kwargs = fit_no_weights_only.call_args.kwargs
        assert call_kwargs["datamodule"] is manager.instantiated_data
        assert call_kwargs["ckpt_path"] == str(ckpt_path.resolve())
        assert "weights_only" not in call_kwargs
