# tests/test_environment_dump_callback.py

"""Comprehensive test suite for EnvironmentDumpCallback.

Automatically detects GPU availability and runs appropriate tests.

Run with:
    pytest tests/test_environment_dump_callback.py -v
    pytest tests/test_environment_dump_callback.py -v -m "not slow"
    pytest tests/test_environment_dump_callback.py -v -k gpu
"""

import os
import sys
import tempfile
import time
import threading
import gc
from pathlib import Path
from unittest.mock import patch
import subprocess

import pytest
import yaml
import torch
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from stable_pretraining.callbacks import EnvironmentDumpCallback  # Update import path


# ==================== GPU Detection ====================

HAS_GPU = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if HAS_GPU else 0
SKIP_GPU_TESTS = not HAS_GPU

# Markers
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ==================== Helper Functions ====================


def get_gpu_name():
    """Get GPU name if available."""
    if HAS_GPU:
        return torch.cuda.get_device_name(0)
    return "No GPU"


def print_test_header():
    """Print test environment info."""
    print("\n" + "=" * 60)
    print("Test Environment:")
    print(
        f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {HAS_GPU}")
    if HAS_GPU:
        print(f"  GPU Count: {GPU_COUNT}")
        print(f"  GPU: {get_gpu_name()}")
        print(f"  CUDA Version: {torch.version.cuda}")
    print("=" * 60 + "\n")


# Print once at module load
print_test_header()


# ==================== Fixtures ====================


@pytest.fixture
def dummy_model():
    """Create a simple Lightning module for testing."""

    class DummyModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = torch.nn.functional.mse_loss(self(x), y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    model = DummyModel()
    yield model

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def dummy_model_gpu():
    """Create a GPU model if available, otherwise skip."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    class DummyModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = torch.nn.functional.mse_loss(self(x), y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    model = DummyModel()
    yield model

    # Cleanup
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@pytest.fixture
def dummy_dataloader():
    """Create a simple dataloader for testing."""
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
    dataloader = DataLoader(dataset, batch_size=10)
    yield dataloader

    # Cleanup
    del dataset
    del dataloader


@pytest.fixture
def dummy_dataloader_gpu():
    """Create a GPU dataloader if available."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
    dataloader = DataLoader(dataset, batch_size=10)
    yield dataloader

    # Cleanup
    del dataset
    del dataloader
    torch.cuda.empty_cache()


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
        # Automatic cleanup by TemporaryDirectory context manager


@pytest.fixture
def mock_subprocess_success():
    """Mock subprocess calls to return successful results."""
    with patch("subprocess.check_output") as mock_check_output:

        def side_effect(cmd, *args, **kwargs):
            # Normalize to list
            cmd_list = cmd if isinstance(cmd, list) else cmd.split()

            # Create a normalized string for pattern matching
            cmd_str = " ".join(str(c) for c in cmd_list)

            # Define exact command patterns and their responses
            patterns = [
                # Pip
                ("pip freeze", "torch==2.0.0\npytorch-lightning==2.0.0\npyyaml==6.0\n"),
                # Git - order from most specific to least specific
                ("git rev-parse --git-dir", ".git\n"),
                ("git rev-parse --abbrev-ref HEAD", "main\n"),
                ("git rev-parse HEAD", "abc123def456\n"),
                ("git status --porcelain", ""),
                (
                    "git config --get remote.origin.url",
                    "https://github.com/user/repo.git\n",
                ),
                # NVIDIA - check for driver_version query specifically
                ("nvidia-smi --query-gpu=driver_version", "525.85.12\n"),
                (
                    "nvidia-smi --query-gpu=name,driver_version,memory.total",
                    f"{get_gpu_name() if HAS_GPU else 'No GPU'}, 525.85.12, 40960 MiB\n",
                ),
            ]

            # Match patterns
            for pattern, response in patterns:
                # Check if all words in pattern are in cmd_str in order
                pattern_words = pattern.split()
                if all(word in cmd_str for word in pattern_words):
                    # Additional check: for git rev-parse, distinguish between different commands
                    if "git rev-parse" in pattern:
                        if "--abbrev-ref" in pattern and "--abbrev-ref" in cmd_str:
                            return response
                        elif "--git-dir" in pattern and "--git-dir" in cmd_str:
                            return response
                        elif (
                            "HEAD" in pattern
                            and "--abbrev-ref" not in cmd_str
                            and "--git-dir" not in cmd_str
                        ):
                            return response
                    else:
                        return response

            return ""

        mock_check_output.side_effect = side_effect
        yield mock_check_output


@pytest.fixture
def clean_env():
    """Save and restore environment variables."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def cleanup_threads():
    """Cleanup any lingering threads after each test."""
    threads_before = set(threading.enumerate())

    yield

    # Wait for new threads to complete
    threads_after = set(threading.enumerate())
    new_threads = threads_after - threads_before

    for thread in new_threads:
        if thread.is_alive() and not thread.daemon:
            thread.join(timeout=5)

    # Force cleanup any remaining threads
    for thread in new_threads:
        if thread.is_alive() and hasattr(thread, "_stop"):
            thread._stop()


@pytest.fixture(autouse=True)
def cleanup_resources():
    """Cleanup resources after each test."""
    yield

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    gc.collect()


# ==================== Basic Functionality Tests ====================


class TestBasicFunctionality:
    """Test basic callback creation and configuration."""

    def test_callback_creation(self):
        """Test that callback can be created with default parameters."""
        callback = EnvironmentDumpCallback()
        try:
            assert callback.filename == "environment.json"
            assert callback.async_dump
        finally:
            del callback

    def test_callback_custom_filename(self):
        """Test callback with custom filename."""
        callback = EnvironmentDumpCallback(filename="custom_env.json")
        try:
            assert callback.filename == "custom_env.json"
        finally:
            del callback

    def test_callback_sync_mode(self):
        """Test callback in synchronous mode."""
        callback = EnvironmentDumpCallback(async_dump=False)
        try:
            assert not callback.async_dump
        finally:
            del callback


# ==================== File Creation Tests ====================


class TestFileCreation:
    """Test file creation in various modes."""

    def test_files_created_sync(
        self, dummy_model, dummy_dataloader, temp_log_dir, mock_subprocess_success
    ):
        """Test that files are created in synchronous mode."""
        callback = EnvironmentDumpCallback(async_dump=False)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                max_epochs=1,
                max_steps=1,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(dummy_model, dummy_dataloader)

            # Check files exist
            env_file = Path(trainer.log_dir) / "environment.json"
            req_file = Path(trainer.log_dir) / "requirements_frozen.txt"

            assert env_file.exists(), "environment.json should exist"
            assert req_file.exists(), "requirements_frozen.txt should exist"
            trainer.fit(dummy_model, dummy_dataloader)

            # Check files exist
            env_file = Path(trainer.log_dir) / "environment_v1.json"
            req_file = Path(trainer.log_dir) / "requirements_frozen_v1.txt"

            assert env_file.exists(), "environment_v1.json should exist"
            assert req_file.exists(), "requirements_frozen_v1.txt should exist"
        finally:
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer

    def test_files_created_async(
        self, dummy_model, dummy_dataloader, temp_log_dir, mock_subprocess_success
    ):
        """Test that files are created in async mode."""
        callback = EnvironmentDumpCallback(async_dump=True)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                max_epochs=1,
                max_steps=1,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(dummy_model, dummy_dataloader)

            # Wait for async thread to complete
            if callback._dump_thread:
                callback._dump_thread.join(timeout=10)

            # Check files exist
            env_file = Path(trainer.log_dir) / "environment.json"
            req_file = Path(trainer.log_dir) / "requirements_frozen.txt"

            assert env_file.exists(), "environment.json should exist"
            assert req_file.exists(), "requirements_frozen.txt should exist"
        finally:
            if callback._dump_thread and callback._dump_thread.is_alive():
                callback._dump_thread.join(timeout=5)
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer


# ==================== Content Validation Tests ====================


class TestContentValidation:
    """Test the content of generated files."""

    def test_environment_yaml_structure(
        self, dummy_model, dummy_dataloader, temp_log_dir, mock_subprocess_success
    ):
        """Test that environment.json has correct structure."""
        callback = EnvironmentDumpCallback(async_dump=False)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                max_epochs=1,
                max_steps=1,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(dummy_model, dummy_dataloader)

            env_file = Path(trainer.log_dir) / "environment.json"
            with open(env_file, "r") as f:
                env_data = yaml.safe_load(f)

            # Check top-level keys
            required_keys = [
                "timestamp",
                "python",
                "system",
                "packages",
                "environment_variables",
            ]
            for key in required_keys:
                assert key in env_data, f"Missing required key: {key}"

            # Check python info
            assert "version" in env_data["python"]
            assert "executable" in env_data["python"]

            # Check packages info
            assert "pip_freeze" in env_data["packages"]
            assert "key_packages" in env_data["packages"]

        finally:
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer

    def test_requirements_content(
        self, dummy_model, dummy_dataloader, temp_log_dir, mock_subprocess_success
    ):
        """Test that requirements_frozen.txt has valid content."""
        callback = EnvironmentDumpCallback(async_dump=False)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                max_epochs=1,
                max_steps=1,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(dummy_model, dummy_dataloader)

            req_file = Path(trainer.log_dir) / "requirements_frozen.txt"
            content = req_file.read_text()

            # Should contain package names with versions
            assert "torch" in content or "pytorch" in content.lower()
            assert "==" in content  # Version specifier
        finally:
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer


# ==================== GPU-Specific Tests ====================


class TestGPUFunctionality:
    """Tests that require GPU."""

    @pytest.mark.skipif(SKIP_GPU_TESTS, reason="GPU not available")
    def test_gpu_info_captured(
        self,
        dummy_model_gpu,
        dummy_dataloader_gpu,
        temp_log_dir,
        mock_subprocess_success,
    ):
        """Test that GPU information is correctly captured."""
        callback = EnvironmentDumpCallback(async_dump=False)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                max_epochs=1,
                max_steps=1,
                accelerator="gpu",
                devices=1,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(dummy_model_gpu, dummy_dataloader_gpu)

            env_file = Path(trainer.log_dir) / "environment.json"
            with open(env_file, "r") as f:
                env_data = yaml.safe_load(f)

            # Check CUDA info if nvidia-smi available
            if env_data.get("cuda"):
                assert "driver_version" in env_data["cuda"]
        finally:
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer
            torch.cuda.empty_cache()

    @pytest.mark.skipif(
        SKIP_GPU_TESTS or GPU_COUNT < 2, reason="Multiple GPUs not available"
    )
    @pytest.mark.slow
    def test_multi_gpu_info(
        self,
        dummy_model_gpu,
        dummy_dataloader_gpu,
        temp_log_dir,
        mock_subprocess_success,
    ):
        """Test GPU info with multiple GPUs."""
        callback = EnvironmentDumpCallback(async_dump=False)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                max_epochs=1,
                max_steps=1,
                accelerator="gpu",
                devices=min(2, GPU_COUNT),
                strategy="ddp",
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(dummy_model_gpu, dummy_dataloader_gpu)

            # Only rank 0 should create files
            if trainer.global_rank == 0:
                env_file = Path(trainer.log_dir) / "environment.json"
                with open(env_file, "r") as f:
                    env_data = yaml.safe_load(f)

                assert env_data["pytorch"]["num_gpus"] >= 2
                assert len(env_data["pytorch"]["gpu_names"]) >= 2
        finally:
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer
            torch.cuda.empty_cache()


# ==================== Stage Tests ====================


class TestStages:
    """Test callback behavior in different stages."""

    def test_only_runs_on_fit_stage(
        self, dummy_model, temp_log_dir, mock_subprocess_success
    ):
        """Test that callback only runs during 'fit' stage."""
        callback = EnvironmentDumpCallback(async_dump=False)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
            )

            # Test validate stage
            callback.setup(trainer, dummy_model, stage="validate")
            time.sleep(0.2)
            env_file = Path(temp_log_dir) / "environment.json"
            assert not env_file.exists(), "Should not create files on validate stage"

            # Test test stage
            callback.setup(trainer, dummy_model, stage="test")
            time.sleep(0.2)
            assert not env_file.exists(), "Should not create files on test stage"

            # Test fit stage
            callback.setup(trainer, dummy_model, stage="fit")
            time.sleep(0.5)
            env_file = Path(trainer.log_dir) / "environment.json"
            assert env_file.exists(), "Should create files on fit stage"
        finally:
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer


# ==================== Information Collection Tests ====================


class TestInformationCollection:
    """Test individual information collection methods."""

    def test_python_info_collection(self):
        """Test _get_python_info method."""
        callback = EnvironmentDumpCallback()

        try:
            info = callback._get_python_info()

            assert "version" in info
            assert "version_info" in info
            assert info["version_info"]["major"] == sys.version_info.major
            assert info["version_info"]["minor"] == sys.version_info.minor
            assert "executable" in info
            assert info["executable"] == sys.executable
        finally:
            del callback

    def test_system_info_collection(self):
        """Test _get_system_info method."""
        callback = EnvironmentDumpCallback()

        try:
            info = callback._get_system_info()

            assert "platform" in info
            assert "system" in info
            assert "hostname" in info
            assert len(info["hostname"]) > 0
        finally:
            del callback

    def test_packages_info_with_mock(self, mock_subprocess_success):
        """Test _get_packages_info with mocked pip freeze."""
        callback = EnvironmentDumpCallback()

        try:
            info = callback._get_packages_info()

            assert "pip_freeze" in info
            assert "key_packages" in info
            assert "torch" in info["key_packages"]
            assert info["key_packages"]["torch"] == "2.0.0"
        finally:
            del callback

    def test_git_info_with_repo(self, mock_subprocess_success):
        """Test _get_git_info when in a git repository."""
        callback = EnvironmentDumpCallback()

        try:
            info = callback._get_git_info()

            assert info is not None
            assert "commit_hash" in info
            assert "branch" in info
            assert info["commit_hash"] == "abc123def456"
            assert info["branch"] == "main"
        finally:
            del callback

    def test_git_info_without_repo(self):
        """Test _get_git_info when not in a git repository."""
        callback = None

        with patch(
            "subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            try:
                callback = EnvironmentDumpCallback()
                info = callback._get_git_info()

                assert info is None
            finally:
                if callback:
                    del callback

    def test_slurm_info_collection(self, clean_env):
        """Test _get_slurm_info with SLURM environment variables."""
        callback = EnvironmentDumpCallback()

        try:
            os.environ["SLURM_JOB_ID"] = "12345"
            os.environ["SLURM_JOB_NAME"] = "test_job"
            os.environ["SLURM_NTASKS"] = "4"

            info = callback._get_slurm_info()

            assert info is not None
            assert info["SLURM_JOB_ID"] == "12345"
            assert info["SLURM_JOB_NAME"] == "test_job"
            assert info["SLURM_NTASKS"] == "4"
        finally:
            del callback

    def test_slurm_info_no_slurm(self, clean_env):
        """Test _get_slurm_info when not running under SLURM."""
        callback = EnvironmentDumpCallback()

        try:
            # Ensure no SLURM vars
            slurm_vars = [k for k in os.environ.keys() if k.startswith("SLURM_")]
            for var in slurm_vars:
                del os.environ[var]

            info = callback._get_slurm_info()

            assert info is None or len(info) == 0
        finally:
            del callback

    def test_env_variables_filtering(self, clean_env):
        """Test that only relevant environment variables are captured."""
        callback = EnvironmentDumpCallback()

        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            os.environ["MY_CUSTOM_VAR"] = "should_not_be_captured"
            os.environ["NCCL_DEBUG"] = "INFO"

            info = callback._get_env_variables()

            assert "CUDA_VISIBLE_DEVICES" in info
            assert "NCCL_DEBUG" in info
            assert "MY_CUSTOM_VAR" not in info
        finally:
            del callback


# ==================== Error Handling Tests ====================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_handles_pip_freeze_failure(self):
        """Test that callback handles pip freeze failures gracefully."""
        callback = None

        with patch(
            "subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "pip"),
        ):
            try:
                callback = EnvironmentDumpCallback()
                info = callback._get_packages_info()

                assert "Error getting pip freeze" in info["pip_freeze"]
                assert info["total_packages"] == 0
            finally:
                if callback:
                    del callback

    def test_handles_nvidia_smi_missing(self):
        """Test that callback handles missing nvidia-smi gracefully."""
        callback = None

        with patch("subprocess.check_output", side_effect=FileNotFoundError()):
            try:
                callback = EnvironmentDumpCallback()
                info = callback._get_cuda_info()

                assert info is None
            finally:
                if callback:
                    del callback

    def test_async_thread_completion(
        self, dummy_model, dummy_dataloader, temp_log_dir, mock_subprocess_success
    ):
        """Test that async thread completes properly."""
        callback = EnvironmentDumpCallback(async_dump=True)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                max_epochs=1,
                max_steps=1,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(dummy_model, dummy_dataloader)

            # Thread should complete
            assert callback._dump_thread is not None
            callback._dump_thread.join(timeout=10)
            assert not callback._dump_thread.is_alive(), "Thread should have completed"
        finally:
            if callback._dump_thread and callback._dump_thread.is_alive():
                callback._dump_thread.join(timeout=5)
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer

    def test_teardown_waits_for_thread(
        self, dummy_model, temp_log_dir, mock_subprocess_success
    ):
        """Test that teardown waits for async thread."""
        callback = EnvironmentDumpCallback(async_dump=True)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
            )

            # Start a thread
            callback.setup(trainer, dummy_model, stage="fit")

            assert callback._dump_thread is not None
            assert callback._dump_thread.is_alive()

            # Call teardown
            callback.teardown(trainer, dummy_model, stage="fit")

            # Thread should be done
            assert not callback._dump_thread.is_alive()
        finally:
            if callback._dump_thread and callback._dump_thread.is_alive():
                callback._dump_thread.join(timeout=5)
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer

    def test_subprocess_timeout_handling(self):
        """Test that subprocess timeouts are handled gracefully."""
        callback = None

        with patch(
            "subprocess.check_output", side_effect=subprocess.TimeoutExpired("cmd", 5)
        ):
            try:
                callback = EnvironmentDumpCallback()

                git_info = callback._get_git_info()
                assert git_info is None

                cuda_info = callback._get_cuda_info()
                assert cuda_info is None
            finally:
                if callback:
                    del callback

    def test_callback_picklable_after_thread_started(
        self, dummy_model, temp_log_dir, mock_subprocess_success
    ):
        """Regression test for issue #416.

        Spawn-mode DataLoader workers must be able to pickle the
        trainer/module graph that contains this callback.
        ``threading.Thread`` holds a ``_thread.lock``, so the callback
        must drop ``_dump_thread`` in ``__getstate__``.
        """
        import pickle

        callback = EnvironmentDumpCallback(async_dump=True)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
            )

            callback.setup(trainer, dummy_model, stage="fit")
            assert callback._dump_thread is not None

            blob = pickle.dumps(callback)
            restored = pickle.loads(blob)
            assert restored._dump_thread is None
            assert restored.filename == callback.filename
            assert restored.async_dump == callback.async_dump
        finally:
            if callback._dump_thread and callback._dump_thread.is_alive():
                callback._dump_thread.join(timeout=5)
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer

    def test_thread_cleanup_on_exception(self, dummy_model, temp_log_dir):
        """Test that threads are cleaned up even when exceptions occur."""
        callback = EnvironmentDumpCallback(async_dump=True)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
            )

            callback.setup(trainer, dummy_model, stage="fit")

            # Simulate an exception
            try:
                raise ValueError("Simulated error")
            except ValueError:
                pass

            # Ensure cleanup still happens
            callback.teardown(trainer, dummy_model, stage="fit")

            if callback._dump_thread:
                assert not callback._dump_thread.is_alive()
        finally:
            if callback._dump_thread and callback._dump_thread.is_alive():
                callback._dump_thread.join(timeout=5)
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests with full training runs."""

    def test_full_training_run(
        self, dummy_model, dummy_dataloader, temp_log_dir, mock_subprocess_success
    ):
        """Integration test with full training run."""
        callback = EnvironmentDumpCallback(async_dump=False)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                max_epochs=2,
                max_steps=5,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(dummy_model, dummy_dataloader)

            # Verify files exist and are valid
            env_file = Path(trainer.log_dir) / "environment.json"
            req_file = Path(trainer.log_dir) / "requirements_frozen.txt"

            assert env_file.exists()
            assert req_file.exists()

            # Verify YAML is valid
            with open(env_file, "r") as f:
                data = yaml.safe_load(f)
            assert data is not None
            assert "timestamp" in data
        finally:
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer

    def test_custom_log_dir(
        self, dummy_model, dummy_dataloader, temp_log_dir, mock_subprocess_success
    ):
        """Test that files are saved to custom log directory."""
        custom_dir = Path(temp_log_dir) / "custom_logs"
        callback = EnvironmentDumpCallback(async_dump=False)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=str(custom_dir),
                max_epochs=1,
                max_steps=1,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(dummy_model, dummy_dataloader)

            env_file = Path(trainer.log_dir) / "environment.json"
            assert env_file.exists()
            assert custom_dir in env_file.parents
        finally:
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer

    def test_no_file_leaks(
        self, dummy_model, dummy_dataloader, temp_log_dir, mock_subprocess_success
    ):
        """Test that no file handles are left open."""
        callback = EnvironmentDumpCallback(async_dump=False)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                max_epochs=1,
                max_steps=1,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(dummy_model, dummy_dataloader)

            # Force garbage collection
            gc.collect()

            # Verify files can be read (not locked)
            env_file = Path(trainer.log_dir) / "environment.json"
            with open(env_file, "r") as f:
                _ = f.read()
        finally:
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer
            gc.collect()

    @pytest.mark.skipif(SKIP_GPU_TESTS, reason="GPU not available")
    def test_full_training_run_gpu(
        self,
        dummy_model_gpu,
        dummy_dataloader_gpu,
        temp_log_dir,
        mock_subprocess_success,
    ):
        """Integration test with GPU training."""
        callback = EnvironmentDumpCallback(async_dump=False)
        trainer = None

        try:
            trainer = Trainer(
                default_root_dir=temp_log_dir,
                max_epochs=1,
                max_steps=3,
                accelerator="gpu",
                devices=1,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(dummy_model_gpu, dummy_dataloader_gpu)

        finally:
            if trainer and hasattr(trainer, "strategy"):
                trainer.strategy.teardown()
            del callback
            del trainer
            torch.cuda.empty_cache()


# ==================== Performance marker ====================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
