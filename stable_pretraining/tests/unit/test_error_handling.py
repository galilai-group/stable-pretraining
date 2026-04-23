import sys
import os
from io import StringIO
import tempfile

import pytest
from loguru import logger

from stable_pretraining.utils.error_handling import (
    catch_errors_decorator,
    catch_errors,
    get_rank,
    is_main_process,
)


@pytest.fixture
def clean_env():
    """Clean environment variables before/after test."""
    old_rank = os.environ.get("RANK")
    yield
    if old_rank is not None:
        os.environ["RANK"] = old_rank
    elif "RANK" in os.environ:
        del os.environ["RANK"]


@pytest.fixture
def temp_log_file():
    """Create and cleanup temporary log file."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log")
    temp_file.close()

    # Setup logger to write to temp file
    logger.remove()
    logger.add(temp_file.name, level="DEBUG")

    yield temp_file.name

    # Cleanup
    logger.remove()
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.mark.unit
def test_catch_errors_basic(temp_log_file):
    """Test basic error catching and logging."""
    captured_stderr = StringIO()
    captured_stdout = StringIO()

    old_stderr = sys.stderr
    old_stdout = sys.stdout

    sys.stderr = captured_stderr
    sys.stdout = captured_stdout

    try:
        with pytest.raises(ValueError):
            with catch_errors():
                raise ValueError("Test error message")

        stderr_content = captured_stderr.getvalue()
        assert "💥 EXCEPTION CAUGHT" in stderr_content
        assert "ValueError" in stderr_content
        assert "Test error message" in stderr_content

        stdout_content = captured_stdout.getvalue()
        assert "💥 EXCEPTION CAUGHT" in stdout_content

    finally:
        sys.stderr = old_stderr
        sys.stdout = old_stdout

    with open(temp_log_file) as f:
        log_content = f.read()
        assert "ValueError" in log_content


@pytest.mark.unit
def test_catch_errors_reraises(temp_log_file):
    """Test that errors are re-raised after logging."""
    with pytest.raises(RuntimeError, match="Critical failure"):
        with catch_errors():
            raise RuntimeError("Critical failure")


@pytest.mark.unit
def test_catch_errors_success_no_error(temp_log_file):
    """Test that no error occurs when code succeeds."""
    result = None
    with catch_errors():
        result = 42

    assert result == 42


@pytest.mark.unit
def test_decorator_version(temp_log_file):
    """Test decorator version of error catcher."""
    captured_stderr = StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured_stderr

    try:

        @catch_errors_decorator()
        def failing_function():
            raise TypeError("Decorator test error")

        with pytest.raises(TypeError):
            failing_function()

        stderr_content = captured_stderr.getvalue()
        assert "💥 EXCEPTION CAUGHT" in stderr_content
        assert "TypeError" in stderr_content
        assert "Decorator test error" in stderr_content

    finally:
        sys.stderr = old_stderr


@pytest.mark.unit
def test_rank_0_no_prefix(temp_log_file, clean_env):
    """Test rank 0 (main process) has no rank prefix."""
    os.environ["RANK"] = "0"

    captured_stderr = StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured_stderr

    try:
        with pytest.raises(ValueError):
            with catch_errors():
                raise ValueError("Rank 0 error")

        stderr_content = captured_stderr.getvalue()
        assert "💥 EXCEPTION CAUGHT" in stderr_content
        assert "[Rank" not in stderr_content

    finally:
        sys.stderr = old_stderr


@pytest.mark.unit
def test_rank_3_has_prefix(temp_log_file, clean_env):
    """Test non-zero rank has rank prefix."""
    os.environ["RANK"] = "3"

    captured_stderr = StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured_stderr

    try:
        with pytest.raises(ValueError):
            with catch_errors():
                raise ValueError("Rank 3 error")

        stderr_content = captured_stderr.getvalue()
        assert "[Rank 3] 💥 EXCEPTION CAUGHT" in stderr_content

    finally:
        sys.stderr = old_stderr


@pytest.mark.unit
def test_traceback_included(temp_log_file):
    """Test that full traceback is included in error message."""
    captured_stderr = StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured_stderr

    def inner_function():
        raise KeyError("Missing key")

    def outer_function():
        inner_function()

    try:
        with pytest.raises(KeyError):
            with catch_errors():
                outer_function()

        stderr_content = captured_stderr.getvalue()
        assert "TRACEBACK:" in stderr_content
        assert "inner_function" in stderr_content
        assert "outer_function" in stderr_content
        assert "KeyError" in stderr_content

    finally:
        sys.stderr = old_stderr


@pytest.mark.unit
def test_get_rank_default(clean_env):
    """Test get_rank returns 0 when RANK not set."""
    if "RANK" in os.environ:
        del os.environ["RANK"]

    assert get_rank() == 0
    assert is_main_process() is True


@pytest.mark.unit
def test_error_no_wandb_dependency(temp_log_file, clean_env):
    """Test that error handling has no wandb dependency."""
    os.environ["RANK"] = "0"

    with pytest.raises(ValueError):
        with catch_errors():
            raise ValueError("Test with no wandb")

    # Should complete without crashing - no wandb involved
