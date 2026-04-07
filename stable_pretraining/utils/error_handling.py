import time
from huggingface_hub.utils import HfHubHTTPError
import requests
from loguru import logger as logging
import sys
import os
import traceback
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any


def get_rank():
    """Get distributed training rank."""
    return int(os.environ.get("RANK", "0"))


def is_main_process():
    """Check if this is the main process."""
    return get_rank() == 0


@contextmanager
def catch_errors():
    """Catch and log errors from all ranks before re-raising.

    Ensures errors appear in Slurm logs, wandb, and everywhere else.
    """
    try:
        yield

    except Exception as e:
        rank = get_rank()
        rank_prefix = f"[Rank {rank}] " if rank > 0 else ""

        error_msg = (
            f"\n{'=' * 80}\n"
            f"{rank_prefix}💥 EXCEPTION CAUGHT\n"
            f"{'=' * 80}\n"
            f"Type: {type(e).__name__}\n"
            f"Message: {str(e)}\n"
            f"{'=' * 80}\n"
            f"TRACEBACK:\n"
            f"{traceback.format_exc()}"
            f"{'=' * 80}\n"
        )

        # Log from ALL ranks (important for debugging distributed issues)
        logging.opt(depth=1).error(error_msg)

        # Direct prints to stderr/stdout (backup for Slurm logs)
        print(error_msg, file=sys.stderr, flush=True)
        print(error_msg, file=sys.stdout, flush=True)

        # Always re-raise
        raise


def catch_errors_decorator():
    """Decorator version of catch_errors.

    Usage:
        @catch_errors_decorator()
        def train():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with catch_errors():
                return func(*args, **kwargs)

        return wrapper

    return decorator


def with_hf_retry_ratelimit(func, *args, delay=10, max_attempts=100, **kwargs):
    """Calls the given function with retry logic for HTTP 429 (Too Many Requests) errors.

    This function attempts to call ``func(*args, **kwargs)``. If a rate-limiting error (HTTP 429)
    is encountered—detected via exception type, status code, or error message—it will wait
    for the duration specified by the HTTP ``Retry-After`` header (if present), or fall back to
    the ``delay`` parameter, and then retry. Retries continue up to ``max_attempts`` times.
    Non-429 errors are immediately re-raised. If all attempts fail due to 429, the last
    exception is raised.

    Exceptions handled:
        - huggingface_hub.utils.HfHubHTTPError
        - requests.exceptions.HTTPError
        - OSError

    429 detection is performed by checking the exception's ``response.status_code`` (if available)
    or by searching for '429' or 'Too Many Requests' in the exception message.

    Args:
        func (callable): The function to call.
        *args: Positional arguments to pass to ``func``.
        delay (int, optional): Default wait time (in seconds) between retries if ``Retry-After``
            is not provided. Defaults to 10.
        max_attempts (int, optional): Maximum number of attempts before giving up. Defaults to 100.
        **kwargs: Keyword arguments to pass to ``func``.

    Returns:
        The return value of ``func(*args, **kwargs)`` if successful.

    Raises:
        Exception: The original exception if a non-429 error occurs, or if all attempts fail.

    Example:
        >>> from transformers import AutoModel
        >>> model = with_hf_retry_ratelimit(
        ...     AutoModel.from_pretrained,
        ...     "facebook/ijepa_vith14_1k",
        ...     delay=10,
        ...     max_attempts=5,
        ... )
    """
    attempts = 0
    while True:
        try:
            return func(*args, **kwargs)
        except (HfHubHTTPError, requests.exceptions.HTTPError, OSError) as e:
            # Try to extract status code and Retry-After
            status_code = None
            retry_after = delay
            if hasattr(e, "response") and e.response is not None:
                status_code = getattr(e.response, "status_code", None)
                retry_after = int(e.response.headers.get("Retry-After", delay))
            # Fallback: parse error message for 429
            if status_code == 429 or "429" in str(e) or "Too Many Requests" in str(e):
                attempts += 1
                if attempts >= max_attempts:
                    raise
                logging.warning(
                    f"429 received. Waiting {retry_after}s before retrying (attempt {attempts}/{max_attempts})..."
                )
                time.sleep(retry_after)
            else:
                raise


def catch_errors_class(exclude_methods=None, include_private=False):
    """Class decorator that wraps all public methods with catch_errors.

    Only wraps methods defined in the class itself, not inherited methods.

    Args:
        exclude_methods: Set/list of method names to exclude
        include_private: If True, wrap private methods too (starting with _)

    Usage:
        @catch_errors_class()
        class MyClass:
            def method(self):
                ...
    """
    if exclude_methods is None:
        exclude_methods = {
            "__new__",
            "__del__",
            "__repr__",
            "__str__",
            "__dict__",
            "__weakref__",
            "__module__",
            "__doc__",
            # PyTorch methods - don't wrap these!
            "get_extra_state",
            "set_extra_state",
            "_apply",
            "_save_to_state_dict",
            "_load_from_state_dict",
        }
    else:
        exclude_methods = set(exclude_methods)

    def decorator(cls):
        # Only iterate over methods defined in THIS class, not inherited
        for attr_name, attr_value in cls.__dict__.items():
            # Skip excluded
            if attr_name in exclude_methods:
                continue

            # Skip private unless requested
            if not include_private and attr_name.startswith("_"):
                continue

            # Only wrap callable methods (not classmethods, staticmethods, properties)
            if callable(attr_value) and not isinstance(
                attr_value, (classmethod, staticmethod, property)
            ):
                wrapped = catch_errors_decorator()(attr_value)
                setattr(cls, attr_name, wrapped)

        return cls

    return decorator
