# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Registry backend with client-server architecture.

The server runs on the login node (persistent, outside SLURM jobs).
Training jobs connect via HTTP using the discovery file.

Users add one line to their ``.bashrc``::

    spt registry ensure

This checks if the server is running and starts it if not.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get_hostname() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        return "127.0.0.1"


def _server_is_alive(url: str, timeout: float = 3.0) -> bool:
    try:
        req = urllib.request.Request(f"{url}/api/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _default_db_path() -> str:
    """Resolve the default registry.db path from config or env."""
    cache_dir = os.environ.get("SPT_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "stable-pretraining"
        )
    return str(Path(cache_dir).resolve() / "registry.db")


def _discovery_path(db_path: str) -> Path:
    return Path(db_path).with_suffix(".server.json")


def _read_discovery(db_path: str) -> dict | None:
    path = _discovery_path(db_path)
    try:
        data = json.loads(path.read_text())
        if "url" in data:
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return None


def _write_discovery(db_path: str, info: dict) -> None:
    disc_path = _discovery_path(db_path)
    disc_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(disc_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(info, f, indent=2)
        os.replace(tmp, str(disc_path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# ensure_server — called from CLI / .bashrc
# ---------------------------------------------------------------------------


def ensure_server(db_path: str | None = None) -> str:
    """Ensure a registry server is running.  Start one if not.

    Designed to be called from ``.bashrc`` or interactively on the
    login node.  Idempotent — safe to call multiple times.

    Returns the server URL.
    """
    if db_path is None:
        db_path = _default_db_path()

    # Already running?
    info = _read_discovery(db_path)
    if info and _server_is_alive(info["url"]):
        return info["url"]

    # Start the server
    port = _find_free_port()
    hostname = _get_hostname()
    url = f"http://{hostname}:{port}"

    server_script = str(Path(__file__).parent / "_server.py")
    cmd = [
        sys.executable,
        server_script,
        "--db",
        str(Path(db_path).resolve()),
        "--port",
        str(port),
    ]

    log_file = Path(db_path).with_suffix(".server.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    try:
        pid = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=log_fd,
            start_new_session=True,
        ).pid
    finally:
        os.close(log_fd)

    # Wait for readiness
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if _server_is_alive(url):
            _write_discovery(
                db_path,
                {
                    "url": url,
                    "pid": pid,
                    "port": port,
                    "hostname": hostname,
                    "db_path": str(Path(db_path).resolve()),
                    "started_at": time.time(),
                },
            )
            return url
        time.sleep(0.3)

    raise RuntimeError(f"Registry server did not start within 15s. Check {log_file}")


# ---------------------------------------------------------------------------
# Public factory — used by RegistryLogger, query API, CLI
# ---------------------------------------------------------------------------

_SERVER_NOT_RUNNING_MSG = """\
Registry server is not running.

Add this to your ~/.bashrc (run once on the login node):

    spt registry ensure

Then re-open your shell or run it manually."""


def RegistryDB(db_path: str) -> "RegistryClient":  # noqa: N802, F821
    """Connect to the registry server for *db_path*.

    The server must already be running (started via
    ``spt registry ensure``).  If not, raises RuntimeError with
    instructions.
    """
    from ._client import RegistryClient

    info = _read_discovery(db_path)
    if info and _server_is_alive(info["url"]):
        return RegistryClient(info["url"])

    raise RuntimeError(_SERVER_NOT_RUNNING_MSG)
