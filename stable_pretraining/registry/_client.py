# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""HTTP client for the registry server.

Implements the same API as ``LocalRegistryDB`` so callers don't need
to know which backend they're using.  Uses only ``urllib`` — no extra
dependencies.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from loguru import logger as logging


class RegistryClient:
    """HTTP client that talks to the auto-managed registry server.

    Args:
        server_url: Base URL, e.g. ``"http://127.0.0.1:8234"``.
        timeout: Per-request timeout in seconds.
        max_retries: Retries on transient failures.
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.db_path = self.server_url  # backward-compat attribute

    # -- low-level HTTP --------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> Any:
        url = f"{self.server_url}/api{path}"
        if params:
            query = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
            if query:
                url = f"{url}?{query}"

        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")

        delay = 0.5
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_data = resp.read().decode("utf-8")
                    if not resp_data:
                        return None
                    return json.loads(resp_data)
            except urllib.error.HTTPError as exc:
                # Read the response body *now* and convert to a plain
                # RuntimeError.  HTTPError holds an _io.BufferedReader
                # in .fp which is not picklable — this breaks submitit.
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                msg = (
                    f"Registry server error: {exc.code} {exc.reason} "
                    f"on {method} {url}\n{body}"
                )
                if exc.code in (500, 502, 503, 504) and attempt < self.max_retries - 1:
                    logging.warning(
                        f"! {msg} (attempt {attempt + 1}/{self.max_retries})"
                    )
                    last_exc = RuntimeError(msg)
                    time.sleep(delay)
                    delay = min(delay * 2, 5.0)
                elif exc.code == 404:
                    raise  # let callers handle 404 specifically
                else:
                    raise RuntimeError(msg) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries - 1:
                    last_exc = RuntimeError(str(exc))
                    time.sleep(delay)
                    delay = min(delay * 2, 5.0)
                else:
                    raise RuntimeError(str(exc)) from None
        raise last_exc  # pragma: no cover

    # -- public API (matches LocalRegistryDB) ----------------------------------

    def insert_run(
        self,
        run_id: str,
        *,
        status: str = "running",
        run_dir: Optional[str] = None,
        config: Optional[dict] = None,
        hparams: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> None:
        self._request(
            "POST",
            "/runs",
            body={
                "run_id": run_id,
                "status": status,
                "run_dir": run_dir,
                "config": config or {},
                "hparams": hparams or {},
                "tags": tags or [],
                "notes": notes or "",
            },
        )

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        encoded_id = urllib.parse.quote(run_id, safe="")
        self._request("PATCH", f"/runs/{encoded_id}", body=fields)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        encoded_id = urllib.parse.quote(run_id, safe="")
        try:
            return self._request("GET", f"/runs/{encoded_id}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def query_runs(
        self,
        *,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        sort_by: Optional[str] = None,
        descending: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        params: dict[str, str] = {}
        if status is not None:
            params["status"] = status
        if tag is not None:
            params["tag"] = tag
        if sort_by is not None:
            params["sort_by"] = sort_by
        params["descending"] = str(descending).lower()
        if limit is not None:
            params["limit"] = str(limit)
        return self._request("GET", "/runs", params=params) or []

    def count(self) -> int:
        result = self._request("GET", "/runs/count")
        return result.get("count", 0) if result else 0

    def close(self) -> None:
        """No-op — kept for API compatibility."""

    def _get_connection(self):
        """Compatibility shim for code that checks internals."""
        raise AttributeError(
            "RegistryClient is an HTTP client — no SQLite connection. "
            "Use LocalRegistryDB for direct SQLite access."
        )
