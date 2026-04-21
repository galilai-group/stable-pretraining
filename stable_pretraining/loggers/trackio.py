# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning logger backed by `Trackio <https://github.com/gradio-app/trackio>`_.

:class:`TrackioLogger` is a drop-in replacement for Lightning's
``WandbLogger``.  It translates Lightning's ``log_hyperparams`` /
``log_metrics`` / ``finalize`` protocol into Trackio's ``init`` /
``log`` / ``finish`` calls.

Three logging targets are supported:

* **Local SQLite** (default) — ``trackio.init()`` writes to
  ``~/.cache/huggingface/trackio/<project>.db``.  Safe for single-node,
  single-process runs; **not** NFS-safe across parallel jobs.
* **Hugging Face Space** — pass ``space_id="user/space"``.  Trackio deploys
  a public dashboard and syncs metrics over HTTPS.
* **Self-hosted Trackio server** — pass ``server_url="http://host:port"``.
  Bypasses ``trackio.init()`` and connects directly to a
  ``trackio show``-style server on your cluster.  All jobs funnel metrics
  through one server process, keeping SQLite local to that node — NFS is
  never touched.  Set ``TRACKIO_SERVER_URL`` env var to make this the
  default across all jobs without touching code.

Only rank-zero performs I/O — on non-zero ranks every method is a
no-op, making it safe for DDP / FSDP / DeepSpeed.

Requeue / resume support is handled by the companion
:class:`~stable_pretraining.callbacks.TrackioCheckpoint` callback,
which persists the run name in the checkpoint so a restarted job can
call ``trackio.init(resume="must")`` to continue the same run.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.rank_zero import rank_zero_only

try:
    import trackio

    TRACKIO_AVAILABLE = True
except ImportError:
    trackio = None
    TRACKIO_AVAILABLE = False


# Env var for cluster-wide self-hosted server default.
_SERVER_URL_ENV = "TRACKIO_SERVER_URL"


class TrackioLogger(Logger):
    """Lightning logger that sends metrics to Trackio.

    Args:
        project: Trackio project name (required).
        name: Run name.  If ``None``, Trackio auto-generates one.
        group: Logical group for related runs (e.g. a sweep).
        space_id: Hugging Face Space ID (``"user/space"``) for remote
            dashboards hosted on HF.  Mutually exclusive with ``server_url``.
        server_url: URL of a self-hosted Trackio server
            (e.g. ``"http://89.169.120.245:7860"``).  When set, bypasses
            ``trackio.init()`` and connects directly to the server — no
            HF API calls, no local SQLite on the training node.  If not
            provided, falls back to the ``TRACKIO_SERVER_URL`` env var.
            Mutually exclusive with ``space_id``.
        auto_log_gpu: Automatically log GPU utilisation metrics.  If
            ``None`` (default), auto-detects CUDA and enables GPU logging
            when a device is available.  Pass ``False`` to force-disable.
        gpu_log_interval: Seconds between GPU metric samples (default 10).
        resume: Resume mode — ``"never"`` (default), ``"allow"``, or
            ``"must"``.  Set automatically by
            :class:`~stable_pretraining.callbacks.TrackioCheckpoint`
            on requeue.
        trackio_kwargs: Extra keyword arguments forwarded to
            ``trackio.init()`` (ignored in ``server_url`` mode).
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        *,
        group: Optional[str] = None,
        space_id: Optional[str] = None,
        server_url: Optional[str] = None,
        auto_log_gpu: Optional[bool] = None,
        gpu_log_interval: float = 10.0,
        resume: str = "never",
        **trackio_kwargs: Any,
    ) -> None:
        if not TRACKIO_AVAILABLE:
            raise ImportError(
                "trackio is required for TrackioLogger but is not installed. "
                "Install it with: pip install trackio"
            )
        # Resolve server_url from env var if not passed explicitly.
        resolved_server_url = server_url or os.environ.get(_SERVER_URL_ENV)

        if resolved_server_url and space_id:
            raise ValueError(
                "Pass either `space_id` (HF Space) or `server_url` "
                "(self-hosted Trackio server), not both."
            )

        super().__init__()
        self._project = project
        self._name = name
        self._group = group
        self._space_id = space_id
        self._server_url = resolved_server_url
        self._auto_log_gpu = _cuda_available() if auto_log_gpu is None else auto_log_gpu
        self._gpu_log_interval = gpu_log_interval
        self._resume = resume
        self._trackio_kwargs = trackio_kwargs
        self._run: Optional[object] = None

    # -- Lightning Logger protocol ---------------------------------------------

    @property
    def name(self) -> str:
        return self._project

    @property
    def version(self) -> Union[str, int]:
        return self._name or ""

    @property
    @rank_zero_experiment
    def experiment(self) -> object:
        """Return the Trackio run, lazily initialised on first access."""
        if self._run is None:
            self._init_run()
        return self._run

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Any]) -> None:
        # Ensure the run exists (config is sent on init).
        if self._run is None:
            config = _params_to_dict(params)
            self._init_run(config=config)
        # If already initialised, Trackio doesn't support updating config
        # after init — the params are still saved by Lightning's CSV/sidecar
        # loggers.

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._run is None:
            self._init_run()
        # Filter to scalar values only — Trackio expects numeric metrics.
        scalar_metrics = {}
        for k, v in metrics.items():
            s = _to_scalar(v)
            if s is not None:
                scalar_metrics[k] = s
        if not scalar_metrics:
            return

        if self._server_url:
            # Direct run.log() — we constructed the Run manually, trackio's
            # module-level state isn't set.
            self._run.log(scalar_metrics, step=step)
        else:
            trackio.log(scalar_metrics, step=step)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._run is None:
            return
        if self._server_url:
            # We own this Run, trackio doesn't know about it — call finish
            # directly to guarantee the remote sender thread drains.
            finish_fn = getattr(self._run, "finish", None)
            if finish_fn is not None:
                finish_fn()
        else:
            trackio.finish()
        self._run = None

    # -- Internal helpers ------------------------------------------------------

    def _init_run(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the run via ``trackio.init()`` or against a self-hosted server.

        For a self-hosted ``server_url``, constructs a :class:`trackio.run.Run`
        directly against the server.
        """
        if self._server_url:
            self._init_run_server_mode(config=config)
        else:
            self._run = trackio.init(
                project=self._project,
                name=self._name,
                group=self._group,
                space_id=self._space_id,
                config=config,
                resume=self._resume,
                auto_log_gpu=self._auto_log_gpu,
                gpu_log_interval=self._gpu_log_interval,
                **self._trackio_kwargs,
            )

    def _init_run_server_mode(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Connect directly to a self-hosted Trackio server.

        Bypasses ``trackio.init()`` (which would try to create/parse an HF
        Space) and manually constructs a :class:`trackio.run.Run` bound to
        a ``gradio_client.Client`` pointing at the server URL.  The
        ``space_id="local/server"`` sentinel activates the remote sender
        thread inside ``Run`` without triggering HF media-upload paths.
        """
        from gradio_client import Client
        from trackio.run import Run
        import trackio.context_vars as _cv

        client = Client(self._server_url, verbose=False)
        self._run = Run(
            url=self._server_url,
            project=self._project,
            name=self._name,
            client=client,
            # Sentinel — activates remote sender thread.  Not a real HF Space.
            space_id="local/server",
            config=config or {},
            auto_log_gpu=bool(self._auto_log_gpu),
            gpu_log_interval=self._gpu_log_interval,
        )
        _cv.current_run.set(self._run)

    # -- Checkpoint resume helpers (used by TrackioCheckpoint) -----------------

    @property
    def resume_info(self) -> Dict[str, Any]:
        """Snapshot of state needed to resume this run after requeue."""
        return {
            "project": self._project,
            "name": self._name,
            "group": self._group,
            "server_url": self._server_url,
        }

    def set_resume(self, name: str) -> None:
        """Configure this logger to resume a previous run on next init.

        Called by the Manager or :class:`TrackioCheckpoint` *before*
        ``experiment`` is accessed, so the run is created with
        ``resume="must"``.
        """
        self._name = name
        self._resume = "must"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_project_df(
    project: str,
    *,
    server_url: Optional[str] = None,
    runs: Optional[list] = None,
) -> Any:
    """Load all runs of a Trackio project as a single pandas DataFrame.

    Works against either a self-hosted Trackio server (over HTTP via
    ``gradio_client``) or the local SQLite DB.

    Args:
        project: Trackio project name.
        server_url: URL of the self-hosted Trackio server
            (e.g. ``"http://89.169.120.245:7860"``).  If ``None``, falls
            back to the ``TRACKIO_SERVER_URL`` env var.  If that's also
            unset, reads from the local SQLite DB.
        runs: Optional list of run names to restrict to.  If ``None``
            (default), loads every run in the project.

    Returns:
        A ``pandas.DataFrame`` with one row per logged step and columns
        for every metric plus ``step``, ``timestamp``, and ``run``.
        Missing metrics are ``NaN``.

    Example::

        import stable_pretraining as spt

        df = spt.loggers.load_project_df(
            "backward_statistics_je",
            server_url="http://89.169.120.245:7860",
        )
        # Pivot: one column per metric, per run
        df.pivot_table(index="step", columns="run", values="loss/total")
    """
    import pandas as pd

    resolved = server_url or os.environ.get(_SERVER_URL_ENV)

    if resolved:
        list_runs, fetch_logs = _remote_fetchers(resolved, project)
    else:
        list_runs, fetch_logs = _local_fetchers(project)

    # Only enumerate runs when the caller didn't specify them.
    target_runs = list(runs) if runs is not None else list_runs()
    frames = []
    for run in target_runs:
        rows = fetch_logs(run)
        if not rows:
            continue
        sub = pd.DataFrame(rows)
        sub["run"] = run
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _remote_fetchers(server_url: str, project: str):
    """Return ``(list_runs_fn, fetch_logs_fn)`` bound to a remote server."""
    from gradio_client import Client

    client = Client(server_url, verbose=False)

    def list_runs():
        return list(client.predict(project, api_name="/get_runs_for_project"))

    def fetch_logs(run: str):
        return client.predict(project, run, api_name="/get_logs")

    return list_runs, fetch_logs


def _local_fetchers(project: str):
    """Return ``(list_runs_fn, fetch_logs_fn)`` bound to local SQLite."""
    from trackio.sqlite_storage import SQLiteStorage

    def list_runs():
        return list(SQLiteStorage.get_runs(project))

    def fetch_logs(run: str):
        return SQLiteStorage.get_logs(project, run)

    return list_runs, fetch_logs


def find_trackio_logger(trainer: Any) -> Optional[TrackioLogger]:
    """Find the unique :class:`TrackioLogger` among trainer loggers.

    Returns ``None`` if no TrackioLogger is configured.

    Raises:
        RuntimeError: If more than one TrackioLogger is attached.
    """
    found = [lg for lg in trainer.loggers if isinstance(lg, TrackioLogger)]
    if len(found) == 0:
        return None
    if len(found) > 1:
        raise RuntimeError(
            f"Found {len(found)} TrackioLoggers attached to the Trainer. "
            "Only one is supported for run resume across requeues."
        )
    return found[0]


def _params_to_dict(params: Any) -> Dict[str, Any]:
    """Normalise hparams to a flat dict for ``trackio.init(config=...)``."""
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(params, DictConfig):
            return OmegaConf.to_container(params, resolve=True)
    except ImportError:
        pass
    if isinstance(params, dict):
        return params
    if hasattr(params, "__dict__"):
        return vars(params)
    return {"params": str(params)}


def _cuda_available() -> bool:
    """Return True if any CUDA GPU is visible.

    Safe to call even when torch isn't installed.  Used to default
    ``auto_log_gpu`` to ``True`` on GPU jobs.
    """
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except ImportError:
        return False


def _to_scalar(v: Any) -> Optional[float]:
    """Coerce a metric value to a Python float, or ``None`` if not scalar."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    item = getattr(v, "item", None)
    if callable(item):
        try:
            numel = getattr(v, "numel", None)
            if callable(numel) and numel() != 1:
                return None
            return float(item())
        except (RuntimeError, ValueError, TypeError):
            return None
    return None
