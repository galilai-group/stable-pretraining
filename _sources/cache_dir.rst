.. _cache_dir:

Cache directory & run layout
============================

stable-pretraining persists every run under a single ``cache_dir`` so
that scanners (registry, web viewer) can find them without external
state. This page documents how that directory is laid out, how a run
gets assigned its own subdirectory, and how preempt/requeue and DDP
behave inside it.

Resolution order
----------------

``cache_dir`` is resolved from, in order:

1. Explicit ``--cache-dir`` flag on a CLI command (e.g. ``spt registry``,
   ``spt web``).
2. ``SPT_CACHE_DIR`` environment variable.
3. ``spt.set(cache_dir=...)`` global config (set in code or via Hydra).
4. Default: ``~/.cache/stable-pretraining``.

Directory layout
----------------

::

    {cache_dir}/
    ├── runs/                                      # one run dir per training launch
    │   └── 20260425/                              # date  (YYYYMMDD)
    │       └── 135323/                            # time  (HHMMSS)
    │           └── a3f1b2c4d5e6/                  # run_id (uuid4 hex, 12 chars)
    │               ├── sidecar.json               # status, hparams, summary, …
    │               ├── metrics.csv                # Lightning CSVLogger output
    │               ├── hparams.yaml
    │               ├── heartbeat                  # touched on every log_metrics
    │               ├── run_meta.json              # used by checkpoint→run_dir lookup
    │               ├── media.jsonl                # one event per log_image / log_video
    │               ├── media/                     # files referenced by media.jsonl
    │               │   └── <safe_tag>/
    │               │       └── <step>_<i>.{png,mp4,…}
    │               └── checkpoints/
    │                   └── last.ckpt
    ├── registry.db                                # SQLite cache (rebuildable from sidecars)
    ├── .slurm_index/                              # SLURM-key → run_dir map (used on requeue)
    └── .rank_handoff/                             # DDP rank-0 → rank-N run_dir handoff


Run-id assignment
-----------------

Every run gets a fresh ``uuid4().hex[:12]`` (48 bits of entropy) — even
under SLURM, where the same job-id can otherwise leak across consecutive
``srun --pty`` invocations. Distinct runs essentially never collide.

The path layout (``YYYYMMDD/HHMMSS/<uuid>``) makes it trivial to ``rm
-rf`` runs from a given day and gives a chronological view in ``ls``.

Resume & requeue
----------------

``Manager`` resumes a run dir via two strategies, in priority order:

**Strategy 1 — explicit checkpoint path.**
If you pass ``ckpt_path=…`` and the directory containing that ckpt has
a ``run_meta.json`` next to it, the recorded ``run_dir`` is reused.

**Strategy 2 — SLURM requeue index.**
On a fresh launch, ``Manager`` writes
``cache_dir/.slurm_index/<SLURM_JOB_ID[_TASK_ID]>`` pointing at the new
``run_dir``. On a SLURM-driven preempt+requeue (detected via
``SLURM_RESTART_COUNT >= 1``), the index file is read and the job
resumes into the same directory — preserving the metrics, sidecar, and
checkpoint history. Interactive ``srun --pty`` reruns never bump
``SLURM_RESTART_COUNT``, so they fall through to a fresh dir even
though they share the SLURM job-id.

Distributed training (DDP)
--------------------------

In multi-rank training every rank is its own process and each one
calls ``Manager.__call__`` independently before the Trainer / Strategy
attaches. To keep all ranks writing into the same ``run_dir``:

* **Rank 0** runs the resume / fresh-create logic, then atomically
  publishes the chosen path to ``cache_dir/.rank_handoff/<launch_key>``
  via temp+``replace``.
* **Rank N** blocks on that file (polling every 50 ms; default timeout
  60 s) and adopts its value.

The ``launch_key`` is shared by every rank in the same launch but
unique between concurrent launches:

================================  =======================================================
Launcher                          Key
================================  =======================================================
SLURM batch / array               ``slurm-<JOB_ID>[_<TASK_ID>]``
``torchrun`` (torchelastic)       ``elastic-<TORCHELASTIC_RUN_ID>``
Local DDP / Lightning subprocess  ``local-<MASTER_ADDR>-<MASTER_PORT>-<pgid>``
Single-process                    ``None`` — handoff is skipped
================================  =======================================================

Rank detection uses ``lightning.pytorch.utilities.rank_zero.rank_zero_only.rank``,
which Lightning initialises from ``RANK`` / ``SLURM_PROCID`` /
``JSM_NAMESPACE_RANK`` at import time and updates from
``Strategy.global_rank`` once a Strategy attaches. Sticking with
Lightning's mechanism keeps detection consistent with every
``@rank_zero_only``-gated logger we ship.

If rank-0 crashes before publishing, rank-N falls back to local
resolution (with a loud warning) rather than blocking forever — only
rank-0 actually writes via ``@rank_zero_only`` loggers, so the worst
case is an orphaned empty rank-N directory, never data loss. Override
the timeout via ``SPT_RANK_HANDOFF_TIMEOUT_S=120`` for unusually slow
NFS clusters.

Media (images, videos)
----------------------

Callbacks calling ``trainer.logger.log_image(...)`` or
``trainer.logger.log_video(...)`` write into the run dir at
``{run_dir}/media/<safe_tag>/<step:08d>_<i>.<ext>`` with a manifest
line per file in ``{run_dir}/media.jsonl``. The web viewer reads that
JSONL and renders one panel per tag with a step slider; one row per
visible run.

Tag separators (``/``) become ``__`` on disk so the tag becomes a single
safe directory. Tags are still stored verbatim in the manifest, so the
UI groups them under the same ``/``-tree as scalar metrics.

Querying the registry
---------------------

``Manager`` writes a ``sidecar.json`` snapshot on every flush (status,
hparams, latest summary metric values, checkpoint path, tags). A
separate scanner walks ``{cache_dir}/runs/**/sidecar.json`` and
upserts each into a SQLite cache (``{cache_dir}/registry.db``):

::

    spt registry scan          # incremental refresh
    spt registry ls            # show all runs
    spt registry show <id>     # full hparams + summary
    spt registry best <metric> # top runs by metric
    spt registry export runs.parquet

The DB is purely a cache — deleting it just means the next ``scan``
takes longer. The sidecars themselves are the source of truth.

See :doc:`cli` for the full command listing and :doc:`api/registry`
for the Python query API.
