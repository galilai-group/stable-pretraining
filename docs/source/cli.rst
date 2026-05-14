.. _cli:

CLI reference
=============

The ``spt`` command is installed by ``pip install -e .`` (or any wheel
install) via the ``[project.scripts]`` entry in ``pyproject.toml``. It
groups together training, log-management, registry queries, and the
web viewer.

::

    spt --help

.. contents:: Subcommands
   :local:
   :depth: 2


``spt run``
-----------

Launch a Hydra-driven training script.

::

    spt run <config> [HYDRA_OVERRIDES...]

Examples
^^^^^^^^

::

    # Single run from configs/myconfig.yaml
    spt run myconfig

    # Multirun (sweep) — also auto-selects the SLURM submitit launcher
    spt run myconfig -m trainer.max_epochs=100,200

    # Override individual keys
    spt run myconfig optim.lr=3e-4 model.backbone=resnet50

Multirun is detected from any of: ``--multirun``/``-m`` flag,
``hydra/launcher=…`` override, ``hydra.sweep…`` override, or a
comma-separated value (``key=a,b,c``). When detected without an explicit
launcher, ``hydra/launcher=submitit_slurm`` is appended.


``spt web``
-----------

Local wandb-like web viewer.

::

    spt web [DIRECTORY] [--host 127.0.0.1] [--port 4242] [--poll 1.0] [--cache-dir DIR]

Without ``DIRECTORY``, scans ``{cache_dir}/runs`` where ``cache_dir`` is
resolved from ``--cache-dir`` > ``SPT_CACHE_DIR`` env var >
``spt.set(cache_dir=...)`` global config. The viewer parses
``sidecar.json`` + ``metrics.csv`` (+ ``media.jsonl`` when present) under
the given tree and renders charts, the run table, and any logged images
or videos. See :doc:`api/web` for the Python entry point.

Examples
^^^^^^^^

::

    # Run on the cluster, port-forward over SSH from your laptop:
    #   ssh -L 4242:127.0.0.1:4242 user@cluster
    spt web

    # Serve a specific tree on a non-default port
    spt web /scratch/my_sweep --port 4243

    # Bind publicly (NO auth — only do this on a trusted network)
    spt web --host 0.0.0.0


``spt dump-csv-logs``
---------------------

Aggregate Hydra-multirun CSV logs and save them in the smallest
losslessly-compressed format among parquet/zstd, parquet/gzip,
parquet/brotli, feather, csv, pickle.

::

    spt dump-csv-logs <input_dir> <output_basename> [{max,last,all}]

The aggregator picks the highest value (``max``), the last row
(``last``), or keeps all rows (``all``) per metric column. Numeric
columns aggregate as the mode says; non-numeric take the last non-null.


``spt registry ...``
--------------------

Query the local run registry. Requires ``cache_dir`` resolved (same
priority as ``spt web``); pass ``--cache-dir`` or set ``SPT_CACHE_DIR``.

``spt registry ls``
^^^^^^^^^^^^^^^^^^^

List runs in a table, optionally filtered/sorted.

::

    spt registry ls [--tag TAG] [--status STATUS] [--alive/--dead]
                    [--sort COL] [-n LIMIT] [--cache-dir DIR]

``spt registry show``
^^^^^^^^^^^^^^^^^^^^^

Dump one run's full sidecar — status, run_dir, hparams, summary metrics,
checkpoint path.

::

    spt registry show <run_id> [--cache-dir DIR]

``spt registry best``
^^^^^^^^^^^^^^^^^^^^^

Top N runs ranked by a summary metric. Defaults to highest first; pass
``--asc`` for losses.

::

    spt registry best <metric> [--tag TAG] [-n N] [--asc] [--cache-dir DIR]

``spt registry export``
^^^^^^^^^^^^^^^^^^^^^^^

Export filtered runs as CSV or Parquet (auto-detected from extension).
``hparams`` and ``summary`` are flattened into columns.

::

    spt registry export <output.{csv,parquet}> [--tag TAG] [--status STATUS]

``spt registry scan``
^^^^^^^^^^^^^^^^^^^^^

Refresh the SQLite cache from sidecar files. Incremental by default
(only re-parses sidecars whose mtime advanced); ``--full`` re-ingests
everything (use after a schema change).

::

    spt registry scan [--full] [--cache-dir DIR]

``spt registry migrate``
^^^^^^^^^^^^^^^^^^^^^^^^

One-shot conversion from a legacy server-backed ``registry.db`` to the
filesystem sidecar layout. Run once after upgrading; subsequent
``scan`` invocations rebuild the cache from the sidecars.

::

    spt registry migrate <legacy_registry.db> [--cache-dir DIR] [--overwrite]
