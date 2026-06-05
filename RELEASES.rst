
Unreleased
----------

**Discoverability improvements**

- ``METHODS.md``: new root-level catalog table covering every method class and
  forward function with columns for forward fn, loss class, key callbacks, and
  paper reference. ``stable_pretraining/forward.py`` module docstring now
  cross-references ``stable_pretraining/methods/`` and ``METHODS.md`` for the full
  ``LightningModule`` catalog.
- Type annotations on the forward functions in ``stable_pretraining/forward.py``:
  ``batch: dict[str, Any]``, ``stage: str``, and ``-> dict[str, torch.Tensor]``
  return types. Added ``stable_pretraining/py.typed`` PEP 561 marker so mypy and
  pyright treat the package as typed.
- Top-level namespace: ``stable_pretraining.methods`` is now exposed as a lazy
  submodule. A curated set of common method classes (``SimCLR``, ``BYOL``,
  ``DINO``, ``DINOv2``, ``VICReg``, ``MAE``, ``NNCLR``, ``SwAV``,
  ``BarlowTwins``) are hoisted into the top-level ``__all__`` and
  ``_LAZY_ATTRS``, making ``import stable_pretraining as spt; spt.SimCLR``
  work without a deep import. The full catalog remains under
  ``stable_pretraining.methods``.
- Agent compatibility files: ``AGENTS.md`` (canonical instructions for all coding
  agents — repository layout, import patterns, core concepts, naming conventions,
  step-by-step guide for adding a new SSL method, and key design decisions);
  ``CLAUDE.md`` updated to point to ``AGENTS.md`` and add Claude-specific workflow
  notes; ``.github/copilot-instructions.md`` and ``.cursor/rules`` added as thin
  wrappers for GitHub Copilot and Cursor respectively.
- ``Module`` docstrings: added or expanded docstrings for ``training_step``,
  ``validation_step``, ``test_step``, ``predict_step``, ``on_train_start``,
  ``rescale_loss_for_grad_acc``, and ``after_manual_backward``.

- ``spt`` CLI: new ``spt web`` subcommand launches a local, dependency-free
  web viewer (stdlib ``http.server`` + Server-Sent Events, NFS-safe). Reads
  ``sidecar.json`` + ``metrics.csv`` (and optional ``media.jsonl``) under a
  given directory and renders a wandb-like UI with a metric tree, multi-run
  filter chips, group-by/sort, light/dark theme toggle, in-chart synced
  cursor tooltip, run-config modal, and a landing-page activity overview.
- ``spt web`` defaults to ``{cache_dir}/runs`` when no path is passed.
- ``RegistryLogger.log_image`` / ``log_video``: new methods matching
  Lightning's ``WandbLogger`` signatures. Existing callbacks gating on
  ``hasattr(logger, "log_image")`` start writing to disk without code
  changes. Files land under ``{run_dir}/media/<safe_tag>/`` with an
  append-only ``media.jsonl`` manifest. The web viewer renders these
  alongside scalar charts in the same metric tree.
- ``Manager._resolve_run_dir`` is now DDP-safe: rank-0 picks the
  ``run_dir`` and atomically publishes it under
  ``{cache_dir}/.rank_handoff/<launch_key>``; non-zero ranks block on
  that handoff and adopt the same value. Prevents ranks from generating
  divergent uuids and writing inconsistent ``.slurm_index`` entries (a
  silent data-loss source on multi-rank preempt+requeue). Rank detection
  uses Lightning's ``rank_zero_only.rank``. Override timeout via
  ``SPT_RANK_HANDOFF_TIMEOUT_S``.
- New documentation pages: :doc:`cache_dir` (run directory layout,
  resume / requeue / DDP semantics, media layout) and :doc:`cli`
  (full ``spt run`` / ``spt web`` / ``spt registry`` reference).
- New API pages: :doc:`api/registry` (RegistryLogger + Registry query)
  and :doc:`api/web` (``serve`` entry point).

**spt web — viewer improvements**

- **State persistence & shareable URLs**: all interactive state
  (selected runs, filters, group-by, sort, x-axis, smoothing, log-y, active
  tab, theme) is serialised to ``localStorage`` on every mutation and restored
  on load. The URL fragment is kept in sync (``#runs=…&tab=…``) so copying the
  address bar produces a link that reopens the exact same selection.

- **Chart value annotations**: each metric chart now has a compact table below
  it showing the last and best value for every visible series. The best row is
  highlighted; clicking a row toggles that series on/off. Lower-is-better is
  inferred automatically from metric names containing ``loss``, ``err``,
  ``perplexity``, or ``ppl``.

- **Runs table view**: a fourth **table** tab renders a horizontally-scrollable
  grid with one row per visible run and one column per hparam/summary key. Cells
  whose values differ across runs are highlighted in amber; identical cells are
  dimmed. Columns are sortable by click and filterable by a search box. The
  run-ID column and header row are sticky.

- **Run display names**: the sidebar and table now show ``display_name`` as the
  primary label with the raw path shown as a dimmed hint. Double-clicking the
  label opens an inline editor; changes are persisted via ``PATCH /api/run-meta``.

- **Notes editing**: the run detail modal has an editable notes textarea that
  auto-resizes to its content. Notes are saved on blur or ``Ctrl+Enter`` via
  ``PATCH /api/run-meta``.

- **Log auto-refresh / live tail**: the ``.out`` and ``.err`` tabs auto-refresh
  every 10 s while the selected run is running or stale, showing a pulsing
  **live** badge. A pause button stops the timer; a refresh button triggers a
  manual fetch. Scroll position is preserved when the user has scrolled up.

- **Elapsed time / duration**: each run row and the detail modal now show the
  elapsed or total duration. Running runs tick forward every 60 s without a full
  re-render. ``RegistryLogger.finalize()`` now records ``ended_at`` in the
  sidecar so completed durations survive restarts.

- **Heartbeat staleness**: runs whose heartbeat file is more than 5 minutes old
  are flagged as **stale** with an amber ⚠ indicator in the sidebar, topbar
  stats, detail modal, figures overview, and activity timeline. Filters,
  group-by, and sort all treat stale as a distinct status value via a centralised
  ``effectiveStatus()`` helper.

- **Scatter plot**: the figures tab renders a scatter plot below the metric
  charts when two or more runs are visible. X and Y axes are independently
  selectable from any numeric ``hparams.*`` or ``summary.*`` key across visible
  runs. Each run contributes one point (its final summary scalar). Axis
  selection is persisted to ``localStorage``.

- **CSV export**: a **download CSV** button in the figures toolbar downloads all
  visible metrics for the currently selected runs as a flat CSV
  (``run_id, run_name, metric, step, epoch, value``). Only the runs and metrics
  currently in view (respecting the metric-search filter) are included. No
  server round-trip — the data is built entirely from the in-memory metrics
  cache.

- **Zoom reset**: drag-selecting a region on any chart zooms all charts
  simultaneously (shared sync key). A **⤢** reset button appears in the chart
  title bar after zooming and resets all charts to their full x-range in one
  click. The drag-selection region is now styled with an accent-coloured border
  and fill.

- **Sidebar resize + virtual scroll**: the sidebar can be dragged to any width
  between 160 px and 600 px; the chosen width is persisted. The
  search/filter/sort controls are now fixed at the top of the sidebar while the
  run list scrolls independently below them. When more than 300 ungrouped runs
  are visible, the run list switches to a virtual-scroll window so SSE updates
  remain smooth at scale.

- **Keyboard shortcuts**: ``/`` focuses metric search, ``r`` focuses run
  search, ``t`` cycles tabs, ``Shift+A`` selects all, ``Shift+C`` clears all,
  ``?`` opens a help popover listing all shortcuts. ``Esc`` exits focused inputs
  and dismisses all modals and popovers.

- **Tag editing**: run tags are now displayed as editable pills in the detail
  modal. Clicking **star** on a pill removes the tag; a dashed ``+ tag`` input
  with autocomplete (populated from tags on other runs) adds new ones. All
  changes are persisted via ``PATCH /api/run-meta`` with optimistic updates and
  revert on failure.

Version 0.1
-----------

- Added `matmul_precision` config parameter to control TensorFloat-32 (TF32) precision on Ampere and newer GPUs.
- Base trainer offering the basic functionalities of stable-SSL (logging, checkpointing, data loading etc).
- Template trainers for supervised and self-supervised learning (general joint embedding, JEPA, and teacher student models).
- Examples of self-supervised learning methods : SimCLR, Barlow Twins, VicReg, DINO, MoCo, SimSiam.
- Classes to load templates neural networks (backbone, projector, etc).
- LARS optimizer.
- Linear warmup schedulers.
- Loss functions: NTXEnt, Barlow Twins, Negative Cosine Similarity, VICReg.
- Base classes for multi-view dataloaders.
- Functionalities to read the loggings and easily export the results.
- RankMe, LiDAR metrics to monitor training.
- Examples of extracting run data from WandB and utilizing it to create figures.
- Fixed a bug in the logging functionality.
