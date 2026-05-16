
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
