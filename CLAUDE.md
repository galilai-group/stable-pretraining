# stable-pretraining — Claude Instructions

> The full agent instructions for this repository are in [`AGENTS.md`](./AGENTS.md).
> Read that file first and treat it as authoritative.

## Claude-specific notes

### Memory and context

- This repo has 30 SSL method classes. When working on any specific method, read its file in `stable_pretraining/methods/` before writing code — do not rely on pattern-matching from other methods, as hyperparameter defaults and architecture choices differ.
- `METHODS.md` is the ground-truth index of all methods. Check it before claiming a method does or does not exist in the library.
- The 9 forward functions in `forward.py` are the composable form; the 30 classes in `methods/` are the batteries-included form. Both are valid entry points — choose based on context.

### Preferred workflow for code changes

1. Read the relevant source file(s) in full before proposing anything
2. State what you found before proposing changes
3. Show diffs, not full file rewrites, for files over 100 lines
4. After any change to `__init__.py`, verify the import still works by tracing the lazy-load path through `_LAZY_ATTRS` and `_LAZY_SUBMODULES` manually

### Docstring style

This project uses Google-style docstrings (configured in `pyproject.toml` under `[tool.ruff.lint.pydocstyle]`). Always match the style in `stable_pretraining/forward.py` (the best-documented file in the codebase) when writing new docstrings — it uses `Args:`, `Returns:`, and `Note:` sections.

### Commands

**Installation:**
```bash
pip install -e ".[dev]"          # includes pytest, ruff, sphinx
pip install -e .                 # core only
```

**Running experiments:**
```bash
spt examples/simclr_cifar10_config.yaml                   # From YAML config
spt examples/simclr_cifar10_config.yaml trainer.max_epochs=50  # With overrides
spt examples/simclr_cifar10_slurm.yaml -m                 # SLURM multirun
```

**Testing:**
```bash
python -m pytest stable_pretraining/tests -m unit --verbose   # Fast unit tests (CI default)
python -m pytest stable_pretraining/tests -m integration      # Integration tests
python -m pytest -m "not slow"                                # Skip slow tests
# Markers: unit, integration, gpu, slow, download, ddp
```

**Linting:**
```bash
ruff check stable_pretraining --fix
ruff format stable_pretraining
pre-commit run --all-files
```

**Registry CLI:**
```bash
spt registry ls                    # List runs
spt registry best val_acc -n 5     # Top 5 by metric
spt registry export sweep.csv      # Export to CSV
spt registry scan --full           # Rebuild SQLite cache
```
