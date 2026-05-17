# stable-pretraining — Claude Instructions

> The full agent instructions for this repository are in [`AGENTS.md`](./AGENTS.md).
> Read that file first and treat it as authoritative. The sections below only
> cover Claude-specific behavior; everything else (commands, naming, design
> decisions) lives in `AGENTS.md` so it stays in one place.

## Claude-specific notes

### Memory and context

- When working on any specific method, read its file in `stable_pretraining/methods/` before writing code — do not rely on pattern-matching from other methods, as hyperparameter defaults and architecture choices differ.
- [`METHODS.md`](./METHODS.md) is the ground-truth index of all methods and forward functions. Check it before claiming a method does or does not exist in the library.
- Forward functions in `forward.py` are the composable form; classes in `methods/` are the batteries-included form. Both are valid entry points — choose based on context.

### Preferred workflow for code changes

1. Read the relevant source file(s) in full before proposing anything
2. State what you found before proposing changes
3. Show diffs, not full file rewrites, for files over 100 lines
4. After any change to `__init__.py`, verify the import still works by tracing the lazy-load path through `_LAZY_ATTRS` and `_LAZY_SUBMODULES` manually

### Docstring style

This project uses Google-style docstrings (configured in `pyproject.toml` under `[tool.ruff.lint.pydocstyle]`). Match the style in `stable_pretraining/forward.py` when writing new docstrings — it uses `Args:`, `Returns:`, and `Note:` sections.
