# Copilot Instructions — stable-pretraining

Read [`AGENTS.md`](../AGENTS.md) at the repository root for full context.

## Quick reference for inline suggestions

- **Forward functions** live in `stable_pretraining/forward.py` — signature is always
  `(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]`
- **Loss classes** follow `{Method}Loss` naming in `stable_pretraining/losses/`
- **Method classes** live in `stable_pretraining/methods/` — 30 available, full list in `METHODS.md`
- All new public symbols need type annotations and a Google-style docstring
- Import from top-level namespace: `import stable_pretraining as spt`
- Batches are always `dict`, never raw tensors — do not unpack them positionally
- `Manager(...)()` is the correct programmatic entry point, not `Trainer.fit(...)`
