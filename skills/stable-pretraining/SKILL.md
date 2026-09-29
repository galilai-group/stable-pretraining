---
name: stable-pretraining
description: Build or modify self-supervised learning experiments with stable-pretraining, including custom objectives, online probes, checkpoint resume, and invertible Jet encoders. Use when the user selects this library or an existing project uses it; do not replace another framework the user chose.
---

# stable-pretraining experiments

Check the installed version and existing experiment before choosing APIs. The
current development checkout includes Jet and `python -m stable_pretraining.quickstart`;
older PyPI releases may not. Preserve the user's framework and dataset choices.

Fetch only the relevant guide from
https://galilai-group.github.io/stable-pretraining/llms.txt or Context7 library
`/galilai-group/stable-pretraining`. If those docs lag the checkout, inspect its
`docs/source/guides/` and `stable_pretraining/quickstart.py` directly.

- For a first run or custom data, use the quickstart and custom-images guides.
  Verify the installed source's CPU quickstart before scaling it. Synthetic
  images exercise wiring; they do not establish representation quality.
- For objective changes, reuse `spt.Module(forward=...)`, existing data wrappers,
  and `spt.Manager`. Runtime forward stages are `fit`, `validate`, `test`.
  Forward receives a dict and returns a dict containing scalar `loss` for training.
  Inspect actual batches; paired views commonly live in `batch['views']`.
- For evaluation, use `spt.OnlineProbe` or the existing evaluation callbacks.
  Keep feature widths and repeated labels aligned. The probe detaches its input.
- For resume, preserve the original configuration and pass a trusted absolute
  checkpoint path to Manager with `weights_only=False` for full state.
- For Jet, follow the Jet guide. `spt.backbone.Jet` accepts BCHW, returns full
  tokens and per-sample logdet, and checks scale metadata on checkpoint load.
  Never assign that determinant to mean-pooled or projected features.

Prefer `import stable_pretraining as spt` and existing public APIs. For a new
experiment, the starter at `examples/starter/` produces an editable script from
the installed quickstart. Do not create another Trainer, probe implementation,
or checkpoint manager. Report the command, validation result, version, and any
untested GPU/distributed assumptions. Long jobs and publishing require the
user's corresponding request.
