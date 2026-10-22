# Third-party components

## CAPI

The CAPI backbone, loss, prediction head, and schedules adapt code from
[facebookresearch/capi](https://github.com/facebookresearch/capi), commit
`98b4fa17ee8eec8810c17022df9a27a44845368b`, distributed under Apache-2.0.
The full license is in `stable_pretraining/backbone/licenses/CAPI-Apache-2.0.txt`
and is included in built distributions. Adaptations integrate the dict-based
Module API, separate student/clustering optimizers, checkpointed EMA progress,
and reduced-precision and rectangular-grid handling.

## Jet

`stable_pretraining/backbone/jet.py` adapts the transformer conditioner and
affine-coupling architecture from [btrude/jet-pytorch](https://github.com/btrude/jet-pytorch),
commit `d71d0dcfdc8b190f9b01795fd188742ce669130d`, distributed under Apache-2.0.
The full license is in `stable_pretraining/backbone/licenses/JET_LICENSE` and is
included in built distributions. The rest of stable-pretraining retains its MIT license.

Changes include BCHW input, indexed permutations, a configurable exponential
scale with a lower floor, FP32-or-higher affine arithmetic and log determinants,
diagnostics, activation checkpointing, and checkpoint configuration validation.
This port uses its own state-dict layout; existing upstream and deepstats
checkpoints require an explicit conversion and cannot be loaded directly.
