# Third-party components

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

## ReSA reference implementation

`stable_pretraining/tests/unit/test_resa.py` contains a single-process reduction of
the ReSA loss and Sinkhorn-Knopp normalization from
[winci-ai/resa_cifar](https://github.com/winci-ai/resa_cifar) (`method/base.py` and
`method/resa.py`), commit `bc5d70a48ed9`, distributed under Apache-2.0 (Copyright
(c) Winci). The tests use it only as a reference to check that `ReSALoss` matches
the official loss and gradients. The full license is in
`stable_pretraining/tests/unit/licenses/RESA_CIFAR_LICENSE`. Tests are not included
in built distributions.
