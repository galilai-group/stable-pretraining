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

## RotNet

`stable_pretraining/backbone/nin.py` and `benchmarks/cifar10/rotnet-nin-eval.py`
adapt the Network-in-Network architecture, the frozen-feature classifiers and
their training configurations from
[gidariss/FeatureLearningRotNet](https://github.com/gidariss/FeatureLearningRotNet),
commit `edafaba6561003292017d0a44272414356c959ff`, distributed under the MIT
license. The full license is in `stable_pretraining/backbone/licenses/ROTNET_LICENSE`
and is included in built distributions. The rest of stable-pretraining retains
its MIT license.

Changes include a rewrite for current PyTorch, `BatchNorm1d` in place of the
original `BatchNorm2d` on flat features in the fully connected classifier, and
rotations applied on-device in `forward.rotnet` instead of in the data loader.
