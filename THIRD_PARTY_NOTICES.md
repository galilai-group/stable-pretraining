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

## MMCR

The MMCR objective and reference-comparison tests follow
[ThomasYerxa/mmcr](https://github.com/ThomasYerxa/mmcr), commit
`db5a25caca93a55583fba5f9647f6eb36fca2efe`, especially
`mmcr/cifar_stl/loss_mmcr.py`. The integration adds an explicit `[B, V, D]`
contract, input checks, FP32-or-higher spectral operations, differentiable
distributed gathering, and the Module/Manager training interface.

MIT License

Copyright (c) 2023 Thomas Yerxa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
