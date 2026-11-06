# LeVJEPA token loader

`stable_pretraining.data.levjepa` feeds a LeVJEPA training step every view's *kept patches* instead of decoded
clips. LeVJEPA keeps about 5% of each view's patches, so the loader draws the views, crops and kept tokens first
(with v2's distributions), then reads and decodes only the JPEG MCUs those patches cover:

- **CPU workers** sample each batch's clips and view parameters, map every kept patch back to the stored frame, and
  splice the MCUs it needs from the frame store's memory-mapped files into one small JPEG per clip.
- **GPU server processes** decode those JPEGs (nvJPEG), and build every kept patch (squash, crop, flip, colour
  jitter, normalize) in Triton kernels, into a ring of patch buffers the training process reads over CUDA IPC.

This is an unreleased addition; install the current source as described in [the quickstart](quickstart.md).

## Install and data

```bash
pip install -e ".[levjepa]"   # stable-datasets[framestore], numba, triton
```

The loader reads *frame stores*: lance datasets of JPEG frames converted by stable-datasets so that any subset of a
frame's MCUs can be decoded on its own. Convert a store once (losslessly, in place, resumable):

```bash
python -m stable_datasets.framestore.convert /data/k400.lance
```

See stable-datasets' frame store guide for the options. The loader refuses a store that has not been converted.
It needs Linux and an NVIDIA GPU; start the MPS daemon (`nvidia-cuda-mps-control -d`) so its GPU servers share the
GPU with training.

## Use

```python
from stable_pretraining.data.levjepa import SharedTokenLoader, ViewRecipe

loader = SharedTokenLoader("/data/k400.lance", batch=64, index="shared", recipe=ViewRecipe())  # v2's views
global_patches, global_idx, local_patches, local_idx = next(loader)
# (64, 160, 768), (64, 160), (256, 32, 768), (256, 32): valid until the next call
```

Patches come normalized and in `PatchEmbed`'s `(c p1 p2)` order; `idx` index each view's flattened `T x N` patch
grid, so the encoder skips patch embedding of dropped tokens. `ViewRecipe` holds the clip length, view sizes and
scales, jitter and kept share; its defaults (`V2`) are LeVJEPA v2's.

Process counts are sized from the node: workers from the CPUs the job may use (two always stay free), two GPU
servers by default. `decoders="nvjpeg:4,cpu:6"` runs four nvJPEG servers and lets six workers also decode on the
CPU when the GPU servers fall behind; `resources=` caps the loader's share of CPUs, memory and GPU memory (by default
all usable CPUs, half the free memory and half the free GPU memory), and it shrinks to what fits instead of failing.

## What it changes, and what it costs

Two documented differences from v2's view builder: chroma is upsampled per MCU (as nvJPEG decodes; about 1/255
mean difference from libjpeg's), and local-view contrast pivots on the mean of the view's kept patches rather than
the full crop's. Training is unaffected within noise: ViT-S trained 3,000 steps from the same initialization, 4 data
seeds per loader (H100), reached a loss of 0.4691 ± 0.0045 with this loader and 0.4715 ± 0.0025 with v2's.

On an H100 node with 23 vCPUs, ViT-S trains at 2,828 clips/s with this loader (bf16, one CUDA graph) against 946
clips/s with v2's DataLoader on the same step; the GPU is busy with the training step 94% of the time. The loader
alone delivers about 8,400 clips/s, limited by the GPU's JPEG engines.
