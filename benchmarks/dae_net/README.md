# DAE-Net: ShapeNet Part four-category benchmark

Categories: Mug, Chair, Lamp and Skateboard.

Train one DAE-Net per category through `spt.DAENet` and `spt.Manager`, then score
semantic parts using the released category-wide branch mapping. This recipe
trains on **all category geometry, including test shapes**; semantic labels are
used only during scoring, and the mapping is fitted on the evaluated labels.
The scores describe full-category co-segmentation, not held-out generalization.

[Paper](https://arxiv.org/abs/2311.13125) ·
[Author implementation and data instructions](https://github.com/czq142857/DAE-Net#datasets-and-weights)

Training datasets and samplers are provided by `spt.data`. Offline evaluation
helpers live alongside `evaluate.py`: `surface_labels.py` loads and matches
surface labels, `occupancy_evaluation.py` accumulates reconstruction counts,
and `part_segmentation_evaluation.py` implements semantic mapping and scoring.

## Install and prepare data

From the repository root:

```bash
python -m pip install -e .
python -m pip install h5py
```

Download the preprocessed ShapeNet Part archive linked by the authors through
[BAE-Net](https://github.com/czq142857/BAE-NET#datasets-and-weights), and extract it
locally. This benchmark does not download data. Point `--data-root` at a directory
containing the extracted `data/` directory. Each category needs:

```text
data/03797390_mug/
  03797390_vox.hdf5
  03797390_vox.txt
  03797390_test_vox.txt
  points/<shape_id>.txt
```

The full HDF5 file supplies both training and test voxels; the paired TXT gives
its row identities. The test TXT selects the scoring shapes. Preparation
validates array shapes/types, IDs and test surface labels, preserves full-file
training order, and removes repeated test IDs in first-occurrence order. It
writes `train.jsonl`, `test.jsonl`, `evaluation_labels.jsonl` and `prepared.json`.

```bash
python benchmarks/dae_net/prepare.py \
  --category mug --data-root /path/to/extracted \
  --output-dir outputs/dae-manifests/mug
```

For a different extraction layout, pass `--category-dir PATH_RELATIVE_TO_DATA_ROOT`.
Use a fresh output directory. Training geometry has no label dependency; the
preparer reads the test surface files to prepare the separate scoring manifest.
Manifest and source identities are checked before training and on resume.

## Train

[config.json](config.json) contains the four category recipes. All use FP32,
seed 42, batch size 16, fixed-rate Adam at `2e-4`, 8,192 queries per shape and
10 epochs per era. An epoch repeats complete ordered category passes until at
least 12,500 exposures are reached. Resolution-64 queries use a deterministic
one-of-four offset; the revival loader visits each shape once using voxels only.

| Category | Synset | Shapes | Branches | Query resolution | Sparsity weight | Epochs | Updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mug | 03797390 | 184 | 8 | 64 | 0.02 | 160 | 125,120 |
| Chair | 03001627 | 3,746 | 16 | 32 | 0.002 | 320 | 299,840 |
| Lamp | 03636649 | 1,546 | 8 | 32 | 0.01 | 160 | 139,200 |
| Skateboard | 04225987 | 152 | 8 | 64 | 0.001 | 160 | 126,240 |

```bash
python benchmarks/dae_net/train.py \
  --category mug --data-root /path/to/extracted \
  --manifest-dir outputs/dae-manifests/mug \
  --output-dir outputs/dae-mug --accelerator gpu
```

Repeat preparation/training with another `--category` for its recipe. Add
`--dry-run` to verify inputs and print the effective settings without training.
No scheduler allocation, W&B account or server-specific path is required by the
scripts. Training writes `run-config.json`, CSV/registry logs and completed-epoch
checkpoints under Manager’s run directory inside the output directory.
`result.json` records the exact completed-epoch checkpoint path. `spt web outputs/dae-mug` opens the local
run viewer. No semantic labels or validation scores select training checkpoints.
The latest completed checkpoint includes any era rollback already applied.

The benchmark directly reads HDF5 with `num_workers=0`. The recorded experiments
below used the same model, sampling recipe and training controller with a uint8
RAM cache. Cache/direct sample equivalence is checked; this is not a claim that
a fresh training run will reproduce scores bit-for-bit. Default-width models
are used throughout. Mixed precision is outside this recipe.

### Single-node DDP

The DDP path uses Linux, one node, FP32, one Adam update per step and
`num_workers=0`. Set `--devices N` for any positive process count that fits
the available hardware and leaves every rank a nonempty batch. For DDP,
`N >= 2`; use CPU/Gloo or one GPU per process with NCCL. For example:

```bash
python benchmarks/dae_net/train.py \
  --category mug --data-root /path/to/extracted \
  --manifest-dir outputs/dae-manifests/mug \
  --output-dir outputs/dae-mug-ddp --accelerator gpu --devices 2 --strategy ddp
```

For an externally allocated node, use
`torchrun --standalone --nproc_per_node=N benchmarks/dae_net/train.py ... --devices N`.
The launcher process count and `--devices` must agree.
Use `--accelerator cpu` for Gloo. One device remains the default.

`--batch-size` always means **global** batch size: 16 on two ranks gives 8 shapes
per rank.
The ordered exposure sequence, query offsets and updates per epoch match the
single-device recipe. Partial global batches are retained, with unequal nonempty
rank shards when needed. A tail smaller than the process count is rejected before
training. The effective config records full and partial rank batch sizes.

Sparsity uses the global existence mean. Reconstruction and other penalties are
weighted by their global element counts. Logged loss describes the global batch;
epoch loss weights each batch by its global shape count, including the tail.
The local scalar used for backpropagation can differ. Era IoU pools counts within
each global batch, then averages the last epoch's batch IoUs equally. Rank zero
traverses the usage dataset once, revives branches and broadcasts weights/ages.
Every rank clears Adam history; rejection restores network weights only.

For a short recovery check, add `--epochs 4 --era-length 2 --batch-size 4
--shapes-per-epoch 184 --limit-train-batches 2 --stop-after-epochs 2`. Repeat with
the same settings, remove `--stop-after-epochs 2`, and add
`--resume outputs/dae-mug-ddp`. Resume requires the same world size, global batch,
sampling protocol, manifest, device/software environment and completed epoch.
Checkpoints store shared era state and separate RNG state for each rank. Old
single-device checkpoints retain their original single-device loading path;
cross-world-size continuation is rejected. Multi-node, elastic world size,
FSDP, mixed precision, gradient accumulation and mid-epoch resume are outside
this path. Custom code should use `spt.Manager`,
`spt.data.DAENetDataModule(..., distributed=True)` and
`Trainer(use_distributed_sampler=False, devices=N,
strategy="ddp_find_unused_parameters_true")`. This strategy keeps gradient buckets
stable across fresh-process recovery; the recipe records it in the run config.
Resume requires the same strategy as the original run.

The four full-category learning results below remain single-GPU results.
Short DDP checks establish execution and recovery only.

### Check execution and recovery

Use a separate directory and explicitly shortened settings for an execution
check. Batch-limited results are marked `smoke` and must not be reported as the
full benchmark. The following first runs two epochs of a four-epoch budget:

```bash
python benchmarks/dae_net/train.py \
  --category mug --data-root /path/to/extracted \
  --manifest-dir outputs/dae-manifests/mug --output-dir outputs/dae-smoke \
  --epochs 4 --era-length 2 --batch-size 1 --shapes-per-epoch 184 \
  --limit-train-batches 2 --stop-after-epochs 2 --accelerator gpu

python benchmarks/dae_net/train.py \
  --category mug --data-root /path/to/extracted \
  --manifest-dir outputs/dae-manifests/mug --output-dir outputs/dae-smoke \
  --epochs 4 --era-length 2 --batch-size 1 --shapes-per-epoch 184 \
  --limit-train-batches 2 --accelerator gpu \
  --resume outputs/dae-smoke
```

For a full run, keep the original recipe and pass its latest completed-epoch
checkpoint (or its output directory containing `result.json`) with `--resume`. Recovery requires unchanged data, settings and the
same device/software environment; incomplete-epoch checkpoints are rejected.
Completing the final optimizer step is insufficient: the controller must finish
the epoch's era decision and RNG capture before the checkpoint is resumable.
`--stop-after-epochs` changes only where this invocation stops, preserving the
total budget in checkpoint identity. Existing checkpoints must be trusted.

## Score a checkpoint

Use the checkpoint path printed by training and saved in `result.json`. This
command evaluates the complete Mug test list, without subsampling points:

```bash
python benchmarks/dae_net/evaluate.py \
  --checkpoint /path/from/result.json/epoch0159.ckpt \
  --run-config outputs/dae-mug/run-config.json --require-final \
  --data-root /path/to/extracted \
  --manifest outputs/dae-manifests/mug/test.jsonl --split test \
  --metric semantic \
  --labels-manifest outputs/dae-manifests/mug/evaluation_labels.jsonl \
  --device cuda --chunk-size 1024 --output-dir outputs/dae-mug-evaluation
```

Use `--device cpu` if needed. `--require-final` checks the completed training
budget and verifies that current weights equal the accepted era snapshot. For
an intermediate checkpoint, omit that flag; `--run-config` still checks its
identity. Historical native checkpoints can be scored with their original
`run-config.json`, or without `--run-config`/`--require-final` when only weights
are available; in that case completion and final-snapshot identity are unverified.
Author PyTorch checkpoints use a different format and are not accepted directly.

The evaluator transforms surface coordinates `(x,y,z) -> (z,y,-x)` without
rescaling, predicts branch occupancy, and fills low-confidence labels from the
nearest point with occupancy greater than 0.4. Raw and postprocessed predictions
receive separate category-wide branch mappings using the released search.
IoU is averaged over fixed semantic parts per shape, then equally over shapes;
a part absent in both prediction and reference has IoU 1. The mapping uses test
labels. These labels do not enter training.

Outputs are `summary.json`, `per_shape.csv`, `per_part.csv`, `mapping.json` and
`predictions.npz`. Use a new output directory. `--max-shapes N` explicitly marks
an evaluation as a prefix smoke check, even if N covers the full list. Offline
occupancy scoring is also available via `--metric occupancy` without the labels
manifest; it is distinct from the online era IoU below.

## Recorded results

One seed; no matched reference-run or multi-seed equivalence claim. These are
existing full-category runs, not scores from the shortened recovery example.
Pretrained checkpoints are not included in this submission. The training
command produces the native checkpoint used by the scoring command above.

| Category | Test shapes | Semantic part mIoU (%) | Raw mIoU (%) |
| --- | ---: | ---: | ---: |
| Mug | 38 | 90.94 | 92.90 |
| Chair | 704 | 86.01 | 84.73 |
| Lamp | 286 | 69.94 | 70.68 |
| Skateboard | 31 | 62.82 | 64.47 |

The primary score includes nearest-confident-point postprocessing. Raw and
postprocessed predictions use separately fitted branch-to-label mappings.

| Category | First-era online IoU (%) | Last accepted-era online IoU (%) | Accepted epoch |
| --- | ---: | ---: | ---: |
| Mug | 78.33 | 94.60 | 130 |
| Chair | 58.51 | 87.52 | 320 |
| Lamp | 43.97 | 71.90 | 150 |
| Skateboard | 62.85 | 91.02 | 160 |

Online IoU averages batch-pooled occupancy IoU during each era's final training
epoch. The first measurement is at epoch 10, not at initialization. Mug and Lamp
rolled back their final candidate eras; final saved weights came from epochs
130 and 150. Epoch numbers above count completed epochs.

Training histories:
[Mug](https://wandb.ai/Brown-Research/dae-net-migration/runs/official-mug-34b46b19),
[Chair](https://wandb.ai/Brown-Research/dae-net-migration/runs/official-chair-282289ef),
[Lamp](https://wandb.ai/Brown-Research/dae-net-migration/runs/official-lamp-cdc5c3ca),
[Skateboard](https://wandb.ai/Brown-Research/dae-net-migration/runs/official-skateboard-a6a3f67e).
Reviewer access to these histories is not yet confirmed. Training and scoring
commands above use local outputs and do not require W&B.

## Verification

The preparation and dry-run paths were checked on all four categories, preserving
the original full-category training identities and Mug/Lamp test identities.
Direct and cached reads were identical for 12 Mug shape/query-offset cases.
On one RTX 3090, the four-epoch smoke command above produced identical model,
Adam, scheduler and controller states when run continuously or resumed after
epoch two. Era decisions were not replaced or forced. This checks eight updates
and a revival boundary, not long-run reproducibility or every rollback outcome.

The maintained evaluator re-scored the existing final Mug checkpoint on all 38
test shapes (107,459 points). Both mIoU values matched the original summary
exactly; the check allowed an absolute tolerance of `1e-6` in fractional units.
The GPU environment used Python 3.11.11, PyTorch 2.11.0+cu128 and Lightning 2.6.1.
This validation did not retrain the full experiment or re-score Lamp.

The submitted core suite covers reference model/loss/gradient behavior,
dict-based forward, Gloo loss/parameter-gradient/Adam parity at two, three and
four processes, era control, and natural single-device/multi-process Manager recovery.
Tests generate their own data; no ShapeNet download or checkpoint is needed.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q \
  stable_pretraining/tests/unit/test_dae_net.py \
  stable_pretraining/tests/unit/test_dae_net_distributed.py \
  stable_pretraining/tests/integration/test_dae_net_resume.py
```

Run on Linux with local process communication available. The dev extra
includes `h5py` for the synthetic HDF5 recovery test. Natural six-epoch
Manager training at one, two, three and four processes is compared against
new-process recovery after epochs three and four; separate local validation also
exercised the public CPU DDP CLI. Extended data/scoring tests remain outside this
core suite.

On two L40S GPUs using NCCL, real Mug data and the default backbone,
continuous four-epoch training matched 2+2 recovery exactly in model, Adam,
scheduler, controller and per-rank RNG state. This used global batch four, two
batches per epoch and two epochs per era, with PyTorch 2.11.0+cu128 and Lightning
2.6.1. These short checks validate execution and recovery; the learning results
above remain the original single-GPU experiments.

The GPU recovery check was run on the packaged implementation. See the
accompanying PR for test, documentation and CI status.
