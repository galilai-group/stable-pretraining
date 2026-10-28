# LeVLJEPA integration and validation

This implementation adds **LeVLJEPA: End-to-End Vision-Language Pretraining
Without Negatives** as a reusable method and composable forward function.
It follows the [official implementation](https://github.com/MLO-lab/LeVLJEPA)
at revision `1fd6545c38407f0ab98af3e59070df483914e92b`.

## Components

| Requirement | Implementation | Verification |
|---|---|---|
| Image and text encoders | User-supplied modules; no compulsory model downloads | Offline tiny encoders, real ViT/GPT-2 example |
| Separate marginal projections | BN/GELU hidden layers and linear outputs | Feature shape and encoder gradient tests |
| Bidirectional prediction | Vision predicts detached text; text predicts detached vision | Independent objective/gradient comparison |
| Anti-collapse regularization | Sliced Epps–Pulley SIGReg on each marginal | Quadrature formula, collapsed inputs, bf16 tests |
| Framework integration | `LeVLJEPA(Module)`, `forward.levljepa`, `LeVLJEPALoss` | Dict API, composable-forward equivalence, exports |
| Training and resume | Standard Module optimizer configuration and Manager | Real optimizer steps and full checkpoint resume |
| Distributed behavior | Shared projection directions; differentiable global moments | Two-process gloo gradient comparison test |
| Data and evaluation | Existing Flickr8k image-level train/validation splits | Exact image overlap rejection; held-out retrieval |
| Documentation | Method catalog, API exports and this guide | Ruff, import tests and diff checks |

The reduced examples validate training and frozen-feature evaluation, not
DataComp-scale pretraining or published checkpoint quality. The ADE20K probe
is a disclosed reduced protocol, not an exact reproduction of the paper.

## Objective and gradients

For image/text encoder outputs, separate pre-projection heads produce `v` and
`t`. The vision predictor `p_v(v)` has the width of `t`, and the text predictor
`p_t(t)` has the width of `v`:

```text
prediction = (MSE(p_v(v), stopgrad(t)) + MSE(p_t(t), stopgrad(v))) / 2
loss = (1 - lambda_vision - lambda_text) * prediction
       + lambda_vision * SIGReg(v) + lambda_text * SIGReg(t)
```

The default weights are 0.02 per modality. An optional cosine alignment uses
the reference's `2 - 2*cos(prediction, target)` formula. Targets are detached
only for prediction; SIGReg continues to train both encoders. No negative
pairs, temperature, teacher, or EMA update is introduced.

The text encoder must return `[B, L, D]` hidden states (tensor, dictionary or
HuggingFace output). Pooling selects the last unmasked position, including
left-padded input. The example explicitly appends EOS before padding. Empty
text masks fail clearly instead of pooling padding.

SIGReg uses random unit directions, a 17-point grid on `[0, 3]`, Gaussian
characteristic-function weighting, and the two-sided trapezoidal statistic
from the reference. It runs in fp32 under mixed precision; explicit half/bfloat16
module conversions reconstruct quadrature constants in fp32 rather than
upcasting already-rounded constants. The random sequence
counter is checkpointed, advances during training only, and does not depend on
unrelated calls to the process RNG.

Under DDP, all ranks share directions and differentiably sum characteristic
function moments before forming the statistic. Global moments are normalized
by the true global count. The statistic is scaled by `global_count/world_size`,
matching the reference's local-batch scale for equal batches. Use equal batch
sizes on all ranks (`drop_last=True`) for the full objective's local means and
DDP gradient averaging. Gradient accumulation averages microbatch objectives;
it does not enlarge the batch used to estimate each SIGReg statistic.
All ranks must enter paired loss evaluation together;
use `stage="predict"` for inference without loss collectives.

The existing `methods.lejepa.SlicedEppsPulley` is not changed. Its construction-
time distributed detection and collective-return handling cannot provide the
runtime and autograd behavior needed here; a private regularizer implements
the same statistic with explicit distributed semantics.

## Framework reuse

`LeVLJEPA` inherits `spt.Module` and supplies the method forward function;
training, optimization and checkpoint resume use the existing Module/Manager
path. Projection and predictor heads use the framework's `backbone.MLP`
with BatchNorm/GELU. Zero-dropout modules are omitted to retain the existing
LeVLJEPA checkpoint parameter names. Three state-layout regression cases
cover linear, one-hidden-layer and two-hidden-layer heads.

The example's `LossHistory` callback records epoch component means for its
JSON report and updates the caption dataset epoch. It does not stop training,
replace EarlyStopping, implement optimization, or implement checkpoint saving.
The small dataset adapter supplies Flickr8k-specific EOS, caption selection
and image-identity checks. Its CLIP control reuses the paired encoder forward
and existing `CLIPLoss`; no separate training framework is introduced. This is
a locally trained CLIP-style control, not teacher-provided scores or a pretrained
OpenAI CLIP checkpoint. The upstream CLIP loss source is unchanged. Both methods
use the same randomly initialized reduced encoders and update budget, although
LeVLJEPA adds cross-modal predictors and therefore has more parameters. Equal sample/update budgets do not imply equal FLOPs. Exploratory validation
searches must disclose each method's candidate budget separately.

## API

```python
import stable_pretraining as spt

model = spt.LeVLJEPA(
    vision_encoder=vision_encoder,  # returns [B, vision_dim]
    text_encoder=text_encoder,      # returns last_hidden_state [B, L, text_dim]
    vision_dim=192,
    text_dim=192,
    embedding_dim=192,
    optim={"optimizer": {"type": "AdamW", "lr": 2e-4}},
)
output = model(batch, stage="fit")
output["loss"].backward()
```

The model returns a dict preserving batch metadata. `embedding` is the raw
vision feature for probes; `image_embedding` and `text_embedding` are the
regularized spaces. `image_prediction` means prediction **from image to text**,
and `text_prediction` means prediction **from text to image**. Loss components
are separately available. Supported stages are `fit`/`train`, `validate`/`val`,
`test`, and `predict`. Call `model.eval()` for inference, as usual.

To compose `spt.Module` directly, use `forward=spt.forward.levljepa` and attach
`vision_encoder`, `text_encoder`, `vision_pre_proj`, `text_pre_proj`,
`projector_vision`, `projector_text`, and `levljepa_loss` as attributes.

## Existing data and reduced training recipe

The example uses [jxie/flickr8k](https://huggingface.co/datasets/jxie/flickr8k),
revision `56f58c967835f7c508d684f36bd7897cca9d7634`. It contains 6,000 train,
1,000 validation and 1,000 test images, with five caption columns. The earlier reduced
experiments use only `caption_0` and seeded subsets of the existing train and
validation splits, without using the test split. The full-data follow-up below
also evaluates an independent test split. No new dataset wrapper is added
to `stable-datasets`. Exact pixel-duplicate images are removed within subsets,
and any exact train/validation image overlap raises an error. Near-duplicates
are not exhaustively detected. The dataset mirror's card is incomplete; use
the original Flickr8k terms for any redistribution. No dataset is bundled here.

The model uses randomly initialized reduced ViT and GPT-2 encoders, smaller
projection/predictor heads, 64-pixel inputs, and shorter captions. This retains
the method's two encoder families and objective while keeping local validation
practical. It is not the published tiny/base architecture recipe.

```bash
pip install -e . pytest ruff
python examples/levljepa_flickr8k.py \
  --train-samples 1024 --val-samples 256 --epochs 30 --seed 42 \
  --accelerator cpu --output outputs/levljepa-flickr8k-seed42
```

The first run downloads the existing dataset and GPT-2 **tokenizer**, not model
weights. `--train-parquet` and `--val-parquet` accept already downloaded Flickr8k
shards with the same schema; `--train-parquet` can take multiple shard paths.
`--tokenizer` also accepts a local directory.

By default, training uses `caption_0`. With `--caption-mode random`, each
training image uses one of its five captions per epoch, seeded by the run seed,
epoch and image index. Caption choices are independent of model RNG consumption
and repeat across objectives for identical image ordering. Validation keeps
`caption_0`; image identity, rather than caption rows, defines the data split.
The example uses zero-worker data loaders so epoch updates reach the dataset.

The script records before/after image-to-text and text-to-image retrieval R@1
and R@5 over the full validation gallery, random R@1, feature standard
deviations, epoch loss components, configuration, selected image hashes,
software versions, and the final step in `results.json`. Retrieval compares
each predictor to targets in the opposite modality. Positives are the paired
image identity; two different images sharing a caption are not merged. Earlier
first-caption experiment records used caption-string equality; their recorded
values are retained as historical results under that older rule. The full
1,000-image test has no duplicate first captions, and the five-caption test
already uses image-owner identities. No retrieval negatives enter the
training objective. A full local checkpoint is saved as `final.ckpt`.

```bash
python examples/levljepa_flickr8k.py \
  --train-samples 1024 --val-samples 256 --epochs 30 --seed 42 \
  --resume /absolute/path/to/final.ckpt \
  --output outputs/levljepa-flickr8k-resumed
```

Keep the architecture/data arguments unchanged when resuming. Manager restores
optimizer state and the step counter; the example's before metric then measures
the loaded model. Resume is not claimed to reproduce stochastic augmentations
bit-for-bit across restarts.

## Reference convention

The prediction term follows the pinned executable reference: elementwise-mean
MSE, then an average of the two directions. A literal Euclidean-sum reading of
paper Eq. (5) differs by a factor of `2*D` for common embedding width `D`, which
changes the effective relative regularization weight. The implementation keeps
the explicit code convention. The paper appendix and reference defaults also
differ in learning rate, regularizer weights and predictor dropout; these
reduced recipes do not claim to reproduce the published training run.

An independent kernel-form/analytic-derivative check and a pinned-author
20-update CPU FP32 comparison complement the training tests. Matching tiny
model updates does not prove equivalence for every architecture and device.

## Fixed Flickr8k training evidence

Four further CPU runs use both training shards: 5,999 unique training
images after exact deduplication, 1,000 validation images, and
1,000 independent test images after the fixed exact-image overlap and
deduplication checks. Training samples one of the five captions for each
visited image per epoch with `--caption-mode random`. Each epoch drops the
incomplete final training batch. Seeds 42/43 use the same reduced architecture,
optimizer and batch size as the paired control above, with 50 fixed epochs /
4,650 optimizer steps. No pretrained weights, hyperparameter search or
test-driven checkpoint selection are used.

Validation keeps `caption_0`. After all four runs finish, the original final
checkpoints are evaluated on all five test captions per image. Image-to-text
retrieval has 5,000 text candidates and five owner-image positives;
text-to-image has 1,000 image candidates and one owner-image positive.
Random R@1 is approximately 0.100% in either direction.

| Method | Seed | Test image→text R@1 | Test text→image R@1 | Test image→text R@5 | Test text→image R@5 |
|---|---|---|---|---|---|
| levljepa | 42 | 1.30% | 0.70% | 5.60% | 3.92% |
| clip | 42 | 2.20% | 2.16% | 9.50% | 8.42% |
| levljepa | 43 | 1.20% | 1.00% | 4.30% | 4.26% |
| clip | 43 | 1.60% | 1.90% | 6.70% | 7.58% |

Every final checkpoint reproduces its stored validation metrics after reload.
Full-data LeVLJEPA seed 42 also resumes from
4,650 to 4,743 steps,
with finite state and matching optimizer, SIGReg and BatchNorm counters.
These runs change both image count and caption diversity, so they do not
isolate either factor. The previous 512-image and current 1,000-image
validation galleries also differ. Two seeds and reduced, randomly initialized
encoders do not establish paper-scale quality or general superiority.

Reproduce each fixed method/seed training run with the complete pinned data:

```bash
python examples/levljepa_flickr8k.py \
  --method levljepa --seed 42 --caption-mode random \
  --train-samples 6000 --val-samples 1000 --batch-size 64 \
  --epochs 50 --threads 2 --accelerator cpu \
  --output /path/outside/test-working-directory/full-levljepa-seed42
```

Repeat with `--method clip` and seeds 42/43, using separate output directories.
The example saves validation metrics; the five-caption test numbers above
come from a separate final-checkpoint audit using the stated retrieval protocol.

## Reproducible frozen ADE20K probe

`examples/levljepa_ade20k.py` reads an existing ADEChallengeData2016 directory
and a saved vision checkpoint; it never downloads datasets or model weights.
It uses the native `Module`/`Manager` path, a frozen timm ViT, a single linear
1×1 head and existing AdamW/cosine configuration. It does not introduce a new
training framework or a stopping/checkpoint callback.

Prepare the original structure outside the repository:

```text
ADEChallengeData2016/
  images/training/*.jpg
  images/validation/*.jpg
  annotations/training/*.png
  annotations/validation/*.png
```

For a native reduced width-192/depth-4, 96px/patch-8 checkpoint:

```bash
python examples/levljepa_ade20k.py \
  --data /data/ADEChallengeData2016 \
  --checkpoint /checkpoints/final.ckpt \
  --output /results/ade-native-seed42 --seed 42
```

Match `--width`, `--depth`, `--heads`, `--image-size` and `--patch-size` to the
checkpoint. Strict state loading rejects mismatches. The CLIP control's native
checkpoint uses the same `vision_encoder.*` prefix. For a random encoder use
`--format random` without a checkpoint and retain the same architecture/seed.

A released author ViT-B/16 encoder can be used as an **evaluation positive
control**, separately from our trained results:

```bash
python examples/levljepa_ade20k.py \
  --data /data/ADEChallengeData2016 \
  --checkpoint /checkpoints/vision_encoder.safetensors --format author \
  --timm-model vit_base_patch16_224 --image-size 224 \
  --max-cache-gib 12 --output /results/ade-author-seed42
```

This example uses ImageNet normalization and deterministic square resizing;
it is a fixed diagnostic choice, not an assertion of the paper's unpublished
segmentation preprocessing details. Backbone features are cached as fp16 in
host RAM, subject to `--max-cache-gib`; no feature dataset is copied to disk.
The head trains for 20 epochs at default lr=0.001, weight_decay=0.0001, batch
size 32, with step-wise cosine decay to zero. Both training and inference
use fp32 arithmetic. Architecture and head configuration remain identical
across matched controls.

Training masks use nearest-neighbor resizing. Original labels 1..150 map to
0..149; original labels 0 and 255 are ignored. At evaluation, logits are
bilinearly resized to each original annotation size, with class chunks
limiting memory. Confusion counts determine pixel accuracy and class mIoU
over classes with nonzero union; the output includes the full confusion
matrix for independent recomputation. This evaluates official validation,
not hidden test labels.

`results.json` records the complete configuration, image counts, optimizer
steps, frozen encoder hash, initial head hash and final metrics. Frozen state
and expected update count are checked. A separate head checkpoint is saved.
Use a fresh output directory per run; existing results are not overwritten.
`--limit-train`, `--limit-val`, `--epochs`, `--device cpu` and `--workers 0`
support a bounded smoke check, which is not a full-data result.

### Recorded limitations

Earlier fixed CC3M-subset encoders (100,741 training images, seed 42) achieved
3.8257% ADE20K mIoU for LeVLJEPA, 4.1407% for CLIP and 1.3311% for a trained
head on random features, using 96px inputs and a 12×12 patch grid. Both
pretrained models beat random features, but LeVLJEPA trails CLIP.

The author-weight ViT-B/16 positive control achieved 28.0490% under a fixed
224px/14×14 ImageNet-normalized diagnostic; a separately recorded CLIP-
normalization check obtained 28.2547%. These are author pretrained weights,
not our training results, and the head protocol is not an exact paper
reproduction. Model capacity, pretraining data and input resolution all differ
from our small encoders, so this comparison does not isolate one cause.

A later CC12M subset used 250,000/10,000/10,000 train/validation/test pairs.
Two fixed 20-epoch runs per objective gave LeVLJEPA test R@1 of 1.17–1.18%
image→text and 1.10–1.19% text→image, versus CLIP 1.78–1.88% and 1.59–1.66%.
A four-candidate-per-method lr/weight-decay search selected on final-epoch
symmetric validation retrieval only; it did not produce a new independent
test result. Its seed-42 selected R@1 was 1.34%/1.26% for LeVLJEPA and
1.74%/1.44% for CLIP. Different gallery sizes and readouts prevent direct
comparison with the Flickr or directional test percentages.

These observations establish executable reduced training, not paper-scale
quality or method superiority. Full datasets, weights, environments, raw
predictions and exploratory experiment adapters remain outside the PR.

## Full-capacity subset follow-up

A separate seed-42 comparison used the pinned author repository's default
architecture: ViT-B/16 at 224px, a 12-layer GPT-2 text encoder, 768-dimensional
embeddings, 2,048-wide projection/prediction heads and 4,096 SIGReg projections.
Both objectives trained from matched encoder/projection initialization on
32,000 CC12M-subset pairs, using a true batch of 32 for 50,000 updates
(1.6 million sample exposures each). This follows the repository-default
profile, not the paper's full-scale training or the released DataComp weights.

Final directional retrieval used a separate 1,000-pair validation gallery with
one paired positive per query; random R@1 is 0.10%:

| Method | Image-to-text R@1 | Text-to-image R@1 |
|---|---:|---:|
| LeVLJEPA | 0.40% | 0.60% |
| CLIP control | 0.90% | 0.80% |

Both final model-only checkpoints passed strict reload checks. These are
single-seed validation results, not a new independent test or a controlled
scaling comparison with the earlier reduced-encoder experiments.

A no-update diagnostic used the first 512 training and first 512 validation
pairs as equal-sized galleries:

| Method | Training image/text R@1 | Validation image/text R@1 |
|---|---:|---:|
| LeVLJEPA | 12.50% / 23.44% | 0.98% / 1.17% |
| CLIP control | 73.83% / 73.24% | 1.56% / 1.56% |

Both methods show substantial training/validation gaps. In four augmented
LeVLJEPA training-mode batches of 32 pairs, the weighted regularizer-to-
prediction encoder-gradient norm ratio was 0.13-0.53 for vision and
0.044-0.097 for text. The combined gradient did not oppose the prediction
gradient in these batches. Feature variance and effective rank did not
indicate a constant representation. These bounded checks support a
poor-generalization diagnosis; they neither isolate its cause nor exclude
other training or implementation problems.

The initial two full-capacity ADE20K attempts failed before head training
because the supplied relative output directory produced a rejected cache path.
Corrected runs with absolute output paths completed: frozen linear-head mIoU
was 0.9731% for LeVLJEPA and 1.3751% for CLIP. The failed attempts remain
separate from these successful reruns. Both used the same head initialization,
20 training epochs and native-resolution confusion metrics, with unchanged
frozen backbones. This is an approximate head protocol, not paper reproduction.

## Expanded-subset diagnosis and controlled follow-up

A seed-42 follow-up expanded the training corpus to 128,000 CC12M-subset pairs
while preserving the 1,000-pair validation gallery and the full-capacity
repository-default architecture. Both methods used batch 32, 50,000 updates
and 1.6 million exposures. This is still much smaller than the paper's
pretraining, and the CLIP control is not an exact OpenCLIP reproduction.

The initial comparison used peak LR 2e-4 for both objectives. A development-
selected LeVLJEPA follow-up used LR 1e-4, as described below; all three runs
used the same data, seed, batch size and fixed update/exposure budget.

| Evaluation | LeVLJEPA LR 2e-4 | LeVLJEPA LR 1e-4 | Fixed CLIP control |
|---|---:|---:|---:|
| Validation image-to-text R@1 | 0.40% | 3.80% | 0.60% |
| Validation text-to-image R@1 | 0.40% | 2.40% | 0.90% |
| Pets frozen CLS test Top-1 | 4.3336% | 18.2066% | 8.2311% |
| Aircraft frozen CLS test Top-1 | 1.8002% | 7.8008% | 4.8905% |
| ADE20K frozen patch validation mIoU | 1.0030% | 3.2488% | 1.1091% |

Classification used L2-normalized CLS features, training-only per-dimension
standardization, a bias-free linear head, AdamW (LR 0.1, weight decay 1e-4),
batch 4,096 and 50 fixed epochs. ADE used the same approximate 20-epoch head
protocol as above. Confusion-derived metrics, identical control head
initialization and frozen-backbone hashes were independently verified.
Neither transfer task supports a paper-level representation-quality claim.

The marginal embeddings retained diversity, but the predictor outputs had
low effective rank from the first recorded 5,000-step validation onward.
A no-update diagnostic recalibrated BatchNorm using only 1,024 training
examples; LeVLJEPA validation R@1 remained 0.40% in both directions. BatchNorm
running statistics alone therefore did not explain this gap. This diagnostic
used BF16 encoders and FP32 heads, separately from the canonical BF16 metrics.
High marginal effective rank does not establish semantic feature quality.

Four fresh development-only ablations used the same data and initialization,
5,000 updates, 1,000 warmup steps and a shortened cosine schedule. The baseline
used LR 2e-4, symmetric SIGReg weights 0.02 and no predictor dropout. Each other
case changed one factor; the cosine floor was one tenth of the peak LR and
the warmup start LR remained 2e-6.

| Changed factor | Image-to-text R@1 | Text-to-image R@1 | Mean bidirectional R@10 |
|---|---:|---:|---:|
| Short-run baseline | 0.20% | 0.10% | 3.50% |
| Predictor dropout 0.1 | 0.70% | 0.60% | 4.20% |
| SIGReg weights 0.01 each | 0.20% | 0.20% | 2.85% |
| Peak LR 1e-4 | 1.10% | 0.40% | 8.05% |

The lower-LR case met a predeclared development continuation gate: at least
0.5 percentage points improvement in mean bidirectional R@1, 2 points in
mean R@10, and positive pairing margins. The selected LR 1e-4 completed a
fixed 50,000-step confirmation; its final checkpoint produced the follow-up
results in the table above. Image/text R@10 was 17.30%/16.60%, versus
7.10%/6.80% for the fixed CLIP control. Strict model-only reload and SHA256
backup checks passed. Classification and ADE used the same frozen encoder
hash; confusion metrics, head initializers and split identities were checked.

The confirmation used an A6000 because the earlier L40S was unavailable;
this is update-matched rather than hardware/FLOP-matched. LeVLJEPA's LR was
selected on development data, whereas this CLIP control retained its fixed
LR 2e-4. The single-seed improvement over this control does not establish
universal method superiority. No held-out retrieval test or downstream test
labels were used to choose the pretraining variant. The paper's LeVLJEPA
Pets/Aircraft linear-probe accuracy is 81.28%/46.38%, and ADE20K mIoU is
23.15%; the follow-up remains far below these differently pretrained models.
The paper has no directly comparable 1,000-gallery retrieval R@1 result.
Training data, model weights and local experimental adapters remain outside
the contribution.

## Verification

Run tests from a disposable working directory: repository test fixtures may
delete `outputs/` relative to their current directory. Retained experiment
artifacts must be kept elsewhere.

```bash
LEV_REPO=/absolute/path/stable-pretraining
cd "$(mktemp -d)"
PYTHONPATH="$LEV_REPO" python -m pytest \
  "$LEV_REPO/stable_pretraining/tests/unit/test_levljepa.py" \
  "$LEV_REPO/stable_pretraining/tests/unit/test_levljepa_ade20k.py" \
  "$LEV_REPO/stable_pretraining/tests/unit/test_clip_loss.py" \
  "$LEV_REPO/stable_pretraining/tests/integration/test_levljepa_training.py" -q

# On a machine with two CUDA GPUs:
PYTHONPATH="$LEV_REPO" python -m pytest \
  "$LEV_REPO/stable_pretraining/tests/integration/test_levljepa_training.py" \
  -m gpu -k nccl -q
```

The tests cover detached targets, formula/gradient agreement, half/bfloat16
quadrature constants, padding, composable/native parity, full-state resume,
real CPU gloo, MPS and two-GPU NCCL/SyncBatchNorm where available. ADE checks
independently verify label mapping, void exclusion and perfect native-mask
confusion counts. GPU-only tests skip on CPU/Mac; local passes are not GitHub
CI results. Unrelated upstream fixture fixes are excluded from this method
contribution, so no clean full-repository result is claimed.

## Sources

- [Pinned author implementation](https://github.com/MLO-lab/LeVLJEPA/tree/1fd6545c38407f0ab98af3e59070df483914e92b)
- [Paper](https://arxiv.org/html/2607.00784v1)
- [Flickr8k mirror revision](https://huggingface.co/datasets/jxie/flickr8k/tree/56f58c967835f7c508d684f36bd7897cca9d7634)
- [ADE20K dataset and download reference](https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/download_ADE20K.sh)


### Predictor dropout and recipe scope

`LeVLJEPA(..., predictor_dropout=0.1)` enables the dropout probability
reported for the paper's cross-modal predictors. The default remains `0.0`,
matching the public default configuration and existing experiments. Dropout
applies only to predictors; pre-projection heads remain unchanged. Parameter
and BatchNorm buffer keys are identical across these settings, so existing
zero-dropout state dictionaries load strictly. Inference requires `.eval()`.

Changing dropout alone does not reproduce the paper's data, batch statistics,
learning rate or regularization recipe. Report these separately. Two independent
single-GPU controls do not form a distributed statistical batch, and gradient
accumulation does not combine their SIGReg or BatchNorm statistics.
