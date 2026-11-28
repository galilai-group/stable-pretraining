# RC-aux: full-model Reacher comparison

This benchmark trains a learnable encoder, action-conditioned latent dynamics,
and a reachability head through `spt.RCaux` and `spt.Manager`. It compares
prediction-only training with RC-aux from the same initial world-model weights
and with exactly the same minibatch indices, optimizer, and number of updates.
Both conditions train from scratch. The `lewm_cont` key in the result JSON is
retained from the original example; here it denotes the prediction-only baseline,
not continuation from a pretrained LeWM checkpoint.

## Reproduction

Use Linux or WSL: the repository's Manager uses POSIX signals. From the repository
root, install the package and collection dependencies:

```bash
python -m pip install -e . pytest h5py 'dm-control==1.0.50' 'mujoco==3.17.0' 'numpy==2.2.6'
python -m benchmarks.rcaux.collect_reacher --output data/rcaux/reacher_states.h5 --episodes 64 --steps 200 --seed 2026
python -m benchmarks.rcaux.reacher_offline --dataset data/rcaux/reacher_states.h5 --output outputs/rcaux/results.json --train-episodes 48 --steps 800 --batch-size 128 --seeds 0 1 2 --queries 512 --candidates 32 --horizon 5 --learning-rate 0.003 --device cpu
```

The collector saves state observations and actions from the actual DeepMind
Control `reacher/easy` simulator under a uniform random policy. It does not
require rendering or downloading image trajectories. This is a newly collected
small dataset, not the paper's official training set. Existing data files are
not overwritten. Simulator and array-library versions are recorded in the HDF5
metadata and copied to the result JSON.

The encoder is a trainable 6-to-6 linear map initialized to the identity. The
dynamics are a residual MLP with two hidden layers of width 128. The auxiliary
head has width 128 and a 16-dimensional budget embedding. Prediction and anchor
weights are 1.0 and 0.1 in both conditions; the reachability weight is 0.35 for
RC-aux and zero for the baseline. The predicted-source pair weight is 0.5.
An identity-anchor regularizer discourages the encoder from collapsing.

The first 48 episodes are used for training and the last 16 are held out. Windows
never cross episode boundaries. Normalization uses only training episodes.
Each condition uses 800 AdamW updates with batch size 128 and learning rate 0.003.
Training uses one CPU thread and flushes subnormal floating-point values for
consistent CPU performance. Seeds 0, 1, and 2 are all reported.

Evaluation asks the model to rank 32 action sequences for each of 512 held-out
start/goal queries. One candidate is the demonstrated sequence; the others are
sampled held-out sequences. Any distractor matching the query window is
replaced by a uniformly sampled different window, so the correct window
appears exactly once. Other distractors are sampled with replacement.
All conditions use identical queries, candidates,
and target slots. Chance top-one accuracy is 3.125%. The report includes
untrained-model results, final held-out prediction error, retrieval metrics,
and training losses every 100 updates. Encoder spread and anchor error are
reported so a small latent prediction error is not considered sufficient
evidence of learning by itself.

Reachability-adjusted cost scores intermediate predicted states only: the
k-th predicted state uses the remaining budget K-k for 1 <= k < K. The
terminal state is excluded. For a one-step rollout the base cost is returned
without a reachability discount.

## Interpretation

This is an offline action-sequence retrieval task, not closed-loop control or a
reproduction of the paper's image-based benchmarks. Temporal-distance labels
and cross-trajectory negatives are training proxies for reachability: a longer
observed path does not prove a shorter path is impossible, and a state from
another trajectory can still be reachable. Under DDP these negatives are local
to each rank, with no latent all-gather. The example uses one process.

The identity anchor and low-dimensional state observations make this a small
implementation benchmark. It cannot establish improved general planning or
obstacle awareness. Report the prediction-only comparison even when RC-aux
does not improve retrieval.

## Tests

```bash
python -m pytest stable_pretraining/tests/unit/test_rcaux.py stable_pretraining/tests/integration/test_rcaux.py -q
```

These tests include a hand-enumerated BCE calculation for both encoded and
predicted-source pairs, actual Module logging, Manager checkpoint restoration
of parameters and AdamW steps, and a two-process CPU Gloo run launched with
`torchrun`. The distributed test checks local batch membership, pair counts,
finite losses, parameter updates, and exact agreement between ranks after
training. It is marked `integration` and `distributed`, so it runs in the
existing CPU integration CI job without requiring two GPUs. Additional unit
tests check remaining-budget indexing, terminal-state exclusion, one-step
rollouts, and unique correct candidates with deterministic sampling.


## Measured results

Mean +/- population standard deviation over all three seeds (0, 1, 2):

| Condition | Held-out latent RMSE (lower is better) | Retrieval top-1 (%) |
|---|---:|---:|
| Untrained world model | 0.7260 +/- 0.0278 | 4.8828 +/- 2.0854 |
| Prediction-only | 0.0669 +/- 0.0009 | 89.5182 +/- 2.0567 |
| RC-aux, ordinary cost | 0.0626 +/- 0.0011 | 91.2109 +/- 2.0360 |
| RC-aux, reachability-adjusted cost | 0.0626 +/- 0.0011 | 90.7552 +/- 1.7924 |

| Seed | Initial RMSE | Prediction-only RMSE | RC-aux RMSE | Baseline top-1 (%) | RC-aux ordinary top-1 (%) | RC-aux adjusted top-1 (%) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.7185 | 0.0681 | 0.0637 | 90.23 | 90.62 | 90.23 |
| 1 | 0.6963 | 0.0661 | 0.0612 | 91.60 | 93.95 | 93.16 |
| 2 | 0.7631 | 0.0664 | 0.0629 | 86.72 | 89.06 | 88.87 |

| Condition | Held-out encoder anchor RMSE | Mean latent standard deviation |
|---|---:|---:|
| Prediction-only | 0.0767 +/- 0.0032 | 0.8956 +/- 0.0008 |
| RC-aux | 0.2287 +/- 0.0209 | 0.8856 +/- 0.0091 |

Both complete models learn: the mean held-out latent RMSE decreases from 0.7260 before training to 0.0669 for prediction-only and 0.0626 for RC-aux. RC-aux with ordinary cost improves retrieval by 1.69 percentage points over the baseline; reachability-adjusted cost improves it by 1.24 points. The adjusted cost performs worse than ordinary cost on the RC-aux model in this experiment. This small three-seed comparison is implementation evidence, not a claim of statistically established planning superiority.

These results were reevaluated from the same six 800-step checkpoints after
correcting remaining-budget indexing and duplicate correct candidates. Before
reevaluation, the archived prediction and retrieval metrics were reproduced
with the old evaluation to verify the checkpoint mapping. Training, prediction
RMSE, encoder diagnostics, and learning curves are unchanged. The candidate
correction affects 5, 1, and 4 queries for seeds 0, 1, and 2 respectively.

Raw per-seed measurements, both training curves, and evaluation revision notes
are in [results.json](results.json). The dataset path is normalized to the
reproduction command above. Training and reevaluation used Linux/WSL, Python
3.12.16, PyTorch 2.15.0+cpu, Lightning 2.6.6, and one CPU thread. Dataset
collection used dm-control 1.0.50, MuJoCo 3.17.0, and NumPy 2.2.6. There are
9,408 training and 3,136 held-out windows.
