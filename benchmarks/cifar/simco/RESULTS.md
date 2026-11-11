# SimCo on CIFAR-10 and CIFAR-100

These single-seed results validate a CIFAR adaptation of SimCo, using the
implementation in this repository. They are **online linear-probe accuracy on a
held-out validation split**, not frozen-encoder or official-test accuracy, and do
not claim reproduction of the paper's accuracy.

| Dataset | Epochs | Optimizer steps | Best online validation top-1 | Final online validation top-1 | Training wall time |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 500 | 87,500 | 88.58% | 88.18% | 9,682.42 s (2.69 h) |
| CIFAR-100 | 500 | 87,500 | 60.76% | 60.34% | 10,357.26 s (2.88 h) |

Each dataset used a fresh initialization, seed 0, and a fixed random partition of
the official training split: 45,000 images for SSL training and 5,000 for
validation. The official test split was unused. The best accuracy is selected
across validation epochs; the final accuracy is the last reported value. The
probe is trained with labels on detached augmented backbone features.

## Configuration

- Encoder: ResNet-18 with the low-resolution stem, initialized without pretrained
  weights; projector dimensions 512 → 128 → 128 with ReLU between linear layers.
- Loss: symmetric SimCo, intra-anchor temperature 0.1 and inter-anchor temperature
  1.0, with global cross-view negatives gathered across ranks.
- Hardware: one node with two NVIDIA RTX A5000 GPUs; DDP and SyncBatchNorm, FP32.
- Global batch size 256, local batch size 128; training drops the final incomplete
  batch. There are 175 optimizer steps per epoch and 87,500 per run.
- Backbone/projector optimizer: SGD, learning rate 0.03, momentum 0.9, weight decay
  0.0005; 10-epoch linear warmup (1,750 steps), followed by cosine decay to zero.
- Online probe: SGD, constant learning rate 0.1, momentum 0.9.
- Python 3.11.17, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, Lightning 2.6.6.

Full protocol and split fingerprints, exact reported metrics, and evidence
limitations are recorded in [results.json](results.json). Run the generic recipe
in [the example](../../../examples/simco_cifar.py) with `--devices 2` under
`torchrun`; the [method guide](../../../docs/source/guides/simco_cifar.md) explains
the data and evaluation protocol.

## Evidence and limits

The result table is derived from user-provided `config.json`, `summary.json`,
`environment.json`, and `requirements_frozen.txt` for the two completed runs.
The canonical protocol hashes were recomputed and match the configuration and
summary. Both summaries report two DDP ranks and 87,500 optimizer steps, with
finite final metrics. The subsequently supplied training logs
`simco-ddp-7787670_0.out` and `simco-ddp-7787670_1.out` each report three independent
DDP tests passing before training (60.87 s and 100.65 s, respectively); their
corresponding error logs report that `max_epochs=500` was reached.

The supplied local copies of both `split_indices.json` files are truncated at
262,144 bytes and cannot be parsed. Reconstructing the configured seed-0
partition reproduces both declared split hashes, and the copied bytes match the
corresponding JSON prefixes. This consistency check does not replace the
original complete split artifacts. A subsequent Oscar validation job,
`simco-validate-7816077`, reports that the original files for both datasets pass
the protocol/split hash and complete, disjoint 45,000/5,000 partition checks.
The validation helper also checks the final checkpoint at epoch 499 and step
87,500, both populated optimizer states, saved online-probe parameters, and the
presence of a metric CSV before printing success. Complete split files,
checkpoints, CSV contents, and per-example predictions have not been transferred
for local inspection; the original-artifact checks are evidenced by the supplied
Oscar stdout. Accuracy has not been independently recomputed from predictions.

The same validation job reports **5 SimCo integration tests passed**, with no
skips, in 95.77 s. These cover short training, restoration of full training state
on a synthetic run, actual two-process loss/gradient/update parity, and a
distributed synthetic end-to-end run. This resume test does not restore either
500-epoch training checkpoint. That job's original unit invocation inherited
GPU/scheduler state and reported 140 failures and 24 errors. Its record remains
in `results.json`; the following clean comparison isolated the remaining
pre-fix failures.

Validation job `simco-validate-7822625` ran units on CPU with inherited
Slurm/elastic/MPI/rank variables cleared. All **81 SimCo unit tests passed** in
27.29 s. The complete suite reported **3,820 passed, 11 failed, 20 skipped**
(170 deselected) in 530.97 s. A pristine upstream snapshot at
`ab836bf699a2be712dfcf980a5eb70d36391e876` in the same environment reported
**3,739 passed, the same 11 failed, 20 skipped** (165 deselected) in 533.56 s.
The comparison records identical failing node IDs and no additional failures;
the 81 additional passes match the added SimCo units. Ten shared failures report
`Cannot use ReduceOp.AVG with Gloo` under PyTorch 2.6.0; the remaining shared
failure compares an unresolved home path with its resolved symlink target.
The complete suite **did not pass in that historical run**. The five integration
tests were not rerun in this job; their passing evidence remains job 7816077.
The original split/checkpoint checks were also reported passing again in job
7822625.

The subsequent source fixes replace the statistics helper's distributed `AVG`
with differentiable `SUM / world_size`, and make the home-directory assertion
compare resolved paths using isolated direct, parent-component and symlink
layouts. Job 7822625 predates these fixes; its counts above remain the historical
record. A subsequent user-provided `unit-comparison.json` reports a **passing
post-fix Oscar CPU run**: both the SimCo unit command and the complete unit-suite
command exited with code 0, with no failing node IDs. No upstream comparison
was needed. The environment is recorded as CPU with inherited
Slurm/elastic/MPI/rank state cleared. The new job ID, test/skip counts, duration,
integration output and source hashes were not supplied. The earlier five
integration passes remain historical evidence from job 7816077; a post-fix
integration rerun and CI status are unverified. These utility/test fixes do not
change the SimCo objective or training recipe, so the two completed 500-epoch
experiments do not need to be rerun.

The source repository was recorded as a working tree with uncommitted changes.
Its base commit does not identify the exact executed SimCo source. These are one
run per dataset, with no uncertainty estimate or matched baseline comparison.
The validation job's source-hash files have not been supplied either.
