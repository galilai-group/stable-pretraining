# FlatNCE implementation and validation

Validated on October 21, 2026 with Python 3.13.3, PyTorch 2.14.1,
Lightning 2.6.6, and macOS CPU/Gloo. CUDA hardware was unavailable.

## Implemented behavior

- `FlatNCELoss` implements the symmetric two-view objective from Algorithm 1
  and Equation 6, excluding self and positive pairs from the negative partition.
- Cross-process candidates use differentiable gathering and rank-aware pairing.
- Low-precision inputs use float32 loss arithmetic, including inside autocast.
- `forward.flatnce` shares SimCLR's two-view encoding and evaluation contract.
- `spt.FlatNCE` reuses SimCLR's model construction and direct tensor interface,
  with a dict interface compatible with `Module`, `Manager`, and probe callbacks.
- Public runtime exports, IDE type exports, the method catalog, and API docs
  include the method and loss.
- The CIFAR-10 runner selects FlatNCE or SimCLR while keeping the remaining
  recipe fixed. It supports local metrics, checkpoints, full-state resume,
  a fixed validation split, and download-free synthetic checks.
- The runner writes resumable state after every completed epoch. A two-epoch
  CPU test checks that `last.ckpt` exists before epoch 2 starts.

The existing NT-Xent loss and the core training loop were left intact.

## Automated checks

All **151 initially selected tests passed** across these commands (no GPU required):

```bash
python -m pytest \
  stable_pretraining/tests/unit/test_flatnce.py \
  stable_pretraining/tests/unit/test_simclr.py \
  stable_pretraining/tests/unit/test_collectives.py \
  stable_pretraining/tests/unit/test_forward_functions.py \
  stable_pretraining/tests/unit/test_method_boundary_contracts.py \
  stable_pretraining/tests/unit/test_module_and_forward_edges.py \
  stable_pretraining/tests/unit/test_static_registry.py \
  stable_pretraining/tests/unit/test_optional_api_contracts.py \
  stable_pretraining/tests/integration/test_flatnce.py -q
# 137 passed

python -m pytest stable_pretraining/tests/unit/test_methods_smoke.py \
  -k 'FlatNCE or SimCLR' -q
# 4 passed

python -m pytest stable_pretraining/tests/regression/test_methods.py \
  -k simclr -q
# 2 passed, including the pinned SimCLR numerical reference

python -m pytest stable_pretraining/tests/distributed/test_collectives_dist.py -q
# 8 passed, with two CPU/Gloo processes per test
```

The FlatNCE tests compare gradients with an independent negative-only
log-partition implementation, including saturated NT-Xent, view swapping,
sample permutations, mixed precision, invalid inputs, full optimizer updates,
probe evaluation, and checkpoint reloads. The distributed test compares DDP
parameter gradients with a global reference, including one image per rank.
The spawn harness now also fails on timed-out or abnormally exited workers.

An interrupted-run test also verifies that resumed FlatNCE training matches
uninterrupted training through nine updates, including model weights, optimizer
momentum, and scheduler state.

Six results/checkpoint reporting tests also pass. Ruff lint, Ruff format checks
on the changed Python/type-stub files, and `git diff --check` passed. A broader
repository test selection is described under limitations below.

## Runner checks

The actual ResNet-18 runner completed these CPU checks:

| Check | Result |
|---|---|
| FlatNCE, synthetic data, two updates | Passed; finite nonzero gradient diagnostics, probe metrics, checkpoint, and summary saved. |
| SimCLR, same synthetic recipe, two updates | Passed; metrics, checkpoint, and summary saved. |
| FlatNCE, load saved checkpoint | Passed; checkpoint restoration completed. The automated interrupted-run test verifies subsequent updates and full-state equivalence. |
| FlatNCE, one-update run | Passed; handles a run too short for warmup plus cosine decay. |
| FlatNCE, real CIFAR-10 subset, two updates | Passed; 16 training images and 16 held-out images, finite gradients, probes, checkpoint, and summary. |
| SimCLR, identical real CIFAR-10 subset, two updates | Passed; same split and settings with the objective changed. |
| FlatNCE, real CIFAR-10, 10 epochs | Passed; 3,510 updates, final linear 49.63%, final k-NN 52.72%. |
| SimCLR, matched real CIFAR-10, 10 epochs | Passed; 3,510 updates, final linear 47.15%, final k-NN 53.51%. |

The real-data checks used `--batch-size 8 --train-samples 16
--validation-size 16 --epochs 1 --max-steps 2 --num-workers 0 --accelerator cpu`
with seed 0 and temperature 0.5. The official CIFAR-10 archive's MD5 was verified
as `c58f30108f718f92721af3b95e74349a` before extraction. The downloaded dataset
is excluded from Git by the repository's `/data/` ignore rule.

| Objective | Completed updates | Mean training loss | Last model gradient norm |
|---|---:|---:|---:|
| FlatNCE | 2 | 1.0 | 21.61635 |
| SimCLR | 2 | 2.61454 | 19.97303 |

The gradient diagnostic is the combined backbone/projector parameter-gradient
norm in float32. These loss and norm values are implementation diagnostics;
their relative magnitudes do not establish which objective learns better.

The complete matched-run results and limitations are documented in
[RESULTS.md](RESULTS.md). They establish short-run learning behavior but do not
reproduce the paper. Multi-seed training, batch-size/temperature sweeps, frozen
linear evaluation, and CUDA execution remain future experiments.

The broad CPU-only unit selection reached 3,117 passes and 16 skips before ten
failures in an existing macOS fork-based visualization test and optional
Trackio/gradio-client tests. Representative failures reproduced on the unchanged
base source. Installing optional extras while preserving existing versions
selected Trackio 0.0.6, which remains incompatible with these tests on Python
3.13. The full repository suite is therefore not claimed to pass locally.
