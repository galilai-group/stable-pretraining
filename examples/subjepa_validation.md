# Sub-JEPA validation

Implementation of [Zhao et al., 2026](https://arxiv.org/abs/2605.09241), checked
against the authors' [regularizer](https://github.com/intcomp/Sub-JEPA/blob/main/subjepa.py)
and [training objective](https://github.com/intcomp/Sub-JEPA/blob/main/train.py).

## Implementation scope

- Shared observation encoder, differentiable next-state targets, prediction MSE
  plus weighted multi-subspace Epps–Pulley regularization.
- Independently sampled, QR-initialized, frozen row-orthonormal projection
  matrices. Matrices and the random-direction counter are saved in state_dict.
- The released code estimates characteristic functions across the batch at each
  time step and averages statistics over time. This implementation follows that
  behavior, rather than pooling time and batch as described in paper equation 6.
- The default predictor is a compact causal transformer with action embeddings.
  It is not a reproduction of every LeWM architectural choice. A custom causal
  predictor can be supplied without changing the objective.
- No EMA, stop-gradient, pretrained encoder, or trainable projection variants.
  Default latent dimension is 192, with 32 subspaces; use 16 for PushT.
  The regularization weight should be tuned on held-out data.

## Reproduce the CPU diagnostic

```bash
python examples/subjepa_digits.py --epochs 20 --seed 42 --output /tmp/subjepa-digits-42
python examples/subjepa_digits.py --epochs 20 --seed 7 --output /tmp/subjepa-digits-7
python -m pytest stable_pretraining/tests/test_subjepa.py -q
```

The example uses the standard sklearn handwritten-digits dataset, normalized
from [0,16] to [0,1]. Each 8x8 image moves on a 16x16 canvas for four frames.
Each step samples an integer displacement in [-3,3] on each axis and clips the
position to the canvas; the action records the **executed** displacement divided
by three. Trajectories are generated once, offline, with a fixed seed.

Split: 1,437 training and 360 held-out source images, stratified by digit label.
Each source image appears in exactly one split. Digit labels and positions are
not supplied to the SSL objective. Positions are used only for a frozen linear
readout after training. Train and validation feature extraction use the same
encoder weights, with no gradient updates on validation data.

The CPU recipe uses a 256→128→24 MLP encoder, one causal transformer layer with
four heads, four subspaces of dimension six, 32 directions per subspace, 17
quadrature nodes, and regularization weight 0.1. AdamW starts at lr=0.001 with
weight decay 0.001; Module's default cosine schedule is retained. Batch size is
64 with drop_last for training. Evaluation covers all held-out trajectories.

Observed on macOS ARM CPU, Python 3.12.12, torch 2.14.1, Lightning 2.6.6,
scikit-learn 1.9.1:

| Held-out metric after 20 epochs | Seed 42 | Seed 7 |
|---|---:|---:|
| Next-latent prediction MSE | 0.046600 | 0.036554 |
| Persistence MSE (predict current latent) | 0.310015 | 0.275753 |
| Shuffled-action prediction MSE | 0.394811 | 0.397116 |
| Mean per-feature standard deviation | 0.579591 | 0.603014 |
| Effective covariance rank / 24 | 4.900630 | 4.235532 |
| Frozen linear position probe R² | 0.968188 | 0.969429 |
| Position probe R² before training | 0.785338 | 0.682408 |

Persistence and shuffled-action baselines use the **same trained encoder and
predictor**, so their latent errors are comparable. Absolute latent MSE before
and after training is not directly comparable because the encoder and its scale
change. Shuffling rolls complete action sequences by one held-out sample.
The readout is sklearn Ridge(alpha=1), fit on training embeddings only.
Effective rank uses normalized eigenvalues of the centered feature covariance.

These results demonstrate action-sensitive prediction and useful nonconstant
representations on a small known-image dataset. They do **not** establish the
paper's Two-Room, Reacher, PushT, or OGB-Cube planning success rates. No planner,
benchmark environment, or paper-scale training run is included in this diagnostic.

## Automated checks

The Sub-JEPA tests cover an independently computed scalar-loop reference
statistic and gradient, frozen orthogonal matrices, checkpoint continuation,
Gaussian-versus-collapsed embeddings, gradients through targets, malformed
configuration/batches, mixed precision, causal action conditioning, public
imports, composable Module usage, and actual Lightning optimizer/validation steps.
A two-process Gloo check compares distributed statistics and gradients with a
single-process reference, including unequal local batch sizes (3 and 5).
