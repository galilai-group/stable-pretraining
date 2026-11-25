# LeHami on a pixel pendulum

CPU-scale check of LeHami's two headline claims (Yalovetzky, Balestriero &
LeCun, 2026): its leapfrog step preserves latent phase-space volume, and its
rollouts keep track of velocity for longer than an unconstrained predictor.
Script: `lehami_pendulum.py`.

All numbers: one seed, 3,000 training steps on CPU (about 25 minutes per
model), 300 training and 60 held-out episodes. Global batch size 32: each run
is a single process with no DDP, so `--batch 32` (the default) is the whole
batch, and SIGReg sees all 32 clips at each frame. Treat differences
under about 0.1 R² as noise.

## Setup

Three models share everything except the predictor: a four-block ConvNet
encoder (the paper's design, narrowed to 16/32/64/128 channels for CPU), a
BatchNorm projector, SIGReg (weight 0.25) and multi-step rollout training on
their own predictions.

| Model | State | Predictor |
|---|---|---|
| `lehami` | 32-d position `q` + 32-d momentum `p` (Savitzky-Golay stencils) | Leapfrog on a separable Hamiltonian `K(p/s) + V(q, u)` |
| `phase-mlp` | same `q`, `p`, same losses | Unconstrained residual MLP `(q, p) + f(q, p, u)` |
| `lewm` | last three 64-d latents | LeWorldModel-style causal transformer |

Each model is rolled out for 60 steps from frame 3 of every held-out episode
without seeing another frame. Probes fit on the model's states of real
training frames read angle and angular velocity off the predicted states. We
use the paper's two probes: distance-weighted 25-nearest-neighbour regression
and ridge regression (alpha 1) on raw features. The latents encode velocity
nonlinearly (on real frames the linear probe reaches R² 0.73, nearest-neighbour
1.00, the same gap the paper reports), so nearest-neighbour is the headline
read-out.

## Conservative pendulum (three rod lengths, 20 Hz)

The rod lengths follow the paper's multi-rod set; the sampling rate is that of
its single-rod set (the paper's multi-rod runs use 10 Hz), so one of our steps
is half a paper step.

Angular-velocity R² of decoded rollouts, nearest-neighbour probe:

| Steps ahead | 1 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|
| `lehami` | 0.92 | 0.97 | **0.88** | **0.79** | **0.41** |
| `phase-mlp` | 0.93 | 0.95 | 0.76 | -0.19 | -0.10 |
| `lewm` | 0.81 | 0.97 | 0.87 | -0.59 | -0.39 |

Against the paper's LeHami at the same elapsed time (its nearest-neighbour
probe on its own 10 Hz multi-rod runs, three seeds, GPU, 100 epochs):

| Time ahead | 0.5 s | 1 s | 2 s | 3 s | 4 s |
|---|---|---|---|---|---|
| Paper, LeHami | 0.97 | 0.94 | 0.74 | - | 0.09 |
| This run, `lehami` | 0.97 | 0.88 | 0.79 | 0.41 | - |

This CPU run reproduces the paper's numbers; it does not exceed them. The
paper's LeWM baseline is not comparable here: its probe reads a single
frame's latent, which cannot hold velocity, while our `lewm` probe reads its
three-frame window.

Angle R² (sin and cos), nearest-neighbour probe:

| Steps ahead | 1 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|
| `lehami` | 1.00 | 0.82 | **0.67** | **0.67** | **0.50** |
| `phase-mlp` | 1.00 | 0.88 | 0.55 | -0.10 | -0.41 |
| `lewm` | 1.00 | 0.90 | 0.65 | -0.15 | -0.52 |

Latent volume, log-determinant per dimension of the k-step rollout's Jacobian
in each model's own state (0 = preserved, negative = contracted):

| Steps | 1 | 10 | 30 | 60 |
|---|---|---|---|---|
| `lehami` | 0.000 | 0.000 | 0.000 | 0.000 |
| `phase-mlp` | -0.03 | -0.36 | -1.12 | -2.32 |
| `lewm` | -0.82 | -7.76 | -22.79 | -45.16 |

- All three track the pendulum about equally well for 10-20 steps (0.5-1 s).
- Beyond that only LeHami keeps reading out velocity and angle; the two
  unconstrained predictors fall to or below the mean-prediction baseline.
- LeHami's volume is exactly preserved by construction; both unconstrained
  predictors contract, the transformer by orders of magnitude.
- With the paper's linear probe LeHami's rollouts read worse (velocity R² 0.19,
  0.28 and -2.4 at 1, 10 and 20 steps): rollouts start from the causal momentum
  estimate, which differs from the centered one the probes are fit on by about
  37%, and a linear read-out is sensitive to that offset while
  nearest-neighbour lookup is not.

## Actuated pendulum (three rod lengths, three masses, 10 Hz)

The paper's swing-up setting: a random torque in [-4, 4], held for 3 frames,
drives the pendulum, and every model is told the torque at each step.
LeHami takes it in through the potential `V(q, u)`. SIGReg is on `q` for
`lehami` and `phase-mlp`, as in the paper's main actuated model (on the joint
state both read velocity no better). In 44% of held-out episodes the torque
pushes the pendulum over the top at least once.

To test whether full turns are the problem, `actuated-noflip` keeps the same
torques but resamples every episode until the pendulum never passes over the
top. Its pendulum also moves more slowly (mean |ω| 2.0 rad/s against 3.9), so
compare models within a task rather than across tasks.

Angular-velocity R² of decoded rollouts, nearest-neighbour probe (last
column: the probe on the model's states of real frames, no rollout):

| Task | Model | 1 | 10 | 20 | 40 | 60 | Real frames |
|---|---|---|---|---|---|---|---|
| actuated | `lehami` | 0.02 | -0.32 | 0.00 | -0.07 | -0.02 | 0.10 |
| actuated | `phase-mlp` | 0.87 | 0.33 | 0.05 | -0.05 | 0.05 | 0.45 |
| actuated | `lewm` | **0.96** | 0.28 | 0.16 | 0.01 | 0.12 | **0.98** |
| no full turns | `lehami` | **0.98** | **0.97** | **0.93** | **0.70** | **0.27** | **0.98** |
| no full turns | `phase-mlp` | 0.97 | 0.96 | 0.85 | 0.36 | -0.15 | **0.98** |
| no full turns | `lewm` | 0.95 | 0.95 | 0.79 | 0.27 | 0.16 | 0.96 |

Angle R² on the task without full turns: `lehami` 0.83 and 0.68 at 20 and 40
steps, `phase-mlp` 0.62 and 0.26, `lewm` 0.57 and 0.29. Latent volume after
60 steps: `lehami` 0.000 on both tasks, `phase-mlp` -15.8 and -9.6, `lewm`
-34.8 and -39.7.

Mean velocity error (rad/s) of the nearest-neighbour probe on real frames of
episodes that never pass over the top, by true speed:

| Trained on | Model | 0-2 rad/s | 2-4 | 4-8 |
|---|---|---|---|---|
| actuated | `lehami` | 0.65 | 1.59 | 3.18 |
| actuated | `phase-mlp` | 0.19 | 0.23 | 0.35 |
| actuated | `lewm` | 0.23 | 0.25 | 0.32 |
| no full turns | `lehami` | 0.17 | 0.22 | 0.48 |
| no full turns | `phase-mlp` | 0.16 | 0.23 | 0.61 |
| no full turns | `lewm` | 0.30 | 0.36 | 0.65 |

- With full turns, LeHami fails before any rollout: its encoder does not
  hold velocity even on real frames, and its error is 3 to 9 times the
  others' even on slow frames of episodes that never flip.
- Without full turns, at the same torque strength, its encoder recovers to
  the others' level and its rollouts again track velocity and angle longest,
  as on the conservative task.
- A plausible cause: a torque that keeps turning the pendulum the same way
  around a full circle does work that no single-valued potential `V(q, u)`
  can describe. Full turns also bring the fastest swings, so this run does
  not fully separate the two.
- Fast swings are a separate limit. Above about 8 rad/s the 5-frame momentum
  window (0.4 s at 10 Hz) spans more than half a turn, and both models that
  use it read velocity badly on the actuated task (errors of 13.3 and 9.7
  rad/s), while `lewm`, which learns its read-out from three raw frames,
  stays at 1.3.
- The paper's own swing-up planning results show the same ordering. In its
  released figure data, LeHami reaches goal success rates of 0.24-0.31
  across goal settings: above a LeWM that encodes one frame (0.10-0.15), but
  below LeWM variants that encode three frames (0.39-0.64), like our `lewm`.

## Reproducing

```bash
P=benchmarks/pendulum/lehami_pendulum.py
for m in lehami phase-mlp lewm; do python $P --model $m --out runs/$m; done
python $P --report runs/lehami runs/phase-mlp runs/lewm
# actuated swing-up setting, with and without full turns
for t in actuated actuated-noflip; do
  for m in lehami phase-mlp lewm; do python $P --task $t --model $m --out runs/$t/$m; done
done
```
