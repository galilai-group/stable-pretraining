"""LeHami on a pixel pendulum: latent volume and long-horizon velocity.

A small CPU-scale check of the two headline claims of LeHami (Yalovetzky,
Balestriero & LeCun, 2026) on the paper's two pendulum settings, rendered as
64x64 frames with the rod drawn to scale:

- ``--task conservative``: a frictionless pendulum with one of three rod
  lengths, sampled at 20 Hz (the paper's multi-rod runs use 10 Hz).
- ``--task actuated``: the swing-up setting. Three rod lengths times three bob
  masses (bob size shows the mass), driven by a random torque held for three
  frames, ``theta'' = -(g / L) sin(theta) + u / (m L^2)``, sampled at 10 Hz.
  Every model is conditioned on the torque.

Three models share the encoder, projector, SIGReg and multi-step rollout
training and differ only in the predictor:

- ``lehami``: :class:`~stable_pretraining.methods.LeHami`, positions from the
  encoder, momenta from Savitzky-Golay stencils, leapfrog on a separable
  Hamiltonian.
- ``phase-mlp``: the same phase-space state and losses, but an unconstrained
  residual MLP in place of the Hamiltonian step. Separates the symplectic
  structure from the momentum targets.
- ``lewm``: a LeWorldModel-style causal transformer over the last three
  latents (64-d, matching LeHami's phase-space size), the paper's baseline.

Evaluation rolls each model out for 60 steps from frame 3 of held-out
episodes, without seeing any further frame (but with the recorded torques on
the actuated task), and reports:

- R^2 of the angular velocity and the angle read from the predicted state by
  probes fit on the model's states of real training frames, with the paper's
  two probes: distance-weighted 25-nearest-neighbour regression (the headline
  read-out, since the latents encode velocity nonlinearly) and ridge
  regression on raw features (alpha = 1).
- Log-determinant per dimension of the Jacobian of the k-step rollout map in
  the model's own state space: 0 means volume is preserved, negative means
  the rollout contracts the latent distribution.

Example::

    python benchmarks/pendulum/lehami_pendulum.py --model lehami --out runs/lehami
    python benchmarks/pendulum/lehami_pendulum.py --model lewm --out runs/lewm
    python benchmarks/pendulum/lehami_pendulum.py --model phase-mlp --out runs/phase_mlp
    python benchmarks/pendulum/lehami_pendulum.py --report runs/lehami runs/phase_mlp runs/lewm
    python benchmarks/pendulum/lehami_pendulum.py --task actuated --model lehami --out runs/act_lehami
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from stable_pretraining.methods.lehami import (
    CAUSAL_WINDOW,
    CENTERED_HALFWIDTH,
    LeHami,
    causal_momentum,
    centered_momentum,
)
from stable_pretraining.methods.lejepa import SlicedEppsPulley

G = 9.81
SUBSTEP = 0.0125
IMG_SIZE = 64
PIVOT = (IMG_SIZE // 2, IMG_SIZE // 2)
LENGTHS = (0.7, 1.0, 1.4)
# Dimensionless initial-velocity bound; L = 1 gets U(-2, 2) rad/s as in the paper.
OMEGA_BOUND = 2.0 / math.sqrt(G)
CLIP_FRAMES = 14
ANCHOR = CAUSAL_WINDOW - 1
HORIZON = 60
LATENT_DIM = 32
TASKS = {
    "conservative": {
        "dt": 0.05,
        "frames": 100,
        "masses": (1.0,),
        "u_max": 0.0,
        "hold": 1,
        "max_bob_r": 4.0,
        # as the paper's conservative multi-rod model
        "sigreg_on": "joint",
    },
    "actuated": {
        "dt": 0.1,
        "frames": 80,
        "masses": (0.5, 1.0, 2.0),
        "u_max": 4.0,
        "hold": 3,
        "max_bob_r": 6.0,
        # as the paper's main actuated model
        "sigreg_on": "q",
    },
}
# Same torques, but only episodes that never pass over the top: tests whether
# full rotations, which a potential V(q, u) cannot represent, are what hurts.
TASKS["actuated-noflip"] = {**TASKS["actuated"], "no_flip": True}


def simulate(theta, omega, length, mass, actions, task) -> np.ndarray:
    """``(theta, omega)`` per frame, RK4 at ``SUBSTEP``.

    ``actions[t]`` is the torque over ``[t, t + 1)``; the last one drives nothing.
    """

    def deriv(th, om, u):
        return om, -(G / length) * math.sin(th) + u / (mass * length**2)

    frames = task["frames"]
    out = np.empty((frames, 2), dtype=np.float32)
    for t in range(frames):
        out[t] = theta, omega
        u = float(actions[t])
        for _ in range(round(task["dt"] / SUBSTEP)):
            k1 = deriv(theta, omega, u)
            k2 = deriv(theta + SUBSTEP / 2 * k1[0], omega + SUBSTEP / 2 * k1[1], u)
            k3 = deriv(theta + SUBSTEP / 2 * k2[0], omega + SUBSTEP / 2 * k2[1], u)
            k4 = deriv(theta + SUBSTEP * k3[0], omega + SUBSTEP * k3[1], u)
            theta += SUBSTEP / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            omega += SUBSTEP / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return out


def geometry(length: float, mass: float, task: dict) -> tuple[float, float]:
    """Rod length and bob radius in pixels: rod to scale, bob radius ~ m^(1/3)."""
    bob_r = min(task["max_bob_r"], 4.0 * mass ** (1 / 3))
    rod_px = (IMG_SIZE // 2 - task["max_bob_r"] - 1) * length / max(LENGTHS)
    return rod_px, bob_r


def render(theta: float, rod_px: float, bob_r: float) -> np.ndarray:
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = PIVOT[0] + rod_px * math.sin(theta)
    y = PIVOT[1] + rod_px * math.cos(theta)
    draw.line([PIVOT, (x, y)], fill=(50, 50, 50), width=2)
    draw.ellipse([x - bob_r, y - bob_r, x + bob_r, y + bob_r], fill=(200, 50, 50))
    draw.ellipse(
        [PIVOT[0] - 2, PIVOT[1] - 2, PIVOT[0] + 2, PIVOT[1] + 2], fill=(50, 50, 200)
    )
    return np.asarray(img)


def make_episodes(episodes: int, seed: int, task_name: str) -> dict:
    """Episodes split evenly over the (length, mass) cells of a task.

    Returns:
        ``video`` ``[N, T, 3, 64, 64]`` uint8, ``state`` ``[N, T, 2]``
        (angle, angular velocity), ``action`` ``[N, T, 1]`` torques, and
        ``length`` / ``mass`` per episode.
    """
    task = TASKS[task_name]
    cells = [(L, m) for L in LENGTHS for m in task["masses"]]
    per_cell = math.ceil(episodes / len(cells))
    rng = np.random.default_rng(seed)
    videos, states, actions, params = [], [], [], []
    for length, mass in cells:
        rod_px, bob_r = geometry(length, mass, task)
        for _ in range(per_cell):
            while True:
                theta = rng.uniform(-math.pi, math.pi)
                omega = rng.uniform(-OMEGA_BOUND, OMEGA_BOUND) * math.sqrt(G / length)
                u = np.zeros(task["frames"], dtype=np.float32)
                if task["u_max"]:
                    blocks = rng.uniform(
                        -task["u_max"],
                        task["u_max"],
                        math.ceil(task["frames"] / task["hold"]),
                    )
                    u = np.repeat(blocks, task["hold"])[: task["frames"]]
                    u = u.astype(np.float32)
                    u[-1] = 0.0
                state = simulate(theta, omega, length, mass, u, task)
                # resample until the pendulum never passes over the top
                if not task.get("no_flip") or np.abs(state[:, 0]).max() < math.pi:
                    break
            videos.append(
                np.stack([render(float(a), rod_px, bob_r) for a in state[:, 0]])
            )
            states.append(state)
            actions.append(u)
            params.append((length, mass))
    video = torch.from_numpy(np.stack(videos)).permute(0, 1, 4, 2, 3).contiguous()
    params = torch.tensor(params)
    return {
        "video": video,
        "state": torch.from_numpy(np.stack(states)),
        "action": torch.from_numpy(np.stack(actions))[..., None],
        "length": params[:, 0],
        "mass": params[:, 1],
    }


def to_float(video: torch.Tensor) -> torch.Tensor:
    return video.float() / 127.5 - 1


def conv_encoder(channels=(16, 32, 64), out_dim: int = 128) -> nn.Module:
    """The paper's four-block ConvNet, narrower for CPU."""
    layers, c_in = [], 3
    for c in channels:
        layers += [
            nn.Conv2d(c_in, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.MaxPool2d(2),
        ]
        c_in = c
    layers += [
        nn.Conv2d(c_in, out_dim, 3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ]
    return nn.Sequential(*layers)


class ResidualPhaseFlow(nn.Module):
    """Unconstrained step ``(q, p) -> (q, p) + MLP(q, p)``, same interface as the flow."""

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        # forward.lehami takes the momentum stencils' frame spacing from flow.dt.
        self.dt = 1.0
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q, p, u=None):
        x = torch.cat([q, p] if u is None else [q, p, u], dim=-1)
        delta = self.net(x)
        return q + delta[:, : self.latent_dim], p + delta[:, self.latent_dim :]


class LeWMStyle(nn.Module):
    """LeWorldModel-style baseline: causal transformer over three latents."""

    def __init__(
        self, encoder: nn.Module, encoder_dim: int, embed_dim: int = 64, action_dim=0
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.history = 3
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, embed_dim),
        )
        width = 256
        # The action taken at each frame is concatenated to that frame's latent.
        self.inp = nn.Linear(embed_dim + action_dim, width)
        self.time = nn.Parameter(torch.randn(1, self.history, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, 8, 4 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.predictor = nn.TransformerEncoder(layer, 4)
        self.out = nn.Linear(width, embed_dim)
        self.register_buffer(
            "causal",
            torch.triu(torch.full((self.history,) * 2, float("-inf")), 1),
            persistent=False,
        )
        self.sigreg = SlicedEppsPulley(num_slices=1024)

    def encode(self, video):
        b, t = video.shape[:2]
        z = self.projector(self.encoder(video.flatten(0, 1)))
        return z.view(b, t, self.embed_dim)

    def step(self, window, actions=None):
        """Next latent from a window ``[B, 3, D]`` and its actions ``[B, 3, A]``."""
        x = window if actions is None else torch.cat([window, actions], dim=-1)
        x = self.predictor(self.inp(x) + self.time, mask=self.causal)
        return self.out(x[:, -1])

    def forward(self, video, actions=None):
        z = self.encode(video)
        window = z[:, : self.history]
        losses = []
        for k in range(z.shape[1] - self.history):
            acts = None if actions is None else actions[:, k : k + self.history]
            nxt = self.step(window, acts)
            losses.append(F.mse_loss(nxt, z[:, self.history + k]))
            window = torch.cat([window[:, 1:], nxt[:, None]], 1)
        sigreg = torch.stack([self.sigreg(z[:, i]) for i in range(z.shape[1])]).mean()
        return torch.stack(losses).mean() + 0.25 * sigreg


def build(kind: str, task_name: str, sigreg_on: str = None) -> nn.Module:
    encoder = conv_encoder()
    action_dim = 1 if TASKS[task_name]["u_max"] else 0
    if kind == "lewm":
        return LeWMStyle(encoder, 128, action_dim=action_dim)
    model = LeHami(
        encoder=encoder,
        action_dim=action_dim,
        latent_dim=LATENT_DIM,
        encoder_dim=128,
        flow_hidden_dim=512,
        lamb=0.25,
        sigreg_on=sigreg_on or TASKS[task_name]["sigreg_on"],
    )
    if kind == "phase-mlp":
        model.flow = ResidualPhaseFlow(
            LATENT_DIM, 512, 2 * LATENT_DIM if action_dim else 0
        )
    return model


def loss_of(kind: str, model: nn.Module, clip, actions) -> torch.Tensor:
    if kind == "lewm":
        # history 3 + 8 rollout steps, as in the paper's baseline
        return model(clip[:, :11], None if actions is None else actions[:, :10])
    batch = {"video": clip} if actions is None else {"video": clip, "action": actions}
    return model(batch, "fit")["loss"]


def sample_clips(data, n, g):
    """Random clips ``[n, CLIP_FRAMES, 3, 64, 64]`` and their torques, or ``None``."""
    episodes, frames = data["video"].shape[:2]
    idx = torch.randint(episodes, (n,), generator=g)
    start = torch.randint(frames - CLIP_FRAMES + 1, (n,), generator=g)
    t = start[:, None] + torch.arange(CLIP_FRAMES)
    actions = data["action"][idx[:, None], t] if data["action"].any() else None
    return to_float(data["video"][idx[:, None], t]), actions


def train(kind, model, data, args, ckpt: Path) -> dict:
    g = torch.Generator().manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    warmup = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * s / args.steps))
        ),
    )
    state = {"step": 0, "seconds": 0.0, "history": []}
    if ckpt.exists():
        saved = torch.load(ckpt, weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        g.set_state(saved["gen"])
        state = saved["state"]
        print(f"resumed at step {state['step']}", flush=True)
    model.train()
    t0, running = time.time() - state["seconds"], 0.0
    for step in range(state["step"] + 1, args.steps + 1):
        loss = loss_of(kind, model, *sample_clips(data, args.batch, g))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running += float(loss.detach())
        if step % 50 == 0:
            row = {"step": step, "loss": running / 50, "seconds": time.time() - t0}
            state["history"].append(row)
            print(
                f"step {step}  loss {row['loss']:.4f}  {row['seconds']:.0f}s",
                flush=True,
            )
            running = 0.0
        if step % 250 == 0 or step == args.steps:
            state.update(step=step, seconds=time.time() - t0)
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "gen": g.get_state(),
                    "state": state,
                },
                ckpt.with_suffix(".tmp"),
            )
            os.replace(ckpt.with_suffix(".tmp"), ckpt)
    return state


# ---- evaluation ------------------------------------------------------------


@torch.no_grad()
def true_states(kind, model, video) -> tuple[torch.Tensor, slice]:
    """Each model's state on real frames ``[N, T', S]`` and the frames it covers."""
    z = torch.cat(
        [model.encode(to_float(video[i : i + 8])) for i in range(0, len(video), 8)]
    )
    t = z.shape[1]
    if kind == "lewm":
        state = torch.cat([z[:, :-2], z[:, 1:-1], z[:, 2:]], dim=-1)
        return state, slice(2, t)
    p = centered_momentum(z)
    state = torch.cat([z[:, CENTERED_HALFWIDTH : t - CENTERED_HALFWIDTH], p], dim=-1)
    return state, slice(CENTERED_HALFWIDTH, t - CENTERED_HALFWIDTH)


def step_fn(kind, model):
    """``step(state, actions, k)``: step ``k`` of a rollout on the flattened state.

    ``actions`` ``[N, T, 1]`` are the episode's torques (or ``None``); the
    step from frame ``ANCHOR + k`` uses the torque at that frame.
    """
    if kind == "lewm":
        d = model.embed_dim

        def step(s, actions, k):
            window = s.view(len(s), 3, d)
            acts = None
            if actions is not None:
                acts = actions[:, ANCHOR - 2 + k : ANCHOR + 1 + k].to(s.dtype)
            nxt = model.step(window, acts)
            return torch.cat([window[:, 1:].flatten(1), nxt], dim=-1)

        return step

    def step(s, actions, k):
        q, p = s.chunk(2, dim=-1)
        u = None
        if actions is not None:
            u = model.action_encoder(actions[:, ANCHOR + k].to(s.dtype))
        q, p = model.flow(q, p, u)
        return torch.cat([q, p], dim=-1)

    return step


@torch.no_grad()
def initial_state(kind, model, video):
    z = model.encode(to_float(video[:, : ANCHOR + 1]))
    if kind == "lewm":
        return z[:, ANCHOR - 2 : ANCHOR + 1].flatten(1)
    return torch.cat([z[:, ANCHOR], causal_momentum(z)], dim=-1)


def rollout(kind, model, video, actions=None, steps=HORIZON) -> torch.Tensor:
    """Predicted states ``[N, steps, S]`` for frames ``ANCHOR + 1 ..``."""
    s, step = initial_state(kind, model, video), step_fn(kind, model)
    out = []
    with torch.no_grad():
        for k in range(steps):
            s = step(s, actions, k)
            out.append(s)
    return torch.stack(out, 1)


def targets(state: torch.Tensor) -> torch.Tensor:
    theta, omega = state[..., 0], state[..., 1]
    return torch.stack([theta.sin(), theta.cos(), omega], dim=-1)


def fit_ridge(x, y, alpha=1.0):
    """Ridge regression on raw features with an unpenalized intercept."""
    x_mean, y_mean = x.double().mean(0), y.double().mean(0)
    xc = x.double() - x_mean
    w = torch.linalg.solve(
        xc.T @ xc + alpha * torch.eye(x.shape[1], dtype=torch.float64),
        xc.T @ (y.double() - y_mean),
    )
    return lambda z: ((z.double() - x_mean) @ w + y_mean).float()


def fit_knn(x, y, k=25):
    """Distance-weighted k-nearest-neighbour regression on raw features."""

    def predict(z):
        out = []
        for i in range(0, len(z), 1024):
            dist, idx = torch.cdist(z[i : i + 1024], x).topk(k, largest=False)
            w = 1 / dist.clamp_min(1e-12)
            out.append((w[..., None] * y[idx]).sum(1) / w.sum(1, keepdim=True))
        return torch.cat(out)

    return predict


def r2(pred, true):
    return 1 - ((pred - true) ** 2).sum(0) / ((true - true.mean(0)) ** 2).sum(0)


def log_volume(kind, model, video, actions=None, ks=(1, 10, 30, 60)) -> dict:
    """Mean log|det J| per dimension of the k-step rollout map, in float64.

    The map is taken with respect to the starting state, under each episode's
    own torques. By the chain rule its log-determinant is the sum of the
    one-step log-determinants along the rollout, so one pass gives every k.
    """
    s0 = initial_state(kind, model, video).double()
    model = model.double()
    step = step_fn(kind, model)
    totals = np.zeros((len(s0), max(ks)))
    for i, s in enumerate(s0):
        acts = None if actions is None else actions[i : i + 1].double()
        for j in range(max(ks)):

            def one_step(x, j=j, acts=acts):
                return step(x[None], acts, j)[0]

            jac = torch.autograd.functional.jacobian(one_step, s)
            totals[i, j] = torch.linalg.slogdet(jac)[1].item() / len(s)
            with torch.no_grad():
                s = one_step(s)
    model.float()
    cumulative = totals.cumsum(1).mean(0)
    return {k: float(cumulative[k - 1]) for k in ks}


def evaluate(kind, model, train_data, test_data) -> dict:
    model.eval()
    x, frames = true_states(kind, model, train_data["video"])
    x, y = x.flatten(0, 1), targets(train_data["state"][:, frames]).flatten(0, 1)
    probes = {"knn": fit_knn(x, y), "linear": fit_ridge(x, y)}
    x_test, frames_test = true_states(kind, model, test_data["video"])
    x_test = x_test.flatten(0, 1)
    y_test = targets(test_data["state"][:, frames_test]).flatten(0, 1)

    actions = test_data["action"] if test_data["action"].any() else None
    pred = rollout(kind, model, test_data["video"], actions)
    future = targets(test_data["state"][:, ANCHOR + 1 : ANCHOR + 1 + HORIZON])
    results = {"true_frames": {}, "rollout": {}}
    for name, probe in probes.items():
        r = r2(probe(x_test), y_test)
        results["true_frames"][name] = {
            "angle_r2": float(r[:2].mean()),
            "omega_r2": float(r[2]),
        }
        rows = []
        for k in range(HORIZON):
            r = r2(probe(pred[:, k]), future[:, k])
            rows.append(
                {"k": k + 1, "angle_r2": float(r[:2].mean()), "omega_r2": float(r[2])}
            )
        results["rollout"][name] = rows
    results["log_volume_per_dim"] = log_volume(
        kind, model, test_data["video"][:6], None if actions is None else actions[:6]
    )
    return results


def report(runs: list[Path]):
    ks = (1, 5, 10, 20, 40, 60)
    for run in runs:
        res = json.loads((run / "results.json").read_text())
        ev = res["eval"]
        print(
            f"\n{res['model']}, {res.get('task', 'conservative')} task "
            f"({res['train']['seconds'] / 60:.0f} min training)"
        )
        for name in ("knn", "linear"):
            roll = {r["k"]: r for r in ev["rollout"][name]}
            true = ev["true_frames"][name]
            print(
                f"  {name:6s} probe, real frames: angle R2 {true['angle_r2']:.2f}, "
                f"omega R2 {true['omega_r2']:.2f}"
            )
            for key, label in (("omega_r2", "omega"), ("angle_r2", "angle")):
                cells = " ".join(f"k{k}={roll[k][key]:.2f}" for k in ks)
                print(f"    {label} R2 by step: {cells}")
        vol = ev["log_volume_per_dim"]
        cells = " ".join(f"k{k}={float(v):+.3f}" for k, v in vol.items())
        print(f"  log|det J| / dim: {cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=("lehami", "phase-mlp", "lewm"))
    parser.add_argument("--task", choices=tuple(TASKS), default="conservative")
    parser.add_argument(
        "--sigreg-on",
        choices=("q", "joint"),
        default=None,
        help="Override the task's SIGReg placement for lehami / phase-mlp.",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path, nargs="+")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--test-episodes", type=int, default=60)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Re-score a finished run from <out>/ckpt.pt; keeps its training log.",
    )
    args = parser.parse_args()
    if args.report:
        return report(args.report)
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    train_data = make_episodes(args.train_episodes, 1000, args.task)
    test_data = make_episodes(args.test_episodes, 2000, args.task)
    torch.manual_seed(args.seed)
    model = build(args.model, args.task, args.sigreg_on)
    print(
        f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
        f"{len(train_data['video'])} train episodes",
        flush=True,
    )
    if args.eval_only:
        saved = torch.load(args.out / "ckpt.pt", weights_only=False)
        model.load_state_dict(saved["model"])
        state = saved["state"]
    else:
        state = train(args.model, model, train_data, args, args.out / "ckpt.pt")
    results = evaluate(args.model, model, train_data, test_data)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "task": args.task,
                "args": {k: str(v) for k, v in vars(args).items()},
                "train": state,
                "eval": results,
            }
        )
    )
    report([args.out])


if __name__ == "__main__":
    main()
