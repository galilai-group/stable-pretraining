import time
import numpy as np
import torch
from probabilistic_ensembles import ReplayBuffer, ProbabilisticEnsemble
from cem_optim import CEMOptimizer

L = 1.0  # pole length (tip distance from the pivot)


class CartpoleSwingup:
    """obs = [x, x_dot, sin(theta), cos(theta), theta_dot]; action in [-1, 1] (scaled to +-10 N)."""

    obs_dim, act_dim = 5, 1
    g, mc, mp, pole_l, fmax, dt, sub = 9.8, 1.0, 0.1, 0.5, 10.0, 0.05, 5

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s = np.array([0.0, 0.0, np.pi, 0.0]) + self.rng.normal(0, 0.05, 4)
        return self._obs()

    def _obs(self):
        x, xd, th, thd = self.s
        return np.array([x, xd, np.sin(th), np.cos(th), thd], dtype=np.float32)

    def step(self, a):
        F = float(np.clip(a[0], -1, 1)) * self.fmax
        x, xd, th, thd = self.s
        h = self.dt / self.sub
        for _ in range(self.sub):
            tm, pml = self.mc + self.mp, self.mp * self.pole_l
            tmp = (F + pml * thd**2 * np.sin(th)) / tm
            thacc = (self.g * np.sin(th) - np.cos(th) * tmp) / (
                self.pole_l * (4 / 3 - self.mp * np.cos(th) ** 2 / tm)
            )
            xacc = tmp - pml * thacc * np.cos(th) / tm
            x, xd, th, thd = x + h * xd, xd + h * xacc, th + h * thd, thd + h * thacc
        self.s = np.array([x, xd, th, thd])
        o = self._obs()
        tip_x, tip_y = o[0] + L * o[2], L * o[3]
        r = float(
            np.exp(-(tip_x**2 + (tip_y - L) ** 2) / L**2) - 0.01 * float(a[0]) ** 2
        )
        return o, r


def cartpole_reward_fn(trajectory, actions):
    """trajectory: (N, P, H+1, 5), actions: (N, H, 1) -> (N,) mean-over-particles return."""
    x, s, c = (trajectory[:, :, 1:, i] for i in (0, 2, 3))
    d2 = (x + L * s) ** 2 + (L * c - L) ** 2  # (N, P, H)
    ctrl = 0.01 * (actions**2).sum(-1)[:, None, :]  # (N, 1, H)
    return (torch.exp(-d2 / L**2) - ctrl).sum(-1).mean(1)


@torch.no_grad()
def one_step_check(ensemble, S, A, N, device):
    """One-step error on transitions the current model has NOT been trained on."""
    S, A, N = (torch.as_tensor(x, device=device) for x in (S, A, N))
    x = torch.cat([S, A], -1)
    preds = []
    for m in ensemble.models:
        m.eval()
        out = m.predict(x) if hasattr(m, "predict") else m(x)
        preds.append(out[0])
    pred = torch.stack(preds).mean(0)
    delta = N - S
    ratio = (pred - delta).pow(2).mean(0).sqrt() / delta.std(0).clamp_min(1e-8)
    print(
        f"  held-out one-step RMSE/std per dim [x, x_dot, sin, cos, th_dot]: {ratio.cpu().numpy().round(2)}"
    )


def run_pets_cartpole():
    env = CartpoleSwingup()
    state_dim, action_dim = env.obs_dim, env.act_dim
    action_low, action_high = (
        np.array([-1.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    )

    # Hyperparameters
    num_models, horizon, num_particles = 5, 25, 20
    num_samples, num_elites, num_iters = 400, 40, 5
    T = 200
    num_random_episodes = 5
    num_episodes = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    buffer = ReplayBuffer(state_dim, action_dim, max_size=100000, device=device)
    ensemble = ProbabilisticEnsemble(
        num_models, state_dim, action_dim, hidden_dim=200, lr=1e-3, device=device
    )
    cem = CEMOptimizer(
        action_dim, action_high, action_low, horizon, num_samples, num_elites, num_iters
    )

    # Random exploration; its mean return is the baseline to beat
    print("Collecting initial random exploration data...", flush=True)
    rng = np.random.default_rng(0)
    random_returns = []
    for i in range(num_random_episodes):
        state, ret = env.reset(seed=i), 0.0
        for _ in range(T):
            action = rng.uniform(-1, 1, action_dim).astype(np.float32)
            next_state, reward = env.step(action)
            buffer.add(state, action, next_state)
            state, ret = next_state, ret + reward
        random_returns.append(ret)
    print(
        f"Random policy: mean return over {T} steps = {np.mean(random_returns):.1f} "
        f"(+/- {np.std(random_returns):.1f}), buffer = {buffer.size}",
        flush=True,
    )

    returns = []
    for ep in range(num_episodes):
        t0 = time.time()
        print(f"\n--- Starting Episode {ep + 1}/{num_episodes} ---", flush=True)

        if ep > 0:  # last episode's transitions are still unseen by the current model
            lo = buffer.ptr - T
            one_step_check(
                ensemble,
                buffer.states[lo : buffer.ptr],
                buffer.actions[lo : buffer.ptr],
                buffer.next_states[lo : buffer.ptr],
                device,
            )

        ensemble.train_ensemble(buffer, batch_size=256, epochs=50)

        state, episode_reward = env.reset(seed=1000 + ep), 0.0
        for _ in range(T):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            action_tensor = cem.optimize(
                state_tensor, ensemble, cartpole_reward_fn, num_particles=num_particles
            )
            action_np = action_tensor.detach().cpu().numpy()
            next_state, reward = env.step(action_np)
            buffer.add(state, action_np, next_state)
            state, episode_reward = next_state, episode_reward + reward

        returns.append(episode_reward)
        print(
            f"Episode {ep + 1} Completed | Steps: {T} | Total Reward: {episode_reward:.2f} "
            f"| random baseline {np.mean(random_returns):.1f} | {time.time() - t0:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    run_pets_cartpole()
