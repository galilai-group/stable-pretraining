import time
import numpy as np
import gymnasium as gym
import torch
from probabilistic_ensembles import ReplayBuffer, ProbabilisticEnsemble
from cem_optim import CEMOptimizer

TIP, OBJ, GOAL = slice(14, 17), slice(17, 20), slice(20, 23)


def pusher_reward_fn(trajectory, actions):
    """trajectory: (N, P, H+1, 23), actions: (N, H, 7) -> (N,) mean-over-particles return."""
    tip = trajectory[:, :, 1:, TIP]
    obj = trajectory[:, :, 1:, OBJ]
    goal = trajectory[
        :, :, :1, GOAL
    ]  # exact goal from the start state (it never changes)
    dist = (obj - goal).norm(dim=-1)  # (N, P, H)
    near = (obj - tip).norm(dim=-1)
    ctrl = 0.1 * (actions**2).sum(-1)[:, None, :]  # (N, 1, H)
    return (-dist - 0.5 * near - ctrl).sum(-1).mean(1)


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
    delta = N - S
    std = delta.std(0)
    ratio = (torch.stack(preds).mean(0) - delta).pow(2).mean(0).sqrt() / std.clamp_min(
        1e-8
    )
    live = std > 1e-6

    def avg(sl):
        v = ratio[sl][live[sl]]
        return f"{v.mean():.2f}" if len(v) else "n/a (static)"

    print(
        f"  held-out one-step RMSE/std: mean {ratio[live].mean():.2f} | joint vel {avg(slice(7, 14))} "
        f"| tip {avg(TIP)} | object {avg(OBJ)}",
        flush=True,
    )


def run_pets_pusher():
    env = gym.make("Pusher-v5")
    state_dim, action_dim = (
        env.observation_space.shape[0],
        env.action_space.shape[0],
    )  # 23, 7
    action_low, action_high = env.action_space.low, env.action_space.high  # +-2
    T = 100

    num_models, horizon, num_particles = 5, 25, 20
    num_samples, num_elites, num_iters = 400, 40, 5
    num_random_episodes, num_episodes = 20, 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    buffer = ReplayBuffer(state_dim, action_dim, max_size=100000, device=device)
    ensemble = ProbabilisticEnsemble(
        num_models, state_dim, action_dim, hidden_dim=200, lr=1e-3, device=device
    )
    cem = CEMOptimizer(
        action_dim, action_high, action_low, horizon, num_samples, num_elites, num_iters
    )

    random_returns, random_moved = [], []
    for _ in range(num_random_episodes):
        state, _ = env.reset()
        obj0, ret, moved = state[OBJ].copy(), 0.0, 0.0
        for _ in range(T):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            buffer.add(state, action, next_state)
            state, ret = next_state, ret + reward
            moved = max(moved, float(np.linalg.norm(state[OBJ] - obj0)))
            if terminated or truncated:
                break
        random_returns.append(ret)
        random_moved.append(moved)
    print(
        f"Random policy: mean return over {T} steps = {np.mean(random_returns):.1f} "
        f"(+/- {np.std(random_returns):.1f}), buffer = {buffer.size}, "
        f"object ever moved in {sum(m > 1e-4 for m in random_moved)}/{num_random_episodes} episodes",
        flush=True,
    )

    returns = []
    for ep in range(num_episodes):
        t0 = time.time()
        print(f"\n--- Starting Episode {ep + 1}/{num_episodes} ---", flush=True)
        if ep > 0:
            lo = buffer.ptr - T
            one_step_check(
                ensemble,
                buffer.states[lo : buffer.ptr],
                buffer.actions[lo : buffer.ptr],
                buffer.next_states[lo : buffer.ptr],
                device,
            )
        ensemble.train_ensemble(buffer, batch_size=256, epochs=50)

        state, _ = env.reset()
        obj0, episode_reward, moved = state[OBJ].copy(), 0.0, 0.0
        for _ in range(T):
            st = torch.as_tensor(state, dtype=torch.float32, device=device)
            action_np = (
                cem.optimize(
                    st, ensemble, pusher_reward_fn, num_particles=num_particles
                )
                .detach()
                .cpu()
                .numpy()
            )
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            buffer.add(state, action_np, next_state)
            state, episode_reward = next_state, episode_reward + reward
            moved = max(moved, float(np.linalg.norm(state[OBJ] - obj0)))
            if terminated or truncated:
                break
        returns.append(episode_reward)
        print(
            f"Episode {ep + 1} Completed | Return: {episode_reward:.2f} | random baseline {np.mean(random_returns):.1f} "
            f"| object moved {moved:.3f} m | {time.time() - t0:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    run_pets_pusher()
