import time
import numpy as np
import gymnasium as gym
import torch
from probabilistic_ensembles import ReplayBuffer, ProbabilisticEnsemble
from cem_optim import CEMOptimizer


def reacher_reward_fn(trajectory, actions):
    """trajectory: (N, P, H+1, 10), actions: (N, H, 2) -> (N,) mean-over-particles return."""
    dist = trajectory[:, :, 1:, 8:10].norm(
        dim=-1
    )  # (N, P, H) fingertip-to-target distance
    ctrl = (actions**2).sum(-1)[:, None, :]  # (N, 1, H)
    return (-dist - ctrl).sum(-1).mean(1)


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
    print(
        f"  held-out one-step RMSE/std: mean {ratio[live].mean():.2f} | joint vel {ratio[6:8].mean():.2f} "
        f"| fingertip-target {ratio[8:10].mean():.2f}",
        flush=True,
    )


def run_pets_reacher():
    env = gym.make("Reacher-v5")
    state_dim, action_dim = (
        env.observation_space.shape[0],
        env.action_space.shape[0],
    )  # 10, 2
    action_low, action_high = env.action_space.low, env.action_space.high  # +-1
    T = 50

    num_models, horizon, num_particles = 5, 25, 20
    num_samples, num_elites, num_iters = 400, 40, 5
    num_random_episodes, num_episodes = 10, 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    buffer = ReplayBuffer(state_dim, action_dim, max_size=100000, device=device)
    ensemble = ProbabilisticEnsemble(
        num_models, state_dim, action_dim, hidden_dim=200, lr=1e-3, device=device
    )
    cem = CEMOptimizer(
        action_dim, action_high, action_low, horizon, num_samples, num_elites, num_iters
    )

    random_returns = []
    for _ in range(num_random_episodes):
        state, _ = env.reset()
        ret = 0.0
        for _ in range(T):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            buffer.add(state, action, next_state)
            state, ret = next_state, ret + reward
            if terminated or truncated:
                break
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
        episode_reward = 0.0
        for _ in range(T):
            st = torch.as_tensor(state, dtype=torch.float32, device=device)
            action_np = (
                cem.optimize(
                    st, ensemble, reacher_reward_fn, num_particles=num_particles
                )
                .detach()
                .cpu()
                .numpy()
            )
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            buffer.add(state, action_np, next_state)
            state, episode_reward = next_state, episode_reward + reward
            if terminated or truncated:
                break
        returns.append(episode_reward)
        print(
            f"Episode {ep + 1} Completed | Return: {episode_reward:.2f} | random baseline {np.mean(random_returns):.1f} "
            f"| final distance {np.linalg.norm(state[8:10]):.3f} m | {time.time() - t0:.0f}s",
            flush=True,
        )
        np.save("reacher_gym_returns.npy", np.array(returns))


if __name__ == "__main__":
    run_pets_reacher()
