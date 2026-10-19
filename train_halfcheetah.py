import gymnasium as gym
import torch
import numpy as np
from probabilistic_ensembles import ReplayBuffer, ProbabilisticEnsemble
from cem_optim import CEMOptimizer


def half_cheetah_reward_fn(trajectory, actions):
    """trajectory: (N, P, H+1, 17), actions: (N, H, 6) -> (N,) mean-over-particles return."""
    vel = trajectory[:, :, 1:, 8]  # forward velocity, (N, P, H)
    ctrl = 0.1 * (actions**2).sum(-1)[:, None, :]  # (N, 1, H)
    return (vel - ctrl).sum(-1).mean(1)


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
    rmse = (pred - delta).pow(2).mean(0).sqrt()
    ratio = rmse / delta.std(0).clamp_min(1e-8)
    print(
        f"  held-out one-step RMSE/std: mean over dims {ratio.mean():.2f} | "
        f"vel (obs[8]) {ratio[8]:.2f} | worst dim {ratio.max():.2f}"
    )


def run_pets_halfcheetah():
    # 1. Environment
    env = gym.make("HalfCheetah-v4")
    state_dim = env.observation_space.shape[0]  # 17
    action_dim = env.action_space.shape[0]  # 6
    action_low, action_high = env.action_space.low, env.action_space.high

    # 2. Hyperparameters
    num_models = 5
    horizon = 30
    num_particles = 20
    num_samples = 500
    num_elites = 50
    num_iters = 5
    T = 200  # steps per episode (env default is 1000)
    num_random_episodes = 10
    num_episodes = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Core components
    buffer = ReplayBuffer(state_dim, action_dim, max_size=100000, device=device)
    ensemble = ProbabilisticEnsemble(
        num_models, state_dim, action_dim, hidden_dim=200, lr=1e-3, device=device
    )
    cem = CEMOptimizer(
        action_dim, action_high, action_low, horizon, num_samples, num_elites, num_iters
    )

    # 4. Random exploration; its mean return is the baseline to beat
    print("Collecting initial random exploration data...")
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
        f"(+/- {np.std(random_returns):.1f}), buffer = {buffer.size}"
    )

    # 5. Online training and MPC loop
    returns = []
    for ep in range(num_episodes):
        print(f"\n--- Starting Episode {ep + 1}/{num_episodes} ---")

        if ep > 0:
            lo = buffer.ptr - T
            one_step_check(
                ensemble,
                buffer.states[lo : buffer.ptr],
                buffer.actions[lo : buffer.ptr],
                buffer.next_states[lo : buffer.ptr],
                device,
            )

        print("Retraining probabilistic ensemble...")
        ensemble.train_ensemble(buffer, batch_size=256, epochs=50)

        state, _ = env.reset()
        episode_reward, step_count = 0.0, 0
        for _ in range(T):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            action_tensor = cem.optimize(
                state_tensor,
                ensemble,
                half_cheetah_reward_fn,
                num_particles=num_particles,
            )
            action_np = action_tensor.detach().cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(action_np)
            buffer.add(state, action_np, next_state)

            state = next_state
            episode_reward += reward
            step_count += 1
            if terminated or truncated:
                break

        returns.append(episode_reward)
        print(
            f"Episode {ep + 1} Completed | Steps: {step_count} | Total Reward: {episode_reward:.2f} "
            f"| random baseline {np.mean(random_returns):.1f}"
        )


if __name__ == "__main__":
    run_pets_halfcheetah()
