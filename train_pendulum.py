import gymnasium as gym
import torch
import numpy as np
from probabilistic_ensembles import ReplayBuffer, ProbabilisticEnsemble
from cem_optim import CEMOptimizer


def run_pets_pendulum():
    # 1. Initialize Environment
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]  # 3: [cos(theta), sin(theta), theta_dot]
    action_dim = env.action_space.shape[0]  # 1: [torque]
    action_low = env.action_space.low
    action_high = env.action_space.high

    # 2. Hyperparameters (following PETS paper recommendations for low-dim tasks)
    num_models = 5  # Ensemble size (B)
    horizon = 25  # Planning horizon (H)
    num_particles = 20  # Particle swarm size (P)
    num_samples = 400  # CEM candidate action sequences (N)
    num_elites = 40  # CEM elite sequences kept (M)
    num_iters = 5  # CEM refinement iterations (K)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    buffer = ReplayBuffer(
        state_dim, action_dim, max_size=50000, device=torch.device("cuda")
    )
    ensemble = ProbabilisticEnsemble(
        num_models,
        state_dim,
        action_dim,
        hidden_dim=200,
        lr=1e-3,
        device=torch.device("cuda"),
    )
    cem = CEMOptimizer(
        action_dim, action_high, action_low, horizon, num_samples, num_elites, num_iters
    )

    print("Collecting initial random exploration data...")
    for _ in range(3):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, next_state)
            state = next_state

    # Define the Reward Function for Particle Trajectories (Pendulum specific)
    def pendulum_reward_fn(trajectory, actions):
        cos_th, sin_th, th_dot = (trajectory[:, :, 1:, i] for i in range(3))
        theta = torch.atan2(sin_th, cos_th)
        u = actions.squeeze(-1)[:, None, :]  # (N, 1, H)
        cost = theta**2 + 0.1 * th_dot**2 + 0.001 * u**2  # (N, P, H)
        return -cost.sum(-1).mean(1)  # (N,)

    # Main Online Training & MPC Loop
    num_episodes = 15
    for ep in range(num_episodes):
        print(f"\n--- Starting Episode {ep + 1}/{num_episodes} ---")

        print("Retraining probabilistic ensemble...")
        ensemble.train_ensemble(buffer, batch_size=256, epochs=50)

        state, _ = env.reset(seed=1000 + ep % 5)
        print(f"start theta = {np.arctan2(state[1], state[0]):.2f}")
        done = False
        episode_reward = 0
        step_count = 0

        while not done and step_count < 200:
            state_tensor = torch.FloatTensor(state).to(device)

            # Plan next action using CEM + TS_inf Particle Propagation
            action_tensor = cem.optimize(
                state_tensor, ensemble, pendulum_reward_fn, num_particles=num_particles
            )
            action_np = action_tensor.detach().cpu().numpy()

            # Execute action in the real physical environment
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # Store transition in replay buffer
            buffer.add(state, action_np, next_state)

            state = next_state
            episode_reward += reward
            step_count += 1

        print(
            f"Episode {ep + 1} Completed | Steps: {step_count} | Total Reward: {episode_reward:.2f}"
        )


if __name__ == "__main__":
    run_pets_pendulum()
