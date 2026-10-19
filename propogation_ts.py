import torch
from propogation_ts import propagate_particles


class CEMOptimizer:
    """The Cross-Entropy Method (CEM) planner for Model Predictive Control."""

    def __init__(
        self,
        action_dim,
        action_high,
        action_low,
        horizon=30,
        num_samples=500,
        num_elites=50,
        num_iters=5,
        scalar_var_init=1.0,
    ):
        self.action_dim = action_dim
        self.action_high = torch.tensor(action_high, dtype=torch.float32)
        self.action_low = torch.tensor(action_low, dtype=torch.float32)

        self.H = horizon  # Planning horizon
        self.N = (
            num_samples  # Number of candidate action sequences sampled per iteration
        )
        self.M = num_elites  # Number of top elite sequences kept
        self.K = num_iters  # Number of CEM refinement iterations
        self.scalar_var_init = scalar_var_init

    def optimize(self, initial_state, ensemble, reward_fn, num_particles=20):
        """Optimizes an action sequence over horizon H using CEM.

        Args:
            initial_state (torch.Tensor): Current state of the environment.
            ensemble (ProbabilisticEnsemble): Trained dynamics ensemble.
            reward_fn (callable): Function taking (states, actions) and returning cumulative reward.
            num_particles (int): Number of particles P used in propagation.

        Returns:
            torch.Tensor: The first optimal action to execute in the environment, shape (action_dim,)
        """
        device = initial_state.device
        self.action_high = self.action_high.to(device)
        self.action_low = self.action_low.to(device)

        # 1. Initialize action distribution parameters: Mean = 0, Variance = Identity * scalar_var_init
        mean = torch.zeros((self.H, self.action_dim), device=device)
        var = (
            torch.ones((self.H, self.action_dim), device=device) * self.scalar_var_init
        )

        for _ in range(self.K):
            # 2. Sample N candidate action sequences from Gaussian distribution: shape (N, H, action_dim)
            std = torch.sqrt(var)
            action_samples = (
                torch.randn((self.N, self.H, self.action_dim), device=device) * std
                + mean
            )

            # Clip actions to valid environment bounds (e.g., joint torque limits)
            action_samples = torch.clamp(
                action_samples, self.action_low, self.action_high
            )

            # 3. Evaluate all N action sequences using particle propagation
            trajectory = propagate_particles(
                initial_state, action_samples, ensemble, num_particles
            )
            batch_rewards = reward_fn(trajectory, action_samples)  # (N,)

            # 4. Select the top M "elite" action sequences
            _, elite_idxs = torch.topk(batch_rewards, self.M, largest=True)
            elites = action_samples[elite_idxs]  # shape: (M, H, action_dim)

            # 5. Update distribution mean and variance to match the elites
            mean = elites.mean(dim=0)
            var = (
                elites.var(dim=0, unbiased=False) + 1e-6
            )  # Add tiny epsilon for numerical stability

        # Return only the very first action of the optimized sequence
        return mean[0]
