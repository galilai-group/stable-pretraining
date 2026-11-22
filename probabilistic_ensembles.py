import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    """Stores environment transition tuples (s, a, s_next) and computes deltas."""

    def __init__(
        self, state_dim, action_dim, max_size=100000, device=torch.device("cpu")
    ):
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def add(self, state, action, next_state):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        states = torch.FloatTensor(self.states[idx]).to(self.device)
        actions = torch.FloatTensor(self.actions[idx]).to(self.device)
        # Target is the state delta: delta_s = s_{t+1} - s_t
        deltas = torch.FloatTensor(self.next_states[idx] - self.states[idx]).to(
            self.device
        )
        return states, actions, deltas


class ProbabilisticEnsembleMember(nn.Module):
    """A single neural network predicting mean and log-variance for state deltas."""

    def __init__(self, state_dim, action_dim, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_layer = nn.Linear(hidden_dim, state_dim)
        self.log_var_layer = nn.Linear(hidden_dim, state_dim)

        # Learnable variance bounds (crucial PETS stabilization trick)
        self.max_log_var = nn.Parameter(torch.ones(state_dim) * 0.5, requires_grad=True)
        self.min_log_var = nn.Parameter(
            torch.ones(state_dim) * -10.0, requires_grad=True
        )

    def forward(self, x):
        h = torch.nn.functional.silu(self.fc1(x))
        h = torch.nn.functional.silu(self.fc2(h))
        h = torch.nn.functional.silu(self.fc3(h))

        mean = self.mean_layer(h)
        raw_log_var = self.log_var_layer(h)

        # Bounded log-variance prevents numerical explosions / division-by-zero
        log_var = self.max_log_var - torch.nn.functional.softplus(
            self.max_log_var - raw_log_var
        )
        log_var = self.min_log_var + torch.nn.functional.softplus(
            log_var - self.min_log_var
        )

        return mean, log_var


class ProbabilisticEnsemble:
    """Manages an ensemble of B models trained with bootstrap sampling and Gaussian NLL loss."""

    def __init__(
        self,
        num_models,
        state_dim,
        action_dim,
        hidden_dim=200,
        lr=1e-3,
        device=torch.device("cpu"),
    ):
        self.num_models = num_models
        self.device = device
        self.models = [
            ProbabilisticEnsembleMember(state_dim, action_dim, hidden_dim).to(device)
            for _ in range(num_models)
        ]
        self.optimizers = [
            optim.Adam(model.parameters(), lr=lr) for model in self.models
        ]

    def gaussian_nll_loss(self, mean, log_var, target, model):
        """Gaussian Negative Log-Likelihood loss with variance regularization."""
        inv_var = torch.exp(-log_var)
        mse = (target - mean) ** 2
        # NLL loss formula: 0.5 * (mse * inv_var + log_var)
        nll = 0.5 * (mse * inv_var + log_var).mean(dim=-1)

        # Regularization penalty on max/min log variance bounds (Chua et al. 2018)
        bound_penalty = 0.01 * (model.max_log_var.sum() - model.min_log_var.sum())
        return nll.mean() + bound_penalty

    def train_ensemble(self, buffer, batch_size=256, epochs=5):
        if buffer.size < batch_size:
            return  # Not enough data yet

        states, actions, targets = buffer.sample(buffer.size)
        inputs = torch.cat([states, actions], dim=-1)
        dataset_size = states.shape[0]

        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            # Bootstrap sample indices with replacement for each ensemble member
            boot_idx = np.random.choice(dataset_size, size=dataset_size, replace=True)
            b_inputs = inputs[boot_idx]
            b_targets = targets[boot_idx]

            model.train()
            perm = torch.randperm(dataset_size, device=self.device)

            for epoch in range(epochs):
                for start in range(0, dataset_size, batch_size):
                    end = start + batch_size
                    batch_idx = perm[start:end]

                    batch_in = b_inputs[batch_idx]
                    batch_tgt = b_targets[batch_idx]

                    optimizer.zero_grad()
                    mean, log_var = model(batch_in)
                    loss = self.gaussian_nll_loss(mean, log_var, batch_tgt, model)
                    loss.backward()
                    optimizer.step()
