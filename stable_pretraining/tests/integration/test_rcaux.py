"""Integration test for RC-aux through the stable-pretraining training loop."""

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining.forward import rcaux
from stable_pretraining.losses import RCauxLoss


class _TrajectoryDataset(Dataset):
    """Small deterministic trajectory dataset with no external dependencies."""

    def __init__(self, size: int = 8) -> None:
        generator = torch.Generator().manual_seed(7)
        self.observation = torch.randn(size, 5, 4, generator=generator)
        self.action = torch.randn(size, 4, 2, generator=generator)

    def __len__(self) -> int:
        return len(self.observation)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation": self.observation[index],
            "action": self.action[index],
        }


class _TinyWorldModel(nn.Module):
    """Encode trajectories and predict two latent states from a shared context."""

    def __init__(self, embedding_dim: int = 3, context_index: int = 2) -> None:
        super().__init__()
        self.context_index = context_index
        self.encoder = nn.Linear(4, embedding_dim)
        self.transition = nn.Sequential(
            nn.Linear(embedding_dim + 2, 8),
            nn.Tanh(),
            nn.Linear(8, embedding_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(batch["observation"])
        state = encoded[:, self.context_index - 1]
        rollout = []
        for offset in range(2):
            action = batch["action"][:, self.context_index - 1 + offset]
            state = self.transition(torch.cat((state, action), dim=-1))
            rollout.append(state)
        return {"embedding": encoded, "prediction": torch.stack(rollout, dim=1)}


@pytest.mark.integration
def test_rcaux_trains_through_spt_module() -> None:
    """Run a real optimizer step through Module and update RC-aux parameters."""
    world_model = _TinyWorldModel()
    module = spt.Module(
        forward=rcaux,
        world_model=world_model,
        rcaux_loss=RCauxLoss(
            embedding_dim=3,
            max_horizon=3,
            hidden_dim=8,
            horizon_dim=2,
        ),
        context_index=2,
        optim={"optimizer": {"type": "AdamW", "lr": 1e-2}},
    )
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }
    loader = DataLoader(_TrajectoryDataset(), batch_size=4, shuffle=False)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(module, train_dataloaders=loader)

    assert trainer.global_step == 1
    assert any(
        not torch.equal(initial_parameters[name], parameter.detach())
        for name, parameter in module.named_parameters()
    )
