"""Integration tests for LeWM training and checkpoint resumption."""

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ViTConfig, ViTModel

from stable_pretraining.methods.lewm import LeWM
from stable_pretraining.methods.lewm_components.architecture import (
    ARPredictor,
    Embedder,
)
from stable_pretraining.methods.lewm_components.jepa import JEPA


class SyntheticSequences(Dataset):
    """Synthetic sequences for training tests."""

    def __len__(self):
        return 8

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return {
            "pixels": torch.randn(4, 3, 32, 32, generator=generator),
            "action": torch.randn(4, 3, generator=generator),
        }


def make_method():
    encoder = ViTModel(
        ViTConfig(
            image_size=32,
            patch_size=8,
            num_channels=3,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
        )
    )

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(
            num_frames=3,
            depth=1,
            heads=2,
            mlp_dim=64,
            input_dim=16,
            hidden_dim=16,
            dim_head=8,
        ),
        action_encoder=Embedder(
            input_dim=3,
            smoothed_dim=8,
            emb_dim=16,
        ),
        projector=torch.nn.Linear(32, 16),
    )

    return LeWM(
        model=model,
        history_size=3,
        num_preds=1,
        optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
    )


def test_lewm_training_and_resume(tmp_path):
    """Train real LeWM components, save a checkpoint, and resume."""
    pl.seed_everything(42)

    loader = DataLoader(SyntheticSequences(), batch_size=2)

    method = make_method()
    trainer = pl.Trainer(
        max_steps=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(method, train_dataloaders=loader)
    assert trainer.global_step == 2

    checkpoint = tmp_path / "lewm.ckpt"
    trainer.save_checkpoint(checkpoint)
    assert checkpoint.exists()

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "state_dict" in saved
    assert saved["optimizer_states"]

    resumed_method = make_method()
    resumed_trainer = pl.Trainer(
        max_steps=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    resumed_trainer.fit(
        resumed_method,
        train_dataloaders=loader,
        ckpt_path=str(checkpoint),
    )

    assert resumed_trainer.global_step == 3
