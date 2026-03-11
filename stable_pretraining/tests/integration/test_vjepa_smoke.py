"""Deterministic smoke test for the V-JEPA training pipeline."""

import types

import lightning as pl
import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.methods.vjepa import VJEPA


class SyntheticVideoDataset(torch.utils.data.Dataset):
    """Fixed synthetic video dataset for deterministic testing.

    All tensors are pre-generated from a seeded RNG so the dataset is fully
    reproducible with no external downloads required.
    """

    def __init__(self, num_samples: int = 64, num_frames: int = 4, seed: int = 0):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.videos = torch.randn(num_samples, 3, num_frames, 224, 224, generator=rng)
        self.labels = torch.randint(0, 10, (num_samples,), generator=rng)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int):
        return {"video": self.videos[idx], "label": self.labels[idx]}


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:`isinstance.treespec, LeafSpec.` is deprecated")
@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:Trying to infer the `batch_size`")
class TestVJEPASmoke:
    """Run VJEPA (vit_tiny) on synthetic video for 3 steps and check determinism."""

    def test_vjepa_3_steps(self):
        """Train VJEPA for 3 steps and assert loss matches expected value."""
        pl.seed_everything(42, workers=True)

        num_frames = 4

        data = spt.data.DataModule(
            train=torch.utils.data.DataLoader(
                dataset=SyntheticVideoDataset(
                    num_samples=64, num_frames=num_frames, seed=0
                ),
                batch_size=4,
                num_workers=0,
                drop_last=True,
                shuffle=True,
            ),
            val=torch.utils.data.DataLoader(
                dataset=SyntheticVideoDataset(
                    num_samples=16, num_frames=num_frames, seed=1
                ),
                batch_size=4,
                num_workers=0,
            ),
        )

        def vjepa_forward(self, batch, stage):
            output = VJEPA.forward(self, batch["video"])
            embedding = output.embedding.detach() if self.training else output.embedding

            self.log(
                f"{stage}/loss",
                output.loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

            return {
                "loss": output.loss,
                "embedding": embedding,
                "label": batch["label"].long(),
            }

        module = VJEPA(
            encoder_name="vit_tiny_patch16_224",
            num_frames=num_frames,
            predictor_embed_dim=192,
            predictor_depth=4,
            num_targets=4,
            target_scale=(0.15, 0.2),
            target_aspect_ratio=(0.75, 1.5),
            context_scale=(1.0, 1.0),
            ema_decay_start=0.996,
            ema_decay_end=1.0,
            pretrained=False,
        )

        module.forward = types.MethodType(vjepa_forward, module)
        module.optim = {
            "optimizer": {
                "type": "AdamW",
                "lr": 6e-4,
                "weight_decay": 0.05,
                "betas": (0.9, 0.95),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        }

        trainer = pl.Trainer(
            max_steps=3,
            num_sanity_val_steps=0,
            callbacks=[
                spt.callbacks.TeacherStudentCallback(
                    update_frequency=1,
                    update_after_backward=True,
                ),
            ],
            logger=False,
            enable_checkpointing=False,
            devices=1,
            accelerator="cpu",
            enable_progress_bar=False,
        )

        manager = spt.Manager(trainer=trainer, module=module, data=data, seed=42)
        manager()

        final_loss = trainer.callback_metrics.get("fit/loss_step")
        assert final_loss is not None, "No loss logged"
        print(f"\nVJEPA final loss after 3 steps: {final_loss.item():.6f}")
        # NOTE: Update this expected value after the first successful reference run.
        expected = torch.tensor(0.422097)  # calibrated on first run
        assert torch.isclose(final_loss.cpu(), expected, atol=1e-4), (
            f"VJEPA loss {final_loss.item():.6f} != expected {expected.item():.6f}"
        )
