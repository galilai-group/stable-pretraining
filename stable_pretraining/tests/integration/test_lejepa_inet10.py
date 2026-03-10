"""Deterministic smoke test for the LeJEPA training pipeline."""

import lightning as pl
import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.lejepa import LeJEPA


@pytest.mark.integration
@pytest.mark.download
@pytest.mark.filterwarnings("ignore:`isinstance.treespec, LeafSpec.` is deprecated")
@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:Trying to infer the `batch_size`")
class TestLeJEPAImagenet10:
    """Run LeJEPA (vit_tiny) on imagenette for 3 steps and check determinism."""

    def test_lejepa_10_steps(self):
        """Train LeJEPA for 3 steps and assert loss matches expected value."""
        pl.seed_everything(42, workers=True)

        # Build data from frgfm/imagenette
        # LeJEPA uses 2 global views (224x224) + 2 local views (96x96)
        train_transform = transforms.MultiViewTransform(
            {
                "global_0": transforms.Compose(
                    transforms.RGB(),
                    transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
                "global_1": transforms.Compose(
                    transforms.RGB(),
                    transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
                "local_0": transforms.Compose(
                    transforms.RGB(),
                    transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.3)),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
                "local_1": transforms.Compose(
                    transforms.RGB(),
                    transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.3)),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
            }
        )

        data = spt.data.DataModule(
            train=torch.utils.data.DataLoader(
                dataset=spt.data.HFDataset(
                    "frgfm/imagenette",
                    split="train",
                    revision="refs/convert/parquet",
                    transform=train_transform,
                ),
                batch_size=16,
                num_workers=0,
                drop_last=True,
                shuffle=True,
            ),
            val=torch.utils.data.DataLoader(
                dataset=spt.data.HFDataset(
                    "frgfm/imagenette",
                    split="validation",
                    revision="refs/convert/parquet",
                    transform=transforms.Compose(
                        transforms.RGB(),
                        transforms.Resize((256, 256)),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToImage(**spt.data.static.ImageNet),
                    ),
                ),
                batch_size=16,
                num_workers=0,
            ),
        )

        # Create LeJEPA module with vit_tiny for fast CPU testing
        # drop_path_rate=0.0 for determinism on CPU
        module = LeJEPA(
            encoder_name="vit_tiny_patch16_224",
            lamb=0.02,
            n_slices=64,
            n_points=17,
            pretrained=False,
            drop_path_rate=0.0,
        )
        module.optim = {
            "optimizer": {
                "type": "AdamW",
                "lr": 5e-4,
                "weight_decay": 0.05,
                "betas": (0.9, 0.999),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        }

        # Create trainer (CPU-compatible, stripped down for testing)
        trainer = pl.Trainer(
            max_steps=3,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            devices=1,
            accelerator="cpu",
            enable_progress_bar=False,
        )

        # Run training
        manager = spt.Manager(trainer=trainer, module=module, data=data, seed=42)
        manager()

        # Verify deterministic loss
        final_loss = trainer.callback_metrics.get("fit/loss_step")
        assert final_loss is not None, "No loss logged"
        print(f"\nLeJEPA final loss after 3 steps: {final_loss.item():.6f}")
        expected = torch.tensor(0.433364)
        assert torch.isclose(final_loss.cpu(), expected, atol=1e-4), (
            f"LeJEPA loss {final_loss.item():.6f} != expected {expected.item():.6f}"
        )
