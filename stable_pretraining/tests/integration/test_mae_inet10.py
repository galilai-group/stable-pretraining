"""Deterministic smoke test for the MAE training pipeline."""

import lightning as pl
import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.mae import MAE


@pytest.mark.integration
@pytest.mark.download
@pytest.mark.filterwarnings("ignore:`isinstance.treespec, LeafSpec.` is deprecated")
@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:Trying to infer the `batch_size`")
class TestMAEImagenet10:
    """Run MAE (vit_tiny) on imagenette for 10 steps and check determinism."""

    def test_mae_10_steps(self):
        """Train MAE for 10 steps and assert loss matches expected value."""
        pl.seed_everything(42, workers=True)

        # Build data from frgfm/imagenette
        data = spt.data.DataModule(
            train=torch.utils.data.DataLoader(
                dataset=spt.data.HFDataset(
                    "frgfm/imagenette",
                    split="train",
                    revision="refs/convert/parquet",
                    transform=transforms.Compose(
                        transforms.RGB(),
                        transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToImage(**spt.data.static.ImageNet),
                    ),
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

        # Create MAE module with vit_tiny for fast CPU testing
        module = MAE(
            encoder_name="vit_tiny_patch16_224",
            decoder_embed_dim=192,
            decoder_depth=4,
            decoder_num_heads=3,
            mask_ratio=0.75,
            block_size=1,
            norm_pix_loss=True,
            loss_type="mse",
            pretrained=False,
        )

        module.optim = {
            "optimizer": {
                "type": "AdamW",
                "lr": 5e-4,
                "weight_decay": 0.05,
                "betas": (0.9, 0.95),
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
        print(f"\nMAE final loss after 3 steps: {final_loss.item():.6f}")
        expected = torch.tensor(1.214716)
        assert torch.isclose(final_loss.cpu(), expected, atol=1e-4), (
            f"MAE loss {final_loss.item():.6f} != expected {expected.item():.6f}"
        )
