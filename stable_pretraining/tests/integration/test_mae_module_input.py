"""Deterministic smoke test for MAE with pre-instantiated nn.Module backbone."""

import types

import lightning as pl
import pytest
import timm
import torch

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.mae import MAE


def load_backbone(backbone_name: str, pretrained: bool = False, img_size: int = 224):
    """Load a backbone from TIMM (simplified version for testing)."""
    backbone = timm.create_model(
        backbone_name, pretrained=pretrained, num_classes=0, img_size=img_size
    )
    for p in backbone.parameters():
        p.requires_grad = True
    return backbone


@pytest.mark.integration
@pytest.mark.download
@pytest.mark.filterwarnings("ignore:`isinstance.treespec, LeafSpec.` is deprecated")
@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:Trying to infer the `batch_size`")
class TestMAEImagenet10LoadModule:
    """Run MAE with a pre-loaded backbone on imagenette for 3 steps.

    Mirrors test_mae_inet10.py but passes an nn.Module instead of a string
    to MAE(), verifying the model_or_model_name interface.
    """

    def test_mae_10_steps_with_loaded_backbone(self):
        """Train MAE (pre-loaded backbone) for 3 steps and assert loss matches."""
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

        # Forward function matching benchmark pattern
        def mae_forward(self, batch, stage):
            output = MAE.forward(self, batch["image"])
            with torch.no_grad():
                features = self.encoder.forward_features(batch["image"])

            self.log(
                f"{stage}/loss",
                output.loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

            return {
                "loss": output.loss,
                "embedding": features[:, 1:].mean(dim=1).detach(),
                **({"label": batch["label"].long()} if "label" in batch else {}),
            }

        # Load backbone externally, then pass nn.Module to MAE
        backbone = load_backbone("vit_tiny_patch16_224", pretrained=False)

        module = MAE(
            model_or_model_name=backbone,
            decoder_embed_dim=192,
            decoder_depth=4,
            decoder_num_heads=3,
            mask_ratio=0.75,
            block_size=1,
            norm_pix_loss=True,
            loss_type="mse",
        )

        module.forward = types.MethodType(mae_forward, module)
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

        # Verify deterministic loss (should match test_mae_inet10 since same
        # architecture, same seed, same data, same init — only difference is
        # the backbone was loaded externally via nn.Module path)
        final_loss = trainer.callback_metrics.get("fit/loss_step")
        assert final_loss is not None, "No loss logged"
        print(
            f"\nMAE (loaded module) final loss after 3 steps: {final_loss.item():.6f}"
        )
        expected = torch.tensor(1.223879)
        assert torch.isclose(final_loss.cpu(), expected, atol=1e-4), (
            f"MAE loss {final_loss.item():.6f} != expected {expected.item():.6f}"
        )
