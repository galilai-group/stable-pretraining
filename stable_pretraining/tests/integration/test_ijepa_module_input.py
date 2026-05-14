"""Deterministic smoke test for IJEPA with pre-instantiated nn.Module backbone."""

import types

import lightning as pl
import pytest
import timm
import torch

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.ijepa import IJEPA


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
class TestIJEPAModuleInput:
    """Run IJEPA with a pre-loaded backbone on imagenette for 3 steps.

    Mirrors test_ijepa_inet10.py but passes an nn.Module instead of a string
    to IJEPA(), verifying the model_or_model_name interface.
    """

    def test_ijepa_3_steps_with_loaded_backbone(self):
        """Train IJEPA (pre-loaded backbone) for 3 steps and assert loss matches."""
        pl.seed_everything(42, workers=True)

        # Build data from frgfm/imagenette
        # IJEPA uses scale=(0.3, 1.0) and no horizontal flip
        data = spt.data.DataModule(
            train=torch.utils.data.DataLoader(
                dataset=spt.data.HFDataset(
                    "frgfm/imagenette",
                    split="train",
                    revision="refs/convert/parquet",
                    transform=transforms.Compose(
                        transforms.RGB(),
                        transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
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
        def ijepa_forward(self, batch, stage):
            output = IJEPA.forward(self, batch["image"], embedding_source="student")
            embedding = output.embedding.mean(dim=1)
            if self.training:
                embedding = embedding.detach()

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
                **({"label": batch["label"].long()} if "label" in batch else {}),
            }

        # Load backbone externally, then pass nn.Module to IJEPA
        backbone = load_backbone("vit_tiny_patch16_224", pretrained=False)

        module = IJEPA(
            model_or_model_name=backbone,
            predictor_embed_dim=192,
            predictor_depth=6,
            num_targets=4,
            target_scale=(0.15, 0.2),
            target_aspect_ratio=(0.75, 1.5),
            context_scale=(0.85, 1.0),
            ema_decay_start=0.996,
            ema_decay_end=1.0,
        )

        module.forward = types.MethodType(ijepa_forward, module)
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

        # Create trainer (CPU-compatible, with EMA callback)
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

        # Run training
        manager = spt.Manager(trainer=trainer, module=module, data=data, seed=42)
        manager()

        # Verify deterministic loss
        final_loss = trainer.callback_metrics.get("fit/loss_step")
        assert final_loss is not None, "No loss logged"
        print(
            f"\nIJEPA (loaded module) final loss after 3 steps: {final_loss.item():.6f}"
        )
        expected = torch.tensor(0.515345)
        assert torch.isclose(final_loss.cpu(), expected, atol=1e-4), (
            f"IJEPA loss {final_loss.item():.6f} != expected {expected.item():.6f}"
        )
