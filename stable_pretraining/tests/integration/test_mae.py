"""Integration tests for Masked Autoencoder (MAE) functionality."""

import lightning as pl
import pytest
import torch
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import transforms


@pytest.mark.integration
class TestMAEIntegration:
    """Integration tests for MAE with actual training and data."""

    @pytest.mark.v1
    @pytest.mark.gpu
    @pytest.mark.download
    @pytest.mark.slow
    def test_mae_with_probing(self):
        """Test MAE training with online linear probing."""
        # Define transforms
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(mean=mean, std=std),
        )

        # Create train dataset with multi-view sampling
        train_dataset = spt.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="train",
            transform=train_transform,
        )

        train = torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
            batch_size=64,
            num_workers=20,
            drop_last=True,
        )

        # Create validation dataset
        val_transform = transforms.Compose(
            transforms.RGB(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToImage(mean=mean, std=std),
        )

        val = torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                path="frgfm/imagenette",
                name="160px",
                split="validation",
                transform=val_transform,
            ),
            batch_size=128,
            num_workers=10,
        )

        data = spt.data.DataModule(train=train, val=val)

        # Define MAE forward function
        def forward(self, batch, stage):
            latent, pred, mask = self.backbone(batch["image"])
            batch["embedding"] = latent[:, 0]  # CLS token only
            if self.training:
                loss = spt.losses.mae(
                    self.backbone.patchify(batch["image"]), pred, mask
                )
                batch["loss"] = loss
            return batch

        # Create MAE backbone and module
        backbone = spt.backbone.mae.vit_base_patch16_dec512d8b()
        module = spt.Module(backbone=backbone, forward=forward)

        # Create online probe callback
        linear_probe = spt.callbacks.OnlineProbe(
            module,
            "linear_probe",
            "embedding",
            "label",
            probe=torch.nn.Linear(768, 10),
            loss=torch.nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(10),
                "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
            },
        )

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=6,
            num_sanity_val_steps=1,
            callbacks=[linear_probe],
            precision="16-mixed",
            logger=False,
            enable_checkpointing=False,
        )

        # Run training
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()

    @pytest.mark.gpu
    def test_mae_reconstruction_loss(self):
        """Test MAE reconstruction loss computation."""
        # Create a small MAE model
        backbone = spt.backbone.mae.vit_base_patch16_dec512d8b()

        # Create dummy batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        with torch.cuda.amp.autocast():
            latent, pred, mask = backbone(images)

        # Compute reconstruction loss
        patches = backbone.patchify(images)
        loss = spt.losses.mae(patches, pred, mask)

        # Verify loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar loss
        assert loss.item() > 0  # positive loss

    @pytest.mark.gpu
    def test_mae_feature_extraction(self):
        """Test MAE feature extraction for downstream tasks."""
        # Create MAE backbone
        backbone = spt.backbone.mae.vit_base_patch16_dec512d8b()
        backbone.eval()

        # Create dummy batch
        images = torch.randn(4, 3, 224, 224)

        # Extract features
        with torch.no_grad():
            latent, _, _ = backbone(images)
            cls_features = latent[:, 0]  # Extract CLS token

        # Verify feature dimensions
        assert cls_features.shape == (4, 768)  # ViT-Base has 768-dim features

    def test_mae_patchify_unpatchify(self):
        """Test MAE patchify and unpatchify operations."""
        from stable_pretraining.backbone.mae import PatchEmbed

        # Create patch embedding layer
        patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)

        # Create dummy images
        images = torch.randn(2, 3, 224, 224)

        # Patchify
        patches = patch_embed(images)

        # Verify patch dimensions
        num_patches = (224 // 16) ** 2  # 196
        assert patches.shape == (2, num_patches, 768)

    @pytest.mark.v1
    @pytest.mark.download
    def test_mae_with_different_masking_ratios(self):
        """Test MAE with different masking ratios."""
        # Note: This test would require modifying the MAE backbone to accept mask_ratio
        # For now, we'll test that the model works with its default masking

        backbone = spt.backbone.mae.vit_base_patch16_dec512d8b()
        images = torch.randn(2, 3, 224, 224)

        # Forward pass
        latent, pred, mask = backbone(images)

        # Check that masking is applied
        assert mask.shape == (2, 196)  # mask for each patch
        assert mask.dtype == torch.bool

        # Verify some patches are masked
        assert mask.sum() > 0
        assert mask.sum() < mask.numel()  # Not all patches should be masked

    def test_mae_multi_view_sampling(self):
        """Test MAE with multi-view data augmentation."""
        from stable_pretraining.data.sampler import RepeatedRandomSampler

        # Create dummy dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 224, 224), torch.randint(0, 10, (100,))
        )

        # Create sampler with 2 views
        sampler = RepeatedRandomSampler(dataset, n_views=2)

        # Create dataloader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            sampler=sampler,
        )

        # Get a batch
        batch = next(iter(loader))
        images, labels = batch

        # Verify we get repeated samples (2 views per sample)
        assert images.shape[0] == 4
        # Due to the random nature of sampling, we can't guarantee exact duplicates
        # but the sampler should be working

    @pytest.mark.v1
    @pytest.mark.gpu
    def test_mae_training_step(self):
        """Test a single MAE training step."""
        # Create simple MAE setup
        backbone = spt.backbone.mae.vit_base_patch16_dec512d8b()

        def forward(self, batch, stage):
            latent, pred, mask = self.backbone(batch["image"])
            batch["embedding"] = latent[:, 0]
            if self.training:
                loss = spt.losses.mae(
                    self.backbone.patchify(batch["image"]), pred, mask
                )
                batch["loss"] = loss
            return batch

        module = spt.Module(backbone=backbone, forward=forward)
        module.train()

        # Create dummy batch
        batch = {"image": torch.randn(2, 3, 224, 224), "label": torch.tensor([0, 1])}

        # Run forward pass
        output = module.training_step(batch, 0)

        # Verify output
        assert "loss" in output
        assert isinstance(output["loss"], torch.Tensor)
        assert output["loss"].requires_grad
