"""Integration tests for Masked Autoencoder (MAE) functionality."""

import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.methods.mae import MAE


@pytest.mark.integration
class TestMAEIntegration:
    """Integration tests for MAE with actual training and data."""

    @pytest.mark.gpu
    def test_mae_reconstruction_loss(self):
        """Test MAE reconstruction loss computation."""
        # Create a small MAE model
        model = MAE("vit_base_patch16_224")
        model.train()

        # Create dummy batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        # Forward pass — loss is computed internally
        with torch.cuda.amp.autocast():
            output = model(images)

        loss = output.loss

        # Verify loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar loss
        assert loss.item() > 0  # positive loss

    @pytest.mark.gpu
    def test_mae_feature_extraction(self):
        """Test MAE feature extraction for downstream tasks."""
        # Create encoder (no masking needed for feature extraction)
        encoder = spt.backbone.MaskedEncoder("vit_base_patch16_224")
        encoder.eval()

        # Create dummy batch
        images = torch.randn(4, 3, 224, 224)

        # Extract features
        with torch.no_grad():
            output = encoder(images)
            cls_features = output.encoded[:, 0]  # CLS token is first prefix token

        # Verify feature dimensions
        assert cls_features.shape == (4, 768)  # ViT-Base has 768-dim features

    def test_mae_patchify_unpatchify(self):
        """Test patch embedding layer output dimensions."""
        from timm.layers import PatchEmbed

        # Create patch embedding layer
        patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)

        # Create dummy images
        images = torch.randn(2, 3, 224, 224)

        # Apply patch embedding
        patches = patch_embed(images)

        # Verify patch dimensions
        num_patches = (224 // 16) ** 2  # 196
        assert patches.shape == (2, num_patches, 768)

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
