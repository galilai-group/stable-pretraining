"""Integration tests for supervised training functionality."""

import pytest
import torch
from transformers import AutoModelForImageClassification

from stable_pretraining.data import transforms


@pytest.mark.integration
class TestSupervisedIntegration:
    """Integration tests for supervised training with actual models and data."""

    @pytest.mark.gpu
    def test_supervised_loss_computation(self):
        """Test supervised loss computation with actual tensors."""
        batch_size = 32
        num_classes = 10
        feature_dim = 512

        # Create classifier
        classifier = torch.nn.Linear(feature_dim, num_classes)

        # Create dummy features and labels
        features = torch.randn(batch_size, feature_dim)
        labels = torch.randint(0, num_classes, (batch_size,))

        # Forward pass
        preds = classifier(features)
        loss = torch.nn.functional.cross_entropy(preds, labels)

        # Verify loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert loss.requires_grad

    @pytest.mark.gpu
    def test_feature_extraction_supervised(self):
        """Test feature extraction in supervised setting."""
        # Load pretrained backbone
        backbone = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-18"
        )
        backbone.classifier[1] = torch.nn.Identity()
        backbone.eval()

        # Create dummy batch
        batch = torch.randn(4, 3, 224, 224)

        # Extract features
        with torch.no_grad():
            output = backbone(batch)
            features = output["logits"]

        # Verify features
        assert features.shape == (4, 512)

    def test_rankme_computation(self):
        """Test RankMe metric computation logic."""
        # Create dummy features
        features = torch.randn(100, 512)

        # Compute singular values for rank estimation
        _, s, _ = torch.svd(features)

        # Normalize singular values
        s_norm = s / s.sum()

        # Compute entropy (simplified RankMe)
        entropy = -(s_norm * torch.log(s_norm + 1e-8)).sum()
        rank_estimate = torch.exp(entropy)

        # Verify computation
        assert isinstance(rank_estimate, torch.Tensor)
        assert rank_estimate.item() > 0
        assert rank_estimate.item() <= min(features.shape)

    def test_data_augmentations_supervised(self):
        """Test data augmentations for supervised training."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Create augmentation pipeline. SPT transforms operate on dict samples
        # via `source`/`target` keys (default: "image").
        augment = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=(5, 5), p=1.0),
            transforms.ToImage(mean=mean, std=std),
        )

        # Create dummy image
        sample = {"image": torch.randn(3, 256, 256)}

        # Apply augmentations multiple times
        aug1 = augment({"image": sample["image"].clone()})["image"]
        aug2 = augment({"image": sample["image"].clone()})["image"]

        # Verify augmentations produce different results
        assert aug1.shape == aug2.shape == (3, 224, 224)
        assert not torch.allclose(aug1, aug2)
