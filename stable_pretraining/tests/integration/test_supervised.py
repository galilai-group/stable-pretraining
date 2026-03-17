"""Integration tests for supervised training functionality."""

import lightning as pl
import pytest
import torch
import torchmetrics
from transformers import AutoConfig, AutoModelForImageClassification

import stable_pretraining as spt
from stable_pretraining.data import transforms


@pytest.mark.integration
class TestSupervisedIntegration:
    """Integration tests for supervised training with actual models and data."""

    @pytest.mark.v1
    @pytest.mark.gpu
    @pytest.mark.download
    @pytest.mark.slow
    def test_supervised_training_with_probing(self):
        """Test supervised training with online probing and RankMe."""
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
            transforms.GaussianBlur(kernel_size=(5, 5), p=1.0),
            transforms.ToImage(mean=mean, std=std),
        )

        # Create train dataset
        train_dataset = spt.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="train",
            transform=train_transform,
        )

        train = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=64, num_workers=20, drop_last=True
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

        # Define supervised forward function
        def forward(self, batch, stage):
            batch["embedding"] = self.backbone(batch["image"])["logits"]
            if self.training:
                preds = self.classifier(batch["embedding"])
                batch["loss"] = torch.nn.functional.cross_entropy(preds, batch["label"])
            return batch

        # Create backbone and classifier
        config = AutoConfig.from_pretrained("microsoft/resnet-18")
        backbone = AutoModelForImageClassification.from_config(config)
        backbone.classifier[1] = torch.nn.Identity()
        classifier = torch.nn.Linear(512, 10)

        # Create module
        module = spt.Module(backbone=backbone, classifier=classifier, forward=forward)

        # Create callbacks
        linear_probe = spt.callbacks.OnlineProbe(
            module,
            "linear_probe",
            "embedding",
            "label",
            probe=torch.nn.Linear(512, 10),
            loss=torch.nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(10),
                "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
            },
        )

        knn_probe = spt.callbacks.OnlineKNN(
            "knn_probe",
            "embedding",
            "label",
            20000,
            metrics=torchmetrics.classification.MulticlassAccuracy(10),
            k=10,
            features_dim=512,
        )

        rankme = spt.callbacks.RankMe(
            module, "rankme", "embedding", 20000, target_shape=512
        )

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=10,
            num_sanity_val_steps=1,
            callbacks=[linear_probe, knn_probe, rankme],
            precision="16-mixed",
            logger=False,
            enable_checkpointing=False,
        )

        # Run training
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()

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

    @pytest.mark.v1
    @pytest.mark.download
    def test_imagenette_loading_supervised(self):
        """Test ImageNette dataset loading for supervised training."""
        transform = transforms.Compose(
            transforms.RGB(),
            transforms.Resize((224, 224)),
            transforms.ToImage(),
        )

        # Load dataset
        dataset = spt.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="train[:100]",  # Small subset
            transform=transform,
        )

        # Test sample
        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape == (3, 224, 224)
        assert 0 <= sample["label"] <= 9

    @pytest.mark.v1
    @pytest.mark.gpu
    def test_mixed_precision_supervised(self):
        """Test supervised training with mixed precision."""
        # Create simple model
        backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        classifier = torch.nn.Linear(64, 10)

        # Create dummy batch
        images = torch.randn(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))

        # Test with autocast
        with torch.cuda.amp.autocast():
            features = backbone(images)
            preds = classifier(features)
            loss = torch.nn.functional.cross_entropy(preds, labels)

        # Verify types
        assert features.dtype == torch.float16 or features.dtype == torch.bfloat16
        assert loss.dtype == torch.float32

    @pytest.mark.v1
    def test_data_augmentations_supervised(self):
        """Test data augmentations for supervised training."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Create augmentation pipeline
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
        image = torch.randn(3, 256, 256)

        # Apply augmentations multiple times
        aug1 = augment(image.clone())
        aug2 = augment(image.clone())

        # Verify augmentations produce different results
        assert aug1.shape == aug2.shape == (3, 224, 224)
        assert not torch.allclose(aug1, aug2)

    @pytest.mark.v1
    @pytest.mark.gpu
    def test_supervised_training_step(self):
        """Test a single supervised training step."""
        # Create simple setup
        backbone = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 32 * 32, 512),
            torch.nn.ReLU(),
        )
        classifier = torch.nn.Linear(512, 10)

        def forward(self, batch, stage):
            batch["embedding"] = self.backbone(batch["image"])
            if self.training:
                preds = self.classifier(batch["embedding"])
                batch["loss"] = torch.nn.functional.cross_entropy(preds, batch["label"])
            return batch

        module = spt.Module(backbone=backbone, classifier=classifier, forward=forward)
        module.train()

        # Create dummy batch
        batch = {
            "image": torch.randn(8, 3, 32, 32),
            "label": torch.randint(0, 10, (8,)),
        }

        # Run training step
        output = module.training_step(batch, 0)

        # Verify output
        assert "loss" in output
        assert output["loss"].requires_grad
        assert "embedding" in output
        assert output["embedding"].shape == (8, 512)
