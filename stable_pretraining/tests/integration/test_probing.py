"""Integration tests for probing functionality (linear probe and KNN)."""

import lightning as pl
import pytest
import torch
import torchmetrics
from transformers import AutoModelForImageClassification

import stable_pretraining as spt
from stable_pretraining.data import transforms


@pytest.mark.integration
class TestProbingIntegration:
    """Integration tests for probing with actual models and data."""

    @pytest.mark.v1
    @pytest.mark.gpu
    @pytest.mark.download
    @pytest.mark.slow
    def test_probing_with_pretrained_backbone(self):
        """Test probing with pretrained ResNet-18 backbone."""
        # Define transforms
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224)),
            transforms.ToImage(mean=mean, std=std),
        )

        # Create train dataloader
        train = torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                path="frgfm/imagenette",
                name="160px",
                split="train[:128]",
                transform=train_transform,
            ),
            batch_size=128,
            shuffle=True,
            num_workers=10,
            drop_last=True,
        )

        # Create validation dataloader
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
                split="validation[:128]",
                transform=val_transform,
            ),
            batch_size=128,
            num_workers=10,
        )

        # Create data module
        data = spt.data.DataModule(train=train, val=val)

        # Define forward function for feature extraction
        def forward(self, batch, stage):
            with torch.inference_mode():
                x = batch["image"]
                batch["embedding"] = self.backbone(x)["logits"]
            return batch

        # Load and modify pretrained backbone
        backbone = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-18"
        )
        backbone.classifier[1] = torch.nn.Identity()

        # Create module with frozen backbone
        module = spt.Module(
            backbone=spt.backbone.EvalOnly(backbone), forward=forward, optim=None
        )

        # Create linear probe callback
        linear_probe = spt.callbacks.OnlineProbe(
            module,
            name="linear_probe",
            input="embedding",
            target="label",
            probe=torch.nn.Linear(512, 10),
            loss=torch.nn.CrossEntropyLoss(),
            metrics=torchmetrics.classification.MulticlassAccuracy(10),
        )

        # Create KNN probe callback
        knn_probe = spt.callbacks.OnlineKNN(
            name="knn_probe",
            input="embedding",
            target="label",
            queue_length=50000,
            metrics=torchmetrics.classification.MulticlassAccuracy(10),
            k=10,
            input_dim=512,
        )

        # Create trainer
        trainer = pl.Trainer(
            max_steps=10,
            num_sanity_val_steps=1,
            callbacks=[linear_probe, knn_probe],
            precision="16",
            logger=False,
            enable_checkpointing=False,
        )

        # Run training
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()
        manager.validate()

    @pytest.mark.gpu
    def test_feature_extraction_with_resnet(self):
        """Test feature extraction using ResNet-18."""
        # Load pretrained model
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

        # Verify feature dimensions
        assert features.shape == (4, 512)  # ResNet-18 outputs 512-dim features

    def test_linear_probe_training(self):
        """Test linear probe training mechanics."""
        # Create dummy features and labels
        features = torch.randn(32, 512)
        labels = torch.randint(0, 10, (32,))

        # Create linear probe
        probe = torch.nn.Linear(512, 10)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Training step
        preds = probe(features)
        loss = loss_fn(preds, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Verify loss is scalar
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_knn_classification(self):
        """Test KNN classification logic."""
        # Create dummy feature bank
        bank_features = torch.randn(100, 512)
        bank_labels = torch.randint(0, 10, (100,))

        # Create query features
        query_features = torch.randn(10, 512)

        # Compute distances
        distances = torch.cdist(query_features, bank_features)

        # Get k nearest neighbors
        k = 5
        _, indices = distances.topk(k, largest=False, dim=1)

        # Get labels of nearest neighbors
        knn_labels = bank_labels[indices]

        # Predict by majority vote
        predictions = []
        for i in range(query_features.shape[0]):
            labels, counts = torch.unique(knn_labels[i], return_counts=True)
            predictions.append(labels[counts.argmax()])

        predictions = torch.stack(predictions)

        # Verify predictions
        assert predictions.shape == (10,)
        assert all(0 <= p <= 9 for p in predictions)

    @pytest.mark.v1
    @pytest.mark.download
    def test_imagenette_dataset_loading(self):
        """Test ImageNette dataset loading."""
        transform = transforms.Compose(
            transforms.RGB(),
            transforms.Resize((224, 224)),
            transforms.ToImage(),
        )

        # Load small subset
        dataset = spt.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="train[:10]",
            transform=transform,
        )

        # Test sample
        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape == (3, 224, 224)
        assert isinstance(sample["label"], int)
        assert 0 <= sample["label"] <= 9  # ImageNette has 10 classes

    def test_eval_only_behavior(self):
        """Test EvalOnly wrapper behavior."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
        )

        # Wrap with EvalOnly
        eval_model = spt.backbone.EvalOnly(model)

        # Test that it's in eval mode
        assert not eval_model.training

        # Test forward pass
        x = torch.randn(2, 10)
        with torch.no_grad():
            output = eval_model(x)

        assert output.shape == (2, 5)

    @pytest.mark.v1
    @pytest.mark.gpu
    def test_mixed_precision_probing(self):
        """Test probing with mixed precision training."""
        # Create simple setup
        features_dim = 128
        num_classes = 5

        # Create probe
        probe = torch.nn.Linear(features_dim, num_classes)

        # Create dummy data
        features = torch.randn(16, features_dim)
        labels = torch.randint(0, num_classes, (16,))

        # Test with autocast
        with torch.cuda.amp.autocast():
            preds = probe(features)
            loss = torch.nn.functional.cross_entropy(preds, labels)

        # Verify types
        assert preds.dtype == torch.float16 or preds.dtype == torch.bfloat16
        assert loss.dtype == torch.float32  # Loss is typically kept in float32
