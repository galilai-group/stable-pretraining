"""Integration tests for probing functionality (linear probe and KNN)."""

import pytest
import torch
from transformers import AutoModelForImageClassification

import stable_pretraining as spt


@pytest.mark.integration
class TestProbingIntegration:
    """Integration tests for probing with actual models and data."""

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
