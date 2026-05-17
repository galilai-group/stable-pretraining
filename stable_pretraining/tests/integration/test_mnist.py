"""Integration tests for MNIST dataset and training functionality."""

import lightning as pl
import pytest
import torch
from omegaconf import OmegaConf

import stable_pretraining as spt


@pytest.mark.integration
class TestMNISTIntegration:
    """Integration tests for MNIST training with actual data loading."""

    @pytest.mark.download
    def test_dictconfig_with_manager(self):
        """Test Manager with DictConfig for MNIST."""
        # Create configuration
        data = OmegaConf.create(
            {
                "_target_": "stable_pretraining.data.DataModule",
                "train": {
                    "dataset": {
                        "_target_": "stable_pretraining.data.HFDataset",
                        "path": "ylecun/mnist",
                        "split": "test",
                        "transform": {
                            "_target_": "stable_pretraining.data.transforms.ToImage",
                        },
                    },
                    "batch_size": 20,
                    "num_workers": 10,
                    "drop_last": True,
                    "shuffle": True,
                },
                "test": {
                    "dataset": {
                        "_target_": "stable_pretraining.data.HFDataset",
                        "path": "ylecun/mnist",
                        "split": "test",
                        "transform": {
                            "_target_": "stable_pretraining.data.transforms.ToImage",
                        },
                    },
                    "batch_size": 20,
                    "num_workers": 10,
                },
            }
        )

        # Create manager
        manager = spt.Manager(
            trainer=pl.Trainer(max_epochs=1), module=pl.LightningModule(), data=data
        )

        assert manager is not None
        assert manager.data == data

    @pytest.mark.download
    def test_mnist_dataset_properties(self):
        """Test MNIST dataset loading and properties."""
        # Load MNIST train dataset
        train_dataset = spt.data.HFDataset(
            path="ylecun/mnist", split="train", transform=spt.data.transforms.ToImage()
        )

        # Load MNIST test dataset
        test_dataset = spt.data.HFDataset(
            path="ylecun/mnist", split="test", transform=spt.data.transforms.ToImage()
        )

        # Check dataset sizes
        assert len(train_dataset) == 60000
        assert len(test_dataset) == 10000

        # Check sample structure
        train_sample = train_dataset[0]
        assert "image" in train_sample
        assert "label" in train_sample
        assert isinstance(train_sample["image"], torch.Tensor)
        assert train_sample["image"].shape == (1, 28, 28)
        assert isinstance(train_sample["label"], int)
        assert 0 <= train_sample["label"] <= 9

    @pytest.mark.download
    def test_dataloader_batching(self):
        """Test DataLoader batching behavior."""
        dataset = spt.data.HFDataset(
            path="ylecun/mnist", split="test", transform=spt.data.transforms.ToImage()
        )

        # Test with different batch sizes
        for batch_size in [1, 10, 32]:
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Use 0 for testing
            )

            batch = next(iter(loader))
            assert batch["image"].shape[0] == batch_size
            assert batch["label"].shape[0] == batch_size

    def test_transform_application(self):
        """Test transform application on MNIST data."""
        # Create transform
        transform = spt.data.transforms.ToImage()

        # Create dataset with transform
        dataset = spt.data.HFDataset(
            path="ylecun/mnist",
            split="test[:10]",  # Small subset for testing
            transform=transform,
        )

        # Get a sample
        sample = dataset[0]

        # Verify transformed data
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].dtype == torch.float32
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    @pytest.mark.gpu
    def test_model_forward_pass(self):
        """Test model forward pass with MNIST data."""
        # Create model
        backbone = spt.backbone.Resnet9(num_classes=10, num_channels=1)

        # Create dummy MNIST batch
        batch = torch.randn(4, 1, 28, 28)

        # Forward pass
        output = backbone(batch)

        # Verify output
        assert output.shape == (4, 10)  # 4 samples, 10 classes

        # Test with CUDA if available
        if torch.cuda.is_available():
            backbone = backbone.cuda()
            batch = batch.cuda()
            output = backbone(batch)
            assert output.is_cuda
            assert output.shape == (4, 10)
