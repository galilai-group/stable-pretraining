"""Integration tests for dataset functionality."""

import minari
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torchvision.transforms import v2

import stable_pretraining as spt


@pytest.mark.integration
class TestDatasetIntegration:
    """Integration tests for datasets with actual data loading."""

    @pytest.mark.download
    def test_hf_datasets(self):
        """Test HuggingFace datasets loading and transformations."""
        # Test basic dataset loading
        dataset1 = spt.data.HFDataset("ylecun/mnist", split="train")

        # Test with transform
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        def t(x):
            x["image"] = transform(x["image"])
            return x

        dataset2 = spt.data.HFDataset("ylecun/mnist", split="train", transform=t)

        # Verify transform is applied correctly
        assert transform(dataset1[0]["image"]).eq(dataset2[0]["image"]).all()

        # Test with column renaming
        dataset3 = spt.data.HFDataset(
            "ylecun/mnist", split="train", rename_columns=dict(image="toto")
        )
        assert transform(dataset3[0]["toto"]).eq(dataset2[0]["image"]).all()

    @pytest.mark.download
    def test_hf_dataloaders(self):
        """Test HuggingFace datasets with DataLoader."""
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        def t(x):
            x["image"] = transform(x["image"])
            return x

        # Create dataset with transform
        dataset = spt.data.HFDataset("ylecun/mnist", split="train", transform=t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)

        # Test batch loading
        for x in loader:
            assert x["image"].shape == (4, 1, 28, 28)
            assert len(x["label"]) == 4
            break

    @pytest.mark.download
    @pytest.mark.slow
    def test_datamodule(self):
        """Test DataModule with full configuration."""
        # Configure train dataset
        train = OmegaConf.create(
            {
                "dataset": {
                    "_target_": "stable_pretraining.data.HFDataset",
                    "path": "ylecun/mnist",
                    "split": "train",
                },
                "batch_size": 20,
                "drop_last": True,
            }
        )

        # Configure test dataset with transform
        test = OmegaConf.create(
            {
                "dataset": {
                    "_target_": "stable_pretraining.data.HFDataset",
                    "path": "ylecun/mnist",
                    "split": "test",
                    "transform": {
                        "_target_": "stable_pretraining.data.transforms.ToImage",
                    },
                },
                "batch_size": 20,
            }
        )

        # Create DataModule
        module = spt.data.DataModule(train=train, test=test, val=test, predict=test)

        # Test data preparation and setup
        module.prepare_data()

        # Test train setup
        module.setup("fit")
        assert not torch.is_tensor(module.train_dataset[0]["image"])
        loader = module.train_dataloader()
        assert loader.drop_last

        # Test test setup
        module.setup("test")
        loader = module.test_dataloader()
        assert torch.is_tensor(module.test_dataset[0]["image"])
        assert not loader.drop_last

        # Test validation setup
        module.setup("validate")
        loader = module.val_dataloader()
        assert not loader.drop_last

        # Test predict setup
        module.setup("predict")
        loader = module.predict_dataloader()
        assert not loader.drop_last

    @pytest.mark.download
    def test_dataset_sizes(self):
        """Test dataset size and splits."""
        # Load train and test splits
        train_dataset = spt.data.HFDataset("ylecun/mnist", split="train")
        test_dataset = spt.data.HFDataset("ylecun/mnist", split="test")

        # MNIST has 60,000 train and 10,000 test samples
        assert len(train_dataset) == 60000
        assert len(test_dataset) == 10000

    @pytest.mark.download
    def test_batch_collation(self):
        """Test batch collation with different batch sizes."""
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        def t(x):
            x["image"] = transform(x["image"])
            return x

        dataset = spt.data.HFDataset("ylecun/mnist", split="train", transform=t)

        # Test different batch sizes
        for batch_size in [1, 16, 32]:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=0,  # Use 0 workers for testing
            )

            batch = next(iter(loader))
            assert batch["image"].shape[0] == batch_size
            assert len(batch["label"]) == batch_size

    @pytest.mark.v1
    def test_fromtensor_dataset(self):
        """Test FromTensorDataset transform logic."""
        mock_data = torch.randn(128, 3, 32, 32)
        trans = spt.data.transforms.ToImage(mean=(0.5,), std=(0.5,))

        # fake torch dataset
        dataset = torch.utils.data.TensorDataset(mock_data)
        data = spt.data.utils.FromTorchDataset(dataset, names=["image"])
        data_trans = spt.data.utils.FromTorchDataset(
            dataset, names=["image"], transform=trans
        )

        assert list(data[0].keys()) == ["image"]
        assert list(data_trans[0].keys()) == ["image"]

    @pytest.mark.download
    def test_minaristep_dataset_bounds(self):
        """Test bounds computation for MinariStepsDataset util."""
        # minari download D4RL/pen/human-v2
        minari_dataset = minari.load_dataset("D4RL/pen/human-v2", download=True)
        dataset = spt.data.MinariStepsDataset(minari_dataset, num_steps=2)

        assert np.all(dataset.bounds >= 0)
        assert len(dataset.bounds) == minari_dataset.total_episodes
        assert dataset.bounds[-1] < minari_dataset.total_steps
        assert len(dataset) < minari_dataset.total_steps

        for idx in range(16):
            batch = dataset[idx]
            assert len(batch["observations"]) == 2
            assert list(batch.keys()) == dataset.NAMES
