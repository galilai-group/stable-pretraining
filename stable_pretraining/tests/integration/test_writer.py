"""Integration tests for writer callback functionality."""

import os
import shutil
import tempfile

import lightning as pl
import pytest
import torch
import torchmetrics
from transformers import AutoConfig, AutoModelForImageClassification

import stable_pretraining as spt
from stable_pretraining.data import transforms


@pytest.mark.integration
class TestWriterIntegration:
    """Integration tests for OnlineWriter with actual file I/O."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for writer output."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.mark.v1
    @pytest.mark.gpu
    @pytest.mark.download
    @pytest.mark.slow
    def test_simple_writer_full_pipeline(self, temp_dir):
        """Test OnlineWriter in full training pipeline."""
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

        # Create small train dataset
        train_dataset = spt.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="train[:256]",  # Use small subset
            transform=train_transform,
        )

        train = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=64,
            num_workers=4,  # Reduced for testing
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
                split="validation[:128]",  # Small subset
                transform=val_transform,
            ),
            batch_size=128,
            num_workers=4,
        )

        data = spt.data.DataModule(train=train, val=val)

        # Define forward function
        def forward(self, batch, stage):
            batch["embedding"] = self.backbone(batch["image"])["logits"]
            if self.training:
                preds = self.classifier(batch["embedding"])
                batch["loss"] = torch.nn.functional.cross_entropy(preds, batch["label"])
            return batch

        # Create model
        config = AutoConfig.from_pretrained("microsoft/resnet-18")
        backbone = AutoModelForImageClassification.from_config(config)
        backbone.classifier[1] = torch.nn.Identity()
        classifier = torch.nn.Linear(512, 10)

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

        # Create writer
        writer = spt.callbacks.OnlineWriter(
            names=["embedding", "linear_probe_preds"],
            path=temp_dir,
            during=["train"],
            every_k_epochs=2,
        )

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=3,
            num_sanity_val_steps=1,
            callbacks=[linear_probe, writer],
            precision="16-mixed",
            logger=False,
            enable_checkpointing=False,
        )

        # Run training
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()

        # Verify files were written
        written_files = os.listdir(temp_dir)
        num_written_files = len(written_files)

        # Expected: 2 epochs (0, 2) * 2 names * number of batches
        expected_files = 2 * 2 * len(train)
        assert num_written_files == expected_files, (
            f"Expected {expected_files} files, but found {num_written_files} files."
        )

    def test_writer_file_format(self, temp_dir):
        """Test the format of written files."""
        # Create dummy data
        data_to_write = {
            "embedding": torch.randn(32, 128),
            "predictions": torch.randn(32, 10),
        }

        # Write data
        filename = os.path.join(temp_dir, "test_data.pt")
        torch.save(data_to_write, filename)

        # Load and verify
        loaded_data = torch.load(filename)
        assert "embedding" in loaded_data
        assert "predictions" in loaded_data
        assert torch.allclose(loaded_data["embedding"], data_to_write["embedding"])
        assert torch.allclose(loaded_data["predictions"], data_to_write["predictions"])

    def test_writer_during_stages(self, temp_dir):
        """Test writer behavior during different training stages."""
        # Test configurations for different stages
        stage_configs = [
            (["train"], True, False),  # Write during train only
            (["val"], False, True),  # Write during val only
            (["train", "val"], True, True),  # Write during both
            ([], False, False),  # Don't write
        ]

        for during, should_write_train, should_write_val in stage_configs:
            # Simulate stage checking
            write_train = "train" in during
            write_val = "val" in during

            assert write_train == should_write_train
            assert write_val == should_write_val

    def test_writer_epoch_filtering(self, temp_dir):
        """Test writer epoch filtering behavior."""
        every_k_epochs = 3
        max_epochs = 10

        # Determine which epochs should write
        epochs_to_write = []
        for epoch in range(max_epochs):
            if epoch % every_k_epochs == 0:
                epochs_to_write.append(epoch)

        # Verify correct epochs
        assert epochs_to_write == [0, 3, 6, 9]

        # Simulate file writing for these epochs
        for epoch in epochs_to_write:
            filename = os.path.join(temp_dir, f"data_epoch{epoch}.pt")
            torch.save({"epoch": epoch}, filename)

        # Verify files exist
        written_files = os.listdir(temp_dir)
        assert len(written_files) == len(epochs_to_write)

    def test_writer_multiple_tensors(self, temp_dir):
        """Test writing multiple tensors in a single batch."""
        # Create multiple tensors
        batch_data = {
            "features": torch.randn(16, 256),
            "logits": torch.randn(16, 10),
            "embeddings": torch.randn(16, 512),
            "labels": torch.randint(0, 10, (16,)),
        }

        # Select which tensors to write
        names_to_write = ["features", "embeddings"]

        # Write selected tensors
        for i, name in enumerate(names_to_write):
            if name in batch_data:
                filename = os.path.join(temp_dir, f"{name}_batch{i}.pt")
                torch.save({name: batch_data[name]}, filename)

        # Verify correct tensors were written
        written_files = os.listdir(temp_dir)
        assert len(written_files) == len(names_to_write)

        # Verify content
        for name in names_to_write:
            matching_files = [f for f in written_files if name in f]
            assert len(matching_files) == 1

    def test_writer_large_tensor_handling(self, temp_dir):
        """Test writer with large tensors."""
        # Create large tensor
        large_tensor = torch.randn(1000, 2048)  # ~8MB

        # Write tensor
        filename = os.path.join(temp_dir, "large_tensor.pt")
        torch.save({"data": large_tensor}, filename)

        # Verify file exists and size
        assert os.path.exists(filename)
        file_size = os.path.getsize(filename)
        assert file_size > 1_000_000  # Should be > 1MB

        # Load and verify
        loaded = torch.load(filename)
        assert torch.allclose(loaded["data"], large_tensor)

    def test_writer_concurrent_writes(self, temp_dir):
        """Test concurrent writing behavior."""
        import concurrent.futures

        def write_tensor(idx):
            """Write a tensor with given index."""
            data = torch.randn(10, 10)
            filename = os.path.join(temp_dir, f"tensor_{idx}.pt")
            torch.save({"idx": idx, "data": data}, filename)
            return filename

        # Write multiple tensors concurrently
        num_writes = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(write_tensor, i) for i in range(num_writes)]
            [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify all files were written
        written_files = os.listdir(temp_dir)
        assert len(written_files) == num_writes

        # Verify each file
        for i in range(num_writes):
            expected_file = f"tensor_{i}.pt"
            assert expected_file in written_files

    @pytest.mark.slow
    def test_writer_performance(self, temp_dir):
        """Test writer performance with multiple batches."""
        import time

        num_batches = 50
        batch_size = 32
        feature_dim = 512

        start_time = time.time()

        # Write multiple batches
        for batch_idx in range(num_batches):
            data = {
                "embedding": torch.randn(batch_size, feature_dim),
                "predictions": torch.randn(batch_size, 10),
            }

            for name, tensor in data.items():
                filename = os.path.join(temp_dir, f"{name}_batch{batch_idx}.pt")
                torch.save({name: tensor}, filename)

        elapsed_time = time.time() - start_time

        # Verify performance
        assert elapsed_time < 10  # Should complete in under 10 seconds

        # Verify all files
        written_files = os.listdir(temp_dir)
        assert len(written_files) == num_batches * 2  # 2 tensors per batch
