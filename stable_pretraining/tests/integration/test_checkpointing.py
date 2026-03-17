"""Integration tests for checkpointing functionality."""

import pytest


@pytest.mark.integration
class TestCheckpointingIntegration:
    """Integration tests for checkpointing with actual training and file I/O."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for checkpoint files."""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.v1
    def test_sklearn_module_checkpointing(self, temp_dir):
        """Test sklearn module checkpointing with SklearnCheckpoint callback."""
        import stable_pretraining as ssl
        import lightning as pl
        import torchvision
        import sklearn.tree
        import wandb
        import torch
        from torch import nn
        from pathlib import Path

        wandb.init(mode="disabled")

        # Create dummy dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 32, 32), torch.randint(low=0, high=10, size=(100,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        # Define forward function
        def forward(self, batch):
            return 0

        # Create module with sklearn component
        module = ssl.module.Module(
            backbone=ssl.backbone.Resnet9(num_classes=10, num_channels=1),
            linear_probe=nn.Linear(10, 10),
            nonlinear_probe=torchvision.ops.MLP(
                in_channels=10,
                hidden_channels=[2048, 2048, 10],
                norm_layer=torch.nn.BatchNorm1d,
            ),
            tree=sklearn.tree.DecisionTreeRegressor(),
            G_scaling=1,
            imbalance_factor=0,
            batch_size=512,
            lr=1,
            forward=forward,
        )

        # Create trainer with SklearnCheckpoint callback
        trainer = pl.Trainer(
            max_epochs=0, accelerator="cpu", enable_checkpointing=False, logger=False
        )

        # Train and save checkpoint
        trainer.fit(module, train_dataloaders=loader)
        module.tree.test = 3

        checkpoint_path = Path(temp_dir) / "test.ckpt"
        trainer.save_checkpoint(str(checkpoint_path))

        # Change attribute and load checkpoint
        module.tree.test = 5
        trainer.fit(module, train_dataloaders=loader, ckpt_path=str(checkpoint_path))

        # With SklearnCheckpoint callback, sklearn attributes are restored
        assert module.tree.test == 3

        # Test loading into a new module without tree
        module2 = ssl.module.Module(
            backbone=ssl.backbone.Resnet9(num_classes=10, num_channels=1),
            linear_probe=nn.Linear(10, 10),
            nonlinear_probe=torchvision.ops.MLP(
                in_channels=10,
                hidden_channels=[2048, 2048, 10],
                norm_layer=torch.nn.BatchNorm1d,
            ),
            G_scaling=1,
            imbalance_factor=0,
            batch_size=512,
            lr=1,
            forward=forward,
        )

        trainer2 = pl.Trainer(
            max_epochs=0, accelerator="cpu", enable_checkpointing=False, logger=False
        )

        trainer2.fit(module2, train_dataloaders=loader, ckpt_path=str(checkpoint_path))
        assert module2.tree.test == 3
