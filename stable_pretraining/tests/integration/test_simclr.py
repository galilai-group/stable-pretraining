"""Integration tests for SimCLR functionality."""

import lightning as pl
import pytest
import torch
import torchmetrics
from transformers import AutoConfig, AutoModelForImageClassification

import stable_pretraining as spt
from stable_pretraining.data import transforms


@pytest.mark.integration
class TestSimCLRIntegration:
    """Integration tests for SimCLR with actual training."""

    @pytest.mark.v1
    @pytest.mark.gpu
    @pytest.mark.download
    @pytest.mark.slow
    def test_simclr_with_probing(self):
        """Test SimCLR training with online probing."""
        # Define transforms with SimCLR augmentations
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
                split="validation",
                transform=val_transform,
            ),
            batch_size=128,
            num_workers=10,
        )

        data = spt.data.DataModule(train=train, val=val)

        # Define SimCLR forward function
        def forward(self, batch, stage):
            batch["embedding"] = self.backbone(batch["image"])["logits"]
            if self.training:
                proj = self.projector(batch["embedding"])
                views = spt.data.fold_views(proj, batch["sample_idx"])
                batch["loss"] = self.simclr_loss(views[0], views[1])
            return batch

        # Create backbone and projector
        config = AutoConfig.from_pretrained("microsoft/resnet-18")
        backbone = AutoModelForImageClassification.from_config(config)
        backbone.classifier[1] = torch.nn.Identity()
        projector = torch.nn.Linear(512, 128)

        # Create module
        module = spt.Module(
            backbone=backbone,
            projector=projector,
            forward=forward,
            simclr_loss=spt.losses.NTXEntLoss(temperature=0.1),
        )

        # Create online probes
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

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=6,
            num_sanity_val_steps=1,
            callbacks=[linear_probe, knn_probe],
            precision="16-mixed",
            logger=False,
            enable_checkpointing=False,
        )

        # Run training
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()

    @pytest.mark.v1
    @pytest.mark.gpu
    def test_simclr_loss_computation(self):
        """Test SimCLR NT-Xent loss computation."""
        batch_size = 16
        feature_dim = 128

        # Create projector and loss
        projector = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, feature_dim),
        )

        simclr_loss = spt.losses.NTXEntLoss(temperature=0.1)

        # Create dummy features
        features = torch.randn(batch_size, 512)

        # Project features
        z = projector(features)

        # Split into two views
        z1 = z[: batch_size // 2]
        z2 = z[batch_size // 2 :]

        # Compute loss
        loss = simclr_loss(z1, z2)

        # Verify loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_multi_view_data_loading(self):
        """Test multi-view data loading for SimCLR."""
        # Create dummy dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 224, 224), torch.randint(0, 10, (100,))
        )

        # Create multi-view sampler
        sampler = spt.data.sampler.RepeatedRandomSampler(dataset, n_views=2)

        # Create dataloader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            sampler=sampler,
        )

        # Get a batch
        images, labels = next(iter(loader))

        # Verify multi-view structure
        assert images.shape[0] == 8  # batch_size
        # Sample indices should have repeats for multi-view

    def test_fold_views_operation(self):
        """Test fold_views operation for multi-view data."""
        batch_size = 16
        n_views = 2
        feature_dim = 128

        # Create multi-view features
        features = torch.randn(batch_size, feature_dim)
        sample_idx = torch.repeat_interleave(
            torch.arange(batch_size // n_views), n_views
        )

        # Fold views
        views = spt.data.fold_views(features, sample_idx)

        # Verify views
        assert len(views) == n_views
        assert all(v.shape == (batch_size // n_views, feature_dim) for v in views)

    @pytest.mark.gpu
    def test_projector_architecture(self):
        """Test different projector architectures for SimCLR."""
        input_dim = 512
        hidden_dim = 256
        output_dim = 128

        # Linear projector
        linear_proj = torch.nn.Linear(input_dim, output_dim)

        # MLP projector (standard for SimCLR)
        mlp_proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        # Test forward pass
        x = torch.randn(32, input_dim)

        linear_out = linear_proj(x)
        mlp_out = mlp_proj(x)

        assert linear_out.shape == (32, output_dim)
        assert mlp_out.shape == (32, output_dim)

    def test_temperature_scaling(self):
        """Test temperature scaling in SimCLR loss."""
        feature_dim = 128

        # Create normalized features
        z1 = torch.nn.functional.normalize(torch.randn(8, feature_dim), dim=1)
        z2 = torch.nn.functional.normalize(torch.randn(8, feature_dim), dim=1)

        # Test different temperatures
        for temp in [0.05, 0.1, 0.5, 1.0]:
            loss_fn = spt.losses.NTXEntLoss(temperature=temp)
            loss = loss_fn(z1, z2)
            assert loss.item() > 0

    @pytest.mark.v1
    @pytest.mark.download
    def test_simclr_augmentations(self):
        """Test SimCLR data augmentations."""
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

        # Load a small dataset
        dataset = spt.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="train[:10]",
            transform=augment,
        )

        # Get augmented samples
        sample1 = dataset[0]
        sample2 = dataset[0]  # Same image, different augmentation

        # Verify augmentations produce different results
        assert sample1["image"].shape == sample2["image"].shape
        assert not torch.allclose(sample1["image"], sample2["image"])

    @pytest.mark.v1
    @pytest.mark.gpu
    def test_simclr_training_step(self):
        """Test a single SimCLR training step."""
        # Create simple setup
        backbone = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(3 * 224 * 224, 512)
        )
        projector = torch.nn.Linear(512, 128)

        def forward(self, batch, stage):
            # Simplified forward for testing
            features = self.backbone(batch["image"])
            batch["embedding"] = features
            if self.training:
                proj = self.projector(features)
                # Simulate two views
                half = proj.shape[0] // 2
                batch["loss"] = torch.nn.functional.mse_loss(proj[:half], proj[half:])
            return batch

        module = spt.Module(
            backbone=backbone,
            projector=projector,
            forward=forward,
        )
        module.train()

        # Create dummy batch
        batch = {
            "image": torch.randn(8, 3, 224, 224),
            "label": torch.randint(0, 10, (8,)),
            "sample_idx": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        }

        # Run training step
        output = module.training_step(batch, 0)

        # Verify output
        assert "loss" in output
        assert output["loss"].requires_grad
