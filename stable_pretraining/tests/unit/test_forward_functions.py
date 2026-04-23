"""Unit tests for all forward functions using benchmark transforms."""

import pytest
import torch
import torch.nn as nn
from PIL import Image
from unittest.mock import Mock

import stable_pretraining as spt
from stable_pretraining import forward as forward_module
from stable_pretraining.data import transforms


def _create_dummy_pil_image(size=(224, 224)):
    """Create a dummy PIL image for testing."""
    return Image.new("RGB", size, color=(128, 128, 128))


def _simclr_transforms():
    return transforms.MultiViewTransform(
        [
            transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.PILGaussianBlur(p=1.0),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(**spt.data.static.ImageNet),
            ),
            transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.PILGaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=0.5, p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(**spt.data.static.ImageNet),
            ),
        ]
    )


def _dino_transforms():
    return transforms.MultiViewTransform(
        {
            "global_1": transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((224, 224), scale=(0.4, 1.0)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.PILGaussianBlur(p=1.0),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(**spt.data.static.ImageNet),
            ),
            "global_2": transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((224, 224), scale=(0.4, 1.0)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.PILGaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=0.5, p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(**spt.data.static.ImageNet),
            ),
            "local_1": transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.PILGaussianBlur(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(**spt.data.static.ImageNet),
            ),
            "local_2": transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.PILGaussianBlur(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(**spt.data.static.ImageNet),
            ),
        }
    )


def _val_transform():
    """Validation transform used across benchmarks."""
    return transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


@pytest.mark.unit
class TestForwardFunctionsWithBenchmarkTransforms:
    """Test forward functions with actual benchmark transforms."""

    def test_simclr_forward_with_benchmark_transforms(self):
        """Test simclr_forward with transforms from simclr-resnet50.py benchmark."""
        # Create sample
        sample = {"image": _create_dummy_pil_image(), "label": torch.tensor([0])}

        # Apply transform
        transform = _simclr_transforms()
        batch = transform(sample)

        # Create minimal module
        module = Mock()
        module.backbone = lambda x: torch.randn(x.shape[0], 512)
        module.projector = lambda x: torch.randn(x.shape[0], 128)
        module.simclr_loss = Mock(return_value=torch.tensor(0.5))
        module.training = True
        module.log = Mock()

        # Forward pass
        result = forward_module.simclr_forward(module, batch, "train")

        # Verify
        assert "embedding" in result
        assert "loss" in result
        assert "label" in result

    def test_simclr_forward_validation(self):
        """Test simclr_forward with validation transform."""
        sample = {"image": _create_dummy_pil_image(), "label": torch.tensor(0)}

        # Apply val transform
        transform = _val_transform()
        batch = transform(sample)

        # Create module
        module = Mock()
        module.backbone = lambda x: torch.randn(x.shape[0], 512)
        module.training = False

        # Forward pass
        result = forward_module.simclr_forward(module, batch, "val")

        assert "embedding" in result
        assert "label" in result
        assert "loss" not in result

    def test_byol_forward_with_benchmark_transforms(self):
        """Test byol_forward with SimCLR-style transforms (2 views)."""
        sample = {"image": _create_dummy_pil_image(), "label": torch.tensor([0])}

        transform = _simclr_transforms()
        batch = transform(sample)

        # Create module with teacher-student wrappers
        module = Mock()
        module.backbone = Mock()
        module.backbone.forward_student = lambda x: torch.randn(x.shape[0], 512)
        module.backbone.forward_teacher = lambda x: torch.randn(x.shape[0], 512)
        module.projector = Mock()
        module.projector.forward_student = lambda x: torch.randn(x.shape[0], 128)
        module.projector.forward_teacher = lambda x: torch.randn(x.shape[0], 128)
        module.predictor = lambda x: torch.randn(x.shape[0], 128)
        module.byol_loss = Mock(return_value=torch.tensor(0.3))
        module.training = True
        module.log = Mock()

        result = forward_module.byol_forward(module, batch, "train")

        assert "embedding" in result
        assert "loss" in result

    def test_vicreg_forward_with_benchmark_transforms(self):
        """Test vicreg_forward with benchmark transforms."""
        sample = {"image": _create_dummy_pil_image()}

        transform = _simclr_transforms()
        batch = transform(sample)

        module = Mock()
        module.backbone = lambda x: torch.randn(x.shape[0], 512)
        module.projector = lambda x: torch.randn(x.shape[0], 8192)
        module.vicreg_loss = Mock(return_value=torch.tensor(0.4))
        module.training = True
        module.log = Mock()

        result = forward_module.vicreg_forward(module, batch, "train")

        assert "embedding" in result
        assert "loss" in result

    def test_barlow_twins_forward_with_benchmark_transforms(self):
        """Test barlow_twins_forward with benchmark transforms."""
        sample = {"image": _create_dummy_pil_image()}

        transform = _simclr_transforms()
        batch = transform(sample)

        module = Mock()
        module.backbone = lambda x: torch.randn(x.shape[0], 512)
        module.projector = lambda x: torch.randn(x.shape[0], 8192)
        module.barlow_loss = Mock(return_value=torch.tensor(0.6))
        module.training = True
        module.log = Mock()

        result = forward_module.barlow_twins_forward(module, batch, "train")

        assert "embedding" in result
        assert "loss" in result

    def test_swav_forward_with_benchmark_transforms(self):
        """Test swav_forward with benchmark transforms."""
        sample = {"image": _create_dummy_pil_image()}

        transform = _simclr_transforms()
        batch = transform(sample)

        module = Mock()
        module.backbone = lambda x: torch.randn(x.shape[0], 512)
        module.projector = lambda x: torch.randn(x.shape[0], 128)
        module.prototypes = nn.Linear(128, 3000, bias=False)
        module.swav_loss = Mock(return_value=torch.tensor(0.7))
        module.training = True
        module.log = Mock()
        module.use_queue = False

        result = forward_module.swav_forward(module, batch, "train")

        assert "embedding" in result
        assert "loss" in result

    def test_nnclr_forward_with_benchmark_transforms(self):
        """Test nnclr_forward with benchmark transforms."""
        sample = {"image": _create_dummy_pil_image()}

        transform = _simclr_transforms()
        batch = transform(sample)

        module = Mock()
        module.backbone = lambda x: torch.randn(x.shape[0], 512)
        module.projector = lambda x: torch.randn(x.shape[0], 256)
        module.predictor = lambda x: torch.randn(x.shape[0], 256)
        module.nnclr_loss = Mock(return_value=torch.tensor(0.5))
        module.training = True
        module.log = Mock()
        module.hparams = Mock()
        module.hparams.support_set_size = 1000
        module.hparams.projection_dim = 256
        module.trainer = Mock()

        # Mock queue callback with empty queue
        module._nnclr_queue_callback = Mock()
        module._nnclr_queue_callback.key = "nnclr_support_set"
        spt.callbacks.queue.OnlineQueue._shared_queues = {
            "nnclr_support_set": Mock(get=Mock(return_value=torch.empty(0, 256)))
        }

        result = forward_module.nnclr_forward(module, batch, "train")

        assert "embedding" in result
        assert "loss" in result

    def test_dino_forward_with_benchmark_transforms(self):
        """Test dino_forward with transforms from dino-resnet18.py benchmark."""
        sample = {"image": _create_dummy_pil_image()}

        # Apply DINO multi-crop transform
        transform = _dino_transforms()
        batch = transform(sample)

        # Verify batch structure
        assert "global_1" in batch
        assert "global_2" in batch
        assert "local_1" in batch
        assert "local_2" in batch

        # Add batch dimension to all images
        for key in batch:
            if "image" in batch[key]:
                batch[key]["image"] = batch[key]["image"].unsqueeze(0)

        # Create module with teacher-student wrappers
        module = Mock()

        # Mock backbone that returns ViT-style output
        class MockViTOutput:
            def __init__(self, batch_size):
                self.last_hidden_state = torch.randn(batch_size, 197, 192)

            def __getitem__(self, idx):
                return self.last_hidden_state[idx]

        def mock_vit_forward(images, interpolate_pos_encoding=False):
            batch_size = images.shape[0]
            return MockViTOutput(batch_size)

        module.backbone = Mock()
        module.backbone.forward_teacher = mock_vit_forward
        module.backbone.forward_student = mock_vit_forward

        # Mock projector
        module.projector = Mock()
        module.projector.forward_teacher = lambda x: torch.randn(x.shape[0], 65536)
        module.projector.forward_student = lambda x: torch.randn(x.shape[0], 65536)

        # Mock loss
        module.dino_loss = Mock()
        module.dino_loss.softmax_center_teacher = Mock(
            return_value=torch.randn(2, 1, 65536)
        )
        module.dino_loss.return_value = torch.tensor(0.8)
        module.dino_loss.update_center = Mock()

        module.training = True
        module.log = Mock()
        module.current_epoch = 10
        module.temperature_teacher = 0.07
        module.warmup_epochs_temperature_teacher = 50
        module.warmup_temperature_teacher = 0.04

        # Forward pass
        result = forward_module.dino_forward(module, batch, "train")

        # Verify
        assert "embedding" in result
        assert "loss" in result

    def test_dino_forward_validation(self):
        """Test dino_forward with validation transform."""
        sample = {"image": _create_dummy_pil_image()}

        transform = _val_transform()
        batch = transform(sample)

        # Add batch dimension
        batch["image"] = batch["image"].unsqueeze(0)

        # Create module
        class MockViTOutput:
            def __init__(self, batch_size):
                self.last_hidden_state = torch.randn(batch_size, 197, 192)

            def __getitem__(self, idx):
                return self.last_hidden_state[idx]

        def mock_vit_forward(images):
            return MockViTOutput(images.shape[0])

        module = Mock()
        module.backbone = Mock()
        module.backbone.forward_teacher = mock_vit_forward
        module.training = False

        result = forward_module.dino_forward(module, batch, "val")

        assert "embedding" in result
        assert "loss" not in result

    def test_dino_forward_raises_on_missing_global_keys(self):
        """Test that DINO raises error when dict lacks 'global' in keys."""
        module = Mock()
        module.training = True

        # Test: Dict views without 'global' in keys
        batch_with_dict_views = {
            "views": {
                "view1": {"image": torch.randn(1, 3, 224, 224)},
                "view2": {"image": torch.randn(1, 3, 224, 224)},
            }
        }

        with pytest.raises(ValueError):
            forward_module.dino_forward(module, batch_with_dict_views, "train")

    def test_dino_forward_raises_on_list_views(self):
        """Test that DINO raises error when given list of views (implicit assumption removed)."""
        module = Mock()
        module.training = True

        # Test: List of views (should raise error requiring explicit dict format)
        batch_with_list_views = {
            "views": [
                {"image": torch.randn(1, 3, 224, 224)},
                {"image": torch.randn(1, 3, 224, 224)},
                {"image": torch.randn(1, 3, 96, 96)},
                {"image": torch.randn(1, 3, 96, 96)},
            ]
        }

        with pytest.raises(ValueError):
            forward_module.dino_forward(module, batch_with_list_views, "train")

    def test_dinov2_forward_raises_on_list_views(self):
        """Test that DINOv2 raises error when given list of views."""
        module = Mock()
        module.training = True

        batch_with_list_views = {
            "views": [
                {"image": torch.randn(1, 3, 224, 224)},
                {"image": torch.randn(1, 3, 224, 224)},
            ]
        }

        with pytest.raises(ValueError):
            forward_module.dinov2_forward(module, batch_with_list_views, "train")

    def test_supervised_forward(self):
        """Test supervised_forward with validation transform."""
        sample = {"image": _create_dummy_pil_image(), "label": torch.tensor(0)}

        transform = _val_transform()
        batch = transform(sample)

        # Add batch dimension to image
        batch["image"] = batch["image"].unsqueeze(0)
        batch["label"] = batch["label"].unsqueeze(0)

        module = Mock()
        module.backbone = lambda x: torch.randn(x.shape[0], 512)
        module.classifier = nn.Linear(512, 10)
        module.supervised_loss = nn.CrossEntropyLoss()
        module.log = Mock()

        result = forward_module.supervised_forward(module, batch, "train")

        assert "embedding" in result
        assert "logits" in result
        assert "loss" in result
