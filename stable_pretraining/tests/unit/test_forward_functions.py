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
        """Test simclr with transforms from simclr-resnet50.py benchmark."""
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
        result = forward_module.simclr(module, batch, "train")

        # Verify
        assert "embedding" in result
        assert "loss" in result
        assert "label" in result

    def test_simclr_forward_validation(self):
        """Test simclr with validation transform."""
        sample = {"image": _create_dummy_pil_image(), "label": torch.tensor(0)}

        # Apply val transform
        transform = _val_transform()
        batch = transform(sample)

        # Create module
        module = Mock()
        module.backbone = lambda x: torch.randn(x.shape[0], 512)
        module.training = False

        # Forward pass
        result = forward_module.simclr(module, batch, "val")

        assert "embedding" in result
        assert "label" in result
        assert "loss" not in result

    def test_byol_forward_with_benchmark_transforms(self):
        """Test byol with SimCLR-style transforms (2 views)."""
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

        result = forward_module.byol(module, batch, "train")

        assert "embedding" in result
        assert "loss" in result

    def test_vicreg_forward_with_benchmark_transforms(self):
        """Test vicreg with benchmark transforms."""
        sample = {"image": _create_dummy_pil_image()}

        transform = _simclr_transforms()
        batch = transform(sample)

        module = Mock()
        module.backbone = lambda x: torch.randn(x.shape[0], 512)
        module.projector = lambda x: torch.randn(x.shape[0], 8192)
        module.vicreg_loss = Mock(return_value=torch.tensor(0.4))
        module.training = True
        module.log = Mock()

        result = forward_module.vicreg(module, batch, "train")

        assert "embedding" in result
        assert "loss" in result

    def test_barlow_twins_forward_with_benchmark_transforms(self):
        """Test barlow_twins with benchmark transforms."""
        sample = {"image": _create_dummy_pil_image()}

        transform = _simclr_transforms()
        batch = transform(sample)

        module = Mock()
        module.backbone = lambda x: torch.randn(x.shape[0], 512)
        module.projector = lambda x: torch.randn(x.shape[0], 8192)
        module.barlow_loss = Mock(return_value=torch.tensor(0.6))
        module.training = True
        module.log = Mock()

        result = forward_module.barlow_twins(module, batch, "train")

        assert "embedding" in result
        assert "loss" in result

    def test_swav_forward_with_benchmark_transforms(self):
        """Test swav with benchmark transforms."""
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

        result = forward_module.swav(module, batch, "train")

        assert "embedding" in result
        assert "loss" in result

    def test_nnclr_forward_with_benchmark_transforms(self):
        """Test nnclr with benchmark transforms."""
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

        result = forward_module.nnclr(module, batch, "train")

        assert "embedding" in result
        assert "loss" in result

    def test_dino_forward_with_benchmark_transforms(self):
        """Test dino with transforms from dino-resnet18.py benchmark."""
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
        result = forward_module.dino(module, batch, "train")

        # Verify
        assert "embedding" in result
        assert "loss" in result

    def test_dino_forward_validation(self):
        """Test dino with validation transform."""
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

        result = forward_module.dino(module, batch, "val")

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
            forward_module.dino(module, batch_with_dict_views, "train")

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
            forward_module.dino(module, batch_with_list_views, "train")

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
            forward_module.dinov2(module, batch_with_list_views, "train")

    def test_supervised_forward(self):
        """Test supervised with validation transform."""
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

        result = forward_module.supervised(module, batch, "train")

        assert "embedding" in result
        assert "logits" in result
        assert "loss" in result


# ---------------------------------------------------------------------------
# EquiMod: three-view forward function
# ---------------------------------------------------------------------------


def _equimod_transforms():
    """[un-augmented, augmented, augmented] CIFAR-style pipelines."""

    def augmented():
        return transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        )

    original = transforms.Compose(
        transforms.RGB(), transforms.ToImage(**spt.data.static.CIFAR10)
    )
    return transforms.MultiViewTransform([original, augmented(), augmented()])


def _equimod_batch(n: int = 4):
    """Collated three-view training batch, as the DataLoader delivers it."""
    torch.manual_seed(0)
    transform = _equimod_transforms()
    samples = [
        transform({"image": _create_dummy_pil_image((32, 32)), "label": i % 10})
        for i in range(n)
    ]
    return torch.utils.data.default_collate(samples)


def _equimod_module(embed_dim: int = 16):
    """A real EquiMod so forward.equimod and EquiMod.forward can be compared."""
    from stable_pretraining.methods import EquiMod

    torch.manual_seed(0)
    encoder = nn.Sequential(
        nn.Conv2d(3, embed_dim, kernel_size=3, stride=2),
        nn.BatchNorm2d(embed_dim),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return EquiMod(
        encoder_name=encoder,
        projector_dims=(32, 8),
        equi_projector_dims=(32, 8),
        param_projector_dims=(8,),
    )


@pytest.mark.unit
class TestEquiModForward:
    """forward.equimod on a composable Module and on the EquiMod class."""

    def test_training_batch_with_benchmark_transforms(self):
        from stable_pretraining.methods.equimod import EquivarianceModule

        batch = _equimod_batch(n=4)
        module = Mock()
        module.backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 16))
        module.projector = nn.Linear(16, 8)
        module.simclr_loss = spt.losses.NTXEntLoss(temperature=0.5)
        module.equi = EquivarianceModule(
            16, projector_dims=(16, 8), param_projector_dims=(8,)
        )
        module.equi_loss = spt.losses.NTXEntLoss(temperature=0.2)
        module.training = True
        module.log = Mock()

        result = forward_module.equimod(module, batch, "fit")

        assert result["embedding"].shape == (8, 16)  # two augmented views
        _, v1, v2 = batch["views"]
        assert torch.equal(result["label"], torch.cat([v1["label"], v2["label"]]))
        assert torch.isfinite(result["loss"])
        logged = {call.args[0] for call in module.log.call_args_list}
        assert logged == {"fit/inv", "fit/equi", "fit/equi_gain", "fit/loss"}

    def test_loss_weight_and_param_keys_are_honoured(self):
        from stable_pretraining.methods.equimod import EquivarianceModule

        def augmented_with_blur():
            return transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.PILGaussianBlur(p=0.5),  # records the 16th/17th entries
                transforms.ToImage(**spt.data.static.CIFAR10),
            )

        torch.manual_seed(0)
        original = transforms.Compose(
            transforms.RGB(), transforms.ToImage(**spt.data.static.CIFAR10)
        )
        transform = transforms.MultiViewTransform(
            [original, augmented_with_blur(), augmented_with_blur()]
        )
        samples = [
            transform({"image": _create_dummy_pil_image((32, 32)), "label": i})
            for i in range(4)
        ]
        batch = torch.utils.data.default_collate(samples)

        module = Mock()
        module.backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 16))
        module.projector = nn.Linear(16, 8)
        module.simclr_loss = spt.losses.NTXEntLoss(temperature=0.5)
        module.equi = EquivarianceModule(
            16,
            projector_dims=(16, 8),
            param_dim=17,
            param_projector_dims=(8,),
            weight=0.5,
            param_keys={"blur": "PILGaussianBlur"},
        )
        module.equi_loss = spt.losses.NTXEntLoss(temperature=0.2)
        module.training = True
        module.log = Mock()

        result = forward_module.equimod(module, batch, "fit")

        logged = {call.args[0]: call.args[1] for call in module.log.call_args_list}
        assert torch.allclose(
            result["loss"], logged["fit/inv"] + 0.5 * logged["fit/equi"], atol=1e-6
        )

    def test_matches_method_class_forward(self):
        model = _equimod_module()
        model.train()
        model.log = Mock()
        batch = _equimod_batch(n=4)
        original, v1, v2 = batch["views"]

        torch.manual_seed(1)
        via_forward = forward_module.equimod(model, batch, "fit")
        torch.manual_seed(1)
        via_class = model(
            v1["image"],
            v2["image"],
            original["image"],
            model.augmentation_params(v1),
            model.augmentation_params(v2),
        )
        assert torch.allclose(via_forward["loss"], via_class.loss, atol=1e-6)
        assert torch.allclose(via_forward["embedding"], via_class.embedding, atol=1e-6)

    def test_eval_mode_skips_loss(self):
        model = _equimod_module()
        model.eval()
        batch = _equimod_batch(n=4)
        result = forward_module.equimod(model, batch, "fit")
        assert "loss" not in result
        assert result["embedding"].shape == (8, 16)

    def test_validation_single_view(self):
        model = _equimod_module()
        model.eval()
        batch = {"image": torch.randn(4, 3, 32, 32), "label": torch.arange(4)}
        result = forward_module.equimod(model, batch, "validate")
        assert result["embedding"].shape == (4, 16)
        assert torch.equal(result["label"], torch.arange(4))
        assert "loss" not in result

    def test_named_views_dict_is_accepted(self):
        model = _equimod_module()
        model.train()
        model.log = Mock()
        batch = _equimod_batch(n=4)
        original, v1, v2 = batch["views"]
        named = {"views": {"original": original, "view_a": v1, "view_b": v2}}
        torch.manual_seed(1)
        from_named = forward_module.equimod(model, named, "fit")
        torch.manual_seed(1)
        from_list = forward_module.equimod(model, batch, "fit")
        assert torch.allclose(from_named["loss"], from_list["loss"], atol=1e-6)

    def test_two_views_is_rejected(self):
        model = _equimod_module()
        model.train()
        batch = _equimod_batch(n=4)
        with pytest.raises(ValueError, match="exactly 3 views"):
            forward_module.equimod(model, {"views": batch["views"][1:]}, "fit")
