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


def _poly_view_transforms(m):
    """M-view augmentation in the style of the SimCLR benchmark transforms."""
    return transforms.MultiViewTransform(
        [
            transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(**spt.data.static.ImageNet),
            )
            for _ in range(m)
        ]
    )


def _val_transform():
    """Validation transform used across benchmarks."""
    return transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def _encoded_views(m, b, as_dict=False):
    """M views of B samples whose pixels all equal ``10 * sample + view``.

    Labels are ``100 + sample``, so every embedding row can be traced back to
    its sample and view and checked against its label.
    """
    views = [
        {
            "image": (10.0 * torch.arange(b) + v).view(b, 1, 1, 1).repeat(1, 3, 4, 4),
            "label": 100 + torch.arange(b),
        }
        for v in range(m)
    ]
    if as_dict:
        return {f"view_{v}": view for v, view in enumerate(views)}
    return {"views": views}


def _poly_view_mock_module(training=True):
    """Mock module whose backbone returns each image's pixel value as ``[B, 2]``."""
    module = Mock()
    module.backbone = lambda x: x.flatten(1)[:, :2]
    module.projector = lambda x: torch.cat([x, x], dim=1)
    module.poly_view_loss = Mock(return_value=torch.tensor(0.5))
    module.training = training
    module.log = Mock()
    return module


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


@pytest.mark.unit
class TestPolyViewForward:
    """Tests for the ``poly_view`` forward function."""

    # --- Input: what the data pipeline produces ---

    def test_train_with_benchmark_transforms(self):
        sample = {"image": _create_dummy_pil_image(), "label": torch.tensor([0])}
        batch = _poly_view_transforms(4)(sample)
        module = _poly_view_mock_module()

        result = forward_module.poly_view(module, batch, "train")

        assert {"embedding", "loss", "label"} <= result.keys()
        module.poly_view_loss.assert_called_once()
        assert len(module.poly_view_loss.call_args.args[0]) == 4

    def test_fewer_than_two_views_raises(self):
        module = _poly_view_mock_module()

        with pytest.raises(ValueError, match="at least 2 views"):
            forward_module.poly_view(module, _encoded_views(1, 3), "train")

    # --- Loss: how the loss is computed through the forward function ---

    @pytest.mark.parametrize("m", [3, 8])
    def test_all_views_reach_loss(self, m):
        module = _poly_view_mock_module()

        forward_module.poly_view(module, _encoded_views(m, 2), "train")

        assert len(module.poly_view_loss.call_args.args[0]) == m

    @pytest.mark.parametrize(
        "loss",
        [
            pytest.param(
                spt.losses.PVCLoss(0.5, aggregation="geometric"), id="geometric"
            ),
            pytest.param(
                spt.losses.PVCLoss(0.5, aggregation="arithmetic"), id="arithmetic"
            ),
            pytest.param(spt.losses.SufficientStatisticsLoss(0.5), id="suff_stats"),
        ],
    )
    def test_two_views_match_simclr(self, loss):
        torch.manual_seed(0)
        module = Mock()
        module.backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 16))
        module.projector = nn.Linear(16, 8)
        module.poly_view_loss = loss
        module.simclr_loss = spt.losses.NTXEntLoss(temperature=0.5)
        module.training = True
        module.log = Mock()
        batch = {
            "views": [
                {"image": torch.randn(6, 3, 4, 4), "label": torch.arange(6)}
                for _ in range(2)
            ]
        }

        poly = forward_module.poly_view(module, batch, "train")
        simclr = forward_module.simclr(module, batch, "train")

        torch.testing.assert_close(poly["loss"], simclr["loss"])
        torch.testing.assert_close(poly["embedding"], simclr["embedding"])

    # --- Output: what callbacks receive ---

    def test_train_output_shapes(self):
        m, b = 4, 3
        module = _poly_view_mock_module()

        result = forward_module.poly_view(module, _encoded_views(m, b), "train")

        assert result["embedding"].shape == (m * b, 2)
        assert result["label"].shape == (m * b,)
        projections = module.poly_view_loss.call_args.args[0]
        assert [p.shape for p in projections] == [(b, 4)] * m

    @pytest.mark.parametrize("as_dict", [False, True], ids=["list", "dict"])
    def test_labels_align_with_embeddings(self, as_dict):
        # OnlineProbe pairs embedding[r] with label[r]; a misordering would
        # silently degrade probe accuracy instead of raising.
        m, b = 4, 3
        module = _poly_view_mock_module()

        batch = _encoded_views(m, b, as_dict)
        result = forward_module.poly_view(module, batch, "train")

        code = result["embedding"][:, 0].long()
        sample, view = code // 10, code % 10
        assert torch.equal(result["label"], 100 + sample)
        rows = torch.arange(m * b)
        assert torch.equal(sample, rows % b), "embeddings must be view-major"
        assert torch.equal(view, rows // b), "embeddings must be view-major"

    def test_logs_train_loss(self):
        module = _poly_view_mock_module()

        result = forward_module.poly_view(module, _encoded_views(4, 3), "train")

        module.log.assert_called_once()
        assert module.log.call_args.args[0] == "train/loss"
        assert module.log.call_args.args[1] is result["loss"]

    # --- Modes: training vs evaluation ---

    def test_validation_single_view(self):
        sample = {"image": _create_dummy_pil_image(), "label": torch.tensor(0)}
        batch = _val_transform()(sample)
        module = _poly_view_mock_module(training=False)

        result = forward_module.poly_view(module, batch, "val")

        assert "embedding" in result
        assert "label" in result
        assert "loss" not in result
        module.poly_view_loss.assert_not_called()

    def test_multiview_eval_mode_has_no_loss(self):
        module = _poly_view_mock_module(training=False)

        result = forward_module.poly_view(module, _encoded_views(4, 3), "val")

        assert result["embedding"].shape == (12, 2)
        assert "loss" not in result
        module.poly_view_loss.assert_not_called()
        module.log.assert_not_called()

    # --- Integration: inside a real spt.Module ---

    def test_through_spt_module(self):
        # A Mock answers any attribute name, so only a real Module checks that
        # the forward function reads the attributes users actually pass in.
        torch.manual_seed(0)
        module = spt.Module(
            forward=forward_module.poly_view,
            backbone=nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 16)),
            projector=nn.Linear(16, 8),
            poly_view_loss=spt.losses.PVCLoss(temperature=0.5),
        )
        module.train()
        batch = {
            "views": [
                {"image": torch.randn(4, 3, 4, 4), "label": torch.arange(4)}
                for _ in range(3)
            ]
        }

        result = module(batch, stage="fit")

        assert result["embedding"].shape == (12, 16)
        assert result["loss"].ndim == 0 and torch.isfinite(result["loss"])
        result["loss"].backward()
        assert module.backbone[1].weight.grad.abs().sum() > 0
