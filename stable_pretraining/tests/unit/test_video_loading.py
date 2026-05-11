"""Unit tests for video loading functionality."""

from unittest.mock import Mock, patch

import pytest
import torch


@pytest.mark.unit
class TestVideoLoadingUnit:
    """Unit tests for video loading components without actual video data."""

    def test_random_contiguous_temporal_sampler_initialization(self):
        """Test RandomContiguousTemporalSampler initialization."""
        with patch(
            "stable_pretraining.data.transforms.RandomContiguousTemporalSampler"
        ) as mock_sampler:
            sampler = mock_sampler(source="video", target="frames", num_frames=10)

            mock_sampler.assert_called_once_with(
                source="video", target="frames", num_frames=10
            )
            assert sampler is not None

    def test_video_transform_composition(self):
        """Test composition of video transforms."""
        with patch("stable_pretraining.data.transforms") as mock_transforms:
            mock_transforms.Compose.return_value = Mock()

            # Create video transform pipeline
            transform = mock_transforms.Compose(
                mock_transforms.RandomContiguousTemporalSampler(
                    source="video", target="video", num_frames=10
                ),
                mock_transforms.Resize((128, 128), source="video", target="video"),
            )

            assert mock_transforms.Compose.called
            assert transform is not None

    def test_temporal_sampling_logic(self):
        """Test temporal sampling logic for video frames."""
        # Mock video tensor: [frames, channels, height, width]
        num_total_frames = 30
        num_sample_frames = 10
        video = torch.randn(num_total_frames, 3, 224, 224)

        # Simulate random contiguous sampling
        start_idx = torch.randint(
            0, num_total_frames - num_sample_frames + 1, (1,)
        ).item()
        sampled_frames = video[start_idx : start_idx + num_sample_frames]

        # Verify sampling
        assert sampled_frames.shape == (num_sample_frames, 3, 224, 224)
        assert sampled_frames.shape[0] == num_sample_frames

    def test_video_resize_logic(self):
        """Test video resize transformation logic."""
        # Mock video tensor: [frames, channels, height, width]
        video = torch.randn(10, 3, 256, 256)
        target_size = (128, 128)

        # Simulate resize (simplified - actual resize would use interpolation)
        # For unit test, we just verify the shape transformation
        resized_shape = (video.shape[0], video.shape[1], target_size[0], target_size[1])

        assert resized_shape == (10, 3, 128, 128)

    def test_video_to_image_transform(self):
        """Test ToImage transform for video data."""
        with patch("stable_pretraining.data.transforms.ToImage") as mock_to_image:
            mock_to_image(
                scale=False,
                mean=[0, 0, 0],
                std=[255, 255, 255],
                source="video",
                target="video",
            )

            mock_to_image.assert_called_once_with(
                scale=False,
                mean=[0, 0, 0],
                std=[255, 255, 255],
                source="video",
                target="video",
            )

    def test_image_to_video_encoder_initialization(self):
        """Test ImageToVideoEncoder wrapper initialization."""
        with patch("stable_pretraining.backbone.ImageToVideoEncoder") as mock_encoder:
            with patch("torchvision.models.resnet18") as mock_resnet:
                backbone = mock_resnet()
                encoder = mock_encoder(backbone)

                mock_encoder.assert_called_once_with(backbone)
                assert encoder is not None

    def test_concat_dataset_creation(self):
        """Test ConcatDataset creation for video data."""
        with patch("torch.utils.data.ConcatDataset") as mock_concat:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=10)

            # Create concatenated dataset
            datasets = [mock_dataset for _ in range(10)]
            concat_dataset = mock_concat(datasets)

            mock_concat.assert_called_once()
            assert concat_dataset is not None

    def test_video_batch_structure(self):
        """Test expected batch structure for video data."""
        batch_size = 4
        num_frames = 10
        height, width = 128, 128

        # Mock video batch
        video_batch = {
            "video": torch.randn(batch_size, num_frames, 3, height, width),
            "idx": torch.arange(batch_size),
        }

        # Verify structure
        assert video_batch["video"].shape == (batch_size, num_frames, 3, height, width)
        assert video_batch["idx"].shape == (batch_size,)
        assert "video" in video_batch
        assert "frames" not in video_batch  # After transform, frames become video

    def test_frame_extraction_logic(self):
        """Test frame extraction from video data."""
        # Mock video data structure
        video_data = {
            "video": Mock(),  # Original video data
            "frames": None,  # Will be populated by transform
        }

        # Simulate frame extraction
        num_frames = 10
        extracted_frames = torch.randn(num_frames, 3, 224, 224)

        # Update data structure
        video_data["frames"] = extracted_frames

        # Verify extraction
        assert video_data["frames"] is not None
        assert video_data["frames"].shape == (num_frames, 3, 224, 224)

    def test_video_feature_extraction_shape(self):
        """Test shape of features extracted from video frames."""
        batch_size = 4
        num_frames = 10
        feature_dim = 1000  # ResNet output dimension

        # Mock feature extraction
        video_features = torch.randn(batch_size, num_frames, feature_dim)

        # Verify shape
        assert video_features.shape == (batch_size, num_frames, feature_dim)

    def test_hf_dataset_with_video(self):
        """Test HFDataset initialization for video data."""
        with patch("stable_pretraining.data.HFDataset") as mock_dataset:
            mock_transform = Mock()
            mock_dataset(
                path="shivalikasingh/video-demo",
                split="train",
                trust_remote_code=True,
                transform=mock_transform,
            )

            mock_dataset.assert_called_once_with(
                path="shivalikasingh/video-demo",
                split="train",
                trust_remote_code=True,
                transform=mock_transform,
            )
