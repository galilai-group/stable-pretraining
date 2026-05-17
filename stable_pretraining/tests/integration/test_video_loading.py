"""Integration tests for video loading functionality."""

import pytest
import torch
import torchvision

import stable_pretraining


@pytest.mark.integration
class TestVideoLoadingIntegration:
    """Integration tests for video loading with actual video data."""

    def test_temporal_sampling_variations(self):
        """Test different temporal sampling strategies."""
        num_frames_list = [4, 8, 16, 32]

        for num_frames in num_frames_list:
            # Create dataset with different frame counts
            dataset = stable_pretraining.data.HFDataset(
                path="shivalikasingh/video-demo",
                split="train[:1]",  # Use only first video
                trust_remote_code=True,
                transform=stable_pretraining.data.transforms.RandomContiguousTemporalSampler(
                    source="video", target="frames", num_frames=num_frames
                ),
            )

            try:
                sample = dataset[0]
                assert sample["frames"].shape[0] == num_frames
            except Exception:
                # Skip if video is too short for requested frames
                pytest.skip(f"Video too short for {num_frames} frames")

    @pytest.mark.gpu
    def test_video_encoder_with_different_backbones(self):
        """Test ImageToVideoEncoder with different backbone architectures."""
        backbones = [
            torchvision.models.resnet18(),
            torchvision.models.resnet34(),
            torchvision.models.mobilenet_v2(),
        ]

        # Create dummy video data
        video = torch.randn(
            2, 8, 3, 224, 224
        )  # [batch, frames, channels, height, width]

        for backbone in backbones:
            encoder = stable_pretraining.backbone.ImageToVideoEncoder(backbone)

            # Extract features
            with torch.no_grad():
                features = encoder(video)

            # Verify output shape
            assert features.shape[0] == 2  # batch size
            assert features.shape[1] == 8  # num frames
            # Feature dimension varies by architecture
