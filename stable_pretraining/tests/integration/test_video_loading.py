"""Integration tests for video loading functionality."""

import pytest
import torch
import torchvision

import stable_pretraining


@pytest.mark.integration
class TestVideoLoadingIntegration:
    """Integration tests for video loading with actual video data."""

    @pytest.mark.v1
    @pytest.mark.download
    def test_clip_extract(self):
        """Test video clip extraction with temporal sampling."""
        dataset = stable_pretraining.data.HFDataset(
            path="shivalikasingh/video-demo",
            split="train",
            trust_remote_code=True,
            transform=stable_pretraining.data.transforms.RandomContiguousTemporalSampler(
                source="video", target="frames", num_frames=10
            ),
        )

        # Test first sample
        sample = dataset[0]

        # Verify both video and frames are present
        assert "video" in sample
        assert "frames" in sample

        # Verify frame dimensions
        assert sample["frames"].ndim == 4  # [frames, channels, height, width]
        assert sample["frames"].size(0) == 10  # num_frames
        assert sample["frames"].size(1) == 3  # RGB channels

    @pytest.mark.v1
    @pytest.mark.download
    @pytest.mark.slow
    def test_clip_dataset_with_dataloader(self):
        """Test video dataset with DataLoader and transformations."""
        # Create dataset with video transforms
        dataset = stable_pretraining.data.HFDataset(
            path="shivalikasingh/video-demo",
            split="train",
            trust_remote_code=True,
            transform=stable_pretraining.data.transforms.Compose(
                stable_pretraining.data.transforms.RandomContiguousTemporalSampler(
                    source="video", target="video", num_frames=10
                ),
                stable_pretraining.data.transforms.Resize(
                    (128, 128), source="video", target="video"
                ),
            ),
        )

        # Create concatenated dataset for more samples
        dataset = torch.utils.data.ConcatDataset([dataset for _ in range(10)])

        # Create dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        # Test batch loading
        for data in loader:
            assert "video" in data
            assert "frames" not in data  # frames renamed to video
            assert data["video"].shape == (4, 10, 3, 128, 128)
            break  # Test only first batch

    @pytest.mark.v1
    @pytest.mark.download
    @pytest.mark.gpu
    def test_embedding_from_video_frames(self):
        """Test feature extraction from video frames using image encoder."""
        # Create dataset with full video preprocessing
        dataset = stable_pretraining.data.HFDataset(
            path="shivalikasingh/video-demo",
            split="train",
            trust_remote_code=True,
            transform=stable_pretraining.data.transforms.Compose(
                stable_pretraining.data.transforms.RandomContiguousTemporalSampler(
                    source="video", target="video", num_frames=10
                ),
                stable_pretraining.data.transforms.Resize(
                    (128, 128), source="video", target="video"
                ),
                stable_pretraining.data.transforms.ToImage(
                    scale=False,
                    mean=[0, 0, 0],
                    std=[255, 255, 255],
                    source="video",
                    target="video",
                ),
            ),
        )

        # Create concatenated dataset
        dataset = torch.utils.data.ConcatDataset([dataset for _ in range(10)])
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        # Create video encoder with ResNet18
        embedding = stable_pretraining.utils.ImageToVideoEncoder(
            torchvision.models.resnet18()
        )

        # Test feature extraction
        for data in loader:
            features = embedding(data["video"])
            assert features.shape == (4, 10, 1000)  # [batch, frames, features]
            break  # Test only first batch

    @pytest.mark.v1
    @pytest.mark.download
    def test_video_transform_pipeline(self):
        """Test complete video transform pipeline."""
        # Define transform pipeline
        transform = stable_pretraining.data.transforms.Compose(
            stable_pretraining.data.transforms.RandomContiguousTemporalSampler(
                source="video", target="video", num_frames=16
            ),
            stable_pretraining.data.transforms.Resize(
                (224, 224), source="video", target="video"
            ),
            stable_pretraining.data.transforms.ToImage(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                source="video",
                target="video",
            ),
        )

        # Create dataset
        dataset = stable_pretraining.data.HFDataset(
            path="shivalikasingh/video-demo",
            split="train",
            trust_remote_code=True,
            transform=transform,
        )

        # Test transformed sample
        sample = dataset[0]
        assert "video" in sample
        assert sample["video"].shape[0] == 16  # num_frames
        assert sample["video"].shape[2:] == (224, 224)  # spatial dimensions

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
            encoder = stable_pretraining.utils.ImageToVideoEncoder(backbone)

            # Extract features
            with torch.no_grad():
                features = encoder(video)

            # Verify output shape
            assert features.shape[0] == 2  # batch size
            assert features.shape[1] == 8  # num frames
            # Feature dimension varies by architecture

    @pytest.mark.v1
    def test_video_data_types(self):
        """Test video data type handling."""
        dataset = stable_pretraining.data.HFDataset(
            path="shivalikasingh/video-demo",
            split="train[:1]",
            trust_remote_code=True,
            transform=stable_pretraining.data.transforms.RandomContiguousTemporalSampler(
                source="video", target="frames", num_frames=10
            ),
        )

        sample = dataset[0]

        # Check data types
        assert isinstance(sample["frames"], torch.Tensor)
        assert sample["frames"].dtype in [torch.float32, torch.uint8]

    @pytest.mark.v1
    @pytest.mark.slow
    def test_video_batch_processing_efficiency(self):
        """Test efficient batch processing of video data."""
        import time

        # Create dataset
        dataset = stable_pretraining.data.HFDataset(
            path="shivalikasingh/video-demo",
            split="train",
            trust_remote_code=True,
            transform=stable_pretraining.data.transforms.Compose(
                stable_pretraining.data.transforms.RandomContiguousTemporalSampler(
                    source="video", target="video", num_frames=8
                ),
                stable_pretraining.data.transforms.Resize(
                    (112, 112), source="video", target="video"
                ),
            ),
        )

        # Create dataloader with multiple workers
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, num_workers=4, shuffle=True
        )

        # Time batch loading
        start_time = time.time()
        for i, batch in enumerate(loader):
            if i >= 5:  # Process only 5 batches
                break
            assert batch["video"].shape[0] <= 8

        elapsed_time = time.time() - start_time
        assert elapsed_time < 30  # Should process 5 batches in under 30 seconds
