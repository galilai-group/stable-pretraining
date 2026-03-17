"""Unit tests for transforms that handle video data with shape (T, C, H, W) uint8."""

import pytest
import torch
import stable_pretraining.data.transforms as transforms


def create_video_uint8(num_frames=8, channels=3, height=64, width=64, seed=42):
    """Create a synthetic video tensor with shape (T, C, H, W) and dtype uint8.

    Args:
        num_frames: Number of frames (T dimension)
        channels: Number of channels (C dimension)
        height: Height of each frame
        width: Width of each frame
        seed: Random seed for reproducibility

    Returns:
        torch.Tensor with shape (T, C, H, W) and dtype torch.uint8
    """
    torch.manual_seed(seed)
    return torch.randint(
        0, 256, (num_frames, channels, height, width), dtype=torch.uint8
    )


def create_video_float(num_frames=8, channels=3, height=64, width=64, seed=42):
    """Create a synthetic video tensor with shape (T, C, H, W) and dtype float32.

    Args:
        num_frames: Number of frames (T dimension)
        channels: Number of channels (C dimension)
        height: Height of each frame
        width: Width of each frame
        seed: Random seed for reproducibility

    Returns:
        torch.Tensor with shape (T, C, H, W) and dtype torch.float32, values in [0, 1]
    """
    torch.manual_seed(seed)
    return torch.rand(num_frames, channels, height, width, dtype=torch.float32)


def create_video_dict(
    num_frames=8, channels=3, height=64, width=64, seed=42, key="video", dtype="uint8"
):
    """Create a sample dictionary with video data.

    Args:
        num_frames: Number of frames
        channels: Number of channels
        height: Frame height
        width: Frame width
        seed: Random seed
        key: Key for video in dictionary
        dtype: "uint8" or "float32"

    Returns:
        Dict with video tensor and idx
    """
    if dtype == "uint8":
        video = create_video_uint8(num_frames, channels, height, width, seed)
    else:
        video = create_video_float(num_frames, channels, height, width, seed)

    return {key: video, "idx": seed}


@pytest.mark.unit
class TestToImageVideo:
    """Test ToImage transform with video data (T, C, H, W)."""

    def test_uint8_video_conversion(self):
        """Test converting uint8 video (T, C, H, W) to float tensor."""
        sample = create_video_dict(num_frames=8, height=64, width=64, dtype="uint8")
        transform = transforms.ToImage(source="video", target="video")

        result = transform(sample)

        # Check output type and shape preserved
        assert isinstance(result["video"], torch.Tensor)
        assert result["video"].dtype == torch.float32
        assert result["video"].shape == (8, 3, 64, 64)
        # Values should be normalized to [0, 1]
        assert result["video"].min() >= 0.0
        assert result["video"].max() <= 1.0

    def test_uint8_video_with_normalization(self):
        """Test ToImage with mean/std normalization on video data."""
        sample = create_video_dict(num_frames=4, height=32, width=32, dtype="uint8")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.ToImage(
            mean=mean, std=std, source="video", target="video"
        )

        result = transform(sample)

        assert result["video"].dtype == torch.float32
        assert result["video"].shape == (4, 3, 32, 32)
        # Normalized values can be outside [0, 1]

    def test_float_video_passthrough(self):
        """Test that float video is handled correctly."""
        sample = create_video_dict(num_frames=6, height=48, width=48, dtype="float32")
        transform = transforms.ToImage(source="video", target="video", scale=False)

        result = transform(sample)

        assert result["video"].dtype == torch.float32
        assert result["video"].shape == (6, 3, 48, 48)

    def test_different_frame_counts(self):
        """Test ToImage with various frame counts."""
        for num_frames in [1, 4, 8, 16, 32]:
            sample = create_video_dict(num_frames=num_frames, dtype="uint8")
            transform = transforms.ToImage(source="video", target="video")

            result = transform(sample)

            assert result["video"].shape[0] == num_frames
            assert result["video"].dtype == torch.float32


@pytest.mark.unit
class TestResizeVideo:
    """Test Resize transform with video data (T, C, H, W)."""

    def test_resize_video_spatial_dimensions(self):
        """Test resizing video spatial dimensions while preserving temporal."""
        sample = create_video_dict(num_frames=8, height=128, width=128, dtype="float32")
        transform = transforms.Resize(size=(64, 64), source="video", target="video")

        result = transform(sample)

        # Temporal dimension should be preserved
        assert result["video"].shape[0] == 8
        # Spatial dimensions should be resized
        assert result["video"].shape[2] == 64
        assert result["video"].shape[3] == 64

    def test_resize_video_uint8(self):
        """Test resizing uint8 video data."""
        sample = create_video_dict(num_frames=4, height=224, width=224, dtype="uint8")
        # Convert to float first as Resize expects float
        sample["video"] = sample["video"].float() / 255.0

        transform = transforms.Resize(size=(112, 112), source="video", target="video")

        result = transform(sample)

        assert result["video"].shape == (4, 3, 112, 112)

    def test_resize_various_sizes(self):
        """Test resizing to various target sizes."""
        sample = create_video_dict(num_frames=4, height=256, width=256, dtype="float32")

        for target_size in [(128, 128), (64, 64), (224, 224), (32, 32)]:
            sample_copy = {"video": sample["video"].clone(), "idx": sample["idx"]}
            transform = transforms.Resize(
                size=target_size, source="video", target="video"
            )

            result = transform(sample_copy)

            assert result["video"].shape == (4, 3, target_size[0], target_size[1])


@pytest.mark.unit
class TestRandomHorizontalFlipVideo:
    """Test RandomHorizontalFlip transform with video data (T, C, H, W)."""

    def test_flip_all_frames_consistently(self):
        """Test that all frames are flipped consistently when p=1.0."""
        sample = create_video_dict(num_frames=4, height=32, width=32, dtype="float32")
        original = sample["video"].clone()

        transform = transforms.RandomHorizontalFlip(
            p=1.0, source="video", target="video"
        )

        result = transform(sample)

        # All frames should be flipped
        for t in range(4):
            # Check that each frame is horizontally flipped
            expected_flip = torch.flip(original[t], dims=[-1])
            assert torch.allclose(result["video"][t], expected_flip)

    def test_no_flip_preserves_video(self):
        """Test that video is unchanged when p=0.0."""
        sample = create_video_dict(num_frames=4, height=32, width=32, dtype="float32")
        original = sample["video"].clone()

        transform = transforms.RandomHorizontalFlip(
            p=0.0, source="video", target="video"
        )

        result = transform(sample)

        assert torch.allclose(result["video"], original)

    def test_flip_video_shape_preserved(self):
        """Test that video shape is preserved after flip."""
        for num_frames in [1, 4, 8, 16]:
            sample = create_video_dict(num_frames=num_frames, dtype="float32")
            transform = transforms.RandomHorizontalFlip(
                p=0.5, source="video", target="video"
            )

            result = transform(sample)

            assert result["video"].shape == sample["video"].shape


@pytest.mark.unit
class TestGaussianBlurVideo:
    """Test GaussianBlur transform with video data (T, C, H, W)."""

    def test_blur_video_shape_preserved(self):
        """Test that video shape is preserved after blur."""
        sample = create_video_dict(num_frames=4, height=64, width=64, dtype="float32")

        transform = transforms.GaussianBlur(
            kernel_size=3, sigma=(0.1, 2.0), p=1.0, source="video", target="video"
        )

        result = transform(sample)

        assert result["video"].shape == (4, 3, 64, 64)
        assert result["video"].dtype == torch.float32

    def test_blur_modifies_video(self):
        """Test that blur actually modifies the video."""
        sample = create_video_dict(num_frames=4, height=64, width=64, dtype="float32")
        original = sample["video"].clone()

        transform = transforms.GaussianBlur(
            kernel_size=5, sigma=(2.0, 2.0), p=1.0, source="video", target="video"
        )

        result = transform(sample)

        # Should be different from original
        assert not torch.allclose(result["video"], original)


@pytest.mark.unit
class TestColorJitterVideo:
    """Test ColorJitter transform with video data (T, C, H, W)."""

    def test_color_jitter_video_shape_preserved(self):
        """Test that video shape is preserved after color jitter."""
        sample = create_video_dict(num_frames=4, height=64, width=64, dtype="float32")

        transform = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=1.0,
            source="video",
            target="video",
        )

        result = transform(sample)

        assert result["video"].shape == (4, 3, 64, 64)

    def test_color_jitter_params_stored(self):
        """Test that color jitter parameters are stored in output."""
        sample = create_video_dict(num_frames=4, height=64, width=64, dtype="float32")

        transform = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=1.0,
            source="video",
            target="video",
        )

        result = transform(sample)

        assert "ColorJitter" in result


@pytest.mark.unit
class TestRandomGrayscaleVideo:
    """Test RandomGrayscale transform with video data (T, C, H, W)."""

    def test_grayscale_video_shape_preserved(self):
        """Test that RGB video converted to grayscale maintains 3 channels."""
        sample = create_video_dict(num_frames=4, height=64, width=64, dtype="float32")

        transform = transforms.RandomGrayscale(p=1.0, source="video", target="video")

        result = transform(sample)

        # Should maintain 3 channels (grayscale replicated)
        assert result["video"].shape == (4, 3, 64, 64)

    def test_grayscale_channels_equal(self):
        """Test that grayscale video has equal channel values."""
        sample = create_video_dict(num_frames=4, height=64, width=64, dtype="float32")

        transform = transforms.RandomGrayscale(p=1.0, source="video", target="video")

        result = transform(sample)

        # All channels should be equal for each frame
        for t in range(4):
            assert torch.allclose(
                result["video"][t, 0], result["video"][t, 1], atol=1e-5
            )
            assert torch.allclose(
                result["video"][t, 1], result["video"][t, 2], atol=1e-5
            )


@pytest.mark.unit
class TestRandomResizedCropVideo:
    """Test RandomResizedCrop transform with video data (T, C, H, W)."""

    def test_crop_video_output_size(self):
        """Test that RandomResizedCrop produces correct output size."""
        sample = create_video_dict(num_frames=4, height=256, width=256, dtype="float32")

        transform = transforms.RandomResizedCrop(
            size=(128, 128), source="video", target="video"
        )

        result = transform(sample)

        # Temporal dimension preserved, spatial resized
        assert result["video"].shape == (4, 3, 128, 128)

    def test_crop_different_sizes(self):
        """Test RandomResizedCrop with various target sizes."""
        sample = create_video_dict(num_frames=4, height=224, width=224, dtype="float32")

        for target_size in [(32, 32), (64, 64), (112, 112), (224, 224)]:
            sample_copy = {"video": sample["video"].clone(), "idx": sample["idx"]}
            transform = transforms.RandomResizedCrop(
                size=target_size, source="video", target="video"
            )

            result = transform(sample_copy)

            assert result["video"].shape == (4, 3, target_size[0], target_size[1])

    def test_crop_params_stored(self):
        """Test that crop parameters are stored in output."""
        sample = create_video_dict(num_frames=4, height=224, width=224, dtype="float32")

        transform = transforms.RandomResizedCrop(
            size=(128, 128), source="video", target="video"
        )

        result = transform(sample)

        assert "RandomResizedCrop" in result
        # Should have top, left, height, width
        assert result["RandomResizedCrop"].shape == (4,)


@pytest.mark.unit
class TestRandomSolarizeVideo:
    """Test RandomSolarize transform with video data (T, C, H, W)."""

    def test_solarize_video_shape_preserved(self):
        """Test that video shape is preserved after solarize."""
        sample = create_video_dict(num_frames=4, height=64, width=64, dtype="float32")

        transform = transforms.RandomSolarize(
            threshold=0.5, p=1.0, source="video", target="video"
        )

        result = transform(sample)

        assert result["video"].shape == (4, 3, 64, 64)

    def test_solarize_modifies_video(self):
        """Test that solarize actually modifies values above threshold."""
        sample = create_video_dict(num_frames=4, height=64, width=64, dtype="float32")
        original = sample["video"].clone()

        transform = transforms.RandomSolarize(
            threshold=0.5, p=1.0, source="video", target="video"
        )

        result = transform(sample)

        # Should be different from original (assuming some values are > threshold)
        assert not torch.allclose(result["video"], original)


@pytest.mark.unit
class TestRandomRotationVideo:
    """Test RandomRotation transform with video data (T, C, H, W)."""

    def test_rotation_video_shape_preserved(self):
        """Test that video shape is preserved after rotation (with no expansion)."""
        sample = create_video_dict(num_frames=4, height=64, width=64, dtype="float32")

        transform = transforms.RandomRotation(
            degrees=45, expand=False, source="video", target="video"
        )

        result = transform(sample)

        assert result["video"].shape == (4, 3, 64, 64)


@pytest.mark.unit
class TestPatchMaskingVideo:
    """Test PatchMasking transform with video data (T, C, H, W)."""

    def test_patch_masking_per_frame(self):
        """Test patch masking on individual frames of video."""
        # PatchMasking works on single images, so we test per-frame
        num_frames = 4
        sample = create_video_dict(
            num_frames=num_frames, height=224, width=224, dtype="float32"
        )

        # Apply to first frame only
        first_frame_sample = {"image": sample["video"][0]}
        transform = transforms.PatchMasking(
            patch_size=16, drop_ratio=0.5, source="image", target="masked_image"
        )

        result = transform(first_frame_sample)

        # Check output shape
        assert result["masked_image"].shape == (3, 224, 224)
        assert result["patch_mask"].shape == (14, 14)

    def test_patch_masking_uint8_frame(self):
        """Test patch masking on uint8 video frame."""
        sample = create_video_dict(num_frames=4, height=224, width=224, dtype="uint8")

        # Apply to single frame
        frame_sample = {"image": sample["video"][0]}
        transform = transforms.PatchMasking(
            patch_size=16, drop_ratio=0.5, source="image", target="masked_image"
        )

        result = transform(frame_sample)

        # Output should be float (normalized)
        assert result["masked_image"].dtype == torch.float32
        assert result["masked_image"].shape == (3, 224, 224)


@pytest.mark.unit
class TestCenterCropVideo:
    """Test CenterCrop transform with video data (T, C, H, W)."""

    def test_center_crop_video(self):
        """Test center crop on video data."""
        sample = create_video_dict(num_frames=4, height=256, width=256, dtype="float32")

        transform = transforms.CenterCrop(
            size=(224, 224), source="video", target="video"
        )

        result = transform(sample)

        assert result["video"].shape == (4, 3, 224, 224)

    def test_center_crop_various_sizes(self):
        """Test center crop with various sizes."""
        sample = create_video_dict(num_frames=4, height=256, width=256, dtype="float32")

        for crop_size in [(128, 128), (200, 200), (64, 64)]:
            sample_copy = {"video": sample["video"].clone(), "idx": sample["idx"]}
            transform = transforms.CenterCrop(
                size=crop_size, source="video", target="video"
            )

            result = transform(sample_copy)

            assert result["video"].shape == (4, 3, crop_size[0], crop_size[1])


@pytest.mark.unit
class TestComposeVideo:
    """Test composing multiple transforms with video data."""

    def test_compose_resize_and_flip(self):
        """Test composing resize and flip transforms on video."""
        sample = create_video_dict(num_frames=4, height=256, width=256, dtype="float32")

        transform = transforms.Compose(
            transforms.Resize(size=(128, 128), source="video", target="video"),
            transforms.RandomHorizontalFlip(p=1.0, source="video", target="video"),
        )

        result = transform(sample)

        assert result["video"].shape == (4, 3, 128, 128)

    def test_compose_multiple_augmentations(self):
        """Test composing multiple augmentation transforms on video."""
        sample = create_video_dict(num_frames=4, height=224, width=224, dtype="float32")

        transform = transforms.Compose(
            transforms.RandomResizedCrop(
                size=(128, 128), source="video", target="video"
            ),
            transforms.RandomHorizontalFlip(p=0.5, source="video", target="video"),
            transforms.GaussianBlur(
                kernel_size=3, sigma=(0.1, 2.0), p=0.5, source="video", target="video"
            ),
        )

        result = transform(sample)

        assert result["video"].shape == (4, 3, 128, 128)
        assert result["video"].dtype == torch.float32

    def test_compose_with_normalization(self):
        """Test composing transforms including normalization."""
        sample = create_video_dict(num_frames=4, height=224, width=224, dtype="uint8")

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose(
            transforms.ToImage(mean=mean, std=std, source="video", target="video"),
            transforms.RandomResizedCrop(
                size=(128, 128), source="video", target="video"
            ),
        )

        result = transform(sample)

        assert result["video"].shape == (4, 3, 128, 128)
        assert result["video"].dtype == torch.float32


@pytest.mark.unit
class TestVideoDataTypes:
    """Test video transforms with different data types."""

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.float16])
    def test_resize_with_different_dtypes(self, dtype):
        """Test Resize transform with different input dtypes."""
        torch.manual_seed(42)
        if dtype == torch.uint8:
            video = torch.randint(0, 256, (4, 3, 128, 128), dtype=torch.uint8)
        elif dtype == torch.float16:
            video = torch.rand(4, 3, 128, 128, dtype=torch.float16)
        else:
            video = torch.rand(4, 3, 128, 128, dtype=torch.float32)

        sample = {"video": video, "idx": 0}

        if dtype == torch.uint8:
            # Convert to float for resize
            sample["video"] = sample["video"].float() / 255.0

        transform = transforms.Resize(size=(64, 64), source="video", target="video")

        result = transform(sample)

        assert result["video"].shape == (4, 3, 64, 64)

    def test_uint8_to_float_pipeline(self):
        """Test typical uint8 to float conversion pipeline for video."""
        sample = create_video_dict(num_frames=8, height=256, width=256, dtype="uint8")

        # Typical preprocessing pipeline
        transform = transforms.Compose(
            transforms.ToImage(source="video", target="video"),
            transforms.Resize(size=(224, 224), source="video", target="video"),
            transforms.RandomHorizontalFlip(p=0.5, source="video", target="video"),
        )

        result = transform(sample)

        assert result["video"].dtype == torch.float32
        assert result["video"].shape == (8, 3, 224, 224)
        assert result["video"].min() >= 0.0
        assert result["video"].max() <= 1.0


@pytest.mark.unit
class TestVideoEdgeCases:
    """Test edge cases for video transforms."""

    def test_single_frame_video(self):
        """Test transforms on single-frame video (T=1)."""
        sample = create_video_dict(num_frames=1, height=64, width=64, dtype="float32")

        transform = transforms.Compose(
            transforms.Resize(size=(32, 32), source="video", target="video"),
            transforms.RandomHorizontalFlip(p=0.5, source="video", target="video"),
        )

        result = transform(sample)

        assert result["video"].shape == (1, 3, 32, 32)

    def test_many_frames_video(self):
        """Test transforms on video with many frames."""
        sample = create_video_dict(num_frames=64, height=64, width=64, dtype="float32")

        transform = transforms.Resize(size=(32, 32), source="video", target="video")

        result = transform(sample)

        assert result["video"].shape == (64, 3, 32, 32)

    def test_grayscale_video(self):
        """Test transforms on grayscale video (T, 1, H, W)."""
        torch.manual_seed(42)
        video = torch.rand(4, 1, 64, 64, dtype=torch.float32)
        sample = {"video": video, "idx": 0}

        transform = transforms.Resize(size=(32, 32), source="video", target="video")

        result = transform(sample)

        assert result["video"].shape == (4, 1, 32, 32)

    def test_rgba_video(self):
        """Test transforms on RGBA video (T, 4, H, W)."""
        torch.manual_seed(42)
        video = torch.rand(4, 4, 64, 64, dtype=torch.float32)
        sample = {"video": video, "idx": 0}

        transform = transforms.Resize(size=(32, 32), source="video", target="video")

        result = transform(sample)

        assert result["video"].shape == (4, 4, 32, 32)

    def test_small_spatial_dimensions(self):
        """Test transforms on video with small spatial dimensions."""
        sample = create_video_dict(num_frames=4, height=16, width=16, dtype="float32")

        transform = transforms.Resize(size=(8, 8), source="video", target="video")

        result = transform(sample)

        assert result["video"].shape == (4, 3, 8, 8)

    def test_non_square_video(self):
        """Test transforms on non-square video."""
        torch.manual_seed(42)
        video = torch.rand(4, 3, 128, 256, dtype=torch.float32)  # Height != Width
        sample = {"video": video, "idx": 0}

        transform = transforms.Resize(size=(64, 128), source="video", target="video")

        result = transform(sample)

        assert result["video"].shape == (4, 3, 64, 128)


@pytest.mark.unit
class TestControlledTransformVideo:
    """Test ControlledTransform with video data for deterministic augmentation."""

    def test_controlled_flip_video_deterministic(self):
        """Test that ControlledTransform makes video flip deterministic."""
        sample = create_video_dict(num_frames=4, height=64, width=64, dtype="float32")

        transform = transforms.ControlledTransform(
            transform=transforms.RandomHorizontalFlip(
                p=0.5, source="video", target="video"
            ),
            seed_offset=0,
        )

        # Apply multiple times with same idx
        result1 = transform({"video": sample["video"].clone(), "idx": 42})
        result2 = transform({"video": sample["video"].clone(), "idx": 42})

        assert torch.allclose(result1["video"], result2["video"])

    def test_controlled_crop_video_deterministic(self):
        """Test that ControlledTransform makes video crop deterministic."""
        sample = create_video_dict(num_frames=4, height=256, width=256, dtype="float32")

        transform = transforms.ControlledTransform(
            transform=transforms.RandomResizedCrop(
                size=(128, 128), source="video", target="video"
            ),
            seed_offset=0,
        )

        # Apply multiple times with same idx
        result1 = transform({"video": sample["video"].clone(), "idx": 123})
        result2 = transform({"video": sample["video"].clone(), "idx": 123})

        assert torch.allclose(result1["video"], result2["video"])

    def test_different_idx_different_augmentation(self):
        """Test that different idx produces different augmentation."""
        sample = create_video_dict(num_frames=4, height=256, width=256, dtype="float32")

        transform = transforms.ControlledTransform(
            transform=transforms.RandomResizedCrop(
                size=(128, 128), source="video", target="video"
            ),
            seed_offset=0,
        )

        result1 = transform({"video": sample["video"].clone(), "idx": 1})
        result2 = transform({"video": sample["video"].clone(), "idx": 2})

        # Should be different (with high probability for random crop)
        # Check the crop parameters are different
        assert not torch.allclose(
            result1["RandomResizedCrop"], result2["RandomResizedCrop"]
        )
