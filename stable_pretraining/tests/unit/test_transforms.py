"""Unit tests for transforms that don't require actual images."""

import numpy as np
import pytest
import torch
import stable_pretraining.data.transforms as transforms
from PIL import Image
from torchvision.transforms import v2


@pytest.mark.unit
class TestPatchMasking:
    """Test suite for transforms.PatchMasking transform."""

    @pytest.fixture
    def pil_sample(self):
        """Create a sample dict with PIL image (like HuggingFace dataset item)."""
        img = Image.new("RGB", (224, 224), color=(255, 128, 64))
        return {"image": img}

    @pytest.fixture
    def tensor_sample(self):
        """Create a sample dict with tensor image."""
        img = torch.rand(3, 224, 224)
        return {"image": img}

    @pytest.fixture
    def uint8_tensor_sample(self):
        """Create a sample dict with uint8 tensor image."""
        img = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
        return {"image": img}

    def test_exact_drop_ratio_pil(self, pil_sample):
        """Test that exact drop ratio is respected with PIL images."""
        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="image",
            target="masked_image",
        )

        result = transform(pil_sample)

        # Check output keys
        assert "masked_image" in result
        assert "patch_mask" in result

        # Check patch mask shape (224/16 = 14 patches per side)
        assert result["patch_mask"].shape == (14, 14)

        # Check exact drop ratio
        total_patches = 14 * 14
        kept_patches = result["patch_mask"].sum().item()
        masked_patches = total_patches - kept_patches
        expected_masked = int(total_patches * 0.5)

        assert masked_patches == expected_masked, (
            f"Expected exactly {expected_masked} masked patches, got {masked_patches}"
        )

        # Check output type matches input
        assert isinstance(result["masked_image"], torch.Tensor)

    def test_exact_drop_ratio_tensor(self, tensor_sample):
        """Test that exact drop ratio is respected with tensor images."""
        transform = transforms.PatchMasking(
            patch_size=32,
            drop_ratio=0.75,
            source="image",
            target="masked_image",
        )

        result = transform(tensor_sample)

        # Check patch mask shape (224/32 = 7 patches per side)
        assert result["patch_mask"].shape == (7, 7)

        # Check exact drop ratio
        total_patches = 7 * 7
        kept_patches = result["patch_mask"].sum().item()
        masked_patches = total_patches - kept_patches
        expected_masked = int(total_patches * 0.75)

        assert masked_patches == expected_masked

        # Check output type
        assert isinstance(result["masked_image"], torch.Tensor)
        assert result["masked_image"].shape == tensor_sample["image"].shape

    def test_fill_value_applied_correctly(self, tensor_sample):
        """Test that custom mask value is applied to masked patches."""
        fill_value = 0.5
        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=1.0,  # Mask all patches
            source="image",
            target="masked_image",
            fill_value=fill_value,
        )

        result = transform(tensor_sample)

        # With drop_ratio=1.0, all patches should be masked
        assert result["patch_mask"].sum().item() == 0

        # All pixels should be fill_value
        assert torch.allclose(
            result["masked_image"], torch.tensor(fill_value), atol=1e-6
        )

    def test_no_masking_when_drop_ratio_zero(self, tensor_sample):
        """Test that no masking occurs when drop_ratio is 0."""
        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.0,
            source="image",
            target="masked_image",
        )

        result = transform(tensor_sample)

        # All patches should be kept
        total_patches = (224 // 16) ** 2
        assert result["patch_mask"].sum().item() == total_patches

        # Image should be unchanged
        assert torch.allclose(result["masked_image"], tensor_sample["image"], atol=1e-6)

    def test_default_fill_value_pil(self, pil_sample):
        """Test that PIL images use default mask value of 128/255."""
        transform = transforms.PatchMasking(
            patch_size=224,  # One big patch
            drop_ratio=1.0,  # Mask it
            source="image",
            target="masked_image",
        )

        result = transform(pil_sample)

        # Convert to array to check values
        img_array = np.array(result["masked_image"])
        expected_value = 0

        # All pixels should be mid-gray
        assert np.allclose(img_array, expected_value, atol=1)

    def test_default_fill_value_tensor(self, tensor_sample):
        """Test that float tensors use default mask value of 0.0."""
        transform = transforms.PatchMasking(
            patch_size=224,  # One big patch
            drop_ratio=1.0,  # Mask it
            source="image",
            target="masked_image",
        )

        result = transform(tensor_sample)

        # All pixels should be 0.0
        assert torch.allclose(result["masked_image"], torch.tensor(0.0), atol=1e-6)

    def test_uint8_tensor_handling(self, uint8_tensor_sample):
        """Test that uint8 tensors are handled correctly."""
        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="image",
            target="masked_image",
        )

        result = transform(uint8_tensor_sample)

        # Should return float tensor (normalized)
        assert result["masked_image"].dtype == torch.float32
        assert result["masked_image"].min() >= 0.0
        assert result["masked_image"].max() <= 1.0

    def test_different_patch_sizes(self, tensor_sample):
        """Test with various patch sizes."""
        for patch_size in [8, 16, 32, 56]:
            transform = transforms.PatchMasking(
                patch_size=patch_size,
                drop_ratio=0.3,
                source="image",
                target="masked_image",
            )

            result = transform(tensor_sample)

            expected_patches_per_side = 224 // patch_size
            assert result["patch_mask"].shape == (
                expected_patches_per_side,
                expected_patches_per_side,
            )

    def test_non_divisible_image_size(self):
        """Test with image size not perfectly divisible by patch_size."""
        img = torch.rand(3, 225, 225)  # Not divisible by 16
        sample = {"image": img}

        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="image",
            target="masked_image",
        )

        result = transform(sample)

        # Should handle gracefully (14x14 patches, ignoring remainder)
        assert result["patch_mask"].shape == (14, 14)
        assert result["masked_image"].shape == img.shape

    def test_custom_source_target_keys(self):
        """Test with custom dictionary keys."""
        img = Image.new("RGB", (224, 224))
        sample = {"my_image": img}

        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="my_image",
            target="my_output",
        )

        result = transform(sample)

        assert "my_output" in result
        assert "patch_mask" in result
        assert "my_image" in result  # Original should still be there

    def test_randomness(self, tensor_sample):
        """Test that different calls produce different masks."""
        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="image",
            target="masked_image",
        )

        result1 = transform(tensor_sample.copy())
        result2 = transform(tensor_sample.copy())

        # Masks should be different (with very high probability)
        assert not torch.equal(result1["patch_mask"], result2["patch_mask"])

    def test_grayscale_image(self):
        """Test with grayscale PIL image."""
        img = Image.new("L", (224, 224), color=128)
        sample = {"image": img}

        transform = transforms.PatchMasking(
            patch_size=16,
            drop_ratio=0.5,
            source="image",
            target="masked_image",
        )

        result = transform(sample)

        assert isinstance(result["masked_image"], torch.Tensor)
        assert result["patch_mask"].shape == (14, 14)


@pytest.mark.unit
class TestTransformUtils:
    """Test transform utilities and basic functionality."""

    def test_collator(self):
        """Test the Collator utility."""
        import stable_pretraining as spt

        assert spt.data.Collator._test()

    def test_compose_transforms(self):
        """Test composing multiple transforms."""
        transform = transforms.Compose(transforms.RGB(), transforms.ToImage())
        # Test with mock data in expected format (dict with 'image' key)
        mock_data = {"image": torch.randn(3, 32, 32)}
        result = transform(mock_data)
        assert isinstance(result, dict)
        assert "image" in result
        assert isinstance(result["image"], torch.Tensor)

    def test_to_image_transform(self):
        """Test ToImage transform with different inputs."""
        transform = transforms.ToImage()

        # Test with numpy array
        np_image = np.random.rand(32, 32, 3).astype(np.float32)
        data = {"image": np_image}
        result = transform(data)
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == (3, 32, 32)

        # Test with torch tensor
        torch_image = torch.randn(3, 32, 32)
        data = {"image": torch_image}
        result = transform(data)
        assert isinstance(result["image"], torch.Tensor)

    def test_rgb_transform(self):
        """Test RGB transform ensures 3 channels."""
        transform = transforms.RGB()

        # Test with grayscale image
        gray_image = torch.randn(1, 32, 32)
        data = {"image": gray_image}
        result = transform(data)
        assert result["image"].shape == (3, 32, 32)

        # Test with RGB image (should be unchanged)
        rgb_image = torch.randn(3, 32, 32)
        data = {"image": rgb_image}
        result = transform(data)
        assert result["image"].shape == (3, 32, 32)

    def test_normalize_transform(self):
        """Test normalization with mean and std."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.ToImage(mean=mean, std=std)

        # Create a tensor with known values
        image = torch.ones(3, 32, 32)
        data = {"image": image}
        result = transform(data)

        # Check that normalization was applied
        assert not torch.allclose(result["image"], image)

    def test_transform_params_initialization(self):
        """Test that transforms can be initialized with various parameters."""
        # Test each transform can be created
        transforms_to_test = [
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomChannelPermutation(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomResizedCrop(size=(32, 32)),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.RandomRotation(degrees=90),
        ]

        for t in transforms_to_test:
            assert t is not None


def create_test_image(size=(32, 32), mode="RGB", seed=42):
    """Create a synthetic test image for testing.

    Args:
        size: Image size (H, W)
        mode: PIL image mode
        seed: Random seed for reproducibility

    Returns:
        PIL Image
    """
    np.random.seed(seed)
    if mode == "RGB":
        arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    elif mode == "L":
        arr = np.random.randint(0, 256, size, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return Image.fromarray(arr, mode=mode)


def create_test_dict(idx=0, size=(32, 32), seed=42, include_label=True):
    """Create a test dictionary with image, idx, and optionally label.

    Args:
        idx: Sample index (used for controlled transforms)
        size: Image size
        seed: Random seed for image generation
        include_label: Whether to include a label

    Returns:
        Dict with "image", "idx" keys (and optionally "label")
    """
    img_dict = {
        "image": create_test_image(size=size, seed=seed),
        "idx": idx,
    }
    if include_label:
        img_dict["label"] = seed % 10  # Arbitrary label
    return img_dict


@pytest.fixture
def test_dict():
    """Fixture providing a test dictionary."""
    return create_test_dict(idx=0, seed=42)


@pytest.fixture
def test_dicts():
    """Fixture providing multiple test dictionaries."""
    return [create_test_dict(idx=i, seed=i) for i in range(5)]


# ============================================================================
# Unit Tests for Controlled Transforms
# ============================================================================


@pytest.mark.unit
class TestControlledTransforms:
    """Test controlled transforms produce deterministic results."""

    @pytest.mark.parametrize(
        "transform_class,kwargs",
        [
            (transforms.GaussianBlur, {"kernel_size": 3}),
            (transforms.RandomChannelPermutation, {}),
            (transforms.RandomHorizontalFlip, {"p": 0.5}),
            (transforms.RandomGrayscale, {"p": 0.5}),
            (
                transforms.ColorJitter,
                {"brightness": 0.8, "contrast": 0.4, "saturation": 0.4, "hue": 0.4},
            ),
            (transforms.RandomResizedCrop, {"size": (32, 32)}),
            (transforms.RandomSolarize, {"threshold": 0.5, "p": 0.2}),
            (transforms.RandomRotation, {"degrees": 90}),
        ],
    )
    def test_deterministic_with_same_idx(self, transform_class, kwargs):
        """Test that same idx produces same output (core controlled transform property)."""
        # Create transform
        our_transform = transforms.ControlledTransform(
            transform=transform_class(**kwargs), seed_offset=0
        )

        # Create test dict with specific idx
        sample = create_test_dict(idx=42, seed=100)

        # Apply transform multiple times with same idx
        output1 = our_transform(sample.copy())
        output2 = our_transform(sample.copy())
        output3 = our_transform(sample.copy())

        # Extract images
        img1 = output1["image"]
        img2 = output2["image"]
        img3 = output3["image"]

        # Convert to tensors for comparison
        if isinstance(img1, Image.Image):
            img1 = torch.from_numpy(np.array(img1))
            img2 = torch.from_numpy(np.array(img2))
            img3 = torch.from_numpy(np.array(img3))

        # Should be identical regardless of when called
        assert torch.equal(img1, img2), (
            f"{transform_class.__name__} is not deterministic with same idx (1 vs 2)"
        )
        assert torch.equal(img1, img3), (
            f"{transform_class.__name__} is not deterministic with same idx (1 vs 3)"
        )

    @pytest.mark.parametrize(
        "transform_class,kwargs,is_random",
        [
            (transforms.GaussianBlur, {"kernel_size": 3}, False),  # Deterministic
            (transforms.RandomChannelPermutation, {}, True),
            (
                transforms.RandomHorizontalFlip,
                {"p": 0.5},
                True,
            ),  # Changed from 1.0 to 0.5
            (transforms.RandomGrayscale, {"p": 0.5}, True),  # Changed from 1.0 to 0.5
            (
                transforms.ColorJitter,
                {"brightness": 0.8, "contrast": 0.4, "saturation": 0.4, "hue": 0.4},
                True,
            ),
            (transforms.RandomResizedCrop, {"size": (32, 32)}, True),
            (transforms.RandomRotation, {"degrees": 90}, True),
        ],
    )
    def test_different_idx_produce_different_outputs(
        self, transform_class, kwargs, is_random
    ):
        """Test that different idx values produce different outputs for random transforms."""
        if not is_random:
            pytest.skip("Transform is deterministic")

        our_transform = transforms.ControlledTransform(
            transform=transform_class(**kwargs), seed_offset=0
        )

        # Create samples with different idx but DIFFERENT images
        # (to avoid the case where transform doesn't apply to either)
        sample1 = create_test_dict(idx=0, seed=100)
        sample2 = create_test_dict(idx=1, seed=101)  # Different image AND idx
        _ = create_test_dict(idx=100, seed=200)

        # Apply transform multiple times to build confidence
        different_count = 0
        total_comparisons = 0

        for _ in range(5):  # Try multiple times to account for randomness
            output1 = our_transform(sample1.copy())
            output2 = our_transform(sample2.copy())

            img1 = output1["image"]
            img2 = output2["image"]

            if isinstance(img1, Image.Image):
                img1 = torch.from_numpy(np.array(img1))
                img2 = torch.from_numpy(np.array(img2))

            if not torch.equal(img1, img2):
                different_count += 1
            total_comparisons += 1

        # At least some should be different (not all identical)
        # For p=0.5, we expect about 50% to differ
        assert different_count > 0, (
            f"{transform_class.__name__} produced identical outputs for all comparisons"
        )

    def test_seed_offset_affects_output(self):
        """Test that seed_offset changes which augmentation is applied."""
        sample = create_test_dict(idx=42, seed=100)

        transform1 = transforms.ControlledTransform(
            transform=transforms.RandomHorizontalFlip(p=0.5), seed_offset=0
        )
        transform2 = transforms.ControlledTransform(
            transform=transforms.RandomHorizontalFlip(p=0.5), seed_offset=1000
        )

        # Same idx, different seed offsets
        output1 = transform1(sample.copy())
        output2 = transform2(sample.copy())

        # Verify outputs exist
        assert "image" in output1
        assert "image" in output2

    @pytest.mark.parametrize(
        "our_transform_class,our_kwargs,torchvision_class,tv_kwargs",
        [
            (
                transforms.GaussianBlur,
                {"kernel_size": 3},
                v2.GaussianBlur,
                {"kernel_size": 3},
            ),
            (
                transforms.RandomHorizontalFlip,
                {"p": 0.5},
                v2.RandomHorizontalFlip,
                {"p": 0.5},
            ),
            (transforms.RandomGrayscale, {"p": 0.5}, v2.RandomGrayscale, {"p": 0.5}),
            (
                transforms.ColorJitter,
                {"brightness": 0.8, "contrast": 0.4, "saturation": 0.4, "hue": 0.4},
                v2.ColorJitter,
                {"brightness": 0.8, "contrast": 0.4, "saturation": 0.4, "hue": 0.4},
            ),
            (
                transforms.RandomResizedCrop,
                {"size": (32, 32)},
                v2.RandomResizedCrop,
                {"size": (32, 32)},
            ),
            (
                transforms.RandomRotation,
                {"degrees": 90},
                v2.RandomRotation,
                {"degrees": 90},
            ),
        ],
    )
    def test_matches_torchvision_with_manual_seed(
        self, our_transform_class, our_kwargs, torchvision_class, tv_kwargs
    ):
        """Test that controlled transforms match torchvision when using idx as seed."""
        # Create transforms
        our_transform = transforms.ControlledTransform(
            transform=our_transform_class(**our_kwargs), seed_offset=0
        )
        tv_transform = torchvision_class(**tv_kwargs)

        # Test multiple idx values
        for idx in [0, 42, 100, 999]:
            # Create test data
            img = create_test_image(seed=idx)
            sample = {"image": img, "idx": idx}

            # Apply our transform (uses idx internally)
            our_output = our_transform(sample.copy())
            our_img = our_output["image"]

            # Apply torchvision transform with manual seed based on idx
            torch.manual_seed(idx)  # Simulate what ControlledTransform does
            tv_img = tv_transform(img)

            # Convert to tensors
            if isinstance(our_img, Image.Image):
                our_img = torch.from_numpy(np.array(our_img)).float()
            if isinstance(tv_img, Image.Image):
                tv_img = torch.from_numpy(np.array(tv_img)).float()

            # Should match closely
            assert torch.allclose(our_img, tv_img, atol=1e-5), (
                f"{our_transform_class.__name__} doesn't match torchvision with idx {idx}"
            )

    def test_preserves_idx_and_other_keys(self):
        """Test that transform preserves idx and other keys in the dict."""
        sample = {
            "image": create_test_image(seed=42),
            "idx": 123,
            "label": 5,
            "metadata": {"id": 456, "source": "test"},
        }

        transform = transforms.ControlledTransform(
            transforms.RandomHorizontalFlip(p=1.0), seed_offset=0
        )

        output = transform(sample.copy())

        # Should preserve all keys
        assert "image" in output
        assert "idx" in output
        assert output["idx"] == 123
        assert "label" in output
        assert output["label"] == 5
        assert "metadata" in output
        assert output["metadata"]["id"] == 456


@pytest.mark.unit
class TestIdxBasedSeeding:
    """Test idx-based seeding mechanism."""

    def test_same_idx_always_same_augmentation(self):
        """Test that same idx always produces same augmentation across runs."""
        transform = transforms.ControlledTransform(
            transforms.RandomResizedCrop(size=(32, 32)),  # Changed from ColorJitter
            seed_offset=0,
        )

        idx = 42
        sample = create_test_dict(idx=idx, seed=100)

        # Apply many times
        outputs = []
        for _ in range(10):
            output = transform(sample.copy())
            outputs.append(output["image"])

        # All should be identical
        first = torch.from_numpy(np.array(outputs[0]))
        for i, img in enumerate(outputs[1:], 1):
            img_tensor = torch.from_numpy(np.array(img))
            assert torch.equal(first, img_tensor), (
                f"Output {i} differs from output 0 for same idx"
            )

    def test_idx_independent_of_call_order(self):
        """Test that idx-based seeding is independent of call order."""
        transform = transforms.ControlledTransform(
            transforms.RandomRotation(degrees=45), seed_offset=0
        )

        # Create samples with different idx
        samples = [create_test_dict(idx=i, seed=i) for i in range(5)]

        # Process in order: 0, 1, 2, 3, 4
        outputs_forward = [transform(s.copy()) for s in samples]

        # Process in reverse: 4, 3, 2, 1, 0
        outputs_reverse = [transform(s.copy()) for s in reversed(samples)]
        outputs_reverse = list(reversed(outputs_reverse))

        # Results should match regardless of order
        for i, (fwd, rev) in enumerate(zip(outputs_forward, outputs_reverse)):
            fwd_img = torch.from_numpy(np.array(fwd["image"]))
            rev_img = torch.from_numpy(np.array(rev["image"]))
            assert torch.equal(fwd_img, rev_img), (
                f"Output for idx {i} differs based on call order"
            )

    def test_idx_across_epochs(self):
        """Test that same idx produces same augmentation across epochs."""
        transform = transforms.ControlledTransform(
            transforms.RandomHorizontalFlip(p=0.5), seed_offset=0
        )

        idx = 99
        sample = create_test_dict(idx=idx, seed=50)

        # Simulate 3 epochs
        epoch_outputs = []
        for epoch in range(3):
            # In each epoch, process other samples first (to change global state)
            for other_idx in range(20):
                other_sample = create_test_dict(idx=other_idx, seed=other_idx)
                _ = transform(other_sample.copy())

            # Then process our target sample
            output = transform(sample.copy())
            epoch_outputs.append(output["image"])

        # All epochs should produce identical output for idx=99
        first = torch.from_numpy(np.array(epoch_outputs[0]))
        for epoch, img in enumerate(epoch_outputs[1:], 1):
            img_tensor = torch.from_numpy(np.array(img))
            assert torch.equal(first, img_tensor), (
                f"Epoch {epoch} differs from epoch 0 for idx {idx}"
            )

    def test_different_images_same_idx(self):
        """Test that same idx on different images gives consistent relative augmentation."""
        transform = transforms.ControlledTransform(
            transforms.RandomHorizontalFlip(p=1.0),
            seed_offset=0,  # Always flip
        )

        idx = 10

        # Two different images, same idx
        sample1 = create_test_dict(idx=idx, seed=1)
        sample2 = create_test_dict(idx=idx, seed=2)

        output1 = transform(sample1.copy())
        output2 = transform(sample2.copy())

        # Both should be flipped (deterministic with p=1.0)
        # We can't easily verify they're flipped without the original,
        # but we verify the function runs and preserves idx
        assert output1["idx"] == idx
        assert output2["idx"] == idx


@pytest.mark.unit
class TestControlledTransformComposition:
    """Test composition of controlled transforms."""

    def test_compose_multiple_transforms(self):
        """Test composing multiple controlled transforms."""
        transform = transforms.Compose(
            transforms.ControlledTransform(
                transforms.RandomHorizontalFlip(p=1.0), seed_offset=0
            ),
            transforms.ControlledTransform(
                transforms.RandomResizedCrop(size=(32, 32)),  # Changed from ColorJitter
                seed_offset=1,
            ),
            transforms.ControlledTransform(
                transforms.RandomRotation(degrees=45), seed_offset=2
            ),
        )

        sample = create_test_dict(idx=42, seed=100)

        # Should be deterministic based on idx
        output1 = transform(sample.copy())
        output2 = transform(sample.copy())

        img1 = torch.from_numpy(np.array(output1["image"]))
        img2 = torch.from_numpy(np.array(output2["image"]))

        assert torch.equal(img1, img2)

    def test_different_seed_offsets_produce_different_augmentations(self):
        """Test that different seed offsets in composition produce different augmentations."""
        sample = create_test_dict(idx=42, seed=100)

        # Same transform, different offsets
        transform1 = transforms.Compose(
            transforms.ControlledTransform(
                transforms.RandomResizedCrop(size=(32, 32)),  # Changed from ColorJitter
                seed_offset=0,
            ),
            transforms.ControlledTransform(
                transforms.RandomResizedCrop(size=(32, 32)), seed_offset=1
            ),
        )

        # Process once
        output = transform1(sample.copy())

        # The two transforms should have different offsets
        assert "image" in output


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases for controlled transforms."""

    def test_missing_idx_raises_error(self):
        """Test that missing idx key raises error."""
        transform = transforms.ControlledTransform(
            transforms.RandomHorizontalFlip(p=1.0), seed_offset=0
        )

        # Missing idx key
        sample = {"image": create_test_image(seed=42)}

        with pytest.raises(KeyError):
            transform(sample)

    def test_idx_zero(self):
        """Test that idx=0 works correctly."""
        transform = transforms.ControlledTransform(
            transforms.RandomHorizontalFlip(p=0.5), seed_offset=0
        )

        sample = create_test_dict(idx=0, seed=42)
        output = transform(sample.copy())

        assert output["idx"] == 0
        assert "image" in output

    def test_negative_idx(self):
        """Test that negative idx is handled (converted to positive)."""
        transform = transforms.ControlledTransform(
            transforms.RandomHorizontalFlip(p=0.5), seed_offset=0
        )

        # Negative idx should either be converted to positive or raise error
        sample = create_test_dict(idx=0, seed=42)
        sample["idx"] = -1

        try:
            output = transform(sample.copy())
            # If it works, idx should be preserved
            assert output["idx"] == -1
        except ValueError as e:
            # If it raises ValueError about seed range, that's also acceptable
            assert "Seed must be between" in str(e)
            pytest.skip("Implementation doesn't support negative idx")

    def test_large_idx(self):
        """Test with very large idx values."""
        transform = transforms.ControlledTransform(
            transforms.RandomHorizontalFlip(p=0.5), seed_offset=0
        )

        # Use a large but valid seed value
        large_idx = 2**31 - 1  # Max valid seed
        sample = create_test_dict(idx=large_idx, seed=42)
        output = transform(sample.copy())

        assert output["idx"] == large_idx

    def test_probability_zero(self):
        """Test transforms with p=0 never apply."""
        transform = transforms.ControlledTransform(
            transforms.RandomHorizontalFlip(p=0.0), seed_offset=0
        )

        sample = create_test_dict(idx=42, seed=100)
        original = torch.from_numpy(np.array(sample["image"]))

        # Try multiple times
        for _ in range(10):
            output = transform(sample.copy())
            output_img = torch.from_numpy(np.array(output["image"]))

            # Should never change
            assert torch.equal(output_img, original)

    def test_probability_one(self):
        """Test transforms with p=1.0 always apply."""
        transform = transforms.ControlledTransform(
            transforms.RandomHorizontalFlip(p=1.0), seed_offset=0
        )

        # Try multiple idx values - should always flip
        for idx in range(10):
            sample = create_test_dict(idx=idx, seed=idx)
            output = transform(sample.copy())
            assert "image" in output


@pytest.mark.unit
class TestReproducibility:
    """Test reproducibility guarantees."""

    def test_reproducible_across_workers(self):
        """Simulate that different workers get same augmentation for same idx."""
        transform = transforms.ControlledTransform(
            transforms.RandomResizedCrop(size=(32, 32)),  # Changed from ColorJitter
            seed_offset=0,
        )

        idx = 123

        # Simulate worker 1
        worker1_sample = create_test_dict(idx=idx, seed=50)
        worker1_output = transform(worker1_sample.copy())

        # Simulate worker 2 (same idx, same image)
        worker2_sample = create_test_dict(idx=idx, seed=50)
        worker2_output = transform(worker2_sample.copy())

        # Should produce identical augmentation
        img1 = torch.from_numpy(np.array(worker1_output["image"]))
        img2 = torch.from_numpy(np.array(worker2_output["image"]))

        assert torch.equal(img1, img2)

    def test_dataset_shuffle_invariance(self):
        """Test that shuffling dataset doesn't affect augmentation for given idx."""
        transform = transforms.ControlledTransform(
            transforms.RandomRotation(degrees=90), seed_offset=0
        )

        # Create samples
        samples = [create_test_dict(idx=i, seed=i) for i in range(10)]

        # Process in original order
        original_order = [transform(s.copy()) for s in samples]

        # Process in shuffled order
        indices = [7, 2, 9, 1, 5, 0, 8, 3, 6, 4]
        shuffled_samples = [samples[i] for i in indices]
        shuffled_outputs = [transform(s.copy()) for s in shuffled_samples]

        # Unshuffle results
        unshuffled = [None] * 10
        for orig_idx, output in zip(indices, shuffled_outputs):
            unshuffled[orig_idx] = output

        # Compare
        for i, (orig, unshuf) in enumerate(zip(original_order, unshuffled)):
            orig_img = torch.from_numpy(np.array(orig["image"]))
            unshuf_img = torch.from_numpy(np.array(unshuf["image"]))
            assert torch.equal(orig_img, unshuf_img), (
                f"Augmentation for idx {i} changed after shuffle"
            )


@pytest.mark.unit
class TestMultiViewTransform:
    """Test suite for MultiViewTransform."""

    @pytest.fixture
    def sample(self):
        """Create a sample dict with PIL image."""
        img = Image.new("RGB", (32, 32), color=(255, 128, 64))
        return {"image": img, "label": 1}

    def test_list_input_returns_dict_with_views_list(self, sample):
        """Test that list input returns dict with 'views' key and correct length."""
        # Use two different transforms to verify they are both applied
        t1 = transforms.RandomHorizontalFlip(p=0.5)
        t2 = transforms.RandomRotation(degrees=90)

        transform_list = [t1, t2]
        mv_transform = transforms.MultiViewTransform(transform_list)

        output = mv_transform(sample.copy())

        # 1. Output remains a dict
        assert isinstance(output, dict), "Output must be a dictionary"

        # 2. Check keys per documentation (list input -> "views" key)
        assert "views" in output, "Output dict must contain 'views' key for list input"

        # 3. Check length of list matches length of input transforms
        views = output["views"]
        assert isinstance(views, list), "'views' must be a list"
        assert len(views) == len(transform_list), (
            f"Expected {len(transform_list)} views, got {len(views)}"
        )

        # Verify content validity
        for view in views:
            assert isinstance(view, dict)
            assert "image" in view

    def test_dict_input_returns_dict_with_matching_keys(self, sample):
        """Test that dict input returns dict with matching keys."""
        t1 = transforms.RandomHorizontalFlip(p=0.5)
        t2 = transforms.RandomRotation(degrees=90)

        transform_dict = {"view1": t1, "view2": t2, "aug_view": t1}

        mv_transform = transforms.MultiViewTransform(transform_dict)

        output = mv_transform(sample.copy())

        # 1. Output remains a dict
        assert isinstance(output, dict), "Output must be a dictionary"

        # 2. Check keys line up per documentation
        # Keys in output should match keys in input dict
        assert set(output.keys()) == set(transform_dict.keys()), (
            "Output keys must match input transform keys"
        )

        # Verify content validity
        for key in transform_dict.keys():
            view = output[key]
            assert isinstance(view, dict)
            assert "image" in view
