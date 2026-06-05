"""Tests for native JAX image augmentations (the kornia alternative on JAX)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.jax]

from stable_pretraining.jax import augment  # noqa: E402


def _imgs(b=4, s=32):
    return jnp.asarray(np.random.RandomState(0).rand(b, s, s, 3).astype("float32"))


def test_normalize_matches_manual():
    x = _imgs()
    out = augment.normalize(x)
    mean = jnp.asarray(augment.IMAGENET_MEAN).reshape(1, 1, 1, 3)
    std = jnp.asarray(augment.IMAGENET_STD).reshape(1, 1, 1, 3)
    np.testing.assert_allclose(np.asarray(out), np.asarray((x - mean) / std), atol=1e-6)


def test_horizontal_flip_p1_reverses_width():
    x = _imgs()
    out = augment.random_horizontal_flip(jax.random.PRNGKey(0), x, p=1.0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x[:, :, ::-1, :]))


def test_horizontal_flip_p0_is_identity():
    x = _imgs()
    out = augment.random_horizontal_flip(jax.random.PRNGKey(0), x, p=0.0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


def test_grayscale_p1_equalizes_channels():
    x = _imgs()
    out = augment.random_grayscale(jax.random.PRNGKey(0), x, p=1.0)
    out = np.asarray(out)
    np.testing.assert_allclose(out[..., 0], out[..., 1], atol=1e-6)
    np.testing.assert_allclose(out[..., 1], out[..., 2], atol=1e-6)


def test_random_resized_crop_shape_and_range():
    x = _imgs(b=4, s=32)
    out = augment.random_resized_crop(jax.random.PRNGKey(0), x, size=24)
    assert out.shape == (4, 24, 24, 3)
    assert np.isfinite(np.asarray(out)).all()


def test_two_crop_shapes_and_differ():
    x = _imgs()
    v1, v2 = augment.two_crop(jax.random.PRNGKey(0), x, size=24)
    assert v1.shape == v2.shape == (4, 24, 24, 3)
    # Independent keys -> the two views should not be identical.
    assert not np.allclose(np.asarray(v1), np.asarray(v2))


def test_simclr_view_is_jittable():
    x = _imgs()
    fn = jax.jit(lambda k, im: augment.simclr_view(k, im, 24))
    out = fn(jax.random.PRNGKey(0), x)
    assert out.shape == (4, 24, 24, 3)


# --------------------------------------------------------------------------- #
# Parity of the deterministic cores vs torchvision.transforms.functional.
# Given the same parameters, the JAX op must match torchvision.
# --------------------------------------------------------------------------- #
def _to_torch(x_nhwc):
    import torch

    return torch.from_numpy(
        np.ascontiguousarray(np.asarray(x_nhwc).transpose(0, 3, 1, 2))
    )


def _to_nhwc(t):
    return np.asarray(t.numpy()).transpose(0, 2, 3, 1)


def test_hflip_parity():
    import torchvision.transforms.functional as F

    x = _imgs()
    tj = np.asarray(augment.hflip(x))
    tt = _to_nhwc(F.hflip(_to_torch(x)))
    np.testing.assert_allclose(tj, tt, atol=1e-6)


def test_grayscale_parity():
    import torchvision.transforms.functional as F

    x = _imgs()
    tj = np.asarray(augment.rgb_to_grayscale(x))
    tt = _to_nhwc(F.rgb_to_grayscale(_to_torch(x), num_output_channels=3))
    np.testing.assert_allclose(tj, tt, atol=1e-5)


def test_normalize_parity():
    import torchvision.transforms.functional as F

    x = _imgs()
    tj = np.asarray(augment.normalize(x))
    tt = _to_nhwc(
        F.normalize(_to_torch(x), augment.IMAGENET_MEAN, augment.IMAGENET_STD)
    )
    np.testing.assert_allclose(tj, tt, atol=1e-5)


def test_adjust_brightness_parity():
    import torchvision.transforms.functional as F

    x = _imgs()
    tj = np.asarray(augment.adjust_brightness(x, 1.4))
    tt = _to_nhwc(F.adjust_brightness(_to_torch(x), 1.4))
    np.testing.assert_allclose(tj, tt, atol=1e-5)


def test_adjust_contrast_parity():
    import torchvision.transforms.functional as F

    x = _imgs()
    tj = np.asarray(augment.adjust_contrast(x, 0.7))
    tt = _to_nhwc(F.adjust_contrast(_to_torch(x), 0.7))
    np.testing.assert_allclose(tj, tt, atol=1e-4)


def test_exact_crop_parity():
    """Size == crop size with an integral box must be an exact crop (== F.crop)."""
    import torchvision.transforms.functional as F

    x = _imgs(b=1, s=16)
    top, left, h, w = 3, 4, 8, 8
    tj = np.asarray(augment.crop_resize_image(x[0], top, left, h, w, size=h))
    tt = _to_nhwc(F.crop(_to_torch(x), top, left, h, w))[0]
    np.testing.assert_allclose(tj, tt, atol=1e-5)


def test_resized_crop_parity_loose():
    """Crop + bilinear resize is close to torchvision (interpolation differs at edges)."""
    import torchvision.transforms.functional as F
    from torchvision.transforms import InterpolationMode

    x = _imgs(b=1, s=24)
    top, left, h, w, size = 2, 3, 16, 16, 12
    tj = np.asarray(augment.crop_resize_image(x[0], top, left, h, w, size))
    tt = _to_nhwc(
        F.resized_crop(
            _to_torch(x), top, left, h, w, [size, size], InterpolationMode.BILINEAR
        )
    )[0]
    # Boundary handling differs between map_coordinates and torchvision resize.
    assert np.abs(tj - tt).mean() < 0.05
