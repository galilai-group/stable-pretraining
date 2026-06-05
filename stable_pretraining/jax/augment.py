"""Native JAX (jnp) image augmentations — the on-device alternative to kornia.

kornia powers GPU augmentation on the *torch* backend, but it operates on
``torch.Tensor`` / ``nn.Module`` and cannot run inside a JAX program. These
functions are pure ``jnp`` and ``jit``/``vmap``-friendly, so augmentation runs
on the *same* accelerator as the model and keeps the GPUs fed (the CPU
torchvision pipeline was the throughput bottleneck in the ImageNette run).

The module is split into:

* **Deterministic cores** (:func:`hflip`, :func:`rgb_to_grayscale`,
  :func:`adjust_brightness`, :func:`adjust_contrast`, :func:`normalize`,
  :func:`crop_resize_image`) — each matches ``torchvision.transforms.functional``
  for the same parameters, so they are regression-tested for parity.
* **Random wrappers** (:func:`random_horizontal_flip`, …) — sample parameters
  per image and call the cores.
* **Composition** (:func:`simclr_view`, :func:`two_crop`) and
  :func:`two_view_transform`, which populates ``batch["views"]`` with the two
  augmentation realisations — the same batch-dict convention as the torch
  ``MultiViewTransform``.

All ops take/return NHWC float images in ``[0, 1]`` and are batched.
"""

import jax
import jax.numpy as jnp

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# ITU-R 601-2 luma weights, matching torchvision.rgb_to_grayscale.
_GRAY_W = jnp.asarray([0.2989, 0.587, 0.114])


# --------------------------------------------------------------------------- #
# Deterministic cores (parity with torchvision.transforms.functional)
# --------------------------------------------------------------------------- #
def normalize(x: jnp.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> jnp.ndarray:
    """Channel-normalize NHWC images: ``(x - mean) / std``."""
    mean = jnp.asarray(mean).reshape(1, 1, 1, -1)
    std = jnp.asarray(std).reshape(1, 1, 1, -1)
    return (x - mean) / std


def hflip(x: jnp.ndarray) -> jnp.ndarray:
    """Flip images left-right (matches ``F.hflip``)."""
    return x[:, :, ::-1, :]


def rgb_to_grayscale(x: jnp.ndarray) -> jnp.ndarray:
    """3-channel luma grayscale (matches ``F.rgb_to_grayscale``)."""
    gray = jnp.sum(x * _GRAY_W.reshape(1, 1, 1, 3), axis=-1, keepdims=True)
    return jnp.broadcast_to(gray, x.shape)


def adjust_brightness(x: jnp.ndarray, factor) -> jnp.ndarray:
    """Brightness jitter: ``clamp(x * factor, 0, 1)`` (matches ``F.adjust_brightness``)."""
    return jnp.clip(x * factor, 0.0, 1.0)


def adjust_contrast(x: jnp.ndarray, factor) -> jnp.ndarray:
    """Contrast jitter toward the grayscale mean (matches ``F.adjust_contrast``).

    ``factor`` may be a scalar or per-image array broadcastable over ``[B,1,1,1]``.
    """
    mean = jnp.mean(rgb_to_grayscale(x), axis=(1, 2, 3), keepdims=True)
    return jnp.clip(x * factor + mean * (1.0 - factor), 0.0, 1.0)


def crop_resize_image(img, top, left, height, width, size):
    """Crop ``img`` ``[H,W,C]`` to a box then bilinearly resize to ``(size, size)``.

    Pixel-unit box. Implemented with continuous grid sampling so it composes
    under ``vmap`` with per-image boxes. When ``size == height == width`` and the
    box is integral, this is an exact crop (grid hits integer pixels).
    """
    c = img.shape[-1]
    # Half-pixel-center sampling (align_corners=False), matching torchvision's
    # bilinear resize: src = box_origin + (dst + 0.5) * box/out - 0.5.
    idx = jnp.arange(size)
    ys = top + (idx + 0.5) * (height / size) - 0.5
    xs = left + (idx + 0.5) * (width / size) - 0.5
    gy, gx = jnp.meshgrid(ys, xs, indexing="ij")
    cols = [
        jax.scipy.ndimage.map_coordinates(
            img[..., k], [gy, gx], order=1, mode="nearest"
        )
        for k in range(c)
    ]
    return jnp.stack(cols, axis=-1)


# --------------------------------------------------------------------------- #
# Random wrappers (per-image parameter sampling)
# --------------------------------------------------------------------------- #
def random_horizontal_flip(key, x: jnp.ndarray, p: float = 0.5) -> jnp.ndarray:
    """Flip each image left-right independently with probability ``p``."""
    flip = jax.random.bernoulli(key, p, (x.shape[0],))
    return jnp.where(flip[:, None, None, None], hflip(x), x)


def random_grayscale(key, x: jnp.ndarray, p: float = 0.2) -> jnp.ndarray:
    """Grayscale each image independently with probability ``p``."""
    apply = jax.random.bernoulli(key, p, (x.shape[0],))
    return jnp.where(apply[:, None, None, None], rgb_to_grayscale(x), x)


def random_brightness_contrast(key, x, brightness=0.4, contrast=0.4):
    """Per-image brightness then contrast jitter (the SimCLR color-jitter core)."""
    b = x.shape[0]
    kb, kc = jax.random.split(key)
    bf = 1.0 + jax.random.uniform(
        kb, (b, 1, 1, 1), minval=-brightness, maxval=brightness
    )
    cf = 1.0 + jax.random.uniform(kc, (b, 1, 1, 1), minval=-contrast, maxval=contrast)
    return adjust_contrast(adjust_brightness(x, bf), cf)


def random_resized_crop(key, x, size, scale=(0.2, 1.0)):
    """Random square-area crop per image, resized to ``(size, size)`` (bilinear)."""
    b, h, w, _ = x.shape
    ka, kt, kl = jax.random.split(key, 3)
    frac = jnp.sqrt(jax.random.uniform(ka, (b,), minval=scale[0], maxval=scale[1]))
    box_h = frac * h  # crop box size in pixels
    box_w = frac * w
    top = jax.random.uniform(kt, (b,)) * (h - box_h)
    left = jax.random.uniform(kl, (b,)) * (w - box_w)
    return jax.vmap(
        lambda im, t, lf, bh, bw: crop_resize_image(im, t, lf, bh, bw, size)
    )(x, top, left, box_h, box_w)


# --------------------------------------------------------------------------- #
# Composition + batch-dict transform
# --------------------------------------------------------------------------- #
def simclr_view(key, x, size, normalize_output=True):
    """One SimCLR view: resized-crop -> hflip -> brightness/contrast -> grayscale -> norm."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x = random_resized_crop(k1, x, size)
    x = random_horizontal_flip(k2, x)
    x = random_brightness_contrast(k3, x)
    x = random_grayscale(k4, x)
    return normalize(x) if normalize_output else x


def two_crop(key, x, size, normalize_output=True):
    """Return two independently augmented SimCLR views of ``x``."""
    k1, k2 = jax.random.split(key)
    return (
        simclr_view(k1, x, size, normalize_output),
        simclr_view(k2, x, size, normalize_output),
    )


def two_view_transform(size: int, normalize_output: bool = True):
    """Build a Module ``transform`` that populates ``batch["views"]`` on device.

    The returned callable ``(module, batch) -> batch`` reads ``batch["image"]``
    (raw NHWC ``[0, 1]``), draws a fresh key from ``module.aug_rngs``, generates
    two augmented views, and writes them back as ``batch["views"]`` — the same
    dict convention the torch ``MultiViewTransform`` uses, so the SimCLR/VICReg/…
    forward functions and all callbacks work unchanged.

    Args:
        size: Output crop side length.
        normalize_output: Apply ImageNet normalization to each view.

    Returns:
        Callable suitable for ``Module(transform=...)``.
    """

    def transform(module, batch: dict) -> dict:
        key = module.aug_rngs()
        v1, v2 = two_crop(key, batch["image"], size, normalize_output)
        view = {"label": batch["label"]} if "label" in batch else {}
        out = dict(batch)
        out["views"] = [{"image": v1, **view}, {"image": v2, **view}]
        return out

    return transform


__all__ = [
    "normalize",
    "hflip",
    "rgb_to_grayscale",
    "adjust_brightness",
    "adjust_contrast",
    "crop_resize_image",
    "random_horizontal_flip",
    "random_grayscale",
    "random_brightness_contrast",
    "random_resized_crop",
    "simclr_view",
    "two_crop",
    "two_view_transform",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
