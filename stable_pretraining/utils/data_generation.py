"""Sample generation utilities for SSL experiments."""

from multiprocessing import Pool

import numpy as np
import torch
import tqdm
from torchvision.transforms import v2

from ..backbone.nn_modules import Normalize


def _apply_inet_transforms(x):
    """Apply ImageNet-style data augmentations to an image.

    Args:
        x: Input image

    Returns:
        Transformed image tensor
    """
    transform = v2.Compose(
        [
            v2.RGB(),
            v2.RandomResizedCrop(size=(224, 224), antialias=True, scale=(0.2, 0.99)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            Normalize(),
        ]
    )
    return transform(x)


def generate_dae_samples(x, n, eps, num_workers=10):
    """Generate samples for Denoising Autoencoder (DAE) training.

    Args:
        x: List of input images
        n: Number of noisy versions per image
        eps: Noise level (variance)
        num_workers: Number of parallel workers

    Returns:
        Tuple of (noisy_images, similarity_matrix)
    """
    with Pool(num_workers) as p:
        x = list(tqdm.tqdm(p.imap(_apply_inet_transforms, x), total=len(x)))
    x = torch.stack(x, 0)
    xtile = torch.repeat_interleave(x, n, dim=0)
    G = xtile.flatten(1).matmul(xtile.flatten(1).T)
    xtile.add_(torch.randn_like(xtile).mul_(torch.sqrt(torch.Tensor([eps]))))
    return xtile, G


def generate_sup_samples(x, y, n, num_workers=10):
    """Generate samples for supervised learning with class structure.

    Only includes classes with at least n samples.

    Args:
        x: List of input images
        y: Class labels
        n: Minimum samples per class
        num_workers: Number of parallel workers

    Returns:
        Tuple of (processed_images, class_similarity_matrix)
    """
    values, counts = np.unique(y, return_counts=True)

    values = values[counts >= n]
    values = np.flatnonzero(np.isin(y, values))
    ys = np.argsort(y[values])
    y = y[values[ys]]
    x = [x[i] for i in values[ys]]

    with Pool(num_workers) as p:
        x = list(tqdm.tqdm(p.imap(_apply_inet_transforms, x), total=len(x)))
    x = torch.stack(x, 0)
    ytile = torch.nn.functional.one_hot(
        torch.from_numpy(y), num_classes=int(np.max(y) + 1)
    )
    G = ytile.flatten(1).matmul(ytile.flatten(1).T)
    return x, G


def generate_dm_samples(x, n, betas, i, num_workers=10):
    """Generate samples for Diffusion Model training.

    Args:
        x: List of input images
        n: Number of noisy versions per timestep
        betas: Noise schedule beta values
        i: Timestep indices to use
        num_workers: Number of parallel workers

    Returns:
        Tuple of (noisy_images, similarity_matrix)
    """
    with Pool(num_workers) as p:
        x = list(tqdm.tqdm(p.imap(_apply_inet_transforms, x), total=len(x)))
    x = torch.stack(x, 0)
    if not torch.is_tensor(betas):
        betas = torch.Tensor(betas)
    alphas = torch.cumprod(1 - betas, 0)
    xtile = torch.repeat_interleave(x, n * len(i), dim=0)
    alphas = torch.repeat_interleave(alphas[i], n).repeat(x.size(0))

    xtile.mul_(alphas.reshape(-1, 1, 1, 1).sqrt().expand_as(xtile))
    G = xtile.flatten(1).matmul(xtile.flatten(1).T)
    eps = (1 - alphas.reshape(-1, 1, 1, 1)).sqrt().expand_as(xtile)
    xtile.add_(torch.randn_like(xtile).mul_(eps))
    return xtile, G


def generate_ssl_samples(x, n, num_workers=10):
    """Generate augmented samples for self-supervised learning.

    Creates n augmented versions of each image.

    Args:
        x: List of input images
        n: Number of augmented versions per image
        num_workers: Number of parallel workers

    Returns:
        Tuple of (augmented_images, similarity_matrix)
    """
    G = torch.kron(torch.eye(len(x)), torch.ones((n, n)))
    xtile = sum([[x[i] for _ in range(n)] for i in range(len(x))], [])
    with Pool(num_workers) as p:
        xtile = list(tqdm.tqdm(p.imap(_apply_inet_transforms, xtile), total=len(xtile)))
    return xtile, G
