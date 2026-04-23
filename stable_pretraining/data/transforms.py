from itertools import islice
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision
from PIL import ImageFilter
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._utils import query_chw
from PIL import Image

from stable_pretraining.data.masking import multi_block_mask


class Transform(v2.Transform):
    """Base transform class extending torchvision v2.Transform with nested data handling."""

    def single_nested_get(self, v, name):
        if name == "":
            return v
        i = name.split(".")
        if i[0].isnumeric():
            i[0] = int(i[0])
        return self.single_nested_get(v[i[0]], ".".join(i[1:]))

    def nested_get(self, v, name):
        if type(name) in [list, tuple]:
            return [self.single_nested_get(v, n) for n in name]
        return self.single_nested_get(v, name)

    def single_nested_set(self, original, value, name):
        if "." not in name:
            if name.isnumeric():
                name = int(name)
            original[name] = value
        else:
            i = name.split(".")
            if i[0].isnumeric():
                i[0] = int(i[0])
            self.single_nested_set(original[i[0]], value, ".".join(i[1:]))

    def nested_set(self, original, value, name):
        if type(name) in [list, tuple]:
            assert type(value) in [list, tuple]
            assert len(value) == len(name)
            return [self.single_nested_set(original, v, n) for v, n in zip(value, name)]
        return self.single_nested_set(original, value, name)

    def get_name(self, x):
        base = self.name
        assert "_" not in base
        if base not in x:
            return base
        ctr = 0
        while f"{base}_{ctr}" in base:
            ctr += 1
        return f"{base}_{ctr}"

    @property
    def name(self):
        return self.__class__.__name__


@torch.jit.unused
def to_image(
    input: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
) -> tv_tensors.Image:
    """See :class:`~torchvision.transforms.v2.ToImage` for details."""
    if isinstance(input, np.ndarray):
        output = torch.from_numpy(np.atleast_3d(input)).transpose(-3, -1).contiguous()
    elif isinstance(input, PIL.Image.Image):
        output = torchvision.transforms.functional.pil_to_tensor(input)
    elif isinstance(input, torch.Tensor):
        output = input
    else:
        raise TypeError(
            f"Input can either be a pure Tensor, a numpy array, or a PIL image, but got {type(input)} instead."
        )
    return tv_tensors.Image(output)


class ToImage(Transform):
    """Convert input to image tensor with optional normalization."""

    def __init__(
        self,
        dtype=torch.float32,
        scale=True,
        mean=None,
        std=None,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__()
        t = [to_image, v2.ToDtype(dtype, scale=scale)]
        if mean is not None and std is not None:
            t.append(v2.Normalize(mean=mean, std=std))
        self.t = v2.Compose(t)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(x, self.t(self.nested_get(x, self.source)), self.target)
        return x


class RandomGrayscale(Transform, v2.RandomGrayscale):
    """Randomly convert image to grayscale with given probability."""

    def __init__(self, p=0.1, source: str = "image", target: str = "image"):
        super().__init__(p)
        self.source = source
        self.target = target

    def _get_params(self, inp: List[Any]) -> Dict[str, Any]:
        num_input_channels, *_ = query_chw([inp])
        return dict(num_input_channels=num_input_channels)

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = False
            self.nested_set(x, self.nested_get(x, self.source), self.target)
            return x
        channels, *_ = query_chw([self.nested_get(x, self.source)])
        self.nested_set(
            x,
            F.rgb_to_grayscale(
                self.nested_get(x, self.source), num_output_channels=channels
            ),
            self.target,
        )
        x[self.get_name(x)] = True
        return x


class Lambda(Transform):
    """Applies a lambda callable to target key and store it in source."""

    def __init__(self, lambd, source: str = "image", target: str = "image"):
        super().__init__()
        self.source = source
        self.target = target
        self.lambd = lambd

    def __call__(self, x) -> Any:
        self.nested_set(x, self.lambd(x), self.target)
        return x


class RoutingTransform(Transform):
    """Applies a routing callable to conditionally apply a transform from many candidates."""

    def __init__(self, router: callable, transforms: Union[list, tuple, dict]):
        self.router = router
        self.transforms = transforms

    def __call__(self, x) -> Any:
        route = self.router(x)
        return self.transforms[route](x)


class WrapTorchTransform(Transform, v2.Lambda):
    """Applies a lambda callable to target key and store it in source."""

    def __init__(self, transform, source: str = "image", target: str = "image"):
        super().__init__(transform)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        self.nested_set(
            x, super().__call__(self.nested_get(x, self.source)), self.target
        )
        return x


class RandomSolarize(Transform, v2.RandomSolarize):
    """Randomly solarize image by inverting pixel values above threshold."""

    def __init__(self, threshold, p=0.5, source: str = "image", target: str = "image"):
        super().__init__(threshold, p)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = False
            return x
        self.nested_set(
            x, F.solarize(self.nested_get(x, self.source), self.threshold), self.target
        )
        x[self.get_name(x)] = True
        return x


class RandomAutocontrast(Transform, v2.RandomAutocontrast):
    """Randomly apply autocontrast to the image."""

    def __init__(self, p=0.5, source: str = "image", target: str = "image"):
        super().__init__(p)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = False
            return x
        self.nested_set(x, F.autocontrast(self.nested_get(x, self.source)), self.target)
        x[self.get_name(x)] = True
        return x


class RandomEqualize(Transform, v2.RandomEqualize):
    """Randomly equalize the histogram of the image."""

    def __init__(self, p=0.5, source: str = "image", target: str = "image"):
        super().__init__(p)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = False
            return x
        self.nested_set(x, F.equalize(self.nested_get(x, self.source)), self.target)
        x[self.get_name(x)] = True
        return x


class RandomInvert(Transform, v2.RandomInvert):
    """Randomly invert the colors of the image."""

    def __init__(self, p=0.5, source: str = "image", target: str = "image"):
        super().__init__(p)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = False
            return x
        self.nested_set(x, F.invert(self.nested_get(x, self.source)), self.target)
        x[self.get_name(x)] = True
        return x


class RandomPosterize(Transform, v2.RandomPosterize):
    """Randomly posterize the image by reducing bits per channel."""

    def __init__(self, bits, p=0.5, source: str = "image", target: str = "image"):
        super().__init__(bits, p)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = torch.Tensor([0])
            return x
        self.nested_set(
            x, F.posterize(self.nested_get(x, self.source), self.bits), self.target
        )
        x[self.get_name(x)] = torch.Tensor([self.bits])
        return x


class RandomAdjustSharpness(Transform, v2.RandomAdjustSharpness):
    """Randomly adjust the sharpness of the image."""

    def __init__(
        self, sharpness_factor, p=0.5, source: str = "image", target: str = "image"
    ):
        super().__init__(sharpness_factor, p)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = torch.Tensor([0.0])
            return x
        self.nested_set(
            x,
            F.adjust_sharpness(self.nested_get(x, self.source), self.sharpness_factor),
            self.target,
        )
        x[self.get_name(x)] = torch.Tensor([self.sharpness_factor])
        return x


class RandomVerticalFlip(Transform, v2.RandomVerticalFlip):
    """Vertically flip the given image randomly with a given probability."""

    def __init__(self, p=0.5, source: str = "image", target: str = "image"):
        super().__init__(p)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p > 0 and torch.rand(1) < self.p:
            candidates = self.nested_get(x, self.source)
            if type(candidates) in [tuple, list]:
                out = [F.vertical_flip(c) for c in candidates]
                self.nested_set(x, out, self.target)
            else:
                self.nested_set(x, F.vertical_flip(candidates), self.target)
            x[self.get_name(x)] = True
        else:
            self.nested_set(x, self.nested_get(x, self.source), self.target)
            x[self.get_name(x)] = False
        return x


class RandomErasing(Transform, v2.RandomErasing):
    """Randomly erase a rectangular region in the image."""

    _NAMES = ["i", "j", "h", "w"]

    def __init__(
        self,
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=0,
        inplace=False,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(p, scale, ratio, value, inplace)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = torch.zeros(4)
            return x
        params = self.make_params([self.nested_get(x, self.source)])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), params), self.target
        )
        x[self.get_name(x)] = torch.Tensor(
            [params["i"], params["j"], params["h"], params["w"]]
        )
        return x


class RandomAffine(Transform, v2.RandomAffine):
    """Randomly apply affine transformation to the image."""

    _NAMES = ["angle", "translate_x", "translate_y", "scale", "shear_x", "shear_y"]

    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        interpolation=InterpolationMode.NEAREST,
        fill=0,
        center=None,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(degrees, translate, scale, shear, interpolation, fill, center)
        self.source = source
        self.target = target

    def __call__(self, x):
        params = self.make_params([self.nested_get(x, self.source)])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), params), self.target
        )
        x[self.get_name(x)] = torch.Tensor(
            [
                params["angle"],
                *params["translate"],
                params["scale"],
                *params["shear"],
            ]
        )
        return x


class RandomPerspective(Transform, v2.RandomPerspective):
    """Randomly apply perspective transformation to the image."""

    def __init__(
        self,
        distortion_scale=0.5,
        p=0.5,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(distortion_scale, p, interpolation, fill)
        self.source = source
        self.target = target

    def __call__(self, x):
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = torch.zeros(8)
            return x
        params = self.make_params([self.nested_get(x, self.source)])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), params), self.target
        )
        coeffs = params["coefficients"]
        x[self.get_name(x)] = torch.Tensor(coeffs)
        return x


class GaussianNoise(Transform, v2.GaussianNoise):
    """Add Gaussian noise to the image (torchvision implementation)."""

    def __init__(
        self, mean=0.0, sigma=0.1, p=0.5, source: str = "image", target: str = "image"
    ):
        super().__init__(mean, sigma)
        self.p = p
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = torch.Tensor([0.0, 0.0])
            return x
        params = self.make_params([self.nested_get(x, self.source)])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), params), self.target
        )
        x[self.get_name(x)] = torch.Tensor([self.mean, self.sigma])
        return x


class GaussianBlur(Transform, v2.GaussianBlur):
    """Apply Gaussian blur to image with random sigma values."""

    _NAMES = ["sigma_x", "sigma_y"]

    def __init__(
        self,
        kernel_size,
        sigma=(0.1, 2.0),
        p=1,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(kernel_size, sigma)
        self.p = p
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = torch.zeros((2,))
            return x
        params = self.make_params([])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), params), self.target
        )
        x[self.get_name(x)] = torch.Tensor(params["sigma"])
        return x


class PILGaussianBlur(Transform):
    """PIL-based Gaussian blur transform with random sigma sampling."""

    _NAMES = ["sigma_x", "sigma_y"]

    def __init__(self, sigma=None, p=1, source: str = "image", target: str = "image"):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
            p (float): probability of applying the transform.
            source (str): source key in the data dictionary.
            target (str): target key in the data dictionary.
        """
        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma
        self.p = p
        self.source = source
        self.target = target

    def __call__(self, x):
        """Applies gaussian blur to an input image.

        Args:
            x (dict): Data dictionary containing the image to transform.

        Returns:
            dict: Data dictionary with blurred image.
        """
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = torch.zeros((1,))
            return x
        sigma = torch.rand((1,)) * (self.sigma[1] - self.sigma[0]) + self.sigma[0]
        x[self.get_name(x)] = sigma
        self.nested_set(
            x,
            self.nested_get(x, self.source).filter(
                ImageFilter.GaussianBlur(radius=sigma.item())
            ),
            self.target,
        )
        return x


class UniformTemporalSubsample(Transform):
    """``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_temporal_subsample``."""

    def __init__(
        self,
        num_samples: int,
        temporal_dim: int = -3,
        source: str = "video",
        target: str = "video",
    ):
        super().__init__(num_samples, temporal_dim)
        self.source = source
        self.target = target

    def forward(self, x: dict) -> torch.Tensor:
        self.nested_set(
            x, super().forward(self, self.nested_get(x, self.source)), self.target
        )
        return x


class RandomContiguousTemporalSampler(Transform):
    """Randomly sample contiguous frames from a video sequence."""

    def __init__(self, source, target, num_frames, frame_subsampling: int = 1):
        self.source = source
        self.target = target
        self.num_frames = num_frames
        self.frame_subsampling = frame_subsampling

    def __call__(self, x):
        metadata = self.nested_get(x, self.source).get_metadata()
        T = int(metadata["video"]["duration"][0] * metadata["video"]["fps"][0])
        covering = self.num_frames * self.frame_subsampling
        start = torch.randint(low=0, high=T - covering, size=(1,)).item()
        video_frames = []  # video frame buffer

        # Seek and return frames
        count = 0
        for frame in islice(
            self.nested_get(x, self.source).seek(start / metadata["video"]["fps"][0]),
            covering,
        ):
            if count % self.frame_subsampling == 0:
                video_frames.append(frame["data"])
            count += 1
        # Stack it into a tensor
        self.nested_set(x, torch.stack(video_frames, 0), self.target)
        x[self.get_name(x)] = start
        return x


class RGB(Transform, v2.RGB):
    """Convert image to RGB format."""

    def __init__(self, source: str = "image", target: str = "image"):
        super().__init__()
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(
            x, F.grayscale_to_rgb(self.nested_get(x, self.source)), self.target
        )
        return x


class Resize(Transform, v2.Resize):
    """Resize image to specified size."""

    def __init__(
        self,
        size,
        interpolation=2,
        max_size=None,
        antialias=True,
        source="image",
        target="image",
    ) -> None:
        super().__init__(size, interpolation, max_size, antialias)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), []), self.target
        )
        return x


class ColorJitter(Transform, v2.ColorJitter):
    """Randomly change brightness, contrast, saturation, and hue of an image."""

    def __init__(
        self,
        brightness=None,
        contrast=None,
        saturation=None,
        hue=None,
        p=1,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) > self.p:
            self.nested_set(x, self.nested_get(x, self.source), self.target)
            x[self.get_name(x)] = torch.zeros(8)
            return x
        params = self.make_params([])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), params), self.target
        )
        brightness_factor = params["brightness_factor"]
        contrast_factor = params["contrast_factor"]
        saturation_factor = params["saturation_factor"]
        hue_factor = params["hue_factor"]
        perm = params["fn_idx"].tolist()
        x[self.get_name(x)] = torch.Tensor(
            [brightness_factor, contrast_factor, saturation_factor, hue_factor] + perm
        )
        return x


class RandomRotation(Transform, v2.RandomRotation):
    """Rotate image by random angle within specified degrees range."""

    def __init__(
        self,
        degrees,
        interpolation=InterpolationMode.NEAREST,
        expand=False,
        center=None,
        fill=0,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(degrees, interpolation, expand, center, fill)
        self.source = source
        self.target = target

    def __call__(self, x):
        angle = self.make_params([])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), angle), self.target
        )
        x[self.get_name(x)] = angle
        return x


class RandomChannelPermutation(Transform, v2.RandomChannelPermutation):
    """Randomly permute the channels of an image."""

    def __init__(self, source: str = "image", target: str = "image"):
        super().__init__()
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        num_channels, *_ = query_chw([self.nested_get(x, self.source)])
        perm = torch.randperm(num_channels)
        self.nested_set(
            x, F.permute_channels(self.nested_get(x, self.source), perm), self.target
        )
        x[self.get_name(x)] = perm
        return x


class RandomCrop(Transform, v2.RandomCrop):
    """Crop a random portion of image and resize it to given size."""

    _NAMES = ["needs_crop", "top", "left", "height", "width", "needs_pad", "padding"]

    def __init__(
        self,
        size,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.source = source
        self.target = target

    def __call__(self, x):
        params = self.make_params([self.nested_get(x, self.source)])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), params), self.target
        )
        values = []
        values.append(params["needs_crop"])
        values.append(params["top"])
        values.append(params["left"])
        values.append(params["height"])
        values.append(params["width"])
        values.append(params["needs_pad"])
        values.extend(params["padding"])
        x[self.get_name(x)] = torch.Tensor(values)
        return x


class RandomHorizontalFlip(Transform, v2.RandomHorizontalFlip):
    """Horizontally flip the given image randomly with a given probability."""

    def __init__(self, p=0.5, source: str = "image", target: str = "image"):
        super().__init__(p)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p > 0 and torch.rand(1) < self.p:
            candidates = self.nested_get(x, self.source)
            if type(candidates) in [tuple, list]:
                out = [F.horizontal_flip(c) for c in candidates]
                self.nested_set(x, out, self.target)
            else:
                self.nested_set(x, F.horizontal_flip(candidates), self.target)
            x[self.get_name(x)] = True
        else:
            self.nested_set(x, self.nested_get(x, self.source), self.target)
            x[self.get_name(x)] = False
        return x


class RandomResizedCrop(Transform, v2.RandomResizedCrop):
    """Crop a random portion of image and resize it to given size."""

    _NAMES = ["top", "left", "height", "width"]

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(size, scale, ratio, interpolation, antialias)
        self.source = source
        self.target = target

    def __call__(self, x):
        params = self.make_params([self.nested_get(x, self.source)])

        candidates = self.nested_get(x, self.source)
        if type(candidates) in [tuple, list]:
            out = [self.transform(c, params) for c in candidates]
            self.nested_set(x, out, self.target)
        else:
            self.nested_set(
                x, self.transform(self.nested_get(x, self.source), params), self.target
            )
        values = []
        values.append(params["top"])
        values.append(params["left"])
        values.append(params["height"])
        values.append(params["width"])
        x[self.get_name(x)] = torch.Tensor(values)
        return x


class PatchMasking(Transform):
    """Randomly masks square patches in an image, similar to patch masking used in Masked Signal Encoding (MSE) tasks.

    This transform operates on a dictionary input, applies patch masking to the image found at the specified `source` key,
    and writes the masked image to the `target` key. It also saves a boolean mask matrix (one entry per patch) to the
    `mask_key` in the dictionary, indicating which patches were masked (False) or kept (True).
    The output image remains in the same format as the input (PIL Image or Tensor), and the masking is performed efficiently
    for both input types.

    Args:
        patch_size (int): The size (in pixels) of each square patch to be masked.
        drop_ratio (float): The exact fraction of patches to randomly mask (set to the mask value).
        source (str): The key in the input dictionary from which to read the image.
        target (str): The key in the output dictionary to which the masked image will be written.
        mask_key (str): The key in the output dictionary to which the boolean patch mask will be written.
        fill_value (float, optional): The value to use for masked patches. If None, defaults to 0.0 for float tensors,
            and 128/255.0 for PIL images (mid-gray). Can be set to any float in [0,1] for normalized images.
    """

    def __init__(
        self,
        patch_size: int = 16,
        drop_ratio: float = 0.5,
        source: str = "image",
        target: str = "image",
        fill_value: float = None,
        mask_key: str = "patch_mask",
    ):
        super().__init__()
        if not 0.0 <= drop_ratio <= 1.0:
            raise ValueError(f"drop_ratio must be in [0, 1], got {drop_ratio}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        self.mask_key = mask_key
        self.patch_size = patch_size
        self.drop_ratio = drop_ratio
        self.source = source
        self.target = target
        self.fill_value = fill_value

    def __call__(self, x):
        img = self.nested_get(x, self.source)
        img_tensor = self._to_tensor(img)
        C, H, W = img_tensor.shape

        # Compute number of patches
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        total_patches = n_patches_h * n_patches_w

        # Generate mask with EXACT drop ratio (not probabilistic)
        n_masked = int(total_patches * self.drop_ratio)
        perm = torch.randperm(total_patches)
        mask_flat = torch.ones(total_patches, dtype=torch.bool)
        mask_flat[perm[:n_masked]] = False  # False = masked
        mask = mask_flat.reshape(n_patches_h, n_patches_w)

        # Determine fill value
        fill_value = self.fill_value if self.fill_value is not None else 0.0
        fill = torch.tensor(
            fill_value, dtype=img_tensor.dtype, device=img_tensor.device
        )

        # Efficient masking via reshape + broadcast (no full-resolution mask needed)
        ps = self.patch_size
        ph, pw = n_patches_h * ps, n_patches_w * ps
        # Reshape patchable region to (C, n_h, ps, n_w, ps) and broadcast small mask
        patches = img_tensor[:, :ph, :pw].reshape(C, n_patches_h, ps, n_patches_w, ps)
        patch_mask = mask[None, :, None, :, None]  # (1, n_h, 1, n_w, 1)
        masked_patches = torch.where(patch_mask, patches, fill)

        # Reconstruct image (keep remainder pixels unchanged)
        if ph == H and pw == W:
            masked_img_out = masked_patches.reshape(C, H, W)
        else:
            masked_img_out = img_tensor.clone()
            masked_img_out[:, :ph, :pw] = masked_patches.reshape(C, ph, pw)

        self.nested_set(x, masked_img_out, self.target)
        self.nested_set(x, mask, self.mask_key)
        return x

    @staticmethod
    def _to_tensor(img):
        if isinstance(img, torch.Tensor):
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            if img.ndim == 3:
                return img
            elif img.ndim == 2:
                return img.unsqueeze(0)
        elif isinstance(img, Image.Image):
            return F.pil_to_tensor(img).float() / 255.0
        else:
            raise TypeError("Unsupported image type")


class CenterCrop(Transform, v2.CenterCrop):
    """Crop the center of an image to the given size."""

    _NAMES = []

    def __init__(self, size, source: str = "image", target: str = "image"):
        super().__init__(size)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), []), self.target
        )
        return x


class random_seed:
    """Context manager that forks the torch RNG, seeds it, and restores on exit.

    Only forks the torch RNG since all transforms use torch random ops.
    Avoids the overhead of saving/restoring Python and NumPy RNG states
    (~7.5KB copied per call previously).
    """

    def __init__(self, seed):
        self.seed = int(seed)
        self._fork = torch.random.fork_rng(enabled=True)

    def __enter__(self):
        self._fork.__enter__()
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *args):
        return self._fork.__exit__(*args)


class ControlledTransform(Transform):
    """Face Landmarks dataset."""

    def __init__(
        self, transform: callable, seed_offset: int = 0, key: Optional[str] = "idx"
    ):
        super().__init__()
        self.seed_offset = seed_offset
        self._transform = transform
        self.key = key

    def __call__(self, x):
        with random_seed(x["idx"] + self.seed_offset):
            x = self._transform(x)
        return x


class Conditional(Transform):
    """Apply transform conditionally based on a data dictionary key."""

    def __init__(self, transform, condition_key, apply_on_true=True):
        super().__init__()
        self._transform = transform
        self.condition_key = condition_key
        self.apply_on_true = apply_on_true

    def __call__(self, x):
        if x[self.condition_key] and self.apply_on_true:
            return self._transform(x)
        elif not x[self.condition_key] and not self.apply_on_true:
            return self._transform(x)
        # if the transform is not applied we still inform the user
        # otherwise collate_fn will complain
        x[self._transform.get_name(x)] = self._transform.BYPASS_VALUE
        return x


class AdditiveGaussian(Transform):
    """Add Gaussian noise to input data."""

    BYPASS_VALUE = False

    def __init__(self, sigma, p=1):
        super().__init__()
        if not torch.is_tensor(sigma):
            sigma = torch.Tensor([sigma])[0]
        self.sigma = sigma
        self.p = p

    def __call__(self, x):
        if self.p == 0 or self.p < torch.rand(1):
            x[self.get_name(x)] = self.BYPASS_VALUE
            return x
        x[self.get_name(x)] = True
        out = torch.randn_like(x["image"]).mul_(self.sigma)
        x["image"] = x["image"].add_(out)
        return x


class Compose(v2.Transform):
    """Compose multiple transforms together in sequence."""

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def __call__(self, sample):
        for a in self.args:
            sample = a(sample)
        return sample


class RoundRobinMultiViewTransform(v2.Transform):
    """Round-robin multi-view transform that cycles through transforms using a counter.

    IMPORTANT: This transform is designed to work with RepeatedRandomSampler, where
    each image index appears multiple times consecutively in the batch. It uses an
    internal counter to apply different augmentations to each repeated occurrence.

    BATCH SIZE NOTE: When using this with RepeatedRandomSampler, the batch_size
    parameter refers to the total number of augmented samples, NOT the number of
    unique images. For example, with batch_size=256 and n_views=2, you get 128
    unique images, each appearing twice with different augmentations.

    How it works:
    1. RepeatedRandomSampler produces indices like [0,0,1,1,2,2,...] (for n_views=2)
    2. DataLoader loads the same image multiple times
    3. This transform applies a different augmentation each time using round-robin

    Args:
        transforms: List of transforms, one for each view. The counter cycles
                   through these transforms in order.

    Example:
        # With RepeatedRandomSampler(dataset, n_views=2)
        transform = RoundRobinMultiViewTransform([
            strong_augmentation,  # Applied to 1st occurrence of each image
            weak_augmentation,    # Applied to 2nd occurrence of each image
        ])

    Warning: The internal counter makes this transform stateful and not thread-safe.
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
        self.n_transforms = len(transforms)
        self.counter = 0

    def __call__(self, sample):
        # Use round-robin to apply transforms
        transform_idx = self.counter % self.n_transforms
        self.counter += 1
        return self.transforms[transform_idx](sample)


class MultiViewTransform(v2.Transform):
    """Creates multiple views from one sample by applying different transforms.

    Takes a single sample and applies different transforms to create multiple
    views, returning a list of complete sample dicts. Preserves all modifications
    each transform makes (masks, augmentation params, metadata, etc.).

    Implementation Note:
        This transform uses shallow copy (dict.copy()) for the input sample before
        applying each transform. This is efficient and safe because:
        - The shallow copy shares references to the original tensors/objects
        - Standard transforms create NEW tensors (e.g., through mul(), resize(),
          crop()) rather than modifying inputs in-place
        - The original sample remains unchanged

    Consequences of shallow copy:
        - Memory efficient: Original tensors are not duplicated unnecessarily
        - Safe with torchvision transforms: All torchvision transforms and our
          custom transforms follow the pattern of creating new tensors
        - Caution: If using custom transforms that modify tensors in-place (using
          operations like mul_(), add_() with underscore), views may interfere with
          each other. Always use non-in-place operations in custom transforms.

    Args:
        transforms: Either a list or dict of transforms.
                   - List: Returns a list of views in the same order
                   - Dict: Returns a dict of views with the same keys

    Returns:
        Dict[str, Union[Dict, List[Dict]]:
            - If transforms is a list: Returns a dict mapping the key "views" to a list of transformed sample dicts
            - If transforms is a dict: Returns a dict of transformed sample dicts with the corresponding keys mapping to the samples.
            Each dict contains NEW tensors, not references to the original.

    Example:
    # List input - returns dict with key "views" mapping to a list of views
    transform = MultiViewTransform([
        strong_augmentation,  # Creates first view with strong aug
        weak_augmentation,    # Creates second view with weak aug
    ])
    # Input: {"image": img, "label": 0}
    # Output: {"views": [{"image": img_strong, "label": 0}, {"image": img_weak, "label": 0}]}

    # Dict input - returns dict of named views
    transform = MultiViewTransform({
        "student": strong_augmentation,
        "teacher": weak_augmentation,
    })
    # Input: {"image": img, "label": 0}
    # Output: {"student": {"image": img_strong, "label": 0},
    #          "teacher": {"image": img_weak, "label": 0}}
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
        self.is_keys_specified = isinstance(transforms, dict)

    def __call__(self, sample):
        """Create multiple views by applying different transforms to the sample."""
        if self.is_keys_specified:
            # Dict input - return dict of views
            views = {}
            for key, transform in self.transforms.items():
                # Copy to avoid transforms modifying the original
                sample_copy = sample.copy()
                # Apply transform to entire dict
                transformed = transform(sample_copy)
                views[key] = transformed
        else:
            # List input - return list of views
            views_list = []
            for transform in self.transforms:
                # Copy to avoid transforms modifying the original
                sample_copy = sample.copy()
                # Apply transform to entire dict
                transformed = transform(sample_copy)
                views_list.append(transformed)

            views = {"views": views_list}

        return views


class ContextTargetsMultiBlockMask(Transform):
    """Transform that adds multi-block masks to batch, with multiple target blocks and one disjoint context block.

    Args:
        patch_size: Size of the patch in patches
        num_blocks: Number of blocks to sample
        context_scale: Scale of the context block
        aspect_ratio: Aspect ratio of the blocks
        min_keep: Minimum number of patches that must be in the block

    """

    def __init__(
        self,
        patch_size=16,
        context_scale=(0.85, 1.0),
        context_aspect_ratio=(1.0, 1.0),
        target_scales=((0.15, 0.2),) * 4,
        target_aspect_ratios=((0.75, 1.5),) * 4,
        min_keep=10,
        source: str = "image",
        target_context: str = "mask_context",
        target_targets: str = "masks_target",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.context_scale = context_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.target_scales = target_scales
        self.target_aspect_ratios = target_aspect_ratios
        self.source = source
        self.target_context = target_context
        self.target_targets = target_targets
        if len(target_scales) != len(target_aspect_ratios):
            raise ValueError(
                "Each scale must have its associated aspect ratio and vice versa.",
                f"Received {len(target_scales)=} {len(target_aspect_ratios)=}",
            )

        self.min_keep = min_keep

    def __call__(self, x):
        source = self.nested_get(x, self.source)
        if isinstance(source, PIL.Image.Image):
            W, H = source.size  # PIL is W,H
        elif isinstance(source, torch.Tensor):
            # assumes H W
            H, W = source.shape[-2:]
        else:
            raise ValueError(
                f"Source must be a PIL.Image.Image or a torch.Tensor, but got {type(source)} instead."
            )

        scales = [self.context_scale, *self.target_scales]
        aspect_ratios = [self.context_aspect_ratio, *self.target_aspect_ratios]
        context_mask, *target_masks = multi_block_mask(
            H // self.patch_size,
            W // self.patch_size,
            block_scales=scales,
            aspect_ratios=aspect_ratios,
            min_keep=self.min_keep,
        )
        # makes targets disjoint with context
        for mask in target_masks:
            context_mask &= ~mask

        x[self.target_context] = torch.nonzero(context_mask.flatten()).squeeze()
        x[self.target_targets] = [
            torch.nonzero(mask.flatten()).squeeze() for mask in target_masks
        ]
        x[self.get_name(x)] = torch.tensor([scales, aspect_ratios])
        return x


class RandomMask(Transform):
    r"""Creates a random MAE-style mask for an image.

    This transform generates a random permutation of all patch indices for an
    input image. It then splits these indices into two disjoint sets:
    'visible' and 'masked', according to the specified `mask_ratio`.

    It also provides an `ids_restore` tensor, which can un-shuffle a sequence
    of patches back to its original 2D grid order. All outputs are added as
    new keys to the sample dictionary.

    Example:
        >>> # xdoctest: +SKIP
        >>> transform = RandomMask(patch_size=16, mask_ratio=0.75)
        >>> sample = {"image": torch.randn(3, 224, 224)}
        >>> result = transform(sample)
        >>> sorted(result.keys())
        ['image', 'ids_restore', 'len_keep', 'mask_masked', 'mask_visible']
        >>> result["len_keep"]
        49
        >>> result["mask_visible"].shape
        torch.Size([49])

    Args:
        patch_size (int): The height and width of each square patch.
        mask_ratio (float): The fraction of patches to be masked (e.g., 0.75).
        source (str): The key in the sample dict for the source image tensor.
        target_visible (str): The key to use when storing visible patch indices.
        target_masked (str): The key to use when storing masked patch indices.
        target_ids_restore (str): The key to use for the restoration indices.
        target_len_keep (str): The key to use for the count of visible patches.
    """

    def __init__(
        self,
        patch_size=16,
        mask_ratio=0.75,
        source: str = "image",
        target_visible: str = "mask_visible",
        target_masked: str = "mask_masked",
        target_ids_restore: str = "ids_restore",
        target_len_keep: str = "len_keep",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.source = source
        self.target_visible = target_visible
        self.target_masked = target_masked
        self.target_ids_restore = target_ids_restore
        self.target_len_keep = target_len_keep

    def __call__(self, x):
        source = self.nested_get(x, self.source)
        if isinstance(source, PIL.Image.Image):
            W, H = source.size  # PIL is W,H
        elif isinstance(source, torch.Tensor):
            # NOTE assumes _HW
            H, W = source.shape[-2:]
        else:
            raise ValueError(
                f"Source must be a PIL.Image.Image or a torch.Tensor, but got {type(source)} instead."
            )

        num_patches = (H // self.patch_size) * (W // self.patch_size)
        len_keep = int(num_patches * (1 - self.mask_ratio))

        # Generate random noise and shuffle indices (like MAE)
        noise = torch.rand(num_patches)
        ids_shuffle = torch.argsort(noise)
        ids_restore = torch.argsort(ids_shuffle)  # inverse permutation

        # Split into visible and masked
        mask_visible = ids_shuffle[:len_keep]  # first len_keep are visible
        mask_masked = ids_shuffle[len_keep:]  # rest are masked

        # Add to sample
        x[self.target_visible] = mask_visible
        x[self.target_masked] = mask_masked
        x[self.target_ids_restore] = (
            ids_restore  # NEW: for reconstructing full sequence
        )
        x[self.target_len_keep] = len_keep

        return x


# class RandomClassSwitch(v2.Transform):
#     def __init__(
#         self,
#         label_key: str,
#         new_key: str,
#         p: float,
#         low: int = -2147483648,
#         high: int = 0,
#     ):
#         super().__init__()
#         self.p = p
#         self.label_key = label_key
#         self.new_key = new_key
#         self.low = low
#         self.high = high

#     def __call__(self, sample: dict):
#         assert type(sample) is dict
#         assert self.label_key in sample
#         assert self.new_key not in sample
#         if self.p > 0 and torch.rand(1) < self.p:
#             if torch.is_tensor(sample[self.label_key]):
#                 sample[self.new_key] = torch.randint(
#                     low=self.low, high=self.high, size=()
#                 )
#             else:
#                 sample[self.new_key] = np.random.randint(low=self.low, high=self.high)
#         else:
#             sample[self.new_key] = sample[self.label_key]
#         return sample
