"""The views a LeVJEPA step trains on.

v2 (LeVJEPA upstream) builds them like this: every frame of a clip is resized whole
to squash x squash (antialiased bilinear, data.decode), then 1 global and num_local local views cropped from it
(ViewBuilder.crop: roi_align of random boxes), flipped, locals colour-jittered, normalized, and an iid `keep` share of
each view's patches kept (PatchEmbed). The token loader draws all of it on the CPU first and decodes only what the
kept patches read.
"""

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class ViewRecipe:
    """A LeVJEPA view recipe: clip length, crop geometry, jitter and kept share; V2 is v2's."""

    frames: int = 16  # T: frames per clip
    patch: int = 16  # patch side (the kernels need 16)
    squash: int = 224  # frames are resized whole to squash x squash before cropping
    global_size: int = 224  # global view side
    local_size: int = 96  # local view side
    num_local: int = 4
    global_scale: tuple = (0.8, 1.0)  # crop area as a share of the squashed frame
    local_scale: tuple = (0.02, 0.4)
    ratio: tuple = (3 / 4, 4 / 3)  # crop aspect ratios (log-uniform)
    flip_p: float = 0.5
    jitter: tuple = (0.4, 0.4, 0.2)  # brightness, contrast, saturation (locals)
    jitter_p: float = 0.8
    gray_p: float = 0.2
    keep: float = 0.05  # share of each view's patches kept, per frame
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    def __post_init__(self):
        if self.patch != 16:
            raise ValueError("the token loader's kernels build 16 x 16 patches")
        for size in (self.global_size, self.local_size):
            if size % self.patch:
                raise ValueError(
                    f"view size {size} is not a multiple of the patch size"
                )
        # the GPU stage squashes a 32 x 32 window around each patch: a patch's crop sub-box plus bilinear taps
        for size, scale in (
            (self.global_size, self.global_scale),
            (self.local_size, self.local_scale),
        ):
            side = min(
                math.sqrt(scale[1] * self.squash**2 * self.ratio[1]), self.squash
            )
            if side / size * self.patch + 4 > 32:
                raise ValueError(
                    f"a {size}-px view's patches cover up to {side / size * self.patch:.1f} px of the "
                    f"squashed frame: more than the GPU stage's 32-px window holds"
                )

    @property
    def keep_global(self):
        return math.ceil(
            self.keep * (self.global_size // self.patch) ** 2 - 1e-9
        )  # 10 of 196 in v2

    @property
    def keep_local(self):
        return math.ceil(
            self.keep * (self.local_size // self.patch) ** 2 - 1e-9
        )  # 2 of 36

    def sampling_ratio(self, scale, size):
        """ViewBuilder.crop's roi_align sampling_ratio for a view."""
        return max(
            1,
            math.ceil(
                math.sqrt(scale[1] * self.squash * self.squash * self.ratio[1]) / size
            ),
        )


V2 = ViewRecipe()
