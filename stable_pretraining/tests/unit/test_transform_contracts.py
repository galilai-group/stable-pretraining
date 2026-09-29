"""Augmentations must preserve routing, metadata, and expected pixel values."""

import numpy as np
import pytest
import torch
from torchvision.transforms import v2

from stable_pretraining.data import transforms as t

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(321)
        yield


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("RandomSolarize", {"threshold": 100}),
        ("RandomAutocontrast", {}),
        ("RandomEqualize", {}),
        ("RandomInvert", {}),
        ("RandomPosterize", {"bits": 3}),
        ("RandomAdjustSharpness", {"sharpness_factor": 2}),
        ("RandomVerticalFlip", {}),
        ("RandomHorizontalFlip", {}),
        ("RandomErasing", {"value": 19}),
        ("RandomPerspective", {"distortion_scale": 0.3}),
        ("GaussianNoise", {"mean": 0.1, "sigma": 0.05}),
    ],
)
@pytest.mark.parametrize("probability", [0, 1])
def test_wrappers_match_torchvision_and_preserve_other_fields(
    name, kwargs, probability
):
    image = torch.randint(0, 256, (3, 12, 16), dtype=torch.uint8)
    if name == "GaussianNoise":
        image = image.float() / 255
    source = image.clone()
    wrapped = getattr(t, name)(p=probability, **kwargs)
    sample = {"image": image, "label": 3, "nested": {"keep": "value"}}
    state = torch.random.get_rng_state()
    if probability == 1 and name in ("RandomErasing", "RandomPerspective"):
        # torchvision samples its probability check even when p=1; our wrapper skips it.
        torch.rand(1)
    result = wrapped(sample)
    torch.random.set_rng_state(state)
    if name == "GaussianNoise":
        expected = v2.GaussianNoise(**kwargs)(source) if probability else source
    else:
        expected = getattr(v2, name)(p=probability, **kwargs)(source)
    torch.testing.assert_close(result["image"], expected)
    assert result["label"] == 3 and result["nested"] == {"keep": "value"}
    assert name in result


def test_repeated_transforms_keep_all_metadata_entries():
    sample = {"image": torch.arange(12, dtype=torch.uint8).reshape(1, 3, 4)}
    for p in [1, 0, 1, 0]:
        t.RandomVerticalFlip(p=p)(sample)
    assert [
        sample[key]
        for key in [
            "RandomVerticalFlip",
            "RandomVerticalFlip_0",
            "RandomVerticalFlip_1",
            "RandomVerticalFlip_2",
        ]
    ] == [True, False, True, False]


def test_nested_routing_and_multiple_images_preserve_sources():
    first = torch.arange(12).reshape(1, 3, 4)
    second = first + 12
    sample = {"views": [{"image": first}, {"image": second}], "output": [None, None]}
    transform = t.RandomVerticalFlip(
        p=1, source=["views.0.image", "views.1.image"], target=["output.0", "output.1"]
    )
    transform(sample)
    torch.testing.assert_close(sample["output"][0], first.flip(-2))
    torch.testing.assert_close(sample["output"][1], second.flip(-2))
    assert sample["views"][0]["image"] is first
    assert sample["views"][1]["image"] is second
    assert t.Transform().nested_get(sample, "") is sample


def test_lambda_routing_and_wrapped_transform_operate_on_selected_data():
    routes = {
        "double": t.Lambda(lambda sample: sample["image"] * 2, target="result"),
        "add": t.WrapTorchTransform(lambda image: image + 3, target="result"),
    }
    routing = t.RoutingTransform(lambda sample: sample["operation"], routes)
    for operation, expected in [("double", 4), ("add", 5)]:
        sample = {"image": torch.tensor(2), "operation": operation}
        result = routing(sample)
        assert result["result"].item() == expected
        assert result["image"].item() == 2


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("RandomAffine", {"degrees": 20}),
        ("RandomCrop", {"size": (6, 7)}),
        ("RandomCrop", {"size": (16, 20), "pad_if_needed": True}),
    ],
)
def test_geometric_transform_matches_reference_with_separate_target(name, kwargs):
    image = torch.arange(3 * 12 * 16, dtype=torch.float32).reshape(3, 12, 16)
    state = torch.random.get_rng_state()
    output = getattr(t, name)(source="input", target="output", **kwargs)(
        {"input": image}
    )
    torch.random.set_rng_state(state)
    expected = getattr(v2, name)(**kwargs)(image)
    torch.testing.assert_close(output["output"], expected)
    assert output["input"] is image
    assert torch.isfinite(output[name]).all()


@pytest.mark.parametrize(
    "enabled,apply_on_true",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_conditional_transform_records_apply_and_bypass(enabled, apply_on_true):
    transform = t.Conditional(
        t.AdditiveGaussian(sigma=0), "enabled", apply_on_true=apply_on_true
    )
    image = torch.ones(3, 4, 4)
    sample = transform({"image": image.clone(), "enabled": enabled})
    torch.testing.assert_close(sample["image"], image)
    assert sample["AdditiveGaussian"] is (enabled == apply_on_true)


@pytest.mark.parametrize("sigma", [0.2, torch.tensor(0.2)])
def test_additive_noise_matches_seeded_draw_and_bypass(sigma):
    image = torch.zeros(3, 4, 4)
    state = torch.random.get_rng_state()
    result = t.AdditiveGaussian(sigma=sigma)({"image": image.clone()})
    torch.random.set_rng_state(state)
    torch.rand(1)
    torch.testing.assert_close(result["image"], torch.randn_like(image) * 0.2)
    assert result["AdditiveGaussian"] is True
    skipped = t.AdditiveGaussian(sigma=sigma, p=0)({"image": image.clone()})
    torch.testing.assert_close(skipped["image"], image)
    assert skipped["AdditiveGaussian"] is False


def test_round_robin_cycles_without_skipping_views():
    transform = t.RoundRobinMultiViewTransform(
        [lambda x: {**x, "view": 0}, lambda x: {**x, "view": 1}]
    )
    samples = [transform({"index": i // 2}) for i in range(6)]
    assert [s["view"] for s in samples] == [0, 1, 0, 1, 0, 1]
    assert [s["index"] for s in samples] == [0, 0, 1, 1, 2, 2]


def test_numpy_image_conversion_preserves_height_width_and_channels():
    image = np.arange(4 * 7 * 3, dtype=np.uint8).reshape(4, 7, 3)
    converted = t.to_image(image)
    torch.testing.assert_close(converted, torch.from_numpy(image).permute(2, 0, 1))
    with pytest.raises(TypeError, match="Input can either"):
        t.to_image("not an image")


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("samples", [2, 7])
def test_uniform_temporal_subsample_handles_axis_and_repeated_frames(axis, samples):
    video = torch.arange(24).reshape(4, 3, 2)
    indices = torch.linspace(0, video.shape[axis] - 1, samples).long()
    result = t.UniformTemporalSubsample(
        samples, temporal_dim=axis, source="input", target="output"
    )({"input": video})
    torch.testing.assert_close(result["output"], video.index_select(axis, indices))
    assert result["input"] is video


class _VideoReader:
    def __init__(self, count):
        self.count = count

    def get_metadata(self):
        return {"video": {"duration": [self.count / 2], "fps": [2]}}

    def seek(self, seconds):
        return iter(
            {"data": torch.full((3, 2, 2), i)}
            for i in range(round(seconds * 2), self.count)
        )


@pytest.mark.parametrize("count,stride", [(4, 1), (7, 2), (12, 2)])
def test_contiguous_sampler_returns_requested_frames_in_order(count, stride):
    transform = t.RandomContiguousTemporalSampler(
        "reader", "video", 4, frame_subsampling=stride
    )
    sample = transform({"reader": _VideoReader(count)})
    start = sample["RandomContiguousTemporalSampler"]
    assert 0 <= start <= count - (3 * stride + 1)
    assert sample["video"].shape == (4, 3, 2, 2)
    assert sample["video"][:, 0, 0, 0].tolist() == [
        start + i * stride for i in range(4)
    ]


def test_contiguous_sampler_rejects_video_too_short():
    with pytest.raises(ValueError, match="frames"):
        t.RandomContiguousTemporalSampler("reader", "video", 4)(
            {"reader": _VideoReader(3)}
        )


@pytest.mark.parametrize("frames,stride", [(0, 1), (2, 0)])
def test_contiguous_sampler_requires_positive_sizes(frames, stride):
    with pytest.raises(ValueError, match="positive"):
        t.RandomContiguousTemporalSampler("reader", "video", frames, stride)


def test_uniform_sampler_rejects_empty_video_and_zero_samples():
    with pytest.raises(ValueError, match="positive"):
        t.UniformTemporalSubsample(0)
    with pytest.raises(ValueError, match="no frames"):
        t.UniformTemporalSubsample(2, temporal_dim=0)(
            {"video": torch.empty(0, 3, 4, 4)}
        )
