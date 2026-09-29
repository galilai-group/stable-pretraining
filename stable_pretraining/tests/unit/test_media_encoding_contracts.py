"""Real codecs and Arrow batches preserve image content and video row boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from PIL import Image

from stable_pretraining.data import images, video

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["gray", "gray_channel", "rgba", "float", "pil_gray"])
def test_rgb_conversion_normalizes_shape_dtype_and_channels(kind):
    base = np.arange(24, dtype=np.uint8).reshape(4, 6)
    source = {
        "gray": base,
        "gray_channel": base[..., None],
        "rgba": np.stack([base, base, base, base], -1),
        "float": base.astype(float) * 20,
        "pil_gray": Image.fromarray(base),
    }[kind]
    result = images._to_rgb_uint8(source)
    expected = (
        np.clip(base.astype(float) * 20, 0, 255).astype(np.uint8)
        if kind == "float"
        else base
    )
    assert result.shape == (4, 6, 3)
    assert result.dtype == np.uint8 and result.flags.c_contiguous
    np.testing.assert_array_equal(result, np.repeat(expected[..., None], 3, -1))


@pytest.mark.parametrize(
    "max_size,expected", [(None, (12, 8)), (6, (6, 4)), (20, (12, 8))]
)
def test_image_encoding_preserves_color_and_aspect_ratio(max_size, expected):
    rgb = np.full((8, 12, 3), [200, 30, 10], dtype=np.uint8)
    blob, width, height = images._encode_rgb(rgb, ".png", [], max_size)
    decoded = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    assert (width, height) == expected
    assert decoded.shape == (height, width, 3)
    np.testing.assert_array_equal(decoded[0, 0], [10, 30, 200])


def test_image_encoder_failure_is_reported(monkeypatch):
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    with pytest.raises(RuntimeError, match="encode failed"):
        images._encode_rgb(np.zeros((2, 2, 3), np.uint8), ".webp", [], None)


@pytest.mark.parametrize("resize,shape", [(None, (8, 12)), (6, (6, 6)), (20, (8, 12))])
def test_video_worker_encodes_real_video_frames(tmp_path, resize, shape):
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (12, 8))
    assert writer.isOpened()
    try:
        for value in [40, 80, 120]:
            writer.write(np.full((8, 12, 3), value, np.uint8))
    finally:
        writer.release()
    result = video._encode_one_video((7, str(path), 100, resize))
    assert result[:6] == ("ok", 7, str(path), 3, *shape)
    for expected, blob in zip([40, 80, 120], result[-1]):
        frame = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        assert frame.shape[:2] == shape
        assert np.abs(frame.astype(float) - expected).max() <= 3


@pytest.mark.parametrize("failure", ["open", "empty", "encode", "exception"])
def test_video_worker_releases_capture_on_every_failure(monkeypatch, failure):
    capture = Mock()
    capture.isOpened.return_value = failure != "open"
    capture.read.return_value = (
        (False, None) if failure == "empty" else (True, np.zeros((4, 4, 3), np.uint8))
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    if failure == "encode":
        monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    elif failure == "exception":
        capture.read.side_effect = OSError("decode failure")
    result = video._encode_one_video((0, "fake.avi", 80, None))
    assert result[0] == "error"
    capture.release.assert_called_once_with()


def test_video_batches_keep_contiguous_offsets_when_corrupt_video_is_skipped():
    records = []
    progress = SimpleNamespace(update=Mock())
    inputs = [
        ("ok", 2, "a", 2, 4, 6, [b"a", b"b"]),
        ("error", 3, "bad", "corrupt"),
        ("ok", 4, "c", 1, 4, 6, [b"c"]),
    ]
    batches = list(video._batch_stream(iter(inputs), records, True, progress, 0))
    assert [record["start_row"] for record in records] == [0, 2]
    assert batches[0].to_pydict() == {
        "video_id": [2, 2],
        "frame_idx": [0, 1],
        "bytes": [b"a", b"b"],
    }
    assert batches[1].to_pydict()["bytes"] == [b"c"]
    assert progress.update.call_count == 3
    with pytest.raises(RuntimeError, match="corrupt"):
        list(video._batch_stream(iter([inputs[1]]), [], False, progress, 0))
