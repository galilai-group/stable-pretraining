"""Independent label and frozen-probe metric checks."""

import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

pytestmark = pytest.mark.unit
EXAMPLE = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "examples" / "levljepa_ade20k.py")
)


def test_ade_labels_keep_last_class_and_ignore_both_void_values():
    raw = np.array([[0, 1, 150, 255]], dtype=np.uint8)
    assert EXAMPLE["_labels"](raw).tolist() == [[255, 0, 149, 255]]


def test_ade_labels_reject_non_ade_categories():
    with pytest.raises(ValueError, match="ADE20K labels"):
        EXAMPLE["_labels"](np.array([[151]], dtype=np.uint8))


def test_frozen_probe_scores_perfect_native_mask_and_ignores_void(tmp_path):
    image = tmp_path / "images" / "validation" / "one.jpg"
    mask = tmp_path / "annotations" / "validation" / "one.png"
    image.parent.mkdir(parents=True)
    mask.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(image)
    Image.fromarray(np.array([[1, 2], [0, 255]], dtype=np.uint8)).save(mask)
    data = EXAMPLE["_Masks"](tmp_path, "validation", 2, None)
    head = nn.Conv2d(2, 150, 1)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-100)
        head.bias[:2].zero_()
        head.weight[0, 0] = 10
        head.weight[1, 1] = 10
    features = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]])
    result = EXAMPLE["_metrics"](head, features, data, torch.device("cpu"))
    assert result["miou"] == result["pixel_accuracy"] == 1.0
    assert result["classes_with_union"] == 2
    confusion = torch.tensor(result["confusion"])
    assert confusion.sum() == 2
    assert confusion[0, 0] == confusion[1, 1] == 1
