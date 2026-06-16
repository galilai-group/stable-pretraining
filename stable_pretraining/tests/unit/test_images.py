"""Unit tests for the Lance-backed image dataset.

Tests are marked ``unit`` and are fully self-cleaning: they only write under
``tmp_path`` (pytest-managed scratch) and don't touch any user directories.
A tiny synthetic ``datasets.Dataset`` is built in-memory (no network, no extra
dep beyond ``datasets`` + ``cv2``, both already core dependencies).
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from stable_pretraining.data.images import (
    LanceImageDataset,
    build_lance_image_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_hf_dataset(n: int = 12, h: int = 48, w: int = 48, n_image_cols: int = 1):
    """Build an in-memory HF dataset with image + label + extra columns.

    ``n_image_cols`` image columns exercise multi-image auto-detection. A
    string ``meta`` column exercises pass-through of non-image fields.
    """
    import datasets
    from datasets import ClassLabel, Features
    from datasets import Image as HFImage
    from datasets import Value

    rng = np.random.default_rng(0)

    def _img(seed_shift):
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = (seed_shift * 7) % 256
        arr[:, :, 1] = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, 0)
        arr[:, :, 2] = rng.integers(0, 256, (h, w), dtype=np.uint8)
        return Image.fromarray(arr)

    data = {"label": [i % 3 for i in range(n)], "meta": [f"row-{i}" for i in range(n)]}
    feats = {"label": ClassLabel(names=["a", "b", "c"]), "meta": Value("string")}
    image_col_names = ["image"] + [f"image{k}" for k in range(1, n_image_cols)]
    for ci, name in enumerate(image_col_names):
        data[name] = [_img(i + 100 * ci) for i in range(n)]
        feats[name] = HFImage()
    return datasets.Dataset.from_dict(data, features=Features(feats)), image_col_names


@pytest.fixture
def hf_dataset():
    ds, _ = _make_hf_dataset(n=12)
    return ds


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildLanceImageDataset:
    """Unit tests for :func:`build_lance_image_dataset`."""

    def test_build_creates_dataset_and_sidecar(self, hf_dataset, tmp_path):
        out = tmp_path / "ds.lance"
        returned = build_lance_image_dataset(
            hf_dataset, out, image_format="webp", quality=80
        )
        assert returned == out
        assert out.is_dir(), "lance dataset directory not created"

        sidecar = tmp_path / "ds.lance.images.json"
        assert sidecar.exists(), "sidecar images.json missing"
        meta = json.loads(sidecar.read_text())
        assert meta["encoding"] == "webp"
        assert meta["quality"] == 80
        assert meta["num_rows"] == 12
        assert meta["image_columns"] == ["image"]
        # All source columns preserved in the recorded order.
        assert set(meta["columns"]) == {"label", "meta", "image"}

    def test_jpeg_format(self, hf_dataset, tmp_path):
        out = tmp_path / "ds.lance"
        build_lance_image_dataset(hf_dataset, out, image_format="jpeg", quality=90)
        meta = json.loads((tmp_path / "ds.lance.images.json").read_text())
        assert meta["encoding"] == "jpeg"

    def test_invalid_format_raises(self, hf_dataset, tmp_path):
        with pytest.raises(ValueError):
            build_lance_image_dataset(
                hf_dataset, tmp_path / "x.lance", image_format="png"
            )

    def test_multiple_image_columns_autodetected(self, tmp_path):
        ds, names = _make_hf_dataset(n=8, n_image_cols=3)
        out = tmp_path / "ds.lance"
        build_lance_image_dataset(ds, out)
        meta = json.loads((tmp_path / "ds.lance.images.json").read_text())
        assert sorted(meta["image_columns"]) == sorted(names)

    def test_depth_style_no_label(self, tmp_path):
        # Multi-image, no label column (e.g. RGB + depth prediction): all
        # image columns are encoded, nothing is singled out as a "label".
        import datasets
        from datasets import Features
        from datasets import Image as HFImage

        imgs = [
            Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)) for _ in range(5)
        ]
        ds = datasets.Dataset.from_dict(
            {"rgb": imgs, "depth": imgs},
            features=Features({"rgb": HFImage(), "depth": HFImage()}),
        )
        out = tmp_path / "ds.lance"
        build_lance_image_dataset(ds, out)
        meta = json.loads((tmp_path / "ds.lance.images.json").read_text())
        assert sorted(meta["image_columns"]) == ["depth", "rgb"]
        rds = LanceImageDataset(out)
        assert set(rds[0].keys()) == {"rgb", "depth", "sample_idx"}
        assert sorted(rds.image_columns) == ["depth", "rgb"]

    def test_image_columns_override(self, tmp_path):
        # Two image features, but only encode "image" and keep the other raw.
        ds, names = _make_hf_dataset(n=6, n_image_cols=2)
        out = tmp_path / "ds.lance"
        build_lance_image_dataset(ds, out, image_columns="image")
        meta = json.loads((tmp_path / "ds.lance.images.json").read_text())
        assert meta["image_columns"] == ["image"]

    def test_image_columns_unknown_raises(self, hf_dataset, tmp_path):
        with pytest.raises(ValueError, match="not in dataset columns"):
            build_lance_image_dataset(
                hf_dataset, tmp_path / "x.lance", image_columns="nope"
            )

    def test_store_dimensions(self, tmp_path):
        ds, _ = _make_hf_dataset(n=6, h=96, w=64)
        out = tmp_path / "ds.lance"
        build_lance_image_dataset(ds, out, store_dimensions=True)
        import lance

        schema_names = lance.dataset(str(out)).schema.names
        assert "image_width" in schema_names and "image_height" in schema_names

    def test_max_size_downscales_preserving_aspect(self, tmp_path):
        ds, _ = _make_hf_dataset(n=4, h=100, w=50)
        out = tmp_path / "ds.lance"
        build_lance_image_dataset(ds, out, max_size=40, store_dimensions=True)
        import lance

        row = lance.dataset(str(out)).take([0]).to_pylist()[0]
        # Longest side (100) scaled to 40 → (40, 20), aspect preserved.
        assert row["image_height"] == 40 and row["image_width"] == 20

    def test_max_size_keeps_small_images(self, tmp_path):
        # Source is 48x48; max_size=64 is larger, so images are untouched.
        ds, _ = _make_hf_dataset(n=4, h=48, w=48)
        out = tmp_path / "ds.lance"
        build_lance_image_dataset(ds, out, max_size=64, store_dimensions=True)
        import lance

        row = lance.dataset(str(out)).take([0]).to_pylist()[0]
        assert row["image_height"] == 48 and row["image_width"] == 48

    def test_overwrite_false_raises_on_existing(self, hf_dataset, tmp_path):
        out = tmp_path / "ds.lance"
        build_lance_image_dataset(hf_dataset, out)
        with pytest.raises(FileExistsError):
            build_lance_image_dataset(hf_dataset, out)

    def test_overwrite_true_replaces(self, hf_dataset, tmp_path):
        out = tmp_path / "ds.lance"
        build_lance_image_dataset(hf_dataset, out, quality=70)
        build_lance_image_dataset(hf_dataset, out, quality=95, overwrite=True)
        meta = json.loads((tmp_path / "ds.lance.images.json").read_text())
        assert meta["quality"] == 95

    def test_no_image_columns_raises(self, tmp_path):
        import datasets

        ds = datasets.Dataset.from_dict({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="no image columns"):
            build_lance_image_dataset(ds, tmp_path / "x.lance")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@pytest.fixture
def built_dataset(hf_dataset, tmp_path):
    out = tmp_path / "ds.lance"
    build_lance_image_dataset(hf_dataset, out, max_size=32, overwrite=True)
    return out


@pytest.mark.unit
class TestLanceImageDataset:
    """Unit tests for :class:`LanceImageDataset` reads."""

    def test_len_matches_num_rows(self, built_dataset):
        ds = LanceImageDataset(built_dataset)
        assert len(ds) == 12

    def test_getitem_keys_and_pil(self, built_dataset):
        ds = LanceImageDataset(built_dataset)
        sample = ds[0]
        assert set(sample.keys()) == {"image", "label", "meta", "sample_idx"}
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].size == (32, 32)  # resize applied (square source)
        assert sample["label"] == 0
        assert sample["meta"] == "row-0"
        assert sample["sample_idx"] == 0

    def test_getitem_tensor_format(self, built_dataset):
        ds = LanceImageDataset(built_dataset, image_format="tensor")
        sample = ds[1]
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].dtype == torch.uint8
        assert tuple(sample["image"].shape) == (32, 32, 3)

    def test_transform_applied(self, built_dataset):
        def transform(sample):
            sample["label"] = sample["label"] + 100
            return sample

        ds = LanceImageDataset(built_dataset, transform=transform)
        assert ds[2]["label"] == 100 + (2 % 3)

    def test_column_subset(self, built_dataset):
        ds = LanceImageDataset(built_dataset, columns=["image", "label"])
        sample = ds[0]
        assert set(sample.keys()) == {"image", "label", "sample_idx"}

    def test_unknown_column_raises(self, built_dataset):
        with pytest.raises(ValueError, match="unknown columns"):
            LanceImageDataset(built_dataset, columns=["nope"])

    def test_out_of_range_raises(self, built_dataset):
        ds = LanceImageDataset(built_dataset)
        with pytest.raises(IndexError):
            _ = ds[len(ds)]
        with pytest.raises(IndexError):
            _ = ds[-1]

    def test_missing_sidecar_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LanceImageDataset(tmp_path / "nonexistent.lance")

    def test_image_columns_property(self, built_dataset):
        ds = LanceImageDataset(built_dataset)
        assert ds.image_columns == ["image"]

    # --- DataLoader smoke test --------------------------------------------

    def test_dataloader_single_process(self, built_dataset):
        from torch.utils.data import DataLoader

        ds = LanceImageDataset(built_dataset, image_format="tensor")

        def collate(xs):
            return {
                k: (
                    torch.stack([x[k] for x in xs])
                    if isinstance(xs[0][k], torch.Tensor)
                    else [x[k] for x in xs]
                )
                for k in xs[0]
            }

        loader = DataLoader(
            ds, batch_size=4, num_workers=0, collate_fn=collate, shuffle=False
        )
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 32, 32, 3)
        assert len(batch["meta"]) == 4
