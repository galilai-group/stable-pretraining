"""Random-access image dataset backed by Lance.

This module mirrors :mod:`stable_pretraining.data.video` for still images. It
turns a HuggingFace dataset (e.g. ``imagenet-1k``) into a Lance dataset of
re-encoded image rows, optimised for fast random-access reads during SSL
dataloading on slow filesystems (e.g. NFS).

    - :func:`build_lance_image_dataset` streams an :class:`datasets.Dataset`
      (or a dataset name to load) into Lance. Image columns are detected
      automatically (any HuggingFace ``Image`` feature) and re-encoded to
      WebP/JPEG bytes; **every other column is preserved as-is**, so datasets
      with extra metadata or several image columns round-trip cleanly.

    - :class:`LanceImageDataset` is a PyTorch :class:`~torch.utils.data.Dataset`
      that decodes one row per ``__getitem__`` and returns a sample dict (image
      columns decoded to PIL/tensor, all other columns passed through).

Encoding benchmark (224x224 photos, single core; see the module-level note):

    ============  ==========  ==========  ===========
    codec/q       encode      decode      size/img
    ============  ==========  ==========  ===========
    webp q85      ~180 img/s  ~1000 img/s smallest
    jpeg q90      ~5500 img/s ~3400 img/s larger
    ============  ==========  ==========  ===========

WebP is the default: it produces the smallest files (least NFS bandwidth per
sample), and its slower decode is easily hidden behind a handful of DataLoader
workers. Switch to ``image_format="jpeg"`` if your pipeline is decode-bound
(many tiny crops, few workers) and disk is cheap.
"""

import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

import cv2
import lance
import numpy as np
import pyarrow as pa
import torch
from loguru import logger as logging

from .datasets import Dataset


# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------

# One-time per-process cache of opened Lance datasets.
# IMPORTANT: must be reset in every DataLoader worker (see ``worker_init``)
# because the Rust/tokio runtime inside a Lance handle is not fork-safe.
_LANCE_CACHE: dict = {}


def _open_dataset(path: str):
    ds = _LANCE_CACHE.get(path)
    if ds is None:
        ds = lance.dataset(path)
        _LANCE_CACHE[path] = ds
    return ds


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _to_rgb_uint8(img) -> np.ndarray:
    """Coerce a decoded sample value into an HxWx3 RGB uint8 array.

    Handles PIL images (the HuggingFace ``Image`` feature decodes to PIL) and
    numpy arrays (e.g. ``Array3D`` features). Grayscale/RGBA inputs are
    converted to RGB.
    """
    from PIL import Image as PILImage

    if isinstance(img, PILImage.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img)
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _encode_rgb(arr: np.ndarray, ext: str, params: list, max_size: Optional[int]):
    """Maybe downscale (aspect-preserving), then encode an RGB array to bytes.

    Unlike :mod:`video` (which resizes to a square), images preserve aspect
    ratio: only when ``max(H, W) > max_size`` is the longest side scaled down
    to ``max_size`` — smaller images are left untouched. Runs in a worker
    thread; cv2 releases the GIL during resize/encode so this parallelises
    across cores.
    """
    H, W = arr.shape[:2]
    if max_size is not None and max(H, W) > int(max_size):
        scale = int(max_size) / float(max(H, W))
        new_w = max(1, round(W * scale))
        new_h = max(1, round(H * scale))
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        H, W = new_h, new_w
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise RuntimeError(f"cv2 {ext} encode failed")
    return buf.tobytes(), int(W), int(H)


def _resolve_dataset(source, split):
    """Return a map-style ``datasets.Dataset`` from a name/path or object."""
    import datasets

    if isinstance(source, (str, Path)):
        ds = datasets.load_dataset(str(source), split=split)
    else:
        ds = source
        if isinstance(ds, datasets.DatasetDict) and split is not None:
            ds = ds[split]
    if isinstance(ds, datasets.DatasetDict):
        raise ValueError(
            "source resolved to a DatasetDict with multiple splits "
            f"({list(ds.keys())}); pass split=... to pick one."
        )
    if not isinstance(ds, datasets.Dataset):
        raise TypeError(
            f"source must be a datasets.Dataset (or a name/path to load), got "
            f"{type(ds).__name__}."
        )
    return ds


def _resolve_image_columns(ds, image_columns) -> list[str]:
    """Pick the columns to encode as images.

    If ``image_columns`` is given (a name or list of names) it is used verbatim
    after validating the columns exist. Otherwise every image column is
    auto-detected: first any HuggingFace ``Image`` feature, falling back to
    sniffing the first row for PIL images.
    """
    if image_columns is not None:
        if isinstance(image_columns, str):
            image_columns = [image_columns]
        image_columns = list(image_columns)
        unknown = [c for c in image_columns if c not in ds.column_names]
        if unknown:
            raise ValueError(
                f"image_columns {unknown} not in dataset columns {ds.column_names}"
            )
        return image_columns

    import datasets

    cols = [n for n, f in ds.features.items() if isinstance(f, datasets.Image)]
    if cols:
        return cols
    # No declared Image feature — sniff the first row for PIL images.
    from PIL import Image as PILImage

    if len(ds) > 0:
        row = ds[0]
        cols = [n for n, v in row.items() if isinstance(v, PILImage.Image)]
    return cols


def build_lance_image_dataset(
    source,
    output_path: Union[str, Path],
    *,
    image_columns: Optional[Union[str, list[str]]] = None,
    split: Optional[str] = None,
    image_format: str = "webp",
    quality: int = 85,
    max_size: Optional[int] = None,
    batch_size: int = 512,
    workers: Optional[int] = None,
    store_dimensions: bool = False,
    overwrite: bool = False,
) -> Path:
    """Build a random-access Lance image dataset from a HuggingFace dataset.

    Image columns are discovered automatically — every column whose
    HuggingFace feature is :class:`datasets.Image` (or, if none are declared,
    every column that decodes to a PIL image) is re-encoded to WebP/JPEG bytes.
    So a plain ``{image, label}`` classification set, a depth-prediction set
    with several image columns (``rgb``, ``depth``, ``mask``, ...) and no
    label, and anything in between all round-trip with **no configuration**.
    Pass ``image_columns=`` only to override the auto-detection.

    **All non-image columns are copied through unchanged** at their native
    arrow type, so labels, captions, bounding boxes, etc. are preserved without
    being singled out.

    The resulting Lance schema is::

        id:                 int64                 # row index, always present
        <image_col>:        binary                # encoded bytes, per image col
        <image_col>_width:  int32   (optional)    # if store_dimensions=True
        <image_col>_height: int32   (optional)
        <other_col>:        <native arrow type>   # every non-image column

    A side-car ``<output_path>.images.json`` records which columns are images,
    the codec/quality, the row count, and the column order so the matching
    :class:`LanceImageDataset` reader needs no arguments.

    Rows are streamed from the dataset in batches and encoded with a thread
    pool (cv2 releases the GIL), so memory is bounded by one batch.

    Args:
        source: A HuggingFace dataset name/path (loaded via
            ``datasets.load_dataset``) or an already-loaded
            ``datasets.Dataset`` / ``DatasetDict``.
        output_path: Target path for the ``.lance`` dataset directory.
        image_columns: Optional explicit column name or list of names to treat
            as images. ``None`` (default) auto-detects them; pass this only to
            override (e.g. to encode just ``"rgb"`` and keep ``"depth"`` raw).
        split: Split to load/select when ``source`` is a name or a
            ``DatasetDict`` (e.g. ``"train"``).
        image_format: ``"webp"`` (default, smallest files) or ``"jpeg"``
            (faster decode).
        quality: Encoder quality in ``[1, 100]``. Default 85 is a good WebP
            sweet-spot for SSL.
        max_size: Cap on the longest image side. If set and an image's
            ``max(H, W) > max_size``, the longest side is scaled down to
            ``max_size`` (aspect-preserving); images already at or below it are
            stored untouched. ``None`` (default) keeps native resolution.
        batch_size: Rows read/encoded per Lance RecordBatch.
        workers: Encode threads. Defaults to ``os.cpu_count()``.
        store_dimensions: If ``True``, also store ``<col>_width`` /
            ``<col>_height`` per image column.
        overwrite: If ``True``, remove any existing dataset at ``output_path``.

    Returns:
        Path to the ``.lance`` dataset directory that was written.
    """
    output_path = Path(output_path)
    if image_format not in ("webp", "jpeg"):
        raise ValueError(f"image_format must be 'webp' or 'jpeg', got {image_format!r}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Pass overwrite=True to replace it."
            )
        import shutil

        shutil.rmtree(output_path)

    ds = _resolve_dataset(source, split)
    image_columns = _resolve_image_columns(ds, image_columns)
    if not image_columns:
        raise ValueError(
            "no image columns detected. Pass image_columns=<name or list>, or "
            "use a dataset that declares datasets.Image feature(s)."
        )
    n_rows = len(ds)
    column_order = list(ds.column_names)
    logging.info(
        f"building Lance image dataset: {n_rows} rows, "
        f"image columns={image_columns}, other columns="
        f"{[c for c in column_order if c not in image_columns]} → {output_path}"
    )

    passthrough = [c for c in column_order if c not in image_columns]
    arrow_schema = ds.features.arrow_schema
    passthrough_types = {name: arrow_schema.field(name).type for name in passthrough}

    # Two views of the same dataset, read in lock-step per batch:
    #   - ``enc_ds`` (default/python format) decodes ONLY the columns we encode
    #     to PIL — no wasted decode of pass-through images.
    #   - ``arrow_ds`` (arrow format) yields pass-through columns at their exact
    #     native arrow type without decoding (handles any dtype, including an
    #     image-feature column the user chose NOT to re-encode — its original
    #     encoded bytes are then stored verbatim).
    enc_ds = ds.select_columns(image_columns)
    arrow_ds = ds.with_format("arrow")

    # Build the output schema in a stable order: id, then every source column
    # in its original order (image columns become binary, with optional dims).
    fields = [pa.field("id", pa.int64())]
    for name in column_order:
        if name in image_columns:
            fields.append(pa.field(name, pa.binary()))
            if store_dimensions:
                fields.append(pa.field(f"{name}_width", pa.int32()))
                fields.append(pa.field(f"{name}_height", pa.int32()))
        else:
            fields.append(pa.field(name, passthrough_types[name]))
    out_schema = pa.schema(fields)

    if image_format == "webp":
        ext = ".webp"
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]
    else:
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    n_workers = int(workers) if workers else max(1, os.cpu_count() or 1)

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    def _batch_stream(executor, task_id):
        for start in range(0, n_rows, batch_size):
            enc_batch = enc_ds[start : start + batch_size]
            pass_tbl = arrow_ds[start : start + batch_size] if passthrough else None
            bs = len(enc_batch[image_columns[0]])
            columns: list = [pa.array(range(start, start + bs), type=pa.int64())]
            for name in column_order:
                if name in image_columns:
                    arrs = [_to_rgb_uint8(im) for im in enc_batch[name]]
                    enc = list(
                        executor.map(
                            lambda a: _encode_rgb(a, ext, params, max_size), arrs
                        )
                    )
                    columns.append(pa.array([e[0] for e in enc], type=pa.binary()))
                    if store_dimensions:
                        columns.append(pa.array([e[1] for e in enc], type=pa.int32()))
                        columns.append(pa.array([e[2] for e in enc], type=pa.int32()))
                else:
                    # Native arrow array (no decode); combine in case the slice
                    # spans multiple chunks so record_batch gets a flat Array.
                    columns.append(pass_tbl.column(name).combine_chunks())
            yield pa.record_batch(columns, schema=out_schema)
            progress.update(task_id, advance=bs)

    t0 = time.time()
    with progress:
        task_id = progress.add_task("encoding", total=n_rows)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            reader = pa.RecordBatchReader.from_batches(
                out_schema, _batch_stream(executor, task_id)
            )
            lance.write_dataset(reader, str(output_path), mode="create")

    elapsed = time.time() - t0
    size_b = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())

    sidecar_path = output_path.with_name(output_path.name + ".images.json")
    sidecar_path.write_text(
        json.dumps(
            {
                "encoding": image_format,
                "quality": int(quality),
                "max_size": int(max_size) if max_size is not None else None,
                "store_dimensions": bool(store_dimensions),
                "num_rows": int(n_rows),
                "image_columns": image_columns,
                "columns": column_order,
            },
            indent=2,
        )
    )
    logging.info(
        f"done in {elapsed:.1f}s. {n_rows} rows, {size_b / 1024**2:.1f} MiB, "
        f"{n_rows / max(elapsed, 1e-9):.0f} img/s. sidecar={sidecar_path}"
    )
    return output_path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LanceImageDataset(Dataset):
    """Random-access image dataset over a Lance file.

    Reads one row per ``__getitem__`` from a dataset built by
    :func:`build_lance_image_dataset`, decodes every image column back to a PIL
    image (or RGB uint8 tensor), and returns a sample dict that also carries
    all the preserved non-image columns plus ``sample_idx``. This makes it a
    drop-in source for the same transform pipelines used with
    :class:`~stable_pretraining.data.HFDataset`.

    Lance is opened lazily per-process (never in the parent) and the handle
    cache is reset in :meth:`worker_init`, because the Rust/tokio runtime
    inside a Lance handle is not fork-safe.

    Args:
        lance_path: Path to a ``.lance`` directory produced by
            :func:`build_lance_image_dataset`. The sidecar
            ``<lance_path>.images.json`` must sit next to it.
        image_format: ``"pil"`` (default) returns each image column as a
            ``PIL.Image``; ``"tensor"`` returns an HxWx3 RGB uint8 tensor.
        columns: Optional subset of source columns to load (image and/or
            non-image). Defaults to every column. ``id`` is never returned.
        transform: Optional callable applied to each sample dict (library
            standard). Receives a dict, returns a dict.

    Returns per item (before optional ``transform``):
        ``<image_col>``  : ``PIL.Image`` or uint8 tensor ``(H, W, 3)`` RGB
        ``<other_col>``  : the preserved value for every non-image column
        ``sample_idx``   : the row index (same as ``idx``)
    """

    def __init__(
        self,
        lance_path: Union[str, Path],
        *,
        image_format: str = "pil",
        columns: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(transform=transform)
        if image_format not in ("pil", "tensor"):
            raise ValueError(
                f"image_format must be 'pil' or 'tensor', got {image_format!r}"
            )
        self.lance_path = str(lance_path)
        self.image_format = image_format

        sidecar = Path(self.lance_path + ".images.json")
        if not sidecar.exists():
            raise FileNotFoundError(
                f"sidecar not found at {sidecar}. "
                f"Was the dataset built with build_lance_image_dataset?"
            )
        meta = json.loads(sidecar.read_text())
        self._num_rows = int(meta["num_rows"])
        self._image_columns = list(meta["image_columns"])
        all_columns = list(meta["columns"])

        if columns is None:
            self._load_columns = all_columns
        else:
            unknown = [c for c in columns if c not in all_columns]
            if unknown:
                raise ValueError(f"unknown columns {unknown}; available: {all_columns}")
            self._load_columns = list(columns)

    def __len__(self) -> int:
        return self._num_rows

    @property
    def image_columns(self) -> list[str]:
        """Names of the columns that are decoded as images."""
        return list(self._image_columns)

    def _decode(self, blob: bytes):
        bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.image_format == "tensor":
            return torch.from_numpy(np.ascontiguousarray(rgb))
        from PIL import Image as PILImage

        return PILImage.fromarray(rgb)

    def __getitem__(self, idx: int) -> dict:
        if not (0 <= idx < self._num_rows):
            raise IndexError(idx)
        ds = _open_dataset(self.lance_path)
        tbl = ds.take([int(idx)], columns=self._load_columns)
        sample = {}
        for col in self._load_columns:
            value = tbl.column(col)[0].as_py()
            if col in self._image_columns:
                sample[col] = self._decode(value)
            else:
                sample[col] = value
        sample["sample_idx"] = int(idx)
        return self.process_sample(sample)

    @staticmethod
    def worker_init(worker_id: int) -> None:
        """Pass as ``worker_init_fn=LanceImageDataset.worker_init``.

        Lance is not fork-safe (its Rust/tokio runtime is inherited with a
        dead state on ``fork``), so every worker must reset the module-level
        handle cache. Also pins cv2 to a single thread so N DataLoader
        workers x M OpenCV threads does not oversubscribe the CPU.
        """
        global _LANCE_CACHE
        _LANCE_CACHE = {}
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
