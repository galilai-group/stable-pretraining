"""Prepare MIMIC-III data, pretrain COSTAR, and train an outcome predictor.

Data: https://zenodo.org/record/8412226/files/all_needed_data.tar.gz
Install this repository with ``pip install -e .``, then run:

    python -m examples.costar_validation --archive /path/to/all_needed_data.tar.gz --output runs/costar
    python -m examples.costar_validation --data-dir runs/costar/data --ssl-epochs 0 --output runs/supervised

--archive also accepts the extracted archive directory. Preparation needs about
40 GB of free disk space for one seed; the large temporary pickle is removed.
Use --target-finetune to continue from the best source model on target data.
SSL uses the repository COSTAR implementation for a fixed epoch budget.
Supervised training selects the lowest validation loss, with patience 20.
Prepared splits contain dynamic_history[N,T,30], static_features[N,44],
current_treatments[N,T,3], next_outcomes[N,T,2], and bool valid_mask[N,T].
Arrays are stored as data/seed{seed}/{split}/{name}.npy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
import gc
import gzip
import math
import mmap
import os
from pathlib import Path
import pickle
import shutil
import struct
import tarfile
import tempfile
from typing import Any

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt

HORIZONS = 11
INPUT_KEYS = ("dynamic_history", "static_features", "valid_mask")
OFFICIAL_PREFIX = (
    "multirun/230814_mimicsyn_all_data/"
    "mimic3_synthetic_age_domain_0-3_all/CRN_NONCAUSAL_TROFF"
)


def require(condition: bool, message: str) -> None:
    """Reject incompatible official data before training."""
    if not condition:
        raise ValueError(message)


class PassiveRecord:
    """Retain stored attributes without running the original class."""

    def __new__(cls, *args: Any, **kwargs: Any) -> PassiveRecord:
        return object.__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._constructor_args = args

    def __setstate__(self, state: Any) -> None:
        for part in state if isinstance(state, tuple) else (state,):
            if isinstance(part, dict):
                self.__dict__.update(part)


class DiskBytes:
    """Refer to an array payload without copying its bytes."""

    def __init__(self, mapping: mmap.mmap, offset: int, length: int) -> None:
        self.mapping, self.offset, self.length = mapping, offset, length


class ArrayRecord:
    """Reconstruct NumPy storage as numeric views or passive object arrays."""

    def __setstate__(self, state: Any) -> None:
        if not isinstance(state, tuple) or len(state) != 5:
            raise pickle.UnpicklingError("Unsupported NumPy array state.")
        version, shape, dtype, fortran, raw = state
        dtype = np.dtype(dtype)
        if version not in (0, 1) or not isinstance(shape, tuple):
            raise pickle.UnpicklingError("Unsupported NumPy array format.")
        if any(not isinstance(size, int) or size < 0 for size in shape):
            raise pickle.UnpicklingError("Invalid NumPy array shape.")
        count, order = math.prod(shape), "F" if fortran else "C"
        if isinstance(raw, list) and dtype.hasobject:
            if count != len(raw):
                raise pickle.UnpicklingError("Object array size mismatch.")
            self.array = np.empty(shape, dtype=dtype, order=order)
            for index, value in enumerate(raw):
                self.array.flat[index] = value
        elif isinstance(raw, (DiskBytes, bytes)) and not dtype.hasobject:
            size = raw.length if isinstance(raw, DiskBytes) else len(raw)
            if count * dtype.itemsize != size:
                raise pickle.UnpicklingError("Numeric array size mismatch.")
            self.array = (
                np.ndarray(shape, dtype, raw.mapping, raw.offset, order=order)
                if isinstance(raw, DiskBytes)
                else np.frombuffer(raw, dtype=dtype).reshape(shape, order=order)
            )
        else:
            raise pickle.UnpicklingError("Unsupported NumPy array storage.")


def _new_array(*args: Any) -> ArrayRecord:
    return ArrayRecord()


def _new_index(cls: type[PassiveRecord], state: Any) -> PassiveRecord:
    index = cls()
    index.__setstate__(state)
    return index


class DataUnpickler(pickle._Unpickler):
    """Use local passive objects and map large pickle byte payloads."""

    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, stream: Any, mapping: mmap.mmap) -> None:
        super().__init__(stream)
        self.stream, self.mapping = stream, mapping

    def find_class(self, module: str, name: str) -> Any:
        constructors = {
            ("builtins", "dict"): dict,
            ("builtins", "int"): int,
            ("builtins", "list"): list,
            ("builtins", "slice"): slice,
            ("collections", "defaultdict"): defaultdict,
            ("numpy", "dtype"): np.dtype,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy.core.multiarray", "_reconstruct"): _new_array,
            ("numpy._core.multiarray", "_reconstruct"): _new_array,
            ("pandas.core.indexes.base", "_new_Index"): _new_index,
            ("typing", "Any"): Any,
        }
        if (module, name) in constructors:
            return constructors[module, name]
        if module.startswith(("src.", "pandas.", "sklearn.", "omegaconf.")):
            return PassiveRecord
        raise pickle.UnpicklingError(f"Unsupported saved global: {module}.{name}")

    def get_extension(self, code: int) -> None:
        raise pickle.UnpicklingError("Pickle extensions are not supported.")

    def _payload(self, size: int) -> None:
        if size < 1024 * 1024:
            self.append(self.read(size))
            return
        frame = self._unframer.current_frame
        if frame is not None:
            if frame.tell() != len(frame.getbuffer()):
                raise pickle.UnpicklingError("Large payload inside a pickle frame.")
            self._unframer.current_frame = None
        offset = self.stream.tell()
        if offset + size > len(self.mapping):
            raise pickle.UnpicklingError("Array payload exceeds the cache file.")
        self.stream.seek(size, os.SEEK_CUR)
        self.append(DiskBytes(self.mapping, offset, size))

    def load_binbytes(self) -> None:
        self._payload(struct.unpack("<I", self.read(4))[0])

    def load_binbytes8(self) -> None:
        self._payload(struct.unpack("<Q", self.read(8))[0])

    dispatch[pickle.BINBYTES[0]] = load_binbytes
    dispatch[pickle.BINBYTES8[0]] = load_binbytes8


def _hydrate(value: Any, seen: dict[int, Any]) -> Any:
    identity = id(value)
    if identity in seen:
        return seen[identity]
    if isinstance(value, ArrayRecord):
        seen[identity] = value.array
        return value.array
    seen[identity] = value
    if isinstance(value, PassiveRecord):
        for key, item in list(vars(value).items()):
            vars(value)[key] = _hydrate(item, seen)
    elif isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _hydrate(item, seen)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _hydrate(item, seen)
    elif isinstance(value, tuple):
        value = tuple(_hydrate(item, seen) for item in value)
        seen[identity] = value
    return value


def load_collection(raw_path: Path, convert: Callable[[Any], None]) -> None:
    """Call a converter that writes its outputs without retaining cache views."""
    with raw_path.open("rb") as stream:
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        loader, collection = None, None
        try:
            loader = DataUnpickler(stream, mapping)
            collection = _hydrate(loader.load(), {})
            if stream.read(1):
                raise pickle.UnpicklingError("Trailing bytes after the saved data.")
            convert(collection)
        finally:
            collection, loader = None, None
            gc.collect()
            mapping.close()


def _factual_arrays(
    raw: dict, mean: np.ndarray, std: np.ndarray, indices: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Retain official input alignment and standardize outcomes using source statistics."""
    keys = (
        "active_entries",
        "vitals",
        "prev_treatments",
        "prev_unscaled_outputs",
        "current_treatments",
        "unscaled_outputs",
        "static_features",
    )
    data = {key: raw[key] if indices is None else raw[key][indices] for key in keys}
    active = data["active_entries"]
    require(active.shape[1:] == (99, 1), "Expected masks [N,99,1].")
    valid = active[..., 0].astype(bool)
    count = len(valid)
    require(
        count > 0
        and np.isin(active, (0, 1)).all()
        and valid[:, 0].all()
        and not np.any(valid[:, 1:] & ~valid[:, :-1]),
        "Each patient needs a nonempty valid prefix.",
    )
    dimensions = (25, 3, 2, 3, 2)
    require(
        all(
            data[key].shape == (count, 99, width)
            for key, width in zip(keys[1:6], dimensions, strict=True)
        )
        and data["static_features"].shape == (count, 44),
        "Unexpected MIMIC feature dimensions.",
    )
    previous = (data["prev_unscaled_outputs"] - mean) / std
    history = np.concatenate((data["vitals"], data["prev_treatments"], previous), -1)
    arrays = {
        "dynamic_history": history.astype(np.float32),
        "static_features": data["static_features"].astype(np.float32),
        "valid_mask": valid,
        "current_treatments": data["current_treatments"].astype(np.float32),
        "next_outcomes": ((data["unscaled_outputs"] - mean) / std).astype(np.float32),
    }
    for key in ("dynamic_history", "current_treatments", "next_outcomes"):
        arrays[key][~valid] = 0
    require(
        all(np.isfinite(value).all() for value in arrays.values()),
        "Nonfinite observed values.",
    )
    return arrays


def prepare_collection(collection: Any, destination: Path, seed: int) -> None:
    """Export the four factual splits from one official collection."""
    require(int(collection.seed) == seed, "Official data seed differs.")
    mean, std = (
        np.asarray(collection.train_scaling_params[key], dtype=np.float64)
        for key in ("output_means", "output_stds")
    )
    require(
        mean.shape == std.shape == (2,)
        and np.isfinite((mean, std)).all()
        and (std > 0).all(),
        "Invalid source outcome normalization.",
    )
    destination.mkdir(parents=True, exist_ok=True)
    require(not any(destination.iterdir()), "Prepare into an empty directory.")
    for name, attribute in (
        ("source_train", "src_train_f"),
        ("source_val", "src_val_f"),
        ("target_train", "test_train_f"),
        ("target_val", "test_val_f"),
    ):
        raw, indices = getattr(collection, attribute).data, None
        if name == "target_train":
            count = len(raw["active_entries"])
            indices = np.random.RandomState(seed).choice(
                count, int(0.1 * count), replace=False
            )
        arrays = _factual_arrays(raw, mean, std, indices)
        directory = destination / name
        directory.mkdir()
        for key, value in arrays.items():
            with (directory / f"{key}.npy").open("xb") as stream:
                np.save(stream, value, allow_pickle=False)


def prepare_data(source: Path, output: Path, seed: int) -> Path:
    """Read one official cohort and write only the factual training splits."""
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=False)
    member_name = f"{OFFICIAL_PREFIX}/{seed}/data/dataset_collection.pt"

    with ExitStack() as stack:
        if source.is_dir():
            stream = stack.enter_context((source / member_name).open("rb"))
        else:
            archive = stack.enter_context(tarfile.open(source, "r|gz"))
            member = next(
                (
                    item
                    for item in archive
                    if item.name.removeprefix("./") == member_name
                ),
                None,
            )
            if member is None or not member.isfile():
                raise FileNotFoundError(f"Missing regular cohort file: {member_name}")
            stream = stack.enter_context(archive.extractfile(member))
        print(f"Preparing seed {seed}...", flush=True)
        # Each .pt is a gzip pickle, with individual array payloads over 2 GB.
        temporary = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="raw-", dir=output)
        )
        raw = Path(temporary) / "dataset.pkl"
        with gzip.GzipFile(fileobj=stream) as decoded, raw.open("xb") as target:
            shutil.copyfileobj(decoded, target, length=8 * 1024 * 1024)
        load_collection(
            raw,
            lambda collection: prepare_collection(
                collection, output / f"seed{seed}", seed
            ),
        )
    return output


class MimicDataset(Dataset):
    """Read a prepared split and optionally construct future treatment/target pairs."""

    def __init__(
        self, data_dir: Path, seed: int, split: str, supervised: bool = True
    ) -> None:
        root = data_dir.expanduser().resolve() / f"seed{seed}" / split
        keys = INPUT_KEYS + (
            ("current_treatments", "next_outcomes") if supervised else ()
        )
        self.arrays = {
            key: np.load(root / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            for key in keys
        }
        self.supervised = supervised
        length = self.arrays["valid_mask"].shape[1]
        future = np.arange(length)[:, None] + np.arange(HORIZONS)[None]
        self.indices, self.in_range = np.minimum(future, length - 1), future < length

    def __len__(self) -> int:
        return len(self.arrays["valid_mask"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = {
            key: torch.from_numpy(self.arrays[key][index].copy()) for key in INPUT_KEYS
        }
        if self.supervised:
            mask = self.arrays["valid_mask"][index][self.indices] & self.in_range
            batch["target_mask"] = torch.from_numpy(mask)
            for source, target in (
                ("current_treatments", "future_treatments"),
                ("next_outcomes", "target"),
            ):
                values = self.arrays[source][index][self.indices].copy()
                values[~mask] = 0
                batch[target] = torch.from_numpy(values)
        return batch


class OutcomeHead(nn.Module):
    """Predict eleven horizons using history, current treatment and preceding plans."""

    def __init__(self) -> None:
        super().__init__()
        self.current = nn.Linear(3, 36)
        self.preceding = nn.Conv1d(3, 36, HORIZONS)
        self.predictor = nn.Sequential(nn.Linear(36, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(
        self, history: torch.Tensor, plans: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, times, horizons, treatments = plans.shape
        history = torch.where(mask.any(-1, keepdim=True), history, 0.0)
        plans = torch.where(mask[..., None], plans, 0.0)
        # Each horizon sees preceding planned treatments, excluding its current A.
        preceding = torch.cat((torch.zeros_like(plans[:, :, :1]), plans[:, :, :-1]), 2)
        preceding = preceding.reshape(batch * times, horizons, treatments).transpose(
            1, 2
        )
        prefix = self.preceding(F.pad(preceding, (HORIZONS - 1, 0)))
        prefix = prefix.transpose(1, 2).reshape(batch, times, horizons, 36)
        prediction = self.predictor(history[:, :, None] + prefix + self.current(plans))
        return torch.where(mask[..., None], prediction, 0.0)


def supervised_forward(
    self: spt.Module, batch: dict[str, torch.Tensor], stage: str
) -> dict[str, torch.Tensor]:
    """Fine-tune the encoder and prediction head with inverse-horizon weighted MSE."""
    tokens, _ = self.encoder(
        batch["dynamic_history"], batch["static_features"], batch["valid_mask"]
    )
    history = self.pooling(tokens, batch["valid_mask"])["history"]
    mask = batch["target_mask"]
    prediction = self.head(history, batch["future_treatments"], mask)
    target = torch.where(mask[..., None], batch["target"], 0.0)
    weights = (
        torch.arange(1, HORIZONS + 1, device=prediction.device).float().reciprocal()
    )
    weights = weights * (HORIZONS / weights.sum())
    errors = F.mse_loss(prediction, target, reduction="none")
    loss = (
        torch.einsum("bthy,h->bthy", errors, weights) * mask[..., None]
    ).sum() / mask.sum()
    self.log(
        "train/loss" if stage == "fit" else "val/loss",
        loss,
        on_step=False,
        on_epoch=True,
        batch_size=len(history),
    )
    return {"loss": loss, "prediction": prediction}


def fit(
    model: pl.LightningModule,
    data_dir: Path,
    args: argparse.Namespace,
    stage: str,
) -> None:
    """Train and restore supervised BEST or SSL LAST weights."""
    pl.seed_everything(args.seed, workers=True)
    supervised = stage != "ssl"
    output = args.output / stage

    def loader(split: str) -> DataLoader:
        return DataLoader(
            MimicDataset(data_dir, args.seed, split, supervised),
            batch_size=args.batch_size,
            shuffle=split.endswith("train"),
        )

    domain = stage if supervised else "source"
    train = loader(domain + "_train")
    val = loader(domain + "_val") if supervised else None
    spt.set(
        cache_dir=str(output.resolve()),
        requeue_checkpoint=False,
        default_callbacks={"env_dump": False},
    )
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints",
        monitor="val/loss" if supervised else None,
        mode="min",
        filename="best",
        save_top_k=int(supervised),
        save_last=not supervised,
        save_weights_only=True,
    )
    callbacks = [checkpoint]
    if supervised:
        callbacks.append(EarlyStopping(monitor="val/loss", patience=20, mode="min"))
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs if supervised else args.ssl_epochs,
        default_root_dir=output,
        callbacks=callbacks,
        logger=False,
        precision="32-true",
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if supervised else 0,
    )
    data = spt.data.DataModule(train=train, val=val)
    if os.name == "nt":
        # Manager's POSIX signal diagnostics are unavailable on Windows.
        trainer.fit(model, datamodule=data)
    else:
        spt.Manager(trainer=trainer, module=model, data=data, seed=args.seed)()
    path = checkpoint.best_model_path or checkpoint.last_model_path
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    )
    print(f"Saved model: {path}")


def main() -> None:
    """Run preparation, optional SSL, source supervision, and optional target tuning."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--archive", type=Path)
    inputs.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/costar"))
    parser.add_argument("--seed", type=int, choices=(17, 43, 44, 91, 95), default=17)
    parser.add_argument("--ssl-epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--target-finetune", action="store_true")
    args = parser.parse_args()
    if args.ssl_epochs < 0 or args.epochs < 1 or args.batch_size < 1:
        parser.error("Use ssl-epochs >= 0, epochs >= 1, and batch-size >= 1.")
    data_dir = (
        prepare_data(args.archive, args.output / "data", args.seed)
        if args.archive
        else args.data_dir
    )
    pl.seed_everything(args.seed, workers=True)

    ssl = spt.COSTAR(
        covariate_dim=25, treatment_dim=3, outcome_dim=2, static_dim=44, lr=args.lr
    )
    if args.ssl_epochs:
        fit(ssl, data_dir, args, "ssl")
    pl.seed_everything(args.seed, workers=True)
    model = spt.Module(
        forward=supervised_forward,
        encoder=ssl.encoder,
        pooling=ssl.pooling,
        head=OutcomeHead(),
        optim={"optimizer": partial(torch.optim.Adam, lr=args.lr)},
    )
    del ssl
    for domain in ("source", "target") if args.target_finetune else ("source",):
        fit(model, data_dir, args, domain)


if __name__ == "__main__":
    main()
