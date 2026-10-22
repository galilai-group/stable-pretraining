"""Select cached-feature classifiers using only a training development split."""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

_spec = importlib.util.spec_from_file_location(
    "directclr_probe_benchmark", Path(__file__).with_name("directclr.py")
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def _label_digest(labels: torch.Tensor) -> str:
    return hashlib.sha256(labels.long().contiguous().numpy().tobytes()).hexdigest()


def _training_data(cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = cache["x_train"], cache["y_train"]
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y) or x.shape[1] == 0:
        raise ValueError("training features and labels must align and be nonempty")
    if (
        y.dtype != torch.int64
        or not x.is_floating_point()
        or not torch.isfinite(x).all()
    ):
        raise ValueError("expected finite features and integer class labels")
    classes = torch.unique(y, sorted=True)
    if not torch.equal(classes, torch.arange(len(classes))):
        raise ValueError("training classes must be contiguous and start at zero")
    return x, y


def _stratified_split(
    labels: torch.Tensor, fraction: float = 0.1, seed: int = 2026
) -> dict:
    if labels.ndim != 1 or not 0 < fraction < 1:
        raise ValueError(
            "expected a label vector and fraction strictly between 0 and 1"
        )
    generator = torch.Generator().manual_seed(seed)
    fit, development = [], []
    for label in torch.unique(labels, sorted=True):
        indices = torch.where(labels == label)[0]
        if len(indices) < 2:
            raise ValueError("each class needs at least two training samples")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        development.append(indices[:count])
        fit.append(indices[count:])
    if not fit:
        raise ValueError("training labels are empty")
    return {
        "fit_indices": torch.cat(fit).sort().values,
        "development_indices": torch.cat(development).sort().values,
        "labels_sha256": _label_digest(labels),
        "fraction": fraction,
        "seed": seed,
    }


def _validate_split(split: dict, labels: torch.Tensor) -> None:
    if split["labels_sha256"] != _label_digest(labels):
        raise ValueError("split label ordering does not match this encoder cache")
    fit, development = split["fit_indices"], split["development_indices"]
    joined = torch.cat((fit, development))
    if (
        fit.dtype != torch.int64
        or development.dtype != torch.int64
        or not len(fit)
        or not len(development)
        or not torch.equal(joined.sort().values, torch.arange(len(labels)))
    ):
        raise ValueError("split must partition all training rows without overlap")
    for indices in (fit, development):
        if not torch.equal(torch.unique(labels[indices]), torch.unique(labels)):
            raise ValueError("both split partitions must contain every class")


def _validate_registered_split(split: dict, labels: torch.Tensor) -> None:
    _validate_split(split, labels)
    if split["seed"] != 2026 or split["fraction"] != 0.1:
        raise ValueError("expected the preregistered seed=2026, fraction=0.1 split")


@torch.inference_mode()
def _accuracy(module, features: torch.Tensor, labels: torch.Tensor) -> dict:
    correct1 = correct5 = 0
    for start in range(0, len(labels), 1024):
        x = nn.functional.normalize(features[start : start + 1024], dim=1)
        scores = module.classifier(x.to(module.device))
        if not torch.isfinite(scores).all():
            raise ValueError("classifier produced nonfinite predictions")
        predicted = scores.topk(min(5, scores.shape[1]), dim=1).indices.cpu()
        y = labels[start : start + 1024]
        correct1 += (predicted[:, 0] == y).sum().item()
        correct5 += (predicted == y[:, None]).any(1).sum().item()
    return {"top1": correct1 / len(labels), "top5": correct5 / len(labels)}


def _select(candidates: list[dict]) -> dict:
    if not candidates or any(
        not math.isfinite(c["development_top1"]) for c in candidates
    ):
        raise ValueError("selection requires finite development scores")
    return min(candidates, key=lambda c: (-c["development_top1"], c["learning_rate"]))


def _probe_args(
    output: Path, lr: float, args: argparse.Namespace
) -> argparse.Namespace:
    return argparse.Namespace(
        output=output,
        probe_lr=lr,
        probe_epochs=args.epochs,
        probe_batch_size=args.batch_size,
        accelerator=args.accelerator,
        cpu_threads=args.cpu_threads,
        workers=0,
        seed=args.seed,
    )


def _fit(x, y, classes, manifest, output, lr, args):
    output.mkdir(parents=True, exist_ok=False)
    probe_args = _probe_args(output, lr, args)
    benchmark._configure(probe_args)
    return benchmark._fit_linear_classifier(x, y, classes, probe_args, manifest)


def _gpu_smoke(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each worker must receive exactly one CUDA device")
    properties = torch.cuda.get_device_properties(0)
    expected = getattr(args, "expected_gpu_name", None)
    if expected is not None and properties.name != expected:
        raise RuntimeError(f"expected {expected}, received {properties.name}")
    toy = argparse.Namespace(**vars(args))
    toy.epochs = 20
    toy.batch_size = 12
    labels = torch.arange(3).repeat_interleave(24)
    features = nn.functional.one_hot(labels, 8).float()
    directory = args.output / ("smoke-" + args.methods[0])
    module = _fit(features, labels, 3, {"synthetic_smoke": True}, directory, 1.0, toy)
    score = _accuracy(module, features, labels)
    if score["top1"] < 0.95:
        raise RuntimeError("synthetic GPU classifier failed to learn separable data")
    benchmark._save_json(
        directory / "gpu-smoke.json",
        {
            "device": properties.name,
            "device_uuid": str(properties.uuid),
            "visible_devices": torch.cuda.device_count(),
            "synthetic_top1": score["top1"],
            "not_representation_quality_evidence": True,
        },
    )
    del module
    gc.collect()
    torch.cuda.empty_cache()
    print("CUDA_CLASSIFIER_SMOKE_PASSED", flush=True)


def _feature_directory(method: str, args: argparse.Namespace) -> Path:
    return (
        args.source
        / (
            "imagenet100-r50-random-seed0"
            if method == "random"
            else f"imagenet100-r50-{getattr(args, 'encoder_epochs', 100)}ep-{method}-seed0"
        )
        / "evaluation"
    )


def _run_method(method: str, args: argparse.Namespace) -> None:
    source = _feature_directory(method, args)
    manifest = json.loads((source / "feature_manifest.json").read_text())
    # The packed cache includes held-out tensors. Selection only accesses the
    # two training keys; final held-out scoring occurs after selection is saved.
    cache = torch.load(source / "features.pt", map_location="cpu", weights_only=True)
    x, y = _training_data(cache)
    split = torch.load(args.split, map_location="cpu", weights_only=True)
    _validate_registered_split(split, y)
    fit, development = split["fit_indices"], split["development_indices"]
    classes = len(torch.unique(y))
    root = args.output / method
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    candidates = []
    for lr in args.learning_rates:
        path = root / ("lr-" + format(lr, ".8g").replace(".", "p"))
        module = _fit(x[fit], y[fit], classes, manifest, path, lr, args)
        score = _accuracy(module, x[development], y[development])
        candidate = {
            "learning_rate": lr,
            "development_top1": score["top1"],
            "development_top5": score["top5"],
        }
        benchmark._save_json(path / "development.json", candidate)
        candidates.append(candidate)
        del module
        gc.collect()
        if args.accelerator == "gpu":
            torch.cuda.empty_cache()
    selected = _select(candidates)
    selection = {
        "method": method,
        "candidates": candidates,
        "selected_learning_rate": selected["learning_rate"],
        "criterion": "final_epoch_development_top1_then_smaller_learning_rate",
        "split_seed": split["seed"],
        "labels_sha256": split["labels_sha256"],
        "fit_samples": len(fit),
        "development_samples": len(development),
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "final_validation_used_for_selection": False,
        "development_images_seen_in_ssl_pretraining": True,
        "encoder_manifest": manifest,
    }
    benchmark._save_json(root / "selection.json", selection)
    module = _fit(
        x, y, classes, manifest, root / "refit", selected["learning_rate"], args
    )
    score = _accuracy(module, cache["x_val"], cache["y_val"])
    benchmark._save_json(
        root / "evaluation.json",
        {
            **selection,
            "linear_top1": score["top1"],
            "linear_top5": score["top5"],
            "validation_samples": len(cache["y_val"]),
            "elapsed_seconds": time.perf_counter() - started,
            "protocol": "cached_center_crop_l2_probe_development_lr_selection_full_train_refit_final_validation",
        },
    )
    torch.save(module.classifier.state_dict(), root / "linear_classifier.pt")
    print("PROBE_SWEEP_METHOD_COMPLETED=" + method, flush=True)


def main() -> None:
    """Prepare the development split or run cached-feature probe selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-split", "sweep"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["directclr", "linear", "full", "random"],
        default=["directclr"],
    )
    parser.add_argument(
        "--learning-rates", nargs="+", type=float, default=[1.0, 3.0, 10.0]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--encoder-epochs",
        type=int,
        default=100,
        help="Encoder run directory schedule; classifier epochs remain independent",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--smoke-first", action="store_true")
    parser.add_argument(
        "--expected-gpu-name",
        default=None,
        help="Optional exact GPU model check for --smoke-first",
    )
    args = parser.parse_args()
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.learning_rates):
        parser.error("learning rates must be finite and positive")
    if min(args.epochs, args.encoder_epochs, args.batch_size, args.cpu_threads) < 1:
        parser.error("epoch, batch and thread counts must be positive")
    if len(set(args.methods)) != len(args.methods) or len(
        set(args.learning_rates)
    ) != len(args.learning_rates):
        parser.error("methods and learning rates must be unique")
    torch.set_num_threads(args.cpu_threads)
    if args.command == "prepare-split":
        source = _feature_directory("directclr", args) / "features.pt"
        _, labels = _training_data(
            torch.load(source, map_location="cpu", weights_only=True)
        )
        if args.split.exists():
            _validate_registered_split(
                torch.load(args.split, weights_only=True), labels
            )
        else:
            args.split.parent.mkdir(parents=True, exist_ok=True)
            split = _stratified_split(labels)
            _validate_registered_split(split, labels)
            torch.save(split, args.split)
        print("TRAINING_DEVELOPMENT_SPLIT_READY", flush=True)
    else:
        if args.smoke_first:
            if args.accelerator != "gpu":
                parser.error("smoke-first requires the GPU accelerator")
            _gpu_smoke(args)
        for method in args.methods:
            _run_method(method, args)


if __name__ == "__main__":
    main()
