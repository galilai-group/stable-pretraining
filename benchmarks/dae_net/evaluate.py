"""Evaluate DAE-Net occupancy reconstruction or semantic surface segmentation."""

import argparse
import csv
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import stable_pretraining as spt

import part_segmentation_evaluation as metrics
from occupancy_evaluation import OccupancyCounts
from surface_labels import DAENetSurfaceLabels


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    run_config: dict[str, Any] | None,
    require_final: bool,
) -> bool | None:
    """Check saved weights and, when configured, completed-epoch identity."""
    state = checkpoint["state_dict"]
    for name, tensor in state.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
    if run_config is None:
        if require_final:
            raise ValueError("Final checkpoint validation requires run-config")
        return None
    if checkpoint.get("hyper_parameters", {}).get("run_settings") != run_config:
        raise ValueError("Checkpoint belongs to a different run configuration")
    for key in ("epochs", "updates_per_epoch", "era_length"):
        if type(run_config.get(key)) is not int or run_config[key] <= 0:
            raise ValueError(f"run-config requires a positive integer {key}")
    epoch, step = checkpoint.get("epoch"), checkpoint.get("global_step")
    if type(epoch) is not int or not 0 <= epoch < run_config["epochs"]:
        raise ValueError("Checkpoint epoch is outside the configured budget")
    if type(step) is not int or step != (epoch + 1) * run_config["updates_per_epoch"]:
        raise ValueError("Checkpoint is not a complete configured epoch")
    complete = epoch == run_config["epochs"] - 1
    if require_final:
        if not complete:
            raise ValueError("Final evaluation requires the final epoch")
        era_length = run_config["era_length"]
        if (epoch + 1) % era_length:
            raise ValueError("Final checkpoint must end a complete era")
        controller = checkpoint.get("callbacks", {}).get("DAENetRevival", {})
        accepted_epoch = controller.get("accepted_epoch")
        accepted = controller.get("accepted_weights")
        if (
            type(accepted_epoch) is not int
            or not 0 <= accepted_epoch <= epoch
            or (accepted_epoch + 1) % era_length
            or not isinstance(accepted, dict)
            or not accepted
        ):
            raise ValueError("Missing or invalid accepted era snapshot")
        current = {
            key.removeprefix("backbone."): value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
        if current.keys() != accepted.keys() or any(
            not torch.equal(current[key], accepted[key]) for key in current
        ):
            raise ValueError("Final weights do not match the accepted era snapshot")
    return complete


@torch.inference_mode()
def _evaluate_shape(
    backbone: torch.nn.Module,
    dataset: Any,
    index: int,
    device: torch.device,
    chunk_size: int,
    threshold: float,
) -> dict[str, Any]:
    sample = dataset[index, 0]
    encoded = backbone.encode(sample["voxels"].unsqueeze(0).to(device))
    resolution = dataset.records[index]["query_resolution"]
    counts = OccupancyCounts()
    query_digest = hashlib.sha256()
    for offset in range(4 if resolution == 64 else 1):
        if offset:
            sample = dataset[index, offset]
        # Hash the actual evaluation inputs, not only the manifest's file paths.
        query_digest.update(sample["points"].numpy().tobytes())
        query_digest.update(sample["occupancy"].numpy().tobytes())
        for start in range(0, len(sample["points"]), chunk_size):
            points = sample["points"][start : start + chunk_size].to(device)
            target = sample["occupancy"][start : start + chunk_size].to(device)
            prediction = backbone.decode(points.unsqueeze(0), encoded)["occupancy_max"][
                0
            ]
            counts.update(prediction, target, threshold)
    return {
        "shape_id": sample["shape_id"],
        "query_resolution": resolution,
        **asdict(counts),
        "iou": counts.iou,
        "queries_sha256": query_digest.hexdigest(),
        "voxels_sha256": hashlib.sha256(sample["voxels"].numpy().tobytes()).hexdigest(),
    }


@torch.inference_mode()
def _evaluate_semantic(
    backbone: torch.nn.Module,
    dataset: Any,
    surfaces: Any,
    num_shapes: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Infer surface branches once, then score raw and released postprocessing."""
    rows, raw_counts, counts = [], [], []
    cache = {
        key: []
        for key in ("points", "labels", "raw_branches", "branches", "confidence")
    }
    offsets = [0]
    for index in range(num_shapes):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        sample, surface = dataset[index], surfaces[index]
        if sample["shape_id"] != surface["shape_id"]:
            raise ValueError("Voxel/surface shape ID mismatch")
        points = surface["points"]
        encoded = backbone.encode(sample["voxels"].unsqueeze(0).to(device))
        predictions, confidences = [], []
        for start in range(0, len(points), chunk_size):
            query = points[start : start + chunk_size]
            parts = backbone.decode(query.unsqueeze(0).to(device), encoded)[
                "part_occupancy"
            ]
            if (
                parts.shape != (1, len(query), backbone.branch_num)
                or not torch.isfinite(parts).all()
                or torch.any(parts < 0)
                or torch.any(parts > 1)
            ):
                raise ValueError("Invalid existence-weighted part occupancies")
            confidence, branch = parts[0].max(dim=-1)
            predictions.append(branch.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
        raw = np.concatenate(predictions)
        confidence = np.concatenate(confidences)
        target = surface["labels"].numpy()
        processed, fallback = metrics.propagate_confident_labels(
            points.numpy(), raw, confidence
        )
        for collection, prediction in ((raw_counts, raw), (counts, processed)):
            collection.append(
                metrics.branch_label_counts(
                    prediction,
                    target,
                    num_branches=backbone.branch_num,
                    num_parts=len(surfaces.label_ids),
                )
            )
        for key, value in zip(
            cache, (points.numpy(), target, raw, processed, confidence)
        ):
            cache[key].append(value)
        offsets.append(offsets[-1] + len(points))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rows.append(
            {
                "shape_id": sample["shape_id"],
                "queries": len(points),
                "no_confident_points": fallback,
                "source_points_sha256": surface["source_points_sha256"],
                "voxels_sha256": hashlib.sha256(
                    sample["voxels"].numpy().tobytes()
                ).hexdigest(),
                "evaluation_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"{index + 1}/{num_shapes} {sample['shape_id']}: {len(points)} surface points",
            flush=True,
        )
    counts, raw_counts = np.stack(counts), np.stack(raw_counts)
    started = time.perf_counter()
    fitted = metrics.fit_branch_mapping(counts, protocol="dae_net_release_v1")
    raw_fitted = metrics.fit_branch_mapping(raw_counts, protocol="dae_net_release_v1")
    mapping_seconds = time.perf_counter() - started
    for index, row in enumerate(rows):
        row["miou"] = float(fitted.scores.shape_miou[index])
        row["raw_miou"] = float(raw_fitted.scores.shape_miou[index])
        for part, label_id in enumerate(surfaces.label_ids):
            row[f"part_{label_id}_iou"] = float(
                fitted.scores.shape_part_iou[index, part]
            )
    per_part = [
        {
            "label_id": label_id,
            "mean_iou": float(fitted.scores.per_part_iou[part]),
            "raw_mean_iou": float(raw_fitted.scores.per_part_iou[part]),
            "pooled_iou": float(fitted.scores.pooled_part_iou[part]),
            "gt_present_shapes": int(fitted.scores.gt_present_shapes[part]),
            "empty_union_shapes": int(fitted.scores.empty_union_shapes[part]),
        }
        for part, label_id in enumerate(surfaces.label_ids)
    ]
    mapping = {
        "protocol": "dae_net_release_v1",
        "label_ids": list(surfaces.label_ids),
        "fit_shape_ids": [row["shape_id"] for row in rows],
        "mapping_uses_evaluation_labels": True,
    }
    for key, result in (("postprocessed", fitted), ("raw", raw_fitted)):
        mapping[key] = {
            "branch_to_label_index": result.mapping.tolist(),
            "branch_to_label_id": [surfaces.label_ids[i] for i in result.mapping],
            "unused_branches": list(result.unused_branches),
            "actual_miou": result.scores.category_miou,
            "reference_reported_miou": result.reference_reported_miou,
        }
    summary = {
        "metric": "semantic_part_miou",
        "semantic_part_miou": fitted.scores.category_miou,
        "raw_semantic_part_miou": raw_fitted.scores.category_miou,
        "reference_reported_miou": fitted.reference_reported_miou,
        "raw_reference_reported_miou": raw_fitted.reference_reported_miou,
        "protocol": "dae_net_release_v1",
        "upstream_commit": "4f4af22519940b774ae47e9c8d9a99fbfd17ac9b",
        "coordinate_transform": "(x,y,z) -> (z,y,-x); no rescaling",
        "postprocessing": "KDTree nearest point with occupancy_max > 0.4; leaf_size=8",
        "confidence_threshold": 0.4,
        "empty_part_iou": 1.0,
        "score_units": "fraction",
        "mapping_uses_evaluation_labels": True,
        "mapping_seconds": mapping_seconds,
        "label_ids": list(surfaces.label_ids),
        "synset": surfaces.synset,
        "labels_manifest": str(surfaces.manifest.resolve()),
        "labels_manifest_sha256": surfaces.manifest_sha256,
        "num_surface_points": offsets[-1],
        "no_confident_shapes": sum(row["no_confident_points"] for row in rows),
        "threshold": None,
    }
    predictions = {key: np.concatenate(value) for key, value in cache.items()}
    predictions.update(
        shape_offsets=np.array(offsets, dtype=np.int64),
        shape_ids=np.array([row["shape_id"] for row in rows]),
        label_ids=np.array(surfaces.label_ids, dtype=np.int64),
        counts=counts,
        raw_counts=raw_counts,
    )
    return (
        rows,
        summary,
        {"per_part": per_part, "mapping": mapping, "predictions": predictions},
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    """Write reconstruction or semantic metrics and artifacts without training.

    Args:
        argv: CLI arguments, or None to read the process command line.

    Raises:
        ValueError: If the checkpoint is not a supported native DAE-Net model.
        FileExistsError: If the output directory contains existing results.

    Note:
        Uses current saved weights, including any completed era rollback.
        Checkpoints must be trusted. Standard backbone widths are required.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--metric", choices=("occupancy", "semantic"), default="occupancy"
    )
    parser.add_argument(
        "--labels-manifest",
        type=Path,
        help="Audited surface-label JSONL for semantic evaluation.",
    )
    parser.add_argument(
        "--run-config", type=Path, help="Frozen training configuration JSON."
    )
    parser.add_argument(
        "--max-shapes", type=int, help="Evaluate a manifest prefix for smoke checks."
    )
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="Require the final configured epoch and era outcome.",
    )
    parser.add_argument(
        "--snapshot-checkpoint",
        action="store_true",
        help="Keep a checkpoint copy in the output directory.",
    )
    args = parser.parse_args(argv)
    if args.chunk_size < 1 or not 0 < args.threshold < 1:
        parser.error("chunk-size must be positive and threshold must be in (0, 1)")
    if args.max_shapes is not None and args.max_shapes < 1:
        parser.error("max-shapes must be positive")
    if args.require_final and args.run_config is None:
        parser.error("require-final requires run-config")
    if (args.metric == "semantic") != (args.labels_manifest is not None):
        parser.error(
            "semantic requires labels-manifest; occupancy does not use labels-manifest"
        )
    if args.metric == "semantic" and args.threshold != 0.5:
        parser.error(
            "threshold controls occupancy only; semantic uses the fixed release protocol"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    started = time.perf_counter()
    run_config = None
    run_config_sha256 = None
    if args.run_config is not None:
        config_bytes = args.run_config.read_bytes()
        run_config = json.loads(config_bytes)
        if not isinstance(run_config, dict):
            raise ValueError("run-config must be a JSON object")
        run_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    checkpoint_path = args.checkpoint
    if args.snapshot_checkpoint:
        snapshot_dir = args.output_dir / "checkpoint"
        snapshot_dir.mkdir(parents=True)
        checkpoint_path = snapshot_dir / args.checkpoint.name
        shutil.copyfile(args.checkpoint, checkpoint_path)
    # An open descriptor survives removal of an older checkpoint by training.
    with checkpoint_path.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    training_budget_complete = _validate_checkpoint(
        checkpoint, run_config, args.require_final
    )
    state = {
        key.removeprefix("backbone."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("backbone.")
    }
    if "encoder.linear_1.bias" not in state:
        raise ValueError("Expected native DAE-Net backbone weights")
    backbone = spt.backbone.DAENetBackbone(
        branch_num=len(state["encoder.linear_1.bias"])
    )
    backbone.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    backbone.to(device).eval()
    manifest_sha256 = _sha256(args.manifest)
    dataset = spt.data.DAENetDataset(
        args.data_root,
        args.manifest,
        split=args.split,
        load_queries=args.metric == "occupancy",
    )
    surfaces = (
        DAENetSurfaceLabels(
            args.data_root, args.labels_manifest, dataset.records, split=args.split
        )
        if args.metric == "semantic"
        else None
    )
    num_available = len(dataset)
    num_evaluated = (
        min(args.max_shapes, num_available)
        if args.max_shapes is not None
        else num_available
    )
    artifacts = None
    if surfaces is not None:
        rows, metric_summary, artifacts = _evaluate_semantic(
            backbone, dataset, surfaces, num_evaluated, device, args.chunk_size
        )
        metric_summary["mapping_fit_split"] = args.split
    else:
        rows = []
        for index in range(num_evaluated):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            shape_started = time.perf_counter()
            row = _evaluate_shape(
                backbone, dataset, index, device, args.chunk_size, args.threshold
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row["evaluation_seconds"] = time.perf_counter() - shape_started
            expected_queries = 32768 if row["query_resolution"] == 64 else 8192
            if row["queries"] != expected_queries:
                raise ValueError(
                    f"Incomplete stored-query coverage for {row['shape_id']}"
                )
            rows.append(row)
            print(
                f"{index + 1}/{num_evaluated} {row['shape_id']}: "
                f"IoU={row['iou']:.6f}, seconds={row['evaluation_seconds']:.3f}",
                flush=True,
            )
        total_union = sum(row["union"] for row in rows)
        metric_summary = {
            "metric": "sampled_occupancy_iou",
            "mean_shape_iou": sum(row["iou"] for row in rows) / len(rows),
            "pooled_iou": sum(row["intersection"] for row in rows) / total_union
            if total_union
            else 1.0,
            "empty_union_shapes": sum(row["union"] == 0 for row in rows),
            "protocol": "All stored queries; strict occupancy_max > threshold; macro average across shapes; empty union IoU=1",
        }
    controller = next(
        (
            state
            for state in checkpoint.get("callbacks", {}).values()
            if isinstance(state, dict) and "accepted_epoch" in state
        ),
        {},
    )
    summary = {
        "num_shapes": len(rows),
        "num_available_shapes": num_available,
        "evaluation_mode": "smoke" if args.max_shapes is not None else "full_manifest",
        "selection": "manifest_prefix" if args.max_shapes is not None else "all",
        "max_shapes": args.max_shapes,
        "training_budget_complete": training_budget_complete,
        "final_checkpoint_verified": args.require_final,
        "run_config": str(args.run_config.resolve()) if args.run_config else None,
        "run_config_sha256": run_config_sha256,
        "run_config_verified": run_config is not None,
        "coverage_verified": True,
        "num_queries": sum(row["queries"] for row in rows),
        "split": args.split,
        "threshold": args.threshold,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_snapshot": (
            str(checkpoint_path.resolve()) if args.snapshot_checkpoint else None
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "accepted_epoch": controller.get("accepted_epoch"),
        "weights": "checkpoint state_dict (current saved backbone)",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "data_root": str(args.data_root.resolve()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "torch_version": str(torch.__version__),
        "precision": "FP32, TF32 disabled",
        "chunk_size": args.chunk_size,
        "evaluation_loop_seconds": sum(row["evaluation_seconds"] for row in rows),
        "mean_shape_seconds_after_first": (
            sum(row["evaluation_seconds"] for row in rows[1:]) / (len(rows) - 1)
            if len(rows) > 1
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary.update(metric_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_shape.csv", rows)
    if artifacts is not None:
        _write_csv(args.output_dir / "per_part.csv", artifacts["per_part"])
        artifacts["mapping"].update(
            checkpoint_sha256=checkpoint_sha256,
            mapping_fit_split=args.split,
            labels_manifest_sha256=surfaces.manifest_sha256,
        )
        (args.output_dir / "mapping.json").write_text(
            json.dumps(artifacts["mapping"], indent=2) + "\n", encoding="utf-8"
        )
        artifacts["predictions"].update(
            checkpoint_sha256=np.array(checkpoint_sha256),
            manifest_sha256=np.array(manifest_sha256),
            labels_manifest_sha256=np.array(surfaces.manifest_sha256),
        )
        prediction_path = args.output_dir / "predictions.npz"
        np.savez_compressed(prediction_path, **artifacts["predictions"])
        summary["predictions_sha256"] = _sha256(prediction_path)
    summary["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
