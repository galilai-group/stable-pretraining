"""Serial reduced-scale CAPI/MAE validation inside one Slurm allocation.

Run inside a Slurm allocation with repository-local cache environment variables.
Datasets, checkpoints, feature caches and full logs remain under .cache/capi/runs.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import time
import types
from pathlib import Path

import lightning as pl
import numpy as np
import timm
import torch
import torchmetrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

import stable_pretraining as spt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".cache/capi/runs"
REPORT = ROOT / ".cache/capi/results"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_CONFIG = dict(
    embed_dim=384,
    depth=12,
    num_heads=6,
    predictor_depth=6,
    patch_size=16,
    n_registers=16,
    num_clusters=16384,
)


class DictDataset(torch.utils.data.Dataset):
    """Expose torchvision examples through the framework's dict contract."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return dict(image=image, label=label, index=index)


class Diagnostics(pl.Callback):
    """Record bounded representation diagnostics and runtime every epoch."""

    def __init__(self, name):
        self.name = name
        self.times = []
        self.start = 0
        self.samples = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.start = time.monotonic()
        self.samples = []
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if len(self.samples) < 8:
            self.samples.append(outputs["embedding"].detach().float().cpu())

    def on_train_epoch_end(self, trainer, pl_module):
        seconds = time.monotonic() - self.start
        self.times.append(seconds)
        x = torch.cat(self.samples)
        x = x - x.mean(0)
        singular = torch.linalg.svdvals(x)
        probs = singular / singular.sum().clamp_min(1e-12)
        rank = (-(probs * probs.clamp_min(1e-12).log()).sum()).exp()
        row = dict(
            epoch=trainer.current_epoch,
            seconds=seconds,
            effective_rank=float(rank),
            feature_variance=float(x.var(0, unbiased=False).mean()),
            peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
        )
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                row[key] = float(value)
        with (OUT / f"{self.name}-epochs.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def _gradients(self):
    modules = (
        ("backbone", "student_head", "clustering")
        if hasattr(self, "clustering")
        else ("encoder", "decoder")
    )
    for name in modules:
        squares = [
            p.grad.detach().float().square().sum()
            for p in getattr(self, name).parameters()
            if p.grad is not None
        ]
        if squares:
            self.log(
                f"train/gradient_norm_{name}",
                torch.stack(squares).sum().sqrt(),
                on_step=True,
            )


def _mae_forward(self, batch, stage):
    output = spt.MAE.forward(self, batch["image"])
    with torch.no_grad():
        features = self.encoder.forward_features(batch["image"])
    self.log(
        f"{stage}/loss",
        output.loss,
        on_step=True,
        on_epoch=True,
        batch_size=len(batch["image"]),
    )
    return dict(
        loss=output.loss,
        embedding=features[:, self.encoder.num_prefix_tokens :].mean(1).detach(),
        label=batch["label"],
    )


def _make_model(method, steps):
    if method == "capi":
        model = spt.CAPI(
            **MODEL_CONFIG, total_steps=steps, warmup_steps=max(1, int(steps * 0.1))
        )
    else:
        encoder = timm.create_model(
            "vit_small_patch16_224", pretrained=False, num_classes=0
        )
        model = spt.MAE(
            encoder,
            decoder_embed_dim=384,
            decoder_depth=4,
            decoder_num_heads=6,
            mask_ratio=0.75,
            norm_pix_loss=True,
        )
        model.forward = types.MethodType(_mae_forward, model)
        model.optim = dict(
            optimizer=dict(
                type="AdamW", lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95)
            ),
            scheduler=dict(
                type="LinearWarmupCosineAnnealing",
                peak_step=0.1,
                start_factor=0.01,
                end_lr=0,
                total_steps=steps,
            ),
            interval="step",
        )
    model.after_manual_backward = types.MethodType(_gradients, model)
    return model


def _data(batch):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    train = datasets.Imagenette(
        str(OUT / "data"),
        split="train",
        size="320px",
        download=not (OUT / "data/imagenette2-320").exists(),
        transform=train_transform,
    )
    val = datasets.Imagenette(
        str(OUT / "data"),
        split="val",
        size="320px",
        download=False,
        transform=test_transform,
    )

    def loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            DictDataset(ds),
            batch_size=batch,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        )

    return spt.data.DataModule(train=loader(train, True), val=loader(val, False))


def _run(method, seed, epochs, batch, name, timing=False, max_batches=None):
    pl.seed_everything(seed, workers=True)
    dm = _data(batch)
    steps_per_epoch = min(len(dm.train), max_batches) if max_batches else len(dm.train)
    model = _make_model(method, epochs * steps_per_epoch)
    diagnostics = Diagnostics(name)
    callbacks = [diagnostics]
    if not timing:
        callbacks += [
            spt.OnlineProbe(
                model,
                name="linear",
                input="embedding",
                target="label",
                probe=torch.nn.Linear(384, 10),
                loss=torch.nn.CrossEntropyLoss(),
                optimizer=dict(type="AdamW", lr=0.025, weight_decay=0),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(
                        10, average="micro"
                    )
                },
            ),
            spt.OnlineKNN(
                name="knn",
                input="embedding",
                target="label",
                input_dim=384,
                queue_length=10000,
                num_classes=10,
                k=20,
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(
                        10, average="micro"
                    )
                },
            ),
        ]
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=pl.pytorch.loggers.CSVLogger(str(OUT / "logs"), name=name),
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0 if timing else 1.0,
        limit_train_batches=max_batches if max_batches else 1.0,
    )
    spt.set(
        cache_dir=str(OUT / "registry"),
        default_callbacks={"env_dump": False},
        requeue_checkpoint=max_batches is None,
    )
    run_start = time.monotonic()
    spt.Manager(trainer=trainer, module=model, data=dm, seed=seed)()
    run_seconds = time.monotonic() - run_start
    expected_steps = epochs * steps_per_epoch
    if len(diagnostics.times) != epochs or trainer.global_step != expected_steps:
        raise RuntimeError(f"{name} did not complete the locked training schedule")
    if method == "capi" and int(model.capi_updates) != expected_steps:
        raise RuntimeError("Teacher progress does not match completed student updates")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError(f"{name} has nonfinite parameters")
    if not timing:
        path = OUT / f"{name}-final.ckpt"
        trainer.save_checkpoint(path)
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        (REPORT / f"{name}-checkpoint.sha256").write_text(f"{digest}  {path.name}\n")
    del model, trainer, dm
    torch.cuda.empty_cache()
    if timing:
        return [max(seconds, run_seconds / epochs) for seconds in diagnostics.times]
    return diagnostics.times


@torch.inference_mode()
def _extract(model, dataset, method):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, num_workers=6, pin_memory=True
    )
    rows, labels = [], []
    model.requires_grad_(False).eval().cuda()
    for images, target in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if method == "capi":
                features, _ = model.teacher.forward_pretrain(
                    images.cuda(non_blocking=True)
                )
            else:
                features = model.encoder.forward_features(
                    images.cuda(non_blocking=True)
                )[:, model.encoder.num_prefix_tokens :]
        rows.append(features.mean(1).float().cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(rows), np.concatenate(labels)


def _fit_probe(x, y, c):
    classifier = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, tol=1e-4).fit(
        x, y
    )
    if classifier.n_iter_.max() >= 1000:
        classifier = LogisticRegression(
            C=c, solver="lbfgs", max_iter=5000, tol=1e-4
        ).fit(x, y)
        if classifier.n_iter_.max() >= 5000:
            raise RuntimeError(f"Probe at C={c} did not converge")
    return classifier


def _probe(train_x, train_y, test_x, test_y):
    train_indices, val_indices = train_test_split(
        np.arange(len(train_y)), test_size=5000, stratify=train_y, random_state=17
    )
    scaler = StandardScaler().fit(train_x[train_indices])
    x, y = scaler.transform(train_x[train_indices]), train_y[train_indices]
    vx = scaler.transform(train_x[val_indices])
    candidates = []
    for c in (0.01, 0.1, 1, 10, 100):
        clf = _fit_probe(x, y, c)
        candidates.append(
            (clf.score(vx, train_y[val_indices]), c, int(clf.n_iter_.max()))
        )
    accuracy, c, niter = max(candidates, key=lambda value: value[0])
    scaler = StandardScaler().fit(train_x)
    clf = _fit_probe(scaler.transform(train_x), train_y, c)
    predictions = clf.predict(scaler.transform(test_x))
    return dict(
        accuracy=float((predictions == test_y).mean()),
        validation_accuracy=accuracy,
        C=c,
        candidates=candidates,
        refit_iterations=int(clf.n_iter_.max()),
    ), predictions


def _evaluate(epochs):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    train = datasets.CIFAR10(
        str(OUT / "data"), train=True, download=True, transform=transform
    )
    test = datasets.CIFAR10(
        str(OUT / "data"), train=False, download=False, transform=transform
    )
    rows, preds = {}, {}
    for name, method, seed in [
        ("capi-seed0", "capi", 0),
        ("capi-seed1", "capi", 1),
        ("mae-seed0", "mae", 0),
        ("random-seed0", "capi", 0),
        ("random-seed1", "capi", 1),
    ]:
        pl.seed_everything(seed, workers=True)
        model = _make_model(method, max(2, epochs * 73))
        if not name.startswith("random"):
            checkpoint = torch.load(
                OUT / f"{name}-final.ckpt", map_location="cpu", weights_only=False
            )
            state = {
                k: v
                for k, v in checkpoint["state_dict"].items()
                if not k.startswith(("callbacks_modules.", "callbacks_metrics."))
            }
            model.load_state_dict(state)
            del checkpoint, state
        if name.startswith("random"):
            path = OUT / f"{name}-initial.ckpt"
            torch.save({"state_dict": model.state_dict(), "seed": seed}, path)
            digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            (REPORT / f"{name}-checkpoint.sha256").write_text(
                f"{digest}  {path.name}\n"
            )
        tx, ty = _extract(model, train, method)
        sx, sy = _extract(model, test, method)
        np.savez(
            OUT / f"{name}-features.npz", train_x=tx, train_y=ty, test_x=sx, test_y=sy
        )
        row, pred = _probe(tx, ty, sx, sy)
        rows[name] = row
        preds[name] = pred
        np.savez(REPORT / f"{name}-predictions.npz", predictions=pred, labels=sy)
        print(name, row, flush=True)
        del model, tx, sx
        torch.cuda.empty_cache()
    rng = np.random.default_rng(17)
    for seed in (0, 1):
        delta = (preds[f"capi-seed{seed}"] == sy).astype(float) - (
            preds[f"random-seed{seed}"] == sy
        )
        estimates = np.array(
            [
                delta[rng.integers(0, len(delta), len(delta))].mean()
                for _ in range(10000)
            ]
        )
        rows[f"capi-seed{seed}"]["improvement_over_random"] = float(delta.mean())
        rows[f"capi-seed{seed}"]["paired_bootstrap_95ci"] = np.quantile(
            estimates, [0.025, 0.975]
        ).tolist()
    rows["capi_summary"] = {
        "mean": float(np.mean([rows[f"capi-seed{s}"]["accuracy"] for s in (0, 1)])),
        "range": [
            min(rows[f"capi-seed{s}"]["accuracy"] for s in (0, 1)),
            max(rows[f"capi-seed{s}"]["accuracy"] for s in (0, 1)),
        ],
    }
    (REPORT / "transfer.json").write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    """Execute timing, lock the protocol, pretrain serially and evaluate transfer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this campaign inside Slurm")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        _evaluate(json.loads((REPORT / "protocol.json").read_text())["epochs"])
        return
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "lightning": pl.__version__,
        "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
    }
    (REPORT / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    batch = 128
    try:
        _run("capi", 0, 1, batch, "smoke-capi", True, max_batches=2)
        _run("mae", 0, 1, batch, "smoke-mae", True, max_batches=2)
        capi_time = _run("capi", 0, 2, batch, "timing-capi", True)
        mae_time = _run("mae", 0, 2, batch, "timing-mae", True)
    except torch.OutOfMemoryError:
        batch = 64
    if batch == 64:
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        _run("capi", 0, 1, batch, "smoke-capi-b64", True, max_batches=2)
        _run("mae", 0, 1, batch, "smoke-mae-b64", True, max_batches=2)
        capi_time = _run("capi", 0, 2, batch, "timing-capi-b64", True)
        mae_time = _run("mae", 0, 2, batch, "timing-mae-b64", True)
    seconds_per_epoch = 2 * max(capi_time) + max(mae_time)
    epochs = min(200, math.floor(16 * 3600 / (1.2 * seconds_per_epoch)))
    protocol = dict(
        epochs=epochs,
        batch_size=batch,
        capi_epoch_seconds=capi_time,
        mae_epoch_seconds=mae_time,
        margin=1.2,
        training_budget_hours=16,
        total_budget_hours=24,
        seeds=[0, 1],
        model=MODEL_CONFIG,
        reference_commit="98b4fa17ee8eec8810c17022df9a27a44845368b",
        schedule_locked_at=time.time(),
    )
    (REPORT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("LOCKED PROTOCOL", protocol, flush=True)
    if epochs < 50:
        raise RuntimeError("Insufficient budget: fewer than 50 epochs fit")
    for method, seed in [("capi", 0), ("capi", 1), ("mae", 0)]:
        _run(method, seed, epochs, batch, f"{method}-seed{seed}")
    _evaluate(epochs)


if __name__ == "__main__":
    main()
