"""
LeVLJEPA frozen ADE20K probe
============================

This reduced linear-head protocol is not an exact paper reproduction. Prepare
ADEChallengeData2016 separately; this example never downloads data or weights.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import lightning as pl
import numpy as np
import stable_pretraining as spt
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class _Masks(Dataset):
    def __init__(self, root: Path, split: str, size: int, limit: int | None):
        self.root, self.split = root, split
        self.files = sorted((root / "images" / split).glob("*.jpg"))
        if not self.files:
            raise ValueError(f"No images found for {split}")
        if limit is not None:
            self.files = self.files[:limit]
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        if any(not self.mask_path(i).is_file() for i in range(len(self))):
            raise ValueError("Each image must have a corresponding annotation PNG")

    def __len__(self) -> int:
        return len(self.files)

    def mask_path(self, i: int) -> Path:
        return self.root / "annotations" / self.split / (self.files[i].stem + ".png")

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        with Image.open(self.files[i]) as image:
            x = self.transform(image.convert("RGB"))
        with Image.open(self.mask_path(i)) as mask:
            raw = np.array(
                mask.resize((self.size, self.size), Image.Resampling.NEAREST)
            )
        return {"image": x, "target": _labels(raw).to(torch.uint8)}


def _labels(raw: np.ndarray) -> torch.Tensor:
    value = torch.as_tensor(raw.astype(np.int64))
    if not ((value >= 0) & ((value <= 150) | (value == 255))).all():
        raise ValueError("Expected ADE20K labels 0..150 or void 255")
    target = value - 1
    target[(value == 0) | (value == 255)] = 255
    return target


class _Cached(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x, self.y = x, y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"features": self.x[i], "target": self.y[i]}


def _state_hash(model: nn.Module) -> str:
    value = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        value.update(key.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


@torch.no_grad()
def _extract(
    model: nn.Module, data: _Masks, device: torch.device, batch: int, workers: int
) -> _Cached:
    x, y = None, torch.empty(len(data), data.size, data.size, dtype=torch.uint8)
    offset = 0
    for b in DataLoader(data, batch_size=batch, num_workers=workers):
        tokens = model.forward_features(b["image"].to(device))
        tokens = tokens[:, model.num_prefix_tokens :]
        grid = math.isqrt(tokens.shape[1])
        if grid * grid != tokens.shape[1]:
            raise ValueError("A square spatial patch grid is required")
        features = tokens.transpose(1, 2).reshape(len(tokens), -1, grid, grid)
        if not torch.isfinite(features).all():
            raise ValueError("Non-finite frozen features")
        if x is None:
            x = torch.empty(len(data), *features.shape[1:], dtype=torch.float16)
        n = len(tokens)
        x[offset : offset + n] = features.cpu().half()
        y[offset : offset + n] = b["target"]
        offset += n
    if x is None or offset != len(data):
        raise ValueError("Incomplete feature extraction")
    return _Cached(x, y)


def _forward(
    self, batch: dict[str, torch.Tensor], stage: str
) -> dict[str, torch.Tensor]:
    logits = self.head(batch["features"].float())
    logits = F.interpolate(
        logits, size=batch["target"].shape[-2:], mode="bilinear", align_corners=False
    )
    target = batch["target"].long()
    loss = (
        F.cross_entropy(logits, target, ignore_index=255)
        if (target != 255).any()
        else logits.sum() * 0
    )
    return {"loss": loss}


@torch.no_grad()
def _metrics(
    head: nn.Module, features: torch.Tensor, data: _Masks, device: torch.device
) -> dict:
    confusion = torch.zeros(150, 150, dtype=torch.int64, device=device)
    for i in range(len(data)):
        with Image.open(data.mask_path(i)) as mask:
            target = _labels(np.array(mask)).to(device)
        logits = head(features[i : i + 1].float().to(device))
        # Class chunks bound memory at native annotation resolution.
        best = torch.full(target.shape, -torch.inf, device=device)
        prediction = torch.zeros_like(target)
        for start in range(0, 150, 25):
            score = F.interpolate(
                logits[:, start : start + 25],
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            )[0]
            value, index = score.max(0)
            prediction = torch.where(value > best, index + start, prediction)
            best = torch.maximum(best, value)
        valid = target != 255
        confusion += torch.bincount(
            target[valid] * 150 + prediction[valid], minlength=150 * 150
        ).reshape(150, 150)
    c = confusion.cpu().double()
    union = c.sum(0) + c.sum(1) - c.diag()
    present = union > 0
    if not present.any():
        raise ValueError("No valid annotation pixels")
    return {
        "miou": float((c.diag()[present] / union[present]).mean()),
        "pixel_accuracy": float(c.diag().sum() / c.sum()),
        "classes_with_union": int(present.sum()),
        "confusion": confusion.cpu().tolist(),
    }


def main() -> None:
    """Train a frozen-feature probe and save its protocol and final metrics."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--format", choices=["native", "author", "random"], default="native")
    p.add_argument(
        "--timm-model",
        help="Optional named timm ViT instead of the reduced architecture",
    )
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-val", type=int)
    p.add_argument("--max-cache-gib", type=float, default=12)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    a = p.parse_args()
    if (
        min(
            a.image_size,
            a.patch_size,
            a.width,
            a.depth,
            a.heads,
            a.epochs,
            a.batch_size,
        )
        < 1
    ):
        p.error("Sizes, epochs and batch size must be positive")
    if a.image_size % a.patch_size and not a.timm_model:
        p.error("Image size must be divisible by patch size")
    if a.format != "random" and a.checkpoint is None:
        p.error("--checkpoint is required unless --format random is used")
    if a.format == "random" and a.checkpoint is not None:
        p.error("Random controls must not load a checkpoint")
    if (
        a.workers < 0
        or a.max_cache_gib <= 0
        or any(n is not None and n < 1 for n in (a.limit_train, a.limit_val))
    ):
        p.error("Workers must be nonnegative; cache and sample limits must be positive")
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "results.json").exists():
        p.error("Choose a new output directory; results.json already exists")
    pl.seed_everything(a.seed, workers=True)
    torch.set_num_threads(2)
    device = torch.device(a.device)
    model = (
        timm.create_model(a.timm_model, pretrained=False, num_classes=0)
        if a.timm_model
        else timm.models.vision_transformer.VisionTransformer(
            img_size=a.image_size,
            patch_size=a.patch_size,
            embed_dim=a.width,
            depth=a.depth,
            num_heads=a.heads,
            num_classes=0,
        )
    )
    if a.checkpoint is not None:
        if a.checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(a.checkpoint))
        else:
            state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
        prefix = "vision_encoder." if a.format == "native" else "encoder."
        model.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)},
            strict=True,
        )
        del state
    model.eval().requires_grad_(False).to(device)
    frozen = _state_hash(model)
    train = _Masks(a.data, "training", a.image_size, a.limit_train)
    val = _Masks(a.data, "validation", a.image_size, a.limit_val)
    grid = a.image_size // model.patch_embed.patch_size[0]
    expected = (len(train) + len(val)) * (
        model.num_features * grid * grid * 2 + a.image_size**2
    )
    if expected > a.max_cache_gib * 1024**3:
        raise ValueError("Estimated RAM cache exceeds --max-cache-gib")
    x = _extract(model, train, device, a.batch_size, a.workers)
    y = _extract(model, val, device, a.batch_size, a.workers)
    pl.seed_everything(a.seed, workers=True)
    head = nn.Conv2d(model.num_features, 150, 1)
    initial_head = _state_hash(head)
    steps = math.ceil(len(x) / a.batch_size) * a.epochs
    module = spt.Module(
        forward=_forward,
        head=head,
        optim={
            "optimizer": {"type": "AdamW", "lr": a.lr, "weight_decay": a.weight_decay},
            "scheduler": {"type": "CosineAnnealingLR", "T_max": steps, "eta_min": 0.0},
            "interval": "step",
        },
    )
    spt.set(
        cache_dir=str(a.output / "cache"),
        progress_bar="none",
        cleanup={"checkpoints": True},
    )
    trainer = pl.Trainer(
        accelerator="gpu" if a.device == "cuda" else "cpu",
        devices=1,
        max_epochs=a.epochs,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(
            train=DataLoader(
                x,
                batch_size=a.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(a.seed),
            )
        ),
        seed=a.seed,
    )()
    head.eval().to(device)
    result = _metrics(head, y.x, val, device)
    if _state_hash(model) != frozen or trainer.global_step != steps:
        raise RuntimeError("Frozen-state or optimizer-step check failed")
    result.update(
        protocol="reduced frozen linear ADE20K probe; not exact paper reproduction",
        configuration={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
        },
        train_images=len(train),
        val_images=len(val),
        steps=steps,
        frozen_encoder_sha256=frozen,
        frozen_encoder_unchanged=True,
        initial_head_sha256=initial_head,
        patch_grid=grid,
        feature_cache="fp16 host RAM; no dataset or encoder-weight copies",
    )
    torch.save(head.state_dict(), a.output / "head.pt")
    (a.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "confusion"}))


if __name__ == "__main__":
    main()
