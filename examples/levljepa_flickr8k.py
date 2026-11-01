"""
LeVLJEPA on Flickr8k
====================

Run ``python examples/levljepa_flickr8k.py --help``. No tracking account,
pretrained weights or external uploads are used. This reduced recipe validates
integration and learning behavior; it is not the paper's DataComp experiment.
Use ``--method clip`` for a control with shared encoders and marginal heads;
CLIP omits LeVLJEPA's cross-modal predictors and uses embedding retrieval.
``--manifest-root`` streams local image/caption pairs without caching images;
``--profile base`` selects ViT-B/16 and 12-layer GPT-2 from random initialization.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import lightning as pl
import timm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from lightning.pytorch.callbacks import Callback
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2
from transformers import AutoTokenizer, GPT2Config, GPT2Model

import stable_pretraining as spt


class CaptionPairs(Dataset):
    """Cache a small image-level subset, with an explicit final EOS token.

    Args:
        rows: Existing HuggingFace dataset rows with ``image`` and ``caption_0``.
        tokenizer: GPT-2 tokenizer.
        image_size: Square input resolution.
        max_length: Padded token length including EOS.
        training: Enable mild image augmentation.
        random_captions: Sample from available captions during training.
        seed: Seed for caption choices, independent of model RNG consumption.
    """

    def __init__(
        self,
        rows,
        tokenizer,
        image_size: int,
        max_length: int,
        training: bool,
        random_captions: bool = False,
        seed: int = 0,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must leave room for content and EOS")
        self.images, self.captions, self.fingerprints = [], [], []
        self.caption_groups = []
        self.seed, self.epoch = seed, 0
        self.random_captions = training and random_captions
        seen = set()
        for row in rows:
            image = row["image"].convert("RGB")
            fingerprint = hashlib.sha256(
                str(image.size).encode() + image.tobytes()
            ).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            self.images.append(image.copy())
            self.captions.append(row["caption_0"])
            captions = (
                [row[f"caption_{i}"] for i in range(5)]
                if self.random_captions
                else [row["caption_0"]]
            )
            if any(not isinstance(c, str) or not c.strip() for c in captions):
                raise ValueError("expected nonempty Flickr8k captions")
            self.caption_groups.append(captions)
            self.fingerprints.append(fingerprint)
        encoded = tokenizer(
            [caption for group in self.caption_groups for caption in group],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 1,
        )["input_ids"]
        self.ids = torch.full(
            (len(encoded), max_length), tokenizer.eos_token_id, dtype=torch.long
        )
        self.mask = torch.zeros_like(self.ids)
        for index, content in enumerate(encoded):
            sequence = content + [tokenizer.eos_token_id]
            self.ids[index, : len(sequence)] = torch.tensor(sequence)
            self.mask[index, : len(sequence)] = 1
        count = 5 if self.random_captions else 1
        self.ids = self.ids.reshape(len(self.images), count, max_length)
        self.mask = self.mask.reshape(len(self.images), count, max_length)
        spatial = (
            [transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))]
            if training
            else [transforms.Resize((image_size, image_size))]
        )
        self.transform = transforms.Compose(
            [
                *spatial,
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Keep caption choices identical across objectives and reproducible on resume.
        caption = random.Random(f"{self.seed}:{self.epoch}:{index}").randrange(
            len(self.caption_groups[index])
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                self.seed + self.epoch * len(self) + index
            )
            pixels = self.transform(self.images[index])
        return {
            "image": pixels,
            "input_ids": self.ids[index, caption],
            "attention_mask": self.mask[index, caption],
        }


class ManifestPairs(Dataset):
    """Stream local pairs with verified image identities and seeded augmentation.

    Args:
        root: Directory containing ``images/`` and JSON split manifests.
        split: Manifest stem, either ``train`` or ``validation``.
        tokenizer: Tokenizer exposing ``eos_token_id`` and batched tokenization.
        samples: Number of leading manifest entries to use, without reshuffling.
        image_size: Square input resolution.
        max_length: Token count including the explicit final EOS token.
        seed: Augmentation seed, independent of model RNG consumption.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        tokenizer,
        samples: int,
        image_size: int,
        max_length: int,
        seed: int,
    ) -> None:
        if split not in {"train", "validation"} or max_length < 2 or samples < 1:
            raise ValueError("invalid manifest split, sample count or token length")
        manifest = root / f"{split}.json"
        self.manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        rows = json.loads(manifest.read_text())
        if not isinstance(rows, list) or len(rows) < samples:
            raise ValueError(
                "manifest must contain at least the requested sample count"
            )
        self.paths, self.fingerprints, captions = [], [], []
        image_root = (root / "images").resolve()
        for row in rows[:samples]:
            path = (image_root / row["file"]).resolve()
            if not path.is_relative_to(image_root):
                raise ValueError("manifest image must stay inside images/")
            caption = row["caption"]
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError("expected nonempty manifest caption")
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"image SHA256 mismatch: {row['file']}")
            self.paths.append(path)
            self.fingerprints.append(digest)
            captions.append(caption)
        if len(set(self.fingerprints)) != samples:
            raise ValueError("duplicate image bytes in manifest split")
        encoded = tokenizer(
            captions,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 1,
        )["input_ids"]
        self.ids = torch.full(
            (samples, max_length), tokenizer.eos_token_id, dtype=torch.long
        )
        self.mask = torch.zeros_like(self.ids)
        for index, content in enumerate(encoded):
            sequence = content + [tokenizer.eos_token_id]
            self.ids[index, : len(sequence)] = torch.tensor(sequence)
            self.mask[index, : len(sequence)] = 1
        self.seed, self.epoch = seed, 0
        spatial = (
            [
                v2.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 4 / 3),
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(),
                v2.RandomApply([v2.ColorJitter(0.2, 0.2, 0.2, 0.02)], p=0.3),
                v2.RandomGrayscale(p=0.05),
            ]
            if split == "train"
            else []
        )
        self.transform = v2.Compose(
            [
                *spatial,
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                *([] if spatial else [v2.Resize((image_size, image_size))]),
                v2.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with Image.open(self.paths[index]) as image, torch.random.fork_rng(devices=[]):
            # Seed CPU transforms without resetting an initialized CUDA generator.
            torch.random.default_generator.manual_seed(
                self.seed + self.epoch * len(self) + index
            )
            pixels = self.transform(image.convert("RGB"))
        return {
            "image": pixels,
            "input_ids": self.ids[index],
            "attention_mask": self.mask[index],
        }


def clip_forward(
    self, batch: dict[str, torch.Tensor], stage: str
) -> dict[str, torch.Tensor]:
    """Train a CLIP control with the same encoders and marginal heads.

    Args:
        self: Module exposing the shared encoder and projection contract.
        batch: Paired image and token tensors.
        stage: Framework training, validation, test, or prediction stage.

    Returns:
        Encoded batch with symmetric contrastive loss outside prediction.
    """
    output = spt.forward.levljepa(self, batch, "predict")
    if stage != "predict":
        output["loss"] = self.clip_loss(
            output["image_embedding"], output["text_embedding"]
        )
    return output


class LossHistory(Callback):
    """Record batch-averaged epoch components for a local experiment report."""

    def __init__(self) -> None:
        self.epochs = []
        self._values = []

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if not pl_module.training:
            raise RuntimeError(
                "Training started in eval mode; restore model.train() after evaluation"
            )
        trainer.train_dataloader.dataset.epoch = trainer.current_epoch
        self._values = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._values.append(
            {
                key: float(outputs[key].detach())
                for key in ("loss", "prediction_loss", "sigreg_vision", "sigreg_text")
                if key in outputs
            }
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._values:
            self.epochs.append(
                {
                    "epoch": trainer.current_epoch,
                    **{
                        key: sum(row[key] for row in self._values) / len(self._values)
                        for key in self._values[0]
                    },
                }
            )
            print(json.dumps(self.epochs[-1]), flush=True)


@torch.no_grad()
def evaluate(
    model: spt.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str = "32-true",
) -> dict:
    """Measure retrieval over the full holdout in the selected objective space.

    Args:
        model: LeVLJEPA instance or CLIP control Module.
        loader: Deterministic image/caption pairs.
        device: Evaluation device.
        precision: Autocast precision, matching the training run.

    Returns:
        R@1/R@5/R@10, random R@1, marginal standard deviations and sample count.
        Positives use paired image identity, never equality of caption strings.
        CaptionPairs deduplicates images, leaving one caption per gallery image.
    """
    was_training = model.training
    model.eval()
    collected = {
        key: []
        for key in (
            "image_embedding",
            "text_embedding",
            "image_prediction",
            "text_prediction",
        )
    }
    for batch in loader:
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16 if precision == "bf16-mixed" else torch.float16,
            enabled=precision != "32-true",
        ):
            output = model(
                {key: value.to(device) for key, value in batch.items()}, stage="predict"
            )
        for key in collected:
            collected[key].append(output[key].float().cpu())
    vectors = {key: torch.cat(value) for key, value in collected.items()}
    samples = vectors["image_embedding"].shape[0]
    # Distinct images can have identical captions; text equality is not identity.
    positives = torch.eye(samples, dtype=torch.bool)
    result = {"samples": samples, "random_r1": 1 / samples}
    pairs = (
        (
            ("image_to_text", "image_embedding", "text_embedding"),
            ("text_to_image", "text_embedding", "image_embedding"),
        )
        if getattr(model, "retrieval_mode", "prediction") == "embedding"
        else (
            ("image_to_text", "image_prediction", "text_embedding"),
            ("text_to_image", "text_prediction", "image_embedding"),
        )
    )
    for name, prediction, target in pairs:
        scores = (
            F.normalize(vectors[prediction], dim=-1)
            @ F.normalize(vectors[target], dim=-1).T
        )
        for k in (1, 5, 10):
            neighbors = scores.topk(min(k, scores.shape[1]), dim=1).indices
            result[f"{name}_r{k}"] = float(
                positives.gather(1, neighbors).any(1).float().mean()
            )
    for modality in ("image", "text"):
        result[f"{modality}_feature_std"] = float(
            vectors[f"{modality}_embedding"].std(dim=0, correction=0).mean()
        )
    model.train(was_training)
    return result


def main() -> None:
    """Train through Manager and save local metrics plus a resumable checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/levljepa-flickr8k")
    )
    parser.add_argument("--method", choices=("levljepa", "clip"), default="levljepa")
    parser.add_argument(
        "--profile",
        choices=("reduced", "base"),
        default="reduced",
        help="base fixes ViT-B/16, GPT-2 12L/12H/768, 224px, 77 tokens and 4096 slices",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        help="Local images/ plus train.json and validation.json; no dataset download",
    )
    parser.add_argument("--dataset", default="jxie/flickr8k")
    parser.add_argument(
        "--revision", default="56f58c967835f7c508d684f36bd7897cca9d7634"
    )
    parser.add_argument("--tokenizer", default="gpt2", help="HF ID or local directory")
    parser.add_argument(
        "--train-parquet",
        nargs="+",
        help="Optional local train shards, Flickr8k columns",
    )
    parser.add_argument("--val-parquet", help="Optional local validation shard")
    parser.add_argument("--caption-mode", choices=("first", "random"), default="first")
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, help="Fixed optimizer-update budget")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--text-length", type=int, default=32)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--projections", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-vision", type=float, default=0.02)
    parser.add_argument("--lambda-text", type=float, default=0.02)
    parser.add_argument("--predictor-dropout", type=float, default=0.0)
    parser.add_argument("--alignment", choices=("mse", "cosine"), default="mse")
    parser.add_argument("--warmup-fraction", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--warmup-start-lr", type=float)
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument(
        "--image-augmentation", choices=("crop", "resize"), default="crop"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-shared-sha256",
        help="Optional required encoder/pre-projection initializer hash",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=("32-true", "bf16-mixed", "16-mixed"),
        default="32-true",
    )
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--resume", type=Path, help="Full local .ckpt, resumes optimizer and step"
    )
    args = parser.parse_args()
    if args.profile == "base":
        args.width, args.layers = 768, 12
        args.image_size, args.text_length, args.projections = 224, 77, 4096
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if not 0 <= args.warmup_fraction < 1:
        parser.error("warmup fraction must lie in [0, 1)")
    if args.warmup_steps is not None and args.warmup_steps < 1:
        parser.error("warmup steps must be positive")
    if args.warmup_steps is not None and args.warmup_fraction:
        parser.error("choose warmup steps or warmup fraction, not both")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max steps must be positive")
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    if args.warmup_start_lr is not None and (
        not math.isfinite(args.warmup_start_lr)
        or not 0 <= args.warmup_start_lr <= args.lr
    ):
        parser.error("warmup start LR must be finite and between zero and peak LR")
    if args.manifest_root and (
        args.train_parquet
        or args.val_parquet
        or args.caption_mode != "first"
        or args.image_augmentation != "crop"
    ):
        parser.error(
            "manifest mode uses its own fixed captions and augmentation recipe"
        )
    if not math.isfinite(args.gradient_clip_val) or args.gradient_clip_val < 0:
        parser.error("gradient clip value must be finite and nonnegative")
    if args.width % 4 or args.image_size % 8 or args.batch_size < 2:
        parser.error("width must be divisible by 4, image size by 8, batch size >=2")
    if (
        min(
            args.train_samples, args.val_samples, args.epochs, args.layers, args.threads
        )
        < 1
    ):
        parser.error("sample counts, epochs, layers and threads must be positive")
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(
            args.resume.resolve(), map_location="cpu", weights_only=False
        )
        saved_schedulers = resume_checkpoint.get("lr_schedulers", [])
        if saved_schedulers:
            saved_warmup = "warmup_steps" in saved_schedulers[0]
            if saved_warmup != bool(args.warmup_fraction or args.warmup_steps):
                parser.error(
                    "resume must use the checkpoint's scheduler family: "
                    "set --warmup-fraction or --warmup-steps for warmup checkpoints, "
                    "or neither for the original cosine-only checkpoints"
                )
    if not args.resume and any(
        (args.output / name).exists() for name in ("results.json", "final.ckpt")
    ):
        parser.error("output contains a completed run; choose a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    pl.seed_everything(args.seed, workers=True)
    spt.set(
        cache_dir=str((args.output / "runs").resolve()),
        progress_bar="none",
        cleanup={"checkpoints": True},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if bool(args.train_parquet) != bool(args.val_parquet):
        parser.error("provide both --train-parquet and --val-parquet")
    if args.manifest_root:
        train = ManifestPairs(
            args.manifest_root,
            "train",
            tokenizer,
            args.train_samples,
            args.image_size,
            args.text_length,
            args.seed,
        )
        val = ManifestPairs(
            args.manifest_root,
            "validation",
            tokenizer,
            args.val_samples,
            args.image_size,
            args.text_length,
            args.seed,
        )
    elif args.train_parquet:
        train_rows = load_dataset(
            "parquet", data_files=args.train_parquet, split="train"
        )
        val_rows = load_dataset("parquet", data_files=args.val_parquet, split="train")
    else:
        train_rows = load_dataset(args.dataset, revision=args.revision, split="train")
        val_rows = load_dataset(
            args.dataset, revision=args.revision, split="validation"
        )
    if not args.manifest_root:
        train_rows = train_rows.shuffle(seed=args.seed).select(
            range(min(args.train_samples, len(train_rows)))
        )
        val_rows = val_rows.shuffle(seed=args.seed).select(
            range(min(args.val_samples, len(val_rows)))
        )
        train = CaptionPairs(
            train_rows,
            tokenizer,
            args.image_size,
            args.text_length,
            True,
            random_captions=args.caption_mode == "random",
            seed=args.seed,
        )
        val = CaptionPairs(
            val_rows, tokenizer, args.image_size, args.text_length, False
        )
    if not args.manifest_root and args.image_augmentation == "resize":
        train.transform = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    if set(train.fingerprints) & set(val.fingerprints):
        raise ValueError("train/validation image overlap detected")
    if len(train) < args.batch_size or len(val) < 2:
        raise ValueError(
            "need at least one full training batch and two validation examples"
        )
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    # Reduced randomly initialized ViT/GPT-2 retain the reference encoder families.
    vision = (
        timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )
        if args.profile == "base"
        else timm.models.vision_transformer.VisionTransformer(
            img_size=args.image_size,
            patch_size=8,
            embed_dim=args.width,
            depth=args.layers,
            num_heads=4,
            num_classes=0,
        )
    )
    text = GPT2Model(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=1024 if args.profile == "base" else args.text_length,
            n_embd=args.width,
            n_layer=args.layers,
            n_head=12 if args.profile == "base" else 4,
            n_inner=args.width * 4,
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
        )
    )
    optim = {"optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": 0.01}}
    total_steps = args.max_steps or len(train_loader) * args.epochs
    if args.warmup_fraction or args.warmup_steps:
        warmup_steps = args.warmup_steps or max(
            1, int(total_steps * args.warmup_fraction)
        )
        if warmup_steps >= total_steps:
            parser.error("warmup must leave at least one decay step")
        optim.update(
            scheduler={
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": total_steps,
                "warmup_start_lr": (
                    args.warmup_start_lr
                    if args.warmup_start_lr is not None
                    else args.lr * 0.01
                ),
                "eta_min": args.lr * 0.1,
            },
            interval="step",
        )
    model = spt.LeVLJEPA(
        vision_encoder=vision,
        text_encoder=text,
        vision_dim=args.width,
        text_dim=args.width,
        embedding_dim=args.width,
        pre_projection_dims=(2048,) if args.profile == "base" else (args.width * 2,),
        predictor_dims=(2048,) * 4 if args.profile == "base" else (args.width * 2,) * 2,
        predictor_dropout=args.predictor_dropout,
        lambda_vision=args.lambda_vision,
        lambda_text=args.lambda_text,
        alignment=args.alignment,
        num_projections=args.projections,
        seed=args.seed,
        optim=optim,
    )
    if args.method == "clip":
        model = spt.Module(
            forward=clip_forward,
            vision_encoder=model.vision_encoder,
            text_encoder=model.text_encoder,
            vision_pre_proj=model.vision_pre_proj,
            text_pre_proj=model.text_pre_proj,
            projector_vision=torch.nn.Identity(),
            projector_text=torch.nn.Identity(),
            clip_loss=spt.losses.CLIPLoss(temperature=0.07),
            retrieval_mode="embedding",
            optim=optim,
        )
    shared = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(
            ("vision_encoder.", "text_encoder.", "vision_pre_proj.", "text_pre_proj.")
        ):
            shared.update(name.encode())
            shared.update(value.cpu().contiguous().numpy().tobytes())
    if (
        args.expected_shared_sha256
        and shared.hexdigest() != args.expected_shared_sha256
    ):
        raise ValueError("shared initializer SHA256 mismatch; training was not started")
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "shared_initializer_sha256": shared.hexdigest(),
                "train_samples": len(train),
                "validation_samples": len(val),
                "planned_steps": total_steps,
                "planned_exposures": total_steps * args.batch_size,
                "manifest_sha256": {
                    "train": train.manifest_sha256,
                    "validation": val.manifest_sha256,
                }
                if args.manifest_root
                else None,
                "selection": "fixed final checkpoint; validation is development data",
            },
            indent=2,
        )
        + "\n"
    )
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["state_dict"])
    device = torch.device(
        "cuda" if args.accelerator != "cpu" and torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    before = evaluate(model, val_loader, device, args.precision)
    history = LossHistory()
    trainer = pl.Trainer(
        max_epochs=-1 if args.max_steps else args.epochs,
        max_steps=args.max_steps or -1,
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[history],
        log_every_n_steps=1,
        gradient_clip_val=args.gradient_clip_val,
    )
    spt.Manager(
        trainer=trainer,
        module=model,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        seed=args.seed,
        ckpt_path=str(args.resume.resolve()) if args.resume else None,
        weights_only=False,
    )()
    # Lightning teardown moves the model to CPU before this external evaluation.
    model.to(device)
    after = evaluate(model, val_loader, device, args.precision)
    trainer.save_checkpoint(str((args.output / "final.ckpt").resolve()))
    report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data": {
            "train_samples": len(train),
            "val_samples": len(val),
            "train_image_hashes": train.fingerprints,
            "val_image_hashes": val.fingerprints,
            "caption_column": (
                "caption"
                if args.manifest_root
                else "caption_0..caption_4; seeded per-image per-epoch training choice"
                if args.caption_mode == "random"
                else "caption_0"
            ),
            "validation_caption_column": "caption"
            if args.manifest_root
            else "caption_0",
            "split": (
                "leading manifest entries; verified image-byte SHA256; disjoint splits"
                if args.manifest_root
                else "existing train/validation; seeded subset; exact image deduplication"
            ),
            "manifest_sha256": {
                "train": train.manifest_sha256,
                "validation": val.manifest_sha256,
            }
            if args.manifest_root
            else None,
        },
        "shared_initializer_sha256": shared.hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "versions": {
            "torch": torch.__version__,
            "lightning": pl.__version__,
            "timm": timm.__version__,
        },
        "before": before,
        "after": after,
        "epochs": history.epochs,
        "global_step": trainer.global_step,
        "scope": "Reduced-data experiment; profile base uses full-capacity encoders, not paper-scale reproduction.",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"before": before, "after": after, "global_step": trainer.global_step},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
