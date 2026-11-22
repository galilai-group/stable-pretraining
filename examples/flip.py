"""Reduced-scale FLIP training and full-candidate retrieval on COCO 2017.
======================================================================

See docs/source/flip.rst for the data layout and experiment commands.
"""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T
from transformers import AutoTokenizer

import stable_pretraining as spt


class TextEncoder(nn.Module):
    """Small bidirectional transformer with first-token pooling and no pooler.

    Like the reference FLIP configuration, padding tokens participate in
    attention. Tokenization adds neither CLS nor SEP. This example trains from
    scratch with six 256-wide blocks, rather than the paper's larger encoder.
    """

    def __init__(self, vocab_size: int, dim: int = 256, depth: int = 6) -> None:
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, dim)
        self.position = nn.Parameter(torch.empty(1, 32, dim))
        nn.init.normal_(self.tokens.weight, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.Sequential(
            *[spt.backbone.TransformerBlock(dim, num_heads=8) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tokens(input_ids) + self.position[:, : input_ids.shape[1]]
        return self.norm(self.blocks(x))[:, 0]


class Captions(Dataset):
    """One row per image; sample one caption for training, retain all for retrieval."""

    def __init__(
        self,
        root: Path,
        image_ids: list[int] | None,
        tokenizer: Callable,
        transform: Callable,
        training: bool = False,
        split: str = "train2017",
        seed: int = 0,
    ) -> None:
        records = json.loads(
            (root / "annotations" / f"captions_{split}.json").read_text()
        )
        files = {row["id"]: row["file_name"] for row in records["images"]}
        if image_ids is None:
            image_ids = sorted(files)
        captions = defaultdict(list)
        for row in records["annotations"]:
            captions[row["image_id"]].append(row["caption"])
        self.image_ids = image_ids
        self.paths = [root / split / files[i] for i in image_ids]
        self.transform, self.training = transform, training
        self.seed = seed
        # Persistent workers must observe epoch changes made in the parent process.
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.offsets = [0]
        texts, owners = [], []
        for image_id in image_ids:
            texts.extend(captions[image_id])
            owners.extend([image_id] * len(captions[image_id]))
            self.offsets.append(len(texts))
        self.caption_ids = torch.tensor(owners)
        self.input_ids = tokenizer(
            texts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        def sample_seed(stream: str) -> int:
            key = f"{self.seed}:{int(self.epoch)}:{self.image_ids[index]}:{stream}"
            return int.from_bytes(
                hashlib.blake2b(key.encode(), digest_size=8).digest(), "little"
            )

        caption = self.offsets[index]
        if self.training:
            caption = random.Random(sample_seed("caption")).randrange(
                caption, self.offsets[index + 1]
            )
        with Image.open(self.paths[index]) as image:
            # Keep augmentation independent of FLIP's torch masking RNG, even
            # with num_workers=0, so masking ablations see the same crops.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(sample_seed("augmentation"))
                image = self.transform(image.convert("RGB"))
        return {"image": image, "input_ids": self.input_ids[caption]}


def split_ids(
    root: Path, train_size: int, dev_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Choose disjoint training/development images from train2017 only."""
    records = json.loads((root / "annotations/captions_train2017.json").read_text())
    ids = sorted(row["id"] for row in records["images"])
    random.Random(seed).shuffle(ids)
    if min(train_size, dev_size) <= 0 or train_size + dev_size > len(ids):
        raise ValueError("Require positive train/dev sizes fitting within train2017")
    # Preserve the development split selected by the original 20k-image pilot.
    start = min(20000, len(ids) - dev_size)
    dev_ids = ids[start : start + dev_size]
    return sorted(set(ids) - set(dev_ids))[:train_size], dev_ids


class EpochSampler(DistributedSampler):
    """Advance worker sampling before the next epoch's iterator prefetches data."""

    def set_epoch(self, epoch: int) -> None:
        """Set the zero-based epoch for shuffling and per-image randomness."""
        super().set_epoch(epoch)
        self.dataset.epoch.fill_(epoch)


class CocoDataModule(spt.data.DataModule):
    """Construct a sampler without padding after Lightning initializes the ranks."""

    def train_dataloader(self) -> DataLoader:
        """Return the loader with a rank-specific, non-padding sampler."""
        rank, world = self.trainer.global_rank, self.trainer.world_size
        seed = self.train_dataset.seed
        self.train_dataset.epoch.share_memory_()
        self.train["sampler"] = EpochSampler(
            self.train_dataset, num_replicas=world, rank=rank, seed=seed, drop_last=True
        )
        self.train["generator"] = torch.Generator().manual_seed(seed + 1000 + rank)
        torch.manual_seed(seed + 10000 + rank)
        return super().train_dataloader()


class Float32CLIPLoss(spt.losses.CLIPLoss):
    """Keep contrastive similarities in FP32 while encoders use mixed precision."""

    def forward(
        self, image: torch.Tensor, text: torch.Tensor, logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """Return FP32 contrastive loss for paired embeddings and a temperature."""
        with torch.autocast(image.device.type, enabled=False):
            return super().forward(
                image.float(), text.float(), logit_scale=logit_scale.float()
            )


def warmup_cosine(
    optimizer: torch.optim.Optimizer, module: spt.Module
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm up for 5% of updates and reach the floor on the final update.

    Args:
        optimizer: AdamW configured with the peak learning rate.
        module: Training module providing the planned number of updates.

    Returns:
        Per-update schedule from 1e-6 to 1e-4 and back to 1e-6.
    """
    steps = module.trainer.estimated_stepping_batches
    warmup = max(1, round(steps * 0.05))

    def factor(step: int) -> float:
        if step < warmup:
            return 0.01 + 0.99 * step / warmup
        fraction = min(1.0, (step - warmup) / max(1, steps - 1 - warmup))
        return 0.01 + 0.99 * (1 + math.cos(math.pi * fraction)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def retrieval_recall(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    query_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> dict[str, float]:
    """Recall against every candidate, counting any matching image ID as positive."""
    hits = torch.zeros(3, dtype=torch.int64)
    for start in range(0, len(queries), 256):
        scores = queries[start : start + 256] @ candidates.T
        # Identical COCO captions tie; use candidate order as in the experiment.
        indices = scores.argsort(dim=1, descending=True, stable=True)[:, :10]
        matches = candidate_ids[indices] == query_ids[start : start + 256, None]
        for j, k in enumerate((1, 5, 10)):
            hits[j] += matches[:, :k].any(dim=1).sum()
    return {f"R@{k}": hits[j].item() / len(queries) for j, k in enumerate((1, 5, 10))}


@torch.inference_mode()
def evaluate(
    model: spt.FLIP, loader: DataLoader, normalize: Callable
) -> dict[str, dict[str, float]]:
    """Evaluate unmasked retrieval in float32 using the training normalization.

    Args:
        model: FLIP model on its inference device.
        loader: Evaluation images as uint8 tensors, with cached caption tokens.
        normalize: Batch transform also used by the training DataModule.

    Returns:
        Full-candidate Recall@1, @5, and @10 in both retrieval directions.
    """
    model.eval()
    dataset = loader.dataset
    images = []
    for batch in loader:
        batch["image"] = batch["image"].to(model.device, non_blocking=True)
        images.append(
            F.normalize(
                model.encode_image(normalize(batch)["image"]).float(), dim=-1
            ).cpu()
        )
    images = torch.cat(images)
    texts = torch.cat(
        [
            F.normalize(
                model.encode_text(tokens.to(model.device, non_blocking=True)).float(),
                dim=-1,
            ).cpu()
            for tokens in dataset.input_ids.split(256)
        ]
    )
    image_ids = torch.tensor(dataset.image_ids)
    return {
        "image_to_text": retrieval_recall(
            images, texts, image_ids, dataset.caption_ids
        ),
        "text_to_image": retrieval_recall(
            texts, images, dataset.caption_ids, image_ids
        ),
    }


def main() -> None:
    """Train FLIP on a COCO subset and evaluate unmasked retrieval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--train-size", type=int, default=117287)
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256, help="Pairs per device.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.train_size < args.batch_size * args.devices:
        parser.error("train-size must cover batch-size * devices (uses drop_last)")
    args.output.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    spt.set(cache_dir=str(args.output.resolve()), progress_bar="simple")
    train_ids, dev_ids = split_ids(args.data, args.train_size, args.dev_size, args.seed)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    normalize = spt.data.transforms.ToImage(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(
                224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(),
            T.PILToTensor(),
        ]
    )
    eval_transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.PILToTensor(),
        ]
    )
    training = Captions(
        args.data, train_ids, tokenizer, train_transform, training=True, seed=args.seed
    )
    evaluation = Captions(args.data, dev_ids, tokenizer, eval_transform)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    eval_loader = DataLoader(evaluation, **loader_args)
    model = spt.FLIP(
        TextEncoder(len(tokenizer)),
        text_dim=256,
        projection_dim=256,
        encoder_name="vit_small_patch16_224",
        mask_ratio=args.mask_ratio,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": 1e-4,
                "weight_decay": 0.2,
                "betas": (0.9, 0.95),
                "exclude_bias_norm": True,
                "exclude_weight_decay_names": (
                    "position",
                    "pos_embed",
                    "cls_token",
                    "mask_token",
                ),
            },
            "scheduler": warmup_cosine,
        },
    )
    model.clip_loss = Float32CLIPLoss()
    trainer = pl.Trainer(
        accelerator="auto",
        devices=args.devices,
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=10,
        precision="bf16-mixed"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "32-true",
        logger=False,
        default_root_dir=str(args.output),
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )
    data = CocoDataModule(
        train=dict(dataset=training, drop_last=True, **loader_args),
        val=eval_loader,
        gpu_transform=normalize,
    )
    spt.Manager(trainer=trainer, module=model, data=data, seed=args.seed)()
    if trainer.is_global_zero:
        model.to(trainer.strategy.root_device)
        validation = Captions(
            args.data, None, tokenizer, eval_transform, split="val2017"
        )
        final_loader = DataLoader(validation, **{**loader_args, "batch_size": 128})
        metrics = evaluate(model, final_loader, normalize)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
