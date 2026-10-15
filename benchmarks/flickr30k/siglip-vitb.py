"""SigLIP ViT-B/16 on Flickr30k: zero-shot, fine-tune, or from-scratch.

Reports image->text and text->image retrieval Recall@{1,5,10} on the
standard Karpathy 1k test split (1,000 images, 5,000 captions).

Modes:

- ``zeroshot``: evaluate ``google/siglip-base-patch16-224`` as released,
  loaded through :class:`stable_pretraining.methods.SigLIP`.
- ``finetune``: fine-tune that checkpoint on the 29k training images (one
  random caption of five per image per step), starting from its learned
  logit scale/bias.
- ``scratch``: same towers, random init, paper logit init (log 10, -10),
  AdamW with beta2 0.95 and weight decay 0.2. Flickr30k is far too small for
  real pretraining; this mode is a learning sanity check. The paper's lr
  1e-3 assumes ~32k batches: at batch 512 it collapses to constant
  embeddings. Of the rates tried (1e-3, 3e-4, 1e-4), 1e-4 gave the best
  retrieval. Without augmentation the model overfits after ~12 epochs, so
  training uses random resized crops (no flips: captions mention left and
  right) and a longer 100-epoch schedule.

Training goes through :func:`stable_pretraining.forward.siglip`. All
training modes clip gradients to norm 1.0, warm up for the first 10% of
steps, and exclude biases, norm parameters, and the logit scale/bias from
weight decay. Retrieval is reported for the last epoch and for the epoch with
the lowest validation loss (all 1,014 Karpathy val images, first caption per
image).

Example::

    python benchmarks/flickr30k/siglip-vitb.py --mode zeroshot --num_devices 1
    srun python benchmarks/flickr30k/siglip-vitb.py --mode finetune --num_devices 2
"""

import argparse
import json
import os
import subprocess
from functools import partial
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from transformers import AutoImageProcessor, AutoTokenizer

import stable_pretraining as spt
from stable_pretraining.methods import SigLIP

MODEL_NAME = "google/siglip-base-patch16-224"
DATASET = "AnyModal/flickr30k"

DEFAULTS = {  # mode: (lr, weight_decay, epochs, min_crop_scale)
    "zeroshot": (0.0, 0.0, 0, 1.0),
    "finetune": (1e-5, 1e-4, 5, 0.9),
    "scratch": (1e-4, 0.2, 100, 0.5),
}


class Tokenize:
    """Tokenize captions the way the SigLIP text tower was trained.

    The text tower pools the last token and was trained with max_length
    padding and no attention mask, so no mask is returned.
    """

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, captions):
        return self.tokenizer(
            captions,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )["input_ids"]


def random_caption_tokens(sample, tokenize):
    captions = sample["original_alt_text"]
    return tokenize(captions[torch.randint(len(captions), ()).item()]).squeeze(0)


def first_caption_tokens(sample, tokenize):
    return tokenize(sample["original_alt_text"][0]).squeeze(0)


def image_transform(image_processor, *extra, min_crop_scale=1.0):
    if min_crop_scale < 1.0:
        resize = spt.data.transforms.RandomResizedCrop(
            (224, 224), scale=(min_crop_scale, 1.0)
        )
    else:
        resize = spt.data.transforms.Resize((224, 224))
    return spt.data.transforms.Compose(
        resize,
        spt.data.transforms.RGB(),
        spt.data.transforms.ToImage(
            mean=image_processor.image_mean, std=image_processor.image_std
        ),
        *extra,
    )


def collate(samples):
    # Keep only tensors; captions are variable-length string lists.
    return {
        "image": torch.stack([s["image"] for s in samples]),
        "tokenized_prompt": torch.stack([s["tokenized_prompt"] for s in samples]),
    }


def load_split(split, transform, max_samples=None):
    ds = spt.data.HFDataset(DATASET, split=split, transform=transform)
    if max_samples is not None:
        ds = spt.data.Subset(ds, range(min(max_samples, len(ds))))
    return ds


@torch.no_grad()
def retrieval_metrics(model, dataset, tokenize, device, batch_size=250):
    """Flickr30k retrieval: R@K for image->text (any of 5 captions) and text->image."""
    model.eval().to(device)
    img_embs, txt_embs, txt_owner = [], [], []
    with torch.autocast(device.type, dtype=torch.bfloat16):
        for start in range(0, len(dataset), batch_size):
            samples = [
                dataset[i] for i in range(start, min(start + batch_size, len(dataset)))
            ]
            images = torch.stack([s["image"] for s in samples]).to(device)
            img_embs.append(model.encode_image(images).float())
            captions = [c for s in samples for c in s["original_alt_text"]]
            txt_owner += [
                start + j for j, s in enumerate(samples) for _ in s["original_alt_text"]
            ]
            txt_embs.append(model.encode_text(tokenize(captions).to(device)).float())

    img = torch.cat(img_embs)
    txt = torch.cat(txt_embs)
    owner = torch.tensor(txt_owner, device=device)
    sims = img @ txt.T  # [N_img, N_txt]

    # image->text: rank of the best-ranked correct caption for each image
    order = sims.argsort(dim=1, descending=True)
    correct = owner[order] == torch.arange(len(img), device=device)[:, None]
    i2t_rank = correct.float().argmax(dim=1)
    # text->image: rank of the caption's own image
    order = sims.T.argsort(dim=1, descending=True)
    t2i_rank = (order == owner[:, None]).float().argmax(dim=1)

    metrics = {"num_images": len(img), "num_captions": len(txt)}
    for k in (1, 5, 10):
        metrics[f"i2t_R@{k}"] = round(100 * (i2t_rank < k).float().mean().item(), 2)
        metrics[f"t2i_R@{k}"] = round(100 * (t2i_rank < k).float().mean().item(), 2)
    metrics["rsum"] = round(sum(v for k, v in metrics.items() if "R@" in k), 2)
    return metrics


class BestValidationWeights(pl.Callback):
    """Keep a CPU copy of the weights from the epoch with the lowest val loss."""

    def __init__(self, model, monitor="validate/loss"):
        self.model = model
        self.monitor = monitor
        self.best_loss = float("inf")
        self.best_epoch = None
        self.weights = None

    def on_validation_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get(self.monitor)
        # `not <` also rejects NaN.
        if trainer.sanity_checking or loss is None or not loss < self.best_loss:
            return
        self.best_loss = loss.item()
        self.best_epoch = trainer.current_epoch
        if trainer.is_global_zero:
            self.weights = {
                k: v.detach().to("cpu", copy=True)
                for k, v in self.model.state_dict().items()
            }


def logit_params(model):
    loss = model.siglip_loss
    return {
        "final_logit_scale": round(loss.logit_scale.exp().item(), 3),
        "final_logit_bias": round(loss.logit_bias.item(), 3),
    }


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["zeroshot", "finetune", "scratch"], required=True
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--global_batch", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--min_crop_scale", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/siglip-flickr30k")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()
    for name, default in zip(
        ("lr", "weight_decay", "num_epochs", "min_crop_scale"), DEFAULTS[args.mode]
    ):
        if getattr(args, name) is None:
            setattr(args, name, default)
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) / args.mode
    pl.seed_everything(args.seed, workers=True)
    # The best-val weights are kept in memory; skip Manager's per-epoch
    # requeue checkpoint (~2.4 GB for ViT-B with AdamW state).
    spt.set(requeue_checkpoint=False)

    tokenize = Tokenize(MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    test_dataset = load_split(
        "test", image_transform(image_processor), args.max_test_samples
    )
    device = torch.device(
        f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
        if torch.cuda.is_available()
        else "cpu"
    )
    result = {
        "mode": args.mode,
        "model": MODEL_NAME,
        "dataset": DATASET,
        "git_sha": git_sha(),
        **{
            k: getattr(args, k)
            for k in (
                "lr",
                "weight_decay",
                "num_epochs",
                "global_batch",
                "num_devices",
                "grad_clip",
                "min_crop_scale",
                "seed",
            )
        },
    }

    model = SigLIP(model_name=MODEL_NAME, pretrained=args.mode != "scratch")

    if args.mode != "zeroshot":
        train_dataset = load_split(
            "train",
            image_transform(
                image_processor,
                spt.data.transforms.Lambda(
                    partial(random_caption_tokens, tokenize=tokenize),
                    target="tokenized_prompt",
                ),
                min_crop_scale=args.min_crop_scale,
            ),
            args.max_train_samples,
        )
        val_dataset = load_split(
            "validation",
            image_transform(
                image_processor,
                spt.data.transforms.Lambda(
                    partial(first_caption_tokens, tokenize=tokenize),
                    target="tokenized_prompt",
                ),
            ),
        )
        loader = partial(
            torch.utils.data.DataLoader,
            batch_size=args.global_batch // args.num_devices,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        # SigLIPLoss gathers text features across ranks, so every rank needs
        # the same batch size. drop_last keeps training batches equal; for
        # validation the DistributedSampler pads ranks to equal length, so
        # the last batch matches across ranks and every val image counts.
        train_dataloader = loader(train_dataset, shuffle=True, drop_last=True)
        data = spt.data.DataModule(train=train_dataloader, val=loader(val_dataset))

        module = spt.Module(
            forward=spt.forward.siglip,
            vision_model=model.vision_model,
            text_model=model.text_model,
            siglip_loss=model.siglip_loss,
            hparams=result,
            optim={
                "optimizer": {
                    "type": "AdamW",
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "betas": (0.9, 0.95),
                    "exclude_bias_norm": True,
                },
                "scheduler": {
                    "type": "LinearWarmupCosineAnnealing",
                    "total_steps": (len(train_dataloader) // args.num_devices)
                    * args.num_epochs,
                    "peak_step": 0.1,
                },
                "interval": "step",
            },
        )
        best_weights = BestValidationWeights(model)
        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            num_sanity_val_steps=0,
            callbacks=[LearningRateMonitor(logging_interval="step"), best_weights],
            precision="bf16-mixed",
            logger=CSVLogger(str(output_dir), name="logs"),
            enable_checkpointing=False,
            devices=args.num_devices,
            accelerator="gpu",
            strategy="ddp" if args.num_devices > 1 else "auto",
            log_every_n_steps=10,
            gradient_clip_val=args.grad_clip,
        )
        spt.Manager(trainer=trainer, module=module, data=data, seed=args.seed)()
        if not trainer.is_global_zero:
            return

    result.update(logit_params(model))
    result["test"] = retrieval_metrics(model, test_dataset, tokenize, device)
    if args.mode != "zeroshot" and best_weights.weights is not None:
        model.load_state_dict(best_weights.weights)
        result["best_val_epoch"] = best_weights.best_epoch
        result["best_val_loss"] = round(best_weights.best_loss, 4)
        result["test_best_val"] = retrieval_metrics(
            model, test_dataset, tokenize, device
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
