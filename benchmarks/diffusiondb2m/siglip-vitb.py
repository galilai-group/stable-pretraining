import argparse
from functools import partial

import lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    SiglipTextModel,
    SiglipVisionModel,
)

import stable_pretraining as spt
from stable_pretraining import forward


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--num_devices", type=int, default=8)
parser.add_argument("--global_batch", type=int, default=4096)
parser.add_argument("--num_epochs", type=int, default=8)
parser.add_argument("--val_percent", type=float, default=0.10)
parser.add_argument("--resume_ckpt_path", type=str, default=None)
args = parser.parse_args()

lr = args.lr
num_devices = args.num_devices
global_batch = args.global_batch
batch_size = global_batch // num_devices
num_epochs = args.num_epochs
val_percent = args.val_percent
resume_ckpt_path = args.resume_ckpt_path

model_name = "google/siglip-base-patch16-224"
tokenizer = AutoTokenizer.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name)
vision_model = SiglipVisionModel.from_pretrained(model_name)
text_model = SiglipTextModel.from_pretrained(model_name)


def tokenize(text: str, tokenizer: AutoTokenizer):
    data = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    return data["input_ids"].squeeze(0), data["attention_mask"].squeeze(0)


image_transform = spt.data.transforms.Compose(
    spt.data.transforms.Resize((224, 224)),
    spt.data.transforms.ToImage(
        mean=image_processor.image_mean,
        std=image_processor.image_std,
    ),
    spt.data.transforms.LambdaTransform(
        fn=partial(tokenize, tokenizer=tokenizer),
        source="prompt",
        targets=("tokenized_prompt", "attention_mask"),
    ),
)


train_base = spt.data.HFDataset(
    "poloclub/diffusiondb",
    "2m_all",
    split="train",
    transform=image_transform,
    remove_columns=[
        "timestamp",
        "user_name",
        "prompt_nsfw",
        "image_nsfw",
        "sampler",
    ],
)

size = len(train_base)
val_n = int(size * val_percent)
val_dataset = spt.data.Subset(train_base, range(0, val_n))
train_dataset = spt.data.Subset(train_base, range(val_n, size))


train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=16,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)


class SigLIPMonitor(pl.Callback):
    """Log retrieval and pairwise sigmoid statistics for SigLIP training."""

    def __init__(self, log_every_n_steps: int = 10):
        super().__init__()
        self.every = log_every_n_steps

    @torch.no_grad()
    def _log(self, trainer: pl.Trainer, pl_module, outputs: dict, stage: str):
        img = F.normalize(outputs["image_embeds"], dim=-1)
        txt = F.normalize(outputs["text_embeds"], dim=-1)

        loss_fn = pl_module.siglip_loss
        logits = loss_fn.logit_scale.exp() * (img @ txt.T) + loss_fn.logit_bias
        batch_size = logits.size(0)
        diag = torch.arange(batch_size, device=logits.device)

        r1_i2t = (logits.argmax(dim=1) == diag).float().mean()
        r1_t2i = (logits.argmax(dim=0) == diag).float().mean()
        pos_prob = torch.sigmoid(logits[diag, diag]).mean()
        cos_pos = F.cosine_similarity(img, txt, dim=-1).mean()

        metrics = {
            f"{stage}/retrieval/R@1_i2t": float(r1_i2t.cpu()),
            f"{stage}/retrieval/R@1_t2i": float(r1_t2i.cpu()),
            f"{stage}/contrast/pos_prob": float(pos_prob.cpu()),
            f"{stage}/align/cos_pos": float(cos_pos.cpu()),
            f"{stage}/config/logit_scale": float(loss_fn.logit_scale.exp().cpu()),
            f"{stage}/config/logit_bias": float(loss_fn.logit_bias.cpu()),
        }
        if batch_size > 1:
            neg = logits.masked_fill(
                torch.eye(batch_size, dtype=torch.bool, device=logits.device),
                float("-inf"),
            )
            top_neg = neg.max(dim=1).values
            margin = logits[diag, diag] - top_neg
            metrics[f"{stage}/contrast/margin"] = float(margin.mean().cpu())

        trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every == 0:
            self._log(trainer, pl_module, outputs, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._log(trainer, pl_module, outputs, "val")


module = spt.Module(
    vision_model=vision_model,
    text_model=text_model,
    forward=forward.siglip_forward,
    siglip_loss=spt.losses.SigLIPLoss(),
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": 1.0e-6,
            "betas": (0.9, 0.98),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "total_steps": (len(train_dataloader) // num_devices) * num_epochs,
            "peak_step": 0.1,
        },
        "interval": "step",
    },
)

wandb_logger = WandbLogger(
    entity="stable-pretraining",
    project="diffusiondb2m-siglip",
    name="siglip-vit-b16-diffusiondb2m-32k",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=num_epochs,
    num_sanity_val_steps=0,
    callbacks=[
        ModelCheckpoint(
            monitor="fit/loss_step",
            mode="min",
            every_n_epochs=1,
            save_top_k=-1,
            dirpath="/your/path/to/checkpoints",
        ),
        LearningRateMonitor(logging_interval="step"),
        SigLIPMonitor(log_every_n_steps=10),
    ],
    precision="bf16-mixed",
    logger=wandb_logger,
    enable_checkpointing=True,
    devices=num_devices,
    accelerator="gpu",
    strategy="ddp",
)

manager = spt.Manager(
    trainer=trainer,
    module=module,
    data=data,
    ckpt_path=resume_ckpt_path,
)

manager()
