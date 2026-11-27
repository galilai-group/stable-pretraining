"""LeWorldModel benchmark on OGBench Cube."""

import argparse

import h5py
import lightning as pl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import ViTConfig, ViTModel

import stable_pretraining as spt
from stable_pretraining.methods.lewm import LeWM
from stable_pretraining.methods.lewm_components.architecture import (
    ARPredictor,
    Embedder,
    MLP,
    SIGReg,
)
from stable_pretraining.methods.lewm_components.jepa import JEPA
import hdf5plugin  # noqa: F401


class CubeDataset(Dataset):
    """Load OGBench Cube trajectories from an HDF5 dataset."""

    def __init__(self, path, episodes, mean, std, frameskip=5):
        self.path = path
        self.frameskip = frameskip
        self.num_frames = 4
        self.samples = []
        self.file = None
        self.mean = mean
        self.std = std

        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

        self.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        with h5py.File(path, "r") as f:
            for ep in episodes:
                offset = int(f["ep_offset"][ep])
                length = int(f["ep_len"][ep])

                window = self.num_frames * frameskip

                for start in range(length - window + 1):
                    self.samples.append(offset + start)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, "r")

        start = self.samples[index]
        end = start + self.num_frames * self.frameskip

        pixels = self.file["pixels"][start : end : self.frameskip]

        pixels = torch.from_numpy(pixels)
        pixels = pixels.permute(0, 3, 1, 2).float() / 255.0
        pixels = (pixels - self.pixel_mean) / self.pixel_std

        actions = self.file["action"][start:end]
        actions = (actions - self.mean) / self.std
        actions = np.nan_to_num(actions, nan=0.0)
        actions = torch.from_numpy(actions).float()
        actions = actions.reshape(self.num_frames, -1)

        return {
            "pixels": pixels,
            "action": actions,
        }


def split_episodes(seed=42):
    rng = np.random.default_rng(seed)
    episodes = rng.permutation(10000)[:1000]

    return episodes[:900], episodes[900:]


def compute_action_stats(path, episodes):
    actions = []

    with h5py.File(path, "r") as f:
        for ep in episodes:
            offset = int(f["ep_offset"][ep])
            length = int(f["ep_len"][ep])

            actions.append(f["action"][offset : offset + length])

    actions = np.concatenate(actions)
    actions = actions[np.isfinite(actions).all(axis=1)]

    mean = actions.mean(axis=0)
    std = actions.std(axis=0).clip(min=1e-6)

    return mean, std


def make_method():
    embed_dim = 192

    encoder = ViTModel(
        ViTConfig(
            image_size=224,
            patch_size=14,
            hidden_size=192,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
        ),
        add_pooling_layer=False,
    )

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(
            num_frames=3,
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.0,
        ),
        action_encoder=Embedder(
            input_dim=25,
            emb_dim=embed_dim,
        ),
        projector=MLP(
            input_dim=embed_dim,
            hidden_dim=2048,
            output_dim=embed_dim,
            norm_fn=nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=embed_dim,
            hidden_dim=2048,
            output_dim=embed_dim,
            norm_fn=nn.BatchNorm1d,
        ),
    )

    return LeWM(
        model=model,
        history_size=3,
        num_preds=1,
        sigreg_weight=0.09,
        sigreg=SIGReg(knots=17, num_proj=1024),
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": 5e-5,
                "weight_decay": 1e-3,
            }
        },
    )


@torch.no_grad()
def evaluate(method, loader, max_batches=100):
    method.eval()
    device = next(method.parameters()).device

    pred_error = 0.0
    baseline_error = 0.0
    shuffled_error = 0.0
    count = 0
    embeddings = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        output = method.model.encode(batch)

        emb = output["emb"]
        act = output["act_emb"]

        context = emb[:, :3]
        target = emb[:, 1:]

        pred = method.model.predict(context, act[:, :3])

        shuffled_pred = method.model.predict(context, act[:, :3].roll(1, dims=0))

        pred_error += (pred - target).square().sum().item()

        baseline_error += (context - target).square().sum().item()

        shuffled_error += (shuffled_pred - target).square().sum().item()

        count += target.numel()
        embeddings.append(emb.cpu())

    pred_mse = pred_error / count
    baseline_mse = baseline_error / count
    shuffled_mse = shuffled_error / count

    embedding_std = torch.cat(embeddings).flatten(0, 1).std(dim=0).mean().item()

    print("\n--- Evaluation Results ---")
    z = torch.cat(embeddings).flatten(0, 1).float()
    z = z - z.mean(dim=0)

    singular_values = torch.linalg.svdvals(z)
    p = singular_values / singular_values.sum().clamp_min(1e-12)

    effective_rank = torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()

    print(f"Effective rank: {effective_rank:.2f} / {z.shape[1]}")
    print(f"Prediction MSE:       {pred_mse:.6f}")
    print(f"Persistence MSE:      {baseline_mse:.6f}")
    print(f"Shuffled action MSE:  {shuffled_mse:.6f}")
    print(f"Embedding STD:        {embedding_std:.6f}")

    if baseline_mse > 0:
        improvement = 100 * (1 - pred_mse / baseline_mse)
        print(f"Improvement:          {improvement:.2f}%")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    train_eps, val_eps = split_episodes()

    mean, std = compute_action_stats(args.data, train_eps)

    train_dataset = CubeDataset(args.data, train_eps, mean, std)

    val_dataset = CubeDataset(args.data, val_eps, mean, std)

    print(f"Training episodes: {len(train_eps)}")
    print(f"Validation episodes: {len(val_eps)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=4,
        pin_memory=True,
    )

    method = make_method()

    print("\n--- Before Training ---")
    method = method.cuda()
    evaluate(method, val_loader)

    # Reset validation sampling for a fair comparison.
    val_loader.generator.manual_seed(42)

    data = spt.data.DataModule(train=train_loader)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=args.steps,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=method,
        data=data,
        seed=42,
    )

    manager()
    trainer.save_checkpoint("lewm_cube.ckpt")

    print("\n--- After Training ---")
    method = method.cuda()
    evaluate(method, val_loader)


if __name__ == "__main__":
    main()
