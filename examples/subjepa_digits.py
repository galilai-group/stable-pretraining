"""Train Sub-JEPA on action-controlled translations of sklearn's digits dataset.

Sub-JEPA on translated handwritten digits
=========================================

Run::

    python examples/subjepa_digits.py --epochs 20 --output /tmp/subjepa-digits

This offline diagnostic uses real handwritten images (8x8 on a 16x16 canvas),
not the paper's Two-Room/PushT benchmark. Train/validation split is by source
image, preventing the same digit image from appearing in both sets. Reports
latent prediction MSE, persistence and shuffled-action baselines, feature
standard deviation, effective rank, and a frozen linear position probe.
The encoder and predictor are deliberately small for CPU validation.
"""

import argparse
import json
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt


class MovingDigits(Dataset):
    """Deterministic offline trajectories of handwritten digits and executed actions.

    Args:
        images: Source digit images [N, 8, 8], normalized to [0, 1].
        seed: Seed for initial positions and independent per-step actions.
        length: Number of observations per trajectory.
    """

    def __init__(self, images: np.ndarray, seed: int, length: int = 4) -> None:
        rng = np.random.default_rng(seed)
        frames, actions, positions = [], [], []
        for digit in images:
            position = rng.integers(0, 9, size=2)
            trajectory, controls, coordinates = [], [], []
            for step in range(length):
                canvas = np.zeros((1, 16, 16), dtype=np.float32)
                y, x = position
                canvas[0, y : y + 8, x : x + 8] = digit
                trajectory.append(canvas)
                coordinates.append(position.copy() / 8)
                if step < length - 1:
                    next_position = np.clip(
                        position + rng.integers(-3, 4, size=2), 0, 8
                    )
                    controls.append((next_position - position) / 3)
                    position = next_position
            frames.append(np.stack(trajectory))
            actions.append(np.stack(controls))
            positions.append(np.stack(coordinates))
        self.images = torch.tensor(np.stack(frames), dtype=torch.float32)
        self.actions = torch.tensor(np.stack(actions), dtype=torch.float32)
        self.positions = torch.tensor(np.stack(positions), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[index],
            "action": self.actions[index],
            "position": self.positions[index],
        }


@torch.no_grad()
def evaluate(
    model: spt.SubJEPA, train: DataLoader, val: DataLoader
) -> dict[str, float]:
    """Evaluate dynamics and a frozen linear position readout on held-out digits.

    Args:
        model: Trained or randomly initialized Sub-JEPA model on CPU.
        train: Training loader used to fit the linear readout only.
        val: Held-out trajectory loader.

    Returns:
        Prediction, collapse, action-use, and position-readout diagnostics.
    """
    model.eval()

    def features(loader):
        zs, positions, actions = [], [], []
        for batch in loader:
            images = batch["image"]
            zs.append(model.backbone(images.flatten(0, 1)).reshape(len(images), 4, -1))
            positions.append(batch["position"])
            actions.append(batch["action"])
        return torch.cat(zs), torch.cat(positions), torch.cat(actions)

    train_z, train_pos, _ = features(train)
    z, pos, actions = features(val)
    prediction = model.predictor(z[:, :-1], actions)
    shuffled = model.predictor(z[:, :-1], actions.roll(1, dims=0))
    flattened = z.flatten(0, 1)
    spectrum = torch.linalg.svdvals(flattened - flattened.mean(0)).square()
    probabilities = spectrum / spectrum.sum().clamp_min(1e-12)
    rank = (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp()
    probe = Ridge(alpha=1.0).fit(
        train_z.flatten(0, 1).numpy(), train_pos.flatten(0, 1).numpy()
    )
    return {
        "prediction_mse": (prediction - z[:, 1:]).square().mean().item(),
        "persistence_mse": (z[:, :-1] - z[:, 1:]).square().mean().item(),
        "shuffled_action_mse": (shuffled - z[:, 1:]).square().mean().item(),
        "embedding_std": flattened.std(0).mean().item(),
        "effective_rank": rank.item(),
        "position_probe_r2": float(
            r2_score(pos.flatten(0, 1).numpy(), probe.predict(flattened.numpy()))
        ),
    }


def main() -> None:
    """Run the reproducible CPU diagnostic and save before/after metrics.

    Returns:
        None. Writes metrics.json and model.pt under the output directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/subjepa-digits"))
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(2)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    spt.set(cache_dir=str(args.output))
    digits = load_digits()
    train_idx, val_idx = train_test_split(
        np.arange(len(digits.images)),
        test_size=0.2,
        stratify=digits.target,
        random_state=args.seed,
    )
    train_data = MovingDigits(digits.images[train_idx] / 16, args.seed)
    val_data = MovingDigits(digits.images[val_idx] / 16, args.seed + 1)
    train = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
    train_eval = DataLoader(train_data, batch_size=128)
    val = DataLoader(val_data, batch_size=128)
    backbone = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, 24),
    )
    model = spt.SubJEPA(
        backbone=backbone,
        action_dim=2,
        embed_dim=24,
        num_subspaces=4,
        num_projections=32,
        regularization_weight=0.1,
        predictor_depth=1,
        num_heads=4,
        max_history=3,
        lr=1e-3,
    )
    before = evaluate(model, train_eval, val)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=args.epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=args.output,
    )
    spt.Manager(
        trainer=trainer, module=model, data=spt.data.DataModule(train=train, val=val)
    )()
    after = evaluate(model, train_eval, val)
    result = {
        "dataset": "sklearn digits with controlled translations",
        "seed": args.seed,
        "epochs": args.epochs,
        "train_trajectories": len(train_data),
        "val_trajectories": len(val_data),
        "before": before,
        "after": after,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(model.state_dict(), args.output / "model.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
