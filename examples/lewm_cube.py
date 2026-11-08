"""Train LeWM on OGBench Cube using Stable Pretraining.

Requires the official LeWM repository's dataset utilities and configuration.

Environment variables:
    LEWM_OFFICIAL_DIR: Path to the official LeWM repository.
    LEWM_DATASET_PATH: Optional path to a local dataset.
    LOCAL_DATASET_DIR: Optional dataset cache directory.

Example:
    export LEWM_OFFICIAL_DIR=/path/to/lewm-official
    export LEWM_DATASET_PATH=/path/to/cube_single_expert.h5

    python examples/lewm_cube.py data=ogb
"""

import os
import sys
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict

from stable_pretraining.methods.lewm_components.architecture import SIGReg


# Locate the official LeWM repository.
OFFICIAL_REPO = (
    Path(os.environ.get("LEWM_OFFICIAL_DIR", "../lewm-official")).expanduser().resolve()
)

if not OFFICIAL_REPO.is_dir():
    raise FileNotFoundError(
        f"Official LeWM repository not found: {OFFICIAL_REPO}. "
        "Set LEWM_OFFICIAL_DIR to its location."
    )

sys.path.insert(0, str(OFFICIAL_REPO))

from utils import get_column_normalizer, get_img_preprocessor  # noqa: E402


@hydra.main(
    version_base=None,
    config_path=str(OFFICIAL_REPO / "config" / "train"),
    config_name="lewm",
)
def main(cfg):
    # Load dataset configuration from the official LeWM config.
    dataset_cfg = OmegaConf.to_container(
        cfg.data.dataset,
        resolve=True,
    )

    dataset_name = dataset_cfg.pop("name")

    # Allow a local dataset path without modifying the official config.
    dataset_name = os.environ.get(
        "LEWM_DATASET_PATH",
        dataset_name,
    )

    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
        **dataset_cfg,
    )

    # Use the official LeWM preprocessing utilities.
    transforms = [
        get_img_preprocessor(
            source="pixels",
            target="pixels",
            img_size=cfg.img_size,
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            transforms.append(get_column_normalizer(dataset, col, col))

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

    dataset.transform = spt.data.transforms.Compose(*transforms)

    # Split the dataset reproducibly.
    generator = torch.Generator().manual_seed(cfg.seed)

    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=generator,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )

    # Resolve the official Hydra model paths against
    # our native Stable Pretraining implementation.
    with open_dict(cfg):
        cfg.model._target_ = "stable_pretraining.methods.lewm_components.jepa.JEPA"

        cfg.model.predictor._target_ = (
            "stable_pretraining.methods.lewm_components.architecture.ARPredictor"
        )

        cfg.model.action_encoder._target_ = (
            "stable_pretraining.methods.lewm_components.architecture.Embedder"
        )

        cfg.model.projector._target_ = (
            "stable_pretraining.methods.lewm_components.architecture.MLP"
        )

        cfg.model.pred_proj._target_ = (
            "stable_pretraining.methods.lewm_components.architecture.MLP"
        )

    # Construct the LeWM architecture.
    model = hydra.utils.instantiate(cfg.model)

    # Configure optimizer and learning rate scheduler.
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }

    # Create the native Stable Pretraining LeWM method.
    method = spt.LeWM(
        model=model,
        history_size=cfg.history_size,
        num_preds=cfg.num_preds,
        sigreg_weight=cfg.loss.sigreg.weight,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        optim=optimizers,
    )

    # Wrap the dataloaders in Stable Pretraining's DataModule.
    data = spt.data.DataModule(
        train=train_loader,
        val=val_loader,
    )

    # Configure Lightning Trainer.
    trainer = pl.Trainer(
        **cfg.trainer,
        num_sanity_val_steps=1,
        logger=False,
        enable_checkpointing=True,
    )

    # Launch training through Stable Pretraining.
    manager = spt.Manager(
        trainer=trainer,
        module=method,
        data=data,
    )

    manager()


if __name__ == "__main__":
    main()
