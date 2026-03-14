"""V-JEPA ViT-Base on UCF-101.

Self-supervised video representation learning via tube-masked spatio-temporal
prediction.  A ViT-Base is trained with the V-JEPA objective: a lightweight
predictor recovers teacher representations of masked spatio-temporal tubes from
the surrounding context encoded by the student.

The UCF-101 dataset is downloaded automatically from HuggingFace
(MichiganNLP/ucf-101) on the first run and extracted to the configured data
directory.

References:
    Bardes et al. "V-JEPA: Latent Video Prediction for Visual Representation
    Learning." ICLR 2024. https://arxiv.org/abs/2404.08471
"""

import sys
import types
import zipfile
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
import torchvision.transforms.functional as TF

import stable_pretraining as spt
from stable_pretraining.methods.vjepa import VJEPA

NUM_FRAMES = 8
SPATIAL_SIZE = 224
NUM_CLASSES = 101
EMBED_DIM = 768  # ViT-Base


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _download_and_extract(data_dir: Path) -> tuple[Path, Path]:
    """Download UCF-101 zip files from HuggingFace and extract them.

    Returns:
        Tuple of (videos_root_dir, annotation_dir) paths.
    """
    from huggingface_hub import hf_hub_download

    videos_dir = data_dir / "UCF-101"
    splits_dir = data_dir / "ucfTrainTestlist"

    if not videos_dir.exists():
        print("Downloading UCF101.zip from HuggingFace (MichiganNLP/ucf-101)...")
        zip_path = hf_hub_download(
            repo_id="MichiganNLP/ucf-101",
            filename="UCF101.zip",
            repo_type="dataset",
            cache_dir=str(data_dir / "hf_cache"),
        )
        print(f"Extracting {zip_path} -> {data_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

    if not splits_dir.exists():
        print(
            "Downloading UCF101TrainTestSplits-RecognitionTask.zip from HuggingFace..."
        )
        splits_zip = hf_hub_download(
            repo_id="MichiganNLP/ucf-101",
            filename="UCF101TrainTestSplits-RecognitionTask.zip",
            repo_type="dataset",
            cache_dir=str(data_dir / "hf_cache"),
        )
        print(f"Extracting {splits_zip} -> {data_dir}")
        with zipfile.ZipFile(splits_zip, "r") as zf:
            zf.extractall(data_dir)

    return videos_dir, splits_dir


class UCF101VideoDataset(torch.utils.data.Dataset):
    """UCF-101 wrapper that returns ``{"video": (C,T,H,W), "label": int}`` dicts.

    Videos are loaded via :class:`torchvision.datasets.UCF101`.  Each clip is
    returned as a ``(C, T, H, W)`` float32 tensor normalised with UCF-101
    channel statistics.  Spatial augmentations (random-resized crop + flip for
    train; resize + centre-crop for val) are applied **consistently across all
    frames** by computing crop parameters once and reapplying them per frame.

    Args:
        root: Path to the extracted ``UCF-101/`` video directory.
        annotation_path: Path to the ``ucfTrainTestlist/`` annotation directory.
        train: If ``True``, use the training split with augmentations;
            otherwise use the validation split.
        frames_per_clip: Number of frames per returned clip.
        step_between_clips: Temporal stride between consecutive clips.
        fold: UCF-101 split fold index (1, 2, or 3).
    """

    _mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    _std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

    def __init__(
        self,
        root: str,
        annotation_path: str,
        train: bool,
        frames_per_clip: int = NUM_FRAMES,
        step_between_clips: int = 4,
        fold: int = 1,
    ):
        self.dataset = torchvision.datasets.UCF101(
            root=root,
            annotation_path=annotation_path,
            frames_per_clip=frames_per_clip,
            step_between_clips=step_between_clips,
            fold=fold,
            train=train,
        )
        self.train = train

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        video, _audio, label = self.dataset[idx]
        # torchvision UCF101 returns video as (T, H, W, C) uint8
        video = video.permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)

        if self.train:
            video = self._train_transform(video)
        else:
            video = self._val_transform(video)

        video = (video - self._mean) / self._std
        return {"video": video, "label": label, "sample_idx": idx}

    @staticmethod
    def _train_transform(video: torch.Tensor) -> torch.Tensor:
        """Consistent random-resized crop + horizontal flip across all T frames."""
        C, T, H, W = video.shape
        # Reference frame for computing crop parameters
        ref = video[:, 0]  # (C, H, W)
        i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(
            ref, scale=(0.5, 1.0), ratio=(0.75, 1.333)
        )
        # Apply the same crop to every frame
        frames = torch.stack(
            [
                TF.resized_crop(video[:, t], i, j, h, w, [SPATIAL_SIZE, SPATIAL_SIZE])
                for t in range(T)
            ]
        )  # (T, C, H, W)
        if torch.rand(1).item() > 0.5:
            frames = TF.hflip(frames)
        return frames.permute(1, 0, 2, 3)  # (C, T, H, W)

    @staticmethod
    def _val_transform(video: torch.Tensor) -> torch.Tensor:
        """Consistent centre-crop across all T frames."""
        C, T, H, W = video.shape
        scale = round(SPATIAL_SIZE * 256 / 224)
        frames = torch.stack(
            [
                TF.center_crop(
                    TF.resize(video[:, t], [scale]), [SPATIAL_SIZE, SPATIAL_SIZE]
                )
                for t in range(T)
            ]
        )  # (T, C, H, W)
        return frames.permute(1, 0, 2, 3)  # (C, T, H, W)


# ---------------------------------------------------------------------------
# Forward function
# ---------------------------------------------------------------------------


def vjepa_forward(self, batch, stage):
    """V-JEPA forward step for Lightning training loop.

    Args:
        self: VJEPA module instance.
        batch: Dict with ``"video"`` ``(B, C, T, H, W)`` and optional ``"label"``.
        stage: Training stage string (``"fit"``, ``"validate"``, etc.).

    Returns:
        Dict with ``"loss"``, ``"embedding"``, and optionally ``"label"``.
    """
    output = VJEPA.forward(self, batch["video"], embedding_source="teacher")
    embedding = output.embedding  # [B, D] mean-pooled

    if self.training:
        embedding = embedding.detach()

    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    self.log(
        f"{stage}/num_targets",
        float(output.num_targets),
        on_step=False,
        on_epoch=True,
        sync_dist=True,
    )

    return {
        "loss": output.loss,
        "embedding": embedding,
        **({"label": batch["label"].long()} if "label" in batch else {}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    num_gpus = torch.cuda.device_count() or 1
    batch_size = 16  # video batches are memory-intensive; adjust per GPU VRAM
    num_workers = 8
    max_epochs = 200

    data_dir = Path(get_data_dir("ucf101"))
    videos_dir, splits_dir = _download_and_extract(data_dir)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=UCF101VideoDataset(
                root=str(videos_dir),
                annotation_path=str(splits_dir),
                train=True,
                frames_per_clip=NUM_FRAMES,
                step_between_clips=4,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=UCF101VideoDataset(
                root=str(videos_dir),
                annotation_path=str(splits_dir),
                train=False,
                frames_per_clip=NUM_FRAMES,
                step_between_clips=8,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    module = VJEPA(
        encoder_name="vit_base_patch16_224",
        num_frames=NUM_FRAMES,
        predictor_embed_dim=384,
        predictor_depth=6,
        num_targets=8,
        target_scale=(0.15, 0.2),
        target_aspect_ratio=(0.75, 1.5),
        context_scale=(1.0, 1.0),
        ema_decay_start=0.996,
        ema_decay_end=1.0,
        pretrained=False,
    )

    module.forward = types.MethodType(vjepa_forward, module)
    module.optim = {
        "optimizer": {
            "type": "AdamW",
            "lr": (lr := 1.5e-4),
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": 15 / max_epochs,
            "start_factor": 0.01,
            "end_lr": lr / 100,
            "total_steps": (len(data.train) // num_gpus) * max_epochs,
        },
        "interval": "step",
    }

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.TeacherStudentCallback(
                update_frequency=1,
                update_after_backward=True,
            ),
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(EMBED_DIM, NUM_CLASSES),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES),
                    "top5": torchmetrics.classification.MulticlassAccuracy(
                        NUM_CLASSES, top_k=5
                    ),
                },
                optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 0.0},
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=4096,
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES)
                },
                input_dim=EMBED_DIM,
                k=20,
            ),
            spt.callbacks.RankMe(
                name="rankme",
                target="embedding",
                queue_length=1000,
                target_shape=EMBED_DIM,
            ),
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(Path(__file__).parent / "checkpoints" / "vjepa-vitb"),
                filename="vjepa-vitb-{epoch:03d}",
                save_top_k=-1,
                every_n_epochs=50,
                save_last=True,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.WandbLogger(
            entity="stable-ssl",
            project="ucf101-methods",
            name="vjepa-vitb-ucf101",
            log_model=False,
        ),
        precision="16-mixed",
        devices=num_gpus,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true" if num_gpus > 1 else "auto",
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
