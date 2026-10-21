"""Small CPU training integration test for PCL."""

import lightning as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stable_pretraining.callbacks import TeacherStudentCallback
from stable_pretraining.methods.pcl import (
    PCL,
    extract_clustering_features,
)
from types import SimpleNamespace
from stable_pretraining.methods.pcl import PCLClusteringCallback
from lightning.pytorch.callbacks import Callback


class TinyEncoder(nn.Module):
    """Small image encoder used for PCL integration tests."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(3, 16)

    def forward(self, x):
        return self.linear(self.pool(x).flatten(1))


class TinyDataset(Dataset):
    """Synthetic dataset providing two augmented views and sample indices."""

    def __len__(self):
        return 16

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)

        image = torch.randn(3, 8, 8, generator=generator)

        return {
            "image": image,
            "view1": image,
            "view2": image + 0.05,
            "indices": torch.tensor(index, dtype=torch.long),
        }


@pytest.mark.unit
def test_pcl_lightning_training_updates_ema(tmp_path):
    torch.manual_seed(7)

    model = PCL(
        encoder_name=TinyEncoder(),
        projector_dims=(32, 8),
        queue_length=16,
        temperature=0.2,
    )

    model.optim = {
        "optimizer": {
            "type": "SGD",
            "lr": 0.1,
        },
    }

    student_before = [p.detach().clone() for p in model.backbone.student.parameters()]
    teacher_before = [p.detach().clone() for p in model.backbone.teacher.parameters()]

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=4,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[TeacherStudentCallback(verbose=False)],
        default_root_dir=tmp_path,
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            TinyDataset(),
            batch_size=4,
            shuffle=False,
            num_workers=0,
        ),
    )

    assert trainer.global_step == 4

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            student_before,
            model.backbone.student.parameters(),
        )
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            teacher_before,
            model.backbone.teacher.parameters(),
        )
    )

    assert model.queue.get().shape[0] == 16


@pytest.mark.unit
def test_extract_clustering_features_preserves_indices():
    torch.manual_seed(7)

    model = PCL(
        encoder_name=TinyEncoder(),
        projector_dims=(32, 8),
        queue_length=16,
    )
    model.train()

    dataloader = DataLoader(
        TinyDataset(),
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    queue_before = model.queue.get().clone()

    features = extract_clustering_features(model, dataloader)

    assert features.shape == (16, 8)
    assert torch.isfinite(features).all()

    torch.testing.assert_close(
        features.norm(dim=1),
        torch.ones(16),
    )

    # Extractor must restore training state.
    assert model.training

    # Extractor must not modify the negative queue.
    torch.testing.assert_close(
        model.queue.get(),
        queue_before,
    )

    # Verify index ordering against individual teacher outputs.
    model.eval()
    dataset = TinyDataset()

    with torch.no_grad():
        for idx in (0, 3, 9, 15):
            image = dataset[idx]["image"].unsqueeze(0)

            expected = model.projector.forward_teacher(
                model.backbone.forward_teacher(image)
            )
            expected = torch.nn.functional.normalize(expected, dim=-1)

            torch.testing.assert_close(
                features[idx],
                expected.squeeze(0),
            )


@pytest.mark.unit
def test_pcl_clustering_callback_respects_warmup():
    torch.manual_seed(7)

    model = PCL(
        encoder_name=TinyEncoder(),
        projector_dims=(32, 8),
        queue_length=16,
    )

    loader = DataLoader(
        TinyDataset(),
        batch_size=4,
        shuffle=False,
    )

    callback = PCLClusteringCallback(
        dataloader=loader,
        num_clusters=3,
        warmup_epochs=1,
    )

    trainer = SimpleNamespace(
        current_epoch=0,
        world_size=1,
    )

    # Epoch 0: warm-up, no clustering.
    callback.on_train_epoch_start(trainer, model)

    assert model.cluster_assignments.numel() == 0

    # Epoch 1: clustering begins.
    trainer.current_epoch = 1
    callback.on_train_epoch_start(trainer, model)

    assert model.cluster_assignments.shape == (16,)
    assert model.cluster_prototypes.shape == (3, 8)
    assert model.cluster_concentrations.shape == (3,)

    assert torch.isfinite(model.cluster_concentrations).all()
    assert (model.cluster_concentrations > 0).all()


class PCLLossRecorder(Callback):
    """Record PCL losses during Lightning training."""

    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.records.append(
            {
                "epoch": trainer.current_epoch,
                "instance_loss": outputs["instance_loss"].detach().cpu().item(),
                "prototype_loss": (
                    None
                    if outputs.get("prototype_loss") is None
                    else outputs["prototype_loss"].detach().cpu().item()
                ),
                "total_loss": outputs["loss"].detach().cpu().item(),
            }
        )


@pytest.mark.unit
def test_pcl_warmup_to_prototype_training(tmp_path):
    """Run warm-up followed by prototype training with clustering refresh."""
    torch.manual_seed(7)

    model = PCL(
        encoder_name=TinyEncoder(),
        projector_dims=(32, 8),
        queue_length=16,
        temperature=0.2,
    )
    model.num_proto_negatives = 1
    model.optim = {"optimizer": {"type": "SGD", "lr": 0.1}}

    dataset = TinyDataset()

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    clustering_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    clustering_callback = PCLClusteringCallback(
        dataloader=clustering_loader,
        num_clusters=(8, 10),
        warmup_epochs=1,
        refresh_every=1,
        seed=7,
    )

    loss_recorder = PCLLossRecorder()

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            TeacherStudentCallback(verbose=False),
            clustering_callback,
            loss_recorder,
        ],
        default_root_dir=tmp_path,
    )

    trainer.fit(model, train_dataloaders=train_loader)

    # Two epochs, eight batches per epoch.
    assert trainer.global_step == 16

    # Clustering must have started after warm-up.
    assert model.cluster_assignments.shape == (16,)
    assert model.cluster_prototypes.shape == (8, 8)
    assert model.cluster_concentrations.shape == (8,)

    # Prototype temperatures must remain positive.
    assert torch.isfinite(model.cluster_concentrations).all()
    assert (model.cluster_concentrations > 0).all()

    warmup_records = [r for r in loss_recorder.records if r["epoch"] == 0]
    prototype_records = [r for r in loss_recorder.records if r["epoch"] == 1]

    assert len(warmup_records) == 8
    assert len(prototype_records) == 8

    # Warm-up: instance-only training.
    for record in warmup_records:
        assert record["prototype_loss"] is None
        assert torch.isfinite(torch.tensor(record["instance_loss"]))
        assert record["total_loss"] == pytest.approx(record["instance_loss"])

    # After clustering: both objectives must contribute.
    for record in prototype_records:
        assert record["prototype_loss"] is not None
        assert torch.isfinite(torch.tensor(record["prototype_loss"]))
        assert record["total_loss"] == pytest.approx(
            record["instance_loss"] + record["prototype_loss"],
            rel=1e-5,
            abs=1e-6,
        )
    assert len(model.additional_clusterings) == 1

    second = model.additional_clusterings[0]

    assert second.assignments.shape == (16,)
    assert second.prototypes.shape == (10, 8)
    assert second.concentrations.shape == (10,)

    assert torch.isfinite(second.concentrations).all()
    assert (second.concentrations > 0).all()


@pytest.mark.unit
def test_pcl_multiresolution_clustering_callback():
    """Refresh multiple clustering resolutions after warm-up."""
    torch.manual_seed(7)

    model = PCL(
        encoder_name=TinyEncoder(),
        projector_dims=(32, 8),
        queue_length=16,
    )

    loader = DataLoader(
        TinyDataset(),
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    callback = PCLClusteringCallback(
        dataloader=loader,
        num_clusters=(3, 5),
        warmup_epochs=1,
        seed=7,
    )

    trainer = SimpleNamespace(
        current_epoch=0,
        world_size=1,
    )

    # Warm-up: no clustering should exist.
    callback.on_train_epoch_start(trainer, model)
    assert model.cluster_assignments.numel() == 0

    # First clustering refresh.
    trainer.current_epoch = 1
    callback.on_train_epoch_start(trainer, model)

    assert model.cluster_prototypes.shape == (3, 8)
    assert len(model.additional_clusterings) == 1

    second = model.additional_clusterings[0]
    assert second.prototypes.shape == (5, 8)

    for concentrations in (
        model.cluster_concentrations,
        second.concentrations,
    ):
        assert torch.isfinite(concentrations).all()
        assert (concentrations > 0).all()
        torch.testing.assert_close(
            concentrations.mean(),
            torch.tensor(0.2),
        )


@pytest.mark.unit
def test_pcl_lightning_checkpoint_resume(tmp_path):
    """Restore PCL clustering, queue, EMA, and optimizer state."""
    torch.manual_seed(7)

    dataset = TinyDataset()
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    cluster_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    def make_pcl():
        model = PCL(
            encoder_name=TinyEncoder(),
            projector_dims=(32, 8),
            queue_length=16,
            temperature=0.2,
        )
        model.num_proto_negatives = 1
        model.optim = {"optimizer": {"type": "AdamW", "lr": 0.001}}
        return model

    def make_callbacks():
        return [
            TeacherStudentCallback(verbose=False),
            PCLClusteringCallback(
                dataloader=cluster_loader,
                num_clusters=(8, 10),
                warmup_epochs=1,
                refresh_every=10,
                seed=7,
            ),
        ]

    def make_trainer(epochs):
        return pl.Trainer(
            max_epochs=epochs,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=make_callbacks(),
            default_root_dir=tmp_path,
        )

    # Phase 1: Warm-up and one prototype-training epoch.
    model = make_pcl()
    trainer = make_trainer(2)
    trainer.fit(model, train_dataloaders=train_loader)

    assert trainer.global_step == 16
    assert len(model.additional_clusterings) == 1

    path = tmp_path / "pcl_resume.ckpt"
    trainer.save_checkpoint(path)

    saved = torch.load(path, weights_only=False)

    assert saved["global_step"] == 16
    assert saved["optimizer_states"]
    assert saved["optimizer_states"][0]["state"]

    # Phase 2: Resume into a fresh PCL model.
    restored = make_pcl()
    resumed_trainer = make_trainer(3)

    resumed_trainer.fit(
        restored,
        train_dataloaders=train_loader,
        ckpt_path=path,
        weights_only=False,
    )

    assert resumed_trainer.global_step == 24
    assert len(restored.additional_clusterings) == 1

    # No refresh occurs in epoch 2 (refresh_every=10),
    # so clustering buffers should match the checkpoint.
    for name in (
        "cluster_assignments",
        "cluster_prototypes",
        "cluster_concentrations",
    ):
        torch.testing.assert_close(
            getattr(restored, name),
            saved["state_dict"][name],
        )

    second = restored.additional_clusterings[0]
    torch.testing.assert_close(
        second.assignments,
        saved["state_dict"]["additional_clusterings.0.assignments"],
    )
    torch.testing.assert_close(
        second.prototypes,
        saved["state_dict"]["additional_clusterings.0.prototypes"],
    )

    assert torch.isfinite(restored.cluster_concentrations).all()
