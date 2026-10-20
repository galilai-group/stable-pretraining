import lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from stable_pretraining.methods.sce import SCE
from stable_pretraining.callbacks.sce_queue import SCEQueueCallback
from stable_pretraining.callbacks.teacher_student import TeacherStudentCallback


class TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {
            "view1": torch.randn(3, 32, 32),
            "view2": torch.randn(3, 32, 32),
        }


def make_backbone():
    """Small CNN to keep integration tests fast."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 64),
    )


def make_model(optim=None):
    return SCE(
        encoder_name=make_backbone(),
        projector_dims=(64, 32),
        queue_length=16,
        optim=optim,
    )


def test_sce_forward_and_queue():
    model = make_model()
    callback = SCEQueueCallback()

    batch = {
        "view1": torch.randn(2, 3, 32, 32),
        "view2": torch.randn(2, 3, 32, 32),
    }

    output = model(batch, stage="fit")

    assert output.queries.shape == (2, 32)
    assert output.keys.shape == (2, 32)
    assert torch.isfinite(output.loss)
    assert model.queue.get().shape == (0, 32)

    callback.on_train_batch_end(
        trainer=None,
        pl_module=model,
        outputs=output,
        batch=batch,
        batch_idx=0,
    )

    assert model.queue.get().shape == (2, 32)


def test_sce_lightning_training():
    torch.manual_seed(42)

    model = make_model(
        optim={"optimizer": {"type": "AdamW", "lr": 1e-3}}
    )

    student_before = model.projector.student[0].weight.detach().clone()
    teacher_before = model.projector.teacher[0].weight.detach().clone()

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=3,
        callbacks=[
            SCEQueueCallback(),
            TeacherStudentCallback(),
        ],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(TinyDataset(), batch_size=2),
    )

    student_after = model.projector.student[0].weight.detach()
    teacher_after = model.projector.teacher[0].weight.detach()

    assert trainer.global_step == 3
    assert model.queue.get().shape == (6, 32)
    assert not torch.equal(student_before, student_after)
    assert not torch.equal(teacher_before, teacher_after)
    assert not model.projector.teacher[0].weight.requires_grad


def test_sce_checkpoint_state():
    torch.manual_seed(42)

    model = make_model()
    model.enqueue_keys(torch.randn(6, 32))

    saved_state = model.state_dict()

    restored = make_model()
    restored.load_state_dict(saved_state)

    assert torch.allclose(model.queue.get(), restored.queue.get())
    assert torch.allclose(
        model.projector.student[0].weight,
        restored.projector.student[0].weight,
    )
    assert torch.allclose(
        model.projector.teacher[0].weight,
        restored.projector.teacher[0].weight,
    )


def test_sce_inference():
    model = make_model()
    model.eval()

    images = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        output = model(images)

    assert output.embedding.shape == (2, 64)
    assert output.embedding.requires_grad is False
    assert model.queue.get().shape == (0, 32)

def test_sce_multiview_input():
    model = make_model()

    batch = {
        "views": [
            {"image": torch.randn(2, 3, 32, 32)},
            {"image": torch.randn(2, 3, 32, 32)},
        ]
    }

    output = model(batch, stage="fit")

    assert output.queries.shape == (2, 32)
    assert output.keys.shape == (2, 32)
    assert torch.isfinite(output.loss)
    assert model.queue.get().shape == (0, 32)