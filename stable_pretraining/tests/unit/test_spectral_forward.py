"""Composable SpectralCL forward and Manager lifecycle regressions."""

import copy

import lightning as pl
import pytest
import torch
from torch import nn

import stable_pretraining as spt

pytestmark = pytest.mark.unit


def _module():
    return spt.Module(
        forward=spt.forward.spectral,
        backbone=nn.Linear(4, 6),
        projector=nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8)),
        spectral_loss=spt.losses.SpectralContrastiveLoss(mu=2.0),
        optim={
            "optimizer": {"type": "SGD", "lr": 0.03, "momentum": 0.9},
            "scheduler": {"type": "ConstantLR", "factor": 1.0, "total_iters": 1},
        },
    )


def _batch(named=False):
    views = [{"image": torch.randn(4, 4), "label": torch.arange(4)} for _ in range(2)]
    return {"left": views[0], "right": views[1]} if named else {"views": views}


@pytest.mark.parametrize("named", [False, True])
def test_composable_forward_matches_explicit_loss_and_gradients(named):
    module = _module().train()
    reference = copy.deepcopy(module)
    batch = _batch(named)
    views = list(batch.values()) if named else batch["views"]
    output = module(batch | {"batch_idx": 7}, stage="fit")
    embeddings = [reference.backbone(view["image"]) for view in views]
    projections = [reference.projector(embedding) for embedding in embeddings]
    expected = reference.spectral_loss(*projections)

    torch.testing.assert_close(output["loss"], expected)
    torch.testing.assert_close(output["embedding"], torch.cat(embeddings))
    assert output["label"].tolist() == list(range(4)) * 2
    output["loss"].backward()
    expected.backward()
    for actual, target in zip(module.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, target.grad)


@pytest.mark.parametrize("stage", ["validate", "test", "predict"])
def test_composable_eval_needs_only_backbone(stage):
    module = spt.Module(forward=spt.forward.spectral, backbone=nn.Linear(4, 6)).eval()
    batch = {"image": torch.randn(3, 4), "label": torch.arange(3)}
    with torch.no_grad():
        output = module(batch, stage=stage)
    assert set(output) == {"embedding", "label"}
    torch.testing.assert_close(output["embedding"], module.backbone(batch["image"]))
    torch.testing.assert_close(output["label"], batch["label"])


def test_empty_eval_batch_has_clear_error():
    with pytest.raises(ValueError, match="at least one image view"):
        _module().eval()({"batch_idx": 0}, stage="validate")


def test_manager_training_and_resume_match_uninterrupted_training(
    tmp_path, monkeypatch
):
    # One identical batch per epoch makes checkpoint continuation independent
    # of dataloader shuffling while exercising optimizer momentum restoration.
    pl.seed_everything(17, workers=True)
    initial = _module()
    batch = _batch(named=True)
    train = torch.utils.data.DataLoader([batch], batch_size=None)
    val = torch.utils.data.DataLoader([batch["left"]], batch_size=None)
    data = spt.data.DataModule(train=train, val=val)
    monkeypatch.setattr(spt.get_config(), "requeue_checkpoint", False)
    monkeypatch.setattr(
        spt.get_config(), "default_callbacks", {"env_dump": False, "logging": False}
    )

    def run(module, epochs, ckpt_path=None):
        trainer = pl.Trainer(
            accelerator="cpu",
            max_epochs=epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        spt.Manager(
            trainer=trainer,
            module=module,
            data=data,
            seed=17,
            ckpt_path=ckpt_path,
            weights_only=False,
        )()
        return trainer

    uninterrupted = copy.deepcopy(initial)
    run(uninterrupted, 2)
    first_epoch = copy.deepcopy(initial)
    first_trainer = run(first_epoch, 1)
    checkpoint = tmp_path / "spectral.ckpt"
    first_trainer.save_checkpoint(checkpoint)
    resumed = copy.deepcopy(initial)
    resumed_trainer = run(resumed, 2, str(checkpoint))

    assert resumed_trainer.global_step == 2
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial.parameters(), resumed.parameters())
    )
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key])
