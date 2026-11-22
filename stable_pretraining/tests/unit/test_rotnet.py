"""Unit tests for RotNet: rotations, the NIN backbone, the method class, and Manager training."""

import copy
import json
from unittest.mock import Mock

import lightning as pl
import numpy as np
import pytest
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining import forward as forward_module
from stable_pretraining.backbone import NetworkInNetwork
from stable_pretraining.forward import _rotate_batch
from stable_pretraining.manager import Manager
from stable_pretraining.methods import RotNet

pytestmark = pytest.mark.unit


def _images(n=4, size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, size, size, generator=g)


def _official_rotate(img: np.ndarray, rot: int) -> np.ndarray:
    """``rotate_img`` from the official RotNet ``dataloader.py`` (HWC arrays)."""
    if rot == 0:
        return img
    if rot == 90:
        return np.flipud(np.transpose(img, (1, 0, 2)))
    if rot == 180:
        return np.fliplr(np.flipud(img))
    if rot == 270:
        return np.transpose(np.flipud(img), (1, 0, 2))
    raise ValueError(rot)


class TestRotateBatch:
    """Rotations match the official data loader."""

    def test_matches_official_rotation(self):
        images = torch.arange(2 * 3 * 5 * 5, dtype=torch.float32).view(2, 3, 5, 5)
        rotated, _ = _rotate_batch(images)
        for r, angle in enumerate((0, 90, 180, 270)):
            for b in range(2):
                hwc = images[b].permute(1, 2, 0).numpy()
                expected = _official_rotate(hwc, angle).copy()
                got = rotated[r * 2 + b].permute(1, 2, 0).numpy()
                np.testing.assert_array_equal(got, expected)

    def test_layout_and_labels(self):
        images = _images(n=3)
        rotated, labels = _rotate_batch(images, (0, 180))
        assert rotated.shape == (6, 3, 32, 32)
        assert labels.tolist() == [0, 0, 0, 1, 1, 1]
        assert torch.equal(rotated[:3], images)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            _rotate_batch(torch.zeros(1, 3, 32, 16))

    def test_rejects_non_right_angles(self):
        with pytest.raises(ValueError, match="multiples of 90"):
            _rotate_batch(_images(n=1), (0, 45))


class TestNetworkInNetwork:
    """The NIN backbone matches the official CIFAR-10 shapes."""

    @pytest.mark.parametrize("num_blocks", [3, 4, 5])
    def test_block_shapes_on_cifar(self, num_blocks):
        nin = NetworkInNetwork(num_blocks=num_blocks, pool=False)
        x = _images(n=2)
        shapes = []
        for block in nin.blocks:
            x = block(x)
            shapes.append(tuple(x.shape))
        assert shapes[0] == (2, 96, 16, 16)
        assert shapes[1:] == [(2, 192, 8, 8)] * (num_blocks - 1)

    def test_pooled_output(self):
        nin = NetworkInNetwork(num_blocks=4)
        assert nin(_images(n=2)).shape == (2, 192)
        assert nin.embed_dim == 192

    def test_batch_norm_scales_follow_pytorch_0_2_init(self):
        nin = NetworkInNetwork(num_blocks=4)
        scales = torch.cat(
            [m.weight for m in nin.modules() if isinstance(m, nn.BatchNorm2d)]
        )
        assert scales.min() >= 0 and scales.max() < 1
        assert scales.std() > 0.1

    def test_rejects_too_few_blocks(self):
        with pytest.raises(ValueError, match="at least 3"):
            NetworkInNetwork(num_blocks=2)


class TestRotNetMethod:
    """The RotNet method class."""

    def test_train_forward_shapes(self):
        model = RotNet().train()
        out = model(_images(n=4))
        assert out.loss.ndim == 0 and torch.isfinite(out.loss)
        assert out.embedding.shape == (4, 192)
        assert out.rotation_logits.shape == (16, 4)
        assert 0.0 <= out.rotation_accuracy.item() <= 1.0

    @pytest.mark.parametrize("feature_block,dim", [(1, 96), (2, 192), (3, 192)])
    def test_embedding_is_pooled_unrotated_backbone_features(self, feature_block, dim):
        model = RotNet(num_blocks=4, feature_block=feature_block).eval()
        images = _images(n=3)
        with torch.no_grad():
            out = model(images)
            expected = model.backbone(images).mean(dim=(2, 3))
        assert model.embed_dim == dim
        assert torch.allclose(out.embedding, expected, atol=1e-5)

    def test_rotation_subset_sets_head_size(self):
        model = RotNet(rotations=(0, 180))
        assert model(_images(n=2)).rotation_logits.shape == (4, 2)

    def test_rotations_without_identity_still_embed_unrotated_images(self):
        model = RotNet(rotations=(90, 270)).eval()
        images = _images(n=2)
        with torch.no_grad():
            out = model(images)
            expected = model.backbone(images).mean(dim=(2, 3))
        assert torch.allclose(out.embedding, expected, atol=1e-5)

    def test_extra_embedding_pass_leaves_batch_norm_stats_alone(self):
        model = RotNet(num_blocks=3, rotations=(90, 270)).train()
        reference = copy.deepcopy(model)
        images = _images(n=4)
        model(images)
        with torch.no_grad():
            reference.backbone(_rotate_batch(images, (90, 270))[0])
        for (name, got), want in zip(
            model.backbone.named_buffers(), reference.backbone.buffers()
        ):
            assert torch.equal(got, want), name
        assert model.backbone.training

    @pytest.mark.parametrize("feature_block", [0, 4])
    def test_rejects_bad_feature_block(self, feature_block):
        with pytest.raises(ValueError, match="feature_block"):
            RotNet(num_blocks=4, feature_block=feature_block)

    @pytest.mark.parametrize("pooled", [True, False])
    def test_custom_encoder(self, pooled):
        layers = [nn.Conv2d(3, 8, 3, padding=1)]
        if pooled:
            layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        model = RotNet(encoder_name=nn.Sequential(*layers))
        out = model(_images(n=2))
        assert model.embed_dim == 8
        assert out.embedding.shape == (2, 8)
        assert out.rotation_logits.shape == (8, 4)

    def test_batch_dict_runs_forward_function(self):
        model = RotNet(num_blocks=3)
        model.log = Mock()
        images = _images(n=2)
        out = model({"image": images, "label": torch.tensor([1, 2])}, stage="fit")
        assert set(out) == {"embedding", "loss", "rotation_logits", "label"}
        logged = {call.args[0] for call in model.log.call_args_list}
        assert logged == {"fit/loss", "fit/rotation_accuracy"}

    def test_learns_to_recognise_rotations(self):
        torch.manual_seed(0)
        model = RotNet(num_blocks=3).train()
        images = _images(n=8, size=16)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = model(images).loss.item()
        for _ in range(30):
            opt.zero_grad()
            out = model(images)
            out.loss.backward()
            opt.step()
        assert out.loss.item() < 0.5 * first
        assert out.rotation_accuracy.item() > 0.9


class TestRotNetForwardFunction:
    """The composable ``forward.rotnet`` function."""

    def _module(self, training=True):
        module = Mock()
        module.backbone = lambda x: torch.randn(x.shape[0], 192, 8, 8)
        module.rotation_head = lambda f: torch.randn(f.shape[0], 4, requires_grad=True)
        module.training = training
        module.log = Mock()
        module.rotnet_loss = nn.CrossEntropyLoss()
        del module.rotations
        return module

    def test_validation_keeps_loss_and_accuracy(self):
        module = self._module(training=False)
        out = forward_module.rotnet(module, {"image": _images(n=2)}, "validate")
        assert "loss" in out and "label" not in out
        logged = {call.args[0] for call in module.log.call_args_list}
        assert "validate/rotation_accuracy" in logged

    def test_requires_rotnet_loss(self):
        module = self._module()
        del module.rotnet_loss
        with pytest.raises(ValueError, match="rotnet_loss"):
            forward_module.rotnet(module, {"image": _images(n=2)}, "fit")

    def test_uses_custom_loss_and_rotations(self):
        module = self._module()
        module.rotations = (0, 90)
        module.rotation_head = lambda f: torch.zeros(f.shape[0], 2)
        module.rotnet_loss = Mock(return_value=torch.tensor(0.25))
        out = forward_module.rotnet(module, {"image": _images(n=3)}, "fit")
        logits, labels = module.rotnet_loss.call_args.args
        assert logits.shape == (6, 2)
        assert labels.tolist() == [0, 0, 0, 1, 1, 1]
        assert out["loss"].item() == 0.25


# --- Manager training and SLURM requeue resume -----------------------------

_SEED = 42
_STEPS_PER_EPOCH = 32 // 8
# Milestones are in optimizer steps (Module steps schedulers every step): one LR
# drop inside each epoch, so a resume that loses scheduler state diverges.
_OPTIM = {
    "optimizer": {"type": "SGD", "lr": 0.1, "momentum": 0.9, "nesterov": True},
    "scheduler": {"type": "MultiStepLR", "milestones": [2, 6], "gamma": 0.2},
}
_TRAINER = {
    "accelerator": "cpu",
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}


class _FakeImages(Dataset):
    """Fixed images and labels, so a resumed run sees the same batches."""

    def __init__(self, num_samples=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(num_samples, 3, 16, 16, generator=g)
        self.labels = torch.randint(0, 10, (num_samples,), generator=g)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


def _data():
    return spt.data.DataModule(
        train=DataLoader(_FakeImages(), batch_size=8, drop_last=True),
        val=DataLoader(_FakeImages(num_samples=16, seed=1), batch_size=8),
    )


def _composable_module():
    """forward.rotnet on a plain Module: the YAML-config form."""
    pl.seed_everything(_SEED, workers=True)
    nin = NetworkInNetwork(num_blocks=3, pool=False)
    return spt.Module(
        backbone=nn.Sequential(*nin.blocks[:2]),
        rotation_head=nn.Sequential(
            nin.blocks[2], nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(192, 4)
        ),
        forward=forward_module.rotnet,
        rotnet_loss=nn.CrossEntropyLoss(),
        optim=_OPTIM,
    )


def _method_class():
    """The RotNet class, trained directly through its batch-dict forward."""
    pl.seed_everything(_SEED, workers=True)
    model = RotNet(encoder_name="nin", num_blocks=3, feature_block=2)
    model.optim = _OPTIM
    return model


_BUILDERS = {"module": _composable_module, "method_class": _method_class}


class _PreemptAfterFirstEpoch(pl.Callback):
    """Stop after epoch 1 while keeping the planned horizon (so the LR schedule matches)."""

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def _run_summary(manager) -> dict:
    """Final metrics of the run this Manager trained, read from its sidecar."""
    sidecar = json.loads((manager._run_dir / "sidecar.json").read_text())
    assert sidecar["status"] == "completed", sidecar["status"]
    return sidecar["summary"]


class TestRotNetUnderManager:
    """Real Manager training for both entry points: probes, logging, and resume."""

    @pytest.mark.parametrize("entry_point", list(_BUILDERS))
    def test_trains_with_probes(self, entry_point):
        module = _BUILDERS[entry_point]()
        accuracy = torchmetrics.classification.MulticlassAccuracy(10)
        probe = spt.callbacks.OnlineProbe(
            module,
            name="linear_probe",
            input="embedding",
            target="label",
            probe=nn.Linear(192, 10),
            loss=nn.CrossEntropyLoss(),
            metrics={"top1": accuracy},
        )
        knn = spt.callbacks.OnlineKNN(
            name="knn_probe",
            input="embedding",
            target="label",
            queue_length=32,
            metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
            input_dim=192,
            k=5,
        )
        trainer = pl.Trainer(
            max_epochs=1, callbacks=[probe, knn], logger=False, **_TRAINER
        )
        manager = Manager(trainer=trainer, module=module, data=_data(), seed=_SEED)
        manager()

        summary = _run_summary(manager)
        for key in (
            "fit/loss_epoch",
            "fit/rotation_accuracy",
            "validate/loss_epoch",
            "validate/rotation_accuracy",
        ):
            assert key in summary, (key, sorted(summary))
            assert torch.isfinite(torch.tensor(summary[key]))
        assert 0.0 <= summary["validate/rotation_accuracy"] <= 1.0
        assert "eval/linear_probe_top1_epoch" in summary, sorted(summary)
        assert "eval/knn_probe_top1" in summary, sorted(summary)

    @pytest.mark.parametrize("entry_point", list(_BUILDERS))
    def test_requeue_resumes_and_matches_uninterrupted(self, entry_point, monkeypatch):
        """A SLURM requeue resumes from last.ckpt and ends where a 2-epoch run ends."""
        builder = _BUILDERS[entry_point]

        def trainer_cfg():
            return OmegaConf.create(
                {"_target_": "lightning.Trainer", "max_epochs": 2, **_TRAINER}
            )

        # Reference: two uninterrupted epochs.
        reference = builder()
        reference_manager = Manager(
            trainer=trainer_cfg(), module=reference, data=_data(), seed=_SEED
        )
        reference_manager()
        reference_loss = _run_summary(reference_manager)["fit/loss_epoch"]

        # Leg 1: the same job plans two epochs but is preempted after the first;
        # the Manager's requeue checkpoint is written at the epoch end.
        monkeypatch.setenv("SLURM_JOB_ID", "424243")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        first = Manager(
            trainer=pl.Trainer(
                max_epochs=2,
                callbacks=[_PreemptAfterFirstEpoch()],
                logger=False,
                **_TRAINER,
            ),
            module=builder(),
            data=_data(),
            seed=_SEED,
        )
        first()
        ckpt_path = first._run_dir / "checkpoints" / "last.ckpt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["global_step"] == _STEPS_PER_EPOCH, ckpt["global_step"]

        # Leg 2: SLURM requeues the job; the Manager must resume into the same run_dir.
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        resumed = builder()
        second = Manager(
            trainer=trainer_cfg(), module=resumed, data=_data(), seed=_SEED
        )
        second()
        assert second._run_dir == first._run_dir
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["global_step"] == 2 * _STEPS_PER_EPOCH, ckpt["global_step"]

        resumed_loss = _run_summary(second)["fit/loss_epoch"]
        assert resumed_loss == pytest.approx(reference_loss, abs=1e-5)
        for key, value in reference.backbone.state_dict().items():
            assert torch.allclose(value, resumed.backbone.state_dict()[key], atol=1e-6)

        if entry_point == "method_class":
            # The checkpoint rebuilds the non-default (3-block) model on its own.
            reloaded = RotNet.load_from_checkpoint(ckpt_path, strict=False)
            assert len(reloaded.backbone) == 2 and len(reloaded.rotation_head) == 4
