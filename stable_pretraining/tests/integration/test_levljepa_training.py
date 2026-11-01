"""Real Manager training/resume and two-process DDP gradient checks."""

import copy
import math
import os
import tempfile
import time
from pathlib import Path

import lightning as pl
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt
from stable_pretraining.tests.unit.test_levljepa import make_batch, make_model


def _make_real_encoder_model() -> spt.Module:
    from timm.models.vision_transformer import VisionTransformer
    from transformers import GPT2Config, GPT2Model

    return spt.LeVLJEPA(
        vision_encoder=VisionTransformer(
            img_size=16,
            patch_size=8,
            embed_dim=16,
            depth=1,
            num_heads=2,
            num_classes=0,
        ),
        text_encoder=GPT2Model(
            GPT2Config(
                vocab_size=16,
                n_positions=4,
                n_embd=16,
                n_layer=1,
                n_head=2,
                n_inner=32,
                attn_pdrop=0,
                resid_pdrop=0,
                embd_pdrop=0,
            )
        ),
        vision_dim=16,
        text_dim=16,
        embedding_dim=8,
        pre_projection_dims=(16,),
        predictor_dims=(16, 16),
        num_projections=16,
    )


class _Pairs(Dataset):
    def __init__(self, real_encoders: bool = False) -> None:
        self.batch = make_batch()
        if real_encoders:
            self.batch["image"] = torch.randn(4, 3, 16, 16)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.batch.items()}


@pytest.mark.integration
@pytest.mark.parametrize("real_encoders", [False, True])
@pytest.mark.parametrize("warmup", [False, True])
def test_manager_training_and_full_checkpoint_resume(tmp_path, real_encoders, warmup):
    spt.set(cache_dir=str(tmp_path / "cache"))
    torch.set_num_threads(1)
    pl.seed_everything(123)
    loader = DataLoader(_Pairs(real_encoders), batch_size=4)
    data = spt.data.DataModule(train=loader, val=loader)
    factory = _make_real_encoder_model if real_encoders else make_model
    model = factory()
    if warmup:
        model.optim = {
            "optimizer": {"type": "AdamW", "lr": 1e-3},
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": 2,
                "max_steps": 6,
                "warmup_start_lr": 1e-5,
                "eta_min": 1e-4,
            },
            "interval": "step",
        }
    initial = copy.deepcopy(model.vision_encoder.state_dict())
    trainer = pl.Trainer(
        gradient_clip_val=1.0 if warmup else 0.0,
        max_steps=3,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    spt.Manager(trainer=trainer, module=model, data=data, seed=123)()
    assert trainer.global_step == 3
    assert any(
        not torch.equal(initial[k], v)
        for k, v in model.vision_encoder.state_dict().items()
    )
    checkpoint = str((tmp_path / "resume.ckpt").resolve())
    trainer.save_checkpoint(checkpoint)
    before = int(model.levljepa_loss.vision_regularizer.step)
    restored = factory()
    restored.optim = copy.deepcopy(model.optim)
    resumed = pl.Trainer(
        gradient_clip_val=1.0 if warmup else 0.0,
        max_steps=5,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    spt.Manager(
        trainer=resumed,
        module=restored,
        data=data,
        seed=123,
        ckpt_path=checkpoint,
        weights_only=False,
    )()
    assert resumed.global_step == 5
    if warmup:
        scheduler = resumed.lr_scheduler_configs[0].scheduler
        assert scheduler.last_epoch == 5
        assert scheduler.warmup_steps == 2 and scheduler.max_steps == 6
        expected_lr = 1e-4 + (1e-3 - 1e-4) * (1 + math.cos(3 * math.pi / 4)) / 2
        assert resumed.optimizers[0].param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert int(restored.levljepa_loss.vision_regularizer.step) == before + 2
    assert all(torch.isfinite(p).all() for p in restored.parameters())
    if real_encoders:
        counts = [
            int(layer.num_batches_tracked)
            for layer in restored.modules()
            if isinstance(layer, torch.nn.BatchNorm1d)
        ]
        assert counts and all(count == 5 for count in counts)
    state = resumed.optimizers[0].state_dict()["state"]
    assert state and all(int(value["step"]) == 5 for value in state.values())


def _ddp_worker(
    rank: int,
    init_file: str,
    real_encoders: bool = False,
    backend: str = "gloo",
    local_rank: int | None = None,
    sync_batchnorm: bool = False,
) -> None:
    torch.set_num_threads(1)
    device_index = rank if local_rank is None else local_rank
    device = (
        torch.device("cuda", device_index) if backend == "nccl" else torch.device("cpu")
    )
    if backend == "nccl":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(19)
    if real_encoders:
        model = _make_real_encoder_model()
        reference = _make_real_encoder_model()
        reference.levljepa_loss.vision_regularizer.distributed = False
        reference.levljepa_loss.text_regularizer.distributed = False
    else:
        model = make_model(distributed=True)
        reference = make_model(distributed=False)
    reference.load_state_dict(model.state_dict())
    if sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    reference.to(device)
    dist.init_process_group(
        backend,
        init_method="env://" if init_file == "env://" else f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(29)
        first, second = make_batch(), make_batch()
        if real_encoders:
            first["image"] = torch.randn(4, 3, 16, 16)
            second["image"] = torch.randn(4, 3, 16, 16)
        first = {key: value.to(device) for key, value in first.items()}
        second = {key: value.to(device) for key, value in second.items()}
        full = {key: torch.cat([first[key], second[key]]) for key in first}
        local = first if rank == 0 else second
        # Deliberately different RNG histories must not change the projection slices.
        torch.manual_seed(100 + rank)
        torch.randn(rank * 31, device=device)
        wrapped = DistributedDataParallel(
            model, device_ids=[device_index] if backend == "nccl" else None
        )
        output = wrapped(local, stage="fit")
        if real_encoders and not sync_batchnorm:
            # Ordinary DDP BatchNorm uses rank-local statistics, not global ones.
            parts = [reference(batch, stage="predict") for batch in (first, second)]
            expected = reference.levljepa_loss(
                *[
                    torch.cat([part[key] for part in parts])
                    for key in (
                        "image_embedding",
                        "text_embedding",
                        "image_prediction",
                        "text_prediction",
                    )
                ]
            )
        else:
            expected = reference(full, stage="fit")
        baseline = (
            0.96 * expected["prediction_loss"]
            + 0.02 * (expected["sigreg_vision"] + expected["sigreg_text"]) / 2
        )
        mean_loss = output["loss"].detach().clone()
        dist.all_reduce(mean_loss)
        torch.testing.assert_close(
            mean_loss / 2, baseline.detach(), rtol=2e-5, atol=2e-6
        )
        output["loss"].backward()
        baseline.backward()
        for (name, actual), (_, target) in zip(
            model.named_parameters(), reference.named_parameters()
        ):
            assert actual.grad is not None, name
            torch.testing.assert_close(
                actual.grad, target.grad, rtol=2e-4, atol=2e-6, msg=name
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.distributed
@pytest.mark.parametrize("real_encoders", [False, True])
def test_real_two_process_ddp_matches_concatenated_batch_gradients(real_encoders):
    if os.name == "nt" or not dist.is_gloo_available():
        pytest.skip("CPU gloo required")
    with tempfile.TemporaryDirectory() as directory:
        context = mp.spawn(
            _ddp_worker,
            args=(str(Path(directory) / "init"), real_encoders),
            nprocs=2,
            join=False,
        )
        try:
            if not context.join(timeout=45):
                if not context.join(timeout=5):
                    raise AssertionError("LeVLJEPA DDP test timed out")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)


def _check_two_gpu_nccl(real_encoders: bool, sync_batchnorm: bool = False) -> None:
    with tempfile.TemporaryDirectory() as directory:
        context = mp.spawn(
            _ddp_worker,
            args=(
                str(Path(directory) / "init"),
                real_encoders,
                "nccl",
                None,
                sync_batchnorm,
            ),
            nprocs=2,
            join=False,
        )
        deadline = time.monotonic() + 300
        try:
            while not context.join(timeout=5):
                if time.monotonic() >= deadline:
                    pytest.fail("Two-GPU NCCL comparison timed out")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("real_encoders", [False, True])
def test_two_gpu_nccl_matches_reference_loss_and_gradients(real_encoders):
    if not dist.is_nccl_available() or torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA GPUs and NCCL required")
    _check_two_gpu_nccl(real_encoders)


@pytest.mark.integration
@pytest.mark.gpu
def test_two_gpu_nccl_sync_batchnorm_matches_global_reference():
    if not dist.is_nccl_available() or torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA GPUs and NCCL required")
    _check_two_gpu_nccl(real_encoders=True, sync_batchnorm=True)


@pytest.mark.integration
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS required")
def test_real_encoders_batchnorm_heads_on_mps():
    torch.manual_seed(123)
    model = _make_real_encoder_model().to("mps")
    batch = make_batch()
    batch["image"] = torch.randn(4, 3, 16, 16)
    batch = {key: value.to("mps") for key, value in batch.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    before = {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
    }
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(batch, stage="fit")["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        optimizer.step()
    assert any(
        not torch.equal(before[name], value.detach().cpu())
        for name, value in model.named_parameters()
    )
    assert int(model.levljepa_loss.vision_regularizer.step) == 2


class _NCCLTrainingAudit(pl.Callback):
    def __init__(self, directory: Path, expected_steps: int) -> None:
        self.directory = directory
        self.expected_steps = expected_steps
        self.backward_checks = 0

    def on_after_backward(self, trainer, pl_module) -> None:
        for name, parameter in pl_module.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        self.backward_checks += 1

    def on_train_end(self, trainer, pl_module) -> None:
        import json

        assert dist.get_backend() == "nccl"
        assert dist.get_world_size() == 2
        assert pl_module.device.type == "cuda"
        assert trainer.global_step == self.expected_steps
        assert self.backward_checks > 0
        assert (
            int(pl_module.levljepa_loss.vision_regularizer.step) == self.expected_steps
        )
        assert int(pl_module.levljepa_loss.text_regularizer.step) == self.expected_steps
        counts = [
            int(layer.num_batches_tracked)
            for layer in pl_module.modules()
            if isinstance(layer, torch.nn.BatchNorm1d)
        ]
        assert counts and all(count == self.expected_steps for count in counts)
        state = trainer.optimizers[0].state_dict()["state"]
        assert state and all(
            int(value["step"]) == self.expected_steps for value in state.values()
        )
        assert all(torch.isfinite(p).all() for p in pl_module.parameters())
        flat = torch.cat([p.detach().flatten() for p in pl_module.parameters()])
        rank_zero = flat.clone()
        dist.broadcast(rank_zero, src=0)
        torch.testing.assert_close(flat, rank_zero, rtol=0, atol=0)
        result = {
            "rank": dist.get_rank(),
            "backend": dist.get_backend(),
            "world_size": dist.get_world_size(),
            "device": str(pl_module.device),
            "global_step": trainer.global_step,
            "optimizer_steps": sorted({int(value["step"]) for value in state.values()}),
            "batchnorm_counts": counts,
            "sigreg_step": int(pl_module.levljepa_loss.vision_regularizer.step),
            "scheduler_step": trainer.lr_scheduler_configs[0].scheduler.last_epoch,
            "backward_checks": self.backward_checks,
            "rank_parameters_identical": True,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        (
            self.directory / f"step-{self.expected_steps}-rank-{dist.get_rank()}.json"
        ).write_text(json.dumps(result, indent=2))


def _run_nccl_manager_resume(directory: Path) -> None:
    from lightning.pytorch.strategies import DDPStrategy

    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    pl.seed_everything(123)
    spt.set(cache_dir=str(directory / "cache"))
    data = spt.data.DataModule(train=DataLoader(_Pairs(True), batch_size=2))
    optim = {
        "optimizer": {"type": "AdamW", "lr": 1e-3},
        "scheduler": {
            "type": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": 2,
            "max_steps": 6,
            "warmup_start_lr": 1e-5,
            "eta_min": 1e-4,
        },
        "interval": "step",
    }
    checkpoint = directory / "resume.ckpt"
    for steps in (3, 5):
        model = _make_real_encoder_model()
        model.optim = copy.deepcopy(optim)
        before = {k: v.clone() for k, v in model.vision_encoder.state_dict().items()}
        trainer = pl.Trainer(
            max_steps=steps,
            accelerator="gpu",
            devices=2,
            strategy=DDPStrategy(process_group_backend="nccl"),
            gradient_clip_val=1.0,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=[_NCCLTrainingAudit(directory, steps)],
        )
        spt.Manager(
            trainer=trainer,
            module=model,
            data=data,
            seed=123,
            ckpt_path=str(checkpoint) if steps == 5 else None,
            weights_only=False,
        )()
        assert any(
            not torch.equal(before[k], v.cpu())
            for k, v in model.vision_encoder.state_dict().items()
        )
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        assert scheduler.last_epoch == steps and scheduler.max_steps == 6
        if steps == 5:
            expected_lr = 1e-4 + (1e-3 - 1e-4) * (1 + math.cos(3 * math.pi / 4)) / 2
            assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(
                expected_lr
            )
        trainer.save_checkpoint(str(checkpoint))


@pytest.mark.integration
@pytest.mark.gpu
def test_two_gpu_nccl_manager_training_and_full_resume(tmp_path):
    import json
    import subprocess
    import sys

    if not dist.is_nccl_available() or torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA GPUs and NCCL required")
    # torchrun initializes callbacks inside each worker rather than pickling the
    # framework's thread locks across a spawn-based Trainer launcher.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SLURM_")
        and key
        not in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_RANK", "LOCAL_WORLD_SIZE"}
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            "--nccl-manager-resume",
            str(tmp_path),
        ],
        env=env,
        check=True,
        timeout=600,
        cwd=tmp_path,
    )
    records = [
        json.loads(path.read_text()) for path in tmp_path.glob("step-*-rank-*.json")
    ]
    assert {(r["global_step"], r["rank"]) for r in records} == {
        (s, r) for s in (3, 5) for r in (0, 1)
    }
    assert all(r["rank_parameters_identical"] for r in records)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--nccl-manager-resume":
        _run_nccl_manager_resume(Path(sys.argv[2]).resolve())
