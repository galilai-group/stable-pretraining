"""Natural DAE-Net controller training and new-process completed-epoch resume."""

import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import stable_pretraining as spt

pytestmark = pytest.mark.integration


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected


class _Queries(spt.data.DAENetDataset):
    def __getitem__(self, key):
        batch = super().__getitem__(key)
        if "points" in batch:
            batch["points"] = batch["points"][:11]
            batch["occupancy"] = batch["occupancy"][:11]
            batch["exposure"] = torch.tensor(key)
        return batch


class _Data(spt.data.DataModule):
    def __init__(self, root, world):
        dataset = _Queries(root, root / "train.jsonl", split="train")
        usage = _Queries(root, root / "train.jsonl", split="train", load_queries=False)
        sampler = spt.data.DAENetExposureSampler(
            dataset, shapes_per_epoch=len(dataset), seed=91
        )
        batch_size = max(3, world + 1)
        batches = spt.data.DAENetDistributedBatchSampler(
            sampler, batch_size, world_size=world
        )
        train = DataLoader(
            dataset, batch_sampler=batches, generator=torch.Generator().manual_seed(91)
        )
        super().__init__(train=train)
        self.usage_loader = DataLoader(
            usage, batch_size=3, generator=torch.Generator().manual_seed(92)
        )

    def usage_dataloader(self):
        return self.usage_loader


class _Trace(pl.Callback):
    def __init__(self, output):
        self.steps = []
        self.output = output

    def on_train_start(self, trainer, pl_module):
        # A fresh rank gets distinct streams; controller restoration runs after
        # this hook on resume, so it must recover each stream from the checkpoint.
        torch.manual_seed(711 + trainer.global_rank)
        random.seed(811 + trainer.global_rank)
        np.random.seed(911 + trainer.global_rank)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        assert torch.isfinite(outputs["loss"])
        self.steps.append(
            _snapshot(
                {
                    "epoch": trainer.current_epoch,
                    "batch": batch_idx,
                    "exposure": batch["exposure"],
                    "loss": outputs["loss"],
                    "network": pl_module.backbone.state_dict(),
                    "optimizer": trainer.optimizers[0].state_dict(),
                    "scheduler": trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                    "draws": (torch.rand(3), random.random(), np.random.random()),
                }
            )
        )
        if trainer.current_epoch == 2 and batch_idx + 1 == trainer.num_training_batches:
            trainer.save_checkpoint(self.output / "before-epoch-end.ckpt")


def _worker(rank, world, root, output, epochs, checkpoint):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root, output = Path(root), Path(output)
    if world > 1:
        os.environ.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE=str(world),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29581",
        )
        dist.init_process_group(
            "gloo",
            init_method=(output / "rendezvous").as_uri(),
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=60),
        )
    try:
        spt.set(cache_dir=str(output), requeue_checkpoint=False)
        from stable_pretraining._config import get_config

        get_config()._default_callbacks = dict(
            env_dump=False, progress_bar=False, trainer_info=False, module_summary=False
        )
        pl.seed_everything(17)
        backbone = spt.backbone.DAENetBackbone(
            branch_num=3, z_dim=4, ef_dim=2, df_dim=6, gf_dim=8
        )
        with torch.no_grad():
            backbone.encoder.linear_1.bias.copy_(torch.tensor([8.0, 8.5, 9.0]))
        model = spt.DAENet(backbone=backbone, era_length=2)
        before = _snapshot(backbone.state_dict())
        trace = _Trace(output)
        saver = ModelCheckpoint(
            dirpath=output / "checkpoints",
            filename="epoch{epoch}",
            auto_insert_metric_name=False,
            save_top_k=-1,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=world,
            strategy="ddp_find_unused_parameters_true" if world > 1 else "auto",
            precision="32-true",
            use_distributed_sampler=False,
            max_epochs=epochs,
            callbacks=[trace, saver],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            default_root_dir=str(output),
        )
        spt.Manager(
            trainer=trainer,
            module=model,
            data=_Data(root, world),
            seed=17,
            ckpt_path=checkpoint,
            weights_only=False,
        )()
        assert any(
            not torch.equal(before[k], v) for k, v in backbone.state_dict().items()
        )
        assert model._era_controller.branch_ages is not None
        torch.save(
            {
                "steps": trace.steps,
                "controller": model._era_controller.state_dict(),
                "checkpoint_dir": saver.dirpath,
            },
            output / f"rank{rank}.pt",
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(world, root, output, epochs, checkpoint=None):
    output.mkdir()
    context = mp.spawn(
        _worker,
        args=(world, str(root), str(output), epochs, checkpoint),
        nprocs=world,
        join=False,
    )
    deadline = time.monotonic() + 240
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("DAE-Net Manager processes exceeded 240 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
    return [
        torch.load(output / f"rank{rank}.pt", weights_only=False)
        for rank in range(world)
    ]


@pytest.mark.slow
@pytest.mark.parametrize(
    "world", [1, *[pytest.param(n, marks=pytest.mark.distributed) for n in (2, 3, 4)]]
)
def test_natural_manager_resume_with_new_processes(tmp_path, world):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(19)
    shapes = max(5, 2 * world + 1)
    with h5py.File(tmp_path / "shapes.hdf5", "w") as source:
        source["voxels"] = rng.integers(0, 2, (shapes, 64, 64, 64, 1), dtype=np.uint8)
        source["points_64"] = rng.integers(0, 64, (shapes, 32768, 3), dtype=np.uint8)
        source["values_64"] = rng.integers(0, 2, (shapes, 32768, 1), dtype=np.uint8)
    (tmp_path / "train.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "shape_id": f"shape-{i}",
                    "source_hdf5": "shapes.hdf5",
                    "row_index": i,
                    "clean_split": "train",
                    "query_resolution": 64,
                }
            )
            + "\n"
            for i in range(shapes)
        )
    )
    baseline = _run(world, tmp_path, tmp_path / "continuous", 6)
    premature = torch.load(
        tmp_path / "continuous/before-epoch-end.ckpt", weights_only=False
    )
    assert premature["global_step"] == 6
    for legacy in (False, True):
        if legacy:
            premature["callbacks"]["DAENetRevival"].pop("completed_epoch")
        with pytest.raises(ValueError, match="completed-epoch"):
            spt.callbacks.DAENetRevival.validate_checkpoint(premature)
    if world > 1:
        states = baseline[0]["controller"]["rank_rng_states"]
        assert len(states) == world
        for rank in range(1, world):
            _assert_equal(baseline[0]["controller"], baseline[rank]["controller"])
            for left, right in zip(baseline[0]["steps"], baseline[rank]["steps"]):
                _assert_equal(left["network"], right["network"])
                _assert_equal(left["optimizer"], right["optimizer"])
            assert not torch.equal(
                states[0]["torch_rng_state"], states[rank]["torch_rng_state"]
            )
    for stop in (3, 4):
        cut = tmp_path / f"cut{stop}"
        interrupted = _run(world, tmp_path, cut, stop)
        checkpoint = Path(interrupted[0]["checkpoint_dir"]) / f"epoch{stop - 1}.ckpt"
        saved = torch.load(checkpoint, weights_only=False)
        spt.callbacks.DAENetRevival.validate_checkpoint(saved)
        saved["callbacks"]["DAENetRevival"].pop("completed_epoch")
        spt.callbacks.DAENetRevival.validate_checkpoint(saved)
        assert saved["callbacks"]["DAENetRevival"]["accepted_epoch"] is not None
        resumed = _run(world, tmp_path, tmp_path / f"resume{stop}", 6, str(checkpoint))
        for rank in range(world):
            _assert_equal(
                resumed[rank]["steps"],
                [s for s in baseline[rank]["steps"] if s["epoch"] >= stop],
            )
            _assert_equal(resumed[rank]["controller"], baseline[rank]["controller"])


def test_validation_loads_weights_from_incomplete_epoch(tmp_path, training_environment):
    backbone = spt.backbone.DAENetBackbone(
        branch_num=2, z_dim=4, ef_dim=2, df_dim=6, gf_dim=8
    )
    model = spt.DAENet(backbone=backbone, era_length=2)
    checkpoint = tmp_path / "incomplete.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "pytorch-lightning_version": pl.__version__,
            "epoch": 0,
            "global_step": 1,
            "callbacks": {"DAENetRevival": model._era_controller.state_dict()},
        },
        checkpoint,
    )
    batch = {
        "voxels": torch.zeros(1, 1, 64, 64, 64),
        "points": torch.zeros(1, 7, 3),
        "occupancy": torch.ones(1, 7, 1),
    }
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    results = trainer.validate(
        model,
        dataloaders=DataLoader([batch], batch_size=None),
        ckpt_path=checkpoint,
        weights_only=False,
        verbose=False,
    )
    assert np.isfinite(results[0]["validate/loss"])


@pytest.fixture
def training_environment(monkeypatch):
    from stable_pretraining._config import get_config

    config = get_config()
    monkeypatch.setattr(config, "_requeue_checkpoint", False)
    monkeypatch.setattr(
        config,
        "_default_callbacks",
        {
            "env_dump": False,
            "progress_bar": False,
            "trainer_info": False,
            "module_summary": False,
        },
    )
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng():
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
