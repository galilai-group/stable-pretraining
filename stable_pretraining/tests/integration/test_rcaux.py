"""Integration test for RC-aux through the stable-pretraining training loop."""

from pathlib import Path
import copy
import os
import subprocess
import sys

import lightning as pl
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt


class _TrajectoryDataset(Dataset):
    """Small deterministic trajectory dataset with no external dependencies."""

    def __init__(self, size: int = 8) -> None:
        generator = torch.Generator().manual_seed(7)
        self.observation = torch.randn(size, 5, 4, generator=generator)
        self.action = torch.randn(size, 4, 2, generator=generator)

    def __len__(self) -> int:
        return len(self.observation)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation": self.observation[index],
            "action": self.action[index],
            "index": torch.tensor(index),
        }


class _TinyWorldModel(nn.Module):
    """Encode trajectories and predict two latent states from a shared context."""

    def __init__(self, embedding_dim: int = 3, context_index: int = 2) -> None:
        super().__init__()
        self.context_index = context_index
        self.encoder = nn.Linear(4, embedding_dim)
        self.transition = nn.Sequential(
            nn.Linear(embedding_dim + 2, 8),
            nn.Tanh(),
            nn.Linear(8, embedding_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(batch["observation"])
        state = encoded[:, self.context_index - 1]
        rollout = []
        for offset in range(2):
            action = batch["action"][:, self.context_index - 1 + offset]
            state = self.transition(torch.cat((state, action), dim=-1))
            rollout.append(state)
        return {"embedding": encoded, "prediction": torch.stack(rollout, dim=1)}


def _module() -> spt.Module:
    return spt.RCaux(
        _TinyWorldModel(),
        embedding_dim=3,
        max_horizon=3,
        context_index=2,
        loss_kwargs={"hidden_dim": 8, "horizon_dim": 2},
        optim={"optimizer": {"type": "AdamW", "lr": 1e-2}},
    )


def _data() -> spt.data.DataModule:
    return spt.data.DataModule(
        train=DataLoader(_TrajectoryDataset(), batch_size=4, shuffle=False)
    )


def _trainer(root: Path, max_steps: int, callbacks=None) -> pl.Trainer:
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=-1,
        max_steps=max_steps,
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        default_root_dir=str(root),
        callbacks=callbacks or [],
    )


class _ResumeState(pl.Callback):
    """Capture restored state before the first resumed optimizer step."""

    def on_train_start(self, trainer, pl_module):
        self.step = trainer.global_step
        self.weights = {
            k: v.detach().clone() for k, v in pl_module.state_dict().items()
        }
        self.optimizer_steps = [
            v["step"].item() for v in trainer.optimizers[0].state.values()
        ]
        self.optimizer_state = copy.deepcopy(trainer.optimizers[0].state_dict())


@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform == "win32", reason="Manager uses POSIX signals; run on Linux or WSL"
)
def test_rcaux_manager_checkpoint_resume(tmp_path) -> None:
    """Restore exact model and AdamW state, then perform two more updates."""
    pl.seed_everything(7)
    first = _module()
    initial = {k: v.detach().clone() for k, v in first.state_dict().items()}
    trainer = _trainer(tmp_path / "first", 4)
    spt.Manager(trainer=trainer, module=first, data=_data(), seed=7)()
    checkpoint = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved = {k: v.detach().clone() for k, v in first.state_dict().items()}
    for prefix in [
        "world_model.encoder.",
        "world_model.transition.",
        "rcaux_loss.reachability_head.",
    ]:
        assert any(
            not torch.equal(initial[k], v)
            for k, v in saved.items()
            if k.startswith(prefix)
        )

    restored = _ResumeState()
    resumed = _module()
    resumed_trainer = _trainer(tmp_path / "resumed", 6, [restored])
    spt.Manager(
        trainer=resumed_trainer,
        module=resumed,
        data=_data(),
        seed=7,
        ckpt_path=str(checkpoint.resolve()),
        weights_only=False,
    )()
    assert restored.step == 4
    assert restored.optimizer_steps and all(
        step == 4 for step in restored.optimizer_steps
    )
    for name, value in saved.items():
        torch.testing.assert_close(restored.weights[name], value, rtol=0, atol=0)
    expected_optimizer = trainer.optimizers[0].state_dict()
    assert (
        restored.optimizer_state["param_groups"] == expected_optimizer["param_groups"]
    )
    for parameter_id, state in expected_optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                restored.optimizer_state["state"][parameter_id][name],
                value,
                rtol=0,
                atol=0,
            )
    assert resumed_trainer.global_step == 6

    assert any(not torch.equal(saved[k], v) for k, v in resumed.state_dict().items())
    assert all(
        state["step"].item() == 6
        for state in resumed_trainer.optimizers[0].state.values()
    )


class _DistributedEvidence(pl.Callback):
    """Persist each real process's local batches, losses, and synchronized weights."""

    def __init__(self, output: str):
        self.output = output
        self.rows = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        assert dist.is_initialized() and dist.get_world_size() == 2
        assert torch.isfinite(outputs["loss"])
        # T=5, H=3: 54 local temporal pairs + 40 local cross-trajectory pairs.
        assert outputs["encoded_pair_count"].item() == 94
        assert outputs["predicted_pair_count"].item() == 12
        self.rows.append(
            {"indices": batch["index"].tolist(), "loss": outputs["loss"].item()}
        )

    def on_train_end(self, trainer, pl_module):
        parameters = torch.cat([p.detach().flatten() for p in pl_module.parameters()])
        peers = [torch.zeros_like(parameters) for _ in range(2)]
        dist.all_gather(peers, parameters)
        torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
        torch.save(
            {
                "rows": self.rows,
                "parameters": parameters.cpu(),
                "global_step": trainer.global_step,
                "rank": dist.get_rank(),
            },
            Path(self.output) / f"rank{dist.get_rank()}.pt",
        )


def _run_ddp(output: str) -> None:
    torch.set_num_threads(1)
    pl.seed_everything(11)
    spt.set(cache_dir=str(Path(output) / "cache"))
    module = _module()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(process_group_backend="gloo"),
        max_steps=3,
        max_epochs=2,
        use_distributed_sampler=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        default_root_dir=output,
        callbacks=[_DistributedEvidence(output)],
    )
    data = spt.data.DataModule(
        train=DataLoader(_TrajectoryDataset(), batch_size=2, shuffle=False)
    )
    spt.Manager(trainer=trainer, module=module, data=data, seed=11)()


@pytest.mark.integration
@pytest.mark.distributed
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo distributed backend is required",
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="Manager uses POSIX signals; run on Linux or WSL"
)
def test_rcaux_two_process_ddp_uses_local_negatives(tmp_path) -> None:
    """Launch two actual CPU ranks with torchrun and inspect both results."""
    pl.seed_everything(11)
    initial = torch.cat([p.detach().flatten() for p in _module().parameters()])
    env = dict(os.environ, OMP_NUM_THREADS="1", LOGURU_LEVEL="ERROR")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            "--ddp-worker",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ranks = [torch.load(tmp_path / f"rank{i}.pt", weights_only=True) for i in range(2)]
    assert [r["rank"] for r in ranks] == [0, 1]
    assert all(r["global_step"] == 3 for r in ranks)
    assert not torch.equal(initial, ranks[0]["parameters"])
    torch.testing.assert_close(
        ranks[0]["parameters"], ranks[1]["parameters"], rtol=0, atol=0
    )
    for left, right in zip(ranks[0]["rows"], ranks[1]["rows"], strict=True):
        assert set(left["indices"]).isdisjoint(right["indices"])
    assert ranks[0]["rows"][0]["loss"] != ranks[1]["rows"][0]["loss"]


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--ddp-worker":
        raise SystemExit("Use --ddp-worker OUTPUT_DIRECTORY")
    _run_ddp(sys.argv[2])
