"""Two-process CPU/Gloo integration coverage for TS2Vec's local-batch training."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.integration
@pytest.mark.distributed
def test_ts2vec_cpu_ddp_training_and_local_instance_negatives(tmp_path: Path) -> None:
    """Run real two-rank DDP forward/backward/step and verify both encoders."""
    repository_root = Path(__file__).resolve().parents[3]
    result_dir = tmp_path / "rank-results"
    result_dir.mkdir()
    env = os.environ.copy()
    env["TS2VEC_DDP_RESULT_DIR"] = str(result_dir)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["GLOO_SOCKET_IFNAME"] = "lo"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=2",
        "--node_rank=0",
        "--master_addr=127.0.0.1",
        f"--master_port={master_port}",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    completed = subprocess.run(
        command,
        cwd=repository_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, (
        "two-process TS2Vec DDP worker failed\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )

    results = [
        torch.load(result_dir / f"rank-{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    for result in results:
        assert result["n_averaged"] == 4
        assert result["finite_grad"]
        assert result["nonzero_grad"]
        assert result["backbone_changed"]
    for state_name in ("backbone", "averaged_backbone"):
        for key, value in results[0][state_name].items():
            torch.testing.assert_close(
                value, results[1][state_name][key], rtol=0, atol=0
            )

    assert results[0]["local_loss"] != pytest.approx(results[0]["global_loss"])
    assert results[1]["local_loss"] != pytest.approx(results[1]["global_loss"])


def _ddp_worker() -> None:
    """Run synchronized TS2Vec optimizer steps on each Gloo rank."""
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    from torch.nn.parallel import DistributedDataParallel
    from torch.optim import AdamW

    from stable_pretraining.callbacks.ts2vec import TS2VecAveraging
    from stable_pretraining.losses.ts2vec import TS2VecLoss
    from stable_pretraining.methods.ts2vec import TS2Vec

    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=30))
    rank = dist.get_rank()
    torch.set_num_threads(1)
    torch.manual_seed(2718 + rank)
    model = TS2Vec(
        input_dims=2,
        output_dims=4,
        hidden_dims=4,
        depth=1,
        alpha=0.5,
        temporal_unit=0,
    )
    ddp_model = DistributedDataParallel(model)
    optimizer = AdamW(model.backbone.parameters(), lr=0.003, weight_decay=0.0)
    averaging = TS2VecAveraging()
    averaging.on_fit_start(SimpleNamespace(optimizers=[optimizer]), model)
    initial = {
        name: value.detach().clone()
        for name, value in model.backbone.state_dict().items()
    }
    snapshots = [initial]

    # Deliberately distinct rank data ensures DDP averages gradients from both ranks.
    finite_grad = True
    nonzero_grad = False
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(100 + rank * 10 + step)
        batch = {"series": torch.randn(2, 8, 2, generator=generator)}
        output = ddp_model(batch, stage="train")
        loss = output["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
        assert grads
        finite_grad &= all(bool(torch.isfinite(grad).all()) for grad in grads)
        nonzero_grad |= any(bool(torch.count_nonzero(grad)) for grad in grads)
        optimizer.step()
        snapshots.append(
            {
                name: value.detach().clone()
                for name, value in model.backbone.state_dict().items()
            }
        )
    averaging.on_fit_end(None, model)
    assert finite_grad and nonzero_grad

    local_changed = any(
        not torch.equal(value, initial[name])
        for name, value in model.backbone.state_dict().items()
    )
    local_embeddings = _instance_embeddings(rank)
    gathered_embeddings = [torch.empty_like(local_embeddings) for _ in range(2)]
    dist.all_gather(gathered_embeddings, local_embeddings)
    instance_loss = TS2VecLoss(alpha=1.0)
    local_first, local_second = local_embeddings[:2], local_embeddings[2:]
    local_loss = instance_loss(local_first, local_second).item()
    global_first = torch.cat([embedding[:2] for embedding in gathered_embeddings])
    global_second = torch.cat([embedding[2:] for embedding in gathered_embeddings])
    local_oracle = _instance_cross_entropy_oracle(local_first, local_second)
    global_oracle = _instance_cross_entropy_oracle(global_first, global_second)
    torch.testing.assert_close(torch.tensor(local_loss), torch.tensor(local_oracle))
    assert local_oracle != pytest.approx(global_oracle)

    expected_average = {
        name: torch.stack([snapshot[name] for snapshot in snapshots]).mean(dim=0)
        for name in initial
    }
    for name, expected in expected_average.items():
        torch.testing.assert_close(
            model.averaged_backbone.module.state_dict()[name],
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    dist.barrier()
    result = {
        "backbone": {
            k: v.detach().cpu() for k, v in model.backbone.state_dict().items()
        },
        "averaged_backbone": {
            k: v.detach().cpu()
            for k, v in model.averaged_backbone.module.state_dict().items()
        },
        "n_averaged": int(model.averaged_backbone.n_averaged.item()),
        "finite_grad": finite_grad,
        "nonzero_grad": nonzero_grad,
        "backbone_changed": local_changed,
        "local_loss": local_oracle,
        "global_loss": global_oracle,
    }
    torch.save(result, Path(os.environ["TS2VEC_DDP_RESULT_DIR"]) / f"rank-{rank}.pt")
    dist.destroy_process_group()


def _instance_embeddings(rank: int):
    """Deterministic paired embeddings whose negatives differ across ranks."""
    import torch

    # Two aligned views of two instances, with rank-dependent directions.
    base = torch.tensor(
        [
            [1.0 + rank, 0.2],
            [0.1, 1.0 + rank],
            [0.8 + rank, 0.3],
            [0.2, 0.7 + rank],
        ]
    )
    return base.unsqueeze(1)


def _instance_cross_entropy_oracle(z1, z2):
    """Compute paired instance InfoNCE directly, without TS2Vec loss helpers."""
    import torch.nn.functional as F

    batch = z1.shape[0]
    features = torch.cat((z1[:, 0], z2[:, 0]), dim=0)
    logits = features @ features.T
    logits.fill_diagonal_(float("-inf"))
    targets = (torch.arange(2 * batch) + batch) % (2 * batch)
    return F.cross_entropy(logits, targets).item()


if __name__ == "__main__" and "--worker" in sys.argv:
    _ddp_worker()
