"""Real two-rank SimCo gradients and updates match a global-batch reference."""

import copy
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from stable_pretraining.losses.simco import SimCoLoss

pytestmark = [
    pytest.mark.integration,
    pytest.mark.distributed,
    pytest.mark.skipif(sys.platform != "linux", reason="Linux/Gloo regression"),
    pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable"),
]


def _global_reference(
    first: torch.Tensor,
    second: torch.Tensor,
    intra_temperature: float,
    inter_temperature: float,
) -> torch.Tensor:
    """Evaluate paper equation (9) directly without production loss helpers."""
    first = F.normalize(first, dim=1)
    second = F.normalize(second, dim=1)
    losses = []
    for query, key in ((first, second), (second, first)):
        scores = query @ key.T
        intra = F.softmax(scores / intra_temperature, dim=1)
        inter = F.softmax(scores / inter_temperature, dim=1)
        ratio = ((1 - inter.diagonal()) / (1 - intra.diagonal())).detach()
        cross_entropy = -F.log_softmax(scores / intra_temperature, dim=1).diagonal()
        losses.append((ratio * cross_entropy).mean())
    return (losses[0] + losses[1]) / 2


def _compare_ddp_to_global_batch(
    rank: int,
    rendezvous: str,
    intra_temperature: float,
    inter_temperature: float,
) -> None:
    """Run real collective backward and DDP reduction on two CPU processes."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        torch.manual_seed(811)
        encoder_projector = nn.Sequential(
            nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 4)
        ).double()
        reference = copy.deepcopy(encoder_projector)
        distributed = DistributedDataParallel(encoder_projector)
        optimizer = torch.optim.SGD(distributed.parameters(), lr=0.03, momentum=0.9)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.03, momentum=0.9
        )
        initial = [parameter.detach().clone() for parameter in reference.parameters()]
        # Rank order deliberately differs from dataset order; positives must
        # follow the same gathered ordering in both views.
        indices = torch.tensor([4, 1, 5, 0, 3, 2]).chunk(2)[rank]
        objective = SimCoLoss(intra_temperature, inter_temperature)

        for step in range(2):
            generator = torch.Generator().manual_seed(313 + step)
            full_views = [
                torch.randn(
                    6, 5, generator=generator, dtype=torch.float64
                ).requires_grad_()
                for _ in range(2)
            ]
            local_views = [
                view.detach()[indices].clone().requires_grad_() for view in full_views
            ]
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)

            reference_projections = reference(torch.cat(full_views)).chunk(2)
            expected = _global_reference(
                *reference_projections, intra_temperature, inter_temperature
            )
            projections = distributed(torch.cat(local_views)).chunk(2)
            actual = objective(*projections)
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)

            expected.backward()
            actual.backward()
            for local_view, full_view in zip(local_views, full_views):
                assert local_view.grad is not None and full_view.grad is not None
                assert local_view.grad.norm() > 0
                # Every rank consumes the entire global loss. Autograd gather
                # sums those copies; DDP averages parameter gradients below.
                torch.testing.assert_close(
                    local_view.grad,
                    full_view.grad[indices] * dist.get_world_size(),
                    rtol=1e-8,
                    atol=1e-10,
                )

            for (name, parameter), (_, expected_parameter) in zip(
                distributed.module.named_parameters(), reference.named_parameters()
            ):
                assert (
                    parameter.grad is not None and expected_parameter.grad is not None
                )
                torch.testing.assert_close(
                    parameter.grad,
                    expected_parameter.grad,
                    rtol=1e-8,
                    atol=1e-10,
                    msg=f"Rank {rank}, step {step}: DDP gradient for {name}",
                )

            optimizer.step()
            reference_optimizer.step()
            for parameter, expected_parameter in zip(
                distributed.module.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(
                    parameter, expected_parameter, rtol=1e-8, atol=1e-10
                )
                torch.testing.assert_close(
                    optimizer.state[parameter]["momentum_buffer"],
                    reference_optimizer.state[expected_parameter]["momentum_buffer"],
                    rtol=1e-8,
                    atol=1e-10,
                )
        assert any(
            not torch.equal(before, after)
            for before, after in zip(initial, distributed.module.parameters())
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("intra_temperature", "inter_temperature"), [(0.1, 1.0), (0.3, 0.3)]
)
def test_two_rank_ddp_matches_global_loss_gradients_and_updates(
    tmp_path: Path, intra_temperature: float, inter_temperature: float
) -> None:
    """Check gather order, both-view gradients, DDP scaling, and SGD momentum."""
    torch.multiprocessing.spawn(
        _compare_ddp_to_global_batch,
        args=(
            (tmp_path / "simco-rendezvous").as_uri(),
            intra_temperature,
            inter_temperature,
        ),
        nprocs=2,
        join=True,
    )


def test_two_rank_example_trains_validates_and_saves_shared_artifacts(
    tmp_path: Path,
) -> None:
    """Exercise the actual Manager entry point under a two-process launcher."""
    repository = Path(__file__).resolve().parents[3]
    output_dir = tmp_path / "distributed-example"
    log_path = tmp_path / "torchrun.log"
    rank_keys = {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "NODE_RANK",
        "JSM_NAMESPACE_RANK",
        "PL_IN_DDP_SUBPROCESS",
    }
    # A test invoked inside an allocation must give torchrun control of the
    # new ranks instead of inheriting Slurm's existing process topology.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in rank_keys
        and not key.startswith(
            (
                "SLURM_",
                "TORCHELASTIC_",
                "MASTER_",
                "SUBMITIT_",
                "PMI_",
                "PMIX_",
                "OMPI_",
            )
        )
    }
    # CI installs dependencies with pip --user. Keep the parent's import policy
    # so the launcher can import the same packages as the pytest process.
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        str(repository / "examples" / "simco_cifar.py"),
        "--synthetic",
        "--accelerator",
        "cpu",
        "--devices",
        "2",
        "--batch-size",
        "16",
        "--workers",
        "0",
        "--epochs",
        "1",
        "--max-steps",
        "2",
        "--warmup-epochs",
        "0",
        "--output-dir",
        str(output_dir),
    ]
    try:
        with log_path.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command,
                cwd=repository,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=True,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        pytest.fail(
            f"Two-rank example failed: {error}\n{log_path.read_text()[-12000:]}"
        )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert list(output_dir.glob("summary*.json")) == [output_dir / "summary.json"]
    assert len(list(output_dir.rglob("sidecar.json"))) == 1
    assert summary["synthetic_wiring_check"] is True
    assert summary["optimizer_steps"] == 2
    assert summary["distributed"] == {
        "world_size": 2,
        "strategy": "DDPStrategy",
        "global_batch_size": 16,
        "per_device_batch_size": 8,
        "sync_batchnorm": False,
    }
    configuration = json.loads((output_dir / "config.json").read_text())
    training = configuration["protocol"]["training"]
    assert training["world_size"] == 2
    assert training["strategy"] == "ddp"
    assert training["global_batch_size"] == 16
    assert training["per_device_batch_size"] == 8
    assert training["validation_sampler_padding"] is False
    assert Path(summary["best_checkpoint"]).is_file()
    checkpoint = torch.load(
        summary["last_checkpoint"], map_location="cpu", weights_only=False
    )
    assert checkpoint["global_step"] == 2
    assert len(checkpoint["optimizer_states"]) == 2
    assert all(state["state"] for state in checkpoint["optimizer_states"])
    assert any(
        name.startswith("callbacks_modules.linear_probe.")
        for name in checkpoint["state_dict"]
    )

    records = []
    records_dir = output_dir / "validation_records"
    for rank in range(2):
        rank_paths = sorted(
            records_dir.glob(f"validation_epoch=0_batch=*_device={rank}.pt")
        )
        assert len(rank_paths) == 2
        records.extend(
            torch.load(path, map_location="cpu", weights_only=True)
            for path in rank_paths
        )
    indices = torch.cat([record["sample_idx"] for record in records])
    torch.testing.assert_close(indices.sort().values, torch.arange(32))
    logits = torch.cat([record["linear_probe_preds"] for record in records])
    labels = torch.cat([record["label"] for record in records])
    assert logits.shape == (32, 10)
    measured_accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    assert summary["metrics"]["eval/linear_probe_top1_accuracy"] == pytest.approx(
        measured_accuracy
    )
