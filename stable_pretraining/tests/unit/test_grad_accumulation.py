"""Regression tests for gradient-accumulation equivalence.

Invariant: with seeded data in a fixed order and a mean-reduction loss, running
with batch ``B/N`` and accumulation factor ``N`` must produce the EXACT same
end-of-epoch parameters as batch ``B`` with no accumulation — the accumulated
gradient is averaged by ``1/N`` so each optimizer step matches a single
full-(effective-)batch update.

``Module.training_step`` averages each optimizer's gradients by its stepping
frequency before the step (see the ``freq > 1`` block). Accumulation can be
requested two equivalent ways, both funnelling to ``_optimizer_frequencies``:
  * ``Trainer(accumulate_grad_batches=N)``  (mapped to frequency N in
    ``on_train_start``; that mapping is covered by
    ``test_manual_optimization_trainer_args.py``), or
  * per-optimizer ``"frequency": N`` in the ``optim`` config (used here, so the
    tests need only a bare ``Trainer`` and stay fast).

Regression for the bug where ``rescale_loss_for_grad_acc`` was dead code and the
loss was back-propagated unscaled, making an accum-N run take ~N× too-large
steps.
"""

import lightning as pl
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt


class _RegressionDS(Dataset):
    """Fixed, seeded (x, y) regression dataset — deterministic sample order."""

    def __init__(self, n: int = 8, in_dim: int = 4, out_dim: int = 4):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, in_dim, generator=g)
        self.y = torch.randn(n, out_dim, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


def _single_fwd(self, batch, stage):
    pred = self.backbone(batch["x"])
    return {"loss": (pred - batch["y"]).pow(2).mean()}


def _multi_fwd(self, batch, stage):
    pred = self.projector(self.backbone(batch["x"]))
    return {"loss": (pred - batch["y"]).pow(2).mean()}


def _trainer():
    return pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


def _params(module):
    return [p.detach().clone() for p in module.parameters()]


def _assert_equal(a, b):
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        assert torch.allclose(pa, pb, atol=1e-6), (pa - pb).abs().max().item()


@pytest.mark.unit
class TestGradAccumulationEquivalence:
    """(batch=B/N, accum=N) must equal (batch=B, accum=1) with seeded data."""

    def _run_single(self, batch_size, frequency, opt_kwargs, n=8):
        torch.manual_seed(123)  # identical init across runs
        module = spt.Module(
            forward=_single_fwd,
            hparams={},
            backbone=nn.Linear(4, 4),
            optim={
                "optimizer": {"type": "SGD", "lr": 0.1, **opt_kwargs},
                "scheduler": {"type": "ConstantLR"},
                "interval": "step",
                "frequency": frequency,
            },
        )
        loader = DataLoader(_RegressionDS(n=n), batch_size=batch_size, shuffle=False)
        _trainer().fit(module, loader)
        return _params(module.backbone)

    def test_single_optimizer_sgd(self):
        """SGD, one optimizer: half-batch + accum-2 == full-batch."""
        full = self._run_single(batch_size=8, frequency=1, opt_kwargs={})
        accum = self._run_single(batch_size=4, frequency=2, opt_kwargs={})
        _assert_equal(full, accum)

    def test_single_optimizer_momentum_multistep(self):
        """SGD+momentum across multiple steps: stateful optimizer still matches.

        16 samples => full: 2 batches/2 steps; accum: 4 batches/2 steps. The
        momentum buffer must evolve identically, which only holds if each step's
        gradient equals the corresponding full-batch gradient.
        """
        kw = {"momentum": 0.9}
        full = self._run_single(batch_size=8, frequency=1, opt_kwargs=kw, n=16)
        accum = self._run_single(batch_size=4, frequency=2, opt_kwargs=kw, n=16)
        _assert_equal(full, accum)

    def test_accum_4_matches_full(self):
        """Higher accumulation factor (4) over an 8-sample epoch."""
        full = self._run_single(batch_size=8, frequency=1, opt_kwargs={})
        accum = self._run_single(batch_size=2, frequency=4, opt_kwargs={})
        _assert_equal(full, accum)

    def _run_multi(self, batch_size, frequency, n=8):
        torch.manual_seed(321)  # identical init across runs
        module = spt.Module(
            forward=_multi_fwd,
            hparams={},
            backbone=nn.Linear(4, 8),
            projector=nn.Linear(8, 4),
            optim={
                "enc": {
                    "modules": r"^backbone(\.|$)",
                    "optimizer": {"type": "SGD", "lr": 0.1},
                    "scheduler": {"type": "ConstantLR"},
                    "interval": "step",
                    "frequency": frequency,
                },
                "head": {
                    "modules": r"^projector(\.|$)",
                    "optimizer": {"type": "SGD", "lr": 0.05},
                    "scheduler": {"type": "ConstantLR"},
                    "interval": "step",
                    "frequency": frequency,
                },
            },
        )
        loader = DataLoader(_RegressionDS(n=n), batch_size=batch_size, shuffle=False)
        _trainer().fit(module, loader)
        # both optimizers' params (backbone + projector)
        return _params(module.backbone) + _params(module.projector)

    def test_multi_optimizer_uniform_frequency(self):
        """Two optimizers (disjoint params), both accum-2: matches full-batch.

        Gradients are averaged per-optimizer, so each optimizer's params match
        their own full-(effective-)batch update.
        """
        full = self._run_multi(batch_size=8, frequency=1)
        accum = self._run_multi(batch_size=4, frequency=2)
        _assert_equal(full, accum)
