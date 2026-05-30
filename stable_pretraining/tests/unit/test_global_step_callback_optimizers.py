"""Regression: callback-owned optimizers must not advance ``trainer.global_step``.

Without the dispatch fix in ``Module.training_step``, every
``LightningOptimizer.step()`` call ticks Lightning's manual-opt step counter
once. With N ``TrainableCallback``-style callbacks (e.g. ``OnlineProbe``)
attached, that meant ``global_step`` advanced by N+1 per actual training
batch — silently breaking ``Trainer(max_steps=...)``, step-based
schedulers, and step-keyed logging.

This test fits a tiny Lightning ``Trainer`` with several callback
optimizers and asserts that ``trainer.global_step`` after training equals
exactly the number of training batches executed.
"""

import lightning as pl
import pytest
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset

import stable_pretraining as spt
from stable_pretraining import callbacks


def _make_module_and_data(num_probes: int):
    """Build a Module with N OnlineProbe callbacks attached."""

    def fwd(self, batch, stage):
        x = batch["image"]
        emb = self.backbone(x)
        out = {"embedding": emb, "label": batch["label"]}
        if stage == "fit":
            # Tiny SSL-style loss so manual_backward has something to do.
            out["loss"] = emb.pow(2).mean()
        return out

    backbone = nn.Linear(4, 8)
    module = spt.Module(
        forward=fwd,
        hparams={},
        backbone=backbone,
        optim={
            "optimizer": {"type": "AdamW", "lr": 1e-3},
            "scheduler": {"type": "ConstantLR"},
            "interval": "step",
        },
    )

    # Attach N OnlineProbe callbacks (each adds its own optimizer + scheduler).
    cb = [
        callbacks.OnlineProbe(
            module,
            name=f"probe_{i}",
            input="embedding",
            target="label",
            probe=nn.Linear(8, 2),
            loss=nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(num_classes=2)
            },
            optimizer={"type": "AdamW", "lr": 1e-3},
        )
        for i in range(num_probes)
    ]

    # Fake batch tiny enough to run on CPU in seconds.
    N = 16
    ds = TensorDataset(
        torch.randn(N, 4),
        torch.randint(0, 2, (N,)),
    )

    class _DictWrap(torch.utils.data.Dataset):
        def __init__(self, inner):
            self.inner = inner

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, i):
            x, y = self.inner[i]
            return {"image": x, "label": y}

    loader = DataLoader(_DictWrap(ds), batch_size=4, shuffle=False)

    return module, loader, cb


class _BatchCounter(pl.Callback):
    """Counts how many ``on_train_batch_end`` events fire (= batches actually trained)."""

    def __init__(self):
        super().__init__()
        self.batches = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batches += 1


@pytest.mark.unit
class TestCallbackOptimizersDoNotAdvanceGlobalStep:
    """``max_steps=N`` means N *training batches* regardless of callback count.

    Without the dispatch fix, with N callback optimizers attached, each batch
    fired ``opt.step()`` N+1 times — each ticking ``trainer.global_step`` —
    so Lightning stopped at ``max_steps`` after only ``max_steps / (N+1)``
    actual training batches. The test catches that regression by comparing
    the trainer's ``global_step`` to the number of ``on_train_batch_end``
    fires.
    """

    @pytest.mark.parametrize("num_probes", [0, 1, 3, 5])
    def test_batches_match_global_step(self, num_probes):
        max_steps = 8
        module, loader, cb = _make_module_and_data(num_probes=num_probes)
        counter = _BatchCounter()
        trainer = pl.Trainer(
            max_steps=max_steps,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu",
            devices=1,
            callbacks=[*cb, counter],
        )
        trainer.fit(module, loader)

        assert counter.batches == max_steps, (
            f"Trainer stopped after only {counter.batches} training batches "
            f"with {num_probes} probes attached (expected {max_steps}). "
            f"trainer.global_step={trainer.global_step}. Each callback "
            f"optimizer's LightningOptimizer.step() advances global_step, "
            f"so max_steps is hit after only max_steps/(num_probes+1) batches."
        )
        assert trainer.global_step == max_steps

    def test_zero_optimizers_eval_only_does_not_crash(self):
        """Zero optimizers: ``Module(optim=None)`` + no callbacks fits cleanly.

        Pure forward-pass fit (e.g. dry-run for shape debugging or eval
        coverage in fit) must complete without errors.
        Lightning returns a ``_MockOptimizer`` from
        ``self.optimizers()`` when ``configure_optimizers`` yields
        nothing, and our ``training_step`` returns early in that case.
        ``global_step`` stays at 0 (which is correct — nothing
        optimized), but ``trainer.current_epoch`` still advances, so
        ``max_epochs`` is the meaningful budget here.
        """
        from stable_pretraining.module import Module
        from stable_pretraining.data import DataModule

        def fwd(self, batch, stage):
            return {"embedding": self.backbone(batch["image"])}

        backbone = nn.Linear(4, 8)
        module = Module(forward=fwd, hparams={}, backbone=backbone, optim=None)

        class _DictWrap(torch.utils.data.Dataset):
            def __init__(self, n=8):
                self.x = torch.randn(n, 4)

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                return {"image": self.x[i]}

        loader = DataLoader(_DictWrap(), batch_size=4, shuffle=False)
        dm = DataModule(train=loader, val=loader)
        counter = _BatchCounter()
        trainer = pl.Trainer(
            max_epochs=2,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu",
            devices=1,
            callbacks=[counter],
        )
        # The key assertion: this finishes (no infinite loop, no error).
        # 2 epochs × 2 batches = 4 forward passes.
        trainer.fit(module, dm)
        assert counter.batches == 4, (
            f"zero-optimizer fit ran {counter.batches} batches, expected 4."
        )
        # global_step stays at 0 because no optimizer ever stepped.
        assert trainer.global_step == 0
        # But the epoch counter does advance.
        assert trainer.current_epoch == 2

    def test_no_main_optimizer_still_ticks_global_step(self):
        """``optim=None`` + callback optimizers still ticks ``global_step``.

        Otherwise ``Trainer(max_steps=...)`` is unreachable and ``fit``
        loops forever.
        Regression for the ``test_image_decoder.py`` infinite-loop bug
        the user spotted ("Epoch 18533/-1 done in 0.0s") after the
        callback-optimizer-counter fix: with no main optimizer there
        was no remaining optimizer to tick ``global_step``. The
        ``_pick_global_step_ticker`` fallback promotes the first
        callback optimizer to ticker in that case.
        """
        from stable_pretraining.module import Module
        from stable_pretraining.data import DataModule

        def fwd(self, batch, stage):
            emb = self.backbone(batch["image"])
            out = {"embedding": emb, "label": batch["label"]}
            if stage == "fit":
                out["loss"] = emb.pow(2).mean()
            return out

        # Module with NO main optimizer — only the callback's will be present.
        backbone = nn.Linear(4, 8)
        module = Module(forward=fwd, hparams={}, backbone=backbone, optim=None)
        cb = callbacks.OnlineProbe(
            module,
            name="probe",
            input="embedding",
            target="label",
            probe=nn.Linear(8, 2),
            loss=nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(num_classes=2)
            },
            optimizer={"type": "AdamW", "lr": 1e-3},
        )

        class _DictWrap(torch.utils.data.Dataset):
            def __init__(self, n=16):
                self.x = torch.randn(n, 4)
                self.y = torch.randint(0, 2, (n,))

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                return {"image": self.x[i], "label": self.y[i]}

        loader = DataLoader(_DictWrap(), batch_size=4, shuffle=False)
        dm = DataModule(train=loader, val=loader)

        max_steps = 3
        counter = _BatchCounter()
        trainer = pl.Trainer(
            max_steps=max_steps,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu",
            devices=1,
            callbacks=[cb, counter],
        )
        trainer.fit(module, dm)
        assert counter.batches == max_steps, (
            f"no-main-opt path: trainer ran {counter.batches} batches "
            f"(expected {max_steps}); global_step={trainer.global_step}. "
            "The callback optimizer must be promoted to global_step ticker "
            "when no main optimizer is present."
        )
        assert trainer.global_step == max_steps

    @pytest.mark.gpu
    def test_amp_scaler_preserved_for_callback_opts(self):
        """fp16-mixed: GradScaler must still see every callback optimizer step.

        The fix swaps Lightning's progress-counter hooks to no-ops around
        callback ``LightningOptimizer.step()`` calls but keeps the
        ``strategy.optimizer_step()`` path intact — so the AMP
        ``GradScaler.step(opt)`` / ``scaler.update()`` machinery still
        runs for every callback optimizer. This guards against a naive
        bypass that would have called ``opt.optimizer.step()`` directly
        and silently broken fp16 training.
        """
        max_steps = 4
        module, loader, cb = _make_module_and_data(num_probes=3)
        counter = _BatchCounter()
        trainer = pl.Trainer(
            max_steps=max_steps,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="gpu",
            devices=1,
            precision="16-mixed",  # the GradScaler path
            callbacks=[*cb, counter],
        )
        trainer.fit(module, loader)
        assert counter.batches == max_steps, (
            f"fp16 path: trainer stopped after {counter.batches} batches "
            f"(expected {max_steps}); global_step={trainer.global_step}."
        )
        assert trainer.global_step == max_steps
        # GradScaler should still be in a healthy state — get_scale() works
        # and the scale hasn't degenerated to 0 from skipped steps.
        scaler = trainer.precision_plugin.scaler
        if scaler is not None:
            assert scaler.get_scale() > 0
