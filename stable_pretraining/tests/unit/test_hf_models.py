import os

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import multiprocessing as mp  # noqa: E402
from loguru import logger  # noqa: E402
from transformers import PretrainedConfig, PreTrainedModel  # noqa: E402
import lightning.pytorch as pl  # noqa: E402
import stable_pretraining as spt  # noqa: E402

# =============================================================================
# 1. Mock HF Model & Config
# =============================================================================


class SimpleHFConfig(PretrainedConfig):
    """Config for a simple HF model used in tests."""

    model_type = "simple_hf"

    def __init__(self, dim=32, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim


class SimpleHFModel(PreTrainedModel):
    """Simple HF model for testing checkpoint export."""

    config_class = SimpleHFConfig
    base_model_prefix = "simple_hf"

    def __init__(self, config):
        super().__init__(config)
        self.proj = nn.Linear(config.dim, config.dim, bias=False)
        self.post_init()

    def forward(self, x):
        return self.proj(x)


# =============================================================================
# 2. SPT-Native Module (Stage-Aware Forward)
# =============================================================================


class MockSPTSystem(spt.Module):
    """Mock SPT module wrapping a HF backbone for integration testing."""

    def __init__(self, cfg):
        super().__init__()
        self.encoder = spt.backbone.MaskedEncoder(
            model_or_model_name=cfg.get("model_name", "vit_tiny_patch16_224"),
            masking=spt.backbone.PatchMasking(mask_ratio=0.5, block_size=1),
            img_size=(224, 224),
            pretrained=False,
        )
        self.hf_backbone = SimpleHFModel(SimpleHFConfig(dim=self.encoder.embed_dim))

    def forward(self, batch, stage: str = "train"):
        """Run forward pass and return loss dict.

        Receives a dictionary batch and returns a dict with 'loss' if
        stage == 'train'.
        """
        x = batch["image"]

        # Latent extraction
        out = self.encoder(x)
        feat = out.encoded if hasattr(out, "encoded") else out

        # Global Average Pool -> Head
        preds = self.hf_backbone(feat.mean(dim=1))

        if stage == "fit":
            # Target a high value to ensure measurable weight drift for the test
            target = torch.ones_like(preds) * 100.0
            loss = torch.nn.functional.mse_loss(preds, target)
            return {"loss": loss, "preds": preds}

        return preds

    def configure_optimizers(self):
        # Use high LR SGD to guarantee weight shift in exactly 1 step
        return torch.optim.SGD(self.parameters(), lr=10.0)


# =============================================================================
# 3. Process Isolation Worker
# =============================================================================


def check_load_fidelity(model_path, input_tensor, expected_output, q):
    """Execution in a spawned process to ensure Zero-Knowledge loading."""
    import traceback

    try:
        from transformers import AutoModel
        import torch

        # Load via trust_remote_code to use bundled .py files
        loaded = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        with torch.no_grad():
            actual = loaded(input_tensor)

        match = torch.allclose(actual, expected_output, atol=1e-5)
        q.put((True, match))
    except Exception as e:
        q.put((False, f"{e}\n{traceback.format_exc()}"))


# =============================================================================
# 4. Standalone Unit Test
# =============================================================================


@pytest.mark.unit
def test_spt_hf_fidelity_flow(tmp_path):
    test_root = tmp_path / "spt_fidelity_artifacts"
    hf_save_dir = test_root / "hf_exports"
    test_root.mkdir(parents=True)

    # Setup baseline data
    res = 224
    sample_img = torch.ones(1, 3, res, res)  # Ones for stable gradients
    batch = {"image": sample_img.repeat(4, 1, 1, 1), "label": torch.zeros(4)}

    # Initialize Model and capture initial state
    model = MockSPTSystem({"res": res})
    with torch.no_grad():
        latents = model.encoder(sample_img)
        feat = latents.encoded if hasattr(latents, "encoded") else latents
        feat_fixed = feat.mean(dim=1).clone()
        init_out = model.hf_backbone(feat_fixed).clone()

    # Configure Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=str(test_root),
        accelerator="cpu",
        devices=1,
        max_steps=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        enable_checkpointing=True,
        logger=False,
    )

    # Inject/Redirect Callback. The default is now per_step=False (writes
    # to ``last/`` and overwrites). This test asserts against a per-step
    # path (``step_{N}/hf_backbone``), so opt it back into per-step mode.
    for cb in trainer.callbacks:
        if "HuggingFaceCheckpointCallback" in cb.__class__.__name__:
            cb.save_dir = hf_save_dir
            cb.per_step = True

    # DataLoader with dict-based batches
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(batch["image"], batch["label"]),
        batch_size=2,
        # Collate into dict to satisfy spt.Module.forward
        collate_fn=lambda x: {
            "image": torch.stack([i[0] for i in x]),
            "label": torch.stack([i[1] for i in x]),
        },
    )

    logger.info("Executing SPT-Native fit...")
    trainer.fit(model, dl)

    # Explicitly trigger save to ensure on_save_checkpoint executes
    trainer.save_checkpoint(test_root / "manual.ckpt")

    # Verify Weight Drift
    with torch.no_grad():
        trained_out = model.hf_backbone(feat_fixed).clone()

    delta = (trained_out - init_out).abs().sum().item()
    logger.info(f"Weight delta after 1 step: {delta:.4f}")
    assert delta > 0, "Weights failed to update! Check gradient flow in forward()."

    # Verify Export and Isolation
    target = hf_save_dir / f"step_{trainer.global_step}" / "hf_backbone"
    assert target.exists(), "HF Export folder missing."

    logger.info("Testing Zero-Knowledge load in fresh process...")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=check_load_fidelity, args=(str(target), feat_fixed, trained_out, q)
    )
    p.start()

    success, result = q.get(timeout=120)
    p.join()

    if not success:
        pytest.fail(f"Isolation test crashed: {result}")

    assert result is True, (
        "The reloaded model prediction differs from the trained state!"
    )
    logger.success("SPT-Native Fidelity and Zero-Knowledge Load Verified.")


# =============================================================================
# 5. Behavior tests: per_step default, folder creation, error robustness
# =============================================================================


def _make_simple_lightning_module():
    """Minimal LightningModule wrapping a single HF submodule for unit tests."""

    class M(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.hf = SimpleHFModel(SimpleHFConfig(dim=8))

        def forward(self, x):
            return self.hf(x)

        def training_step(self, batch, batch_idx):
            x, _ = batch
            out = self(x)
            return torch.nn.functional.mse_loss(out, torch.zeros_like(out))

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-3)

    return M()


def _trivial_loader():
    x = torch.randn(2, 8)
    y = torch.zeros(2)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=1
    )


@pytest.mark.unit
def test_hf_callback_default_uses_last_subdir(tmp_path):
    """Default (per_step=False) overwrites a single 'last/' subdir.

    It must not accumulate per-step folders.
    """
    from stable_pretraining.callbacks.hf_models import HuggingFaceCheckpointCallback

    save_dir = tmp_path / "hf"
    cb = HuggingFaceCheckpointCallback(save_dir=str(save_dir))
    assert cb.per_step is False, "default must be per_step=False"

    model = _make_simple_lightning_module()
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        enable_checkpointing=True,
        logger=False,
        callbacks=[cb],
    )
    trainer.fit(model, _trivial_loader())
    trainer.save_checkpoint(tmp_path / "manual1.ckpt")
    trainer.save_checkpoint(tmp_path / "manual2.ckpt")

    # Only "last" should exist; no step_N subdirs
    assert (save_dir / "last").exists(), "default save_dir/last must be created"
    step_dirs = [p.name for p in save_dir.iterdir() if p.name.startswith("step_")]
    assert step_dirs == [], (
        f"per_step=False must NOT create step_N subdirs, got {step_dirs}"
    )
    assert (save_dir / "last" / "hf").exists(), "submodule subdir under last/ missing"


@pytest.mark.unit
def test_hf_callback_per_step_keeps_step_subdirs(tmp_path):
    """per_step=True keeps each step's snapshot in its own folder."""
    from stable_pretraining.callbacks.hf_models import HuggingFaceCheckpointCallback

    save_dir = tmp_path / "hf"
    cb = HuggingFaceCheckpointCallback(save_dir=str(save_dir), per_step=True)
    assert cb.per_step is True

    model = _make_simple_lightning_module()
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        enable_checkpointing=True,
        logger=False,
        callbacks=[cb],
    )
    trainer.fit(model, _trivial_loader())
    trainer.save_checkpoint(tmp_path / "manual1.ckpt")
    trainer.save_checkpoint(tmp_path / "manual2.ckpt")

    step_dirs = sorted(p.name for p in save_dir.iterdir() if p.name.startswith("step_"))
    assert len(step_dirs) >= 1, (
        f"per_step=True must create step_N subdirs, got {step_dirs}"
    )
    # Should NOT create "last" when per_step=True
    assert not (save_dir / "last").exists(), "per_step=True should not write 'last/'"


@pytest.mark.unit
def test_hf_callback_creates_missing_save_dir(tmp_path):
    """The callback must create a missing save_dir on first export.

    Even when ``save_dir`` is a deeply-nested path that doesn't yet exist,
    the callback should ``mkdir -p`` it rather than crash.
    """
    from stable_pretraining.callbacks.hf_models import HuggingFaceCheckpointCallback

    # Three nested levels of nonexistent dirs
    save_dir = tmp_path / "does" / "not" / "exist" / "yet" / "hf"
    assert not save_dir.exists()

    cb = HuggingFaceCheckpointCallback(save_dir=str(save_dir))
    model = _make_simple_lightning_module()
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
        max_steps=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        enable_checkpointing=True,
        logger=False,
        callbacks=[cb],
    )
    trainer.fit(model, _trivial_loader())
    trainer.save_checkpoint(tmp_path / "manual.ckpt")

    assert save_dir.exists(), "callback must create deeply-nested missing save_dir"
    assert (save_dir / "last" / "hf").exists()


@pytest.mark.unit
def test_hf_callback_default_swallows_errors(tmp_path):
    """Default raise_on_error=False keeps training alive on export errors.

    An export failure is logged but does NOT propagate, so training
    continues. Enforces the 'callbacks must not kill training' contract.
    """
    from stable_pretraining.callbacks.hf_models import HuggingFaceCheckpointCallback

    cb = HuggingFaceCheckpointCallback(save_dir=str(tmp_path / "hf"))
    assert cb.raise_on_error is False

    # Force a guaranteed failure: monkey-patch _do_export to always raise.
    def _boom(*a, **kw):
        raise FileNotFoundError("simulated missing dir during export")

    cb._do_export = _boom

    model = _make_simple_lightning_module()
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        enable_checkpointing=True,
        logger=False,
        callbacks=[cb],
    )
    # Should NOT raise — error is logged and swallowed.
    trainer.fit(model, _trivial_loader())
    trainer.save_checkpoint(tmp_path / "manual.ckpt")
    # Confirm we actually reached the end of training
    assert trainer.current_epoch >= 0


@pytest.mark.unit
def test_hf_callback_raise_on_error_propagates(tmp_path):
    """raise_on_error=True restores the legacy fail-fast behavior.

    Failures kill training. Useful for unit tests that need to assert
    export correctness.
    """
    from stable_pretraining.callbacks.hf_models import HuggingFaceCheckpointCallback

    cb = HuggingFaceCheckpointCallback(
        save_dir=str(tmp_path / "hf"), raise_on_error=True
    )

    def _boom(*a, **kw):
        raise FileNotFoundError("simulated missing dir during export")

    cb._do_export = _boom

    model = _make_simple_lightning_module()
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
        max_steps=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        enable_checkpointing=True,
        logger=False,
        callbacks=[cb],
    )
    with pytest.raises(FileNotFoundError):
        trainer.fit(model, _trivial_loader())
        trainer.save_checkpoint(tmp_path / "manual.ckpt")


if __name__ == "__main__":
    test_spt_hf_fidelity_flow()
