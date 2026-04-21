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

    # Inject/Redirect Callback
    for cb in trainer.callbacks:
        if "HuggingFaceCheckpointCallback" in cb.__class__.__name__:
            cb.save_dir = hf_save_dir

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


if __name__ == "__main__":
    test_spt_hf_fidelity_flow()
