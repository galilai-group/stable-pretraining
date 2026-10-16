"""Unit tests for the SigLIP method class."""

import math
import types

import lightning as pl
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from stable_pretraining import forward
from stable_pretraining.methods import SigLIP

VOCAB, SEQ, IMG = 50, 8, 16


def _tiny_siglip_config():
    from transformers import SiglipConfig, SiglipTextConfig, SiglipVisionConfig

    vision = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=IMG,
        patch_size=8,
    )
    text = SiglipTextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=SEQ,
    )
    return SiglipConfig(text_config=text.to_dict(), vision_config=vision.to_dict())


class _TinyText(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.emb = nn.EmbeddingBag(VOCAB, dim)

    def forward(self, input_ids, attention_mask=None):
        return self.emb(input_ids)


def _tiny_custom_model(**kwargs):
    vision = nn.Sequential(nn.Flatten(), nn.Linear(3 * IMG * IMG, 16))
    return SigLIP(vision_encoder=vision, text_encoder=_TinyText(16), **kwargs)


def _batch(b=4):
    return (
        torch.randn(b, 3, IMG, IMG),
        torch.randint(0, VOCAB, (b, SEQ)),
    )


@pytest.mark.unit
class TestSigLIP:
    """Construction, forward, and training-loop behavior of ``SigLIP``."""

    def test_hf_config_forward_and_backward(self, tmp_path):
        """Random-init towers from a HF config produce a loss that trains every part."""
        _tiny_siglip_config().save_pretrained(tmp_path)
        model = SigLIP(model_name=str(tmp_path))
        images, ids = _batch()

        out = model(images=images, input_ids=ids)
        out.loss.backward()

        assert out.loss.ndim == 0 and torch.isfinite(out.loss)
        assert torch.allclose(out.image_embeds.norm(dim=-1), torch.ones(4), atol=1e-5)
        assert torch.allclose(out.text_embeds.norm(dim=-1), torch.ones(4), atol=1e-5)
        for name in ("vision_model", "text_model", "siglip_loss"):
            grads = [p.grad for p in getattr(model, name).parameters()]
            assert any(g is not None and g.abs().sum() > 0 for g in grads), name
        assert model.siglip_loss.logit_scale.item() == pytest.approx(math.log(10))
        assert model.siglip_loss.logit_bias.item() == pytest.approx(-10.0)

    def test_pretrained_loads_checkpoint_logit_scale_and_bias(self, tmp_path):
        """pretrained=True copies the checkpoint's learned scale/bias into the loss."""
        from transformers import SiglipModel

        ckpt = SiglipModel(_tiny_siglip_config())
        with torch.no_grad():
            ckpt.logit_scale.fill_(4.5)
            ckpt.logit_bias.fill_(-12.5)
        ckpt.save_pretrained(tmp_path)

        model = SigLIP(model_name=str(tmp_path), pretrained=True)

        assert model.siglip_loss.logit_scale.item() == pytest.approx(4.5)
        assert model.siglip_loss.logit_bias.item() == pytest.approx(-12.5)
        ref = ckpt.vision_model.state_dict()
        for k, v in model.vision_model.state_dict().items():
            assert torch.equal(v, ref[k]), k

    def test_pretrained_requires_model_name(self):
        """pretrained=True without a checkpoint name is rejected."""
        with pytest.raises(ValueError, match="model_name"):
            SigLIP(pretrained=True)

    def test_custom_encoders_must_be_paired(self):
        """Passing only one custom encoder is rejected."""
        with pytest.raises(ValueError, match="both"):
            SigLIP(vision_encoder=nn.Identity())

    def test_pretrained_rejects_custom_encoders(self):
        """pretrained=True cannot be combined with custom encoders."""
        with pytest.raises(ValueError, match="pretrained"):
            _tiny_custom_model(pretrained=True)

    def test_single_modality_returns_embeddings_without_loss(self):
        """Embedding one modality returns zero loss and leaves the other None."""
        model = _tiny_custom_model().eval()
        images, ids = _batch()

        out = model(images=images)
        assert out.text_embeds is None
        assert out.image_embeds.shape == (4, 16)
        assert out.loss.item() == 0.0

        out = model(input_ids=ids)
        assert out.image_embeds is None
        assert out.text_embeds.shape == (4, 16)

        with pytest.raises(ValueError):
            model()

    def test_matches_siglip_loss_on_embeddings(self):
        """Method-class loss equals SigLIPLoss applied to its own embeddings."""
        from stable_pretraining.losses import SigLIPLoss

        torch.manual_seed(0)
        model = _tiny_custom_model(init_logit_scale=1.0, init_logit_bias=-2.0)
        images, ids = _batch()

        out = model(images=images, input_ids=ids)
        ref = SigLIPLoss(init_logit_scale=1.0, init_logit_bias=-2.0)(
            out.image_embeds, out.text_embeds
        )
        assert torch.allclose(out.loss, ref, atol=1e-6)

    def test_trains_with_lightning_via_forward_function(self):
        """The class plugs into ``forward.siglip`` and runs a Lightning fit loop."""

        class Pairs(Dataset):
            def __init__(self):
                g = torch.Generator().manual_seed(0)
                self.images = torch.randn(16, 3, IMG, IMG, generator=g)
                self.ids = torch.randint(0, VOCAB, (16, SEQ), generator=g)

            def __len__(self):
                return 16

            def __getitem__(self, i):
                return {"image": self.images[i], "tokenized_prompt": self.ids[i]}

        torch.manual_seed(0)
        model = _tiny_custom_model()
        model.forward = types.MethodType(forward.siglip, model)
        model.optim = {"optimizer": {"type": "AdamW", "lr": 1e-2}}
        initial_bias = model.siglip_loss.logit_bias.item()

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        loader = DataLoader(Pairs(), batch_size=8)
        trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

        assert torch.isfinite(trainer.callback_metrics["fit/loss_epoch"])
        assert "validate/loss" in trainer.callback_metrics
        assert model.siglip_loss.logit_bias.item() != initial_bias
