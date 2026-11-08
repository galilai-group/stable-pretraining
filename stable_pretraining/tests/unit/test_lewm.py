"""Tests for the LeWM method."""

import torch
from torch import nn

from stable_pretraining.methods.lewm import LeWM


class TinyWorldModel(nn.Module):
    """Minimal world model for testing LeWM training."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 8)
        self.action_encoder = nn.Linear(3, 8)
        self.predictor = nn.Linear(8, 8)

    def encode(self, batch):
        return {
            "emb": self.encoder(batch["pixels"]),
            "act_emb": self.action_encoder(batch["action"]),
        }

    def predict(self, emb, act_emb):
        return self.predictor(emb + act_emb)


def test_lewm_forward_backward():
    """LeWM computes finite losses and propagates gradients."""
    model = TinyWorldModel()

    method = LeWM(
        model=model,
        history_size=3,
        num_preds=1,
        sigreg_weight=0.09,
    )

    batch = {
        "pixels": torch.randn(4, 4, 4),
        "action": torch.randn(4, 4, 3),
    }

    output = method.forward(batch, stage="train")

    for key in ("pred_loss", "sigreg_loss", "loss"):
        assert key in output
        assert torch.isfinite(output[key]).all()

    expected = output["pred_loss"] + 0.09 * output["sigreg_loss"]
    torch.testing.assert_close(output["loss"], expected)

    output["loss"].backward()

    assert model.encoder.weight.grad is not None
    assert model.predictor.weight.grad is not None
    assert model.action_encoder.weight.grad is not None


def test_lewm_public_api():
    """LeWM is accessible from the package root."""
    import stable_pretraining as spt

    assert spt.LeWM is LeWM
