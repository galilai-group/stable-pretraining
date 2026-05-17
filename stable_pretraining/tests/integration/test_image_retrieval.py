"""Integration tests for image retrieval functionality."""

import pytest
import torch
from transformers import AutoModel


@pytest.mark.integration
class TestImageRetrievalIntegration:
    """Integration tests for image retrieval with actual models and data."""

    @pytest.mark.gpu
    def test_feature_extraction_with_dino(self):
        """Test feature extraction using DINO model."""
        # Load model
        backbone = AutoModel.from_pretrained("facebook/dino-vits16")

        # Create dummy batch
        batch = {"image": torch.randn(2, 3, 224, 224)}

        # Extract features
        with torch.inference_mode():
            output = backbone(pixel_values=batch["image"])
            cls_embed = output.last_hidden_state[:, 0, :]

        # Verify output shape
        assert cls_embed.shape == (2, 384)  # DINO ViT-S/16 has 384-dim features
