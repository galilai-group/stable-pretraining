"""Unit tests for SALT method and MAE with MultiBlockMasking."""

import pytest
import tempfile
import torch

from stable_pretraining.backbone import MultiBlockMasking, MaskedEncoder
from stable_pretraining.methods.mae import MAE
from stable_pretraining.methods.salt import SALT, SALTOutput


pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.fixture
def small_images():
    return torch.randn(2, 3, 224, 224)


class TestInit:
    """Test SALT initialization."""

    def test_default_init(self):
        model = SALT("vit_tiny_patch16_224")
        assert model.embed_dim == 192
        assert model.student is not None
        assert model.teacher is not None
        assert model.predictor is not None

    def test_with_teacher_state_dict(self):
        # Create a teacher state dict from a MaskedEncoder
        encoder = MaskedEncoder("vit_tiny_patch16_224", masking=None)
        state_dict = encoder.state_dict()

        model = SALT(
            "vit_tiny_patch16_224",
            teacher_state_dict=state_dict,
        )
        # Teacher should be loaded from state_dict
        assert model.teacher is not None

    def test_teacher_is_frozen(self):
        model = SALT("vit_tiny_patch16_224")
        for param in model.teacher.parameters():
            assert not param.requires_grad


class TestForwardShapes:
    """Test output shapes in train and eval modes."""

    def test_train_output_shape(self, small_images):
        model = SALT("vit_tiny_patch16_224")
        model.train()
        output = model(small_images)

        assert isinstance(output, SALTOutput)
        assert output.loss.dim() == 0  # scalar
        assert output.embedding.shape == (2, 192)
        assert output.predictions is not None
        assert output.targets is not None
        assert output.num_targets > 0
        assert output.num_context > 0

    def test_eval_output_shape(self, small_images):
        model = SALT("vit_tiny_patch16_224")
        model.eval()

        with torch.no_grad():
            output = model(small_images)

        assert isinstance(output, SALTOutput)
        assert output.loss.item() == 0.0
        assert output.embedding.shape == (2, 192)
        assert output.predictions is None
        assert output.targets is None
        assert output.num_targets == 0


class TestTeacherFrozen:
    """Test that teacher parameters don't require grad."""

    def test_teacher_no_grad(self):
        model = SALT("vit_tiny_patch16_224")
        for param in model.teacher.parameters():
            assert not param.requires_grad

    def test_teacher_stays_eval(self):
        model = SALT("vit_tiny_patch16_224")
        model.train()
        # Teacher should stay in eval mode even when model is in train mode
        assert not model.teacher.training


class TestStudentTrainable:
    """Test that student and predictor have gradients."""

    def test_student_has_grad(self, small_images):
        model = SALT("vit_tiny_patch16_224")
        model.train()
        output = model(small_images)
        output.loss.backward()

        # Student should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.student.parameters()
        )
        assert has_grad

    def test_predictor_has_grad(self, small_images):
        model = SALT("vit_tiny_patch16_224")
        model.train()
        output = model(small_images)
        output.loss.backward()

        # Predictor should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.predictor.parameters()
        )
        assert has_grad


class TestLoss:
    """Test loss properties."""

    def test_loss_nonzero(self, small_images):
        model = SALT("vit_tiny_patch16_224")
        model.train()
        output = model(small_images)
        assert output.loss.item() > 0

    def test_loss_finite(self, small_images):
        model = SALT("vit_tiny_patch16_224")
        model.train()
        output = model(small_images)
        assert torch.isfinite(output.loss)

    def test_loss_differentiable(self, small_images):
        model = SALT("vit_tiny_patch16_224")
        model.train()
        output = model(small_images)
        output.loss.backward()
        # Should not raise


class TestFromCheckpoint:
    """Test the from_checkpoint factory method."""

    def test_from_checkpoint(self, small_images):
        # Create a Stage 1 MAE model and save its checkpoint
        stage1 = MAE(
            "vit_tiny_patch16_224",
            decoder_embed_dim=128,
            decoder_depth=4,
            decoder_num_heads=8,
            masking=MultiBlockMasking(num_targets=4),
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            # Save as a state dict with "encoder." prefix (matching MAE structure)
            state_dict = {
                f"encoder.{k}": v for k, v in stage1.encoder.state_dict().items()
            }
            torch.save({"state_dict": state_dict}, f.name)

            # Load SALT from checkpoint
            stage2 = SALT.from_checkpoint(
                f.name,
                encoder_name="vit_tiny_patch16_224",
                predictor_embed_dim=384,
                predictor_depth=12,
                predictor_num_heads=16,
            )

        # Verify it works
        stage2.train()
        output = stage2(small_images)
        assert output.loss.item() > 0


class TestEvalMode:
    """Test eval mode behavior."""

    def test_eval_zero_loss(self, small_images):
        model = SALT("vit_tiny_patch16_224")
        model.eval()
        with torch.no_grad():
            output = model(small_images)
        assert output.loss.item() == 0.0

    def test_eval_all_patches(self, small_images):
        model = SALT("vit_tiny_patch16_224")
        model.eval()
        with torch.no_grad():
            output = model(small_images)
        # In eval, all patches are used (196 for 224x224 with patch_size=16)
        assert output.num_context == 196
        assert output.num_targets == 0

    def test_eval_cls_embedding(self, small_images):
        model = SALT("vit_tiny_patch16_224")
        model.eval()
        with torch.no_grad():
            output = model(small_images)
        assert output.embedding.shape == (2, 192)


class TestMAEWithMultiBlockMasking:
    """Test MAE accepts custom masking."""

    def test_mae_custom_masking(self):
        masking = MultiBlockMasking(num_targets=4)
        model = MAE(
            "vit_tiny_patch16_224",
            decoder_embed_dim=128,
            decoder_depth=4,
            decoder_num_heads=8,
            masking=masking,
        )
        assert model.masking is masking

    def test_mae_custom_masking_forward(self, small_images):
        masking = MultiBlockMasking(num_targets=4)
        model = MAE(
            "vit_tiny_patch16_224",
            decoder_embed_dim=128,
            decoder_depth=4,
            decoder_num_heads=8,
            masking=masking,
        )
        model.train()
        output = model(small_images)
        assert output.loss.item() > 0
        assert output.num_masked > 0

    def test_mae_custom_masking_eval(self, small_images):
        masking = MultiBlockMasking(num_targets=4)
        model = MAE(
            "vit_tiny_patch16_224",
            decoder_embed_dim=128,
            decoder_depth=4,
            decoder_num_heads=8,
            masking=masking,
        )
        model.eval()
        with torch.no_grad():
            output = model(small_images)
        assert output.loss.item() == 0.0
        assert output.num_masked == 0

    def test_mae_default_masking_still_works(self, small_images):
        """Ensure default PatchMasking still works when no masking is provided."""
        model = MAE(
            "vit_tiny_patch16_224",
            decoder_embed_dim=128,
            decoder_depth=4,
            decoder_num_heads=8,
        )
        model.train()
        output = model(small_images)
        assert output.loss.item() > 0
