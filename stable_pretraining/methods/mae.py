"""MAE: Masked Autoencoders Are Scalable Vision Learners.

Self-supervised learning via reconstructing masked patches from visible patches.

References:
    He et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.
    https://arxiv.org/abs/2111.06377

Example::

    from stable_pretraining.backbone import MAE
    import lightning as pl

    # Create model
    model = MAE("vit_base_patch16_224", mask_ratio=0.75)

    # Training
    model.train()
    output = model(images)
    output.loss.backward()

    # Get encoder for downstream
    encoder = model.encoder
"""

from dataclasses import dataclass

import torch

from stable_pretraining.backbone import MAEDecoder, MaskedEncoder, PatchMasking
from stable_pretraining.utils import MAELoss
from stable_pretraining import Module


@dataclass
class MAEOutput:
    """Output from MAE forward pass.

    :ivar loss: Reconstruction loss (MSE on masked patches)
    :ivar predictions: Reconstructed patches [B, N, patch_dim]
    :ivar mask: Binary mask where 1=masked, 0=visible [B, N]
    :ivar num_masked: Number of masked patches
    :ivar num_visible: Number of visible patches
    """

    loss: torch.Tensor
    predictions: torch.Tensor
    mask: torch.Tensor
    num_masked: int
    num_visible: int


class MAE(Module):
    """MAE: Masked Autoencoders Are Scalable Vision Learners.

    Architecture:
        - **Encoder**: ViT processing only visible (unmasked) patches
        - **Decoder**: Lightweight transformer reconstructing masked patches
        - **Target**: Normalized pixel values of masked patches

    :param encoder_name: timm model name (e.g., "vit_base_patch16_224")
    :param decoder_embed_dim: Decoder hidden dimension (default: 512)
    :param decoder_depth: Number of decoder blocks (default: 8)
    :param decoder_num_heads: Decoder attention heads (default: 16)
    :param mask_ratio: Fraction of patches to mask (default: 0.75)
    :param block_size: Masking block size, 1=random (default: 1)
    :param norm_pix_loss: Normalize target pixels per patch (default: True)
    :param loss_type: Loss type for MAELoss (default: 'mse')
    :param pretrained: Load pretrained encoder weights

    Example::

        # Basic usage
        model = MAE("vit_base_patch16_224", mask_ratio=0.75)
        images = torch.randn(4, 3, 224, 224)

        model.train()
        output = model(images)
        output.loss.backward()

        model.eval()
        output = model(images)  # Full reconstruction, zero loss

    Example with Lightning::

        class MAELightning(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = MAE("vit_base_patch16_224")

            def training_step(self, batch, batch_idx):
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                return self.model(images).loss

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1.5e-4)
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mask_ratio: float = 0.75,
        block_size: int = 1,
        norm_pix_loss: bool = True,
        loss_type: str = "mse",
        pretrained: bool = False,
    ):
        super().__init__()

        # Encoder with masking
        self.masking = PatchMasking(mask_ratio=mask_ratio, block_size=block_size)
        self.encoder = MaskedEncoder(
            encoder_name, masking=self.masking, pretrained=pretrained
        )

        embed_dim = self.encoder.embed_dim
        num_patches = self.encoder.default_grid_h * self.encoder.default_grid_w
        patch_size = self.encoder.patch_size_h
        in_chans = self.encoder.patch_embed.proj.in_channels
        patch_dim = patch_size * patch_size * in_chans

        # Decoder
        self.decoder = MAEDecoder(
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            output_dim=patch_dim,
            num_patches=num_patches,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
        )

        # Loss
        self.loss_fn = MAELoss(
            patch_size=patch_size,
            loss_type=loss_type,
            mask_only=True,
            patch_normalize=norm_pix_loss,
        )

    def forward(self, images: torch.Tensor) -> MAEOutput:
        """Forward pass.

        Training: masks patches, encodes visible, decodes all, loss on masked.
        Eval: no masking, full encode/decode, zero loss.

        :param images: Input images [B, C, H, W]
        :return: MAEOutput with loss and reconstructions
        """
        enc_out = self.encoder(images)

        # Decode (output_masked_only=False gives full reconstruction)
        encoded_patches = enc_out.encoded[:, self.encoder.num_prefix_tokens :]
        predictions = self.decoder(
            encoded_patches,
            enc_out.mask,
            ids_keep=enc_out.ids_keep,
            output_masked_only=False,
        )

        if self.training:
            loss = self.loss_fn(predictions, images, enc_out.mask)
            num_masked = int(enc_out.mask.sum(dim=1)[0].item())
        else:
            loss = torch.tensor(0.0, device=images.device)
            num_masked = 0

        return MAEOutput(
            loss=loss,
            predictions=predictions,
            mask=enc_out.mask,
            num_masked=num_masked,
            num_visible=enc_out.mask.shape[1] - num_masked,
        )
