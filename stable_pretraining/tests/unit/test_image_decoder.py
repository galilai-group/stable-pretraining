"""Unit tests for the image decoders and the OnlineImageDecoder callback.

Exercises three things:
1. ``CNNImageDecoder`` and ``ViTImageDecoder`` produce correct-shape outputs.
2. Both decoders raise clear, actionable errors on shape mismatch (wrong
   rank or wrong feature dim) — this is the most common user mistake.
3. ``OnlineImageDecoder`` plugs into a Lightning trainer end-to-end and
   reduces reconstruction loss on a tiny overfit batch.
"""

import pytest
import torch
import torch.nn as nn

import stable_pretraining as spt
from stable_pretraining.backbone.decoders import (
    CNNImageDecoder,
    ViTImageDecoder,
    build_image_decoder,
)


@pytest.mark.unit
class TestCNNImageDecoder:
    """Shape and validation behaviour of the CNN vector->image decoder."""

    def test_output_shape(self):
        dec = CNNImageDecoder(embed_dim=64, img_size=32, out_chans=3, base_channels=64)
        z = torch.randn(2, 64)
        y = dec(z)
        assert y.shape == (2, 3, 32, 32)

    def test_depth_single_channel(self):
        dec = CNNImageDecoder(embed_dim=32, img_size=64, out_chans=1, base_channels=64)
        z = torch.randn(4, 32)
        assert dec(z).shape == (4, 1, 64, 64)

    def test_rejects_token_grid_input(self):
        dec = CNNImageDecoder(embed_dim=32, img_size=32, base_channels=64)
        z = torch.randn(2, 16, 32)  # (B, P, D) — wrong rank
        with pytest.raises(ValueError, match="2-D input"):
            dec(z)

    def test_rejects_wrong_embed_dim(self):
        dec = CNNImageDecoder(embed_dim=32, img_size=32, base_channels=64)
        with pytest.raises(ValueError, match="expects D=32"):
            dec(torch.randn(2, 17))

    def test_rejects_bad_img_size(self):
        with pytest.raises(ValueError, match="power of two"):
            CNNImageDecoder(embed_dim=32, img_size=48)  # 48/4 = 12, not pow2


@pytest.mark.unit
class TestViTImageDecoder:
    """Shape and validation behaviour of the ViT tokens->image decoder."""

    def test_output_shape(self):
        dec = ViTImageDecoder(
            embed_dim=64,
            img_size=32,
            patch_size=8,
            out_chans=3,
            decoder_dim=64,
            depth=2,
            num_heads=4,
        )
        tokens = torch.randn(2, 16, 64)  # 4x4 grid
        assert dec(tokens).shape == (2, 3, 32, 32)

    def test_depth_single_channel(self):
        dec = ViTImageDecoder(
            embed_dim=32,
            img_size=64,
            patch_size=16,
            out_chans=1,
            decoder_dim=32,
            depth=2,
            num_heads=4,
        )
        # 64/16 = 4 -> 16 tokens
        tokens = torch.randn(3, 16, 32)
        assert dec(tokens).shape == (3, 1, 64, 64)

    def test_rejects_pooled_vector(self):
        dec = ViTImageDecoder(
            embed_dim=32,
            img_size=32,
            patch_size=8,
            decoder_dim=32,
            depth=1,
            num_heads=4,
        )
        with pytest.raises(ValueError, match="3-D input"):
            dec(torch.randn(2, 32))

    def test_rejects_wrong_token_count(self):
        dec = ViTImageDecoder(
            embed_dim=32,
            img_size=32,
            patch_size=8,
            decoder_dim=32,
            depth=1,
            num_heads=4,
        )
        with pytest.raises(ValueError, match="expects P=16 tokens"):
            dec(torch.randn(2, 15, 32))

    def test_rejects_bad_patch_size(self):
        with pytest.raises(ValueError, match="must be divisible"):
            ViTImageDecoder(
                embed_dim=32,
                img_size=32,
                patch_size=7,
                decoder_dim=32,
                depth=1,
                num_heads=4,
            )


@pytest.mark.unit
class TestBuildImageDecoder:
    """Auto-selection and error paths of ``build_image_decoder``."""

    def test_auto_resolves_to_cnn_without_patch_size(self):
        dec = build_image_decoder(embed_dim=32, image_shape=(3, 32, 32))
        assert isinstance(dec, CNNImageDecoder)

    def test_auto_resolves_to_vit_with_patch_size(self):
        dec = build_image_decoder(
            embed_dim=32,
            image_shape=(3, 32, 32),
            patch_size=8,
            decoder_kwargs=dict(decoder_dim=32, depth=1, num_heads=4),
        )
        assert isinstance(dec, ViTImageDecoder)

    def test_explicit_vit_requires_patch_size(self):
        with pytest.raises(ValueError, match="requires a patch_size"):
            build_image_decoder(embed_dim=32, image_shape=(3, 32, 32), kind="vit")

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown kind"):
            build_image_decoder(embed_dim=32, image_shape=(3, 32, 32), kind="bogus")

    def test_non_square_image_rejected(self):
        with pytest.raises(ValueError, match="square images"):
            build_image_decoder(embed_dim=32, image_shape=(3, 32, 16))

    def test_wrong_arity_image_shape_rejected(self):
        with pytest.raises(ValueError, match=r"\(C, H, W\)"):
            build_image_decoder(embed_dim=32, image_shape=(32, 32))


@pytest.mark.unit
class TestGradientDetachment:
    """Verify encoder is fully detached from decoder loss.

    The defining property of a probe: the encoder must not receive any
    gradient from the decoder's reconstruction loss. If this regresses,
    the decoder is effectively co-training the encoder via the probe path,
    which would invalidate everything that uses an online probe.
    """

    def test_encoder_gets_no_gradient_from_decoder(self):
        import lightning as pl

        embed_dim, img_size, C, B = 32, 32, 3, 4
        torch.manual_seed(0)
        encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C * img_size * img_size, embed_dim),
        )
        enc_params = list(encoder.parameters())

        def forward(self, batch, stage):
            batch["embedding"] = self.encoder(batch["image"])
            return batch

        module = spt.Module(encoder=encoder, forward=forward, optim=None)
        x = torch.randn(B, C, img_size, img_size)

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return B

            def __getitem__(self, idx):
                return {"image": x[idx]}

        dl = torch.utils.data.DataLoader(_DS(), batch_size=B)
        data = spt.data.DataModule(train=dl, val=dl)

        cb = spt.callbacks.OnlineImageDecoder(
            module=module,
            name="recon",
            input="embedding",
            target="image",
            image_shape=(C, img_size, img_size),
            embed_dim=embed_dim,
            decoder_kwargs=dict(base_channels=32),
        )

        captured = {"enc_max": None, "dec_nonzero": None}

        class _GradInspector(pl.pytorch.callbacks.Callback):
            """Inspect grads after backward, before optimizer zeros them.

            This is the only window where probe-detachment is testable.
            """

            def on_before_optimizer_step(self, trainer, pl_module, optimizer):
                if captured["enc_max"] is not None:
                    return  # only sample the first step
                enc_max = 0.0
                for p in enc_params:
                    if p.grad is not None:
                        enc_max = max(enc_max, p.grad.abs().max().item())
                dec_params = list(pl_module.callbacks_modules["recon"].parameters())
                dec_nz = sum(
                    1
                    for p in dec_params
                    if p.grad is not None and torch.any(p.grad != 0)
                )
                captured["enc_max"] = enc_max
                captured["dec_nonzero"] = dec_nz

        trainer = pl.Trainer(
            max_steps=1,
            num_sanity_val_steps=0,
            callbacks=[cb, _GradInspector()],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()

        assert captured["enc_max"] is not None, "grad inspector never fired"
        # Encoder grads must be exactly zero — the decoder loss is detached
        # from the encoder by OnlineProbe.wrap_forward.
        assert captured["enc_max"] == 0.0, (
            f"encoder grad leaked from decoder loss: max|grad|={captured['enc_max']}"
        )
        # Sanity-check the other direction: decoder DID receive gradient,
        # so the zero result above is not a false positive from a broken
        # backward path.
        assert captured["dec_nonzero"] > 0, (
            "decoder params got no gradient — backprop is broken, so the "
            "encoder-grad assertion above is meaningless"
        )


@pytest.mark.unit
class TestOverfitDrivesLossDown:
    """Verify the decoder actually trains (loss drops over a few steps).

    If the optimizer isn't wired up, or the decoder is detached from its
    own loss, the unit tests above still pass (forward shapes are fine, no
    gradient flows to the encoder). This test catches that class of bug by
    asserting the loss actually decreases over a handful of steps on a
    fixed batch.
    """

    def test_cnn_loss_decreases_on_fixed_batch(self):
        import lightning as pl

        embed_dim, img_size, C, B = 32, 32, 3, 8
        torch.manual_seed(0)
        encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C * img_size * img_size, embed_dim),
        )
        # Freeze encoder so optimizer can't change its output across steps;
        # any loss reduction must come from decoder updates.
        for p in encoder.parameters():
            p.requires_grad_(False)

        def forward(self, batch, stage):
            with torch.no_grad():
                batch["embedding"] = self.encoder(batch["image"])
            return batch

        module = spt.Module(encoder=encoder, forward=forward, optim=None)
        x = torch.randn(B, C, img_size, img_size)

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return B

            def __getitem__(self, idx):
                return {"image": x[idx]}

        dl = torch.utils.data.DataLoader(_DS(), batch_size=B)
        data = spt.data.DataModule(train=dl, val=dl)

        losses = []

        class _Probe(pl.pytorch.callbacks.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                m = trainer.callback_metrics.get("train/recon_loss")
                if m is not None:
                    losses.append(float(m))

        cb = spt.callbacks.OnlineImageDecoder(
            module=module,
            name="recon",
            input="embedding",
            target="image",
            image_shape=(C, img_size, img_size),
            embed_dim=embed_dim,
            decoder_kwargs=dict(base_channels=64),
        )
        trainer = pl.Trainer(
            max_steps=50,
            num_sanity_val_steps=0,
            callbacks=[cb, _Probe()],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()

        assert len(losses) >= 10, f"only captured {len(losses)} losses"
        head = sum(losses[:5]) / 5
        tail = sum(losses[-5:]) / 5
        assert tail < head * 0.5, (
            f"recon loss did not drop enough: head={head:.4f} tail={tail:.4f} "
            f"(expected tail < 0.5 * head). Full series: {losses}"
        )


@pytest.mark.unit
class TestOnlineImageDecoderEndToEnd:
    """Verify the callback hooks into a real LightningModule + Trainer."""

    def _make_module(self, embed_dim, img_size, channels):
        # Tiny "encoder": ground-truth target image is generated alongside the
        # synthetic batch; the encoder outputs a fixed-D embedding so the
        # decoder has something to regress from.
        torch.manual_seed(0)
        encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * img_size * img_size, embed_dim),
        )

        def forward(self, batch, stage):
            batch["embedding"] = self.encoder(batch["image"])
            return batch

        module = spt.Module(encoder=encoder, forward=forward, optim=None)
        return module

    def _make_data(self, batch_size, channels, img_size):
        # One fixed batch we re-yield -- enough to confirm the loss drops.
        x = torch.randn(batch_size, channels, img_size, img_size)

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return batch_size

            def __getitem__(self, idx):
                return {"image": x[idx]}

        dl = torch.utils.data.DataLoader(_DS(), batch_size=batch_size)
        return spt.data.DataModule(train=dl, val=dl)

    def test_cnn_decoder_runs_and_logs(self):
        import lightning as pl

        embed_dim, img_size, C, B = 64, 32, 3, 4
        module = self._make_module(embed_dim, img_size, C)
        data = self._make_data(B, C, img_size)

        cb = spt.callbacks.OnlineImageDecoder(
            module=module,
            name="recon",
            input="embedding",
            target="image",
            image_shape=(C, img_size, img_size),
            embed_dim=embed_dim,
            decoder_kwargs=dict(base_channels=64),
        )

        trainer = pl.Trainer(
            max_steps=3,
            num_sanity_val_steps=0,
            callbacks=[cb],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()
        # Probe module is registered and produced an output buffer
        assert "recon" in module.callbacks_modules

    def test_vit_decoder_runs(self):
        import lightning as pl

        # ViT path needs a token-grid input. Stub an encoder that emits one.
        embed_dim, img_size, patch_size, C, B = 32, 32, 8, 3, 2
        P = (img_size // patch_size) ** 2

        def forward(self, batch, stage):
            B_ = batch["image"].shape[0]
            # Pretend tokens; gradient path is irrelevant because the
            # callback detaches before decoding anyway.
            batch["tokens"] = self.encoder(batch["image"]).view(B_, P, embed_dim)
            return batch

        torch.manual_seed(0)
        encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C * img_size * img_size, P * embed_dim),
        )
        module = spt.Module(encoder=encoder, forward=forward, optim=None)
        data = self._make_data(B, C, img_size)

        cb = spt.callbacks.OnlineImageDecoder(
            module=module,
            name="recon_vit",
            input="tokens",
            target="image",
            image_shape=(C, img_size, img_size),
            embed_dim=embed_dim,
            patch_size=patch_size,
            decoder_kwargs=dict(decoder_dim=32, depth=2, num_heads=4),
        )

        trainer = pl.Trainer(
            max_steps=3,
            num_sanity_val_steps=0,
            callbacks=[cb],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()
        assert "recon_vit" in module.callbacks_modules

    def test_multiple_decoders_have_unique_keys(self):
        """Stack two decoders and confirm all per-callback slots stay distinct.

        Each prediction key, module, metric, and optimizer slot must be
        name-scoped. Mirrors the cvjepa.py usage where one decoder is
        attached per camera view (RGB + depth) — all sharing the same input
        embedding.
        """
        import lightning as pl

        embed_dim, img_size, C, B = 64, 32, 3, 4
        module = self._make_module(embed_dim, img_size, C)
        # Add a depth-like target to the batch so a 1-channel decoder has
        # something to regress against.
        depth = torch.randn(B, 1, img_size, img_size)
        x = torch.randn(B, C, img_size, img_size)

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return B

            def __getitem__(self, idx):
                return {"image": x[idx], "depth": depth[idx]}

        dl = torch.utils.data.DataLoader(_DS(), batch_size=B)
        data = spt.data.DataModule(train=dl, val=dl)

        cb_rgb = spt.callbacks.OnlineImageDecoder(
            module=module,
            name="recon_rgb",
            input="embedding",
            target="image",
            image_shape=(C, img_size, img_size),
            embed_dim=embed_dim,
            decoder_kwargs=dict(base_channels=32),
        )
        cb_depth = spt.callbacks.OnlineImageDecoder(
            module=module,
            name="recon_depth",
            input="embedding",
            target="depth",
            image_shape=(1, img_size, img_size),
            embed_dim=embed_dim,
            decoder_kwargs=dict(base_channels=32),
        )

        # Capture outputs from the forward to verify both prediction keys
        # land in the same outputs dict without collision.
        captured = {}

        class _Capture(pl.pytorch.callbacks.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                # outputs here is the dict returned by training_step
                if isinstance(outputs, dict):
                    captured["keys"] = set(outputs.keys())

        trainer = pl.Trainer(
            max_steps=1,
            num_sanity_val_steps=0,
            callbacks=[cb_rgb, cb_depth, _Capture()],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()

        # Both modules registered under distinct keys
        assert "recon_rgb" in module.callbacks_modules
        assert "recon_depth" in module.callbacks_modules
        assert (
            module.callbacks_modules["recon_rgb"]
            is not module.callbacks_modules["recon_depth"]
        )
        # Distinct metric namespaces
        assert "recon_rgb" in module.callbacks_metrics
        assert "recon_depth" in module.callbacks_metrics
        # Distinct prediction keys in the forward outputs dict
        assert "recon_rgb_preds" in captured["keys"]
        assert "recon_depth_preds" in captured["keys"]

    def test_shape_mismatch_raises_clear_error(self):
        """If image_shape doesn't match the target tensor, MSE wrapper fires."""
        import lightning as pl

        embed_dim, img_size, C, B = 64, 32, 3, 4
        module = self._make_module(embed_dim, img_size, C)
        data = self._make_data(B, C, img_size)

        # Claim 64x64 reconstruction but the actual target is 32x32 -> mismatch
        cb = spt.callbacks.OnlineImageDecoder(
            module=module,
            name="bad",
            input="embedding",
            target="image",
            image_shape=(C, 64, 64),
            embed_dim=embed_dim,
            decoder_kwargs=dict(base_channels=32),
        )
        trainer = pl.Trainer(
            max_steps=1,
            num_sanity_val_steps=0,
            callbacks=[cb],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        with pytest.raises(ValueError, match="shape disagree"):
            manager()
