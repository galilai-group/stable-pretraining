"""MaskFeat ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

from masked import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.maskfeat import MaskFeat


def main():
    batch_size = 128
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)
    module = MaskFeat(
        encoder_name="vit_small_patch16_224",
        patch_size=16,
        mask_ratio=0.4,
        n_hog_bins=9,
    )
    # MaskFeat paper (MViT-S, ImageNet-1k, 300 ep): AdamW lr 2e-3, weight
    # decay 0.05, betas (0.9, 0.999).
    attach_forward_and_optim(
        module,
        MaskFeat,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": 2e-3,
                "weight_decay": 0.05,
                "betas": (0.9, 0.999),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )
    callbacks = standard_callbacks(module, embed_dim=module.embed_dim)
    trainer = standard_trainer(
        callbacks, max_epochs=max_epochs, log_name="maskfeat-vits-inet10"
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
