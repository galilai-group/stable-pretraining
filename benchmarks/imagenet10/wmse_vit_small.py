"""W-MSE ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

from two_view import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.wmse import WMSE


def main():
    batch_size = 256
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)
    # W-MSE paper (ResNet50, ImageNet-1k, 200 ep): AdamW lr 2e-3, weight
    # decay 1e-6, very small whitening dim (4-128 depending on sub-batch).
    module = WMSE(
        encoder_name="vit_small_patch16_224",
        projector_dims=(1024, 64),
        eps=1e-3,
    )
    attach_forward_and_optim(
        module,
        WMSE,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": 2e-3,
                "weight_decay": 1e-6,
                "betas": (0.9, 0.95),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )
    callbacks = standard_callbacks(module, embed_dim=module.embed_dim)
    trainer = standard_trainer(
        callbacks, max_epochs=max_epochs, log_name="wmse-vits-inet10"
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
