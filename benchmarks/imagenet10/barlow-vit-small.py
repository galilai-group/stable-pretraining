"""Barlow Twins ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

from two_view import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.barlow_twins import BarlowTwins


def main():
    batch_size = 256
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)

    module = BarlowTwins(
        encoder_name="vit_small_patch16_224",
        projector_dims=(2048, 2048, 2048),
        lambd=5.1e-3,
    )
    # ViT-S/16: use AdamW (LARS + lr 0.2 is the ResNet50 recipe and collapses on ViT).
    attach_forward_and_optim(
        module,
        BarlowTwins,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": 5e-4,
                "weight_decay": 0.05,
                "betas": (0.9, 0.95),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )

    callbacks = standard_callbacks(module, embed_dim=module.embed_dim)
    trainer = standard_trainer(callbacks, max_epochs=max_epochs, log_name="barlow-vits-inet10")
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
