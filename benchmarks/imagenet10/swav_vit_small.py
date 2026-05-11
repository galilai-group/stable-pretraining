"""SwAV ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B.

Uses multi-crop (2 global + 4 local) — vanilla 2-view SwAV without a queue
collapses, since the Sinkhorn batch is too small to enforce a balanced
prototype distribution.
"""

from multicrop import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.swav import SwAV


def main():
    batch_size = 128
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(
        batch_size=batch_size, num_workers=8, n_global=2, n_local=4
    )

    module = SwAV(
        encoder_name="vit_small_patch16_224",
        projector_dims=(2048, 128),
        n_prototypes=3000,
        temperature=0.1,
        sinkhorn_iterations=3,
        epsilon=0.05,
    )
    attach_forward_and_optim(
        module,
        SwAV,
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
        use_local=True,
    )

    callbacks = standard_callbacks(module, embed_dim=module.embed_dim)
    trainer = standard_trainer(
        callbacks, max_epochs=max_epochs, log_name="swav-vits-inet10"
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
