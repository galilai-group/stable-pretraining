"""PIRL ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B.

PIRL only needs a single image per sample (the jigsaw is generated
internally), but reuses the two-view dataloader for convenience and
ignores ``view2``.
"""

from two_view import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.pirl import PIRL


def main():
    batch_size = 256
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)
    module = PIRL(
        encoder_name="vit_small_patch16_224",
        projector_dim=128,
        queue_length=8192,
        temperature=0.07,
        lambda_pirl=0.5,
        jigsaw_grid=4,
    )
    attach_forward_and_optim(
        module,
        PIRL,
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
    trainer = standard_trainer(
        callbacks, max_epochs=max_epochs, log_name="pirl-vits-inet10"
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
