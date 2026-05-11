"""iGPT (AIM-style) ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B.

Note: this is the modern AIM-style autoregressive variant — causal ViT with
next-patch MSE prediction. The classical 2020 iGPT (pixel-cluster
classification) needs a separate tokenizer.
"""

from masked import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.igpt import iGPT


def main():
    batch_size = 64
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)

    module = iGPT(
        encoder_name="vit_small_patch16_224",
        patch_size=16,
        image_size=224,
    )
    # AIM paper: AdamW lr 1e-3 with cosine, weight decay 0.05.
    attach_forward_and_optim(
        module,
        iGPT,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.05,
                "betas": (0.9, 0.95),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )

    callbacks = standard_callbacks(module, embed_dim=module.embed_dim)
    trainer = standard_trainer(callbacks, max_epochs=max_epochs, log_name="igpt-vits-inet10")
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
