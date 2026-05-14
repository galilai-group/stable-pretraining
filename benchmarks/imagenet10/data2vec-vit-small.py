"""data2vec ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

from masked import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.data2vec import Data2Vec


def main():
    batch_size = 128
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)

    # data2vec paper (vision, ViT-B/16, ImageNet-1k): AdamW lr 1.5e-3,
    # weight decay 0.05, betas (0.9, 0.98), top_k = ALL blocks averaged,
    # EMA τ 0.9998 → 1.0 over training.
    module = Data2Vec(
        encoder_name="vit_small_patch16_224",
        top_k_blocks=12,
        mask_ratio=0.6,
        ema_decay_start=0.9998,
        ema_decay_end=1.0,
    )
    attach_forward_and_optim(
        module,
        Data2Vec,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": 1.5e-3,
                "weight_decay": 0.05,
                "betas": (0.9, 0.98),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )

    callbacks = [
        spt.callbacks.TeacherStudentCallback(update_frequency=1, update_after_backward=True),
        *standard_callbacks(module, embed_dim=module.embed_dim),
    ]
    trainer = standard_trainer(callbacks, max_epochs=max_epochs, log_name="data2vec-vits-inet10")
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
