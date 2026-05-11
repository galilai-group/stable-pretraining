"""MoCo v2 ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

from two_view import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.mocov2 import MoCov2


def main():
    batch_size = 256
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)
    module = MoCov2(
        encoder_name="vit_small_patch16_224",
        projector_dims=(2048, 128),
        queue_length=8192,
        temperature=0.2,
    )
    attach_forward_and_optim(
        module,
        MoCov2,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": 1.5e-4,
                "weight_decay": 0.1,
                "betas": (0.9, 0.95),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )
    callbacks = [
        spt.callbacks.TeacherStudentCallback(update_frequency=1, update_after_backward=True),
        *standard_callbacks(module, embed_dim=module.embed_dim),
    ]
    trainer = standard_trainer(callbacks, max_epochs=max_epochs, log_name="mocov2-vits-inet10")
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
