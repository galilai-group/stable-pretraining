"""MSN ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

from two_view import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.msn import MSN


def main():
    batch_size = 256
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)
    module = MSN(
        encoder_name="vit_small_patch16_224",
        n_prototypes=1024,
        mask_ratio=0.6,
        temperature_student=0.1,
        temperature_teacher=0.025,
        me_max_weight=1.0,
    )
    attach_forward_and_optim(
        module,
        MSN,
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
    callbacks = [
        spt.callbacks.TeacherStudentCallback(
            update_frequency=1, update_after_backward=True
        ),
        *standard_callbacks(module, embed_dim=module.embed_dim),
    ]
    trainer = standard_trainer(
        callbacks, max_epochs=max_epochs, log_name="msn-vits-inet10"
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
