"""DINO ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

from multicrop import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.dino import DINO


def main():
    batch_size = 128
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8, n_global=2, n_local=6)

    module = DINO(
        encoder_name="vit_small_patch16_224",
        encoder_kwargs={"dynamic_img_size": True},
        projector_hidden_dim=2048,
        projector_bottleneck_dim=256,
        n_prototypes=65536,
        temperature_student=0.1,
        temperature_teacher_warmup=0.04,
        temperature_teacher=0.07,
        warmup_epochs_temperature_teacher=10,
        ema_decay_start=0.996,
        ema_decay_end=1.0,
    )
    attach_forward_and_optim(
        module,
        DINO,
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

    callbacks = [
        spt.callbacks.TeacherStudentCallback(update_frequency=1, update_after_backward=True),
        *standard_callbacks(module, embed_dim=module.embed_dim),
    ]
    trainer = standard_trainer(callbacks, max_epochs=max_epochs, log_name="dino-vits-inet10")
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
