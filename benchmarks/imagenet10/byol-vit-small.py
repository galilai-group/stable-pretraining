"""BYOL ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

from two_view import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.byol import BYOL


def main():
    batch_size = 256
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)

    module = BYOL(
        encoder_name="vit_small_patch16_224",
        projector_dims=(4096, 256),
        predictor_dims=(4096, 256),
        ema_decay_start=0.99,
        ema_decay_end=1.0,
    )
    attach_forward_and_optim(
        module,
        BYOL,
        optim={
            "optimizer": {
                "type": "LARS",
                "lr": 0.5 * batch_size / 256,
                "weight_decay": 1e-6,
                "clip_lr": True,
                "eta": 0.02,
                "exclude_bias_n_norm": True,
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )

    # BYOL needs the EMA teacher updated each step
    callbacks = [
        spt.callbacks.TeacherStudentCallback(update_frequency=1, update_after_backward=True),
        *standard_callbacks(module, embed_dim=module.embed_dim),
    ]
    trainer = standard_trainer(callbacks, max_epochs=max_epochs, log_name="byol-vits-inet10")
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
