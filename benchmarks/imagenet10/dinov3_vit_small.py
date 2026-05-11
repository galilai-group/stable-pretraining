"""DINOv3 ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B.

Two global views only (no local) for the short verification budget.
"""

from multicrop import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.dinov3 import DINOv3


def main():
    batch_size = 128
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(
        batch_size=batch_size, num_workers=8, n_global=2, n_local=6
    )
    module = DINOv3(
        encoder_name="vit_small_patch16_224",
        n_register_tokens=4,
        koleo_weight=0.1,
        projector_hidden_dim=2048,
        projector_bottleneck_dim=256,
        n_cls_prototypes=8192,
        n_patch_prototypes=2048,
        mask_ratio=0.3,
        patch_loss_weight=1.0,
        temperature_student=0.1,
        temperature_teacher=0.07,
        ema_decay_start=0.996,
        ema_decay_end=1.0,
    )
    attach_forward_and_optim(
        module,
        DINOv3,
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
        spt.callbacks.TeacherStudentCallback(
            update_frequency=1, update_after_backward=True
        ),
        *standard_callbacks(module, embed_dim=module.embed_dim),
    ]
    trainer = standard_trainer(
        callbacks, max_epochs=max_epochs, log_name="dinov3-vits-inet10"
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
