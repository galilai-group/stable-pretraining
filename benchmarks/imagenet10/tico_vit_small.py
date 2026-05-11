"""TiCO ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

from two_view import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.tico import TiCO


def main():
    batch_size = 256
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)
    module = TiCO(
        encoder_name="vit_small_patch16_224",
        projector_dims=(2048, 256),
        beta=0.9,
        rho=20.0,
    )
    # TiCO paper (ResNet50, ImageNet-1k, 100 ep): LARS lr 0.3·bs/256,
    # weight decay 1e-5, eta 0.001. Adapted to bs=256 → lr 0.3.
    attach_forward_and_optim(
        module,
        TiCO,
        optim={
            "optimizer": {
                "type": "LARS",
                "lr": 0.3 * batch_size / 256,
                "weight_decay": 1e-5,
                "clip_lr": True,
                "eta": 0.001,
                "exclude_bias_n_norm": True,
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )
    callbacks = standard_callbacks(module, embed_dim=module.embed_dim)
    trainer = standard_trainer(callbacks, max_epochs=max_epochs, log_name="tico-vits-inet10")
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
