"""SimSiam ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

from two_view import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.simsiam import SimSiam


def main():
    batch_size = 256
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)

    module = SimSiam(
        encoder_name="vit_small_patch16_224",
        projector_dim=2048,
        predictor_hidden_dim=512,
    )
    # SimSiam paper recipe: SGD with momentum 0.9, lr scales as 0.05 * bs/256,
    # weight decay 1e-4. Predictor LR is fixed (no schedule) but spt's optimizer
    # config doesn't expose per-param-group schedules cleanly, so we keep one
    # cosine schedule on backbone+projector+predictor — close to the paper.
    attach_forward_and_optim(
        module,
        SimSiam,
        optim={
            "optimizer": {
                "type": "SGD",
                "lr": 0.05 * batch_size / 256,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
    )

    callbacks = standard_callbacks(module, embed_dim=module.embed_dim)
    trainer = standard_trainer(
        callbacks, max_epochs=max_epochs, log_name="simsiam-vits-inet10"
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
