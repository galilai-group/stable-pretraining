"""IWM ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

import types

import torch

from two_view import make_imagenette_data, standard_callbacks, standard_trainer

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.iwm import IWM


def iwm_train_transform():
    """Source/target views from Garrido et al. 2024, Appendix A.1.

    A shared crop and flip (I'), then independent color jitters on each view,
    with the destructive augmentations on the source only, and finally the
    action between them. Where the paper is ambiguous: views are built from I'
    as in A.1 (Section 3 instead describes the source as a further-augmented
    target); source order follows BYOL (jitter, grayscale, blur, solarize); and
    solarize uses A.1's p=0.2 (Table S2's default row lists 0.1).
    """
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(),  # [0, 1] floats so solarize's 0.5 threshold is half brightness
        transforms.MultiViewTransform(
            {
                "target": transforms.Compose(
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
                "source": transforms.Compose(
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.2),
                    transforms.RandomSolarize(threshold=0.5, p=0.2),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
            }
        ),
        transforms.IWMAction(),
    )


def iwm_forward(self, batch, stage):
    if self.training:
        output = IWM.forward(
            self,
            batch["source"]["image"],
            batch["target"]["image"],
            batch["action"],
            embedding_source="student",
        )
        label = batch["source"].get("label")
    else:
        output = IWM.forward(self, batch["image"])
        label = batch.get("label")
    embedding = output.embedding.mean(dim=1)
    if self.training:
        embedding = embedding.detach()
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    return {
        "loss": output.loss,
        "embedding": embedding,
        **({"label": label.long()} if label is not None else {}),
    }


def main():
    num_gpus = torch.cuda.device_count() or 1
    batch_size = 64
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    data = make_imagenette_data(
        batch_size=batch_size, train_transform=iwm_train_transform()
    )

    # Same encoder, predictor size and optimization as ijepa-vit-small.py, not
    # the paper's ImageNet recipe (betas 0.999, 40-epoch warmup, weight decay
    # 0.04 -> 0.4). Depth 12 is the paper's IWM^Inv_{12,384} setting.
    module = IWM(
        model_or_model_name="vit_small_patch16_224",
        predictor_embed_dim=384,
        predictor_depth=12,
        action_dim=7,
        num_targets=4,
        ema_decay_start=0.996,
        ema_decay_end=1.0,
        pretrained=False,
    )
    module.forward = types.MethodType(iwm_forward, module)
    module.optim = {
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.04,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": 2 / max_epochs,
            "start_factor": 0.01,
            "end_lr": 6e-5,
            "total_steps": (len(data.train) // num_gpus) * max_epochs,
        },
        "interval": "step",
    }

    callbacks = [
        spt.callbacks.TeacherStudentCallback(
            update_frequency=1, update_after_backward=True
        ),
        *standard_callbacks(module, module.embed_dim),
    ]
    trainer = standard_trainer(callbacks, max_epochs, "iwm-vits-inet10")
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
