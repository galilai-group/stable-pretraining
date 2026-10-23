"""Local Deep InfoMax (JSD) on CIFAR10 with the DCGAN-style encoder of the paper.

Pretrains the encoder with local DIM (plus the optional prior matching term),
then evaluates it as in Table 1 of Hjelm et al. (ICLR 2019): the encoder is
frozen and a one-hidden-layer classifier is trained on each of the ``conv``,
``fc`` and ``Y`` features. Settings not stated in the paper follow the
reference implementation (https://github.com/rdevon/DIM, ``basic32x32``
config, cortex defaults): inputs normalized to [-1, 1] without augmentation,
Adam(lr=1e-4, betas=(0.5, 0.999)), batch size 64, 1000 epochs. Training
metrics are written by the ``Manager`` run registry (view them with ``spt web``).

Usage::

    python local-dim-dcgan.py                       # DIM(L): prior weight 0.1
    python local-dim-dcgan.py --prior-weight 0      # local-only ablation
    python local-dim-dcgan.py --eval-only runs/local-dim/encoder.pt
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor
from torch import nn

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

MEAN_STD = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
LOCAL_LAYER = "layer1"
EVAL_LAYERS = {"conv": "layer2", "fc": "layer4"}
EVAL_DIMS = {"conv": 256 * 4 * 4, "fc": 1024, "Y": 64}
ADAM = {"lr": 1e-4, "betas": (0.5, 0.999)}  # cortex defaults


def build_encoder(global_dim: int = 64) -> nn.Sequential:
    """DCGAN-style encoder of the ``basic32x32`` config, layer names as in the reference."""

    def conv(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )

    return nn.Sequential(
        OrderedDict(
            layer0=conv(3, 64),  # 16x16
            layer1=conv(64, 128),  # 8x8, local feature map
            layer2=conv(128, 256),  # 4x4, "conv" evaluation features
            layer3=nn.Flatten(),
            layer4=nn.Sequential(
                nn.Linear(256 * 4 * 4, 1024, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
            ),  # "fc" evaluation features
            layer5=nn.Linear(1024, global_dim),  # global feature Y
        )
    )


def build_classifier(in_dim: int) -> nn.Sequential:
    """Evaluation classifier of the reference ``ClassificationEval``.

    Linear(200) -> Dropout(0.1) -> BN -> ReLU -> Linear(10), in the layer order
    of the reference ``Convnet`` blocks.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, 200),
        nn.Dropout(0.1),
        nn.BatchNorm1d(200),
        nn.ReLU(),
        nn.Linear(200, 10),
    )


def pretrain(args) -> nn.Module:
    """Train the encoder with local DIM and return it."""
    data_dir = get_data_dir("cifar10")
    transform = transforms.Compose(transforms.RGB(), transforms.ToImage(**MEAN_STD))
    train_dataset = spt.data.FromTorchDataset(
        torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True),
        names=["image", "label"],
        transform=transform,
    )
    val_dataset = spt.data.FromTorchDataset(
        torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True),
        names=["image", "label"],
        transform=transform,
    )
    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
        ),
        val=torch.utils.data.DataLoader(
            val_dataset, batch_size=256, num_workers=args.num_workers
        ),
    )

    encoder = build_encoder()
    module = spt.Module(
        forward=forward.local_dim,
        backbone=spt.backbone.HiddenStateExtractor(
            encoder, [LOCAL_LAYER, *EVAL_LAYERS.values()]
        ),
        local_layer=LOCAL_LAYER,
        dim_critic=spt.backbone.LocalGlobalCritic(
            local_channels=128, global_dim=64, embed_dim=2048
        ),
        dim_loss=spt.losses.LocalDIMLoss(),
        prior_loss=spt.losses.DIMPriorLoss(spt.backbone.MLP(64, [1000, 200, 1])),
        prior_weight=args.prior_weight,
        optim={
            "optimizer": {"type": "Adam", **ADAM},
            "scheduler": {"type": "ConstantLR", "factor": 1.0},
            "interval": "epoch",
        },
        hparams=vars(args),
    )

    # As with ``main.py local classifier`` in the reference code, classifiers on
    # the conv / fc / Y features are trained alongside DIM for monitoring only;
    # reported numbers come from evaluate().
    inputs = {**EVAL_LAYERS, "Y": "embedding"}
    callbacks = [
        spt.callbacks.OnlineProbe(
            module,
            name=f"probe_{name}",
            input=inputs[name],
            target="label",
            probe=build_classifier(dim),
            loss=nn.CrossEntropyLoss(),
            optimizer={"type": "Adam", **ADAM},
            metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
        )
        for name, dim in EVAL_DIMS.items()
    ] + [
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        limit_train_batches=args.limit_train_batches,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    spt.Manager(trainer=trainer, module=module, data=data)()

    torch.save(encoder.state_dict(), args.out_dir / "encoder.pt")
    return encoder


@torch.no_grad()
def extract_features(encoder: nn.Module, train: bool, device) -> dict:
    """Frozen ``conv``, ``fc`` and ``Y`` features and labels of a CIFAR10 split."""
    dataset = torchvision.datasets.CIFAR10(
        root=str(get_data_dir("cifar10")),
        train=train,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(**MEAN_STD),
            ]
        ),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=500, num_workers=4)
    extractor = spt.backbone.HiddenStateExtractor(encoder, list(EVAL_LAYERS.values()))
    extractor.to(device).eval()

    feats = {name: [] for name in [*EVAL_LAYERS, "Y"]}
    labels = []
    for images, targets in loader:
        out = extractor(images.to(device))
        for name, layer in EVAL_LAYERS.items():
            feats[name].append(out.hidden_states[layer].flatten(1))
        feats["Y"].append(out.last_hidden_state)
        labels.append(targets.to(device))
    extractor.remove_hooks()
    return {k: torch.cat(v) for k, v in feats.items()}, torch.cat(labels)


def train_classifier(x_train, y_train, x_test, y_test, epochs, batch_size=64):
    """Train the reference evaluation classifier; return per-epoch test accuracy.

    Matches ``ClassificationEval`` of the reference implementation: Adam(1e-4,
    betas=(0.5, 0.999)), batch size 64, MultiStepLR(milestones=[50, 100],
    gamma=0.1).
    """
    classifier = build_classifier(x_train.size(1)).to(x_train.device)
    optimizer = torch.optim.Adam(classifier.parameters(), **ADAM)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 100], gamma=0.1)

    accuracies = []
    for _ in range(epochs):
        # cortex steps the scheduler at the start of every training epoch.
        scheduler.step()
        classifier.train()
        for idx in torch.randperm(len(x_train), device=x_train.device).split(
            batch_size
        ):
            loss = nn.functional.cross_entropy(classifier(x_train[idx]), y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        classifier.eval()
        with torch.no_grad():
            preds = torch.cat([classifier(x).argmax(1) for x in x_test.split(1000)])
        accuracies.append((preds == y_test).float().mean().item() * 100)
    return accuracies


def evaluate(encoder: nn.Module, args) -> dict:
    """Table 1 protocol: test accuracy averaged over the last 100 classifier epochs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_feats, train_labels = extract_features(encoder, train=True, device=device)
    test_feats, test_labels = extract_features(encoder, train=False, device=device)

    results = {}
    for name in train_feats:
        accuracies = train_classifier(
            train_feats[name],
            train_labels,
            test_feats[name],
            test_labels,
            epochs=args.eval_epochs,
        )
        last = accuracies[-min(100, len(accuracies)) :]
        results[name] = {
            "mean_last_100": sum(last) / len(last),
            "final": accuracies[-1],
            "best": max(accuracies),
        }
        print(f"{name:>4}: {results[name]}")

    with open(args.out_dir / "eval.json", "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, **results}, f)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/local-dim"))
    parser.add_argument("--eval-only", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only is None:
        encoder = pretrain(args)
    else:
        encoder = build_encoder()
        encoder.load_state_dict(torch.load(args.eval_only, map_location="cpu"))
    evaluate(encoder, args)


if __name__ == "__main__":
    main()
