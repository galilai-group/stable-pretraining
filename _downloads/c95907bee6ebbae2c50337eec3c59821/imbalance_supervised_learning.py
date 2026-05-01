"""
Imbalanced Supervised Learning
==============================

This example demonstrates how to use stable-SSL to train a supervised model
on CIFAR10 with class imbalance.
"""

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig
from torchvision import transforms

import stable_pretraining as spt
from stable_pretraining.supervised import Supervised


class MyCustomSupervised(Supervised):
    """Custom supervised example model for an imbalanced dataset."""

    def initialize_train_loader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            root=self.config.root, train=True, download=True, transform=transform
        )
        distribution = np.exp(np.linspace(0, self.config.distribution, 10))
        distribution /= np.sum(distribution)
        trainset = spt.base.resample_classes(trainset, distribution)
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.config.optim.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        return trainloader

    def initialize_test_loader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        testset = torchvision.datasets.CIFAR10(
            root=self.config.root, train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.config.optim.batch_size, num_workers=2
        )
        return testloader

    def initialize_modules(self):
        self.model = spt.utils.nn.resnet9()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self):
        """The compute_loss method is called during training on each mini-batch.

        stable-SSL automatically stores the output of the data loader as `self.data`,
        which you can access directly within this function.
        """
        preds = self.forward(self.data[0])
        print(self.data[1][:4])
        self.log(
            {"train/step/acc1": self.metrics["train/step/acc1"](preds, self.data[1])},
            commit=False,
        )
        return F.cross_entropy(preds, self.data[1])


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    args = spt.get_args(cfg)

    print("--- Arguments ---")
    print(args)

    # while we provide a lot of config parameters (e.g. `optim.batch_size`), you can
    # also pass arguments directly when calling your model, they will be logged and
    #  accessible from within the model as `self.config.root` (in this example)
    trainer = MyCustomSupervised(args, root="~/data")
    trainer()


def visualization():
    import matplotlib.pyplot as plt
    import seaborn
    from matplotlib import colormaps

    seaborn.set(font_scale=2)

    cmap = colormaps.get_cmap("cool")

    configs, values = spt.reader.jsonl_project("experiment_llm")
    distris = {j: i for i, j in enumerate(np.unique(configs["distribution"]))}
    print(distris)
    fig, axs = plt.subplots(1, 1, sharey="all", sharex="all", figsize=(10, 7))

    for (_, c), v in zip(configs.iterrows(), values):
        if c["distribution"] > 0.01:
            continue
        axs.plot(
            v[-1]["eval/epoch/acc1_by_class"],
            c=cmap(np.sqrt(np.sqrt(c["optim.weight_decay"] / 10))),
            linewidth=3,
        )
        print(
            "(",
            c["optim.weight_decay"],
            ",",
            np.round(100 * np.array(v[-1]["eval/epoch/acc1_by_class"]), 2),
            ")",
        )

    plt.ylabel("test accuracy")
    plt.xlabel("class index")
    plt.tight_layout()
    plt.savefig("imbalance_classification.png")
    plt.close()


if __name__ == "__main__":
    main()
    visualization()
