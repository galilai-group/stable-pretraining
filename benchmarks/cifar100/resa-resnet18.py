"""ReSA training on CIFAR-100 with a ResNet-18.

Runs the shared CIFAR recipe in ``benchmarks/cifar10/resa-resnet18.py`` with
``--dataset cifar100``; every other flag is passed through. Paper (Table 1,
ResNet-18, 1000 epochs): 72.21 linear / 66.83 k-NN.

Usage:
    python benchmarks/cifar100/resa-resnet18.py
    python benchmarks/cifar100/resa-resnet18.py --epochs 2 --limit-train-batches 20  # smoke test
"""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    recipe = Path(__file__).resolve().parents[1] / "cifar10" / "resa-resnet18.py"
    sys.argv = [str(recipe), "--dataset", "cifar100", *sys.argv[1:]]
    runpy.run_path(str(recipe), run_name="__main__")
