"""Offline linear evaluation of a DirectPred (or BYOL) CIFAR-10 checkpoint.

Evaluates a checkpoint written by ``directpred-resnet18.py`` the way the
DirectPred CIFAR-10 numbers are reported: the frozen online encoder embeds the
un-augmented training and test images, the features are standardized with the
training statistics, and a linear classifier is trained on them with Adam
(lr 3e-4, weight decay 1e-3, batch 128) for 50 epochs. Both the best test
top-1 over the epochs and the final-epoch top-1 are printed.

Usage:
    python benchmarks/cifar10/directpred-linear-eval.py --ckpt path/to/last.ckpt
    python benchmarks/cifar10/directpred-linear-eval.py --ckpt ... --encoder target
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T

import stable_pretraining as spt

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--encoder", choices=["online", "target"], default="online")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=str, default=None, help="optional JSON file")
args = parser.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Same normalization as directpred-resnet18.py, without augmentation.
transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)
data_dir = str(get_data_dir("cifar10"))
train_set = torchvision.datasets.CIFAR10(
    data_dir, train=True, transform=transform, download=True
)
test_set = torchvision.datasets.CIFAR10(
    data_dir, train=False, transform=transform, download=True
)

# The online encoder is the student of the backbone wrapper, the EMA target is
# the teacher.
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
prefix = "backbone.student." if args.encoder == "online" else "backbone.teacher."
state = {
    k[len(prefix) :]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)
}
encoder = spt.backbone.from_torchvision("resnet18", low_resolution=True)
encoder.fc = nn.Identity()
encoder.load_state_dict(state, strict=True)
encoder.to(device).eval()


@torch.no_grad()
def features(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=8)
    xs, ys = [], []
    for x, y in loader:
        xs.append(encoder(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


x_train, y_train = features(train_set)
x_test, y_test = features(test_set)
mean, std = x_train.mean(0), x_train.std(0, unbiased=False).clamp(min=1e-12)
x_train = ((x_train - mean) / std).to(device)
x_test = ((x_test - mean) / std).to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

classifier = nn.Linear(x_train.size(1), 10).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4, weight_decay=1e-3)
accuracies = []
for _ in range(args.epochs):
    perm = torch.randperm(x_train.size(0), device=device)
    for i in range(0, x_train.size(0), 128):
        idx = perm[i : i + 128]
        loss = F.cross_entropy(classifier(x_train[idx]), y_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        correct = (classifier(x_test).argmax(1) == y_test).float().mean().item()
    accuracies.append(100.0 * correct)

best = max(accuracies)
result = {
    "ckpt": args.ckpt,
    "encoder": args.encoder,
    "best_top1": best,
    "best_epoch": accuracies.index(best) + 1,
    "final_top1": accuracies[-1],
}
print(
    f"{args.encoder} encoder: best top-1 {best:.2f} (epoch {result['best_epoch']}), "
    f"final top-1 {accuracies[-1]:.2f}"
)
if args.output:
    with open(args.output, "w") as f:
        json.dump({**result, "history": accuracies}, f, indent=1)
