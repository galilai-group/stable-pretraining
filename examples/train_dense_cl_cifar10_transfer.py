from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

DATA = Path("data")
WEIGHTS = Path("/oscar/home/eslin/densecl_cifar10_backbone.pth")
OUTPUT = Path("/oscar/home/eslin/densecl_cifar10_transfer.pth")
BATCH_SIZE = 128
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

train_set = datasets.CIFAR10(
    DATA, train=True, download=False, transform=train_transform
)
val_set = datasets.CIFAR10(DATA, train=False, download=False, transform=val_transform)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

model = resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
result = model.load_state_dict(checkpoint["state_dict"], strict=False)
assert set(result.missing_keys) == {"fc.weight", "fc.bias"}, result
assert not result.unexpected_keys, result

model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"loss={total_loss / len(train_loader):.4f} "
        f"val_accuracy={100 * correct / total:.2f}%",
        flush=True,
    )

torch.save(model.state_dict(), OUTPUT)
print("Saved:", OUTPUT, flush=True)
