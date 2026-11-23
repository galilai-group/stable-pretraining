from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import resnet18
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.transforms import functional as TF

ROOT = Path("data/VOC/VOCdevkit/VOC2007")
WEIGHTS = Path("/oscar/home/eslin/densecl_cifar10_backbone.pth")

CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(CLASSES)}


class VOCDataset(Dataset):
    def __init__(self, root, split="trainval"):
        self.root = Path(root)
        split_file = self.root / "ImageSets" / "Main" / f"{split}.txt"
        self.ids = split_file.read_text().split()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = Image.open(self.root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")

        xml = ET.parse(self.root / "Annotations" / f"{image_id}.xml").getroot()

        boxes, labels = [], []
        for obj in xml.findall("object"):
            name = obj.findtext("name")
            if name not in CLASS_TO_ID:
                continue

            box = obj.find("bndbox")
            xmin = float(box.findtext("xmin")) - 1
            ymin = float(box.findtext("ymin")) - 1
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(CLASS_TO_ID[name])

        image = TF.to_tensor(image)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    dataset = VOCDataset(ROOT, split="trainval")
    print("VOC trainval images:", len(dataset), flush=True)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
    )

    # Re-create the CIFAR-style ResNet-18 architecture used by DenseCL.
    resnet = resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()

    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    result = resnet.load_state_dict(checkpoint["state_dict"], strict=False)
    assert set(result.missing_keys) == {"fc.weight", "fc.bias"}, result
    assert not result.unexpected_keys, result
    print("Loaded pretrained DenseCL backbone.", flush=True)

    # spatial feature maps for detector.
    backbone = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
    )
    backbone.out_channels = 512

    anchors = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )

    model = FasterRCNN(
        backbone,
        num_classes=len(CLASSES) + 1,  # 0 is background
        rpn_anchor_generator=anchors,
        min_size=400,
        max_size=600,
    ).to(device)

    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
    )

    images, targets = next(iter(loader))
    images = [image.to(device) for image in images]
    targets = [
        {key: value.to(device) for key, value in target.items()} for target in targets
    ]

    optimizer.zero_grad(set_to_none=True)
    losses = model(images, targets)
    total_loss = sum(losses.values())

    print(
        "Losses:",
        {key: round(value.item(), 4) for key, value in losses.items()},
        flush=True,
    )
    assert torch.isfinite(total_loss), f"Non-finite loss: {total_loss.item()}"

    total_loss.backward()
    optimizer.step()

    print(
        f"SMOKE TEST PASSED: one training step, loss={total_loss.item():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
