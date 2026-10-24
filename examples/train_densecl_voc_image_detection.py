from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
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
OUTPUT_DIR = Path("/oscar/home/eslin/voc_detection_checkpoints")

EPOCHS = 5
BATCH_SIZE = 2
NUM_WORKERS = 1

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
    def __init__(self, root, split):
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

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        return TF.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def box_iou(box, boxes):
    """IoU between one box and a tensor of boxes."""
    if len(boxes) == 0:
        return torch.empty(0)

    left_top = torch.maximum(box[:2], boxes[:, :2])
    right_bottom = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=1)

    area_a = (box[2:] - box[:2]).clamp(min=0).prod()
    area_b = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    return intersection / (area_a + area_b - intersection).clamp(min=1e-8)


def evaluate_map(model, loader, device):
    """VOC 2007 11-point interpolated AP at IoU >= 0.5."""
    model.eval()
    ground_truth = {c: {} for c in range(1, len(CLASSES) + 1)}
    detections = {c: [] for c in range(1, len(CLASSES) + 1)}

    with torch.no_grad():
        for images, targets in loader:
            predictions = model([image.to(device) for image in images])

            for target, pred in zip(targets, predictions):
                image_id = int(target["image_id"].item())
                gt_boxes = target["boxes"]
                gt_labels = target["labels"]

                for c in ground_truth:
                    mask = gt_labels == c
                    boxes = gt_boxes[mask]
                    ground_truth[c][image_id] = {
                        "boxes": boxes,
                        "matched": torch.zeros(len(boxes), dtype=torch.bool),
                    }

                for box, label, score in zip(
                    pred["boxes"].cpu(),
                    pred["labels"].cpu(),
                    pred["scores"].cpu(),
                ):
                    c = int(label)
                    if c in detections:
                        detections[c].append((float(score), image_id, box))

    aps = []
    for c in ground_truth:
        total_gt = sum(len(item["boxes"]) for item in ground_truth[c].values())
        if total_gt == 0:
            continue

        ranked = sorted(detections[c], key=lambda x: x[0], reverse=True)
        tp = np.zeros(len(ranked))
        fp = np.zeros(len(ranked))

        for i, (_, image_id, box) in enumerate(ranked):
            gt = ground_truth[c][image_id]
            overlaps = box_iou(box, gt["boxes"])

            if len(overlaps) and overlaps.max().item() >= 0.5:
                best = int(overlaps.argmax())
                if not gt["matched"][best]:
                    tp[i] = 1
                    gt["matched"][best] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        if len(ranked) == 0:
            aps.append(0.0)
            continue

        recall = np.cumsum(tp) / total_gt
        precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)

        # VOC 2007 AP: average maximum precision at 11 recall levels.
        ap = np.mean(
            [
                precision[recall >= threshold].max()
                if np.any(recall >= threshold)
                else 0.0
                for threshold in np.linspace(0, 1, 11)
            ]
        )
        aps.append(float(ap))

    model.train()
    return float(np.mean(aps)) if aps else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    train_set = VOCDataset(ROOT, "train")
    val_set = VOCDataset(ROOT, "val")
    print(f"Train images: {len(train_set)}", flush=True)
    print(f"Validation images: {len(val_set)}", flush=True)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    resnet = resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()

    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    result = resnet.load_state_dict(checkpoint["state_dict"], strict=False)
    assert set(result.missing_keys) == {"fc.weight", "fc.bias"}, result
    assert not result.unexpected_keys, result
    print("Loaded pretrained DenseCL backbone.", flush=True)

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
        num_classes=len(CLASSES) + 1,
        rpn_anchor_generator=anchors,
        min_size=400,
        max_size=600,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [
                {k: v.to(device) for k, v in target.items()} for target in targets
            ]

            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            total_loss = sum(losses.values())

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")

            total_loss.backward()
            optimizer.step()
            loss_sum += total_loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} "
                    f"step {step}/{len(train_loader)} "
                    f"loss={total_loss.item():.4f}",
                    flush=True,
                )

        avg_loss = loss_sum / len(train_loader)
        map50 = evaluate_map(model, val_loader, device)

        print(
            f"Epoch {epoch}/{EPOCHS} complete: "
            f"train_loss={avg_loss:.4f} val_mAP50={map50:.4f}",
            flush=True,
        )

        save_path = OUTPUT_DIR / f"epoch_{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "val_map50": map50,
            },
            save_path,
        )
        print("Saved:", save_path, flush=True)


if __name__ == "__main__":
    main()
