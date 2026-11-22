"""Frozen-backbone k-NN evaluation of PCL on CIFAR-10."""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from stable_pretraining.methods.pcl import PCL


@torch.inference_mode()
def extract_features(encoder, loader, device):
    features = []
    labels = []

    encoder.eval()

    for images, targets in loader:
        images = images.to(device)
        embeddings = encoder(images)
        embeddings = F.normalize(embeddings, dim=1)

        features.append(embeddings.cpu())
        labels.append(targets)

    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def knn_accuracy(
    train_features,
    train_labels,
    test_features,
    test_labels,
    device,
    k=20,
    temperature=0.07,
):
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    correct = 0
    num_classes = 10

    for start in range(0, len(test_features), 256):
        features = test_features[start : start + 256].to(device)
        targets = test_labels[start : start + 256].to(device)

        similarity = features @ train_features.T
        scores, indices = similarity.topk(k, dim=1)

        neighbors = train_labels[indices]
        weights = (scores / temperature).exp()

        class_scores = torch.zeros(features.shape[0], num_classes, device=device)
        class_scores.scatter_add_(1, neighbors, weights)

        predictions = class_scores.argmax(dim=1)
        correct += (predictions == targets).sum().item()

    return correct / len(test_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/pcl-cifar10/last.ckpt",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--random-baseline", action="store_true")
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet50"],
        default="resnet18",
    )
    parser.add_argument(
        "--queue-length",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = PCL(
        encoder_name=args.backbone,
        low_resolution=True,
        projector_dims=(512, 128),
        queue_length=args.queue_length,
        temperature=0.2,
    )

    if not args.random_baseline:
        # Trusted checkpoint produced by our own training run.
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    model.to(device)
    encoder = model.backbone.teacher

    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)

    accuracy = knn_accuracy(
        train_features,
        train_labels,
        test_features,
        test_labels,
        device,
        k=args.k,
    )

    name = "Random baseline" if args.random_baseline else "PCL"
    print(f"{name} k-NN top-1 accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
