"""Train CluMuT on the Hugging Face CIFAR-10 dataset.

The dataset is cached under ``--data-root`` on first run.
Use ``--max-epochs 1 --limit-train-batches 5`` for a short smoke run.
"""

import argparse
import re
from functools import partial
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import stable_pretraining as spt


class TwoViewTransform:
    """Apply the same augmentation pipeline independently to two views."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image = sample["image"]
        return {
            "views": (self.transform(image), self.transform(image)),
            "label": sample["label"],
        }


class SingleViewTransform:
    """Return one deterministic normalized image for linear evaluation."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return {"image": self.transform(sample["image"]), "label": sample["label"]}


class EncoderCheckpointCallback(pl.Callback):
    """Save encoder-only snapshots at fixed pretraining epoch intervals."""

    def __init__(self, directory: Path, interval: int):
        self.directory = directory
        self.interval = interval
        self.saved_checkpoints = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.interval != 0 and epoch != trainer.max_epochs:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"encoder_epoch_{epoch:03d}.pt"
        state = {
            name: value.detach().cpu()
            for name, value in pl_module.backbone.state_dict().items()
        }
        torch.save(state, path)
        self.saved_checkpoints.append((epoch, path))
        print(f"Saved encoder checkpoint: {path}")


class TrainingCheckpointCallback(pl.Callback):
    """Save one rolling full checkpoint for resuming training."""

    def __init__(self, directory: Path, interval: int):
        self.directory = directory
        self.interval = interval

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.interval != 0 and epoch != trainer.max_epochs:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "last.ckpt"
        trainer.save_checkpoint(str(path))
        print(f"Saved resumable training checkpoint: {path}")


class CluMuTDiagnosticsCallback(pl.Callback):
    """Report CluMuT loss, assignment, correlation, and gradient diagnostics."""

    def __init__(self, log_every: int = 100, initial_batches: int = 3):
        self.log_every = log_every
        self.initial_batches = initial_batches
        self.gradient_squared_norms = {}
        self.hooks = []

    def on_fit_start(self, trainer, pl_module):
        def capture_gradient(name):
            def hook(gradient):
                squared_norm = gradient.detach().float().square().sum()
                self.gradient_squared_norms[name] = (
                    self.gradient_squared_norms.get(name, 0.0) + squared_norm
                )

            return hook

        for name, parameter in pl_module.projector.named_parameters():
            self.hooks.append(parameter.register_hook(capture_gradient("projector")))
        self.hooks.append(
            pl_module.prototypes.register_hook(capture_gradient("prototypes"))
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.gradient_squared_norms.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not isinstance(outputs, dict):
            return

        epoch = trainer.current_epoch + 1
        step = trainer.global_step
        mutual = outputs["mutual_targeting_loss"]
        identitization = outputs["identitization_loss"]
        total = outputs["loss"]
        h_a, h_b = outputs["h_a"], outputs["h_b"]
        q_a, q_b = outputs["q_a"], outputs["q_b"]
        correlation = outputs["correlation"]

        finite_losses = torch.isfinite(torch.stack((mutual, identitization, total)))
        if not finite_losses.all():
            print(f"WARNING epoch={epoch} step={step}: non-finite CluMuT loss")
        if not torch.isfinite(q_a).all() or not torch.isfinite(q_b).all():
            print(f"WARNING epoch={epoch} step={step}: non-finite assignments")
        if correlation is not None and not torch.isfinite(correlation).all():
            print(f"WARNING epoch={epoch} step={step}: non-finite correlation matrix")

        entropy_a = -(q_a * q_a.clamp_min(torch.finfo(q_a.dtype).tiny).log()).sum(-1)
        entropy_b = -(q_b * q_b.clamp_min(torch.finfo(q_b.dtype).tiny).log()).sum(-1)
        concentration_a = (q_a.max(dim=-1).values >= 0.99).float().mean()
        concentration_b = (q_b.max(dim=-1).values >= 0.99).float().mean()
        if min(entropy_a.mean().item(), entropy_b.mean().item()) < 0.1:
            print(f"WARNING epoch={epoch} step={step}: assignment entropy is very low")
        if max(concentration_a.item(), concentration_b.item()) >= 0.9:
            print(
                f"WARNING epoch={epoch} step={step}: most samples assign >=99% "
                "probability to one prototype"
            )

        should_log = step <= self.initial_batches or step % self.log_every == 0
        if not should_log:
            return

        with torch.no_grad():
            diag = correlation.diagonal()
            off_mask = ~torch.eye(
                correlation.shape[0], device=correlation.device, dtype=torch.bool
            )
            h_a_std = h_a.float().std().item()
            h_b_std = h_b.float().std().item()
            h_a_prototype_std = h_a.float().std(dim=0).mean().item()
            h_b_prototype_std = h_b.float().std(dim=0).mean().item()
            diagonal_mean = diag.mean().item()
            diagonal_min = diag.min().item()
            diagonal_max = diag.max().item()
            off_diagonal = correlation[off_mask]
            off_diagonal_mean = off_diagonal.abs().mean().item()
            off_diagonal_min = off_diagonal.min().item()
            off_diagonal_max = off_diagonal.max().item()
        gradient_norms = {
            name: squared_norm.sqrt().item()
            for name, squared_norm in self.gradient_squared_norms.items()
        }
        print(
            f"CluMuT diagnostic epoch={epoch} step={step} "
            f"batch={q_a.shape[0]} mutual={mutual.item():.6g} "
            f"identitization={identitization.item():.6g} total={total.item():.6g} "
            f"h_a={tuple(h_a.shape)} h_b={tuple(h_b.shape)} "
            f"q_a={tuple(q_a.shape)} q_b={tuple(q_b.shape)} "
            f"h_a_std={h_a_std:.6f} h_b_std={h_b_std:.6f} "
            f"h_a_mean_prototype_std={h_a_prototype_std:.6f} "
            f"h_b_mean_prototype_std={h_b_prototype_std:.6f} "
            f"q_a_range=({q_a.min().item():.4g},{q_a.max().item():.4g}) "
            f"q_b_range=({q_b.min().item():.4g},{q_b.max().item():.4g}) "
            f"entropy=({entropy_a.mean().item():.4g},{entropy_b.mean().item():.4g}) "
            f"corr_diag_mean={diagonal_mean:.6f} "
            f"corr_diag_minmax={diagonal_min:.6f}/{diagonal_max:.6f} "
            f"corr_offdiag_abs={off_diagonal_mean:.4g} "
            f"corr_offdiag_minmax={off_diagonal_min:.6f}/{off_diagonal_max:.6f} "
            f"grad_norms={gradient_norms}"
        )

    def on_fit_end(self, trainer, pl_module):
        for hook in self.hooks:
            hook.remove()


def extract_features(encoder, loader, device):
    """Extract frozen encoder features and labels to CPU tensors."""
    features = []
    labels = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            features.append(encoder(images).cpu())
            labels.append(batch["label"].cpu())
    return torch.cat(features), torch.cat(labels)


def linear_evaluate(
    encoder,
    train_loader,
    test_loader,
    device,
    epochs,
    log_every,
    pretrain_epoch,
    seed,
    classifier_lr,
):
    """Train a linear classifier on frozen features and periodically report test accuracy."""
    if log_every < 1:
        raise ValueError("linear_eval_log_every must be positive")
    torch.manual_seed(seed)
    encoder.requires_grad_(False)
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)
    classifier = nn.Linear(train_features.shape[1], 10).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=classifier_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy = 0.0
    best_epoch = 0
    final_accuracy = None

    for epoch in range(epochs):
        classifier.train()
        permutation = torch.randperm(train_features.shape[0])
        for indices in permutation.split(train_loader.batch_size):
            features = train_features[indices].to(device)
            labels = train_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(classifier(features), labels).backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % log_every == 0 or epoch + 1 == epochs:
            classifier.eval()
            correct = 0
            with torch.no_grad():
                for features, labels in zip(
                    test_features.split(test_loader.batch_size),
                    test_labels.split(test_loader.batch_size),
                ):
                    predictions = classifier(features.to(device)).argmax(dim=1).cpu()
                    correct += (predictions == labels).sum().item()
            accuracy = correct / len(test_labels)
            final_accuracy = accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch + 1
            print(
                f"Pretraining epoch {pretrain_epoch} | "
                f"linear eval epoch {epoch + 1}/{epochs}: "
                f"CIFAR-10 test top-1 accuracy: {accuracy:.4%} | "
                f"best so far: {best_accuracy:.4%} at epoch {best_epoch}"
            )

    if final_accuracy is not None:
        print(
            f"Pretraining epoch {pretrain_epoch} | linear evaluation complete: "
            f"best test top-1 accuracy {best_accuracy:.4%} at epoch {best_epoch}; "
            f"final test top-1 accuracy {final_accuracy:.4%}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train CluMuT on CIFAR-10.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--linear-eval-epochs", type=int, default=100)
    parser.add_argument("--linear-eval-log-every", type=int, default=10)
    parser.add_argument("--linear-eval-lr", type=float, default=0.01)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--checkpoint-dir", default="./clumut_checkpoints")
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume CluMuT training from a Lightning .ckpt checkpoint.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        help="Evaluate a saved encoder without running self-supervised pretraining.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lambd",
        type=float,
        default=0.05,
        help="Off-diagonal correlation scale in the CluMuT identitization loss.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print loss, assignment, correlation, and gradient diagnostics.",
    )
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.linear_eval_lr <= 0:
        parser.error("--linear-eval-lr must be positive")
    if args.encoder_checkpoint and args.linear_eval_epochs < 1:
        parser.error("--linear-eval-epochs must be positive with --encoder-checkpoint")
    pl.seed_everything(args.seed, workers=True)

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    optim = None
    if not args.encoder_checkpoint:
        dataset = spt.data.HFDataset(
            "uoft-cs/cifar10",
            split="train",
            cache_dir=args.data_root,
            rename_columns={"img": "image"},
            transform=TwoViewTransform(transform),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

        steps_per_epoch = len(loader)
        total_steps = max(2, steps_per_epoch * args.max_epochs)
        warmup_steps = min(steps_per_epoch * 10, total_steps - 1)
        optim = {
            "optimizer": partial(
                spt.optim.LARS,
                lr=0.5,
                momentum=0.9,
                weight_decay=1e-4,
                exclude_bias_n_norm=True,
            ),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": total_steps,
                "warmup_start_lr": 0.0,
                "eta_min": 0.0005,
            },
            "interval": "step",
        }

    model = spt.CluMuT(
        encoder_name="resnet18",
        low_resolution=True,
        projector_dims=(1024, 1024),
        n_prototypes=1024,
        temperature=0.1,
        lambd=args.lambd,
        optim=optim,
    )
    checkpoint_dir = Path(args.checkpoint_dir) / f"batch_size_{args.batch_size}"
    evaluation_checkpoints = []
    if args.encoder_checkpoint:
        checkpoint_path = Path(args.encoder_checkpoint)
        if not checkpoint_path.is_file():
            parser.error(f"encoder checkpoint does not exist: {checkpoint_path}")
        match = re.search(r"encoder_epoch_(\d+)\.pt$", checkpoint_path.name)
        if match is None:
            parser.error(
                "--encoder-checkpoint filename must follow encoder_epoch_NNN.pt"
            )
        evaluation_checkpoints.append((int(match.group(1)), checkpoint_path))
    else:
        checkpoint_callback = EncoderCheckpointCallback(
            checkpoint_dir, args.checkpoint_every
        )
        training_checkpoint_callback = TrainingCheckpointCallback(
            checkpoint_dir, args.checkpoint_every
        )
        callbacks = [checkpoint_callback, training_checkpoint_callback]
        if args.diagnostics:
            callbacks.append(CluMuTDiagnosticsCallback())
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            limit_train_batches=(
                args.limit_train_batches
                if args.limit_train_batches is not None
                else 1.0
            ),
            accelerator=args.accelerator,
            callbacks=callbacks,
        )
        trainer.fit(
            model,
            train_dataloaders=loader,
            ckpt_path=args.resume_from_checkpoint,
        )
        evaluation_checkpoints = checkpoint_callback.saved_checkpoints

    if args.linear_eval_epochs > 0:
        linear_transform = SingleViewTransform(evaluation_transform)
        linear_train = spt.data.HFDataset(
            "uoft-cs/cifar10",
            split="train",
            cache_dir=args.data_root,
            rename_columns={"img": "image"},
            transform=linear_transform,
        )
        linear_test = spt.data.HFDataset(
            "uoft-cs/cifar10",
            split="test",
            cache_dir=args.data_root,
            rename_columns={"img": "image"},
            transform=linear_transform,
        )
        train_eval_loader = DataLoader(
            linear_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_eval_loader = DataLoader(
            linear_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        use_cuda = args.accelerator in ("gpu", "cuda") or (
            args.accelerator == "auto" and torch.cuda.is_available()
        )
        use_mps = args.accelerator == "mps" and torch.backends.mps.is_available()
        device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
        model.backbone.to(device)
        for pretrain_epoch, checkpoint_path in evaluation_checkpoints:
            encoder_state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model.backbone.load_state_dict(encoder_state)
            linear_evaluate(
                model.backbone,
                train_eval_loader,
                test_eval_loader,
                device,
                args.linear_eval_epochs,
                args.linear_eval_log_every,
                pretrain_epoch,
                args.seed,
                args.linear_eval_lr,
            )


if __name__ == "__main__":
    main()
