# """Export a torchvision-style ResNet backbone from a stable-pretraining checkpoint.

# Example:
#     python examples/export_densecl_backbone.py \
#         --checkpoint /path/to/last.ckpt \
#         --output /path/to/densecl_backbone.pth

# The output is intended for downstream transfer tooling, not for resuming
# the original self-supervised training run.
# """

# from __future__ import annotations

# import argparse
# from collections.abc import Mapping
# from pathlib import Path

# import torch


# # Try longer prefixes first so nested backbones are handled correctly.
# DEFAULT_PREFIXES = (
#     "module.backbone.model.",
#     "backbone.model.",
#     "module.backbone.",
#     "model.backbone.",
#     "backbone.",
# )

# RESNET_PREFIXES = (
#     "conv1.",
#     "bn1.",
#     "layer1.",
#     "layer2.",
#     "layer3.",
#     "layer4.",
# )


# def load_checkpoint(path: Path) -> Mapping:
#     """Load a local checkpoint and return its model state dictionary."""
#     checkpoint = torch.load(
#         path,
#         map_location="cpu",
#         weights_only=False,
#     )

#     if not isinstance(checkpoint, Mapping):
#         raise TypeError(f"Expected a checkpoint dictionary, got {type(checkpoint)}")

#     state_dict = checkpoint.get("state_dict", checkpoint)

#     if not isinstance(state_dict, Mapping):
#         raise TypeError("Checkpoint 'state_dict' is not a dictionary.")

#     return state_dict


# def extract_resnet_backbone(
#     state_dict: Mapping,
#     prefix: str | None = None,
# ) -> dict[str, torch.Tensor]:
#     """Extract ResNet stem and residual layers from a checkpoint.

#     The output uses torchvision-style names such as conv1.weight and
#     layer1.0.conv1.weight, which the official DenseCL conversion script
#     expects before converting weights to Detectron2 format.
#     """
#     prefixes = (prefix,) if prefix is not None else DEFAULT_PREFIXES

#     for candidate in prefixes:
#         extracted = {}

#         for key, value in state_dict.items():
#             if not isinstance(key, str) or not key.startswith(candidate):
#                 continue

#             backbone_key = key[len(candidate) :]

#             if not backbone_key.startswith(RESNET_PREFIXES):
#                 continue

#             if not isinstance(value, torch.Tensor):
#                 continue

#             extracted[backbone_key] = value.detach().cpu()

#         if extracted:
#             break
#     else:
#         examples = [key for key in state_dict.keys() if isinstance(key, str)][:20]

#         raise ValueError(
#             "Could not find torchvision-style ResNet backbone weights. "
#             f"Tried prefixes: {prefixes}. Example checkpoint keys: {examples}. "
#             "Inspect the checkpoint keys and pass --prefix if necessary."
#         )

#     required = ("conv1.weight", "bn1.weight", "layer1.0.conv1.weight")
#     missing = [key for key in required if key not in extracted]

#     if missing:
#         raise ValueError(
#             f"Extracted weights are missing required ResNet keys: {missing}"
#         )

#     return extracted


# def main() -> None:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--checkpoint", required=True, type=Path)
#     parser.add_argument("--output", required=True, type=Path)
#     parser.add_argument(
#         "--prefix",
#         default=None,
#         help=(
#             "Exact state-dict prefix to strip, e.g. 'backbone.'. "
#             "If omitted, common prefixes are tried automatically."
#         ),
#     )
#     args = parser.parse_args()

#     if not args.checkpoint.is_file():
#         parser.error(f"Checkpoint does not exist: {args.checkpoint}")

#     state_dict = load_checkpoint(args.checkpoint)
#     backbone = extract_resnet_backbone(state_dict, args.prefix)

#     args.output.parent.mkdir(parents=True, exist_ok=True)
#     torch.save(
#         {
#             "state_dict": backbone,
#             "author": "stable-pretraining",
#             "source_checkpoint": str(args.checkpoint),
#         },
#         args.output,
#     )

#     print(f"Exported {len(backbone)} backbone tensors.")
#     print(f"Input:  {args.checkpoint}")
#     print(f"Output: {args.output}")
#     print(f"Stem shape: {tuple(backbone['conv1.weight'].shape)}")

#     # Official DenseCL detector configs use ResNet-50. But a ResNet-18/34 checkpoint is not compatible with those configs.
#     has_resnet50_depth = (
#         "layer1.2.conv1.weight" in backbone
#         and "layer2.3.conv1.weight" in backbone
#         and "layer3.5.conv1.weight" in backbone
#         and "layer4.2.conv1.weight" in backbone
#     )
#     if not has_resnet50_depth:
#         print(
#             "WARNING: the extracted backbone does not appear to be ResNet-50. "
#             "Do not load it into the official DenseCL ResNet-50 detector "
#             "configs without adapting the architecture."
#         )


# if __name__ == "__main__":
#     main()


"""Export a sequential ResNet-18 backbone from a stable-pretraining checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a checkpoint dictionary.")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint state_dict is not a dictionary.")

    # Convert the sequential backbone's names into torchvision ResNet names.
    mapping = {
        "backbone.0.": "conv1.",
        "backbone.1.": "bn1.",
        "backbone.4.": "layer1.",
        "backbone.5.": "layer2.",
        "backbone.6.": "layer3.",
        "backbone.7.": "layer4.",
    }

    backbone = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue

        for source, target in mapping.items():
            if key.startswith(source):
                backbone[target + key[len(source) :]] = value.detach().cpu()
                break

    required = (
        "conv1.weight",
        "bn1.weight",
        "layer1.0.conv1.weight",
        "layer2.0.conv1.weight",
        "layer3.0.conv1.weight",
        "layer4.0.conv1.weight",
    )
    missing = [key for key in required if key not in backbone]
    if missing:
        raise ValueError(f"Missing expected backbone weights: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": backbone,
            "author": "stable-pretraining",
            "source_checkpoint": str(args.checkpoint),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "architecture": "ResNet-18",
        },
        args.output,
    )

    print(f"Exported {len(backbone)} backbone tensors.")
    print(f"Source epoch: {checkpoint.get('epoch')}")
    print(f"Source step: {checkpoint.get('global_step')}")
    print(f"Output: {args.output}")
    print(f"Stem shape: {tuple(backbone['conv1.weight'].shape)}")


if __name__ == "__main__":
    main()
