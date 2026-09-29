import os
import torch
import timm
from typing import Optional, Tuple
from huggingface_hub import create_repo, get_token, upload_folder
from transformers import (
    ViTConfig,
    ViTModel,
    DeiTConfig,
    DeiTModel,
    SwinConfig,
    SwinModel,
    ConvNextConfig,
    ConvNextModel,
)
import safetensors.torch


def push_timm_to_hf(
    model_name: str,
    model: torch.nn.Module,
    repo_id: str,
    hf_token: Optional[str] = None,
    private: bool = False,
    validate: bool = True,
    batch_size: int = 2,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    device: Optional[str] = None,
    strict: bool = False,
) -> str:
    family_map = {
        "vit": (ViTConfig, ViTModel),
        "deit": (DeiTConfig, DeiTModel),
        "swin": (SwinConfig, SwinModel),
        "convnext": (ConvNextConfig, ConvNextModel),
    }
    hf_token = hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or get_token()
    if not hf_token:
        raise RuntimeError(
            "Hugging Face token not found. Pass hf_token or set HUGGINGFACE_HUB_TOKEN."
        )

    family = next((fam for fam in family_map if fam in model_name.lower()), None)
    repo_url = f"https://huggingface.co/{repo_id}"
    local_dir = f"./{repo_id.replace('/', '__')}"
    os.makedirs(local_dir, exist_ok=True)

    try:
        create_repo(repo_id, token=hf_token, private=private, exist_ok=True)
    except Exception as e:
        print(f"Repo creation warning: {e}")

    # Model card (README)
    readme = f"""---
tags:
- timm
- vision
- {family or "custom"}
license: apache-2.0
---

# Model: {repo_id}
- TIMM model: `{model_name}`
- TIMM version: {timm.__version__}
- Architecture: {family or "custom"}
- Converted: {"Transformers" if family else "PyTorch"}
- Example usage:
```python
from transformers import AutoModel, AutoImageProcessor
model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained("{repo_id}")"""
    with open(f"{local_dir}/README.md", "w") as f:
        f.write(readme)
    if family:
        config_cls, model_cls = family_map[family]
        image_size = _normalize_img_size(getattr(model, "img_size", 224))
        in_chans = getattr(model, "in_chans", 3)
        num_labels = getattr(model, "num_classes", 1000)
        config = config_cls(
            image_size=image_size[0] if isinstance(image_size, tuple) else image_size,
            num_channels=in_chans,
            num_labels=num_labels,
        )
        hf_model = model_cls(config)
        try:
            incompatible = hf_model.load_state_dict(model.state_dict(), strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError("State dict keys do not match the Transformers model")
        except Exception as e:
            print(f"State dict mapping failed: {e}")
            family = None

        if family and validate:
            _validate_timm_vs_hf(
                timm_model=model,
                hf_model=hf_model,
                family=family,
                batch_size=batch_size,
                atol=atol,
                rtol=rtol,
                device=device,
                strict=strict,
            )

        if family:
            safetensors.torch.save_file(
                hf_model.state_dict(), f"{local_dir}/model.safetensors"
            )
            hf_model.config.save_pretrained(local_dir)
            _save_image_processor(local_dir, image_size, family)
            upload_folder(
                repo_id=repo_id,
                folder_path=local_dir,
                token=hf_token,
                commit_message="Push TIMM-converted model",
            )
            return repo_url

    readme = readme.split("- Example usage:")[0].replace(
        "- Converted: Transformers", "- Converted: PyTorch"
    )
    readme += "\nWeights are saved in `pytorch_model.bin` for the original TIMM architecture.\n"
    with open(f"{local_dir}/README.md", "w") as f:
        f.write(readme)
    torch.save(model.state_dict(), f"{local_dir}/pytorch_model.bin")
    with open(f"{local_dir}/model_type.txt", "w") as f:
        f.write(
            f"TIMM model: {model_name}\nNo direct Transformers conversion available.\n"
        )
    upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        token=hf_token,
        commit_message="Push plain TIMM PyTorch model",
    )
    print(
        f"WARNING: {model_name} not natively supported for Transformers conversion. Uploaded as PyTorch weights."
    )
    return repo_url


def _normalize_img_size(img_size) -> Tuple[int, int]:
    if isinstance(img_size, (list, tuple)):
        if len(img_size) == 2:
            return int(img_size[0]), int(img_size[1])
        return int(img_size[0]), int(img_size[0])
    return int(img_size), int(img_size)


def _validate_timm_vs_hf(
    timm_model: torch.nn.Module,
    hf_model: torch.nn.Module,
    family: str,
    batch_size: int,
    atol: float,
    rtol: float,
    device: Optional[str],
    strict: bool,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    timm_model = timm_model.to(device).eval()
    hf_model = hf_model.to(device).eval()

    torch.manual_seed(42)
    img_size = _normalize_img_size(getattr(timm_model, "img_size", 224))
    h, w = img_size
    x = torch.rand(batch_size, 3, h, w, device=device, dtype=torch.float32)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    x_norm = (x - mean) / std

    with torch.no_grad():
        feats_timm = _extract_features_timm(timm_model, x_norm, family)
        feats_hf = _extract_features_hf(hf_model, x_norm, family)

    if feats_timm.shape != feats_hf.shape:
        min_last = min(feats_timm.shape[-1], feats_hf.shape[-1])
        feats_timm = feats_timm[..., :min_last]
        feats_hf = feats_hf[..., :min_last]

    cos_sim, max_abs_diff = _compare_tensors(feats_timm, feats_hf)
    print(
        f"[Sanity Check] Cosine similarity: {cos_sim:.6f}, Max abs diff: {max_abs_diff:.6g}"
    )
    thresh = atol + rtol * feats_timm.abs().max().item()
    if not (cos_sim >= 0.999 or max_abs_diff <= thresh):
        msg = f"Sanity check failed: cosine={cos_sim:.6f}, max_abs_diff={max_abs_diff:.6g}, thresh={thresh:.6g}"
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)


def _extract_features_timm(
    model: torch.nn.Module, x: torch.Tensor, family: str
) -> torch.Tensor:
    if family in ("vit", "deit"):
        out = model.forward_features(x)
        if isinstance(out, dict):
            if "cls_token" in out:
                return out["cls_token"]
            if "x" in out and out["x"].ndim == 3:
                return out["x"][:, 0]
            if "x_norm_cls" in out:
                return out["x_norm_cls"]
            raise RuntimeError("Unexpected forward_features dict for ViT/DeiT.")
        return out[:, 0] if out.ndim == 3 else out
    elif family == "swin":
        out = model.forward_features(x)
        if isinstance(out, dict) and "x" in out:
            tokens = out["x"]
            return tokens.mean(dim=1)
        return out.mean(dim=1) if out.ndim == 3 else out.mean(dim=[2, 3])
    elif family == "convnext":
        features = model.forward_features(x)
        if isinstance(features, dict) and "x" in features:
            features = features["x"]
        return features.mean(dim=[2, 3])
    raise NotImplementedError(f"Sanity check not implemented for {family}")


def _extract_features_hf(
    model: torch.nn.Module, x: torch.Tensor, family: str
) -> torch.Tensor:
    if family in ("vit", "deit"):
        out = model(x)
        return out.last_hidden_state[:, 0]
    elif family == "swin":
        out = model(x)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state.mean(dim=1)
    elif family == "convnext":
        out = model(x)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state.mean(dim=[2, 3])
    raise NotImplementedError(f"Sanity check not implemented for {family}")


def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a = a.view(a.size(0), -1).float().cpu()
    b = b.view(b.size(0), -1).float().cpu()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return float(cos.mean()), float((a - b).abs().max())


def _save_image_processor(
    local_dir: str, img_size: Tuple[int, int], family: str
) -> None:
    h, w = img_size
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    try:
        import json

        proc_cfg = {
            "_class_name": "AutoImageProcessor",
            "do_resize": True,
            "size": {"height": h, "width": w},
            "do_center_crop": False,
            "do_normalize": True,
            "image_mean": image_mean,
            "image_std": image_std,
        }
        with open(os.path.join(local_dir, "preprocessor_config.json"), "w") as f:
            json.dump(proc_cfg, f, indent=2)
    except Exception as e:
        print(f"Image processor fallback save warning: {e}")


if __name__ == "__main__":
    timm_model_name = "vit_base_patch16_224"
    model = timm.create_model(timm_model_name, pretrained=True)
    repo_id = "your-username/my-vit-base-patch16-224"
    url = push_timm_to_hf(
        timm_model_name, model, repo_id, private=True, validate=True, strict=False
    )
    print(f"Model pushed to: {url}")
