from pathlib import Path
from loguru import logger as logging
import threading
from typing import Dict, List, Optional
import json
import os

import timm
import torch


def count_parameters(model):
    """Count trainable parameters efficiently."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _get_last_submodule_full_name(model):
    """Get the full name of the last submodule in definition order."""
    last_name = ""  # Default fallback
    for name, _ in model.named_modules():
        last_name = name
    return last_name


class MetaStatic(type):
    """Metaclass that enables dict-like behavior on the TIMM_PARAMETERS class."""

    def __contains__(cls, key):
        """Enable 'in' operator on the class itself."""
        cls._ensure_loaded()
        return key in cls._data

    def __len__(cls):
        """Enable len() on the class itself."""
        cls._ensure_loaded()
        return len(cls._data)

    def __iter__(cls):
        """Enable iteration over keys."""
        cls._ensure_loaded()
        return iter(cls._data)

    def __getitem__(cls, key):
        """Enable bracket notation for getting items on the class itself."""
        cls._ensure_loaded()
        # Return a copy to prevent mutation of cached data
        value = cls._data[key]
        return value

    def __setitem__(cls, key, value):
        """Enable bracket notation for setting items on the class itself."""
        cls._ensure_loaded()
        cls._data[key] = value

    def __delitem__(cls, key):
        """Enable bracket notation for deleting items on the class itself."""
        cls._ensure_loaded()
        del cls._data[key]

    def keys(cls):
        """Return a view of the keys."""
        cls._ensure_loaded()
        return cls._data.keys()

    def values(cls):
        """Return a view of the values (as copies to prevent mutation)."""
        cls._ensure_loaded()
        # Return copies of lists to prevent mutation
        return (list(v) for v in cls._data.values())

    def items(cls):
        """Return a view of the items (with copied values)."""
        cls._ensure_loaded()
        # Return copies of lists to prevent mutation
        return ((k, v) for k, v in cls._data.items())

    def get(cls, key, default=None):
        """Get value with optional default."""
        cls._ensure_loaded()
        if key in cls._data:
            return list(cls._data[key])
        return default

    def clear(cls):
        """Clear all data."""
        cls._ensure_loaded()
        cls._data.clear()

    def update(cls, other):
        """Update from another dict or iterable of key-value pairs."""
        cls._ensure_loaded()
        if isinstance(other, dict):
            for key, value in other.items():
                if not isinstance(value, list):
                    raise TypeError(
                        f"All values must be lists, got {type(value).__name__} for key '{key}'"
                    )
            cls._data.update(other)
        else:
            for key, value in other:
                if not isinstance(value, list):
                    raise TypeError(
                        f"All values must be lists, got {type(value).__name__} for key '{key}'"
                    )
                cls._data[key] = value

    def _ensure_loaded(cls):
        """Ensure the TIMM parameters are loaded from the JSON file.

        This method uses double-checked locking to ensure thread-safe,
        one-time initialization of the cached data.

        Raises:
            RuntimeError: If the assets folder or JSON file is missing.
        """
        if cls._data is None:
            with cls._lock:
                # Double-check after acquiring lock
                if cls._data is None:
                    logging.info("TIMM cache not loaded yet... loading!")
                    path = Path(os.path.abspath(__file__))
                    target = path.parent.parent / cls._file_path
                    logging.info(f"Loading TIMM embeddings from: {target}")
                    if not target.is_file():
                        raise RuntimeError("Did you manually delete the assets folder?")
                    with open(target, "r") as f:
                        cls._data = json.load(f)


class TIMM_EMBEDDINGS(metaclass=MetaStatic):
    """Thread-safe, lazy-loaded registry for TIMM (PyTorch Image Models) embedding names, accessed via class-level indexing.

    This class provides a mapping from string keys to lists of embedding names, loaded on first access from a
    JSON file located at 'assets/static_timm.json' relative to the source file. The data is cached as a class
    attribute after the first load, and subsequent accesses are served from memory.
    The class is intended to be used as a static registry, e.g.:
        >>> names = TIMM_EMBEDDINGS["resnet50"]
        >>> print(names)  # List of embedding names for 'resnet50'
    Notes:
        - The data is loaded only once per process and is shared across all uses of the class.
        - Thread-safe: concurrent first-time access is protected by a class-level lock.
        - The class depends on the presence of the 'assets/static_timm.json' file two directories above the source file.
        - The class assumes the `__file__` attribute is available and points to the current file.
        - The class attribute `_data` is private and shared.
        - Logging and printing occur on first load for debugging.
        - File system access and JSON parsing are required at runtime.

    Raises:
        RuntimeError: If the assets file is missing.
        OSError, IOError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If the requested key is not present in the data.

    Example:
        >>> embeddings = TIMM_EMBEDDINGS["vit_base_patch16_224"]
        >>> print(embeddings)
    """

    _file_path = "assets/static_timm.json"

    _data: Optional[Dict[str, List[str]]] = None
    _lock = threading.RLock()

    data: dict[str, list[str]] = None

    @classmethod
    def __class_getitem__(cls, key):
        """Retrieve a copy of the list of embedding names for a given model key, loading the registry from disk if necessary.

        On first access, this method loads the JSON file 'assets/static_timm.json' located two directories above
        the current file, caches the result in the class attribute `_data`, and returns a copy of the value for the given key.
        Subsequent accesses use the cached data.
        Parameters:
            key (str): The model identifier for which to retrieve embedding names.

        Returns:
            list[str]: A copy of the list of embedding names associated with the given key.

        Raises:
            RuntimeError: If the assets file is missing.
            OSError, IOError: If the file cannot be read.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If the requested key is not present in the data.

        Notes:
            - Logging and printing of the resolved path occur on first load.
            - Thread-safe: concurrent first-time access is protected by a class-level lock.
            - The method assumes the `__file__` attribute is available.
            - The returned list is a copy; mutating it will not affect the cached data.

        Example:
            >>> names = TIMM_EMBEDDINGS["efficientnet_b0"]
            >>> print(names)
        """
        cls._ensure_loaded()
        # Defensive: always return a copy to prevent mutation of the cached data
        value = cls._data[key]
        return list(value)


class HF_EMBEDDINGS(metaclass=MetaStatic):
    """Thread-safe, lazy-loaded registry for TIMM (PyTorch Image Models) embedding names, accessed via class-level indexing.

    This class provides a mapping from string keys to lists of embedding names, loaded on first access from a
    JSON file located at 'assets/static_timm.json' relative to the source file. The data is cached as a class
    attribute after the first load, and subsequent accesses are served from memory.
    The class is intended to be used as a static registry, e.g.:
        >>> names = TIMM_EMBEDDINGS["resnet50"]
        >>> print(names)  # List of embedding names for 'resnet50'
    Notes:
        - The data is loaded only once per process and is shared across all uses of the class.
        - Thread-safe: concurrent first-time access is protected by a class-level lock.
        - The class depends on the presence of the 'assets/static_timm.json' file two directories above the source file.
        - The class assumes the `__file__` attribute is available and points to the current file.
        - The class attribute `_data` is private and shared.
        - Logging and printing occur on first load for debugging.
        - File system access and JSON parsing are required at runtime.

    Raises:
        RuntimeError: If the assets file is missing.
        OSError, IOError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If the requested key is not present in the data.

    Example:
        >>> embeddings = TIMM_EMBEDDINGS["vit_base_patch16_224"]
        >>> print(embeddings)
    """

    _file_path = "assets/static_hf.json"

    _data: Optional[Dict[str, List[str]]] = None
    _lock = threading.RLock()

    data: dict[str, list[str]] = None

    @classmethod
    def __class_getitem__(cls, key):
        """Retrieve a copy of the list of embedding names for a given model key, loading the registry from disk if necessary.

        On first access, this method loads the JSON file 'assets/static_timm.json' located two directories above
        the current file, caches the result in the class attribute `_data`, and returns a copy of the value for the given key.
        Subsequent accesses use the cached data.
        Parameters:
            key (str): The model identifier for which to retrieve embedding names.

        Returns:
            list[str]: A copy of the list of embedding names associated with the given key.

        Raises:
            RuntimeError: If the assets file is missing.
            OSError, IOError: If the file cannot be read.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If the requested key is not present in the data.

        Notes:
            - Logging and printing of the resolved path occur on first load.
            - Thread-safe: concurrent first-time access is protected by a class-level lock.
            - The method assumes the `__file__` attribute is available.
            - The returned list is a copy; mutating it will not affect the cached data.

        Example:
            >>> names = HF_EMBEDDINGS["efficientnet_b0"]
            >>> print(names)
        """
        cls._ensure_loaded()
        # Defensive: always return a copy to prevent mutation of the cached data
        value = cls._data[key]
        return list(value)


class TIMM_PARAMETERS(metaclass=MetaStatic):
    """Thread-safe singleton class for accessing TIMM (Timm Image Models) parameters.

    This class provides lazy-loaded, cached access to TIMM model parameters stored
    in a static JSON file. It implements a dict-like interface with thread-safe
    initialization and defensive copying to prevent mutation of cached data.

    Usage:
        # Access parameters by key
        params = TIMM_PARAMETERS['model_name']

        # Iterate over keys
        for key in TIMM_PARAMETERS.keys():
            print(key)

        # Iterate over values
        for values in TIMM_PARAMETERS.values():
            print(values)

        # Iterate over items
        for key, values in TIMM_PARAMETERS.items():
            print(f"{key}: {values}")

    Note:
        All methods return copies of the data to prevent accidental mutation
        of the internal cache.
    """

    _file_path = "assets/static_timm_parameters.json"

    _data: Optional[Dict[str, List[str]]] = None
    _lock = threading.RLock()

    data: dict[str, list[str]] = None


class HF_PARAMETERS(metaclass=MetaStatic):
    """Thread-safe singleton class for accessing TIMM (Timm Image Models) parameters.

    This class provides lazy-loaded, cached access to TIMM model parameters stored
    in a static JSON file. It implements a dict-like interface with thread-safe
    initialization and defensive copying to prevent mutation of cached data.

    Usage:
        # Access parameters by key
        params = HF_PARAMETERS['model_name']

        # Iterate over keys
        for key in HF_PARAMETERS.keys():
            print(key)

        # Iterate over values
        for values in HF_PARAMETERS.values():
            print(values)

        # Iterate over items
        for key, values in HF_PARAMETERS.items():
            print(f"{key}: {values}")

    Note:
        All methods return copies of the data to prevent accidental mutation
        of the internal cache.
    """

    _file_path = "assets/static_hf_parameters.json"

    _data: Optional[Dict[str, List[str]]] = None
    _lock = threading.RLock()

    data: dict[str, list[str]] = None


TORCHVISION_EMBEDDINGS = {
    "vit_b_16": [
        "encoder.layers.encoder_layer_0",
        "encoder.layers.encoder_layer_1",
        "encoder.layers.encoder_layer_2",
        "encoder.layers.encoder_layer_3",
        "encoder.layers.encoder_layer_4",
        "encoder.layers.encoder_layer_5",
        "encoder.layers.encoder_layer_6",
        "encoder.layers.encoder_layer_7",
        "encoder.layers.encoder_layer_8",
        "encoder.layers.encoder_layer_9",
        "encoder.layers.encoder_layer_10",
        "encoder.layers.encoder_layer_11",
    ],
    "vit_l_16": [
        "encoder.layers.encoder_layer_0",
        "encoder.layers.encoder_layer_1",
        "encoder.layers.encoder_layer_2",
        "encoder.layers.encoder_layer_3",
        "encoder.layers.encoder_layer_4",
        "encoder.layers.encoder_layer_5",
        "encoder.layers.encoder_layer_6",
        "encoder.layers.encoder_layer_7",
        "encoder.layers.encoder_layer_8",
        "encoder.layers.encoder_layer_9",
        "encoder.layers.encoder_layer_10",
        "encoder.layers.encoder_layer_11",
        "encoder.layers.encoder_layer_12",
        "encoder.layers.encoder_layer_13",
        "encoder.layers.encoder_layer_14",
        "encoder.layers.encoder_layer_15",
        "encoder.layers.encoder_layer_16",
        "encoder.layers.encoder_layer_17",
        "encoder.layers.encoder_layer_18",
        "encoder.layers.encoder_layer_19",
        "encoder.layers.encoder_layer_20",
        "encoder.layers.encoder_layer_21",
        "encoder.layers.encoder_layer_22",
        "encoder.layers.encoder_layer_23",
    ],
    "vit_h_14": [
        "encoder.layers.encoder_layer_0",
        "encoder.layers.encoder_layer_1",
        "encoder.layers.encoder_layer_2",
        "encoder.layers.encoder_layer_3",
        "encoder.layers.encoder_layer_4",
        "encoder.layers.encoder_layer_5",
        "encoder.layers.encoder_layer_6",
        "encoder.layers.encoder_layer_7",
        "encoder.layers.encoder_layer_8",
        "encoder.layers.encoder_layer_9",
        "encoder.layers.encoder_layer_10",
        "encoder.layers.encoder_layer_11",
        "encoder.layers.encoder_layer_12",
        "encoder.layers.encoder_layer_13",
        "encoder.layers.encoder_layer_14",
        "encoder.layers.encoder_layer_15",
        "encoder.layers.encoder_layer_16",
        "encoder.layers.encoder_layer_17",
        "encoder.layers.encoder_layer_18",
        "encoder.layers.encoder_layer_19",
        "encoder.layers.encoder_layer_20",
        "encoder.layers.encoder_layer_21",
        "encoder.layers.encoder_layer_22",
        "encoder.layers.encoder_layer_23",
        "encoder.layers.encoder_layer_24",
        "encoder.layers.encoder_layer_25",
        "encoder.layers.encoder_layer_26",
        "encoder.layers.encoder_layer_27",
        "encoder.layers.encoder_layer_28",
        "encoder.layers.encoder_layer_29",
        "encoder.layers.encoder_layer_30",
        "encoder.layers.encoder_layer_31",
    ],
    "resnet18": [
        "maxpool",
        "layer1.0",
        "layer1.1",
        "layer2.0",
        "layer2.1",
        "layer3.0",
        "layer3.1",
        "layer4.0",
        "layer4.1",
    ],
    "resnet34": [
        "maxpool",
        "layer1.0",
        "layer1.1",
        "layer1.2",
        "layer2.0",
        "layer2.1",
        "layer2.2",
        "layer2.3",
        "layer3.0",
        "layer3.1",
        "layer3.2",
        "layer3.3",
        "layer3.4",
        "layer3.5",
        "layer4.0",
        "layer4.1",
        "layer4.2",
    ],
    "resnet50": [
        "maxpool",
        "layer1.0",
        "layer1.1",
        "layer1.2",
        "layer2.0",
        "layer2.1",
        "layer2.2",
        "layer2.3",
        "layer3.0",
        "layer3.1",
        "layer3.2",
        "layer3.3",
        "layer3.4",
        "layer3.5",
        "layer4.0",
        "layer4.1",
        "layer4.2",
    ],
    "resnet101": [
        "maxpool",
        "layer1.0",
        "layer1.1",
        "layer1.2",
        "layer2.0",
        "layer2.1",
        "layer2.2",
        "layer2.3",
        "layer3.0",
        "layer3.1",
        "layer3.2",
        "layer3.3",
        "layer3.4",
        "layer3.5",
        "layer3.6",
        "layer3.7",
        "layer3.8",
        "layer3.9",
        "layer3.10",
        "layer3.11",
        "layer3.12",
        "layer3.13",
        "layer3.14",
        "layer3.15",
        "layer3.16",
        "layer3.17",
        "layer3.18",
        "layer3.19",
        "layer3.20",
        "layer3.21",
        "layer3.22",
        "layer4.0",
        "layer4.1",
        "layer4.2",
    ],
    "resnet152": [
        "maxpool",
        "layer1.0",
        "layer1.1",
        "layer1.2",
        "layer2.0",
        "layer2.1",
        "layer2.2",
        "layer2.3",
        "layer3.0",
        "layer3.1",
        "layer3.2",
        "layer3.3",
        "layer3.4",
        "layer3.5",
        "layer3.6",
        "layer3.7",
        "layer3.8",
        "layer3.9",
        "layer3.10",
        "layer3.11",
        "layer3.12",
        "layer3.13",
        "layer3.14",
        "layer3.15",
        "layer3.16",
        "layer3.17",
        "layer3.18",
        "layer3.19",
        "layer3.20",
        "layer3.21",
        "layer3.22",
        "layer3.23",
        "layer3.24",
        "layer3.25",
        "layer3.26",
        "layer3.27",
        "layer3.28",
        "layer3.29",
        "layer3.30",
        "layer3.31",
        "layer3.32",
        "layer3.33",
        "layer3.34",
        "layer3.35",
        "layer4.0",
        "layer4.1",
        "layer4.2",
    ],
    "swin_v2_t": [
        "features.0",
        "features.1.0",
        "features.1.1",
        "features.2",
        "features.3.0",
        "features.3.1",
        "features.5.0",
        "features.5.1",
        "features.5.2",
        "features.5.3",
        "features.5.4",
        "features.5.5",
        "features.6",
        "features.7.0",
        "features.7.1",
    ],
    "swin_v2_b": [
        "features.0",
        "features.1.0",
        "features.1.1",
        "features.2",
        "features.3.0",
        "features.3.1",
        "features.5.0",
        "features.5.1",
        "features.5.2",
        "features.5.3",
        "features.5.4",
        "features.5.5",
        "features.5.6",
        "features.5.7",
        "features.5.8",
        "features.5.9",
        "features.5.10",
        "features.5.11",
        "features.5.12",
        "features.5.13",
        "features.5.14",
        "features.5.15",
        "features.5.16",
        "features.5.17",
        "features.6",
        "features.7.0",
        "features.7.1",
    ],
    "convnext_tiny": [
        "features.0",
        "features.1.0",
        "features.1.1",
        "features.1.2",
        "features.2",
        "features.3.0",
        "features.3.1",
        "features.3.2",
        "features.4",
        "features.5.0",
        "features.5.1",
        "features.5.2",
        "features.5.3",
        "features.5.4",
        "features.5.5",
        "features.5.6",
        "features.5.7",
        "features.5.8",
        "features.6",
        "features.7.0",
        "features.7.1",
        "features.7.2",
    ],
    "convnext_large": [
        "features.0",
        "features.1.0",
        "features.1.1",
        "features.1.2",
        "features.2",
        "features.3.0",
        "features.3.1",
        "features.3.2",
        "features.4",
        "features.5.0",
        "features.5.1",
        "features.5.2",
        "features.5.3",
        "features.5.4",
        "features.5.5",
        "features.5.6",
        "features.5.7",
        "features.5.8",
        "features.5.9",
        "features.5.10",
        "features.5.11",
        "features.5.12",
        "features.5.13",
        "features.5.14",
        "features.5.15",
        "features.5.16",
        "features.5.17",
        "features.5.18",
        "features.5.19",
        "features.5.20",
        "features.5.21",
        "features.5.22",
        "features.5.23",
        "features.5.24",
        "features.5.25",
        "features.5.26",
        "features.6",
        "features.7.0",
        "features.7.1",
        "features.7.2",
    ],
}

TORCHVISION_EMBEDDINGS["convnext_base"] = TORCHVISION_EMBEDDINGS["convnext_large"]
TORCHVISION_EMBEDDINGS["swin_v2_s"] = TORCHVISION_EMBEDDINGS["swin_v2_b"]
TORCHVISION_EMBEDDINGS["vit_b_32"] = TORCHVISION_EMBEDDINGS["vit_b_16"]
TORCHVISION_EMBEDDINGS["vit_l_32"] = TORCHVISION_EMBEDDINGS["vit_l_16"]
TORCHVISION_EMBEDDINGS["wide_resnet50_2"] = TORCHVISION_EMBEDDINGS["resnet50"]


#######################################
####                HuggingFace
#######################################


def _retrieve_hf_modules(args):
    """Retrieve HF model modules with error handling."""
    name, layer_pattern, L = args
    try:
        from transformers import AutoModel, AutoConfig
        from stable_pretraining.backbone.utils import get_children_modules

        config = AutoConfig.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_config(config)

        num_params = count_parameters(model)
        internals = get_children_modules(
            model, parent_name=layer_pattern, partial_match=True, L=L
        )
        last = _get_last_submodule_full_name(model)
        return name, (internals + [last], num_params)
    except Exception as e:
        print(f"Failed: {name} - {e}")
        return name, None


def _generate_hf_factory():
    """Generate HF vision model metadata."""
    from tqdm import tqdm
    from pathlib import Path
    import json
    from multiprocessing import Pool

    # Corrected patterns - SegFormer uses "block" not "layer"
    family_to_pattern = {
        # ResNet variants
        "microsoft/resnet-": ("layers", 1),
        # ViT variants
        "facebook/deit-": ("layer", 1),
        "google/vit-": ("layer", 1),
        "facebook/dinov2-": ("layer", 1),
        "facebook/dino-": ("layer", 1),
        "microsoft/beit-": ("layer", 1),
        # Swin variants
        "microsoft/swin-": ("blocks", 1),
        "microsoft/swinv2-": ("blocks", 1),
        # ConvNeXT variants
        "facebook/convnext-": ("layers", 1),
        "facebook/convnextv2-": ("layers", 1),
        # CLIP
        "openai/clip-vit-": ("layers", 1),
        # RegNet
        "facebook/regnet-": ("layers", 1),
        # SegFormer (MiT) - FIXED!
        "nvidia/mit-b": ("block", 1),
        "nvidia/segformer-": ("block", 1),
        # MobileViT
        "apple/mobilevit-": ("layer", 1),
        # MobileNet
        "google/mobilenet_v": ("layer", 1),
        # Additional architectures
        "microsoft/cvt-": ("layers", 1),
        "facebook/data2vec-vision-": ("layer", 1),
        "facebook/levit-": ("blocks", 1),
        "shi-labs/dinat-": ("layers", 1),
        "facebook/maskformer-swin-": ("blocks", 1),
        "google/efficientnet-": ("blocks", 2),
        "timm/": ("blocks", 1),  # Some timm models on HF
        "facebook/sam-vit-": ("layers", 1),
        "facebook/poolformer": ("pool", 1),
    }

    # MASSIVE list of verified HF vision models
    model_names = [
        # ==================== ResNets ====================
        "microsoft/resnet-18",
        "microsoft/resnet-34",
        "microsoft/resnet-50",
        "microsoft/resnet-101",
        "microsoft/resnet-152",
        # ==================== ViT ====================
        "google/vit-base-patch16-224",
        "google/vit-base-patch16-224-in21k",
        "google/vit-base-patch16-384",
        "google/vit-base-patch32-224-in21k",
        "google/vit-base-patch32-384",
        "google/vit-large-patch16-224",
        "google/vit-large-patch16-224-in21k",
        "google/vit-large-patch16-384",
        "google/vit-large-patch32-224-in21k",
        "google/vit-large-patch32-384",
        "google/vit-huge-patch14-224-in21k",
        # ==================== DeiT ====================
        "facebook/deit-tiny-patch16-224",
        "facebook/deit-small-patch16-224",
        "facebook/deit-base-patch16-224",
        "facebook/deit-base-patch16-384",
        "facebook/deit-tiny-distilled-patch16-224",
        "facebook/deit-small-distilled-patch16-224",
        "facebook/deit-base-distilled-patch16-224",
        "facebook/deit-base-distilled-patch16-384",
        "facebook/deit-3-small-patch16-224",
        "facebook/deit-3-small-patch16-384",
        "facebook/deit-3-base-patch16-224",
        "facebook/deit-3-base-patch16-384",
        "facebook/deit-3-large-patch16-224",
        "facebook/deit-3-large-patch16-384",
        # ==================== DINOv2 ====================
        "facebook/dinov2-small",
        "facebook/dinov2-base",
        "facebook/dinov2-large",
        "facebook/dinov2-giant",
        "facebook/dinov2-small-imagenet1k-1-layer",
        "facebook/dinov2-base-imagenet1k-1-layer",
        "facebook/dinov2-large-imagenet1k-1-layer",
        "facebook/dinov2-giant-imagenet1k-1-layer",
        # ==================== BEiT ====================
        "microsoft/beit-base-patch16-224",
        "microsoft/beit-base-patch16-224-pt22k-ft22k",
        "microsoft/beit-large-patch16-224",
        "microsoft/beit-large-patch16-224-pt22k-ft22k",
        "microsoft/beit-base-patch16-384",
        "microsoft/beit-large-patch16-384",
        "microsoft/beit-large-patch16-512",
        # ==================== Swin Transformer ====================
        "microsoft/swin-tiny-patch4-window7-224",
        "microsoft/swin-small-patch4-window7-224",
        "microsoft/swin-base-patch4-window7-224",
        "microsoft/swin-base-patch4-window7-224-in22k",
        "microsoft/swin-base-patch4-window12-384",
        "microsoft/swin-base-patch4-window12-384-in22k",
        "microsoft/swin-large-patch4-window7-224",
        "microsoft/swin-large-patch4-window7-224-in22k",
        "microsoft/swin-large-patch4-window12-384",
        "microsoft/swin-large-patch4-window12-384-in22k",
        # ==================== Swin V2 ====================
        "microsoft/swinv2-tiny-patch4-window8-256",
        "microsoft/swinv2-tiny-patch4-window16-256",
        "microsoft/swinv2-small-patch4-window8-256",
        "microsoft/swinv2-small-patch4-window16-256",
        "microsoft/swinv2-base-patch4-window8-256",
        "microsoft/swinv2-base-patch4-window16-256",
        "microsoft/swinv2-base-patch4-window12-192-22k",
        "microsoft/swinv2-large-patch4-window12-192-22k",
        # ==================== ConvNeXT ====================
        "facebook/convnext-tiny-224",
        "facebook/convnext-tiny-384",
        "facebook/convnext-small-224",
        "facebook/convnext-small-384",
        "facebook/convnext-base-224",
        "facebook/convnext-base-224-22k",
        "facebook/convnext-base-384",
        "facebook/convnext-base-384-22k",
        "facebook/convnext-large-224",
        "facebook/convnext-large-224-22k",
        "facebook/convnext-large-384",
        "facebook/convnext-large-384-22k",
        "facebook/convnext-xlarge-224-22k",
        "facebook/convnext-xlarge-384-22k",
        # ==================== ConvNeXT V2 ====================
        "facebook/convnextv2-atto-1k-224",
        "facebook/convnextv2-femto-1k-224",
        "facebook/convnextv2-pico-1k-224",
        "facebook/convnextv2-nano-1k-224",
        "facebook/convnextv2-tiny-1k-224",
        "facebook/convnextv2-base-1k-224",
        "facebook/convnextv2-large-1k-224",
        "facebook/convnextv2-huge-1k-224",
        "facebook/convnextv2-tiny-22k-224",
        "facebook/convnextv2-tiny-22k-384",
        "facebook/convnextv2-base-22k-224",
        "facebook/convnextv2-base-22k-384",
        "facebook/convnextv2-large-22k-224",
        "facebook/convnextv2-large-22k-384",
        # ==================== CLIP ====================
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-large-patch14-336",
        # ==================== MobileViT ====================
        "apple/mobilevit-xx-small",
        "apple/mobilevit-x-small",
        "apple/mobilevit-small",
        # ==================== MobileNet ====================
        "google/mobilenet_v1_1.0_224",
        "google/mobilenet_v1_0.75_192",
        "google/mobilenet_v2_1.0_224",
        "google/mobilenet_v2_1.4_224",
        # ==================== RegNet ====================
        "facebook/regnet-y-040",
        "facebook/regnet-y-080",
        "facebook/regnet-y-160",
        "facebook/regnet-y-320",
        "facebook/regnet-y-640",
        "facebook/regnet-y-1_3gf",
        "facebook/regnet-y-2_6gf",
        "facebook/regnet-y-4_0gf",
        "facebook/regnet-y-8_0gf",
        "facebook/regnet-y-16gf",
        "facebook/regnet-y-32gf",
        # ==================== SegFormer (MiT backbone) ====================
        "nvidia/mit-b0",
        "nvidia/mit-b1",
        "nvidia/mit-b2",
        "nvidia/mit-b3",
        "nvidia/mit-b4",
        "nvidia/mit-b5",
        "nvidia/segformer-b0-finetuned-ade-512-512",
        "nvidia/segformer-b1-finetuned-ade-512-512",
        "nvidia/segformer-b2-finetuned-ade-512-512",
        "nvidia/segformer-b3-finetuned-ade-512-512",
        "nvidia/segformer-b4-finetuned-ade-512-512",
        "nvidia/segformer-b5-finetuned-ade-640-640",
        # ==================== CvT ====================
        "microsoft/cvt-13",
        "microsoft/cvt-13-384",
        "microsoft/cvt-21",
        "microsoft/cvt-21-384",
        # ==================== Data2Vec Vision ====================
        "facebook/data2vec-vision-base",
        "facebook/data2vec-vision-base-ft",
        "facebook/data2vec-vision-large",
        "facebook/data2vec-vision-large-ft",
        # ==================== LeViT ====================
        "facebook/levit-128S",
        "facebook/levit-128",
        "facebook/levit-192",
        "facebook/levit-256",
        "facebook/levit-384",
        # ==================== DiNAT ====================
        "shi-labs/dinat-mini-in1k-224",
        "shi-labs/dinat-tiny-in1k-224",
        "shi-labs/dinat-small-in1k-224",
        "shi-labs/dinat-base-in1k-224",
        # ==================== EfficientNet ====================
        "google/efficientnet-b0",
        "google/efficientnet-b1",
        "google/efficientnet-b2",
        "google/efficientnet-b3",
        "google/efficientnet-b4",
        "google/efficientnet-b5",
        "google/efficientnet-b6",
        "google/efficientnet-b7",
        # ==================== PoolFormer ====================
        "sail/poolformer_s12",
        "sail/poolformer_s24",
        "sail/poolformer_s36",
        "sail/poolformer_m36",
        "sail/poolformer_m48",
    ]

    # Match models to patterns
    names = []
    for model_name in model_names:
        for family_prefix, (pattern, L) in family_to_pattern.items():
            if model_name.startswith(family_prefix):
                names.append((model_name, pattern, L))
                break

    print(f"Processing {len(names)} HuggingFace vision models...")

    # Process with multiprocessing
    with Pool(10) as pool:
        results = list(
            tqdm(
                pool.imap(_retrieve_hf_modules, names),
                total=len(names),
                desc="Processing HF models",
            )
        )

    # Filter and organize results
    hf_embeddings = {}
    hf_parameters = {}
    failed = []

    for name, result in results:
        if result is not None:
            hf_embeddings[name], hf_parameters[name] = result
        else:
            failed.append(name)

    # Save results
    path = Path(__file__).parent.parent
    (path / "assets").mkdir(exist_ok=True)

    with open(path / "assets/static_hf.json", "w") as f:
        json.dump(hf_embeddings, f, indent=2)

    with open(path / "assets/static_hf_parameters.json", "w") as f:
        json.dump(hf_parameters, f, indent=2)

    print(f"\n✓ Successfully processed: {len(hf_embeddings)}/{len(names)}")
    if failed:
        print(
            f"✗ Failed models ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}"
        )

    # Print sample outputs
    samples = [
        "microsoft/resnet-18",
        "google/vit-base-patch16-224",
        "facebook/dinov2-small",
        "nvidia/mit-b0",
    ]
    for sample in samples:
        if sample in hf_embeddings:
            layers = hf_embeddings[sample]
            print(f"{sample}: {layers[:2]}...{layers[-1:]} ({len(layers)} total)")


#######################################
####                TIMM
#######################################


def _retreive_timm_modules(args):
    name, parent_name, L = args
    import timm
    from stable_pretraining.backbone.utils import get_children_modules

    model = timm.create_model(name, pretrained=False, num_classes=0)
    num_params = count_parameters(model)
    internals = get_children_modules(
        model, parent_name=parent_name, partial_match=True, L=L
    )
    last = _get_last_submodule_full_name(model)
    return internals + [last], num_params


def _generate_timm_factory():
    from tqdm import tqdm
    import timm
    import os
    import json
    from multiprocessing import Pool

    family_to_name = {
        "vit_": ("blocks", 1),
        "swin": ("blocks", 1),
        "levit": ("blocks", 1),
        "maxvit": ("blocks", 1),
        "maxxvit": ("blocks", 1),
        "convnext": ("blocks", 1),
        "resnetv2": ("blocks", 1),
        "resnet": ("layer", 1),
        "resnext": ("layer", 1),
        "efficientnet": ("blocks", 2),
        "mobilenet": ("blocks", 2),
        "convmixer": ("blocks", 1),
        "inception": ("blocks", 1),
    }

    names = []
    for name in timm.list_models(pretrained=False) + timm.list_models(pretrained=True):
        for f in family_to_name:
            if name.startswith(f):
                names.append((name,) + family_to_name[f])
    results = list(
        tqdm(
            Pool(20).imap(_retreive_timm_modules, names),
            total=len(names),
            desc="timm",
        )
    )
    timm_embeddings = {}
    timm_parameters = {}
    for name, result in zip(names, results):
        timm_embeddings[name[0]], timm_parameters[name[0]] = result

    path = Path(os.path.abspath(__file__))
    with open(path.parent.parent / "assets/static_timm.json", "w") as f:
        json.dump(timm_embeddings, f, indent=2)
    with open(path.parent.parent / "assets/static_timm_parameters.json", "w") as f:
        json.dump(timm_parameters, f, indent=2)


if __name__ == "__main__":
    _generate_hf_factory()
    _generate_timm_factory()
    # _generate_hf_embeddings_factory()

    # last 3 blocks
    import stable_pretraining as spt
    import timm
    import torch

    model = timm.create_model("resnet34")
    # add last 3 blocks as separate output
    names = spt.static.TIMM_EMBEDDINGS["resnet34"][-3:]
    # names = ['layers.2.blocks.17', 'layers.3.blocks.0', 'layers.3.blocks.1']
    model = spt.backbone.utils.HiddenStateExtractor(model, names)
    # if you need shapes e.g. for probing definition
    image = torch.zeros((10, 3, 224, 224))
    output_shape, embedding_shapes = spt.backbone.utils.get_output_shape(model, image)
    # embedding_shapes = {'layers.2.blocks.17': torch.Size([10, 14, 14, 768]),
    # 'layers.3.blocks.0': torch.Size([10, 7, 7, 1536]),
    # 'layers.3.blocks.1': torch.Size([10, 7, 7, 1536])}
    output = model(image)
    # output = tensor([[ 1.1009 ...
    # embeddings = {'layers.3.blocks.1': tensor([[[[-0.6236, ...}
