"""Load verified, unmodified EEG-JEPA sources from an external cache."""

import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

REFERENCE_URL = "https://github.com/SWF-hao/EEG-JEPA-official"
REFERENCE_REVISION = "925b1194d5abf1e95ca2a2e7416b5a5d268845df"
SOURCE_HASHES = {
    "models/criss_cross_transformer.py": "90e90b8c98ca7eb543c44e32afc29674d3f71cb090692ad9f82db3a43edf334c",
    "models/cbramod_jepa.py": "49b8fa6774778bc0d1e17c633a3c71e0818f97def0635f44db62341c3b7a5573",
    "models/eeg_jepa.py": "9acf3fc96a8f631edf6be7e401336daac89706a753d71f23732b44f6afb4f600",
    "masks/nmet.py": "80f35c3df59d99e69a4bf8f80a1f3cf6e7dd54d3479e900da0220382eaaabd66",
    "trainer.py": "d13c63c0744536e7cb3d1ca8457ac4a0c619672c6dd9977a83ee9678c779ca3d",
}


def load_reference(root: Path) -> SimpleNamespace:
    """Verify the pinned sources and load without retaining generic import aliases.

    Args:
        root: Extracted reference archive, outside any Git checkout.

    Returns:
        Reference model, mask and trainer modules.

    Note:
        No downloads or reference modifications occur here.
    """
    root = Path(root).expanduser().resolve()
    for relative, expected in SOURCE_HASHES.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference differs from {REFERENCE_REVISION}: {relative}")
    return _load_verified(root)


@lru_cache
def _load_verified(root: Path) -> SimpleNamespace:
    prefix = "_spt_eeg_jepa_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    package = ModuleType(prefix)
    package.__path__ = [str(root)]
    sys.modules[prefix] = package
    aliases = {
        key: value
        for key, value in sys.modules.items()
        if key == "models" or key.startswith("models.")
    }
    for key in aliases:
        del sys.modules[key]
    models = ModuleType("models")
    models.__path__ = [str(root / "models")]
    sys.modules["models"] = models
    loaded = {}
    try:
        for relative in SOURCE_HASHES:
            name = relative.removesuffix(".py").replace("/", "_")
            spec = importlib.util.spec_from_file_location(
                f"{prefix}.{name}", root / relative
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            if relative.startswith("models/"):
                sys.modules[relative.removesuffix(".py").replace("/", ".")] = module
            spec.loader.exec_module(module)
            loaded[relative] = module
    finally:
        for key in list(sys.modules):
            if key == "models" or key.startswith("models."):
                del sys.modules[key]
        sys.modules.update(aliases)
    return SimpleNamespace(
        model=loaded["models/eeg_jepa.py"],
        masks=loaded["masks/nmet.py"],
        trainer=loaded["trainer.py"],
    )
