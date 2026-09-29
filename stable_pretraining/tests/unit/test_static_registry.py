"""Model metadata caches must load once and protect their stored values."""

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from torch import nn

from stable_pretraining import static

pytestmark = pytest.mark.unit


@pytest.fixture(
    params=[
        static.TIMM_EMBEDDINGS,
        static.HF_EMBEDDINGS,
        static.TIMM_PARAMETERS,
        static.HF_PARAMETERS,
    ]
)
def registry(request, tmp_path, monkeypatch):
    cls = request.param
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"a": ["encoder.0", "encoder.1"], "b": ["head"]}))
    monkeypatch.setattr(cls, "_data", None)
    monkeypatch.setattr(cls, "_file_path", str(path))
    return cls


def test_registry_mapping_access_and_defensive_copies(registry):
    assert "a" in registry and "missing" not in registry
    assert len(registry) == 2
    assert list(registry) == list(registry.keys()) == ["a", "b"]
    assert registry.get("missing", "fallback") == "fallback"
    for value in [
        registry["a"],
        registry.get("a"),
        next(registry.values()),
        dict(registry.items())["a"],
    ]:
        value.append("corruption")
    assert registry["a"] == ["encoder.0", "encoder.1"]
    with pytest.raises(KeyError):
        registry["missing"]


def test_registry_changes_and_validation(registry):
    registry["c"] = ["block"]
    registry.update({"d": ["layer"]})
    registry.update([("e", ["norm"])])
    assert registry["e"] == ["norm"]
    del registry["c"]
    assert "c" not in registry
    for invalid in [{"bad": "not a list"}, [("bad", 1)]]:
        with pytest.raises(TypeError, match="must be lists"):
            registry.update(invalid)
    assert "bad" not in registry
    registry.clear()
    assert len(registry) == 0


def test_registry_load_is_shared_between_threads(registry, monkeypatch):
    read = Mock(wraps=static.json.load)
    monkeypatch.setattr(static.json, "load", read)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: registry["a"], range(12)))
    assert values == [["encoder.0", "encoder.1"]] * 12
    assert read.call_count == 1


@pytest.mark.parametrize("registry", [static.TIMM_PARAMETERS, static.HF_PARAMETERS])
def test_parameter_registry_supports_numeric_values(registry, monkeypatch):
    monkeypatch.setattr(registry, "_data", {"tiny": 17})
    assert registry["tiny"] == registry.get("tiny") == 17
    assert list(registry.values()) == [17]
    assert dict(registry.items()) == {"tiny": 17}


@pytest.mark.parametrize("problem", ["missing", "invalid_json"])
def test_registry_failed_load_can_be_retried(tmp_path, monkeypatch, problem):
    registry = static.TIMM_EMBEDDINGS
    path = tmp_path / "metadata.json"
    monkeypatch.setattr(registry, "_data", None)
    monkeypatch.setattr(registry, "_file_path", str(path))
    if problem == "invalid_json":
        path.write_text("{")
    with pytest.raises((RuntimeError, json.JSONDecodeError)):
        len(registry)
    path.write_text('{"model": ["encoder"]}')
    assert registry["model"] == ["encoder"]


def test_model_metadata_extraction_counts_only_trainable_parameters(monkeypatch):
    model = nn.ModuleDict(
        {"blocks": nn.Sequential(nn.Linear(3, 2), nn.ReLU()), "head": nn.Linear(2, 1)}
    )
    model.head.requires_grad_(False)
    create = Mock(return_value=model)
    monkeypatch.setattr(static.timm, "create_model", create)
    names, count = static._retreive_timm_modules(("tiny", "blocks", 1))
    assert names == ["blocks.0", "blocks.1", "head"]
    assert count == 8
    create.assert_called_once_with("tiny", pretrained=False, num_classes=0)


def test_hf_metadata_handles_model_failure_and_success(monkeypatch):
    from transformers import AutoConfig, AutoModel

    monkeypatch.setattr(
        AutoConfig, "from_pretrained", Mock(side_effect=OSError("unavailable"))
    )
    assert static._retrieve_hf_modules(("tiny", "blocks", 1)) == ("tiny", None)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: object())
    model = nn.ModuleDict(
        {"blocks": nn.Sequential(nn.Linear(3, 2)), "head": nn.Identity()}
    )
    monkeypatch.setattr(AutoModel, "from_config", lambda _: model)
    assert static._retrieve_hf_modules(("tiny", "blocks", 1)) == (
        "tiny",
        (["blocks.0", "head"], 8),
    )


def test_hf_catalog_generation_keeps_successes_and_reports_failures(
    monkeypatch, tmp_path, capsys
):
    from multiprocessing.dummy import Pool
    import multiprocessing

    monkeypatch.setattr(multiprocessing, "Pool", Pool)
    monkeypatch.setattr(static, "__file__", str(tmp_path / "package" / "static.py"))

    def retrieve(args):
        name = args[0]
        return (
            (name, (["encoder", "head"], 17))
            if name.startswith("google/vit")
            else (name, None)
        )

    monkeypatch.setattr(static, "_retrieve_hf_modules", retrieve)
    static._generate_hf_factory()
    names = json.loads((tmp_path / "assets/static_hf.json").read_text())
    counts = json.loads((tmp_path / "assets/static_hf_parameters.json").read_text())
    assert names and names.keys() == counts.keys()
    assert all(name.startswith("google/vit") for name in names)
    assert all(value == ["encoder", "head"] for value in names.values())
    assert set(counts.values()) == {17}
    assert "Failed models" in capsys.readouterr().out


def test_timm_catalog_generation_routes_model_families(monkeypatch, tmp_path):
    from multiprocessing.dummy import Pool
    import multiprocessing

    pools = []

    def pool(size):
        p = Pool(2)
        pools.append(p)
        return p

    monkeypatch.setattr(multiprocessing, "Pool", pool)
    monkeypatch.setattr(static, "__file__", str(tmp_path / "package/static.py"))
    (tmp_path / "assets").mkdir()
    monkeypatch.setattr(
        static.timm, "list_models", lambda **kw: ["vit_tiny", "resnet18", "unrelated"]
    )
    seen = []

    def retrieve(args):
        seen.append(args)
        return [args[1]], 23

    monkeypatch.setattr(static, "_retreive_timm_modules", retrieve)
    try:
        static._generate_timm_factory()
    finally:
        for p in pools:
            p.close()
            p.join()
    assert ("vit_tiny", "blocks", 1) in seen
    assert ("resnet18", "layer", 1) in seen
    assert json.loads((tmp_path / "assets/static_timm.json").read_text()) == {
        "vit_tiny": ["blocks"],
        "resnet18": ["layer"],
    }
    assert json.loads(
        (tmp_path / "assets/static_timm_parameters.json").read_text()
    ) == {"vit_tiny": 23, "resnet18": 23}
