"""DataModule must support independent stage configurations and real loaders."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from stable_pretraining.data.module import DataModule, DictFormat

pytestmark = pytest.mark.unit


class TinyDataset(Dataset):
    """Small deterministic dataset that records the attached trainer."""

    column_names = ["value"]

    def __init__(self, size=8):
        self.size = size
        self.trainer = None

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"value": index}

    def set_pl_trainer(self, trainer):
        self.trainer = trainer


def _config(**overrides):
    return {
        "dataset": {"_target_": __name__ + ".TinyDataset", "size": 8},
        "batch_size": 3,
        "pin_memory": False,
        **overrides,
    }


@pytest.mark.parametrize(
    "stage,key",
    [("fit", "train"), ("validate", "val"), ("test", "test"), ("predict", "predict")],
)
@pytest.mark.parametrize("prebuilt", [False, True])
def test_stage_loader_preserves_samples_and_attaches_trainer(stage, key, prebuilt):
    source = DataLoader(TinyDataset(), batch_size=3) if prebuilt else _config()
    module = DataModule(**{key: source})
    trainer = SimpleNamespace(world_size=1)
    module.set_pl_trainer(trainer)
    module.setup(stage)
    loader = getattr(module, key + "_dataloader")()
    assert [int(x) for batch in loader for x in batch["value"]] == list(range(8))
    assert loader.dataset.trainer is trainer
    if prebuilt:
        assert loader is source
    assert module.state_dict() == {}
    module.load_state_dict({})
    module.teardown(stage)


@pytest.mark.parametrize("train_prebuilt", [False, True])
def test_fit_supports_mixed_prebuilt_and_configured_loaders(train_prebuilt):
    loader = DataLoader(TinyDataset(), batch_size=2)
    module = DataModule(
        train=loader if train_prebuilt else _config(),
        val=_config() if train_prebuilt else loader,
    )
    module.set_pl_trainer(SimpleNamespace(world_size=1))
    module.setup("fit")
    assert len(module.train_dataloader().dataset) == 8
    assert len(module.val_dataloader().dataset) == 8


def test_standard_torch_dataset_does_not_require_huggingface_metadata():
    cfg = _config(
        dataset={
            "_target_": "torch.utils.data.TensorDataset",
            "_args_": [{"_target_": "torch.arange", "end": 8}],
        }
    )
    module = DataModule(test=cfg)
    module.setup("test")
    assert torch.cat([batch[0] for batch in module.test_dataloader()]).tolist() == list(
        range(8)
    )


def test_loader_configuration_is_copied_and_partial_sampler_gets_dataset():
    cfg = OmegaConf.create(
        _config(
            sampler={
                "_target_": "torch.utils.data.SequentialSampler",
                "_partial_": True,
            }
        )
    )
    module = DataModule(train=cfg)
    cfg.batch_size = 8
    module.set_pl_trainer(SimpleNamespace(world_size=1))
    module.setup("fit")
    loader = module.train_dataloader()
    assert loader.batch_size == 3
    assert list(loader.sampler) == list(range(8))
    assert module.val_dataloader() == []


@pytest.mark.parametrize(
    "config", [{}, {"dataset": {"_target_": "builtins.list"}, "misspelled_option": 2}]
)
def test_invalid_loader_config_reports_value_error(config):
    with pytest.raises(ValueError):
        DataModule(train=config)


def test_datamodule_requires_a_stage_and_rejects_invalid_setup():
    with pytest.raises(ValueError, match="none"):
        DataModule()
    module = DataModule(test=DataLoader(TinyDataset()))
    with pytest.raises(ValueError, match="Invalid stage"):
        module.setup("unknown")


def test_dict_dataset_wrapper_preserves_column_values():
    wrapped = DictFormat(
        [(torch.tensor([1, 2]), 3), (torch.tensor([4, 5]), 6)], ["image", "label"]
    )
    assert len(wrapped) == 2
    assert wrapped[torch.tensor(1)]["label"] == 6
    torch.testing.assert_close(wrapped[0]["image"], torch.tensor([1, 2]))


class TinyStream(torch.utils.data.IterableDataset):
    """An ordinary PyTorch stream without Hugging Face metadata."""

    def __iter__(self):
        yield from ({"value": value} for value in range(8))


@pytest.mark.parametrize("dataset", [TinyDataset(), TinyStream()])
@pytest.mark.parametrize(
    "stage,key",
    [("fit", "train"), ("validate", "val"), ("test", "test"), ("predict", "predict")],
)
def test_prebuilt_torch_datasets_work_inside_loader_configs(dataset, stage, key):
    config = {"dataset": dataset, "batch_size": 3, "pin_memory": False}
    module = DataModule(**{key: config})
    module.set_pl_trainer(SimpleNamespace(world_size=1))
    module.setup(stage)
    loader = getattr(module, key + "_dataloader")()
    assert [int(value) for batch in loader for value in batch["value"]] == list(
        range(8)
    )
    assert isinstance(loader.dataset, type(dataset))
    assert config["dataset"] is dataset


@pytest.mark.parametrize("streaming", [False, True])
def test_hf_dataset_configs_still_use_the_standard_dataset_contract(streaming):
    import datasets
    from stable_pretraining.data.datasets import HFIterableDataset, HFMapDataset

    source = datasets.Dataset.from_dict({"value": list(range(8))})
    dataset = (
        HFIterableDataset(source.to_iterable_dataset())
        if streaming
        else HFMapDataset(source)
    )
    module = DataModule(test={"dataset": dataset, "batch_size": 3, "pin_memory": False})
    module.setup("test")
    assert [
        int(value) for batch in module.test_dataloader() for value in batch["value"]
    ] == list(range(8))
