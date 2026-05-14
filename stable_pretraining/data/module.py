import copy
import inspect
from typing import Iterable, Optional, Union

import hydra
import lightning as pl
import torch
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from .sampler import RepeatedRandomSampler
from .datasets import HFMapDataset, HFIterableDataset


class DictFormat(Dataset):
    """Format dataset with named columns for dictionary-style access.

    Args:
        dataset (Iterable): Dataset to be wrapped.
        names (Iterable): Column names for the dataset.
    """

    def __init__(self, dataset: Iterable, names: Iterable):
        self.dataset = dataset
        self.names = names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        values = self.dataset[idx]
        sample = {k: v for k, v in zip(self.names, values)}
        return sample


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling train/val/test/predict dataloaders."""

    def __init__(
        self,
        train: Optional[Union[dict, DictConfig, DataLoader]] = None,
        test: Optional[Union[dict, DictConfig, DataLoader]] = None,
        val: Optional[Union[dict, DictConfig, DataLoader]] = None,
        predict: Optional[Union[dict, DictConfig, DataLoader]] = None,
        **kwargs,
    ):
        super().__init__()
        if train is None and test is None and val is None and predict is None:
            raise ValueError("They can't all be none")
        logging.info("Setting up DataModule")
        if train is None:
            logging.warning(
                "! train was not passed to DataModule, it is required "
                "unless you only validate"
            )
        self.train = self._format_data_conf(train, "train")
        self.test = self._format_data_conf(test, "test")
        if val is None:
            logging.warning(
                "! val was not passed to DataModule, it is required "
                "unless you set `num_sanity_val_steps=0` and `val_check_interval=0`"
            )
        self.val = self._format_data_conf(val, "val")
        self.predict = self._format_data_conf(predict, "predict")
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._trainer = None

    @staticmethod
    def _format_data_conf(conf: Union[dict, DictConfig], stage: str):
        if conf is None:
            return None
        if isinstance(conf, DataLoader):
            return conf
        elif isinstance(conf["dataset"], (HFMapDataset, HFIterableDataset)):
            logging.info(f"  {stage} already has an instantiated dataset")
        elif type(conf) is dict:
            logging.info(f"  {stage} has `dict` type and no instantiated dataset")
            conf = OmegaConf.create(conf)
            logging.info(f"  {stage} created DictConfig")
        elif type(conf) is not DictConfig:
            raise ValueError(f"`{conf}` must be a dict of DictConfig")
        sign = inspect.signature(DataLoader)
        # we check that user gives the required parameters
        for k, param in sign.parameters.items():
            if param.default is param.empty and k not in conf:
                raise ValueError(f"conf must specify a value for {k}")
        # we check that user doesn't give extra parameters
        for k in conf.keys():
            if k not in sign.parameters:
                raise ValueError(f"{k} given in conf is not a DataLoader kwarg")
        conf = copy.deepcopy(conf)
        logging.success(f"✓ {stage} conf is valid and saved")
        return conf

    def setup(self, stage):
        # TODO: should we move some to prepare_data?
        if stage not in ["fit", "validate", "test", "predict"]:
            raise ValueError(f"Invalid stage {stage}")
        d = None
        if stage == "fit" and not isinstance(self.train, DataLoader):
            self.train_dataset = d = hydra.utils.instantiate(
                self.train.dataset, _convert_="object", _recursive_=True
            )
            self.val_dataset = hydra.utils.instantiate(
                self.val.dataset, _convert_="object", _recursive_=True
            )
        elif stage == "test" and not isinstance(self.test, DataLoader):
            self.test_dataset = d = hydra.utils.instantiate(
                self.test.dataset, _convert_="object", _recursive_=True
            )
        elif stage == "validate" and not isinstance(self.val, DataLoader):
            self.val_dataset = d = hydra.utils.instantiate(
                self.val.dataset, _convert_="object", _recursive_=True
            )
        elif stage == "predict" and not isinstance(self.predict, DataLoader):
            self.predict_dataset = d = hydra.utils.instantiate(
                self.predict.dataset, _convert_="object", _recursive_=True
            )
        logging.success(f"✓ dataset for {stage} loaded")
        if d is not None:
            logging.info(f"  length: {len(d)}")
            logging.info(f"  columns: {d.column_names}")
        else:
            logging.info("  setup was done by user")

    def _get_loader_kwargs(self, config, dataset):
        kwargs = dict()
        for k, v in config.items():
            if k == "dataset":
                continue
            if type(v) in [dict, DictConfig] and "_target_" in v:
                kwargs[k] = hydra.utils.instantiate(v, _convert_="object")
                if "_partial_" in v:
                    kwargs[k] = kwargs[k](dataset)
            else:
                kwargs[k] = v
        return kwargs

    def train_dataloader(self):
        """Return the training DataLoader."""
        if isinstance(self.train, DataLoader):
            loader = self.train
        else:
            kwargs = self._get_loader_kwargs(self.train, self.train_dataset)
            loader = DataLoader(dataset=self.train_dataset, **kwargs)
        if hasattr(loader.dataset, "set_pl_trainer"):
            loader.dataset.set_pl_trainer(self._trainer)
        else:
            logging.warning("could not set pl_trainer to train dataset")
        if (
            self.trainer is not None
            and self.trainer.world_size > 1
            and isinstance(loader.sampler, RepeatedRandomSampler)
        ):
            sampler = RepeatedRandomSampler(
                loader.sampler._data_source_len,
                loader.sampler.n_views,
                loader.sampler.replacement,
                loader.sampler.seed,
                loader.sampler.pass_view_idx,
            )
            loader = DataLoader(
                dataset=loader.dataset,
                sampler=sampler,
                batch_size=loader.batch_size,
                shuffle=False,
                num_workers=loader.num_workers,
                collate_fn=loader.collate_fn,
                pin_memory=loader.pin_memory,
                drop_last=loader.drop_last,
                timeout=loader.timeout,
                worker_init_fn=loader.worker_init_fn,
                prefetch_factor=loader.prefetch_factor,
                persistent_workers=loader.persistent_workers,
                pin_memory_device=loader.pin_memory_device,
                in_order=loader.in_order,
            )
        return loader

    def val_dataloader(self):
        """Return the validation DataLoader (or an empty list if unset)."""
        if self.val is None:
            return []
        if isinstance(self.val, DataLoader):
            if hasattr(self.val.dataset, "set_pl_trainer"):
                self.val.dataset.set_pl_trainer(self._trainer)
            else:
                logging.warning("could not set pl_trainer to train dataset")
            return self.val
        kwargs = self._get_loader_kwargs(self.val, self.val_dataset)
        if hasattr(self.val_dataset, "set_pl_trainer"):
            self.val_dataset.set_pl_trainer(self._trainer)
        else:
            logging.warning("could not set pl_trainer to train dataset")
        return DataLoader(dataset=self.val_dataset, **kwargs)

    def test_dataloader(self):
        """Return the test DataLoader."""
        if isinstance(self.test, DataLoader):
            if hasattr(self.test.dataset, "set_pl_trainer"):
                self.test.dataset.set_pl_trainer(self._trainer)
            else:
                logging.warning("could not set pl_trainer to train dataset")
            return self.test
        kwargs = self._get_loader_kwargs(self.test, self.test_dataset)
        if hasattr(self.test_dataset, "set_pl_trainer"):
            self.test_dataset.set_pl_trainer(self._trainer)
        else:
            logging.warning("could not set pl_trainer to train dataset")
        return DataLoader(dataset=self.test_dataset, **kwargs)

    def predict_dataloader(self):
        """Return the prediction DataLoader."""
        if isinstance(self.predict, DataLoader):
            if hasattr(self.predict.dataset, "set_pl_trainer"):
                self.predict.dataset.set_pl_trainer(self._trainer)
            else:
                logging.warning("could not set pl_trainer to train dataset")
            return self.predict
        kwargs = self._get_loader_kwargs(self.predict, self.predict_dataset)

        if hasattr(self.predict_dataset, "set_pl_trainer"):
            self.predict_dataset.set_pl_trainer(self._trainer)
        else:
            logging.warning("could not set pl_trainer to train dataset")
        return DataLoader(dataset=self.predict_dataset, **kwargs)

    def teardown(self, stage: str):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return

    def set_pl_trainer(self, trainer: pl.Trainer):
        self._trainer = trainer
