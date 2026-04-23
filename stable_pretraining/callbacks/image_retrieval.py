import types
from typing import List, Union

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from torchmetrics.retrieval.base import RetrievalMetric

from .registry import log_dict as _spt_log_dict
from .utils import format_metrics_as_dict


def wrap_validation_step(fn, input, name):
    def ffn(
        self,
        batch,
        batch_idx,
        fn=fn,
        name=name,
        input=input,
    ):
        batch = fn(batch, batch_idx)

        with torch.no_grad():
            norm = self.callbacks_modules[name]["normalizer"](batch[input])
            norm = torch.nn.functional.normalize(norm, dim=1, p=2)

        idx = self.all_gather(batch["sample_idx"])
        norm = self.all_gather(norm)

        if self.local_rank == 0:
            self.embeds[idx] = norm

        return batch

    return ffn


class ImageRetrieval(Callback):
    """Image Retrieval evaluator for self-supervised learning.

    The implementation follows:
      1. https://github.com/facebookresearch/dino/blob/main/eval_image_retrieval.py
    """

    NAME = "ImageRetrieval"

    def __init__(
        self,
        pl_module,
        name: str,
        input: str,
        query_col: str,
        retrieval_col: str | List[str],
        metrics,
        features_dim: Union[tuple[int], list[int], int],
        normalizer: str = None,
    ) -> None:
        logging.info(f"Setting up callback ({self.NAME})")
        logging.info(f"\t- {input=}")
        logging.info(f"\t- {query_col=}")
        logging.info("\t- caching modules into `callbacks_modules`")
        if name in pl_module.callbacks_modules:
            raise ValueError(f"{name=} already used in callbacks")
        if type(features_dim) in [list, tuple]:
            features_dim = np.prod(features_dim)

        if normalizer is not None and normalizer not in ["batch_norm", "layer_norm"]:
            raise ValueError(
                "`normalizer` has to be one of `batch_norm` or `layer_norm`"
            )

        if normalizer == "batch_norm":
            normalizer = torch.nn.BatchNorm1d(features_dim, affine=False)
        elif normalizer == "layer_norm":
            normalizer = torch.nn.LayerNorm(
                features_dim, elementwise_affine=False, bias=False
            )
        else:
            normalizer = torch.nn.Identity()

        pl_module.callbacks_modules[name] = torch.nn.ModuleDict(
            {
                "normalizer": normalizer,
            }
        )

        logging.info(
            f"`callbacks_modules` now contains ({list(pl_module.callbacks_modules.keys())})"
        )

        if not isinstance(retrieval_col, list):
            retrieval_col = [retrieval_col]

        for k, metric in metrics.items():
            if not isinstance(metric, RetrievalMetric):
                raise ValueError(
                    f"Only `RetrievalMetric` is supported for {self.NAME} callback, but got {metric} for {k}"
                )

        logging.info("\t- caching metrics into `callbacks_metrics`")
        pl_module.callbacks_metrics[name] = format_metrics_as_dict(metrics)

        logging.info("\t- wrapping the `validation_step`")
        fn = wrap_validation_step(pl_module.validation_step, input, name)
        pl_module.validation_step = types.MethodType(fn, pl_module)

        self.name = name
        self.features_dim = features_dim

        self.query_col = query_col
        self.retrieval_col = retrieval_col

        #
        pl_module.embeds = None

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"ImageRetrieval[name={self.name}]"

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # register buffer on rank 0
        val_dataset = pl_module.trainer.datamodule.val.dataset
        dataset_size = len(val_dataset)
        if pl_module.local_rank == 0:
            device = pl_module.device
            pl_module.embeds = torch.zeros(
                (dataset_size, self.features_dim), device=device
            )
        return

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if pl_module.local_rank == 0:
            logging.info(f"Computing results for {self.name} callback")

            val_dataset = pl_module.trainer.datamodule.val.dataset.dataset

            if len(pl_module.embeds) != len(val_dataset):
                logging.warning(
                    f"Expected {len(val_dataset)} embeddings, but got {len(pl_module.embeds)}. Skipping evaluation."
                )
                return

            is_query = torch.tensor(
                val_dataset[self.query_col], device=pl_module.device
            ).squeeze()

            query_idx = torch.nonzero(is_query)
            query = pl_module.embeds[is_query]
            gallery = pl_module.embeds[~is_query]
            score = query @ gallery.t()
            # ranks = torch.argsort(-score, dim=1)

            preds = []
            targets = []
            indexes = []

            for idx, q_idx in enumerate(query_idx):
                # add query idx to the indexes
                indexes.append(q_idx.repeat(len(gallery)))

                # build target for query
                target = torch.zeros(
                    len(gallery), dtype=torch.bool, device=pl_module.device
                )

                for col in self.retrieval_col:
                    ret_idx = val_dataset[q_idx][col]
                    if ret_idx:
                        target[ret_idx] = True

                targets.append(target)
                preds.append(score[idx])

            preds = torch.cat(preds)
            targets = torch.cat(targets)
            indexes = torch.cat(indexes)

            logs = {}
            for k, metric in pl_module.callbacks_metrics[self.name]["_val"].items():
                res = metric(preds, targets, indexes=indexes)
                logs[f"eval/{self.name}_{k}"] = res.item() * 100

            _spt_log_dict(logs, on_epoch=True, rank_zero_only=True)

            logging.info(f"Finished computing results for {self.name} callback")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        pl_module.embeds = None
