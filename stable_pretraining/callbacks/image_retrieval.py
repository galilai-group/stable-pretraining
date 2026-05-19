import types
from typing import List, Optional, Union

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from torchmetrics.retrieval.base import RetrievalMetric

from .registry import log_dict as _spt_log_dict
from .utils import format_metrics_as_dict


def wrap_validation_step(fn, input, name, callback):
    """Wrap ``validation_step`` to populate the callback's embedding buffer.

    The wrapped function captures the ``ImageRetrieval`` instance via the
    closure so embeddings live on ``callback.embeds`` instead of mutating
    ``pl_module.embeds`` — keeping multiple ImageRetrieval instances isolated.
    """

    def ffn(
        self,
        batch,
        batch_idx,
        fn=fn,
        name=name,
        input=input,
        callback=callback,
    ):
        batch = fn(batch, batch_idx)

        with torch.no_grad():
            norm = self.callbacks_modules[name]["normalizer"](batch[input])
            norm = torch.nn.functional.normalize(norm, dim=1, p=2)

        idx = self.all_gather(batch["sample_idx"])
        norm = self.all_gather(norm)

        if self.local_rank == 0:
            # Lazy allocation: infer ``features_dim`` from the first batch's
            # normalized embeddings if the user didn't specify it upfront.
            if callback.embeds is None:
                inferred_dim = int(norm.shape[-1])
                if (
                    callback.features_dim is not None
                    and callback.features_dim != inferred_dim
                ):
                    raise ValueError(
                        f"ImageRetrieval[{name}]: features_dim="
                        f"{callback.features_dim} but first batch produced "
                        f"embeddings of dim {inferred_dim}. Either pass the "
                        f"correct features_dim or omit it for inference."
                    )
                dataset_size = len(self.trainer.datamodule.val.dataset)
                callback.embeds = torch.zeros(
                    (dataset_size, inferred_dim), device=norm.device
                )
                callback.features_dim = inferred_dim
            callback.embeds[idx] = norm

        return batch

    return ffn


class ImageRetrieval(Callback):
    """Image Retrieval evaluator for self-supervised learning.

    The implementation follows:
      1. https://github.com/facebookresearch/dino/blob/main/eval_image_retrieval.py

    Args:
        pl_module: The ``spt.LightningModule`` to evaluate against.
        name: Unique identifier (used as key in ``callbacks_modules`` and
            ``callbacks_metrics``). Two instances with the same ``name`` raise.
        input: Key in ``batch`` containing per-sample embeddings.
        query_col: Boolean column on the val dataset marking query rows.
        retrieval_col: Single column name or list — each value is a list of
            gallery indices that count as relevant for that query.
        metrics: ``torchmetrics.retrieval.RetrievalMetric`` instances keyed by
            display name.
        features_dim: Output dimension of the embedding. If ``None`` (default)
            the dimension is inferred from the first validation batch. If
            provided, it must match what the model emits — a mismatch raises.
        normalizer: ``"batch_norm"``, ``"layer_norm"``, or ``None`` (identity)
            applied to embeddings before L2-normalization. When using
            ``"batch_norm"`` or ``"layer_norm"``, ``features_dim`` must be set
            explicitly because the normalizer module needs to be built at
            ``__init__`` time.
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
        features_dim: Optional[Union[tuple[int], list[int], int]] = None,
        normalizer: str = None,
    ) -> None:
        logging.info(f"Setting up callback ({self.NAME})")
        logging.info(f"\t- {input=}")
        logging.info(f"\t- {query_col=}")
        logging.info("\t- caching modules into `callbacks_modules`")
        if name in pl_module.callbacks_modules:
            raise ValueError(f"{name=} already used in callbacks")
        if features_dim is not None and type(features_dim) in [list, tuple]:
            features_dim = int(np.prod(features_dim))

        if normalizer is not None and normalizer not in ["batch_norm", "layer_norm"]:
            raise ValueError(
                "`normalizer` has to be one of `batch_norm` or `layer_norm`"
            )

        # batch_norm / layer_norm need a known input dim at construction time;
        # identity is dim-agnostic and lets us defer.
        if normalizer in ("batch_norm", "layer_norm") and features_dim is None:
            raise ValueError(
                f"normalizer={normalizer!r} requires features_dim to be "
                "specified explicitly; only normalizer=None supports "
                "inference at first batch."
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

        self.name = name
        self.features_dim = features_dim
        self.query_col = query_col
        self.retrieval_col = retrieval_col
        # Embedding buffer lives on the callback instance (not on pl_module)
        # so multiple ImageRetrieval callbacks don't clobber each other's state.
        self.embeds: Optional[torch.Tensor] = None

        logging.info("\t- wrapping the `validation_step`")
        fn = wrap_validation_step(pl_module.validation_step, input, name, callback=self)
        pl_module.validation_step = types.MethodType(fn, pl_module)

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"ImageRetrieval[name={self.name}]"

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Eagerly allocate the embeds buffer if ``features_dim`` was given.

        Skipped otherwise — allocation happens lazily on the first batch
        inside the wrapped ``validation_step``.
        """
        if pl_module.local_rank == 0 and self.features_dim is not None:
            val_dataset = pl_module.trainer.datamodule.val.dataset
            dataset_size = len(val_dataset)
            self.embeds = torch.zeros(
                (dataset_size, self.features_dim), device=pl_module.device
            )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if pl_module.local_rank == 0:
            logging.info(f"Computing results for {self.name} callback")

            val_dataset = pl_module.trainer.datamodule.val.dataset.dataset

            if self.embeds is None:
                logging.warning(
                    f"{self.name}: no embeddings collected (zero validation "
                    "batches?). Skipping evaluation."
                )
                return

            if len(self.embeds) != len(val_dataset):
                logging.warning(
                    f"Expected {len(val_dataset)} embeddings, but got "
                    f"{len(self.embeds)}. Skipping evaluation."
                )
                return

            is_query = torch.tensor(
                val_dataset[self.query_col], device=pl_module.device
            ).squeeze()

            query_idx = torch.nonzero(is_query)
            query = self.embeds[is_query]
            gallery = self.embeds[~is_query]
            score = query @ gallery.t()

            # Pre-extract the retrieval-target columns once. Per-row
            # ``val_dataset[q_idx][col]`` triggers the dataset's transform
            # pipeline — re-decoding the image for every query just to read
            # an index list. Column-wise access on the underlying HF dataset
            # bypasses the transform pipeline entirely and returns native
            # Python lists, turning O(queries × image-decodes) into one
            # column fetch.
            ret_cache = {col: val_dataset[col] for col in self.retrieval_col}

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

                row = int(q_idx.item())
                for col in self.retrieval_col:
                    ret_idx = ret_cache[col][row]
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

        self.embeds = None
