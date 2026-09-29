import lightning as pl
from loguru import logger as logging


class EmbeddingCache(pl.pytorch.Callback):
    """Cache embedding from a module given their names.

    Args:
    module_names (list of str): List of module names to hook (e.g., ['layer1', 'encoder.block1']).
    add_to_forward_output (bool): If True, enables merging cached outputs into the dict returned by forward.
    """

    def __init__(self, module_names: list, add_to_forward_output: bool = True):
        super().__init__()
        logging.info("Init of EmbeddingCache callback with")
        logging.info(f"\t - {len(module_names)} module names")
        logging.info(f"\t - {add_to_forward_output}")
        self.module_names = module_names
        self.add_to_forward_output = add_to_forward_output
        self.hooks = []

    def setup(self, trainer, pl_module, stage=None):
        logging.info("Setup of EmbeddingCache")
        if hasattr(pl_module, "embedding_cache"):
            raise RuntimeError("A embedding_cache is already present")
        modules = []
        for name in self.module_names:
            module = self._get_module_by_name(pl_module, name)
            if module is None:
                raise ValueError(f"Module '{name}' not found in LightningModule.")
            modules.append(module)
        pl_module.embedding_cache = {}
        for name, module in zip(self.module_names, modules):
            hook = module.register_forward_hook(self._make_hook(name, pl_module))
            self.hooks.append(hook)
        logging.info("\t - adding forward hook")
        self.hooks.append(pl_module.register_forward_hook(self.forward_hook_fn))

    def teardown(self, trainer, pl_module, stage=None):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if hasattr(pl_module, "embedding_cache"):
            del pl_module.embedding_cache
        if hasattr(pl_module, "_addembedding_cache_to_forward"):
            del pl_module._addembedding_cache_to_forward

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        pl_module.embedding_cache.clear()

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        pl_module.embedding_cache.clear()

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        pl_module.embedding_cache.clear()

    def _make_hook(self, name, pl_module):
        def hook(module, input, output):
            pl_module.embedding_cache[name] = output

        return hook

    def _get_module_by_name(self, pl_module, name):
        module = pl_module
        for attr in name.split("."):
            if not hasattr(module, attr):
                return None
            module = getattr(module, attr)
        return module

    def forward_hook_fn(self, pl_module, args, outputs) -> None:
        """Perform probe training step."""
        # Extract batch from args tuple (it's the first argument to forward)
        if not self.add_to_forward_output:
            return outputs
        outputs.update(pl_module.embedding_cache)
        return outputs
