import pytorch_lightning as pl


class HPMetricLogger(pl.Callback):
    """Callback to log a specific metric for hyperparameter optimization."""

    def __init__(self, metric_name):
        super().__init__()
        self.metric_name = metric_name

    def on_validation_epoch_end(self, trainer, pl_module):
        hp_metric = trainer.callback_metrics[self.metric_name]
        if getattr(pl_module, "hp_metric", None) is None:
            pl_module.hp_metric = hp_metric
        else:
            pl_module.hp_metric = min(pl_module.hp_metric, hp_metric)
