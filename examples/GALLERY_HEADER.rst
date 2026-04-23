Examples
========

Configuration examples for stable-pretraining.


# Bayesian Hyperparameter Search with Optuna
Sweeping over a search space is very easy with Hydra and Optuna. `hp_search.yaml` provides an example configuration for performing hyperparameter optimization using Optuna's TPE sampler (bayesian optimization).

First, make sure to install Optuna if you haven't already:
```
pip install optuna
```

Then, make sure to register the callback HPMetricLogger, which will make sure your hyperparameter optimization metric is logged correctly. More complex logic can also be implemented in this callback.

```
from spt.callbacks import HPMetricLogger

callbacks=[HPMetricLogger(metric_name="eval/some_metric")]
```

Finally, make sure your train script contains this code to return the hp_metric to Optuna:
```
...
manager = spt.Manager(...)
manager()


if hasattr(module, "hp_metric"):
    result = module.hp_metric.item()
    if np.isnan(result):
        logger.warning("HP Metric is NaN, returning inf for optimization.")
        result = float("inf")
    logger.info(f"HP Metric: {result}")
    return result
```

Now you can simply run the hyperparameter search and it will automatically run multiple trials:
```
python train.py --config-name=hydra_hp_search
```
It is recommended to use the EarlyStopping callback in combination with hyperparameter optimization to avoid wasting resources on bad trials.
