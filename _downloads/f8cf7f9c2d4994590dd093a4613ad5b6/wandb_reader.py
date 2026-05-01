"""
Reading WandB Runs
==================

Retrieve run data from wandb using the stable_pretraining library.
"""

import stable_pretraining as spt

config, df = spt.utils.reader.wandb_run(
    "excap", "single_dataset_sequential", "p67ng6bq"
)
print(df)
configs, dfs = spt.utils.reader.wandb_project("excap", "single_dataset_sequential")
print(dfs)
