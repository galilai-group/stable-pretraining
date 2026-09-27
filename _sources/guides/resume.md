# Resume training with full state

Use the same model, optimizer, scheduler, callback, and dataset configuration.
Set the new epoch limit above the completed epoch count. From the quickstart:

```bash
python -m stable_pretraining.quickstart --epochs 1 --cache-dir ./first-run
```

Locate the checkpoint under the printed run directory, then resume:

```bash
python -m stable_pretraining.quickstart --epochs 2 --cache-dir ./resumed-run --resume /absolute/path/to/checkpoint.ckpt
```

Replace the checkpoint path with the actual file from your run. The quickstart
passes its resolved path to `spt.Manager(..., ckpt_path=..., weights_only=False)`.
It delegates checkpoint restoration to Lightning. Use only a trusted checkpoint
for full-state loading. A new invocation creates a new run directory; SLURM
requeue has its own Manager-managed continuation behavior.

The regression suite checks deterministic CPU restart at an epoch boundary,
including optimizer, scheduler, EMA teacher, probe, queue, and callback state.
This is not a promise of bitwise equivalence for shuffled data resumed mid-epoch
or across different devices/precision settings.

For Jet, preserve `scale_parameterization` and `scale_eps`. Its state dict
records those settings and rejects incompatible transforms on load. A bounded
checkpoint cannot be silently reinterpreted as an unbounded-scale experiment.
