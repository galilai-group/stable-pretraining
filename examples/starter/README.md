# An editable SSL experiment

Install the current stable-pretraining source checkout first; the packaged
quickstart is unreleased. Copy this directory into your own project, then extract
the runnable example from that installed package:

```bash
python -c "from importlib.resources import files; from pathlib import Path; Path('train.py').write_text(files('stable_pretraining').joinpath('quickstart.py').read_text())"
python train.py --cache-dir ./runs-cache
```

This gives you a standalone, editable script rather than a new training engine.
It delegates training, online probing, checkpoints, and registry logging to
stable-pretraining. Modify its backbone or forward callable for your research.

```bash
python train.py --data ./images --epochs 2 --cache-dir ./runs-cache
spt web ./runs-cache/runs
```

The image directory needs `train/<class>/*` and `val/<class>/*`. The script's
16x16 crops and tiny encoder are for onboarding. Select an appropriate research
configuration before treating results as a benchmark. Save the installed
package version/source revision with your experiment.

The accompanying `AGENTS.md` is opt-in project guidance. Remove or adapt it if
this project does not use stable-pretraining.
