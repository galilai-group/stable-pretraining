# Run your first SSL experiment

This guide targets the current source checkout. The packaged quickstart and Jet
are unreleased additions; an older PyPI release does not include them.

```bash
git clone https://github.com/galilai-group/stable-pretraining.git
cd stable-pretraining
python -m pip install -e .
python -m stable_pretraining.quickstart
spt web ./spt-quickstart/runs
```

The command uses synthetic images, a small convolutional encoder, SimCLR,
an online classification probe, and a CPU Trainer through `spt.Manager`.
It completes four backbone optimizer steps and validates on separate synthetic
images. No model or dataset download is needed. A successful run prints
`Completed 4 optimizer steps` and a command for the local viewer.

The registry stores metrics under the selected cache directory and Lightning
writes a checkpoint in the run directory. Inspect `fit/loss` and
`eval/probe_accuracy` in the viewer. These synthetic metrics verify that training
and evaluation work; they are not a benchmark.

```bash
python -m stable_pretraining.quickstart --epochs 2 --cache-dir ./my-first-run
python -m stable_pretraining.quickstart --method jet-entropy --cache-dir ./jet-demo
```

The second command exercises MSE prediction minus exact full-token logdet with
the invertible Jet encoder. It is a demonstration, not a tuned training recipe.

For a published release, install `python -m pip install stable-pretraining`
and consult documentation for that release. Check what is installed with:

```bash
python -c "import importlib.metadata as m; print(m.version('stable-pretraining'))"
```

For an editable experiment, see the [starter project](https://github.com/galilai-group/stable-pretraining/tree/main/examples/starter).
Next: [custom images](custom_images.md), [online evaluation](online_evaluation.md),
[checkpoint/resume](resume.md), and [Jet](jet.md).
