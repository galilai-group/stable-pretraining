# This experiment uses stable-pretraining

Use `import stable_pretraining as spt`. Check the installed package version and
the experiment configuration before changing APIs or loading a checkpoint.

Keep training, validation, optimizers, online probes, and checkpoint lifecycle
in `spt.Module`, `spt.data.DataModule`, the existing callbacks, and `spt.Manager`.
Put the experiment's novel computation in its forward callable. It receives a
dict batch and runtime stage `fit`, `validate`, or `test`, and returns a dict
with a scalar tensor `loss` during training.

Inspect a real batch before assuming view layout. The starter's SimCLR path
uses `batch['views']`, a list of two image/label dicts. Preserve matching feature
and label batch lengths for online evaluation.

Run the short CPU example after changes. Test nontrivial new mathematical
behavior against an independent reference. For Jet, preserve the full token
output for inverse/logdet checks; pooled features are not invertible.

Reuse the existing experiment's dependencies and user choices. Do not launch
long training runs, sweeps, or publish results unless requested.
