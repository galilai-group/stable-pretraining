"""Reproduce the full-model RC-aux comparison from the repository root.

Run ``python -m benchmarks.rcaux.reacher_offline --help`` for the experiment
options. The shared example trains the encoder, dynamics, and reachability
head through ``spt.Manager`` and writes per-seed metrics and learning curves.
"""

from examples.rcaux_reacher_offline import main


if __name__ == "__main__":
    main()
