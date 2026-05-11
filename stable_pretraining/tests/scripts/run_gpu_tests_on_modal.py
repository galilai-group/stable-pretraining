r"""Run GPU-marked tests on Modal.

Most of the FSDP / distributed test suite is gated with ``@pytest.mark.gpu``
because PyTorch's FSDP requires a non-CPU accelerator at forward time, and
the keystone equivalence tests need ``world_size >= 2``. This script makes
that suite runnable from any laptop by spinning up a Modal container with
the requested number of GPUs, mounting the working tree, and streaming
``pytest`` output back.

Setup (once per machine)
------------------------

::

    uv pip install modal
    modal setup     # opens a browser; drops a token at ~/.modal.toml

Usage
-----

::

    # Full GPU suite on the default tier (2× A10G ≈ $0.30 for ~5 min)
    modal run stable_pretraining/tests/scripts/run_gpu_tests_on_modal.py

    # Just the equivalence suite
    modal run stable_pretraining/tests/scripts/run_gpu_tests_on_modal.py \\
        --test stable_pretraining/tests/distributed/test_ddp_fsdp2_equivalence.py

    # One specific test
    modal run stable_pretraining/tests/scripts/run_gpu_tests_on_modal.py \\
        --test "stable_pretraining/tests/distributed/test_ddp_fsdp2_equivalence.py::test_supervised_one_step_equivalence"

    # Bigger / faster GPUs for the memory regression suite
    modal run stable_pretraining/tests/scripts/run_gpu_tests_on_modal.py --gpu A100:2 \\
        --test stable_pretraining/tests/distributed/test_fsdp2_memory.py

    # Override pytest marker (default: ``gpu``)
    modal run stable_pretraining/tests/scripts/run_gpu_tests_on_modal.py --marker "gpu and not slow"

GPU tiers and approximate cost (full equivalence suite, ~1 min wall clock)
-------------------------------------------------------------------------

================  ================================  ===================
``--gpu`` value   Hardware                          Approx. cost / run
================  ================================  ===================
``T4:2``          2x NVIDIA T4 (16 GB each)         ~$0.10
``L4:2``          2x NVIDIA L4 (24 GB each)         ~$0.15
``A10G:2``        2x NVIDIA A10G (24 GB each)       ~$0.20  (default)
``A100:2``        2x NVIDIA A100 (40 GB each)       ~$0.70
================  ================================  ===================

Costs are per-second on Modal; a 1-minute run on 2x A10G is roughly $0.20.
The first build takes longer (~5 min) because the PyTorch wheel is fetched
fresh; subsequent runs reuse the cached image and start in <30 s.

Notes:
-----
- The image build mounts the working tree via ``add_local_dir`` honoring
  ``.gitignore`` — the venv, build artifacts, and run outputs aren't shipped.
- ``--index-url https://pypi.org/simple/`` overrides Modal's
  ``pypi-mirror.modal.local`` whose package metadata can lag upstream PyPI
  (we hit this on the ``lightning`` quarantine of 2026-04).
- ``setuptools_scm`` is configured in ``pyproject.toml`` and shells out to
  the ``git`` binary at install time, so we install ``git`` in the image.
"""

from __future__ import annotations

# Self-exec: ``python scripts/run_gpu_tests_on_modal.py ...`` is equivalent
# to ``modal run scripts/run_gpu_tests_on_modal.py ...``. This block runs
# **before** ``import modal`` so the launcher path doesn't require the
# Modal Python library locally — it only needs the ``modal`` CLI on
# ``$PATH``. ``modal run`` imports this file as a regular module
# (``__name__ != "__main__"``), so it doesn't re-trigger this branch.
if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(subprocess.call(["modal", "run", __file__, *sys.argv[1:]]))


import modal  # noqa: E402 — must follow the __main__ guard above


image = (
    modal.Image.debian_slim(python_version="3.10")
    # ``setuptools_scm`` shells out to ``git`` at install time to read the
    # version from the tag history.
    .apt_install("git")
    # Ship the working tree (no git push needed). Honor ``.gitignore`` so
    # the venv, build artifacts, and run outputs don't get uploaded.
    .add_local_dir(
        ".",
        "/repo",
        copy=True,
        ignore=modal.FilePatternMatcher.from_file(".gitignore"),
    )
    # Use plain ``pip`` here, not ``uv``. The Modal sandbox has no
    # preinstalled torch so the resolver walks the full lightning → torch →
    # nvidia-* CUDA wheel graph from scratch; uv's strict resolver finds a
    # combinatorial conflict that pip's looser resolver doesn't.
    # ``--index-url`` bypasses Modal's preconfigured mirror.
    .run_commands(
        'cd /repo && pip install --index-url https://pypi.org/simple/ -e ".[dev]"'
    )
)

app = modal.App("stable-pretraining-gpu-tests", image=image)


# Modal pins the GPU tier at decoration time, not call time, so we need one
# registered function per supported tier. ``serialized=True`` lets us nest
# the ``@app.function`` decorator inside a factory (otherwise Modal's
# AST-based serialization requires the decorated function at module scope).
def _make_runner(gpu: str):
    # ``name=`` is required: without it, every registration shares the same
    # qualified name (``_make_runner.<locals>.run_pytest``) and they
    # overwrite each other in Modal's app registry — silently routing every
    # call to the last-registered tier.
    safe_name = f"run_pytest_{gpu.replace(':', 'x').replace('-', '_').lower()}"

    @app.function(gpu=gpu, timeout=60 * 30, serialized=True, name=safe_name)
    def run_pytest(test_filter: str = "", marker: str = "gpu") -> int:
        # Body is self-contained on purpose: ``serialized=True`` cloudpickles
        # the closure, and the remote container can't ``import`` this script
        # (it lives in ``scripts/`` which isn't on the container's PYTHONPATH).
        # Anything referenced from the outer module namespace would fail to
        # deserialize.
        import subprocess

        args = ["pytest", "-v", "--tb=short", "-m", marker]
        if test_filter:
            args.append(test_filter)
        else:
            args.append("stable_pretraining/tests/distributed/")

        print(f"[modal] running: {' '.join(args)}", flush=True)
        return subprocess.run(args, cwd="/repo").returncode

    return run_pytest


_SUPPORTED_GPUS = ("T4:2", "L4:2", "A10G:2", "A100:2")
_RUNNERS = {gpu: _make_runner(gpu) for gpu in _SUPPORTED_GPUS}


@app.local_entrypoint()
def main(test: str = "", marker: str = "gpu", gpu: str = "A10G:2"):
    """Spawn the test run on Modal and stream results back.

    Args:
        test: Pytest path filter (e.g. ``path/to/test.py::test_name``).
            Empty string means run the entire ``tests/distributed/`` tree.
        marker: Pytest marker expression. Default ``gpu`` runs everything
            gated behind ``@pytest.mark.gpu``. Pass e.g.
            ``"gpu and not slow"`` to narrow further.
        gpu: GPU tier — one of ``T4:2``, ``L4:2``, ``A10G:2``, ``A100:2``.
            See module docstring for cost guidance.
    """
    if gpu not in _RUNNERS:
        raise SystemExit(
            f"Unknown --gpu={gpu!r}. Supported: {sorted(_RUNNERS)}.\n"
            "Edit _RUNNERS in this file to add more tiers."
        )

    print(f"[modal] gpu={gpu}  marker={marker!r}  test={test or '<all>'}")
    rc = _RUNNERS[gpu].remote(test_filter=test, marker=marker)
    if rc != 0:
        raise SystemExit(rc)
    print("[modal] all tests passed ✓")
