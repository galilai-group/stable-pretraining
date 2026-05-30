"""Tests for the implicit performance defaults set by ``stable_pretraining``.

Covers:

- ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` env var set on import
- DataModule applies ``pin_memory=True`` and ``persistent_workers=True``
  defaults to dict-spec configs (without overriding user-provided values)
- cuDNN SDPA preference is applied on Hopper (gpu-marked; on non-Hopper
  the assertion is just that nothing crashed)
"""

import os

import pytest
import torch


@pytest.mark.unit
class TestEnvDefaults:
    """``PYTORCH_CUDA_ALLOC_CONF`` should be set after importing stable_pretraining."""

    def test_expandable_segments_set(self):
        import stable_pretraining  # noqa: F401 — import has the side effect

        assert "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""), (
            "Expected stable_pretraining import to set "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        )


@pytest.mark.unit
class TestDataLoaderDefaults:
    """``DataModule._get_loader_kwargs`` injects sensible defaults."""

    def _make_dm(self, conf):
        from stable_pretraining.data import DataModule

        # We need a real DataModule but the dict path runs through
        # ``_format_data_conf`` validation. Build a minimal valid dict.
        return DataModule(train=conf, val=conf)

    def test_pin_memory_defaults_on(self):
        from omegaconf import OmegaConf
        from stable_pretraining.data import DataModule

        conf = OmegaConf.create(
            {
                "dataset": {"_target_": "stable_pretraining.data.HFDataset"},
                "batch_size": 4,
            }
        )
        dm = DataModule.__new__(DataModule)
        kwargs = dm._get_loader_kwargs(conf, dataset=None)
        assert kwargs["pin_memory"] is True

    def test_pin_memory_respects_user(self):
        from omegaconf import OmegaConf
        from stable_pretraining.data import DataModule

        conf = OmegaConf.create(
            {
                "dataset": {"_target_": "stable_pretraining.data.HFDataset"},
                "batch_size": 4,
                "pin_memory": False,
            }
        )
        dm = DataModule.__new__(DataModule)
        kwargs = dm._get_loader_kwargs(conf, dataset=None)
        assert kwargs["pin_memory"] is False  # user override wins

    def test_persistent_workers_with_workers(self):
        from omegaconf import OmegaConf
        from stable_pretraining.data import DataModule

        conf = OmegaConf.create(
            {
                "dataset": {"_target_": "stable_pretraining.data.HFDataset"},
                "batch_size": 4,
                "num_workers": 4,
            }
        )
        dm = DataModule.__new__(DataModule)
        kwargs = dm._get_loader_kwargs(conf, dataset=None)
        assert kwargs["persistent_workers"] is True

    def test_persistent_workers_not_set_without_workers(self):
        from omegaconf import OmegaConf
        from stable_pretraining.data import DataModule

        conf = OmegaConf.create(
            {
                "dataset": {"_target_": "stable_pretraining.data.HFDataset"},
                "batch_size": 4,
                "num_workers": 0,
            }
        )
        dm = DataModule.__new__(DataModule)
        kwargs = dm._get_loader_kwargs(conf, dataset=None)
        # PyTorch errors if persistent_workers=True with num_workers=0,
        # so we must NOT inject the default in that case.
        assert "persistent_workers" not in kwargs

    def test_prefetch_factor_unchanged(self):
        from omegaconf import OmegaConf
        from stable_pretraining.data import DataModule

        # prefetch_factor stays on the PyTorch default — spt doesn't touch it.
        conf = OmegaConf.create(
            {
                "dataset": {"_target_": "stable_pretraining.data.HFDataset"},
                "batch_size": 4,
                "num_workers": 4,
            }
        )
        dm = DataModule.__new__(DataModule)
        kwargs = dm._get_loader_kwargs(conf, dataset=None)
        assert "prefetch_factor" not in kwargs


@pytest.mark.unit
class TestCudnnSdpaPreference:
    """The cuDNN preference flip should run cleanly regardless of GPU."""

    def test_deferred_init_does_not_crash(self):
        # Just touching a lazy attribute triggers ``_do_deferred_init``
        # which contains the cuDNN preference flip.
        import stable_pretraining as spt

        _ = spt.Module  # noqa: B018 — access triggers deferred init
        # No assertion about preferred_sdp_backend here: on non-Hopper
        # or older torch the flip is a no-op. The test just guarantees
        # that the flip code does not raise.

    @pytest.mark.gpu
    def test_cudnn_preferred_on_hopper(self):
        # On Hopper (sm_90+) the deferred init should have set the
        # backend to "cudnn". Skip otherwise.
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            pytest.skip("not Hopper or newer")
        if not hasattr(torch.backends.cuda, "preferred_sdp_backend"):
            pytest.skip("torch version lacks preferred_sdp_backend")
        import stable_pretraining as spt

        _ = spt.Module  # noqa: B018 — trigger deferred init
        # The attribute may be a string or enum depending on torch version.
        pref = torch.backends.cuda.preferred_sdp_backend
        assert "cudnn" in str(pref).lower()
