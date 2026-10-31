"""Unit tests for GPU-side batched transforms (``data.gpu_transforms``)."""

import pytest
import torch

from stable_pretraining.data import gpu_transforms as gt


def _rand_batch(batch_size=4, channels=3, size=32, device="cpu"):
    """Synthetic float batch in [0, 1] with the standard (B, C, H, W) layout."""
    return {
        "image": torch.rand(batch_size, channels, size, size, device=device),
        "label": torch.randint(0, 10, (batch_size,), device=device),
    }


@pytest.mark.unit
class TestToDevice:
    """ToDevice should walk all tensor leaves and be idempotent."""

    def test_cpu_to_cpu_is_identity(self):
        batch = _rand_batch()
        before_image_id = id(batch["image"])
        before_label_id = id(batch["label"])
        out = gt.ToDevice(device="cpu", non_blocking=False)(batch)
        assert out["image"].device.type == "cpu"
        assert out["label"].device.type == "cpu"
        assert id(out["image"]) == before_image_id  # idempotent — no copy
        assert id(out["label"]) == before_label_id

    def test_auto_device_resolves(self):
        # device=None should pick the right device automatically.
        op = gt.ToDevice(device=None, non_blocking=False)
        expected_type = "cuda" if torch.cuda.is_available() else "cpu"
        assert op.device.type == expected_type
        batch = _rand_batch()  # CPU batch
        out = op(batch)
        assert out["image"].device.type == expected_type

    def test_walks_nested_dict(self):
        batch = {
            "views": {"a": torch.rand(2, 3, 8, 8), "b": torch.rand(2, 3, 8, 8)},
            "label": torch.zeros(2),
        }
        out = gt.ToDevice(device="cpu", non_blocking=False)(batch)
        assert isinstance(out["views"], dict)
        assert out["views"]["a"].device.type == "cpu"

    def test_walks_list(self):
        batch = {
            "views": [torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)],
            "label": torch.zeros(2),
        }
        out = gt.ToDevice(device="cpu", non_blocking=False)(batch)
        assert isinstance(out["views"], list)
        assert all(v.device.type == "cpu" for v in out["views"])

    def test_ignores_non_tensors(self):
        batch = {"image": torch.rand(2, 3, 8, 8), "meta": "some_string", "n": 7}
        out = gt.ToDevice(device="cpu", non_blocking=False)(batch)
        assert out["meta"] == "some_string"
        assert out["n"] == 7

    @pytest.mark.gpu
    def test_cpu_to_cuda_moves_tensors(self):
        batch = _rand_batch()
        out = gt.ToDevice(device="cuda", non_blocking=True)(batch)
        assert out["image"].device.type == "cuda"
        assert out["label"].device.type == "cuda"

    @pytest.mark.gpu
    def test_cuda_to_cuda_is_idempotent(self):
        batch = _rand_batch(device="cuda")
        before_image_ptr = batch["image"].data_ptr()
        out = gt.ToDevice(device="cuda")(batch)
        assert out["image"].data_ptr() == before_image_ptr  # zero-copy


@pytest.mark.unit
class TestGPUNormalize:
    """GPUNormalize is pure pointwise (x - mean) / std on a batched tensor."""

    def test_shapes_match(self):
        op = gt.GPUNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        batch = _rand_batch(channels=3)
        out = op(batch)
        assert out["image"].shape == batch["image"].shape

    def test_matches_manual(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        op = gt.GPUNormalize(mean=mean, std=std)
        batch = _rand_batch(channels=3, size=16)
        original = batch["image"].clone()
        out = op(batch)
        m = torch.tensor(mean).view(1, 3, 1, 1)
        s = torch.tensor(std).view(1, 3, 1, 1)
        expected = (original - m) / s
        assert torch.allclose(out["image"], expected, atol=1e-6)

    def test_source_target_routing(self):
        op = gt.GPUNormalize(mean=[0.0], std=[1.0], source="raw", target="normed")
        batch = {"raw": torch.rand(2, 1, 8, 8)}
        out = op(batch)
        assert "normed" in out
        assert torch.allclose(out["normed"], batch["raw"])  # identity normalize

    def test_handles_list_of_views(self):
        op = gt.GPUNormalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        views = [torch.rand(2, 3, 8, 8) for _ in range(3)]
        batch = {"image": views}
        out = op(batch)
        assert isinstance(out["image"], list)
        assert len(out["image"]) == 3
        for v_in, v_out in zip(views, out["image"]):
            assert torch.allclose(v_out, v_in)

    def test_handles_dict_of_views(self):
        op = gt.GPUNormalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        views = {"a": torch.rand(2, 3, 8, 8), "b": torch.rand(2, 3, 8, 8)}
        batch = {"image": views}
        out = op(batch)
        assert isinstance(out["image"], dict)
        assert set(out["image"].keys()) == {"a", "b"}


@pytest.mark.unit
class TestKorniaWrappers:
    """Kornia-backed transforms should produce expected output shapes."""

    def test_random_resized_crop(self):
        op = gt.GPURandomResizedCrop(size=24, scale=(0.5, 1.0))
        batch = _rand_batch(size=32)
        out = op(batch)
        assert out["image"].shape == (4, 3, 24, 24)

    def test_random_horizontal_flip(self):
        # p=1.0 forces flip; we can verify by comparing to torch.flip on width axis.
        op = gt.GPURandomHorizontalFlip(p=1.0)
        batch = _rand_batch(size=8)
        original = batch["image"].clone()
        out = op(batch)
        assert out["image"].shape == original.shape
        assert torch.allclose(out["image"], torch.flip(original, dims=[-1]))

    def test_color_jitter(self):
        op = gt.GPUColorJitter(brightness=0.4, contrast=0.4, p=1.0)
        batch = _rand_batch(size=16)
        out = op(batch)
        assert out["image"].shape == batch["image"].shape

    def test_gaussian_blur(self):
        op = gt.GPUGaussianBlur(kernel_size=5, sigma=(0.5, 1.0), p=1.0)
        batch = _rand_batch(size=32)
        out = op(batch)
        assert out["image"].shape == batch["image"].shape

    def test_random_grayscale(self):
        op = gt.GPURandomGrayscale(p=1.0)
        batch = _rand_batch(size=16)
        out = op(batch)
        # kornia returns grayscale broadcast to 3 channels — shape preserved
        assert out["image"].shape == batch["image"].shape

    def test_random_solarize(self):
        op = gt.GPURandomSolarize(p=1.0)
        batch = _rand_batch(size=16)
        out = op(batch)
        assert out["image"].shape == batch["image"].shape

    def test_random_erasing(self):
        op = gt.GPURandomErasing(p=1.0)
        batch = _rand_batch(size=16)
        out = op(batch)
        assert out["image"].shape == batch["image"].shape

    def test_list_of_views_supported(self):
        op = gt.GPURandomHorizontalFlip(p=1.0)
        views = [torch.rand(2, 3, 8, 8) for _ in range(2)]
        out = op({"image": views})
        assert isinstance(out["image"], list)
        assert all(v.shape == (2, 3, 8, 8) for v in out["image"])


@pytest.mark.unit
class TestGPUCompose:
    """GPUCompose chains transforms and (optionally) compiles tensor ops."""

    def test_order_is_preserved(self):
        # Two normalize ops: first divides by 2, second subtracts 0.25.
        # Order matters: ((x/2) - 0.25) != ((x - 0.25)/2)
        op1 = gt.GPUNormalize(mean=[0.0], std=[2.0])  # x / 2
        op2 = gt.GPUNormalize(mean=[0.25], std=[1.0])  # x - 0.25
        chain = gt.GPUCompose([op1, op2], compile=False)
        batch = {"image": torch.ones(1, 1, 4, 4)}
        out = chain(batch)
        # 1.0 -> 0.5 -> 0.25
        assert torch.allclose(out["image"], torch.full_like(out["image"], 0.25))

    def test_compile_disabled_on_cpu(self):
        # ``compile=True`` is gated on CUDA availability; on CPU it should
        # silently fall back to eager.
        chain = gt.GPUCompose(
            [gt.GPUNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
            compile=True,
        )
        # Whether the inner _compiled_op is set depends on cuda availability:
        if torch.cuda.is_available():
            assert chain.transforms[0]._compiled_op is not None
        else:
            assert chain.transforms[0]._compiled_op is None
        batch = _rand_batch(size=16)
        out = chain(batch)
        assert out["image"].shape == batch["image"].shape

    def test_mixed_todevice_and_normalize(self):
        chain = gt.GPUCompose(
            [
                gt.ToDevice(device="cpu", non_blocking=False),
                gt.GPUNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
            compile=False,
        )
        batch = _rand_batch(size=16)
        out = chain(batch)
        assert out["image"].shape == batch["image"].shape

    def test_to_device_does_not_corrupt_forward(self):
        """Regression: a method named ``_apply`` collides with ``nn.Module._apply``.

        When the inner tensor op was named ``_apply``, ``.to("cuda")`` (which
        recursively calls ``Module._apply(fn=convert)``) accidentally invoked
        the dispatch path with the conversion closure, causing kornia ops to
        receive a function instead of a tensor on the next forward.
        """
        chain = gt.GPUCompose(
            [
                gt.GPUNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                gt.GPURandomHorizontalFlip(p=0.5),
            ],
            compile=False,
        )
        chain.to("cpu")  # exercises Module._apply on the cpu path
        batch = _rand_batch(size=16)
        out = chain(batch)
        assert out["image"].shape == batch["image"].shape

    def test_buffers_move_with_module(self):
        op = gt.GPUNormalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        chain = gt.GPUCompose([op], compile=False)
        # mean/std should be registered as buffers (so .to(device) moves them)
        buffer_names = [n for n, _ in chain.named_buffers()]
        assert any("mean" in n for n in buffer_names)
        assert any("std" in n for n in buffer_names)

    @pytest.mark.gpu
    def test_compile_true_runs_on_cuda(self):
        chain = gt.GPUCompose(
            [
                gt.GPUNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                gt.GPURandomHorizontalFlip(p=0.5),
            ],
            compile=True,
        ).to("cuda")
        batch = _rand_batch(size=32, device="cuda")
        out = chain(batch)
        assert out["image"].device.type == "cuda"
        assert out["image"].shape == batch["image"].shape

    @pytest.mark.gpu
    def test_full_ssl_pipeline_gpu(self):
        chain = gt.GPUCompose(
            [
                gt.ToDevice(device="cuda"),
                gt.GPURandomResizedCrop(size=24, scale=(0.5, 1.0)),
                gt.GPURandomHorizontalFlip(p=0.5),
                gt.GPUColorJitter(brightness=0.4, contrast=0.4, p=0.8),
                gt.GPUGaussianBlur(kernel_size=5, sigma=(0.1, 2.0), p=0.5),
                gt.GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
            compile=True,
        ).to("cuda")
        batch = _rand_batch(size=32)
        out = chain(batch)
        assert out["image"].shape == (4, 3, 24, 24)
        assert out["image"].device.type == "cuda"


@pytest.mark.unit
class TestParamRecording:
    """Kornia-backed transforms should record sampled params into the batch dict."""

    def test_hflip_records_params(self):
        op = gt.GPURandomHorizontalFlip(p=0.5)
        batch = _rand_batch(size=16)
        out = op(batch)
        assert "GPURandomHorizontalFlip" in out
        # kornia exposes a ParamItem / dict with at least one tensor leaf;
        # we don't pin the exact schema — just that something got recorded.
        assert out["GPURandomHorizontalFlip"] is not None

    def test_random_resized_crop_records_params(self):
        op = gt.GPURandomResizedCrop(size=24, scale=(0.5, 1.0))
        batch = _rand_batch(size=32)
        out = op(batch)
        assert "GPURandomResizedCrop" in out
        assert out["GPURandomResizedCrop"] is not None

    def test_color_jitter_records_params(self):
        op = gt.GPUColorJitter(brightness=0.4, contrast=0.4, p=1.0)
        batch = _rand_batch(size=16)
        out = op(batch)
        assert "GPUColorJitter" in out

    def test_multiple_same_class_get_unique_keys(self):
        # Two flips back-to-back should land under distinct keys
        op1 = gt.GPURandomHorizontalFlip(p=0.5)
        op2 = gt.GPURandomHorizontalFlip(p=0.5)
        chain = gt.GPUCompose([op1, op2], compile=False)
        batch = _rand_batch(size=8)
        out = chain(batch)
        assert "GPURandomHorizontalFlip" in out
        assert "GPURandomHorizontalFlip_0" in out

    def test_multiview_records_list_of_params(self):
        op = gt.GPURandomHorizontalFlip(p=0.5)
        views = [torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)]
        batch = {"image": views}
        out = op(batch)
        params = out["GPURandomHorizontalFlip"]
        assert isinstance(params, list)
        assert len(params) == 2  # one per view

    def test_record_params_disabled(self):
        # Build a wrapper with record_params=False via the internal API
        from stable_pretraining.data.gpu_transforms import _KorniaWrap
        import kornia.augmentation as K

        op = _KorniaWrap(kornia_op=K.RandomHorizontalFlip(p=0.5), record_params=False)
        batch = _rand_batch(size=8)
        out = op(batch)
        assert "_KorniaWrap" not in out


@pytest.mark.unit
class TestStackedMultiView:
    """One chain on (n_views*B, ...) — the fast path for symmetric SSL."""

    def test_produces_n_views(self):
        chain = gt.GPUCompose(
            [gt.GPUNormalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])],
            compile=False,
        )
        smv = gt.StackedMultiView(chain, n_views=2)
        batch = _rand_batch(size=16)
        out = smv(batch)
        assert "views" in out
        assert "image" not in out  # source consumed
        assert len(out["views"]) == 2
        for v in out["views"]:
            assert v["image"].shape == (4, 3, 16, 16)
            assert torch.equal(v["label"], batch["label"])

    def test_three_views(self):
        chain = gt.GPUCompose(
            [gt.GPUNormalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])],
            compile=False,
        )
        smv = gt.StackedMultiView(chain, n_views=3)
        out = smv(_rand_batch(size=8))
        assert len(out["views"]) == 3

    def test_runs_under_random_aug(self):
        # If kornia samples per-sample params, different views should
        # generally differ (flip with p=1 forces every sample to flip).
        chain = gt.GPUCompose([gt.GPURandomHorizontalFlip(p=1.0)], compile=False)
        smv = gt.StackedMultiView(chain, n_views=2)
        out = smv({"image": torch.rand(2, 3, 4, 4)})
        # With p=1.0 BOTH views are flipped; the input was the same; both
        # outputs should equal the same flipped tensor.
        assert torch.allclose(out["views"][0]["image"], out["views"][1]["image"])


@pytest.mark.unit
class TestMultiView:
    """Per-view chains for asymmetric SSL (BYOL, DINO student/teacher)."""

    def test_per_view_chains(self):
        # Chain A halves; chain B doubles. After application, view 0 == x/2,
        # view 1 == x*2 — so views are distinguishable.
        chain_a = gt.GPUCompose(
            [gt.GPUNormalize(mean=[0.0, 0.0, 0.0], std=[2.0, 2.0, 2.0])],
            compile=False,
        )
        chain_b = gt.GPUCompose(
            [gt.GPUNormalize(mean=[0.0, 0.0, 0.0], std=[0.5, 0.5, 0.5])],
            compile=False,
        )
        mv = gt.MultiView([chain_a, chain_b])
        x = torch.ones(2, 3, 4, 4)
        out = mv({"image": x, "label": torch.zeros(2)})
        assert len(out["views"]) == 2
        assert torch.allclose(out["views"][0]["image"], torch.full_like(x, 0.5))
        assert torch.allclose(out["views"][1]["image"], torch.full_like(x, 2.0))
        assert "image" not in out


@pytest.mark.unit
class TestDataModuleGpuTransform:
    """Module.on_after_batch_transfer falls back to trainer.datamodule.gpu_transform."""

    def _make_module_and_dm(self, dm_gpu_transform):
        from stable_pretraining.module import Module
        from stable_pretraining.data import DataModule
        from torch.utils.data import DataLoader, TensorDataset

        def fwd(self, batch, stage):
            return {"loss": torch.tensor(0.0)}

        # A trivial DataLoader so DataModule construction succeeds.
        loader = DataLoader(TensorDataset(torch.zeros(4, 3)), batch_size=2)
        dm = DataModule(train=loader, val=loader, gpu_transform=dm_gpu_transform)
        module = Module(forward=fwd, hparams={})
        return module, dm

    def test_dm_callable_resolution(self):
        """A callable gpu_transform on the DataModule is found via the fallback path."""
        seen = {}

        def gt_fn(batch):
            seen["called"] = True
            return batch

        module, dm = self._make_module_and_dm(gt_fn)
        # Stub trainer that exposes the datamodule reference and a stage flag.
        module._trainer = type(
            "T",
            (),
            {
                "datamodule": dm,
                "training": True,
                "validating": False,
                "sanity_checking": False,
                "testing": False,
                "predicting": False,
            },
        )()
        module.on_after_batch_transfer({"image": torch.zeros(2, 3)}, dataloader_idx=0)
        assert seen.get("called") is True

    def test_dm_dict_routes_by_stage(self):
        train_seen, val_seen = {}, {}

        def train_t(batch):
            train_seen["x"] = True
            return batch

        def val_t(batch):
            val_seen["x"] = True
            return batch

        module, dm = self._make_module_and_dm({"train": train_t, "val": val_t})
        T = type(
            "T",
            (),
            {
                "datamodule": dm,
                "training": False,
                "validating": True,
                "sanity_checking": False,
                "testing": False,
                "predicting": False,
            },
        )()
        module._trainer = T
        module.on_after_batch_transfer({}, dataloader_idx=0)
        assert val_seen.get("x") is True
        assert "x" not in train_seen


@pytest.mark.unit
class TestDatasetGpuTransform:
    """``dataset.gpu_transform`` is the preferred wiring point.

    Resolution: Module override > dataset.gpu_transform > DataModule.gpu_transform.
    """

    def _make_module_with_dataset(self, gpu_transform):
        from stable_pretraining.module import Module
        from stable_pretraining.data import FromTorchDataset
        from torch.utils.data import DataLoader, TensorDataset

        def fwd(self, batch, stage):
            return {"loss": torch.tensor(0.0)}

        spt_ds = FromTorchDataset(
            TensorDataset(torch.zeros(4, 3)),
            names=["image"],
            gpu_transform=gpu_transform,
        )
        loader = DataLoader(spt_ds, batch_size=2)
        module = Module(forward=fwd, hparams={})
        return module, loader

    def test_dataset_gpu_transform_constructor(self):
        from stable_pretraining.data import FromTorchDataset
        from torch.utils.data import TensorDataset

        marker = lambda b: b  # noqa: E731
        ds = FromTorchDataset(
            TensorDataset(torch.zeros(2, 3)),
            names=["image"],
            gpu_transform=marker,
        )
        assert ds.gpu_transform is marker

    def test_dataset_gpu_transform_picked_up_during_training(self):
        seen = {}

        def gt_fn(batch):
            seen["stage"] = "train"
            return batch

        module, loader = self._make_module_with_dataset(gt_fn)
        module._trainer = type(
            "T",
            (),
            {
                "datamodule": None,
                "training": True,
                "validating": False,
                "sanity_checking": False,
                "testing": False,
                "predicting": False,
                "train_dataloader": loader,
            },
        )()
        module.on_after_batch_transfer({}, dataloader_idx=0)
        assert seen.get("stage") == "train"

    def test_dataset_dropped_during_pickling(self):
        # gpu_transform is stripped in __getstate__ to avoid serialising
        # an nn.Module into every worker spawn.
        import pickle
        from stable_pretraining.data import FromTorchDataset
        from torch.utils.data import TensorDataset

        ds = FromTorchDataset(
            TensorDataset(torch.zeros(2, 3)),
            names=["image"],
            gpu_transform=lambda b: b,
        )
        assert ds.gpu_transform is not None
        restored = pickle.loads(pickle.dumps(ds))
        assert restored.gpu_transform is None  # dropped on pickle

    def test_dataset_wins_over_datamodule(self):
        from stable_pretraining.data import DataModule

        ds_seen, dm_seen = {}, {}

        def ds_t(batch):
            ds_seen["x"] = True
            return batch

        def dm_t(batch):
            dm_seen["x"] = True
            return batch

        module, loader = self._make_module_with_dataset(ds_t)
        dm = DataModule(train=loader, val=loader, gpu_transform=dm_t)
        module._trainer = type(
            "T",
            (),
            {
                "datamodule": dm,
                "training": True,
                "validating": False,
                "sanity_checking": False,
                "testing": False,
                "predicting": False,
                "train_dataloader": loader,
            },
        )()
        module.on_after_batch_transfer({}, dataloader_idx=0)
        assert ds_seen.get("x") is True
        assert "x" not in dm_seen


@pytest.mark.unit
class TestToImageRGB:
    """ToImage(rgb=True) collapses RGB() + ToImage() into one call."""

    def test_grayscale_to_rgb_pil(self):
        from PIL import Image
        from stable_pretraining.data import transforms as cpu_tf

        gray = Image.new("L", (8, 8), color=128)
        t = cpu_tf.ToImage(rgb=True, scale=True)
        out = t({"image": gray})
        assert out["image"].shape[0] == 3  # 3 channels

    def test_rgb_false_default_passthrough(self):
        from PIL import Image
        from stable_pretraining.data import transforms as cpu_tf

        gray = Image.new("L", (8, 8), color=128)
        t = cpu_tf.ToImage(scale=True)  # rgb=False default
        out = t({"image": gray})
        assert out["image"].shape[0] == 1  # stays 1 channel

    def test_rgb_true_passthrough_for_rgb_input(self):
        from PIL import Image
        from stable_pretraining.data import transforms as cpu_tf

        rgb_img = Image.new("RGB", (8, 8), color=(10, 20, 30))
        t = cpu_tf.ToImage(rgb=True, scale=True)
        out = t({"image": rgb_img})
        assert out["image"].shape[0] == 3  # already 3 channels


@pytest.mark.unit
class TestModuleHook:
    """Module.on_after_batch_transfer should be a passthrough when nothing is wired up."""

    def test_no_gpu_transform_is_passthrough(self):
        from stable_pretraining.module import Module

        def fwd(self, batch, stage):
            return {"loss": batch["image"].mean()}

        m = Module(forward=fwd, hparams={})
        batch = _rand_batch(size=8)
        out = m.on_after_batch_transfer(batch, dataloader_idx=0)
        assert out is batch  # exact passthrough

    def test_module_level_gpu_transform_is_rejected(self):
        """Setting ``gpu_transform`` on the Module should fail loudly at on_train_start."""
        from stable_pretraining.module import Module

        def fwd(self, batch, stage):
            return {"loss": torch.tensor(0.0)}

        m = Module(forward=fwd, hparams={}, gpu_transform=lambda b: b)
        # Stub the bits ``on_train_start`` would touch before our guard.
        with pytest.raises(RuntimeError, match="not supported"):
            m.on_train_start()


# ---------------------------------------------------------------------------
# Video / multi-crop / decode / prefetch
# ---------------------------------------------------------------------------


def _static_clip(batch_size=2, frames=6, size=40):
    """Clips whose frames are all identical (any per-frame augmentation shows up)."""
    img = torch.rand(batch_size, 1, 3, size, size)
    return img.expand(batch_size, frames, 3, size, size).contiguous()


def _max_frame_diff(views):
    return max((v - v[:, :1]).abs().max().item() for v in views)


LEVJEPA_GROUPS = [
    dict(n_views=1, size=24, scale=(0.8, 1.0)),
    dict(
        n_views=3,
        size=12,
        scale=(0.02, 0.4),
        hflip_p=0.5,
        color_jitter=(0.4, 0.4, 0.2, 0.1),
        color_jitter_p=0.8,
        grayscale_p=0.2,
    ),
]


@pytest.mark.unit
class TestRandomResizedCropMode:
    """GPURandomResizedCrop forwards kornia's cropping_mode."""

    @pytest.mark.parametrize("mode", ["slice", "resample"])
    def test_mode_forwarded_and_shape(self, mode):
        op = gt.GPURandomResizedCrop(size=16, cropping_mode=mode)
        assert op.kornia_op.flags["cropping_mode"] == mode
        out = op(_rand_batch(size=32))
        assert out["image"].shape == (4, 3, 16, 16)

    def test_default_unchanged(self):
        assert (
            gt.GPURandomResizedCrop(size=16).kornia_op.flags["cropping_mode"] == "slice"
        )


@pytest.mark.unit
class TestGPUVideoSequential:
    """One parameter set per clip, shared by its frames."""

    def test_frames_share_params(self):
        import kornia.augmentation as K

        aug = gt.GPUVideoSequential(
            K.RandomResizedCrop((16, 16), scale=(0.1, 0.5), cropping_mode="resample"),
            gt.GPURandomHorizontalFlip(),
            K.ColorJitter(0.4, 0.4, 0.2, 0.1, p=1.0),
        )
        out = aug({"image": _static_clip()})["image"]
        assert out.shape == (2, 6, 3, 16, 16)
        assert _max_frame_diff([out]) < 1e-6

    def test_image_wrapper_varies_per_frame(self):
        # Contrast: the image wrapper on flattened frames samples per frame.
        clip = _static_clip()
        flat = clip.view(-1, 3, 40, 40)
        out = gt.GPURandomResizedCrop(size=16, scale=(0.1, 0.5))({"image": flat})[
            "image"
        ]
        assert _max_frame_diff([out.view(2, 6, 3, 16, 16)]) > 1e-3


@pytest.mark.unit
class TestGPUMultiCrop:
    """Batched multi-crop for images and clips."""

    def test_clip_shapes_dtype_and_labels(self):
        clip = (_static_clip() * 255).to(torch.uint8)
        batch = {"image": clip, "label": torch.tensor([3, 7])}
        out = gt.GPUMultiCrop(LEVJEPA_GROUPS)(batch)
        assert "image" not in out
        views = out["views"]
        assert len(views) == 4
        assert views[0]["image"].shape == (2, 6, 3, 24, 24)
        assert all(v["image"].shape == (2, 6, 3, 12, 12) for v in views[1:])
        assert all(v["image"].dtype == torch.uint8 for v in views)
        assert torch.equal(views[2]["label"], torch.tensor([3, 7]))
        assert len(out["GPUMultiCrop"]) == 2

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="backend"):
            gt.GPUMultiCrop(LEVJEPA_GROUPS, backend="cuda")

    def test_triton_backend_requires_cuda_uint8(self):
        with pytest.raises(RuntimeError, match="triton"):
            gt.GPUMultiCrop(LEVJEPA_GROUPS, backend="triton")({"image": _static_clip()})

    @pytest.mark.gpu
    @pytest.mark.parametrize("seed", range(6))
    def test_fused_matches_torch(self, seed):
        # Padded batch of two sizes, every colour op at full probability: the
        # fused kernel and the torch path draw identical params (same seed).
        groups = [
            dict(n_views=2, size=(20, 28), scale=(0.3, 1.0), hflip_p=0.5),
            dict(
                n_views=5,
                size=12,
                scale=(0.02, 0.6),
                hflip_p=0.5,
                color_jitter=(0.4, 0.4, 0.2, 0.1),
                color_jitter_p=1.0,
                grayscale_p=0.3,
            ),
        ]
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(0, 256, (3, 4, 3, 36, 44), generator=g, dtype=torch.uint8)
        x = (
            torch.nn.functional.avg_pool2d(x.view(12, 3, 36, 44).float(), 3, 1, 1)
            .round()
            .to(torch.uint8)
            .view(3, 4, 3, 36, 44)
            .cuda()
        )
        sizes = torch.tensor([[36, 44], [30, 44], [36, 37]], device="cuda")
        outs = []
        for backend in ("triton", "torch"):
            torch.manual_seed(seed)
            mc = gt.GPUMultiCrop(
                groups, size_key="size", output="groups", backend=backend
            )
            outs.append(mc({"image": x.clone(), "size": sizes})["views"])
        for a, b in zip(*outs):
            d = (a.int() - b.int()).abs()
            assert a.shape == b.shape and a.dtype == torch.uint8
            assert d.float().mean() < 0.05 and d.max() <= 3

    @pytest.mark.gpu
    def test_fused_is_default_on_cuda_uint8(self):
        mc = gt.GPUMultiCrop(LEVJEPA_GROUPS)
        x = (_static_clip() * 255).to(torch.uint8).cuda()
        assert mc._use_fused(x) and not mc._use_fused(x.float())
        out = mc({"image": x})
        assert _max_frame_diff([v["image"].float() for v in out["views"]]) == 0

    def test_frames_share_params(self):
        out = gt.GPUMultiCrop(LEVJEPA_GROUPS)({"image": _static_clip()})
        assert _max_frame_diff([v["image"] for v in out["views"]]) < 1e-6

    def test_images_and_groups_output(self):
        x = torch.rand(5, 3, 32, 32)
        out = gt.GPUMultiCrop(LEVJEPA_GROUPS, output="groups")({"image": x})
        g0, g1 = out["views"]
        assert g0.shape == (5, 1, 3, 24, 24) and g1.shape == (5, 3, 3, 12, 12)
        assert g1.dtype == torch.float32 and 0 <= g1.min() and g1.max() <= 1

    def test_full_crop_is_exact(self):
        # scale/ratio pinned to the whole square image at native size: roi_align
        # then samples exactly the pixel centres.
        x = torch.rand(3, 2, 3, 20, 20)
        mc = gt.GPUMultiCrop(
            [dict(n_views=2, size=20, scale=(1.0, 1.0), ratio=(1.0, 1.0))]
        )
        for v in mc({"image": x.clone()})["views"]:
            assert torch.allclose(v["image"], x, atol=1e-6)

    def test_views_differ(self):
        out = gt.GPUMultiCrop([dict(n_views=2, size=16, scale=(0.05, 0.3))])(
            {"image": torch.rand(4, 3, 64, 64)}
        )
        assert not torch.allclose(out["views"][0]["image"], out["views"][1]["image"])

    def test_padding_never_sampled(self):
        # Valid 30x20 region of ones; padding replicates the edge (GPUDecodeJPEG's
        # convention) and a sentinel lies past it -- crops must never reach it.
        x = torch.full((4, 3, 48, 48), -5.0)
        x[:, :, :31, :21] = 1.0
        batch = {"image": x, "image_size": torch.tensor([[30, 20]] * 4)}
        mc = gt.GPUMultiCrop(
            [dict(n_views=3, size=8, scale=(0.05, 1.0))], size_key="image_size"
        )
        for v in mc(batch)["views"]:
            assert torch.allclose(v["image"], torch.ones_like(v["image"]))

    def test_hue_matches_torchvision(self):
        from torchvision.transforms.v2 import functional as F

        x = torch.rand(2, 3, 16, 16)
        for h in (-0.1, 0.05, 0.3):
            hsv = gt._rgb_to_hsv(x)
            hsv[..., 0, :, :] = (hsv[..., 0, :, :] + h) % 1.0
            ours = gt._hsv_to_rgb(hsv)
            assert torch.allclose(ours, F.adjust_hue(x, h), atol=1e-4)

    def test_hsv_roundtrip(self):
        x = torch.rand(4, 3, 8, 8)
        assert torch.allclose(gt._hsv_to_rgb(gt._rgb_to_hsv(x)), x, atol=1e-5)

    def test_unknown_group_key_raises(self):
        with pytest.raises(ValueError, match="unknown view-group keys"):
            gt.GPUMultiCrop([dict(n_views=1, size=8, scale=(0.5, 1.0), jitter=1)])

    @pytest.mark.gpu
    def test_cuda(self):
        clip = (_static_clip(batch_size=3) * 255).to(torch.uint8).cuda()
        out = gt.GPUMultiCrop(LEVJEPA_GROUPS)({"image": clip})
        assert all(v["image"].is_cuda for v in out["views"])
        assert _max_frame_diff([v["image"].float() for v in out["views"]]) == 0


def _jpeg(color, h=24, w=32):
    from torchvision.io import encode_jpeg

    img = torch.zeros(3, h, w, dtype=torch.uint8)
    img[:] = torch.tensor(color, dtype=torch.uint8).view(3, 1, 1)
    return encode_jpeg(img, quality=95)


@pytest.mark.unit
class TestGPUDecodeJPEG:
    """Batched JPEG decode (CPU fallback in unit tests)."""

    def test_images_match_torchvision(self):
        from torchvision.io import decode_jpeg

        enc = [_jpeg((10 * i, 50, 200)) for i in range(4)]
        out = gt.GPUDecodeJPEG(device="cpu")(
            {"image_bytes": [bytes(e.numpy()) for e in enc]}
        )
        assert out["image"].shape == (4, 3, 24, 32)
        assert torch.equal(out["image"], torch.stack(decode_jpeg(enc)))
        assert out["image_size"].tolist() == [[24, 32]] * 4

    def test_clip_layout_from_default_collate(self):
        from torch.utils.data import default_collate

        # B=2 samples with T=3 frames each; frame (b, t) has red = 40 * (3b + t).
        samples = [
            {
                "video_bytes": [
                    bytes(_jpeg((40 * (3 * b + t), 0, 0)).numpy()) for t in range(3)
                ]
            }
            for b in range(2)
        ]
        batch = default_collate(samples)
        out = gt.GPUDecodeJPEG(source="video_bytes", target="video", device="cpu")(
            batch
        )
        assert out["video"].shape == (2, 3, 3, 24, 32)
        red = out["video"][:, :, 0].float().mean((-2, -1))
        expect = torch.tensor(
            [[40.0 * (3 * b + t) for t in range(3)] for b in range(2)]
        )
        assert torch.allclose(red, expect, atol=3)

    def test_mixed_sizes_are_padded(self):
        enc = [
            bytes(_jpeg((255, 0, 0), 16, 16).numpy()),
            bytes(_jpeg((0, 255, 0), 24, 40).numpy()),
        ]
        out = gt.GPUDecodeJPEG(device="cpu")({"image_bytes": enc})
        assert out["image"].shape == (2, 3, 24, 40)
        assert out["image_size"].tolist() == [[16, 16], [24, 40]]
        first = out["image"][0]
        assert torch.equal(first[:, 16:], first[:, 15:16].expand(3, 8, 40))
        assert torch.equal(first[:, :16, 16:], first[:, :16, 15:16].expand(3, 16, 24))

    @pytest.mark.gpu
    def test_cuda_matches_cpu(self):
        enc = [bytes(_jpeg((30 * i, 90, 160)).numpy()) for i in range(6)]
        cpu = gt.GPUDecodeJPEG(device="cpu")({"image_bytes": list(enc)})["image"]
        gpu = gt.GPUDecodeJPEG(device="cuda")({"image_bytes": list(enc)})["image"]
        assert gpu.is_cuda
        assert (gpu.cpu().int() - cpu.int()).abs().max() <= 3


class _Double(torch.nn.Module):
    def forward(self, batch):
        batch["image"] = batch["image"] * 2
        return batch


def _loader(n=6, batch_size=2):
    from stable_pretraining.data import FromTorchDataset
    from torch.utils.data import DataLoader, TensorDataset

    ds = FromTorchDataset(
        TensorDataset(torch.arange(n * 3, dtype=torch.float32).view(n, 3)),
        names=["image"],
        gpu_transform=_Double(),
    )
    return DataLoader(ds, batch_size=batch_size)


@pytest.mark.unit
class TestGPUPrefetcher:
    """Prefetching wrapper, its DataModule switch and Module integration."""

    def test_cpu_path_applies_transform(self):
        pf = gt.GPUPrefetcher(_loader(), transform=_Double(), device="cpu")
        batches = list(pf)
        assert len(pf) == 3 and len(batches) == 3
        assert all(isinstance(b, gt._PrefetchedBatch) for b in batches)
        assert torch.equal(batches[0]["image"], torch.arange(6.0).view(2, 3) * 2)
        assert pf.batch_size == 2  # attribute fall-through to the DataLoader

    def test_module_skips_prefetched_batches(self):
        from stable_pretraining.module import Module

        def boom(batch):
            raise AssertionError("GPU transform must not run twice")

        module = Module(
            forward=lambda self, b, s: {"loss": torch.tensor(0.0)}, hparams={}
        )
        module._trainer = type(
            "T",
            (),
            {
                "datamodule": type("DM", (), {"gpu_transform": boom})(),
                "training": True,
                "validating": False,
                "sanity_checking": False,
                "testing": False,
                "predicting": False,
            },
        )()
        out = module.on_after_batch_transfer(
            gt._PrefetchedBatch(image=torch.zeros(1)), 0
        )
        assert type(out) is dict

    def test_marker_survives_lightning_transfer(self):
        from lightning.fabric.utilities.apply_func import move_data_to_device

        moved = move_data_to_device(gt._PrefetchedBatch(image=torch.zeros(2)), "cpu")
        assert isinstance(moved, gt._PrefetchedBatch)

    def test_datamodule_switch_resolves_dataset_transform(self):
        from stable_pretraining.data import DataModule

        dm = DataModule(train=_loader(), val=_loader(), gpu_prefetch=True)
        loader = dm.train_dataloader()
        assert isinstance(loader, gt.GPUPrefetcher)
        assert isinstance(loader.transform, _Double)

    @pytest.mark.gpu
    @pytest.mark.parametrize("background", [True, False])
    def test_cuda_prefetch_matches_synchronous(self, background):
        pf = gt.GPUPrefetcher(
            _loader(n=8),
            transform=_Double(),
            device="cuda",
            depth=2,
            background=background,
        )
        got = [b["image"] for b in pf]
        assert all(t.is_cuda for t in got)
        expect = torch.arange(24.0).view(8, 3) * 2
        assert torch.equal(torch.cat(got).cpu(), expect)

    @pytest.mark.gpu
    def test_background_thread_stops_on_early_exit(self):
        import threading

        it = iter(gt.GPUPrefetcher(_loader(n=40), transform=_Double(), device="cuda"))
        next(it)
        it.close()
        assert not any(t.name == "GPUPrefetcher" for t in threading.enumerate())

    @pytest.mark.gpu
    def test_background_errors_reach_the_consumer(self):
        class Boom(torch.nn.Module):
            def forward(self, batch):
                raise RuntimeError("bad batch")

        with pytest.raises(RuntimeError, match="bad batch"):
            list(gt.GPUPrefetcher(_loader(), transform=Boom(), device="cuda"))
