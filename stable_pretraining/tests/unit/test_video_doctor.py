"""Unit tests for ``spt video-pipeline doctor`` (stable_pretraining.data.video_doctor)."""

import json
import os

import cv2
import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from stable_pretraining import cli
from stable_pretraining.data import video_doctor as vd
from stable_pretraining.data.video import build_lance_video_dataset
from stable_pretraining.tests.unit.test_video import _write_mp4

pytestmark = pytest.mark.unit
runner = CliRunner()


def _jpeg(img: np.ndarray, *flags: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90, *flags])
    assert ok
    return buf.tobytes()


class TestJpegHeader:
    """Frame-header parsing used by the store checks."""

    img = np.random.default_rng(0).integers(0, 256, (40, 56, 3), dtype=np.uint8)

    def test_baseline_420(self):
        h = vd.jpeg_header(_jpeg(self.img))
        assert h == {"height": 40, "width": 56, "chroma": "4:2:0", "progressive": False}

    def test_444_and_progressive(self):
        h = vd.jpeg_header(
            _jpeg(
                self.img,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
                cv2.IMWRITE_JPEG_PROGRESSIVE,
                1,
            )
        )
        assert h["chroma"] == "4:4:4" and h["progressive"]

    def test_gray(self):
        assert vd.jpeg_header(_jpeg(self.img[..., 0]))["chroma"] == "gray"

    def test_not_jpeg(self):
        png = cv2.imencode(".png", self.img)[1].tobytes()
        assert vd.jpeg_header(png) is None
        assert vd.jpeg_header(b"\xff\xd8\x00") is None


def test_usable_cpus_is_bounded():
    n = vd.usable_cpus()
    assert 0 < n <= (os.cpu_count() or 1)


class TestReport:
    """Report rendering, status and JSON."""

    def _report(self):
        r = vd.Report()
        r.add("a", vd.OK, "fine")
        r.add("b", vd.WARN, "meh", "do x")
        r.throughput = {"stage 1": 100.0, "stage 2": 50.0}
        r.bottleneck = "stage 2"
        return r

    def test_format_marks_fixes_and_bottleneck(self):
        text = self._report().format()
        assert "✓ a" in text and "! b" in text and "→ do x" in text
        assert "← bottleneck" in text.splitlines()[-3]
        assert "0 failed, 1 warnings." in text

    def test_ok_and_json(self):
        r = self._report()
        assert r.ok
        r.add("c", vd.FAIL, "broken")
        d = json.loads(r.to_json())
        assert d["ok"] is False and d["bottleneck"] == "stage 2"
        assert [c["status"] for c in d["checks"]] == ["ok", "warn", "fail"]


class TestStoreChecks:
    """Lance store validation."""

    @pytest.fixture
    def videos(self, tmp_path):
        d = tmp_path / "videos"
        for i in range(3):
            _write_mp4(d / f"{i}.mp4", n_frames=40)
        return d

    def _store_checks(self, store):
        r = vd.Report()
        vd._check_store(r, str(store), hw_ok=True, device=0)
        return {c.name: c for c in r.checks}

    def test_webp_store_fails(self, videos, tmp_path):
        out = tmp_path / "w.lance"
        build_lance_video_dataset(videos, out, workers=1)
        checks = self._store_checks(out)
        assert checks["store encoding"].status == vd.FAIL
        assert "encoding='jpeg'" in checks["store encoding"].fix

    def test_jpeg_store_passes(self, videos, tmp_path):
        out = tmp_path / "j.lance"
        build_lance_video_dataset(videos, out, encoding="jpeg", workers=1)
        checks = self._store_checks(out)
        assert checks["store encoding"].status == vd.OK
        assert checks["store frames"].status == vd.OK
        assert "4:2:0" in checks["store frames"].detail

    def test_missing_store_fails(self, tmp_path):
        checks = self._store_checks(tmp_path / "nope.lance")
        assert checks["store"].status == vd.FAIL


class TestCLI:
    """The spt video-pipeline doctor command."""

    def _patch(self, monkeypatch, status):
        def fake(**kwargs):
            r = vd.Report()
            r.add("x", status, "detail", "fix")
            fake.kwargs = kwargs
            return r

        monkeypatch.setattr(vd, "run_doctor", fake)
        return fake

    def test_exit_code_follows_failures(self, monkeypatch):
        self._patch(monkeypatch, vd.WARN)
        assert runner.invoke(cli.app, ["video-pipeline", "doctor"]).exit_code == 0
        self._patch(monkeypatch, vd.FAIL)
        assert runner.invoke(cli.app, ["video-pipeline", "doctor"]).exit_code == 1

    def test_options_and_json(self, monkeypatch):
        fake = self._patch(monkeypatch, vd.OK)
        res = runner.invoke(
            cli.app,
            [
                "video-pipeline",
                "doctor",
                "--store",
                "s.lance",
                "--bench",
                "--workers",
                "3",
                "--json",
            ],
        )
        assert res.exit_code == 0, res.output
        assert json.loads(res.output)["ok"] is True
        assert fake.kwargs == dict(
            store="s.lance", bench=True, workers=3, batch=96, device=0
        )


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU path")
def test_no_gpu_fails_cleanly():
    r = vd.run_doctor()
    gpu = next(c for c in r.checks if c.name == "GPU")
    assert gpu.status == vd.FAIL and not r.ok


@pytest.mark.gpu
def test_gpu_doctor_and_synthetic_bench():
    r = vd.run_doctor(bench=True, batch=8)
    names = {c.name: c for c in r.checks}
    assert names["GPU"].status == vd.OK
    assert names["torchvision CUDA ops"].status == vd.OK
    assert names["nvJPEG (torchvision)"].status == vd.OK
    assert set(r.throughput) == {
        "GPU decode (GPUDecodeJPEG)",
        "GPU multi-crop (LeVJEPA recipe)",
    }
    assert r.bottleneck in r.throughput


@pytest.mark.parametrize(
    "cap,archs,ok",
    [
        ((9, 0), ["sm_80", "sm_90"], True),
        ((8, 9), ["sm_80", "sm_86", "sm_90"], True),  # L40S runs sm_86 SASS
        ((8, 9), ["sm_90", "sm_100"], False),  # newer major: no
        ((10, 3), ["sm_100"], True),
        ((12, 0), ["sm_90", "sm_100"], False),  # Blackwell consumer needs sm_120
        ((12, 0), ["sm_90", "compute_90"], True),  # PTX JIT
        ((9, 0), ["sm_90a"], False),
    ],
)
def test_arch_supported(cap, archs, ok):
    assert vd._arch_supported(cap, archs)[0] is ok
