"""CLI arguments, exit statuses, and exported values must survive dispatch."""

import subprocess
import sys
from unittest.mock import Mock

import pandas as pd
import pytest
from typer.testing import CliRunner

from stable_pretraining import cli
from stable_pretraining._config import get_config
from stable_pretraining.loggers import csv_log_reader

pytestmark = pytest.mark.unit
runner = CliRunner()


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ([], []),
        (["trainer.max_epochs=3"], ["trainer.max_epochs=3"]),
        (["lr=0.1,0.2"], ["-m", "lr=0.1,0.2", "hydra/launcher=submitit_slurm"]),
        (
            ["hydra/launcher=basic", "seed=1,2"],
            ["-m", "hydra/launcher=basic", "seed=1,2"],
        ),
    ],
)
def test_training_command_preserves_arguments_without_shell_interpretation(
    tmp_path, monkeypatch, overrides, expected
):
    config = tmp_path / "config with spaces.yaml"
    config.write_text("trainer: {}\n")
    launch = Mock()
    monkeypatch.setattr(cli.subprocess, "run", launch)
    result = runner.invoke(cli.app, ["run", str(config), *overrides])
    assert result.exit_code == 0, result.output
    launch.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "stable_pretraining.run",
            "--config-path",
            str(tmp_path),
            "--config-name",
            config.stem,
            *expected,
        ],
        check=True,
    )


@pytest.mark.parametrize("flag", ["-m", "--multirun"])
def test_explicit_multirun_flag_is_forwarded_once(tmp_path, monkeypatch, flag):
    (tmp_path / "config.yaml").write_text("{}")
    monkeypatch.chdir(tmp_path)
    launch = Mock()
    monkeypatch.setattr(cli.subprocess, "run", launch)
    cli.run("config", [flag])
    command = launch.call_args.args[0]
    assert command.count("-m") == 2  # Python's module flag and Hydra's multirun flag.
    assert command[-1] == "hydra/launcher=submitit_slurm"


@pytest.mark.parametrize(
    "error,code",
    [(subprocess.CalledProcessError(7, "train"), 7), (KeyboardInterrupt(), 130)],
)
def test_training_command_propagates_failure_status(tmp_path, monkeypatch, error, code):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    monkeypatch.setattr(cli.subprocess, "run", Mock(side_effect=error))
    result = runner.invoke(cli.app, ["run", str(config)])
    assert result.exit_code == code


def test_missing_training_config_does_not_launch_process(tmp_path, monkeypatch):
    launch = Mock()
    monkeypatch.setattr(cli.subprocess, "run", launch)
    result = runner.invoke(cli.app, ["run", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 1
    assert "Could not find config" in result.output
    launch.assert_not_called()


@pytest.mark.parametrize("aggregation", ["all", "last", "max"])
def test_csv_export_aggregates_values_without_mutating_collected_data(
    tmp_path, monkeypatch, aggregation
):
    frame = pd.DataFrame(
        {"loss": [3.0, 1.0], "label": ["first", "last"], "empty": [None, None]}
    )
    original = frame.copy(deep=True)
    monkeypatch.setattr(
        csv_log_reader.CSVLogAutoSummarizer, "collect", lambda *a: frame
    )

    def save(data, output):
        path = output + ".csv.gz"
        data.to_csv(path, index=False, compression="gzip")
        return path

    monkeypatch.setattr(csv_log_reader, "save_best_compressed", save)
    result = runner.invoke(
        cli.app, ["dump-csv-logs", str(tmp_path), str(tmp_path / "export"), aggregation]
    )
    assert result.exit_code == 0, result.output
    exported = pd.read_csv(tmp_path / "export.csv.gz")
    assert (
        exported.loss.tolist()
        == {"all": [3.0, 1.0], "last": [1.0], "max": [3.0]}[aggregation]
    )
    assert exported.label.tolist() == (
        ["first", "last"] if aggregation == "all" else ["last"]
    )
    assert exported["empty"].isna().all()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "problem", ["missing", "file", "aggregation", "empty", "read_error", "write_error"]
)
def test_csv_export_reports_failures(tmp_path, monkeypatch, problem):
    directory = tmp_path
    agg = "all"
    collect = Mock(return_value=pd.DataFrame({"loss": [1.0]}))
    save = Mock(return_value="export.csv")
    if problem == "missing":
        directory = tmp_path / "missing"
    elif problem == "file":
        directory = tmp_path / "file"
        directory.touch()
    elif problem == "aggregation":
        agg = "invalid"
    elif problem == "empty":
        collect.return_value = pd.DataFrame()
    elif problem == "read_error":
        collect.side_effect = FileNotFoundError("file disappeared")
    else:
        save.side_effect = OSError("disk full")
    monkeypatch.setattr(csv_log_reader.CSVLogAutoSummarizer, "collect", collect)
    monkeypatch.setattr(csv_log_reader, "save_best_compressed", save)
    result = runner.invoke(
        cli.app, ["dump-csv-logs", str(directory), str(tmp_path / "export"), agg]
    )
    assert result.exit_code == 1
    if problem != "write_error":
        save.assert_not_called()


@pytest.mark.parametrize("source", ["explicit", "flag", "environment", "config"])
def test_web_directory_precedence_and_server_options(tmp_path, monkeypatch, source):
    import stable_pretraining.web as web

    root = tmp_path / source
    (root / "runs").mkdir(parents=True)
    monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
    serve = Mock()
    monkeypatch.setattr(web, "serve", serve)
    args = ["web", "--host", "127.0.0.2", "--port", "8123", "--poll", "0.25"]
    if source == "explicit":
        args.append(str(root))
        monkeypatch.setenv("SPT_CACHE_DIR", str(tmp_path / "wrong"))
    elif source == "flag":
        args.extend(["--cache-dir", str(root)])
        monkeypatch.setenv("SPT_CACHE_DIR", str(tmp_path / "wrong"))
    elif source == "environment":
        monkeypatch.setenv("SPT_CACHE_DIR", str(root))
    else:
        monkeypatch.setattr(get_config(), "_cache_dir", str(root))
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    serve.assert_called_once_with(
        root if source == "explicit" else root / "runs",
        host="127.0.0.2",
        port=8123,
        poll_interval=0.25,
    )


@pytest.mark.parametrize("problem", ["unconfigured", "missing", "bind_error"])
def test_web_command_reports_configuration_and_bind_errors(
    tmp_path, monkeypatch, problem
):
    import stable_pretraining.web as web

    serve = Mock(side_effect=OSError("port occupied"))
    monkeypatch.setattr(web, "serve", serve)
    args = ["web"]
    if problem == "unconfigured":
        monkeypatch.setattr(cli, "_resolve_cache_dir_only", lambda _: None)
    else:
        args.append(str(tmp_path / "missing" if problem == "missing" else tmp_path))
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    if problem != "bind_error":
        serve.assert_not_called()
