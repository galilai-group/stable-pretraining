import pandas as pd
import yaml
from unittest.mock import patch
from stable_pretraining.loggers import CSVLogAutoSummarizer
import pytest


@pytest.mark.unit
class TestInit:
    """Test initialization of CSVLogAutoSummarizer."""

    def test_default_initialization(self):
        summarizer = CSVLogAutoSummarizer()
        assert summarizer.agg is None
        assert summarizer.monitor_keys is None
        assert summarizer.include_globs == []
        assert summarizer.exclude_globs == []
        assert summarizer.max_workers == 10

    def test_custom_initialization(self):
        def agg_func(x):
            return x

        monitor_keys = ["val_loss", "val_accuracy"]
        include_globs = ["run*/**"]
        exclude_globs = ["**/checkpoints/**"]
        max_workers = 4

        summarizer = CSVLogAutoSummarizer(
            agg=agg_func,
            monitor_keys=monitor_keys,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            max_workers=max_workers,
        )

        assert summarizer.agg == agg_func
        assert summarizer.monitor_keys == monitor_keys
        assert summarizer.include_globs == include_globs
        assert summarizer.exclude_globs == exclude_globs
        assert summarizer.max_workers == max_workers


@pytest.mark.unit
class TestFindMetricsFiles:
    """Test metrics file discovery."""

    def test_find_metrics_files_basic(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()
        summarizer.base_dir = tmp_path

        # Create test files
        (tmp_path / "run1").mkdir()
        (tmp_path / "run1" / "metrics.csv").touch()
        (tmp_path / "run2").mkdir()
        (tmp_path / "run2" / "metrics.csv").touch()

        files = summarizer._find_metrics_files()

        assert len(files) == 2
        assert all(f.name == "metrics.csv" for f in files)

    def test_find_metrics_files_with_include_glob(self, tmp_path):
        summarizer = CSVLogAutoSummarizer(include_globs=["run1/**"])
        summarizer.base_dir = tmp_path

        (tmp_path / "run1").mkdir()
        (tmp_path / "run1" / "metrics.csv").touch()
        (tmp_path / "run2").mkdir()
        (tmp_path / "run2" / "metrics.csv").touch()

        files = summarizer._find_metrics_files()

        assert len(files) == 1
        assert "run1" in str(files[0])

    def test_find_metrics_files_with_exclude_glob(self, tmp_path):
        summarizer = CSVLogAutoSummarizer(exclude_globs=["run2/**"])
        summarizer.base_dir = tmp_path

        (tmp_path / "run1").mkdir()
        (tmp_path / "run1" / "metrics.csv").touch()
        (tmp_path / "run2").mkdir()
        (tmp_path / "run2" / "metrics.csv").touch()

        files = summarizer._find_metrics_files()

        assert len(files) == 1
        assert "run1" in str(files[0])

    def test_find_metrics_files_empty_directory(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()
        summarizer.base_dir = tmp_path

        files = summarizer._find_metrics_files()

        assert len(files) == 0

    def test_find_metrics_files_nested_csv(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()
        summarizer.base_dir = tmp_path

        nested = tmp_path / "run" / "version_0" / "csv"
        nested.mkdir(parents=True)
        (nested / "metrics.csv").touch()

        files = summarizer._find_metrics_files()

        assert len(files) == 1
        assert files[0].name == "metrics.csv"

    def test_find_metrics_files_removes_duplicates(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()
        summarizer.base_dir = tmp_path

        (tmp_path / "metrics.csv").touch()

        files = summarizer._find_metrics_files()

        # Should deduplicate based on resolved path
        assert len(files) == len(set(f.resolve() for f in files))


@pytest.mark.unit
class TestFindRunRoot:
    """Test finding the run root directory."""

    def test_find_run_root_with_hydra_in_parent(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        hydra_dir = tmp_path / "run1" / ".hydra"
        hydra_dir.mkdir(parents=True)
        (hydra_dir / "config.yaml").touch()

        metrics_file = tmp_path / "run1" / "metrics.csv"
        metrics_file.touch()

        root = summarizer._find_run_root(metrics_file)

        assert root == tmp_path / "run1"

    def test_find_run_root_without_hydra(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        metrics_file = tmp_path / "no_hydra" / "metrics.csv"
        metrics_file.parent.mkdir(parents=True)
        metrics_file.touch()

        root = summarizer._find_run_root(metrics_file)

        assert root == metrics_file.parent

    def test_find_run_root_nested_with_hydra_at_top(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        hydra_dir = tmp_path / "experiment" / ".hydra"
        hydra_dir.mkdir(parents=True)
        (hydra_dir / "config.yaml").touch()

        metrics_file = tmp_path / "experiment" / "version_0" / "metrics.csv"
        metrics_file.parent.mkdir(parents=True)
        metrics_file.touch()

        root = summarizer._find_run_root(metrics_file)

        assert root == tmp_path / "experiment"

    def test_find_run_root_multiple_hydra_dirs(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        # Create nested .hydra directories
        outer_hydra = tmp_path / "outer" / ".hydra"
        outer_hydra.mkdir(parents=True)
        (outer_hydra / "config.yaml").touch()

        inner_hydra = tmp_path / "outer" / "inner" / ".hydra"
        inner_hydra.mkdir(parents=True)
        (inner_hydra / "config.yaml").touch()

        metrics_file = tmp_path / "outer" / "inner" / "metrics.csv"
        metrics_file.touch()

        root = summarizer._find_run_root(metrics_file)

        # Should find the nearest one
        assert root == tmp_path / "outer" / "inner"


@pytest.mark.unit
class TestLoadYaml:
    """Test YAML loading."""

    def test_load_yaml_valid_dict(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        yaml_file = tmp_path / "config.yaml"
        data = {"key1": "value1", "key2": 42}
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        result = summarizer._load_yaml(yaml_file)

        assert result == data

    def test_load_yaml_valid_list(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        yaml_file = tmp_path / "overrides.yaml"
        data = ["override1", "override2"]
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        result = summarizer._load_yaml(yaml_file)

        assert result == data

    def test_load_yaml_scalar_value(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        yaml_file = tmp_path / "value.yaml"
        with open(yaml_file, "w") as f:
            f.write("42")

        result = summarizer._load_yaml(yaml_file)

        assert result == 42

    def test_load_yaml_missing_file(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        yaml_file = tmp_path / "nonexistent.yaml"
        result = summarizer._load_yaml(yaml_file)

        assert result == {}

    def test_load_yaml_empty_file(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        yaml_file = tmp_path / "empty.yaml"
        yaml_file.touch()

        result = summarizer._load_yaml(yaml_file)

        assert result == {}

    def test_load_yaml_invalid_yaml(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        yaml_file = tmp_path / "invalid.yaml"
        with open(yaml_file, "w") as f:
            f.write("invalid: yaml: content:")

        result = summarizer._load_yaml(yaml_file)

        assert result == {}

    def test_load_yaml_permission_error(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        yaml_file = tmp_path / "protected.yaml"
        yaml_file.touch()
        yaml_file.chmod(0o000)

        result = summarizer._load_yaml(yaml_file)

        assert result == {}

        # Cleanup
        yaml_file.chmod(0o644)


@pytest.mark.unit
class TestFindHparams:
    """Test finding hparams.yaml."""

    def test_find_hparams_in_same_dir(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        hparams_data = {"seed": 42, "lr": 0.001}
        with open(tmp_path / "hparams.yaml", "w") as f:
            yaml.dump(hparams_data, f)

        result = summarizer._find_hparams(tmp_path)

        assert result == hparams_data

    def test_find_hparams_in_parent_dir(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        hparams_data = {"seed": 123}
        with open(tmp_path / "hparams.yaml", "w") as f:
            yaml.dump(hparams_data, f)

        subdir = tmp_path / "subdir" / "subsubdir"
        subdir.mkdir(parents=True)

        result = summarizer._find_hparams(subdir)

        assert result == hparams_data

    def test_find_hparams_not_found(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        result = summarizer._find_hparams(tmp_path)

        assert result == {}

    def test_find_hparams_uses_first_found(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        # Create hparams in parent
        parent_hparams = {"seed": 1}
        with open(tmp_path / "hparams.yaml", "w") as f:
            yaml.dump(parent_hparams, f)

        # Create hparams in subdir
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        child_hparams = {"seed": 2}
        with open(subdir / "hparams.yaml", "w") as f:
            yaml.dump(child_hparams, f)

        result = summarizer._find_hparams(subdir)

        # Should find the one in subdir first
        assert result == child_hparams


@pytest.mark.unit
class TestReadData:
    """Test reading and processing CSV data."""

    def test_read_data_basic(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        csv_file = tmp_path / "metrics.csv"
        df = pd.DataFrame({"step": [0, 1, 2], "train_loss": [1.5, 1.2, 0.9]})
        df.to_csv(csv_file, index=False)

        result = summarizer._read_data(csv_file)

        assert result is not None
        assert "step" in result.columns
        assert "train_loss" in result.columns
        assert len(result) == 3

    def test_read_data_with_hparams(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        csv_file = tmp_path / "metrics.csv"
        df = pd.DataFrame({"step": [0, 1], "loss": [1.5, 1.2]})
        df.to_csv(csv_file, index=False)

        hparams = {"seed": 42, "lr": 0.001}
        with open(tmp_path / "hparams.yaml", "w") as f:
            yaml.dump(hparams, f)

        result = summarizer._read_data(csv_file)

        assert "config/seed" in result.columns
        assert "config/lr" in result.columns
        assert result["config/seed"].iloc[0] == 42
        assert result["config/lr"].iloc[0] == 0.001

    def test_read_data_removes_unnamed_columns(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        csv_file = tmp_path / "metrics.csv"
        with open(csv_file, "w") as f:
            f.write("step,loss,Unnamed: 2,Unnamed: 3\n")
            f.write("0,1.5,,\n")
            f.write("1,1.2,,\n")

        result = summarizer._read_data(csv_file)

        assert result is not None
        unnamed_cols = [col for col in result.columns if "Unnamed" in col]
        assert len(unnamed_cols) == 0

    def test_read_data_strips_column_names(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        csv_file = tmp_path / "metrics.csv"
        with open(csv_file, "w") as f:
            f.write(" step , loss \n")
            f.write("0,1.5\n")

        result = summarizer._read_data(csv_file)

        assert "step" in result.columns
        assert "loss" in result.columns
        assert " step " not in result.columns

    def test_read_data_converts_numeric(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        csv_file = tmp_path / "metrics.csv"
        df = pd.DataFrame({"step": ["0", "1", "2"], "loss": ["1.5", "1.2", "0.9"]})
        df.to_csv(csv_file, index=False)

        result = summarizer._read_data(csv_file)

        assert pd.api.types.is_numeric_dtype(result["step"])
        assert pd.api.types.is_numeric_dtype(result["loss"])

    def test_read_data_forward_fills_nan(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        csv_file = tmp_path / "metrics.csv"
        df = pd.DataFrame({"step": [0, 1, 2], "loss": [1.5, None, 0.9]})
        df.to_csv(csv_file, index=False)

        result = summarizer._read_data(csv_file)

        # After ffill, the NaN at index 1 should be filled with 1.5
        assert result["loss"].iloc[1] == 1.5

    def test_read_data_all_nan_metrics_warning(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        csv_file = tmp_path / "metrics.csv"
        df = pd.DataFrame({"step": [0, 1, 2], "loss": [None, None, None]})
        df.to_csv(csv_file, index=False)

        with patch("loguru.logger.warning") as mock_warning:
            summarizer._read_data(csv_file)
            mock_warning.assert_called()

    def test_read_data_hparams_prefixed_with_config(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        csv_file = tmp_path / "metrics.csv"
        df = pd.DataFrame({"step": [0]})
        df.to_csv(csv_file, index=False)

        hparams = {"model": "resnet", "optimizer": "adam"}
        with open(tmp_path / "hparams.yaml", "w") as f:
            yaml.dump(hparams, f)

        result = summarizer._read_data(csv_file)

        assert "config/model" in result.columns
        assert "config/optimizer" in result.columns
        assert "model" not in result.columns
        assert "optimizer" not in result.columns


@pytest.mark.unit
class TestMergeMetricsFiles:
    """Test merging multiple metrics files."""

    def test_merge_metrics_files_single_file(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        run_root = tmp_path / "run"
        run_root.mkdir()

        metrics_file = run_root / "metrics.csv"
        df = pd.DataFrame({"step": [0, 1, 2], "loss": [1.5, 1.2, 0.9]})
        df.to_csv(metrics_file, index=False)

        result = summarizer._merge_metrics_files((run_root, [metrics_file]))

        assert result is not None
        assert len(result) == 3
        assert "root" in result.columns
        assert all(result["root"] == run_root)

    def test_merge_metrics_files_multiple_files(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        run_root = tmp_path / "run"
        run_root.mkdir()

        metrics1 = run_root / "metrics_1.csv"
        df1 = pd.DataFrame({"step": [0, 1], "loss": [1.5, 1.2]})
        df1.to_csv(metrics1, index=False)

        metrics2 = run_root / "metrics_2.csv"
        df2 = pd.DataFrame({"step": [2, 3], "loss": [0.9, 0.7]})
        df2.to_csv(metrics2, index=False)

        result = summarizer._merge_metrics_files((run_root, [metrics1, metrics2]))

        assert result is not None
        assert len(result) == 4
        assert "root" in result.columns

    def test_merge_metrics_files_with_agg_function(self, tmp_path):
        def agg_func(df):
            return df.groupby("root").agg({"loss": "mean"}).reset_index()

        summarizer = CSVLogAutoSummarizer(agg=agg_func)

        run_root = tmp_path / "run"
        run_root.mkdir()

        metrics_file = run_root / "metrics.csv"
        df = pd.DataFrame({"step": [0, 1, 2], "loss": [1.5, 1.2, 0.9]})
        df.to_csv(metrics_file, index=False)

        result = summarizer._merge_metrics_files((run_root, [metrics_file]))

        assert result is not None
        assert "loss" in result.columns
        assert len(result) == 1  # Aggregated to single row

    def test_merge_metrics_files_empty_file_list(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        run_root = tmp_path / "run"
        run_root.mkdir()

        result = summarizer._merge_metrics_files((run_root, []))

        assert result is None

    def test_merge_metrics_files_filters_none_and_empty(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        run_root = tmp_path / "run"
        run_root.mkdir()

        # Create one valid and one empty file
        valid_file = run_root / "valid.csv"
        df = pd.DataFrame({"step": [0], "loss": [1.5]})
        df.to_csv(valid_file, index=False)

        empty_file = run_root / "empty.csv"
        pd.DataFrame().to_csv(empty_file, index=False)

        with patch.object(summarizer, "_read_data") as mock_read:
            mock_read.side_effect = [df, pd.DataFrame(), None]
            result = summarizer._merge_metrics_files(
                (run_root, [valid_file, empty_file])
            )

            assert result is not None
            assert len(result) == 1


@pytest.mark.unit
class TestCollect:
    """Test the main collect method."""

    def test_collect_single_directory_string(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        (tmp_path / "run").mkdir()
        metrics = pd.DataFrame({"step": [0], "loss": [1.0]})
        metrics.to_csv(tmp_path / "run" / "metrics.csv", index=False)

        with patch.object(summarizer, "_single_collect") as mock_collect:
            mock_collect.return_value = pd.DataFrame({"data": [1]})
            result = summarizer.collect(tmp_path)

            mock_collect.assert_called_once_with(tmp_path)
            assert isinstance(result, pd.DataFrame)

    def test_collect_multiple_directories_list(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"

        with patch.object(summarizer, "_single_collect") as mock_collect:
            mock_collect.side_effect = [
                pd.DataFrame({"data": [1]}),
                pd.DataFrame({"data": [2]}),
            ]
            result = summarizer.collect([dir1, dir2])

            assert mock_collect.call_count == 2
            assert isinstance(result, pd.DataFrame)

    def test_collect_tuple_of_directories(self, tmp_path):
        summarizer = CSVLogAutoSummarizer()

        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"

        with patch.object(summarizer, "_single_collect") as mock_collect:
            mock_collect.side_effect = [
                pd.DataFrame({"data": [1]}),
                pd.DataFrame({"data": [2]}),
            ]
            summarizer.collect((dir1, dir2))

            assert mock_collect.call_count == 2
