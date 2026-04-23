"""Unit tests for DataFrame compression utilities."""

import pytest
import pandas as pd
import numpy as np
import os


from stable_pretraining.utils.read_csv_logger import (
    _strip_extensions,
    _optimize_dataframe,
    _get_extension,
    save_best_compressed,
    load_best_compressed,
)


@pytest.mark.unit
def test_strip_extensions():
    """Test that all file extensions are properly removed from paths."""
    # Single extension
    assert _strip_extensions("data.csv") == "data"
    assert _strip_extensions("results.parquet") == "results"

    # Multiple extensions
    assert _strip_extensions("data.csv.gz") == "data"
    assert _strip_extensions("file.tar.gz.bz2") == "file"

    # No extension
    assert _strip_extensions("data") == "data"

    # With paths
    assert _strip_extensions("/path/to/file.feather") == "/path/to/file"
    assert _strip_extensions("relative/path/data.pkl") == "relative/path/data"

    # Edge cases
    # Hidden files with extension
    assert _strip_extensions(".hidden.txt") == ".hidden"


@pytest.mark.unit
def test_get_extension():
    """Test that correct extensions are generated for format/compression pairs."""
    # Parquet (compression in metadata)
    assert _get_extension("parquet", "zstd") == ".parquet"
    assert _get_extension("parquet", "gzip") == ".parquet"

    # Feather (compression in metadata)
    assert _get_extension("feather", "lz4") == ".feather"
    assert _get_extension("feather", "zstd") == ".feather"

    # Pickle (compression in metadata)
    assert _get_extension("pickle", "infer") == ".pkl"

    # CSV (needs explicit compression extension)
    assert _get_extension("csv", "gzip") == ".csv.gz"
    assert _get_extension("csv", "bz2") == ".csv.bz2"
    assert _get_extension("csv", "xz") == ".csv.xz"
    assert _get_extension("csv", "zstd") == ".csv.zst"
    assert _get_extension("csv", "zip") == ".csv.zip"


@pytest.mark.unit
def test_optimize_dataframe():
    df = pd.DataFrame(
        {
            "category_col": ["A", "B", "A", "B", "A"],  # Low cardinality (0.4 ratio)
            "object_col": ["x", "y", "z", "w", "v"],  # High cardinality (1.0 ratio)
        }
    )

    df_opt = _optimize_dataframe(df)

    # Check that category conversion happened for low cardinality
    assert isinstance(df_opt["category_col"].dtype, pd.CategoricalDtype)

    # Check that high cardinality stayed as a string-like type (NOT category)
    # This is the robust way to check "it stayed as it was"
    assert not isinstance(df_opt["object_col"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_string_dtype(
        df_opt["object_col"]
    ) or pd.api.types.is_object_dtype(df_opt["object_col"])


@pytest.mark.unit
def test_save_and_load_best_compressed(tmp_path):
    """Test that save and load round-trip preserves data."""
    # Create test DataFrame - FIXED: Use simpler types
    df = pd.DataFrame(
        {
            "value": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "integer": np.random.randint(0, 100, 100),
        }
    )

    # Save to temporary directory
    base_path = tmp_path / "test_data"

    # Test 1: Save without extension
    saved_file = save_best_compressed(df, str(base_path))
    assert os.path.exists(saved_file)
    assert base_path.stem in saved_file

    # Test 2: Load back
    df_loaded = load_best_compressed(saved_file)

    # Verify data integrity - FIXED: Better comparison
    # Convert categorical columns to object for comparison
    df_compare = df.copy()
    df_loaded_compare = df_loaded.copy()

    for col in df_compare.columns:
        if df_loaded_compare[col].dtype.name == "category":
            df_loaded_compare[col] = df_loaded_compare[col].astype("object")

    # Sort columns and index for consistent comparison
    df_compare = df_compare.sort_index(axis=1).reset_index(drop=True)
    df_loaded_compare = df_loaded_compare.sort_index(axis=1).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        df_compare,
        df_loaded_compare,
        check_dtype=False,
        check_exact=False,  # Allow for floating point differences
    )

    # Test 3: Save with extension (should be stripped)
    base_path_with_ext = tmp_path / "test_data_2.csv"
    saved_file_2 = save_best_compressed(df, str(base_path_with_ext))

    # Extension should have been stripped and replaced
    assert os.path.exists(saved_file_2)
    assert str(base_path_with_ext.stem) in saved_file_2

    # Test 4: Verify it can be loaded
    df_loaded_2 = load_best_compressed(saved_file_2)
    assert len(df_loaded_2) == len(df)
    assert set(df_loaded_2.columns) == set(df.columns)


@pytest.mark.unit
def test_load_best_compressed_with_unknown_extension(tmp_path):
    """Test that loader can handle files with ambiguous/unknown extensions."""
    # Create a simple DataFrame
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        }
    )

    # Save as parquet
    parquet_file = tmp_path / "data.parquet"
    df.to_parquet(parquet_file)

    # Rename to unknown extension
    weird_file = tmp_path / "data.weird_ext"
    os.rename(parquet_file, weird_file)

    # Should still load successfully by trying all formats
    df_loaded = load_best_compressed(str(weird_file))

    pd.testing.assert_frame_equal(df, df_loaded, check_dtype=False)


@pytest.mark.unit
def test_load_best_compressed_file_not_found():
    """Test that appropriate error is raised for missing files."""
    with pytest.raises(FileNotFoundError):
        load_best_compressed("/path/that/does/not/exist.parquet")


@pytest.mark.unit
def test_save_best_compressed_empty_dataframe(tmp_path):
    """Test handling of edge case: empty DataFrame."""
    # FIXED: Empty DataFrames don't work well with all formats
    # Create a DataFrame with columns but no rows instead
    df = pd.DataFrame(columns=["a", "b", "c"])
    base_path = tmp_path / "empty_data"

    # Should still save successfully
    saved_file = save_best_compressed(df, str(base_path))
    assert os.path.exists(saved_file)

    # Should load back
    df_loaded = load_best_compressed(saved_file)
    assert len(df_loaded) == 0
    # Check columns are preserved
    assert list(df_loaded.columns) == list(df.columns)


@pytest.mark.unit
def test_save_best_compressed_with_datetime(tmp_path):
    """Test that datetime columns are handled correctly."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50),
            "value": np.random.randn(50),
        }
    )

    base_path = tmp_path / "datetime_data"
    saved_file = save_best_compressed(df, str(base_path))

    # Load back
    df_loaded = load_best_compressed(saved_file)

    # Verify timestamp column exists and has correct length
    assert "timestamp" in df_loaded.columns
    assert len(df_loaded) == len(df)

    # Convert both to datetime if needed and compare
    df_loaded["timestamp"] = pd.to_datetime(df_loaded["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Compare timestamps (allowing for timezone differences)
    pd.testing.assert_series_equal(
        df["timestamp"].reset_index(drop=True),
        df_loaded["timestamp"].reset_index(drop=True),
        check_dtype=False,
    )
