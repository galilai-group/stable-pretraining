from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import concurrent.futures

# ========== Internal Functions ==========


def _optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Optimizing DataFrame dtypes...")
    df_opt = df.copy()

    if len(df_opt) == 0:
        return df_opt

    # 1. Downcast Numerics
    for col in df_opt.select_dtypes(include=["float"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")
    for col in df_opt.select_dtypes(include=["integer"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")

    # 2. Handle Objects, Strings, and Paths
    for col in df_opt.columns:
        # Use a more efficient check for Path objects
        sample_val = df_opt[col].iloc[0] if not df_opt[col].empty else None

        # Check for Path objects (PosixPath/WindowsPath)
        if (
            isinstance(sample_val, Path)
            or df_opt[col].apply(lambda x: isinstance(x, Path)).any()
        ):
            logger.warning(f"Column '{col}' contains Path objects, converting to str.")
            df_opt[col] = df_opt[col].astype(str)

        # Handle Categorical conversion for low-cardinality strings/objects
        if pd.api.types.is_object_dtype(df_opt[col]) or pd.api.types.is_string_dtype(
            df_opt[col]
        ):
            # Skip if it's already a category
            if pd.api.types.is_categorical_dtype(df_opt[col]):
                continue

            num_unique = df_opt[col].nunique()
            num_total = len(df_opt)
            if num_total > 0 and (num_unique / num_total) < 0.5:
                df_opt[col] = df_opt[col].astype("category")

    logger.info("Optimization complete.")
    return df_opt


def _save_variant(df, base_path, fmt, comp):
    """Save DataFrame in a specific format with compression and return file size.

    Creates a file with naming pattern: {base_path}__{format}__{compression}.{ext}

    Args:
        df: DataFrame to save.
        base_path: Base path/name for the output file (without extension).
        fmt: File format. Supported values: 'parquet', 'csv', 'feather', 'pickle'.
        comp: Compression algorithm. Valid options depend on format:
            - parquet: 'brotli', 'gzip', 'snappy', 'zstd', 'none'
            - csv: 'gzip', 'bz2', 'xz', 'zstd', 'zip'
            - feather: 'zstd', 'lz4', 'uncompressed'
            - pickle: 'infer', 'gzip', 'bz2', 'xz', 'zstd'

    Returns:
        Tuple of (filename, size_in_bytes) if successful, (None, None) if failed.

    Notes:
        - CSV files are saved without index
        - File extensions are automatically appended based on format
        - Failures are logged but don't raise exceptions (returns None, None)
        - Unknown formats trigger a warning and return None, None

    Examples:
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> filename, size = _save_variant(df, "data/output", "parquet", "zstd")
        >>> filename
        'data/output__parquet__zstd.parquet'
        >>> size > 0
        True

        >>> # CSV with gzip compression
        >>> filename, size = _save_variant(df, "data/output", "csv", "gzip")
        >>> filename
        'data/output__csv__gzip.csv.gz'
    """
    filename = f"{base_path}__{fmt}__{comp}"
    try:
        if fmt == "parquet":
            filename += ".parquet"
            df.to_parquet(filename, compression=comp)
        elif fmt == "csv":
            ext = {
                "gzip": "gz",
                "bz2": "bz2",
                "xz": "xz",
                "zstd": "zst",
                "zip": "zip",
            }.get(comp, comp)
            filename += f".csv.{ext}"
            df.to_csv(filename, compression=comp, index=False)
        elif fmt == "feather":
            filename += ".feather"
            df.to_feather(filename, compression=comp)
        elif fmt == "pickle":
            filename += ".pkl"
            df.to_pickle(filename, compression=comp)
        else:
            logger.warning(f"Unknown format: {fmt}")
            return None, None
        size = os.path.getsize(filename)
        logger.debug(f"Saved {filename} ({size / 1024:.2f} KB)")
        return filename, size
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return None, None


def _get_trials():
    # Add more combinations as needed
    return [
        # Parquet
        ("parquet", "brotli"),
        ("parquet", "zstd"),
        ("parquet", "gzip"),
        ("parquet", "snappy"),
        # Feather
        ("feather", "zstd"),
        ("feather", "lz4"),
        ("feather", "uncompressed"),
        # CSV
        ("csv", "gzip"),
        ("csv", "bz2"),
        ("csv", "xz"),
        ("csv", "zstd"),
        ("csv", "zip"),
        # Pickle
        ("pickle", "infer"),
        # Add more if you want (e.g., external compression with subprocess)
    ]


def _parse_filename(filename):
    """Extract format and compression from filename following the naming convention.

    Parses filenames created by _save_variant with pattern:
    {base_path}__{format}__{compression}.{extension}

    Args:
        filename: Filename or path to parse. Can be absolute path or basename.

    Returns:
        Tuple of (format, compression) extracted from the filename.

    Raises:
        ValueError: If filename doesn't contain at least 3 parts separated by '__'.

    Examples:
        >>> _parse_filename("data__parquet__zstd.parquet")
        ('parquet', 'zstd')

        >>> _parse_filename("/path/to/output__csv__gzip.csv.gz")
        ('csv', 'gzip')

        >>> _parse_filename("output__feather__lz4.feather")
        ('feather', 'lz4')

        >>> _parse_filename("invalid_filename.csv")
        Traceback (most recent call last):
        ValueError: Filename invalid_filename.csv does not match expected pattern.

    Notes:
        - Only the basename is used; directory paths are stripped
        - The compression is extracted before any file extensions
        - This is the inverse operation of the naming in _save_variant
    """
    basename = os.path.basename(filename)
    parts = basename.split("__")
    if len(parts) < 3:
        raise ValueError(f"Filename {filename} does not match expected pattern.")
    fmt = parts[1]
    comp = parts[2].split(".")[0]
    return fmt, comp


# ========== User-Facing Functions ==========
def _get_extension(fmt: str, comp: str) -> str:
    """Get the appropriate file extension for format/compression combination."""
    if fmt == "parquet":
        return ".parquet"
    elif fmt == "feather":
        return ".feather"
    elif fmt == "pickle":
        # Pickle compression is in the file itself
        return ".pkl"
    elif fmt == "csv":
        # CSV needs compression-specific extensions
        comp_ext = {
            "gzip": ".gz",
            "bz2": ".bz2",
            "xz": ".xz",
            "zstd": ".zst",
            "zip": ".zip",
        }.get(comp, "")
        return f".csv{comp_ext}"
    return ""


def _save_variant(
    df: pd.DataFrame, base_path: str, fmt: str, comp: str
) -> tuple[Optional[str], Optional[int]]:
    """Save DataFrame in a specific format with compression and return file size.

    Creates a file with clean naming: {base_path}.{ext}
    Uses a temporary suffix during trials to avoid conflicts.

    Args:
        df: DataFrame to save.
        base_path: Base path/name for the output file (without extension).
        fmt: File format. Supported values: 'parquet', 'csv', 'feather', 'pickle'.
        comp: Compression algorithm. Valid options depend on format.

    Returns:
        Tuple of (filename, size_in_bytes) if successful, (None, None) if failed.
    """
    ext = _get_extension(fmt, comp)
    # Use temporary suffix to distinguish trial files
    filename = f"{base_path}_trial_{fmt}_{comp}{ext}"

    try:
        if fmt == "parquet":
            df.to_parquet(filename, compression=comp)
        elif fmt == "csv":
            df.to_csv(filename, compression=comp, index=False)
        elif fmt == "feather":
            df.to_feather(filename, compression=comp)
        elif fmt == "pickle":
            df.to_pickle(filename, compression=comp)
        else:
            logger.warning(f"Unknown format: {fmt}")
            return None, None

        size = os.path.getsize(filename)
        logger.debug(f"Saved {filename} ({size / 1024:.2f} KB)")
        return filename, size
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        return None, None


def _strip_extensions(base_path: str) -> str:
    """Remove all file extensions from base path to ensure clean naming.

    Args:
        base_path: Path that may contain extensions.

    Returns:
        Path with all extensions removed.

    Examples:
        >>> _strip_extensions("data.parquet")
        'data'
        >>> _strip_extensions("results.csv.gz")
        'results'
        >>> _strip_extensions("/path/to/file.feather")
        '/path/to/file'
        >>> _strip_extensions("data")
        'data'
    """
    path = Path(base_path)
    parent = path.parent
    name = path.name

    # Keep removing suffixes until there are none left
    while True:
        name_path = Path(name)
        if not name_path.suffix:
            break
        name = name_path.stem

    # Reconstruct with original directory
    if str(parent) != ".":
        return str(parent / name)
    return name


def save_best_compressed(df: pd.DataFrame, base_path: str = "data") -> str:
    """Optimize DataFrame and save using the most space-efficient format/compression combination.

    This function:
    1. Optimizes DataFrame dtypes to reduce memory usage
    2. Tries multiple format/compression combinations in parallel
    3. Keeps only the smallest resulting file with a clean filename
    4. Automatically cleans up all trial files

    The following format/compression combinations are tested:
    - Parquet: brotli, zstd, gzip, snappy
    - Feather: zstd, lz4, uncompressed
    - CSV: gzip, bz2, xz, zstd, zip
    - Pickle: infer

    Args:
        df: DataFrame to save. Will be optimized before saving.
        base_path: Base path/name for output file. Any extensions will be
            automatically removed and replaced with the optimal format.
            Default: 'data'

    Returns:
        Filename of the best-compressed file (smallest size) with clean extension.

    Raises:
        RuntimeError: If all save attempts fail (no files were created).

    Notes:
        - Uses parallel processing (ThreadPoolExecutor) for I/O-bound operations
        - All trial files except the smallest are automatically deleted
        - Original DataFrame is not modified (optimization works on a copy)
        - Final file uses standard extensions (e.g., .parquet, .csv.gz, .feather)
        - Any extensions in base_path are automatically stripped
        - If multiple formats produce the same size, the first one sorted is kept

    Performance:
        - Time: Depends on number of trials and DataFrame size
        - Memory: Peak usage ~2x DataFrame size (original + optimized copy)
        - Disk: Temporarily creates all trial files before cleanup

    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "timestamp": pd.date_range("2024-01-01", periods=1000),
        ...         "value": np.random.randn(1000),
        ...         "category": np.random.choice(["A", "B", "C"], 1000),
        ...     }
        ... )

        >>> # Extension automatically added
        >>> best_file = save_best_compressed(df, "results/experiment_01")
        >>> print(best_file)
        'results/experiment_01.parquet'

        >>> # Extensions in input are stripped
        >>> best_file = save_best_compressed(df, "data.csv")
        >>> print(best_file)
        'data.parquet'  # .csv was stripped, optimal format chosen

        >>> # Works with multiple extensions
        >>> best_file = save_best_compressed(df, "output.csv.gz")
        >>> print(best_file)
        'output.feather'  # All extensions stripped

    See Also:
        _optimize_dataframe: For details on DataFrame optimization
        _get_trials: For the complete list of format/compression combinations
        load_best_compressed: Smart loader that auto-detects format
    """
    # Strip any extensions from base_path
    base_path_clean = _strip_extensions(base_path)

    if base_path_clean != base_path:
        logger.info(f"Stripped extensions: '{base_path}' → '{base_path_clean}'")

    logger.info("Starting best-compression save process...")
    df_opt = _optimize_dataframe(df)
    options = _get_trials()
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Store futures with their corresponding format/compression
        future_to_options = {
            executor.submit(_save_variant, df_opt, base_path_clean, fmt, comp): (
                fmt,
                comp,
            )
            for fmt, comp in options
        }

        for future in concurrent.futures.as_completed(future_to_options):
            fmt, comp = future_to_options[
                future
            ]  # Get the format/compression for this future
            filename, size = future.result()
            if filename and size:
                results.append((size, filename, fmt, comp))

    if not results:
        logger.error("No files were saved!")
        raise RuntimeError("No files were saved!")

    results.sort()
    best_size, best_trial_file, best_fmt, best_comp = results[0]

    # Rename best file to clean name
    best_ext = _get_extension(best_fmt, best_comp)
    final_filename = f"{base_path_clean}{best_ext}"

    try:
        os.rename(best_trial_file, final_filename)
        logger.success(
            f"Best compression: {best_fmt}/{best_comp} "
            f"({best_size / 1024:.2f} KB) → {final_filename}"
        )
    except Exception as e:
        logger.error(f"Failed to rename {best_trial_file} to {final_filename}: {e}")
        final_filename = best_trial_file

    # Cleanup other trial files
    for _, fname, _, _ in results[1:]:
        try:
            os.remove(fname)
            logger.debug(f"Removed {fname}")
        except Exception as e:
            logger.warning(f"Could not remove {fname}: {e}")

    return final_filename


def load_best_compressed(filename: str) -> pd.DataFrame:
    """Load a DataFrame from a compressed file with automatic format detection.

    Automatically detects the file format from the extension and tries multiple
    read methods if needed. This is a smart loader that works with any compressed
    DataFrame file, not just those created by save_best_compressed.

    Args:
        filename: Path to the compressed file. Standard extensions are recognized:
            - .parquet → Parquet format
            - .feather → Feather format
            - .pkl, .pickle → Pickle format
            - .csv, .csv.gz, .csv.bz2, etc. → CSV format

    Returns:
        Loaded DataFrame with original dtypes preserved.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        ValueError: If no read method successfully loads the file.

    Supported Formats:
        - Parquet (compression auto-detected)
        - Feather (compression auto-detected)
        - CSV (compression auto-detected from extension)
        - Pickle (compression auto-detected)

    Notes:
        - Extension-based detection is tried first for speed
        - If extension is ambiguous/missing, tries all formats sequentially
        - Compression is handled automatically by pandas
        - Works with any properly formatted DataFrame file, not just from this module

    Examples:
        >>> # Load with clear extension
        >>> df = load_best_compressed("data.parquet")

        >>> # Load compressed CSV
        >>> df = load_best_compressed("results.csv.gz")

        >>> # Load with full path
        >>> df = load_best_compressed("/path/to/experiment.feather")

        >>> # Even works with ambiguous extensions (tries all methods)
        >>> df = load_best_compressed("mystery_file.dat")

    See Also:
        save_best_compressed: Saves DataFrame with optimal compression
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    logger.info(f"Loading {filename}...")

    # Extension-based detection
    lower_filename = filename.lower()

    # Define read strategies in order of likelihood
    strategies = []

    if lower_filename.endswith(".parquet"):
        strategies.append(("parquet", pd.read_parquet))
    elif lower_filename.endswith(".feather"):
        strategies.append(("feather", pd.read_feather))
    elif lower_filename.endswith((".pkl", ".pickle")):
        strategies.append(("pickle", pd.read_pickle))
    elif ".csv" in lower_filename:
        strategies.append(("csv", pd.read_csv))

    # Fallback: try all methods if extension doesn't match
    if not strategies:
        logger.warning("Unknown extension, trying all formats...")
        strategies = [
            ("parquet", pd.read_parquet),
            ("feather", pd.read_feather),
            ("pickle", pd.read_pickle),
            ("csv", pd.read_csv),
        ]
    else:
        # Add other formats as fallbacks
        all_methods = [
            ("parquet", pd.read_parquet),
            ("feather", pd.read_feather),
            ("pickle", pd.read_pickle),
            ("csv", pd.read_csv),
        ]
        # Add methods we haven't tried yet
        for method in all_methods:
            if method not in strategies:
                strategies.append(method)

    # Try each strategy
    last_error = None
    for fmt, read_func in strategies:
        try:
            logger.debug(f"Trying to load as {fmt}...")
            df = read_func(filename)
            logger.success(f"Successfully loaded as {fmt} format")
            return df
        except Exception as e:
            logger.debug(f"Failed to load as {fmt}: {e}")
            last_error = e
            continue

    # If we get here, nothing worked
    logger.error(f"Could not load {filename} with any known format")
    raise ValueError(
        f"Failed to load {filename}. Tried: {', '.join(s[0] for s in strategies)}. "
        f"Last error: {last_error}"
    )


class CSVLogAutoSummarizer:
    """Automatically discovers and summarizes PyTorch Lightning CSVLogger metrics from Hydra multirun sweeps.

    Handles arbitrary directory layouts, sparse metrics, multiple versions (preemption/resume), and aggregates
    config/hparams metadata into a single DataFrame.
    Features:
    - Recursively finds metrics CSVs using common patterns.
    - Infers run root by searching for .hydra/config.yaml (falls back to metrics parent).
    - Handles multiple metrics files per run with configurable grouping strategies:
      latest_mtime, latest_epoch, latest_step, or merge.
    - Robust CSV parsing (delimiter sniffing, repeated-header cleanup, type coercion).
    - Sparse-metrics aware: last values are last non-NaN; best values ignore NaNs.
    - Loads and flattens Hydra config, overrides (list or dict), and hparams metadata.

    Args:
    base_dir: Root directory to search for metrics files.
    monitor_keys: Metrics to summarize (if None, auto-infer).
    include_globs: Only include files matching these globs (relative to base_dir).
    exclude_globs: Exclude files matching these globs (e.g., '**/checkpoints/**').
    forward_fill_last: If True, forward-fill the frame (after sorting) before computing last.* summaries.

    """

    METRICS_PATTERNS = ["**/metrics.csv", "**/csv/metrics.csv", "**/*metrics*.csv"]

    def __init__(
        self,
        agg: Optional[callable] = None,
        monitor_keys: Optional[List[str]] = None,
        include_globs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
        max_workers: Optional[int] = 10,
    ):
        self.agg = agg
        self.monitor_keys = monitor_keys
        self.include_globs = include_globs or []
        self.exclude_globs = exclude_globs or []
        self.max_workers = max_workers

    def collect(self, base_dir) -> pd.DataFrame:
        """Discover, summarize, and aggregate all runs into a DataFrame.

        Args:
        base_dir: Root directory to search for metrics files.

        Returns:
            pd.DataFrame: One row per selected metrics source (per file or per run root),
            with flattened columns such as:
            - last.val_accuracy
            - best.val_loss, best.val_loss.step, best.val_loss.epoch
            - config.optimizer.lr, override.0 (or override as joined string), hparams.seed
            Also includes 'metrics_path' and 'run_root'.
        """
        if type(base_dir) not in [list, tuple]:
            base_dir = [base_dir]

        dfs = []
        for b in base_dir:
            dfs.append(self._single_collect(b))
        return pd.concat(dfs, ignore_index=True)

    def _single_collect(self, base_dir) -> pd.DataFrame:
        """Discover, summarize, and aggregate all runs into a DataFrame.

        Args:
        base_dir: Root directory to search for metrics files.

        Returns:
            pd.DataFrame: One row per selected metrics source (per file or per run root),
            with flattened columns such as:
            - last.val_accuracy
            - best.val_loss, best.val_loss.step, best.val_loss.epoch
            - config.optimizer.lr, override.0 (or override as joined string), hparams.seed
            Also includes 'metrics_path' and 'run_root'.
        """
        self.base_dir = Path(base_dir)
        metrics_files = self._find_metrics_files()
        run_root_to_files: Dict[Path, List[Path]] = {}
        for f in tqdm(metrics_files, desc="compiling root paths"):
            root = self._find_run_root(f)
            run_root_to_files.setdefault(root, []).append(f)
        items = list(run_root_to_files.items())
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            summaries = list(
                tqdm(
                    executor.map(self._merge_metrics_files, items),
                    desc="Loading",
                    total=len(items),
                )
            )
        summaries = [s for s in summaries if s is not None]
        if not summaries:
            return pd.DataFrame()
        return pd.concat(summaries, ignore_index=True)

    # ----------------------------
    # Discovery and selection
    # ----------------------------
    def _find_metrics_files(self) -> List[Path]:
        """Recursively find all metrics CSV files matching known patterns, applying include/exclude globs."""
        files: List[Path] = []
        for pat in self.METRICS_PATTERNS:
            files.extend(self.base_dir.rglob(pat))
        # Unique files
        files = list({f.resolve() for f in files if f.is_file()})
        rel_files = [f.relative_to(self.base_dir) for f in files]
        if self.include_globs:
            files = [
                f
                for f, rel in zip(files, rel_files)
                if any(rel.match(g) for g in self.include_globs)
            ]
        if self.exclude_globs:
            files = [
                f
                for f, rel in zip(files, rel_files)
                if not any(rel.match(g) for g in self.exclude_globs)
            ]
        logger.info(f"Discovered {len(files)} metrics files.")
        return files

    def _find_run_root(self, metrics_file: Path) -> Path:
        """Walk up to nearest ancestor with .hydra/config.yaml; else use immediate parent."""
        for parent in [metrics_file.parent] + list(metrics_file.parents):
            hydra_dir = parent / ".hydra"
            if (hydra_dir / "config.yaml").exists():
                return parent
        return metrics_file.parent

    def _merge_metrics_files(self, args) -> Optional[pd.DataFrame]:
        """Merge multiple metrics CSVs for a run root, deduplicate by step/epoch.

        Keep the last occurrence for any duplicated step/epoch pair.
        """
        root, files = args
        dfs = [self._read_data(f) for f in files]
        dfs = [df for df in dfs if df is not None and not df.empty]
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        df["root"] = root
        if self.agg is not None:
            df = self.agg(df)
            if isinstance(df, pd.Series):
                df = df.to_frame().T
            elif not isinstance(df, pd.DataFrame):
                raise RuntimeError("Can only be series or dataframe")
        return df

    # ----------------------------
    # CSV reading and summarization
    # ----------------------------
    def _read_data(self, path: Path) -> Optional[pd.DataFrame]:
        """Robustly read a metrics CSV, handling delimiter, repeated headers, and sparse rows.

        Returns None if file cannot be read or is empty.
        """
        df = pd.read_csv(path)
        # Drop unnamed columns, strip column names
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()
        # Try to convert numeric columns where possible
        df = df.apply(pd.to_numeric)
        df.ffill(inplace=True)
        # Diagnostics if all values are NaN (excluding axis/time cols)
        metric_cols = [c for c in df.columns if c not in ("step", "epoch", "time")]
        if metric_cols and df[metric_cols].isna().all().all():
            logger.warning(f"All metric values are NaN in {path}")
            logger.debug(f"Dtypes:\n{df.dtypes}\nHead:\n{df.head()}")
        hparams = self._find_hparams(path.parent / "hparams.yaml")
        for k, v in hparams.items():
            # Convert lists/dicts to string representation
            if isinstance(v, (list, dict)):
                df[f"config/{k}"] = str(v)
            else:
                df[f"config/{k}"] = v
        # hparams = self._find_hparams(path.parent / "hparams.yaml")
        # for k, v in list(hparams.items()):
        #     hparams[f"config/{k}"] = v
        #     del hparams[k]
        # df[list(hparams.keys())] = list(hparams.values())
        return df

    # ----------------------------
    # Metadata loading and flattening
    # ----------------------------
    def _load_yaml(self, path: Path) -> Any:
        """Load a YAML file. Returns the parsed object, which may be dict, list, or scalar.

        Returns {} if file is missing or unreadable.
        """
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                obj = yaml.safe_load(f)
            return {} if obj is None else obj
        except Exception as e:
            logger.warning(f"Failed to load YAML {path}: {e}")
            return {}

    def _find_hparams(self, start_dir: Path) -> Any:
        """Search upward from start_dir for hparams.yaml (first found). Return the parsed object (dict/list/scalar) or {}."""
        for parent in [start_dir] + list(start_dir.parents):
            hp = parent / "hparams.yaml"
            if hp.exists():
                return self._load_yaml(hp)
        return {}
