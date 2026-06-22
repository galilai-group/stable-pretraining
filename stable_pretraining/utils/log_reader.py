"""Unified log reader for local and wandb logs."""

import re
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

try:
    import jsonlines
except ModuleNotFoundError:
    logger.warning(
        "jsonlines module is not installed, local log reading will not work."
    )

try:
    import omegaconf
except ModuleNotFoundError:
    logger.warning("omegaconf module is not installed, config loading will not work.")

try:
    import wandb as wandbapi
    from tqdm.contrib.logging import logging_redirect_tqdm
except ModuleNotFoundError:
    logger.warning(
        "Wandb module is not installed, make sure to not use wandb for logging "
        "or an error will be thrown."
    )
    wandbapi = None
    logging_redirect_tqdm = None


# ============================================================================
# Common Utilities
# ============================================================================


def alphanum_key(key: str) -> List[Union[int, str]]:
    """Convert a string to a list of mixed numbers and strings for natural sorting."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split("([0-9]+)", key)]


def natural_sort(values: List[str]) -> List[str]:
    """Sort a list of strings naturally (handling numbers properly)."""
    return sorted(values, key=alphanum_key)


def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested config dictionaries into a single level.

    Args:
        config: Nested configuration dictionary

    Returns:
        Flattened configuration dictionary
    """
    flat_config = config.copy()
    for name in ["log", "data", "model", "optim", "hardware"]:
        if name in flat_config:
            for k, v in flat_config[name].items():
                flat_config[f"{name}.{k}"] = v
            del flat_config[name]
    return flat_config


# ============================================================================
# Abstract Base Class for Log Readers
# ============================================================================


class LogReader(ABC):
    """Abstract base class for log readers."""

    @abstractmethod
    def read(self, *args, **kwargs):
        """Read logs from source."""
        pass


# ============================================================================
# Local Log Reader
# ============================================================================


class LocalLogReader(LogReader):
    """Reader for local jsonl log files."""

    def __init__(self, num_workers: int = 8):
        """Initialize local log reader.

        Args:
            num_workers: Number of parallel workers for reading logs
        """
        self.num_workers = num_workers

    def read(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load values from a single run directory.

        Args:
            path: Path to the run directory

        Returns:
            List of log entries
        """
        _path = Path(path)
        if not _path.is_dir():
            raise ValueError(f"The provided path ({path}) is not a directory!")

        values = []
        logs_files = list(_path.glob("logs_rank_*.jsonl"))
        logger.info(f"Reading .jsonl files from {_path}")
        logger.info(f"\t=> {len(logs_files)} ranks detected")

        for log_file in logs_files:
            rank = int(log_file.stem.split("rank_")[1])
            for obj in jsonlines.open(log_file).iter(type=dict, skip_invalid=True):
                obj["rank"] = rank
                values.append(obj)

        logger.info(f"\t=> total length of logs: {len(values)}")
        return values

    def read_project(
        self, folder: Union[str, Path]
    ) -> Tuple[pd.DataFrame, List[List[Dict[str, Any]]]]:
        """Load configs and values from all runs in a folder.

        Args:
            folder: Path to the project folder

        Returns:
            Tuple of (configs DataFrame, list of values for each run)
        """
        if not Path(folder).is_dir():
            raise ValueError(f"The provided folder ({folder}) is not a directory!")

        runs = list(Path(folder).rglob("*/hparams.yaml"))
        configs = []
        values = []

        if logging_redirect_tqdm:
            with logging_redirect_tqdm():
                args = [run.parent for run in runs]
                with Pool(self.num_workers) as p:
                    results = list(tqdm(p.imap(self.read, args), total=len(runs)))
                for c, v in results:
                    configs.append(flatten_config(c))
                    values.append(v)
        else:
            args = [run.parent for run in runs]
            with Pool(self.num_workers) as p:
                results = list(tqdm(p.imap(self.read, args), total=len(runs)))
            for v in results:
                values.append(v)

        config_df = pd.DataFrame(configs) if configs else pd.DataFrame()
        return config_df, values

    def read_config(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load config from a single run directory.

        Args:
            path: Path to the run directory

        Returns:
            Configuration dictionary
        """
        _path = Path(path)
        if not _path.is_dir():
            raise ValueError(f"The provided path ({path}) is not a directory!")

        config_path = _path / ".hydra" / "config.yaml"
        if config_path.exists():
            return omegaconf.OmegaConf.load(config_path)

        hparams_path = _path / "hparams.yaml"
        if hparams_path.exists():
            return omegaconf.OmegaConf.load(hparams_path)

        raise FileNotFoundError(f"No config file found in {path}")


# ============================================================================
# WandB Log Reader
# ============================================================================


class WandbLogReader(LogReader):
    """Reader for Weights & Biases logs."""

    def __init__(self, num_workers: int = 10):
        """Initialize wandb log reader.

        Args:
            num_workers: Number of parallel workers for downloading runs
        """
        if wandbapi is None:
            raise ImportError(
                "wandb is not installed. Please install it to use WandbLogReader."
            )
        self.num_workers = num_workers
        self.api = wandbapi.Api()

    def read(
        self,
        entity: str,
        project: str,
        run_id: str,
        min_step: int = 0,
        max_step: int = -1,
        keys: Optional[List[str]] = None,
        _tqdm_disable: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Download data for a single wandb run.

        Args:
            entity: Wandb entity name
            project: Wandb project name
            run_id: Run ID
            min_step: Minimum step to download
            max_step: Maximum step to download (-1 for all)
            keys: Specific keys to download
            _tqdm_disable: Whether to disable tqdm progress bar

        Returns:
            Tuple of (data DataFrame, config dict)
        """
        run = self.api.run(f"{entity}/{project}/{run_id}")

        if max_step == -1:
            max_step = run.lastHistoryStep + 1
        if min_step < 0:
            min_step = max_step + min_step

        if keys is not None and "_step" not in keys:
            keys.append("_step")

        data = []
        for row in tqdm(
            run.scan_history(keys=keys, min_step=min_step, max_step=max_step),
            total=max_step,
            desc=f"Downloading run: {run.name}",
            disable=_tqdm_disable,
        ):
            data.append(row)

        df = pd.DataFrame(data)
        if "_step" in df.columns:
            df.set_index("_step", inplace=True)

        return df, run.config

    def read_project(
        self,
        entity: str,
        project: str,
        filters: Optional[Dict[str, Any]] = None,
        order: str = "+created_at",
        per_page: int = 50,
        include_sweeps: bool = True,
        min_step: int = 0,
        max_step: int = -1,
        keys: Optional[List[str]] = None,
        return_summary: bool = False,
    ) -> Union[Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]], pd.DataFrame]:
        """Download configs and data from a wandb project.

        Args:
            entity: Wandb entity name
            project: Wandb project name
            filters: Optional filters for runs
            order: Sort order for runs
            per_page: Number of runs per page
            include_sweeps: Whether to include sweep runs
            min_step: Minimum step to download
            max_step: Maximum step to download
            keys: Specific keys to download
            return_summary: If True, return only summary DataFrame

        Returns:
            If return_summary is False: Tuple of (dfs dict, configs dict)
            If return_summary is True: Summary DataFrame
        """
        runs = self.api.runs(
            f"{entity}/{project}",
            filters=filters,
            order=order,
            per_page=per_page,
            include_sweeps=include_sweeps,
        )
        logger.info(f"Found {len(runs)} runs for project {project}")

        if return_summary:
            data = []
            for run in tqdm(runs, desc=f"Loading project summary: {project}"):
                run_data = dict()
                run_data.update(run.summary._json_dict)
                run_data.update(run.config)
                run_data.update({"tags": run.tags})
                run_data.update({"name": run.name})
                run_data.update({"created_at": run.created_at})
                run_data.update({"id": run.id})
                data.append(run_data)
            return pd.DataFrame.from_records(data)

        def _run_packed(args):
            return self.read(*args, _tqdm_disable=True)

        with Pool(self.num_workers) as p:
            results = list(
                tqdm(
                    p.imap(
                        _run_packed,
                        [
                            (entity, project, r.id, min_step, max_step, keys)
                            for r in runs
                        ],
                    ),
                    total=len(runs),
                    desc=f"Downloading project: {project}",
                )
            )

        dfs = {}
        configs = {}
        for r, (df, config) in zip(runs, results):
            dfs[f"{entity}/{project}/{r.id}"] = df
            configs[f"{entity}/{project}/{r.id}"] = config

        return dfs, configs


# ============================================================================
# Table Formatter
# ============================================================================


class TableFormatter:
    """Format experiment results as tables for analysis."""

    @staticmethod
    def create_table(
        dfs: Dict[str, pd.DataFrame],
        configs: Dict[str, Any],
        value: str,
        row: str,
        column: str,
        agg: Callable,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Format a pandas DataFrame as a table given the user args.

        Args:
            dfs: Dictionary of DataFrames (one per run)
            configs: Dictionary of configs (one per run)
            value: Name of the column in dfs to use as values
            row: Name of the column in configs to use as row
            column: Name of the column in configs to use as column
            agg: Aggregator function if many values are present
            filters: Optional filters to apply to the data

        Returns:
            Formatted table as DataFrame
        """
        logger.info(f"Creating table from {len(configs)} runs.")
        filters = filters or {}

        df = pd.DataFrame(configs).T

        for id in df.index.values:
            assert id in dfs, f"Run {id} not found in dfs"

        for k, v in filters.items():
            if not isinstance(v, (tuple, list)):
                v = [v]
            s = df[k].isin(v)
            df = df.loc[s]
            logger.info(f"After filtering {k}, {len(df)} runs are left.")

        rows = natural_sort(df[row].astype(str).unique())
        logger.info(f"Found rows: {rows}")
        columns = natural_sort(df[column].astype(str).unique())
        logger.info(f"Found columns: {columns}")

        output = pd.DataFrame(columns=columns, index=rows)

        for r in rows:
            for c in columns:
                cell_runs = (df[row].astype(str) == r) & (df[column].astype(str) == c)
                n = np.count_nonzero(cell_runs)
                samples = []
                logger.info(f"Number of runs for cell ({r}, {c}): {n}")

                for id in df[cell_runs].index.values:
                    if value not in dfs[id].columns:
                        logger.info(f"Run {id} missing {value}, skipping....")
                        continue
                    samples.append(dfs[id][value].values.reshape(-1))

                if len(samples) == 0:
                    output.loc[r, c] = np.nan
                else:
                    output.loc[r, c] = agg(np.concatenate(samples))

        return output

    @staticmethod
    def tabulate_runs(
        configs: pd.DataFrame,
        runs: List[Any],
        value: str,
        ignore: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create a pivot table from configs and runs for a specific value.

        Args:
            configs: DataFrame of run configurations
            runs: List of run data
            value: Value to extract from runs
            ignore: Columns to ignore

        Returns:
            Pivot table as DataFrame
        """
        ignore = ignore or ["hardware.port"]
        res = configs.copy()

        for col in configs.columns:
            if len(configs[col].unique()) == 1 or col in ignore:
                res = res.drop(col, axis=1)

        variables = res.columns
        print("Remaining columns:", variables)
        res["_index"] = res.index

        rows = input("Which to use as rows? ").split(",")
        table = pd.pivot_table(
            res,
            index=rows,
            columns=[v for v in variables if v not in rows],
            values="_index",
        )

        def fn(i):
            try:
                i = int(i)
                return runs[i][value][-1]
            except (ValueError, IndexError, KeyError):
                return np.nan

        return table.map(fn)


# ============================================================================
# Convenience Functions
# ============================================================================


def read_local_logs(
    path: Union[str, Path], num_workers: int = 8
) -> List[Dict[str, Any]]:
    """Convenience function to read local logs.

    Args:
        path: Path to the run directory
        num_workers: Number of parallel workers

    Returns:
        List of log entries
    """
    reader = LocalLogReader(num_workers=num_workers)
    return reader.read(path)


def read_local_project(
    folder: Union[str, Path], num_workers: int = 8
) -> Tuple[pd.DataFrame, List[List[Dict[str, Any]]]]:
    """Convenience function to read a local project.

    Args:
        folder: Path to the project folder
        num_workers: Number of parallel workers

    Returns:
        Tuple of (configs DataFrame, list of values)
    """
    reader = LocalLogReader(num_workers=num_workers)
    return reader.read_project(folder)


def read_wandb_run(
    entity: str,
    project: str,
    run_id: str,
    min_step: int = 0,
    max_step: int = -1,
    keys: Optional[List[str]] = None,
    num_workers: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function to read a wandb run.

    Args:
        entity: Wandb entity name
        project: Wandb project name
        run_id: Run ID
        min_step: Minimum step to download
        max_step: Maximum step to download
        keys: Specific keys to download
        num_workers: Number of parallel workers

    Returns:
        Tuple of (data DataFrame, config dict)
    """
    reader = WandbLogReader(num_workers=num_workers)
    return reader.read(entity, project, run_id, min_step, max_step, keys)


def read_wandb_project(
    entity: str,
    project: str,
    filters: Optional[Dict[str, Any]] = None,
    order: str = "+created_at",
    per_page: int = 50,
    include_sweeps: bool = True,
    min_step: int = 0,
    max_step: int = -1,
    keys: Optional[List[str]] = None,
    num_workers: int = 10,
    return_summary: bool = False,
) -> Union[Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]], pd.DataFrame]:
    """Convenience function to read a wandb project.

    Args:
        entity: Wandb entity name
        project: Wandb project name
        filters: Optional filters for runs
        order: Sort order for runs
        per_page: Number of runs per page
        include_sweeps: Whether to include sweep runs
        min_step: Minimum step to download
        max_step: Maximum step to download
        keys: Specific keys to download
        num_workers: Number of parallel workers
        return_summary: If True, return only summary DataFrame

    Returns:
        If return_summary is False: Tuple of (dfs dict, configs dict)
        If return_summary is True: Summary DataFrame
    """
    reader = WandbLogReader(num_workers=num_workers)
    return reader.read_project(
        entity,
        project,
        filters,
        order,
        per_page,
        include_sweeps,
        min_step,
        max_step,
        keys,
        return_summary,
    )


def create_results_table(
    dfs: Dict[str, pd.DataFrame],
    configs: Dict[str, Any],
    value: str,
    row: str,
    column: str,
    agg: Callable = np.mean,
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Convenience function to create a results table.

    Args:
        dfs: Dictionary of DataFrames (one per run)
        configs: Dictionary of configs (one per run)
        value: Name of the column in dfs to use as values
        row: Name of the column in configs to use as row
        column: Name of the column in configs to use as column
        agg: Aggregator function (default: mean)
        filters: Optional filters to apply

    Returns:
        Formatted table as DataFrame
    """
    formatter = TableFormatter()
    return formatter.create_table(dfs, configs, value, row, column, agg, filters)
