#!/usr/bin/env python
"""Command-line interface for Stable SSL training."""

import sys
from pathlib import Path
import subprocess
import typer
from typing import List, Optional
import pandas as pd

app = typer.Typer(
    name="spt",
    help="Stable SSL Training CLI",
    add_completion=True,
)


# ========== CONFIG RUNNER COMMAND ==========


def _find_config_file(config_spec: str) -> tuple[Optional[str], Optional[str]]:
    """Find config file from path or name."""
    config_path = Path(config_spec)

    if config_path.exists():
        config_path = config_path.resolve()
        return str(config_path.parent), config_path.stem

    if not config_spec.endswith((".yaml", ".yml")):
        config_spec = f"{config_spec}.yaml"

    config_path = Path.cwd() / config_spec
    if config_path.exists():
        return str(Path.cwd()), config_path.stem

    return None, None


def _needs_multirun(overrides: List[str]) -> bool:
    """Detect if multirun mode is needed."""
    if not overrides:
        return False

    overrides_str = " ".join(overrides)

    return (
        "--multirun" in overrides
        or "-m" in overrides
        or "hydra/launcher=" in overrides_str
        or "hydra.sweep" in overrides_str
        or any("=" in o and "," in o.split("=", 1)[1] for o in overrides if "=" in o)
    )


@app.command()
def run(
    config: str = typer.Argument(..., help="Config file path or name"),
    overrides: Optional[List[str]] = typer.Argument(None, help="Hydra overrides"),
):
    """Execute experiment with the specified config.

    Examples:
      spt run config.yaml

      spt run config.yaml -m

      spt run config.yaml trainer.max_epochs=100
    """
    overrides = overrides or []

    config_path, config_name = _find_config_file(config)

    if config_path is None:
        typer.echo(f"Error: Could not find config file '{config}'", err=True)
        raise typer.Exit(code=1)

    cmd = [
        sys.executable,
        "-m",
        "stable_pretraining.run",
        "--config-path",
        config_path,
        "--config-name",
        config_name,
    ]

    if _needs_multirun(overrides):
        cmd.append("-m")
        overrides = [o for o in overrides if o not in ["-m", "--multirun"]]
        if not any("hydra/launcher=" in o for o in overrides):
            overrides.append("hydra/launcher=submitit_slurm")
        typer.echo("Running in multirun mode")

    if overrides:
        cmd.extend(overrides)

    typer.echo(f"Config: {config_name} from {config_path}")
    typer.echo("-" * 50)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(code=e.returncode)
    except KeyboardInterrupt:
        typer.echo("\nInterrupted", err=True)
        raise typer.Exit(code=130)


# ========== CSV COMPRESSION COMMAND ==========


@app.command(name="dump-csv-logs")
def dump_csv_logs(
    dir: str = typer.Argument(..., help="Input CSV file directory"),
    output_name: str = typer.Argument(..., help="Base name for compressed output"),
    agg: str = typer.Argument(
        default="all", help="Aggregation method: 'max' or 'last' or 'all'"
    ),
):
    """Compress CSV logs to the smallest possible format with aggregation."""
    from stable_pretraining.utils.read_csv_logger import (
        save_best_compressed,
        CSVLogAutoSummarizer,
    )

    # ========== Input Validation ==========
    dir_path = Path(dir)
    if not dir_path.exists():
        typer.echo(f"Error: Directory '{dir}' does not exist", err=True)
        raise typer.Exit(code=1)

    if not dir_path.is_dir():
        typer.echo(f"Error: '{dir}' is not a directory", err=True)
        raise typer.Exit(code=1)

    if agg not in ["max", "last", "all"]:
        typer.echo(f"Error: Invalid aggregation '{agg}'. Use 'max' or 'last'", err=True)
        raise typer.Exit(code=1)

    # ========== Define Aggregation Functions ==========
    def _agg_max(df: pd.DataFrame) -> pd.DataFrame:
        """Apply max to numeric columns, last value to others."""
        result = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                result[col] = df[col].max()
            else:
                # For non-numeric, take last non-null value
                result[col] = (
                    df[col].dropna().iloc[-1] if not df[col].dropna().empty else None
                )
        return pd.DataFrame([result])

    def _agg_last(df: pd.DataFrame) -> pd.DataFrame:
        """Take the last row."""
        return df.iloc[[-1]].copy()

    def _agg_all(df: pd.DataFrame) -> pd.DataFrame:
        """Take the last row."""
        return df

    if agg == "max":
        agg_func = _agg_max
    elif agg == "last":
        agg_func = _agg_last
    else:
        agg_func = _agg_all

    # ========== Process Data ==========
    try:
        typer.echo(f"Reading CSV logs from: {dir}")
        df = CSVLogAutoSummarizer().collect(dir)

        if df.empty:
            typer.echo("Warning: Collected DataFrame is empty", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Loaded DataFrame: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

        # Apply aggregation
        typer.echo(f"Applying '{agg}' aggregation...")
        df_agg = agg_func(df)
        typer.echo(
            f"Aggregated to: {df_agg.shape[0]:,} rows x {df_agg.shape[1]:,} columns"
        )

        # Save with best compression
        typer.echo("Finding best compression format...")
        best_file = save_best_compressed(df_agg, output_name)

        typer.echo(f"Success! Best compressed file: {best_file}")

    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during processing: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
