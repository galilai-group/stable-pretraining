#!/usr/bin/env python
"""Command-line interface for Stable SSL training."""

import os
import sys
from pathlib import Path
import subprocess
import typer
from typing import List, Optional

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
    import pandas as pd

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


# ========== REGISTRY COMMANDS ==========

registry_app = typer.Typer(help="Query the local run registry.")
app.add_typer(registry_app, name="registry")


def _open_registry(db_path: Optional[str]):
    """Open the registry DB without importing the full spt package.

    Directly uses the lightweight registry._db and registry.query modules
    (only sqlite3/json/pathlib) to avoid loading torch/lightning/hydra
    on every CLI invocation.
    """
    from stable_pretraining.registry._db import RegistryDB
    from stable_pretraining.registry.query import Registry

    if db_path is None:
        import os

        cache_dir = os.environ.get("SPT_CACHE_DIR")
        if cache_dir is None:
            try:
                from stable_pretraining._config import get_config

                cache_dir = get_config().cache_dir
            except Exception:
                pass
        if cache_dir is None:
            typer.echo(
                "Error: No --db path and no cache_dir configured. "
                "Pass --db or set SPT_CACHE_DIR env var.",
                err=True,
            )
            raise typer.Exit(code=1)
        db_path = str(Path(cache_dir).resolve() / "registry.db")

    return Registry(RegistryDB(db_path))


@registry_app.command(name="ls")
def registry_ls(
    tag: Optional[str] = typer.Option(None, help="Filter by tag"),
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    sort: Optional[str] = typer.Option(
        None, "--sort", help="Sort by column or summary.<key>"
    ),
    limit: Optional[int] = typer.Option(None, "-n", help="Max rows"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to registry.db"),
):
    """List runs in the registry."""
    reg = _open_registry(db)
    runs = reg.query(
        tag=tag,
        status=status,
        sort_by=sort,
        descending=True,
        limit=limit,
    )

    if not runs:
        typer.echo("No runs found.")
        raise typer.Exit()

    # Simple table
    rows = []
    for r in runs:
        row = {
            "run_id": r.run_id,
            "status": r.status,
            "tags": ", ".join(r.tags) if r.tags else "",
        }
        # Add top summary metrics
        for k, v in list(r.summary.items())[:5]:
            if isinstance(v, float):
                row[k] = f"{v:.4f}"
            else:
                row[k] = str(v)
        rows.append(row)

    import pandas as pd

    df = pd.DataFrame(rows)
    typer.echo(df.to_string(index=False))
    reg.close()


@registry_app.command()
def show(
    run_id: str = typer.Argument(..., help="Run ID to display"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to registry.db"),
):
    """Show details for a single run."""
    reg = _open_registry(db)
    run = reg.get(run_id)
    if run is None:
        typer.echo(f"Run '{run_id}' not found.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"run_id:          {run.run_id}")
    typer.echo(f"status:          {run.status}")
    typer.echo(f"run_dir:         {run.run_dir}")
    typer.echo(f"checkpoint_path: {run.checkpoint_path}")
    typer.echo(f"tags:            {run.tags}")
    typer.echo(f"notes:           {run.notes}")

    if run.summary:
        typer.echo("\nSummary:")
        for k, v in sorted(run.summary.items()):
            typer.echo(f"  {k}: {v}")

    if run.hparams:
        typer.echo(f"\nHparams ({len(run.hparams)} keys):")
        for k, v in sorted(run.hparams.items()):
            typer.echo(f"  {k}: {v}")

    reg.close()


@registry_app.command()
def best(
    metric: str = typer.Argument(..., help="Summary metric to rank by (e.g. val_acc)"),
    tag: Optional[str] = typer.Option(None, help="Filter by tag"),
    n: int = typer.Option(5, "-n", help="Number of top runs"),
    ascending: bool = typer.Option(False, "--asc", help="Sort ascending (for loss)"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to registry.db"),
):
    """Show top N runs ranked by a summary metric."""
    reg = _open_registry(db)
    runs = reg.query(
        tag=tag,
        status="completed",
        sort_by=f"summary.{metric}",
        descending=not ascending,
        limit=n,
    )

    if not runs:
        typer.echo("No completed runs found.")
        raise typer.Exit()

    # Filter out runs that don't have the metric
    runs = [r for r in runs if metric in r.summary]
    if not runs:
        typer.echo(f"No runs have metric '{metric}' in summary.")
        raise typer.Exit()

    rows = []
    for r in runs:
        val = r.summary.get(metric, "N/A")
        if isinstance(val, float):
            val = f"{val:.6f}"
        rows.append(
            {
                "run_id": r.run_id,
                metric: val,
                "tags": ", ".join(r.tags) if r.tags else "",
                "run_dir": r.run_dir or "",
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows)
    typer.echo(df.to_string(index=False))
    reg.close()


@registry_app.command()
def export(
    output: str = typer.Argument(
        "runs.csv", help="Output file path (.csv or .parquet)"
    ),
    tag: Optional[str] = typer.Option(None, help="Filter by tag"),
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to registry.db"),
):
    """Export runs to CSV or Parquet with flattened hparams/summary columns."""
    reg = _open_registry(db)
    df = reg.to_dataframe(tag=tag, status=status)

    if df.empty:
        typer.echo("No runs to export.")
        raise typer.Exit()

    output_path = Path(output)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    typer.echo(f"Exported {len(df)} runs to {output_path}")
    reg.close()


@registry_app.command()
def ensure(
    db: Optional[str] = typer.Option(None, "--db", help="Path to registry.db"),
):
    """Ensure the registry server is running (start if needed).

    Add this to your ~/.bashrc so the server auto-starts on login::

        spt registry ensure

    Idempotent — safe to call multiple times.
    """
    from stable_pretraining.registry._db import ensure_server, _default_db_path

    db_path = db or _default_db_path()
    try:
        url = ensure_server(db_path)
        typer.echo(f"✓ Registry server running at {url}")
    except Exception as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(code=1)


@registry_app.command()
def status(
    db: Optional[str] = typer.Option(None, "--db", help="Path to registry.db"),
):
    """Check if the registry server is running."""
    from stable_pretraining.registry._db import (
        _read_discovery,
        _server_is_alive,
        _default_db_path,
    )

    db_path = db or _default_db_path()
    info = _read_discovery(db_path)
    if info and _server_is_alive(info["url"]):
        typer.echo(f"✓ Running at {info['url']} (pid {info.get('pid', '?')})")
    elif info:
        typer.echo(
            f"✗ Discovery file exists but server at {info['url']} is not responding."
        )
        typer.echo("  Run: spt registry ensure")
        raise typer.Exit(code=1)
    else:
        typer.echo(f"✗ No server found (no discovery file at {db_path}).")
        typer.echo("  Run: spt registry ensure")
        raise typer.Exit(code=1)


@registry_app.command()
def stop(
    db: Optional[str] = typer.Option(None, "--db", help="Path to registry.db"),
):
    """Stop the registry server."""
    import signal

    from stable_pretraining.registry._db import (
        _read_discovery,
        _discovery_path,
        _default_db_path,
    )

    db_path = db or _default_db_path()
    info = _read_discovery(db_path)
    if not info:
        typer.echo("No server found.")
        return

    pid = info.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            typer.echo(f"Sent SIGTERM to pid {pid}")
        except ProcessLookupError:
            typer.echo(f"Process {pid} already dead")
        except PermissionError:
            typer.echo(f"Cannot kill pid {pid} (permission denied)", err=True)

    # Clean up discovery file
    try:
        _discovery_path(db_path).unlink()
    except FileNotFoundError:
        pass
    typer.echo("✓ Server stopped")


@registry_app.command()
def sync(
    scan_dir: Optional[str] = typer.Argument(
        None, help="Directory to scan (default: cache_dir)"
    ),
    db: Optional[str] = typer.Option(None, "--db", help="Path to registry.db"),
):
    """Sync sidecar files from interrupted runs into the registry.

    When training jobs lose connection to the server (e.g. during SLURM
    preemption), they save a ``.registry_sidecar.json`` in their run
    directory.  This command finds and ingests those files.
    """
    from stable_pretraining.registry._db import RegistryDB, _default_db_path

    db_path = db or _default_db_path()

    try:
        client = RegistryDB(db_path)
    except RuntimeError as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(code=1)

    # Determine scan root
    if scan_dir is None:
        import os

        scan_dir = os.environ.get("SPT_CACHE_DIR")
        if scan_dir is None:
            try:
                from stable_pretraining._config import get_config

                scan_dir = get_config().cache_dir
            except Exception:
                pass
        if scan_dir is None:
            typer.echo(
                "Error: No scan directory. Pass a path or set SPT_CACHE_DIR.", err=True
            )
            raise typer.Exit(code=1)

    import json

    sidecars = list(Path(scan_dir).rglob(".registry_sidecar.json"))
    if not sidecars:
        typer.echo(f"No sidecar files found under {scan_dir}")
        return

    synced = 0
    for path in sidecars:
        try:
            data = json.loads(path.read_text())
            run_id = data.get("run_id")
            if not run_id:
                continue

            # Insert or update
            client.insert_run(
                run_id,
                status=data.get("status", "interrupted"),
                run_dir=data.get("run_dir"),
                config=data.get("config"),
                hparams=data.get("hparams"),
                tags=data.get("tags"),
                notes=data.get("notes"),
            )
            if data.get("summary"):
                client.update_run(run_id, summary=data["summary"])

            # Remove sidecar after successful sync
            path.unlink()
            synced += 1
            typer.echo(f"  ✓ {run_id}")
        except Exception as exc:
            typer.echo(f"  ✗ {path}: {exc}", err=True)

    typer.echo(f"Synced {synced}/{len(sidecars)} sidecar files.")


if __name__ == "__main__":
    app()
