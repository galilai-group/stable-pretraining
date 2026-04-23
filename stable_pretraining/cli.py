#!/usr/bin/env python
"""Command-line interface for Stable SSL training."""

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


def _resolve_cache_and_db(db: Optional[str], cache: Optional[str]) -> tuple[Path, Path]:
    """Resolve ``(cache_dir, db_path)`` for CLI commands.

    Preference order: explicit flags > ``SPT_CACHE_DIR`` env var >
    ``spt.set(cache_dir=...)`` global config.
    """
    import os

    resolved_cache: Optional[str] = cache
    if resolved_cache is None:
        resolved_cache = os.environ.get("SPT_CACHE_DIR")
    if resolved_cache is None:
        try:
            from stable_pretraining._config import get_config

            resolved_cache = get_config().cache_dir
        except Exception:
            pass
    if resolved_cache is None and db is None:
        typer.echo(
            "Error: No --cache-dir / --db and no cache_dir configured. "
            "Pass --cache-dir or set SPT_CACHE_DIR env var.",
            err=True,
        )
        raise typer.Exit(code=1)

    cache_path = (
        Path(resolved_cache).expanduser().resolve()
        if resolved_cache is not None
        else Path(db).expanduser().resolve().parent  # type: ignore[arg-type]
    )
    db_path = (
        Path(db).expanduser().resolve()
        if db is not None
        else cache_path / "registry.db"
    )
    return cache_path, db_path


def _open_registry(
    db: Optional[str] = None,
    cache: Optional[str] = None,
    *,
    scan: bool = True,
):
    """Open a read-only :class:`Registry` after an optional lazy scan."""
    from stable_pretraining.registry import open_registry

    cache_path, db_path = _resolve_cache_and_db(db, cache)
    try:
        return open_registry(db_path=db_path, cache_dir=cache_path, scan=scan)
    except FileNotFoundError:
        typer.echo(
            f"No registry cache found at {db_path}. Run `spt registry scan` first.",
            err=True,
        )
        raise typer.Exit(code=1)


@registry_app.command(name="ls")
def registry_ls(
    tag: Optional[str] = typer.Option(None, help="Filter by tag"),
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    alive: Optional[bool] = typer.Option(
        None, "--alive/--dead", help="Filter by heartbeat-based liveness"
    ),
    sort: Optional[str] = typer.Option(
        None, "--sort", help="Sort by column or summary.<key>"
    ),
    limit: Optional[int] = typer.Option(None, "-n", help="Max rows"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to registry.db"),
    cache: Optional[str] = typer.Option(None, "--cache-dir", help="Cache dir root"),
):
    """List runs in the registry."""
    reg = _open_registry(db, cache)
    runs = reg.query(
        tag=tag,
        status=status,
        alive=alive,
        sort_by=sort,
        descending=True,
        limit=limit,
    )

    if not runs:
        typer.echo("No runs found.")
        reg.close()
        raise typer.Exit()

    rows = []
    for r in runs:
        row = {
            "run_id": r.run_id,
            "status": r.status,
            "alive": "yes" if r.alive else "no",
            "tags": ", ".join(r.tags) if r.tags else "",
        }
        for k, v in list(r.summary.items())[:5]:
            row[k] = f"{v:.4f}" if isinstance(v, float) else str(v)
        rows.append(row)

    import pandas as pd

    df = pd.DataFrame(rows)
    typer.echo(df.to_string(index=False))
    reg.close()


@registry_app.command()
def show(
    run_id: str = typer.Argument(..., help="Run ID to display"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to registry.db"),
    cache: Optional[str] = typer.Option(None, "--cache-dir", help="Cache dir root"),
):
    """Show details for a single run."""
    reg = _open_registry(db, cache)
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
    cache: Optional[str] = typer.Option(None, "--cache-dir", help="Cache dir root"),
):
    """Show top N runs ranked by a summary metric."""
    reg = _open_registry(db, cache)
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
    cache: Optional[str] = typer.Option(None, "--cache-dir", help="Cache dir root"),
):
    """Export runs to CSV or Parquet with flattened hparams/summary columns."""
    reg = _open_registry(db, cache)
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
def scan(
    full: bool = typer.Option(
        False, "--full", help="Re-ingest every sidecar regardless of mtime."
    ),
    db: Optional[str] = typer.Option(None, "--db", help="Path to registry.db"),
    cache: Optional[str] = typer.Option(None, "--cache-dir", help="Cache dir root"),
):
    """Scan ``{cache_dir}/runs`` and refresh the registry cache.

    Normally incremental — only sidecars whose mtime advanced since
    the last scan are re-parsed.  Pass ``--full`` to re-ingest every
    sidecar (useful if the schema changed or the DB was rebuilt).
    """
    from stable_pretraining.registry._scanner import scan as run_scan
    from stable_pretraining.registry._store import Store

    cache_path, db_path = _resolve_cache_and_db(db, cache)

    with Store(db_path, readonly=False) as store:
        report = run_scan(cache_path, store, full=full)

    typer.echo(str(report))
    if report.total_sidecars == 0:
        typer.echo(
            f"(no sidecars found under {cache_path / 'runs'}; "
            "is this the right cache_dir?)"
        )


@registry_app.command()
def migrate(
    src_db: str = typer.Argument(..., help="Legacy registry.db to migrate from"),
    cache: Optional[str] = typer.Option(None, "--cache-dir", help="Cache dir root"),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite sidecars that already exist"
    ),
):
    """Write sidecar files from a legacy server-backed ``registry.db``.

    After migration, delete the old DB and run ``spt registry scan --full``
    to rebuild the cache from the filesystem.
    """
    import json
    import sqlite3

    from stable_pretraining.registry import _sidecar as sidecar_mod

    cache_path, _ = _resolve_cache_and_db(db=None, cache=cache)

    src = Path(src_db).expanduser().resolve()
    if not src.is_file():
        typer.echo(f"Source DB not found: {src}", err=True)
        raise typer.Exit(code=1)

    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM runs").fetchall()
    conn.close()

    written = skipped = missing_dir = 0
    for row in rows:
        run_dir = row["run_dir"]
        if not run_dir:
            missing_dir += 1
            continue
        run_dir_p = Path(run_dir)
        if not run_dir_p.is_dir():
            missing_dir += 1
            continue
        dest = sidecar_mod.sidecar_path(run_dir_p)
        if dest.exists() and not overwrite:
            skipped += 1
            continue

        def _decode(v, default):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    return default
            return v if v is not None else default

        data = sidecar_mod.make_sidecar(
            run_id=row["run_id"],
            run_dir=str(run_dir_p),
            status=row["status"] or "unknown",
            created_at=row["created_at"] or None,
            hparams=_decode(row["hparams"], {}),
            summary=_decode(row["summary"], {}),
            tags=_decode(row["tags"], []),
            notes=row["notes"] or "",
            checkpoint_path=row["checkpoint_path"],
        )
        sidecar_mod.write_sidecar(run_dir_p, data)
        written += 1

    typer.echo(
        f"migrate: {written} sidecars written, {skipped} already existed, "
        f"{missing_dir} rows had no/missing run_dir"
    )
    typer.echo(
        "Next: delete the old DB and run "
        f"`spt registry scan --full --cache-dir {cache_path}`"
    )


if __name__ == "__main__":
    app()
