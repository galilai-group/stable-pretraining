# your_module.py

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import threading
import time

import lightning as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger

import tempfile
import shutil
import json


class EnvironmentDumpCallback(Callback):
    """Dumps complete environment configuration to enable exact reproduction.

    DDP-safe: only runs on rank 0.
    Uses loguru for comprehensive logging of all operations.

    Args:
    filename: Name of the file to save environment info
    async_dump: If True, runs the dump in a background thread (non-blocking)
    """

    def __init__(self, filename: str = "environment.json", async_dump: bool = True):
        super().__init__()
        self.filename = filename
        self.async_dump = async_dump
        self._dump_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

        logger.info(
            f"EnvironmentDumpCallback initialized (filename={filename}, async={async_dump})"
        )

    @rank_zero_only
    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Called when training starts - runs dump in background."""
        if stage != "fit":
            logger.info(f"Skipping environment dump on {stage=}")
            return

        logger.info("=" * 70)
        logger.info("🔍 Starting environment configuration dump...")
        logger.info("=" * 70)

        self._start_time = time.time()

        # 🔥 CRITICAL: Get log_dir in main thread BEFORE starting background thread
        # Use trainer.log_dir for the run-specific versioned directory
        # Falls back to default_root_dir if no logger is configured
        log_dir = Path(trainer.default_root_dir)
        logger.info(f"EnvironmentDumpCallback  {log_dir=}")

        if self.async_dump:
            logger.info(
                "⚡ Running environment dump in background thread (non-blocking)"
            )
            logger.info(f"📁 Log directory: {log_dir}")
            self._dump_thread = threading.Thread(
                target=self._dump_environment,
                args=(log_dir,),  # Pass log_dir, not trainer!
                daemon=False,
                name="EnvironmentDump",
            )
            self._dump_thread.start()
            logger.success("✓ Background thread started successfully")
        else:
            logger.info("🔄 Running environment dump synchronously (blocking)")
            self._dump_environment(log_dir)

    @rank_zero_only
    def teardown(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Ensure the dump thread completes if still running."""
        if self._dump_thread is not None and self._dump_thread.is_alive():
            logger.info("⏳ Waiting for environment dump thread to complete...")
            self._dump_thread.join(timeout=30)

            if self._dump_thread.is_alive():
                logger.warning(
                    "⚠️  Environment dump thread did not complete within 30 seconds"
                )
            else:
                logger.success("✓ Environment dump thread completed successfully")

    def _get_versioned_path(self, base_path: Path) -> Path:
        """Get next available versioned path if file exists.

        Args:
            base_path: Original path (e.g., /logs/environment.json)

        Returns:
            Path with version number if needed (e.g., /logs/environment_v2.json)

        Examples:
            environment.json → environment.json (if doesn't exist)
            environment.json → environment_v1.json (if exists)
            environment.json → environment_v2.json (if v1 exists)
        """
        if not base_path.exists():
            return base_path

        # File exists, find next version
        stem = base_path.stem  # "environment"
        suffix = base_path.suffix  # ".json"
        parent = base_path.parent

        version = 1
        while True:
            versioned_path = parent / f"{stem}_v{version}{suffix}"
            if not versioned_path.exists():
                logger.info(
                    f"📋 File exists, using version {version}: {versioned_path.name}"
                )
                return versioned_path
            version += 1

            # Safety check (avoid infinite loop)
            if version > 1000:
                logger.warning(
                    f"⚠️  Too many versions ({version}), using timestamp instead"
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                return parent / f"{stem}_{timestamp}{suffix}"

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON/YAML serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    def _dump_environment(self, log_dir: Path) -> None:
        """Collect and dump all environment information."""
        try:
            logger.info("📊 Collecting environment information...")

            # Collect all information
            logger.debug("  → Collecting Python info...")
            python_info = self._get_python_info()
            logger.success(
                f"  ✓ Python {python_info['version_info']['major']}.{python_info['version_info']['minor']}.{python_info['version_info']['micro']}"
            )

            logger.debug("  → Collecting system info...")
            system_info = self._get_system_info()
            logger.success(
                f"  ✓ System: {system_info['system']} {system_info['release']} ({system_info['machine']})"
            )
            logger.info(f"     Hostname: {system_info['hostname']}")

            logger.debug("  → Collecting package info (this may take a few seconds)...")
            packages_start = time.time()
            packages_info = self._get_packages_info()
            packages_time = time.time() - packages_start
            logger.success(
                f"  ✓ Packages: {packages_info['total_packages']} packages collected in {packages_time:.2f}s"
            )

            logger.debug("  → Collecting CUDA driver info...")
            cuda_info = self._get_cuda_info()
            if cuda_info:
                logger.success(f"  ✓ NVIDIA Driver: {cuda_info['driver_version']}")
            else:
                logger.debug("  ℹ️  nvidia-smi not available or no GPU detected")

            logger.debug("  → Collecting git repository info...")
            git_info = self._get_git_info()
            if git_info:
                logger.success(
                    f"  ✓ Git: {git_info['branch']} @ {git_info['commit_hash'][:8]}"
                )
                if git_info.get("remote_url"):
                    logger.info(f"     Remote: {git_info['remote_url']}")
                if git_info["has_uncommitted_changes"]:
                    logger.warning("     ⚠️  Working directory has uncommitted changes!")
                    logger.debug(f"     Changes:\n{git_info['uncommitted_changes']}")
            else:
                logger.debug("  ℹ️  Not in a git repository or git not available")

            logger.debug("  → Collecting SLURM job info...")
            slurm_info = self._get_slurm_info()
            if slurm_info:
                logger.success(
                    f"  ✓ SLURM Job: {slurm_info.get('SLURM_JOB_ID', 'N/A')}"
                )
                logger.info(f"     Name: {slurm_info.get('SLURM_JOB_NAME', 'N/A')}")
                logger.info(
                    f"     Partition: {slurm_info.get('SLURM_JOB_PARTITION', 'N/A')}"
                )
                logger.info(
                    f"     Nodes: {slurm_info.get('SLURM_JOB_NUM_NODES', 'N/A')}"
                )
                logger.info(f"     Tasks: {slurm_info.get('SLURM_NTASKS', 'N/A')}")
            else:
                logger.debug("  ℹ️  Not running under SLURM")

            logger.debug("  → Collecting environment variables...")
            env_vars = self._get_env_variables()
            logger.success(
                f"  ✓ Environment: {len(env_vars)} relevant variables captured"
            )

            # Build complete environment info
            env_info = {
                "timestamp": datetime.now().isoformat(),
                "python": python_info,
                "system": system_info,
                "packages": packages_info,
                "cuda": cuda_info,
                "git": git_info,
                "slurm": slurm_info,
                "environment_variables": env_vars,
            }

            logger.info("🔄 Serializing environment data...")
            env_info = self._make_serializable(env_info)

            # Determine save location
            save_dir = Path(log_dir)
            save_path = self._get_versioned_path(save_dir / self.filename)
            req_path = self._get_versioned_path(save_dir / "requirements_frozen.txt")

            logger.info(f"📁 Save directory: {save_dir}")

            # Create directory if needed
            if not save_dir.exists():
                logger.debug(f"Creating directory: {save_dir}")
                save_dir.mkdir(parents=True, exist_ok=True)

            # Use JSON (faster than YAML)
            json_path = save_path.with_suffix(".json")

            # Write to temp first
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as tmp_f:
                logger.info("💾 Writing environment to temp file...")
                json.dump(env_info, tmp_f, indent=2)
                tmp_path = tmp_f.name

            # Atomic move
            logger.info(f"💾 Moving to: {json_path.name}")
            shutil.move(tmp_path, json_path)
            logger.success(
                f"✓ Environment file saved ({json_path.stat().st_size / 1024:.1f} KB)"
            )

            # Same for requirements
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            ) as tmp_f:
                tmp_f.write(packages_info["pip_freeze"])
                tmp_req_path = tmp_f.name

            shutil.move(tmp_req_path, req_path)
            logger.success(
                f"✓ Requirements file saved ({req_path.stat().st_size / 1024:.1f} KB)"
            )

        except Exception as e:
            logger.error(f"❌ Error during environment dump: {e}")
            logger.exception("Full traceback:")

    def _get_python_info(self) -> Dict[str, Any]:
        """Get Python version and executable path."""
        return {
            "version": sys.version,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
            },
            "executable": sys.executable,
            "prefix": sys.prefix,
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system/platform information."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        }

    def _get_packages_info(self) -> Dict[str, Any]:
        """Get installed packages information."""
        try:
            logger.debug("     Running 'pip freeze'...")
            pip_freeze = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
            logger.debug("     Parsing package list...")
        except subprocess.TimeoutExpired:
            logger.warning("     ⚠️  pip freeze timed out after 30 seconds")
            pip_freeze = "Error: pip freeze timed out"
        except subprocess.CalledProcessError as e:
            logger.warning(f"     ⚠️  pip freeze failed: {e}")
            pip_freeze = f"Error getting pip freeze: {str(e)}"

        # Parse into dict for key packages
        key_packages = {}
        for line in pip_freeze.split("\n"):
            if line.strip() and not line.startswith("#"):
                if "==" in line:
                    pkg, ver = line.split("==", 1)
                    key_packages[pkg.strip()] = ver.strip()

        return {
            "pip_freeze": pip_freeze,
            "key_packages": key_packages,
            "total_packages": len(key_packages),
        }

    def _get_cuda_info(self) -> Optional[Dict[str, Any]]:
        """Get CUDA information from nvidia-smi if available."""
        try:
            logger.debug("     Running nvidia-smi...")
            nvidia_smi = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )

            driver_version = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=10,
                )
                .strip()
                .split("\n")[0]
            )

            return {
                "nvidia_smi_output": nvidia_smi.strip(),
                "driver_version": driver_version,
            }
        except FileNotFoundError:
            logger.debug("     nvidia-smi not found in PATH")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("     ⚠️  nvidia-smi timed out")
            return None
        except subprocess.CalledProcessError as e:
            logger.warning(f"     ⚠️  nvidia-smi failed: {e}")
            return None

    def _get_git_info(self) -> Optional[Dict[str, Any]]:
        """Get git repository information if available."""
        try:
            logger.debug("     Checking git repository...")
            git_dir = subprocess.check_output(
                ["git", "rev-parse", "--git-dir"],
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd(),
                timeout=5,
            ).strip()

            if not git_dir:
                return None

            logger.debug("     Getting commit hash...")
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            ).strip()

            logger.debug("     Getting branch name...")
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            ).strip()

            logger.debug("     Checking for uncommitted changes...")
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            ).strip()

            logger.debug("     Getting remote URL...")
            try:
                remote_url = subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"],
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=5,
                ).strip()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                remote_url = None

            return {
                "commit_hash": commit_hash,
                "branch": branch,
                "has_uncommitted_changes": bool(status),
                "uncommitted_changes": status if status else None,
                "remote_url": remote_url,
            }
        except FileNotFoundError:
            logger.debug("     git not found in PATH")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("     ⚠️  git command timed out")
            return None
        except subprocess.CalledProcessError:
            logger.debug("     Not in a git repository")
            return None

    def _get_slurm_info(self) -> Optional[Dict[str, Any]]:
        """Get SLURM job information if running under SLURM."""
        slurm_vars = [
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_PARTITION",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_NUM_NODES",
            "SLURM_NTASKS",
            "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE",
            "SLURM_GPUS_PER_NODE",
            "SLURM_SUBMIT_DIR",
            "SLURM_CLUSTER_NAME",
            "SLURM_RESTART_COUNT",
        ]

        slurm_info = {}
        for var in slurm_vars:
            value = os.environ.get(var)
            if value is not None:
                slurm_info[var] = value

        if slurm_info:
            logger.debug(f"     Found {len(slurm_info)} SLURM variables")

        return slurm_info if slurm_info else None

    def _get_env_variables(self) -> Dict[str, str]:
        """Get relevant environment variables (filtered for safety)."""
        relevant_prefixes = [
            "CUDA_",
            "NCCL_",
            "OMP_",
            "MKL_",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "PATH",
            "MASTER_ADDR",
            "MASTER_PORT",
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
        ]

        env_vars = {}
        for key, value in os.environ.items():
            if any(key.startswith(prefix) for prefix in relevant_prefixes):
                env_vars[key] = value
                logger.debug(f"     Captured: {key}")

        return env_vars
