"""Download utilities for fetching datasets and model weights.

This module provides functions for downloading files from URLs with progress tracking,
caching, and concurrent download support.
"""

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Union
from urllib.parse import urlparse

import rich.progress
from filelock import FileLock
from loguru import logger as logging
from requests_cache import CachedSession
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm


def bulk_download(
    urls: Iterable[str],
    dest_folder: Union[str, Path],
    backend: str = "filesystem",
    cache_dir: str = "~/.stable_pretraining/",
):
    """Download multiple files concurrently.

    Example::

        import stable_pretraining

        stable_pretraining.data.bulk_download(
            [
                "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
            ],
            "todelete",
        )

    Args:
        urls (Iterable[str]): List of URLs to download
        dest_folder (Union[str, Path]): Destination folder for downloads
        backend (str, optional): Storage backend type. Defaults to "filesystem".
        cache_dir (str, optional): Cache directory path. Defaults to "~/.stable_pretraining/".
    """
    num_workers = len(urls)
    filenames = [os.path.basename(urlparse(url).path) for url in urls]
    with rich.progress.Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        refresh_per_second=5,
    ) as progress:
        futures = []
        with multiprocessing.Manager() as manager:
            _progress = manager.dict()  # Shared dictionary for progress
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for i in range(num_workers):
                    task_id = filenames[i]
                    future = executor.submit(
                        download,
                        urls[i],
                        dest_folder,
                        backend,
                        cache_dir,
                        False,
                        _progress,
                        task_id,
                    )
                    futures.append(future)
                rich_tasks = {}

                while not all(future.done() for future in futures):
                    for task_id in list(_progress.keys()):
                        if task_id in rich_tasks:
                            progress.update(
                                rich_tasks[task_id],
                                completed=_progress[task_id]["progress"],
                            )
                        else:
                            rich_tasks[task_id] = progress.add_task(
                                f"[green]{task_id}",
                                total=_progress[task_id]["total"],
                                visible=True,
                            )
                    time.sleep(0.01)


def download(
    url,
    dest_folder,
    backend="filesystem",
    cache_dir="~/.stable_pretraining/",
    progress_bar=True,
    _progress_dict=None,
    _task_id=None,
):
    """Download a file from a URL with progress tracking.

    Args:
        url: URL to download from
        dest_folder: Destination folder for the download
        backend: Storage backend type
        cache_dir: Cache directory path
        progress_bar: Whether to show progress bar
        _progress_dict: Internal dictionary for progress tracking
        _task_id: Internal task ID for bulk downloads

    Returns:
        Path to the downloaded file or None if download failed
    """
    try:
        filename = os.path.basename(urlparse(url).path)
        dest_folder = Path(dest_folder)
        dest_folder.mkdir(exist_ok=True, parents=True)
        local_filename = dest_folder / filename
        lock_filename = dest_folder / f"{filename}.lock"
        # Use a file lock to prevent concurrent downloads
        with FileLock(lock_filename):
            # Download the file
            session = CachedSession(cache_dir, backend=backend)
            logging.info(f"Downloading: {url}")
            response = session.head(url)
            total_size = int(response.headers.get("content-length", 0))
            logging.info(f"Total size: {total_size}")

            response = session.get(url, stream=True)
            downloaded_size = 0
            # Write the file to the destination folder
            with (
                open(local_filename, "wb") as f,
                tqdm(
                    desc=local_filename.name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    disable=not progress_bar,
                ) as bar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    bar.update(len(chunk))
                    if _progress_dict is not None:
                        _progress_dict[_task_id] = {
                            "progress": downloaded_size,
                            "total": total_size,
                        }
            if downloaded_size == total_size:
                logging.info("Download complete and successful!")
            else:
                logging.error("Download incomplete or corrupted.")
            return local_filename
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        raise e
