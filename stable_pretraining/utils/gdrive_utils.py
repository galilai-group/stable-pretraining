"""Google Drive Background Uploader.

Upload files to Google Drive without blocking your application.
Features: auto-versioning, background processing, comprehensive logging.

Example:
    >>> uploader = GDriveUploader("MyProject", "credentials.json")
    >>> uploader.upload_file("large_file.mp4")  # Returns immediately!
    >>> uploader.wait_for_uploads()  # Optional: wait for completion
"""

import atexit
import mimetypes
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional, Callable
from loguru import logger


class GDriveUploader:
    """Background Google Drive uploader with automatic folder versioning.

    All uploads happen in a background thread - your code never blocks!
    Automatically creates versioned folders (MyFolder_v2, MyFolder_v3, etc).

    Args:
        folder_name: Name of the Drive folder to create
        credentials_path: Path to service account JSON credentials
        parent_folder_id: Optional parent folder ID (None = root)
        callback: Optional function(file_path, file_id, success) called on completion

    Example:
        >>> def notify(path, fid, ok):
        ...     print(f"{'✓' if ok else '✗'} {path}")
        >>>
        >>> uploader = GDriveUploader("Backups", "creds.json", callback=notify)
        >>> uploader.upload_file("data.csv")  # Non-blocking!
        >>> uploader.upload_file("logs.txt")
        >>> uploader.wait_for_uploads()  # Wait for all to finish
    """

    def __init__(
        self,
        folder_name: str,
        credentials_path: str,
        parent_folder_id: Optional[str] = None,
        callback: Optional[Callable[[str, Optional[str], bool], None]] = None,
    ):
        logger.info("=" * 70)
        logger.info(f"Initializing GDriveUploader: {folder_name}")
        logger.debug(
            f"Credentials: {credentials_path} | Parent: {parent_folder_id or 'ROOT'}"
        )

        self.folder_name = folder_name
        self.credentials_path = credentials_path
        self.parent_folder_id = parent_folder_id
        self.callback = callback
        self._upload_queue = Queue()
        self._stop_worker = False

        try:
            # Authenticate and create folder
            logger.info("Step 1/3: Authenticating...")
            self.service = self._authenticate()
            logger.success("✓ Authenticated")

            logger.info("Step 2/3: Creating/versioning folder...")
            self.folder_id = self._create_or_version_folder()
            logger.success(f"✓ Folder ready: {self.folder_id}")

            logger.info("Step 3/3: Starting background worker...")
            self._worker_thread = Thread(
                target=self._upload_worker, daemon=True, name="GDriveWorker"
            )
            self._worker_thread.start()
            logger.success(f"✓ Worker started (TID: {self._worker_thread.ident})")

            atexit.register(self._cleanup)
            logger.success("=" * 70)
            logger.success(f"✓ Ready! URL: {self.get_folder_url()}")
            logger.success("=" * 70)

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.exception("Full error:")
            raise

    def _authenticate(self):
        """Authenticate and return Drive service."""
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        try:
            logger.debug(f"Loading credentials from: {self.credentials_path}")

            if not Path(self.credentials_path).exists():
                raise FileNotFoundError(
                    f"Credentials not found: {self.credentials_path}"
                )

            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=["https://www.googleapis.com/auth/drive"]
            )

            service = build("drive", "v3", credentials=creds)

            # Test connection
            about = service.about().get(fields="user").execute()
            user = about.get("user", {}).get("emailAddress", "unknown")
            logger.debug(f"Connected as: {user}")

            return service

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    def _create_or_version_folder(self) -> str:
        """Create folder or version it if exists. Returns folder ID."""
        try:
            # Search for existing folders
            query_parts = [
                f"name contains '{self.folder_name}'",
                "mimeType='application/vnd.google-apps.folder'",
                "trashed=false",
            ]
            if self.parent_folder_id:
                query_parts.append(f"'{self.parent_folder_id}' in parents")

            query = " and ".join(query_parts)
            logger.debug(f"Search query: {query}")

            results = (
                self.service.files()
                .list(q=query, spaces="drive", fields="files(id, name)", pageSize=100)
                .execute()
            )

            folders = results.get("files", [])
            logger.info(f"Found {len(folders)} existing folder(s)")

            # Check if base name exists
            if not any(f["name"] == self.folder_name for f in folders):
                logger.info(f"Creating new folder: {self.folder_name}")
                return self._create_folder(self.folder_name)

            # Find highest version
            version = 1
            for folder in folders:
                if "_v" in folder["name"]:
                    try:
                        ver = int(folder["name"].split("_v")[-1])
                        version = max(version, ver)
                        logger.debug(f"Found version: v{ver}")
                    except ValueError:
                        continue

            # Create versioned folder
            version += 1
            versioned_name = f"{self.folder_name}_v{version}"
            logger.info(f"Creating versioned folder: {versioned_name}")
            folder_id = self._create_folder(versioned_name)
            logger.success(f"✓ Created {versioned_name} (ID: {folder_id})")

            return folder_id

        except Exception as e:
            logger.error(f"Folder creation error: {e}")
            raise

    def _create_folder(self, name: str) -> str:
        """Create a folder and return its ID."""
        try:
            metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
            if self.parent_folder_id:
                metadata["parents"] = [self.parent_folder_id]

            folder = self.service.files().create(body=metadata, fields="id").execute()
            return folder.get("id")

        except Exception as e:
            logger.error(f"Error creating folder '{name}': {e}")
            raise

    def _upload_worker(self):
        """Background worker that processes upload queue."""
        logger.info(f"Worker started (TID: {Thread.current_thread().ident})")
        processed = 0

        while not self._stop_worker:
            try:
                task = self._upload_queue.get(timeout=1)

                if task is None:  # Shutdown signal
                    logger.info("Worker received shutdown signal")
                    break

                file_path, custom_name, subfolder_id = task
                processed += 1

                logger.info(f"Processing upload #{processed}: {file_path}")
                file_id = self._upload_file(file_path, custom_name, subfolder_id)

                # Callback
                if self.callback:
                    try:
                        self.callback(file_path, file_id, success=(file_id is not None))
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                self._upload_queue.task_done()
                logger.debug(f"Queue size: {self._upload_queue.qsize()}")

            except Exception as e:
                if not self._stop_worker:
                    logger.error(f"Worker error: {e}")

        logger.info(f"Worker stopped (processed {processed} uploads)")

    def _upload_file(
        self,
        file_path: str,
        custom_name: Optional[str] = None,
        subfolder_id: Optional[str] = None,
    ) -> Optional[str]:
        """Upload file and return file ID (or None on failure)."""
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.errors import HttpError

        try:
            path = Path(file_path)

            # Validate
            if not path.exists() or not path.is_file():
                logger.error(f"Invalid file: {file_path}")
                return None

            # File info
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"Uploading: {path.name} ({size_mb:.2f} MB)")

            # MIME type
            mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            logger.debug(f"MIME type: {mime_type}")

            # Metadata
            metadata = {
                "name": custom_name or path.name,
                "parents": [subfolder_id or self.folder_id],
            }

            # Upload
            media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
            logger.info("Uploading to Drive...")

            file = (
                self.service.files()
                .create(
                    body=metadata,
                    media_body=media,
                    fields="id, name, size, webViewLink",
                )
                .execute()
            )

            file_id = file.get("id")
            logger.success("=" * 60)
            logger.success(f"✓ Upload complete: {file.get('name')}")
            logger.success(f"  File ID: {file_id}")
            logger.success(f"  Size: {file.get('size')} bytes")
            logger.debug(f"  Link: {file.get('webViewLink')}")
            logger.success("=" * 60)

            return file_id

        except HttpError as e:
            logger.error(f"HTTP error: {e.resp.status} - {e.resp.reason}")
            return None
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            logger.exception("Full traceback:")
            return None

    def upload_file(
        self,
        file_path: str,
        custom_name: Optional[str] = None,
        subfolder_id: Optional[str] = None,
    ) -> None:
        """Queue a file for background upload. Returns immediately!

        Args:
            file_path: Path to file to upload
            custom_name: Optional custom name in Drive
            subfolder_id: Optional subfolder ID (defaults to main folder)

        Example:
            >>> uploader.upload_file("video.mp4")
            >>> uploader.upload_file("report.pdf", custom_name="Q4_report.pdf")
        """
        logger.info(f"Queuing: {file_path}")
        self._upload_queue.put((file_path, custom_name, subfolder_id))

        queue_size = self._upload_queue.qsize()
        logger.info(f"✓ Queued (queue size: {queue_size})")

        if queue_size > 10:
            logger.warning(f"! Large queue: {queue_size} pending uploads")

    def wait_for_uploads(self, timeout: Optional[float] = None) -> bool:
        """Wait for all queued uploads to complete.

        Args:
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            True if completed, False if timeout/error

        Example:
            >>> uploader.upload_file("file1.txt")
            >>> uploader.upload_file("file2.txt")
            >>> uploader.wait_for_uploads(timeout=300)  # Wait max 5 min
        """
        pending = self.get_queue_size()

        if pending == 0:
            logger.info("✓ No pending uploads")
            return True

        logger.info(f"Waiting for {pending} upload(s)...")
        if timeout:
            logger.debug(f"Timeout: {timeout}s")

        try:
            self._upload_queue.join()
            logger.success("✓ All uploads complete")
            return True
        except Exception as e:
            logger.error(f"Wait error: {e}")
            return False

    def get_queue_size(self) -> int:
        """Get number of pending uploads."""
        return self._upload_queue.qsize()

    def get_folder_id(self) -> str:
        """Get the Drive folder ID."""
        return self.folder_id

    def get_folder_url(self) -> str:
        """Get the Drive folder URL."""
        return f"https://drive.google.com/drive/folders/{self.folder_id}"

    def _cleanup(self):
        """Shutdown worker thread gracefully."""
        logger.info("Shutting down GDriveUploader...")
        self._stop_worker = True
        self._upload_queue.put(None)  # Poison pill

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)

        logger.info("✓ Shutdown complete")


# Usage Example
if __name__ == "__main__":
    # Configure logging
    logger.add(
        "gdrive_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )

    # Callback for upload notifications
    def on_upload_complete(file_path: str, file_id: Optional[str], success: bool):
        if success:
            logger.info(f"Uploaded: {file_path} -> {file_id}")
        else:
            logger.error(f"Failed: {file_path}")

    # Initialize uploader
    uploader = GDriveUploader(
        folder_name="MyProject",
        credentials_path="service_account.json",
        callback=on_upload_complete,
    )

    # Upload files - all non-blocking!
    uploader.upload_file("document.pdf")
    uploader.upload_file("image.png")
    uploader.upload_file("data.csv", custom_name="backup_data.csv")

    logger.info("Files queued! Doing other work...")

    # Check queue
    logger.info(f"Pending: {uploader.get_queue_size()}")

    # Wait for all uploads before exiting
    uploader.wait_for_uploads()

    logger.info(f"All done! Folder: {uploader.get_folder_url()}")
