"""Failed downloads cannot replace a previously valid file."""

from unittest.mock import MagicMock, Mock

import pytest
import requests

from stable_pretraining.data import download as download_function
import importlib

downloads = importlib.import_module(download_function.__module__)
pytestmark = pytest.mark.unit


@pytest.fixture
def session(monkeypatch):
    response = Mock()
    response.headers = {"content-length": "4"}
    response.iter_content.return_value = [b"ab", b"", b"cd"]
    session = Mock(head=Mock(return_value=response), get=Mock(return_value=response))
    monkeypatch.setattr(downloads, "CachedSession", Mock(return_value=session))
    return session, response


@pytest.mark.parametrize("length", ["4", None])
def test_download_streams_bytes_and_reports_progress(session, tmp_path, length):
    connection, response = session
    response.headers = {} if length is None else {"content-length": length}
    progress = {}
    result = downloads.download(
        "https://example.test/file.bin?token=fake",
        tmp_path,
        progress_bar=False,
        _progress_dict=progress,
        _task_id="file",
    )
    assert result == tmp_path / "file.bin"
    assert result.read_bytes() == b"abcd"
    assert progress["file"]["progress"] == 4
    connection.close.assert_called_once_with()


@pytest.mark.parametrize("failure", ["http", "truncated", "stream", "interrupt"])
def test_failed_download_preserves_existing_file_and_removes_temporary_files(
    session, tmp_path, failure
):
    connection, response = session
    destination = tmp_path / "file.bin"
    destination.write_bytes(b"previous-valid-file")
    error = ValueError
    if failure == "http":
        error = requests.HTTPError
        response.raise_for_status.side_effect = error("503")
    elif failure == "truncated":
        response.iter_content.return_value = [b"ab"]
    else:
        error = KeyboardInterrupt if failure == "interrupt" else OSError

        def chunks(**kwargs):
            yield b"ab"
            raise error("connection lost")

        response.iter_content.side_effect = chunks
    with pytest.raises(error):
        downloads.download(
            "https://example.test/file.bin", tmp_path, progress_bar=False
        )
    assert destination.read_bytes() == b"previous-valid-file"
    assert {p.name for p in tmp_path.iterdir()} <= {"file.bin", "file.bin.lock"}
    connection.close.assert_called_once_with()


@pytest.mark.parametrize("fails", [False, True])
def test_bulk_download_waits_and_propagates_worker_errors(monkeypatch, tmp_path, fails):
    future = Mock()
    future.done.side_effect = [False, False, True]
    if fails:
        future.result.side_effect = OSError("worker failed")
    manager = MagicMock()
    manager.__enter__.return_value.dict.return_value = {
        "file.bin": {"progress": 1, "total": 4}
    }
    monkeypatch.setattr(downloads.multiprocessing, "Manager", lambda: manager)
    executor = MagicMock()
    executor.__enter__.return_value.submit.return_value = future
    monkeypatch.setattr(downloads, "ProcessPoolExecutor", lambda **kw: executor)
    monkeypatch.setattr(downloads.time, "sleep", lambda _: None)
    urls = iter(["https://example.test/file.bin"])
    if fails:
        with pytest.raises(OSError, match="worker failed"):
            downloads.bulk_download(urls, tmp_path)
    else:
        downloads.bulk_download(urls, tmp_path)
        future.result.assert_called_once_with()
    assert (
        executor.__enter__.return_value.submit.call_args.args[1]
        == "https://example.test/file.bin"
    )


def test_bulk_download_empty_input_does_not_start_processes(monkeypatch, tmp_path):
    pool = Mock(side_effect=AssertionError("must not start a pool"))
    monkeypatch.setattr(downloads, "ProcessPoolExecutor", pool)
    downloads.bulk_download([], tmp_path)
    pool.assert_not_called()


def test_http_compression_length_is_not_compared_to_decoded_content(session, tmp_path):
    connection, response = session
    response.headers = {"content-encoding": "gzip", "content-length": "24"}
    result = downloads.download(
        "https://example.test/file.bin", tmp_path, progress_bar=False
    )
    assert result.read_bytes() == b"abcd"
    connection.head.assert_not_called()
