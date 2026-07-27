from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


class AtomicJsonOutput:
    """Reserve one report path and publish only complete JSON artifacts.

    The reservation replaces any prior report with a small, atomic
    ``in_progress`` marker while retaining the prior complete artifact beside
    it.  A failed invocation leaves a non-favorable ``failed`` marker and the
    retained prior artifact; a successful commit atomically replaces the
    marker and removes the retained copy.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        self.invocation_id = f"evaluation_{uuid.uuid4().hex}"
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._backup_path = self.path.with_name(
            f".{self.path.name}.previous.{self.invocation_id}"
        )
        self._lock_fd: int | None = None
        self._has_backup = False
        self._committed = False

    def __enter__(self) -> AtomicJsonOutput:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.path.is_file():
            raise IsADirectoryError(f"evaluation output is not a file: {self.path}")
        self._lock_fd = os.open(
            self._lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(self._lock_fd, (self.invocation_id + "\n").encode("ascii"))
            os.fsync(self._lock_fd)
            if self.path.exists():
                os.replace(self.path, self._backup_path)
                self._has_backup = True
            self._write_marker("in_progress")
        except BaseException:
            if self._has_backup:
                os.replace(self._backup_path, self.path)
                self._has_backup = False
            self._release_lock()
            raise
        return self

    def commit(self, value: Any, *, sort_keys: bool = False) -> str:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=sort_keys,
        ) + "\n"
        self.commit_text(rendered)
        return rendered

    def commit_text(self, rendered: str) -> None:
        if self._lock_fd is None:
            raise RuntimeError("evaluation output is not reserved")
        if self._committed:
            raise RuntimeError("evaluation output was already committed")
        _write_text_atomic(self.path, rendered)
        self._committed = True
        if self._has_backup:
            self._backup_path.unlink(missing_ok=True)
            self._has_backup = False

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        failures: list[BaseException] = []
        try:
            if not self._committed:
                self._write_marker("failed")
        except BaseException as marker_error:
            failures.append(marker_error)
        try:
            self._release_lock()
        except BaseException as release_error:
            failures.append(release_error)
        if failures:
            if isinstance(exc, BaseException):
                for failure in failures:
                    exc.add_note(
                        "evaluation-output cleanup failure: "
                        f"{type(failure).__name__}: {failure}"
                    )
                return False
            if all(isinstance(failure, Exception) for failure in failures):
                raise ExceptionGroup(
                    "evaluation-output cleanup failed",
                    [
                        failure
                        for failure in failures
                        if isinstance(failure, Exception)
                    ],
                )
            raise BaseExceptionGroup(
                "evaluation-output cleanup failed",
                failures,
            )
        return False

    def _write_marker(self, state: str) -> None:
        marker = {
            "evaluation_artifact": {
                "schema_version": 1,
                "completion_state": state,
                "invocation_id": self.invocation_id,
                "previous_artifact": (
                    self._backup_path.name if self._has_backup else None
                ),
            }
        }
        _write_text_atomic(
            self.path,
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        )

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
        self._lock_path.unlink(missing_ok=True)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
