from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from agent_libos import Runtime
from agent_libos.substrate import LocalResourceProviderSubstrate


def close_runtime(runtime: Runtime, *, timeout_s: float = 15.0) -> dict[str, Any]:
    """Retry a deliberately deferred Runtime shutdown until ownership is released."""

    deadline = time.monotonic() + timeout_s
    result = runtime.close()
    while not result.get("ok") and time.monotonic() < deadline:
        time.sleep(0.01)
        result = runtime.close()
    if not result.get("ok"):
        raise AssertionError(f"runtime did not close before test deadline: {result}")
    return result


@contextmanager
def temporary_runtime() -> Iterator[Runtime]:
    runtime = Runtime.open("local")
    try:
        yield runtime
    finally:
        runtime.close()


@contextmanager
def workspace_runtime() -> Iterator[tuple[Runtime, Path]]:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(root))
        try:
            yield runtime, root
        finally:
            runtime.close()
