#!/usr/bin/env python3
"""Fail when an MCP pytest node can fall outside the release marker closure."""

from __future__ import annotations

import contextlib
import inspect
import io
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


class _Collection:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.items = list(session.items)


def _looks_like_mcp(item: Any) -> bool:
    path = Path(str(item.path)).resolve()
    if path.name.lower().startswith("test_mcp"):
        return True
    if "mcp" in str(item.nodeid).lower():
        return True
    if any("mcp" in str(name).lower() for name in item.fixturenames):
        return True
    try:
        source = inspect.getsource(item.obj)
    except (OSError, TypeError):
        return False
    return "mcp" in source.lower()


def main() -> int:
    captured = _Collection()
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = pytest.main(
            ["--collect-only", "-q", str(ROOT / "tests")],
            plugins=[captured],
        )
    if status != pytest.ExitCode.OK:
        sys.stderr.write(stdout.getvalue())
        sys.stderr.write(stderr.getvalue())
        return int(status)

    expected = {item.nodeid for item in captured.items if _looks_like_mcp(item)}
    marked = {item.nodeid for item in captured.items if "mcp" in item.keywords}
    transport = {
        item.nodeid for item in captured.items if "mcp_transport" in item.keywords
    }
    missing = sorted(expected - marked)
    unclosed_transport = sorted(transport - marked)
    if missing or unclosed_transport:
        for nodeid in missing:
            print(f"MCP closure check failed: unmarked MCP node: {nodeid}", file=sys.stderr)
        for nodeid in unclosed_transport:
            print(
                f"MCP closure check failed: transport node lacks mcp marker: {nodeid}",
                file=sys.stderr,
            )
        return 1
    if not marked or not transport:
        print("MCP closure check failed: marker closure is empty", file=sys.stderr)
        return 1
    print(
        f"validated MCP pytest closure: {len(marked)} product nodes; "
        f"{len(transport)} frozen-SDK transport nodes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
