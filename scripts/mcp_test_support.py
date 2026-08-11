from __future__ import annotations

import importlib.util
from collections.abc import Callable


# Keep this list aligned with ``project.optional-dependencies.mcp``.  Checking
# the complete import surface prevents an explicitly requested MCP test run
# from turning into a green session made entirely of dependency skips.
MCP_OPTIONAL_IMPORTS: tuple[tuple[str, str], ...] = (
    ("anyio", "anyio"),
    ("httpcore2", "httpcore2"),
    ("httpx2", "httpx2"),
    ("mcp", "mcp"),
    ("keyring", "keyring"),
    ("opentelemetry-api", "opentelemetry"),
)


def missing_mcp_optional_dependencies(
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> tuple[str, ...]:
    """Return missing distributions required by the ``mcp`` project extra."""

    missing: list[str] = []
    for distribution, module in MCP_OPTIONAL_IMPORTS:
        try:
            available = find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(distribution)
    return tuple(missing)


def mcp_dependency_error(missing: tuple[str, ...]) -> str:
    joined = ", ".join(missing)
    return (
        f"--run-mcp requires the complete MCP optional dependency set; missing: {joined}. "
        "Install it with `uv sync --frozen --extra mcp`."
    )
