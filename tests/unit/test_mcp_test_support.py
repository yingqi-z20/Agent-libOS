from __future__ import annotations

from pathlib import Path
import re
import tomllib

import pytest

from scripts.mcp_test_support import (
    MCP_OPTIONAL_IMPORTS,
    missing_mcp_optional_dependencies,
)
import tests.conftest as repository_conftest


def test_mcp_dependency_probe_checks_every_distribution_in_the_extra() -> None:
    probed: list[str] = []

    def find_spec(module: str) -> object | None:
        probed.append(module)
        return None if module in {"httpcore2", "opentelemetry"} else object()

    assert missing_mcp_optional_dependencies(find_spec) == (
        "httpcore2",
        "opentelemetry-api",
    )
    assert probed == [module for _distribution, module in MCP_OPTIONAL_IMPORTS]


def test_mcp_dependency_probe_stays_closed_over_project_extra() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        re.match(r"[A-Za-z0-9_.-]+", requirement).group(0).replace("_", "-").lower()
        for requirement in project["project"]["optional-dependencies"]["mcp"]
    }

    assert {distribution for distribution, _module in MCP_OPTIONAL_IMPORTS} == declared


def test_direct_pytest_run_mcp_fails_before_collection_when_extra_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        @staticmethod
        def getoption(name: str, default: object = None) -> object:
            assert name == "--run-mcp"
            return True

    monkeypatch.setattr(
        repository_conftest,
        "missing_mcp_optional_dependencies",
        lambda: ("mcp", "httpx2"),
    )

    with pytest.raises(pytest.UsageError, match="missing: mcp, httpx2"):
        repository_conftest.pytest_configure(Config())  # type: ignore[arg-type]


def test_direct_pytest_without_run_mcp_does_not_require_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        @staticmethod
        def getoption(name: str, default: object = None) -> object:
            assert name == "--run-mcp"
            return False

    monkeypatch.setattr(
        repository_conftest,
        "missing_mcp_optional_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("must not probe")),
    )

    repository_conftest.pytest_configure(Config())  # type: ignore[arg-type]
