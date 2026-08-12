from __future__ import annotations

import os
import json
import inspect
import shutil
from pathlib import Path

import pytest
from scripts.agent_outputs import cleanup_agent_outputs, snapshot_agent_outputs
from agent_libos.utils.yaml_loader import load_yaml_mapping
from scripts.mcp_test_support import (
    mcp_dependency_error,
    missing_mcp_optional_dependencies,
)

LANE_DIRS = {
    "unit": "unit",
    "runtime": "runtime",
    "security": "security",
    "self_evolution": "self_evolution",
    "providers": "providers",
    "benchmark": "benchmarks",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-real-deno",
        action="store_true",
        default=False,
        help="skip tests that execute a real deno binary",
    )
    parser.addoption(
        "--run-real-llm",
        action="store_true",
        default=False,
        help="run tests that spend real LLM/provider calls",
    )
    parser.addoption(
        "--run-postgres",
        action="store_true",
        default=False,
        help="run PostgreSQL runtime store integration tests",
    )
    parser.addoption(
        "--run-mcp",
        action="store_true",
        default=False,
        help="run real MCP SDK integration tests",
    )
    parser.addoption(
        "--keep-agent-outputs",
        action="store_true",
        default=False,
        help="preserve files written under agent_outputs during this pytest run",
    )
    parser.addoption(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help="fail the pytest session when any selected test is skipped",
    )


def pytest_configure(config: pytest.Config) -> None:
    if not bool(config.getoption("--run-mcp", default=False)):
        return
    missing = missing_mcp_optional_dependencies()
    if missing:
        raise pytest.UsageError(mcp_dependency_error(missing))


def pytest_sessionstart(session: pytest.Session) -> None:
    config = session.config
    if _skip_agent_outputs_cleanup(config):
        return
    root = Path(config.rootpath) / "agent_outputs"
    config._agent_outputs_baseline = snapshot_agent_outputs(root)  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if not _skip_agent_outputs_cleanup(config):
        root = Path(config.rootpath) / "agent_outputs"
        baseline = getattr(config, "_agent_outputs_baseline", set())
        cleanup_agent_outputs(root, baseline=set(baseline))
    if bool(config.getoption("--fail-on-skip", default=False)):
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        skipped = list(getattr(reporter, "stats", {}).get("skipped", ()))
        if skipped and exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    root = Path(config.rootpath)
    skip_real_deno = bool(config.getoption("--skip-real-deno"))
    run_real_llm = bool(config.getoption("--run-real-llm"))
    run_postgres = bool(config.getoption("--run-postgres"))
    run_mcp = bool(config.getoption("--run-mcp"))
    invariant_marks = _load_invariant_marks(root)

    for item in items:
        _mark_lane(root, item)
        # ``mcp`` is the product-area closure marker, not merely a synonym for
        # tests which import the optional SDK.  Apply it before pytest evaluates
        # ``-m`` so the release job cannot lose a new MCP regression simply
        # because it was added outside tests/providers/test_mcp*.py.
        if _is_mcp_test_item(root, item):
            item.add_marker(pytest.mark.mcp)
        for invariant_id in invariant_marks.get(item.nodeid.replace("\\", "/"), ()):
            item.add_marker(pytest.mark.invariant(invariant_id))
        if "real_deno" in item.keywords:
            if skip_real_deno:
                item.add_marker(pytest.mark.skip(reason="real Deno tests skipped by --skip-real-deno"))
            elif shutil.which("deno") is None:
                item.add_marker(pytest.mark.skip(reason="deno not installed"))
        if "real_llm" in item.keywords:
            if not run_real_llm:
                item.add_marker(pytest.mark.skip(reason="real LLM tests require --run-real-llm"))
            elif not _has_real_llm_environment():
                item.add_marker(pytest.mark.skip(reason="real LLM environment is not configured"))
        if "postgres" in item.keywords:
            if not run_postgres:
                item.add_marker(pytest.mark.skip(reason="PostgreSQL tests require --run-postgres"))
            elif not os.getenv("AGENT_LIBOS_POSTGRES_DSN"):
                item.add_marker(pytest.mark.skip(reason="AGENT_LIBOS_POSTGRES_DSN is not configured"))
        if "mcp_transport" in item.keywords:
            if not run_mcp:
                item.add_marker(pytest.mark.skip(reason="MCP integration tests require --run-mcp"))


def _skip_agent_outputs_cleanup(config: pytest.Config) -> bool:
    if getattr(config, "workerinput", None) is not None:
        return True
    if bool(config.getoption("--keep-agent-outputs", default=False)):
        return True
    if os.getenv("AGENT_LIBOS_KEEP_AGENT_OUTPUTS"):
        return True
    return False


def _mark_lane(root: Path, item: pytest.Item) -> None:
    try:
        rel = Path(str(item.fspath)).resolve().relative_to(root)
    except ValueError:
        return
    parts = rel.parts
    if len(parts) < 2 or parts[0] != "tests":
        return
    for marker, directory in LANE_DIRS.items():
        if parts[1] == directory:
            item.add_marker(getattr(pytest.mark, marker))
            return


def _is_mcp_test_item(root: Path, item: pytest.Item) -> bool:
    """Classify every MCP regression independently of its lane or directory.

    The filename/function/parameter-id rule is intentionally broad.  Source
    inspection closes generic files such as CLI, GUI, storage, and release
    contracts whose test names do not necessarily repeat the subsystem name.
    Explicit markers remain authoritative for tests generated by helpers whose
    source cannot be inspected.
    """

    if "mcp" in item.keywords:
        return True
    try:
        relative = Path(str(item.fspath)).resolve().relative_to(root)
    except ValueError:
        return False
    if relative.name.lower().startswith("test_mcp"):
        return True
    if "mcp" in item.nodeid.lower():
        return True
    if any("mcp" in name.lower() for name in getattr(item, "fixturenames", ())):
        return True
    target = getattr(item, "obj", None)
    if target is None:
        return False
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return False
    return "mcp" in source.lower()


def _has_real_llm_environment() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        and (os.getenv("OPENAI_LANGUAGE_MODEL") or os.getenv("OPENAI_MODEL"))
    )


def _load_invariant_marks(root: Path) -> dict[str, list[str]]:
    manifest = root / "tests" / "invariants.yaml"
    if not manifest.exists():
        return {}
    text = manifest.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = load_yaml_mapping(text)
    marks: dict[str, list[str]] = {}
    for invariant in data.get("invariants", []):
        invariant_id = str(invariant.get("id", "")).strip()
        if not invariant_id:
            continue
        for node_id in invariant.get("node_ids", []):
            if isinstance(node_id, str) and node_id:
                marks.setdefault(node_id.replace("\\", "/"), []).append(invariant_id)
    return marks
