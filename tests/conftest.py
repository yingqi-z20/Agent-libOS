from __future__ import annotations

import os
import json
import shutil
from pathlib import Path

import pytest
from scripts.agent_outputs import cleanup_agent_outputs, snapshot_agent_outputs
from agent_libos.utils.yaml_loader import load_yaml_mapping

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
        "--run-real-deno",
        action="store_true",
        default=False,
        help="deprecated; real Deno tests run by default when deno is installed",
    )
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
        "--keep-agent-outputs",
        action="store_true",
        default=False,
        help="preserve files written under agent_outputs during this pytest run",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    config = session.config
    if _skip_agent_outputs_cleanup(config):
        return
    root = Path(config.rootpath) / "agent_outputs"
    config._agent_outputs_baseline = snapshot_agent_outputs(root)  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if _skip_agent_outputs_cleanup(config):
        return
    root = Path(config.rootpath) / "agent_outputs"
    baseline = getattr(config, "_agent_outputs_baseline", set())
    cleanup_agent_outputs(root, baseline=set(baseline))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    root = Path(config.rootpath)
    skip_real_deno = bool(config.getoption("--skip-real-deno"))
    run_real_llm = bool(config.getoption("--run-real-llm"))
    invariant_marks = _load_invariant_marks(root)

    for item in items:
        _mark_lane(root, item)
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
