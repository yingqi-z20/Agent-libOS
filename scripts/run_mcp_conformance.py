#!/usr/bin/env python3
"""Run the official MCP 2026-07-28 Tools-only client conformance gate.

The official conformance runner starts its own scenario server and appends the
server URL to ``--command``.  This file therefore has two entry points:

* the default release gate, which fetches and verifies one immutable official
  conformance commit before running the allowlisted scenarios; and
* ``_client``, the private command driven by that runner.  It uses the real
  Agent libOS Runtime, registry, protected operations, and SDK provider.

No expected-failure baseline is accepted.  A scenario must produce non-empty
checks, every requirement-level check must be observed, and FAILURE/WARNING
results fail the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


OFFICIAL_CONFORMANCE_REPOSITORY = (
    "https://github.com/modelcontextprotocol/conformance.git"
)
OFFICIAL_CONFORMANCE_COMMIT = "81eb1c3edaed87d7fd585d7b80186da7a2960660"
OFFICIAL_CONFORMANCE_PACKAGE_VERSION = "0.2.0-alpha.10"
MCP_PROTOCOL_REVISION = "2026-07-28"


class ConformanceGateError(RuntimeError):
    """A stable, user-actionable conformance gate failure."""


@dataclass(frozen=True)
class ScenarioContract:
    """Checks that prove an official client scenario was actually exercised."""

    required_success_ids: frozenset[str]
    required_present_ids: frozenset[str] = frozenset()
    required_skipped_ids: frozenset[str] = frozenset()


# This is deliberately not an upstream suite name.  The official ``draft`` and
# ``all`` suites include OAuth, MRTR, and other product surfaces that 1.2.1 does
# not implement.  The allowlist is the complete non-auth, non-MRTR client set at
# the pinned commit that exercises the shipped Tools-only modern surface.
OFFICIAL_TOOLS_ONLY_SCENARIOS: Mapping[str, ScenarioContract] = {
    "tools_call": ScenarioContract(
        required_success_ids=frozenset(
            {"tool-add-numbers", "wire-schema-valid"}
        ),
    ),
    "request-metadata": ScenarioContract(
        required_success_ids=frozenset(
            {
                "sep-2575-http-client-sends-version-header",
                "sep-2575-client-populates-meta",
                "sep-2575-client-sends-client-info",
                "sep-2575-http-version-header-matches-meta",
                "sep-2575-client-retry-supported-version",
            }
        ),
        required_present_ids=frozenset(
            {
                "sep-2575-client-declares-roots-capability",
                "sep-2575-client-declares-sampling-capability",
                "sep-2575-client-declares-elicitation-capability",
            }
        ),
        # Agent libOS intentionally advertises none of these unsupported client
        # capabilities.  The official scenario reports that as SKIPPED.
        required_skipped_ids=frozenset(
            {
                "sep-2575-client-declares-roots-capability",
                "sep-2575-client-declares-sampling-capability",
                "sep-2575-client-declares-elicitation-capability",
            }
        ),
    ),
    "http-standard-headers": ScenarioContract(
        required_success_ids=frozenset(
            {"sep-2243-client-includes-standard-headers"}
        ),
    ),
    "http-custom-headers": ScenarioContract(
        required_success_ids=frozenset(
            {
                "sep-2243-client-supports-custom-headers",
                "sep-2243-client-mirrors-designated-params",
                "sep-2243-client-encode-values",
                "sep-2243-client-base64-unsafe",
                "sep-2243-client-omit-null",
            }
        ),
    ),
    "http-invalid-tool-headers": ScenarioContract(
        required_success_ids=frozenset(
            {
                "sep-2243-client-reject-invalid-tool",
                "sep-2243-x-mcp-header-not-empty",
                "sep-2243-x-mcp-header-charset",
                "sep-2243-x-mcp-header-unique",
                "sep-2243-x-mcp-header-primitive-only",
            }
        ),
    ),
    "json-schema-ref-no-deref": ScenarioContract(
        required_success_ids=frozenset({"sep-2106-no-network-ref-deref"}),
    ),
    "json-schema-2020-12-preservation": ScenarioContract(
        required_success_ids=frozenset(
            {
                "json-schema-2020-12-client-tool-found",
                "json-schema-2020-12-client-echo-completed",
                "json-schema-2020-12-client-$schema-preserved",
                "json-schema-2020-12-client-$defs-preserved",
                "json-schema-2020-12-client-additionalProperties-preserved",
                "sep-2106-client-composition-keywords-preserved",
                "sep-2106-client-conditional-keywords-preserved",
                "sep-2106-client-anchor-keyword-preserved",
                "wire-schema-valid",
            }
        ),
    ),
}


@dataclass(frozen=True)
class _ToolPlan:
    tool_id: str
    mcp_name: str


_SCENARIO_TOOLS: Mapping[str, tuple[_ToolPlan, ...]] = {
    "tools_call": (_ToolPlan("add_numbers", "add_numbers"),),
    # Manifest v2 requires a non-empty Host allowlist even though this scenario
    # stops after discovery and never consumes the placeholder.
    "request-metadata": (_ToolPlan("unused", "conformance_unused"),),
    "http-standard-headers": (_ToolPlan("test_headers", "test_headers"),),
    "http-custom-headers": (
        _ToolPlan("custom_headers", "test_custom_headers"),
        _ToolPlan("custom_headers_null", "test_custom_headers_null"),
    ),
    # Invalid server-advertised tools are deliberately absent from Host
    # authority.  Calling the retained valid tool proves they did not poison it.
    "http-invalid-tool-headers": (_ToolPlan("valid", "valid_tool"),),
    "json-schema-ref-no-deref": (_ToolPlan("lookup_user", "lookup_user"),),
    "json-schema-2020-12-preservation": (
        _ToolPlan("schema_source", "json_schema_2020_12_tool"),
        _ToolPlan("schema_echo", "json_schema_echo"),
    ),
}

_SERVER_ID = "official-conformance"
_SAFE_HARNESS_COMMAND_PART = re.compile(r"^[A-Za-z0-9_./:\\-]+$")
_INHERITED_HARNESS_ENV = (
    # Executable discovery and stable text handling are the only ambient
    # inputs the pinned checkout/build/runner needs.  Windows process creation
    # additionally requires these fixed system fields.
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
)


def _run_checked(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        command = " ".join(argv[:3])
        output = (completed.stderr or completed.stdout).strip()
        if len(output) > 4_000:
            output = output[-4_000:]
        raise ConformanceGateError(
            f"command failed ({completed.returncode}): {command}\n{output}"
        )
    return completed


def _required_executable(name: str, env: Mapping[str, str]) -> str:
    executable = shutil.which(name, path=env.get("PATH"))
    if executable is None:
        raise ConformanceGateError(
            f"official MCP conformance requires {name} on the allowed PATH"
        )
    # Freeze discovery before changing cwd into the untrusted checkout.  A
    # relative PATH entry must never turn into checkout-controlled execution.
    return os.path.abspath(executable)


def _verify_official_checkout(
    checkout: Path,
    *,
    env: Mapping[str, str],
) -> None:
    if not (checkout / ".git").exists():
        raise ConformanceGateError(
            f"official conformance checkout has no .git directory: {checkout}"
        )
    revision = _run_checked(
        (_required_executable("git", env), "rev-parse", "HEAD"),
        cwd=checkout,
        env=env,
    ).stdout.strip()
    if revision != OFFICIAL_CONFORMANCE_COMMIT:
        raise ConformanceGateError(
            "official conformance checkout revision mismatch: "
            f"expected {OFFICIAL_CONFORMANCE_COMMIT}, got {revision}"
        )
    try:
        package = json.loads((checkout / "package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConformanceGateError("cannot read official package.json") from exc
    if package.get("version") != OFFICIAL_CONFORMANCE_PACKAGE_VERSION:
        raise ConformanceGateError(
            "official conformance package version mismatch: "
            f"expected {OFFICIAL_CONFORMANCE_PACKAGE_VERSION}, "
            f"got {package.get('version')!r}"
        )
    if not (checkout / "package-lock.json").is_file():
        raise ConformanceGateError("official checkout is missing package-lock.json")


@contextmanager
def _official_checkout(
    selected: Path | None,
    *,
    env: Mapping[str, str],
) -> Iterator[Path]:
    if selected is not None:
        checkout = selected.resolve()
        _verify_official_checkout(checkout, env=env)
        yield checkout
        return

    with tempfile.TemporaryDirectory(prefix="agent-libos-mcp-conformance-") as root:
        checkout = Path(root) / "official"
        git = _required_executable("git", env)
        _run_checked((git, "init", str(checkout)), env=env)
        _run_checked(
            (
                git,
                "-C",
                str(checkout),
                "remote",
                "add",
                "origin",
                OFFICIAL_CONFORMANCE_REPOSITORY,
            ),
            env=env,
        )
        _run_checked(
            (
                git,
                "-C",
                str(checkout),
                "fetch",
                "--depth",
                "1",
                "origin",
                OFFICIAL_CONFORMANCE_COMMIT,
            ),
            env=env,
        )
        _run_checked(
            (
                git,
                "-C",
                str(checkout),
                "checkout",
                "--detach",
                OFFICIAL_CONFORMANCE_COMMIT,
            ),
            env=env,
        )
        _verify_official_checkout(checkout, env=env)
        yield checkout


def _prepare_official_runner(
    checkout: Path,
    *,
    env: Mapping[str, str],
) -> Path:
    npm = _required_executable("npm", env)
    _required_executable("node", env)
    # Ignore lifecycle scripts during installation, then run only the pinned
    # repository's explicit build target.  package-lock.json fixes every npm
    # artifact used by the official runner.
    _run_checked(
        (npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"),
        cwd=checkout,
        env=env,
    )
    _run_checked(
        (npm, "run", "build", "--silent"),
        cwd=checkout,
        env=env,
    )
    entrypoint = checkout / "dist" / "index.js"
    if not entrypoint.is_file():
        raise ConformanceGateError("official conformance build produced no dist/index.js")
    return entrypoint


def _prepared_official_runner(checkout: Path) -> Path:
    entrypoint = checkout / "dist" / "index.js"
    if not entrypoint.is_file():
        raise ConformanceGateError(
            "--skip-prepare requires an existing official dist/index.js"
        )
    return entrypoint


def _client_command() -> str:
    # Preserve a virtual-environment launcher instead of resolving its symlink
    # to the base interpreter (which would lose the environment's packages).
    parts = (
        os.path.abspath(sys.executable),
        str(Path(__file__).resolve()),
        "_client",
    )
    # The pinned official runner tokenizes --command with spaces and launches
    # through a shell.  Reject paths that it cannot represent safely.
    for part in parts:
        if not _SAFE_HARNESS_COMMAND_PART.fullmatch(part):
            raise ConformanceGateError(
                "official conformance cannot safely represent the client command path: "
                f"{part!r}"
            )
    return " ".join(parts)


def _sanitized_client_environment(isolation_root: Path) -> dict[str, str]:
    """Build the checkout/build/runner/client environment from an allowlist."""

    root = isolation_root.resolve()
    home = root / "home"
    temporary = root / "tmp"
    cache = root / "cache"
    config = root / "config"
    npm_cache = cache / "npm"
    npm_userconfig = root / "npmrc"
    for directory in (home, temporary, cache, config, npm_cache):
        directory.mkdir(parents=True, exist_ok=True)

    selected = {
        name: os.environ[name]
        for name in _INHERITED_HARNESS_ENV
        if os.environ.get(name)
    }
    selected.setdefault("PATH", os.defpath)
    selected.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
            "XDG_CACHE_HOME": str(cache),
            "XDG_CONFIG_HOME": str(config),
            "npm_config_cache": str(npm_cache),
            "npm_config_userconfig": str(npm_userconfig),
            "npm_config_update_notifier": "false",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONNOUSERSITE": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
        }
    )
    return selected


def _load_checks(result_root: Path) -> list[dict[str, Any]]:
    paths = sorted(result_root.rglob("checks.json"))
    if len(paths) != 1:
        raise ConformanceGateError(
            f"official scenario emitted {len(paths)} checks.json files; expected exactly one"
        )
    try:
        value = json.loads(paths[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConformanceGateError("official checks.json is unreadable") from exc
    if not isinstance(value, list) or not value:
        raise ConformanceGateError("official scenario returned an empty check set")
    if not all(isinstance(item, dict) for item in value):
        raise ConformanceGateError("official checks.json must contain only objects")
    return value


def _validate_checks(
    scenario: str,
    checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = OFFICIAL_TOOLS_ONLY_SCENARIOS[scenario]
    statuses_by_id: dict[str, set[str]] = {}
    failures: list[str] = []
    for index, check in enumerate(checks):
        check_id = check.get("id")
        status = check.get("status")
        if not isinstance(check_id, str) or not isinstance(status, str):
            raise ConformanceGateError(
                f"{scenario}: check {index} has no string id/status"
            )
        if status not in {"SUCCESS", "FAILURE", "WARNING", "SKIPPED", "INFO"}:
            raise ConformanceGateError(
                f"{scenario}: check {check_id} has unknown status {status!r}"
            )
        statuses_by_id.setdefault(check_id, set()).add(status)
        if status in {"FAILURE", "WARNING"}:
            failures.append(f"{check_id}={status}")
    if failures:
        raise ConformanceGateError(
            f"{scenario}: official conformance failures: {', '.join(failures)}"
        )
    missing_success = sorted(
        check_id
        for check_id in contract.required_success_ids
        if "SUCCESS" not in statuses_by_id.get(check_id, set())
    )
    missing_present = sorted(
        check_id
        for check_id in contract.required_present_ids
        if check_id not in statuses_by_id
    )
    missing_skipped = sorted(
        check_id
        for check_id in contract.required_skipped_ids
        if "SKIPPED" not in statuses_by_id.get(check_id, set())
    )
    if missing_success or missing_present or missing_skipped:
        raise ConformanceGateError(
            f"{scenario}: incomplete official evidence "
            f"success={missing_success}, present={missing_present}, "
            f"skipped={missing_skipped}"
        )
    if "wire-schema-harness-error" in statuses_by_id:
        raise ConformanceGateError(
            f"{scenario}: official harness reported a wire-schema harness error"
        )
    return {
        "scenario": scenario,
        "check_count": len(checks),
        "success_count": sum(
            check.get("status") == "SUCCESS" for check in checks
        ),
        "skipped_count": sum(
            check.get("status") == "SKIPPED" for check in checks
        ),
        "check_ids": sorted(statuses_by_id),
    }


def _run_official_scenario(
    *,
    node: str,
    runner: Path,
    checkout: Path,
    scenario: str,
    output_root: Path,
    timeout_ms: int,
    env: Mapping[str, str],
) -> dict[str, Any]:
    scenario_output = output_root / scenario
    scenario_output.mkdir(parents=True, exist_ok=False)
    # Intentionally no --expected-failures argument.
    _run_checked(
        (
            node,
            str(runner),
            "client",
            "--command",
            _client_command(),
            "--scenario",
            scenario,
            "--spec-version",
            MCP_PROTOCOL_REVISION,
            "--timeout",
            str(timeout_ms),
            "--output-dir",
            str(scenario_output),
            "--verbose",
        ),
        cwd=checkout,
        env=env,
    )
    return _validate_checks(scenario, _load_checks(scenario_output))


def run_gate(args: argparse.Namespace) -> int:
    selected = tuple(args.scenario or OFFICIAL_TOOLS_ONLY_SCENARIOS)
    unknown = sorted(set(selected) - set(OFFICIAL_TOOLS_ONLY_SCENARIOS))
    if unknown:
        raise ConformanceGateError(
            "unsupported conformance scenario selection: " + ", ".join(unknown)
        )
    if len(set(selected)) != len(selected):
        raise ConformanceGateError("conformance scenarios must not be repeated")
    if args.timeout_ms < 1_000 or args.timeout_ms > 120_000:
        raise ConformanceGateError("--timeout-ms must be between 1000 and 120000")

    output_parent = Path(args.output_dir).resolve()
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = output_parent / f"{run_name}-{os.getpid()}"
    output_root.mkdir(parents=True, exist_ok=False)

    selected_checkout = (
        None if args.official_checkout is None else Path(args.official_checkout)
    )
    with tempfile.TemporaryDirectory(
        prefix="agent-libos-mcp-conformance-env-"
    ) as isolation:
        environment = _sanitized_client_environment(Path(isolation))
        with _official_checkout(selected_checkout, env=environment) as checkout:
            runner = (
                _prepared_official_runner(checkout)
                if args.skip_prepare
                else _prepare_official_runner(checkout, env=environment)
            )
            node = _required_executable("node", environment)
            results = [
                _run_official_scenario(
                    node=node,
                    runner=runner,
                    checkout=checkout,
                    scenario=scenario,
                    output_root=output_root,
                    timeout_ms=args.timeout_ms,
                    env=environment,
                )
                for scenario in selected
            ]

    summary = {
        "gate": "official-mcp-tools-only-client",
        "official_repository": OFFICIAL_CONFORMANCE_REPOSITORY,
        "official_commit": OFFICIAL_CONFORMANCE_COMMIT,
        "official_package_version": OFFICIAL_CONFORMANCE_PACKAGE_VERSION,
        "protocol_revision": MCP_PROTOCOL_REVISION,
        "complete_allowlist": set(selected) == set(OFFICIAL_TOOLS_ONLY_SCENARIOS),
        "scenarios": results,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"official MCP Tools-only client conformance passed: "
        f"{len(results)}/{len(selected)} scenarios; artifacts={output_root}"
    )
    return 0


def _manifest(server_url: str, scenario: str) -> dict[str, Any]:
    tools = []
    for tool in _SCENARIO_TOOLS[scenario]:
        tools.append(
            {
                "tool_id": tool.tool_id,
                "mcp_name": tool.mcp_name,
                "right": "read",
                "rollback_class": "no_rollback_required",
                "state_mutation": False,
                "information_flow": True,
                # The official scenario server is the fixture source.  Empty
                # Host schemas keep authority fixed while the live schema is
                # exercised as untrusted provider metadata in each operation.
                "input_schema": {},
            }
        )
    return {
        "schema_version": 2,
        "server_id": _SERVER_ID,
        "transport": "streamable_http",
        "http": {"url": server_url},
        "tools": tools,
        "timeout_s": 20,
        "max_request_bytes": 1_048_576,
        "max_response_bytes": 4_194_304,
        "protocol_mode": MCP_PROTOCOL_REVISION,
    }


def _context_tool_calls(scenario: str) -> list[dict[str, Any]]:
    raw = os.environ.get("MCP_CONFORMANCE_CONTEXT")
    if raw is None:
        raise ConformanceGateError(f"{scenario}: MCP_CONFORMANCE_CONTEXT is missing")
    try:
        context = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConformanceGateError(
            f"{scenario}: MCP_CONFORMANCE_CONTEXT is not JSON"
        ) from exc
    if not isinstance(context, dict) or context.get("name") != scenario:
        raise ConformanceGateError(f"{scenario}: conformance context name mismatch")
    calls = context.get("toolCalls")
    if not isinstance(calls, list) or not calls:
        raise ConformanceGateError(f"{scenario}: conformance toolCalls are missing")
    selected: list[dict[str, Any]] = []
    for item in calls:
        if not isinstance(item, dict):
            raise ConformanceGateError(f"{scenario}: invalid toolCalls item")
        name = item.get("name")
        arguments = item.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ConformanceGateError(f"{scenario}: invalid toolCalls shape")
        selected.append({"name": name, "arguments": arguments})
    return selected


def _live_schema(listing: Mapping[str, Any], mcp_name: str) -> dict[str, Any]:
    tools = listing.get("tools")
    if not isinstance(tools, list):
        raise ConformanceGateError("Agent libOS MCP listing returned no tools array")
    for item in tools:
        if not isinstance(item, dict) or item.get("mcp_name") != mcp_name:
            continue
        live = item.get("live")
        schema = live.get("input_schema") if isinstance(live, dict) else None
        if isinstance(schema, dict):
            return schema
    raise ConformanceGateError(f"live MCP schema was not observed for {mcp_name}")


def _call_or_fail(
    runtime: Any,
    pid: str,
    tool_by_name: Mapping[str, _ToolPlan],
    name: str,
    arguments: dict[str, Any],
) -> None:
    tool = tool_by_name.get(name)
    if tool is None:
        raise ConformanceGateError(f"scenario requested unexpected tool {name!r}")
    result = runtime.mcp.call_tool(
        pid,
        _SERVER_ID,
        tool.tool_id,
        arguments,
    )
    if not result.ok:
        status = getattr(result.status, "value", str(result.status))
        raise ConformanceGateError(f"Agent libOS MCP call failed with status={status}")


def run_client_adapter(server_url: str) -> int:
    from agent_libos import Runtime
    from agent_libos.models import CapabilityRight

    scenario = os.environ.get("MCP_CONFORMANCE_SCENARIO", "")
    protocol = os.environ.get("MCP_CONFORMANCE_PROTOCOL_VERSION", "")
    if scenario not in OFFICIAL_TOOLS_ONLY_SCENARIOS:
        raise ConformanceGateError(f"unallowlisted official scenario: {scenario!r}")
    if protocol != MCP_PROTOCOL_REVISION:
        raise ConformanceGateError(
            f"official runner requested protocol {protocol!r}, "
            f"expected {MCP_PROTOCOL_REVISION}"
        )
    if not (server_url.startswith("http://localhost:") or server_url.startswith("http://127.0.0.1:")):
        raise ConformanceGateError("official scenario server must be loopback HTTP")

    runtime = Runtime.open(":memory:")
    try:
        runtime.mcp.register_server(
            _manifest(server_url, scenario),
            actor="mcp-conformance",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal=f"official MCP conformance {scenario}",
        )
        tool_by_name = {
            item.mcp_name: item for item in _SCENARIO_TOOLS[scenario]
        }
        for tool in _SCENARIO_TOOLS[scenario]:
            runtime.capability.grant(
                pid,
                f"mcp:{_SERVER_ID}:{tool.tool_id}",
                [CapabilityRight.READ],
                issued_by="mcp-conformance",
            )

        if scenario == "tools_call":
            _call_or_fail(
                runtime,
                pid,
                tool_by_name,
                "add_numbers",
                {"a": 2, "b": 3},
            )
        elif scenario == "request-metadata":
            runtime.mcp.discover(
                _SERVER_ID,
                actor=None,
                require_capability=False,
            )
        elif scenario == "http-standard-headers":
            _call_or_fail(
                runtime,
                pid,
                tool_by_name,
                "test_headers",
                {},
            )
        elif scenario == "http-custom-headers":
            for call in _context_tool_calls(scenario):
                _call_or_fail(
                    runtime,
                    pid,
                    tool_by_name,
                    call["name"],
                    call["arguments"],
                )
        elif scenario == "http-invalid-tool-headers":
            _call_or_fail(
                runtime,
                pid,
                tool_by_name,
                "valid_tool",
                {"region": "us-west1"},
            )
        elif scenario == "json-schema-ref-no-deref":
            runtime.mcp.list_tools(
                _SERVER_ID,
                actor=None,
                require_capability=False,
                refresh=True,
            )
        elif scenario == "json-schema-2020-12-preservation":
            listing = runtime.mcp.list_tools(
                _SERVER_ID,
                actor=None,
                require_capability=False,
                refresh=True,
            )
            schema = _live_schema(listing, "json_schema_2020_12_tool")
            _call_or_fail(
                runtime,
                pid,
                tool_by_name,
                "json_schema_echo",
                {"schema": schema},
            )
        else:  # pragma: no cover - map and dispatch are tested together
            raise ConformanceGateError(f"no adapter for scenario {scenario}")
    finally:
        runtime.close()
    print(json.dumps({"scenario": scenario, "status": "completed"}))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Agent libOS against the pinned official MCP 2026-07-28 "
            "Tools-only client conformance scenarios."
        )
    )
    parser.add_argument(
        "--official-checkout",
        help="reuse an existing checkout at the exact pinned official commit",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="reuse dist/index.js in --official-checkout after commit verification",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(OFFICIAL_TOOLS_ONLY_SCENARIOS),
        help="run a selected allowlisted scenario (repeatable; default: all)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30_000,
        help="official per-client timeout in milliseconds (default: 30000)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPOSITORY_ROOT / ".benchmark_runs" / "mcp-conformance"),
        help="untracked parent directory for official result artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    selected = list(sys.argv[1:] if argv is None else argv)
    try:
        if selected and selected[0] == "_client":
            if len(selected) != 2:
                raise ConformanceGateError("_client requires exactly one server URL")
            return run_client_adapter(selected[1])
        args = _parser().parse_args(selected)
        if args.skip_prepare and args.official_checkout is None:
            raise ConformanceGateError(
                "--skip-prepare requires --official-checkout"
            )
        return run_gate(args)
    except ConformanceGateError as exc:
        print(f"MCP conformance gate failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
