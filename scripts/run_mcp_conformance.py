#!/usr/bin/env python3
"""Run the official MCP 2026-07-28 client conformance gate.

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
import hashlib
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
_MAX_OFFICIAL_CHECKS_BYTES = 2 * 1_024 * 1_024
_MAX_OFFICIAL_CHECK_COUNT = 4_096


class ConformanceGateError(RuntimeError):
    """A stable, user-actionable conformance gate failure."""


@dataclass(frozen=True)
class ScenarioContract:
    """Checks that prove an official client scenario was actually exercised."""

    required_success_ids: frozenset[str]
    required_present_ids: frozenset[str] = frozenset()
    required_skipped_ids: frozenset[str] = frozenset()
    required_success_names: frozenset[str] = frozenset()
    required_skipped_names: frozenset[str] = frozenset()
    expected_success_count: int | None = None
    expected_skipped_count: int | None = None


# This is deliberately not an upstream suite name.  Every entry is reviewed at
# the pinned commit and has explicit proof obligations below; newly added
# upstream scenarios cannot silently enter a release gate.
OFFICIAL_CLIENT_SCENARIOS: Mapping[str, ScenarioContract] = {
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
                "sep-2575-client-declares-elicitation-capability",
            }
        ),
        required_present_ids=frozenset(
            {
                "sep-2575-client-declares-roots-capability",
                "sep-2575-client-declares-sampling-capability",
                "sep-2575-client-declares-elicitation-capability",
            }
        ),
        # Exact-v3 advertises only governed Elicitation. Roots and Sampling
        # remain intentionally unsupported and are reported as SKIPPED.
        required_skipped_ids=frozenset(
            {
                "sep-2575-client-declares-roots-capability",
                "sep-2575-client-declares-sampling-capability",
            }
        ),
    ),
    "http-standard-headers": ScenarioContract(
        required_success_ids=frozenset(
            {"sep-2243-client-includes-standard-headers"}
        ),
        # Exact-v3 negotiates through server/discover and dispatches a
        # Host-allowlisted Tool without the legacy initialize/initialized and
        # standalone tools/list paths.  Resources and Prompts are implemented,
        # so every one of their method/name headers must be observed rather
        # than silently remaining in the fixture's SKIPPED projection.
        required_success_names=frozenset(
            {
                "ClientMcpMethodHeader_tools_call",
                "ClientMcpNameHeader_tools_call",
                "ClientMcpMethodHeader_resources_list",
                "ClientMcpMethodHeader_resources_read",
                "ClientMcpNameHeader_resources_read",
                "ClientMcpMethodHeader_prompts_list",
                "ClientMcpMethodHeader_prompts_get",
                "ClientMcpNameHeader_prompts_get",
            }
        ),
        required_skipped_names=frozenset(
            {
                "ClientMcpMethodHeader_initialize",
                "ClientMcpMethodHeader_notifications_initialized",
                "ClientMcpMethodHeader_tools_list",
            }
        ),
        expected_success_count=8,
        expected_skipped_count=3,
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
    "sep-2322-client-request-state": ScenarioContract(
        required_success_ids=frozenset(
            {
                "sep-2322-client-request-state-echoed",
                "sep-2322-client-jsonrpc-id-different",
                "sep-2322-client-no-state-omitted",
                "sep-2322-client-parallel-isolation",
                "sep-2322-default-result-type-complete",
            }
        ),
    ),
    "auth/pre-registration": ScenarioContract(
        required_success_ids=frozenset(
            {
                "prm-pathbased-requested",
                "authorization-server-metadata",
                "authorization-request",
                "pkce-code-challenge-sent",
                "pkce-s256-method-used",
                "token-request",
                "pkce-code-verifier-sent",
                "pkce-verifier-matches-challenge",
                "pre-registration-auth",
                "valid-bearer-token",
            }
        ),
    ),
    "auth/basic-cimd": ScenarioContract(
        required_success_ids=frozenset(
            {
                "prm-pathbased-requested",
                "authorization-server-metadata",
                "authorization-request",
                "pkce-code-challenge-sent",
                "pkce-s256-method-used",
                "token-request",
                "pkce-code-verifier-sent",
                "pkce-verifier-matches-challenge",
                "cimd-client-id-used",
                "valid-bearer-token",
            }
        ),
    ),
}

OFFICIAL_OAUTH_SCENARIOS = frozenset(
    {"auth/pre-registration", "auth/basic-cimd"}
)
OFFICIAL_TOOL_SCENARIOS = frozenset(
    scenario
    for scenario in OFFICIAL_CLIENT_SCENARIOS
    if scenario != "sep-2322-client-request-state"
    and scenario not in OFFICIAL_OAUTH_SCENARIOS
)
OFFICIAL_MRTR_SCENARIO = "sep-2322-client-request-state"

# These are the remaining authorization-code core/draft scenarios in the
# immutable upstream commit.  The fixed-upstream pre-registration and CIMD
# scenarios run through a separate checked harness which obtains both fixture
# origins before Runtime construction.  The scenarios below expose only an
# untrusted Resource Server URL through the generic runner and start the
# Authorization Server on another random origin.  Treating Protected Resource
# Metadata or a 401 as permission to trust that issuer would turn discovery
# into authority.  They are reported as reviewed-but-unavailable, never xfailed
# or represented as official passes.  The separate real TLS gate covers the
# production pinning path.  Optional extension and older-spec scenarios are
# inventoried separately as reviewed product exclusions.
OFFICIAL_OAUTH_SCENARIOS_REVIEWED_BUT_NOT_RUNNABLE = (
    "auth/metadata-default",
    "auth/metadata-var1",
    "auth/metadata-var2",
    "auth/metadata-var3",
    "auth/scope-from-www-authenticate",
    "auth/scope-from-scopes-supported",
    "auth/scope-omitted-when-undefined",
    "auth/scope-step-up",
    "auth/scope-retry-limit",
    "auth/token-endpoint-auth-basic",
    "auth/token-endpoint-auth-post",
    "auth/token-endpoint-auth-none",
    "auth/resource-mismatch",
    "auth/offline-access-scope",
    "auth/offline-access-not-supported",
    "auth/authorization-server-migration",
    "auth/iss-supported",
    "auth/iss-not-advertised",
    "auth/iss-supported-missing",
    "auth/iss-wrong-issuer",
    "auth/iss-unexpected",
    "auth/iss-normalized",
    "auth/metadata-issuer-mismatch",
)
OFFICIAL_OAUTH_SCENARIOS_REVIEWED_OUT_OF_SCOPE = (
    "auth/2025-03-26-oauth-metadata-backcompat",
    "auth/2025-03-26-oauth-endpoint-fallback",
    "auth/client-credentials-jwt",
    "auth/client-credentials-basic",
    "auth/enterprise-managed-authorization",
    "auth/dpop",
    "auth/dpop-nonce",
    "auth/wif-jwt-bearer",
)
OFFICIAL_OAUTH_AUTHORITY_GAP_CODE = "runner_omits_host_pinned_expected_issuer"
OFFICIAL_OAUTH_AUTHORITY_GAP = (
    "the pinned official client runner provides only an untrusted resource "
    "server URL and no Host-reviewed expected issuer; Agent libOS will not "
    "convert remote discovery into authority"
)
OFFICIAL_OAUTH_OUT_OF_SCOPE_REASON_CODE = "unsupported_oauth_extensions_or_backcompat"
OFFICIAL_OAUTH_OUT_OF_SCOPE_REASON = (
    "Agent libOS v3 OAuth is a Host-preconfigured authorization-code client; "
    "the reviewed 2025-03-26 backcompat, client-credentials, enterprise-managed "
    "authorization, DPoP, and workload-identity scenarios are not product claims"
)
RUNTIME_OAUTH_TLS_REGRESSION_NODE = (
    "tests/providers/test_mcp_oauth_runtime_tls.py::"
    "test_runtime_oauth_pkce_tls_and_bearer_transport_end_to_end"
)


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
    reveal_failure_output: bool = True,
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
        suffix = f"\n{output}" if reveal_failure_output else "; output omitted"
        raise ConformanceGateError(
            f"command failed ({completed.returncode}): {command}{suffix}"
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
    path = paths[0]
    if path.is_symlink():
        raise ConformanceGateError("official checks.json must not be a symlink")
    try:
        if path.stat().st_size > _MAX_OFFICIAL_CHECKS_BYTES:
            raise ConformanceGateError("official checks.json exceeds the evidence bound")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConformanceGateError("official checks.json is unreadable") from exc
    if not isinstance(value, list) or not value:
        raise ConformanceGateError("official scenario returned an empty check set")
    if len(value) > _MAX_OFFICIAL_CHECK_COUNT:
        raise ConformanceGateError("official scenario returned too many checks")
    if not all(isinstance(item, dict) for item in value):
        raise ConformanceGateError("official checks.json must contain only objects")
    return value


def _validate_checks(
    scenario: str,
    checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = OFFICIAL_CLIENT_SCENARIOS[scenario]
    statuses_by_id: dict[str, set[str]] = {}
    statuses_by_name: dict[str, set[str]] = {}
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
        check_name = check.get("name")
        if isinstance(check_name, str):
            statuses_by_name.setdefault(check_name, set()).add(status)
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
    missing_success_names = sorted(
        check_name
        for check_name in contract.required_success_names
        if "SUCCESS" not in statuses_by_name.get(check_name, set())
    )
    missing_skipped_names = sorted(
        check_name
        for check_name in contract.required_skipped_names
        if "SKIPPED" not in statuses_by_name.get(check_name, set())
    )
    success_count = sum(check.get("status") == "SUCCESS" for check in checks)
    skipped_count = sum(check.get("status") == "SKIPPED" for check in checks)
    unexpected_counts: list[str] = []
    if (
        contract.expected_success_count is not None
        and success_count != contract.expected_success_count
    ):
        unexpected_counts.append(
            f"success_count={success_count} expected={contract.expected_success_count}"
        )
    if (
        contract.expected_skipped_count is not None
        and skipped_count != contract.expected_skipped_count
    ):
        unexpected_counts.append(
            f"skipped_count={skipped_count} expected={contract.expected_skipped_count}"
        )
    if (
        missing_success
        or missing_present
        or missing_skipped
        or missing_success_names
        or missing_skipped_names
        or unexpected_counts
    ):
        raise ConformanceGateError(
            f"{scenario}: incomplete official evidence "
            f"success={missing_success}, present={missing_present}, "
            f"skipped={missing_skipped}, success_names={missing_success_names}, "
            f"skipped_names={missing_skipped_names}, counts={unexpected_counts}"
        )
    if "wire-schema-harness-error" in statuses_by_id:
        raise ConformanceGateError(
            f"{scenario}: official harness reported a wire-schema harness error"
        )
    return {
        "scenario": scenario,
        "check_count": len(checks),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "check_ids": sorted(statuses_by_id),
    }


def _project_durable_check(check: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only bounded, non-transient upstream conformance evidence."""

    check_id = check.get("id")
    status = check.get("status")
    if not isinstance(check_id, str) or not isinstance(status, str):
        raise ConformanceGateError(
            "official check cannot be projected without string id/status"
        )
    if len(check_id.encode("utf-8")) > 512 or len(status) > 32:
        raise ConformanceGateError("official check id/status exceeds evidence bounds")
    projected: dict[str, Any] = {"id": check_id, "status": status}
    raw_references = check.get("specReferences")
    if raw_references is None:
        return projected
    if not isinstance(raw_references, list) or len(raw_references) > 32:
        raise ConformanceGateError("official check spec references are invalid")
    references: list[dict[str, str]] = []
    for raw_reference in raw_references:
        if not isinstance(raw_reference, dict):
            raise ConformanceGateError("official check spec reference is invalid")
        reference: dict[str, str] = {}
        for key, limit in (("id", 512), ("url", 2_048)):
            value = raw_reference.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or len(value.encode("utf-8")) > limit:
                raise ConformanceGateError(
                    f"official check spec reference {key} is invalid"
                )
            reference[key] = value
        if not reference:
            raise ConformanceGateError("official check spec reference is empty")
        references.append(reference)
    projected["specReferences"] = references
    return projected


def _persist_durable_scenario_evidence(
    result_root: Path,
    checks: Sequence[Mapping[str, Any]],
) -> str:
    """Replace raw runner artifacts with a deterministic safe projection."""

    projected = [_project_durable_check(check) for check in checks]
    projected.sort(
        key=lambda check: json.dumps(
            check,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    canonical = json.dumps(
        projected,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    evidence_sha256 = hashlib.sha256(canonical).hexdigest()
    paths = sorted(result_root.rglob("checks.json"))
    target = paths[0] if len(paths) == 1 else result_root / "checks.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(projected, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for artifact in sorted(result_root.rglob("*")):
        if artifact.is_file() and artifact != target:
            artifact.unlink()
    return evidence_sha256


def _discard_raw_scenario_evidence(result_root: Path) -> None:
    """Remove unreviewed raw upstream output when a scenario fails closed."""

    for artifact in sorted(result_root.rglob("*")):
        if artifact.is_file():
            artifact.unlink()


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
    try:
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
            reveal_failure_output=False,
        )
        checks = _load_checks(scenario_output)
        summary = _validate_checks(scenario, checks)
        summary["evidence_sha256"] = _persist_durable_scenario_evidence(
            scenario_output,
            checks,
        )
        return summary
    except Exception:
        _discard_raw_scenario_evidence(scenario_output)
        raise


def _run_fixed_upstream_oauth_scenario(
    *,
    node: str,
    checkout: Path,
    scenario: str,
    output_root: Path,
    timeout_ms: int,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Run one pinned OAuth fixture with Host pins supplied out of band."""

    if scenario not in OFFICIAL_OAUTH_SCENARIOS:
        raise ConformanceGateError(f"unreviewed fixed-upstream OAuth scenario: {scenario}")
    tsx_loader = checkout / "node_modules" / "tsx" / "dist" / "loader.mjs"
    if not tsx_loader.is_file():
        raise ConformanceGateError(
            "fixed-upstream OAuth harness requires the pinned tsx dependency"
        )
    scenario_output = output_root / scenario
    scenario_output.mkdir(parents=True, exist_ok=False)
    helper_output = scenario_output / "fixed-upstream"
    try:
        _run_checked(
            (
                node,
                "--import",
                str(tsx_loader),
                str(REPOSITORY_ROOT / "scripts" / "mcp_conformance_oauth_harness.mts"),
                str(checkout),
                scenario,
                os.path.abspath(sys.executable),
                str(Path(__file__).resolve()),
                str(timeout_ms),
                str(helper_output),
            ),
            cwd=checkout,
            env=env,
            reveal_failure_output=False,
        )
        checks = _load_checks(scenario_output)
        summary = _validate_checks(scenario, checks)
        summary["evidence_sha256"] = _persist_durable_scenario_evidence(
            scenario_output,
            checks,
        )
        return summary
    except Exception:
        _discard_raw_scenario_evidence(scenario_output)
        raise


def run_gate(args: argparse.Namespace) -> int:
    selected = tuple(args.scenario or OFFICIAL_CLIENT_SCENARIOS)
    unknown = sorted(set(selected) - set(OFFICIAL_CLIENT_SCENARIOS))
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
                (
                    _run_fixed_upstream_oauth_scenario(
                        node=node,
                        checkout=checkout,
                        scenario=scenario,
                        output_root=output_root,
                        timeout_ms=args.timeout_ms,
                        env=environment,
                    )
                    if scenario in OFFICIAL_OAUTH_SCENARIOS
                    else _run_official_scenario(
                        node=node,
                        runner=runner,
                        checkout=checkout,
                        scenario=scenario,
                        output_root=output_root,
                        timeout_ms=args.timeout_ms,
                        env=environment,
                    )
                )
                for scenario in selected
            ]

    summary = {
        "gate": "official-mcp-client",
        "official_repository": OFFICIAL_CONFORMANCE_REPOSITORY,
        "official_commit": OFFICIAL_CONFORMANCE_COMMIT,
        "official_package_version": OFFICIAL_CONFORMANCE_PACKAGE_VERSION,
        "protocol_revision": MCP_PROTOCOL_REVISION,
        "complete_allowlist": set(selected) == set(OFFICIAL_CLIENT_SCENARIOS),
        "oauth_review": {
            "status": "pinned_preregistration_and_cimd_run_remaining_unavailable",
            "reason_code": OFFICIAL_OAUTH_AUTHORITY_GAP_CODE,
            "reason": OFFICIAL_OAUTH_AUTHORITY_GAP,
            "official_scenarios_run": sorted(
                set(selected) & OFFICIAL_OAUTH_SCENARIOS
            ),
            "official_scenarios_reviewed": list(
                OFFICIAL_OAUTH_SCENARIOS_REVIEWED_BUT_NOT_RUNNABLE
            ),
            "official_scenarios_out_of_scope": list(
                OFFICIAL_OAUTH_SCENARIOS_REVIEWED_OUT_OF_SCOPE
            ),
            "out_of_scope_reason_code": OFFICIAL_OAUTH_OUT_OF_SCOPE_REASON_CODE,
            "out_of_scope_reason": OFFICIAL_OAUTH_OUT_OF_SCOPE_REASON,
            "runtime_tls_regression_node": RUNTIME_OAUTH_TLS_REGRESSION_NODE,
        },
        "scenarios": results,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"official MCP client conformance passed: "
        f"{len(results)}/{len(selected)} scenarios; artifacts={output_root}"
    )
    return 0


def _manifest(server_url: str, scenario: str) -> dict[str, Any]:
    modern_surface = scenario in {"request-metadata", "http-standard-headers"}
    tools = []
    for tool in _SCENARIO_TOOLS[scenario]:
        tools.append(
            {
                "tool_id": tool.tool_id,
                "mcp_name": tool.mcp_name,
                "right": "read",
                "rollback_class": "no_rollback_required",
                **(
                    {"rollback_status": "not_required"}
                    if scenario == "request-metadata"
                    else {}
                ),
                "state_mutation": False,
                "information_flow": True,
                # The official scenario server is the fixture source.  Empty
                # Host schemas keep authority fixed while the live schema is
                # exercised as untrusted provider metadata in each operation.
                "input_schema": {},
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 3 if modern_surface else 2,
        "server_id": _SERVER_ID,
        "transport": "streamable_http",
        "http": {"url": server_url},
        "tools": tools,
        "timeout_s": 20,
        "max_request_bytes": 1_048_576,
        "max_response_bytes": 4_194_304,
        "protocol_mode": MCP_PROTOCOL_REVISION,
    }
    if scenario == "http-standard-headers":
        # The pinned SEP-2243 fixture has one reviewed Resource and Prompt
        # selector for exercising Mcp-Method/Mcp-Name.  They are explicit Host
        # allowlist entries; live discovery cannot add any other authority.
        manifest.update(
            {
                "resources": [
                    {
                        "resource_id": "header-resource",
                        "remote_uri": "file:///path/to/file%20name.txt",
                        "right": "read",
                        "information_flow": True,
                        "model_visible": False,
                        "mime_types": ["text/plain"],
                    }
                ],
                "prompts": [
                    {
                        "prompt_id": "header-prompt",
                        "mcp_name": "test_prompt",
                        "argument_names": [],
                    }
                ],
                "subscriptions": [],
            }
        )
    return manifest


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
    if scenario not in OFFICIAL_CLIENT_SCENARIOS:
        raise ConformanceGateError(f"unallowlisted official scenario: {scenario!r}")
    if protocol != MCP_PROTOCOL_REVISION:
        raise ConformanceGateError(
            f"official runner requested protocol {protocol!r}, "
            f"expected {MCP_PROTOCOL_REVISION}"
        )
    if not (server_url.startswith("http://localhost:") or server_url.startswith("http://127.0.0.1:")):
        raise ConformanceGateError("official scenario server must be loopback HTTP")

    if scenario in OFFICIAL_OAUTH_SCENARIOS:
        return _run_oauth_runtime_client_adapter(server_url, scenario)
    if scenario == OFFICIAL_MRTR_SCENARIO:
        return _run_mrtr_runtime_client_adapter(server_url, scenario)

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
        if scenario in {"request-metadata", "http-standard-headers"}:
            runtime.capability.grant(
                pid,
                f"mcp_server:{_SERVER_ID}",
                [CapabilityRight.EXECUTE],
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
            from agent_libos.mcp.types import McpComplete

            metadata_result = runtime.mcp.call_tool(
                pid,
                _SERVER_ID,
                "unused",
                {},
            )
            if not isinstance(metadata_result, McpComplete):
                raise ConformanceGateError(
                    "request-metadata v3 Tool did not complete"
                )
        elif scenario == "http-standard-headers":
            from agent_libos.mcp.types import McpComplete

            tool_result = runtime.mcp.call_tool(
                pid,
                _SERVER_ID,
                "test_headers",
                {},
            )
            if not isinstance(tool_result, McpComplete):
                raise ConformanceGateError(
                    "standard-header scenario Tool did not complete"
                )
            resources = runtime.mcp.list_resources(_SERVER_ID)
            if [item.resource_id for item in resources.items] != [
                "header-resource"
            ]:
                raise ConformanceGateError(
                    "standard-header scenario did not observe its Resource"
                )
            resource_result = runtime.mcp.read_resource(
                _SERVER_ID,
                "header-resource",
            )
            if not isinstance(resource_result, McpComplete):
                raise ConformanceGateError(
                    "standard-header scenario Resource did not complete"
                )
            prompts = runtime.mcp.list_prompts(_SERVER_ID)
            if [item.prompt_id for item in prompts.items] != ["header-prompt"]:
                raise ConformanceGateError(
                    "standard-header scenario did not observe its Prompt"
                )
            prompt_result = runtime.mcp.get_prompt(
                _SERVER_ID,
                "header-prompt",
                arguments={},
            )
            if not isinstance(prompt_result, McpComplete):
                raise ConformanceGateError(
                    "standard-header scenario Prompt did not complete"
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


def _run_oauth_runtime_client_adapter(server_url: str, scenario: str) -> int:
    """Drive the pinned pre-registration fixture through Runtime-owned OAuth."""

    import http.client
    from dataclasses import replace
    from urllib.parse import urlsplit

    import anyio

    from agent_libos import Runtime
    from agent_libos.config import DEFAULT_CONFIG
    from agent_libos.mcp import (
        InMemoryMcpCredentialBroker,
        McpOAuthProfile,
        McpOAuthRegistrationMode,
        McpOAuthStatusKind,
        McpOAuthTokenEndpointAuthMethod,
        PinnedMcpOAuthHttpTransport,
    )
    from agent_libos.mcp.types import McpComplete
    from agent_libos.models import CapabilityRight
    from agent_libos.substrate import LocalResourceProviderSubstrate

    raw_context = os.environ.get("MCP_CONFORMANCE_CONTEXT", "")
    try:
        context = json.loads(raw_context)
    except json.JSONDecodeError as exc:
        raise ConformanceGateError(
            f"{scenario}: fixed-upstream OAuth context is invalid"
        ) from exc
    expected_context_keys = {
        "name",
        "client_id",
        "client_secret",
        "registration_mode",
        "token_endpoint_auth_method",
        "trusted_resource_url",
        "trusted_issuer",
        "trusted_prm_url",
        "trusted_as_metadata_url",
    }
    if not isinstance(context, dict) or set(context) != expected_context_keys:
        raise ConformanceGateError(
            f"{scenario}: fixed-upstream OAuth context shape is invalid"
        )
    if context.get("name") != scenario or context.get("trusted_resource_url") != server_url:
        raise ConformanceGateError(
            f"{scenario}: fixed-upstream OAuth resource binding changed"
        )
    for field in expected_context_keys - {"name", "client_secret"}:
        if not isinstance(context.get(field), str) or not context[field]:
            raise ConformanceGateError(
                f"{scenario}: fixed-upstream OAuth {field} is invalid"
            )
    if context["registration_mode"] == "preregistered":
        if (
            context["token_endpoint_auth_method"] != "client_secret_basic"
            or not isinstance(context.get("client_secret"), str)
            or not context["client_secret"]
        ):
            raise ConformanceGateError(
                f"{scenario}: pre-registration credentials are invalid"
            )
    elif context["registration_mode"] == "cimd":
        if (
            context["token_endpoint_auth_method"] != "none"
            or context.get("client_secret") is not None
            or not str(context["client_id"]).startswith("https://")
        ):
            raise ConformanceGateError(f"{scenario}: CIMD configuration is invalid")
    else:
        raise ConformanceGateError(
            f"{scenario}: unsupported OAuth registration mode"
        )

    resource = urlsplit(server_url)
    issuer = urlsplit(context["trusted_issuer"])
    prm = urlsplit(context["trusted_prm_url"])
    metadata = urlsplit(context["trusted_as_metadata_url"])
    if (
        resource.scheme != "http"
        or resource.hostname != "localhost"
        or resource.port is None
        or resource.path != "/mcp"
        or resource.query
        or resource.fragment
    ):
        raise ConformanceGateError(
            f"{scenario}: fixed-upstream OAuth resource URL is not pinned loopback"
        )
    if (
        issuer.scheme != "http"
        or issuer.hostname != "localhost"
        or issuer.port is None
        or issuer.path not in {"", "/"}
        or issuer.query
        or issuer.fragment
    ):
        raise ConformanceGateError(
            f"{scenario}: fixed-upstream OAuth issuer is not pinned loopback"
        )
    resource_origin = f"http://localhost:{resource.port}"
    issuer_origin = f"http://localhost:{issuer.port}"
    if resource_origin == issuer_origin:
        raise ConformanceGateError(
            f"{scenario}: fixed-upstream OAuth origins were not separated"
        )
    if (
        context["trusted_prm_url"]
        != f"{resource_origin}/.well-known/oauth-protected-resource/mcp"
        or context["trusted_as_metadata_url"]
        != f"{issuer_origin}/.well-known/oauth-authorization-server"
        or prm.hostname != "localhost"
        or metadata.hostname != "localhost"
    ):
        raise ConformanceGateError(
            f"{scenario}: fixed-upstream OAuth metadata pins changed"
        )

    pinned_ports = {resource.port, issuer.port}

    def resolve_pinned(host: str, port: int, _deadline: float) -> tuple[str, ...]:
        if host != "localhost" or port not in pinned_ports:
            raise ConformanceGateError(
                f"{scenario}: OAuth transport attempted an unpinned endpoint"
            )
        return ("127.0.0.1",)

    broker = InMemoryMcpCredentialBroker()
    temporary = tempfile.TemporaryDirectory(
        prefix="agent-libos-mcp-conformance-oauth-runtime-"
    )
    substrate = LocalResourceProviderSubstrate(Path(temporary.name))
    substrate.mcp_credential_broker = broker
    substrate.mcp_oauth_transport = PinnedMcpOAuthHttpTransport(
        resolver=resolve_pinned,
        allow_loopback_http=True,
    )
    runtime = Runtime.open(
        ":memory:",
        substrate=substrate,
        config=replace(
            DEFAULT_CONFIG,
            mcp=replace(DEFAULT_CONFIG.mcp, oauth_enabled=True),
        ),
    )
    client_secret = context.get("client_secret")
    try:
        profile = McpOAuthProfile(
            profile_id="official-pre-registration",
            server_id=_SERVER_ID,
            resource_uri=server_url,
            expected_issuer=issuer_origin,
            redirect_uri="http://127.0.0.1:8765/callback",
            client_id=context["client_id"],
            registration_mode=McpOAuthRegistrationMode(
                context["registration_mode"]
            ),
            token_endpoint_auth_method=McpOAuthTokenEndpointAuthMethod(
                context["token_endpoint_auth_method"]
            ),
            protected_resource_metadata_url=context["trusted_prm_url"],
            authorization_server_metadata_url=context[
                "trusted_as_metadata_url"
            ],
            allowed_endpoint_origins=(issuer_origin,),
            allow_loopback_http=True,
        )
        profile_kwargs: dict[str, Any] = {"actor": "mcp-conformance-host"}
        if isinstance(client_secret, str):
            profile_kwargs["client_secret"] = client_secret.encode("utf-8")
        provisional = runtime.mcp.add_oauth_profile(profile, **profile_kwargs)
        if provisional.status is not McpOAuthStatusKind.AUTHORIZATION_REQUIRED:
            raise ConformanceGateError(
                f"{scenario}: OAuth profile was not authorization-required"
            )
        runtime.mcp.register_server(
            {
                "schema_version": 3,
                "server_id": _SERVER_ID,
                "transport": "streamable_http",
                "protocol_mode": MCP_PROTOCOL_REVISION,
                "http": {"url": server_url},
                "tools": [
                    {
                        "tool_id": "test",
                        "mcp_name": "test-tool",
                        "right": "read",
                        "rollback_class": "no_rollback_required",
                        "rollback_status": "not_required",
                        "state_mutation": False,
                        "information_flow": True,
                        "input_schema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    }
                ],
                "auth_profile_id": profile.profile_id,
                "subscriptions": [],
                "timeout_s": 10,
                "max_request_bytes": 65_536,
                "max_response_bytes": 1_048_576,
            },
            actor="mcp-conformance-host",
            require_capability=False,
        )
        challenge = runtime.mcp.auth_begin(
            profile.profile_id,
            actor="mcp-conformance-host",
        )
        authorization = urlsplit(challenge.authorization_url)
        if (
            authorization.scheme != "http"
            or authorization.hostname != "localhost"
            or authorization.port != issuer.port
        ):
            raise ConformanceGateError(
                f"{scenario}: authorization challenge escaped the pinned issuer"
            )
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            authorization.port,
            timeout=5,
        )
        try:
            target = authorization.path + (
                f"?{authorization.query}" if authorization.query else ""
            )
            connection.request(
                "GET",
                target,
                headers={"Accept": "application/json", "Host": authorization.netloc},
            )
            response = connection.getresponse()
            response.read()
            callback = response.getheader("Location")
        finally:
            connection.close()
        if response.status != 302 or not callback:
            raise ConformanceGateError(
                f"{scenario}: authorization fixture did not return a callback"
            )
        authorized = runtime.mcp.auth_complete(
            challenge.challenge_id,
            callback,
            actor="mcp-conformance-host",
        )
        if authorized.status is not McpOAuthStatusKind.AUTHORIZED:
            raise ConformanceGateError(
                f"{scenario}: OAuth token exchange did not authorize"
            )

        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal=f"official MCP conformance {scenario}",
        )
        runtime.capability.grant(
            pid,
            f"mcp:{_SERVER_ID}:test",
            [CapabilityRight.READ],
            issued_by="mcp-conformance-host",
        )
        runtime.capability.grant(
            pid,
            f"mcp_server:{_SERVER_ID}",
            [CapabilityRight.EXECUTE],
            issued_by="mcp-conformance-host",
        )
        result = runtime.mcp.call_tool(pid, _SERVER_ID, "test", {})
        if not isinstance(result, McpComplete) or "test" not in repr(result):
            raise ConformanceGateError(
                f"{scenario}: authorized MCP Tool call failed"
            )
        public_evidence = repr(
            (
                runtime.audit.trace(),
                runtime.events.list(),
                runtime.store.list_external_effects(),
                runtime.uow.mcp_auth.get(profile.profile_id),
            )
        )
        if isinstance(client_secret, str) and client_secret in public_evidence:
            raise ConformanceGateError(
                f"{scenario}: pre-registration secret entered public evidence"
            )
        if anyio.run(runtime._mcp_connection_supervisor.snapshot):
            raise ConformanceGateError(
                f"{scenario}: OAuth conformance leaked a supervised connection"
            )
    finally:
        runtime.close()
        broker.close()
        temporary.cleanup()
    print(json.dumps({"scenario": scenario, "status": "completed"}))
    return 0


def _run_mrtr_runtime_client_adapter(server_url: str, scenario: str) -> int:
    """Exercise official MRTR through Runtime v3, Human, Store, and Provider."""

    import anyio

    from agent_libos import Runtime
    from agent_libos.mcp.oauth import InMemoryMcpCredentialBroker
    from agent_libos.mcp.types import McpComplete, McpInputRequired
    from agent_libos.models import CapabilityRight
    from agent_libos.substrate import LocalResourceProviderSubstrate

    tool_names = (
        "test_mrtr_echo_state",
        "test_mrtr_no_state",
        "test_mrtr_unrelated",
        "test_mrtr_no_result_type",
    )
    temporary = tempfile.TemporaryDirectory(prefix="agent-libos-mcp-mrtr-runtime-")
    broker = InMemoryMcpCredentialBroker()
    substrate = LocalResourceProviderSubstrate(Path(temporary.name))
    substrate.mcp_credential_broker = broker
    runtime = Runtime.open(":memory:", substrate=substrate)
    try:
        runtime.mcp.register_server(
            {
                "schema_version": 3,
                "server_id": _SERVER_ID,
                "transport": "streamable_http",
                "http": {"url": server_url},
                "protocol_mode": MCP_PROTOCOL_REVISION,
                "tools": [
                    {
                        "tool_id": name,
                        "mcp_name": name,
                        "right": "read",
                        "rollback_class": "no_rollback_required",
                        "rollback_status": "not_required",
                        "state_mutation": False,
                        "information_flow": True,
                        "input_schema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    }
                    for name in tool_names
                ],
                "timeout_s": 20,
                "max_request_bytes": 1_048_576,
                "max_response_bytes": 4_194_304,
            },
            actor="mcp-conformance",
            require_capability=False,
        )
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="official MCP MRTR conformance through Runtime v3",
        )
        for name in tool_names:
            runtime.capability.grant(
                pid,
                f"mcp:{_SERVER_ID}:{name}",
                [CapabilityRight.READ],
                issued_by="mcp-conformance",
            )
        for resource, rights in (
            (f"mcp_server:{_SERVER_ID}", [CapabilityRight.EXECUTE]),
            ("human:owner", [CapabilityRight.WRITE]),
        ):
            runtime.capability.grant(
                pid,
                resource,
                rights,
                issued_by="mcp-conformance",
            )

        def initial(tool_name: str) -> McpInputRequired:
            value = runtime.mcp.call_tool(pid, _SERVER_ID, tool_name, {})
            if not isinstance(value, McpInputRequired):
                raise ConformanceGateError(
                    f"{tool_name}: Runtime did not durably capture input_required"
                )
            if (
                len(value.input_requests) != 1
                or value.human_request_id is None
                or value.human_revision is None
                or value.human_preview_sha256 is None
            ):
                raise ConformanceGateError(
                    f"{tool_name}: Runtime returned an incomplete Human fence"
                )
            return value

        def respond(value: McpInputRequired) -> McpComplete[Any]:
            request_id = value.input_requests[0].request_id
            result = runtime.mcp.respond_continuation(
                value.continuation_id,
                expected_revision=value.revision,
                responses={
                    request_id: {
                        "action": "accept",
                        "content": {"confirmed": True},
                    }
                },
                human_request_id=value.human_request_id or "",
                human_expected_revision=value.human_revision or 0,
                human_preview_sha256=value.human_preview_sha256 or "",
                actor="mcp-conformance",
            )
            if not isinstance(result, McpComplete):
                raise ConformanceGateError(
                    "official MRTR continuation did not complete"
                )
            return result

        echo = initial("test_mrtr_echo_state")
        unrelated = runtime.mcp.call_tool(
            pid, _SERVER_ID, "test_mrtr_unrelated", {}
        )
        if not isinstance(unrelated, McpComplete):
            raise ConformanceGateError("unrelated MRTR Tool did not complete")
        respond(echo)
        respond(initial("test_mrtr_no_state"))
        no_type = runtime.mcp.call_tool(
            pid, _SERVER_ID, "test_mrtr_no_result_type", {}
        )
        if not isinstance(no_type, McpComplete):
            raise ConformanceGateError(
                "missing resultType did not default to complete"
            )

        actions = [record.action for record in runtime.audit.trace()]
        if actions.count("primitive.mcp.call") != 4:
            raise ConformanceGateError(
                "official MRTR initial Tools were not dispatched exactly once"
            )
        if actions.count("primitive.mcp.continuation.respond") != 2:
            raise ConformanceGateError(
                "official MRTR retries bypassed the protected continuation facade"
            )
        if anyio.run(runtime._mcp_connection_supervisor.snapshot):
            raise ConformanceGateError("official MRTR leaked a transport session")
    finally:
        runtime.close()
        broker.close()
        temporary.cleanup()
    print(json.dumps({"scenario": scenario, "status": "completed"}))
    return 0
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Agent libOS against the pinned official MCP 2026-07-28 "
            "reviewed client conformance scenarios."
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
        choices=tuple(OFFICIAL_CLIENT_SCENARIOS),
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
