from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts import run_mcp_conformance as gate


_EXPECTED_SCENARIOS = (
    "tools_call",
    "request-metadata",
    "http-standard-headers",
    "http-custom-headers",
    "http-invalid-tool-headers",
    "json-schema-ref-no-deref",
    "json-schema-2020-12-preservation",
    "sep-2322-client-request-state",
    "auth/pre-registration",
    "auth/basic-cimd",
)

_EXPECTED_REVIEWED_OAUTH_SCENARIOS = (
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

_EXPECTED_OUT_OF_SCOPE_OAUTH_SCENARIOS = (
    "auth/2025-03-26-oauth-metadata-backcompat",
    "auth/2025-03-26-oauth-endpoint-fallback",
    "auth/client-credentials-jwt",
    "auth/client-credentials-basic",
    "auth/enterprise-managed-authorization",
    "auth/dpop",
    "auth/dpop-nonce",
    "auth/wif-jwt-bearer",
)


def _successful_checks(scenario: str) -> list[dict[str, str]]:
    contract = gate.OFFICIAL_CLIENT_SCENARIOS[scenario]
    if contract.required_success_names:
        repeated_id = min(contract.required_success_ids)
        checks = [
            {"id": repeated_id, "name": name, "status": "SUCCESS"}
            for name in sorted(contract.required_success_names)
        ]
        checks.extend(
            {"id": repeated_id, "name": name, "status": "SKIPPED"}
            for name in sorted(contract.required_skipped_names)
        )
    else:
        checks = [
            {"id": check_id, "status": "SUCCESS"}
            for check_id in sorted(contract.required_success_ids)
        ]
        checks.extend(
            {"id": check_id, "status": "SKIPPED"}
            for check_id in sorted(contract.required_skipped_ids)
        )
    checks.extend(
        {"id": check_id, "status": "INFO"}
        for check_id in sorted(
            contract.required_present_ids - contract.required_skipped_ids
        )
    )
    return checks


def test_official_conformance_source_and_reviewed_allowlist_are_immutable() -> None:
    assert gate.OFFICIAL_CONFORMANCE_REPOSITORY == (
        "https://github.com/modelcontextprotocol/conformance.git"
    )
    assert gate.OFFICIAL_CONFORMANCE_COMMIT == (
        "81eb1c3edaed87d7fd585d7b80186da7a2960660"
    )
    assert gate.OFFICIAL_CONFORMANCE_PACKAGE_VERSION == "0.2.0-alpha.10"
    assert gate.MCP_PROTOCOL_REVISION == "2026-07-28"
    assert tuple(gate.OFFICIAL_CLIENT_SCENARIOS) == _EXPECTED_SCENARIOS
    assert set(gate._SCENARIO_TOOLS) == gate.OFFICIAL_TOOL_SCENARIOS
    assert gate.OFFICIAL_MRTR_SCENARIO in gate.OFFICIAL_CLIENT_SCENARIOS
    assert gate.OFFICIAL_MRTR_SCENARIO not in gate._SCENARIO_TOOLS
    assert gate.OFFICIAL_OAUTH_SCENARIOS == {
        "auth/pre-registration",
        "auth/basic-cimd",
    }
    assert gate.OFFICIAL_OAUTH_SCENARIOS <= set(gate.OFFICIAL_CLIENT_SCENARIOS)
    assert gate.OFFICIAL_OAUTH_SCENARIOS.isdisjoint(gate._SCENARIO_TOOLS)


def test_official_oauth_inventory_separates_safe_pinned_runs_from_unavailable() -> None:
    assert (
        gate.OFFICIAL_OAUTH_SCENARIOS_REVIEWED_BUT_NOT_RUNNABLE
        == _EXPECTED_REVIEWED_OAUTH_SCENARIOS
    )
    assert not set(_EXPECTED_REVIEWED_OAUTH_SCENARIOS).intersection(
        gate.OFFICIAL_OAUTH_SCENARIOS
    )
    assert (
        gate.OFFICIAL_OAUTH_SCENARIOS_REVIEWED_OUT_OF_SCOPE
        == _EXPECTED_OUT_OF_SCOPE_OAUTH_SCENARIOS
    )
    assert not set(_EXPECTED_OUT_OF_SCOPE_OAUTH_SCENARIOS).intersection(
        gate.OFFICIAL_OAUTH_SCENARIOS
        | set(_EXPECTED_REVIEWED_OAUTH_SCENARIOS)
    )
    reviewed_inventory = (
        set(gate.OFFICIAL_OAUTH_SCENARIOS)
        | set(_EXPECTED_REVIEWED_OAUTH_SCENARIOS)
        | set(_EXPECTED_OUT_OF_SCOPE_OAUTH_SCENARIOS)
    )
    assert len(reviewed_inventory) == 33
    assert gate.OFFICIAL_OAUTH_AUTHORITY_GAP_CODE == (
        "runner_omits_host_pinned_expected_issuer"
    )
    assert "Host-reviewed expected issuer" in gate.OFFICIAL_OAUTH_AUTHORITY_GAP
    assert gate.OFFICIAL_OAUTH_OUT_OF_SCOPE_REASON_CODE == (
        "unsupported_oauth_extensions_or_backcompat"
    )
    assert "authorization-code client" in gate.OFFICIAL_OAUTH_OUT_OF_SCOPE_REASON
    assert gate.RUNTIME_OAUTH_TLS_REGRESSION_NODE.endswith(
        "test_runtime_oauth_pkce_tls_and_bearer_transport_end_to_end"
    )


def test_fixed_upstream_oauth_adapter_uses_runtime_pins_and_no_dcr() -> None:
    source = inspect.getsource(gate._run_oauth_runtime_client_adapter)
    for required in (
        "Runtime.open",
        "PinnedMcpOAuthHttpTransport",
        'context.get("trusted_resource_url")',
        'context["trusted_issuer"]',
        'context["trusted_prm_url"]',
        'context["trusted_as_metadata_url"]',
        "runtime.mcp.add_oauth_profile",
        "runtime.mcp.auth_begin",
        "runtime.mcp.auth_complete",
        "runtime.mcp.call_tool",
        "pre-registration secret entered public evidence",
    ):
        assert required in source
    for forbidden in (
        "registration_endpoint",
        "McpOAuthRegistrationMode.DCR",
        "manager.begin(",
        "manager.complete(",
        "access_token(",
    ):
        assert forbidden not in source

    harness = (
        gate.REPOSITORY_ROOT / "scripts" / "mcp_conformance_oauth_harness.mts"
    ).read_text(encoding="utf-8")
    for required in (
        'scenario.authServer?.getUrl()',
        'shell: false',
        'trusted_resource_url: resource.href',
        'trusted_issuer: issuer.origin',
        'registrationMode = "preregistered"',
        'registrationMode = "cimd"',
        'const durableChecks = checks.map',
        'id: check.id',
        'status: check.status',
        'output omitted',
    ):
        assert required in harness
    for forbidden in (
        "details: check.details",
        "name: check.name",
        "description: check.description",
        "timestamp: check.timestamp",
        "source: check.source",
        'writeFile(path.join(output, "stdout.txt")',
        'writeFile(path.join(output, "stderr.txt")',
    ):
        assert forbidden not in harness
    assert "--expected-failures" not in harness


def test_durable_official_evidence_drops_raw_details_logs_and_secrets(
    tmp_path: Path,
) -> None:
    scenario = tmp_path / "scenario"
    result = scenario / "upstream-timestamped-result"
    result.mkdir(parents=True)
    raw_checks = [
        {
            "id": "check-id",
            "status": "SUCCESS",
            "name": "transient name",
            "description": "transient description",
            "timestamp": "future timestamp",
            "details": {
                "state": "private-state",
                "code": "private-code",
                "access_token": "private-token",
            },
            "source": {"logs": ["private log"]},
            "specReferences": [
                {
                    "id": "RFC-TEST",
                    "url": "https://example.invalid/spec",
                    "private": "drop-me",
                }
            ],
        }
    ]
    checks_path = result / "checks.json"
    checks_path.write_text(json.dumps(raw_checks), encoding="utf-8")
    (result / "stdout.txt").write_text("private stdout", encoding="utf-8")
    (result / "stderr.txt").write_text("private stderr", encoding="utf-8")

    digest = gate._persist_durable_scenario_evidence(scenario, raw_checks)

    assert len(digest) == 64
    int(digest, 16)
    assert [path for path in scenario.rglob("*") if path.is_file()] == [checks_path]
    assert json.loads(checks_path.read_text(encoding="utf-8")) == [
        {
            "id": "check-id",
            "status": "SUCCESS",
            "specReferences": [
                {"id": "RFC-TEST", "url": "https://example.invalid/spec"}
            ],
        }
    ]


def test_failed_official_scenario_discards_every_raw_artifact(tmp_path: Path) -> None:
    result = tmp_path / "scenario" / "upstream-result"
    result.mkdir(parents=True)
    for name in ("checks.json", "stdout.txt", "stderr.txt", "trace.log"):
        (result / name).write_text("private raw evidence", encoding="utf-8")

    gate._discard_raw_scenario_evidence(tmp_path / "scenario")

    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_durable_official_evidence_digest_ignores_upstream_check_order(
    tmp_path: Path,
) -> None:
    checks = [
        {"id": "b", "status": "INFO"},
        {"id": "a", "status": "SUCCESS"},
    ]

    first = gate._persist_durable_scenario_evidence(tmp_path / "first", checks)
    second = gate._persist_durable_scenario_evidence(
        tmp_path / "second", list(reversed(checks))
    )

    assert first == second


def test_official_mrtr_adapter_uses_only_the_runtime_durable_facade() -> None:
    source = inspect.getsource(gate._run_mrtr_runtime_client_adapter)
    for required in (
        "Runtime.open",
        "register_server",
        "runtime.mcp.call_tool",
        "runtime.mcp.respond_continuation",
        "human_request_id",
        "primitive.mcp.continuation.respond",
    ):
        assert required in source
    for forbidden in (
        "from mcp.client",
        "MemoryRepository",
        "McpContinuationManager(",
        "manager.respond(",
    ):
        assert forbidden not in source


def test_request_metadata_uses_exact_v3_elicitation_contract() -> None:
    contract = gate.OFFICIAL_CLIENT_SCENARIOS["request-metadata"]
    elicitation = "sep-2575-client-declares-elicitation-capability"
    assert elicitation in contract.required_success_ids
    assert elicitation not in contract.required_skipped_ids
    assert contract.required_skipped_ids == {
        "sep-2575-client-declares-roots-capability",
        "sep-2575-client-declares-sampling-capability",
    }
    manifest = gate._manifest("http://127.0.0.1:1234/mcp", "request-metadata")
    assert manifest["schema_version"] == 3
    assert manifest["protocol_mode"] == gate.MCP_PROTOCOL_REVISION


@pytest.mark.parametrize("scenario", _EXPECTED_SCENARIOS)
def test_each_allowlisted_scenario_requires_nonempty_official_evidence(
    scenario: str,
) -> None:
    summary = gate._validate_checks(scenario, _successful_checks(scenario))

    assert summary["scenario"] == scenario
    assert summary["check_count"] >= 1
    assert summary["success_count"] >= 1


@pytest.mark.parametrize("status", ["FAILURE", "WARNING"])
def test_official_failure_or_warning_cannot_be_baselined(status: str) -> None:
    checks = _successful_checks("tools_call")
    checks.append({"id": "wire-regression", "status": status})

    with pytest.raises(gate.ConformanceGateError, match="official conformance failures"):
        gate._validate_checks("tools_call", checks)


def test_missing_requirement_level_check_fails_closed() -> None:
    with pytest.raises(gate.ConformanceGateError, match="incomplete official evidence"):
        gate._validate_checks(
            "tools_call",
            [{"id": "wire-schema-valid", "status": "SUCCESS"}],
        )


def test_unknown_check_status_fails_closed() -> None:
    with pytest.raises(gate.ConformanceGateError, match="unknown status"):
        gate._validate_checks(
            "tools_call",
            [{"id": "tool-add-numbers", "status": "PASSED"}],
        )


def test_official_scenario_command_has_no_expected_failure_escape_hatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def fake_run_checked(
        argv: tuple[str, ...],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        captured.append((tuple(argv), dict(kwargs["env"])))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(gate, "_run_checked", fake_run_checked)
    monkeypatch.setattr(
        gate,
        "_load_checks",
        lambda _root: [
            {"id": "tool-add-numbers", "status": "SUCCESS"},
            {"id": "wire-schema-valid", "status": "SUCCESS"},
        ],
    )

    environment = gate._sanitized_client_environment(tmp_path / "isolation")
    summary = gate._run_official_scenario(
        node="/usr/bin/node",
        runner=tmp_path / "dist" / "index.js",
        checkout=tmp_path,
        scenario="tools_call",
        output_root=tmp_path / "results",
        timeout_ms=30_000,
        env=environment,
    )

    assert summary["success_count"] == 2
    assert len(captured) == 1
    command, runner_environment = captured[0]
    assert runner_environment == environment
    assert command[2:5] == ("client", "--command", gate._client_command())
    assert "--scenario" in command
    assert "--spec-version" in command
    assert gate.MCP_PROTOCOL_REVISION in command
    assert "--expected-failures" not in command
    assert "--force" not in command


def test_scenario_subprocess_failure_omits_untrusted_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ("official", "scenario"),
            1,
            "private stdout state",
            "private stderr token",
        ),
    )

    with pytest.raises(gate.ConformanceGateError) as captured:
        gate._run_checked(
            ("official", "scenario"),
            env={"PATH": "/allowed/bin"},
            reveal_failure_output=False,
        )

    message = str(captured.value)
    assert "output omitted" in message
    assert "private" not in message


def test_fixed_upstream_oauth_command_uses_pinned_loader_and_no_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    loader = checkout / "node_modules" / "tsx" / "dist" / "loader.mjs"
    loader.parent.mkdir(parents=True)
    loader.write_text("// pinned fixture loader\n", encoding="utf-8")
    captured: list[tuple[str, ...]] = []

    def fake_run_checked(
        argv: tuple[str, ...],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        captured.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(gate, "_run_checked", fake_run_checked)
    monkeypatch.setattr(
        gate,
        "_load_checks",
        lambda _root: _successful_checks("auth/pre-registration"),
    )
    summary = gate._run_fixed_upstream_oauth_scenario(
        node="/allowed/bin/node",
        checkout=checkout,
        scenario="auth/pre-registration",
        output_root=tmp_path / "results",
        timeout_ms=30_000,
        env={"PATH": "/allowed/bin"},
    )

    assert summary["scenario"] == "auth/pre-registration"
    assert len(captured) == 1
    command = captured[0]
    assert command[:3] == ("/allowed/bin/node", "--import", str(loader))
    assert "mcp_conformance_oauth_harness.mts" in command[3]
    assert "--expected-failures" not in command


def test_client_command_preserves_virtual_environment_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = tmp_path / "venv" / "bin" / "python"
    original_resolve = Path.resolve

    def reject_launcher_resolution(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Path:
        if path == launcher:
            raise AssertionError("virtual-environment launcher must not be resolved")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(gate.sys, "executable", str(launcher))
    monkeypatch.setattr(Path, "resolve", reject_launcher_resolution)

    command = gate._client_command()

    assert command.split(" ", 1)[0] == str(launcher.absolute())


def test_official_checkout_verification_binds_commit_and_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "package.json").write_text(
        json.dumps({"version": gate.OFFICIAL_CONFORMANCE_PACKAGE_VERSION}),
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text("{}\n", encoding="utf-8")
    environment = gate._sanitized_client_environment(tmp_path / "isolation")
    observed_environments: list[dict[str, str]] = []

    def fake_run_checked(*_args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed_environments.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(
            (), 0, gate.OFFICIAL_CONFORMANCE_COMMIT + "\n", ""
        )

    monkeypatch.setattr(gate, "_run_checked", fake_run_checked)

    gate._verify_official_checkout(tmp_path, env=environment)
    assert observed_environments == [environment]

    (tmp_path / "package.json").write_text(
        json.dumps({"version": "future"}),
        encoding="utf-8",
    )
    with pytest.raises(gate.ConformanceGateError, match="package version mismatch"):
        gate._verify_official_checkout(tmp_path, env=environment)


def test_client_environment_is_allowlisted_and_uses_isolated_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/safe/bin")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("HOME", "/ambient/home")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "not-a-real-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-a-real-secret")
    monkeypatch.setenv("DEMO_TOKEN", "not-a-real-secret")
    monkeypatch.setenv("CUSTOM_DATABASE_DSN", "not-a-real-dsn")
    monkeypatch.setenv("PYTHONPATH", "/ambient/pythonpath")
    monkeypatch.setenv("PYTHONHOME", "/ambient/pythonhome")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid")
    monkeypatch.setenv("TRACEPARENT", "00-secret")
    monkeypatch.setenv("TRACESTATE", "vendor=secret")
    monkeypatch.setenv("BAGGAGE", "secret=value")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=secret")
    monkeypatch.setenv("MCP_CONFORMANCE_CONTEXT", "ambient-context")
    monkeypatch.setenv("npm_config_cache", "/ambient/npm-cache")

    isolation = tmp_path / "isolation"
    selected = gate._sanitized_client_environment(isolation)

    assert selected["PATH"] == "/safe/bin"
    assert selected["LANG"] == "C.UTF-8"
    assert selected["PYTHONNOUSERSITE"] == "1"
    assert Path(selected["HOME"]) == isolation / "home"
    assert Path(selected["USERPROFILE"]) == isolation / "home"
    assert Path(selected["TMPDIR"]) == isolation / "tmp"
    assert Path(selected["npm_config_cache"]) == isolation / "cache" / "npm"
    assert Path(selected["npm_config_userconfig"]) == isolation / "npmrc"
    for forbidden in (
        "OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "DEMO_TOKEN",
        "CUSTOM_DATABASE_DSN",
        "PYTHONPATH",
        "PYTHONHOME",
        "HTTPS_PROXY",
        "TRACEPARENT",
        "TRACESTATE",
        "BAGGAGE",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "MCP_CONFORMANCE_CONTEXT",
    ):
        assert forbidden not in selected


def test_checkout_and_build_subprocesses_receive_only_the_isolated_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "not-a-real-secret")
    monkeypatch.setenv("SERVICE_DSN", "not-a-real-dsn")
    monkeypatch.setenv("PYTHONPATH", "/ambient/pythonpath")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid")
    environment = gate._sanitized_client_environment(tmp_path / "isolation")
    captured: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def fake_executable(name: str, _env: Any) -> str:
        return f"/allowed/bin/{name}"

    def fake_run_checked(
        argv: tuple[str, ...],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        captured.append((tuple(argv), dict(kwargs["env"])))
        return subprocess.CompletedProcess(argv, 0, "", "")

    verified_environments: list[dict[str, str]] = []

    def fake_verify(_checkout: Path, *, env: Any) -> None:
        verified_environments.append(dict(env))

    monkeypatch.setattr(gate, "_required_executable", fake_executable)
    monkeypatch.setattr(gate, "_run_checked", fake_run_checked)
    monkeypatch.setattr(gate, "_verify_official_checkout", fake_verify)

    with gate._official_checkout(None, env=environment):
        pass

    checkout_commands = [command for command, _env in captured]
    assert [command[1] for command in checkout_commands] == [
        "init",
        "-C",
        "-C",
        "-C",
    ]
    assert all(observed == environment for _command, observed in captured)
    assert verified_environments == [environment]

    captured.clear()
    checkout = tmp_path / "checkout"
    entrypoint = checkout / "dist" / "index.js"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("// fixture\n", encoding="utf-8")

    assert gate._prepare_official_runner(checkout, env=environment) == entrypoint
    assert captured[0][0] == (
        "/allowed/bin/npm",
        "ci",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund",
    )
    assert captured[1][0] == (
        "/allowed/bin/npm",
        "run",
        "build",
        "--silent",
    )
    assert all(observed == environment for _command, observed in captured)
    for _command, observed in captured:
        assert "AWS_ACCESS_KEY_ID" not in observed
        assert "SERVICE_DSN" not in observed
        assert "PYTHONPATH" not in observed
        assert "HTTP_PROXY" not in observed


def test_client_manifests_pin_modern_mode_and_never_grant_unlisted_tools() -> None:
    invalid = gate._manifest(
        "http://127.0.0.1:1234/mcp",
        "http-invalid-tool-headers",
    )

    assert invalid["schema_version"] == 2
    assert invalid["protocol_mode"] == "2026-07-28"
    assert invalid["http"] == {"url": "http://127.0.0.1:1234/mcp"}
    assert [tool["mcp_name"] for tool in invalid["tools"]] == ["valid_tool"]
    assert all(tool["input_schema"] == {} for tool in invalid["tools"])

    headers = gate._manifest(
        "http://127.0.0.1:1234/mcp",
        "http-standard-headers",
    )
    assert headers["schema_version"] == 3
    assert headers["protocol_mode"] == gate.MCP_PROTOCOL_REVISION
    assert headers["resources"] == [
        {
            "resource_id": "header-resource",
            "remote_uri": "file:///path/to/file%20name.txt",
            "right": "read",
            "information_flow": True,
            "model_visible": False,
            "mime_types": ["text/plain"],
        }
    ]
    assert headers["prompts"] == [
        {
            "prompt_id": "header-prompt",
            "mcp_name": "test_prompt",
            "argument_names": [],
        }
    ]
    assert headers["subscriptions"] == []


def test_standard_header_scenario_requires_every_modern_request_path() -> None:
    contract = gate.OFFICIAL_CLIENT_SCENARIOS["http-standard-headers"]

    assert contract.expected_success_count == 8
    assert contract.expected_skipped_count == 3
    assert contract.required_success_names == {
        "ClientMcpMethodHeader_tools_call",
        "ClientMcpNameHeader_tools_call",
        "ClientMcpMethodHeader_resources_list",
        "ClientMcpMethodHeader_resources_read",
        "ClientMcpNameHeader_resources_read",
        "ClientMcpMethodHeader_prompts_list",
        "ClientMcpMethodHeader_prompts_get",
        "ClientMcpNameHeader_prompts_get",
    }
    assert contract.required_skipped_names == {
        "ClientMcpMethodHeader_initialize",
        "ClientMcpMethodHeader_notifications_initialized",
        "ClientMcpMethodHeader_tools_list",
    }
    incomplete = _successful_checks("http-standard-headers")
    incomplete[0] = {
        **incomplete[0],
        "status": "SKIPPED",
    }
    with pytest.raises(gate.ConformanceGateError, match="success_names="):
        gate._validate_checks("http-standard-headers", incomplete)


def test_custom_header_context_is_shape_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MCP_CONFORMANCE_CONTEXT",
        json.dumps(
            {
                "name": "http-custom-headers",
                "toolCalls": [
                    {"name": "test_custom_headers", "arguments": {"region": "x"}}
                ],
            }
        ),
    )

    assert gate._context_tool_calls("http-custom-headers") == [
        {"name": "test_custom_headers", "arguments": {"region": "x"}}
    ]

    monkeypatch.setenv("MCP_CONFORMANCE_CONTEXT", "{}")
    with pytest.raises(gate.ConformanceGateError, match="context name mismatch"):
        gate._context_tool_calls("http-custom-headers")


def test_schema_preservation_reads_only_live_provider_projection() -> None:
    listing = {
        "tools": [
            {
                "mcp_name": "json_schema_2020_12_tool",
                "input_schema": {},
                "live": {
                    "input_schema": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "$defs": {"value": {"type": "string"}},
                    }
                },
            }
        ]
    }

    assert gate._live_schema(listing, "json_schema_2020_12_tool")["$defs"] == {
        "value": {"type": "string"}
    }


def test_checks_loader_rejects_empty_or_ambiguous_artifacts(tmp_path: Path) -> None:
    with pytest.raises(gate.ConformanceGateError, match="0 checks.json"):
        gate._load_checks(tmp_path)

    first = tmp_path / "first" / "checks.json"
    first.parent.mkdir()
    first.write_text("[]\n", encoding="utf-8")
    with pytest.raises(gate.ConformanceGateError, match="empty check set"):
        gate._load_checks(tmp_path)

    second = tmp_path / "second" / "checks.json"
    second.parent.mkdir()
    second.write_text("[]\n", encoding="utf-8")
    with pytest.raises(gate.ConformanceGateError, match="2 checks.json"):
        gate._load_checks(tmp_path)


def test_checks_loader_bounds_untrusted_upstream_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = tmp_path / "checks.json"
    checks.write_text('[{"id":"check","status":"SUCCESS"}]', encoding="utf-8")
    monkeypatch.setattr(gate, "_MAX_OFFICIAL_CHECKS_BYTES", 8)
    with pytest.raises(gate.ConformanceGateError, match="evidence bound"):
        gate._load_checks(tmp_path)

    monkeypatch.setattr(gate, "_MAX_OFFICIAL_CHECKS_BYTES", 1_024)
    monkeypatch.setattr(gate, "_MAX_OFFICIAL_CHECK_COUNT", 1)
    checks.write_text(
        json.dumps(
            [
                {"id": "first", "status": "SUCCESS"},
                {"id": "second", "status": "SUCCESS"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(gate.ConformanceGateError, match="too many checks"):
        gate._load_checks(tmp_path)
