from __future__ import annotations

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
)


def _successful_checks(scenario: str) -> list[dict[str, str]]:
    contract = gate.OFFICIAL_TOOLS_ONLY_SCENARIOS[scenario]
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


def test_official_conformance_source_and_tools_only_allowlist_are_immutable() -> None:
    assert gate.OFFICIAL_CONFORMANCE_REPOSITORY == (
        "https://github.com/modelcontextprotocol/conformance.git"
    )
    assert gate.OFFICIAL_CONFORMANCE_COMMIT == (
        "81eb1c3edaed87d7fd585d7b80186da7a2960660"
    )
    assert gate.OFFICIAL_CONFORMANCE_PACKAGE_VERSION == "0.2.0-alpha.10"
    assert gate.MCP_PROTOCOL_REVISION == "2026-07-28"
    assert tuple(gate.OFFICIAL_TOOLS_ONLY_SCENARIOS) == _EXPECTED_SCENARIOS
    assert set(gate._SCENARIO_TOOLS) == set(_EXPECTED_SCENARIOS)
    assert "sep-2322-client-request-state" not in gate.OFFICIAL_TOOLS_ONLY_SCENARIOS
    assert not any(name.startswith("auth/") for name in gate.OFFICIAL_TOOLS_ONLY_SCENARIOS)


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
