from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from agent_libos.api.cli import cli as cli_entrypoint
from agent_libos.api.cli import main as cli_main
from agent_libos.config import DEFAULT_CONFIG


def _assessment(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "assessment_id": "assessment-1",
        "job_id": "job-1",
        "kind": "approval",
        "status": "success",
        "domain": "filesystem",
        "action_id": "filesystem.read",
        "pid": "pid-1",
        "request_id": "request-1",
        "operation_id": "operation-1",
        "effect_id": "effect-1",
        "shadow_outcome": "require_human",
        "reason_codes": ["semantic.policy.no_matching_rule"],
        "ood": False,
        "abstain": False,
        "confidence_bps": 8_750,
        "calibration_bucket": "high",
        "classifier_id": "scripted",
        "classifier_version": "v1",
        "artifact_sha256": "1" * 64,
        "input_sha256": "2" * 64,
        "feature_snapshot_sha256": "3" * 64,
        "policy_sha256": "4" * 64,
        "tenant_bucket_sha256": "a" * 64,
        "created_at": "2026-08-05T00:00:00Z",
        "completed_at": "2026-08-05T00:00:01Z",
        "latency_ms": 12,
        "input_tokens": 21,
        "output_tokens": 8,
        "cost_microunits": 42,
        "human_outcome": "approved",
        "findings": [{"code": "semantic.policy.no_matching_rule"}],
        "data_findings": [],
        "matched_rule_ids": [],
        "proven_predicates": ["exact_binding"],
        "missing_predicates": ["ceiling_rule"],
        "source_refs_sha256": "5" * 64,
        "data_labels_sha256": "6" * 64,
        "sink_identity_sha256": "7" * 64,
        "tool_schema_sha256": "8" * 64,
        "provider_spec_sha256": "9" * 64,
        "manifest_sha256": "b" * 64,
        "action_sha256": "c" * 64,
        "resource_sha256": "d" * 64,
        "args_sha256": "e" * 64,
        "state_sha256": "f" * 64,
        "projection_sha256": "0" * 64,
        # These fields are intentionally outside the public v1 contract.
        "projection_json": {"intent": "SECRET-SENTINEL"},
        "prompt": "SECRET-SENTINEL",
        "reasoning": "SECRET-SENTINEL",
        "job_error": "SECRET-SENTINEL",
        "error_code": "SECRET-SENTINEL",
        "raw_human_response": "SECRET-SENTINEL",
        "body": "SECRET-SENTINEL",
        "content": "SECRET-SENTINEL",
        "raw_content": "SECRET-SENTINEL",
        "usage": {
            "input_tokens": 999_999,
            "private_provider_field": "SECRET-SENTINEL",
        },
    }
    value.update(overrides)
    return value


class _FakeSemantic:
    def __init__(self) -> None:
        self.query_args: dict[str, Any] | None = None

    def status(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "mode": "shadow",
            "adapter": "scripted",
            "profile_id": "semantic-review",
            "queue": {
                "queued": 1,
                "leased": 2,
                "succeeded": 3,
                "failed": 4,
                "cancelled": 5,
                "capture_failures": 6,
                "raw_projection": "SECRET-SENTINEL",
            },
            "assessments": {
                "total": 7,
                "success": 3,
                "error": 4,
                "ood": 1,
                "would_issue_exact_once": 0,
                "would_deny": 1,
                "require_human": 6,
                "by_status": {
                    "success": 3,
                    "skipped_policy": 0,
                    "egress_blocked": 0,
                    "timeout": 1,
                    "provider_error": 1,
                    "provider_outcome_unknown": 0,
                    "invalid_schema": 1,
                    "ood": 1,
                    "abstained": 0,
                    "stale_input": 0,
                },
                "by_domain": {
                    "filesystem": 2,
                    "shell": 1,
                    "git": 1,
                    "jsonrpc": 1,
                    "mcp": 0,
                    "runtime": 1,
                    "unknown": 1,
                },
            },
            "actual_auto_approval": {
                "numerator": 0,
                "denominator": 0,
                "rate": None,
            },
            "prompt": "SECRET-SENTINEL",
        }

    def query_assessments(self, **kwargs: Any) -> dict[str, Any]:
        self.query_args = kwargs
        return {
            "schema_version": 1,
            "items": [_assessment()],
            "next_cursor": "cursor-2",
        }

    def get_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        if assessment_id == "missing":
            return None
        return _assessment(assessment_id=assessment_id)


class _FakeRuntime:
    def __init__(self) -> None:
        self.semantic = _FakeSemantic()
        self.shutdown_calls: list[tuple[str, str]] = []

    def shutdown(self, *, actor: str, reason: str) -> dict[str, bool]:
        self.shutdown_calls.append((actor, reason))
        return {"ok": True}


def _install_runtime(monkeypatch: pytest.MonkeyPatch) -> _FakeRuntime:
    runtime = _FakeRuntime()
    monkeypatch.setattr(
        "agent_libos.api.cli.load_config_from_project_root",
        lambda: DEFAULT_CONFIG,
    )
    monkeypatch.setattr(
        "agent_libos.api.cli.Runtime.open",
        lambda *_args, **_kwargs: runtime,
    )
    return runtime


def test_semantic_status_emits_exact_v2_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _install_runtime(monkeypatch)

    cli_main(["semantic", "status"])

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema_version": 2,
        "mode": "shadow",
        "adapter": "scripted",
        "profile_id": "semantic-review",
        "queue": {
            "queued": 1,
            "leased": 2,
            "succeeded": 3,
            "failed": 4,
            "cancelled": 5,
            "capture_failures": 6,
        },
        "assessments": {
            "total": 7,
            "success": 3,
            "error": 4,
            "ood": 1,
            "would_issue_exact_once": 0,
            "would_deny": 1,
            "require_human": 6,
            "by_status": {
                "success": 3,
                "skipped_policy": 0,
                "egress_blocked": 0,
                "timeout": 1,
                "provider_error": 1,
                "provider_outcome_unknown": 0,
                "invalid_schema": 1,
                "ood": 1,
                "abstained": 0,
                "stale_input": 0,
            },
            "by_domain": {
                "filesystem": 2,
                "shell": 1,
                "git": 1,
                "jsonrpc": 1,
                "mcp": 0,
                "runtime": 1,
                "unknown": 1,
            },
        },
        "actual_auto_approval": {
            "numerator": 0,
            "denominator": 0,
            "rate": None,
        },
    }
    assert "SECRET-SENTINEL" not in json.dumps(payload)
    assert runtime.shutdown_calls == [("cli", "cli.command_complete")]


def test_semantic_status_uses_zero_counters_and_null_rate_for_empty_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _install_runtime(monkeypatch)
    runtime.semantic.status = lambda: {  # type: ignore[method-assign]
        "schema_version": 2,
        "mode": "off",
        "adapter": "deterministic",
        "profile_id": None,
        "queue": {},
        "assessments": {
            "by_status": {
                "success": 0,
                "skipped_policy": 0,
                "egress_blocked": 0,
                "timeout": 0,
                "provider_error": 0,
                "provider_outcome_unknown": 0,
                "invalid_schema": 0,
                "ood": 0,
                "abstained": 0,
                "stale_input": 0,
            },
            "by_domain": {
                "filesystem": 0,
                "shell": 0,
                "git": 0,
                "jsonrpc": 0,
                "mcp": 0,
                "runtime": 0,
                "unknown": 0,
            },
        },
        "actual_auto_approval": {},
    }

    cli_main(["semantic", "status"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["queue"] == {
        "queued": 0,
        "leased": 0,
        "succeeded": 0,
        "failed": 0,
        "cancelled": 0,
        "capture_failures": 0,
    }
    assert payload["assessments"] == {
        "total": 0,
        "success": 0,
        "error": 0,
        "ood": 0,
        "would_issue_exact_once": 0,
        "would_deny": 0,
        "require_human": 0,
        "by_status": {
            "success": 0,
            "skipped_policy": 0,
            "egress_blocked": 0,
            "timeout": 0,
            "provider_error": 0,
            "provider_outcome_unknown": 0,
            "invalid_schema": 0,
            "ood": 0,
            "abstained": 0,
            "stale_input": 0,
        },
        "by_domain": {
            "filesystem": 0,
            "shell": 0,
            "git": 0,
            "jsonrpc": 0,
            "mcp": 0,
            "runtime": 0,
            "unknown": 0,
        },
    }
    assert payload["actual_auto_approval"] == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
    }


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("mode",), "enforce"),
        (("adapter",), "unknown"),
        (("profile_id",), "../classifier"),
        (("profile_id",), True),
        (("queue", "queued"), True),
        (("queue", "leased"), "2"),
        (("queue", "failed"), -1),
        (("assessments", "total"), True),
        (("assessments", "success"), "3"),
        (("assessments", "error"), -1),
        (("assessments", "by_status", "success"), True),
        (("assessments", "by_domain", "filesystem"), "2"),
        (("actual_auto_approval", "numerator"), 1),
        (("actual_auto_approval", "denominator"), 1),
        (("actual_auto_approval", "rate"), 0.0),
    ],
)
def test_semantic_status_rejects_malformed_v2_fields(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    invalid: object,
) -> None:
    runtime = _install_runtime(monkeypatch)
    payload = runtime.semantic.status()
    target: dict[str, Any] = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid
    runtime.semantic.status = lambda: payload  # type: ignore[method-assign]

    with pytest.raises(TypeError):
        cli_main(["semantic", "status"])

    assert runtime.shutdown_calls == [("cli", "cli.command_complete")]


@pytest.mark.parametrize("aggregate", ["by_status", "by_domain"])
def test_semantic_status_requires_complete_consistent_aggregate_mappings(
    monkeypatch: pytest.MonkeyPatch,
    aggregate: str,
) -> None:
    runtime = _install_runtime(monkeypatch)
    payload = runtime.semantic.status()
    payload["assessments"][aggregate].pop(next(iter(payload["assessments"][aggregate])))
    runtime.semantic.status = lambda: payload  # type: ignore[method-assign]

    with pytest.raises(TypeError):
        cli_main(["semantic", "status"])


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("assessments", "success"), 4),
        (("assessments", "would_issue_exact_once"), 1),
        (("assessments", "ood"), 2),
    ],
)
def test_semantic_status_rejects_inconsistent_derived_totals(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    invalid: int,
) -> None:
    runtime = _install_runtime(monkeypatch)
    payload = runtime.semantic.status()
    target: dict[str, Any] = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid
    runtime.semantic.status = lambda: payload  # type: ignore[method-assign]

    with pytest.raises(TypeError, match="inconsistent"):
        cli_main(["semantic", "status"])


def test_semantic_assessments_pushes_bounded_filters_and_redacts_detail_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _install_runtime(monkeypatch)

    cli_main(
        [
            "semantic",
            "assessments",
            "--pid",
            "pid-1",
            "--request-id",
            "request-1",
            "--operation-id",
            "operation-1",
            "--kind",
            "approval",
            "--status",
            "success",
            "--domain",
            "filesystem",
            "--action-id",
            "filesystem.read",
            "--tenant-bucket-sha256",
            "a" * 64,
            "--after",
            "cursor-1",
            "--limit",
            "17",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert runtime.semantic.query_args == {
        "pid": "pid-1",
        "request_id": "request-1",
        "operation_id": "operation-1",
        "kind": "approval",
        "status": "success",
        "domain": "filesystem",
        "action_id": "filesystem.read",
        "tenant_bucket_sha256": "a" * 64,
        "after": "cursor-1",
        "limit": 17,
    }
    assert payload["schema_version"] == 1
    assert payload["next_cursor"] == "cursor-2"
    assert payload["items"] == [
        {key: value for key, value in _assessment().items() if key in payload["items"][0]}
    ]
    assert "findings" not in payload["items"][0]
    assert "projection_json" not in payload["items"][0]
    assert "SECRET-SENTINEL" not in json.dumps(payload)


def test_semantic_assessment_page_cannot_exceed_requested_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_runtime(monkeypatch)
    runtime.semantic.query_assessments = lambda **_kwargs: {  # type: ignore[method-assign]
        "items": [_assessment(), _assessment(assessment_id="assessment-2")],
        "next_cursor": None,
    }

    with pytest.raises(TypeError, match="page items"):
        cli_main(["semantic", "assessments", "--limit", "1"])

    assert runtime.shutdown_calls == [("cli", "cli.command_complete")]


def test_semantic_assessments_accept_json_safe_integer_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _install_runtime(monkeypatch)
    ceiling = (1 << 53) - 1
    runtime.semantic.query_assessments = lambda **_kwargs: {  # type: ignore[method-assign]
        "schema_version": 1,
        "items": [
            _assessment(
                input_tokens=ceiling,
                output_tokens=ceiling,
                cost_microunits=ceiling,
                latency_ms=ceiling,
            )
        ],
        "next_cursor": None,
    }

    cli_main(["semantic", "assessments"])

    item = json.loads(capsys.readouterr().out)["items"][0]
    assert item["input_tokens"] == ceiling
    assert item["output_tokens"] == ceiling
    assert item["cost_microunits"] == ceiling
    assert item["latency_ms"] == ceiling


@pytest.mark.parametrize(
    "field",
    ["input_tokens", "output_tokens", "cost_microunits", "latency_ms"],
)
def test_semantic_assessments_reject_non_safe_integer_metrics(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    runtime = _install_runtime(monkeypatch)
    runtime.semantic.query_assessments = lambda **_kwargs: {  # type: ignore[method-assign]
        "schema_version": 1,
        "items": [_assessment(**{field: 1 << 53})],
        "next_cursor": None,
    }

    with pytest.raises(TypeError, match=field):
        cli_main(["semantic", "assessments"])

    assert runtime.shutdown_calls == [("cli", "cli.command_complete")]


def test_semantic_show_emits_detail_without_private_payloads(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_runtime(monkeypatch)

    cli_main(["semantic", "show", "assessment-visible"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["assessment"]["assessment_id"] == "assessment-visible"
    assert payload["assessment"]["findings"] == [
        {
            "code": "semantic.policy.no_matching_rule",
            "severity": None,
            "confidence_bps": None,
            "evidence_sha256": None,
            "source": None,
        }
    ]
    assert payload["assessment"]["manifest_sha256"] == "b" * 64
    assert payload["assessment"]["action_sha256"] == "c" * 64
    assert payload["assessment"]["resource_sha256"] == "d" * 64
    assert payload["assessment"]["args_sha256"] == "e" * 64
    assert payload["assessment"]["state_sha256"] == "f" * 64
    assert payload["assessment"]["projection_sha256"] == "0" * 64
    assert "projection_json" not in payload["assessment"]
    assert "prompt" not in payload["assessment"]
    assert "reasoning" not in payload["assessment"]
    assert "job_error" not in payload["assessment"]
    assert "error_code" not in payload["assessment"]
    assert "raw_human_response" not in payload["assessment"]
    assert "body" not in payload["assessment"]
    assert "content" not in payload["assessment"]
    assert "raw_content" not in payload["assessment"]
    assert "SECRET-SENTINEL" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("field", "span_start", "span_end"),
    [
        ("SECRET-SENTINEL-INVENTED-LOCATOR", None, None),
        ("root_goal", None, None),
        ("approval.request", 0, 1),
        ("redacted_intent", None, None),
        ("redacted_intent", False, 1),
        ("redacted_intent", 0, 2_001),
    ],
    ids=(
        "invented-locator",
        "wrong-kind-locator",
        "coarse-span",
        "missing-redacted-span",
        "boolean-redacted-span",
        "oversized-redacted-span",
    ),
)
def test_semantic_show_rejects_invalid_data_locator_without_echo(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    span_start: int | None,
    span_end: int | None,
) -> None:
    runtime = _install_runtime(monkeypatch)
    data_finding = {
        "category": "source_code",
        "field": field,
        "span_start": span_start,
        "span_end": span_end,
        "sensitivity_floor": "confidential",
        "integrity_ceiling": "unknown",
        "trust_ceiling": "untrusted",
        "confidence_bps": 9_200,
        "evidence_sha256": "f" * 64,
    }
    runtime.semantic.get_assessment = lambda _assessment_id: _assessment(  # type: ignore[method-assign]
        data_findings=[data_finding]
    )

    with pytest.raises(TypeError, match="invalid data finding"):
        cli_main(["semantic", "show", "assessment-unsafe"])

    assert "SECRET-SENTINEL" not in capsys.readouterr().out
    assert runtime.shutdown_calls == [("cli", "cli.command_complete")]


def test_semantic_show_allows_nullable_optional_provenance_digests(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _install_runtime(monkeypatch)
    runtime.semantic.get_assessment = lambda _assessment_id: _assessment(  # type: ignore[method-assign]
        manifest_sha256=None,
        resource_sha256=None,
        args_sha256=None,
        state_sha256=None,
    )

    cli_main(["semantic", "show", "assessment-visible"])

    assessment = json.loads(capsys.readouterr().out)["assessment"]
    assert assessment["manifest_sha256"] is None
    assert assessment["resource_sha256"] is None
    assert assessment["args_sha256"] is None
    assert assessment["state_sha256"] is None
    assert assessment["action_sha256"] == "c" * 64
    assert assessment["projection_sha256"] == "0" * 64


@pytest.mark.parametrize("field", ["action_sha256", "projection_sha256"])
@pytest.mark.parametrize("value", [None, "SECRET-SENTINEL"])
def test_semantic_show_requires_valid_binding_digests_without_echo(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: object,
) -> None:
    runtime = _install_runtime(monkeypatch)
    runtime.semantic.get_assessment = lambda _assessment_id: _assessment(  # type: ignore[method-assign]
        **{field: value}
    )

    with pytest.raises(TypeError, match="sha256 digest"):
        cli_main(["semantic", "show", "assessment-unsafe"])

    assert "SECRET-SENTINEL" not in capsys.readouterr().out
    assert runtime.shutdown_calls == [("cli", "cli.command_complete")]


def test_semantic_show_rejects_unknown_human_outcome_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _install_runtime(monkeypatch)
    runtime.semantic.get_assessment = lambda _assessment_id: _assessment(  # type: ignore[method-assign]
        human_outcome="SECRET-SENTINEL"
    )

    with pytest.raises(TypeError, match="invalid human outcome"):
        cli_main(["semantic", "show", "assessment-unsafe"])

    assert "SECRET-SENTINEL" not in capsys.readouterr().out
    assert runtime.shutdown_calls == [("cli", "cli.command_complete")]


def test_semantic_show_missing_uses_stable_cli_error_and_still_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _install_runtime(monkeypatch)

    with pytest.raises(SystemExit) as raised:
        cli_entrypoint(["semantic", "show", "missing"])

    assert raised.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema_version": 1,
        "error": {
            "type": "NotFound",
            "message": "semantic assessment not found: missing",
        },
    }
    assert runtime.shutdown_calls == [("cli", "cli.command_complete")]


@pytest.mark.parametrize("limit", ["0", "101", "nan"])
def test_semantic_assessment_limit_is_bounded(
    limit: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli_main(["semantic", "assessments", "--limit", limit])

    assert raised.value.code == 2
    assert "semantic assessment limit" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option",
    [
        ("--kind", "unknown"),
        ("--status", "unknown"),
        ("--domain", "unknown-domain"),
    ],
)
def test_semantic_assessment_enums_are_rejected_before_runtime_open(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: tuple[str, str],
) -> None:
    monkeypatch.setattr(
        "agent_libos.api.cli.Runtime.open",
        lambda *_args, **_kwargs: pytest.fail("invalid filters must not open Runtime"),
    )

    with pytest.raises(SystemExit) as raised:
        cli_main(["semantic", "assessments", option[0], option[1]])

    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["semantic", "assessments", "--pid", ""],
        ["semantic", "assessments", "--pid", "x" * 513],
        ["semantic", "assessments", "--request-id", "line\nbreak"],
        ["semantic", "assessments", "--operation-id", "x" * 513],
        ["semantic", "assessments", "--operation-id", "del\x7fchar"],
        ["semantic", "assessments", "--after", "line\nbreak"],
        ["semantic", "assessments", "--after", "x" * 2_049],
        ["semantic", "show", "x" * 513],
    ],
)
def test_semantic_text_inputs_are_bounded(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli_main(argv)

    assert raised.value.code == 2
    assert "must contain" in capsys.readouterr().err


class _MigrationPayload:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


def _install_migration_module(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, object, dict[str, Any]]],
) -> None:
    module = ModuleType("agent_libos.storage.semantic_v5_migration")

    def plan(target: object, **kwargs: Any) -> _MigrationPayload:
        calls.append(("plan", target, kwargs))
        return _MigrationPayload(
            {
                "schema_version": 1,
                "backend": "sqlite",
                "from_version": 4,
                "to_version": 5,
                "statements": ["ALTER TABLE human_requests ADD COLUMN revision"],
                "plan_sha256": "a" * 64,
            }
        )

    def apply(target: object, **kwargs: Any) -> _MigrationPayload:
        calls.append(("apply", target, kwargs))
        return _MigrationPayload(
            {
                "schema_version": 1,
                "backend": "sqlite",
                "from_version": 4,
                "to_version": 5,
                "plan_sha256": "a" * 64,
                "applied": True,
                "already_applied": False,
            }
        )

    module.plan_store_v5_migration = plan  # type: ignore[attr-defined]
    module.apply_store_v5_migration = apply  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_store_migration_dry_run_executes_before_runtime_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, object, dict[str, Any]]] = []
    _install_migration_module(monkeypatch, calls)
    monkeypatch.setattr(
        "agent_libos.api.cli.load_config_from_project_root",
        lambda: DEFAULT_CONFIG,
    )
    monkeypatch.setattr(
        "agent_libos.api.cli.Runtime.open",
        lambda *_args, **_kwargs: pytest.fail("migration must not open Runtime"),
    )
    database = tmp_path / "runtime.sqlite"

    cli_main(["--db", str(database), "store", "migrate", "--to", "5", "--dry-run"])

    assert calls == [
        (
            "plan",
            str(database),
            {"sqlite_backup": None, "postgres_snapshot_confirmed": False},
        )
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["plan_sha256"] == "a" * 64


def test_store_migration_apply_forwards_digest_and_backup_before_runtime_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, object, dict[str, Any]]] = []
    _install_migration_module(monkeypatch, calls)
    monkeypatch.setattr(
        "agent_libos.api.cli.load_config_from_project_root",
        lambda: DEFAULT_CONFIG,
    )
    monkeypatch.setattr(
        "agent_libos.api.cli.Runtime.open",
        lambda *_args, **_kwargs: pytest.fail("migration must not open Runtime"),
    )
    database = tmp_path / "runtime.sqlite"
    backup = tmp_path / "runtime-v4.sqlite.backup"

    cli_main(
        [
            "--db",
            str(database),
            "store",
            "migrate",
            "--to",
            "5",
            "--apply",
            "--expected-plan-sha256",
            "A" * 64,
            "--sqlite-backup",
            str(backup),
        ]
    )

    assert calls == [
        (
            "apply",
            str(database),
            {
                "expected_plan_sha256": "a" * 64,
                "sqlite_backup": backup,
                "postgres_snapshot_confirmed": False,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out)["applied"] is True


def test_store_migration_apply_requires_plan_digest_before_importing_migrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "agent_libos.api.cli.load_config_from_project_root",
        lambda: DEFAULT_CONFIG,
    )
    monkeypatch.delitem(
        sys.modules,
        "agent_libos.storage.semantic_v5_migration",
        raising=False,
    )

    with pytest.raises(SystemExit) as raised:
        cli_main(
            [
                "--db",
                str(tmp_path / "runtime.sqlite"),
                "store",
                "migrate",
                "--to",
                "5",
                "--apply",
            ]
        )

    assert raised.value.code == 2
    assert "requires --expected-plan-sha256" in capsys.readouterr().err
