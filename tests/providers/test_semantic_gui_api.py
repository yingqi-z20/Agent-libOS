from __future__ import annotations

import http.client
import json
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from agent_libos.api.gui.server import GuiHTTPServer
from agent_libos.config import DEFAULT_CONFIG


class _FakeSemanticService:
    def __init__(self) -> None:
        self.query_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []
        self.status_calls = 0
        self.actual_auto_approval: dict[str, Any] = {
            "numerator": 0,
            "denominator": 1,
            "rate": 0.0,
        }
        self.assessment = {
            "schema_version": 1,
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
            "reason_codes": ["sensitive_data"],
            "ood": False,
            "abstain": False,
            "confidence_bps": 8700,
            "calibration_bucket": "high",
            "input_tokens": 120,
            "output_tokens": 24,
            "cost_microunits": 75,
            "classifier_id": "classifier-1",
            "classifier_version": "v1",
            "artifact_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "feature_snapshot_sha256": "c" * 64,
            "policy_sha256": "d" * 64,
            "tenant_bucket_sha256": "6" * 64,
            "created_at": "2026-08-05T00:00:00Z",
            "completed_at": "2026-08-05T00:00:01Z",
            "latency_ms": 1000,
            "human_outcome": "approved",
            "findings": [
                {
                    "code": "sensitive_data",
                    "severity": "high",
                    "confidence_bps": 8700,
                    "evidence_sha256": "e" * 64,
                    "source": "model",
                    "explanation": "RAW_SECRET_FINDING",
                }
            ],
            "data_findings": [
                {
                    "category": "credential",
                    "field": "approval.request",
                    "span_start": None,
                    "span_end": None,
                    "sensitivity_floor": "secret",
                    "integrity_ceiling": "unknown",
                    "trust_ceiling": "untrusted",
                    "confidence_bps": 9500,
                    "evidence_sha256": "f" * 64,
                    "raw_value": "RAW_SECRET_DATA",
                }
            ],
            "matched_rule_ids": [],
            "proven_predicates": ["binding_current"],
            "missing_predicates": ["low_risk"],
            "source_refs_sha256": "1" * 64,
            "data_labels_sha256": "2" * 64,
            "sink_identity_sha256": "3" * 64,
            "tool_schema_sha256": "4" * 64,
            "provider_spec_sha256": "5" * 64,
            "manifest_sha256": "7" * 64,
            "action_sha256": "8" * 64,
            "resource_sha256": "9" * 64,
            "args_sha256": "0" * 64,
            "state_sha256": "a" * 64,
            "projection_sha256": "b" * 64,
            "raw_prompt": "RAW_SECRET_PROMPT",
            "raw_response": "RAW_SECRET_RESPONSE",
            "reasoning": "RAW_SECRET_REASONING",
            "job_error": "RAW_SECRET_JOB_ERROR",
            "error_code": "RAW_SECRET_ERROR_CODE",
            "raw_human_response": "RAW_SECRET_HUMAN_RESPONSE",
            "human_response": {"body": "RAW_SECRET_HUMAN_BODY"},
            "body": "RAW_SECRET_BODY",
            "content": "RAW_SECRET_CONTENT",
            "provider_payload": {"secret": "RAW_SECRET_PROVIDER"},
            "usage": {
                "input_tokens": 999_999,
                "private_provider_field": "RAW_SECRET_USAGE",
            },
        }
        self.query_items: list[dict[str, Any]] = [self.assessment]

    def status(self) -> dict[str, Any]:
        self.status_calls += 1
        return {
            "schema_version": 3,
            "mode": "enforce_deny",
            "adapter": "scripted",
            "profile_id": "semantic-profile",
            "queue": {
                "queued": 1,
                "leased": 2,
                "succeeded": 3,
                "failed": 4,
                "cancelled": 5,
                "capture_failures": 6,
                "raw_projection": "RAW_SECRET_STATUS",
            },
            "assessments": {
                "total": 10,
                "success": 7,
                "error": 3,
                "ood": 2,
                "would_issue_exact_once": 1,
                "would_deny": 2,
                "require_human": 7,
                "by_status": {
                    "success": 7,
                    "skipped_policy": 0,
                    "egress_blocked": 0,
                    "timeout": 1,
                    "provider_error": 0,
                    "provider_outcome_unknown": 0,
                    "invalid_schema": 0,
                    "ood": 2,
                    "abstained": 0,
                    "stale_input": 0,
                },
                "by_domain": {
                    "filesystem": 4,
                    "shell": 1,
                    "git": 1,
                    "jsonrpc": 1,
                    "mcp": 1,
                    "runtime": 1,
                    "unknown": 1,
                },
            },
            "control": {
                "catalog_version": 1,
                "active_epoch_id": "epoch-1",
                "active_epoch_sha256": "7" * 64,
                "generation": 1,
                "state": "active",
                "trip_reason_code": None,
            },
            "flow": {
                "schema_version": 1,
                "available": True,
                "counts": {
                    "entities": 1,
                    "activities": 1,
                    "edges": 1,
                    "label_assertions": 0,
                },
                "coverage": {
                    "complete": 1,
                    "partial": 0,
                    "unknown": 0,
                    "conflict": 0,
                    "stale": 0,
                },
                "capture_failures": 0,
                "legacy_history": {
                    "present": False,
                    "source_schema_version": None,
                    "assessment_count": 0,
                    "coverage": None,
                    "evidence_sha256": None,
                    "created_at": None,
                },
            },
            "machine": {
                "eligible": 1,
                "issued": 0,
                "consumed": 0,
                "succeeded": 0,
                "failed": 0,
                "unknown": 0,
                "expired": 0,
                "revoked": 0,
                "race_lost": 0,
                "denied": 0,
            },
            "actual_auto_approval": self.actual_auto_approval,
            "review_metrics": {
                "reviewed": 0,
                "safe": 0,
                "unsafe": 0,
                "unsafe_rate": None,
                "issued_reviewed": 0,
                "issued_review_rate": None,
            },
            "prompt": "RAW_SECRET_PROMPT",
        }

    def query_assessments(self, **kwargs: Any) -> dict[str, Any]:
        self.query_calls.append(kwargs)
        return {
            "items": self.query_items,
            "next_cursor": "cursor-2",
            "raw_page": "RAW_SECRET_PAGE",
        }

    def get_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        self.get_calls.append(assessment_id)
        if assessment_id == "missing":
            return None
        return self.assessment

    def flow_status(self) -> dict[str, Any]:
        return self.status()["flow"]

    def query_flow_entities(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "items": [
                {
                    "schema_version": 1,
                    "entity_id": "entity-1",
                    "kind": "provider_result",
                    "pid": "pid-1",
                    "tenant_bucket_sha256": "1" * 64,
                    "content_sha256": "2" * 64,
                    "version_sha256": "3" * 64,
                    "provenance_sha256": "4" * 64,
                    "baseline_labels": {
                        "sensitivity": "normal",
                        "trust_level": "verified",
                        "integrity": "checked",
                    },
                    "identity_present": True,
                    "identity_mixed": False,
                    "coverage": "complete",
                    "created_at": "2026-08-05T00:00:00Z",
                    "raw_content": "RAW_SECRET_FLOW",
                }
            ],
            "next_cursor": None,
        }

    def query_flow_edges(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "items": [
                {
                    "schema_version": 1,
                    "edge_id": "edge-1",
                    "relation": "direct",
                    "source_node_id": "activity-1",
                    "source_node_type": "activity",
                    "target_node_id": "entity-1",
                    "target_node_type": "entity",
                    "pid": "pid-1",
                    "provenance_sha256": "5" * 64,
                    "created_at": "2026-08-05T00:00:00Z",
                }
            ],
            "next_cursor": None,
        }

    def query_flow_lineage(self, node_id: str, **_kwargs: Any) -> dict[str, Any]:
        entity = self.query_flow_entities()["items"][0]
        edge = self.query_flow_edges()["items"][0]
        return {
            "schema_version": 1,
            "root_node_id": node_id,
            "direction": "upstream",
            "items": [
                {"depth": 1, "edge": edge, "node_type": "entity", "node": entity}
            ],
            "effective_labels": entity["baseline_labels"],
            "coverage": "complete",
            "next_cursor": None,
            "truncated": False,
        }

    def query_machine_settlements(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "items": [
                {
                    "schema_version": 1,
                    "settlement_id": "settlement-1",
                    "assessment_id": "assessment-1",
                    "job_id": "job-1",
                    "request_id": "request-1",
                    "request_revision": 0,
                    "pid": "pid-1",
                    "operation_id": "operation-1",
                    "effect_id": "effect-1",
                    "epoch_id": "epoch-1",
                    "policy_sha256": "6" * 64,
                    "tenant_bucket_sha256": "7" * 64,
                    "action_id": "filesystem.read",
                    "outcome": "require_human",
                    "capability_id": None,
                    "binding_sha256": "8" * 64,
                    "decision_sha256": "9" * 64,
                    "matched_rule_id": None,
                    "reason_codes": ["flow_coverage_incomplete"],
                    "created_at": "2026-08-05T00:00:00Z",
                }
            ],
            "next_cursor": None,
        }

    def query_policy_epochs(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "items": [
                {
                    "schema_version": 1,
                    "epoch_id": "epoch-1",
                    "generation": 1,
                    "catalog_version": 1,
                    "policy_sha256": "6" * 64,
                    "expected_previous_sha256": None,
                    "created_at": "2026-08-05T00:00:00Z",
                }
            ],
            "next_cursor": None,
        }

    def control_status(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "revision": 1,
            "generation": 1,
            "mode": "enforce_deny",
            "active_epoch_id": "epoch-1",
            "active_policy_sha256": "6" * 64,
            "tripped": False,
            "trip_code": None,
            "updated_at": "2026-08-05T00:00:00Z",
        }

    def query_control_history(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "items": [self.control_status()],
            "next_cursor": None,
        }

    def query_health_events(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "items": [
                {
                    "schema_version": 1,
                    "event_id": "health-1",
                    "event_kind": "capture_failed",
                    "severity": "warning",
                    "epoch_id": "epoch-1",
                    "tenant_bucket_sha256": "7" * 64,
                    "evidence_sha256": "a" * 64,
                    "created_at": "2026-08-05T00:00:00Z",
                }
            ],
            "next_cursor": None,
        }

    def metrics(self, **_kwargs: Any) -> dict[str, Any]:
        status = self.status()
        review_metrics = dict(status["review_metrics"])
        review_metrics.update(
            {"issued_reviewed": 0, "issued_review_rate": None}
        )
        return {
            "schema_version": 1,
            "window": None,
            "action_id": None,
            "tenant_bucket_sha256": None,
            "epoch_id": None,
            "risk": None,
            "machine": status["machine"],
            "actual_auto_approval": status["actual_auto_approval"],
            "review_metrics": review_metrics,
        }


class _FakeGuiService:
    def __init__(self, semantic: _FakeSemanticService) -> None:
        self.token = "test-token"
        self.runtime = type(
            "FakeRuntime",
            (),
            {"config": DEFAULT_CONFIG, "semantic": semantic},
        )()

    @contextmanager
    def runtime_user(self, *, serialize: bool = True) -> Iterator[Any]:
        del serialize
        yield self.runtime

    def record_internal_error(
        self,
        _error: BaseException,
        *,
        action: str,
        target: str,
    ) -> dict[str, Any]:
        del action, target
        return {"type": "InternalError", "message": "internal GUI server error"}


@pytest.fixture
def semantic_gui_server() -> Iterator[tuple[Any, _FakeSemanticService]]:
    semantic = _FakeSemanticService()
    server = GuiHTTPServer(("127.0.0.1", 0), _FakeGuiService(semantic))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, semantic
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _request(server: Any, method: str, path: str) -> tuple[int, dict[str, Any]]:
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=30)
    connection.request(
        method,
        path,
        body=b"{}" if method != "GET" else None,
        headers={
            "Authorization": "Bearer test-token",
            "Content-Type": "application/json",
        },
    )
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, payload


def test_semantic_status_is_v3_and_payload_free(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    expected = _FakeSemanticService.status(semantic)
    semantic.status_calls = 0

    status, payload = _request(server, "GET", "/api/semantic/status")

    assert status == 200
    expected.pop("prompt")
    expected["queue"].pop("raw_projection")
    assert payload == expected
    assert "RAW_SECRET" not in json.dumps(payload)
    assert semantic.status_calls == 1


@pytest.mark.parametrize(
    "path",
    (
        "/api/semantic/flow/status",
        "/api/semantic/flow/entities?pid=pid-1&kind=provider_result&limit=1",
        "/api/semantic/flow/edges?relation=direct&node_id=entity-1&limit=1",
        "/api/semantic/flow/lineage/entity-1?direction=upstream&max_depth=8&limit=1",
        "/api/semantic/settlements?outcome=require_human&limit=1",
        "/api/semantic/policy/epochs?limit=1",
        "/api/semantic/control",
        "/api/semantic/control/history?limit=1",
        "/api/semantic/health?severity=warning&limit=1",
        "/api/semantic/metrics?action_id=filesystem.read&risk=low",
    ),
)
def test_semantic_phase_four_read_only_endpoints_are_bounded_and_payload_free(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    path: str,
) -> None:
    server, _semantic = semantic_gui_server

    status, payload = _request(server, "GET", path)

    assert status == 200
    assert payload["schema_version"] in {1, 3}
    assert "RAW_SECRET" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("service_method", "path", "item_field"),
    (
        (
            "query_flow_entities",
            "/api/semantic/flow/entities?limit=1",
            "kind",
        ),
        (
            "query_flow_edges",
            "/api/semantic/flow/edges?limit=1",
            "relation",
        ),
        (
            "query_machine_settlements",
            "/api/semantic/settlements?limit=1",
            "outcome",
        ),
        (
            "query_machine_settlements",
            "/api/semantic/settlements?limit=1",
            "reason_codes",
        ),
        (
            "query_health_events",
            "/api/semantic/health?limit=1",
            "event_kind",
        ),
    ),
)
def test_semantic_read_only_item_enums_fail_closed_without_echo(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    service_method: str,
    path: str,
    item_field: str,
) -> None:
    server, semantic = semantic_gui_server
    page = getattr(semantic, service_method)()
    sentinel = f"RAW_SECRET_{item_field.upper()}_SENTINEL"
    page["items"][0][item_field] = (
        [sentinel] if item_field == "reason_codes" else sentinel
    )
    setattr(semantic, service_method, lambda **_kwargs: page)

    status, payload = _request(server, "GET", path)

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"
    assert sentinel not in json.dumps(payload)


@pytest.mark.parametrize(
    ("service_method", "path", "field"),
    (
        ("control_status", "/api/semantic/control", "trip_code"),
        ("metrics", "/api/semantic/metrics", "risk"),
    ),
)
def test_semantic_read_only_scalar_enums_fail_closed_without_echo(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    service_method: str,
    path: str,
    field: str,
) -> None:
    server, semantic = semantic_gui_server
    value = getattr(semantic, service_method)()
    sentinel = f"RAW_SECRET_{field.upper()}_SENTINEL"
    value[field] = sentinel
    if field == "trip_code":
        value["tripped"] = True
    setattr(semantic, service_method, lambda **_kwargs: value)

    status, payload = _request(server, "GET", path)

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"
    assert sentinel not in json.dumps(payload)


@pytest.mark.parametrize(
    "path",
    (
        "/api/semantic/flow/entities?unknown=value",
        "/api/semantic/flow/entities?limit=101",
        "/api/semantic/flow/edges?relation=write",
        "/api/semantic/flow/lineage/entity-1?max_depth=17",
        "/api/semantic/settlements?outcome=approved",
        "/api/semantic/policy/epochs?state=active",
        "/api/semantic/control?state=active",
        "/api/semantic/health?severity=error",
        "/api/semantic/metrics?tenant_bucket_sha256=not-a-digest",
        "/api/semantic/metrics?risk=unknown",
    ),
)
def test_semantic_phase_four_read_only_endpoints_reject_unknown_or_unbounded_queries(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    path: str,
) -> None:
    server, _semantic = semantic_gui_server

    status, payload = _request(server, "GET", path)

    assert status == 400
    assert payload["ok"] is False


def test_semantic_phase_four_page_requires_explicit_schema_version(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    page = semantic.query_flow_entities()
    page.pop("schema_version")
    semantic.query_flow_entities = lambda **_kwargs: page  # type: ignore[method-assign]

    status, payload = _request(server, "GET", "/api/semantic/flow/entities")

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"


def test_semantic_health_rejects_unknown_host_event_kind_without_echo(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    page = semantic.query_health_events()
    page["items"][0]["event_kind"] = "RAW_SECRET_HEALTH_KIND_SENTINEL"
    semantic.query_health_events = lambda **_kwargs: page  # type: ignore[method-assign]

    status, payload = _request(server, "GET", "/api/semantic/health")

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"
    assert "RAW_SECRET" not in json.dumps(payload)


def test_semantic_http_surface_has_no_review_or_policy_mutation_endpoint(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, _semantic = semantic_gui_server

    for path in (
        "/api/semantic/review/import",
        "/api/semantic/policy/activate",
        "/api/semantic/control/revoke",
    ):
        status, _payload = _request(server, "POST", path)
        assert status == 404


def test_semantic_metrics_exposes_issued_only_review_coverage(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    metrics = semantic.metrics()
    metrics["machine"]["eligible"] = 1
    metrics["machine"]["issued"] = 1
    metrics["actual_auto_approval"] = {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    metrics["review_metrics"] = {
        "reviewed": 1,
        "safe": 1,
        "unsafe": 0,
        "unsafe_rate": 0.0,
        # The one label belongs to a non-issued settlement and therefore must
        # not claim canary grant-review coverage.
        "issued_reviewed": 0,
        "issued_review_rate": 0.0,
    }
    semantic.metrics = lambda **_kwargs: metrics  # type: ignore[method-assign]

    status, payload = _request(server, "GET", "/api/semantic/metrics")

    assert status == 200
    assert payload["review_metrics"]["reviewed"] == 1
    assert payload["review_metrics"]["issued_reviewed"] == 0
    assert payload["review_metrics"]["issued_review_rate"] == 0.0


def test_semantic_metrics_rejects_review_coverage_above_issued_grants(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    metrics = semantic.metrics()
    metrics["review_metrics"].update(
        {"issued_reviewed": 1, "issued_review_rate": 1.0}
    )
    semantic.metrics = lambda **_kwargs: metrics  # type: ignore[method-assign]

    status, payload = _request(server, "GET", "/api/semantic/metrics")

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"


def test_semantic_metrics_rejects_ratio_denominator_drift_from_eligible(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    metrics = semantic.metrics()
    metrics["actual_auto_approval"] = {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
    }
    semantic.metrics = lambda **_kwargs: metrics  # type: ignore[method-assign]

    status, payload = _request(server, "GET", "/api/semantic/metrics")

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"


def test_semantic_status_rejects_inconsistent_actual_auto_approval_metrics(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    semantic.actual_auto_approval = {
        "numerator": 2,
        "denominator": 1,
        "rate": 1.0,
    }

    status, payload = _request(server, "GET", "/api/semantic/status")

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"


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
        (("assessments", "success"), "7"),
        (("assessments", "error"), -1),
        (("assessments", "by_status", "success"), True),
        (("assessments", "by_domain", "filesystem"), "4"),
        (("actual_auto_approval", "numerator"), 1),
        (("actual_auto_approval", "denominator"), 2),
        (("actual_auto_approval", "rate"), 0.5),
    ],
)
def test_semantic_status_api_rejects_malformed_v3_fields(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    path: tuple[str, ...],
    invalid: object,
) -> None:
    server, semantic = semantic_gui_server
    payload = _FakeSemanticService.status(semantic)
    target: dict[str, Any] = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid
    semantic.status = lambda: payload  # type: ignore[method-assign]

    status, response = _request(server, "GET", "/api/semantic/status")

    assert status == 500
    assert response["error"]["code"] == "invalid_semantic_service_response"


@pytest.mark.parametrize("aggregate", ["by_status", "by_domain"])
def test_semantic_status_api_requires_complete_aggregate_mappings(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    aggregate: str,
) -> None:
    server, semantic = semantic_gui_server
    payload = _FakeSemanticService.status(semantic)
    payload["assessments"][aggregate].pop(next(iter(payload["assessments"][aggregate])))
    semantic.status = lambda: payload  # type: ignore[method-assign]

    status, response = _request(server, "GET", "/api/semantic/status")

    assert status == 500
    assert response["error"]["code"] == "invalid_semantic_service_response"


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("assessments", "success"), 8),
        (("assessments", "would_issue_exact_once"), 2),
        (("assessments", "ood"), 3),
    ],
)
def test_semantic_status_api_rejects_inconsistent_derived_totals(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    path: tuple[str, ...],
    invalid: int,
) -> None:
    server, semantic = semantic_gui_server
    payload = _FakeSemanticService.status(semantic)
    target: dict[str, Any] = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid
    semantic.status = lambda: payload  # type: ignore[method-assign]

    status, response = _request(server, "GET", "/api/semantic/status")

    assert status == 500
    assert response["error"]["code"] == "invalid_semantic_service_response"


def test_semantic_assessment_list_forwards_strict_filters_and_sanitizes_items(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server

    status, payload = _request(
        server,
        "GET",
        "/api/semantic/assessments?pid=pid-1&request_id=request-1"
        "&operation_id=operation-1&kind=approval&status=success"
        "&domain=filesystem&action_id=filesystem.read"
        f"&tenant_bucket_sha256={'6' * 64}&after=cursor-1&limit=7",
    )

    assert status == 200
    assert semantic.query_calls == [
        {
            "pid": "pid-1",
            "request_id": "request-1",
            "operation_id": "operation-1",
            "kind": "approval",
            "status": "success",
            "domain": "filesystem",
            "action_id": "filesystem.read",
            "tenant_bucket_sha256": "6" * 64,
            "after": "cursor-1",
            "limit": 7,
        }
    ]
    assert payload["schema_version"] == 1
    assert payload["next_cursor"] == "cursor-2"
    assert len(payload["items"]) == 1
    assert set(payload["items"][0]) == {
        "assessment_id",
        "job_id",
        "kind",
        "status",
        "domain",
        "action_id",
        "pid",
        "request_id",
        "operation_id",
        "effect_id",
        "shadow_outcome",
        "reason_codes",
        "ood",
        "abstain",
        "confidence_bps",
        "calibration_bucket",
        "input_tokens",
        "output_tokens",
        "cost_microunits",
        "classifier_id",
        "classifier_version",
        "artifact_sha256",
        "input_sha256",
        "feature_snapshot_sha256",
        "policy_sha256",
        "tenant_bucket_sha256",
        "created_at",
        "completed_at",
        "latency_ms",
        "human_outcome",
    }
    assert payload["items"][0]["action_id"] == "filesystem.read"
    assert payload["items"][0]["calibration_bucket"] == "high"
    assert payload["items"][0]["input_tokens"] == 120
    assert payload["items"][0]["tenant_bucket_sha256"] == "6" * 64
    assert payload["items"][0]["human_outcome"] == "approved"
    assert "RAW_SECRET" not in json.dumps(payload)


def test_semantic_assessment_list_uses_bounded_default_page(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server

    status, _payload = _request(server, "GET", "/api/semantic/assessments")

    assert status == 200
    assert semantic.query_calls == [
        {
            "pid": None,
            "request_id": None,
            "operation_id": None,
            "kind": None,
            "status": None,
            "domain": None,
            "action_id": None,
            "tenant_bucket_sha256": None,
            "after": None,
            "limit": 50,
        }
    ]


def test_semantic_assessment_cost_and_tenant_fields_are_nullable(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    semantic.query_items = [
        {
            **semantic.assessment,
            "input_tokens": None,
            "output_tokens": None,
            "cost_microunits": None,
            "tenant_bucket_sha256": None,
        }
    ]

    status, payload = _request(server, "GET", "/api/semantic/assessments")

    assert status == 200
    item = payload["items"][0]
    assert item["input_tokens"] is None
    assert item["output_tokens"] is None
    assert item["cost_microunits"] is None
    assert item["tenant_bucket_sha256"] is None


def test_semantic_assessment_projection_rejects_nested_payload_in_scalar_field(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    semantic.query_items = [
        {
            **semantic.assessment,
            "classifier_id": {"secret": "RAW_SECRET_NESTED_SCALAR"},
        }
    ]

    status, payload = _request(server, "GET", "/api/semantic/assessments")

    assert status == 500
    assert "RAW_SECRET" not in json.dumps(payload)


def test_semantic_assessment_projection_rejects_arbitrary_human_outcome(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    semantic.query_items = [
        {
            **semantic.assessment,
            "human_outcome": "RAW_SECRET_HUMAN_OUTCOME",
        }
    ]

    status, payload = _request(server, "GET", "/api/semantic/assessments")

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"
    assert "RAW_SECRET" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("kind", "RAW_SECRET_KIND_SENTINEL"),
        ("status", "RAW_SECRET_STATUS_SENTINEL"),
        ("domain", "RAW_SECRET_DOMAIN_SENTINEL"),
        ("shadow_outcome", "RAW_SECRET_OUTCOME_SENTINEL"),
        ("calibration_bucket", "RAW_SECRET_CALIBRATION_SENTINEL"),
        ("reason_codes", ["RAW_SECRET_REASON_CODE_SENTINEL"]),
        ("human_outcome", "RAW_SECRET_HUMAN_OUTCOME_SENTINEL"),
    ),
)
def test_semantic_assessment_summary_rejects_unknown_enums_without_echo(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    field: str,
    invalid: object,
) -> None:
    server, semantic = semantic_gui_server
    semantic.query_items = [{**semantic.assessment, field: invalid}]

    status, payload = _request(server, "GET", "/api/semantic/assessments")

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"
    assert "RAW_SECRET" not in json.dumps(payload)


def test_semantic_assessment_page_cannot_exceed_requested_limit(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    semantic.query_items = [
        semantic.assessment,
        {**semantic.assessment, "assessment_id": "assessment-2"},
    ]

    status, payload = _request(
        server,
        "GET",
        "/api/semantic/assessments?limit=1",
    )

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"


def test_semantic_assessment_projection_accepts_json_safe_integer_ceiling(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    ceiling = (1 << 53) - 1
    semantic.query_items = [
        {
            **semantic.assessment,
            "input_tokens": ceiling,
            "output_tokens": ceiling,
            "cost_microunits": ceiling,
            "latency_ms": ceiling,
        }
    ]

    status, payload = _request(server, "GET", "/api/semantic/assessments")

    assert status == 200
    item = payload["items"][0]
    assert item["input_tokens"] == ceiling
    assert item["output_tokens"] == ceiling
    assert item["cost_microunits"] == ceiling
    assert item["latency_ms"] == ceiling


@pytest.mark.parametrize(
    "field",
    ["input_tokens", "output_tokens", "cost_microunits", "latency_ms"],
)
def test_semantic_assessment_projection_rejects_non_safe_integer_metrics(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    field: str,
) -> None:
    server, semantic = semantic_gui_server
    semantic.query_items = [{**semantic.assessment, field: 1 << 53}]

    status, payload = _request(server, "GET", "/api/semantic/assessments")

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"


@pytest.mark.parametrize(
    "suffix",
    [
        "?unknown=value",
        "?pid=one&pid=two",
        "?after=",
        "?limit=0",
        "?limit=101",
        "?limit=not-an-integer",
        "?after=" + ("x" * 2049),
        "?after=line%0Abreak",
        "?pid=line%0Abreak",
        "?request_id=" + ("x" * 513),
        "?operation_id=line%09break",
        "?operation_id=del%7Fchar",
        "?kind=unknown",
        "?status=unknown",
        "?domain=unknown-domain",
        "?action_id=filesystem",
        "?action_id=Filesystem.Read",
        "?tenant_bucket_sha256=" + ("a" * 63),
        "?tenant_bucket_sha256=" + ("A" * 64),
    ],
)
def test_semantic_assessment_list_rejects_invalid_query_before_service_call(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    suffix: str,
) -> None:
    server, semantic = semantic_gui_server

    status, payload = _request(server, "GET", f"/api/semantic/assessments{suffix}")

    assert status == 400
    assert "error" in payload
    assert semantic.query_calls == []


def test_semantic_assessment_detail_is_typed_and_payload_free(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server

    status, payload = _request(server, "GET", "/api/semantic/assessments/assessment-1")

    assert status == 200
    assert semantic.get_calls == ["assessment-1"]
    assessment = payload["assessment"]
    assert payload["schema_version"] == 1
    assert assessment["findings"] == [
        {
            "code": "sensitive_data",
            "severity": "high",
            "confidence_bps": 8700,
            "evidence_sha256": "e" * 64,
            "source": "model",
        }
    ]
    assert assessment["data_findings"] == [
        {
            "category": "credential",
            "field": "approval.request",
            "span_start": None,
            "span_end": None,
            "sensitivity_floor": "secret",
            "integrity_ceiling": "unknown",
            "trust_ceiling": "untrusted",
            "confidence_bps": 9500,
            "evidence_sha256": "f" * 64,
        }
    ]
    assert assessment["source_refs_sha256"] == "1" * 64
    assert assessment["manifest_sha256"] == "7" * 64
    assert assessment["action_sha256"] == "8" * 64
    assert assessment["resource_sha256"] == "9" * 64
    assert assessment["args_sha256"] == "0" * 64
    assert assessment["state_sha256"] == "a" * 64
    assert assessment["projection_sha256"] == "b" * 64
    assert "RAW_SECRET" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("code", "RAW_SECRET_FINDING_CODE_SENTINEL"),
        ("severity", "RAW_SECRET_FINDING_SEVERITY_SENTINEL"),
        ("source", "RAW_SECRET_FINDING_SOURCE_SENTINEL"),
        ("confidence_bps", "RAW_SECRET_FINDING_CONFIDENCE_SENTINEL"),
        ("evidence_sha256", "RAW_SECRET_FINDING_EVIDENCE_SENTINEL"),
    ),
)
def test_semantic_assessment_detail_rejects_unknown_finding_fields_without_echo(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    field: str,
    invalid: object,
) -> None:
    server, semantic = semantic_gui_server
    finding = {**semantic.assessment["findings"][0], field: invalid}
    semantic.assessment = {**semantic.assessment, "findings": [finding]}

    status, payload = _request(
        server,
        "GET",
        "/api/semantic/assessments/assessment-1",
    )

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"
    assert "RAW_SECRET" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("category", "RAW_SECRET_DATA_CATEGORY_SENTINEL"),
        ("sensitivity_floor", "RAW_SECRET_SENSITIVITY_SENTINEL"),
        ("integrity_ceiling", "RAW_SECRET_INTEGRITY_SENTINEL"),
        ("trust_ceiling", "RAW_SECRET_TRUST_SENTINEL"),
    ),
)
def test_semantic_assessment_detail_rejects_unknown_data_enums_without_echo(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    field: str,
    invalid: object,
) -> None:
    server, semantic = semantic_gui_server
    finding = {**semantic.assessment["data_findings"][0], field: invalid}
    semantic.assessment = {**semantic.assessment, "data_findings": [finding]}

    status, payload = _request(
        server,
        "GET",
        "/api/semantic/assessments/assessment-1",
    )

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"
    assert "RAW_SECRET" not in json.dumps(payload)


@pytest.mark.parametrize("field", ("proven_predicates", "missing_predicates"))
def test_semantic_assessment_detail_rejects_unknown_predicates_without_echo(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    field: str,
) -> None:
    server, semantic = semantic_gui_server
    semantic.assessment = {
        **semantic.assessment,
        field: ["RAW_SECRET_PREDICATE_SENTINEL"],
    }

    status, payload = _request(
        server,
        "GET",
        "/api/semantic/assessments/assessment-1",
    )

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"
    assert "RAW_SECRET" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("field", "span_start", "span_end"),
    [
        ("RAW_SECRET_INVENTED_LOCATOR", None, None),
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
def test_semantic_assessment_detail_rejects_invalid_data_locator_without_echo(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    field: str,
    span_start: int | None,
    span_end: int | None,
) -> None:
    server, semantic = semantic_gui_server
    semantic.assessment["data_findings"][0].update(
        field=field,
        span_start=span_start,
        span_end=span_end,
    )

    status, payload = _request(server, "GET", "/api/semantic/assessments/assessment-1")

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"
    assert "RAW_SECRET" not in json.dumps(payload)


def test_semantic_assessment_detail_allows_nullable_optional_provenance_digests(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, semantic = semantic_gui_server
    for field in (
        "manifest_sha256",
        "resource_sha256",
        "args_sha256",
        "state_sha256",
    ):
        semantic.assessment[field] = None

    status, payload = _request(server, "GET", "/api/semantic/assessments/assessment-1")

    assert status == 200
    assessment = payload["assessment"]
    assert assessment["manifest_sha256"] is None
    assert assessment["resource_sha256"] is None
    assert assessment["args_sha256"] is None
    assert assessment["state_sha256"] is None
    assert assessment["action_sha256"] == "8" * 64
    assert assessment["projection_sha256"] == "b" * 64


@pytest.mark.parametrize("field", ["action_sha256", "projection_sha256"])
@pytest.mark.parametrize("value", [None, "RAW_SECRET_INVALID_DIGEST"])
def test_semantic_assessment_detail_requires_valid_binding_digests_without_echo(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    field: str,
    value: object,
) -> None:
    server, semantic = semantic_gui_server
    semantic.assessment[field] = value

    status, payload = _request(server, "GET", "/api/semantic/assessments/assessment-1")

    assert status == 500
    assert payload["error"]["code"] == "invalid_semantic_service_response"
    assert "RAW_SECRET" not in json.dumps(payload)


def test_semantic_assessment_detail_returns_404_when_missing(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
) -> None:
    server, _semantic = semantic_gui_server

    status, payload = _request(server, "GET", "/api/semantic/assessments/missing")

    assert status == 404
    assert "not found" in payload["error"]["message"]


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/api/semantic/status"),
        ("POST", "/api/semantic/assessments"),
        ("PUT", "/api/semantic/assessments/assessment-1"),
        ("DELETE", "/api/semantic/assessments/assessment-1"),
        ("POST", "/api/semantic/flow/entities"),
        ("PUT", "/api/semantic/flow/lineage/entity-1"),
        ("POST", "/api/semantic/settlements"),
        ("POST", "/api/semantic/policy/epochs"),
        ("POST", "/api/semantic/control"),
        ("DELETE", "/api/semantic/control/history"),
        ("POST", "/api/semantic/health"),
        ("PUT", "/api/semantic/metrics"),
    ],
)
def test_semantic_api_has_no_write_routes(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    method: str,
    path: str,
) -> None:
    server, semantic = semantic_gui_server

    status, _payload = _request(server, method, path)

    assert status == 404
    assert semantic.status_calls == 0
    assert semantic.query_calls == []
    assert semantic.get_calls == []


@pytest.mark.parametrize(
    "path",
    [
        "/api/semantic/status?unknown=",
        "/api/semantic/assessments/assessment-1?unknown=value",
    ],
)
def test_semantic_non_collection_routes_reject_query_parameters(
    semantic_gui_server: tuple[Any, _FakeSemanticService],
    path: str,
) -> None:
    server, semantic = semantic_gui_server

    status, _payload = _request(server, "GET", path)

    assert status == 400
    assert semantic.status_calls == 0
    assert semantic.get_calls == []
