from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import threading
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import pytest

from agent_libos import Runtime
from agent_libos.config import (
    AgentLibOSConfig,
    DataFlowDefaults,
    SemanticDefaults,
)
from agent_libos.models import (
    DeterministicDenyDecision,
    EventType,
    HumanRequest,
    HumanRequestStatus,
    ObjectMetadata,
    ObjectType,
    ProcessStatus,
    SemanticControlStateV1,
    SemanticHardDenyRuleV1,
    SemanticPolicyEpochV1,
    SemanticReasonCode,
    SemanticRuntimeMode,
    SemanticTripCode,
    SinkTrustLevel,
    SinkTrustRule,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.capability.effect_binding import canonical_effect_hash
from agent_libos.semantic.exact_request import (
    decode_exact_semantic_approval_request,
)
from agent_libos.semantic.enforcement import SemanticAuthorityControlView
from agent_libos.utils.ids import utc_now


pytestmark = pytest.mark.security


def _external_request(runtime: Runtime, *, malformed: bool = False) -> tuple[str, str]:
    pid = runtime.process.spawn(goal="exercise semantic Human settlement")
    payload: dict[str, Any] = {
        "type": "external_operation_approval",
        "question": "Allow the exact read?",
        "requested_once_capability": {
            "subject": pid,
            "resource": "filesystem:workspace:reports/phase3.txt",
            "rights": ["read"],
            "constraints": {},
        },
        "context": {
            "adapter": "filesystem",
            "authority_operation": "filesystem.read",
            "primitive": "runtime.filesystem.read_text",
            "operation": "read_text",
            "pid": pid,
            "resource": "filesystem:workspace:reports/phase3.txt",
            "right": "read",
            "risk": "harmless",
            "target_state_version": None,
        },
    }
    request_id = runtime.human.query_authority_request(
        pid,
        "owner",
        payload,
        blocking=True,
        authority_origin="external_operation",
    )
    if malformed:
        with runtime.human.requests.transaction():
            current = runtime.human.get(request_id)
            malformed_payload = dict(current.payload)
            malformed_payload.pop("context", None)
            malformed_payload.pop("effect_binding", None)
            runtime.human.requests.replace_current(
                current,
                payload=malformed_payload,
            )
    return pid, request_id


def _replace_payload(
    runtime: Runtime,
    request_id: str,
    mutation: str,
) -> HumanRequest:
    with runtime.human.requests.transaction():
        current = runtime.human.get(request_id)
        payload = deepcopy(current.payload)
        once = payload["requested_once_capability"]
        constraints = once["constraints"]
        if mutation == "duplicate_rights":
            once["rights"] = ["read", "read"]
        elif mutation == "multiple_rights":
            once["rights"] = ["read", "write"]
        elif mutation == "unknown_payload_field":
            payload["machine_allow"] = True
        elif mutation == "unknown_once_field":
            once["uses"] = 1
        elif mutation == "delegable":
            once["delegable"] = True
        elif mutation == "invalid_expiry":
            once["expires_at"] = "not-a-time"
        elif mutation == "naive_expiry":
            once["expires_at"] = "2026-08-07T12:00:00"
        elif mutation == "unknown_constraint":
            constraints["semantic_magic"] = "allow"
        elif mutation == "null_constraint":
            constraints["git_remote"] = None
        else:  # pragma: no cover - test helper misuse
            raise AssertionError(f"unknown mutation: {mutation}")
        return runtime.human.requests.replace_current(
            current,
            payload=payload,
            updated_at=utc_now(),
        )


def _replace_exact_operation(
    runtime: Runtime,
    request_id: str,
    *,
    action_id: str,
    right: str,
) -> HumanRequest:
    with runtime.human.requests.transaction():
        current = runtime.human.get(request_id)
        payload = deepcopy(current.payload)
        context = payload["context"]
        once = payload["requested_once_capability"]
        context["authority_operation"] = action_id
        context["right"] = right
        once["rights"] = [right]
        binding = {
            "effect_id": payload["effect_binding"]["effect_id"],
            "canonical_args_hash": canonical_effect_hash(context),
            "target_state_version": context.get("target_state_version"),
        }
        payload["effect_binding"] = binding
        once["constraints"] = {"approval_binding": dict(binding)}
        return runtime.human.requests.replace_current(
            current,
            payload=payload,
            updated_at=utc_now(),
        )


def _activate_deny_preflight(
    runtime: Runtime,
    *,
    tripped: bool = False,
    matching_rule: bool = False,
) -> None:
    epoch = SemanticPolicyEpochV1(
        epoch_id="epoch-phase3-test",
        generation=1,
        expected_previous_sha256=None,
        tenant_bucket_sha256s=(),
        auto_approval_rules=(),
        hard_deny_rules=(
            SemanticHardDenyRuleV1(
                rule_id="unrelated-hard-deny",
                authority_operation=(
                    "filesystem.read" if matching_rule else "filesystem.delete"
                ),
                resource=(
                    "filesystem:workspace:reports/phase3.txt"
                    if matching_rule
                    else "filesystem:workspace:never/*"
                ),
                rights=(("read",) if matching_rule else ("delete",)),
            ),
        ),
        created_at=utc_now(),
    )
    control = SemanticControlStateV1(
        revision=1 if tripped else 0,
        generation=epoch.generation,
        mode=SemanticRuntimeMode.ENFORCE_DENY,
        active_epoch_id=epoch.epoch_id,
        active_policy_sha256=epoch.canonical_sha256(),
        tripped=tripped,
        trip_code=(SemanticTripCode.UNSAFE_REVIEW if tripped else None),
        updated_at=utc_now(),
    )

    class _Control:
        def authority_view(self) -> SemanticAuthorityControlView:
            return SemanticAuthorityControlView(control=control, epoch=epoch)

    runtime.semantic._config = replace(
        runtime.semantic._config,
        mode=SemanticRuntimeMode.ENFORCE_DENY.value,
        policy_epoch=epoch,
    )
    runtime.semantic._mode = SemanticRuntimeMode.ENFORCE_DENY.value
    runtime.semantic._control = _Control()
    runtime.semantic._hard_deny_facts_resolver = None


def _enforce_deny_config(
    *,
    sink_rules: tuple[SinkTrustRule, ...] = (),
) -> AgentLibOSConfig:
    epoch = SemanticPolicyEpochV1(
        epoch_id="epoch-phase3-production",
        generation=1,
        expected_previous_sha256=None,
        tenant_bucket_sha256s=(),
        auto_approval_rules=(),
        hard_deny_rules=(
            SemanticHardDenyRuleV1(
                rule_id="deny-unrelated-delete",
                authority_operation="filesystem.delete",
                resource="filesystem:workspace:never/*",
                rights=("delete",),
            ),
        ),
        created_at=utc_now(),
    )
    return AgentLibOSConfig(
        data_flow=DataFlowDefaults(sink_rules=sink_rules),
        semantic=SemanticDefaults(
            mode="enforce_deny",
            adapter="deterministic",
            policy_epoch=epoch,
            max_concurrency=1,
        )
    )


def _external_write_request(
    runtime: Runtime,
    *,
    pid: str | None = None,
    source_oids: tuple[str, ...] = (),
) -> tuple[str, str]:
    selected_pid = pid or runtime.process.spawn(
        goal="exercise semantic DataFlow denial"
    )
    resource = "filesystem:workspace:reports/phase3-output.txt"
    request_id = runtime.human.query_authority_request(
        selected_pid,
        "owner",
        {
            "type": "external_operation_approval",
            "question": "Allow the exact write?",
            "requested_once_capability": {
                "subject": selected_pid,
                "resource": resource,
                "rights": ["write"],
                "constraints": {},
            },
            "context": {
                "adapter": "filesystem",
                "authority_operation": "filesystem.write",
                "primitive": "runtime.filesystem.write_text",
                "operation": "write_text",
                "pid": selected_pid,
                "path": "reports/phase3-output.txt",
                "resource": resource,
                "right": "write",
                "risk": "high",
                "target_state_version": None,
            },
        },
        blocking=True,
        authority_origin="external_operation",
        source_oids=source_oids,
    )
    return selected_pid, request_id


def _assess_request(runtime: Runtime, request_id: str) -> Any:
    for _attempt in range(8):
        page = runtime.uow.semantic.query_semantic_assessments(
            after=None,
            limit=2,
            request_id=request_id,
        )
        if page.records:
            assert len(page.records) == 1
            return page.records[0]
        assert runtime.semantic.process_one()
    raise AssertionError("semantic request assessment did not complete")


def _replace_exact_context_field(
    runtime: Runtime,
    request_id: str,
    *,
    key: str,
    value: Any,
) -> HumanRequest:
    with runtime.human.requests.transaction():
        current = runtime.human.get(request_id)
        payload = deepcopy(current.payload)
        context = payload["context"]
        context[key] = value
        binding = {
            "effect_id": payload["effect_binding"]["effect_id"],
            "canonical_args_hash": canonical_effect_hash(context),
            "target_state_version": context.get("target_state_version"),
        }
        payload["effect_binding"] = binding
        payload["requested_once_capability"]["constraints"][
            "approval_binding"
        ] = dict(binding)
        return runtime.human.requests.replace_current(
            current,
            payload=payload,
            updated_at=utc_now(),
        )


def _deny(runtime: Runtime, request: HumanRequest) -> DeterministicDenyDecision:
    return DeterministicDenyDecision(
        request_id=request.request_id,
        request_revision=request.revision,
        pid=request.pid,
        effect_id=runtime.human._semantic_effect_identity(request),
        reason_codes=(SemanticReasonCode.POLICY_HARD_DENY,),
        policy_sha256="1" * 64,
        evidence_sha256="2" * 64,
        decided_at=utc_now(),
    )


def _bind_policy(
    runtime: Runtime,
    *,
    deny: bool,
    recorder: Any | None = None,
) -> None:
    runtime.human._host_semantic_policy_preflight = (
        (lambda request: _deny(runtime, request))
        if deny
        else (lambda _request: None)
    )
    runtime.human._host_semantic_mode_reader = lambda: "enforce_deny"
    runtime.human._host_semantic_settlement_recorder = recorder or (
        lambda _request, **_kwargs: {"settlement_id": "semset_test"}
    )


def test_canonical_preview_is_identity_safe_revision_bound_and_optional_off() -> None:
    runtime = Runtime.open("local")
    try:
        pid, request_id = _external_request(runtime)
        request = runtime.human.get(request_id)
        view = runtime.human.public_request_view(request)

        preview = view["approval_preview"]
        assert set(preview) == {
            "schema_version",
            "request_id",
            "revision",
            "pid",
            "action_id",
            "resource_display",
            "resource_sha256",
            "rights",
            "effect_id",
            "canonical_args_sha256",
            "argument_projection",
            "target_state_sha256",
            "risk",
            "source_labels",
            "expires_at",
        }
        assert preview["request_id"] == request_id
        assert preview["revision"] == request.revision
        assert preview["pid"] == pid
        assert preview["risk"] == "low"
        assert set(preview["source_labels"]) == {
            "sensitivity",
            "integrity",
            "trust_level",
            "identity_present",
            "identity_mixed",
        }
        assert len(view["preview_sha256"]) == 64

        # The default-off compatibility path accepts the historical response
        # shape and still never persists the derived preview into the payload.
        approved = runtime.human.approve(request_id, {"approved": True})
        assert approved.status == HumanRequestStatus.APPROVED
        assert "approval_preview" not in approved.payload
        assert "preview_sha256" not in approved.payload
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_rights",
        "multiple_rights",
        "unknown_payload_field",
        "unknown_once_field",
        "delegable",
        "invalid_expiry",
        "naive_expiry",
        "unknown_constraint",
        "null_constraint",
    ),
)
def test_exact_request_decoder_rejects_authority_shape_ambiguity(
    mutation: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        _pid, request_id = _external_request(runtime)
        request = _replace_payload(runtime, request_id, mutation)

        with pytest.raises(ValidationError):
            decode_exact_semantic_approval_request(request)

        _activate_deny_preflight(runtime)
        decision = runtime.semantic.deterministic_deny_preflight(request)
        assert decision is not None
        assert decision.reason_codes == (SemanticReasonCode.MALFORMED_REQUEST,)
        assert runtime.human.public_request_view(request).get("approval_preview") is None
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("action_id", "right"),
    (
        ("future.read", "read"),
        ("filesystem.write", "write"),
    ),
)
def test_well_formed_unsupported_or_high_risk_request_remains_with_human(
    action_id: str,
    right: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        _pid, request_id = _external_request(runtime)
        request = _replace_exact_operation(
            runtime,
            request_id,
            action_id=action_id,
            right=right,
        )
        decoded = decode_exact_semantic_approval_request(request)
        assert decoded.action_id == action_id
        assert decoded.right == right

        _activate_deny_preflight(runtime)
        assert runtime.semantic.deterministic_deny_preflight(request) is None
        assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
    finally:
        runtime.close()


def test_real_semantic_manager_malformed_deny_terminalizes_machine_policy() -> None:
    runtime = Runtime.open("local")
    try:
        pid, request_id = _external_request(runtime)
        request = _replace_payload(runtime, request_id, "duplicate_rights")
        _activate_deny_preflight(runtime)
        runtime.human._host_semantic_policy_preflight = (
            runtime.semantic.deterministic_deny_preflight
        )
        runtime.human._host_semantic_mode_reader = lambda: "enforce_deny"
        runtime.human._host_semantic_settlement_recorder = (
            lambda *_args, **_kwargs: {
                "schema_version": 1,
                "settlement_id": "semset_real_manager_deny",
                "outcome": "denied",
            }
        )

        result = runtime.human.approve(
            request_id,
            {"approved": True, "source": "test_human"},
        )

        assert result.status is HumanRequestStatus.REJECTED
        assert result.decision is not None
        assert result.decision["source"] == "machine_policy"
        assert result.decision["deterministic_deny"]["reason_codes"] == [
            SemanticReasonCode.MALFORMED_REQUEST.value
        ]
        assert result.decision["settlement_receipt"]["outcome"] == "denied"
        assert runtime.process.get(pid).status is ProcessStatus.RUNNABLE
        assert not any(
            capability.resource == "filesystem:workspace:reports/phase3.txt"
            for capability in runtime.capability.list_subject(pid)
        )
    finally:
        runtime.close()


def test_production_runtime_rechecks_captured_binding_inside_human_approve() -> None:
    runtime = Runtime.open("local", config=_enforce_deny_config())
    try:
        # Keep the queued capture stable so this test exercises the Human
        # response path rather than racing the background assessment worker.
        assert runtime.semantic.shutdown()
        pid, request_id = _external_request(runtime)
        request = _replace_exact_context_field(
            runtime,
            request_id,
            key="operation",
            value="read_bytes",
        )
        view = runtime.human.public_request_view(request)

        result = runtime.human.approve(
            request_id,
            {"approved": True, "source": "test_human"},
            expected_revision=request.revision,
            preview_sha256=view["preview_sha256"],
        )

        assert result.status is HumanRequestStatus.REJECTED
        assert result.decision is not None
        assert result.decision["source"] == "machine_policy"
        assert result.decision["deterministic_deny"]["reason_codes"] == [
            SemanticReasonCode.STALE_BINDING.value
        ]
        assert runtime.process.get(pid).status is ProcessStatus.RUNNABLE
        assert not any(
            capability.resource == "filesystem:workspace:reports/phase3.txt"
            for capability in runtime.capability.list_subject(pid)
        )
        settlements = runtime.uow.semantic.query_semantic_machine_settlements(
            after=None,
            limit=2,
            request_id=request_id,
        )
        assert len(settlements.records) == 1
        assert settlements.records[0].outcome == "denied"
        assert settlements.records[0].reason_codes == (
            SemanticReasonCode.STALE_BINDING,
        )
    finally:
        runtime.close()


def test_production_runtime_executes_only_explicit_data_flow_deny() -> None:
    resource = "filesystem:workspace:reports/phase3-output.txt"
    tenant = "phase3-source-tenant"
    runtime = Runtime.open(
        "local",
        config=_enforce_deny_config(
            sink_rules=(
                SinkTrustRule(
                    pattern=resource,
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="normal",
                    tenants=("different-tenant",),
                ),
                SinkTrustRule(
                    pattern="human:owner:terminal",
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="normal",
                    tenants=(tenant,),
                ),
            )
        ),
    )
    try:
        assert runtime.semantic.shutdown()
        pid = runtime.process.spawn(goal="create exact tenant source")
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"value": "ordinary source"},
            metadata=ObjectMetadata(tenant=tenant),
        )
        pid, request_id = _external_write_request(
            runtime,
            pid=pid,
            source_oids=(source.oid,),
        )
        request = runtime.human.get(request_id)
        view = runtime.human.public_request_view(request)

        result = runtime.human.approve(
            request_id,
            {"approved": True, "source": "test_human"},
            expected_revision=request.revision,
            preview_sha256=view["preview_sha256"],
        )

        assert result.status is HumanRequestStatus.REJECTED
        assert result.decision is not None
        assert result.decision["deterministic_deny"]["reason_codes"] == [
            SemanticReasonCode.DATA_FLOW_DENIED.value
        ]
        assert not any(
            capability.resource == resource
            for capability in runtime.capability.list_subject(pid)
        )
    finally:
        runtime.close()


def test_worker_executes_data_flow_deny_outside_auto_catalog() -> None:
    """Phase 3 deny predicates are independent of Phase 4's allow catalog."""

    resource = "filesystem:workspace:reports/phase3-output.txt"
    tenant = "phase3-worker-source-tenant"
    runtime = Runtime.open(
        "local",
        config=_enforce_deny_config(
            sink_rules=(
                SinkTrustRule(
                    pattern=resource,
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="normal",
                    tenants=("different-tenant",),
                ),
                SinkTrustRule(
                    pattern="human:owner:terminal",
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="normal",
                    tenants=(tenant,),
                ),
            )
        ),
    )
    try:
        assert runtime.semantic.shutdown()
        pid = runtime.process.spawn(goal="exercise worker DataFlow denial")
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"value": "ordinary source"},
            metadata=ObjectMetadata(tenant=tenant),
        )
        _pid, request_id = _external_write_request(
            runtime,
            pid=pid,
            source_oids=(source.oid,),
        )

        for _attempt in range(8):
            if runtime.human.get(request_id).status is not HumanRequestStatus.PENDING:
                break
            assert runtime.semantic.process_one()

        result = runtime.human.get(request_id)
        assert result.status is HumanRequestStatus.REJECTED
        assert result.decision is not None
        assert result.decision["source"] == "machine_policy"
        assert result.decision["deterministic_deny"]["reason_codes"] == [
            SemanticReasonCode.DATA_FLOW_DENIED.value
        ]
        assert not any(
            capability.resource == resource
            for capability in runtime.capability.list_subject(pid)
        )
    finally:
        runtime.close()


def test_unconfigured_data_flow_sink_remains_with_human() -> None:
    tenant = "phase3-unconfigured-sink-tenant"
    runtime = Runtime.open(
        "local",
        config=_enforce_deny_config(
            sink_rules=(
                SinkTrustRule(
                    pattern="human:owner:terminal",
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="normal",
                    tenants=(tenant,),
                ),
            )
        ),
    )
    try:
        assert runtime.semantic.shutdown()
        pid = runtime.process.spawn(goal="retain unconfigured Sink with Human")
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"value": "ordinary source"},
            metadata=ObjectMetadata(tenant=tenant),
        )
        _pid, request_id = _external_write_request(
            runtime,
            pid=pid,
            source_oids=(source.oid,),
        )
        request = runtime.human.get(request_id)

        assert runtime.semantic.deterministic_deny_preflight(request) is None
        for _attempt in range(8):
            page = runtime.uow.semantic.query_semantic_assessments(
                after=None,
                limit=2,
                request_id=request_id,
            )
            if page.records:
                break
            assert runtime.semantic.process_one()

        assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
        assert (
            runtime.uow.semantic.query_semantic_machine_settlements(
                after=None,
                limit=2,
                request_id=request_id,
            ).records
            == ()
        )
    finally:
        runtime.close()


def test_production_data_flow_resolver_error_stays_with_human(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local", config=_enforce_deny_config())
    try:
        assert runtime.semantic.shutdown()
        _pid, request_id = _external_write_request(runtime)
        request = runtime.human.get(request_id)

        def fail_registry(_sink: Any) -> Any:
            raise RuntimeError("injected sink registry outage")

        monkeypatch.setattr(
            runtime.data_flow,
            "resolve_sink_trust",
            fail_registry,
        )

        assert runtime.semantic.deterministic_deny_preflight(request) is None
        assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
    finally:
        runtime.close()


def test_malformed_machine_deny_retains_digest_only_identifiers() -> None:
    sentinel = "secret_sentinel"
    runtime = Runtime.open("local", config=_enforce_deny_config())
    try:
        assert runtime.semantic.shutdown()
        _pid, request_id = _external_request(runtime)
        with runtime.human.requests.transaction():
            current = runtime.human.get(request_id)
            payload = deepcopy(current.payload)
            payload["unknown_secret_field"] = sentinel
            payload["context"]["authority_operation"] = f"{sentinel}.read"
            payload["context"]["operation_id"] = f"op_{sentinel}"
            binding = dict(payload["effect_binding"])
            binding["effect_id"] = f"eff_{sentinel}"
            payload["effect_binding"] = binding
            payload["requested_once_capability"]["constraints"][
                "approval_binding"
            ] = dict(binding)
            current = runtime.human.requests.replace_current(
                current,
                payload=payload,
                updated_at=utc_now(),
            )

        settled, _evidence = (
            runtime.human._issue_host_semantic_machine_settlement_port().terminalize(
                request_id,
                expected_revision=current.revision,
                status=HumanRequestStatus.REJECTED,
                decision={"approved": False},
                responder="policy:semantic:hard-deny",
                authority_applier=None,
                audit_action="semantic.policy.deny",
            )
        )

        assert settled.decision is not None
        deny = settled.decision["deterministic_deny"]
        assert deny["effect_id"].startswith("unbound:")
        settlements = runtime.uow.semantic.query_semantic_machine_settlements(
            after=None,
            limit=2,
            request_id=request_id,
        )
        assert len(settlements.records) == 1
        record = settlements.records[0]
        assert record.action_id == "runtime.malformed_external_operation"
        assert record.operation_id is None
        assert record.effect_id.startswith("unbound:")

        retained = json.dumps(
            {
                "semantic_settlement": record.to_dict(),
                "semantic_api": runtime.semantic.query_machine_settlements(
                    request_id=request_id,
                    limit=2,
                ),
                "semantic_events": [
                    event.payload
                    for event in runtime.events.list()
                    if event.type is EventType.SEMANTIC_POLICY_RESPONSE
                    and event.payload.get("request_id") == request_id
                ],
                "semantic_audit": [
                    item.decision
                    for item in runtime.audit.trace()
                    if item.action == "semantic.policy.deny"
                    and item.target == f"human_request:{request_id}"
                ],
            },
            sort_keys=True,
            default=str,
        )
        assert sentinel not in retained
    finally:
        runtime.close()


def test_human_outcome_link_joins_assessment_completed_before_human() -> None:
    runtime = Runtime.open("local", config=_enforce_deny_config())
    try:
        assert runtime.semantic.shutdown()
        _pid, request_id = _external_request(runtime)
        assessment = _assess_request(runtime, request_id)
        assert assessment.human_outcome == "pending"

        request = runtime.human.get(request_id)
        view = runtime.human.public_request_view(request)
        rejected = runtime.human.reject(
            request_id,
            {"approved": False, "source": "test_human"},
            expected_revision=request.revision,
            preview_sha256=view["preview_sha256"],
        )

        link = runtime.uow.semantic.get_semantic_human_outcome_link_for_request(
            request_id
        )
        assert link is not None
        assert link.request_revision == rejected.revision
        assert link.assessment_id == assessment.assessment_id
        assert link.job_id == assessment.job_id
        assert link.settlement_id is None
        assert link.outcome == "rejected"
        assert link.source == "human"
        assert len(link.decision_sha256) == 64
        assert runtime.uow.semantic.get_semantic_assessment(
            assessment.assessment_id
        ) == assessment
    finally:
        runtime.close()


def test_human_outcome_link_joins_late_assessment_without_rewrite() -> None:
    runtime = Runtime.open("local", config=_enforce_deny_config())
    try:
        assert runtime.semantic.shutdown()
        _pid, request_id = _external_request(runtime)
        request = runtime.human.get(request_id)
        view = runtime.human.public_request_view(request)
        runtime.human.reject(
            request_id,
            {"approved": False, "source": "test_human"},
            expected_revision=request.revision,
            preview_sha256=view["preview_sha256"],
        )
        before = runtime.uow.semantic.get_semantic_human_outcome_link_for_request(
            request_id
        )
        assert before is not None
        assert before.assessment_id is None
        assert before.job_id is not None

        assessment = _assess_request(runtime, request_id)
        assert assessment.job_id == before.job_id
        assert assessment.human_outcome == "rejected"
        assert (
            runtime.uow.semantic.get_semantic_human_outcome_link_for_request(
                request_id
            )
            == before
        )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("outcome", "expected_request", "expected_process", "expected_capabilities"),
    (
        (
            "approve",
            HumanRequestStatus.APPROVED,
            ProcessStatus.RUNNABLE,
            1,
        ),
        (
            "reject",
            HumanRequestStatus.REJECTED,
            ProcessStatus.RUNNABLE,
            0,
        ),
        (
            "cancel",
            HumanRequestStatus.CANCELLED,
            ProcessStatus.KILLED,
            0,
        ),
    ),
)
def test_shadow_human_outcome_link_failure_matches_off_and_survives_reopen(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_request: HumanRequestStatus,
    expected_process: ProcessStatus,
    expected_capabilities: int,
) -> None:
    configs = {
        "off": AgentLibOSConfig(
            semantic=SemanticDefaults(mode="off", adapter="deterministic")
        ),
        "shadow": AgentLibOSConfig(
            semantic=SemanticDefaults(
                mode="shadow",
                adapter="deterministic",
                max_concurrency=1,
            )
        ),
    }

    def snapshot(runtime: Runtime, pid: str, request_id: str) -> dict[str, Any]:
        request = runtime.human.get(request_id)
        request_events = [
            (event.type.value, event.payload.get("status"))
            for event in runtime.events.list(target=pid)
            if event.payload.get("request_id") == request_id
        ]
        request_audits = [
            record.action
            for record in runtime.audit.trace(
                target=f"human_request:{request_id}"
            )
        ]
        capabilities = [
            (
                capability.resource,
                tuple(sorted(capability.rights)),
                capability.uses_remaining,
                capability.delegable,
                capability.revocable,
                capability.status.value,
            )
            for capability in runtime.capability.list_subject(pid)
            if capability.resource
            == "filesystem:workspace:reports/phase3.txt"
        ]
        return {
            "request_status": request.status,
            "request_revision": request.revision,
            "process_status": runtime.process.get(pid).status,
            "request_events": request_events,
            "request_audits": request_audits,
            "capabilities": capabilities,
        }

    snapshots: dict[str, dict[str, Any]] = {}
    with TemporaryDirectory() as directory:
        for mode, config in configs.items():
            database = f"{directory}/{mode}-{outcome}.db"
            runtime = Runtime.open(database, config=config)
            try:
                assert runtime.semantic.shutdown()
                pid, request_id = _external_request(runtime)

                def fail_append(_record: Any) -> Any:
                    raise RuntimeError(
                        "injected semantic outcome-link failure"
                    )

                monkeypatch.setattr(
                    runtime.uow.semantic,
                    "append_semantic_human_outcome_link",
                    fail_append,
                )
                request = runtime.human.get(request_id)
                if outcome == "approve":
                    view = runtime.human.public_request_view(request)
                    runtime.human.approve(
                        request_id,
                        {"approved": True, "source": "test_human"},
                        expected_revision=request.revision,
                        preview_sha256=view["preview_sha256"],
                    )
                elif outcome == "reject":
                    view = runtime.human.public_request_view(request)
                    runtime.human.reject(
                        request_id,
                        {"approved": False, "source": "test_human"},
                        expected_revision=request.revision,
                        preview_sha256=view["preview_sha256"],
                    )
                else:
                    runtime.human.interrupt(
                        pid,
                        "cancel",
                        {"reason": "test Human cancellation"},
                    )

                assert (
                    runtime.uow.semantic.get_semantic_human_outcome_link_for_request(
                        request_id
                    )
                    is None
                )
                if mode == "shadow":
                    assert runtime.semantic.status()["queue"][
                        "capture_failures"
                    ] >= 1
                snapshots[mode] = snapshot(runtime, pid, request_id)
            finally:
                runtime.close()

            reopened = Runtime.open(database, config=config)
            try:
                assert reopened.semantic.shutdown()
                assert snapshot(reopened, pid, request_id) == snapshots[mode]
                assert (
                    reopened.uow.semantic.get_semantic_human_outcome_link_for_request(
                        request_id
                    )
                    is None
                )
            finally:
                reopened.close()

    assert snapshots["shadow"] == snapshots["off"]
    assert snapshots["shadow"]["request_status"] is expected_request
    assert snapshots["shadow"]["request_revision"] == 1
    assert snapshots["shadow"]["process_status"] is expected_process
    assert len(snapshots["shadow"]["capabilities"]) == expected_capabilities


def test_observational_link_and_health_statement_failures_restore_savepoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(
        "local",
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(
                mode="shadow",
                adapter="deterministic",
                max_concurrency=1,
            )
        )
    )
    try:
        assert runtime.semantic.shutdown()
        pid, request_id = _external_request(runtime)
        request = runtime.human.get(request_id)
        view = runtime.human.public_request_view(request)
        state = {
            "aborted": False,
            "link_savepoints": 0,
            "health_savepoints": 0,
        }
        original_uow_transaction = runtime.uow.transaction
        original_health_transaction = runtime.semantic._repository.transaction

        @contextmanager
        def link_savepoint(*, include_object_payloads: bool = False) -> Any:
            state["link_savepoints"] += 1
            try:
                with original_uow_transaction(
                    include_object_payloads=include_object_payloads
                ):
                    yield runtime.uow
            except BaseException:
                # Model PostgreSQL's aborted-statement flag being cleared by
                # ROLLBACK TO SAVEPOINT before the caller handles the error.
                state["aborted"] = False
                raise

        @contextmanager
        def health_savepoint(*, include_object_payloads: bool = False) -> Any:
            state["health_savepoints"] += 1
            try:
                with original_health_transaction(
                    include_object_payloads=include_object_payloads
                ):
                    yield runtime.semantic._repository
            except BaseException:
                state["aborted"] = False
                raise

        monkeypatch.setattr(runtime.uow, "transaction", link_savepoint)
        monkeypatch.setattr(
            runtime.semantic._repository,
            "transaction",
            health_savepoint,
        )

        def fail_link(_record: Any) -> Any:
            state["aborted"] = True
            raise RuntimeError("injected PostgreSQL link statement failure")

        def fail_health(_record: Any) -> Any:
            state["aborted"] = True
            raise RuntimeError("injected PostgreSQL health statement failure")

        original_emit = runtime.events.emit

        def emit_after_savepoint(*args: Any, **kwargs: Any) -> Any:
            if state["aborted"]:
                raise RuntimeError("current transaction is aborted")
            return original_emit(*args, **kwargs)

        monkeypatch.setattr(
            runtime.uow.semantic,
            "append_semantic_human_outcome_link",
            fail_link,
        )
        monkeypatch.setattr(
            runtime.semantic._repository,
            "append_semantic_health_event",
            fail_health,
        )
        monkeypatch.setattr(runtime.events, "emit", emit_after_savepoint)

        approved = runtime.human.approve(
            request_id,
            {"approved": True, "source": "test_human"},
            expected_revision=request.revision,
            preview_sha256=view["preview_sha256"],
        )

        assert approved.status is HumanRequestStatus.APPROVED
        assert runtime.process.get(pid).status is ProcessStatus.RUNNABLE
        assert state == {
            "aborted": False,
            "link_savepoints": 1,
            "health_savepoints": 1,
        }
        assert any(
            event.type is EventType.HUMAN_RESPONSE
            and event.payload.get("request_id") == request_id
            for event in runtime.events.list(target=pid)
        )
    finally:
        runtime.close()


def test_machine_policy_outcome_link_failure_remains_atomic_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local", config=_enforce_deny_config())
    try:
        assert runtime.semantic.shutdown()
        pid, request_id = _external_request(runtime)
        request = _replace_payload(runtime, request_id, "duplicate_rights")
        before_events = {event.event_id for event in runtime.events.list()}
        before_audit = {record.record_id for record in runtime.audit.trace()}

        def fail_append(_record: Any) -> Any:
            raise RuntimeError("injected machine outcome-link failure")

        monkeypatch.setattr(
            runtime.uow.semantic,
            "append_semantic_human_outcome_link",
            fail_append,
        )
        with pytest.raises(ValidationError, match="outcome link failed"):
            runtime.human.approve(
                request_id,
                {"approved": True, "source": "test_human"},
            )

        assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
        assert runtime.human.get(request_id).revision == request.revision
        assert runtime.process.get(pid).status is ProcessStatus.WAITING_HUMAN
        assert not any(
            capability.resource == "filesystem:workspace:reports/phase3.txt"
            for capability in runtime.capability.list_subject(pid)
        )
        assert {event.event_id for event in runtime.events.list()} == before_events
        assert {record.record_id for record in runtime.audit.trace()} == before_audit
        assert (
            runtime.uow.semantic.get_semantic_human_outcome_link_for_request(
                request_id
            )
            is None
        )
        assert runtime.uow.semantic.query_semantic_machine_settlements(
            after=None,
            limit=2,
            request_id=request_id,
        ).records == ()
    finally:
        runtime.close()


def test_human_outcome_link_survives_runtime_reopen() -> None:
    config = _enforce_deny_config()
    with TemporaryDirectory() as directory:
        database = f"{directory}/semantic-human-outcome.db"
        runtime = Runtime.open(database, config=config)
        try:
            assert runtime.semantic.shutdown()
            _pid, request_id = _external_request(runtime)
            request = runtime.human.get(request_id)
            view = runtime.human.public_request_view(request)
            runtime.human.reject(
                request_id,
                {"approved": False, "source": "test_human"},
                expected_revision=request.revision,
                preview_sha256=view["preview_sha256"],
            )
            expected = (
                runtime.uow.semantic.get_semantic_human_outcome_link_for_request(
                    request_id
                )
            )
            assert expected is not None
        finally:
            runtime.close()

        reopened = Runtime.open(database, config=config)
        try:
            assert (
                reopened.uow.semantic.get_semantic_human_outcome_link_for_request(
                    request_id
                )
                == expected
            )
        finally:
            reopened.close()


def test_human_terminal_cas_has_one_outcome_link_winner() -> None:
    runtime = Runtime.open("local", config=_enforce_deny_config())
    try:
        assert runtime.semantic.shutdown()
        _pid, request_id = _external_request(runtime)
        request = runtime.human.get(request_id)
        preview_sha256 = runtime.human.public_request_view(request)[
            "preview_sha256"
        ]
        barrier = threading.Barrier(3)
        successes: list[HumanRequest] = []
        failures: list[BaseException] = []

        def decide(approve: bool) -> None:
            barrier.wait()
            try:
                method = runtime.human.approve if approve else runtime.human.reject
                successes.append(
                    method(
                        request_id,
                        {"approved": approve, "source": "race_test"},
                        expected_revision=request.revision,
                        preview_sha256=preview_sha256,
                    )
                )
            except BaseException as exc:
                failures.append(exc)

        threads = (
            threading.Thread(target=decide, args=(True,)),
            threading.Thread(target=decide, args=(False,)),
        )
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        assert len(successes) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], ValidationError)
        link = runtime.uow.semantic.get_semantic_human_outcome_link_for_request(
            request_id
        )
        assert link is not None
        assert link.outcome == successes[0].status.value
        assert link.request_revision == successes[0].revision
        page = runtime.uow.semantic.query_semantic_human_outcome_links(
            after=None,
            limit=2,
            request_id=request_id,
        )
        assert page.records == (link,)
    finally:
        runtime.close()


def test_human_outcome_link_records_process_cancellation() -> None:
    runtime = Runtime.open("local", config=_enforce_deny_config())
    try:
        assert runtime.semantic.shutdown()
        pid, request_id = _external_request(runtime)

        assert runtime.human.cancel_pending_for_process(
            pid,
            actor="runtime:test",
            reason="process terminated",
        ) == [request_id]

        request = runtime.human.get(request_id)
        link = runtime.uow.semantic.get_semantic_human_outcome_link_for_request(
            request_id
        )
        assert request.status is HumanRequestStatus.CANCELLED
        assert link is not None
        assert link.request_revision == request.revision
        assert link.outcome == "cancelled"
        assert link.source == "cancel"
        assert link.settlement_id is None
    finally:
        runtime.close()


def test_off_mode_does_not_append_semantic_human_outcome_link() -> None:
    runtime = Runtime.open("local")
    try:
        pid, request_id = _external_request(runtime)

        assert runtime.human.cancel_pending_for_process(
            pid,
            actor="runtime:test",
            reason="process terminated",
        ) == [request_id]

        assert (
            runtime.uow.semantic.get_semantic_human_outcome_link_for_request(
                request_id
            )
            is None
        )
        page = runtime.uow.semantic.query_semantic_human_outcome_links(
            after=None,
            limit=2,
            request_id=request_id,
        )
        assert page.records == ()
    finally:
        runtime.close()


def test_durable_trip_disables_new_deterministic_deny_proof() -> None:
    runtime = Runtime.open("local")
    try:
        _pid, request_id = _external_request(runtime)
        request = _replace_payload(runtime, request_id, "duplicate_rights")
        _activate_deny_preflight(runtime, tripped=True)

        assert runtime.semantic.deterministic_deny_preflight(request) is None
        assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("field", "reason"),
    (
        ("binding_current", SemanticReasonCode.STALE_BINDING),
        ("target_state_current", SemanticReasonCode.DIGEST_DRIFT),
        ("manifest_current", SemanticReasonCode.STALE_MANIFEST),
        ("policy_current", SemanticReasonCode.STALE_POLICY),
        ("data_flow_allowed", SemanticReasonCode.DATA_FLOW_DENIED),
    ),
)
def test_live_host_deny_fact_requires_explicit_false_proof(
    field: str,
    reason: SemanticReasonCode,
) -> None:
    runtime = Runtime.open("local")
    try:
        _pid, request_id = _external_request(runtime)
        request = runtime.human.get(request_id)
        _activate_deny_preflight(runtime)
        facts: dict[str, Any] = {
            "schema_version": 1,
            "binding_current": None,
            "target_state_current": None,
            "manifest_current": None,
            "policy_current": None,
            "data_flow_allowed": None,
            "evidence_sha256": "4" * 64,
        }
        facts[field] = False
        runtime.semantic._hard_deny_facts_resolver = (
            lambda _request, _job: dict(facts)
        )

        decision = runtime.semantic.deterministic_deny_preflight(request)

        assert decision is not None
        assert decision.reason_codes == (reason,)
    finally:
        runtime.close()


def test_live_host_deny_fact_error_remains_with_human() -> None:
    runtime = Runtime.open("local")
    try:
        _pid, request_id = _external_request(runtime)
        request = runtime.human.get(request_id)
        _activate_deny_preflight(runtime)

        def fail(_request: HumanRequest, _job: Any) -> Mapping[str, Any]:
            raise RuntimeError("live target resolver unavailable")

        runtime.semantic._hard_deny_facts_resolver = fail

        assert runtime.semantic.deterministic_deny_preflight(request) is None
        assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
    finally:
        runtime.close()


def test_downstream_host_validation_error_is_not_malformed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        _pid, request_id = _external_request(runtime)
        request = runtime.human.get(request_id)
        _activate_deny_preflight(runtime)
        monkeypatch.setattr(
            runtime.semantic,
            "_prepare_human_approval",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValidationError("injected Host manifest repository failure")
            ),
        )

        assert runtime.semantic.deterministic_deny_preflight(request) is None
        assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
    finally:
        runtime.close()


def test_exact_static_hard_deny_survives_unrelated_live_resolver_error() -> None:
    runtime = Runtime.open("local")
    try:
        _pid, request_id = _external_request(runtime)
        request = runtime.human.get(request_id)
        _activate_deny_preflight(runtime, matching_rule=True)

        def fail(_request: HumanRequest, _job: Any) -> Mapping[str, Any]:
            raise RuntimeError("unrelated live target resolver unavailable")

        runtime.semantic._hard_deny_facts_resolver = fail
        decision = runtime.semantic.deterministic_deny_preflight(request)

        assert decision is not None
        assert decision.reason_codes == (SemanticReasonCode.POLICY_HARD_DENY,)
    finally:
        runtime.close()


def test_human_deny_preflight_never_takes_worker_claim_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Human owns Store before preflight; taking worker lock would deadlock."""

    runtime = Runtime.open("local")
    try:
        _pid, request_id = _external_request(runtime)
        request = runtime.human.get(request_id)
        _activate_deny_preflight(runtime)

        class _ForbiddenManagerLock:
            def __enter__(self) -> None:
                raise AssertionError(
                    "deny preflight acquired the worker claim manager lock"
                )

            def __exit__(self, *_args: Any) -> None:
                return None

        monkeypatch.setattr(runtime.semantic, "_lock", _ForbiddenManagerLock())

        # No hard fact is present, so this remains Human-owned.  The important
        # invariant is that the preflight reaches durable control without the
        # worker claim lock while the caller may already own the Store.
        assert runtime.semantic.deterministic_deny_preflight(request) is None
        assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
    finally:
        runtime.close()


def test_enforcement_requires_exact_display_revision_and_preview_digest() -> None:
    runtime = Runtime.open("local")
    try:
        _pid, request_id = _external_request(runtime)
        _bind_policy(runtime, deny=False)
        request = runtime.human.get(request_id)
        view = runtime.human.public_request_view(request)

        with pytest.raises(ValidationError, match="expected_revision"):
            runtime.human.approve(request_id, {"approved": True})
        with pytest.raises(ValidationError, match="revision conflict"):
            runtime.human.approve(
                request_id,
                {"approved": True},
                expected_revision=request.revision + 1,
                preview_sha256=view["preview_sha256"],
            )
        with pytest.raises(ValidationError, match="preview changed"):
            runtime.human.approve(
                request_id,
                {"approved": True},
                expected_revision=request.revision,
                preview_sha256="0" * 64,
            )

        assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        approved = runtime.human.approve(
            request_id,
            {"approved": True},
            expected_revision=request.revision,
            preview_sha256=view["preview_sha256"],
        )
        assert approved.status == HumanRequestStatus.APPROVED
    finally:
        runtime.close()


def test_human_approve_hard_deny_commits_machine_receipt_without_authority() -> None:
    runtime = Runtime.open("local")
    try:
        pid, request_id = _external_request(runtime)
        recorded: list[dict[str, Any]] = []

        def recorder(
            request: HumanRequest,
            **kwargs: Any,
        ) -> Mapping[str, Any]:
            recorded.append({"request": request, **kwargs})
            return {"settlement_id": "semset_deny"}

        _bind_policy(runtime, deny=True, recorder=recorder)
        request = runtime.human.get(request_id)
        view = runtime.human.public_request_view(request)
        result = runtime.human.approve(
            request_id,
            {"approved": True, "source": "test_human"},
            expected_revision=request.revision,
            preview_sha256=view["preview_sha256"],
        )

        assert result.status == HumanRequestStatus.REJECTED
        assert result.decision is not None
        assert result.decision["source"] == "machine_policy"
        assert result.decision["settlement_receipt"] == {
            "settlement_id": "semset_deny"
        }
        assert len(recorded) == 1
        assert recorded[0]["status"] == HumanRequestStatus.REJECTED
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert not any(
            capability.resource == "filesystem:workspace:reports/phase3.txt"
            for capability in runtime.capability.list_subject(pid)
        )
        assert any(
            event.type == EventType.SEMANTIC_POLICY_RESPONSE
            and event.payload.get("request_id") == request_id
            for event in runtime.events.list(target=pid)
        )
        assert not any(
            event.type == EventType.HUMAN_RESPONSE
            and event.payload.get("request_id") == request_id
            for event in runtime.events.list(target=pid)
        )
        assert any(
            record.action == "semantic.policy.deny"
            and record.target == f"human_request:{request_id}"
            for record in runtime.audit.trace()
        )
    finally:
        runtime.close()


def test_machine_deny_recorder_failure_rolls_back_request_process_and_evidence() -> None:
    runtime = Runtime.open("local")
    try:
        pid, request_id = _external_request(runtime, malformed=True)

        def fail_recorder(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
            raise RuntimeError("injected settlement failure")

        _bind_policy(runtime, deny=True, recorder=fail_recorder)
        request = runtime.human.get(request_id)
        before_events = {event.event_id for event in runtime.events.list()}
        before_audit = {record.record_id for record in runtime.audit.trace()}

        with pytest.raises(ValidationError, match="settlement recording failed"):
            runtime.human._issue_host_semantic_machine_settlement_port().terminalize(
                request_id,
                expected_revision=request.revision,
                status=HumanRequestStatus.REJECTED,
                decision={"approved": False},
                responder="policy:semantic:test",
                authority_applier=None,
                audit_action="semantic.policy.deny",
            )

        assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        assert runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN
        assert {event.event_id for event in runtime.events.list()} == before_events
        assert {record.record_id for record in runtime.audit.trace()} == before_audit
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["off", "shadow", "enforce_deny"])
def test_machine_approval_is_unreachable_outside_canary_auto(mode: str) -> None:
    runtime = Runtime.open("local")
    try:
        pid, request_id = _external_request(runtime)
        request = runtime.human.get(request_id)
        authority_calls = 0

        def forbidden_authority(_request: HumanRequest) -> Mapping[str, Any]:
            nonlocal authority_calls
            authority_calls += 1
            return {"capability_id": "must-not-exist"}

        runtime.human._host_semantic_policy_preflight = lambda _request: None
        runtime.human._host_semantic_mode_reader = lambda: mode
        runtime.human._host_semantic_settlement_recorder = (
            lambda *_args, **_kwargs: {"settlement_id": "must-not-exist"}
        )

        with pytest.raises(ValidationError, match="canary_auto"):
            runtime.human._issue_host_semantic_machine_settlement_port().terminalize(
                request_id,
                expected_revision=request.revision,
                status=HumanRequestStatus.APPROVED,
                decision={"assessment_sha256": "3" * 64},
                responder="policy:semantic:test",
                authority_applier=forbidden_authority,
                audit_action="semantic.policy.auto_approve",
            )

        assert authority_calls == 0
        assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
        assert not any(
            capability.resource == "filesystem:workspace:reports/phase3.txt"
            for capability in runtime.capability.list_subject(pid)
        )
    finally:
        runtime.close()
