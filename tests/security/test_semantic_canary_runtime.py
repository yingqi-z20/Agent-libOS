from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from agent_libos import Runtime
from agent_libos.capability.effect_binding import (
    APPROVAL_BINDING_KEY,
    approval_binding_sha256,
    canonical_effect_hash,
)
from agent_libos.config import (
    AgentLibOSConfig,
    DataFlowDefaults,
    LLMDefaults,
    LLMProfile,
    SemanticDefaults,
)
from agent_libos.llm.client import LLMCompletion
from agent_libos.llm.profiles import LLMProfileRegistry
from agent_libos.models import (
    CapabilityRight,
    CapabilityDecision,
    CapabilityEffect,
    CapabilityStatus,
    DataFlowContext,
    DataLabels,
    HumanRequestStatus,
    ObjectMetadata,
    ObjectType,
    SemanticApprovalRule,
    SemanticPolicyEpochV1,
    SinkTrustLevel,
    SinkTrustRule,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    SemanticAuthorityTripDeferred,
    ValidationError,
)
from agent_libos.sdk import protected_operations as protected_operations_module
from agent_libos.semantic.enforcement import HostSemanticRateBudget
from agent_libos.storage import SemanticAssessmentJobStatus
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import to_jsonable


pytestmark = pytest.mark.security

_TENANT = "tenant-canary-security"
_PROFILE_ID = "semantic-canary-security"
_MODEL = "semantic-canary-model"
_RULE_ID = "canary-reports-read"


class _SuccessfulSemanticClient:
    """Single-attempt structured classifier with no authority surface."""

    semantic_single_attempt = True

    def __init__(self) -> None:
        self.model = _MODEL
        self.timeout = 5.0
        self.store = False
        self.prompt_cache_key = None
        self.prompt_cache_retention = None
        self.responses_previous_response_id = False
        self.max_retries = 0
        self.calls: list[dict[str, Any]] = []

    def complete_with_metadata(self, **kwargs: Any) -> LLMCompletion:
        self.calls.append(dict(kwargs))
        return LLMCompletion(
            content=json.dumps(
                {
                    "schema_version": 1,
                    "status": "success",
                    "findings": [],
                    "data_findings": [],
                    "confidence_bps": 10_000,
                    "calibration_bucket": "very_high",
                    "ood": False,
                    "abstain": False,
                },
                sort_keys=True,
            ),
            tool_calls=[],
            model=_MODEL,
            request_id="semantic-canary-request",
            response_id="semantic-canary-response",
        )


def _tenant_bucket(tenant: str) -> str:
    return hmac.new(
        b"agent-libos-semantic-canary-test-key",
        tenant.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _canary_config(
    *,
    classifier_sink_tenants: tuple[str, ...] = (_TENANT,),
    auto_approval_rules: tuple[SemanticApprovalRule, ...] | None = None,
    additional_sink_rules: tuple[SinkTrustRule, ...] = (),
) -> AgentLibOSConfig:
    profile = LLMProfile(
        model=_MODEL,
        api_mode="chat",
        timeout_s=5.0,
        max_retries=0,
        store=False,
        prompt_cache_key=None,
        prompt_cache_retention=None,
        responses_previous_response_id=False,
        fallback_json_actions=False,
        max_tokens=2_048,
    )
    llm = LLMDefaults(
        default_profile_id="default",
        profiles={
            "default": LLMProfile(model="default-test-model"),
            _PROFILE_ID: profile,
        },
    )
    profile_config = AgentLibOSConfig(llm=llm)
    profile_sha256 = LLMProfileRegistry(
        SimpleNamespace(),
        config=profile_config,
    ).profile_snapshot(_PROFILE_ID).identity_sha256
    selected_rules = (
        auto_approval_rules
        if auto_approval_rules is not None
        else (
            SemanticApprovalRule(
                rule_id=_RULE_ID,
                authority_operation="filesystem.read",
                resource="filesystem:workspace:reports/*",
                rights=("read",),
            ),
        )
    )
    epoch = SemanticPolicyEpochV1(
        epoch_id="epoch-canary-security",
        generation=1,
        expected_previous_sha256=None,
        tenant_bucket_sha256s=(_tenant_bucket(_TENANT),),
        auto_approval_rules=selected_rules,
        hard_deny_rules=(),
        classifier_profile_id=_PROFILE_ID,
        classifier_profile_sha256=profile_sha256,
        classifier_model_sha256=hashlib.sha256(
            _MODEL.encode("utf-8")
        ).hexdigest(),
        created_at=utc_now(),
    )
    return AgentLibOSConfig(
        llm=llm,
        semantic=SemanticDefaults(
            mode="canary_auto",
            adapter="external",
            external_profile_id=_PROFILE_ID,
            policy_epoch=epoch,
            max_concurrency=1,
        ),
        data_flow=DataFlowDefaults(
            sink_rules=(
                SinkTrustRule(
                    pattern=f"llm:{_PROFILE_ID}",
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="normal",
                    tenants=classifier_sink_tenants,
                    identity_sha256=profile_sha256,
                ),
                SinkTrustRule(
                    pattern="human:owner:terminal",
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="normal",
                    tenants=(_TENANT,),
                ),
                *additional_sink_rules,
            )
        ),
    )


def _external_shadow_config() -> AgentLibOSConfig:
    """Reuse the frozen classifier profile without enabling an epoch."""

    canary = _canary_config()
    return replace(
        canary,
        semantic=replace(
            canary.semantic,
            mode="shadow",
            policy_epoch=None,
        ),
    )


def _non_catalog_approval_payload(
    pid: str,
    *,
    action_id: str,
    resource_override: str | None = None,
) -> dict[str, Any]:
    cases: dict[str, tuple[str, str, str, str, dict[str, Any]]] = {
        "filesystem.read": (
            "filesystem",
            "read_text",
            "filesystem:workspace:outside-ceiling/shadow.txt",
            "read",
            {},
        ),
        "filesystem.write": (
            "filesystem",
            "write_text",
            "filesystem:workspace:reports/shadow.txt",
            "write",
            {},
        ),
        "git.write": ("git", "stage", "git:workspace", "write", {}),
        "shell.run": (
            "shell",
            "run",
            "shell:command:shadow-exact",
            "execute",
            {
                "argv": ["git", "status"],
                "argv_sha256": hashlib.sha256(b"git\0status").hexdigest(),
                "cwd": ".",
            },
        ),
        "jsonrpc.call": (
            "jsonrpc",
            "call",
            "jsonrpc:shadow-endpoint:inspect",
            "read",
            {
                "endpoint_id": "shadow-endpoint",
                "method_id": "inspect",
                "params_sha256": "a" * 64,
                "registry_spec_sha256": "b" * 64,
                "registry_generation": 1,
            },
        ),
        "mcp.call": (
            "mcp",
            "call_tool",
            "mcp:shadow-server:inspect",
            "read",
            {
                "server_id": "shadow-server",
                "tool_id": "inspect",
                "arguments_sha256": "c" * 64,
                "registry_spec_sha256": "d" * 64,
                "registry_generation": 1,
            },
        ),
    }
    adapter, operation, resource, right, extra = cases[action_id]
    if resource_override is not None:
        resource = resource_override
    context = {
        "adapter": adapter,
        "operation": operation,
        "authority_operation": action_id,
        "pid": pid,
        "resource": resource,
        "right": right,
        **extra,
    }
    return {
        "type": "external_operation_approval",
        "context": context,
        "requested_once_capability": {
            "subject": pid,
            "resource": resource,
            "rights": [right],
            "constraints": {},
            "delegable": False,
        },
        "_agent_libos_data_flow_context": DataFlowContext(
            labels=DataLabels(
                sensitivity="normal",
                integrity="verified",
                trust_level="verified",
                tenant=_TENANT,
            )
        ).to_dict(),
    }


def _drain_semantic(runtime: Runtime) -> None:
    while runtime.semantic.process_one():
        pass


def _drain_semantic_until(
    runtime: Runtime,
    done: Callable[[], bool],
    *,
    timeout_s: float = 5.0,
) -> None:
    """Drain a stopped worker while asynchronous capture finishes publishing."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        progressed = runtime.semantic.process_one()
        if done():
            return
        if not progressed:
            time.sleep(0.01)
    pytest.fail("semantic work did not reach the expected state before timeout")


def _json_text(value: Any) -> str:
    try:
        normalized = to_jsonable(value)
    except TypeError:
        # Some immutable evidence records intentionally expose MappingProxyType
        # fields, which dataclasses.asdict cannot deepcopy. repr still exposes
        # every retained scalar and is sufficient for the zero-presence oracle.
        return repr(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout


def _init_git_repository(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(root, "config", "user.name", "Agent libOS Semantic Test")
    _git(root, "config", "user.email", "semantic@example.test")
    _git(root, "config", "core.autocrlf", "false")
    (root / "tracked.txt").write_text("reviewed base\n", encoding="utf-8")
    _git(root, "add", "--", "tracked.txt")
    _git(root, "commit", "-q", "-m", "reviewed base")


def _git_rule(action_id: str) -> SemanticApprovalRule:
    right = "diff" if action_id == "git.diff" else "read"
    return SemanticApprovalRule(
        rule_id=f"canary-{action_id.replace('.', '-')}",
        authority_operation=action_id,
        resource="git:workspace",
        rights=(right,),
    )


def _semantic_manifest(rule: SemanticApprovalRule) -> dict[str, Any]:
    return {
        "approval_policy": {
            "semantic_auto_approval": {
                "schema_version": 1,
                "rules": [rule.to_dict()],
            }
        }
    }


def _bind_complete_git_lineage(
    runtime: Runtime,
    *,
    pid: str,
    root: Path,
    flow: DataFlowContext,
) -> None:
    """Label every conservative carrier read by catalog-v1 Git snapshots."""

    runtime.data_flow.bind_written_file_digest(
        pid=pid,
        normalized_path=".",
        content_sha256=hashlib.sha256(
            b"semantic-canary-reviewed-worktree-root"
        ).hexdigest(),
        context=flow,
    )
    runtime.data_flow.bind_written_file(
        pid=pid,
        normalized_path="tracked.txt",
        content=(root / "tracked.txt").read_bytes(),
        context=flow,
    )
    state = runtime.git.provider.repository_state()
    if state.head_oid is not None:
        runtime.git._bind_git_lineage(  # noqa: SLF001 - Host test fixture.
            pid=pid,
            state=state,
            carrier_kind="commit",
            carrier_id=state.head_oid,
            context=flow,
        )
    runtime.git._bind_git_lineage(  # noqa: SLF001 - Host test fixture.
        pid=pid,
        state=state,
        carrier_kind="index",
        carrier_id=state.index_sha256,
        context=flow,
    )
    runtime.git._bind_repository_content_lineage(  # noqa: SLF001
        pid=pid,
        context=flow,
    )


def _invoke_git_catalog_action(runtime: Runtime, pid: str, action_id: str) -> Any:
    if action_id == "git.diff":
        return runtime.git.diff(pid)
    if action_id == "git.read":
        return runtime.git.status(pid)
    raise AssertionError(f"unsupported Git canary action: {action_id}")


def _issue_exact_git_capability(
    runtime: Runtime,
    *,
    root: Path,
    action_id: str,
    client: _SuccessfulSemanticClient,
) -> tuple[str, DataFlowContext, Any, Any]:
    assert runtime.semantic.shutdown()
    runtime.llms.set_test_client(_PROFILE_ID, client)
    rule = _git_rule(action_id)
    pid = runtime.process.spawn(
        goal=f"Issue an exact reviewed {action_id} canary grant.",
        authority_manifest=_semantic_manifest(rule),
    )
    _drain_semantic(runtime)
    client.calls.clear()
    flow = DataFlowContext(
        labels=DataLabels(
            sensitivity="normal",
            integrity="verified",
            trust_level="verified",
            tenant=_TENANT,
        )
    )
    _bind_complete_git_lineage(runtime, pid=pid, root=root, flow=flow)
    right = CapabilityRight.DIFF if action_id == "git.diff" else CapabilityRight.READ
    runtime.capability.set_permission_policy(
        subject=pid,
        resource="git:workspace",
        rights=[right],
        policy=runtime.capability.ASK_EACH_TIME,
        issued_by="test.host",
    )
    with runtime.data_flow.activate(flow):
        with pytest.raises(HumanApprovalRequired) as required:
            _invoke_git_catalog_action(runtime, pid, action_id)
    request_id = required.value.request_id
    _drain_semantic(runtime)
    assert runtime.human.get(request_id).status is HumanRequestStatus.APPROVED
    settlements = runtime.uow.semantic.query_semantic_machine_settlements(
        after=None,
        limit=20,
        request_id=request_id,
    ).records
    assert len(settlements) == 1
    settlement = settlements[0]
    assert settlement.outcome == "issued"
    assert settlement.capability_id is not None
    capability = runtime.store.get_capability(settlement.capability_id)
    assert capability is not None
    assert capability.uses_remaining == 1
    return pid, flow, settlement, capability


def _lifecycle_observation(
    *,
    settlement: Any,
    capability: Any,
    outcome: str,
    outcome_id: str,
    phase: str = "provider_request",
    error_type: str | None = None,
) -> dict[str, Any]:
    binding = capability.constraints[APPROVAL_BINDING_KEY]
    metadata = capability.metadata["semantic_auto_approval"]
    authority = [
        {
            "capability_id": capability.cap_id,
            "binding_sha256": settlement.binding_sha256,
            "policy_epoch_id": settlement.epoch_id,
            "policy_epoch_sha256": settlement.policy_sha256,
            "tenant_bucket_sha256": settlement.tenant_bucket_sha256,
            "assessment_id": settlement.assessment_id,
            "settlement_id": settlement.settlement_id,
            "budget_bucket_id": metadata["budget_bucket_id"],
            "matched_rule_id": settlement.matched_rule_id,
            "issued_at": binding["issued_at"],
            "expires_at": binding["expires_at"],
            "outcome_id": outcome_id,
        }
    ]
    identity = {
        "schema_version": 1,
        "outcome": outcome,
        "effect_id": settlement.effect_id,
        "authority": authority,
    }
    notification_id = "semantic-lifecycle:" + hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        **identity,
        "notification_id": notification_id,
        "pid": settlement.pid,
        "contract_name": "filesystem.read",
        "phase": phase,
        "error_type": error_type,
    }


def _lifecycle_outcome_id(
    *,
    settlement: Any,
    capability: Any,
    outcome: str,
) -> str:
    identity = {
        "schema_version": 1,
        "lifecycle_slot": (
            "consumed" if outcome == "consumed" else "terminal"
        ),
        "effect_id": settlement.effect_id,
        "settlement_id": settlement.settlement_id,
        "capability_id": capability.cap_id,
        "binding_sha256": settlement.binding_sha256,
    }
    return "semantic-outcome:" + hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _issue_exact_canary_capability(
    runtime: Runtime,
    *,
    target: Path,
    client: _SuccessfulSemanticClient,
    suffix: str,
) -> tuple[str, str, DataFlowContext, Any, dict[str, Any]]:
    """Issue one real machine grant through capture, classifier, and CAS."""

    assert runtime.semantic.shutdown()
    runtime.llms.set_test_client(_PROFILE_ID, client)
    pid = runtime.process.spawn(
        goal=f"Issue an exact canary grant for {suffix}.",
        authority_manifest={
            "approval_policy": {
                "semantic_auto_approval": {
                    "schema_version": 1,
                    "rules": [
                        {
                            "rule_id": _RULE_ID,
                            "authority_operation": "filesystem.read",
                            "resource": "filesystem:workspace:reports/*",
                            "rights": ["read"],
                        }
                    ],
                }
            }
        },
    )
    relative = f"reports/{suffix}.txt"
    resource = runtime.filesystem.resource_for(relative)
    flow = DataFlowContext(
        labels=DataLabels(
            sensitivity="normal",
            integrity="verified",
            trust_level="verified",
            tenant=_TENANT,
        )
    )
    runtime.data_flow.bind_written_file(
        pid=pid,
        normalized_path=relative,
        content=target.read_bytes(),
        context=flow,
    )
    runtime.capability.set_permission_policy(
        subject=pid,
        resource=resource,
        rights=[CapabilityRight.READ],
        policy=runtime.capability.ASK_EACH_TIME,
        issued_by="test.host",
    )
    with runtime.data_flow.activate(flow):
        with pytest.raises(HumanApprovalRequired) as required:
            runtime.filesystem.read_text(pid, relative)
    request_id = required.value.request_id
    pending = runtime.human.get(request_id)
    context = dict(pending.payload["context"])
    _drain_semantic(runtime)
    assert runtime.human.get(request_id).status is HumanRequestStatus.APPROVED
    settlements = runtime.uow.semantic.query_semantic_machine_settlements(
        after=None,
        limit=20,
        request_id=request_id,
    ).records
    assert len(settlements) == 1
    capability_id = settlements[0].capability_id
    assert capability_id is not None
    capability = runtime.store.get_capability(capability_id)
    assert capability is not None
    return pid, resource, flow, capability, context


def _machine_decision(
    capability: Any,
    *,
    subject: str | None = None,
    resource: str | None = None,
    right: str = "read",
    context: dict[str, Any] | None = None,
) -> CapabilityDecision:
    return CapabilityDecision(
        subject=subject or capability.subject,
        resource=resource or capability.resource,
        right=right,
        allowed=True,
        effect=CapabilityEffect.ALLOW,
        reason="security structural trip probe",
        matched_capability_ids=[capability.cap_id],
        selected_capability_id=capability.cap_id,
        consume_capability_id=capability.cap_id,
        context=context or {},
    )


def _assert_non_catalog_request_remains_human(
    runtime: Runtime,
    *,
    pid: str,
    request_id: str,
) -> None:
    """Run the real worker and prove it cannot settle catalog-external work."""

    _drain_semantic(runtime)
    assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
    assert (
        runtime.uow.semantic.query_semantic_machine_settlements(
            after=None,
            limit=20,
            request_id=request_id,
        ).records
        == ()
    )
    assert not [
        capability
        for capability in runtime.store.list_capabilities(subject=pid)
        if capability.metadata.get("semantic_auto_approval") is not None
    ]
    assessments = runtime.uow.semantic.query_semantic_assessments(
        after=None,
        limit=20,
        request_id=request_id,
    ).records
    assert len(assessments) == 1
    assert assessments[0].shadow_outcome == "require_human", _json_text(
        assessments[0]
    )


def test_external_shadow_assesses_five_domains_and_ceiling_miss_without_settlement(
    tmp_path: Path,
) -> None:
    """Auto-approval eligibility cannot suppress Shadow classifier coverage."""

    client = _SuccessfulSemanticClient()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = Runtime.open(
        tmp_path / "external-shadow-non-catalog.sqlite",
        config=_external_shadow_config(),
        substrate=LocalResourceProviderSubstrate(workspace),
    )
    try:
        assert runtime.semantic.shutdown()
        runtime.llms.set_test_client(_PROFILE_ID, client)
        rule = SemanticApprovalRule(
            rule_id=_RULE_ID,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:reports/*",
            rights=("read",),
        )
        pid = runtime.process.spawn(
            goal="Observe Human-owned operations across all approval domains.",
            authority_manifest=_semantic_manifest(rule),
        )
        _drain_semantic(runtime)
        client.calls.clear()

        request_ids = tuple(
            runtime.human.query_authority_request(
                pid,
                "host",
                _non_catalog_approval_payload(
                    pid,
                    action_id=action_id,
                    resource_override=resource_override,
                ),
                authority_origin="external_operation",
            )
            for action_id, resource_override in (
                ("filesystem.write", None),
                ("git.write", None),
                ("shell.run", None),
                ("jsonrpc.call", None),
                ("mcp.call", None),
                # Catalog action, but deliberately outside the Task ceiling.
                (
                    "filesystem.read",
                    "filesystem:workspace:outside-ceiling/shadow.txt",
                ),
            )
        )
        _drain_semantic_until(
            runtime,
            lambda: len(client.calls) == len(request_ids),
        )

        assert len(client.calls) == len(request_ids)
        for request_id in request_ids:
            assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
            assessments = runtime.uow.semantic.query_semantic_assessments(
                after=None,
                limit=10,
                request_id=request_id,
            ).records
            assert len(assessments) == 1
            assert assessments[0].status == "success"
            assert assessments[0].shadow_outcome == "require_human"
            assert (
                runtime.uow.semantic.query_semantic_machine_settlements(
                    after=None,
                    limit=10,
                    request_id=request_id,
                ).records
                == ()
            )
    finally:
        runtime.close()


@pytest.mark.parametrize("action_id", ("git.read", "git.diff"))
def test_real_git_canary_exact_request_issues_consumes_and_succeeds(
    action_id: str,
    tmp_path: Path,
) -> None:
    """Exercise exact Git catalog authority through the real provider boundary."""

    workspace = tmp_path / action_id.replace(".", "-")
    workspace.mkdir()
    _init_git_repository(workspace)
    if action_id == "git.diff":
        (workspace / "tracked.txt").write_text(
            "reviewed base\nreviewed worktree change\n",
            encoding="utf-8",
        )
    rule = _git_rule(action_id)
    client = _SuccessfulSemanticClient()
    runtime = Runtime.open(
        tmp_path / f"{action_id}.sqlite",
        config=_canary_config(auto_approval_rules=(rule,)),
        substrate=LocalResourceProviderSubstrate(workspace),
        semantic_tenant_bucketer=_tenant_bucket,
    )
    try:
        assert runtime.semantic.shutdown()
        runtime.llms.set_test_client(_PROFILE_ID, client)
        pid = runtime.process.spawn(
            goal=f"Inspect the exact reviewed repository using {action_id}.",
            authority_manifest=_semantic_manifest(rule),
        )
        _drain_semantic(runtime)
        client.calls.clear()
        flow = DataFlowContext(
            labels=DataLabels(
                sensitivity="normal",
                integrity="verified",
                trust_level="verified",
                tenant=_TENANT,
            )
        )
        _bind_complete_git_lineage(
            runtime,
            pid=pid,
            root=workspace,
            flow=flow,
        )
        right = (
            CapabilityRight.DIFF
            if action_id == "git.diff"
            else CapabilityRight.READ
        )
        runtime.capability.set_permission_policy(
            subject=pid,
            resource="git:workspace",
            rights=[right],
            policy=runtime.capability.ASK_EACH_TIME,
            issued_by="test.host",
        )

        def invoke() -> Any:
            if action_id == "git.diff":
                return runtime.git.diff(pid)
            return runtime.git.status(pid)

        with runtime.data_flow.activate(flow):
            with pytest.raises(HumanApprovalRequired) as required:
                invoke()
        request_id = required.value.request_id
        pending = runtime.human.get(request_id)
        assert pending.status is HumanRequestStatus.PENDING
        assert pending.payload["context"]["authority_operation"] == action_id
        assert pending.payload["context"]["resource"] == "git:workspace"
        assert pending.payload["context"]["right"] == right.value

        _drain_semantic(runtime)

        approved = runtime.human.get(request_id)
        assert approved.status is HumanRequestStatus.APPROVED, json.dumps(
            {
                "assessments": runtime.semantic.query_assessments(
                    request_id=request_id,
                    limit=20,
                ),
                "settlements": runtime.semantic.query_machine_settlements(
                    request_id=request_id,
                    limit=20,
                ),
                "flow": runtime.semantic.query_flow_entities(
                    pid=pid,
                    limit=50,
                ),
                "health": runtime.semantic.query_health_events(limit=50),
                "client_calls": len(client.calls),
            },
            sort_keys=True,
        )
        settlements = runtime.uow.semantic.query_semantic_machine_settlements(
            after=None,
            limit=20,
            request_id=request_id,
        ).records
        assert len(settlements) == 1
        settlement = settlements[0]
        assert settlement.outcome == "issued"
        assert settlement.matched_rule_id == rule.rule_id
        assert settlement.capability_id is not None
        capability = runtime.store.get_capability(settlement.capability_id)
        assert capability is not None
        assert capability.rights == {right.value}
        assert capability.uses_remaining == 1
        assert capability.delegable is False

        with runtime.data_flow.activate(flow):
            result = invoke()
        if action_id == "git.diff":
            assert "reviewed worktree change" in result.patch
        else:
            assert result.repository_id
        consumed = runtime.store.get_capability(settlement.capability_id)
        assert consumed is not None
        assert consumed.uses_remaining == 0
        assert consumed.active is False
        outcomes = runtime.uow.semantic.query_semantic_machine_outcomes(
            after=None,
            limit=20,
            settlement_id=settlement.settlement_id,
        ).records
        assert {item.outcome for item in outcomes} == {
            "issued",
            "consumed",
            "succeeded",
        }
        assert len(client.calls) == 1
        assert not [
            request
            for request in runtime.human.list(pid)
            if request.payload.get("type") == "data_release_approval"
        ]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "drift",
    ("repository_identity", "ref", "state", "labels"),
)
def test_real_git_canary_drift_blocks_before_protected_provider_dispatch(
    drift: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live Git/lineage mismatch cannot consume or dispatch an issued grant."""

    workspace = tmp_path / f"git-drift-{drift}"
    workspace.mkdir()
    _init_git_repository(workspace)
    rule = _git_rule("git.read")
    runtime = Runtime.open(
        tmp_path / f"git-drift-{drift}.sqlite",
        config=_canary_config(auto_approval_rules=(rule,)),
        substrate=LocalResourceProviderSubstrate(workspace),
        semantic_tenant_bucketer=_tenant_bucket,
    )
    try:
        pid, flow, settlement, capability = _issue_exact_git_capability(
            runtime,
            root=workspace,
            action_id="git.read",
            client=_SuccessfulSemanticClient(),
        )

        if drift == "repository_identity":
            replacement = tmp_path / "replacement-repository"
            replacement.mkdir()
            _init_git_repository(replacement)
            runtime.git.provider = LocalResourceProviderSubstrate(replacement).git
        elif drift == "ref":
            (workspace / "tracked.txt").write_text(
                "reviewed base\nnew ref content\n",
                encoding="utf-8",
            )
            _git(workspace, "add", "--", "tracked.txt")
            _git(workspace, "commit", "-q", "-m", "ref drift")
        elif drift == "state":
            (workspace / "tracked.txt").write_text(
                "reviewed base\nunbound state drift\n",
                encoding="utf-8",
            )
        elif drift == "labels":
            runtime.data_flow.bind_written_file_digest(
                pid=pid,
                normalized_path=".",
                content_sha256=hashlib.sha256(
                    b"semantic-canary-reviewed-worktree-root"
                ).hexdigest(),
                context=DataFlowContext(
                    labels=DataLabels(
                        sensitivity="secret",
                        integrity="verified",
                        trust_level="verified",
                        tenant=_TENANT,
                    )
                ),
            )
        else:  # pragma: no cover - parameter list is the closed matrix.
            raise AssertionError(f"unknown Git drift case: {drift}")

        provider_boundary_calls: list[tuple[str, ...]] = []
        provider = runtime.git.provider
        original_run = provider.run

        def tracked_run(args: Any, **kwargs: Any) -> Any:
            if protected_operations_module._CURRENT_BOUNDARY.get() is not None:
                provider_boundary_calls.append(tuple(args))
            return original_run(args, **kwargs)

        monkeypatch.setattr(provider, "run", tracked_run)
        with runtime.data_flow.activate(flow):
            with pytest.raises(HumanApprovalRequired) as fallback:
                runtime.git.status(pid)
        assert fallback.value.request_id != settlement.request_id
        assert (
            runtime.human.get(fallback.value.request_id).status
            is HumanRequestStatus.PENDING
        )

        persisted = runtime.store.get_capability(capability.cap_id)
        assert persisted is not None
        assert persisted.uses_remaining == 1
        assert persisted.active is True
        assert provider_boundary_calls == []
        outcomes = runtime.uow.semantic.query_semantic_machine_outcomes(
            after=None,
            limit=20,
            settlement_id=settlement.settlement_id,
        ).records
        assert [item.outcome for item in outcomes] == ["issued"]
        control = runtime.uow.semantic.get_semantic_control_state()
        assert control is not None
        if control.tripped:
            assert drift == "labels"
            assert control.trip_code == "critical_high_grant"
        else:
            assert control.tripped is False
            assert control.trip_code is None
    finally:
        runtime.close()


def test_git_write_request_cannot_enter_catalog_or_reach_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real Git mutation stays Human-owned despite a successful classifier."""

    workspace = tmp_path / "git-write-unreachable"
    workspace.mkdir()
    _init_git_repository(workspace)
    rule = _git_rule("git.read")
    client = _SuccessfulSemanticClient()
    runtime = Runtime.open(
        tmp_path / "git-write-unreachable.sqlite",
        config=_canary_config(
            auto_approval_rules=(rule,),
            additional_sink_rules=(
                SinkTrustRule(
                    pattern="git:workspace",
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="normal",
                    tenants=(_TENANT,),
                ),
            ),
        ),
        substrate=LocalResourceProviderSubstrate(workspace),
        semantic_tenant_bucketer=_tenant_bucket,
    )
    try:
        assert runtime.semantic.shutdown()
        runtime.llms.set_test_client(_PROFILE_ID, client)
        pid = runtime.process.spawn(
            goal="Inspect the reviewed repository without mutation authority.",
            authority_manifest=_semantic_manifest(rule),
        )
        _drain_semantic(runtime)
        client.calls.clear()
        runtime.capability.set_permission_policy(
            subject=pid,
            resource="git:workspace",
            rights=[CapabilityRight.WRITE],
            policy=runtime.capability.ASK_EACH_TIME,
            issued_by="test.host",
        )
        (workspace / "tracked.txt").write_text(
            "reviewed base\nmutation must remain Human-owned\n",
            encoding="utf-8",
        )
        expected_state_token = runtime.git._state_token(  # noqa: SLF001
            runtime.git.provider.repository_state()
        ).token
        flow = DataFlowContext(
            labels=DataLabels(
                sensitivity="normal",
                integrity="verified",
                trust_level="verified",
                tenant=_TENANT,
            )
        )
        provider_boundary_calls: list[tuple[str, ...]] = []
        provider = runtime.git.provider
        original_run = provider.run

        def tracked_run(args: Any, **kwargs: Any) -> Any:
            if protected_operations_module._CURRENT_BOUNDARY.get() is not None:
                provider_boundary_calls.append(tuple(args))
            return original_run(args, **kwargs)

        monkeypatch.setattr(provider, "run", tracked_run)
        with runtime.data_flow.activate(flow):
            with pytest.raises(HumanApprovalRequired) as required:
                runtime.git.stage(
                    pid,
                    ["tracked.txt"],
                    expected_state_token,
                )
        request_id = required.value.request_id
        pending = runtime.human.get(request_id)
        assert pending.status is HumanRequestStatus.PENDING
        assert pending.payload["context"]["authority_operation"] == "git.write"
        assert pending.payload["context"]["right"] == "write"

        _assert_non_catalog_request_remains_human(
            runtime,
            pid=pid,
            request_id=request_id,
        )
        assert client.calls == []
        assert provider_boundary_calls == []
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("case", "expected_action"),
    (
        ("filesystem.write", "filesystem.write"),
        ("shell.run", "shell.run"),
        ("jsonrpc.call", "jsonrpc.call"),
        ("mcp.call", "mcp.call"),
    ),
)
def test_real_non_catalog_primitive_worker_matrix_remains_human_owned(
    case: str,
    expected_action: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every catalog-external primitive reaches evidence, never settlement."""

    workspace = tmp_path / case.replace(".", "-")
    workspace.mkdir()
    client = _SuccessfulSemanticClient()
    additional_sink_rules: tuple[SinkTrustRule, ...] = ()
    if case == "filesystem.write":
        additional_sink_rules = (
            SinkTrustRule(
                pattern="filesystem:workspace:reports/*",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="normal",
                tenants=(_TENANT,),
            ),
        )
    elif case == "shell.run":
        additional_sink_rules = (
            SinkTrustRule(
                pattern="shell:*",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="normal",
                tenants=(_TENANT,),
            ),
        )
    elif case == "jsonrpc.call":
        additional_sink_rules = (
            SinkTrustRule(
                pattern="jsonrpc:semantic-hidden:inspect",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="normal",
                tenants=(_TENANT,),
            ),
        )
    elif case == "mcp.call":
        additional_sink_rules = (
            SinkTrustRule(
                pattern="mcp:semantic-hidden:inspect",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="normal",
                tenants=(_TENANT,),
            ),
        )
    runtime = Runtime.open(
        tmp_path / f"{case}.sqlite",
        config=_canary_config(additional_sink_rules=additional_sink_rules),
        substrate=LocalResourceProviderSubstrate(workspace),
        semantic_tenant_bucketer=_tenant_bucket,
    )
    try:
        assert runtime.semantic.shutdown()
        runtime.llms.set_test_client(_PROFILE_ID, client)
        rule = SemanticApprovalRule(
            rule_id=_RULE_ID,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:reports/*",
            rights=("read",),
        )
        pid = runtime.process.spawn(
            goal=f"Route {case} through Human authority only.",
            authority_manifest=_semantic_manifest(rule),
        )
        _drain_semantic(runtime)
        client.calls.clear()
        flow = DataFlowContext(
            labels=DataLabels(
                sensitivity="normal",
                integrity="verified",
                trust_level="verified",
                tenant=_TENANT,
            )
        )
        provider_calls: list[str] = []

        def forbidden_provider(*_args: Any, **_kwargs: Any) -> Any:
            provider_calls.append(case)
            raise AssertionError(
                f"{case} reached its provider before Human settlement"
            )

        if case == "filesystem.write":
            resource = runtime.filesystem.resource_for("reports/new.txt")
            runtime.capability.set_permission_policy(
                subject=pid,
                resource=resource,
                rights=[CapabilityRight.WRITE],
                policy=runtime.capability.ASK_EACH_TIME,
                issued_by="test.host",
            )
            monkeypatch.setattr(
                runtime.filesystem.provider,
                "write_text",
                forbidden_provider,
            )

            def invoke() -> Any:
                return runtime.filesystem.write_text(
                    pid,
                    "reports/new.txt",
                    "Human-owned write\n",
                )

        elif case == "shell.run":
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.allowlist_auto_else_ask_level,
                issued_by="test.host",
            )
            monkeypatch.setattr(
                runtime.shell.provider,
                "run",
                forbidden_provider,
            )

            def invoke() -> Any:
                return runtime.shell.run(
                    pid,
                    ["curl", "https://example.test"],
                )

        elif case == "jsonrpc.call":
            resource = "jsonrpc:semantic-hidden:inspect"
            runtime.capability.set_permission_policy(
                subject=pid,
                resource=resource,
                rights=[CapabilityRight.READ],
                policy=runtime.capability.ASK_EACH_TIME,
                issued_by="test.host",
            )
            monkeypatch.setattr(
                runtime.jsonrpc.provider,
                "call",
                forbidden_provider,
            )

            def invoke() -> Any:
                return runtime.jsonrpc.call(
                    pid,
                    "semantic-hidden",
                    "inspect",
                    {"scope": "reviewed"},
                )

        elif case == "mcp.call":
            resource = "mcp:semantic-hidden:inspect"
            runtime.capability.set_permission_policy(
                subject=pid,
                resource=resource,
                rights=[CapabilityRight.READ],
                policy=runtime.capability.ASK_EACH_TIME,
                issued_by="test.host",
            )
            monkeypatch.setattr(
                runtime.mcp.provider,
                "call_tool",
                forbidden_provider,
            )

            def invoke() -> Any:
                return runtime.mcp.call_tool(
                    pid,
                    "semantic-hidden",
                    "inspect",
                    {"scope": "reviewed"},
                )

        else:  # pragma: no cover - parameter list is the closed matrix.
            raise AssertionError(f"unknown non-catalog case: {case}")

        with runtime.data_flow.activate(flow):
            with pytest.raises(HumanApprovalRequired) as required:
                invoke()
        request_id = required.value.request_id
        pending = runtime.human.get(request_id)
        assert pending.status is HumanRequestStatus.PENDING
        assert pending.payload["context"]["authority_operation"] == expected_action

        _assert_non_catalog_request_remains_human(
            runtime,
            pid=pid,
            request_id=request_id,
        )
        assert client.calls == []
        assert provider_calls == []
    finally:
        runtime.close()


def test_real_git_push_worker_remains_human_owned_and_never_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Push may perform Host preflight, but cannot dispatch the remote effect."""

    workspace = tmp_path / "git-push-unreachable"
    remote = tmp_path / "git-push-remote.git"
    workspace.mkdir()
    _init_git_repository(workspace)
    remote.mkdir()
    _git(remote, "init", "--bare", "-q")
    _git(workspace, "remote", "add", "origin", remote.as_uri())
    rule = _git_rule("git.read")
    config = _canary_config(
        auto_approval_rules=(rule,),
        additional_sink_rules=(
            SinkTrustRule(
                pattern="git_remote:workspace:origin",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="normal",
                tenants=(_TENANT,),
            ),
        ),
    )
    config = replace(
        config,
        git=replace(config.git, allow_file_remotes=True),
    )
    runtime = Runtime.open(
        tmp_path / "git-push-unreachable.sqlite",
        config=config,
        substrate=LocalResourceProviderSubstrate(workspace),
        semantic_tenant_bucketer=_tenant_bucket,
    )
    try:
        assert runtime.semantic.shutdown()
        runtime.llms.set_test_client(_PROFILE_ID, _SuccessfulSemanticClient())
        pid = runtime.process.spawn(
            goal="Inspect the repository without remote mutation authority.",
            authority_manifest=_semantic_manifest(rule),
        )
        _drain_semantic(runtime)
        remote_resource = runtime.git.remote_resource("origin")
        runtime.capability.set_permission_policy(
            subject=pid,
            resource=remote_resource,
            rights=[CapabilityRight.WRITE],
            policy=runtime.capability.ASK_EACH_TIME,
            issued_by="test.host",
        )
        expected_state_token = runtime.git._state_token(  # noqa: SLF001
            runtime.git.provider.repository_state()
        ).token
        flow = DataFlowContext(
            labels=DataLabels(
                sensitivity="normal",
                integrity="verified",
                trust_level="verified",
                tenant=_TENANT,
            )
        )
        dispatched_pushes: list[tuple[str, ...]] = []
        provider = runtime.git.provider
        original_run = provider.run

        def reject_push(args: Any, **kwargs: Any) -> Any:
            selected = tuple(args)
            if selected and selected[0] == "push":
                dispatched_pushes.append(selected)
                raise AssertionError("Git push reached the provider")
            return original_run(args, **kwargs)

        monkeypatch.setattr(provider, "run", reject_push)
        with runtime.data_flow.activate(flow):
            with pytest.raises(HumanApprovalRequired) as required:
                runtime.git.push(
                    pid,
                    "origin",
                    "refs/heads/main",
                    expected_state_token,
                    local_ref="refs/heads/main",
                )
        request_id = required.value.request_id
        pending = runtime.human.get(request_id)
        assert pending.status is HumanRequestStatus.PENDING
        assert pending.payload["context"]["authority_operation"] == "git.write"
        assert pending.payload["context"]["operation"] == "push"

        _assert_non_catalog_request_remains_human(
            runtime,
            pid=pid,
            request_id=request_id,
        )
        assert dispatched_pushes == []
    finally:
        runtime.close()


def test_transient_classifier_source_refs_never_enter_evidence_surfaces(
    tmp_path: Path,
) -> None:
    """The Host carries live source refs to preflight but persists only digests."""

    raw_origin = "TRANSIENT_SEMANTIC_SOURCE_IDENTITY_73b30ac0"
    raw_payload = "TRANSIENT_SEMANTIC_SOURCE_PAYLOAD_ef4de019"
    workspace = tmp_path / "transient-source-ref"
    target = workspace / "reports" / "source-bound.txt"
    target.parent.mkdir(parents=True)
    target.write_text("reviewed source-bound report\n", encoding="utf-8")
    client = _SuccessfulSemanticClient()
    runtime = Runtime.open(
        tmp_path / "transient-source-ref.sqlite",
        config=_canary_config(),
        substrate=LocalResourceProviderSubstrate(workspace),
        semantic_tenant_bucketer=_tenant_bucket,
    )
    try:
        assert runtime.semantic.shutdown()
        runtime.llms.set_test_client(_PROFILE_ID, client)
        rule = SemanticApprovalRule(
            rule_id=_RULE_ID,
            authority_operation="filesystem.read",
            resource="filesystem:workspace:reports/*",
            rights=("read",),
        )
        pid = runtime.process.spawn(
            goal="Read the exact source-bound canary report.",
            authority_manifest=_semantic_manifest(rule),
        )
        _drain_semantic(runtime)
        client.calls.clear()
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"value": raw_payload},
            metadata=ObjectMetadata(
                sensitivity="normal",
                integrity="verified",
                trust_level="verified",
                origin=raw_origin,
                tenant=_TENANT,
            ),
        )
        flow = runtime.data_flow.context_from_source_oids(
            pid,
            [source.oid],
            include_current=False,
        )
        relative = "reports/source-bound.txt"
        resource = runtime.filesystem.resource_for(relative)
        runtime.data_flow.bind_written_file(
            pid=pid,
            normalized_path=relative,
            content=target.read_bytes(),
            context=flow,
        )
        runtime.capability.set_permission_policy(
            subject=pid,
            resource=resource,
            rights=[CapabilityRight.READ],
            policy=runtime.capability.ASK_EACH_TIME,
            issued_by="test.host",
        )
        with runtime.data_flow.activate(flow):
            with pytest.raises(HumanApprovalRequired) as required:
                runtime.filesystem.read_text(pid, relative)
        request_id = required.value.request_id

        _drain_semantic(runtime)

        assert runtime.human.get(request_id).status is HumanRequestStatus.APPROVED
        terminal_statuses = tuple(
            status.value
            for status in SemanticAssessmentJobStatus
            if status
            not in {
                SemanticAssessmentJobStatus.QUEUED,
                SemanticAssessmentJobStatus.CLAIMED,
            }
        )
        jobs = runtime.uow.semantic.query_semantic_assessment_jobs(
            statuses=terminal_statuses,
            projection_expires_before=None,
            limit=500,
        )
        assert jobs
        assert all(job.projection == {} for job in jobs)
        assessments = runtime.uow.semantic.query_semantic_assessments(
            after=None,
            limit=500,
            pid=pid,
        ).records
        settlements = runtime.uow.semantic.query_semantic_machine_settlements(
            after=None,
            limit=500,
            pid=pid,
        ).records
        assert len(settlements) == 1
        flow_entities = runtime.uow.semantic.query_semantic_flow_entities(
            after=None,
            limit=500,
            pid=pid,
        ).records
        flow_activities = runtime.uow.semantic.query_semantic_flow_activities(
            after=None,
            limit=500,
            pid=pid,
        ).records
        flow_edges = runtime.uow.semantic.query_semantic_flow_edges(
            after=None,
            limit=500,
            pid=pid,
        ).records
        flow_assertions = (
            runtime.uow.semantic.query_semantic_flow_label_assertions(
                after=None,
                limit=500,
            ).records
        )
        classifier_effects = [
            effect
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider == "llm" and effect.operation == "complete"
        ]
        assert classifier_effects
        classifier_effect_ids = {
            effect.effect_id for effect in classifier_effects
        }
        classifier_events = [
            event
            for event in runtime.store.list_events()
            if event.payload.get("effect_id") in classifier_effect_ids
            or event.payload.get("sink") == f"llm:{_PROFILE_ID}"
        ]
        classifier_audit = [
            record
            for record in runtime.store.list_audit()
            if record.target == f"llm:{_PROFILE_ID}"
            or (record.decision or {}).get("effect_id")
            in classifier_effect_ids
            or (record.decision or {}).get("sink") == f"llm:{_PROFILE_ID}"
        ]
        evidence_surfaces = {
            "jobs": jobs,
            "assessments": assessments,
            "flow_entities": flow_entities,
            "flow_activities": flow_activities,
            "flow_edges": flow_edges,
            "flow_assertions": flow_assertions,
            "settlements": settlements,
            "classifier_effects": classifier_effects,
            "classifier_events": classifier_events,
            "classifier_audit": classifier_audit,
            "llm_calls": runtime.store.list_llm_calls(pid=pid, limit=100),
        }
        retained = _json_text(evidence_surfaces)
        outbound = _json_text(client.calls)
        for raw_value in (source.oid, raw_origin, raw_payload):
            assert raw_value not in retained
            assert raw_value not in outbound
        assert flow.source_refs_hash() in retained
        assert not [
            request
            for request in runtime.human.list(pid)
            if request.payload.get("type") == "data_release_approval"
        ]
    finally:
        runtime.close()


def test_real_runtime_canary_exact_read_is_atomic_consumed_and_not_replayable() -> None:
    """Exercise the production Runtime composition from capture through dispatch."""

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        target = root / "reports" / "canary.txt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"reviewed canary input\n")
        client = _SuccessfulSemanticClient()
        runtime = Runtime.open(
            "local",
            config=_canary_config(),
            substrate=LocalResourceProviderSubstrate(root),
            semantic_tenant_bucketer=_tenant_bucket,
        )
        try:
            # Keep the external adapter and full Runtime wiring, but claim jobs
            # synchronously so the assertions do not depend on worker timing.
            assert runtime.semantic.shutdown()
            runtime.llms.set_test_client(_PROFILE_ID, client)
            pid = runtime.process.spawn(
                goal="Read the exact reviewed canary report.",
                authority_manifest={
                    "approval_policy": {
                        "semantic_auto_approval": {
                            "schema_version": 1,
                            "rules": [
                                {
                                    "rule_id": _RULE_ID,
                                    "authority_operation": "filesystem.read",
                                    "resource": "filesystem:workspace:reports/*",
                                    "rights": ["read"],
                                }
                            ],
                        }
                    }
                },
            )
            relative = "reports/canary.txt"
            resource = runtime.filesystem.resource_for(relative)
            flow = DataFlowContext(
                labels=DataLabels(
                    sensitivity="normal",
                    integrity="verified",
                    trust_level="verified",
                    tenant=_TENANT,
                )
            )
            runtime.data_flow.bind_written_file(
                pid=pid,
                normalized_path=relative,
                content=target.read_bytes(),
                context=flow,
            )
            label_binding_before = runtime.store.get_file_label_binding(relative)
            runtime.capability.set_permission_policy(
                subject=pid,
                resource=resource,
                rights=[CapabilityRight.READ],
                policy=runtime.capability.ASK_EACH_TIME,
                issued_by="test.host",
            )

            with runtime.data_flow.activate(flow):
                with pytest.raises(HumanApprovalRequired) as required:
                    runtime.filesystem.read_text(pid, relative)
            request_id = required.value.request_id
            pending = runtime.human.get(request_id)
            assert pending.status is HumanRequestStatus.PENDING

            _drain_semantic(runtime)

            approved = runtime.human.get(request_id)
            assert approved.status is HumanRequestStatus.APPROVED, json.dumps(
                {
                    "assessment": runtime.semantic.query_assessments(
                        request_id=request_id,
                        limit=20,
                    ),
                    "settlements": [
                        item.to_dict()
                        for item in runtime.uow.semantic.query_semantic_machine_settlements(
                            after=None,
                            limit=20,
                            request_id=request_id,
                        ).records
                    ],
                    "data_flow_decisions": [
                        repr(item)
                        for item in runtime.store.list_data_flow_decisions(pid=pid)
                    ],
                },
                sort_keys=True,
            )
            assessment_page = runtime.uow.semantic.query_semantic_assessments(
                after=None,
                limit=20,
                request_id=request_id,
            )
            assert len(assessment_page.records) == 1
            assessment = assessment_page.records[0]
            assert assessment.status == "success"
            assert assessment.confidence_bps == 10_000
            assert assessment.calibration_bucket == "very_high"
            assert assessment.human_outcome == "approved"

            settlement_page = (
                runtime.uow.semantic.query_semantic_machine_settlements(
                    after=None,
                    limit=20,
                    request_id=request_id,
                )
            )
            assert len(settlement_page.records) == 1
            settlement = settlement_page.records[0]
            assert settlement.outcome == "issued"
            assert settlement.capability_id is not None
            capability = runtime.store.get_capability(settlement.capability_id)
            assert capability is not None
            assert capability.uses_remaining == 1
            semantic_metadata = capability.metadata["semantic_auto_approval"]
            budget_bucket_id = semantic_metadata["budget_bucket_id"]
            reserved_budget = runtime.uow.semantic.get_semantic_rate_budget(
                budget_bucket_id
            )
            assert reserved_budget is not None
            assert reserved_budget.inflight_count == 1

            with runtime.data_flow.activate(flow):
                result = runtime.filesystem.read_text(pid, relative)
            assert result.content == "reviewed canary input\n"
            consumed = runtime.store.get_capability(settlement.capability_id)
            assert consumed is not None
            assert consumed.uses_remaining == 0
            assert consumed.active is False

            outcomes = runtime.uow.semantic.query_semantic_machine_outcomes(
                after=None,
                limit=20,
                settlement_id=settlement.settlement_id,
            ).records
            assert {item.outcome for item in outcomes} == {
                "issued",
                "consumed",
                "succeeded",
            }
            released_budget = runtime.uow.semantic.get_semantic_rate_budget(
                budget_bucket_id
            )
            assert released_budget is not None
            assert released_budget.inflight_count == 0
            assert released_budget.minute_count == 1
            assert released_budget.day_count == 1

            # The exhausted semantic grant cannot be replayed. ASK remains the
            # policy, so retry creates a fresh Human request instead of silently
            # widening authority or installing always_allow.
            with runtime.data_flow.activate(flow):
                with pytest.raises(HumanApprovalRequired) as replay:
                    runtime.filesystem.read_text(pid, relative)
            assert replay.value.request_id != request_id
            assert (
                runtime.capability.permission_policy(
                    pid,
                    resource,
                    CapabilityRight.READ,
                )
                == runtime.capability.ASK_EACH_TIME
            )
            assert runtime.store.get_file_label_binding(relative) == label_binding_before
            assert not [
                request
                for request in runtime.human.list(pid)
                if request.payload.get("type") == "data_release_approval"
            ]
            # Root-goal classification may remain entirely local/metadata-only;
            # the exact approval must still make one bounded classifier call.
            assert len(client.calls) >= 1
            outbound = json.dumps(
                client.calls,
                ensure_ascii=False,
                sort_keys=True,
            )
            assert _TENANT not in outbound

            # A second issuance exercises the Host lifecycle bridge without
            # relying on SDK object internals. One exact failed notification
            # releases its inflight slot; the identical notification is a
            # no-op and cannot decrement a shared bucket twice.
            replay_request_id = replay.value.request_id
            _drain_semantic(runtime)
            replay_settlements = (
                runtime.uow.semantic.query_semantic_machine_settlements(
                    after=None,
                    limit=20,
                    request_id=replay_request_id,
                ).records
            )
            assert len(replay_settlements) == 1
            replay_settlement = replay_settlements[0]
            assert replay_settlement.outcome == "issued"
            assert replay_settlement.capability_id is not None
            replay_capability = runtime.store.get_capability(
                replay_settlement.capability_id
            )
            assert replay_capability is not None
            replay_budget_id = replay_capability.metadata[
                "semantic_auto_approval"
            ]["budget_bucket_id"]
            replay_reserved = runtime.uow.semantic.get_semantic_rate_budget(
                replay_budget_id
            )
            assert replay_reserved is not None
            assert replay_reserved.inflight_count == 1
            assert replay_reserved.minute_count == 2

            terminal_id = _lifecycle_outcome_id(
                settlement=replay_settlement,
                capability=replay_capability,
                outcome="failed",
            )
            forged_id = _lifecycle_observation(
                settlement=replay_settlement,
                capability=replay_capability,
                outcome="failed",
                outcome_id="semantic-outcome:" + "f" * 64,
                error_type="RuntimeError",
            )
            with pytest.raises(
                ValidationError,
                match="lifecycle outcome identity changed",
            ):
                runtime.semantic.record_machine_lifecycle(forged_id)
            assert (
                runtime.uow.semantic.get_semantic_rate_budget(
                    replay_budget_id
                )
                == replay_reserved
            )

            unknown_field = _lifecycle_observation(
                settlement=replay_settlement,
                capability=replay_capability,
                outcome="failed",
                outcome_id=terminal_id,
                error_type="RuntimeError",
            )
            unknown_field["authority"][0]["permit"] = True
            unknown_identity = {
                "schema_version": 1,
                "outcome": unknown_field["outcome"],
                "effect_id": unknown_field["effect_id"],
                "authority": unknown_field["authority"],
            }
            unknown_field["notification_id"] = (
                "semantic-lifecycle:"
                + hashlib.sha256(
                    json.dumps(
                        unknown_identity,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            )
            with pytest.raises(
                ValidationError,
                match="lifecycle authority item is malformed",
            ):
                runtime.semantic.record_machine_lifecycle(unknown_field)
            assert (
                runtime.uow.semantic.get_semantic_rate_budget(
                    replay_budget_id
                )
                == replay_reserved
            )

            failed = _lifecycle_observation(
                settlement=replay_settlement,
                capability=replay_capability,
                outcome="failed",
                outcome_id=terminal_id,
                error_type="RuntimeError",
            )
            runtime.semantic.record_machine_lifecycle(failed)
            released_once = runtime.uow.semantic.get_semantic_rate_budget(
                replay_budget_id
            )
            assert released_once is not None
            assert released_once.inflight_count == 0
            runtime.semantic.record_machine_lifecycle(failed)
            released_twice = runtime.uow.semantic.get_semantic_rate_budget(
                replay_budget_id
            )
            assert released_twice == released_once

            failed_outcomes = (
                runtime.uow.semantic.query_semantic_machine_outcomes(
                    after=None,
                    limit=20,
                    settlement_id=replay_settlement.settlement_id,
                ).records
            )
            assert [item.outcome for item in failed_outcomes].count("failed") == 1

            # Unknown provider state commits the durable trip before trying
            # the shared terminal slot. The contradictory terminal identity
            # then conflicts append-only, while the already released budget
            # remains exactly zero.
            unknown = _lifecycle_observation(
                settlement=replay_settlement,
                capability=replay_capability,
                outcome="provider_outcome_unknown",
                outcome_id=terminal_id,
                phase="dispatch",
                error_type="TimeoutError",
            )
            with pytest.raises(
                ValidationError,
                match="append-only semantic machine outcome identity conflicts",
            ):
                runtime.semantic.record_machine_lifecycle(unknown)
            durable_control = runtime.uow.semantic.get_semantic_control_state()
            assert durable_control is not None
            assert durable_control.tripped is True
            assert durable_control.trip_code == "provider_outcome_unknown"
            after_conflict = runtime.uow.semantic.get_semantic_rate_budget(
                replay_budget_id
            )
            assert after_conflict is not None
            assert after_conflict.inflight_count == 0
        finally:
            runtime.close()


@pytest.mark.parametrize(
    "drift",
    (
        "mixed_identity",
        "tenant_change",
        "label_digest_change",
        "unkeyed_identity",
        "sink_uncleared",
    ),
)
def test_live_identity_or_sink_drift_blocks_classifier_egress(
    drift: str,
) -> None:
    """Captured identity flags never replace live label and Sink preflight."""

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        target = root / "reports" / "identity-drift.txt"
        target.parent.mkdir(parents=True)
        target.write_text("identity-bound canary input\n", encoding="utf-8")
        client = _SuccessfulSemanticClient()
        sink_tenants = (
            ("different-cleared-tenant",)
            if drift == "sink_uncleared"
            else (_TENANT,)
        )
        runtime = Runtime.open(
            "local",
            config=_canary_config(classifier_sink_tenants=sink_tenants),
            substrate=LocalResourceProviderSubstrate(root),
            semantic_tenant_bucketer=_tenant_bucket,
        )
        try:
            assert runtime.semantic.shutdown()
            runtime.llms.set_test_client(_PROFILE_ID, client)
            pid = runtime.process.spawn(
                goal="Read one identity-bound report.",
                authority_manifest={
                    "approval_policy": {
                        "semantic_auto_approval": {
                            "schema_version": 1,
                            "rules": [
                                {
                                    "rule_id": _RULE_ID,
                                    "authority_operation": "filesystem.read",
                                    "resource": "filesystem:workspace:reports/*",
                                    "rights": ["read"],
                                }
                            ],
                        }
                    }
                },
            )
            # Isolate the approval dispatch count from root-goal assessment.
            _drain_semantic(runtime)
            client.calls.clear()

            relative = "reports/identity-drift.txt"
            resource = runtime.filesystem.resource_for(relative)
            flow = DataFlowContext(
                labels=DataLabels(
                    sensitivity="normal",
                    integrity="verified",
                    trust_level="verified",
                    tenant=_TENANT,
                )
            )
            runtime.data_flow.bind_written_file(
                pid=pid,
                normalized_path=relative,
                content=target.read_bytes(),
                context=flow,
            )
            runtime.capability.set_permission_policy(
                subject=pid,
                resource=resource,
                rights=[CapabilityRight.READ],
                policy=runtime.capability.ASK_EACH_TIME,
                issued_by="test.host",
            )

            with runtime.data_flow.activate(flow):
                with pytest.raises(HumanApprovalRequired) as required:
                    runtime.filesystem.read_text(pid, relative)
            request_id = required.value.request_id
            captured = runtime.human.get(request_id)
            assert captured.status is HumanRequestStatus.PENDING

            if drift != "sink_uncleared":
                payload = deepcopy(captured.payload)
                labels = payload["_agent_libos_data_flow_context"]["labels"]
                if drift == "mixed_identity":
                    labels["tenant"] = "mixed"
                    labels["principal"] = "mixed"
                elif drift == "tenant_change":
                    labels["tenant"] = "tenant-changed-after-capture"
                elif drift == "label_digest_change":
                    labels["integrity"] = "untrusted"
                    labels["trust_level"] = "untrusted"
                elif drift == "unkeyed_identity":
                    labels["tenant"] = None
                    labels["principal"] = "unkeyed-principal"
                else:  # pragma: no cover - parameter list is the closed matrix.
                    raise AssertionError(f"unknown drift case: {drift}")
                captured = runtime.human.requests.replace_current(
                    captured,
                    payload=payload,
                )

            _drain_semantic(runtime)

            assessment_page = runtime.uow.semantic.query_semantic_assessments(
                after=None,
                limit=20,
                request_id=request_id,
            )
            assert len(assessment_page.records) == 1
            assessment = assessment_page.records[0]
            assert assessment.status == "egress_blocked"
            assert runtime.human.get(request_id).status is HumanRequestStatus.PENDING
            assert client.calls == []

            settlements = (
                runtime.uow.semantic.query_semantic_machine_settlements(
                    after=None,
                    limit=20,
                    request_id=request_id,
                ).records
            )
            assert all(item.outcome != "issued" for item in settlements)
            bucket_id = HostSemanticRateBudget.bucket_id_for(
                epoch_id="epoch-canary-security",
                tenant_bucket_sha256=_tenant_bucket(_TENANT),
                rule_id=_RULE_ID,
            )
            assert (
                runtime.uow.semantic.get_semantic_rate_budget(bucket_id) is None
            )
            assert not [
                capability
                for capability in runtime.capability.list_subject(pid)
                if capability.metadata.get("semantic_auto_approval") is not None
            ]
            assert not [
                request
                for request in runtime.human.list(pid)
                if request.payload.get("type") == "data_release_approval"
            ]
        finally:
            runtime.close()


@pytest.mark.parametrize("mismatch_phase", ("prepare", "dispatch"))
def test_protected_effect_mismatch_trips_after_rollback_and_survives_reopen(
    mismatch_phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferred Host trips commit outside the rolled-back effect transaction."""

    database = tmp_path / f"semantic-{mismatch_phase}-trip.sqlite"
    workspace = tmp_path / f"workspace-{mismatch_phase}"
    target = workspace / "reports" / "deferred-trip.txt"
    target.parent.mkdir(parents=True)
    target.write_text("deferred trip canary input\n", encoding="utf-8")
    config = _canary_config()
    substrate = LocalResourceProviderSubstrate(workspace)
    client = _SuccessfulSemanticClient()
    runtime = Runtime.open(
        database,
        config=config,
        substrate=substrate,
        semantic_tenant_bucketer=_tenant_bucket,
    )
    closed = False
    try:
        assert runtime.semantic.shutdown()
        runtime.llms.set_test_client(_PROFILE_ID, client)
        pid = runtime.process.spawn(
            goal=f"Exercise a {mismatch_phase} authority fence.",
            authority_manifest={
                "approval_policy": {
                    "semantic_auto_approval": {
                        "schema_version": 1,
                        "rules": [
                            {
                                "rule_id": _RULE_ID,
                                "authority_operation": "filesystem.read",
                                "resource": "filesystem:workspace:reports/*",
                                "rights": ["read"],
                            }
                        ],
                    }
                }
            },
        )
        relative = "reports/deferred-trip.txt"
        resource = runtime.filesystem.resource_for(relative)
        flow = DataFlowContext(
            labels=DataLabels(
                sensitivity="normal",
                integrity="verified",
                trust_level="verified",
                tenant=_TENANT,
            )
        )
        runtime.data_flow.bind_written_file(
            pid=pid,
            normalized_path=relative,
            content=target.read_bytes(),
            context=flow,
        )
        runtime.capability.set_permission_policy(
            subject=pid,
            resource=resource,
            rights=[CapabilityRight.READ],
            policy=runtime.capability.ASK_EACH_TIME,
            issued_by="test.host",
        )
        with runtime.data_flow.activate(flow):
            with pytest.raises(HumanApprovalRequired) as required:
                runtime.filesystem.read_text(pid, relative)
        request_id = required.value.request_id
        _drain_semantic(runtime)
        assert runtime.human.get(request_id).status is HumanRequestStatus.APPROVED
        settlements = (
            runtime.uow.semantic.query_semantic_machine_settlements(
                after=None,
                limit=20,
                request_id=request_id,
            ).records
        )
        assert len(settlements) == 1
        settlement = settlements[0]
        assert settlement.capability_id is not None

        original_recheck = (
            runtime.capability.require_semantic_approval_current
        )

        def mismatch_recheck(
            decision: Any,
            *,
            phase: str,
            effect_id: str | None,
            allow_reserved: bool,
        ) -> None:
            selected_effect = effect_id
            if phase == mismatch_phase:
                selected_effect = "eff_host-forged-mismatch"
            original_recheck(
                decision,
                phase=phase,
                effect_id=selected_effect,
                allow_reserved=allow_reserved,
            )

        monkeypatch.setattr(
            runtime.capability,
            "require_semantic_approval_current",
            mismatch_recheck,
        )
        provider_boundary_calls: list[str] = []
        provider = runtime.filesystem.provider
        original_state = provider.state
        original_read_bytes = provider.read_bytes

        def tracked_state(*args: Any, **kwargs: Any) -> Any:
            if protected_operations_module._CURRENT_BOUNDARY.get() is not None:
                provider_boundary_calls.append("state")
            return original_state(*args, **kwargs)

        def tracked_read_bytes(*args: Any, **kwargs: Any) -> Any:
            if protected_operations_module._CURRENT_BOUNDARY.get() is not None:
                provider_boundary_calls.append("read")
            return original_read_bytes(*args, **kwargs)

        monkeypatch.setattr(provider, "state", tracked_state)
        monkeypatch.setattr(provider, "read_bytes", tracked_read_bytes)

        with runtime.data_flow.activate(flow):
            with pytest.raises(
                SemanticAuthorityTripDeferred,
                match="durable safety trip",
            ):
                runtime.filesystem.read_text(pid, relative)

        # The failed prepare/dispatch transaction cannot consume the grant,
        # and the provider callback cannot run. The local latch and durable
        # trip are deliberately outside that rolled-back Unit of Work.
        persisted_capability = runtime.store.get_capability(
            settlement.capability_id
        )
        assert persisted_capability is not None
        assert persisted_capability.uses_remaining == 1
        assert provider_boundary_calls == []
        control = runtime.uow.semantic.get_semantic_control_state()
        assert control is not None
        assert control.tripped is True
        assert control.trip_code == "unauthorized_effect"

        runtime.close()
        closed = True
        reopened = Runtime.open(
            database,
            config=config,
            substrate=LocalResourceProviderSubstrate(workspace),
            semantic_tenant_bucketer=_tenant_bucket,
        )
        try:
            assert reopened.semantic.shutdown()
            reopened_control = (
                reopened.uow.semantic.get_semantic_control_state()
            )
            assert reopened_control is not None
            assert reopened_control.tripped is True
            assert reopened_control.trip_code == "unauthorized_effect"
        finally:
            reopened.close()
    finally:
        if not closed:
            runtime.close()


@pytest.mark.parametrize(
    ("violation", "expected_trip"),
    (
        ("malformed", "binding_mismatch"),
        ("schema_downgrade", "binding_mismatch"),
        ("malformed_high_risk", "critical_high_grant"),
        ("replayed_use", "replay_detected"),
        ("delegable", "critical_high_grant"),
        ("extra_write_right", "critical_high_grant"),
        ("resource_drift", "unauthorized_effect"),
        ("right_drift", "unauthorized_effect"),
        ("effect_drift", "unauthorized_effect"),
    ),
)
def test_real_capability_manager_structural_violation_trip_matrix(
    violation: str,
    expected_trip: str,
    tmp_path: Path,
) -> None:
    """Persisted BindingV2 corruption trips the closed Host safety code."""

    suffix = f"manager-{violation}"
    workspace = tmp_path / f"workspace-{violation}"
    target = workspace / "reports" / f"{suffix}.txt"
    target.parent.mkdir(parents=True)
    target.write_text("manager structural trip input\n", encoding="utf-8")
    runtime = Runtime.open(
        tmp_path / f"{violation}.sqlite",
        config=_canary_config(),
        substrate=LocalResourceProviderSubstrate(workspace),
        semantic_tenant_bucketer=_tenant_bucket,
    )
    try:
        pid, resource, _flow, capability, context = (
            _issue_exact_canary_capability(
                runtime,
                target=target,
                client=_SuccessfulSemanticClient(),
                suffix=suffix,
            )
        )
        selected = capability
        decision_subject = pid
        decision_resource = resource
        decision_right = "read"
        effect_id: str | None = None

        if violation in {"malformed", "schema_downgrade", "malformed_high_risk"}:
            raw = deepcopy(
                capability.constraints[APPROVAL_BINDING_KEY]
            )
            if violation == "schema_downgrade":
                raw["schema_version"] = 1
            else:
                raw["unexpected_authority_field"] = True
            if violation == "malformed_high_risk":
                raw["authority_operation"] = "filesystem.write"
            selected = replace(
                capability,
                constraints={
                    **capability.constraints,
                    APPROVAL_BINDING_KEY: raw,
                },
            )
        elif violation == "replayed_use":
            selected = replace(
                capability,
                uses_remaining=0,
                status=CapabilityStatus.REVOKED,
            )
        elif violation == "delegable":
            selected = replace(capability, delegable=True)
        elif violation == "extra_write_right":
            selected = replace(capability, rights={"read", "write"})
        elif violation == "resource_drift":
            decision_resource = resource + ".drift"
        elif violation == "right_drift":
            decision_right = "write"
        elif violation == "effect_drift":
            effect_id = "eff_host-forged-effect"
        else:  # pragma: no cover - parameter list is the closed matrix.
            raise AssertionError(f"unknown structural violation: {violation}")

        if selected != capability:
            runtime.store.update_capability(selected)
        decision = _machine_decision(
            selected,
            subject=decision_subject,
            resource=decision_resource,
            right=decision_right,
            context=context,
        )
        with pytest.raises(CapabilityDenied):
            runtime.capability.require_semantic_approval_current(
                decision,
                phase="authorize",
                effect_id=effect_id,
                allow_reserved=False,
            )

        control = runtime.uow.semantic.get_semantic_control_state()
        assert control is not None
        assert control.tripped is True
        assert control.trip_code == expected_trip
        assert (
            runtime.capability._semantic_authority_locally_tripped  # noqa: SLF001
            is True
        )
    finally:
        runtime.close()


@pytest.mark.parametrize("denial", ("off", "stale_generation", "expired"))
def test_off_stale_or_expired_grant_denies_without_safety_trip(
    denial: str,
    tmp_path: Path,
) -> None:
    """Expected revocation/staleness is denial, not an unsafe-canary signal."""

    suffix = f"expected-denial-{denial}"
    workspace = tmp_path / f"workspace-{denial}"
    target = workspace / "reports" / f"{suffix}.txt"
    target.parent.mkdir(parents=True)
    target.write_text("expected semantic denial input\n", encoding="utf-8")
    runtime = Runtime.open(
        tmp_path / f"expected-{denial}.sqlite",
        config=_canary_config(),
        substrate=LocalResourceProviderSubstrate(workspace),
        semantic_tenant_bucketer=_tenant_bucket,
    )
    try:
        _pid, _resource, _flow, capability, context = (
            _issue_exact_canary_capability(
                runtime,
                target=target,
                client=_SuccessfulSemanticClient(),
                suffix=suffix,
            )
        )
        selected = capability
        if denial == "off":
            runtime.semantic.set_mode("off")
        else:
            raw = deepcopy(
                capability.constraints[APPROVAL_BINDING_KEY]
            )
            if denial == "stale_generation":
                raw["control_generation"] += 1
            elif denial == "expired":
                raw["issued_at"] = "1999-12-31T23:59:00+00:00"
                raw["expires_at"] = "2000-01-01T00:00:00+00:00"
            else:  # pragma: no cover - parameter list is the closed matrix.
                raise AssertionError(f"unknown expected denial: {denial}")
            metadata = deepcopy(capability.metadata)
            metadata["semantic_auto_approval"]["binding_sha256"] = (
                approval_binding_sha256(raw)
            )
            selected = replace(
                capability,
                constraints={
                    **capability.constraints,
                    APPROVAL_BINDING_KEY: raw,
                },
                metadata=metadata,
                expires_at=(
                    raw["expires_at"]
                    if denial == "expired"
                    else capability.expires_at
                ),
                issued_at=(
                    raw["issued_at"]
                    if denial == "expired"
                    else capability.issued_at
                ),
            )
            runtime.store.update_capability(selected)

        with pytest.raises(CapabilityDenied):
            runtime.capability.require_semantic_approval_current(
                _machine_decision(selected, context=context),
                phase="authorize",
                effect_id=None,
                allow_reserved=False,
            )
        control = runtime.uow.semantic.get_semantic_control_state()
        assert control is not None
        assert control.tripped is False
        assert control.trip_code is None
        assert (
            runtime.capability._semantic_authority_locally_tripped  # noqa: SLF001
            is False
        )
    finally:
        runtime.close()
