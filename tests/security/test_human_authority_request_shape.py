from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import AgentLibOSConfig, SemanticDefaults
from agent_libos.human.manager import HumanObjectManager
from agent_libos.models import (
    CapabilityRight,
    HumanRequestStatus,
    SemanticHardDenyRuleV1,
    SemanticPolicyEpochV1,
)
from agent_libos.models.exceptions import HumanApprovalRequired, ValidationError
from agent_libos.runtime.boundary_descriptors import (
    HUMAN_BOUNDARIES,
    HUMAN_CONTROL_MUTATION_ADMISSION_BOUNDARIES,
)
from agent_libos.runtime.syscall_descriptors import BUILTIN_SYSCALL_NAMES
from agent_libos.semantic.exact_request import (
    decode_exact_semantic_approval_request,
    decode_host_human_approval_request,
)
from agent_libos.utils.ids import utc_now


pytestmark = pytest.mark.security


class _SecondReadAuthorityMapping(Mapping[str, Any]):
    """Expose an authority field only if the boundary reads the mapping twice."""

    def __init__(self, pid: str) -> None:
        self.reads = 0
        self._current: dict[str, Any] = {}
        self._safe = {
            "type": "question",
            "question": "Keep the admitted snapshot",
        }
        self._second = _question_with_smuggled_authority(pid)

    def __iter__(self) -> Iterator[str]:
        self.reads += 1
        self._current = self._safe if self.reads == 1 else self._second
        return iter(self._current)

    def __len__(self) -> int:
        return len(self._safe)

    def __getitem__(self, key: str) -> Any:
        return self._current[key]


def _question_with_smuggled_authority(pid: str) -> dict[str, object]:
    return {
        "type": "question",
        "question": "Which colour should I use?",
        "requested_once_capability": {
            "subject": pid,
            "resource": "object:smuggled-human-authority",
            "rights": [CapabilityRight.READ.value],
        },
    }


def _external_request(pid: str, resource: str) -> dict[str, object]:
    return {
        "type": "external_operation_approval",
        "question": "Allow the exact read?",
        "requested_once_capability": {
            "subject": pid,
            "resource": resource,
            "rights": [CapabilityRight.READ.value],
            "constraints": {},
        },
        "context": {
            "adapter": "filesystem",
            "authority_operation": "filesystem.read",
            "primitive": "runtime.filesystem.read_text",
            "operation": "read_text",
            "pid": pid,
            "resource": resource,
            "right": CapabilityRight.READ.value,
            "target_state_version": None,
        },
    }


def test_generic_human_question_rejects_smuggled_authority_at_admission() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="reject authority hidden in a question")

        with pytest.raises(ValidationError, match="generic human query.*authority"):
            runtime.human.query(
                pid,
                "owner",
                _question_with_smuggled_authority(pid),
                blocking=True,
            )

        assert runtime.human.list(pid) == []
        assert not runtime.capability.check(
            pid,
            "object:smuggled-human-authority",
            CapabilityRight.READ,
        )
    finally:
        runtime.close()


def test_typed_authority_request_is_host_control_not_model_boundary() -> None:
    assert not hasattr(
        HumanObjectManager,
        "issue_host_semantic_machine_settlement_port",
    )
    assert "human.query_authority_request" not in {
        descriptor.name for descriptor in HUMAN_BOUNDARIES
    }
    assert "human.query_authority_request" not in BUILTIN_SYSCALL_NAMES
    assert (
        "human",
        "query_authority_request",
        "control.human.query_authority_request",
    ) in HUMAN_CONTROL_MUTATION_ADMISSION_BOUNDARIES


def test_ordinary_human_question_can_still_be_answered() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="preserve ordinary Human questions")
        request_id = runtime.human.query(
            pid,
            "owner",
            {"type": "question", "question": "Which colour should I use?"},
            blocking=True,
        )

        approved = runtime.human.approve(
            request_id,
            {"approved": True, "answer": "blue"},
        )

        assert approved.status == HumanRequestStatus.APPROVED
        assert runtime.human.answer_for_request(request_id) == "blue"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "field",
    [
        "requested_permission",
        "requested_once_capability",
        "requested_capability",
        "effect_binding",
    ],
)
def test_generic_query_rejects_every_authority_shaping_field(field: str) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="reject generic Human authority")
        with pytest.raises(ValidationError, match="generic human query.*authority"):
            runtime.human.query(
                pid,
                "owner",
                {
                    "type": "approval",
                    "question": "Looks ordinary",
                    field: {},
                },
            )
    finally:
        runtime.close()


def test_generic_query_cannot_forge_host_authority_origin() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="reject forged Human origin")
        payload = _external_request(pid, "filesystem:workspace:reports/forged.txt")
        payload["_agent_libos_authority_request_origin"] = "external_operation"

        with pytest.raises(ValidationError, match="cannot supply Host authority origin"):
            runtime.human.query(pid, "owner", payload)
    finally:
        runtime.close()


def test_query_admission_uses_one_mapping_snapshot() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="snapshot Human admission input")
        hostile = _SecondReadAuthorityMapping(pid)

        request_id = runtime.human.query(pid, "owner", hostile)

        persisted = runtime.human.get(request_id)
        assert hostile.reads == 1
        assert persisted.payload["question"] == "Keep the admitted snapshot"
        assert "requested_once_capability" not in persisted.payload
        runtime.human.approve(
            request_id,
            {"approved": True, "answer": "snapshot preserved"},
        )
        assert not runtime.capability.check(
            pid,
            "object:smuggled-human-authority",
            CapabilityRight.READ,
        )
    finally:
        runtime.close()


def test_reentrant_capture_cannot_inherit_host_authority_origin() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="contain reentrant Human capture")
        attempts: list[Exception | str] = []

        def capture(_request: object) -> None:
            try:
                runtime.human.query(
                    pid,
                    "owner",
                    _question_with_smuggled_authority(pid),
                )
            except Exception as exc:  # asserted below
                attempts.append(exc)
            else:  # pragma: no cover - exploit regression signal
                attempts.append("unexpectedly accepted")

        runtime.human.set_request_capture(capture)
        runtime.human.query_authority_request(
            pid,
            "owner",
            _external_request(
                pid,
                "filesystem:workspace:reports/reentrant.txt",
            ),
            authority_origin="external_operation",
        )

        assert len(attempts) == 1
        assert isinstance(attempts[0], ValidationError)
        assert "generic human query" in str(attempts[0])
        assert len(runtime.human.list(pid)) == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("response_path", ["direct", "presentation", "terminal"])
def test_forged_durable_question_never_applies_authority(response_path: str) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="reject forged durable Human authority")
        resource = "object:forged-durable-human-authority"
        request_id = runtime.human.query(
            pid,
            "owner",
            {"type": "question", "question": "Ordinary when admitted"},
        )
        with runtime.human.requests.transaction():
            current = runtime.human.get(request_id)
            payload = dict(current.payload)
            payload["requested_once_capability"] = {
                "subject": pid,
                "resource": resource,
                "rights": [CapabilityRight.READ.value],
            }
            runtime.human.requests.replace_current(current, payload=payload)

        with pytest.raises(ValidationError, match="missing Host origin"):
            if response_path == "direct":
                runtime.human.approve(
                    request_id,
                    {"approved": True, "answer": "blue"},
                )
            elif response_path == "presentation":
                runtime.human.approve_for_presentation(
                    request_id,
                    presentation="gui",
                    decision={"approved": True, "answer": "blue"},
                )
            else:
                runtime.human.process_next_terminal(auto_answer="blue")

        assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        assert not runtime.capability.check(pid, resource, CapabilityRight.READ)
    finally:
        runtime.close()


def test_reopened_forged_durable_question_never_applies_authority(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "human-authority-shape.sqlite"
    runtime = Runtime.open(db_path)
    resource = "object:reopened-forged-human-authority"
    try:
        pid = runtime.process.spawn(goal="reject reopened Human authority")
        request_id = runtime.human.query(
            pid,
            "owner",
            {"type": "question", "question": "Ordinary when admitted"},
        )
        with runtime.human.requests.transaction():
            current = runtime.human.get(request_id)
            payload = dict(current.payload)
            payload["requested_once_capability"] = {
                "subject": pid,
                "resource": resource,
                "rights": [CapabilityRight.READ.value],
            }
            runtime.human.requests.replace_current(current, payload=payload)
    finally:
        runtime.close()

    reopened = Runtime.open(db_path)
    try:
        with pytest.raises(ValidationError, match="missing Host origin"):
            reopened.human.approve(
                request_id,
                {"approved": True, "answer": "blue"},
            )
        assert reopened.human.get(request_id).status == HumanRequestStatus.PENDING
        assert not reopened.capability.check(pid, resource, CapabilityRight.READ)
    finally:
        reopened.close()


def test_exact_external_approval_remains_revision_and_preview_fenced() -> None:
    epoch = SemanticPolicyEpochV1(
        epoch_id="epoch-human-authority-shape",
        generation=1,
        expected_previous_sha256=None,
        tenant_bucket_sha256s=(),
        auto_approval_rules=(),
        hard_deny_rules=(
            SemanticHardDenyRuleV1(
                rule_id="unrelated-deny",
                authority_operation="filesystem.delete",
                resource="filesystem:workspace:never/*",
                rights=(CapabilityRight.DELETE.value,),
            ),
        ),
        created_at=utc_now(),
    )
    runtime = Runtime.open(
        "local",
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(
                mode="enforce_deny",
                adapter="deterministic",
                policy_epoch=epoch,
                max_concurrency=1,
            )
        ),
    )
    try:
        pid = runtime.process.spawn(goal="preserve exact external Human approval")
        resource = "filesystem:workspace:reports/authority-shape.txt"
        request_id = runtime.human.query_authority_request(
            pid=pid,
            human="owner",
            authority_origin="external_operation",
            request=_external_request(pid, resource),
            blocking=True,
        )
        pending = runtime.human.get(request_id)
        preview = runtime.human.canonical_approval_preview(pending)

        with pytest.raises(ValidationError, match="expected_revision"):
            runtime.human.approve(request_id, {"approved": True})

        approved = runtime.human.approve(
            request_id,
            {"approved": True},
            expected_revision=pending.revision,
            preview_sha256=preview.canonical_sha256(),
        )

        assert approved.status == HumanRequestStatus.APPROVED
        grants = [
            capability
            for capability in runtime.capability.capabilities_for(pid)
            if capability.resource == resource
        ]
        assert len(grants) == 1
        assert grants[0].uses_remaining == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["off", "shadow"])
@pytest.mark.parametrize("operation", ["directory_delete", "git_pr_list"])
def test_scoped_primitive_human_approval_remains_compatible(
    mode: str,
    operation: str,
) -> None:
    runtime = Runtime.open(
        "local",
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(
                mode=mode,
                adapter="deterministic",
                max_concurrency=1,
            )
        ),
    )
    try:
        pid = runtime.process.spawn(goal="preserve scoped Human approvals")
        if operation == "directory_delete":
            resource = runtime.filesystem.directory_resource_for_path(
                "reports/scoped-review"
            )
            runtime.capability.set_permission_policy(
                subject=pid,
                resource=resource,
                rights=[CapabilityRight.DELETE],
                policy=CapabilityManager.ASK_EACH_TIME,
                issued_by="test.host",
            )
            invoke = lambda: runtime.filesystem.delete_directory(  # noqa: E731
                pid,
                "reports/scoped-review",
                recursive=True,
            )
        else:
            resource = "git_pr:workspace:*"
            runtime.capability.set_permission_policy(
                subject=pid,
                resource=resource,
                rights=[CapabilityRight.READ],
                policy=CapabilityManager.ASK_EACH_TIME,
                issued_by="test.host",
            )
            invoke = lambda: runtime.git.list_pull_requests(pid)  # noqa: E731

        with pytest.raises(HumanApprovalRequired):
            invoke()

        request = runtime.human.pending()[0]
        host_request = decode_host_human_approval_request(request)
        assert host_request.resource == resource
        assert "*" in host_request.resource
        with pytest.raises(ValidationError, match="resource must be exact"):
            decode_exact_semantic_approval_request(request)
        preview = runtime.human.canonical_approval_preview(request)
        assert preview.resource_sha256

        approved = runtime.human.approve(request.request_id, {"approved": True})
        assert approved.status is HumanRequestStatus.APPROVED
        assert any(
            capability.resource == resource and capability.uses_remaining == 1
            for capability in runtime.capability.capabilities_for(pid)
        )
        assert not runtime.uow.semantic.query_semantic_machine_settlements(
            after=None,
            limit=2,
            request_id=request.request_id,
        ).records
    finally:
        runtime.close()


def test_active_scoped_approval_is_human_fenced_and_not_machine_denied(
    tmp_path: Path,
) -> None:
    epoch = SemanticPolicyEpochV1(
        epoch_id="epoch-scoped-human-only",
        generation=1,
        expected_previous_sha256=None,
        tenant_bucket_sha256s=(),
        auto_approval_rules=(),
        hard_deny_rules=(
            SemanticHardDenyRuleV1(
                rule_id="unrelated-scoped-human-deny",
                authority_operation="filesystem.delete",
                resource="filesystem:workspace:never/*",
                rights=(CapabilityRight.DELETE.value,),
            ),
        ),
        created_at=utc_now(),
    )
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(
            mode="enforce_deny",
            adapter="deterministic",
            policy_epoch=epoch,
            max_concurrency=1,
        )
    )
    database = tmp_path / "scoped-human-reopen.sqlite"
    runtime = Runtime.open(database, config=config)
    try:
        pid = runtime.process.spawn(goal="keep scoped authority Human-only")
        resource = "filesystem:workspace:reports/scoped/*"
        request_id = runtime.human.query_authority_request(
            pid,
            "owner",
            _external_request(pid, resource),
            authority_origin="external_operation",
        )
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=config)
    try:
        request = reopened.human.get(request_id)
        preview = reopened.human.canonical_approval_preview(request)

        assert reopened.semantic.deterministic_deny_preflight(request) is None
        with pytest.raises(ValidationError, match="expected_revision"):
            reopened.human.approve(request_id, {"approved": True})

        approved = reopened.human.approve(
            request_id,
            {"approved": True},
            expected_revision=request.revision,
            preview_sha256=preview.canonical_sha256(),
        )
        assert approved.status is HumanRequestStatus.APPROVED
        assert not reopened.uow.semantic.query_semantic_machine_settlements(
            after=None,
            limit=2,
            request_id=request_id,
        ).records
    finally:
        reopened.close()


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_long_host_resource_is_redacted_not_rejected(mode: str) -> None:
    runtime = Runtime.open(
        "local",
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(
                mode=mode,
                adapter="deterministic",
                max_concurrency=1,
            )
        ),
    )
    try:
        pid = runtime.process.spawn(goal="preview a long Host resource safely")
        resource = "filesystem:workspace:" + ("long-segment/" * 190) + "item.txt"
        assert 2_048 < len(resource) < 65_536
        request_id = runtime.human.query_authority_request(
            pid,
            "owner",
            _external_request(pid, resource),
            authority_origin="external_operation",
        )
        request = runtime.human.get(request_id)

        assert decode_host_human_approval_request(request).resource == resource
        with pytest.raises(ValidationError, match="resource must be exact"):
            decode_exact_semantic_approval_request(request)
        preview = runtime.human.canonical_approval_preview(request)
        assert preview.resource_display == "<redacted>"
        assert len(preview.resource_sha256) == 64

        approved = runtime.human.approve(request_id, {"approved": True})
        assert approved.status is HumanRequestStatus.APPROVED
    finally:
        runtime.close()
