from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos.mcp.manifest import (
    McpResourceSpec,
    McpServerManifestV3,
    canonical_mcp_v3_manifest_json,
)
from agent_libos.models import (
    ExternalEffectRecord,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.human import HumanRequest, HumanRequestStatus
from agent_libos.models.mcp import (
    McpProtocolMode,
    McpServerSpec,
    McpStdioTransportSpec,
    McpToolSpec,
    canonical_mcp_server_spec_json,
)
from agent_libos.storage import (
    McpAuthMetadataRecord,
    McpContinuationRecord,
    McpRemoteTaskRecord,
    McpSideEffectPreparationRecord,
    McpSubscriptionRecord,
    SQLiteStore,
    UnitOfWork,
)


H0 = "0" * 64
H1 = "1" * 64
H2 = "2" * 64
T0 = "2026-08-11T00:00:00Z"
T1 = "2026-08-11T00:00:01Z"
EXPIRY = "2026-08-12T00:00:00Z"
_V1_REGISTRY_GOLDEN = (
    '{"http": null, "max_request_bytes": 1024, "max_response_bytes": 4096, '
    '"metadata": {}, "schema_version": 1, "server_id": "legacy.v1", '
    '"stdio": {"args": [], "command": "demo-v1", "cwd": null, "env": {}}, '
    '"timeout_s": 10.0, "tools": [{"information_flow": true, '
    '"input_schema": {}, "mcp_name": "echo", "metadata": {}, "right": '
    '"execute", "rollback_class": "none", "rollback_status": null, '
    '"state_mutation": false, "tool_id": "tool.v1"}], "transport": "stdio"}'
)
_V2_REGISTRY_GOLDEN = (
    '{"http": null, "max_request_bytes": 1024, "max_response_bytes": 4096, '
    '"metadata": {}, "protocol_mode": "legacy", "schema_version": 2, '
    '"server_id": "legacy.v2", "stdio": {"args": [], "command": "demo-v1", '
    '"cwd": null, "env": {}}, "timeout_s": 10.0, "tools": '
    '[{"information_flow": true, "input_schema": {}, "mcp_name": "echo", '
    '"metadata": {}, "right": "execute", "rollback_class": "none", '
    '"rollback_status": null, "state_mutation": false, "tool_id": "tool.v1"}], '
    '"transport": "stdio"}'
)


def _continuation(**changes: object) -> McpContinuationRecord:
    value = McpContinuationRecord(
        continuation_id="continuation-local-1",
        server_id="server.v3",
        server_spec_sha256=H0,
        server_generation=4,
        owner_id="pid-owner",
        auth_principal_sha256=H1,
        auth_scope_sha256=H2,
        request_sha256=H0,
        effect_id="effect-mcp-1",
        capability_sha256=H1,
        data_flow_sha256=H2,
        human_request_id="human-mcp-1",
        broker_ref="mcp-continuation:opaque-1",
        broker_value_sha256=H1,
        status="input_required",
        revision=0,
        expires_at=EXPIRY,
        metadata={"automatic_retry_disabled": True},
        created_at=T0,
        updated_at=T0,
    )
    return replace(value, **changes)


def _remote_task(**changes: object) -> McpRemoteTaskRecord:
    value = McpRemoteTaskRecord(
        task_ref="remote-task-local-1",
        server_id="server.v3",
        server_spec_sha256=H0,
        server_generation=4,
        owner_id="pid-owner",
        auth_principal_sha256=H1,
        auth_scope_sha256=H2,
        origin_request_sha256=H0,
        origin_effect_id="effect-mcp-1",
        human_request_id=None,
        broker_ref="mcp-task:opaque-1",
        remote_id_sha256=H1,
        status="working",
        revision=0,
        expires_at=EXPIRY,
        poll_interval_ms=500,
        status_message_sha256=None,
        result_ref=None,
        result_sha256=None,
        metadata={"automatic_retry_disabled": True},
        created_at=T0,
        updated_at=T0,
    )
    return replace(value, **changes)


def _pending_mcp_effect(*, effect_id: str, pid: str) -> ExternalEffectRecord:
    return ExternalEffectRecord(
        effect_id=effect_id,
        record_id=None,
        event_id=None,
        pid=pid,
        provider="mcp",
        operation="tools/call",
        target="mcp:server.v3:tool",
        rollback_class=ExternalEffectRollbackClass.UNKNOWN,
        rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
        state_mutation=False,
        information_flow=True,
        provider_metadata={},
        created_at=T0,
        effect_state="pending",
        transaction_state="dispatched",
        updated_at=T0,
    )


def _preparation(**changes: object) -> McpSideEffectPreparationRecord:
    value = McpSideEffectPreparationRecord(
        preparation_id="mcp-preparation-1",
        operation_kind="continuation",
        operation_id="continuation-local-1",
        operation_revision=None,
        server_id="server.v3",
        server_spec_sha256=H0,
        server_generation=4,
        owner_id="pid-owner",
        auth_principal_sha256=H1,
        auth_scope_sha256=H2,
        human_request_id="human-mcp-1",
        human_preview_sha256=H0,
        broker_ref="mcp-continuation:opaque-1",
        broker_value_sha256=H1,
        result_ref=None,
        result_sha256=None,
        status="prepared",
        revision=0,
        expires_at=EXPIRY,
        metadata={"cleanup_mode": "abort", "retire_refs": ()},
        created_at=T0,
        updated_at=T0,
    )
    return replace(value, **changes)


def _subscription(**changes: object) -> McpSubscriptionRecord:
    value = McpSubscriptionRecord(
        subscription_id="subscription-local-1",
        server_id="server.v3",
        server_spec_sha256=H0,
        server_generation=4,
        owner_id="pid-owner",
        auth_principal_sha256=H1,
        auth_scope_sha256=H2,
        requested_filter_sha256=H0,
        acknowledged_filter_sha256=None,
        status="starting",
        queue_limit=32,
        event_max_bytes=4096,
        received_count=0,
        dropped_count=0,
        revision=0,
        last_event_at=None,
        metadata={"automatic_retry_disabled": True},
        created_at=T0,
        updated_at=T0,
    )
    return replace(value, **changes)


def _auth(**changes: object) -> McpAuthMetadataRecord:
    value = McpAuthMetadataRecord(
        profile_id="oauth-profile-1",
        server_id="server.v3",
        server_spec_sha256=H0,
        server_generation=4,
        status="authorization_required",
        issuer_sha256=H0,
        resource_sha256=H1,
        audience_sha256=H2,
        scopes_sha256=H0,
        principal_sha256=None,
        expires_at=None,
        credential_generation=0,
        revision=0,
        metadata={"reason_code": "authorization_required"},
        created_at=T0,
        updated_at=T0,
    )
    return replace(value, **changes)


def _insert_human_request(unit: UnitOfWork, request_id: str) -> None:
    unit.processes.insert_human_request(
        HumanRequest(
            request_id=request_id,
            pid="pid-owner",
            human="owner",
            payload={"type": "question", "context": {}},
            status=HumanRequestStatus.PENDING,
            decision=None,
            blocking=True,
            created_at=T0,
            updated_at=T0,
        )
    )


def test_v7_records_reopen_and_keep_all_remote_secrets_out_of_store(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mcp-v7.sqlite"
    sentinel_values = (
        "access-token-SENTINEL",
        "refresh-token-SENTINEL",
        "pkce-verifier-SENTINEL",
        "oauth-state-SENTINEL",
        "remote-task-id-SENTINEL",
        "provider-result-SENTINEL",
        "notification-body-SENTINEL",
    )
    store = SQLiteStore(path)
    unit = UnitOfWork(store)
    _insert_human_request(unit, "human-mcp-1")
    unit.mcp_continuations.insert(_continuation())
    unit.mcp_remote_tasks.insert(_remote_task())
    unit.mcp_subscriptions.insert(_subscription())
    unit.mcp_auth.insert(_auth())
    store.close()

    raw = path.read_bytes()
    for sentinel in sentinel_values:
        assert sentinel.encode("utf-8") not in raw

    reopened = SQLiteStore(path)
    try:
        unit = UnitOfWork(reopened)
        assert unit.mcp_continuations.get("continuation-local-1") == _continuation()
        assert unit.mcp_remote_tasks.get("remote-task-local-1") == _remote_task()
        assert unit.mcp_subscriptions.get("subscription-local-1") == _subscription()
        assert unit.mcp_auth.get("oauth-profile-1") == _auth()
    finally:
        reopened.close()


def test_side_effect_preparation_reopens_before_side_effects_and_commits_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mcp-v7-preparation.sqlite"
    preparation = _preparation()
    store = SQLiteStore(path)
    unit = UnitOfWork(store)
    assert unit.mcp_side_effects.insert(preparation) == preparation
    store.close()

    assert b"provider-secret-MUST-NOT-PERSIST" not in path.read_bytes()
    reopened = SQLiteStore(path)
    try:
        unit = UnitOfWork(reopened)
        assert unit.mcp_side_effects.get(preparation.preparation_id) == preparation
        assert reopened.get_human_request(preparation.human_request_id) is None
        _insert_human_request(unit, preparation.human_request_id)
        continuation = _continuation()
        with pytest.raises(ValidationError, match="reserved side-effect slot"):
            unit.mcp_continuations.insert(continuation)
        assert unit.mcp_side_effects.commit(
            preparation.preparation_id,
            expected_revision=0,
            replacement=continuation,
        )
        assert unit.mcp_continuations.get(continuation.continuation_id) == continuation
        retirement = unit.mcp_side_effects.get(preparation.preparation_id)
        assert retirement == replace(
            preparation,
            status="cleaning",
            revision=1,
            metadata={
                "automatic_retry_disabled": True,
                "cleanup_mode": "retire",
                "retire_refs": (),
            },
        )
    finally:
        reopened.close()

    verified = SQLiteStore(path)
    try:
        unit = UnitOfWork(verified)
        assert unit.mcp_continuations.get("continuation-local-1") == _continuation()
        assert unit.mcp_side_effects.list() == (retirement,)
        assert unit.mcp_side_effects.delete(
            retirement.preparation_id,
            expected_revision=retirement.revision,
        )
    finally:
        verified.close()


def test_side_effect_preparation_cleanup_claim_is_single_use_and_revision_fenced() -> None:
    store = SQLiteStore(":memory:")
    try:
        repository = UnitOfWork(store).mcp_side_effects
        prepared = repository.insert(_preparation())
        assert not repository.delete(prepared.preparation_id, expected_revision=0)
        cleaning = replace(
            prepared,
            status="cleaning",
            revision=1,
            metadata={
                "cleanup_mode": "abort",
                "retire_refs": (),
                "reason_code": "runtime_restart",
            },
            updated_at=T1,
        )
        with pytest.raises(ValidationError, match="cannot forge retirement"):
            repository.compare_and_swap(
                prepared.preparation_id,
                expected_revision=0,
                replacement=replace(
                    cleaning,
                    metadata={
                        "cleanup_mode": "retire",
                        "retire_refs": (),
                    },
                ),
            )
        with pytest.raises(ValidationError, match="changed retirement ownership"):
            repository.compare_and_swap(
                prepared.preparation_id,
                expected_revision=0,
                replacement=replace(
                    cleaning,
                    metadata={
                        "cleanup_mode": "abort",
                        "retire_refs": ("opaque:forged",),
                    },
                ),
            )
        assert repository.compare_and_swap(
            prepared.preparation_id,
            expected_revision=0,
            replacement=cleaning,
        )
        assert not repository.compare_and_swap(
            prepared.preparation_id,
            expected_revision=0,
            replacement=cleaning,
        )
        assert not repository.delete(prepared.preparation_id, expected_revision=0)
        assert repository.delete(prepared.preparation_id, expected_revision=1)
        assert repository.get(prepared.preparation_id) is None
    finally:
        store.close()


def test_side_effect_retirement_metadata_is_closed_paired_and_bounded() -> None:
    with pytest.raises(ValidationError, match="at most two refs"):
        _preparation(
            metadata={
                "cleanup_mode": "abort",
                "retire_refs": ("opaque:1", "opaque:2", "opaque:3"),
            }
        )
    with pytest.raises(ValidationError, match="Human binding is incomplete"):
        _preparation(
            metadata={
                "cleanup_mode": "abort",
                "retire_refs": (),
                "retire_human_request_id": "human-mcp-1",
            }
        )
    with pytest.raises(ValidationError, match="unsupported fields"):
        _preparation(
            metadata={
                "cleanup_mode": "abort",
                "retire_refs": (),
                "token": "provider-secret-MUST-NOT-PERSIST",
            }
        )


def test_side_effect_preparation_followup_cas_rebinds_one_human_round() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        _insert_human_request(unit, "human-mcp-1")
        current = unit.mcp_continuations.insert(
            _continuation(status="dispatching")
        )
        preparation = unit.mcp_side_effects.insert(
            _preparation(
                preparation_id="mcp-preparation-round-2",
                operation_revision=0,
                human_request_id="human-mcp-2",
                human_preview_sha256=H2,
                broker_ref="mcp-continuation:opaque-2",
                broker_value_sha256=H2,
                metadata={
                    "cleanup_mode": "abort",
                    "retire_refs": ("mcp-continuation:opaque-1",),
                    "retire_human_request_id": "human-mcp-1",
                    "retire_human_preview_sha256": H1,
                },
                created_at=T1,
                updated_at=T1,
            )
        )
        _insert_human_request(unit, "human-mcp-2")
        target = replace(
            current,
            status="input_required",
            human_request_id="human-mcp-2",
            broker_ref="mcp-continuation:opaque-2",
            broker_value_sha256=H2,
            revision=1,
            updated_at=T1,
        )
        assert not unit.mcp_side_effects.commit(
            preparation.preparation_id,
            expected_revision=1,
            replacement=target,
        )
        assert unit.mcp_side_effects.get(preparation.preparation_id) == preparation
        assert unit.mcp_side_effects.commit(
            preparation.preparation_id,
            expected_revision=0,
            replacement=target,
        )
        assert unit.mcp_continuations.get(current.continuation_id) == target
        retirement = unit.mcp_side_effects.get(preparation.preparation_id)
        assert retirement == replace(
            preparation,
            status="cleaning",
            revision=1,
            metadata={
                "automatic_retry_disabled": True,
                "cleanup_mode": "retire",
                "retire_refs": (current.broker_ref,),
                "retire_human_request_id": "human-mcp-1",
                "retire_human_preview_sha256": H1,
            },
        )
        assert unit.mcp_side_effects.delete(
            retirement.preparation_id,
            expected_revision=retirement.revision,
        )
    finally:
        store.close()


def test_two_prepared_mcp_handoffs_join_one_sqlite_transaction_and_rollback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mcp-v7-composite-handoff.sqlite"
    store = SQLiteStore(path)
    unit = UnitOfWork(store)
    continuation = _continuation()
    continuation_preparation = unit.mcp_side_effects.insert(_preparation())
    task = _remote_task(
        task_ref="remote-task-atomic-handoff",
        broker_ref="mcp-task:opaque-atomic-handoff",
        remote_id_sha256=H2,
    )
    task_preparation = unit.mcp_side_effects.insert(
        _preparation(
            preparation_id="mcp-preparation-task-atomic-handoff",
            operation_kind="remote_task",
            operation_id=task.task_ref,
            human_request_id=None,
            human_preview_sha256=None,
            broker_ref=task.broker_ref,
            broker_value_sha256=task.remote_id_sha256,
        )
    )
    _insert_human_request(unit, "human-mcp-1")
    pending_effect = _pending_mcp_effect(
        effect_id=task.origin_effect_id,
        pid=task.owner_id,
    )
    unit.evidence.insert_external_effect(pending_effect)
    committed_effect = replace(
        pending_effect,
        rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
        rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
        effect_state="finalized",
        transaction_state="committed",
        provider_receipt={
            "mcp_durable_result": {
                "kind": "remote_task",
                "task_ref": task.task_ref,
            }
        },
        updated_at=T1,
    )

    with pytest.raises(RuntimeError, match="rollback composite handoff"):
        with unit.transaction():
            assert unit.mcp_side_effects.commit(
                continuation_preparation.preparation_id,
                expected_revision=0,
                replacement=continuation,
            )
            assert unit.mcp_side_effects.commit(
                task_preparation.preparation_id,
                expected_revision=0,
                replacement=task,
            )
            assert unit.evidence.finalize_external_effect(
                pending_effect.effect_id,
                committed_effect,
            )
            raise RuntimeError("rollback composite handoff")
    assert unit.mcp_continuations.get(continuation.continuation_id) is None
    assert unit.mcp_remote_tasks.get(task.task_ref) is None
    assert unit.mcp_side_effects.get(
        continuation_preparation.preparation_id
    ) == continuation_preparation
    assert unit.mcp_side_effects.get(task_preparation.preparation_id) == task_preparation
    assert unit.evidence.get_external_effect(pending_effect.effect_id) == pending_effect
    store.close()

    reopened = SQLiteStore(path)
    try:
        unit = UnitOfWork(reopened)
        assert unit.mcp_continuations.get(continuation.continuation_id) is None
        assert unit.mcp_remote_tasks.get(task.task_ref) is None
        assert unit.evidence.get_external_effect(pending_effect.effect_id) == pending_effect
        with unit.transaction():
            assert unit.mcp_side_effects.commit(
                continuation_preparation.preparation_id,
                expected_revision=0,
                replacement=continuation,
            )
            assert unit.mcp_side_effects.commit(
                task_preparation.preparation_id,
                expected_revision=0,
                replacement=task,
            )
            assert unit.evidence.finalize_external_effect(
                pending_effect.effect_id,
                committed_effect,
            )
        assert unit.mcp_continuations.get(continuation.continuation_id) == continuation
        assert unit.mcp_remote_tasks.get(task.task_ref) == task
        assert unit.evidence.get_external_effect(pending_effect.effect_id) == committed_effect
        for preparation in (continuation_preparation, task_preparation):
            retirement = unit.mcp_side_effects.get(preparation.preparation_id)
            assert retirement is not None and retirement.status == "cleaning"
            assert unit.mcp_side_effects.delete(
                retirement.preparation_id,
                expected_revision=retirement.revision,
            )
    finally:
        reopened.close()


def test_remote_task_lookup_uses_server_and_hashed_remote_id_and_accepts_zero_poll() -> None:
    store = SQLiteStore(":memory:")
    try:
        repository = UnitOfWork(store).mcp_remote_tasks
        record = repository.insert(_remote_task(poll_interval_ms=0))
        assert repository.get_by_remote_id_sha256(
            record.server_id,
            record.remote_id_sha256,
        ) == record
        assert repository.get_by_remote_id_sha256(
            "another.server",
            record.remote_id_sha256,
        ) is None
        with pytest.raises(ValidationError, match="identity conflicts"):
            repository.insert(
                _remote_task(
                    task_ref="remote-task-duplicate-remote-id",
                    broker_ref="mcp-task:opaque-duplicate-remote-id",
                )
            )
    finally:
        store.close()


def test_terminal_mcp_retention_is_bounded_revision_fenced_and_keeps_human_audit() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        _insert_human_request(unit, "human-mcp-1")
        continuation = unit.mcp_continuations.insert(_continuation())
        assert unit.mcp_continuations.count_active() == 1
        dispatching = replace(
            continuation,
            status="dispatching",
            revision=1,
            updated_at=T1,
        )
        assert unit.mcp_continuations.compare_and_swap(
            continuation.continuation_id,
            expected_revision=0,
            replacement=dispatching,
        )
        complete = replace(
            dispatching,
            status="complete",
            broker_ref=None,
            broker_value_sha256=None,
            revision=2,
            updated_at="2026-08-11T00:00:02Z",
        )
        assert unit.mcp_continuations.compare_and_swap(
            continuation.continuation_id,
            expected_revision=1,
            replacement=complete,
        )
        assert unit.mcp_continuations.count_active() == 0
        assert unit.mcp_continuations.list_terminal(limit=1) == (complete,)
        with pytest.raises(ValidationError, match="terminal MCP operation"):
            unit.mcp_side_effects.insert(
                _preparation(
                    preparation_id="mcp-preparation-terminal",
                    operation_revision=2,
                    human_request_id="human-terminal-new",
                    human_preview_sha256=H2,
                    broker_ref=None,
                    broker_value_sha256=None,
                )
            )
        assert not unit.mcp_continuations.delete_terminal(
            continuation.continuation_id,
            expected_revision=1,
        )
        with pytest.raises(ValidationError, match="durable side-effect retirement"):
            unit.mcp_continuations.delete_terminal(
                continuation.continuation_id,
                expected_revision=2,
            )
        terminal_retirement = unit.mcp_side_effects.insert(
            _preparation(
                preparation_id="mcp-preparation-terminal-retirement",
                operation_revision=2,
                human_request_id=None,
                human_preview_sha256=None,
                broker_ref=None,
                broker_value_sha256=None,
                metadata={
                    "cleanup_mode": "abort",
                    "retire_refs": (),
                    "retire_human_request_id": "human-mcp-1",
                    "retire_human_preview_sha256": H1,
                },
            )
        )
        assert unit.mcp_side_effects.commit_terminal(
            terminal_retirement.preparation_id,
            expected_revision=0,
        )
        assert unit.mcp_continuations.get(continuation.continuation_id) is None
        cleaning = unit.mcp_side_effects.get(terminal_retirement.preparation_id)
        assert cleaning is not None and cleaning.status == "cleaning"
        assert cleaning.metadata["cleanup_mode"] == "retire"
        assert unit.mcp_side_effects.delete(
            cleaning.preparation_id,
            expected_revision=cleaning.revision,
        )
        assert unit.processes.get_human_request("human-mcp-1") is not None

        attention_continuation = unit.mcp_continuations.insert(
            _continuation(
                continuation_id="mcp-continuation-needs-attention",
                human_request_id="human-mcp-1",
                status="needs_attention",
                updated_at="2026-08-11T00:00:03Z",
            )
        )
        assert unit.mcp_continuations.count_active() == 0
        assert attention_continuation in unit.mcp_continuations.list_terminal(
            limit=10,
        )
        attention_retirement = unit.mcp_side_effects.insert(
            _preparation(
                preparation_id="mcp-preparation-attention-retirement",
                operation_id=attention_continuation.continuation_id,
                operation_revision=0,
                human_request_id=None,
                human_preview_sha256=None,
                broker_ref=None,
                broker_value_sha256=None,
                metadata={
                    "cleanup_mode": "abort",
                    "retire_refs": (attention_continuation.broker_ref,),
                    "retire_human_request_id": "human-mcp-1",
                    "retire_human_preview_sha256": H1,
                },
            )
        )
        assert unit.mcp_side_effects.commit_terminal(
            attention_retirement.preparation_id,
            expected_revision=0,
        )
        attention_cleaning = unit.mcp_side_effects.get(
            attention_retirement.preparation_id
        )
        assert attention_cleaning is not None
        assert unit.mcp_side_effects.delete(
            attention_cleaning.preparation_id,
            expected_revision=attention_cleaning.revision,
        )

        task = unit.mcp_remote_tasks.insert(_remote_task())
        with pytest.raises(ValidationError, match="not terminal"):
            unit.mcp_remote_tasks.delete_terminal(
                task.task_ref,
                expected_revision=0,
            )
        terminal_task = replace(
            task,
            status="completed",
            broker_ref=None,
            revision=1,
            updated_at=T1,
        )
        assert unit.mcp_remote_tasks.compare_and_swap(
            task.task_ref,
            expected_revision=0,
            replacement=terminal_task,
        )
        assert unit.mcp_remote_tasks.count_active() == 0
        assert unit.mcp_remote_tasks.list_terminal(limit=1) == (terminal_task,)
        assert unit.mcp_remote_tasks.delete_terminal(
            task.task_ref,
            expected_revision=1,
        )
        attention_task = unit.mcp_remote_tasks.insert(
            _remote_task(
                task_ref="remote-task-needs-attention",
                remote_id_sha256=H2,
                broker_ref="mcp-task:opaque-needs-attention",
                status="needs_attention",
                updated_at="2026-08-11T00:00:03Z",
            )
        )
        assert unit.mcp_remote_tasks.count_active() == 0
        assert attention_task in unit.mcp_remote_tasks.list_terminal(limit=10)
        task_retirement = unit.mcp_side_effects.insert(
            _preparation(
                preparation_id="mcp-preparation-task-attention-retirement",
                operation_kind="remote_task",
                operation_id=attention_task.task_ref,
                operation_revision=0,
                human_request_id=None,
                human_preview_sha256=None,
                broker_ref=None,
                broker_value_sha256=None,
                metadata={
                    "cleanup_mode": "abort",
                    "retire_refs": (attention_task.broker_ref,),
                },
            )
        )
        assert unit.mcp_side_effects.commit_terminal(
            task_retirement.preparation_id,
            expected_revision=0,
        )
        task_cleaning = unit.mcp_side_effects.get(task_retirement.preparation_id)
        assert task_cleaning is not None
        assert unit.mcp_side_effects.delete(
            task_cleaning.preparation_id,
            expected_revision=task_cleaning.revision,
        )
    finally:
        store.close()


def test_continuation_cas_is_revision_owner_generation_and_expiry_fenced() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        for request_id in ("human-mcp-1", "human-mcp-2", "human-mcp-3"):
            _insert_human_request(unit, request_id)
        repository = unit.mcp_continuations
        record = repository.insert(_continuation())
        dispatching = replace(
            record,
            status="dispatching",
            revision=1,
            updated_at=T1,
        )
        assert repository.compare_and_swap(
            record.continuation_id,
            expected_revision=0,
            replacement=dispatching,
        )
        assert not repository.compare_and_swap(
            record.continuation_id,
            expected_revision=0,
            replacement=dispatching,
        )
        with pytest.raises(ValidationError, match="fenced identity"):
            repository.compare_and_swap(
                record.continuation_id,
                expected_revision=1,
                replacement=replace(
                    dispatching,
                    server_generation=5,
                    revision=2,
                ),
            )
        with pytest.raises(ValidationError, match="fenced identity"):
            repository.compare_and_swap(
                record.continuation_id,
                expected_revision=1,
                replacement=replace(
                    dispatching,
                    expires_at="2026-08-13T00:00:00Z",
                    revision=2,
                    updated_at="2026-08-11T00:00:02Z",
                ),
            )
        second_round = replace(
            dispatching,
            status="input_required",
            human_request_id="human-mcp-2",
            broker_ref="mcp-continuation:opaque-2",
            broker_value_sha256=H2,
            revision=2,
            updated_at="2026-08-11T00:00:02Z",
        )
        assert repository.compare_and_swap(
            record.continuation_id,
            expected_revision=1,
            replacement=second_round,
        )
        with pytest.raises(ValidationError, match="new input round"):
            repository.compare_and_swap(
                record.continuation_id,
                expected_revision=2,
                replacement=replace(
                    second_round,
                    human_request_id="human-mcp-3",
                    revision=3,
                    updated_at="2026-08-11T00:00:03Z",
                ),
            )
        assert repository.list(
            owner_id="pid-owner",
            server_id="server.v3",
            server_generation=4,
            expired_before="2026-08-13T00:00:00Z",
        ) == (second_round,)
        assert repository.list(server_generation=5) == ()
    finally:
        store.close()


def test_remote_task_dispatch_claim_and_terminal_clear_broker_once() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        _insert_human_request(unit, "human-remote-task-injected")
        repository = unit.mcp_remote_tasks
        working = repository.insert(_remote_task())
        assert repository.count() == 1
        assert repository.count(owner_id="pid-owner") == 1
        assert repository.count(owner_id="another-owner") == 0
        with pytest.raises(ValidationError, match="introduced only"):
            repository.compare_and_swap(
                working.task_ref,
                expected_revision=0,
                replacement=replace(
                    working,
                    human_request_id="human-remote-task-injected",
                    revision=1,
                    updated_at=T1,
                ),
            )
        dispatching = replace(
            working,
            status="cancel_dispatching",
            revision=1,
            updated_at=T1,
        )
        assert repository.compare_and_swap(
            working.task_ref,
            expected_revision=0,
            replacement=dispatching,
        )
        terminal = replace(
            dispatching,
            status="cancelled",
            broker_ref=None,
            revision=2,
            updated_at="2026-08-11T00:00:02Z",
        )
        assert repository.compare_and_swap(
            working.task_ref,
            expected_revision=1,
            replacement=terminal,
        )
        with pytest.raises(ValidationError, match="transition"):
            repository.compare_and_swap(
                working.task_ref,
                expected_revision=2,
                replacement=replace(
                    terminal,
                    status="working",
                    revision=3,
                    updated_at="2026-08-11T00:00:03Z",
                ),
            )
    finally:
        store.close()


def test_remote_task_count_is_exact_beyond_the_bounded_list_page() -> None:
    store = SQLiteStore(":memory:")
    try:
        repository = UnitOfWork(store).mcp_remote_tasks
        for index in range(501):
            repository.insert(
                _remote_task(
                    task_ref=f"remote-task-count-{index:03d}",
                    broker_ref=f"mcp-task:opaque-count-{index:03d}",
                    remote_id_sha256=hashlib.sha256(
                        f"remote-task-count-{index:03d}".encode("utf-8")
                    ).hexdigest(),
                )
            )
        assert repository.count() == 501
        assert repository.count(owner_id="pid-owner") == 501
        assert len(repository.list(limit=500)) == 500
        with pytest.raises(ValidationError, match="between 1 and 500"):
            repository.list(limit=501)
    finally:
        store.close()


def test_input_required_remote_task_requires_durable_human_request_binding() -> None:
    with pytest.raises(ValidationError, match="requires a Human request id"):
        _remote_task(status="input_required")
    record = _remote_task(
        status="input_required",
        human_request_id="human-remote-task-1",
    )
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        with pytest.raises(ValidationError, match="durable state"):
            unit.mcp_remote_tasks.insert(record)
        for request_id in ("human-remote-task-1", "human-remote-task-2"):
            _insert_human_request(unit, request_id)
        repository = unit.mcp_remote_tasks
        assert repository.insert(record) == record
        with pytest.raises(ValidationError, match="identity conflicts"):
            repository.insert(
                replace(
                    record,
                    task_ref="remote-task-local-2",
                    broker_ref="mcp-task:opaque-2",
                    remote_id_sha256=H2,
                )
            )
        with pytest.raises(ValidationError, match="new input round"):
            repository.compare_and_swap(
                record.task_ref,
                expected_revision=0,
                replacement=replace(
                    record,
                    status="working",
                    human_request_id="human-remote-task-2",
                    revision=1,
                    updated_at=T1,
                ),
            )
    finally:
        store.close()


def test_human_request_binding_is_unique_across_continuations_and_remote_tasks() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        for request_id in (
            "human-cross-table-continuation",
            "human-cross-table-task",
        ):
            _insert_human_request(unit, request_id)
        continuation = unit.mcp_continuations.insert(
            _continuation(human_request_id="human-cross-table-continuation")
        )
        with pytest.raises(ValidationError, match="already bound"):
            unit.mcp_remote_tasks.insert(
                _remote_task(
                    task_ref="remote-task-cross-table-conflict",
                    human_request_id=continuation.human_request_id,
                    broker_ref="mcp-task:opaque-cross-table-conflict",
                    remote_id_sha256=H2,
                    status="input_required",
                )
            )

        task = unit.mcp_remote_tasks.insert(
            _remote_task(
                task_ref="remote-task-cross-table-owner",
                human_request_id="human-cross-table-task",
                broker_ref="mcp-task:opaque-cross-table-owner",
                remote_id_sha256=H2,
                status="input_required",
            )
        )
        with pytest.raises(ValidationError, match="already bound"):
            unit.mcp_continuations.insert(
                _continuation(
                    continuation_id="continuation-cross-table-conflict",
                    human_request_id=task.human_request_id,
                    broker_ref="mcp-continuation:opaque-cross-table-conflict",
                )
            )

        dispatching = replace(
            continuation,
            status="dispatching",
            revision=1,
            updated_at=T1,
        )
        assert unit.mcp_continuations.compare_and_swap(
            continuation.continuation_id,
            expected_revision=0,
            replacement=dispatching,
        )
        with pytest.raises(ValidationError, match="already bound"):
            unit.mcp_continuations.compare_and_swap(
                continuation.continuation_id,
                expected_revision=1,
                replacement=replace(
                    dispatching,
                    status="input_required",
                    human_request_id=task.human_request_id,
                    broker_ref="mcp-continuation:opaque-cross-table-round",
                    revision=2,
                    updated_at="2026-08-11T00:00:02Z",
                ),
            )

        working = unit.mcp_remote_tasks.insert(
            _remote_task(
                task_ref="remote-task-cross-table-cas",
                broker_ref="mcp-task:opaque-cross-table-cas",
                remote_id_sha256=H0,
            )
        )
        with pytest.raises(ValidationError, match="already bound"):
            unit.mcp_remote_tasks.compare_and_swap(
                working.task_ref,
                expected_revision=0,
                replacement=replace(
                    working,
                    status="input_required",
                    human_request_id=continuation.human_request_id,
                    revision=1,
                    updated_at="2026-08-11T00:00:02Z",
                ),
            )
    finally:
        store.close()


def test_credential_broker_slots_are_unique_across_all_mcp_operations() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        _insert_human_request(unit, "human-mcp-1")
        continuation = unit.mcp_continuations.insert(_continuation())
        with pytest.raises(ValidationError, match="already bound"):
            unit.mcp_remote_tasks.insert(
                _remote_task(
                    task_ref="remote-task-cross-broker",
                    broker_ref=continuation.broker_ref,
                    remote_id_sha256=H2,
                )
            )
        with pytest.raises(ValidationError, match="already bound"):
            unit.mcp_remote_tasks.insert(
                _remote_task(
                    task_ref="remote-task-cross-result",
                    broker_ref="mcp-task:opaque-cross-result",
                    remote_id_sha256=H2,
                    result_ref=continuation.broker_ref,
                    result_sha256=H0,
                )
            )
        with pytest.raises(ValidationError, match="must be distinct"):
            _remote_task(
                broker_ref="mcp-task:shared-slot",
                result_ref="mcp-task:shared-slot",
                result_sha256=H0,
            )
    finally:
        store.close()


def test_subscription_counters_are_monotonic_and_events_are_not_persisted() -> None:
    store = SQLiteStore(":memory:")
    try:
        repository = UnitOfWork(store).mcp_subscriptions
        starting = repository.insert(_subscription())
        active = replace(
            starting,
            acknowledged_filter_sha256=H1,
            status="active",
            received_count=4,
            dropped_count=1,
            revision=1,
            last_event_at=T1,
            updated_at=T1,
        )
        assert repository.compare_and_swap(
            starting.subscription_id,
            expected_revision=0,
            replacement=active,
        )
        with pytest.raises(ValidationError, match="cannot decrease"):
            repository.compare_and_swap(
                starting.subscription_id,
                expected_revision=1,
                replacement=replace(
                    active,
                    received_count=3,
                    revision=2,
                    updated_at="2026-08-11T00:00:02Z",
                ),
            )
        columns = {
            row["name"] for row in store.conn.execute("PRAGMA table_info(mcp_subscriptions)")
        }
        assert not columns & {"event_json", "notification_json", "content_json", "body"}
    finally:
        store.close()


def test_auth_metadata_has_no_secret_carrier_and_cas_is_generation_monotonic() -> None:
    with pytest.raises(ValidationError, match="unsupported fields"):
        _auth(metadata={"token": "access-token-SENTINEL"})
    with pytest.raises(ValidationError, match="diagnostic code"):
        _auth(metadata={"reason_code": "access token SENTINEL"})
    # Shape validation is insufficient: a reflected secret may itself look
    # like a short lowercase diagnostic code.
    for key in (
        "dispatch_state",
        "last_error_code",
        "reason_code",
        "request_method",
        "retry_class",
    ):
        with pytest.raises(ValidationError, match="allowlisted diagnostic code"):
            _auth(metadata={key: "secret"})

    assert _auth(
        metadata={
            "automatic_retry_disabled": True,
            "dispatch_state": "not_started",
            "reason_code": "authorization_required",
            "request_method": "initialize",
            "retry_class": "not_retryable",
        }
    ).metadata["reason_code"] == "authorization_required"

    store = SQLiteStore(":memory:")
    try:
        repository = UnitOfWork(store).mcp_auth
        required = repository.insert(_auth())
        authorized = replace(
            required,
            status="authorized",
            principal_sha256=H1,
            expires_at=EXPIRY,
            credential_generation=1,
            revision=1,
            metadata={},
            updated_at=T1,
        )
        assert repository.compare_and_swap(
            required.profile_id,
            expected_revision=0,
            replacement=authorized,
        )
        columns = {
            row["name"] for row in store.conn.execute("PRAGMA table_info(mcp_auth_metadata)")
        }
        assert not columns & {
            "token", "access_token", "refresh_token", "client_secret",
            "authorization_code", "pkce_verifier", "state", "broker_ref",
        }
    finally:
        store.close()


def test_malformed_persisted_metadata_fails_closed() -> None:
    store = SQLiteStore(":memory:")
    try:
        UnitOfWork(store).mcp_auth.insert(_auth())
        store.conn.execute(
            "UPDATE mcp_auth_metadata SET metadata_json = ? WHERE profile_id = ?",
            ('{"token":"secret"}', "oauth-profile-1"),
        )
        with pytest.raises(ValidationError, match="persisted MCP v7 metadata"):
            UnitOfWork(store).mcp_auth.get("oauth-profile-1")
    finally:
        store.close()


def test_v1_v2_registry_identity_stays_stable_while_v3_round_trips(
    tmp_path: Path,
) -> None:
    v1 = McpServerSpec(
        schema_version=1,
        server_id="legacy.v1",
        transport="stdio",
        stdio=McpStdioTransportSpec(command="demo-v1"),
        timeout_s=10.0,
        max_request_bytes=1024,
        max_response_bytes=4096,
        tools=[
            McpToolSpec(
                tool_id="tool.v1",
                mcp_name="echo",
                right="execute",
                rollback_class="none",
                state_mutation=False,
                information_flow=True,
            )
        ],
    )
    v2 = replace(
        v1,
        schema_version=2,
        server_id="legacy.v2",
        protocol_mode=McpProtocolMode.LEGACY,
    )
    v3 = McpServerManifestV3(
        schema_version=3,
        server_id="modern.v3",
        transport="stdio",
        timeout_s=10.0,
        max_request_bytes=1024,
        max_response_bytes=4096,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        resources=(McpResourceSpec(resource_id="readme", remote_uri="docs://readme"),),
        stdio=McpStdioTransportSpec(command="demo-v3"),
    )
    assert canonical_mcp_server_spec_json(v1) == _V1_REGISTRY_GOLDEN
    assert canonical_mcp_server_spec_json(v2) == _V2_REGISTRY_GOLDEN
    expected = {
        "legacy.v1": _V1_REGISTRY_GOLDEN,
        "legacy.v2": _V2_REGISTRY_GOLDEN,
        "modern.v3": canonical_mcp_v3_manifest_json(v3),
    }
    path = tmp_path / "mcp-registry-v7.sqlite"
    store = SQLiteStore(path)
    unit = UnitOfWork(store)
    extension = unit.extensions
    extension.upsert_mcp_server(v1, registered_by="host", created_at=T0)
    extension.upsert_mcp_server(v2, registered_by="host", created_at=T0)
    with unit.transaction():
        assert extension.compare_and_swap_mcp_v3_server(
            v3,
            expected_current_sha256=None,
            registered_by="host",
            created_at=T0,
        )
    before_stale = extension.get_mcp_registry_binding(v3.server_id)
    assert not extension.compare_and_swap_mcp_v3_server(
        replace(v3, timeout_s=11.0),
        expected_current_sha256="f" * 64,
        registered_by="host",
        created_at=T1,
    )
    assert extension.get_mcp_registry_binding(v3.server_id) == before_stale
    current_sha256 = hashlib.sha256(expected["modern.v3"].encode("utf-8")).hexdigest()
    v3 = replace(v3, timeout_s=11.0)
    assert extension.compare_and_swap_mcp_v3_server(
        v3,
        expected_current_sha256=current_sha256,
        registered_by="host",
        created_at=T1,
    )
    expected["modern.v3"] = canonical_mcp_v3_manifest_json(v3)
    rows = {
        row["server_id"]: row["spec_json"]
        for row in store.conn.execute("SELECT server_id, spec_json FROM mcp_servers")
    }
    assert rows == expected
    assert [item[0].server_id for item in extension.list_mcp_servers()] == [
        "legacy.v1", "legacy.v2"
    ]
    assert [item[0].server_id for item in extension.list_mcp_v3_servers()] == [
        "modern.v3"
    ]
    store.close()

    reopened = SQLiteStore(path)
    try:
        extension = UnitOfWork(reopened).extensions
        assert extension.get_mcp_v3_server("modern.v3")[0] == v3
        assert extension.get_mcp_server("modern.v3") is None
        assert {
            item[0].server_id for item in extension.list_mcp_server_manifests()
        } == set(expected)
        reopened.conn.execute(
            "UPDATE mcp_servers SET spec_json = ? WHERE server_id = ?",
            (expected["modern.v3"] + " ", "modern.v3"),
        )
        reopened.conn.commit()
        with pytest.raises(ValidationError, match="Manifest v3 is not canonical"):
            extension.get_mcp_server_manifest("modern.v3")
        with pytest.raises(ValidationError, match="Manifest v3 is not canonical"):
            extension.list_mcp_server_manifests()
        with pytest.raises(ValidationError, match="Manifest v3 is not canonical"):
            extension.compare_and_swap_mcp_v3_server(
                replace(v3, timeout_s=12.0),
                expected_current_sha256=hashlib.sha256(
                    expected["modern.v3"].encode("utf-8")
                ).hexdigest(),
                registered_by="host",
                created_at=T1,
            )
    finally:
        reopened.close()


def test_v7_schema_has_no_payload_or_oauth_secret_columns() -> None:
    store = SQLiteStore(":memory:")
    try:
        forbidden = {
            "token", "secret", "pkce", "verifier", "oauth_state",
            "authorization_code", "raw", "content", "body", "event_json",
            "remote_task_id", "request_state",
        }
        for table in (
            "mcp_continuations", "mcp_remote_tasks", "mcp_subscriptions",
            "mcp_auth_metadata", "mcp_side_effect_preparations",
        ):
            columns = {
                str(row["name"]).casefold()
                for row in store.conn.execute(f"PRAGMA table_info({table})")
            }
            assert not columns & forbidden
        for table in ("mcp_continuations", "mcp_remote_tasks"):
            foreign_keys = [
                dict(row)
                for row in store.conn.execute(f"PRAGMA foreign_key_list({table})")
            ]
            assert foreign_keys == [
                {
                    "id": 0,
                    "seq": 0,
                    "table": "human_requests",
                    "from": "human_request_id",
                    "to": "request_id",
                    "on_update": "NO ACTION",
                    "on_delete": "NO ACTION",
                    "match": "NONE",
                }
            ]
    finally:
        store.close()
