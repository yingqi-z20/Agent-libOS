from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import (
    PERMITTED_EFFECTS_POLICY_PROVENANCE_KEY,
    AgentProcess,
    AuditRecord,
    CapabilityRight,
    CapabilityStatus,
    ChildProcessWait,
    CheckpointPayloadDeliveryAttempt,
    CheckpointPayloadDeliveryAttemptState,
    ContextMaterializationManifest,
    DataFlowContext,
    DataFlowDecision,
    DataLabels,
    DataSourceRef,
    Event,
    EventPriority,
    EventType,
    ExitedProcessOutcome,
    ExternalEffectClassification,
    ExternalEffectRecord,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    FileLabelBinding,
    HumanRequest,
    HumanRequestStatus,
    HumanProcessWait,
    JsonRpcEndpointSpec,
    JsonRpcMethodSpec,
    JsonRpcTransportResult,
    KilledProcessOutcome,
    LLMCallRecord,
    McpHttpTransportSpec,
    McpProviderCallResult,
    McpProviderTool,
    McpServerSpec,
    McpStdioTransportSpec,
    McpToolSpec,
    McpToolListResult,
    ObjectMetadata,
    ObjectLifecycleState,
    ObjectPatch,
    ObjectTaskStatus,
    ObjectType,
    OperationOutcome,
    OperationState,
    PausedProcessWait,
    ProcessExecutionToken,
    ProcessMessageKind,
    ProcessMessageStatus,
    ProcessSignal,
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
    SinkTrustSpec,
    ToolProcessWait,
    ToolCandidateStatus,
)
from agent_libos.runtime.runtime import Runtime
from agent_libos.runtime.checkpoint_reconciliation import (
    CHECKPOINT_RESTORE_PLAN_ANCHOR_VERSION,
    CheckpointRestorePlan,
)
from agent_libos.runtime.object_task_state import ObjectTaskStateService
from agent_libos.runtime.process_manager import ProcessManager
from agent_libos.runtime.resource_manager import ResourceManager
from agent_libos.storage import StoreAssemblyReadiness, StoreAssemblyReservation
from agent_libos.process_execution import (
    bind_process_execution,
    trusted_process_execution_takeover,
    trusted_post_exec_completion_mutation,
    trusted_terminal_process_mutation,
)
from agent_libos.sdk import (
    ProtectedOperation,
    ProtectedOperationContract,
    ProtectedOperationInvocation,
    ResourcePolicy,
)
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import dumps
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    ProcessError,
    ProcessRevisionConflict,
    ProcessWaitRequired,
    RuntimePublicationPending,
    RuntimeRecoveryRequired,
    ValidationError,
)
from agent_libos.models.snapshot import SnapshotRows


_THREAD_SYNC_TIMEOUT_S = 30.0


STORE_BACKENDS = [
    "sqlite-memory",
    "sqlite-file",
    pytest.param("postgres", marks=pytest.mark.postgres),
]

PERSISTENT_STORE_BACKENDS = [
    "sqlite-file",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_runtime_store_assembly_readiness_probe_is_nonblocking(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        store = runtime.store
        assert (
            store.probe_runtime_assembly_readiness()
            is StoreAssemblyReadiness.READY
        )
        with store.locked():
            assert (
                store.probe_runtime_assembly_readiness()
                is StoreAssemblyReadiness.CURRENT_THREAD_LOCKED
            )
        with store.transaction():
            assert (
                store.probe_runtime_assembly_readiness()
                is StoreAssemblyReadiness.ACTIVE_TRANSACTION
            )

        acquired = threading.Event()
        release = threading.Event()

        def hold_store_lock() -> None:
            with store.locked():
                acquired.set()
                assert release.wait(timeout=5)

        holder = threading.Thread(target=hold_store_lock, daemon=True)
        holder.start()
        assert acquired.wait(timeout=5)
        try:
            assert (
                store.probe_runtime_assembly_readiness()
                is StoreAssemblyReadiness.LOCK_BUSY
            )
        finally:
            release.set()
            holder.join(timeout=5)
        assert not holder.is_alive()


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_runtime_store_assembly_reservation_is_exact_and_fail_fast(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        store = runtime.store
        reservation = StoreAssemblyReservation("contract-reservation")
        assert (
            store.reserve_runtime_assembly(reservation)
            is StoreAssemblyReadiness.READY
        )
        assert (
            store.probe_runtime_assembly_readiness()
            is StoreAssemblyReadiness.LOCK_BUSY
        )

        with pytest.raises(RuntimeError, match="assembly is reserved"):
            with store.locked():
                pass
        with pytest.raises(RuntimeError, match="assembly is reserved"):
            with store.transaction():
                pass
        with pytest.raises(RuntimeError, match="assembly is reserved"):
            store._query("SELECT 1")

        claim_entered = threading.Event()
        release_claim = threading.Event()
        worker_errors: list[BaseException] = []

        def claim_and_query() -> None:
            try:
                with store.claim_runtime_assembly(reservation):
                    with store.locked():
                        assert store._query("SELECT 1")
                    claim_entered.set()
                    assert release_claim.wait(timeout=5)
            except BaseException as error:
                worker_errors.append(error)

        worker = threading.Thread(target=claim_and_query, daemon=True)
        worker.start()
        assert claim_entered.wait(timeout=5)
        try:
            with pytest.raises(RuntimeError, match="already claimed"):
                with store.claim_runtime_assembly(reservation):
                    pass
            with pytest.raises(RuntimeError, match="actively claimed"):
                store.release_runtime_assembly_reservation(reservation)
            with pytest.raises(RuntimeError, match="assembly is reserved"):
                store._query("SELECT 1")
        finally:
            release_claim.set()
            worker.join(timeout=5)

        assert not worker.is_alive()
        assert worker_errors == []
        assert store.release_runtime_assembly_reservation(reservation) is False
        assert store.release_runtime_assembly_reservation(reservation) is False
        assert (
            store.probe_runtime_assembly_readiness()
            is StoreAssemblyReadiness.READY
        )


def _registry_contract_endpoint(endpoint_id: str = "binding-jsonrpc") -> JsonRpcEndpointSpec:
    return JsonRpcEndpointSpec(
        schema_version=1,
        endpoint_id=endpoint_id,
        url="https://api.example.test/jsonrpc",
        headers={},
        methods=[
            JsonRpcMethodSpec(
                method_id="echo",
                rpc_method="demo.echo",
                right="read",
                rollback_class="no_rollback_required",
                state_mutation=False,
                information_flow=True,
            )
        ],
        timeout_s=1.0,
        max_request_bytes=1024,
        max_response_bytes=2048,
    )


def _registry_contract_mcp_server(server_id: str = "binding-mcp") -> McpServerSpec:
    return McpServerSpec(
        schema_version=1,
        server_id=server_id,
        transport="stdio",
        stdio=McpStdioTransportSpec(
            command="python3",
            args=["-m", "demo_mcp"],
            env={},
        ),
        tools=[
            McpToolSpec(
                tool_id="echo",
                mcp_name="demo.echo",
                right="read",
                rollback_class="no_rollback_required",
                state_mutation=False,
                information_flow=True,
            )
        ],
        timeout_s=1.0,
        max_request_bytes=1024,
        max_response_bytes=2048,
    )


def _registry_contract_http_mcp_server(
    server_id: str = "binding-mcp-http",
) -> McpServerSpec:
    return McpServerSpec(
        schema_version=1,
        server_id=server_id,
        transport="streamable_http",
        http=McpHttpTransportSpec(
            url="https://old.example.test/mcp",
            headers={},
        ),
        tools=[
            McpToolSpec(
                tool_id="echo",
                mcp_name="demo.echo",
                right="read",
                rollback_class="no_rollback_required",
                state_mutation=False,
                information_flow=True,
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "additionalProperties": False,
                },
            )
        ],
        timeout_s=1.0,
        max_request_bytes=1024,
        max_response_bytes=2048,
    )


class _RegistryDispatchBarrierProvider:
    def __init__(self) -> None:
        self.jsonrpc_calls = 0
        self.jsonrpc_urls: list[str] = []
        self.mcp_call_tool_calls = 0
        self.mcp_list_tools_calls = 0
        self.preflight_hook: Any | None = None

    def call(
        self,
        endpoint: JsonRpcEndpointSpec,
        _method: JsonRpcMethodSpec,
        request_body: bytes,
        **_kwargs: Any,
    ) -> JsonRpcTransportResult:
        self.jsonrpc_calls += 1
        self.jsonrpc_urls.append(endpoint.url)
        request = json.loads(request_body.decode("utf-8"))
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {"ok": True},
            }
        ).encode("utf-8")
        return JsonRpcTransportResult(
            status_code=200,
            body=body,
            elapsed_s=0.01,
            response_bytes=len(body),
        )

    def validate_and_call(
        self,
        _server: McpServerSpec,
        _tool: McpToolSpec,
        arguments: dict[str, Any],
        **_kwargs: Any,
    ) -> McpProviderCallResult:
        self.mcp_call_tool_calls += 1
        return McpProviderCallResult(
            structured_content={"echo": dict(arguments)},
            response_bytes=512,
            duration_s=0.01,
            list_request_bytes=8,
            list_response_bytes=512,
            call_request_bytes=8,
            call_response_bytes=512,
            call_started=True,
        )

    def list_tools(
        self,
        server: McpServerSpec,
        **_kwargs: Any,
    ) -> McpToolListResult:
        self.mcp_list_tools_calls += 1
        return McpToolListResult(
            server_id=server.server_id,
            tools=[
                McpProviderTool(
                    name="demo.echo",
                    input_schema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "additionalProperties": False,
                    },
                )
            ],
            response_bytes=512,
            duration_s=0.01,
        )

    def classify_external_effect(
        self,
        operation: str,
        _context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if (
            isinstance(result, dict)
            and result.get("preflight") is True
            and self.preflight_hook is not None
        ):
            hook = self.preflight_hook
            self.preflight_hook = None
            hook()
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            state_mutation=operation == "call_tool",
            information_flow=True,
        )


@pytest.mark.parametrize("kind", STORE_BACKENDS)
@pytest.mark.parametrize(
    "failure_type",
    [RuntimeError, KeyboardInterrupt],
    ids=["exception", "base_exception"],
)
def test_outer_store_mutation_guard_rejection_rolls_back_before_return(
    kind: str,
    failure_type: type[BaseException],
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        namespace = f"guard-rejected-{failure_type.__name__.lower()}"
        interruption = failure_type("injected outer mutation guard rejection")
        original_guard = runtime.store._admission_commit_guard

        @contextlib.contextmanager
        def reject_commit() -> Iterator[None]:
            raise interruption
            yield

        runtime.store._admission_commit_guard = reject_commit
        try:
            with pytest.raises(failure_type) as caught:
                runtime.store._execute(
                    """
                    INSERT INTO object_namespaces (
                        namespace, parent_namespace, metadata_json,
                        created_by, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (namespace, None, "{}", "test", "1", "1"),
                )
        finally:
            runtime.store._admission_commit_guard = original_guard

        assert caught.value is interruption
        assert runtime.store.select_table_rows(
            "object_namespaces",
            "namespace = ?",
            (namespace,),
        ) == []


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_outer_mutation_helper_joins_explicit_and_nested_transactions(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        original_guard = runtime.store._admission_commit_guard
        calls: list[str] = []

        @contextlib.contextmanager
        def count_commit() -> Iterator[None]:
            calls.append("enter")
            try:
                with original_guard():
                    yield
            finally:
                calls.append("exit")

        runtime.store._admission_commit_guard = count_commit
        try:
            with runtime.store.transaction():
                runtime.store._execute(
                    "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                    ("guard-outer", None, "{}", "test", "1", "1"),
                )
                with runtime.store.transaction():
                    runtime.store._execute(
                        "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                        ("guard-inner", None, "{}", "test", "1", "1"),
                    )
        finally:
            runtime.store._admission_commit_guard = original_guard

        assert calls == ["enter", "exit"]
        assert {
            row["namespace"]
            for row in runtime.store.select_table_rows(
                "object_namespaces",
                "namespace IN (?, ?)",
                ("guard-outer", "guard-inner"),
            )
        } == {"guard-outer", "guard-inner"}


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_direct_cas_mutations_share_outer_commit_guard(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="outer mutation direct CAS contract")
        capability = runtime.capability.issue_trusted(
            pid,
            "custom:outer-mutation-contract",
            [CapabilityRight.READ],
            issued_by="test",
            uses_remaining=2,
        )

        def reject(call: Callable[[], object], label: str) -> None:
            interruption = RuntimeError(f"injected {label} commit rejection")
            original_guard = runtime.store._admission_commit_guard

            @contextlib.contextmanager
            def reject_commit() -> Iterator[None]:
                raise interruption
                yield

            runtime.store._admission_commit_guard = reject_commit
            try:
                with pytest.raises(RuntimeError) as caught:
                    call()
            finally:
                runtime.store._admission_commit_guard = original_guard
            assert caught.value is interruption

        reject(
            lambda: runtime.store.consume_capability_uses(capability.cap_id),
            "capability consume",
        )
        unchanged = runtime.store.get_capability(capability.cap_id)
        assert unchanged is not None and unchanged.uses_remaining == 2

        reservation_id = "outer-mutation-reservation"
        reserved = runtime.store.reserve_capability_uses(
            capability.cap_id,
            reservation_id,
            reserved_by="test",
            reason="outer mutation contract",
            created_at=utc_now(),
        )
        assert reserved is not None and reserved.uses_remaining == 1
        reject(
            lambda: runtime.store.commit_capability_use_reservation(
                reservation_id,
                updated_at=utc_now(),
            ),
            "capability reservation",
        )
        reservation = runtime.store.select_table_rows(
            "capability_use_reservations",
            "reservation_id = ?",
            (reservation_id,),
        )
        assert [row["status"] for row in reservation] == ["reserved"]

        resume_token = "outer-mutation-resume"
        runtime.store.upsert_llm_pending_action(
            pid,
            {
                "wait_type": "human",
                "request_id": "outer-mutation-human",
                "resume_token": resume_token,
                "action": {"action": "ask_human", "question": "continue?"},
                "data_flow_context": DataFlowContext().to_dict(),
                "content_preview": "",
                "tool_call_count": 1,
                "status": "pending",
            },
        )
        reject(
            lambda: runtime.store.claim_llm_pending_action(
                pid,
                resume_token=resume_token,
            ),
            "pending action claim",
        )
        pending = runtime.store.get_llm_pending_action(pid)
        assert pending is not None and pending["status"] == "pending"


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_snapshot_repository_exec_round_trip_uses_canonical_aggregate(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="snapshot repository contract")
        handle = runtime.memory.create_object(
            pid,
            ObjectType.SUMMARY,
            {"version": 1},
            name="snapshot-contract",
            immutable=False,
        )
        process_rows = runtime.uow.snapshots.load_process_snapshot_rows([pid])
        rows, payloads = runtime.uow.snapshots.capture_checkpoint_rows(
            process_rows.processes,
            object_oids=[handle.oid],
            namespace_names=[runtime.memory.process_namespace(pid)],
        )

        assert isinstance(rows, SnapshotRows)
        assert [row["pid"] for row in rows.processes] == [pid]
        assert payloads[handle.oid] == {"version": 1}

        state = runtime.process_exec_state.capture(pid)
        runtime.memory.update_object(
            pid,
            handle,
            ObjectPatch(payload={"version": 2}),
        )
        runtime.process_exec_state.restore(state, fence_execution=False)

        restored = runtime.memory.get_object_by_name(pid, "snapshot-contract")
        assert restored.payload == {"version": 1}


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_checkpoint_repository_restore_reconstructs_payload_after_reopen(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        pid = runtime.process.spawn(goal="checkpoint repository reopen")
        handle = runtime.memory.create_object(
            pid,
            ObjectType.SUMMARY,
            {"version": 1},
            name="reopen-snapshot-contract",
            immutable=False,
        )
        checkpoint_id = runtime.checkpoint.create(
            pid,
            "repository reopen contract",
            actor=pid,
        )
        runtime.memory.update_object(
            pid,
            handle,
            ObjectPatch(payload={"version": 2}),
        )
        runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            reopened.checkpoint.restore(
                "test",
                checkpoint_id,
                require_capability=False,
            )
            restored = reopened.store.get_object(handle.oid)
            assert restored is not None
            assert reopened.store.object_payload(handle.oid) == {"version": 1}
            # Reopen revokes handles to volatile payloads. Restore reconstructs
            # the checkpoint payload but deliberately preserves revoke-wins.
            restored_handle = reopened.store.get_capability(handle.capability_id)
            assert (
                restored_handle is None
                or restored_handle.status == CapabilityStatus.REVOKED
            )
            assert not reopened.capability.check(
                pid,
                f"object:{handle.oid}",
                CapabilityRight.READ,
            )
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_checkpoint_repository_restore_rolls_back_rows_and_payloads(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="checkpoint repository rollback")
        handle = runtime.memory.create_object(
            pid,
            ObjectType.SUMMARY,
            {"version": 1},
            name="rollback-snapshot-contract",
            immutable=False,
        )
        checkpoint_id = runtime.checkpoint.create(
            pid,
            "repository rollback contract",
            actor=pid,
        )
        runtime.memory.update_object(
            pid,
            handle,
            ObjectPatch(payload={"version": 2}),
        )
        original_insert = runtime.checkpoint._insert_row

        def fail_process_insert(cursor: object, table: str, row: dict[str, object]) -> None:
            if table == "processes":
                raise RuntimeError("injected snapshot repository rollback")
            original_insert(cursor, table, row)

        monkeypatch.setattr(runtime.checkpoint, "_insert_row", fail_process_insert)

        with pytest.raises(RuntimeError, match="snapshot repository rollback"):
            runtime.checkpoint.restore(
                "test",
                checkpoint_id,
                require_capability=False,
            )

        current = runtime.memory.get_object_by_name(
            pid,
            "rollback-snapshot-contract",
        )
        assert current.payload == {"version": 2}
        assert runtime.process.get(pid).pid == pid


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_provider_registry_bindings_are_versioned_and_independent(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        endpoint = _registry_contract_endpoint()
        server = _registry_contract_mcp_server()
        absent_jsonrpc = runtime.store.get_jsonrpc_registry_binding(endpoint.endpoint_id)
        absent_mcp = runtime.store.get_mcp_registry_binding(server.server_id)

        assert absent_jsonrpc["registry_generation"] == 0
        assert absent_mcp["registry_generation"] == 0
        assert len(absent_jsonrpc["registry_spec_sha256"]) == 64
        assert len(absent_mcp["registry_spec_sha256"]) == 64

        runtime.store.upsert_jsonrpc_endpoint(
            endpoint,
            registered_by="test",
            created_at=utc_now(),
        )
        first = runtime.store.get_jsonrpc_registry_binding(endpoint.endpoint_id)
        assert first == {
            "registry_generation": 1,
            "registry_spec_sha256": hashlib.sha256(
                dumps(endpoint).encode("utf-8")
            ).hexdigest(),
        }
        assert runtime.store.get_mcp_registry_binding(server.server_id) == absent_mcp

        runtime.store.upsert_jsonrpc_endpoint(
            endpoint,
            registered_by="test",
            created_at=utc_now(),
        )
        repeated = runtime.store.get_jsonrpc_registry_binding(endpoint.endpoint_id)
        assert repeated["registry_generation"] == 2
        assert repeated["registry_spec_sha256"] == first["registry_spec_sha256"]

        runtime.store.delete_jsonrpc_endpoint(endpoint.endpoint_id)
        deleted = runtime.store.get_jsonrpc_registry_binding(endpoint.endpoint_id)
        assert deleted["registry_generation"] == 3
        assert deleted["registry_spec_sha256"] == absent_jsonrpc["registry_spec_sha256"]

        runtime.store.upsert_mcp_server(
            server,
            registered_by="test",
            created_at=utc_now(),
        )
        current_mcp = runtime.store.get_mcp_registry_binding(server.server_id)
        assert current_mcp == {
            "registry_generation": 1,
            "registry_spec_sha256": hashlib.sha256(
                dumps(server).encode("utf-8")
            ).hexdigest(),
        }


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_provider_registry_generation_survives_reopen(kind: str, tmp_path: Path) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        endpoint = _registry_contract_endpoint("durable-binding")
        runtime = Runtime.open(target, config=config)
        try:
            runtime.store.upsert_jsonrpc_endpoint(
                endpoint,
                registered_by="test",
                created_at=utc_now(),
            )
            expected = runtime.store.get_jsonrpc_registry_binding(endpoint.endpoint_id)
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            assert reopened.store.get_jsonrpc_registry_binding(endpoint.endpoint_id) == expected
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
@pytest.mark.parametrize("adapter", ["jsonrpc", "mcp"])
def test_typed_provider_spec_integer_timeout_is_canonical_across_reopen(
    kind: str,
    adapter: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed construction spelling cannot invalidate a live registry binding."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        provider = _RegistryDispatchBarrierProvider()
        try:
            pid = runtime.process.spawn(goal=f"canonical {adapter} registry spec")
            if adapter == "jsonrpc":
                spec = replace(
                    _registry_contract_endpoint("canonical-timeout-jsonrpc"),
                    # This contract exercises integer/float canonicalization,
                    # not the provider deadline. Leave enough dispatch budget
                    # for a loaded Windows runner.
                    timeout_s=30,
                )
                assert type(spec.timeout_s) is int
                runtime.jsonrpc.register_endpoint(
                    spec,
                    actor="test.host",
                    require_capability=False,
                )
                runtime.capability.grant(
                    pid,
                    runtime.jsonrpc.method_resource(spec.endpoint_id, "echo"),
                    [CapabilityRight.READ],
                    issued_by="test.host",
                )
                table = "jsonrpc_endpoints"
                id_column = "endpoint_id"
                item_id = spec.endpoint_id
                first_binding = runtime.store.get_jsonrpc_registry_binding(item_id)
            else:
                spec = replace(
                    _registry_contract_http_mcp_server("canonical-timeout-mcp"),
                    timeout_s=30,
                )
                assert type(spec.timeout_s) is int
                runtime.mcp.register_server(
                    spec,
                    actor="test.host",
                    require_capability=False,
                )
                runtime.capability.grant(
                    pid,
                    runtime.mcp.server_resource(spec.server_id),
                    [CapabilityRight.READ, CapabilityRight.EXECUTE],
                    issued_by="test.host",
                )
                runtime.capability.grant(
                    pid,
                    runtime.mcp.tool_resource(spec.server_id, "echo"),
                    [CapabilityRight.READ],
                    issued_by="test.host",
                )
                table = "mcp_servers"
                id_column = "server_id"
                item_id = spec.server_id
                first_binding = runtime.store.get_mcp_registry_binding(item_id)

            raw_rows = runtime.store._query(
                f"SELECT spec_json FROM {table} WHERE {id_column} = ?",
                (item_id,),
            )
            assert len(raw_rows) == 1
            assert type(json.loads(raw_rows[0]["spec_json"])["timeout_s"]) is float
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            monkeypatch.setattr(
                reopened.jsonrpc if adapter == "jsonrpc" else reopened.mcp,
                "_validate_runtime_resolution",
                lambda _spec, **_kwargs: ("93.184.216.34",),
            )
            if adapter == "jsonrpc":
                reopened.jsonrpc.provider = provider
                loaded, _metadata = reopened.jsonrpc._load_endpoint(item_id)
                current_binding = reopened.store.get_jsonrpc_registry_binding(item_id)
                assert current_binding == first_binding
                assert current_binding["registry_spec_sha256"] == (
                    reopened.jsonrpc._endpoint_spec_sha256(loaded)
                )
                assert reopened.jsonrpc.call(pid, item_id, "echo", {"value": 1}).ok
                assert provider.jsonrpc_calls == 1
            else:
                reopened.mcp.provider = provider
                loaded, _metadata = reopened.mcp._load_server(item_id)
                current_binding = reopened.store.get_mcp_registry_binding(item_id)
                assert current_binding == first_binding
                assert current_binding["registry_spec_sha256"] == (
                    reopened.mcp._server_spec_sha256(loaded)
                )
                refreshed = reopened.mcp.list_tools(
                    item_id,
                    actor=pid,
                    require_capability=True,
                    refresh=True,
                )
                assert refreshed["refreshed"] is True
                assert reopened.mcp.call_tool(
                    pid,
                    item_id,
                    "echo",
                    {"text": "hello"},
                ).ok
                assert provider.mcp_list_tools_calls == 1
                assert provider.mcp_call_tool_calls == 1
        finally:
            reopened.close()


def _run_joined_registry_mutation(mutation: Any) -> None:
    errors: list[BaseException] = []

    def run() -> None:
        try:
            mutation()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    if errors:
        raise errors[0]


@pytest.mark.parametrize("kind", STORE_BACKENDS)
@pytest.mark.parametrize("mutation", ["same-spec", "replace", "unregister"])
@pytest.mark.parametrize("adapter", ["jsonrpc", "mcp"])
def test_provider_approval_revalidates_registry_binding_before_first_dispatch(
    kind: str,
    mutation: str,
    adapter: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        provider = _RegistryDispatchBarrierProvider()
        pid = runtime.process.spawn(goal=f"{adapter} live registry approval")
        resolution_calls: list[str] = []

        if adapter == "jsonrpc":
            endpoint = _registry_contract_endpoint("approval-dispatch-jsonrpc")
            runtime.jsonrpc.provider = provider
            runtime.jsonrpc.register_endpoint(
                endpoint,
                actor="test.host",
                require_capability=False,
            )
            resource = runtime.jsonrpc.method_resource(endpoint.endpoint_id, "echo")
            runtime.capability.set_permission_policy(
                pid,
                resource,
                [CapabilityRight.READ],
                runtime.capability.ASK_EACH_TIME,
                issued_by="test.host",
            )
            with pytest.raises(HumanApprovalRequired):
                runtime.jsonrpc.call(pid, endpoint.endpoint_id, "echo", {"value": 1})
            runtime.human.drain_terminal_queue(auto_approve=True)

            def mutate_registry() -> None:
                if mutation == "unregister":
                    runtime.jsonrpc.unregister_endpoint(
                        endpoint.endpoint_id,
                        actor="test.host",
                        require_capability=False,
                    )
                    return
                replacement = endpoint
                if mutation == "replace":
                    replacement = replace(
                        endpoint,
                        url="https://new.example.test/jsonrpc",
                        methods=[
                            replace(
                                endpoint.methods[0],
                                rpc_method="dangerous.mutate",
                            )
                        ],
                    )
                runtime.jsonrpc.register_endpoint(
                    replacement,
                    actor="test.host",
                    replace=True,
                    require_capability=False,
                )

            monkeypatch.setattr(
                runtime.jsonrpc,
                "_require_header_environment",
                lambda _spec: _run_joined_registry_mutation(mutate_registry),
            )
            monkeypatch.setattr(
                runtime.jsonrpc,
                "_validate_runtime_resolution",
                lambda spec: resolution_calls.append(spec.url)
                or ("93.184.216.34",),
            )
            invoke = lambda: runtime.jsonrpc.call(
                pid,
                endpoint.endpoint_id,
                "echo",
                {"value": 1},
            )
            provider_calls = lambda: provider.jsonrpc_calls
            operation_name = "primitive.jsonrpc.call"
        else:
            server = _registry_contract_http_mcp_server("approval-dispatch-mcp")
            runtime.mcp.provider = provider
            runtime.mcp.register_server(
                server,
                actor="test.host",
                require_capability=False,
            )
            resource = runtime.mcp.tool_resource(server.server_id, "echo")
            runtime.capability.set_permission_policy(
                pid,
                resource,
                [CapabilityRight.READ],
                runtime.capability.ASK_EACH_TIME,
                issued_by="test.host",
            )
            with pytest.raises(HumanApprovalRequired):
                runtime.mcp.call_tool(
                    pid,
                    server.server_id,
                    "echo",
                    {"text": "hello"},
                )
            runtime.human.drain_terminal_queue(auto_approve=True)

            def mutate_registry() -> None:
                if mutation == "unregister":
                    runtime.mcp.unregister_server(
                        server.server_id,
                        actor="test.host",
                        require_capability=False,
                    )
                    return
                replacement = server
                if mutation == "replace":
                    replacement = replace(
                        server,
                        http=McpHttpTransportSpec(
                            url="https://new.example.test/mcp",
                            headers={},
                        ),
                        tools=[
                            replace(
                                server.tools[0],
                                mcp_name="dangerous.mutate",
                            )
                        ],
                    )
                runtime.mcp.register_server(
                    replacement,
                    actor="test.host",
                    replace=True,
                    require_capability=False,
                )

            monkeypatch.setattr(
                runtime.mcp,
                "_require_runtime_environment",
                lambda _spec, **_kwargs: _run_joined_registry_mutation(mutate_registry),
            )
            monkeypatch.setattr(
                runtime.mcp,
                "_validate_runtime_resolution",
                lambda spec, **_kwargs: resolution_calls.append(spec.http.url)
                or ("93.184.216.34",),
            )
            invoke = lambda: runtime.mcp.call_tool(
                pid,
                server.server_id,
                "echo",
                {"text": "hello"},
            )
            provider_calls = lambda: provider.mcp_call_tool_calls
            operation_name = "primitive.mcp.call"

        one_shot = [
            capability
            for capability in runtime.store.list_capabilities(subject=pid)
            if capability.resource == resource and capability.uses_remaining == 1
        ]
        assert len(one_shot) == 1

        with pytest.raises(
            CapabilityDenied,
            match="provider registry binding changed before protected dispatch",
        ):
            invoke()

        assert resolution_calls == []
        assert provider_calls() == 0
        assert [
            effect
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider == adapter
        ] == []
        restored = runtime.store.get_capability(one_shot[0].cap_id)
        assert restored is not None
        assert restored.active
        assert restored.uses_remaining == 1
        operation = next(
            record
            for record in runtime.store.list_operations(pid=pid)
            if record.name == operation_name
        )
        assert operation.state == OperationState.TERMINAL
        assert operation.outcome == OperationOutcome.DENIED


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_provider_registry_mutation_serializes_after_inflight_phase(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        provider = _RegistryDispatchBarrierProvider()
        runtime.jsonrpc.provider = provider
        pid = runtime.process.spawn(goal="registry phase linearization")
        endpoint = replace(
            _registry_contract_endpoint("inflight-registry-jsonrpc"),
            # The synchronization barrier below may wait for 30 seconds on a
            # loaded CI host; this contract test is not exercising deadlines.
            timeout_s=60.0,
        )
        runtime.jsonrpc.register_endpoint(
            endpoint,
            actor="test.host",
            require_capability=False,
        )
        resource = runtime.jsonrpc.method_resource(endpoint.endpoint_id, "echo")
        runtime.capability.set_permission_policy(
            pid,
            resource,
            [CapabilityRight.READ],
            runtime.capability.ASK_EACH_TIME,
            issued_by="test.host",
        )
        with pytest.raises(HumanApprovalRequired):
            runtime.jsonrpc.call(pid, endpoint.endpoint_id, "echo", {"value": 1})
        runtime.human.drain_terminal_queue(auto_approve=True)
        initial_binding = runtime.store.get_jsonrpc_registry_binding(
            endpoint.endpoint_id
        )
        replacement = replace(
            endpoint,
            url="https://new.example.test/jsonrpc",
            methods=[
                replace(endpoint.methods[0], rpc_method="dangerous.mutate")
            ],
        )

        release_mutation = threading.Event()
        mutation_attempted = threading.Event()
        mutation_done = threading.Event()
        mutation_errors: list[BaseException] = []

        def mutate_registry() -> None:
            try:
                assert release_mutation.wait(timeout=_THREAD_SYNC_TIMEOUT_S)
                mutation_attempted.set()
                runtime.jsonrpc.register_endpoint(
                    replacement,
                    actor="test.host",
                    replace=True,
                    require_capability=False,
                )
                mutation_done.set()
            except BaseException as error:
                mutation_errors.append(error)

        worker = threading.Thread(target=mutate_registry)
        worker.start()
        original_revalidate = ProtectedOperation._revalidate_provider_registry_binding
        phase_count = 0

        def revalidate_with_barrier(operation: ProtectedOperation) -> None:
            nonlocal phase_count
            original_revalidate(operation)
            if (
                operation.contract.name == "primitive.jsonrpc.call"
                and operation.invocation.pid == pid
            ):
                phase_count += 1
                if phase_count == 2:
                    release_mutation.set()
                    assert mutation_attempted.wait(
                        timeout=_THREAD_SYNC_TIMEOUT_S
                    )
                    assert not mutation_done.is_set()

        monkeypatch.setattr(
            ProtectedOperation,
            "_revalidate_provider_registry_binding",
            revalidate_with_barrier,
        )
        monkeypatch.setattr(
            runtime.jsonrpc,
            "_validate_runtime_resolution",
            lambda _spec: ("93.184.216.34",),
        )
        try:
            result = runtime.jsonrpc.call(
                pid,
                endpoint.endpoint_id,
                "echo",
                {"value": 1},
            )
        finally:
            release_mutation.set()
            worker.join(timeout=_THREAD_SYNC_TIMEOUT_S)

        assert not worker.is_alive()
        assert mutation_errors == []
        assert mutation_done.is_set()
        assert result.ok
        assert phase_count == 2
        assert provider.jsonrpc_calls == 1
        assert provider.jsonrpc_urls == [endpoint.url]
        current_binding = runtime.store.get_jsonrpc_registry_binding(
            endpoint.endpoint_id
        )
        assert current_binding["registry_generation"] > initial_binding[
            "registry_generation"
        ]
        assert current_binding["registry_spec_sha256"] != initial_binding[
            "registry_spec_sha256"
        ]


@pytest.mark.parametrize("kind", STORE_BACKENDS)
@pytest.mark.parametrize("mutation", ["same-spec", "replace", "unregister"])
def test_mcp_list_tools_revalidates_registry_binding_before_first_dispatch(
    kind: str,
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        provider = _RegistryDispatchBarrierProvider()
        runtime.mcp.provider = provider
        server = _registry_contract_http_mcp_server("list-dispatch-mcp")
        runtime.mcp.register_server(
            server,
            actor="test.host",
            require_capability=False,
        )
        pid = runtime.process.spawn(goal="MCP list live registry binding")
        runtime.capability.grant(
            pid,
            runtime.mcp.server_resource(server.server_id),
            [CapabilityRight.READ, CapabilityRight.EXECUTE],
            issued_by="test.host",
        )

        def mutate_registry() -> None:
            if mutation == "unregister":
                runtime.mcp.unregister_server(
                    server.server_id,
                    actor="test.host",
                    require_capability=False,
                )
                return
            replacement = server
            if mutation == "replace":
                replacement = replace(
                    server,
                    http=McpHttpTransportSpec(
                        url="https://new.example.test/mcp",
                        headers={},
                    ),
                )
            runtime.mcp.register_server(
                replacement,
                actor="test.host",
                replace=True,
                require_capability=False,
            )

        provider.preflight_hook = lambda: _run_joined_registry_mutation(
            mutate_registry
        )
        resolution_calls: list[str] = []
        monkeypatch.setattr(
            runtime.mcp,
            "_validate_runtime_resolution",
            lambda spec, **_kwargs: resolution_calls.append(spec.http.url)
            or ("93.184.216.34",),
        )

        with pytest.raises(
            CapabilityDenied,
            match="provider registry binding changed before protected dispatch",
        ):
            runtime.mcp.list_tools(
                server.server_id,
                actor=pid,
                require_capability=True,
                refresh=True,
            )

        assert resolution_calls == []
        assert provider.mcp_list_tools_calls == 0
        assert [
            effect
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider == "mcp"
        ] == []
        operation = next(
            record
            for record in runtime.store.list_operations(pid=pid)
            if record.name == "primitive.mcp.list_tools"
        )
        assert operation.state == OperationState.TERMINAL
        assert operation.outcome == OperationOutcome.DENIED


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_effect_policy_provenance_blocks_deny_all_downgrade_after_reopen(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        deny_all_pid = runtime.process.spawn(
            goal="persisted deny-all effect ceiling",
            authority_manifest={"permitted_effects": []},
        )
        intact_pid = runtime.process.spawn(
            goal="intact deny-all effect ceiling",
            authority_manifest={"permitted_effects": []},
        )
        manifest = runtime.authority_manifests.get_for_process(deny_all_pid)
        assert manifest is not None
        runtime.store._execute(
            """
            UPDATE authority_manifests
               SET permitted_effects_json = ?
             WHERE manifest_id = ?
            """,
            ("[]", manifest.manifest_id),
        )
        runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            with pytest.raises(ValidationError, match="authority manifest hash mismatch"):
                reopened.authority_manifests.get_for_process(deny_all_pid)
            assert (
                reopened.authority_manifests.get_for_process(intact_pid).permitted_effects
                == []
            )
            with pytest.raises(CapabilityDenied, match="does not permit effect class"):
                reopened.authority_manifests.assert_effect(
                    intact_pid,
                    "jsonrpc.call",
                )
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_legacy_effect_policy_fallback_requires_legacy_provenance_after_reopen(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        pid = runtime.process.spawn(goal="legacy unrestricted effect policy")
        manifest = runtime.authority_manifests.get_for_process(pid)
        assert manifest is not None
        legacy_metadata = dict(manifest.metadata)
        legacy_metadata.pop(PERMITTED_EFFECTS_POLICY_PROVENANCE_KEY)
        legacy_manifest = replace(
            manifest,
            permitted_effects=None,
            permitted_effects_policy_schema_version=1,
            manifest_hash="",
            metadata=legacy_metadata,
        )
        legacy_hash = runtime.authority_manifests._legacy_hash(legacy_manifest)
        runtime.store._execute(
            """
            UPDATE authority_manifests
               SET permitted_effects_json = ?, metadata_json = ?, manifest_hash = ?
             WHERE manifest_id = ?
            """,
            (
                "[]",
                dumps(legacy_metadata),
                legacy_hash,
                manifest.manifest_id,
            ),
        )
        runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            restored = reopened.authority_manifests.get_for_process(pid)
            assert restored is not None
            assert restored.permitted_effects_policy_schema_version == 1
            assert restored.permitted_effects is None
            reopened.authority_manifests.assert_effect(pid, "jsonrpc.call")
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_bound_worker_token_cannot_fall_back_to_cross_pid_host_write(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        source = runtime.process.spawn(goal="worker token owner")
        target = runtime.process.spawn(goal="cross-pid mutation target")
        token = runtime.store.claim_execution(source, owner_id="test.worker")
        assert token is not None
        before = runtime.process.get(target)

        with bind_process_execution(token):
            with pytest.raises(ProcessRevisionConflict, match="cannot mutate"):
                runtime.store.patch_process(
                    target,
                    {"status_message": "must not commit"},
                    expected_revision=before.revision,
                )

        unchanged = runtime.process.get(target)
        assert unchanged.revision == before.revision
        assert unchanged.status_message == before.status_message

        with bind_process_execution(token):
            with pytest.raises(ProcessRevisionConflict, match="revision conflict"):
                runtime.store.patch_process_control(
                    target,
                    {"status_message": "stale control revision"},
                    expected_revision=unchanged.revision - 1,
                    allowed_statuses={ProcessStatus.RUNNABLE},
                    reason="test stale control revision",
                )
            with pytest.raises(ProcessRevisionConflict, match="control status conflict"):
                runtime.store.patch_process_control(
                    target,
                    {"status_message": "wrong control status"},
                    expected_revision=unchanged.revision,
                    allowed_statuses={ProcessStatus.PAUSED},
                    reason="test wrong control status",
                )
            controlled = runtime.store.patch_process_control(
                target,
                {"status_message": "explicit host control"},
                expected_revision=unchanged.revision,
                allowed_statuses={ProcessStatus.RUNNABLE},
                reason="test explicit cross-pid control",
            )
        assert controlled.status_message == "explicit host control"
        assert runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE)


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_claim_execution_rolls_back_process_and_high_water_on_counter_failure(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="atomic execution claim")
        before = runtime.process.get(pid)
        counter_names = runtime.store._process_concurrency_counter_names(pid)

        def counter_values() -> dict[str, int]:
            rows = runtime.store._query(
                """
                SELECT counter_name, value
                  FROM runtime_counters
                 WHERE counter_name IN (?, ?, ?)
                """,
                counter_names,
            )
            return {str(row["counter_name"]): int(row["value"]) for row in rows}

        before_counters = counter_values()
        original_raise = runtime.store._raise_runtime_counter_floor
        attempted_counters: list[str] = []

        def fail_after_execution_high_water(
            cursor: object,
            counter_name: str,
            floor: int,
        ) -> None:
            original_raise(cursor, counter_name, floor)
            attempted_counters.append(counter_name)
            if counter_name == counter_names[1]:
                raise RuntimeError("injected execution high-water failure")

        monkeypatch.setattr(
            runtime.store,
            "_raise_runtime_counter_floor",
            fail_after_execution_high_water,
        )
        with pytest.raises(RuntimeError, match="injected execution high-water failure"):
            runtime.store.claim_execution(pid, owner_id="test.worker")
        assert attempted_counters == list(counter_names[:2])

        monkeypatch.setattr(
            runtime.store,
            "_raise_runtime_counter_floor",
            original_raise,
        )
        runtime.store.insert_event(
            Event(
                event_id="event-unrelated-after-failed-claim",
                type=EventType.PROCESS_SIGNAL,
                source="test",
                target=pid,
                payload={"signal": "unrelated"},
                priority=EventPriority.NORMAL,
                created_at=utc_now(),
            )
        )

        assert runtime.process.get(pid) == before
        assert counter_values() == before_counters


@pytest.mark.parametrize("kind", STORE_BACKENDS)
@pytest.mark.parametrize("mismatch", ["generation", "owner_id", "lease_id"])
def test_worker_process_mutation_rejects_each_stale_token_field(
    kind: str,
    mismatch: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal=f"reject stale {mismatch}")
        token = runtime.store.claim_execution(pid, owner_id="test.worker")
        assert token is not None
        process = runtime.process.get(pid)
        replacements: dict[str, object] = {
            "generation": token.generation + 1,
            "owner_id": f"{token.owner_id}.stale",
            "lease_id": f"{token.lease_id}.stale",
        }
        stale = replace(token, **{mismatch: replacements[mismatch]})

        with bind_process_execution(stale):
            with pytest.raises(ProcessRevisionConflict, match="stale process execution token"):
                runtime.store.patch_process(
                    pid,
                    {"status_message": f"stale {mismatch}"},
                    expected_revision=process.revision,
                )

        with bind_process_execution(token):
            updated = runtime.store.patch_process(
                pid,
                {"status_message": "exact token accepted"},
                expected_revision=process.revision,
            )
        assert updated.status_message == "exact token accepted"
        assert runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE)


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_process_semantic_state_rejects_generic_store_bypasses_before_write(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="reject semantic process-state bypass")
        before = runtime.process.get(pid)
        semantic_patches = (
            {"status": ProcessStatus.EXITED},
            {"wait_state": ChildProcessWait(child_pid="pid_forged")},
            {"outcome": ExitedProcessOutcome(result_oid="obj_forged")},
            {"state_generation": before.state_generation + 1},
        )

        for patch in semantic_patches:
            with pytest.raises(
                ValidationError,
                match="semantic state must use ProcessTransitionService",
            ):
                runtime.store.patch_process(
                    pid,
                    patch,
                    expected_revision=before.revision,
                )
            unchanged = runtime.process.get(pid)
            assert unchanged.status == before.status
            assert unchanged.wait_state == before.wait_state
            assert unchanged.outcome == before.outcome
            assert unchanged.state_generation == before.state_generation
            assert unchanged.revision == before.revision

        with pytest.raises(
            ValidationError,
            match="semantic state must use ProcessTransitionService",
        ):
            runtime.store.patch_process_control(
                pid,
                {"status": ProcessStatus.KILLED},
                expected_revision=before.revision,
                allowed_statuses={before.status},
                reason="generic control patch must not bypass typed state",
            )

        invalid_update = replace(
            before,
            status=ProcessStatus.EXITED,
            outcome=None,
        )
        with pytest.raises(ValidationError, match="exited requires"):
            runtime.store.update_process(invalid_update)

        valid_but_bypassing_update = replace(
            before,
            status=ProcessStatus.EXITED,
            outcome=ExitedProcessOutcome(result_oid="obj_forged"),
            state_generation=before.state_generation + 1,
        )
        with pytest.raises(
            ValidationError,
            match="semantic state must use ProcessTransitionService",
        ):
            runtime.store.update_process(valid_but_bypassing_update)
        after_update_rejections = runtime.process.get(pid)
        assert after_update_rejections.status == before.status
        assert after_update_rejections.outcome == before.outcome
        assert after_update_rejections.state_generation == before.state_generation
        assert after_update_rejections.revision == before.revision

        invalid_insert = replace(
            before,
            pid=f"{pid}_forged",
            status=ProcessStatus.EXITED,
            outcome=None,
        )
        with pytest.raises(ValidationError, match="exited requires"):
            runtime.store.insert_process(invalid_insert)
        assert runtime.store.get_process(invalid_insert.pid) is None


@pytest.mark.parametrize("kind", STORE_BACKENDS)
@pytest.mark.parametrize(
    ("status", "wait_state"),
    [
        (ProcessStatus.RUNNABLE, None),
        (ProcessStatus.PAUSED, PausedProcessWait()),
        (
            ProcessStatus.WAITING_EVENT,
            ChildProcessWait(child_pid="pid_unresolved"),
        ),
        (
            ProcessStatus.WAITING_HUMAN,
            HumanProcessWait(request_ids=("hreq_unresolved",)),
        ),
        (
            ProcessStatus.WAITING_TOOL,
            ToolProcessWait(operation_id="op_unresolved"),
        ),
    ],
    ids=("runnable", "paused", "waiting-event", "waiting-human", "waiting-tool"),
)
def test_process_exec_epoch_commit_requires_admission_token_before_write(
    kind: str,
    status: ProcessStatus,
    wait_state: object,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal=f"reject tokenless exec commit from {status.value}")
        if status is not ProcessStatus.RUNNABLE:
            current = runtime.process.get(pid)
            runtime.process_transitions.transition(
                pid,
                status,
                expected_revision=current.revision,
                expected_status=current.status,
                expected_state_generation=current.state_generation,
                wait_state=wait_state,  # type: ignore[arg-type]
            )
        before = runtime.process.get(pid)
        raw_before, counters_before = _raw_process_row_and_counters(runtime, pid)

        with pytest.raises(
            ProcessRevisionConflict,
            match="requires its exact admission token",
        ):
            runtime.uow.processes.commit_process_exec_epoch(
                pid,
                publication_id="publication_missing",
                expected_revision=before.revision,
            )

        raw_after, counters_after = _raw_process_row_and_counters(runtime, pid)
        assert raw_after == raw_before
        assert counters_after == counters_before
        assert runtime.process.get(pid) == before


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_process_exec_epoch_commit_cas_requires_every_token_field(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="exact process exec commit token")
        before_admission = runtime.process.get(pid)
        token = runtime.uow.processes.claim_host_process_exec(
            pid,
            owner_id="test.exec.commit",
            expected_revision=before_admission.revision,
            expected_state_generation=before_admission.state_generation,
            expected_execution_generation=before_admission.execution_generation,
        )
        assert token is not None
        publication_id = "publication_exact_exec_commit"
        _insert_process_exec_commit_publication(
            runtime,
            publication_id=publication_id,
            pid=pid,
            token=token,
        )
        running = runtime.process.get(pid)
        raw_before, counters_before = _raw_process_row_and_counters(runtime, pid)
        replacements: dict[str, object] = {
            "generation": token.generation + 1,
            "owner_id": f"{token.owner_id}.stale",
            "lease_id": f"{token.lease_id}.stale",
        }

        for field_name, replacement in replacements.items():
            stale = replace(token, **{field_name: replacement})
            with bind_process_execution(stale):
                with pytest.raises(
                    ProcessRevisionConflict,
                    match="publication admission token conflict",
                ):
                    runtime.uow.processes.commit_process_exec_epoch(
                        pid,
                        publication_id=publication_id,
                        expected_revision=running.revision,
                    )
            raw_after, counters_after = _raw_process_row_and_counters(runtime, pid)
            assert raw_after == raw_before
            assert counters_after == counters_before

        with bind_process_execution(token):
            committed = runtime.uow.processes.commit_process_exec_epoch(
                pid,
                publication_id=publication_id,
                expected_revision=running.revision,
            )
        assert committed.status is ProcessStatus.RUNNABLE
        assert committed.wait_state is None
        assert committed.outcome is None
        assert committed.state_generation == running.state_generation + 1
        assert committed.execution_generation == token.generation
        assert committed.execution_owner_id is None
        assert committed.execution_lease_id is None


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_process_exec_epoch_commit_requires_exact_applying_publication(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="publication-bound process exec commit")
        other_pid = runtime.process.spawn(goal="foreign publication target")
        before_admission = runtime.process.get(pid)
        token = runtime.uow.processes.claim_host_process_exec(
            pid,
            owner_id="test.exec.commit",
            expected_revision=before_admission.revision,
            expected_state_generation=before_admission.state_generation,
            expected_execution_generation=before_admission.execution_generation,
        )
        assert token is not None
        running = runtime.process.get(pid)
        raw_before, counters_before = _raw_process_row_and_counters(runtime, pid)

        cases: list[tuple[str, dict[str, object] | None]] = [
            ("missing", None),
            ("wrong_kind", {"kind": "process_launch"}),
            ("wrong_pid", {"pid": other_pid}),
            ("wrong_state", {"state": "planning", "phase": "planned"}),
            ("wrong_phase", {"phase": "tools_configured"}),
            (
                "wrong_generation",
                {
                    "plan_overrides": {
                        "admission_execution_generation": token.generation + 1
                    }
                },
            ),
            (
                "wrong_owner",
                {
                    "plan_overrides": {
                        "admission_execution_owner_id": f"{token.owner_id}.forged"
                    }
                },
            ),
            (
                "wrong_lease",
                {
                    "plan_overrides": {
                        "admission_execution_lease_id": f"{token.lease_id}.forged"
                    }
                },
            ),
            (
                "invalid_plan",
                {"plan_overrides": {"admission_execution_generation": True}},
            ),
        ]
        for case_name, overrides in cases:
            publication_id = f"publication_exec_commit_{case_name}"
            if overrides is not None:
                selected = dict(overrides)
                publication_pid = str(selected.pop("pid", pid))
                _insert_process_exec_commit_publication(
                    runtime,
                    publication_id=publication_id,
                    pid=publication_pid,
                    token=token,
                    **selected,  # type: ignore[arg-type]
                )
            with bind_process_execution(token):
                with pytest.raises(ProcessRevisionConflict, match="publication"):
                    runtime.uow.processes.commit_process_exec_epoch(
                        pid,
                        publication_id=publication_id,
                        expected_revision=running.revision,
                    )
            raw_after, counters_after = _raw_process_row_and_counters(runtime, pid)
            assert raw_after == raw_before
            assert counters_after == counters_before
            assert runtime.process.get(pid) == running


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_generic_store_cannot_rewind_wait_generation_to_revive_stale_token(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="reject process wait generation rewind")
        first = runtime.process_transitions.transition(
            pid,
            ProcessStatus.WAITING_EVENT,
            expected_revision=runtime.process.get(pid).revision,
            wait_state=ChildProcessWait(child_pid="pid_child"),
        )
        stale = runtime.process_transitions.wait_token(first)
        runnable = runtime.process_transitions.wake(stale, control=False)
        second = runtime.process_transitions.transition(
            pid,
            ProcessStatus.WAITING_EVENT,
            expected_revision=runnable.revision,
            wait_state=ChildProcessWait(child_pid="pid_child"),
        )

        with pytest.raises(
            ValidationError,
            match="semantic state must use ProcessTransitionService",
        ):
            runtime.store.patch_process(
                pid,
                {"state_generation": stale.state_generation},
                expected_revision=second.revision,
            )
        unchanged = runtime.process.get(pid)
        assert unchanged.state_generation == second.state_generation
        assert unchanged.revision == second.revision
        with pytest.raises(ProcessRevisionConflict, match="stale process wait token"):
            runtime.process_transitions.wake(stale, control=False)


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_terminal_process_bookkeeping_requires_exact_declared_fence(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="exact terminal bookkeeping fence")
        token = runtime.store.claim_execution(pid, owner_id="test.worker")
        assert token is not None
        assert runtime.store.complete_execution(
            token,
            status=ProcessStatus.KILLED,
            outcome=KilledProcessOutcome(),
        )
        terminal = runtime.process.get(pid)

        with pytest.raises(ProcessRevisionConflict, match="terminal process"):
            runtime.store.patch_process(
                pid,
                {"status_message": "undeclared terminal write"},
                expected_revision=terminal.revision,
            )

        with bind_process_execution(token):
            with pytest.raises(ValueError, match="reason must be non-empty"):
                with trusted_terminal_process_mutation(
                    pid,
                    expected_revision=terminal.revision,
                    expected_generation=terminal.execution_generation,
                    allowed_statuses={ProcessStatus.KILLED},
                    execution_token=token,
                    reason="",
                ):
                    pass
            with trusted_terminal_process_mutation(
                pid,
                expected_revision=terminal.revision + 1,
                expected_generation=terminal.execution_generation,
                allowed_statuses={ProcessStatus.KILLED},
                execution_token=token,
                reason="reject a stale declared terminal revision",
            ):
                with pytest.raises(ProcessRevisionConflict, match="fence conflict"):
                    runtime.store.patch_process(
                        pid,
                        {"status_message": "wrong revision scope"},
                        expected_revision=terminal.revision,
                    )
            with trusted_terminal_process_mutation(
                pid,
                expected_revision=terminal.revision,
                expected_generation=terminal.execution_generation,
                allowed_statuses={ProcessStatus.EXITED},
                execution_token=token,
                reason="reject a wrong declared terminal status",
            ):
                with pytest.raises(ProcessRevisionConflict, match="fence conflict"):
                    runtime.store.patch_process(
                        pid,
                        {"status_message": "wrong status scope"},
                        expected_revision=terminal.revision,
                    )
            with trusted_terminal_process_mutation(
                pid,
                expected_revision=terminal.revision,
                expected_generation=terminal.execution_generation,
                allowed_statuses={ProcessStatus.KILLED},
                execution_token=token,
                reason="append exact terminal bookkeeping",
            ):
                updated = runtime.store.patch_process(
                    pid,
                    {"status_message": "exact terminal bookkeeping"},
                    expected_revision=terminal.revision,
                )

        assert updated.status == ProcessStatus.KILLED
        assert updated.status_message == "exact terminal bookkeeping"
        assert updated.revision == terminal.revision + 1
        assert updated.execution_generation == terminal.execution_generation + 1
        assert updated.execution_owner_id is None
        assert updated.execution_lease_id is None


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_successful_exec_fences_active_worker_and_returns_process_to_queue(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="before exec")
        token = runtime.store.claim_execution(pid, owner_id="test.worker")
        assert token is not None

        with bind_process_execution(token):
            executed = runtime.exec_process(
                pid,
                "base-agent:v0",
                goal="after successful exec",
            )

        assert executed.status == ProcessStatus.RUNNABLE
        assert executed.execution_generation > token.generation
        assert executed.execution_owner_id is None
        assert executed.execution_lease_id is None
        assert runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE) is False
        assert runtime.store.release_execution(token) is False

        with bind_process_execution(token):
            with pytest.raises(ProcessRevisionConflict, match="stale process execution token"):
                runtime.store.patch_process(
                    pid,
                    {"status_message": "old image write"},
                    expected_revision=executed.revision,
                )

        next_token = runtime.store.claim_execution(pid, owner_id="test.next-worker")
        assert next_token is not None
        assert next_token.generation > executed.execution_generation
        assert runtime.store.complete_execution(next_token, status=ProcessStatus.RUNNABLE)


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_exec_tool_persists_one_result_handle_after_fencing_its_worker(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="exec tool result")
        token = runtime.store.claim_execution(pid, owner_id="test.exec.worker")
        assert token is not None

        with bind_process_execution(token):
            result = runtime.tools.call(
                pid,
                "exec_process",
                {"image": "base-agent:v0", "goal": "post exec result"},
            )

        assert result.ok, result.error
        assert result.result_handle is not None
        process = runtime.process.get(pid)
        publication, committed = _committed_exec_publication(runtime, pid)
        capability = runtime.store.get_capability(result.result_handle.capability_id)
        obj = runtime.store.get_object(result.result_handle.oid)
        assert process.status == ProcessStatus.RUNNABLE
        assert process.execution_generation == token.generation + 1
        assert process.execution_owner_id is None
        assert process.execution_lease_id is None
        assert process.revision == int(committed["revision"]) + 1
        assert result.result_handle.capability_id in process.capabilities
        assert capability is not None and capability.metadata.get("object_handle") is True
        assert obj is not None and obj.type == ObjectType.TOOL_RESULT
        assert obj.provenance.created_from_action == "tool.exec_process"
        assert result.result_handle.oid not in {root.oid for root in process.memory_view.roots}

        with bind_process_execution(token):
            with trusted_post_exec_completion_mutation(
                pid,
                publication_id=publication["publication_id"],
                operation_id=publication["plan"]["operation_id"],
                expected_revision=process.revision,
                expected_generation=process.execution_generation,
                execution_token=token,
                reason="reject duplicate exec completion",
            ):
                with pytest.raises(ProcessRevisionConflict, match="completion fence"):
                    runtime.store.patch_process(
                        pid,
                        {"status_message": "duplicate completion"},
                        expected_revision=process.revision,
                    )
            with pytest.raises(ProcessRevisionConflict, match="stale process execution token"):
                runtime.store.patch_process(
                    pid,
                    {"status_message": "ordinary old-token write"},
                    expected_revision=process.revision,
                )


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_post_exec_completion_rejects_forgery_and_arbitrary_process_patch(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="fence forgery")
        token = runtime.store.claim_execution(pid, owner_id="test.exec.worker")
        assert token is not None
        with bind_process_execution(token):
            runtime.exec_process(pid, "base-agent:v0", goal="committed exec")

        process = runtime.process.get(pid)
        publication, committed = _committed_exec_publication(runtime, pid)
        assert process.revision == int(committed["revision"])

        with bind_process_execution(token):
            with trusted_post_exec_completion_mutation(
                pid,
                publication_id=publication["publication_id"],
                operation_id=publication["plan"]["operation_id"],
                expected_revision=process.revision,
                expected_generation=process.execution_generation,
                execution_token=token,
                reason="reject arbitrary post-exec patch",
            ):
                with pytest.raises(ProcessRevisionConflict, match="only append a ToolResult"):
                    runtime.store.patch_process(
                        pid,
                        {"status_message": "must not commit"},
                        expected_revision=process.revision,
                    )
            with trusted_post_exec_completion_mutation(
                pid,
                publication_id=f"{publication['publication_id']}.forged",
                operation_id=publication["plan"]["operation_id"],
                expected_revision=process.revision,
                expected_generation=process.execution_generation,
                execution_token=token,
                reason="reject forged publication",
            ):
                with pytest.raises(ProcessRevisionConflict, match="publication fence"):
                    runtime.store.patch_process(
                        pid,
                        {"status_message": "forged publication"},
                        expected_revision=process.revision,
                    )

        forged = replace(token, owner_id=f"{token.owner_id}.forged")
        with bind_process_execution(forged):
            with trusted_post_exec_completion_mutation(
                pid,
                publication_id=publication["publication_id"],
                operation_id=publication["plan"]["operation_id"],
                expected_revision=process.revision,
                expected_generation=process.execution_generation,
                execution_token=forged,
                reason="reject forged worker token",
            ):
                with pytest.raises(ProcessRevisionConflict, match="completion fence"):
                    runtime.store.patch_process(
                        pid,
                        {"status_message": "forged token"},
                        expected_revision=process.revision,
                    )
        unchanged = runtime.process.get(pid)
        assert unchanged.revision == process.revision
        assert unchanged.status_message is None


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_host_exec_capture_race_preserves_the_winning_worker_lease(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Host exec that loses admission cannot replay its RUNNABLE snapshot."""

    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="exec admission race")
        original_capture = runtime.process_exec_state.capture
        winning_tokens = []
        before_publication_ids = {
            str(item["publication_id"])
            for item in runtime.store.list_runtime_publications(pid=pid)
        }

        def claim_after_snapshot(selected_pid: str) -> object:
            state = original_capture(selected_pid)
            token = runtime.store.claim_execution(
                selected_pid,
                owner_id="concurrent.worker",
            )
            assert token is not None
            winning_tokens.append(token)
            return state

        monkeypatch.setattr(
            runtime.process_exec_state,
            "capture",
            claim_after_snapshot,
        )

        with pytest.raises(ProcessRevisionConflict, match="exec admission conflict"):
            runtime.exec_process(pid, "base-agent:v0", goal="must not publish")

        assert len(winning_tokens) == 1
        winner = winning_tokens[0]
        claimed = runtime.process.get(pid)
        assert claimed.status == ProcessStatus.RUNNING
        assert claimed.execution_generation == winner.generation
        assert claimed.execution_owner_id == winner.owner_id
        assert claimed.execution_lease_id == winner.lease_id
        assert not [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
            and item["publication_id"] not in before_publication_ids
        ]
        assert runtime.store.complete_execution(
            winner,
            status=ProcessStatus.RUNNABLE,
        )


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_worker_exec_admission_cas_loses_to_concurrent_completion_without_replay(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion that wins the exact worker CAS cannot be undone by exec."""

    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="worker exec race")
        token = runtime.store.claim_execution(pid, owner_id="concurrent.worker")
        assert token is not None
        original_claim = runtime.store.claim_worker_process_exec
        completion_results: list[bool] = []
        before_publication_ids = {
            str(item["publication_id"])
            for item in runtime.store.list_runtime_publications(pid=pid)
        }

        def complete_before_admission(*args: object, **kwargs: object) -> object:
            completion_results.append(
                runtime.store.complete_execution(
                    token,
                    status=ProcessStatus.RUNNABLE,
                )
            )
            return original_claim(*args, **kwargs)

        monkeypatch.setattr(
            runtime.store,
            "claim_worker_process_exec",
            complete_before_admission,
        )
        with bind_process_execution(token):
            with pytest.raises(ProcessRevisionConflict, match="exec admission conflict"):
                runtime.exec_process(pid, "base-agent:v0", goal="must not replay")

        assert completion_results == [True]
        completed = runtime.process.get(pid)
        assert completed.status == ProcessStatus.RUNNABLE
        assert completed.execution_generation >= token.generation
        assert completed.execution_owner_id is None
        assert completed.execution_lease_id is None
        assert runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE) is False
        assert not [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
            and item["publication_id"] not in before_publication_ids
        ]
        next_token = runtime.store.claim_execution(pid, owner_id="next.worker")
        assert next_token is not None
        assert next_token.generation > completed.execution_generation
        assert runtime.store.complete_execution(next_token, status=ProcessStatus.RUNNABLE)


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_failed_host_exec_fences_its_internal_admission_lease(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host rollback restores RUNNABLE state without reviving its exec token."""

    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="before Host exec")
        before = runtime.process.get(pid)
        original_claim = runtime.store.claim_host_process_exec
        admission_tokens = []

        def record_claim(*args: object, **kwargs: object) -> object:
            token = original_claim(*args, **kwargs)
            assert token is not None
            admission_tokens.append(token)
            return token

        def fail_skills(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected failed Host exec")

        monkeypatch.setattr(
            runtime.store,
            "claim_host_process_exec",
            record_claim,
        )
        monkeypatch.setattr(runtime.image_boot, "_configure_skills", fail_skills)

        with pytest.raises(RuntimeError, match="injected failed Host exec"):
            runtime.exec_process(pid, "base-agent:v0", goal="must roll back")

        assert len(admission_tokens) == 1
        admission = admission_tokens[0]
        restored = runtime.process.get(pid)
        assert restored.status == ProcessStatus.RUNNABLE
        assert restored.execution_generation > admission.generation
        assert restored.execution_generation > before.execution_generation
        assert restored.execution_owner_id is None
        assert restored.execution_lease_id is None
        assert runtime.store.complete_execution(
            admission,
            status=ProcessStatus.RUNNABLE,
        ) is False
        next_token = runtime.store.claim_execution(pid, owner_id="next.worker")
        assert next_token is not None
        assert next_token.generation > restored.execution_generation
        assert runtime.store.complete_execution(
            next_token,
            status=ProcessStatus.RUNNABLE,
        )


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_active_exec_admission_rejects_tokenless_host_patch_before_write(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="exec patch fence")

        def block_skills(*_args: object, **_kwargs: object) -> None:
            entered.set()
            assert release.wait(timeout=_THREAD_SYNC_TIMEOUT_S)

        def run_exec() -> None:
            try:
                runtime.exec_process(pid, "base-agent:v0", goal="committed replacement")
            except BaseException as error:
                errors.append(error)

        monkeypatch.setattr(runtime.image_boot, "_configure_skills", block_skills)
        worker = threading.Thread(target=run_exec)
        worker.start()
        assert entered.wait(timeout=_THREAD_SYNC_TIMEOUT_S)
        before_row, before_counters = _raw_process_row_and_counters(runtime, pid)

        with pytest.raises(
            ProcessRevisionConflict,
            match="process exec admission rejects a non-owner process write",
        ):
            runtime.process.set_working_directory(pid, "host-winner")

        after_row, after_counters = _raw_process_row_and_counters(runtime, pid)
        assert after_row == before_row
        assert after_counters == before_counters

        release.set()
        worker.join(timeout=_THREAD_SYNC_TIMEOUT_S)
        assert not worker.is_alive()
        assert errors == []
        committed = runtime.process.get(pid)
        assert committed.status == ProcessStatus.RUNNABLE
        assert committed.working_directory != "host-winner"
        publication = [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
        ][-1]
        assert publication["state"] == "committed"
        assert publication["phase"] == "committed"
        assert runtime.lifecycle.state == "open"


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_ordinary_running_process_allows_tokenless_host_patch_without_exec_publication(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="ordinary scheduler compatibility")
        token = runtime.store.claim_execution(pid, owner_id="ordinary.worker")
        assert token is not None

        updated = runtime.process.set_working_directory(pid, "ordinary-host-patch")

        assert updated.status == ProcessStatus.RUNNING
        assert updated.execution_generation == token.generation
        assert updated.execution_owner_id == token.owner_id
        assert updated.execution_lease_id == token.lease_id
        assert updated.working_directory == "ordinary-host-patch"
        assert runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE)


def test_takeover_helpers_allow_unleased_running_but_reject_partial_lease() -> None:
    unleased = SimpleNamespace(
        pid="pid_unleased_running",
        status=ProcessStatus.RUNNING,
        execution_owner_id=None,
        execution_lease_id=None,
    )
    with ProcessManager._signal_takeover_scope(
        unleased,
        ProcessSignal.CANCEL,
        reason_text=None,
        reason_action=None,
        require_host_resume=False,
    ):
        pass
    with ObjectTaskStateService._runner_takeover_scope(unleased):
        pass
    with ResourceManager._resource_limit_takeover_scope(unleased):
        pass

    for partial in (
        SimpleNamespace(
            pid="pid_owner_only",
            status=ProcessStatus.RUNNING,
            execution_owner_id="worker",
            execution_lease_id=None,
        ),
        SimpleNamespace(
            pid="pid_lease_only",
            status=ProcessStatus.RUNNING,
            execution_owner_id=None,
            execution_lease_id="lease",
        ),
    ):
        with pytest.raises(ProcessError, match="incomplete execution lease"):
            ProcessManager._signal_takeover_scope(
                partial,
                ProcessSignal.CANCEL,
                reason_text=None,
                reason_action=None,
                require_host_resume=False,
            )
        with pytest.raises(RuntimeError, match="incomplete execution lease"):
            ObjectTaskStateService._runner_takeover_scope(partial)
        with pytest.raises(ValidationError, match="incomplete execution lease"):
            ResourceManager._resource_limit_takeover_scope(partial)


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_process_execution_takeover_rejects_another_pid_without_writing(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        target = runtime.process.spawn(goal="exact takeover target")
        other = runtime.process.spawn(goal="must remain unchanged")
        token = runtime.store.claim_execution(target, owner_id="takeover.target")
        assert token is not None
        source = runtime.process.get(target)
        before_row, before_counters = _raw_process_row_and_counters(runtime, other)

        with pytest.raises(
            ProcessRevisionConflict,
            match="takeover cannot mutate another process",
        ):
            with trusted_process_execution_takeover(
                target,
                source_revision=source.revision,
                source_state_generation=source.state_generation,
                source_execution_token=token,
                intended_status=ProcessStatus.PAUSED,
                reason="test exact takeover target",
                nonce=f"test-{uuid4().hex}",
                wait_kind="paused",
            ):
                runtime.process.set_working_directory(other, "forbidden")

        after_row, after_counters = _raw_process_row_and_counters(runtime, other)
        assert after_row == before_row
        assert after_counters == before_counters
        assert runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE)


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_incomplete_process_execution_takeover_rolls_back_all_preparation(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="rollback incomplete takeover")
        token = runtime.store.claim_execution(pid, owner_id="takeover.preparation")
        assert token is not None
        source = runtime.process.get(pid)
        before_row, before_counters = _raw_process_row_and_counters(runtime, pid)
        before_counts = {
            table: int(runtime.store._query(f"SELECT COUNT(*) AS count FROM {table}")[0]["count"])
            for table in ("objects", "capabilities", "events", "audit_records")
        }

        with pytest.raises(
            RuntimeError,
            match="takeover did not commit its final state",
        ):
            with runtime.store.transaction(include_object_payloads=True):
                with trusted_process_execution_takeover(
                    pid,
                    source_revision=source.revision,
                    source_state_generation=source.state_generation,
                    source_execution_token=token,
                    intended_status=ProcessStatus.PAUSED,
                    reason="test incomplete takeover",
                    nonce=f"test-{uuid4().hex}",
                    reason_text="prepared but not committed",
                    reason_action="process.signal.reason",
                    wait_kind="paused",
                ):
                    runtime.process._create_flow_text_carrier(
                        recipient_pid=pid,
                        text="prepared but not committed",
                        payload_field="reason",
                        object_type=ObjectType.MESSAGE,
                        title="Process signal reason",
                        tags=["process_signal", "reason"],
                        created_from_action="process.signal.reason",
                        source_pid=pid,
                    )

        after_row, after_counters = _raw_process_row_and_counters(runtime, pid)
        after_counts = {
            table: int(runtime.store._query(f"SELECT COUNT(*) AS count FROM {table}")[0]["count"])
            for table in ("objects", "capabilities", "events", "audit_records")
        }
        assert after_row == before_row
        assert after_counters == before_counters
        assert after_counts == before_counts
        assert runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE)


@pytest.mark.parametrize("kind", STORE_BACKENDS)
@pytest.mark.parametrize("control", ["pause", "cancel"])
def test_exec_rollback_cas_preserves_trusted_control_winner_and_fences_runtime(
    kind: str,
    control: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    primary = RuntimeError(f"injected late exec failure after {control}")
    errors: list[BaseException] = []
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="control wins exec race")

        def fail_after_control(*_args: object, **_kwargs: object) -> None:
            entered.set()
            assert release.wait(timeout=_THREAD_SYNC_TIMEOUT_S)
            raise primary

        def run_exec() -> None:
            try:
                runtime.exec_process(pid, "base-agent:v0", goal="must not overwrite control")
            except BaseException as error:
                errors.append(error)

        monkeypatch.setattr(runtime.image_boot, "_configure_skills", fail_after_control)
        worker = threading.Thread(target=run_exec)
        worker.start()
        assert entered.wait(timeout=_THREAD_SYNC_TIMEOUT_S)
        admitted = runtime.process.get(pid)
        assert admitted.status == ProcessStatus.RUNNING
        assert admitted.execution_owner_id is not None
        assert admitted.execution_lease_id is not None

        reason = f"trusted {control} wins"
        if control == "pause":
            runtime.process.pause(pid, reason)
        else:
            runtime.process.cancel(pid, reason)
        winner = runtime.process.get(pid)
        if control == "pause":
            assert winner.status == ProcessStatus.PAUSED
            assert isinstance(winner.wait_state, PausedProcessWait)
            assert winner.outcome is None
            reason_oid = winner.wait_state.reason_oid
        else:
            assert winner.status == ProcessStatus.KILLED
            assert winner.wait_state is None
            assert isinstance(winner.outcome, KilledProcessOutcome)
            reason_oid = winner.outcome.reason_oid
        assert winner.execution_generation > admitted.execution_generation
        assert winner.state_generation > admitted.state_generation
        assert winner.execution_owner_id is None
        assert winner.execution_lease_id is None
        assert reason_oid is not None
        assert runtime.store.get_object(reason_oid) is not None

        release.set()
        worker.join(timeout=_THREAD_SYNC_TIMEOUT_S)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeRecoveryRequired)

        persisted = runtime.process.get(pid)
        assert persisted == winner
        assert runtime.store.get_object(reason_oid) is not None
        publication = runtime.store.get_runtime_publication(
            errors[0].publication_id,
        )
        assert publication is not None
        operation = runtime.store.get_operation(publication["plan"]["operation_id"])
        assert publication["state"] == "failed"
        assert publication["phase"] == "compensation_failed"
        assert not any(
            phase.get("phase") == "compensation_applied"
            for phase in publication["receipt"].get("phases", [])
        )
        assert operation is not None
        assert operation.state == OperationState.TERMINAL
        assert operation.outcome == OperationOutcome.UNKNOWN
        assert runtime.lifecycle.state == "close_failed"


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_exec_rollback_preserves_resource_limit_takeover_winner(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="resource limit wins")

        def fail_after_limit(*_args: object, **_kwargs: object) -> None:
            entered.set()
            assert release.wait(timeout=_THREAD_SYNC_TIMEOUT_S)
            raise RuntimeError("injected late exec failure after resource limit")

        def run_exec() -> None:
            try:
                runtime.exec_process(pid, "base-agent:v0", goal="must preserve limit kill")
            except BaseException as error:
                errors.append(error)

        monkeypatch.setattr(runtime.image_boot, "_configure_skills", fail_after_limit)
        worker = threading.Thread(target=run_exec)
        worker.start()
        assert entered.wait(timeout=_THREAD_SYNC_TIMEOUT_S)

        runtime.resources.kill_if_exceeded(pid, reason="resource ceiling reached")
        winner = runtime.process.get(pid)
        assert winner.status == ProcessStatus.KILLED
        assert isinstance(winner.outcome, KilledProcessOutcome)
        assert winner.outcome.code == "resource_limit_exceeded"

        release.set()
        worker.join(timeout=_THREAD_SYNC_TIMEOUT_S)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeRecoveryRequired)
        assert runtime.process.get(pid) == winner
        publication = runtime.store.get_runtime_publication(errors[0].publication_id)
        assert publication is not None
        operation = runtime.store.get_operation(publication["plan"]["operation_id"])
        assert publication["state"] == "failed"
        assert publication["phase"] == "compensation_failed"
        assert operation is not None
        assert operation.outcome == OperationOutcome.UNKNOWN
        assert runtime.lifecycle.state == "close_failed"


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_exec_recovery_requires_startup_lease_before_reading_live_publication(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="online recovery fence")

        def block_skills(*_args: object, **_kwargs: object) -> None:
            entered.set()
            assert release.wait(timeout=_THREAD_SYNC_TIMEOUT_S)

        def run_exec() -> None:
            try:
                runtime.exec_process(pid, "base-agent:v0", goal="commit after gate check")
            except BaseException as error:
                errors.append(error)

        monkeypatch.setattr(runtime.image_boot, "_configure_skills", block_skills)
        worker = threading.Thread(target=run_exec)
        worker.start()
        assert entered.wait(timeout=_THREAD_SYNC_TIMEOUT_S)
        publication = [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
        ][-1]
        before = runtime.store.get_runtime_publication(publication["publication_id"])
        query_calls: list[object] = []
        original_query = runtime.image_boot._publications.query_runtime_publication_recovery

        def record_query(*args: object, **kwargs: object) -> object:
            query_calls.append((args, kwargs))
            return original_query(*args, **kwargs)

        monkeypatch.setattr(
            runtime.image_boot._publications,
            "query_runtime_publication_recovery",
            record_query,
        )
        with pytest.raises(
            RuntimeError,
            match="runtime recovery requires the active startup recovery lease",
        ):
            runtime.image_boot.recover_incomplete_publications()
        assert query_calls == []
        assert runtime.store.get_runtime_publication(publication["publication_id"]) == before

        release.set()
        worker.join(timeout=_THREAD_SYNC_TIMEOUT_S)
        assert not worker.is_alive()
        assert errors == []
        committed = runtime.store.get_runtime_publication(publication["publication_id"])
        assert committed is not None
        assert committed["state"] == "committed"
        assert committed["phase"] == "committed"
        assert runtime.lifecycle.state == "open"


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_host_exec_post_admission_base_exception_compensates_before_propagating(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The admission commit/return gap is inside the durable outcome boundary."""

    interruption = KeyboardInterrupt("injected after exec admission commit")
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="admission ack gap")
        original_transaction = runtime.store.transaction
        injected = False

        @contextlib.contextmanager
        def interrupt_after_admission_commit(
            *args: object,
            **kwargs: object,
        ) -> Iterator[object]:
            nonlocal injected
            with original_transaction(*args, **kwargs) as cursor:
                yield cursor
            publications = [
                item
                for item in runtime.store.list_runtime_publications(pid=pid)
                if item["kind"] == "process_exec"
            ]
            process = runtime.store.get_process(pid)
            if (
                not injected
                and runtime.store._transaction_depth == 0
                and publications
                and publications[-1]["state"] == "planning"
                and process is not None
                and process.status == ProcessStatus.RUNNING
                and str(process.execution_owner_id or "").endswith(":process.exec")
            ):
                injected = True
                raise interruption

        monkeypatch.setattr(
            runtime.store,
            "transaction",
            interrupt_after_admission_commit,
        )
        with pytest.raises(KeyboardInterrupt) as caught:
            runtime.exec_process(pid, "base-agent:v0", goal="must compensate")
        assert caught.value is interruption
        assert injected is True

        restored = runtime.process.get(pid)
        assert restored.status == ProcessStatus.RUNNABLE
        assert restored.execution_owner_id is None
        assert restored.execution_lease_id is None
        publication = [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
        ][-1]
        operation = runtime.store.get_operation(publication["plan"]["operation_id"])
        assert publication["state"] == "rolled_back"
        assert publication["phase"] == "compensated"
        assert publication["operation_reconciled"] is True
        assert operation is not None
        assert operation.state == OperationState.TERMINAL
        assert operation.outcome == OperationOutcome.FAILED
        assert runtime.lifecycle.state == "open"
        next_token = runtime.store.claim_execution(pid, owner_id="after.admission.ack")
        assert next_token is not None
        assert runtime.store.complete_execution(next_token, status=ProcessStatus.RUNNABLE)


@pytest.mark.parametrize("kind", STORE_BACKENDS)
@pytest.mark.parametrize(
    "interruption",
    [
        pytest.param(RuntimeError("after terminal exec commit"), id="exception"),
        pytest.param(KeyboardInterrupt("after terminal exec commit"), id="base-exception"),
    ],
)
def test_exec_terminal_commit_confirmation_preserves_durable_truth_and_primary(
    kind: str,
    interruption: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed exec is never compensated because its commit ack was lost."""

    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="commit ack gap")
        original_commit = runtime.image_boot._commit_exec_publication

        def interrupt_after_commit(*args: object, **kwargs: object) -> None:
            original_commit(*args, **kwargs)
            raise interruption

        monkeypatch.setattr(
            runtime.image_boot,
            "_commit_exec_publication",
            interrupt_after_commit,
        )
        with pytest.raises(type(interruption)) as caught:
            runtime.exec_process(pid, "base-agent:v0", goal="durably committed")
        assert caught.value is interruption

        committed = runtime.process.get(pid)
        publication = [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
        ][-1]
        operation = runtime.store.get_operation(publication["plan"]["operation_id"])
        assert committed.goal_oid is not None
        assert committed.status == ProcessStatus.RUNNABLE
        assert committed.execution_owner_id is None
        assert committed.execution_lease_id is None
        assert publication["state"] == "committed"
        assert publication["phase"] == "committed"
        assert publication["operation_reconciled"] is True
        assert operation is not None
        assert operation.state == OperationState.TERMINAL
        assert operation.outcome == OperationOutcome.SUCCEEDED
        assert runtime.lifecycle.state == "open"


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_exec_terminal_commit_confirmation_accepts_monotonic_successor_claim(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A next worker may advance the row before the commit ack is diagnosed."""

    interruption = RuntimeError("after terminal exec commit and successor claim")
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="commit successor")
        original_commit = runtime.image_boot._commit_exec_publication
        successors = []

        def claim_after_commit(*args: object, **kwargs: object) -> None:
            original_commit(*args, **kwargs)
            successor = runtime.store.claim_execution(pid, owner_id="next.worker")
            assert successor is not None
            successors.append(successor)
            raise interruption

        monkeypatch.setattr(
            runtime.image_boot,
            "_commit_exec_publication",
            claim_after_commit,
        )
        with pytest.raises(RuntimeError) as caught:
            runtime.exec_process(pid, "base-agent:v0", goal="durably committed")
        assert caught.value is interruption
        assert len(successors) == 1

        successor = successors[0]
        claimed = runtime.process.get(pid)
        publication = [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
        ][-1]
        assert claimed.status == ProcessStatus.RUNNING
        assert claimed.execution_generation == successor.generation
        assert claimed.execution_owner_id == successor.owner_id
        assert claimed.execution_lease_id == successor.lease_id
        assert publication["state"] == "committed"
        assert publication["operation_reconciled"] is True
        assert runtime.lifecycle.state == "open"
        assert runtime.store.complete_execution(successor, status=ProcessStatus.RUNNABLE)


@pytest.mark.parametrize("kind", STORE_BACKENDS)
@pytest.mark.parametrize(
    "secondary",
    [
        pytest.param(RuntimeError("after rollback terminal commit"), id="exception"),
        pytest.param(
            KeyboardInterrupt("after rollback terminal commit"),
            id="base-exception",
        ),
    ],
)
def test_exec_rollback_commit_confirmation_preserves_original_primary(
    kind: str,
    secondary: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost rollback ack cannot replace the error that caused compensation."""

    primary = ValueError("injected exec application failure")
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="rollback ack gap")
        original_rollback = runtime.image_boot._rollback_failed_exec

        def fail_skills(*_args: object, **_kwargs: object) -> None:
            raise primary

        def interrupt_after_rollback(*args: object, **kwargs: object) -> None:
            original_rollback(*args, **kwargs)
            raise secondary

        monkeypatch.setattr(runtime.image_boot, "_configure_skills", fail_skills)
        monkeypatch.setattr(
            runtime.image_boot,
            "_rollback_failed_exec",
            interrupt_after_rollback,
        )
        with pytest.raises(ValueError) as caught:
            runtime.exec_process(pid, "base-agent:v0", goal="must roll back")
        assert caught.value is primary

        restored = runtime.process.get(pid)
        publication = [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
        ][-1]
        operation = runtime.store.get_operation(publication["plan"]["operation_id"])
        assert restored.status == ProcessStatus.RUNNABLE
        assert restored.execution_owner_id is None
        assert restored.execution_lease_id is None
        assert publication["state"] == "rolled_back"
        assert publication["phase"] == "compensated"
        assert publication["operation_reconciled"] is True
        assert operation is not None
        assert operation.state == OperationState.TERMINAL
        assert operation.outcome == OperationOutcome.FAILED
        assert runtime.lifecycle.state == "open"


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_exec_rollback_commit_confirmation_accepts_monotonic_successor_claim(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback ack failure cannot erase a worker claimed after compensation."""

    primary = ValueError("injected exec application failure")
    secondary = RuntimeError("after rollback terminal commit and successor claim")
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="rollback successor")
        original_rollback = runtime.image_boot._rollback_failed_exec
        successors = []

        def fail_skills(*_args: object, **_kwargs: object) -> None:
            raise primary

        def claim_after_rollback(*args: object, **kwargs: object) -> None:
            original_rollback(*args, **kwargs)
            successor = runtime.store.claim_execution(pid, owner_id="next.worker")
            assert successor is not None
            successors.append(successor)
            raise secondary

        monkeypatch.setattr(runtime.image_boot, "_configure_skills", fail_skills)
        monkeypatch.setattr(
            runtime.image_boot,
            "_rollback_failed_exec",
            claim_after_rollback,
        )
        with pytest.raises(ValueError) as caught:
            runtime.exec_process(pid, "base-agent:v0", goal="must roll back")
        assert caught.value is primary
        assert len(successors) == 1

        successor = successors[0]
        claimed = runtime.process.get(pid)
        publication = [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
        ][-1]
        assert claimed.status == ProcessStatus.RUNNING
        assert claimed.execution_generation == successor.generation
        assert claimed.execution_owner_id == successor.owner_id
        assert claimed.execution_lease_id == successor.lease_id
        assert publication["state"] == "rolled_back"
        assert publication["operation_reconciled"] is True
        assert runtime.lifecycle.state == "open"
        assert runtime.store.complete_execution(successor, status=ProcessStatus.RUNNABLE)


@pytest.mark.parametrize(
    "interruption",
    [
        pytest.param(
            KeyboardInterrupt("injected Host exec interruption"),
            id="keyboard-interrupt",
        ),
        pytest.param(
            asyncio.CancelledError("injected Host exec cancellation"),
            id="cancelled-error",
        ),
    ],
)
def test_host_exec_base_exception_restores_runnable_and_preserves_control_flow(
    interruption: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="interrupt Host exec")

        def interrupt_skills(*_args: object, **_kwargs: object) -> None:
            raise interruption

        monkeypatch.setattr(
            runtime.image_boot,
            "_configure_skills",
            interrupt_skills,
        )

        with pytest.raises(type(interruption)) as caught:
            runtime.exec_process(pid, "base-agent:v0", goal="must roll back")
        assert caught.value is interruption

        restored = runtime.process.get(pid)
        assert restored.status == ProcessStatus.RUNNABLE
        assert restored.execution_owner_id is None
        assert restored.execution_lease_id is None
        publication = [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
        ][-1]
        assert publication["state"] == "rolled_back"
        assert publication["phase"] == "compensated"
        assert runtime.lifecycle.state == "open"
        token = runtime.store.claim_execution(pid, owner_id="after.interruption")
        assert token is not None
        assert runtime.store.complete_execution(token, status=ProcessStatus.RUNNABLE)
    finally:
        runtime.close()


def test_host_exec_cancelled_terminalization_groups_pending_and_fences_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    interruption = asyncio.CancelledError("injected Host exec cancellation")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="pending cancellation")
        original_update = runtime.store.update_operation

        def cancel_skills(*_args: object, **_kwargs: object) -> None:
            raise interruption

        def fail_rolled_back_operation(
            record: object,
            *,
            expected_states: object = None,
        ) -> bool:
            metadata = getattr(record, "metadata", {})
            if metadata.get("runtime_publication_state") == "rolled_back":
                raise RuntimeError("injected exec terminalization failure")
            return original_update(record, expected_states=expected_states)

        monkeypatch.setattr(runtime.image_boot, "_configure_skills", cancel_skills)
        monkeypatch.setattr(
            runtime.store,
            "update_operation",
            fail_rolled_back_operation,
        )

        with pytest.raises(BaseExceptionGroup) as caught:
            runtime.exec_process(pid, "base-agent:v0", goal="must remain pending")
        pending: list[BaseException] = [caught.value]
        leaves: list[BaseException] = []
        while pending:
            item = pending.pop()
            if isinstance(item, BaseExceptionGroup):
                pending.extend(item.exceptions)
            else:
                leaves.append(item)
        assert any(item is interruption for item in leaves)
        signals = [
            item for item in leaves if isinstance(item, RuntimePublicationPending)
        ]
        assert len(signals) == 1
        publication = runtime.store.get_runtime_publication(
            signals[0].publication_id
        )
        operation = runtime.store.get_operation(signals[0].operation_id)
        assert publication is not None
        assert publication["state"] == "rollback_pending"
        assert publication["phase"] == "compensation_applied"
        assert operation is not None
        assert operation.state == OperationState.RUNNING
        assert operation.outcome == OperationOutcome.PENDING
        assert runtime.lifecycle.state == "close_failed"
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_host_exec_compensation_base_exception_preserves_both_and_fences_runtime(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = RuntimeError("injected late Host exec failure")
    cleanup = KeyboardInterrupt("injected Host exec compensation interruption")
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="fence failed rollback")

        def fail_skills(*_args: object, **_kwargs: object) -> None:
            raise primary

        def interrupt_restore(*_args: object, **_kwargs: object) -> None:
            raise cleanup

        monkeypatch.setattr(runtime.image_boot, "_configure_skills", fail_skills)
        monkeypatch.setattr(
            runtime.process_exec_state,
            "restore",
            interrupt_restore,
        )

        with pytest.raises(BaseExceptionGroup) as caught:
            runtime.exec_process(pid, "base-agent:v0", goal="must fence")
        pending: list[BaseException] = [caught.value]
        leaves: list[BaseException] = []
        while pending:
            item = pending.pop()
            if isinstance(item, BaseExceptionGroup):
                pending.extend(item.exceptions)
            else:
                leaves.append(item)
        assert any(item is primary for item in leaves)
        assert any(item is cleanup for item in leaves)
        assert any(isinstance(item, RuntimeRecoveryRequired) for item in leaves)
        assert runtime.lifecycle.state == "close_failed"
        publication = [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
        ][-1]
        assert publication["state"] == "failed"
        assert publication["phase"] == "compensation_failed"
        with pytest.raises(RuntimeError, match="state=close_failed"):
            runtime.process.spawn(goal="blocked after uncertain exec rollback")


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_failed_exec_fences_superseded_worker_token(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="before failed exec")
        token = runtime.store.claim_execution(pid, owner_id="test.worker")
        assert token is not None
        stale_completion_results: list[bool] = []

        def fail_skills(*_args: object, **_kwargs: object) -> None:
            stale_completion_results.append(
                runtime.store.complete_execution(
                    token,
                    status=ProcessStatus.RUNNABLE,
                )
            )
            raise RuntimeError("injected late exec failure")

        monkeypatch.setattr(runtime.image_boot, "_configure_skills", fail_skills)
        with bind_process_execution(token):
            with pytest.raises(RuntimeError, match="injected late exec failure"):
                runtime.exec_process(
                    pid,
                    "base-agent:v0",
                    goal="must roll back",
                )

        assert stale_completion_results == [False]
        restored = runtime.process.get(pid)
        assert restored.status == ProcessStatus.RUNNABLE
        assert restored.execution_generation > token.generation
        assert restored.execution_owner_id is None
        assert restored.execution_lease_id is None
        assert runtime.store.complete_execution(
            token,
            status=ProcessStatus.RUNNABLE,
        ) is False
        next_token = runtime.store.claim_execution(pid, owner_id="after.failed.exec")
        assert next_token is not None
        assert next_token.generation > restored.execution_generation
        assert runtime.store.complete_execution(next_token, status=ProcessStatus.RUNNABLE)


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_reopen_fails_closed_stale_running_execution_with_audit(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        pid = runtime.process.spawn(goal="leave a stale execution lease")
        token = runtime.store.claim_execution(pid, owner_id="runtime_that_crashed")
        assert token is not None
        runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            process = reopened.process.get(pid)
            assert process.status == ProcessStatus.PAUSED
            assert process.status_message == "stale_execution_recovery"
            assert process.execution_owner_id is None
            assert process.execution_lease_id is None
            assert process.execution_generation > token.generation
            records = reopened.audit.trace(target=f"process:{pid}")
            assert any(record.action == "stale_execution_recovery" for record in records)
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_restore_high_water_and_fork_identity_survive_reopen(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        pid = runtime.process.spawn(goal="persistent restore high-water")
        token = runtime.store.claim_execution(pid, owner_id="pre-restore-worker")
        assert token is not None
        checkpoint_id = runtime.checkpoint.create(
            pid,
            "capture stale process concurrency identity",
            actor=pid,
        )
        found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
        assert found is not None
        stale_snapshot_revision = int(found[1]["rows"]["processes"][0]["revision"])
        assert runtime.store.complete_execution(token)
        before = runtime.process.get(pid)

        runtime.checkpoint.restore("test", checkpoint_id, require_capability=False)
        first = runtime.process.get(pid)
        runtime.checkpoint.restore("test", checkpoint_id, require_capability=False)
        second = runtime.process.get(pid)
        forked = runtime.checkpoint.fork_from_checkpoint(
            "test",
            checkpoint_id,
            require_capability=False,
        )
        fork_pid = str(forked["fork_root_pid"])
        fork_process = runtime.process.get(fork_pid)

        assert first.revision > before.revision
        assert first.execution_generation > before.execution_generation
        assert second.revision > first.revision
        assert second.execution_generation > first.execution_generation
        assert first.execution_owner_id is first.execution_lease_id is None
        assert second.execution_owner_id is second.execution_lease_id is None
        assert runtime.store.complete_execution(token) is False
        assert runtime.store.release_execution(token) is False
        with pytest.raises(ProcessRevisionConflict):
            runtime.store.patch_process(
                pid,
                {"status_message": "snapshot ABA writer"},
                expected_revision=stale_snapshot_revision,
            )
        assert fork_process.revision == 0
        assert fork_process.execution_generation == 0
        assert fork_process.execution_owner_id is None
        assert fork_process.execution_lease_id is None
        runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            reopened_before = reopened.process.get(pid)
            reopened.checkpoint.restore(
                "test",
                checkpoint_id,
                require_capability=False,
            )
            restored = reopened.process.get(pid)
            persisted_fork = reopened.process.get(fork_pid)

            assert restored.revision > reopened_before.revision
            assert restored.revision > second.revision
            assert restored.execution_generation > reopened_before.execution_generation
            assert restored.execution_generation > second.execution_generation
            assert restored.execution_owner_id is None
            assert restored.execution_lease_id is None
            assert persisted_fork.revision == fork_process.revision
            assert persisted_fork.execution_generation == fork_process.execution_generation
            assert persisted_fork.execution_owner_id is None
            assert persisted_fork.execution_lease_id is None
            assert reopened.store.complete_execution(token) is False
            assert reopened.store.release_execution(token) is False
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_restore_serializes_paused_writer_behind_new_epoch(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        pid = runtime.process.spawn(goal="restore writer barrier")
        checkpoint_id = runtime.checkpoint.create(
            pid,
            "writer barrier snapshot",
            actor=pid,
        )
        current = runtime.process.get(pid)
        current = runtime.store.patch_process(
            pid,
            {"status_message": "pre-restore current state"},
            expected_revision=current.revision,
        )
        stale_revision = current.revision
        restore_holds_store = threading.Event()
        writer_attempted = threading.Event()
        writer_done = threading.Event()
        writer_errors: list[BaseException] = []
        original_prepare = runtime.checkpoint._prepare_restored_process_rows

        def pause_restore(*args: object, **kwargs: object) -> object:
            restore_holds_store.set()
            assert writer_attempted.wait(timeout=5)
            assert writer_done.wait(timeout=0.1) is False
            return original_prepare(*args, **kwargs)

        def stale_writer() -> None:
            try:
                assert restore_holds_store.wait(timeout=5)
                writer_attempted.set()
                runtime.store.patch_process(
                    pid,
                    {"status_message": "stale writer committed"},
                    expected_revision=stale_revision,
                )
            except BaseException as error:
                writer_errors.append(error)
            finally:
                writer_done.set()

        monkeypatch.setattr(
            runtime.checkpoint,
            "_prepare_restored_process_rows",
            pause_restore,
        )
        writer = threading.Thread(target=stale_writer)
        writer.start()
        try:
            runtime.checkpoint.restore("test", checkpoint_id, require_capability=False)
            writer.join(timeout=5)
            restored = runtime.process.get(pid)

            assert not writer.is_alive()
            assert len(writer_errors) == 1
            assert isinstance(writer_errors[0], ProcessRevisionConflict)
            assert restored.revision > stale_revision
            assert restored.status_message != "stale writer committed"
        finally:
            writer.join(timeout=5)
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
@pytest.mark.parametrize("publication_state", ["applying", "failed"])
def test_reopen_compensates_incomplete_process_launch_publication(
    kind: str,
    publication_state: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        now = utc_now()
        pid = f"pid-partial-{uuid4().hex}"
        publication_id = f"publication-partial-{uuid4().hex}"
        runtime.store.insert_runtime_publication(
            publication_id=publication_id,
            kind="process_launch",
            pid=pid,
            owner_instance_id="crashed-runtime",
            plan={"pid": pid, "launch_kind": "spawn", "image_id": "base-agent:v0"},
        )
        runtime.store.insert_process(
            AgentProcess(
                pid=pid,
                parent_pid=None,
                image_id="base-agent:v0",
                status=ProcessStatus.CREATED,
                goal_oid=None,
                memory_view=None,
                capabilities=[],
                loaded_skills={},
                tool_table={},
                event_cursor=None,
                checkpoint_head=None,
                resource_budget=ResourceBudget(),
                resource_usage=ResourceUsage(),
                created_at=now,
                updated_at=now,
            )
        )
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state="applying",
            phase="process_inserted",
            expected_states={"planning"},
        )
        if publication_state == "failed":
            assert runtime.store.advance_runtime_publication(
                publication_id,
                state="rollback_pending",
                phase="compensating",
                expected_states={"applying"},
            )
            assert runtime.store.advance_runtime_publication(
                publication_id,
                state="failed",
                phase="compensation_failed",
                error={"code": "publication_compensation_failed"},
                expected_states={"rollback_pending"},
            )
        runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            assert reopened.store.get_process(pid) is None
            publication = reopened.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["state"] == "rolled_back"
            assert publication["phase"] == "startup_compensated"
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
@pytest.mark.parametrize("publication_state", ["applying", "failed"])
def test_reopen_compensates_incomplete_process_exec_publication(
    kind: str,
    publication_state: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        pid = runtime.process.spawn(goal="before interrupted exec")
        operation = runtime.operations.start(
            kind="runtime",
            name="process.exec",
            actor=pid,
            pid=pid,
        )
        before = runtime.process_exec_state.capture(pid)
        publication_id = f"publication-exec-{uuid4().hex}"
        process = runtime.process.get(pid)
        with runtime.store.transaction():
            admission_token = runtime.store.claim_host_process_exec(
                pid,
                owner_id="crashed-runtime:process.exec",
                expected_revision=process.revision,
                expected_state_generation=process.state_generation,
                expected_execution_generation=process.execution_generation,
            )
            assert admission_token is not None
            runtime.store.insert_runtime_publication(
                publication_id=publication_id,
                kind="process_exec",
                pid=pid,
                owner_instance_id="crashed-runtime",
                plan={
                    "pid": pid,
                    "image_id": "coding-agent:v0",
                    "before_snapshot": before.snapshot.to_mapping(),
                    "before_tool_ids": sorted(before.tool_ids),
                    "operation_id": operation.operation_id,
                    "operation_binding_version": 1,
                    "admission_execution_generation": admission_token.generation,
                    "admission_execution_owner_id": admission_token.owner_id,
                    "admission_execution_lease_id": admission_token.lease_id,
                },
            )
            runtime.operations.bind_runtime_publication(
                operation.operation_id,
                publication_id=publication_id,
                publication_kind="process_exec",
                expected_kind="runtime",
                expected_name="process.exec",
                expected_actor=pid,
                expected_pid=pid,
            )
        process = runtime.process.get(pid)
        with bind_process_execution(admission_token):
            runtime.store.patch_process(
                pid,
                {"image_id": "coding-agent:v0"},
                expected_revision=process.revision,
            )
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state="applying",
            phase="process_exec_applied",
            expected_states={"planning"},
        )
        if publication_state == "failed":
            assert runtime.store.advance_runtime_publication(
                publication_id,
                state="rollback_pending",
                phase="compensating",
                expected_states={"applying"},
            )
            assert runtime.store.advance_runtime_publication(
                publication_id,
                state="failed",
                phase="compensation_failed",
                error={"code": "process_exec_compensation_failed"},
                expected_states={"rollback_pending"},
            )
        runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            assert reopened.process.get(pid).image_id == "base-agent:v0"
            publication = reopened.store.get_runtime_publication(publication_id)
            assert publication is not None and publication["state"] == "rolled_back"
            recovered_operation = reopened.store.get_operation(operation.operation_id)
            assert recovered_operation is not None
            assert recovered_operation.outcome == OperationOutcome.FAILED
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_reopen_reconciles_committed_exec_publication_operation(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        pid = runtime.process.spawn(goal="committed exec operation convergence")
        operation = runtime.operations.start(
            kind="runtime",
            name="process.exec",
            actor=pid,
            pid=pid,
        )
        publication_id = f"publication-exec-committed-{uuid4().hex}"
        with runtime.store.transaction():
            runtime.store.insert_runtime_publication(
                publication_id=publication_id,
                kind="process_exec",
                pid=pid,
                owner_instance_id="runtime-that-crashed",
                plan={
                    "pid": pid,
                    "operation_id": operation.operation_id,
                    "operation_binding_version": 1,
                },
            )
            runtime.operations.bind_runtime_publication(
                operation.operation_id,
                publication_id=publication_id,
                publication_kind="process_exec",
                expected_kind="runtime",
                expected_name="process.exec",
                expected_actor=pid,
                expected_pid=pid,
            )
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state="committed",
            phase="committed",
            expected_states={"planning"},
        )
        runtime.close()

        for _attempt in range(2):
            reopened = Runtime.open(target, config=config)
            try:
                recovered = reopened.store.get_operation(operation.operation_id)
                assert recovered is not None
                assert recovered.outcome == OperationOutcome.SUCCEEDED
                assert recovered.metadata["runtime_publication_id"] == publication_id
            finally:
                reopened.close()


@pytest.mark.parametrize("operation_reconciled", [False, True])
@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_payload_delivery_compensation_requires_open_exact_attempt(
    kind: str,
    operation_reconciled: bool,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal=f"{kind} payload attempt fencing")
        owner_instance_id = f"test.payload-owner.{uuid4().hex}"
        publication_id = f"publication-payload-fence-{uuid4().hex}"
        writer = runtime.uow.checkpoint_restore_publications
        writer.insert_runtime_publication(
            publication_id=publication_id,
            kind="checkpoint_restore",
            pid=pid,
            owner_instance_id=owner_instance_id,
            plan={"checkpoint_id": f"checkpoint-{uuid4().hex}"},
        )
        pending_receipt = {
            "phases": [],
            "artifacts": [],
            "payload_delivery": {"state": "pending"},
        }
        with runtime.store.transaction() as cursor:
            updated = cursor.execute(
                "UPDATE runtime_publications SET state = ?, phase = ?, "
                "receipt_json = ?, operation_reconciled = ?, "
                "payload_delivery_state = ?, updated_at = ? "
                "WHERE publication_id = ?",
                (
                    "committed",
                    "reconciled",
                    dumps(pending_receipt),
                    int(operation_reconciled),
                    "pending",
                    utc_now(),
                    publication_id,
                ),
            )
            assert updated.rowcount == 1

        attempt = CheckpointPayloadDeliveryAttempt(
            started_at=utc_now(),
            attempt_id=f"payload-attempt-{uuid4().hex}",
            owner_instance_id=owner_instance_id,
        )
        assert writer.begin_checkpoint_payload_delivery_attempt(attempt)

        def assign_and_complete() -> None:
            assert writer.transition_payload_delivery(
                publication_id,
                expected_delivery_state="pending",
                delivery_state="confirmed",
                delivery_attempt=attempt,
                owner_instance_id=owner_instance_id,
            )
            assert writer.transition_payload_delivery(
                publication_id,
                expected_delivery_state="confirmed",
                delivery_state="completed",
                expected_attempt=attempt,
                delivery_attempt=attempt,
            )

        assert writer.transition_payload_delivery(
            publication_id,
            expected_delivery_state="pending",
            delivery_state="confirmed",
            delivery_attempt=attempt,
            owner_instance_id=owner_instance_id,
        )
        assert writer.transition_payload_delivery(
            publication_id,
            expected_delivery_state="confirmed",
            delivery_state="pending",
            expected_attempt=attempt,
        )
        compensated = runtime.uow.publications.get_runtime_publication(publication_id)
        assert compensated is not None
        assert compensated["payload_delivery_state"] == "pending"
        assert compensated["payload_delivery_attempt_id"] is None
        assert compensated["payload_delivery_started_at"] is None
        assert compensated["operation_reconciled"] is operation_reconciled

        assign_and_complete()
        assert writer.transition_payload_delivery(
            publication_id,
            expected_delivery_state="completed",
            delivery_state="pending",
            expected_attempt=attempt,
        )
        assign_and_complete()

        if not operation_reconciled:
            assert not writer.ack_checkpoint_payload_delivery_attempt(attempt)
            assert (
                runtime.uow.publications.get_checkpoint_payload_delivery_attempt_state(
                    attempt
                )
                is CheckpointPayloadDeliveryAttemptState.PREPARING
            )
            with runtime.store.transaction() as cursor:
                repaired = cursor.execute(
                    "UPDATE runtime_publications SET operation_reconciled = 1, "
                    "updated_at = ? WHERE publication_id = ? "
                    "AND operation_reconciled = 0",
                    (utc_now(), publication_id),
                )
                assert repaired.rowcount == 1

        assert writer.ack_checkpoint_payload_delivery_attempt(attempt)
        assert (
            runtime.uow.publications.get_checkpoint_payload_delivery_attempt_state(
                attempt
            )
            is CheckpointPayloadDeliveryAttemptState.ACKED
        )
        acked = runtime.store.select_table_rows(
            "checkpoint_payload_delivery_attempts",
            "attempt_id = ?",
            (attempt.attempt_id,),
        )
        assert len(acked) == 1
        assert acked[0]["state"] == "acked"
        before = runtime.uow.publications.get_runtime_publication(publication_id)
        assert before is not None

        assert not writer.transition_payload_delivery(
            publication_id,
            expected_delivery_state="completed",
            delivery_state="pending",
            expected_attempt=attempt,
        )
        assert runtime.uow.publications.get_runtime_publication(publication_id) == before
        assert before["operation_reconciled"] is True


def test_payload_delivery_ack_readback_divergence_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend("sqlite-memory", tmp_path) as runtime:
        pid = runtime.process.spawn(goal="payload acknowledgement readback rollback")
        owner_instance_id = f"test.payload-owner.{uuid4().hex}"
        attempt = CheckpointPayloadDeliveryAttempt(
            started_at=utc_now(),
            attempt_id=f"payload-attempt-{uuid4().hex}",
            owner_instance_id=owner_instance_id,
        )
        writer = runtime.uow.checkpoint_restore_publications
        publication_id = f"publication-payload-readback-{uuid4().hex}"
        writer.insert_runtime_publication(
            publication_id=publication_id,
            kind="checkpoint_restore",
            pid=pid,
            owner_instance_id=owner_instance_id,
            plan={"checkpoint_id": f"checkpoint-{uuid4().hex}"},
        )
        assert writer.begin_checkpoint_payload_delivery_attempt(attempt)
        completed_receipt = {
            "phases": [],
            "artifacts": [],
            "payload_delivery": {"state": "completed"},
            "payload_delivery_attempt": {
                "attempt_id": attempt.attempt_id,
                "started_at": attempt.started_at,
            },
        }
        with runtime.store.transaction() as cursor:
            updated = cursor.execute(
                "UPDATE runtime_publications SET state = ?, phase = ?, "
                "receipt_json = ?, operation_reconciled = 1, "
                "payload_delivery_state = ?, payload_delivery_attempt_id = ?, "
                "payload_delivery_started_at = ?, updated_at = ? "
                "WHERE publication_id = ?",
                (
                    "committed",
                    "reconciled",
                    dumps(completed_receipt),
                    "completed",
                    attempt.attempt_id,
                    attempt.started_at,
                    utc_now(),
                    publication_id,
                ),
            )
            assert updated.rowcount == 1

        monkeypatch.setattr(
            runtime.store,
            "_checkpoint_payload_delivery_attempt_row",
            lambda _cursor, _attempt_id: {
                "attempt_id": attempt.attempt_id,
                "owner_instance_id": attempt.owner_instance_id,
                "state": "preparing",
                "started_at": attempt.started_at,
                "acked_at": None,
            },
            raising=False,
        )
        with pytest.raises(
            ValidationError,
            match="checkpoint payload delivery attempt readback diverged",
        ):
            writer.ack_checkpoint_payload_delivery_attempt(attempt)

        rows = runtime.store.select_table_rows(
            "checkpoint_payload_delivery_attempts",
            "attempt_id = ?",
            (attempt.attempt_id,),
        )
        assert len(rows) == 1
        assert rows[0]["state"] == "preparing"


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_payload_delivery_transition_readback_divergence_rolls_back(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal=f"{kind} payload transition rollback")
        owner_instance_id = f"test.payload-owner.{uuid4().hex}"
        publication_id = f"publication-payload-rollback-{uuid4().hex}"
        writer = runtime.uow.checkpoint_restore_publications
        writer.insert_runtime_publication(
            publication_id=publication_id,
            kind="checkpoint_restore",
            pid=pid,
            owner_instance_id=owner_instance_id,
            plan={"checkpoint_id": f"checkpoint-{uuid4().hex}"},
        )
        pending_receipt = {
            "phases": [],
            "artifacts": [],
            "payload_delivery": {"state": "pending"},
        }
        with runtime.store.transaction() as cursor:
            cursor.execute(
                "UPDATE runtime_publications SET state = ?, phase = ?, "
                "receipt_json = ?, operation_reconciled = 1, "
                "payload_delivery_state = ?, updated_at = ? "
                "WHERE publication_id = ?",
                (
                    "committed",
                    "reconciled",
                    dumps(pending_receipt),
                    "pending",
                    utc_now(),
                    publication_id,
                ),
            )
        attempt = CheckpointPayloadDeliveryAttempt(
            started_at=utc_now(),
            attempt_id=f"payload-attempt-{uuid4().hex}",
            owner_instance_id=owner_instance_id,
        )
        assert writer.begin_checkpoint_payload_delivery_attempt(attempt)

        def fail_readback(*_args: object, **_kwargs: object) -> None:
            raise ValidationError(
                "checkpoint payload delivery transition readback diverged"
            )

        monkeypatch.setattr(
            runtime.store,
            "_require_checkpoint_payload_delivery_readback",
            fail_readback,
        )
        with pytest.raises(
            ValidationError,
            match="checkpoint payload delivery transition readback diverged",
        ):
            writer.transition_payload_delivery(
                publication_id,
                expected_delivery_state="pending",
                delivery_state="confirmed",
                delivery_attempt=attempt,
                owner_instance_id=owner_instance_id,
            )

        rows = runtime.store.select_table_rows(
            "runtime_publications",
            "publication_id = ?",
            (publication_id,),
        )
        assert len(rows) == 1
        assert rows[0]["payload_delivery_state"] == "pending"
        assert rows[0]["payload_delivery_attempt_id"] is None
        assert rows[0]["payload_delivery_started_at"] is None


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_empty_expected_states_reject_all_typed_cas_mutations(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal=f"{kind} empty expected states")

        launch_publication_id = f"publication-launch-cas-{uuid4().hex}"
        runtime.uow.publications.insert_runtime_publication(
            publication_id=launch_publication_id,
            kind="process_launch",
            pid=f"launch-{uuid4().hex}",
            owner_instance_id="test.empty-expected-states",
            plan={"launch_kind": "spawn"},
        )
        launch_before = runtime.uow.publications.get_runtime_publication(
            launch_publication_id
        )
        assert launch_before is not None
        plan_update = {
            "boot_kind": "image",
            "materialized_workspace_root": "/workspace/empty-expected-states",
        }
        assert not runtime.uow.publications.update_runtime_publication_plan(
            launch_publication_id,
            plan_update,
            expected_states=[],
        )
        assert (
            runtime.uow.publications.get_runtime_publication(launch_publication_id)
            == launch_before
        )
        assert runtime.uow.publications.update_runtime_publication_plan(
            launch_publication_id,
            plan_update,
            expected_states=None,
        )

        checkpoint_publication_id = f"publication-checkpoint-cas-{uuid4().hex}"
        checkpoint_plan = CheckpointRestorePlan(
            checkpoint_id=f"checkpoint-cas-{uuid4().hex}",
            pid=pid,
            actor=pid,
            operation_id=f"operation-cas-{uuid4().hex}",
            snapshot_version=1,
            snapshot_sha256="a" * 64,
            current_pids=(pid,),
            snapshot_pids=(pid,),
            scoped_pids=(pid,),
            stale_tool_ids=(),
            finalizer_work_items=(),
        ).to_mapping()
        checkpoint_writer = runtime.uow.checkpoint_restore_publications
        checkpoint_writer.insert_runtime_publication(
            publication_id=checkpoint_publication_id,
            kind="checkpoint_restore",
            pid=pid,
            owner_instance_id="test.empty-expected-states",
            plan=checkpoint_plan,
        )
        plan_anchor = {
            "artifact_id": (
                f"{checkpoint_publication_id}:checkpoint_restore_plan:"
                f"v{CHECKPOINT_RESTORE_PLAN_ANCHOR_VERSION}"
            ),
            "artifact_type": "checkpoint_restore_plan_anchor",
            "anchor_version": CHECKPOINT_RESTORE_PLAN_ANCHOR_VERSION,
            "plan_sha256": hashlib.sha256(
                dumps(checkpoint_plan).encode("utf-8")
            ).hexdigest(),
        }
        checkpoint_before_artifact = runtime.uow.publications.get_runtime_publication(
            checkpoint_publication_id
        )
        assert checkpoint_before_artifact is not None
        assert not checkpoint_writer.record_runtime_publication_artifact(
            checkpoint_publication_id,
            plan_anchor,
            expected_states=[],
        )
        assert (
            runtime.uow.publications.get_runtime_publication(
                checkpoint_publication_id
            )
            == checkpoint_before_artifact
        )
        assert checkpoint_writer.record_runtime_publication_artifact(
            checkpoint_publication_id,
            plan_anchor,
            expected_states=None,
        )

        checkpoint_before_advance = runtime.uow.publications.get_runtime_publication(
            checkpoint_publication_id
        )
        assert checkpoint_before_advance is not None
        assert not checkpoint_writer.advance_runtime_publication(
            checkpoint_publication_id,
            state="reconciliation_pending",
            phase="main_state_committed",
            receipt={"phase": "main_state_committed"},
            expected_states=[],
            expected_phase="planned",
        )
        assert (
            runtime.uow.publications.get_runtime_publication(
                checkpoint_publication_id
            )
            == checkpoint_before_advance
        )
        assert checkpoint_writer.advance_runtime_publication(
            checkpoint_publication_id,
            state="reconciliation_pending",
            phase="main_state_committed",
            receipt={"phase": "main_state_committed"},
            expected_states=None,
            expected_phase="planned",
        )

        operation = runtime.operations.start(
            kind="runtime",
            name="contract.empty_expected_states",
            actor=pid,
            pid=pid,
        )
        operation_before = runtime.uow.evidence.get_operation(operation.operation_id)
        assert operation_before is not None
        operation_update = replace(
            operation_before,
            metadata={**operation_before.metadata, "cas_probe": True},
            updated_at=utc_now(),
        )
        assert not runtime.uow.evidence.update_operation(
            operation_update,
            expected_states=[],
        )
        assert (
            runtime.uow.evidence.get_operation(operation.operation_id)
            == operation_before
        )
        assert runtime.uow.evidence.update_operation(
            operation_update,
            expected_states=None,
        )
        operation_after = runtime.uow.evidence.get_operation(operation.operation_id)
        assert operation_after is not None
        assert operation_after.metadata["cas_probe"] is True
        runtime.operations.finish("succeeded", operation_id=operation.operation_id)


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_runtime_publication_recovery_claim_fences_stale_lease(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        publication_id = f"publication-claim-{uuid4().hex}"
        current = runtime.store.insert_runtime_publication(
            publication_id=publication_id,
            kind="process_exec",
            pid="pid-claim-contract",
            owner_instance_id="runtime-that-crashed",
            plan={"pid": "pid-claim-contract"},
        )
        first = runtime.store.claim_runtime_publication_recovery(
            publication_id,
            claimant_instance_id="recovery-a",
            expected_owner_instance_id=current["owner_instance_id"],
            expected_state=current["state"],
            classification="compensate_process_exec",
            max_attempts=3,
        )
        assert first is not None
        first_lease = first["receipt"]["recovery"]["lease_id"]
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state="failed",
            phase="first_attempt_failed",
            expected_states={"rollback_pending"},
            recovery_lease_id=first_lease,
        )
        failed = runtime.store.get_runtime_publication(publication_id)
        assert failed is not None
        second = runtime.store.claim_runtime_publication_recovery(
            publication_id,
            claimant_instance_id="recovery-b",
            expected_owner_instance_id=failed["owner_instance_id"],
            expected_state=failed["state"],
            classification="compensate_process_exec",
            max_attempts=3,
        )
        assert second is not None
        second_lease = second["receipt"]["recovery"]["lease_id"]
        assert second_lease != first_lease
        assert not runtime.store.advance_runtime_publication(
            publication_id,
            state="rolled_back",
            phase="stale_terminal",
            expected_states={"rollback_pending"},
            recovery_lease_id=first_lease,
        )
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state="rolled_back",
            phase="current_terminal",
            expected_states={"rollback_pending"},
            recovery_lease_id=second_lease,
        )


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_reopen_restores_jit_handle_for_incomplete_process_exec_publication(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        pid = runtime.process.spawn(
            image="toolmaker-agent:v0",
            goal="keep registered jit across exec recovery",
        )
        candidate_id = runtime.tools.propose(
            pid,
            {
                "name": "exec_recovery_echo",
                "description": "Echo one value after recovery.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            source_code=(
                "export function run(args, libos) { return { value: args.value }; }"
            ),
        )
        candidate = runtime.store.get_tool_candidate(candidate_id)
        assert candidate is not None
        candidate.status = ToolCandidateStatus.VALIDATED
        candidate.validation = {"ok": True, "language": "typescript"}
        runtime.store.update_tool_candidate(candidate)
        handle = runtime.tools.register(pid, candidate_id)
        before = runtime.process_exec_state.capture(pid)
        publication_id = f"publication-exec-jit-{uuid4().hex}"
        process = runtime.process.get(pid)
        with runtime.store.transaction():
            admission_token = runtime.store.claim_host_process_exec(
                pid,
                owner_id="crashed-runtime:process.exec",
                expected_revision=process.revision,
                expected_state_generation=process.state_generation,
                expected_execution_generation=process.execution_generation,
            )
            assert admission_token is not None
            runtime.store.insert_runtime_publication(
                publication_id=publication_id,
                kind="process_exec",
                pid=pid,
                owner_instance_id="crashed-runtime",
                plan={
                    "pid": pid,
                    "image_id": "coding-agent:v0",
                    "before_snapshot": before.snapshot.to_mapping(),
                    "before_tool_ids": sorted(before.tool_ids),
                    "admission_execution_generation": admission_token.generation,
                    "admission_execution_owner_id": admission_token.owner_id,
                    "admission_execution_lease_id": admission_token.lease_id,
                },
            )
        process = runtime.process.get(pid)
        with bind_process_execution(admission_token):
            runtime.store.patch_process(
                pid,
                {
                    "image_id": "coding-agent:v0",
                    "tool_table": {},
                    "model_tool_table": {},
                },
                expected_revision=process.revision,
            )
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state="applying",
            phase="process_exec_applied",
            expected_states={"planning"},
        )
        runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            restored = reopened.process.get(pid)
            assert restored.image_id == "toolmaker-agent:v0"
            assert restored.tool_table["exec_recovery_echo"] == handle.tool_id
            assert reopened.tools.loaded_tool_handle(handle.tool_id) is not None
            assert reopened.tools.jit_source(handle.tool_id) is not None
            publication = reopened.store.get_runtime_publication(publication_id)
            assert publication is not None and publication["state"] == "rolled_back"
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_runtime_publication_terminal_phase_preserves_original_error(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        publication_id = f"publication-error-{uuid4().hex}"
        pid = f"pid-error-{uuid4().hex}"
        runtime.store.insert_runtime_publication(
            publication_id=publication_id,
            kind="process_launch",
            pid=pid,
            owner_instance_id="test-runtime",
            plan={"pid": pid},
        )
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state="rollback_pending",
            phase="compensating",
            error={"code": "launch_failed", "error_type": "InjectedFailure"},
            expected_states={"planning"},
        )
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state="rolled_back",
            phase="compensated",
            receipt={"phase": "compensated", "pid": pid},
            expected_states={"rollback_pending"},
        )

        publication = runtime.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["error"] == {
            "code": "launch_failed",
            "error_type": "InjectedFailure",
        }


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_llm_call_limit_selects_latest_window_then_returns_chronologically(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        records = [
            ("call-old", "pid-a", "2026-01-01T00:00:01+00:00"),
            ("call-tie-a", "pid-a", "2026-01-01T00:00:02+00:00"),
            ("call-tie-b", "pid-a", "2026-01-01T00:00:02+00:00"),
            ("call-new", "pid-a", "2026-01-01T00:00:03+00:00"),
            ("call-global-new", "pid-b", "2026-01-01T00:00:04+00:00"),
        ]
        for call_id, pid, created_at in records:
            runtime.store.insert_llm_call(
                LLMCallRecord(
                    call_id=call_id,
                    pid=pid,
                    image_id=None,
                    purpose="latest-window",
                    status="ok",
                    created_at=created_at,
                )
            )

        assert [call.call_id for call in runtime.store.list_llm_calls(limit=3)] == [
            "call-tie-b",
            "call-new",
            "call-global-new",
        ]
        assert [call.call_id for call in runtime.store.list_llm_calls(pid="pid-a", limit=2)] == [
            "call-tie-b",
            "call-new",
        ]


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_llm_cache_usage_counters_survive_backend_reopen(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(goal="cache usage persistence")
            runtime.store.insert_llm_call(
                LLMCallRecord(
                    call_id="cache-usage-call",
                    pid=pid,
                    image_id="base-agent:v0",
                    purpose="cache-usage-contract",
                    status="ok",
                    usage={
                        "prompt_tokens": 64,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 32,
                    },
                    created_at=utc_now(),
                )
            )
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            call = reopened.store.list_llm_calls(pid=pid)[0]
            assert call.usage == {
                "prompt_tokens": 64,
                "cache_read_tokens": 0,
                "cache_write_tokens": 32,
            }
        finally:
            reopened.close()


@contextlib.contextmanager
def _runtime_for_backend(kind: str, tmp_path: Path) -> Iterator[Runtime]:
    target: str | Path | None
    config = AgentLibOSConfig()
    postgres_context = contextlib.nullcontext(None)
    if kind == "sqlite-memory":
        target = "local"
    elif kind == "sqlite-file":
        target = tmp_path / "contract.sqlite"
    elif kind == "postgres":
        postgres_context = _postgres_schema_dsn()
        target = None
    else:
        raise AssertionError(f"unknown backend: {kind}")

    with postgres_context as dsn:
        if dsn is not None:
            config = AgentLibOSConfig(runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn))
            target = dsn
        runtime = Runtime.open(target, config=config)
        try:
            yield runtime
        finally:
            runtime.close()


def _raw_process_row_and_counters(
    runtime: Runtime,
    pid: str,
) -> tuple[dict[str, object], dict[str, int]]:
    process_rows = runtime.store._query(
        "SELECT * FROM processes WHERE pid = ?",
        (pid,),
    )
    assert len(process_rows) == 1
    counter_names = runtime.store._process_concurrency_counter_names(pid)
    counter_rows = runtime.store._query(
        """
        SELECT counter_name, value
          FROM runtime_counters
         WHERE counter_name IN (?, ?, ?)
         ORDER BY counter_name
        """,
        counter_names,
    )
    return (
        dict(process_rows[0]),
        {
            str(row["counter_name"]): int(row["value"])
            for row in counter_rows
        },
    )


def _insert_process_exec_commit_publication(
    runtime: Runtime,
    *,
    publication_id: str,
    pid: str,
    token: ProcessExecutionToken,
    kind: str = "process_exec",
    state: str = "applying",
    phase: str = "skills_configured",
    plan_overrides: dict[str, object] | None = None,
) -> None:
    plan: dict[str, object] = {
        "pid": pid,
        "admission_execution_generation": token.generation,
        "admission_execution_owner_id": token.owner_id,
        "admission_execution_lease_id": token.lease_id,
    }
    plan.update(plan_overrides or {})
    initial_phase = phase if state == "planning" else "planned"
    runtime.store.insert_runtime_publication(
        publication_id=publication_id,
        kind=kind,
        pid=pid,
        owner_instance_id="test.exec.commit",
        plan=plan,
        phase=initial_phase,
    )
    if state != "planning":
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state=state,
            phase=phase,
            receipt={"phase": phase},
            expected_states={"planning"},
        )


def _committed_exec_publication(
    runtime: Runtime,
    pid: str,
) -> tuple[dict[str, object], dict[str, object]]:
    publications = [
        publication
        for publication in runtime.store.list_runtime_publications(
            states={"committed"},
            pid=pid,
        )
        if publication["kind"] == "process_exec"
    ]
    assert len(publications) == 1
    publication = publications[0]
    committed = [
        phase
        for phase in publication["receipt"].get("phases", [])
        if phase.get("phase") == "committed"
    ]
    assert len(committed) == 1
    return publication, committed[0]


@contextlib.contextmanager
def _persistent_target(kind: str, tmp_path: Path) -> Iterator[tuple[str | Path | None, AgentLibOSConfig]]:
    if kind == "sqlite-file":
        yield tmp_path / "payload.sqlite", AgentLibOSConfig()
        return
    if kind == "postgres":
        with _postgres_schema_dsn() as dsn:
            yield dsn, AgentLibOSConfig(runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn))
        return
    raise AssertionError(f"unknown persistent backend: {kind}")


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_contract_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield _dsn_with_search_path(dsn, schema)
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


def _dsn_with_search_path(dsn: str, schema: str) -> str:
    parsed = urlsplit(dsn)
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != "options"]
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_capability_status_transition_uses_expected_state_cas(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal=f"{kind} capability lifecycle CAS")
        defended = runtime.capability.grant(
            pid,
            "shell:defended",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )
        defended = replace(
            defended,
            status=CapabilityStatus.DISABLED,
            metadata={**defended.metadata, "disabled_by": "defender"},
        )
        runtime.store.update_capability(defended)

        missed = runtime.store.transition_capability_status(
            defended.cap_id,
            expected_status=CapabilityStatus.ACTIVE,
            status=CapabilityStatus.EXEC_REVOKED,
            metadata={"exec_rollback_token": "must-not-publish"},
        )

        assert missed is None
        assert runtime.store.get_capability(defended.cap_id) == defended

        active = runtime.capability.grant(
            pid,
            "shell:active",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )
        transitioned = runtime.store.transition_capability_status(
            active.cap_id,
            expected_status=CapabilityStatus.ACTIVE,
            status=CapabilityStatus.EXEC_REVOKED,
            metadata={"exec_rollback_token": "published"},
        )

        assert transitioned is not None
        assert transitioned.status == CapabilityStatus.EXEC_REVOKED
        assert transitioned.metadata == {"exec_rollback_token": "published"}
        assert runtime.store.get_capability(active.cap_id) == transitioned


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_runtime_store_contract_core_records(kind: str, tmp_path: Path) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal=f"{kind} store contract")
        handle = runtime.memory.create_object(
            pid=pid,
            object_type=ObjectType.ARTIFACT,
            payload={"backend": kind, "items": [1, 2, 3]},
            name="contract.payload",
        )
        runtime.capability.grant(pid, "filesystem:workspace:*", [CapabilityRight.READ], issued_by="test")
        runtime.capability.grant(pid, "human:owner", [CapabilityRight.WRITE], issued_by="test")
        runtime.messages.post(sender="human:owner", recipient_pid=pid, subject="contract", body="hello")
        runtime.human.output(pid, "visible status", channel="gui")
        runtime.ratings.upsert(pid, score=5, comment="works")

        endpoint = JsonRpcEndpointSpec(
            schema_version=1,
            endpoint_id="contract-jsonrpc",
            url="https://api.example.test/jsonrpc",
            headers={},
            methods=[
                JsonRpcMethodSpec(
                    method_id="echo",
                    rpc_method="demo.echo",
                    right="read",
                    rollback_class="no_rollback_required",
                    state_mutation=False,
                    information_flow=True,
                )
            ],
            timeout_s=1.0,
            max_request_bytes=1024,
            max_response_bytes=2048,
        )
        runtime.store.upsert_jsonrpc_endpoint(endpoint, registered_by="test", created_at=utc_now())
        mcp_server = McpServerSpec(
            schema_version=1,
            server_id="contract-mcp",
            transport="stdio",
            stdio=McpStdioTransportSpec(command="python3", args=["-m", "demo_mcp"], env={}),
            tools=[
                McpToolSpec(
                    tool_id="echo",
                    mcp_name="demo.echo",
                    right="read",
                    rollback_class="no_rollback_required",
                    state_mutation=False,
                    information_flow=True,
                )
            ],
            timeout_s=1.0,
            max_request_bytes=1024,
            max_response_bytes=2048,
        )
        runtime.store.upsert_mcp_server(mcp_server, registered_by="test", created_at=utc_now())
        runtime.store.insert_llm_call(
            LLMCallRecord(
                call_id="contract-llm-call",
                pid=pid,
                image_id="base-agent:v0",
                purpose="contract",
                status="ok",
                messages=[{"role": "user", "content": kind}],
                response_content="ok",
                created_at=utc_now(),
            )
        )
        checkpoint_id = runtime.checkpoint.create(pid, "store contract", require_capability=False)
        operation = runtime.operations.start(
            kind="runtime",
            name="contract.operation",
            actor=pid,
            pid=pid,
            expected_roles=["result"],
        )
        assert runtime.operations.link_evidence(
            "result",
            "contract-operation-result",
            "result",
            operation_id=operation.operation_id,
        ) is not None
        assert runtime.operations.link_evidence(
            "result",
            "contract-operation-result",
            "result",
            operation_id=operation.operation_id,
        ) is None
        runtime.operations.finish("succeeded", operation_id=operation.operation_id)
        cas_operation = runtime.operations.start(
            kind="runtime",
            name="contract.cas",
            actor=pid,
            pid=pid,
        )
        runtime.operations.wait(operation_id=cas_operation.operation_id)
        waiting_snapshot = runtime.store.get_operation(cas_operation.operation_id)
        assert waiting_snapshot is not None
        assert runtime.operations.resume(cas_operation.operation_id).state == OperationState.RUNNING
        stale_terminal = replace(
            waiting_snapshot,
            state=OperationState.TERMINAL,
            outcome=OperationOutcome.SUCCEEDED,
            completed_at=utc_now(),
            updated_at=utc_now(),
        )
        assert runtime.store.update_operation(
            stale_terminal,
            expected_states=[OperationState.WAITING.value],
        ) is False
        runtime.operations.finish("succeeded", operation_id=cas_operation.operation_id)
        manifest = ContextMaterializationManifest(
            materialization_id="contract-context-manifest",
            pid=pid,
            view_id="contract-view",
            policy="contract",
            budget_tokens=64,
            rendered_tokens=12,
            rendered_sha256="a" * 64,
            context_generation="generation-1",
            context_oid=None,
            context_version=None,
            objects=[
                {
                    "oid": handle.oid,
                    "version": 1,
                    "type": ObjectType.ARTIFACT.value,
                    "disposition": "included",
                    "reason": "selected",
                    "transform": "verbatim",
                    "tokens": 12,
                    "rendered_sha256": "b" * 64,
                }
            ],
            created_at=utc_now(),
        )
        runtime.store.insert_context_materialization_manifest(manifest)

        assert runtime.process.get(pid) is not None
        assert runtime.memory.get_object(pid, handle).payload["backend"] == kind
        assert runtime.store.list_capabilities(subject=pid)
        assert runtime.audit.trace()
        assert runtime.events.list(target=pid)
        assert runtime.human.list(pid=pid)
        assert runtime.messages.list(pid)
        assert runtime.ratings.get(pid).score == 5
        assert runtime.store.get_jsonrpc_endpoint("contract-jsonrpc")[0].method_by_id("echo") is not None
        assert runtime.store.get_mcp_server("contract-mcp")[0].tool_by_id("echo") is not None
        assert runtime.store.list_llm_calls(pid=pid)[0].call_id == "contract-llm-call"
        assert runtime.store.get_checkpoint_snapshot(checkpoint_id) is not None
        stored_operation = runtime.store.get_operation(operation.operation_id)
        assert stored_operation is not None
        assert stored_operation.outcome.value == "succeeded"
        assert runtime.store.list_operation_evidence(operation_ids=[operation.operation_id])
        assert runtime.store.get_context_materialization_manifest(manifest.materialization_id) == manifest


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_runtime_store_contract_data_flow_registry_decisions_and_file_labels(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        now = utc_now()
        source = DataSourceRef("contract-source", 1, "a" * 64)
        labels = DataLabels(
            sensitivity="secret",
            trust_level="verified",
            integrity="checked",
            tenant="tenant-a",
            principal="principal-a",
        )
        trust = SinkTrustSpec(
            trust_id="contract-trust-v1",
            pattern="filesystem:workspace:secure/*",
            trust_level="trusted",
            max_sensitivity="secret",
            generation=1,
            created_by="test",
            created_at=now,
            tenants=("tenant-a",),
            principals=("principal-a",),
        )

        assert runtime.store.get_sink_trust_generation() == 0
        assert runtime.store.register_sink_trust(trust) == trust
        assert runtime.store.get_sink_trust_generation() == 1
        assert runtime.store.inspect_sink_trust(trust.pattern) == trust

        replacement = SinkTrustSpec(
            trust_id="contract-trust-v2",
            pattern=trust.pattern,
            trust_level="trusted",
            max_sensitivity="restricted",
            generation=2,
            created_by="test",
            created_at=utc_now(),
            tenants=("tenant-a",),
            principals=("principal-a",),
        )
        with pytest.raises(ValidationError, match="already exists"):
            runtime.store.register_sink_trust(replacement)
        assert runtime.store.get_sink_trust_generation() == 1
        assert runtime.store.inspect_sink_trust(trust.pattern) == trust

        runtime.store.register_sink_trust(replacement, replace=True)
        historical = runtime.store.get_sink_trust(trust.trust_id)
        assert historical is not None
        assert historical.active is False
        assert historical.deactivated_at == replacement.created_at
        assert runtime.store.inspect_sink_trust(trust.pattern) == replacement
        assert runtime.store.list_sink_trust() == [replacement]

        decision = DataFlowDecision(
            decision_id="contract-data-flow-decision",
            pid="contract-pid",
            sink="filesystem:workspace:secure/result.txt",
            direction="egress",
            outcome="allow",
            reason="trusted_sink_clearance",
            labels=labels,
            source_refs=(source,),
            payload_hash="b" * 64,
            trust_id=replacement.trust_id,
            trust_hash=replacement.spec_hash,
            registry_generation=2,
            created_at=utc_now(),
        )
        runtime.store.insert_data_flow_decision(decision)
        assert runtime.store.get_data_flow_decision(decision.decision_id) == decision
        assert runtime.store.list_data_flow_decisions(pid=decision.pid) == [decision]
        decision_row = runtime.store.select_table_rows(
            "data_flow_decisions",
            "decision_id = ?",
            (decision.decision_id,),
        )[0]
        assert "payload_json" not in decision_row
        assert "payload" not in decision_row

        first = FileLabelBinding(
            binding_id="contract-file-label-v1",
            normalized_path="secure/result.txt",
            content_sha256="c" * 64,
            labels=labels,
            source_refs=(source,),
            generation=1,
            tombstoned=False,
            active=True,
            created_by="contract-pid",
            created_at=utc_now(),
        )
        runtime.store.upsert_file_label_binding(first)
        stale = replace(first, binding_id="contract-file-label-stale", content_sha256="e" * 64)
        with pytest.raises(ValidationError, match="generation conflict"):
            runtime.store.upsert_file_label_binding(stale)
        assert runtime.store.get_file_label_binding(first.normalized_path) == first
        assert runtime.store.get_file_label_binding_by_id(first.binding_id) == first

        second = FileLabelBinding(
            binding_id="contract-file-label-v2",
            normalized_path=first.normalized_path,
            content_sha256="d" * 64,
            labels=labels,
            source_refs=(source,),
            generation=2,
            tombstoned=False,
            active=True,
            created_by="contract-pid",
            created_at=utc_now(),
        )
        runtime.store.upsert_file_label_binding(second)

        assert runtime.store.get_file_label_binding(first.normalized_path) == second
        assert runtime.store.get_file_label_binding_by_id(first.binding_id) == replace(
            first,
            active=False,
            superseded_at=second.created_at,
        )
        assert runtime.store.get_file_label_binding_by_id(second.binding_id) == second
        assert runtime.store.get_file_label_binding_generation(first.normalized_path) == 2
        history = runtime.store.list_file_label_bindings(
            normalized_path=first.normalized_path,
            include_history=True,
        )
        assert [item.binding_id for item in history] == [second.binding_id, first.binding_id]
        assert history[1].active is False

        tombstone = runtime.store.tombstone_file_label_binding(
            first.normalized_path,
            binding_id="contract-file-label-tombstone",
            created_by="test",
            created_at=utc_now(),
        )
        assert tombstone is not None
        assert tombstone.tombstoned is True
        assert runtime.store.get_file_label_binding(first.normalized_path) is None
        assert runtime.store.get_file_label_binding_by_id(second.binding_id) == replace(
            second,
            active=False,
            superseded_at=tombstone.created_at,
        )
        assert runtime.store.get_file_label_binding_by_id(tombstone.binding_id) == tombstone
        assert runtime.store.get_file_label_binding_generation(first.normalized_path) == 3
        assert runtime.store.list_file_label_bindings(
            normalized_path=first.normalized_path,
            include_history=True,
            include_tombstones=True,
        )[0] == tombstone

        assert runtime.store.unregister_sink_trust(
            replacement.pattern,
            generation=3,
            deactivated_at=utc_now(),
        )
        assert runtime.store.get_sink_trust_generation() == 3
        assert runtime.store.inspect_sink_trust(replacement.pattern) is None


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_process_message_ack_is_unread_cas_and_preserves_envelope(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="message ack CAS")
        posted = runtime.messages.post(
            sender="test.host",
            recipient_pid=pid,
            channel="control",
            correlation_id="job-1",
            subject="preserve subject",
            body="preserve body",
            payload={"preserve": True},
            metadata={"custom": "original"},
        )
        selected = runtime.store.get_process_message(posted.message_id)
        assert selected is not None
        expected_metadata = dict(selected.metadata)
        concurrent_metadata = {**expected_metadata, "concurrent": "preserved"}
        assert runtime.store.update_process_message_metadata(
            posted.message_id,
            recipient_pid=pid,
            expected_metadata=expected_metadata,
            metadata=concurrent_metadata,
            updated_at="metadata-updated",
        )

        assert runtime.store.ack_process_message(
            posted.message_id,
            recipient_pid=pid,
            acked_at="first-ack",
            updated_at="first-ack-updated",
        )
        acked = runtime.store.get_process_message(posted.message_id)
        assert acked is not None
        assert acked.status == ProcessMessageStatus.ACKED
        assert acked.acked_at == "first-ack"
        assert acked.updated_at == "first-ack-updated"
        assert acked.sender == "test.host"
        assert acked.channel == "control"
        assert acked.correlation_id == "job-1"
        assert acked.subject == "preserve subject"
        assert acked.body == "preserve body"
        assert acked.payload == {"preserve": True}
        assert acked.metadata == concurrent_metadata

        assert not runtime.store.ack_process_message(
            posted.message_id,
            recipient_pid=pid,
            acked_at="second-ack",
            updated_at="second-ack-updated",
        )
        assert runtime.store.get_process_message(posted.message_id) == acked


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_process_message_status_set_queries_filter_before_limit_and_count(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="message status-set query")
        superseded = runtime.messages.post(
            sender="test.host",
            recipient_pid=pid,
            channel="status-set",
            subject="excluded restore history",
        )
        runtime.store.update_process_message(
            replace(
                superseded,
                status=ProcessMessageStatus.SUPERSEDED_BY_RESTORE,
                created_at="2000-01-01T00:00:00+00:00",
                updated_at="2000-01-01T00:00:00+00:00",
            )
        )
        unread = runtime.messages.post(
            sender="test.host",
            recipient_pid=pid,
            channel="status-set",
            subject="visible unread",
        )
        runtime.store.update_process_message(
            replace(
                unread,
                created_at="2001-01-01T00:00:00+00:00",
                updated_at="2001-01-01T00:00:00+00:00",
            )
        )
        acknowledged = runtime.messages.post(
            sender="test.host",
            recipient_pid=pid,
            channel="status-set",
            subject="visible acknowledged",
        )
        runtime.store.update_process_message(
            replace(
                acknowledged,
                created_at="2002-01-01T00:00:00+00:00",
                updated_at="2002-01-01T00:00:00+00:00",
            )
        )
        runtime.messages.ack(pid, [acknowledged.message_id])

        all_messages = runtime.store.list_process_messages(
            pid,
            status=None,
            statuses=None,
            channel="status-set",
        )
        assert [message.message_id for message in all_messages] == [
            superseded.message_id,
            unread.message_id,
            acknowledged.message_id,
        ]
        assert runtime.store.count_process_messages(
            pid,
            status=None,
            statuses=None,
            channel="status-set",
        ) == 3

        visible_statuses = (
            ProcessMessageStatus.UNREAD,
            ProcessMessageStatus.ACKED,
            ProcessMessageStatus.UNREAD,
        )
        first_visible = runtime.store.list_process_messages(
            pid,
            statuses=visible_statuses,
            channel="status-set",
            limit=1,
        )
        assert [message.message_id for message in first_visible] == [unread.message_id]
        assert runtime.store.count_process_messages(
            pid,
            statuses=visible_statuses,
            channel="status-set",
        ) == 2

        assert runtime.store.list_process_messages(pid, statuses=()) == []
        assert runtime.store.count_process_messages(pid, statuses=()) == 0
        with pytest.raises(ValidationError, match="cannot combine status and statuses"):
            runtime.store.list_process_messages(
                pid,
                status=ProcessMessageStatus.UNREAD,
                statuses=(ProcessMessageStatus.UNREAD,),
            )
        with pytest.raises(ValidationError, match="cannot combine status and statuses"):
            runtime.store.count_process_messages(
                pid,
                status=ProcessMessageStatus.UNREAD,
                statuses=(ProcessMessageStatus.UNREAD,),
            )


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_checkpoint_rejects_non_json_native_payload_without_type_loss(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="strict checkpoint object payload")
        handle = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"coordinates": (1, 2)},
            name="tuple-payload",
        )

        with pytest.raises(
            ValidationError,
            match="checkpoint snapshot contains non-JSON value tuple",
        ):
            runtime.checkpoint.create(
                pid,
                "must not stringify or coerce payload types",
                actor=pid,
            )

        assert runtime.store.list_checkpoints(pid) == []
        assert runtime.process.get(pid).checkpoint_head is None
        stored = runtime.store.get_object(handle.oid)
        assert stored is not None
        assert stored.payload == {"coordinates": (1, 2)}
        assert isinstance(stored.payload["coordinates"], tuple)


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_exec_snapshot_rows_and_payload_share_one_store_snapshot(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="atomic exec snapshot payload")
        handle = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"version": 1},
            name="exec-snapshot-payload",
            immutable=False,
        )
        object_rows_selected = threading.Event()
        writer_started = threading.Event()
        writer_finished = threading.Event()
        writer_errors: list[BaseException] = []
        snapshots = runtime.uow.snapshots
        original_owned_rows = snapshots._owned_object_rows

        def coordinated_owned_rows(selected_pid: str) -> list[dict[str, object]]:
            rows = original_owned_rows(selected_pid)
            object_rows_selected.set()
            assert writer_started.wait(timeout=5)
            # Snapshot capture owns the transaction here, so the payload
            # update cannot commit between its row and cache reads.
            writer_finished.wait(timeout=0.2)
            return rows

        monkeypatch.setattr(snapshots, "_owned_object_rows", coordinated_owned_rows)

        def update_payload() -> None:
            try:
                assert object_rows_selected.wait(timeout=5)
                writer_started.set()
                runtime.memory.update_object(
                    pid,
                    handle,
                    ObjectPatch(payload={"version": 2}),
                )
            except BaseException as exc:
                writer_errors.append(exc)
            finally:
                writer_finished.set()

        writer = threading.Thread(target=update_payload, daemon=True)
        writer.start()
        rows, payloads = snapshots.capture_process_exec_snapshot_rows(
            pid,
            process_namespace=runtime.memory.process_namespace(pid),
        )
        writer.join(timeout=5)

        assert not writer.is_alive()
        assert writer_errors == []
        object_row = next(row for row in rows.objects if row["oid"] == handle.oid)
        assert object_row["version"] == 1
        assert payloads[handle.oid] == {"version": 1}
        current = runtime.store.get_object(handle.oid)
        assert current is not None
        assert current.version == 2
        assert current.payload == {"version": 2}


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_legacy_process_message_envelope_marker_remains_user_payload(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="legacy message payload")
        message = runtime.messages.post(
            sender="test.host",
            recipient_pid=pid,
            payload={"placeholder": True},
            metadata={"custom": "preserved"},
        )
        legacy_payload = {
            "__agent_libos_process_message_v1__": True,
            "payload": {"original": "user-data"},
            "metadata": {"data_labels": {"sensitivity": "secret"}},
        }
        runtime.store._execute(
            "UPDATE process_messages SET payload_json = ? WHERE message_id = ?",
            (json.dumps(legacy_payload), message.message_id),
        )

        decoded = runtime.store.get_process_message(message.message_id)

        assert decoded is not None
        assert decoded.payload == legacy_payload
        assert decoded.metadata["custom"] == "preserved"




@pytest.mark.parametrize(
    "raw_context",
    [
        pytest.param(json.dumps({"labels": {"trust_level": "trusted"}}), id="partial"),
        pytest.param("{malformed-json", id="malformed"),
    ],
)
@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_noncanonical_pending_flow_context_fails_closed_on_reopen(
    kind: str,
    raw_context: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        assert target is not None
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="invalid pending context")
            runtime.store.upsert_llm_pending_action(
                pid,
                {
                    "resume_token": "invalid-pending-context-token",
                    "wait_type": "message",
                    "filters": {},
                    "action": {"action": "receive_process_messages"},
                    "data_flow_context": DataFlowContext().to_dict(),
                    "content_preview": "",
                    "tool_call_count": 1,
                    "status": "pending",
                },
            )
        finally:
            runtime.close()

        if kind == "sqlite-file":
            with sqlite3.connect(target) as connection:
                connection.execute(
                    "UPDATE llm_pending_actions SET data_flow_context_json = ? WHERE pid = ?",
                    (raw_context, pid),
                )
                connection.commit()
        else:
            import psycopg

            with psycopg.connect(str(target), autocommit=True) as connection:
                connection.execute(
                    "UPDATE llm_pending_actions SET data_flow_context_json = %s WHERE pid = %s",
                    (raw_context, pid),
                )

        with pytest.raises(ValidationError, match="pending LLM action"):
            Runtime.open(target, config=config)


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_minimal_complete_pending_flow_context_is_normalized_without_failure(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        assert target is not None
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="minimal pending context")
            runtime.store.upsert_llm_pending_action(
                pid,
                {
                    "resume_token": "minimal-pending-context-token",
                    "wait_type": "message",
                    "filters": {},
                    "action": {"action": "receive_process_messages"},
                    "data_flow_context": DataFlowContext().to_dict(),
                    "content_preview": "",
                    "tool_call_count": 1,
                    "status": "pending",
                },
            )
        finally:
            runtime.close()

        minimal = json.dumps(
            {
                "labels": {
                    "sensitivity": "normal",
                    "trust_level": "trusted",
                    "integrity": "verified",
                }
            }
        )
        if kind == "sqlite-file":
            with sqlite3.connect(target) as connection:
                connection.execute(
                    "UPDATE llm_pending_actions SET data_flow_context_json = ? WHERE pid = ?",
                    (minimal, pid),
                )
                connection.commit()
        else:
            import psycopg

            with psycopg.connect(str(target), autocommit=True) as connection:
                connection.execute(
                    "UPDATE llm_pending_actions SET data_flow_context_json = %s WHERE pid = %s",
                    (minimal, pid),
                )

        reopened = Runtime.open(target, config=config)
        try:
            assert reopened.process.get(pid).status == ProcessStatus.RUNNABLE
            pending = reopened.store.get_llm_pending_action(pid)
            assert pending is not None
            assert pending["status"] == "pending"
            labels = DataFlowContext.from_dict(pending["data_flow_context"]).labels
            assert labels.sensitivity.value == "normal"
            assert labels.trust_level.value == "trusted"
            assert labels.integrity.value == "verified"
        finally:
            reopened.close()




def test_released_process_result_wraps_corrupt_metadata_as_validation_error(
    tmp_path: Path,
) -> None:
    target = tmp_path / "corrupt-released-result.sqlite"
    runtime = Runtime.open(target)
    try:
        parent = runtime.process.spawn(image="base-agent:v0", goal="wait corrupt result")
        child = runtime.process.spawn_child(parent, "produce corrupt result")
        result = runtime.memory.create_object(
            child,
            ObjectType.SUMMARY,
            {"done": True},
            name="corrupt.process.result",
        )
        runtime.process.exit(child, result=result)
    finally:
        runtime.close()

    with sqlite3.connect(target) as connection:
        connection.execute(
            "UPDATE objects SET metadata_json = ? WHERE oid = ?",
            ("{malformed-json", result.oid),
        )
        connection.commit()

    reopened = Runtime.open(target)
    try:
        with pytest.raises(
            ValidationError,
            match=f"invalid persisted metadata for released object {result.oid}",
        ):
            reopened.process.wait(parent, child)
    finally:
        reopened.close()




@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_pending_action_rejects_incomplete_flow_labels(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(image="base-agent:v0", goal="reject partial pending labels")

        with pytest.raises(ValidationError, match="complete security labels"):
            runtime.store.upsert_llm_pending_action(
                pid,
                {
                    "wait_type": "message",
                    "filters": {},
                    "action": {"action": "receive_process_messages"},
                    "data_flow_context": {"labels": {"trust_level": "untrusted"}},
                    "status": "pending",
                },
            )

        assert runtime.store.get_llm_pending_action(pid) is None


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_gui_snapshot_visibility_filters_before_limit_with_stable_ties(
    kind: str,
    tmp_path: Path,
) -> None:
    event_specs = [
        ("gui-event-00", EventType.PROCESS_CREATED, {"marker": "visible"}),
        ("gui-event-01", EventType.HUMAN_OUTPUT, {"purpose": "gui_presentation"}),
        ("gui-event-02", EventType.HUMAN_OUTPUT, {"purpose": "gui_presentation_extra"}),
        ("gui-event-03", EventType.DATA_FLOW_DECISION, {"sink": "human:owner:gui"}),
        ("gui-event-04", EventType.DATA_FLOW_DECISION, {"sink": "human:owner:gui-extra"}),
        ("gui-event-05", EventType.HUMAN_OUTPUT, {"purpose": "terminal_output"}),
        ("gui-event-06", EventType.PROCESS_SIGNAL, {"signal": "resume"}),
    ]
    audit_specs = [
        ("gui-audit-00", "test.visible", "process:pid", {"marker": "visible"}),
        ("gui-audit-01", "human.output", "human:owner:terminal", {"purpose": "gui_presentation"}),
        ("gui-audit-02", "human.output", "human:owner:terminal", {"purpose": "gui_presentation_extra"}),
        ("gui-audit-03", "data_flow.egress", "human:owner:gui", {"sink": "human:owner:gui"}),
        ("gui-audit-04", "data_flow.egress", "human:owner:gui-extra", {"sink": "human:owner:gui-extra"}),
        ("gui-audit-05", "data_flow.egress", "service:archive", {"sink": "service:archive"}),
        ("gui-audit-06", "test.visible", "process:pid", None),
    ]

    with _runtime_for_backend(kind, tmp_path) as runtime:
        timestamp = utc_now()
        for event_id, event_type, payload in event_specs:
            runtime.store.insert_event(
                Event(
                    event_id=event_id,
                    type=event_type,
                    source="test",
                    target="pid",
                    payload=payload,
                    priority=EventPriority.NORMAL,
                    created_at=timestamp,
                )
            )
        for record_id, action, target, decision in audit_specs:
            runtime.store.insert_audit(
                AuditRecord(
                    record_id=record_id,
                    timestamp=timestamp,
                    actor="test",
                    action=action,
                    target=target,
                    input_refs=[],
                    output_refs=[],
                    capability_refs=[],
                    decision=decision,
                    correlation_id=None,
                )
            )

        events = runtime.store.list_events(
            limit=3,
            include_gui_presentation=False,
        )
        event_page_one = runtime.store.list_events(
            limit=2,
            after_event_id="gui-event-00",
            include_gui_presentation=False,
        )
        event_page_two = runtime.store.list_events(
            limit=2,
            after_event_id="gui-event-04",
            include_gui_presentation=False,
        )
        audit = runtime.store.list_audit(
            limit=3,
            include_gui_presentation=False,
        )
        audit_match_any = runtime.store.list_audit(
            actor="test",
            target="human:owner:terminal",
            match_any=True,
            include_gui_presentation=False,
        )

        assert [item.event_id for item in events] == [
            "gui-event-04",
            "gui-event-05",
            "gui-event-06",
        ]
        assert [item.event_id for item in event_page_one] == [
            "gui-event-02",
            "gui-event-04",
        ]
        assert [item.event_id for item in event_page_two] == [
            "gui-event-05",
            "gui-event-06",
        ]
        assert [item.record_id for item in audit] == [
            "gui-audit-04",
            "gui-audit-05",
            "gui-audit-06",
        ]
        assert [item.record_id for item in audit_match_any] == [
            "gui-audit-00",
            "gui-audit-02",
            "gui-audit-04",
            "gui-audit-05",
            "gui-audit-06",
        ]
        all_event_ids = {item.event_id for item in runtime.store.list_events()}
        all_audit_ids = {item.record_id for item in runtime.store.list_audit()}
        assert {item[0] for item in event_specs}.issubset(all_event_ids)
        assert {item[0] for item in audit_specs}.issubset(all_audit_ids)




def test_sqlite_tree_file_label_query_is_indexed_and_batched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_config = AgentLibOSConfig()
    config = replace(
        base_config,
        data_flow=replace(base_config.data_flow, file_binding_list_limit=2),
    )
    runtime = Runtime.open(tmp_path / "tree-labels.sqlite", config=config)
    try:
        labels = DataLabels(sensitivity="confidential")
        paths = [
            "tree",
            "tree/a.txt",
            "tree/nested/b.txt",
            "tree/nested/c.txt",
            "tree/z.txt",
            "treehouse/outside.txt",
        ]
        for index, path in enumerate(paths):
            runtime.store.upsert_file_label_binding(
                FileLabelBinding(
                    binding_id=f"tree-binding-{index}",
                    normalized_path=path,
                    content_sha256=f"{index + 1:064x}",
                    labels=labels,
                    source_refs=(),
                    generation=1,
                    tombstoned=False,
                    active=True,
                    created_by="test",
                    created_at=utc_now(),
                )
            )

        original_query = runtime.store._query
        captured: list[tuple[str, tuple[object, ...]]] = []

        def tracked_query(sql: str, params: object = ()) -> list[object]:
            selected_params = tuple(params)  # type: ignore[arg-type]
            if "FROM file_label_bindings" in sql and "LIMIT" in sql:
                captured.append((sql, selected_params))
            return original_query(sql, selected_params)

        monkeypatch.setattr(runtime.store, "_query", tracked_query)

        found = runtime.store.list_file_label_bindings_for_tree("tree")

        assert [binding.normalized_path for binding in found] == paths[:-1]
        assert len(captured) >= 3
        plan = list(
            runtime.store.conn.execute(
                f"EXPLAIN QUERY PLAN {captured[0][0]}",
                captured[0][1],
            )
        )
        details = "\n".join(str(row[3]) for row in plan)
        assert "SEARCH file_label_bindings USING INDEX idx_file_label_tree_scan" in details
        assert "normalized_path>?" in details or "normalized_path>? AND normalized_path<?" in details
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_tree_file_label_prefix_is_backend_collation_safe(
    kind: str,
    tmp_path: Path,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        root = "树_%!"
        expected = [root, f"{root}/é.txt", f"{root}/子/ß.txt"]
        for index, path in enumerate(
            [*expected, f"{root}house/outside.txt", "树/outside.txt"]
        ):
            runtime.store.upsert_file_label_binding(
                FileLabelBinding(
                    binding_id=f"collation-tree-binding-{index}",
                    normalized_path=path,
                    content_sha256=f"{index + 1:064x}",
                    labels=DataLabels(sensitivity="confidential"),
                    source_refs=(),
                    generation=1,
                    tombstoned=False,
                    active=True,
                    created_by="test",
                    created_at=utc_now(),
                )
            )

        found = runtime.store.list_file_label_bindings_for_tree(root)

        assert [binding.normalized_path for binding in found] == expected


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_data_flow_evidence_and_label_bindings_persist_across_reopen(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        created_at = utc_now()
        source = DataSourceRef("persisted-source", 3, "a" * 64)
        labels = DataLabels(sensitivity="restricted", tenant="tenant-a")
        trust = SinkTrustSpec(
            trust_id="persisted-trust",
            pattern="filesystem:workspace:archive/*",
            trust_level="trusted",
            max_sensitivity="restricted",
            generation=1,
            created_by="test",
            created_at=created_at,
            tenants=("tenant-a",),
        )
        decision = DataFlowDecision(
            decision_id="persisted-decision",
            pid="persisted-pid",
            sink="filesystem:workspace:archive/result.txt",
            direction="egress",
            outcome="allow",
            reason="trusted_sink_clearance",
            labels=labels,
            source_refs=(source,),
            payload_hash="b" * 64,
            registry_generation=1,
            created_at=created_at,
            trust_id=trust.trust_id,
            trust_hash=trust.spec_hash,
        )
        binding = FileLabelBinding(
            binding_id="persisted-file-label",
            normalized_path="archive/result.txt",
            content_sha256="c" * 64,
            labels=labels,
            source_refs=(source,),
            generation=1,
            tombstoned=False,
            active=True,
            created_by="persisted-pid",
            created_at=created_at,
        )

        runtime = Runtime.open(target, config=config)
        try:
            runtime.store.register_sink_trust(trust)
            runtime.store.insert_data_flow_decision(decision)
            runtime.store.upsert_file_label_binding(binding)
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            assert reopened.store.get_sink_trust_generation() == 1
            assert reopened.store.get_sink_trust(trust.trust_id) == trust
            assert reopened.store.get_data_flow_decision(decision.decision_id) == decision
            assert reopened.store.get_file_label_binding(binding.normalized_path) == binding
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_llm_context_label_history_persists_and_merges_monotonically(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        store_closed = False
        try:
            first = runtime.store.merge_llm_context_label_history(
                "pid-context-labels",
                DataLabels(
                    sensitivity="restricted",
                    tenant="tenant-a",
                    principal="analyst-a",
                ),
            )
            merged = runtime.store.merge_llm_context_label_history(
                "pid-context-labels",
                DataLabels(
                    sensitivity="secret",
                    tenant="tenant-b",
                    principal="analyst-a",
                ),
            )

            assert first.sensitivity.value == "restricted"
            assert merged.sensitivity.value == "secret"
            assert merged.tenant == "mixed"
            assert merged.principal == "analyst-a"
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            restored = reopened.store.get_llm_context_label_history(
                "pid-context-labels"
            )
            assert restored == merged
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_runtime_store_event_limit_returns_newest_matching_events_in_order(kind: str, tmp_path: Path) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal="bounded event history")
        emitted = [
            runtime.events.emit(
                EventType.EXTERNAL_WRITE,
                source="event-limit-test",
                target=pid,
                payload={"index": index},
            )
            for index in range(5)
        ]
        runtime.events.emit(
            EventType.EXTERNAL_WRITE,
            source="event-limit-test",
            target="another-process",
            payload={"index": 99},
        )

        selected = runtime.events.list(target=pid, limit=2)
        previous = runtime.events.list(target=pid, limit=2, before_event_id=selected[0].event_id)
        following = runtime.events.list(target=pid, limit=2, after_event_id=emitted[1].event_id)

        assert [event.event_id for event in selected] == [emitted[-2].event_id, emitted[-1].event_id]
        assert [event.payload["index"] for event in selected] == [3, 4]
        assert [event.event_id for event in previous] == [emitted[-4].event_id, emitted[-3].event_id]
        assert [event.payload["index"] for event in previous] == [1, 2]
        assert [event.event_id for event in following] == [emitted[2].event_id, emitted[3].event_id]


@pytest.mark.parametrize("kind", STORE_BACKENDS)
def test_runtime_store_process_snapshot_queries_are_bounded_and_batched(kind: str, tmp_path: Path) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        parent = runtime.process.spawn(goal="snapshot parent")
        child = runtime.process.spawn_child(parent, goal="snapshot child")
        third = runtime.process.spawn(goal="snapshot third")
        terminal = runtime.process.get(third)
        runtime.process_transitions.transition(
            third,
            ProcessStatus.EXITED,
            expected_revision=terminal.revision,
            outcome=ExitedProcessOutcome(),
        )
        messages = [
            runtime.messages.post(sender="test", recipient_pid=child, body=f"message-{index}")
            for index in range(3)
        ]
        interrupt = runtime.messages.post(
            sender="test",
            recipient_pid=child,
            kind=ProcessMessageKind.INTERRUPT,
            body="interrupt",
        )
        runtime.ratings.upsert(child, score=4, comment="batch")
        for index in range(3):
            runtime.store.insert_llm_call(
                LLMCallRecord(
                    call_id=f"snapshot-call-{index}",
                    pid=child,
                    image_id="base-agent:v0",
                    purpose="snapshot",
                    status="ok",
                    usage={"total_tokens": 10 + index},
                    created_at=utc_now(),
                )
            )

        listed = runtime.store.list_processes(limit=2)
        active_first = runtime.store.list_processes(limit=2, active_first=True)
        ancestors = runtime.store.get_processes_with_ancestors([child])
        activity = runtime.store.get_process_activity_summaries(
            [parent, child, third],
            recent_message_limit=2,
            recent_llm_call_limit=2,
        )
        ratings = runtime.store.get_agent_ratings_for_processes(
            [child, third],
            rater=runtime.config.runtime.default_human,
        )

        assert len(listed) == 2
        assert third not in {process.pid for process in active_first}
        assert {process.pid for process in ancestors} == {parent, child}
        assert activity[child]["unread_message_count"] == 4
        assert activity[child]["interrupt_count"] == 1
        assert activity[child]["llm_call_count"] == 2
        assert activity[child]["token_total"] == 23
        assert activity[parent]["llm_call_count"] == 0
        assert activity[parent]["token_total"] == 0
        assert [message.message_id for message in activity[child]["messages"]] == [
            messages[-1].message_id,
            interrupt.message_id,
        ]
        assert activity[third]["messages"] == []
        assert ratings[child].score == 4
        assert third not in ratings


def test_sqlite_file_store_rejects_concurrent_active_runtime_and_reopens_after_close(tmp_path: Path) -> None:
    db_path = tmp_path / "leased.sqlite"
    runtime = Runtime.open(db_path)
    try:
        with pytest.raises(ValidationError, match="already open"):
            Runtime.open(db_path)
    finally:
        runtime.close()

    reopened = Runtime.open(db_path)
    try:
        assert reopened.store is not None
    finally:
        reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_object_payload_is_runtime_only_across_reopen(kind: str, tmp_path: Path) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(goal=f"{kind} payload reopen")
            handle = runtime.memory.create_object(
                pid=pid,
                object_type=ObjectType.ARTIFACT,
                payload={"secret": "runtime-only", "backend": kind},
                name="runtime.only.payload",
            )
            live_object = runtime.memory.get_object(pid, handle)
            assert live_object.payload == {"secret": "runtime-only", "backend": kind}
            persisted = runtime.uow.objects.get_persisted_object_state(handle.oid)
            assert persisted is not None
            assert persisted.oid == handle.oid
            assert persisted.lifecycle_state is ObjectLifecycleState.LIVE
            assert persisted.version == live_object.version
            assert persisted.payload_present is True
            assert persisted.recovered_after_reopen is False
            row = runtime.store.select_table_rows("objects", "oid = ?", (handle.oid,))[0]
            assert json.loads(row["payload_json"]) == {"storage": "runtime_memory", "present": True}
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            obj = reopened.store.get_object(handle.oid)
            assert obj is None
            row = reopened.store.select_table_rows("objects", "oid = ?", (handle.oid,))[0]
            assert row["lifecycle_state"] == "released"
            assert json.loads(row["payload_json"]) == {
                "storage": "runtime_memory",
                "present": False,
                "recovered_after_reopen": True,
            }
            assert reopened.store.is_recovered_object_payload(handle.oid)
            persisted = reopened.uow.objects.get_persisted_object_state(handle.oid)
            assert persisted is not None
            assert persisted.oid == handle.oid
            assert persisted.lifecycle_state is ObjectLifecycleState.RELEASED
            assert persisted.version == live_object.version
            assert persisted.payload_present is False
            assert persisted.recovered_after_reopen is True
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_prepared_protected_operation_restores_reservation_across_reopen(
    kind: str,
    tmp_path: Path,
) -> None:
    class Provider:
        def classify_external_effect(self, _operation, _context, _result):
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
            )

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(goal=f"{kind} prepared recovery")
            capability = runtime.capability.issue_trusted(
                pid,
                "test:prepared-recovery",
                [CapabilityRight.READ],
                issued_by="test",
                uses_remaining=1,
            )
            decision = runtime.capability.require(
                pid,
                capability.resource,
                CapabilityRight.READ,
                consume=False,
            )
            contract = ProtectedOperationContract(
                name="primitive.test.prepared_recovery",
                provider="test",
                operation="read",
                evidence_roles=("audit", "event", "effect"),
                resource_policy=ResourcePolicy.NONE,
                information_flow=True,
            )
            runtime.protected_operations.register_contract(contract)
            invocation = ProtectedOperationInvocation(
                pid=pid,
                actor=pid,
                target=capability.resource,
                decisions=(decision,),
                canonical_args={"item": "secret"},
                observation={"item_sha256": "safe"},
            )
            operation = runtime.protected_operations.start(
                contract,
                invocation,
                provider=Provider(),
            )
            operation.__enter__()
            effect_id = operation.effect_id
            assert effect_id is not None
            scope = operation._operation_cm
            assert scope is not None
            interrupted = RuntimeError("simulated runtime crash")
            scope.__exit__(type(interrupted), interrupted, interrupted.__traceback__)
            operation._operation_cm = None
            runtime.store.close()
            store_closed = True

            reopened = Runtime.open(target, config=config)
            try:
                restored = reopened.store.get_capability(capability.cap_id)
                assert restored is not None
                assert restored.uses_remaining == 1
                assert reopened.store.get_external_effect(effect_id) is None
            finally:
                reopened.close()
        finally:
            if not store_closed:
                runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_STORE_BACKENDS)
def test_prepared_recovery_handler_store_write_rolls_back_with_intent_and_reservation(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _runtime_for_backend(kind, tmp_path) as runtime:
        pid = runtime.process.spawn(goal=f"{kind} prepared recovery rollback")
        contract_name = "primitive.test.prepared_recovery_rollback"
        capability = runtime.capability.issue_trusted(
            pid,
            "test:prepared-recovery-rollback",
            [CapabilityRight.READ],
            issued_by="test",
            uses_remaining=1,
        )
        decision = runtime.capability.require(
            pid,
            capability.resource,
            CapabilityRight.READ,
            consume=False,
        )
        now = utc_now()
        repair_marker = HumanRequest(
            request_id=f"hreq-prepared-repair-{uuid4().hex}",
            pid=pid,
            human="owner",
            payload={"type": "approval", "question": "prepared repair marker"},
            status=HumanRequestStatus.PENDING,
            decision=None,
            blocking=False,
            created_at=now,
            updated_at=now,
        )
        runtime.store.insert_human_request(repair_marker)
        fail_recovery = [True]

        def recover(effect: ExternalEffectRecord) -> None:
            marker = runtime.store.get_human_request(repair_marker.request_id)
            assert marker is not None
            marker.status = HumanRequestStatus.DELIVERED
            marker.decision = {"prepared_repair_effect_id": effect.effect_id}
            marker.updated_at = utc_now()
            runtime.store.update_human_request(marker)
            if fail_recovery[0]:
                raise RuntimeError("prepared repair store write failed")

        runtime.protected_operations.register_prepared_recovery(
            "test_prepared_repair_rollback",
            recover,
        )
        contract = ProtectedOperationContract(
            name=contract_name,
            provider="test",
            operation="prepared_recovery_rollback",
            evidence_roles=("audit", "event", "effect"),
            resource_policy=ResourcePolicy.NONE,
            state_mutation=True,
            prepared_recovery="test_prepared_repair_rollback",
        )
        runtime.protected_operations.register_contract(contract)

        class Provider:
            def classify_external_effect(self, _operation, _context, _result):
                return ExternalEffectClassification(
                    rollback_class=(
                        ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
                    ),
                    rollback_status=(
                        ExternalEffectRollbackStatus.NOT_REQUIRED
                    ),
                    state_mutation=True,
                    information_flow=False,
                )

        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=capability.resource,
            decisions=(decision,),
            canonical_args={"item": "prepared repair"},
            observation={"item_sha256": "safe"},
        )
        operation = runtime.protected_operations.start(
            contract,
            invocation,
            provider=Provider(),
        )
        operation.__enter__()
        effect_id = operation.effect_id
        assert effect_id is not None
        reservation_id = operation._reservation_ids[0]
        scope = operation._operation_cm
        assert scope is not None
        interrupted = RuntimeError("simulated runtime crash")
        scope.__exit__(type(interrupted), interrupted, interrupted.__traceback__)
        operation._operation_cm = None
        monkeypatch.setattr(
            runtime.protected_operations,
            "_require_recovery_lease",
            lambda: None,
        )

        with pytest.raises(RuntimeError, match="prepared repair store write failed"):
            runtime.protected_operations.recover_prepared()

        rolled_back_marker = runtime.store.get_human_request(
            repair_marker.request_id
        )
        assert rolled_back_marker is not None
        assert rolled_back_marker.status == HumanRequestStatus.PENDING
        assert rolled_back_marker.decision is None
        retained_effect = runtime.store.get_external_effect(effect_id)
        assert retained_effect is not None
        assert retained_effect.transaction_state == "prepared"
        retained_reservation = runtime.store.get_capability_use_reservation(
            reservation_id
        )
        assert retained_reservation is not None
        assert retained_reservation["status"] == "reserved"
        retained_capability = runtime.store.get_capability(capability.cap_id)
        assert retained_capability is not None
        assert retained_capability.uses_remaining == 0

        fail_recovery[0] = False
        summary = runtime.protected_operations.recover_prepared()
        assert summary.total_count == 1
        assert summary.sample_effect_ids == (effect_id,)
        committed_marker = runtime.store.get_human_request(repair_marker.request_id)
        assert committed_marker is not None
        assert committed_marker.status == HumanRequestStatus.DELIVERED
        assert committed_marker.decision == {
            "prepared_repair_effect_id": effect_id,
        }
        assert runtime.store.get_external_effect(effect_id) is None
        restored_reservation = runtime.store.get_capability_use_reservation(
            reservation_id
        )
        assert restored_reservation is not None
        assert restored_reservation["status"] == "restored"
        restored_capability = runtime.store.get_capability(capability.cap_id)
        assert restored_capability is not None
        assert restored_capability.uses_remaining == 1
