from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from agent_libos.models import (
    CapabilityRight,
    DataFlowContext,
    DataLabels,
    EventType,
    ObjectMetadata,
    ObjectType,
    SinkTrustLevel,
    SinkTrustRule,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    ValidationError,
)
from agent_libos.llm.actions import _model_facing_error
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.substrate import CommandMetrics, SubprocessTimeoutExpired
from agent_libos.tools.base import (
    BaseAgentTool,
    SyncAgentTool,
    ToolContext,
    ToolExecutionError,
    ToolPolicy,
)
from agent_libos.utils.serde import dumps
from tests.support.fakes import RecordingActionClient
from tests.support.runtime import workspace_runtime


class _NoArgs(BaseModel):
    pass


class _FailAfterSecretReadTool(SyncAgentTool[_NoArgs]):
    name = "fail_after_secret_read"
    description = "Read a labeled Object, then return a derived failure."
    args_schema = _NoArgs

    def run(self, args: _NoArgs, ctx: ToolContext) -> None:
        runtime = ctx.runtime
        source = runtime.store.get_object_by_name(
            "hidden-secret",
            namespace=runtime.memory.resolve_namespace(ctx.pid),
        )
        assert source is not None
        runtime.data_flow.observe_ingress(
            runtime.data_flow.context_from_trusted_source_oids([source.oid])
        )
        raise ToolExecutionError(f"derived failure: {source.payload['value']}")


class _DomainFailAfterSecretReadTool(SyncAgentTool[_NoArgs]):
    description = "Raise a domain exception after observing a labeled Object."
    args_schema = _NoArgs

    def __init__(self, error_type: type[Exception]) -> None:
        self._error_type = error_type
        self.name = f"{error_type.__name__.lower()}_after_secret_read"

    def run(self, args: _NoArgs, ctx: ToolContext) -> None:
        runtime = ctx.runtime
        source = runtime.store.get_object_by_name(
            "hidden-secret",
            namespace=runtime.memory.resolve_namespace(ctx.pid),
        )
        assert source is not None
        runtime.data_flow.observe_ingress(
            runtime.data_flow.context_from_trusted_source_oids([source.oid])
        )
        raise self._error_type(
            f"provider/tool diagnostic contains {source.payload['value']}"
        )


class _AsyncFailAfterSecretReadTool(BaseAgentTool[_NoArgs]):
    name = "async_fail_after_secret_read"
    description = "Read a labeled Object in an async task, then fail."
    args_schema = _NoArgs

    async def execute(self, args: _NoArgs, ctx: ToolContext) -> None:
        runtime = ctx.runtime
        source = runtime.store.get_object_by_name(
            "hidden-secret",
            namespace=runtime.memory.resolve_namespace(ctx.pid),
        )
        assert source is not None
        runtime.data_flow.observe_ingress(
            runtime.data_flow.context_from_trusted_source_oids([source.oid])
        )
        raise ToolExecutionError(f"async derived failure: {source.payload['value']}")


class _AsyncTimeoutAfterSecretReadTool(BaseAgentTool[_NoArgs]):
    name = "async_timeout_after_secret_read"
    description = "Read a labeled Object in an async task, then time out."
    args_schema = _NoArgs
    policy = ToolPolicy(timeout_s=0.01)

    async def execute(self, args: _NoArgs, ctx: ToolContext) -> None:
        runtime = ctx.runtime
        source = runtime.store.get_object_by_name(
            "hidden-secret",
            namespace=runtime.memory.resolve_namespace(ctx.pid),
        )
        assert source is not None
        runtime.data_flow.observe_ingress(
            runtime.data_flow.context_from_trusted_source_oids([source.oid])
        )
        await asyncio.Event().wait()


class _StrictCountOutput(BaseModel):
    count: int


class _InvalidOutputAfterSecretReadTool(SyncAgentTool[_NoArgs]):
    name = "invalid_output_after_secret_read"
    description = "Read a labeled Object, then return schema-invalid output."
    args_schema = _NoArgs
    output_schema = _StrictCountOutput

    def run(self, args: _NoArgs, ctx: ToolContext) -> dict[str, str]:
        runtime = ctx.runtime
        source = runtime.store.get_object_by_name(
            "hidden-secret",
            namespace=runtime.memory.resolve_namespace(ctx.pid),
        )
        assert source is not None
        runtime.data_flow.observe_ingress(
            runtime.data_flow.context_from_trusted_source_oids([source.oid])
        )
        return {"count": source.payload["value"]}


class _WaitAfterSecretReadTool(BaseAgentTool[_NoArgs]):
    description = "Observe a labeled Object, then exercise one supported wait family."
    args_schema = _NoArgs

    def __init__(self, wait_kind: str, *, child_pid: str | None = None) -> None:
        self.wait_kind = wait_kind
        self.child_pid = child_pid
        self.name = f"wait_after_secret_read_{wait_kind}"

    async def execute(self, args: _NoArgs, ctx: ToolContext) -> dict[str, str]:
        runtime = ctx.runtime
        source = runtime.store.get_object_by_name(
            "hidden-secret",
            namespace=runtime.memory.resolve_namespace(ctx.pid),
        )
        assert source is not None
        if self.wait_kind == "human":
            request_id = runtime.human.ask(
                ctx.pid,
                "Flow-context wait probe",
                blocking=True,
            )
            runtime.data_flow.observe_ingress(
                runtime.data_flow.context_from_trusted_source_oids([source.oid])
            )
            raise HumanApprovalRequired(request_id, "flow wait")

        runtime.data_flow.observe_ingress(
            runtime.data_flow.context_from_trusted_source_oids([source.oid])
        )
        if self.wait_kind == "child":
            assert self.child_pid is not None
            runtime.process.wait(ctx.pid, self.child_pid, timeout=None)
            raise AssertionError("running child wait unexpectedly returned")
        if self.wait_kind == "message":
            runtime.messages.receive(ctx.pid, block=True, channel="flow-probe")
            raise AssertionError("empty message wait unexpectedly returned")
        return {"value": source.payload["value"]}


def _secret_object(runtime, pid: str, *, name: str = "hidden-secret"):
    return runtime.memory.create_object(
        pid,
        ObjectType.EVIDENCE,
        {"value": "DATA_FLOW_SECRET_SENTINEL"},
        metadata=ObjectMetadata(sensitivity="secret"),
        name=name,
    )


def _tenant_manifest(allowed_tenants: list[str]) -> dict[str, object]:
    return {
        "data_flow_policy": {
            "schema_version": 1,
            "allowed_tenants": allowed_tenants,
            "allowed_principals": [],
        }
    }


def _install_mocked_jit_runner(
    runtime,
    monkeypatch: pytest.MonkeyPatch,
    runner,
) -> None:
    sandbox = runtime.tools.sandbox

    def validate_statically(
        source_code: str,
        tests: list[dict[str, object]],
        *_args: object,
        **_kwargs: object,
    ):
        assert tests == []
        return sandbox.static_check(source_code)

    monkeypatch.setattr(sandbox, "run_tests", validate_statically)
    monkeypatch.setattr(sandbox, "arun_source", runner)


def _bind_ambient_secret_file(
    runtime,
    root,
    pid: str,
    *,
    relative: str = "ambient-tenant-secret.txt",
) -> tuple[str, str]:
    content = "AMBIENT_TENANT_SECRET_SENTINEL"
    (root / relative).write_text(content, encoding="utf-8")
    runtime.data_flow.bind_written_file(
        pid=pid,
        normalized_path=relative,
        content=content.encode("utf-8"),
        context=DataFlowContext(
            labels=DataLabels(sensitivity="secret", tenant="tenant-a")
        ),
    )
    runtime.filesystem.grant_path(
        pid,
        relative,
        [CapabilityRight.READ],
        issued_by="test.host",
    )
    return relative, content


def test_memory_read_tool_taints_its_result_even_when_source_is_not_in_view() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="read labeled Object")
        source = _secret_object(runtime, pid)
        runtime.tools.configure_process_tools(
            pid,
            ["read_memory_object"],
            assigned_by="test.host",
        )
        process = runtime.process.get(pid)
        assert source.oid not in {
            handle.oid for handle in (process.memory_view.roots if process.memory_view else [])
        }

        result = runtime.tools.call(
            pid,
            "read_memory_object",
            {"name": "hidden-secret"},
        )

        assert result.ok, result.error
        assert result.result_handle is not None
        stored = runtime.store.get_object(result.result_handle.oid)
        assert stored is not None
        assert stored.metadata.sensitivity == "secret"
        assert source.oid in stored.provenance.parent_oids


def test_sync_message_tool_preserves_explicit_carrier_labels_for_later_egress() -> None:
    with workspace_runtime() as (runtime, root):
        parent = runtime.process.spawn(goal="send a labeled process message")
        runtime.capability.grant(
            parent,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        child = runtime.spawn_child_process(parent, "read the labeled process message")
        secret = _secret_object(runtime, parent, name="message-secret")
        runtime.messages.send_from_process(
            parent,
            child,
            body="MESSAGE_SECRET_SENTINEL",
            source_oids=[secret.oid],
        )
        runtime.tools.configure_process_tools(
            child,
            ["read_process_messages"],
            assigned_by="test.host",
        )

        result = runtime.tools.call(child, "read_process_messages", {})

        assert result.ok, result.error
        assert result.result_handle is not None
        stored = runtime.store.get_object(result.result_handle.oid)
        assert stored is not None
        assert stored.metadata.sensitivity == "secret"
        output = "message-export.txt"
        runtime.filesystem.grant_path(
            child,
            output,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
            runtime.filesystem.write_text(
                child,
                output,
                "MESSAGE_SECRET_SENTINEL",
                source_oids=[result.result_handle.oid],
            )
        assert not (root / output).exists()


def test_memory_metadata_tools_taint_create_list_and_append_results() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="label memory metadata results")
        runtime.tools.configure_process_tools(
            pid,
            [
                "create_memory_object",
                "list_memory_namespace",
                "append_memory_object",
            ],
            assigned_by="test.host",
        )
        empty_namespace = runtime.memory.create_namespace(pid, "empty-metadata-results")
        empty_result = runtime.tools.call(
            pid,
            "list_memory_namespace",
            {"namespace": empty_namespace.namespace},
        )
        assert empty_result.ok and empty_result.payload["objects"] == []
        assert empty_result.result_handle is not None
        empty_stored = runtime.store.get_object(empty_result.result_handle.oid)
        assert empty_stored is not None
        assert empty_stored.metadata.sensitivity == "normal"

        created_result = runtime.tools.call(
            pid,
            "create_memory_object",
            {
                "name": "secret-created-name",
                "type": "observation",
                "payload": {"ok": True},
                "metadata": {"sensitivity": "secret"},
            },
        )
        assert created_result.ok
        created_object = runtime.store.get_object(created_result.payload["oid"])
        assert created_object is not None
        assert created_object.metadata.sensitivity == "secret"

        listed_result = runtime.tools.call(pid, "list_memory_namespace", {})
        assert listed_result.ok
        assert "secret-created-name" in {
            item["name"] for item in listed_result.payload["objects"]
        }
        listed_oids = {item["oid"] for item in listed_result.payload["objects"]}

        append_target = runtime.memory.create_object(
            pid,
            ObjectType.OBSERVATION,
            {"entries": []},
            metadata=ObjectMetadata(sensitivity="secret"),
            name="secret-append-target",
            immutable=False,
        )
        appended_result = runtime.tools.call(
            pid,
            "append_memory_object",
            {"name": "secret-append-target", "entry": {"public": True}},
        )
        assert appended_result.ok and appended_result.payload["length"] == 1

        for kind, result, derived_value, expected_sources in (
            (
                "create",
                created_result,
                created_result.payload["name"],
                {created_object.oid},
            ),
            ("list", listed_result, "secret-created-name", listed_oids),
            (
                "append",
                appended_result,
                str(appended_result.payload["length"]),
                {append_target.oid},
            ),
        ):
            assert result.result_handle is not None
            stored = runtime.store.get_object(result.result_handle.oid)
            assert stored is not None and stored.metadata.sensitivity == "secret", kind
            assert expected_sources <= set(stored.provenance.parent_oids)
            path = f"{kind}-metadata-leak.txt"
            runtime.filesystem.grant_path(
                pid,
                path,
                [CapabilityRight.WRITE],
                issued_by="test.host",
            )
            with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
                runtime.filesystem.write_text(
                    pid,
                    path,
                    derived_value,
                    source_oids=[result.result_handle.oid],
                )
            assert not (root / path).exists()


def test_jit_namespace_list_taints_secret_object_name_metadata() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="JIT list metadata labels")
        _secret_object(runtime, pid, name="secret-listed-name")
        runtime.filesystem.grant_path(
            pid,
            "jit-list-metadata-leak.txt",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        async def exercise() -> None:
            session = LibOSSyscallSession(runtime, pid)
            listing = await session.handle("memory.list_namespace", {})
            derived_name = next(
                item["name"]
                for item in listing["objects"]
                if item["name"] == "secret-listed-name"
            )
            with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
                await session.handle(
                    "filesystem.write_text",
                    {"path": "jit-list-metadata-leak.txt", "text": derived_name},
                )

        asyncio.run(exercise())
        assert not (root / "jit-list-metadata-leak.txt").exists()


def test_jit_append_taints_secret_target_length_metadata() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="JIT append metadata labels")
        runtime.memory.create_object(
            pid,
            ObjectType.OBSERVATION,
            {"entries": []},
            metadata=ObjectMetadata(sensitivity="secret"),
            name="secret-jit-append-target",
            immutable=False,
        )
        runtime.filesystem.grant_path(
            pid,
            "jit-append-metadata-leak.txt",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        async def exercise() -> None:
            session = LibOSSyscallSession(runtime, pid)
            appended = await session.handle(
                "memory.append_object",
                {"name": "secret-jit-append-target", "entry": {"public": True}},
            )
            assert appended["length"] == 1
            with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
                await session.handle(
                    "filesystem.write_text",
                    {
                        "path": "jit-append-metadata-leak.txt",
                        "text": str(appended["length"]),
                    },
                )

        asyncio.run(exercise())
        assert not (root / "jit-append-metadata-leak.txt").exists()


def test_jit_empty_namespace_list_keeps_normal_flow() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="JIT empty list labels")
        namespace = runtime.memory.create_namespace(pid, "jit-empty-list")
        runtime.filesystem.grant_path(
            pid,
            "jit-empty-list.txt",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        async def exercise() -> None:
            session = LibOSSyscallSession(runtime, pid)
            listing = await session.handle(
                "memory.list_namespace",
                {"namespace": namespace.namespace},
            )
            assert listing["objects"] == []
            written = await session.handle(
                "filesystem.write_text",
                {"path": "jit-empty-list.txt", "text": "0"},
            )
            assert written["bytes_written"] == 1

        asyncio.run(exercise())
        assert (root / "jit-empty-list.txt").read_text(encoding="utf-8") == "0"


def test_sync_tool_failure_keeps_worker_labels_and_cannot_be_reexported() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="labeled tool failure")
        source = _secret_object(runtime, pid)
        handle = runtime.tools.register_tool(
            _FailAfterSecretReadTool(),
            registered_by="test.host",
            ephemeral=True,
        )
        runtime.tools.configure_process_tools(pid, [handle], assigned_by="test.host")

        result = runtime.tools.call(pid, handle, {})

        assert not result.ok
        assert (result.error or "").startswith("execution_error: ToolExecutionError")
        assert "DATA_FLOW_SECRET_SENTINEL" not in (result.error or "")
        assert result.result_handle is not None
        stored = runtime.store.get_object(result.result_handle.oid)
        assert stored is not None
        assert stored.metadata.sensitivity == "secret"
        assert source.oid in stored.provenance.parent_oids

        runtime.filesystem.grant_path(
            pid,
            "failure-leak.txt",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
            runtime.filesystem.write_text(
                pid,
                "failure-leak.txt",
                result.error,
                source_oids=[result.result_handle.oid],
            )
        assert not (root / "failure-leak.txt").exists()


@pytest.mark.parametrize("error_type", [ValidationError, CapabilityDenied])
def test_post_dispatch_domain_exception_text_never_reaches_any_tool_sink(
    error_type: type[Exception],
) -> None:
    secret = "DATA_FLOW_SECRET_SENTINEL"
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="opaque tool failure")
        source = _secret_object(runtime, pid)
        handle = runtime.tools.register_tool(
            _DomainFailAfterSecretReadTool(error_type),
            registered_by="test.host",
            ephemeral=True,
        )
        runtime.tools.configure_process_tools(pid, [handle], assigned_by="test.host")

        result = runtime.tools.call(pid, handle, {})

        assert not result.ok
        assert result.result_handle is not None
        durable = runtime.store.get_object(result.result_handle.oid)
        assert durable is not None
        failed_events = [
            event
            for event in runtime.events.list(target=pid)
            if event.payload.get("call_id") == result.call_id
            and event.type is EventType.TOOL_FAILED
        ]
        failed_audit = [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "tool.call"
            and record.decision.get("tool") == handle.name
            and record.decision.get("ok") is False
        ]
        assert len(failed_events) == 1
        assert len(failed_audit) == 1
        action_feedback = _model_facing_error(result)
        failure = durable.payload["failure"]
        correlation_id = failure["error"]["correlation_id"]
        assert correlation_id.startswith("corr_")
        assert failed_audit[0].correlation_id == correlation_id
        for observation in (
            failure["internal_error"],
            failed_events[0].payload["error"],
            failed_audit[0].decision["error"],
        ):
            assert observation["correlation_id"] == correlation_id
            assert observation["error_type"] == error_type.__name__
            assert observation["exception_text"]["bytes"] > 0
            assert len(observation["exception_text"]["sha256"]) == 64
        serialized = dumps(
            {
                "payload": result.payload,
                "error": result.error,
                "action_feedback": action_feedback,
                "durable": durable.payload,
                "events": [event.payload for event in failed_events],
                "audit": [record.decision for record in failed_audit],
            }
        )
        assert action_feedback == result.error
        assert source.oid in durable.provenance.parent_oids
        assert secret not in serialized
        assert "provider/tool diagnostic contains" not in serialized


def test_async_tool_failure_keeps_child_task_labels() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="async labeled failure")
        source = _secret_object(runtime, pid)
        handle = runtime.tools.register_tool(
            _AsyncFailAfterSecretReadTool(),
            registered_by="test.host",
            ephemeral=True,
        )
        runtime.tools.configure_process_tools(pid, [handle], assigned_by="test.host")

        result = runtime.tools.call(pid, handle, {})

        assert not result.ok
        assert (result.error or "").startswith("execution_error: ToolExecutionError")
        assert "DATA_FLOW_SECRET_SENTINEL" not in (result.error or "")
        assert result.result_handle is not None
        stored = runtime.store.get_object(result.result_handle.oid)
        assert stored is not None
        assert stored.metadata.sensitivity == "secret"
        assert source.oid in stored.provenance.parent_oids


def test_async_tool_timeout_keeps_cancelled_child_task_labels() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="async labeled timeout")
        source = _secret_object(runtime, pid)
        handle = runtime.tools.register_tool(
            _AsyncTimeoutAfterSecretReadTool(),
            registered_by="test.host",
            ephemeral=True,
        )
        runtime.tools.configure_process_tools(pid, [handle], assigned_by="test.host")

        result = runtime.tools.call(pid, handle, {})

        assert not result.ok
        assert (result.error or "").startswith("timeout: TimeoutError")
        assert result.result_handle is not None
        stored = runtime.store.get_object(result.result_handle.oid)
        assert stored is not None
        assert stored.metadata.sensitivity == "secret"
        assert source.oid in stored.provenance.parent_oids


def test_sync_tool_output_validation_failure_keeps_worker_labels() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="label schema-invalid tool output",
        )
        source = _secret_object(runtime, pid)
        handle = runtime.tools.register_tool(
            _InvalidOutputAfterSecretReadTool(),
            registered_by="test.host",
            ephemeral=True,
        )
        runtime.tools.configure_process_tools(pid, [handle], assigned_by="test.host")

        result = runtime.tools.call(pid, handle, {})

        assert not result.ok
        assert result.result_handle is not None
        assert "DATA_FLOW_SECRET_SENTINEL" not in str(result.payload)
        stored = runtime.store.get_object(result.result_handle.oid)
        assert stored is not None
        assert stored.metadata.sensitivity == "secret"
        assert source.oid in stored.provenance.parent_oids


@pytest.mark.parametrize(
    ("primitive_name", "async_name", "sync_name", "call_args"),
    [
        ("jsonrpc", "acall", "call", ("endpoint", "method", {})),
        ("mcp", "acall_tool", "call_tool", ("server", "tool", {})),
        ("shell", "arun", "run", (["git", "status"],)),
    ],
    ids=("jsonrpc", "mcp", "shell"),
)
@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_async_primitive_wrappers_merge_worker_data_flow_context(
    primitive_name: str,
    async_name: str,
    sync_name: str,
    call_args: tuple[object, ...],
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(goal=f"async {primitive_name} ingress")
        source = _secret_object(runtime, pid)
        source_context = runtime.data_flow.context_from_trusted_source_oids(
            [source.oid]
        )
        primitive = getattr(runtime, primitive_name)
        marker = object()

        def observed_call(*_args: object, **_kwargs: object) -> object:
            runtime.data_flow.observe_ingress(source_context)
            if outcome == "failure":
                raise RuntimeError("worker failed after labeled ingress")
            return marker

        monkeypatch.setattr(primitive, sync_name, observed_call)

        async def exercise() -> None:
            with runtime.data_flow.activate(DataFlowContext()):
                invocation = getattr(primitive, async_name)(pid, *call_args)
                if outcome == "failure":
                    with pytest.raises(
                        RuntimeError,
                        match="worker failed after labeled ingress",
                    ):
                        await invocation
                else:
                    assert await invocation is marker
                returned = runtime.data_flow.current_context()
                assert returned.labels.sensitivity.value == "secret"
                assert source.oid in {item.oid for item in returned.source_refs}

        asyncio.run(exercise())


@pytest.mark.parametrize("failure_kind", ["exception", "timeout"])
def test_jit_failure_paths_persist_labeled_result_carrier(
    failure_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(
            image="toolmaker-agent:v0",
            goal=f"JIT labeled {failure_kind}",
        )
        source = _secret_object(runtime, pid)
        source_context = runtime.data_flow.context_from_trusted_source_oids(
            [source.oid]
        )

        async def fail_after_ingress(*_args: object, **_kwargs: object) -> None:
            runtime.data_flow.observe_ingress(source_context)
            if failure_kind == "timeout":
                raise SubprocessTimeoutExpired(
                    "JIT timed out after labeled ingress",
                    metrics=CommandMetrics(
                        wall_seconds=1.0,
                        killed=True,
                        limit_kind="wall_time",
                    ),
                )
            raise RuntimeError("JIT failed after DATA_FLOW_SECRET_SENTINEL")

        _install_mocked_jit_runner(runtime, monkeypatch, fail_after_ingress)
        candidate = runtime.tools.propose(
            pid,
            {
                "name": f"jit_labeled_{failure_kind}",
                "description": "Exercise labeled JIT failure persistence.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            source_code="export function run(args, libos) { return {}; }",
        )
        runtime.tools.register(pid, candidate)

        result = runtime.tools.call(pid, f"jit_labeled_{failure_kind}", {})

        assert not result.ok
        assert result.result_handle is not None
        stored = runtime.store.get_object(result.result_handle.oid)
        assert stored is not None
        assert stored.metadata.sensitivity == "secret"
        assert source.oid in stored.provenance.parent_oids


@pytest.mark.parametrize("wait_kind", ["human", "child", "message", "success"])
def test_tool_waits_preserve_observed_context_for_durable_resume(
    wait_kind: str,
) -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal=f"preserve {wait_kind} wait flow context",
        )
        source = _secret_object(runtime, pid)
        child_pid = (
            runtime.process.spawn_child(pid, "flow wait child")
            if wait_kind == "child"
            else None
        )
        if wait_kind == "human":
            runtime.capability.grant(
                pid,
                f"human:{runtime.config.runtime.default_human}",
                [CapabilityRight.WRITE],
                issued_by="test.host",
            )
        tool = _WaitAfterSecretReadTool(wait_kind, child_pid=child_pid)
        handle = runtime.tools.register_tool(
            tool,
            registered_by="test.host",
            ephemeral=True,
        )
        runtime.tools.configure_process_tools(
            pid,
            [handle],
            assigned_by="test.host",
        )
        # This test isolates labels observed inside the tool. A secret Object
        # creation event is itself labeled and would correctly block the
        # preceding normal-trust LLM turn, so acknowledge setup events first.
        process = runtime.process.get(pid)
        process.event_cursor = runtime.store.list_events(target=pid)[-1].event_id
        runtime.store.update_process(process)
        runtime.llm.client = RecordingActionClient([{"action": tool.name}])

        result = runtime.run_process_once(pid)

        caller_context = runtime.data_flow.current_context()
        assert caller_context.labels.sensitivity.value == "normal"
        assert caller_context.source_refs == ()
        if wait_kind == "success":
            assert result["ok"]
            assert runtime.store.get_llm_pending_action(pid) is None
            result_oid = result["result"]["result_oid"]
            stored_result = runtime.store.get_object(result_oid)
            assert stored_result is not None
            assert stored_result.metadata.sensitivity == "secret"
            assert source.oid in stored_result.provenance.parent_oids
            return

        expected_flag = {
            "human": "waiting_human",
            "child": "waiting_event",
            "message": "waiting_message",
        }[wait_kind]
        assert result[expected_flag]
        pending = runtime.store.get_llm_pending_action(pid)
        assert pending is not None
        pending_context = DataFlowContext.from_dict(pending["data_flow_context"])
        assert pending_context.labels.sensitivity.value == "secret"
        assert source.oid in {item.oid for item in pending_context.source_refs}
        wait_audits = [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action.startswith("llm.action_waiting_")
        ]
        assert wait_audits
        visible_message = str(wait_audits[-1].decision.get("message") or "")
        assert source.oid not in visible_message
        assert "data_flow_context" not in visible_message


def test_jit_deferred_lifecycle_failure_persists_labeled_failure_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "JIT_DEFERRED_SECRET_SENTINEL"
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(
            image="toolmaker-agent:v0",
            goal="preserve deferred JIT failure labels",
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"value": sentinel},
            metadata=ObjectMetadata(sensitivity="secret"),
            name="hidden-secret",
        )

        async def defer_secret_derived_exec(
            _source: str,
            _args: dict[str, object],
            **kwargs: object,
        ) -> dict[str, object]:
            syscall_handler = kwargs["syscall_handler"]

            async def serve() -> dict[str, object]:
                read = await syscall_handler(
                    "memory.read_object",
                    {"name": "hidden-secret"},
                )
                value = read["payload"]["value"]
                await syscall_handler(
                    "process.exec",
                    {
                        "image": value,
                        "goal": value,
                        "preserve_memory": True,
                    },
                )
                return {"observed": value}

            return await asyncio.create_task(serve())

        _install_mocked_jit_runner(runtime, monkeypatch, defer_secret_derived_exec)
        candidate = runtime.tools.propose(
            pid,
            {
                "name": "jit_labeled_deferred_failure",
                "description": "Exercise labeled deferred lifecycle failure.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            source_code="export function run(args, libos) { return {}; }",
        )
        runtime.tools.register(pid, candidate)
        audit_start = len(runtime.audit.trace())
        event_start = len(runtime.events.list())

        result = runtime.tools.call(pid, "jit_labeled_deferred_failure", {})

        assert not result.ok
        assert result.result_handle is not None
        assert (result.error or "").startswith(
            "lifecycle_error: CapabilityDenied"
        )
        assert sentinel not in (result.error or "")
        assert sentinel not in str(result.payload)
        assert result.payload["policy_decision"] == "lifecycle_error"
        assert result.payload["error"]["type"] == "CapabilityDenied"
        assert result.payload["error"]["retryable"] is False
        stored_failure = runtime.store.get_object(result.result_handle.oid)
        assert stored_failure is not None
        assert stored_failure.type == ObjectType.TOOL_RESULT
        assert stored_failure.metadata.sensitivity == "secret"
        assert source.oid in stored_failure.provenance.parent_oids
        assert sentinel not in str(stored_failure.payload)
        failure = stored_failure.payload["failure"]
        assert stored_failure.payload["ok"] is False
        assert failure["ok"] is False
        assert failure["policy_decision"] == "lifecycle_error"
        assert failure["error"]["type"] == "CapabilityDenied"
        internal_error = failure["internal_error"]
        assert internal_error["error_type"] == "CapabilityDenied"
        assert internal_error["exception_text"]["bytes"] > 0
        assert len(internal_error["exception_text"]["sha256"]) == 64
        assert runtime.process.get(pid).image_id == "toolmaker-agent:v0"
        retained_tool_results = [
            obj
            for obj in runtime.store.list_objects()
            if obj.type == ObjectType.TOOL_RESULT
        ]
        assert [obj.oid for obj in retained_tool_results] == [result.result_handle.oid]

        new_audits = runtime.audit.trace()[audit_start:]
        discarded_success_results = [
            record
            for record in new_audits
            if record.action == "memory.delete_object"
            and record.decision.get("reason") == "tool_result.scope_discard"
        ]
        assert discarded_success_results
        tool_audits = [
            record
            for record in new_audits
            if record.action == "tool.call"
            and record.decision.get("tool") == "jit_labeled_deferred_failure"
        ]
        assert tool_audits[-1].output_refs == [result.result_handle.oid]
        assert tool_audits[-1].decision["error"]["error_type"] == "CapabilityDenied"
        assert sentinel not in str(tool_audits[-1].decision)
        failure_events = [
            event
            for event in runtime.events.list()[event_start:]
            if event.type == EventType.TOOL_FAILED
            and event.payload.get("call_id") == result.call_id
        ]
        assert failure_events[-1].payload["result_oid"] == result.result_handle.oid
        assert failure_events[-1].payload["error"]["error_type"] == "CapabilityDenied"
        assert sentinel not in str(failure_events[-1].payload)
        assert not any(
            record.action == "process.exec" and record.actor == pid
            for record in new_audits
        )
        caller_context = runtime.data_flow.current_context()
        assert caller_context.labels.sensitivity.value == "normal"
        assert caller_context.source_refs == ()


def test_jit_object_read_taints_later_filesystem_egress() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="JIT labeled read")
        _secret_object(runtime, pid)
        runtime.filesystem.grant_path(
            pid,
            "leak.txt",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        async def exercise() -> None:
            session = LibOSSyscallSession(runtime, pid)
            read = await session.handle(
                "memory.read_object",
                {"name": "hidden-secret"},
            )
            assert read["payload"]["value"] == "DATA_FLOW_SECRET_SENTINEL"
            with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
                await session.handle(
                    "filesystem.write_text",
                    {"path": "leak.txt", "text": "DATA_FLOW_SECRET_SENTINEL"},
                )

        asyncio.run(exercise())

        assert not (root / "leak.txt").exists()
        denied = runtime.store.list_data_flow_decisions(pid=pid, outcome="deny")
        assert denied and denied[-1].labels.sensitivity.value == "secret"


def test_jit_create_and_append_inherit_all_observed_object_labels() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="derive labeled Objects")
        source = _secret_object(runtime, pid)
        runtime.memory.create_object(
            pid,
            ObjectType.OBSERVATION,
            {"entries": []},
            name="scratch",
            immutable=False,
        )

        async def exercise() -> tuple[str, str]:
            session = LibOSSyscallSession(runtime, pid)
            await session.handle("memory.read_object", {"name": "hidden-secret"})
            created = await session.handle(
                "memory.create_object",
                {
                    "name": "derived-copy",
                    "type": "observation",
                    "payload": {"value": "DATA_FLOW_SECRET_SENTINEL"},
                },
            )
            appended = await session.handle(
                "memory.append_object",
                {
                    "name": "scratch",
                    "entry": {"value": "DATA_FLOW_SECRET_SENTINEL"},
                },
            )
            return created["oid"], appended["oid"]

        created_oid, appended_oid = asyncio.run(exercise())
        created = runtime.store.get_object(created_oid)
        appended = runtime.store.get_object(appended_oid)
        assert created is not None and created.metadata.sensitivity == "secret"
        assert appended is not None and appended.metadata.sensitivity == "secret"
        assert source.oid in created.provenance.parent_oids
        assert source.oid in appended.provenance.parent_oids


def test_jit_derivations_inherit_labels_without_object_source_refs() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="derive path-only labels")
        relative = "host-classified.txt"
        content = "PATH_ONLY_SECRET_SENTINEL"
        (root / relative).write_text(content, encoding="utf-8")
        binding = runtime.data_flow.bind_written_file(
            pid=pid,
            normalized_path=relative,
            content=content.encode("utf-8"),
            context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
        )
        runtime.filesystem.grant_path(
            pid,
            relative,
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        runtime.memory.create_object(
            pid,
            ObjectType.OBSERVATION,
            {"entries": []},
            name="path-scratch",
            immutable=False,
        )

        async def exercise() -> tuple[str, str]:
            session = LibOSSyscallSession(runtime, pid)
            await session.handle("filesystem.read_text", {"path": relative})
            created = await session.handle(
                "memory.create_object",
                {
                    "name": "path-derived",
                    "type": "observation",
                    "payload": {"value": content},
                },
            )
            appended = await session.handle(
                "memory.append_object",
                {"name": "path-scratch", "entry": {"value": content}},
            )
            return created["oid"], appended["oid"]

        created_oid, appended_oid = asyncio.run(exercise())
        created = runtime.store.get_object(created_oid)
        appended = runtime.store.get_object(appended_oid)
        assert created is not None and created.metadata.sensitivity == "secret"
        assert appended is not None and appended.metadata.sensitivity == "secret"
        durable_ref = (
            f"{runtime.data_flow.FILE_BINDING_SOURCE_REF_PREFIX}{binding.binding_id}"
        )
        assert durable_ref in created.provenance.source_refs
        assert durable_ref in appended.provenance.source_refs


def test_jit_send_message_preserves_ambient_labels_and_denies_restricted_recipient() -> None:
    with workspace_runtime() as (runtime, root):
        parent = runtime.process.spawn(
            image="base-agent:v0",
            goal="send ambient tenant data",
            authority_manifest=_tenant_manifest(["tenant-a"]),
        )
        allowed = runtime.process.spawn_child(
            parent,
            "allowed receiver",
            authority_manifest=_tenant_manifest(["tenant-a"]),
        )
        restricted = runtime.process.spawn_child(
            parent,
            "restricted receiver",
            authority_manifest=_tenant_manifest([]),
        )
        relative, content = _bind_ambient_secret_file(runtime, root, parent)

        async def exercise() -> dict[str, object]:
            session = LibOSSyscallSession(runtime, parent)
            read = await session.handle("filesystem.read_text", {"path": relative})
            assert read["content"] == content
            flow = runtime.data_flow.current_context()
            assert flow.labels.sensitivity.value == "secret"
            assert flow.labels.tenant == "tenant-a"
            assert len(flow.source_refs) == 1
            assert flow.source_refs[0].oid.startswith(
                runtime.data_flow.FILE_BINDING_SOURCE_REF_PREFIX
            )

            sent = await session.handle(
                "process.send_message",
                {"recipient_pid": allowed, "body": content},
            )
            with pytest.raises(CapabilityDenied, match="data_flow_policy"):
                await session.handle(
                    "process.send_message",
                    {"recipient_pid": restricted, "body": content},
                )
            return sent

        sent = asyncio.run(exercise())
        persisted = runtime.store.get_process_message(str(sent["message_id"]))
        assert persisted is not None
        assert persisted.body == content
        assert persisted.metadata["data_labels"]["sensitivity"] == "secret"
        assert persisted.metadata["data_labels"]["tenant"] == "tenant-a"
        assert runtime.messages.unread(restricted) == []

        post_audits = [
            record
            for record in runtime.audit.trace()
            if record.action == "process.message.post"
        ]
        assert any(
            record.target == f"process:{allowed}"
            and record.decision["data_labels"]["tenant"] == "tenant-a"
            for record in post_audits
        )
        assert not any(record.target == f"process:{restricted}" for record in post_audits)
        post_events = [
            event
            for event in runtime.events.list()
            if event.type == EventType.PROCESS_MESSAGE_POSTED
        ]
        assert any(
            event.target == allowed
            and event.payload["data_labels"]["tenant"] == "tenant-a"
            for event in post_events
        )
        assert not any(event.target == restricted for event in post_events)
        assert any(
            record.action == "syscall.request"
            and record.target == "process.send_message"
            for record in runtime.audit.trace(actor=parent)
        )


@pytest.mark.parametrize(
    "syscall_name",
    ["process.read_messages", "process.receive_messages"],
    ids=["read", "receive"],
)
def test_jit_receive_observes_labels_before_ack(
    syscall_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        sender = runtime.process.spawn(
            goal="send classified message",
            authority_manifest=_tenant_manifest(["tenant-a"]),
        )
        receiver = runtime.process.spawn_child(
            sender,
            "receive classified message",
            authority_manifest=_tenant_manifest(["tenant-a"]),
        )
        message = runtime.messages.send_from_process(
            sender,
            receiver,
            body="MESSAGE_SECRET_SENTINEL",
            source_context=DataFlowContext(
                labels=DataLabels(sensitivity="secret", tenant="tenant-a")
            ),
        )

        def fail_observation(pid: str, messages: object) -> list[str]:
            assert pid == receiver
            assert [item.message_id for item in messages] == [message.message_id]
            persisted = runtime.store.get_process_message(message.message_id)
            assert persisted is not None and persisted.status.value == "unread"
            raise RuntimeError("injected label observation failure")

        monkeypatch.setattr(runtime.messages, "observe_labels", fail_observation)
        session = LibOSSyscallSession(runtime, receiver)
        with pytest.raises(RuntimeError, match="observation failure"):
            asyncio.run(session.handle(syscall_name, {"block": False}))

        persisted = runtime.store.get_process_message(message.message_id)
        assert persisted is not None and persisted.status.value == "unread"
        assert "label_carrier_oid" not in persisted.metadata
        assert not any(
            event.type == EventType.PROCESS_MESSAGE_ACKED
            for event in runtime.events.list(target=receiver)
        )
        receiver_audits = runtime.audit.trace(actor=receiver)
        assert not any(record.action == "process.message.ack" for record in receiver_audits)
        assert any(
            record.action == "syscall.request" and record.target == syscall_name
            for record in receiver_audits
        )
        assert not any(
            record.action == "syscall.result" and record.target == syscall_name
            for record in receiver_audits
        )


@pytest.mark.parametrize(
    ("syscall_name", "ack"),
    [("process.read_messages", True), ("process.receive_messages", False)],
    ids=["read-ack", "receive-no-ack"],
)
def test_jit_receive_taints_session_and_denies_forward(
    syscall_name: str,
    ack: bool,
) -> None:
    with workspace_runtime() as (runtime, _root):
        sender = runtime.process.spawn(
            goal="sender",
            authority_manifest=_tenant_manifest(["tenant-a"]),
        )
        receiver = runtime.process.spawn_child(
            sender,
            "JIT receiver",
            authority_manifest=_tenant_manifest(["tenant-a"]),
        )
        restricted = runtime.process.spawn_child(
            receiver,
            "restricted descendant",
            authority_manifest=_tenant_manifest([]),
        )
        secret = runtime.memory.create_object(
            sender,
            ObjectType.ARTIFACT,
            {"value": "MESSAGE_SECRET_SENTINEL"},
            metadata=ObjectMetadata(sensitivity="secret", tenant="tenant-a"),
        )
        sender_process = runtime.process.get(sender)
        assert sender_process.memory_view is not None
        sender_process.memory_view.roots.append(secret)
        runtime.store.update_process(sender_process)
        message = runtime.messages.send_from_process(
            sender,
            receiver,
            body="MESSAGE_SECRET_SENTINEL",
        )

        async def exercise() -> None:
            session = LibOSSyscallSession(runtime, receiver)
            read = await session.handle(
                syscall_name,
                {"block": False, "ack": ack},
            )
            assert read["messages"][0]["body"] == "MESSAGE_SECRET_SENTINEL"
            flow = runtime.data_flow.current_context()
            assert flow.labels.sensitivity.value == "secret"
            assert flow.labels.tenant == "tenant-a"

            await session.handle(
                "process.read_messages",
                {
                    "message_ids": [message.message_id],
                    "include_acked": ack,
                    "ack": False,
                },
            )
            with pytest.raises(CapabilityDenied, match="data_flow_policy"):
                await session.handle(
                    "process.send_message",
                    {
                        "recipient_pid": restricted,
                        "body": "MESSAGE_SECRET_SENTINEL",
                    },
                )

        asyncio.run(exercise())

        persisted = runtime.store.get_process_message(message.message_id)
        assert persisted is not None
        carrier_oid = persisted.metadata.get("label_carrier_oid")
        assert carrier_oid
        carrier = runtime.store.get_object(str(carrier_oid))
        assert carrier is not None and carrier.type == ObjectType.MESSAGE
        assert carrier.metadata.sensitivity == "secret"
        assert carrier.metadata.tenant == "tenant-a"
        assert "label_carrier" in carrier.metadata.tags
        assert f"process_message:{message.message_id}" in carrier.provenance.source_refs
        assert secret.oid in carrier.provenance.parent_oids
        receiver_process = runtime.process.get(receiver)
        assert receiver_process.memory_view is not None
        assert [handle.oid for handle in receiver_process.memory_view.roots].count(str(carrier_oid)) == 1
        assert runtime.messages.unread(restricted) == []

        ack_events = [
            event
            for event in runtime.events.list(target=receiver)
            if event.type == EventType.PROCESS_MESSAGE_ACKED
        ]
        ack_audits = [
            record
            for record in runtime.audit.trace(actor=receiver)
            if record.action == "process.message.ack"
        ]
        if ack:
            assert persisted.status.value == "acked"
            assert ack_events[-1].payload["message_ids"] == [message.message_id]
            assert ack_audits[-1].decision["message_ids"] == [message.message_id]
        else:
            assert persisted.status.value == "unread"
            assert ack_events == []
            assert ack_audits == []


def test_jit_receive_taints_tool_result_across_sandbox_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, _root):
        receiver = runtime.process.spawn(
            image="toolmaker-agent:v0",
            goal="return a classified message",
            authority_manifest=_tenant_manifest(["tenant-a"]),
        )
        sender = runtime.process.spawn_child(
            receiver,
            "message sender",
            authority_manifest=_tenant_manifest(["tenant-a"]),
        )
        message = runtime.messages.send_from_process(
            sender,
            receiver,
            body="SANDBOX_TASK_SECRET_SENTINEL",
            source_context=DataFlowContext(
                labels=DataLabels(sensitivity="secret", tenant="tenant-a")
            ),
        )
        async def run_in_sandbox_task(
            _source: str,
            _args: dict[str, object],
            **kwargs: object,
        ) -> dict[str, object]:
            syscall_handler = kwargs["syscall_handler"]

            async def serve() -> dict[str, object]:
                read = await syscall_handler("process.read_messages", {})
                return {"body": read["messages"][0]["body"]}

            return await asyncio.create_task(serve())

        _install_mocked_jit_runner(runtime, monkeypatch, run_in_sandbox_task)
        candidate = runtime.tools.propose(
            receiver,
            {
                "name": "jit_receive_secret",
                "description": "Return one received process message.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            source_code="export function run(args, libos) { return {}; }",
        )
        runtime.tools.register(receiver, candidate)
        result = runtime.tools.call(receiver, "jit_receive_secret", {})

        assert result.ok, result.error
        assert result.payload == {"body": "SANDBOX_TASK_SECRET_SENTINEL"}
        assert result.result_handle is not None
        stored_result = runtime.store.get_object(result.result_handle.oid)
        assert stored_result is not None
        assert stored_result.metadata.sensitivity == "secret"
        assert stored_result.metadata.tenant == "tenant-a"
        persisted_message = runtime.store.get_process_message(message.message_id)
        assert persisted_message is not None
        carrier_oid = str(persisted_message.metadata["label_carrier_oid"])
        assert carrier_oid in stored_result.provenance.parent_oids


@pytest.mark.parametrize("operation", ["exec", "fork", "spawn"])
def test_jit_lifecycle_goal_preserves_ambient_labels(operation: str) -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal=f"JIT {operation} source",
            authority_manifest=_tenant_manifest(["tenant-a"]),
        )
        runtime.capability.grant(
            pid,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        relative, content = _bind_ambient_secret_file(runtime, root, pid)
        session = LibOSSyscallSession(runtime, pid)

        async def exercise() -> dict[str, object] | None:
            async def syscall_server() -> dict[str, object]:
                await session.handle("filesystem.read_text", {"path": relative})
                flow = runtime.data_flow.current_context()
                assert flow.labels.sensitivity.value == "secret"
                assert flow.labels.tenant == "tenant-a"
                assert len(flow.source_refs) == 1
                assert flow.source_refs[0].oid.startswith(
                    runtime.data_flow.FILE_BINDING_SOURCE_REF_PREFIX
                )
                if operation == "exec":
                    return await session.handle(
                        "process.exec",
                        {
                            "image": "base-agent:v0",
                            "goal": content,
                            "preserve_memory": False,
                            "preserve_capabilities": True,
                        },
                    )
                if operation == "fork":
                    return await session.handle(
                        "process.fork",
                        {"goal": content, "include_parent_roots": False},
                    )
                return await session.handle("process.spawn_child", {"goal": content})

            result = await asyncio.create_task(syscall_server())
            assert runtime.data_flow.current_context().labels.sensitivity.value == "normal"
            if operation == "exec":
                await session.apply_deferred_lifecycle()
            return result

        result = asyncio.run(exercise())
        target_pid = pid if operation == "exec" else str(result["child_pid"])
        target = runtime.process.get(target_pid)
        goal = runtime.store.get_object(target.goal_oid)
        assert goal is not None
        assert goal.payload == {"text": content}
        assert goal.metadata.sensitivity == "secret"
        assert goal.metadata.tenant == "tenant-a"
        assert target.memory_view is not None
        assert {handle.oid for handle in target.memory_view.roots} == {goal.oid}

        expected_event = {
            "exec": EventType.PROCESS_EXEC,
            "fork": EventType.PROCESS_FORKED,
            "spawn": EventType.PROCESS_CREATED,
        }[operation]
        assert any(
            event.type == expected_event and event.target == target_pid
            for event in runtime.events.list()
        )
        expected_audit = {
            "exec": "process.exec",
            "fork": "process.fork",
            "spawn": "process.spawn_child",
        }[operation]
        lifecycle_audits = [
            record
            for record in runtime.audit.trace()
            if record.action == expected_audit and record.target == f"process:{target_pid}"
        ]
        assert lifecycle_audits and goal.oid in lifecycle_audits[-1].output_refs


@pytest.mark.parametrize("operation", ["exec", "fork", "spawn"])
def test_jit_lifecycle_rejects_tenant_goal_before_transition(operation: str) -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal=f"restricted JIT {operation}",
            authority_manifest=_tenant_manifest([]),
        )
        runtime.capability.grant(
            pid,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        relative, content = _bind_ambient_secret_file(runtime, root, pid)
        session = LibOSSyscallSession(runtime, pid)
        original = runtime.process.get(pid)
        original_process_ids = {process.pid for process in runtime.store.list_processes()}
        original_roots = [
            handle.oid for handle in (original.memory_view.roots if original.memory_view else [])
        ]

        async def exercise() -> None:
            async def syscall_server() -> None:
                await session.handle("filesystem.read_text", {"path": relative})
                if operation == "exec":
                    await session.handle(
                        "process.exec",
                        {
                            "image": "base-agent:v0",
                            "goal": content,
                            "preserve_memory": False,
                            "preserve_capabilities": True,
                        },
                    )
                    return
                syscall_name = "process.fork" if operation == "fork" else "process.spawn_child"
                args = (
                    {"goal": content, "include_parent_roots": False}
                    if operation == "fork"
                    else {"goal": content}
                )
                await session.handle(syscall_name, args)

            if operation == "exec":
                await asyncio.create_task(syscall_server())
                assert runtime.data_flow.current_context().labels.sensitivity.value == "normal"
                with pytest.raises(CapabilityDenied, match="data_flow_policy"):
                    await session.apply_deferred_lifecycle()
            else:
                with pytest.raises(CapabilityDenied, match="data_flow_policy"):
                    await asyncio.create_task(syscall_server())

        asyncio.run(exercise())

        after = runtime.process.get(pid)
        assert after.image_id == original.image_id
        assert after.goal_oid == original.goal_oid
        assert [
            handle.oid for handle in (after.memory_view.roots if after.memory_view else [])
        ] == original_roots
        assert {process.pid for process in runtime.store.list_processes()} == original_process_ids

        expected_event = {
            "exec": EventType.PROCESS_EXEC,
            "fork": EventType.PROCESS_FORKED,
            "spawn": EventType.PROCESS_CREATED,
        }[operation]
        assert not any(
            event.type == expected_event
            and (
                operation != "spawn"
                or event.source == pid
            )
            for event in runtime.events.list()
        )
        expected_audit = {
            "exec": "process.exec",
            "fork": "process.fork",
            "spawn": "process.spawn_child",
        }[operation]
        assert not any(
            record.action == expected_audit
            for record in runtime.audit.trace(actor=pid)
        )
        syscall_target = {
            "exec": "process.exec",
            "fork": "process.fork",
            "spawn": "process.spawn_child",
        }[operation]
        assert any(
            record.action == "syscall.request" and record.target == syscall_target
            for record in runtime.audit.trace(actor=pid)
        )
        if operation == "exec":
            failed_boots = [
                record
                for record in runtime.audit.trace()
                if record.action == "image.boot.failed"
                and record.target == f"process:{pid}"
            ]
            assert failed_boots
            assert failed_boots[-1].decision["rolled_back"] is True
            assert failed_boots[-1].decision["phase"] == "process.exec"
        else:
            assert not any(
                record.action == "syscall.result" and record.target == syscall_target
                for record in runtime.audit.trace(actor=pid)
            )


def test_llm_memory_create_inherits_ambient_labels_without_source_refs() -> None:
    with workspace_runtime() as (runtime, _root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="derive ambient labels")
        runtime.tools.configure_process_tools(
            pid,
            ["create_memory_object"],
            assigned_by="test.host",
        )

        result = runtime.tools.call(
            pid,
            "create_memory_object",
            {"name": "ambient-derived", "type": "summary", "payload": {"ok": True}},
            context_metadata={
                "data_flow_context": DataFlowContext(
                    labels=DataLabels(sensitivity="secret")
                )
            },
        )

        assert result.ok, result.error
        created = runtime.store.get_object_by_name(
            "ambient-derived",
            namespace=runtime.memory.resolve_namespace(pid),
        )
        assert created is not None and created.metadata.sensitivity == "secret"


def test_ambiguous_file_write_keeps_conservative_path_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="ambiguous labeled write")
        source = _secret_object(runtime, pid)
        trusted_path = "trusted/ambiguous.txt"
        trusted_sink = runtime.filesystem.resource_for_path(trusted_path)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=trusted_sink,
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        runtime.filesystem.grant_path(
            pid,
            trusted_path,
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test.host",
        )
        original_write = runtime.filesystem.provider.write_text

        def write_then_fail(*args, **kwargs) -> None:
            original_write(*args, **kwargs)
            raise RuntimeError("provider result became ambiguous after write")

        monkeypatch.setattr(runtime.filesystem.provider, "write_text", write_then_fail)
        with pytest.raises(RuntimeError, match="ambiguous after write"):
            runtime.filesystem.write_text(
                pid,
                trusted_path,
                "DATA_FLOW_SECRET_SENTINEL",
                source_oids=[source.oid],
            )

        assert (root / trusted_path).read_text(encoding="utf-8") == (
            "DATA_FLOW_SECRET_SENTINEL"
        )
        binding = runtime.store.get_file_label_binding(trusted_path)
        assert binding is not None
        assert binding.labels.sensitivity.value == "secret"
        assert source.oid in {ref.oid for ref in binding.source_refs}

        monkeypatch.setattr(runtime.filesystem.provider, "write_text", original_write)
        runtime.filesystem.grant_path(
            pid,
            "leak.txt",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        read = runtime.filesystem.read_text(pid, trusted_path)
        with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
            runtime.filesystem.write_text(pid, "leak.txt", read.content)

        assert not (root / "leak.txt").exists()


def test_file_to_object_conversion_preserves_path_labels() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(image="base-agent:v0", goal="import labeled file")
        source = _secret_object(runtime, pid)
        relative = "classified.txt"
        content = "DATA_FLOW_SECRET_SENTINEL"
        (root / relative).write_text(content, encoding="utf-8")
        binding = runtime.data_flow.bind_written_file(
            pid=pid,
            normalized_path=relative,
            content=content.encode("utf-8"),
            context=runtime.data_flow.context_from_trusted_source_oids([source.oid]),
        )
        runtime.filesystem.grant_path(
            pid,
            relative,
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        runtime.tools.configure_process_tools(
            pid,
            ["create_object_from_file"],
            assigned_by="test.host",
        )

        result = runtime.tools.call(
            pid,
            "create_object_from_file",
            {"name": "imported-secret", "path": relative},
        )

        assert result.ok, result.error
        imported = runtime.store.get_object_by_name(
            "imported-secret",
            namespace=runtime.memory.resolve_namespace(pid),
        )
        assert imported is not None
        assert imported.metadata.sensitivity == "secret"
        assert source.oid in imported.provenance.parent_oids
        assert (
            f"{runtime.data_flow.FILE_BINDING_SOURCE_REF_PREFIX}{binding.binding_id}"
            in imported.provenance.source_refs
        )
