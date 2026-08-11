from __future__ import annotations

from dataclasses import replace

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG
from agent_libos.mcp import (
    InMemoryMcpCredentialBroker,
    McpInputRequired,
    McpRemoteTask,
    McpTasksExtensionSpec,
)
from agent_libos.mcp.manifest import MCP_TASKS_EXTENSION_ID
from agent_libos.models import CapabilityRight, ResourceBudget
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.sdk.protected_operations import ProtectedOperation

from tests.runtime.test_mcp_v3_durable_facade import (
    _InitialInputRequiredProvider,
    _InitialTaskProvider,
    _approved_input_responses,
    _manifest,
    _substrate,
)


class _ContinuationTaskProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def continue_tool(
        self,
        server,
        mcp_name: str,
        arguments: dict,
        input_responses: dict,
        request_state: str | None,
        *,
        deadline: float,
    ) -> dict:
        del server, mcp_name, arguments, input_responses, request_state, deadline
        self.calls += 1
        return {
            "resultType": "task",
            "taskId": "private-continuation-task",
            "status": "working",
            "createdAt": "2030-01-01T00:00:00Z",
            "lastUpdatedAt": "2030-01-01T00:00:01Z",
            "ttlMs": 60_000,
            "pollIntervalMs": 1,
        }


class _SimulatedProcessDeath(BaseException):
    """Bypass ordinary cleanup at one exact protected-result handoff."""


def _crash_after_initial_capture_commit(
    monkeypatch,
    *,
    result_type: type[McpInputRequired] | type[McpRemoteTask],
) -> list[str]:
    """Crash after the protected outer transaction, before primitive return."""

    original_complete = ProtectedOperation.complete
    committed_effect_ids: list[str] = []

    def complete_then_crash(self, *args, **kwargs):
        result = original_complete(self, *args, **kwargs)
        if (
            not committed_effect_ids
            and self.contract.name == "primitive.mcp.call"
            and isinstance(result, result_type)
        ):
            assert isinstance(self.effect_id, str) and self.effect_id
            committed_effect_ids.append(self.effect_id)
            raise _SimulatedProcessDeath(
                "after protected capture commit, before primitive return"
            )
        return result

    monkeypatch.setattr(ProtectedOperation, "complete", complete_then_crash)
    return committed_effect_ids


def _sqlite_storage_bytes(database) -> bytes:
    return b"".join(
        path.read_bytes()
        for path in sorted(database.parent.glob(f"{database.name}*"))
        if path.is_file()
    )


def _effect_for(runtime: Runtime, *, kind: str) -> object:
    matches = [
        effect
        for effect in runtime.store.list_external_effects()
        if effect.provider == "mcp"
        and effect.provider_receipt.get("mcp_durable_result", {}).get("kind")
        == kind
    ]
    assert len(matches) == 1
    return matches[0]


def _grant_tool(runtime: Runtime, server_id: str, tool_id: str) -> str:
    pid = runtime.process.spawn(
        image="base-agent:v0",
        goal="recover an atomically published MCP result",
        resource_budget=ResourceBudget(max_mcp_bytes=64_000),
    )
    runtime.capability.grant(
        pid,
        f"mcp:{server_id}:{tool_id}",
        [CapabilityRight.READ],
        issued_by="test",
    )
    runtime.capability.grant(
        pid,
        f"mcp_server:{server_id}",
        [CapabilityRight.EXECUTE],
        issued_by="test",
    )
    runtime.capability.grant(
        pid,
        "human:owner",
        [CapabilityRight.WRITE],
        issued_by="test",
    )
    return pid


def test_initial_input_required_reopens_from_exact_effect_receipt_without_replay(
    tmp_path,
) -> None:
    database = tmp_path / "capture-continuation.sqlite"
    broker = InMemoryMcpCredentialBroker()
    initial = Runtime.open(database, substrate=_substrate(tmp_path, broker))
    provider = _InitialInputRequiredProvider(
        initial._mcp_v3_tool_provider.result_adapter,
        initial,
    )
    try:
        initial.mcp.register_server(_manifest(), actor="runtime", require_capability=False)
        pid = _grant_tool(initial, "durable-mrtr", "review")
        initial.mcp._modern_tool_provider = provider  # noqa: SLF001
        pending = initial.mcp.call_tool(
            pid,
            "durable-mrtr",
            "review",
            {"document": "release-notes"},
        )
        assert isinstance(pending, McpInputRequired)
        effect_id = _effect_for(initial, kind="input_required").effect_id
        assert initial.mcp.recover_durable_result(effect_id) == pending
    finally:
        initial.close()

    reopened = Runtime.open(database, substrate=_substrate(tmp_path, broker))
    try:
        recovered = reopened.mcp.recover_durable_result(effect_id)
        assert recovered == pending
        assert provider.calls == 1
        assert b"broker-only-round-state" not in database.read_bytes()
    finally:
        reopened.close()
        broker.close()


def test_initial_task_reopens_from_exact_effect_receipt_without_remote_id(
    tmp_path,
) -> None:
    database = tmp_path / "capture-task.sqlite"
    digest = "b" * 64
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
            remote_task_poll_min_interval_s=0.000001,
        )
    )
    manifest = replace(
        _manifest(),
        server_id="capture-task",
        tasks_extension=McpTasksExtensionSpec(
            extension_id=MCP_TASKS_EXTENSION_ID,
            spec_sha256=digest,
        ),
        tools=(
            replace(
                _manifest().tools[0],
                tool_id="begin-task",
                mcp_name="fixture.begin_task",
                input_schema={
                    "type": "object",
                    "properties": {"mode": {"type": "string", "enum": ["input"]}},
                    "required": ["mode"],
                    "additionalProperties": False,
                },
            ),
        ),
    )
    broker = InMemoryMcpCredentialBroker()
    initial = Runtime.open(
        database,
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    provider = _InitialTaskProvider(
        initial._mcp_v3_tool_provider.result_adapter,
        initial,
    )
    try:
        initial.mcp.register_server(manifest, actor="runtime", require_capability=False)
        pid = _grant_tool(initial, "capture-task", "begin-task")
        initial.mcp._modern_tool_provider = provider  # noqa: SLF001
        task = initial.mcp.call_tool(
            pid,
            "capture-task",
            "begin-task",
            {"mode": "input"},
        )
        assert isinstance(task, McpRemoteTask)
        effect_id = _effect_for(initial, kind="remote_task").effect_id
        assert initial.mcp.recover_durable_result(effect_id) == task
    finally:
        initial.close()

    reopened = Runtime.open(
        database,
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    try:
        recovered = reopened.mcp.recover_durable_result(effect_id)
        assert recovered == task
        assert provider.calls == 1
        database_bytes = database.read_bytes()
        assert b"private-input-task" not in database_bytes
        assert b"taskId" not in repr(recovered).encode("utf-8")
    finally:
        reopened.close()
        broker.close()


def test_initial_input_required_post_commit_crash_reopens_without_replay(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "capture-continuation-post-commit-crash.sqlite"
    broker = InMemoryMcpCredentialBroker()
    initial = Runtime.open(database, substrate=_substrate(tmp_path, broker))
    provider = _InitialInputRequiredProvider(
        initial._mcp_v3_tool_provider.result_adapter,
        initial,
    )
    committed_effect_ids = _crash_after_initial_capture_commit(
        monkeypatch,
        result_type=McpInputRequired,
    )
    try:
        initial.mcp.register_server(_manifest(), actor="runtime", require_capability=False)
        pid = _grant_tool(initial, "durable-mrtr", "review")
        initial.mcp._modern_tool_provider = provider  # noqa: SLF001
        with pytest.raises(
            _SimulatedProcessDeath,
            match="after protected capture commit",
        ):
            initial.mcp.call_tool(
                pid,
                "durable-mrtr",
                "review",
                {"document": "release-notes"},
            )

        assert provider.calls == 1
        assert len(committed_effect_ids) == 1
        effect_id = committed_effect_ids[0]
        effect = next(
            effect
            for effect in initial.store.list_external_effects()
            if effect.effect_id == effect_id
        )
        assert effect.effect_state == "finalized"
        assert effect.transaction_state == "committed"
        recovered_before_reopen = initial.mcp.recover_durable_result(effect_id)
        assert isinstance(recovered_before_reopen, McpInputRequired)
        assert (
            initial.uow.mcp_continuations.get(
                recovered_before_reopen.continuation_id
            )
            is not None
        )
        preparations = initial.uow.mcp_side_effects.list(
            operation_kind="continuation"
        )
        assert len(preparations) == 1
        assert preparations[0].status == "cleaning"
        assert b"broker-only-round-state" not in _sqlite_storage_bytes(database)
    finally:
        initial.close()

    reopened = Runtime.open(database, substrate=_substrate(tmp_path, broker))
    try:
        recovered = reopened.mcp.recover_durable_result(effect_id)
        assert recovered == recovered_before_reopen
        assert provider.calls == 1
        assert reopened.uow.mcp_side_effects.list(
            operation_kind="continuation"
        ) == ()
        assert b"broker-only-round-state" not in _sqlite_storage_bytes(database)
    finally:
        reopened.close()
        broker.close()


def test_initial_remote_task_post_commit_crash_reopens_without_replay(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "capture-task-post-commit-crash.sqlite"
    digest = "e" * 64
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
            remote_task_poll_min_interval_s=0.000001,
        )
    )
    manifest = replace(
        _manifest(),
        server_id="capture-task-crash",
        tasks_extension=McpTasksExtensionSpec(
            extension_id=MCP_TASKS_EXTENSION_ID,
            spec_sha256=digest,
        ),
        tools=(
            replace(
                _manifest().tools[0],
                tool_id="begin-task",
                mcp_name="fixture.begin_task",
                input_schema={
                    "type": "object",
                    "properties": {"mode": {"type": "string", "enum": ["input"]}},
                    "required": ["mode"],
                    "additionalProperties": False,
                },
            ),
        ),
    )
    broker = InMemoryMcpCredentialBroker()
    initial = Runtime.open(
        database,
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    provider = _InitialTaskProvider(
        initial._mcp_v3_tool_provider.result_adapter,
        initial,
    )
    committed_effect_ids = _crash_after_initial_capture_commit(
        monkeypatch,
        result_type=McpRemoteTask,
    )
    try:
        initial.mcp.register_server(manifest, actor="runtime", require_capability=False)
        pid = _grant_tool(initial, "capture-task-crash", "begin-task")
        initial.mcp._modern_tool_provider = provider  # noqa: SLF001
        with pytest.raises(
            _SimulatedProcessDeath,
            match="after protected capture commit",
        ):
            initial.mcp.call_tool(
                pid,
                "capture-task-crash",
                "begin-task",
                {"mode": "input"},
            )

        assert provider.calls == 1
        assert len(committed_effect_ids) == 1
        effect_id = committed_effect_ids[0]
        effect = next(
            effect
            for effect in initial.store.list_external_effects()
            if effect.effect_id == effect_id
        )
        assert effect.effect_state == "finalized"
        assert effect.transaction_state == "committed"
        recovered_before_reopen = initial.mcp.recover_durable_result(effect_id)
        assert isinstance(recovered_before_reopen, McpRemoteTask)
        assert (
            initial.uow.mcp_remote_tasks.get(recovered_before_reopen.task_ref)
            is not None
        )
        preparations = initial.uow.mcp_side_effects.list(
            operation_kind="remote_task"
        )
        assert len(preparations) == 1
        assert preparations[0].status == "cleaning"
        database_bytes = _sqlite_storage_bytes(database)
        assert b"private-input-task" not in database_bytes
        assert b"lastUpdatedAt" not in database_bytes
        assert b"taskId" not in database_bytes
    finally:
        initial.close()

    reopened = Runtime.open(
        database,
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    try:
        recovered = reopened.mcp.recover_durable_result(effect_id)
        assert recovered == recovered_before_reopen
        assert provider.calls == 1
        assert reopened.uow.mcp_side_effects.list(
            operation_kind="remote_task"
        ) == ()
        database_bytes = _sqlite_storage_bytes(database)
        assert b"private-input-task" not in database_bytes
        assert b"lastUpdatedAt" not in database_bytes
        assert b"taskId" not in database_bytes
    finally:
        reopened.close()
        broker.close()


def test_continuation_task_handoff_is_atomic_and_recoverable_by_origin_effect(
    tmp_path,
) -> None:
    database = tmp_path / "capture-continuation-task.sqlite"
    digest = "c" * 64
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
        )
    )
    manifest = replace(
        _manifest(),
        tasks_extension=McpTasksExtensionSpec(
            extension_id=MCP_TASKS_EXTENSION_ID,
            spec_sha256=digest,
        ),
    )
    broker = InMemoryMcpCredentialBroker()
    initial = Runtime.open(
        database,
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    initial_provider = _InitialInputRequiredProvider(
        initial._mcp_v3_tool_provider.result_adapter,
        initial,
    )
    continuation_provider = _ContinuationTaskProvider()
    try:
        initial.mcp.register_server(manifest, actor="runtime", require_capability=False)
        pid = _grant_tool(initial, "durable-mrtr", "review")
        initial.mcp._modern_tool_provider = initial_provider  # noqa: SLF001
        initial.mcp._modern_continuation_provider = continuation_provider  # noqa: SLF001
        pending = initial.mcp.call_tool(
            pid,
            "durable-mrtr",
            "review",
            {"document": "release-notes"},
        )
        assert isinstance(pending, McpInputRequired)
        effect_id = _effect_for(initial, kind="input_required").effect_id
        assert pending.human_request_id is not None
        assert pending.human_revision is not None
        assert pending.human_preview_sha256 is not None
        task = initial.mcp.respond_continuation(
            pending.continuation_id,
            expected_revision=pending.revision,
            responses=_approved_input_responses(),
            human_request_id=pending.human_request_id,
            human_expected_revision=pending.human_revision,
            human_preview_sha256=pending.human_preview_sha256,
            actor=pid,
        )
        assert isinstance(task, McpRemoteTask)
        assert initial.mcp.recover_durable_result(effect_id) == task
    finally:
        initial.close()

    reopened = Runtime.open(
        database,
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    try:
        assert reopened.mcp.recover_durable_result(effect_id) == task
        assert initial_provider.calls == 1
        assert continuation_provider.calls == 1
        assert b"private-continuation-task" not in database.read_bytes()
    finally:
        reopened.close()
        broker.close()


def test_continuation_task_raw_result_handoff_crash_recovers_without_replay(
    tmp_path,
) -> None:
    """A protected Task result must be durable before raw control returns.

    This is the historical split point: ``_run_modern_read`` had already
    finalized the continuation-response effect, but the continuation manager
    had not yet parsed the raw Task mapping or created its preparation.  A
    process exit there left a live remote Task with no local recovery handle.
    """

    database = tmp_path / "capture-continuation-task-raw-handoff.sqlite"
    digest = "d" * 64
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
        )
    )
    manifest = replace(
        _manifest(),
        tasks_extension=McpTasksExtensionSpec(
            extension_id=MCP_TASKS_EXTENSION_ID,
            spec_sha256=digest,
        ),
    )
    broker = InMemoryMcpCredentialBroker()
    initial = Runtime.open(
        database,
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    initial_provider = _InitialInputRequiredProvider(
        initial._mcp_v3_tool_provider.result_adapter,
        initial,
    )
    continuation_provider = _ContinuationTaskProvider()
    try:
        initial.mcp.register_server(manifest, actor="runtime", require_capability=False)
        pid = _grant_tool(initial, "durable-mrtr", "review")
        initial.mcp._modern_tool_provider = initial_provider  # noqa: SLF001
        initial.mcp._modern_continuation_provider = (  # noqa: SLF001
            continuation_provider
        )
        pending = initial.mcp.call_tool(
            pid,
            "durable-mrtr",
            "review",
            {"document": "release-notes"},
        )
        assert isinstance(pending, McpInputRequired)
        effect_id = _effect_for(initial, kind="input_required").effect_id
        assert pending.human_request_id is not None
        assert pending.human_revision is not None
        assert pending.human_preview_sha256 is not None

        run_modern_read = initial.mcp._run_modern_read  # noqa: SLF001

        def crash_after_protected_result(*args, **kwargs):
            result = run_modern_read(*args, **kwargs)
            if kwargs.get("operation") == "continuation.respond":
                raise _SimulatedProcessDeath(
                    "after protected Task result, before manager handoff"
                )
            return result

        initial.mcp._run_modern_read = crash_after_protected_result  # noqa: SLF001
        with pytest.raises(_SimulatedProcessDeath, match="before manager handoff"):
            initial.mcp.respond_continuation(
                pending.continuation_id,
                expected_revision=pending.revision,
                responses=_approved_input_responses(),
                human_request_id=pending.human_request_id,
                human_expected_revision=pending.human_revision,
                human_preview_sha256=pending.human_preview_sha256,
                actor=pid,
            )
        assert continuation_provider.calls == 1
        response_effects = [
            effect
            for effect in initial.store.list_external_effects()
            if effect.provider == "mcp"
            and effect.operation == "continuation.respond"
        ]
        assert len(response_effects) == 1
        assert response_effects[0].effect_id != effect_id
        assert response_effects[0].effect_state == "finalized"
        assert response_effects[0].transaction_state == "committed"
        tasks = initial.uow.mcp_remote_tasks.list()
        assert len(tasks) == 1
        assert tasks[0].origin_effect_id == response_effects[0].effect_id
        assert response_effects[0].provider_receipt["mcp_durable_result"] == {
            "kind": "remote_task",
            "task_ref": tasks[0].task_ref,
        }
        assert (
            initial.mcp.recover_durable_result(response_effects[0].effect_id).task_ref
            == tasks[0].task_ref
        )
        assert initial.uow.mcp_side_effects.list(operation_kind="remote_task") == ()
    finally:
        initial.close()

    reopened = Runtime.open(
        database,
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    try:
        recovered = reopened.mcp.recover_durable_result(effect_id)
        assert isinstance(recovered, McpRemoteTask)
        assert (
            reopened.mcp.recover_durable_result(response_effects[0].effect_id)
            == recovered
        )
        assert continuation_provider.calls == 1
        assert b"private-continuation-task" not in database.read_bytes()
    finally:
        reopened.close()
        broker.close()


class _ContinuationCompleteProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def continue_tool(
        self,
        server,
        mcp_name: str,
        arguments: dict,
        input_responses: dict,
        request_state: str | None,
        *,
        deadline: float,
    ) -> dict:
        del server, mcp_name, arguments, input_responses, request_state, deadline
        self.calls += 1
        return {
            "resultType": "complete",
            "content": [{"type": "text", "text": "private-complete-result"}],
        }


class _ContinuationNextRoundProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def continue_tool(
        self,
        server,
        mcp_name: str,
        arguments: dict,
        input_responses: dict,
        request_state: str | None,
        *,
        deadline: float,
    ) -> dict:
        del server, mcp_name, arguments, input_responses, request_state, deadline
        self.calls += 1
        if self.calls == 1:
            return {
                "resultType": "input_required",
                "requestState": "private-second-round-state",
                "inputRequests": {
                    "second-provider-key": {
                        "method": "elicitation/create",
                        "params": {
                            "mode": "form",
                            "message": "Approve the second round?",
                            "requestedSchema": {
                                "type": "object",
                                "properties": {"approved": {"type": "boolean"}},
                                "required": ["approved"],
                            },
                        },
                    }
                },
            }
        return {
            "resultType": "complete",
            "content": [
                {"type": "text", "text": "private-second-round-complete"}
            ],
        }


def _initial_continuation_for_publication_crash(
    tmp_path,
    database_name: str,
    continuation_provider,
    *,
    config: AgentLibOSConfig = DEFAULT_CONFIG,
    manifest=None,
):
    database = tmp_path / database_name
    broker = InMemoryMcpCredentialBroker()
    runtime = Runtime.open(
        database,
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    initial_provider = _InitialInputRequiredProvider(
        runtime._mcp_v3_tool_provider.result_adapter,
        runtime,
    )
    runtime.mcp.register_server(
        _manifest() if manifest is None else manifest,
        actor="runtime",
        require_capability=False,
    )
    pid = _grant_tool(runtime, "durable-mrtr", "review")
    runtime.mcp._modern_tool_provider = initial_provider  # noqa: SLF001
    runtime.mcp._modern_continuation_provider = (  # noqa: SLF001
        continuation_provider
    )
    pending = runtime.mcp.call_tool(
        pid,
        "durable-mrtr",
        "review",
        {"document": "release-notes"},
    )
    assert isinstance(pending, McpInputRequired)
    initial_effect_id = _effect_for(runtime, kind="input_required").effect_id
    return (
        database,
        broker,
        runtime,
        initial_provider,
        pid,
        pending,
        initial_effect_id,
    )


def _crash_after_continuation_result(runtime: Runtime) -> None:
    run_modern_read = runtime.mcp._run_modern_read  # noqa: SLF001

    def crash_after_protected_result(*args, **kwargs):
        result = run_modern_read(*args, **kwargs)
        if kwargs.get("operation") == "continuation.respond":
            raise _SimulatedProcessDeath(
                "after protected continuation result, before Host return"
            )
        return result

    runtime.mcp._run_modern_read = crash_after_protected_result  # noqa: SLF001


def _respond_to_pending(runtime: Runtime, pending: McpInputRequired, pid: str):
    assert pending.human_request_id is not None
    assert pending.human_revision is not None
    assert pending.human_preview_sha256 is not None
    return runtime.mcp.respond_continuation(
        pending.continuation_id,
        expected_revision=pending.revision,
        responses=_approved_input_responses(),
        human_request_id=pending.human_request_id,
        human_expected_revision=pending.human_revision,
        human_preview_sha256=pending.human_preview_sha256,
        actor=pid,
    )


def _single_response_effect(runtime: Runtime):
    selected = [
        effect
        for effect in runtime.store.list_external_effects()
        if effect.provider == "mcp" and effect.operation == "continuation.respond"
    ]
    assert len(selected) == 1
    return selected[0]


def test_continuation_complete_postcommit_crash_is_terminal_without_replay(
    tmp_path,
) -> None:
    provider = _ContinuationCompleteProvider()
    (
        database,
        broker,
        initial,
        initial_provider,
        pid,
        pending,
        initial_effect_id,
    ) = _initial_continuation_for_publication_crash(
        tmp_path,
        "capture-continuation-complete-crash.sqlite",
        provider,
    )
    try:
        _crash_after_continuation_result(initial)
        with pytest.raises(_SimulatedProcessDeath, match="before Host return"):
            _respond_to_pending(initial, pending, pid)
        response_effect = _single_response_effect(initial)
        assert response_effect.effect_id != initial_effect_id
        assert response_effect.effect_state == "finalized"
        assert response_effect.transaction_state == "committed"
        assert "mcp_durable_result" not in response_effect.provider_receipt
        record = initial.uow.mcp_continuations.get(pending.continuation_id)
        assert record is not None and record.status == "complete"
        assert record.broker_ref is None
        assert initial.uow.mcp_side_effects.list(operation_kind="continuation") == ()
        assert provider.calls == 1
    finally:
        initial.close()

    reopened = Runtime.open(database, substrate=_substrate(tmp_path, broker))
    try:
        with pytest.raises(NotFound, match="receipt"):
            reopened.mcp.recover_durable_result(response_effect.effect_id)
        with pytest.raises(ValidationError):
            reopened.mcp.recover_durable_result(initial_effect_id)
        with pytest.raises(ValidationError):
            reopened.mcp.get_continuation(pending.continuation_id, actor=pid)
        assert provider.calls == 1
        assert initial_provider.calls == 1
        database_bytes = database.read_bytes()
        assert b"private-complete-result" not in database_bytes
        assert b"broker-only-round-state" not in database_bytes
    finally:
        reopened.close()
        broker.close()


def test_continuation_next_round_postcommit_crash_recovers_and_continues_once(
    tmp_path,
) -> None:
    provider = _ContinuationNextRoundProvider()
    (
        database,
        broker,
        initial,
        initial_provider,
        pid,
        pending,
        initial_effect_id,
    ) = _initial_continuation_for_publication_crash(
        tmp_path,
        "capture-continuation-next-round-crash.sqlite",
        provider,
    )
    try:
        _crash_after_continuation_result(initial)
        with pytest.raises(_SimulatedProcessDeath, match="before Host return"):
            _respond_to_pending(initial, pending, pid)
        response_effect = _single_response_effect(initial)
        assert response_effect.effect_id != initial_effect_id
        assert response_effect.effect_state == "finalized"
        assert response_effect.transaction_state == "committed"
        receipt = response_effect.provider_receipt["mcp_durable_result"]
        assert receipt == {
            "kind": "input_required",
            "continuation_id": pending.continuation_id,
        }
        record = initial.uow.mcp_continuations.get(pending.continuation_id)
        assert record is not None and record.status == "input_required"
        assert record.revision == pending.revision + 2
        assert record.human_request_id != pending.human_request_id
        assert initial.uow.mcp_side_effects.list(operation_kind="continuation") == ()
        assert provider.calls == 1
    finally:
        initial.close()

    reopened = Runtime.open(database, substrate=_substrate(tmp_path, broker))
    try:
        recovered = reopened.mcp.recover_durable_result(response_effect.effect_id)
        assert isinstance(recovered, McpInputRequired)
        assert recovered.continuation_id == pending.continuation_id
        assert reopened.mcp.recover_durable_result(initial_effect_id) == recovered
        assert provider.calls == 1
        assert initial_provider.calls == 1
        reopened.mcp._modern_continuation_provider = provider  # noqa: SLF001
        completed = _respond_to_pending(reopened, recovered, pid)
        assert completed.value == {
            "content": [
                {"type": "text", "text": "private-second-round-complete"}
            ]
        }
        assert provider.calls == 2
        database_bytes = database.read_bytes()
        assert b"private-second-round-state" not in database_bytes
        assert b"private-second-round-complete" not in database_bytes
    finally:
        reopened.close()
        broker.close()


@pytest.mark.parametrize("failure_kind", ["exception", "base_exception"])
def test_continuation_task_protected_outer_rollback_never_publishes_partial_ref(
    tmp_path,
    monkeypatch,
    failure_kind: str,
) -> None:
    """Effect evidence failure rolls back both MCP main-row settlements."""

    digest = "e" * 64
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=digest,
        )
    )
    manifest = replace(
        _manifest(),
        tasks_extension=McpTasksExtensionSpec(
            extension_id=MCP_TASKS_EXTENSION_ID,
            spec_sha256=digest,
        ),
    )
    provider = _ContinuationTaskProvider()
    (
        database,
        broker,
        initial,
        initial_provider,
        pid,
        pending,
        _initial_effect_id,
    ) = _initial_continuation_for_publication_crash(
        tmp_path,
        f"capture-continuation-task-outer-rollback-{failure_kind}.sqlite",
        provider,
        config=config,
        manifest=manifest,
    )
    persisted_evidence = ProtectedOperation._persist_evidence
    injected = False
    owned_refs: tuple[str, ...] = ()

    def fail_first_response_evidence(operation, evidence):
        nonlocal injected, owned_refs
        if (
            operation.contract.operation == "continuation.respond"
            and not injected
        ):
            preparations = initial.uow.mcp_side_effects.list(status="cleaning")
            assert {item.operation_kind for item in preparations} == {
                "continuation",
                "remote_task",
            }
            owned_refs = tuple(
                sorted(
                    reference
                    for item in preparations
                    for reference in (item.broker_ref, item.result_ref)
                    if reference is not None
                )
            )
            assert owned_refs
            injected = True
            if failure_kind == "base_exception":
                raise _SimulatedProcessDeath(
                    "after deferred rows, before protected evidence"
                )
            raise RuntimeError("after deferred rows, before protected evidence")
        return persisted_evidence(operation, evidence)

    monkeypatch.setattr(
        ProtectedOperation,
        "_persist_evidence",
        fail_first_response_evidence,
    )
    try:
        if failure_kind == "base_exception":
            with pytest.raises(_SimulatedProcessDeath, match="before protected evidence"):
                _respond_to_pending(initial, pending, pid)
        else:
            with pytest.raises(ValidationError, match="unknown outcome"):
                _respond_to_pending(initial, pending, pid)
        assert injected
        assert provider.calls == 1
        assert initial.uow.mcp_remote_tasks.list() == ()
        assert all(
            "mcp_durable_result" not in effect.provider_receipt
            for effect in initial.store.list_external_effects()
            if effect.operation == "continuation.respond"
        )
        current = initial.uow.mcp_continuations.get(pending.continuation_id)
        assert current is not None
        if failure_kind == "base_exception":
            assert current.status == "dispatching"
            preparations = initial.uow.mcp_side_effects.list(status="prepared")
            assert {item.operation_kind for item in preparations} == {
                "continuation",
                "remote_task",
            }
            for reference in owned_refs:
                assert broker.get_secret(reference)
        else:
            assert current.status == "needs_attention"
            assert initial.uow.mcp_side_effects.list() == ()
            for reference in owned_refs:
                with pytest.raises(ValidationError, match="unavailable"):
                    broker.get_secret(reference)
    finally:
        initial.close()

    reopened = Runtime.open(
        database,
        substrate=_substrate(tmp_path, broker),
        config=config,
    )
    try:
        current = reopened.uow.mcp_continuations.get(pending.continuation_id)
        assert current is not None and current.status == "needs_attention"
        assert reopened.uow.mcp_remote_tasks.list() == ()
        assert reopened.uow.mcp_side_effects.list() == ()
        for reference in owned_refs:
            with pytest.raises(ValidationError, match="unavailable"):
                broker.get_secret(reference)
        assert provider.calls == 1
        assert initial_provider.calls == 1
        database_bytes = database.read_bytes()
        assert b"private-continuation-task" not in database_bytes
        assert b"taskId" not in database_bytes
    finally:
        reopened.close()
        broker.close()
