from __future__ import annotations
import base64
import asyncio
import pytest
import http.client
import json
import tempfile
import threading
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any
from jsonschema import Draft202012Validator
from agent_libos.api.gui.server import (
    GuiEventBroadcaster,
    GuiRequestHandler,
    GuiRuntimeService,
    GuiServerError,
    _BoundedSeenKeys,
    _shutdown_gui_service_before_exit,
    _sse_payload_data,
    create_gui_http_server,
    serve,
)
from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import (
    AgentLibOSConfig,
    DEFAULT_CONFIG,
    GuiDefaults,
    LLMProfile,
    RuntimeDefaults,
)
from agent_libos.models import (
    AuditRecord,
    CapabilityRight,
    Event,
    EventPriority,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    HumanRequest,
    HumanRequestStatus,
    LLMCallRecord,
    ObjectMetadata,
    ObjectPatch,
    ObjectType,
    McpProviderTool,
    McpConnectionInfo,
    McpDiscoveryResult,
    McpProtocolEra,
    McpProtocolMode,
    McpToolListResult,
    ProcessSignal,
    ProcessStatus,
    ResourceUsage,
    SinkTrustLevel,
    SinkTrustRule,
    TaskRunLedgerItem,
    TaskRunLedgerKind,
    TaskRunLink,
    TaskRunRetention,
    TaskRunSpecV1,
    TaskRunStatus,
    process_outcome_to_mapping,
    process_wait_state_to_mapping,
)
from agent_libos.evidence.payload_retention import (
    PayloadRetentionTier,
    external_effect_payload_retention_tier,
    llm_call_payload_sha256,
    retain_llm_call_payload,
)
from agent_libos.mcp import McpSubscriptionEvent
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, HumanResponseRequired, ProcessWaitRequired, ValidationError
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import dumps, to_jsonable
from agent_libos.runtime.runtime import Runtime
from agent_libos.runtime.syscalls import LibOSSyscallSession
from tests.support.checkpoints import checkpoint_cli_json
from tests.support.mcp import MCP_TEST_STDIO_COMMAND, MCP_TEST_STDIO_COMMAND_YAML
from tests.support.skills import write_skill_package


_BASE64URL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
# GUI requests can include runtime and SQLite transitions. Shared Windows
# runners under xdist need more headroom than a local loopback-only request.
_GUI_TEST_HTTP_TIMEOUT_S = 30.0


def _noncanonical_base64url_alias(segment: str) -> str:
    padding = "=" * (-len(segment) % 4)
    decoded = base64.b64decode(
        segment + padding,
        altchars=b"-_",
        validate=True,
    )
    assert base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") == segment
    for replacement in _BASE64URL_ALPHABET:
        candidate = segment[:-1] + replacement
        if candidate != segment and base64.b64decode(
            candidate + padding,
            altchars=b"-_",
            validate=True,
        ) == decoded:
            return candidate
    raise AssertionError("base64url segment has no non-canonical alias")


def test_gui_mcp_continuation_inspect_survives_runtime_reopen(
    tmp_path: Path,
) -> None:
    from agent_libos.mcp import InMemoryMcpCredentialBroker, McpInputRequired
    from agent_libos.models import CapabilityRight
    from agent_libos.substrate import LocalResourceProviderSubstrate
    from tests.unit.test_mcp_v3_continuations import _binding, _input_required

    database = tmp_path / "gui-mcp-continuation-reopen.sqlite"
    broker = InMemoryMcpCredentialBroker()

    def substrate() -> LocalResourceProviderSubstrate:
        selected = LocalResourceProviderSubstrate(tmp_path)
        selected.mcp_credential_broker = broker
        return selected

    initial = Runtime.open(database, substrate=substrate())
    try:
        owner_id = initial.process.spawn(
            image="base-agent:v0",
            goal="recover MCP continuation in GUI",
            authority_manifest={
                "authorized_capabilities": [{
                    "resource": "human:owner",
                    "rights": [CapabilityRight.WRITE.value],
                }]
            },
        )
        pending = initial._mcp_continuation_manager.capture_input_required(
            _binding(owner_id=owner_id),
            _input_required(state="PRIVATE-REOPEN-REQUEST-STATE"),
            expires_at=None,
        )
        assert isinstance(pending, McpInputRequired)
    finally:
        initial.close()

    assert all(
        b"PRIVATE-REOPEN-REQUEST-STATE" not in candidate.read_bytes()
        for candidate in database.parent.glob(f"{database.name}*")
        if candidate.is_file()
    )
    reopened = Runtime.open(database, substrate=substrate())
    server = create_gui_http_server(
        runtime=reopened,
        port=0,
        token="reopen-token",
        auto_run=False,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(
            host,
            port,
            timeout=_GUI_TEST_HTTP_TIMEOUT_S,
        )
        conn.request(
            "POST",
            f"/api/mcp/continuations/{pending.continuation_id}/inspect",
            body="{}",
            headers={
                "Authorization": "Bearer reopen-token",
                "Content-Type": "application/json",
            },
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        conn.close()

        assert response.status == 200, body
        assert body["kind"] == "input_required"
        assert body["continuation_id"] == pending.continuation_id
        assert body["revision"] == pending.revision
        assert body["human_request_id"] == pending.human_request_id
        assert body["human_preview_sha256"] == pending.human_preview_sha256
        assert "PRIVATE-REOPEN-REQUEST-STATE" not in dumps(body)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.service.shutdown()
        server.server_close()
        reopened.close()
        broker.close()


def test_gui_mcp_task_get_recovers_nonzero_revision_after_runtime_reopen(
    tmp_path: Path,
) -> None:
    import time

    from agent_libos.mcp import InMemoryMcpCredentialBroker
    from agent_libos.substrate import LocalResourceProviderSubstrate
    from tests.unit.test_mcp_v3_tasks import (
        _Boundary,
        _TASKS_DIGEST,
        _binding,
        _task_result,
    )

    class TasksProvider:
        mcp_manifest_schema_version = 3
        mcp_protocol_revision = "2026-07-28"

        async def get_remote_task(
            self, server: Any, remote_task_id: str, *, deadline: float
        ) -> dict[str, Any]:
            raise AssertionError((server, remote_task_id, deadline))

        async def update_remote_task(
            self,
            server: Any,
            remote_task_id: str,
            response: dict[str, Any],
            *,
            deadline: float,
        ) -> dict[str, Any]:
            raise AssertionError((server, remote_task_id, response, deadline))

        async def cancel_remote_task(
            self, server: Any, remote_task_id: str, *, deadline: float
        ) -> dict[str, Any]:
            raise AssertionError((server, remote_task_id, deadline))

    database = tmp_path / "gui-mcp-task-reopen.sqlite"
    broker = InMemoryMcpCredentialBroker()
    provider = TasksProvider()
    config = AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=_TASKS_DIGEST,
            remote_task_poll_min_interval_s=0.000001,
        )
    )

    def substrate() -> LocalResourceProviderSubstrate:
        selected = LocalResourceProviderSubstrate(tmp_path)
        selected.mcp_credential_broker = broker
        selected.mcp_tasks_provider = provider
        return selected

    raw_input_required = _task_result(
        status="input_required",
        pollIntervalMs=1,
        inputRequests={
            "remote-input": {
                "method": "elicitation/create",
                "params": {
                    "mode": "form",
                    "message": "Approve the reopened Task?",
                    "requestedSchema": {
                        "type": "object",
                        "properties": {"approved": {"type": "boolean"}},
                        "required": ["approved"],
                    },
                },
            }
        },
    )
    initial = Runtime.open(database, substrate=substrate(), config=config)
    try:
        owner_id = initial.process.spawn(
            image="base-agent:v0",
            goal="recover MCP Task in GUI",
            authority_manifest={
                "authorized_capabilities": [{
                    "resource": "human:owner",
                    "rights": [CapabilityRight.WRITE.value],
                }]
            },
        )
        manager = initial._mcp_remote_task_manager
        binding = _binding(owner_id=owner_id)
        task = manager.capture_task(
            binding,
            _task_result(pollIntervalMs=1),
        )
        first_boundary = _Boundary()
        first_boundary.get_results.append(raw_input_required)
        manager.boundary = first_boundary
        time.sleep(0.02)
        first_observation = asyncio.run(
            manager.get(
                task.task_ref,
                expected_revision=task.revision,
                binding=binding,
                deadline=time.monotonic() + 5,
            )
        )
        assert first_observation.revision > 0
    finally:
        initial.close()

    assert all(
        b"remote-bearer-id" not in candidate.read_bytes()
        for candidate in database.parent.glob(f"{database.name}*")
        if candidate.is_file()
    )
    reopened = Runtime.open(database, substrate=substrate(), config=config)
    second_boundary = _Boundary()
    second_boundary.get_results.append(raw_input_required)
    reopened._mcp_remote_task_manager.boundary = second_boundary
    server = create_gui_http_server(
        runtime=reopened,
        port=0,
        token="task-reopen-token",
        auto_run=False,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.02)
        host, port = server.server_address
        conn = http.client.HTTPConnection(
            host,
            port,
            timeout=_GUI_TEST_HTTP_TIMEOUT_S,
        )
        conn.request(
            "POST",
            f"/api/mcp/remote-tasks/{task.task_ref}/get",
            body="{}",
            headers={
                "Authorization": "Bearer task-reopen-token",
                "Content-Type": "application/json",
            },
        )
        response = conn.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        conn.close()

        assert response.status == 200, body
        assert body["kind"] == "remote_task"
        assert body["task_ref"] == task.task_ref
        assert body["revision"] > first_observation.revision
        assert body["status"] == "input_required"
        assert body["human_request_id"]
        assert "remote-bearer-id" not in dumps(body)
        assert len(second_boundary.get_calls) == 1
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.service.shutdown()
        server.server_close()
        reopened.close()
        broker.close()


def test_gui_validates_user_profiles_before_opening_owned_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = tmp_path / "invalid-profiles.json"
    profiles.write_text("not-json", encoding="utf-8")
    opened = False
    original_open = Runtime.open

    def record_open(*args: object, **kwargs: object) -> Runtime:
        nonlocal opened
        opened = True
        return original_open("local")

    monkeypatch.setattr("agent_libos.api.gui.server.Runtime.open", record_open)

    with pytest.raises(ValidationError, match="invalid LLM profiles JSON"):
        GuiRuntimeService(db="local", auto_run=False, llm_profiles_file=profiles)

    assert opened is False


def test_gui_closes_owned_runtime_when_profile_registration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Runtime] = []
    original_open = Runtime.open

    def record_open(*args: object, **kwargs: object) -> Runtime:
        runtime = original_open("local")
        opened.append(runtime)
        return runtime

    def fail_registration(*args: object, **kwargs: object) -> dict[str, Any]:
        raise RuntimeError("injected profile registration failure")

    monkeypatch.setattr("agent_libos.api.gui.server.Runtime.open", record_open)
    monkeypatch.setattr(GuiRuntimeService, "_register_user_llm_profiles", fail_registration)

    with pytest.raises(RuntimeError, match="injected profile registration failure"):
        GuiRuntimeService(
            db="local",
            auto_run=False,
            llm_profiles_file=tmp_path / "missing-profiles.json",
        )

    assert len(opened) == 1
    assert opened[0].lifecycle.closed is True


def test_gui_closes_owned_runtime_when_http_server_bind_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Runtime] = []
    original_open = Runtime.open
    bind_error = OSError('injected bind failure')

    def record_open(*args: object, **kwargs: object) -> Runtime:
        runtime = original_open('local')
        opened.append(runtime)
        return runtime

    def fail_bind(*_args: object, **_kwargs: object) -> None:
        raise bind_error

    monkeypatch.setattr('agent_libos.api.gui.server.Runtime.open', record_open)
    monkeypatch.setattr('agent_libos.api.gui.server.GuiHTTPServer', fail_bind)

    with pytest.raises(OSError) as raised:
        create_gui_http_server(
            db='local',
            port=0,
            auto_run=False,
            llm_profiles_file=tmp_path / 'missing-profiles.json',
        )

    assert raised.value is bind_error
    assert len(opened) == 1
    assert opened[0].lifecycle.closed is True


def test_gui_bind_failure_does_not_close_borrowed_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open('local')
    bind_error = OSError('injected bind failure')

    def fail_bind(*_args: object, **_kwargs: object) -> None:
        raise bind_error

    monkeypatch.setattr('agent_libos.api.gui.server.GuiHTTPServer', fail_bind)
    try:
        with pytest.raises(OSError) as raised:
            create_gui_http_server(
                runtime=runtime,
                port=0,
                auto_run=False,
                llm_profiles_file=tmp_path / 'missing-profiles.json',
            )

        assert raised.value is bind_error
        assert runtime.lifecycle.closed is False
    finally:
        runtime.shutdown(actor='test', reason='test.cleanup')


@pytest.mark.parametrize('failure_point', ['ready', 'print', 'serve'])
def test_serve_closes_owned_runtime_for_startup_and_serve_failures(
    failure_point: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = create_gui_http_server(
        db='local',
        port=0,
        auto_run=False,
        llm_profiles_file=tmp_path / 'missing-profiles.json',
    )
    runtime = server.service.runtime
    injected_error = RuntimeError(f'injected {failure_point} failure')
    monkeypatch.setattr(
        'agent_libos.api.gui.server.create_gui_http_server',
        lambda **_kwargs: server,
    )

    def fail(_value: object | None = None, **_kwargs: object) -> None:
        raise injected_error

    if failure_point == 'print':
        monkeypatch.setattr('agent_libos.api.gui.server.print', fail, raising=False)
        ready = None
    else:
        ready = fail if failure_point == 'ready' else lambda _payload: None
        if failure_point == 'serve':
            monkeypatch.setattr(server, 'serve_forever', fail)

    with pytest.raises(RuntimeError) as raised:
        serve(
            port=0,
            token=None,
            auto_run=False,
            max_quanta=None,
            ready=ready,
        )

    assert raised.value is injected_error
    assert runtime.lifecycle.closed is True
    assert server.socket.fileno() == -1


def test_serve_treats_keyboard_interrupt_as_graceful_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = create_gui_http_server(
        db="local",
        port=0,
        auto_run=False,
        llm_profiles_file=tmp_path / "missing-profiles.json",
    )
    runtime = server.service.runtime
    monkeypatch.setattr(
        "agent_libos.api.gui.server.create_gui_http_server",
        lambda **_kwargs: server,
    )

    def interrupt() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "serve_forever", interrupt)

    serve(
        port=0,
        token=None,
        auto_run=False,
        max_quanta=None,
        ready=lambda _payload: None,
    )

    assert runtime.lifecycle.closed is True
    assert server.socket.fileno() == -1


def test_gui_rejects_mismatched_config_for_borrowed_runtime(
    tmp_path: Path,
) -> None:
    runtime_config = replace(
        DEFAULT_CONFIG,
        llm=replace(
            DEFAULT_CONFIG.llm,
            profiles={
                **DEFAULT_CONFIG.llm.profiles,
                "runtime-owned": LLMProfile(model="runtime-model"),
            },
        ),
    )
    runtime = Runtime.open("local", config=runtime_config)
    profiles = tmp_path / "llm-profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": {
                    "runtime-owned": {
                        "model": "shadow-model",
                        "api_key_env": "SHADOW_API_KEY",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        with pytest.raises(ValidationError, match="must match the supplied Runtime config"):
            GuiRuntimeService(
                runtime=runtime,
                config=DEFAULT_CONFIG,
                auto_run=False,
                llm_profiles_file=profiles,
            )
        assert runtime.llms.profile("runtime-owned").model == "runtime-model"
    finally:
        runtime.shutdown(actor="test", reason="test.cleanup")


def test_terminal_purge_keeps_later_gui_presentations_outside_run_links(
    tmp_path: Path,
) -> None:
    """A later Host GUI session must not reopen a purged Run payload scope."""

    database = tmp_path / "post-terminal-gui-presentation.sqlite"
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    runtime = Runtime.open(database, config=config)
    service: GuiRuntimeService | None = None
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal={"goal": "POST_TERMINAL_GUI_PRESENTATION_GOAL"},
                display_title="Post-terminal GUI presentation",
                image_id="base-agent:v0",
                retention=TaskRunRetention.PURGE_ON_TERMINAL,
            ),
            client_request_id="create:post-terminal-gui-presentation",
        )
        assert created.root_pid is not None
        pid = created.root_pid
        runtime.capability.grant(
            pid,
            runtime.config.runtime.default_human_resource,
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        request_id = runtime.human.query(
            pid,
            runtime.config.runtime.default_human,
            {
                "type": "question",
                "question": "POST_TERMINAL_GUI_PRESENTATION_CANARY",
            },
            blocking=False,
        )
        terminal = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel:post-terminal-gui-presentation",
        )
        record = runtime.store.get_task_run(created.run_id)
        assert terminal.status is TaskRunStatus.CANCELLED
        assert record is not None and record.payloads_purged_at is not None

        def presentation_effects(selected_runtime: Runtime) -> list[Any]:
            return sorted(
                (
                    effect
                    for effect in selected_runtime.store.list_external_effects(
                        pid=pid
                    )
                    if effect.provider == "human"
                    and effect.operation == "write"
                    and effect.provider_metadata.get("context", {}).get("purpose")
                    == "gui_presentation"
                    and effect.provider_metadata.get("request_id") == request_id
                ),
                key=lambda effect: (effect.created_at, effect.effect_id),
            )

        def operation_id_for(selected_runtime: Runtime, effect_id: str) -> str:
            evidence = selected_runtime.uow.evidence.list_operation_evidence(
                evidence_types=("external_effect",),
                evidence_id=effect_id,
                limit=2,
            )
            assert len(evidence) == 1 and evidence[0].role == "effect"
            return evidence[0].operation_id

        service = GuiRuntimeService(
            runtime=runtime,
            auto_run=False,
            token="post-terminal-session-one",
        )
        first_effects = presentation_effects(runtime)
        assert len(first_effects) == 1
        first_effect = first_effects[0]

        provider_type = type(service._human_presentation_provider)
        service._human_presentation_provider = provider_type()
        service.snapshot()
        exact_effects = presentation_effects(runtime)
        assert len(exact_effects) == 2
        legacy_effect = exact_effects[1]
        first_operation_id = operation_id_for(runtime, first_effect.effect_id)
        legacy_operation_id = operation_id_for(runtime, legacy_effect.effect_id)

        # Simulate a development build that had already projected the later
        # GUI observation.  Both links are append-only and must survive repair.
        legacy_effect_item = runtime.store.append_task_run_ledger_item(
            TaskRunLedgerItem(
                item_id=f"legacy-effect-{legacy_effect.effect_id}",
                run_id=created.run_id,
                seq=0,
                kind=TaskRunLedgerKind.EFFECT,
                status="finalized:committed",
                label="Legacy post-purge GUI effect projection",
                occurred_at=utc_now(),
                pid=pid,
                effect_id=legacy_effect.effect_id,
            )
        )
        legacy_effect_link = TaskRunLink(
            link_id=f"legacy-effect-link-{legacy_effect.effect_id}",
            run_id=created.run_id,
            ledger_seq=legacy_effect_item.seq,
            evidence_type="external_effect",
            evidence_id=legacy_effect.effect_id,
            role="effect",
            created_at=legacy_effect_item.occurred_at,
        )
        runtime.store.insert_task_run_link(legacy_effect_link)
        legacy_operation_item = runtime.store.append_task_run_ledger_item(
            TaskRunLedgerItem(
                item_id=f"legacy-operation-{legacy_operation_id}",
                run_id=created.run_id,
                seq=0,
                kind=TaskRunLedgerKind.PROCESS,
                status="succeeded",
                label="Legacy post-purge GUI operation projection",
                occurred_at=utc_now(),
                pid=pid,
                operation_id=legacy_operation_id,
            )
        )
        legacy_operation_link = TaskRunLink(
            link_id=f"legacy-operation-link-{legacy_operation_id}",
            run_id=created.run_id,
            ledger_seq=legacy_operation_item.seq,
            evidence_type="operation",
            evidence_id=legacy_operation_id,
            role="operation",
            created_at=legacy_operation_item.occurred_at,
        )
        runtime.store.insert_task_run_link(legacy_operation_link)

        class StateChangingGuiProvider(provider_type):
            @staticmethod
            def classify_external_effect(
                operation: str,
                context: dict[str, Any],
                result: Any,
            ) -> ExternalEffectClassification:
                classification = provider_type.classify_external_effect(
                    operation,
                    context,
                    result,
                )
                return replace(classification, state_mutation=True)

        service._human_presentation_provider = StateChangingGuiProvider()
        service.snapshot()
        all_presentations = presentation_effects(runtime)
        assert len(all_presentations) == 3
        disguised_effect = all_presentations[2]
        disguised_operation_id = operation_id_for(
            runtime,
            disguised_effect.effect_id,
        )
        assert disguised_effect.state_mutation is True

        runtime.task_runs.list_ledger(created.run_id, limit=100)
        links = runtime.store.list_task_run_links(created.run_id)
        linked = {(link.evidence_type, link.evidence_id) for link in links}

        assert ("external_effect", first_effect.effect_id) not in linked
        assert ("operation", first_operation_id) not in linked
        assert ("external_effect", legacy_effect.effect_id) in linked
        assert ("operation", legacy_operation_id) in linked
        assert ("external_effect", disguised_effect.effect_id) in linked
        assert ("operation", disguised_operation_id) in linked
        assert legacy_effect_link in links
        assert legacy_operation_link in links

        retained_legacy = runtime.store.get_external_effect(
            legacy_effect.effect_id
        )
        retained_disguised = runtime.store.get_external_effect(
            disguised_effect.effect_id
        )
        retained_first = runtime.store.get_external_effect(first_effect.effect_id)
        assert retained_legacy is not None
        assert retained_disguised is not None
        assert retained_first is not None
        assert (
            external_effect_payload_retention_tier(retained_legacy)
            is PayloadRetentionTier.HASH_ONLY
        )
        assert (
            external_effect_payload_retention_tier(retained_disguised)
            is PayloadRetentionTier.HASH_ONLY
        )
        assert (
            external_effect_payload_retention_tier(retained_first)
            is PayloadRetentionTier.FULL
        )
        assert retained_first.record_id is not None
        assert retained_first.event_id is not None

        assert service.shutdown(timeout_s=2.0) is True
        service = None
    finally:
        if service is not None:
            service.shutdown(timeout_s=2.0)
        runtime.close()

    reopened = Runtime.open(database, config=config)
    reopened_service: GuiRuntimeService | None = None
    try:
        before_reopen_session = {
            effect.effect_id for effect in reopened.store.list_external_effects(pid=pid)
        }
        reopened_service = GuiRuntimeService(
            runtime=reopened,
            auto_run=False,
            token="post-terminal-session-two",
        )
        after_reopen_session = {
            effect.effect_id for effect in reopened.store.list_external_effects(pid=pid)
        }
        new_effect_ids = after_reopen_session - before_reopen_session
        assert len(new_effect_ids) == 1
        reopened_effect_id = next(iter(new_effect_ids))
        reopened_operation_id = operation_id_for(reopened, reopened_effect_id)

        reopened.task_runs.list_ledger(created.run_id, limit=100)
        reopened_links = reopened.store.list_task_run_links(created.run_id)
        reopened_linked = {
            (link.evidence_type, link.evidence_id) for link in reopened_links
        }
        assert ("external_effect", first_effect.effect_id) not in reopened_linked
        assert ("operation", first_operation_id) not in reopened_linked
        assert ("external_effect", reopened_effect_id) not in reopened_linked
        assert ("operation", reopened_operation_id) not in reopened_linked
        assert legacy_effect_link in reopened_links
        assert legacy_operation_link in reopened_links
        assert ("external_effect", disguised_effect.effect_id) in reopened_linked
        assert ("operation", disguised_operation_id) in reopened_linked
        repaired_legacy = reopened.store.get_external_effect(
            legacy_effect.effect_id
        )
        repaired_disguised = reopened.store.get_external_effect(
            disguised_effect.effect_id
        )
        assert repaired_legacy is not None
        assert repaired_disguised is not None
        assert (
            external_effect_payload_retention_tier(repaired_legacy)
            is PayloadRetentionTier.HASH_ONLY
        )
        assert (
            external_effect_payload_retention_tier(repaired_disguised)
            is PayloadRetentionTier.HASH_ONLY
        )
        assert reopened_service.shutdown(timeout_s=2.0) is True
        reopened_service = None
    finally:
        if reopened_service is not None:
            reopened_service.shutdown(timeout_s=2.0)
        reopened.close()

def _gui_provider_trace_record(
    pid: str,
    call_id: str,
    *,
    created_at: str = "2026-08-03T00:00:00+00:00",
    reasoning_text: str = "Provider reasoning",
) -> LLMCallRecord:
    trace = {
        "kind": "provider_trace",
        "schema_version": 1,
        "coverage": "complete",
        "selected_attempt": 1,
        "limited": False,
        "omitted_attempts": 0,
        "attempts": [
            {
                "sequence": 1,
                "kind": "initial",
                "api": "responses",
                "status": "ok",
                "reasoning": {
                    "availability": "returned",
                    "blocks": [
                        {
                            "type": "reasoning_text",
                            "source": "output.reasoning",
                            "text": reasoning_text,
                        }
                    ],
                },
                "output": "final answer",
                "tool_calls": [
                    {
                        "id": "tool-call-1",
                        "name": "read_text_file",
                        "arguments": '{"path":"PRIVATE_TOOL_ARGUMENT"}',
                    }
                ],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 7,
                    "total_tokens": 19,
                    "ignored_provider_metric": "PRIVATE_USAGE_VALUE",
                },
                "model": "trace-model",
                "request_id": "request-1",
                "response_id": "response-1",
                "started_at": created_at,
                "completed_at": created_at,
                "duration_ms": 4,
                "error": None,
            }
        ],
    }
    return LLMCallRecord(
        call_id=call_id,
        pid=pid,
        image_id="base-agent:v0",
        purpose="agent_action",
        status="ok",
        api="responses",
        model="trace-model",
        request_id="request-1",
        response_id="response-1",
        messages=[{"role": "user", "content": "PRIVATE_PROMPT"}],
        tools=[{"name": "read_text_file", "description": "PRIVATE_TOOL_SCHEMA"}],
        request_options={
            "provider_trace_summary": {
                "schema_version": 1,
                "coverage": "complete",
                "attempt_count": 1,
                "recorded_attempt_count": 1,
                "selected_attempt": 1,
                "status_counts": {"ok": 1, "error": 0},
                "limited": False,
                "omitted_attempts": 0,
            },
            "authorization": "PRIVATE_AUTHORIZATION",
        },
        response_content="final answer",
        tool_calls=[{"name": "read_text_file", "arguments": {"path": "PRIVATE_TOOL_ARGUMENT"}}],
        reasoning=trace,
        usage={"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
        raw_response={
            "id": "response-1",
            "encrypted_content": "PRIVATE_ENCRYPTED_BLOB",
            "output_text": "final answer",
        },
        observability={},
        error=None,
        created_at=created_at,
        completed_at=created_at,
    )


class TestGuiServer:

    def setup_method(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.llm_profiles_file = Path(self.temp_dir.name) / 'llm-profiles.json'
        self.server = create_gui_http_server(
            db='local',
            port=0,
            token='test-token',
            auto_run=False,
            llm_profiles_file=self.llm_profiles_file,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def teardown_method(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.service.shutdown()
        self.server.server_close()
        self.temp_dir.cleanup()

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        token: str = 'test-token',
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        conn = http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=_GUI_TEST_HTTP_TIMEOUT_S,
        )
        headers = {'Authorization': f'Bearer {token}'}
        headers.update(extra_headers or {})
        payload = None
        if body is not None:
            payload = json.dumps(body).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        decoded = json.loads(data.decode('utf-8')) if data else None
        return (response.status, decoded)

    def request_raw(
        self,
        method: str,
        path: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=_GUI_TEST_HTTP_TIMEOUT_S,
        )
        headers = {'Authorization': 'Bearer test-token'}
        headers.update(extra_headers or {})
        conn.request(method, path, headers=headers)
        response = conn.getresponse()
        data = response.read()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        conn.close()
        return response.status, response_headers, data

    def request_json_text(self, method: str, path: str, raw: str) -> tuple[int, Any]:
        return self.request_json_bytes(method, path, raw.encode('utf-8'))

    def request_json_bytes(self, method: str, path: str, raw: bytes) -> tuple[int, Any]:
        conn = http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=_GUI_TEST_HTTP_TIMEOUT_S,
        )
        conn.request(
            method,
            path,
            body=raw,
            headers={'Authorization': 'Bearer test-token', 'Content-Type': 'application/json'},
        )
        response = conn.getresponse()
        data = response.read()
        conn.close()
        decoded = json.loads(data.decode('utf-8')) if data else None
        return response.status, decoded

    def test_capability_list_limit_is_validated_and_applied_before_decode(self) -> None:
        runtime = self.server.service.runtime
        configured_limit = runtime.config.capability.list_limit
        for index in range(configured_limit + 25):
            runtime.capability.grant(
                "gui-capability-list-subject",
                f"object:gui-capability-list-{index:04d}",
                [CapabilityRight.READ],
                issued_by="test",
            )

        decoded: list[str] = []
        decode = runtime.store._row_to_capability
        runtime.store._row_to_capability = lambda row: (
            decoded.append(str(row["cap_id"])),
            decode(row),
        )[1]

        status, default_page = self.request("GET", "/api/capabilities")
        assert status == 200
        assert len(default_page) == configured_limit
        assert len(decoded) == configured_limit

        decoded.clear()
        status, selected_page = self.request("GET", "/api/capabilities?limit=7")
        assert status == 200
        assert len(selected_page) == 7
        assert len(decoded) == 7

        decoded.clear()
        status, subject_page = self.request(
            "GET",
            "/api/capabilities?subject=gui-capability-list-subject&limit=7",
        )
        assert status == 200
        assert len(subject_page) == 7
        assert len(decoded) == 7

        for invalid in (0, configured_limit + 1):
            status, body = self.request(
                "GET",
                f"/api/capabilities?limit={invalid}",
            )
            assert status == 400
            assert "limit must" in body["error"]["message"]

        # Existing clients continue to receive the legacy array, while the
        # explicit page mode exposes a cursor envelope for complete GUI walks.
        assert isinstance(subject_page, list)
        runtime.store._row_to_capability = decode
        expected_count = configured_limit + 25
        seen_ids: list[str] = []
        after: str | None = None
        while True:
            page_path = (
                "/api/capabilities?mode=page"
                "&subject=gui-capability-list-subject&limit=7"
            )
            if after is not None:
                page_path += f"&after={after}"
            status, page = self.request("GET", page_path)
            assert status == 200
            assert set(page) == {"items", "next_after", "has_more"}
            page_ids = [item["cap_id"] for item in page["items"]]
            assert len(page_ids) <= 7
            assert not set(page_ids).intersection(seen_ids)
            seen_ids.extend(page_ids)
            if not page["has_more"]:
                assert page["next_after"] is None
                break
            assert page["next_after"]
            assert page["next_after"] != after
            after = page["next_after"]

        assert len(seen_ids) == expected_count

    def test_auth_health_snapshot_and_process_flow(self) -> None:
        status, _body = self.request('GET', '/api/health', token='wrong')
        assert status == 401
        status, health = self.request('GET', '/api/health')
        assert status == 200
        assert health['ok']
        assert not health['scheduler']['auto_run']
        assert health['scheduler']['default_max_quanta'] is None
        status, _profile = self.request(
            'POST',
            '/api/llm-profiles',
            {'profile_id': 'gui-spawn', 'model': 'gui-spawn-model', 'api_key_env': 'GUI_SPAWN_API_KEY'},
        )
        assert status == 200
        status, spawned = self.request(
            'POST',
            '/api/processes',
            {'goal': 'inspect README', 'auto_run': False, 'llm_profile': 'gui-spawn'},
        )
        assert status == 200
        pid = spawned['pid']
        assert spawned['process']['llm_profile_id'] == 'gui-spawn'
        status, message = self.request('POST', f'/api/processes/{pid}/message', {'body': 'hello', 'auto_run': False})
        assert status == 200
        assert message['message']['body'] == 'hello'
        status, interrupt = self.request('POST', f'/api/processes/{pid}/interrupt', {'body': 'stop', 'auto_run': False})
        assert status == 200
        assert interrupt['message']['kind'] == 'interrupt'
        status, snapshot = self.request('GET', '/api/snapshot')
        assert status == 200
        schema = json.loads((Path(__file__).resolve().parents[2] / 'docs' / 'gui_api_schema.json').read_text(encoding='utf-8'))
        Draft202012Validator(
            {
                '$schema': schema['$schema'],
                '$defs': schema['$defs'],
                '$ref': '#/$defs/snapshotResponse',
            }
        ).validate(snapshot)
        assert len(snapshot['processes']) == 1
        assert snapshot['processes'][0]['llm_profile_id'] == 'gui-spawn'
        assert snapshot['processes'][0]['unread_message_count'] >= 2
        assert 'tools' in snapshot
        assert 'images' in snapshot
        assert any((profile['profile_id'] == 'gui-spawn' for profile in snapshot['llm_profiles']))
        assert any((image['image_id'] == 'base-agent:v0' for image in snapshot['images']))

    def test_llm_trace_api_keeps_snapshots_content_free_and_chunks_on_demand(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image="base-agent:v0", goal="trace API")
        reasoning_text = "推理轨迹" * 12_000
        record = _gui_provider_trace_record(
            pid,
            "llm-call-trace-api",
            reasoning_text=reasoning_text,
        )
        record = replace(
            record,
            raw_response={
                **(record.raw_response or {}),
                "headers": [
                    ["Authorization", "Bearer PRIVATE_HEADER_AUTHORIZATION"],
                    [
                        "Authorization",
                        "Bearer PRIVATE_HEADER_TRIPLE_AUTHORIZATION",
                        "PRIVATE_HEADER_TRIPLE_METADATA",
                    ],
                    ["x-api-key", "PRIVATE_HEADER_API_KEY"],
                    ["content-type", "application/json"],
                    {
                        "name": "Authorization",
                        "value": "Bearer PRIVATE_HEADER_OBJECT_AUTHORIZATION",
                    },
                    {
                        "key": "set-cookie",
                        "data": ["PRIVATE_HEADER_OBJECT_COOKIE"],
                    },
                ],
            },
        )
        runtime.store.insert_llm_call(record)
        self.server.service.publish_runtime_changes("test.llm_trace")
        replayed = dumps(
            [
                {"event": event.event, "data": event.data}
                for event in self.server.service.broadcaster.replay_after(0)
                if event.event in {"snapshot", "llm_call.appended"}
            ]
        )
        assert reasoning_text not in replayed
        assert "PRIVATE_PROMPT" not in replayed
        assert "PRIVATE_TOOL_ARGUMENT" not in replayed

        snapshot_status, snapshot = self.request("GET", "/api/snapshot")
        page_status, page = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls?limit=50",
        )
        detail_status, detail = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls/{record.call_id}",
        )

        assert snapshot_status == page_status == detail_status == 200
        assert snapshot["schema_version"] == 3
        assert page == {
            "schema_version": 1,
            "items": [page["items"][0]],
            "next_cursor": None,
            "has_more": False,
        }
        summary = page["items"][0]
        assert summary["attempt_count"] == 1
        assert summary["coverage"] == "complete"
        assert summary["reasoning_availability"] == "returned"
        assert summary["payload_retention_tier"] == "full"
        assert detail["call"] == summary
        assert detail["attempts"][0]["tool_names"] == ["read_text_file"]
        assert detail["attempts"][0]["reasoning_blocks"][0]["type"] == "reasoning_text"

        outward = dumps({"snapshot": snapshot, "page": page, "detail": detail})
        for private in (
            "PRIVATE_PROMPT",
            "PRIVATE_TOOL_SCHEMA",
            "PRIVATE_TOOL_ARGUMENT",
            "PRIVATE_AUTHORIZATION",
            "PRIVATE_ENCRYPTED_BLOB",
            reasoning_text,
        ):
            assert private not in outward

        descriptor = next(
            item
            for item in detail["content"]
            if item["field"] == "attempt_reasoning"
        )
        assert descriptor["availability"] == "available"
        assert descriptor["cursor"]
        assembled = ""
        cursor = descriptor["cursor"]
        while cursor is not None:
            status, chunk = self.request(
                "GET",
                f"/api/processes/{pid}/llm-calls/{record.call_id}/content"
                f"?field=attempt_reasoning&attempt_sequence=1&limit=32768&cursor={cursor}",
            )
            assert status == 200
            assembled += chunk["content"]
            cursor = chunk["next_cursor"]
        assert assembled == reasoning_text

        raw_descriptor = next(
            item for item in detail["content"] if item["field"] == "raw_response"
        )
        status, raw_chunk = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls/{record.call_id}/content"
            f"?field=raw_response&cursor={raw_descriptor['cursor']}",
        )
        assert status == 200
        assert "PRIVATE_ENCRYPTED_BLOB" not in raw_chunk["content"]
        assert "PRIVATE_HEADER_AUTHORIZATION" not in raw_chunk["content"]
        assert "PRIVATE_HEADER_TRIPLE_AUTHORIZATION" not in raw_chunk["content"]
        assert "PRIVATE_HEADER_TRIPLE_METADATA" not in raw_chunk["content"]
        assert "PRIVATE_HEADER_API_KEY" not in raw_chunk["content"]
        assert "PRIVATE_HEADER_OBJECT_AUTHORIZATION" not in raw_chunk["content"]
        assert "PRIVATE_HEADER_OBJECT_COOKIE" not in raw_chunk["content"]
        assert '"kind": "redacted"' in raw_chunk["content"]
        assert '"content-type"' in raw_chunk["content"]
        assert '"application/json"' in raw_chunk["content"]

        status, headers, _payload = self.request_raw(
            "GET",
            f"/api/processes/{pid}/llm-calls/{record.call_id}",
        )
        assert status == 200
        assert headers["cache-control"] == "no-store"
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"

    @pytest.mark.parametrize(
        ("provider_status", "expected_status"),
        [
            (429, 429),
            (True, None),
            ("429", None),
            (99, None),
            (600, None),
        ],
    )
    def test_llm_trace_attempt_error_only_exposes_valid_http_status_codes(
        self,
        provider_status: Any,
        expected_status: int | None,
    ) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image="base-agent:v0", goal="trace error")
        record = _gui_provider_trace_record(
            pid,
            f"llm-call-status-{str(provider_status).lower()}",
        )
        assert isinstance(record.reasoning, dict)
        attempt = record.reasoning["attempts"][0]
        attempt["status"] = "error"
        attempt["error"] = {
            "error_type": "ProviderStatusError",
            "message_bytes": 17,
            "message_sha256": "a" * 64,
            "status_code": provider_status,
        }
        runtime.store.insert_llm_call(record)

        status, detail = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls/{record.call_id}",
        )
        assert status == 200
        assert detail["attempts"][0]["error"] == {
            "error_type": "ProviderStatusError",
            "message_bytes": 17,
            "message_sha256": "a" * 64,
            "status_code": expected_status,
        }

    def test_legacy_llm_reasoning_only_reveals_explicit_readable_blocks(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image="base-agent:v0", goal="legacy trace")
        record = replace(
            _gui_provider_trace_record(pid, "llm-call-legacy-reasoning"),
            request_options={},
            reasoning={
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "LEGACY_READABLE_SUMMARY",
                    }
                ],
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": "LEGACY_READABLE_REASONING",
                    }
                ],
                "encrypted_content": "LEGACY_ENCRYPTED_SECRET",
                "signature": "LEGACY_SIGNATURE_SECRET",
                "opaque_blob": {"text": "LEGACY_OPAQUE_SECRET"},
                "unrelated": {"text": "LEGACY_UNRELATED_SECRET"},
            },
            raw_response={
                "apiKey": "RAW_API_KEY_SECRET",
                "access_token": "RAW_ACCESS_TOKEN_SECRET",
                "refreshToken": "RAW_REFRESH_TOKEN_SECRET",
                "id_token": "RAW_ID_TOKEN_SECRET",
                "session_token": "RAW_SESSION_TOKEN_SECRET",
                "Cookie": "RAW_COOKIE_SECRET",
                "Set-Cookie": "RAW_SET_COOKIE_SECRET",
                "input_tokens": 17,
            },
        )
        runtime.store.insert_llm_call(record)

        page_status, page = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls?limit=50",
        )
        detail_status, detail = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls/{record.call_id}",
        )
        assert page_status == detail_status == 200
        assert page["items"][0]["coverage"] == "legacy_final_only"
        assert detail["attempts"][0]["reasoning_availability"] == "returned"
        block_types = {
            block["type"]
            for block in detail["attempts"][0]["reasoning_blocks"]
        }
        assert {"summary_text", "reasoning_text", "opaque"} <= block_types

        outward = dumps({"page": page, "detail": detail})
        for private in (
            "LEGACY_READABLE_SUMMARY",
            "LEGACY_READABLE_REASONING",
            "LEGACY_ENCRYPTED_SECRET",
            "LEGACY_SIGNATURE_SECRET",
            "LEGACY_OPAQUE_SECRET",
            "LEGACY_UNRELATED_SECRET",
        ):
            assert private not in outward

        reasoning_descriptor = next(
            item
            for item in detail["content"]
            if item["field"] == "attempt_reasoning"
        )
        status, reasoning = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls/{record.call_id}/content"
            "?field=attempt_reasoning&attempt_sequence=1"
            f"&cursor={reasoning_descriptor['cursor']}",
        )
        assert status == 200
        assert set(reasoning["content"].split("\n\n")) == {
            "LEGACY_READABLE_SUMMARY",
            "LEGACY_READABLE_REASONING",
        }
        for private in (
            "LEGACY_ENCRYPTED_SECRET",
            "LEGACY_SIGNATURE_SECRET",
            "LEGACY_OPAQUE_SECRET",
            "LEGACY_UNRELATED_SECRET",
        ):
            assert private not in reasoning["content"]

        raw_descriptor = next(
            item for item in detail["content"] if item["field"] == "raw_response"
        )
        status, raw = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls/{record.call_id}/content"
            f"?field=raw_response&cursor={raw_descriptor['cursor']}",
        )
        assert status == 200
        for private in (
            "RAW_API_KEY_SECRET",
            "RAW_ACCESS_TOKEN_SECRET",
            "RAW_REFRESH_TOKEN_SECRET",
            "RAW_ID_TOKEN_SECRET",
            "RAW_SESSION_TOKEN_SECRET",
            "RAW_COOKIE_SECRET",
            "RAW_SET_COOKIE_SECRET",
        ):
            assert private not in raw["content"]
        assert raw["content"].count('"kind": "redacted"') == 7
        assert '"input_tokens": 17' in raw["content"]

    def test_legacy_responses_reasoning_configuration_is_not_provider_content(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image="base-agent:v0", goal="legacy config")
        record = replace(
            _gui_provider_trace_record(pid, "llm-call-legacy-config"),
            request_options={},
            reasoning={"effort": "high", "summary": "auto"},
        )
        runtime.store.insert_llm_call(record)

        status, detail = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls/{record.call_id}",
        )
        assert status == 200
        assert detail["call"]["coverage"] == "legacy_final_only"
        assert detail["call"]["reasoning_availability"] == "not_returned"
        assert detail["attempts"][0]["reasoning_availability"] == "not_returned"
        assert detail["attempts"][0]["reasoning_blocks"] == []
        descriptor = next(
            item
            for item in detail["content"]
            if item["field"] == "attempt_reasoning"
        )
        assert descriptor["availability"] == "not_returned"
        assert descriptor["cursor"] is None
        assert "auto" not in dumps(detail)

    def test_llm_trace_list_uses_stable_keyset_and_rejects_cross_process_reads(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image="base-agent:v0", goal="trace pagination")
        other_pid = runtime.process.spawn(image="base-agent:v0", goal="other trace")
        for call_id in ("llm-call-a", "llm-call-b", "llm-call-c"):
            runtime.store.insert_llm_call(_gui_provider_trace_record(pid, call_id))
        runtime.store.insert_llm_call(
            _gui_provider_trace_record(other_pid, "llm-call-other")
        )

        status, first = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls?limit=2",
        )
        assert status == 200
        assert [item["call_id"] for item in first["items"]] == [
            "llm-call-c",
            "llm-call-b",
        ]
        assert first["has_more"] is True

        runtime.store.insert_llm_call(
            _gui_provider_trace_record(
                pid,
                "llm-call-newer",
                created_at="2026-08-03T01:00:00+00:00",
            )
        )
        status, second = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls?limit=2&cursor={first['next_cursor']}",
        )
        assert status == 200
        assert [item["call_id"] for item in second["items"]] == ["llm-call-a"]
        assert second["has_more"] is False

        status, _cross = self.request(
            "GET",
            f"/api/processes/{other_pid}/llm-calls/llm-call-a",
        )
        assert status == 404
        status, detail = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls/llm-call-a",
        )
        assert status == 200
        messages_cursor = next(
            item for item in detail["content"] if item["field"] == "messages"
        )["cursor"]
        for scoped_pid, scoped_call in (
            (pid, "llm-call-b"),
            (other_pid, "llm-call-other"),
        ):
            status, wrong_scope = self.request(
                "GET",
                f"/api/processes/{scoped_pid}/llm-calls/{scoped_call}/content"
                f"?field=messages&cursor={messages_cursor}",
            )
            assert status == 404
            assert wrong_scope["error"]["code"] == "llm_call_not_found"
        cursor_prefix, signature = first["next_cursor"].rsplit(".", maxsplit=1)
        tampered = f"{cursor_prefix}.{_noncanonical_base64url_alias(signature)}"
        status, invalid = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls?limit=2&cursor={tampered}",
        )
        assert status == 400
        assert invalid["error"]["code"] == "invalid_cursor"

    def test_llm_trace_content_cursor_detects_retention_change(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image="base-agent:v0", goal="trace retention")
        record = _gui_provider_trace_record(pid, "llm-call-retention")
        runtime.store.insert_llm_call(record)
        status, detail = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls/{record.call_id}",
        )
        assert status == 200
        descriptor = next(
            item for item in detail["content"] if item["field"] == "response_content"
        )
        assert descriptor["cursor"]

        expected_sha256 = llm_call_payload_sha256(record)
        retained = retain_llm_call_payload(
            record,
            PayloadRetentionTier.SUMMARY,
            provider_chain_head=False,
        )
        assert runtime.store.update_llm_call_payload_retention(
            retained,
            expected_payload_sha256=expected_sha256,
            expected_tier=PayloadRetentionTier.FULL,
        )

        status, changed = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls/{record.call_id}/content"
            f"?field=response_content&cursor={descriptor['cursor']}",
        )
        assert status == 409
        assert changed["error"]["code"] == "content_changed"

        status, refreshed = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls/{record.call_id}",
        )
        assert status == 200
        assert refreshed["call"]["payload_retention_tier"] == "summary"
        assert refreshed["call"]["reasoning_availability"] == "not_persisted"
        assert refreshed["attempts"] == []
        response_descriptor = next(
            item for item in refreshed["content"] if item["field"] == "response_content"
        )
        assert response_descriptor["availability"] == "not_persisted"
        assert response_descriptor["cursor"] is None

    def test_process_handlers_preserve_typed_wait_and_outcome_discriminators(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = self.server.service.runtime
        status, spawned = self.request(
            "POST",
            "/api/processes",
            {"goal": "typed GUI state", "auto_run": False},
        )
        assert status == 200
        waiting_pid = spawned["pid"]
        runtime.process.pause(waiting_pid, "review")

        status, listed = self.request("GET", "/api/processes")
        assert status == 200
        waiting = next(process for process in listed if process["pid"] == waiting_pid)
        assert waiting["wait_state"]["schema_version"] == 1
        assert waiting["wait_state"]["kind"] == "paused"
        assert waiting["outcome"] is None

        monkeypatch.setattr(
            runtime,
            "set_process_working_directory",
            lambda *args, **kwargs: runtime.process.get(waiting_pid),
        )
        status, changed_directory = self.request(
            "POST",
            f"/api/processes/{waiting_pid}/cd",
            {"path": "."},
        )
        assert status == 200
        assert changed_directory["wait_state"]["schema_version"] == 1
        assert changed_directory["wait_state"]["kind"] == "paused"

        monkeypatch.setattr(
            runtime,
            "exec_process",
            lambda *args, **kwargs: runtime.process.get(waiting_pid),
        )
        status, executed = self.request(
            "POST",
            f"/api/processes/{waiting_pid}/exec",
            {
                "image": "base-agent:v0",
                "goal": "typed exec response",
                "confirmed": True,
                "auto_run": False,
            },
        )
        assert status == 200
        assert executed["process"]["wait_state"]["schema_version"] == 1
        assert executed["process"]["wait_state"]["kind"] == "paused"

        status, terminal_spawned = self.request(
            "POST",
            "/api/processes",
            {"goal": "typed GUI outcome", "auto_run": False},
        )
        assert status == 200
        terminal_pid = terminal_spawned["pid"]
        runtime.process.exit(terminal_pid, message="done")
        status, terminal = self.request("GET", f"/api/processes/{terminal_pid}")
        assert status == 200
        assert terminal["wait_state"] is None
        assert terminal["outcome"] == {
            "schema_version": 1,
            "kind": "exited",
            "result_oid": terminal["outcome"]["result_oid"],
        }
        assert terminal["outcome"]["result_oid"]

    def test_operation_list_detail_and_evidence_resolution_endpoints(self) -> None:
        status, created = self.request(
            'POST',
            '/api/processes',
            {'goal': 'explain endpoint', 'image': 'base-agent:v0', 'auto_run': False},
        )
        assert status == 200
        pid = created['pid']

        status, listed = self.request('GET', f'/api/operations?pid={pid}&limit=100')
        assert status == 200
        operation = next(item for item in listed['operations'] if item['name'] == 'process.spawn')
        status, explained = self.request('GET', f"/api/operations/{operation['operation_id']}")
        assert status == 200
        assert explained['root']['operation_id'] == operation['operation_id']
        assert explained['evidence_complete'] is True
        audit = next(item for item in explained['evidence'] if item['evidence_type'] == 'audit')

        status, resolved = self.request(
            'GET',
            f"/api/operations/resolve?kind=audit&id={audit['evidence_id']}",
        )
        assert status == 200
        assert resolved['root']['operation_id'] == operation['operation_id']
        status, missing = self.request('GET', '/api/operations/op_missing')
        assert status == 404
        assert missing['error']['type'] == 'NotFound'

        runtime = self.server.service.runtime
        first = runtime.operations.start(kind='runtime', name='first', actor=pid, pid=pid)
        second = runtime.operations.start(kind='runtime', name='second', actor=pid, pid=pid)
        runtime.operations.link_evidence('audit', 'shared-http-audit', 'audit', operation_id=first.operation_id)
        runtime.operations.link_evidence('audit', 'shared-http-audit', 'audit', operation_id=second.operation_id)
        runtime.operations.finish('succeeded', operation_id=first.operation_id)
        runtime.operations.finish('succeeded', operation_id=second.operation_id)
        status, ambiguous = self.request(
            'GET',
            '/api/operations/resolve?kind=audit&id=shared-http-audit',
        )
        assert status == 409
        assert set(ambiguous['error']['candidates']) == {first.operation_id, second.operation_id}

    def test_snapshot_keeps_new_pending_human_request_ahead_of_bounded_history(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='pending must stay visible')
        now = utc_now()
        for index in range(runtime.config.gui.snapshot_collection_max_items + 1):
            runtime.store.insert_human_request(
                HumanRequest(
                    request_id=f'hreq_history_{index:04d}',
                    pid=pid,
                    human='owner',
                    payload={'type': 'question', 'question': f'history {index}'},
                    status=HumanRequestStatus.REJECTED,
                    decision={'approved': False},
                    blocking=False,
                    created_at=now,
                    updated_at=now,
                )
            )
        pending_id = 'hreq_pending_latest'
        runtime.store.insert_human_request(
            HumanRequest(
                request_id=pending_id,
                pid=pid,
                human='owner',
                payload={'type': 'question', 'question': 'must remain visible'},
                status=HumanRequestStatus.PENDING,
                decision=None,
                blocking=True,
                created_at=now,
                updated_at=now,
            )
        )

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        assert pending_id in {request['request_id'] for request in snapshot['human_requests']}

    def test_repeated_snapshot_reuses_unchanged_human_presentation_evidence(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='reuse an unchanged GUI presentation',
        )
        request_id = runtime.human.query(
            pid,
            runtime.config.runtime.default_human,
            {'type': 'question', 'question': 'UNCHANGED_GUI_PRESENTATION'},
            blocking=False,
        )

        first = self.server.service.snapshot()
        effects_after_first = runtime.store.list_external_effects(pid=pid)
        events_after_first = len(runtime.events.list())
        audit_after_first = len(runtime.audit.trace())
        decisions_after_first = len(runtime.store.list_data_flow_decisions(pid=pid))

        second = self.server.service.snapshot()

        assert next(
            item for item in first['human_requests'] if item['request_id'] == request_id
        ) == next(
            item for item in second['human_requests'] if item['request_id'] == request_id
        )
        presentation_effects = [
            effect
            for effect in effects_after_first
            if effect.provider == 'human'
            and effect.provider_metadata.get('context', {}).get('purpose')
            == 'gui_presentation'
        ]
        assert len(presentation_effects) == 1
        assert runtime.store.list_external_effects(pid=pid) == effects_after_first
        assert len(runtime.events.list()) == events_after_first
        assert len(runtime.audit.trace()) == audit_after_first
        assert len(runtime.store.list_data_flow_decisions(pid=pid)) == decisions_after_first

    def test_delivered_output_snapshot_remains_visible_after_source_mutation_and_exit(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='present a frozen output after the process advances',
        )
        runtime.capability.grant(
            pid,
            runtime.config.runtime.default_human_resource,
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.PROCESS_STATE,
            {'step': 1},
            metadata=ObjectMetadata(sensitivity='normal'),
            immutable=False,
        )
        delivered: list[str] = []
        runtime.human.provider.output_sink = delivered.append
        sentinel = 'GUI_DELIVERED_OUTPUT_SNAPSHOT_SENTINEL'

        output = runtime.human.output(
            pid,
            sentinel,
            source_oids=[source.oid],
        )
        runtime.memory.update_object(
            pid,
            source,
            ObjectPatch(payload={'step': 2}),
            expected_version=1,
        )
        runtime.process.exit(pid, message='advance after Human output')

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        assert delivered == [sentinel]
        presented = next(
            item
            for item in snapshot['human_requests']
            if item['request_id'] == output['request_id']
        )
        assert presented['payload']['message'] == sentinel
        assert '_agent_libos_output_snapshot_sha256' not in presented['payload']
        assert runtime.human.is_request_withheld_for_presentation(
            output['request_id'],
            presentation='gui',
        ) is False

    def test_delivered_output_snapshot_digest_mismatch_fails_closed(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='reject a mutated delivered output snapshot',
        )
        runtime.capability.grant(
            pid,
            runtime.config.runtime.default_human_resource,
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        runtime.human.provider.output_sink = lambda _message: None
        output = runtime.human.output(pid, 'ORIGINAL_OUTPUT_SNAPSHOT')
        request = runtime.human.get(output['request_id'])
        request.payload = dict(request.payload)
        request.payload['message'] = 'MUTATED_OUTPUT_SNAPSHOT_SENTINEL'
        runtime.human.requests.update(request)

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        encoded = json.dumps(snapshot, sort_keys=True)
        assert 'MUTATED_OUTPUT_SNAPSHOT_SENTINEL' not in encoded
        presented = next(
            item
            for item in snapshot['human_requests']
            if item['request_id'] == output['request_id']
        )
        assert presented['payload']['release_required'] is True
        assert runtime.human.is_request_withheld_for_presentation(
            output['request_id'],
            presentation='gui',
        ) is True

    def test_delivered_output_snapshot_still_obeys_current_sink_clearance(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='retain labels on a frozen output snapshot',
        )
        runtime.capability.grant(
            pid,
            runtime.config.runtime.default_human_resource,
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        pattern = f'human:{human}:{channel}'
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=pattern,
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity='secret',
            ),
            actor='test.host',
            replace=True,
            require_capability=False,
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_FROZEN_SECRET_OUTPUT_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        runtime.human.provider.output_sink = lambda _message: None
        output = runtime.human.output(
            pid,
            'GUI_FROZEN_SECRET_OUTPUT_SENTINEL',
            source_oids=[source.oid],
        )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=pattern,
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity='normal',
            ),
            actor='test.host',
            replace=True,
            require_capability=False,
        )

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        encoded = json.dumps(snapshot, sort_keys=True)
        assert 'GUI_FROZEN_SECRET_OUTPUT_SENTINEL' not in encoded
        presented = next(
            item
            for item in snapshot['human_requests']
            if item['request_id'] == output['request_id']
        )
        assert presented['payload']['release_required'] is True

    def test_new_gui_presentation_session_does_not_reuse_an_old_receipt(self) -> None:
        service = self.server.service
        runtime = service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='bind GUI presentation receipts to one service session',
        )
        request_id = runtime.human.query(
            pid,
            runtime.config.runtime.default_human,
            {'type': 'question', 'question': 'SESSION_BOUND_GUI_PRESENTATION'},
            blocking=False,
        )

        service.snapshot()
        first_effects = [
            effect
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider == 'human'
            and effect.provider_metadata.get('context', {}).get('purpose')
            == 'gui_presentation'
        ]
        old_provider = service._human_presentation_provider
        new_provider = type(old_provider)()
        old_key = ('gui', request_id, id(old_provider))
        receipts = runtime.human.presentation._receipts
        old_receipt = receipts[old_key]
        # Deterministically simulate a recycled object-id key.  Receipt
        # identity, not just the integer key, must bind the old session.
        receipts[
            ('gui', request_id, id(new_provider))
        ] = old_receipt
        service._human_presentation_provider = new_provider

        service.snapshot()
        second_effects = [
            effect
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider == 'human'
            and effect.provider_metadata.get('context', {}).get('purpose')
            == 'gui_presentation'
        ]

        assert len(first_effects) == 1
        assert len(second_effects) == 2

    def test_cached_gui_projection_linearizes_with_sink_policy_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = self.server.service
        runtime = service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='linearize cached GUI presentation policy',
        )
        sentinel = 'GUI_CACHED_POLICY_LINEARIZATION_SENTINEL'
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': sentinel},
            metadata=ObjectMetadata(sensitivity='normal'),
        )
        human = runtime.config.runtime.default_human
        pattern = f'human:{human}:{runtime.config.runtime.terminal_channel}'

        def register(max_sensitivity: str) -> None:
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=pattern,
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity=max_sensitivity,
                ),
                actor='test.host',
                replace=True,
                require_capability=False,
            )

        register('normal')
        request_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': sentinel},
            source_oids=[source.oid],
            blocking=False,
        )
        assert sentinel in json.dumps(service.snapshot(), sort_keys=True)

        original_receipt_check = runtime.human._presentation_was_delivered
        receipt_checked = threading.Event()
        allow_cached_return = threading.Event()
        blocked_once = False

        def checked_receipt(*args: object, **kwargs: object) -> bool:
            nonlocal blocked_once
            delivered = original_receipt_check(*args, **kwargs)
            selected = args[0] if args else None
            if (
                isinstance(selected, HumanRequest)
                and selected.request_id == request_id
                and not blocked_once
            ):
                blocked_once = True
                receipt_checked.set()
                assert allow_cached_return.wait(timeout=5)
            return delivered

        monkeypatch.setattr(
            runtime.human,
            '_presentation_was_delivered',
            checked_receipt,
        )
        snapshot_box: list[dict[str, Any]] = []
        snapshot_thread = threading.Thread(
            target=lambda: snapshot_box.append(service.snapshot()),
            daemon=True,
        )
        snapshot_thread.start()
        assert receipt_checked.wait(timeout=5)

        mutation_started = threading.Event()
        mutation_done = threading.Event()

        def downgrade_sink() -> None:
            mutation_started.set()
            register('public')
            mutation_done.set()

        mutation_thread = threading.Thread(target=downgrade_sink, daemon=True)
        mutation_thread.start()
        assert mutation_started.wait(timeout=5)
        assert mutation_done.wait(timeout=0.1) is False

        allow_cached_return.set()
        snapshot_thread.join(timeout=5)
        mutation_thread.join(timeout=5)

        assert snapshot_thread.is_alive() is False
        assert mutation_thread.is_alive() is False
        assert sentinel in json.dumps(snapshot_box[0], sort_keys=True)
        assert sentinel not in json.dumps(service.snapshot(), sort_keys=True)

    def test_human_presentation_evidence_does_not_starve_snapshot_causal_windows(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='preserve causal snapshot markers',
        )
        marker_event = runtime.events.emit(
            EventType.PROCESS_CREATED,
            source=pid,
            target=pid,
            payload={'marker': 'before-gui-presentation-burst'},
        )
        marker_audit = runtime.audit.record(
            actor=pid,
            action='test.gui.causal_marker',
            target=f'process:{pid}',
            decision={'marker': 'before-gui-presentation-burst'},
        )
        old_event_window = (
            runtime.config.gui.snapshot_event_limit
            + runtime.config.gui.snapshot_collection_max_items * 8
        )
        old_audit_window = (
            runtime.config.gui.snapshot_audit_limit
            + runtime.config.gui.snapshot_collection_max_items * 8
        )
        burst_size = max(old_event_window, old_audit_window) * 2 + 1
        burst_timestamp = '9999-12-31T23:59:59+00:00'
        for index in range(burst_size):
            runtime.store.insert_event(
                Event(
                    event_id=f'evt_gui_presentation_burst_{index:05d}',
                    type=EventType.HUMAN_OUTPUT,
                    source='human:owner',
                    target=pid,
                    payload={'purpose': 'gui_presentation'},
                    priority=EventPriority.NORMAL,
                    created_at=burst_timestamp,
                )
            )
            runtime.store.insert_audit(
                AuditRecord(
                    record_id=f'audit_gui_presentation_burst_{index:05d}',
                    timestamp=burst_timestamp,
                    actor='human:owner',
                    action='human.output',
                    target=f'human:{runtime.config.runtime.default_human}:terminal',
                    input_refs=[],
                    output_refs=[],
                    capability_refs=[],
                    decision={'purpose': 'gui_presentation'},
                    correlation_id=None,
                )
            )

        event_queries: list[dict[str, Any]] = []
        audit_queries: list[dict[str, Any]] = []
        original_list_events = runtime.store.list_events
        original_list_audit = runtime.store.list_audit

        def tracked_events(*args: Any, **kwargs: Any) -> list[Event]:
            event_queries.append(dict(kwargs))
            return original_list_events(*args, **kwargs)

        def tracked_audit(*args: Any, **kwargs: Any) -> list[AuditRecord]:
            audit_queries.append(dict(kwargs))
            return original_list_audit(*args, **kwargs)

        monkeypatch.setattr(runtime.store, 'list_events', tracked_events)
        monkeypatch.setattr(runtime.store, 'list_audit', tracked_audit)
        snapshot = self.server.service.snapshot()

        assert marker_event.event_id in {item['event_id'] for item in snapshot['events']}
        assert marker_audit.record_id in {item['record_id'] for item in snapshot['audit']}
        assert event_queries == [
            {
                'target': None,
                'limit': runtime.config.gui.snapshot_event_limit,
                'before_event_id': None,
                'after_event_id': None,
                'include_gui_presentation': False,
            }
        ]
        assert audit_queries == [
            {
                'limit': runtime.config.gui.snapshot_audit_limit,
                'actor': None,
                'target': None,
                'match_any': False,
                'include_gui_presentation': False,
            }
        ]

    def test_snapshot_completed_release_history_cannot_crowd_out_withheld_parent(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='pending Human release pair must stay visible',
        )
        human = runtime.config.runtime.default_human
        now = utc_now()
        limit = runtime.config.gui.snapshot_collection_max_items
        for index in range(limit + 2):
            runtime.store.insert_human_request(
                HumanRequest(
                    request_id=f'hreq_completed_release_{index:04d}',
                    pid=pid,
                    human=human,
                    payload={
                        'type': 'data_release_approval',
                        'question': f'completed release {index}',
                    },
                    status=HumanRequestStatus.APPROVED,
                    decision={'approved': True},
                    blocking=False,
                    created_at=now,
                    updated_at=now,
                )
            )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_PENDING_PAIR_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=(
                    f'human:{human}:'
                    f'{runtime.config.runtime.terminal_channel}'
                ),
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        parent_id = runtime.human.query(
            pid,
            human,
            {
                'type': 'question',
                'question': 'GUI_PENDING_PAIR_SECRET_SENTINEL',
            },
            source_oids=[source.oid],
        )

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        requests = snapshot['human_requests']
        assert len(requests) == limit
        assert 'GUI_PENDING_PAIR_SECRET_SENTINEL' not in json.dumps(
            requests,
            sort_keys=True,
        )
        parent = next(item for item in requests if item['request_id'] == parent_id)
        release = next(
            item
            for item in requests
            if item.get('release_for_request_id') == parent_id
            and item['status'] == HumanRequestStatus.PENDING.value
        )
        assert parent['payload']['release_required'] is True
        assert requests.index(release) < requests.index(parent)

    def test_presentation_window_does_not_release_a_cropped_approved_parent(self) -> None:
        service = self.server.service
        runtime = service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='cropped GUI parent must remain withheld',
        )
        human = runtime.config.runtime.default_human
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=(
                    f'human:{human}:'
                    f'{runtime.config.runtime.terminal_channel}'
                ),
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        first_source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_FIRST_WINDOW_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        cropped_source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_CROPPED_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'GUI_FIRST_WINDOW_SECRET_SENTINEL'},
            source_oids=[first_source.oid],
        )
        cropped_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'GUI_CROPPED_SECRET_SENTINEL'},
            source_oids=[cropped_source.oid],
        )
        cropped_view = service.human_request_view(runtime.human.get(cropped_id))
        assert cropped_view['payload']['release_required'] is True
        cropped_release_id = cropped_view['release_request_id']
        cropped_release = runtime.human.approve(
            cropped_release_id,
            {'approved': True, 'source': 'test.gui'},
        )
        release_resource = cropped_release.payload['requested_once_capability']['resource']
        release_capability = next(
            capability
            for capability in runtime.store.list_capabilities(subject=pid)
            if capability.resource == release_resource
        )
        assert release_capability.uses_remaining == 1
        cropped_effects_before = [
            effect.effect_id
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider_metadata.get('context', {}).get('request_id') == cropped_id
        ]

        views, has_more = runtime.human.list_for_presentation_window(
            presentation='gui',
            provider=service._human_presentation_provider,
            limit=2,
        )

        assert has_more is True
        assert cropped_id not in {
            request['request_id'] for request in views
        }
        assert len(views) == 2
        assert views[0].get('release_for_request_id') == views[1]['request_id']
        assert views[1]['payload']['release_required'] is True
        release_capability_after = next(
            capability
            for capability in runtime.store.list_capabilities(subject=pid)
            if capability.cap_id == release_capability.cap_id
        )
        assert release_capability_after.uses_remaining == 1
        assert [
            effect.effect_id
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider_metadata.get('context', {}).get('request_id') == cropped_id
        ] == cropped_effects_before
        assert runtime.human.is_request_withheld_for_presentation(
            cropped_id,
            presentation='gui',
        ) is True

        parent_before_denial = to_jsonable(runtime.human.get(cropped_id))
        process_before_denial = to_jsonable(runtime.process.get(pid))
        denied_status, denied = self.request(
            'POST',
            f'/api/human-requests/{cropped_id}/respond',
            {'approved': True, 'answer': 'not presented', 'auto_run': False},
        )
        assert denied_status == 409
        assert 'not been released' in denied['error']['message']
        assert to_jsonable(runtime.human.get(cropped_id)) == parent_before_denial
        assert to_jsonable(runtime.process.get(pid)) == process_before_denial

    def test_presentation_window_reports_pair_expansion_without_raw_lookahead(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='logical Human presentation expansion',
        )
        human = runtime.config.runtime.default_human
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=(
                    f'human:{human}:'
                    f'{runtime.config.runtime.terminal_channel}'
                ),
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_EXPANSION_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        first_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'GUI_EXPANSION_SECRET_SENTINEL'},
            source_oids=[source.oid],
        )
        second_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'ordinary second parent'},
        )

        views, has_more = runtime.human.list_for_presentation_window(
            presentation='gui',
            provider=self.server.service._human_presentation_provider,
            limit=2,
        )

        assert has_more is True
        assert [view.get('release_for_request_id') for view in views] == [first_id, None]
        assert views[1]['request_id'] == first_id
        assert second_id not in {view['request_id'] for view in views}

    def test_human_request_views_redact_conditional_payload_before_exact_release(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='redact conditional Human request')
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI DATA_FLOW_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f'human:{human}:{channel}',
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {
                'type': 'question',
                'question': 'GUI DATA_FLOW_SECRET_SENTINEL',
                'note': 'GUI_NOTE_SECRET_SENTINEL',
                'custom': {
                    'arbitrary_leaf': 'GUI_NESTED_SECRET_SENTINEL',
                },
            },
            source_oids=[source.oid],
        )

        status, snapshot = self.request('GET', '/api/snapshot')
        list_status, listed = self.request('GET', '/api/human-requests')
        process_status, process_list = self.request(
            'GET',
            f'/api/processes/{pid}/human-requests',
        )

        assert status == list_status == process_status == 200
        snapshot_encoded = json.dumps(snapshot, sort_keys=True)
        assert 'GUI DATA_FLOW_SECRET_SENTINEL' not in snapshot_encoded
        assert 'GUI_NOTE_SECRET_SENTINEL' not in snapshot_encoded
        assert 'GUI_NESTED_SECRET_SENTINEL' not in snapshot_encoded
        for projection in (snapshot['human_requests'], listed, process_list):
            encoded = json.dumps(projection, sort_keys=True)
            assert 'GUI DATA_FLOW_SECRET_SENTINEL' not in encoded
            assert 'GUI_NOTE_SECRET_SENTINEL' not in encoded
            assert 'GUI_NESTED_SECRET_SENTINEL' not in encoded
            parent = next(item for item in projection if item['request_id'] == request_id)
            assert parent['payload']['release_required'] is True
            assert parent['payload']['payload_observation']['redacted'] is True
            assert parent['payload']['payload_observation']['metadata_only'] is True
            release = next(
                item
                for item in projection
                if item['payload'].get('type') == 'data_release_approval'
            )
            assert release['payload']['context']['payload_sha256']
            assert release['release_for_request_id'] == request_id

        releases = [
            item
            for item in runtime.human.list()
            if item.payload.get('type') == 'data_release_approval'
        ]
        assert len(releases) == 1
        release = releases[0]
        assert release.payload['context']['sink'] == f'human:{human}:gui'
        assert release.payload['context']['operation'] == 'human.gui.present'

        parent_before_denial = to_jsonable(runtime.human.get(request_id))
        process_before_denial = to_jsonable(runtime.process.get(pid))
        requests_before_denial = to_jsonable(runtime.human.list(pid=pid))
        capabilities_before_denial = to_jsonable(
            runtime.store.list_capabilities(subject=pid)
        )
        decisions_before_denial = to_jsonable(
            runtime.store.list_data_flow_decisions(pid=pid)
        )
        withheld_status, withheld = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'too early', 'auto_run': False},
        )
        assert withheld_status == 409
        assert 'not been released' in withheld['error']['message']
        assert to_jsonable(runtime.human.get(request_id)) == parent_before_denial
        assert to_jsonable(runtime.process.get(pid)) == process_before_denial
        assert to_jsonable(runtime.human.list(pid=pid)) == requests_before_denial
        assert to_jsonable(runtime.store.list_capabilities(subject=pid)) == capabilities_before_denial
        assert to_jsonable(runtime.store.list_data_flow_decisions(pid=pid)) == decisions_before_denial

        response_status, _response = self.request(
            'POST',
            f'/api/human-requests/{release.request_id}/respond',
            {'approved': True, 'auto_run': False},
        )
        assert response_status == 200

        released_status, released_snapshot = self.request('GET', '/api/snapshot')
        repeated_status, repeated_snapshot = self.request('GET', '/api/snapshot')
        assert released_status == repeated_status == 200
        for projection in (released_snapshot['human_requests'], repeated_snapshot['human_requests']):
            parent = next(item for item in projection if item['request_id'] == request_id)
            assert parent['payload']['question'] == 'GUI DATA_FLOW_SECRET_SENTINEL'
            assert parent['payload']['note'] == 'GUI_NOTE_SECRET_SENTINEL'
            assert parent['payload']['custom']['arbitrary_leaf'] == 'GUI_NESTED_SECRET_SENTINEL'
        assert len([
            item
            for item in runtime.human.list()
            if item.payload.get('type') == 'data_release_approval'
        ]) == 1
        release_caps = [
            capability
            for capability in runtime.store.list_capabilities(subject=pid)
            if capability.resource == release.payload['requested_once_capability']['resource']
        ]
        assert len(release_caps) == 1
        assert release_caps[0].uses_remaining == 0
        presentation_effects = [
            effect
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider == 'human'
            and effect.provider_metadata.get('context', {}).get('purpose') == 'gui_presentation'
        ]
        assert len(presentation_effects) == 1
        presentation_flow = presentation_effects[0].provider_metadata['data_flow']
        assert presentation_flow['sink'] == f'human:{human}:gui'
        assert presentation_flow['release_capability_id'] == release_caps[0].cap_id

        answered_status, answered = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'released answer', 'auto_run': False},
        )
        assert answered_status == 200
        assert answered['request']['status'] == 'approved'
        assert 'answer' not in answered['request']['decision']
        assert answered['request']['payload']['release_required'] is True
        decision_release = next(
            item
            for item in runtime.human.list(pid=pid)
            if item.payload.get('type') == 'data_release_approval'
            and item.request_id != release.request_id
        )
        assert decision_release.status == HumanRequestStatus.PENDING
        runtime.human.approve(
            decision_release.request_id,
            {'approved': True, 'source': 'test.gui'},
        )
        decision_status, decision_snapshot = self.request('GET', '/api/snapshot')
        assert decision_status == 200
        decision_parent = next(
            item
            for item in decision_snapshot['human_requests']
            if item['request_id'] == request_id
        )
        assert decision_parent['decision']['answer'] == 'released answer'
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE

    def test_gui_presentation_release_does_not_suspend_runnable_process(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='keep runnable during GUI presentation release',
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_PRESENTATION_RELEASE_SECRET'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f'human:{human}:{channel}',
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {
                'type': 'question',
                'question': 'GUI_PRESENTATION_RELEASE_SECRET',
            },
            blocking=False,
            source_oids=[source.oid],
        )
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        parent = next(
            item for item in snapshot['human_requests']
            if item['request_id'] == request_id
        )
        assert parent['payload']['release_required'] is True
        release = next(
            item for item in runtime.human.list(pid=pid)
            if item.payload.get('type') == 'data_release_approval'
        )
        assert release.payload['_agent_libos_data_release_presentation'] == 'gui'
        assert release.blocking is False
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE

    def test_gui_conditional_release_survives_reopen_without_duplicate(self) -> None:
        db_path = Path(self.temp_dir.name) / 'gui-human-release.sqlite'
        runtime = Runtime.open(str(db_path))
        service = GuiRuntimeService(
            runtime=runtime,
            token='reopen-one',
            auto_run=False,
        )
        pid = runtime.process.spawn(image='base-agent:v0', goal='reopen GUI release')
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_REOPEN_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        human = runtime.config.runtime.default_human
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f'human:{human}:{runtime.config.runtime.terminal_channel}',
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {
                'type': 'question',
                'question': 'GUI_REOPEN_SECRET_SENTINEL',
            },
            source_oids=[source.oid],
        )
        first = service.snapshot()
        first_encoded = json.dumps(first['human_requests'], sort_keys=True)
        assert 'GUI_REOPEN_SECRET_SENTINEL' not in first_encoded
        release_ids = [
            item.request_id
            for item in runtime.human.list()
            if item.payload.get('type') == 'data_release_approval'
        ]
        assert len(release_ids) == 1
        release_id = release_ids[0]
        service.close()
        runtime.close()

        reopened = Runtime.open(str(db_path))
        reopened_server = create_gui_http_server(
            runtime=reopened,
            port=0,
            token='reopen-two',
            auto_run=False,
        )
        reopened_thread = threading.Thread(
            target=reopened_server.serve_forever,
            daemon=True,
        )
        reopened_thread.start()
        reopened_host, reopened_port = reopened_server.server_address

        def reopened_request(
            method: str,
            path: str,
            body: dict[str, Any] | None = None,
        ) -> tuple[int, Any]:
            conn = http.client.HTTPConnection(
                reopened_host,
                reopened_port,
                timeout=_GUI_TEST_HTTP_TIMEOUT_S,
            )
            headers = {'Authorization': 'Bearer reopen-two'}
            payload = None
            if body is not None:
                payload = json.dumps(body).encode('utf-8')
                headers['Content-Type'] = 'application/json'
            conn.request(method, path, body=payload, headers=headers)
            response = conn.getresponse()
            data = response.read()
            conn.close()
            decoded = json.loads(data.decode('utf-8')) if data else None
            return response.status, decoded

        try:
            reopened_status, reopened_snapshot = reopened_request('GET', '/api/snapshot')
            assert reopened_status == 200
            assert 'GUI_REOPEN_SECRET_SENTINEL' not in json.dumps(
                reopened_snapshot['human_requests'],
                sort_keys=True,
            )
            reopened_release_ids = [
                item.request_id
                for item in reopened.human.list()
                if item.payload.get('type') == 'data_release_approval'
            ]
            assert reopened_release_ids == [release_id]

            parent_before_denial = to_jsonable(reopened.human.get(request_id))
            process_before_denial = to_jsonable(reopened.process.get(pid))
            requests_before_denial = to_jsonable(reopened.human.list(pid=pid))
            capabilities_before_denial = to_jsonable(
                reopened.store.list_capabilities(subject=pid)
            )
            decisions_before_denial = to_jsonable(
                reopened.store.list_data_flow_decisions(pid=pid)
            )
            withheld_status, withheld = reopened_request(
                'POST',
                f'/api/human-requests/{request_id}/respond',
                {'approved': True, 'answer': 'too early', 'auto_run': False},
            )
            assert withheld_status == 409
            assert 'not been released' in withheld['error']['message']
            assert to_jsonable(reopened.human.get(request_id)) == parent_before_denial
            assert to_jsonable(reopened.process.get(pid)) == process_before_denial
            assert to_jsonable(reopened.human.list(pid=pid)) == requests_before_denial
            assert to_jsonable(reopened.store.list_capabilities(subject=pid)) == capabilities_before_denial
            assert to_jsonable(reopened.store.list_data_flow_decisions(pid=pid)) == decisions_before_denial

            reopened.human.approve(
                release_id,
                {'approved': True, 'source': 'test.gui'},
            )

            parent_before_consumption = to_jsonable(reopened.human.get(request_id))
            process_before_consumption = to_jsonable(reopened.process.get(pid))
            requests_before_consumption = to_jsonable(reopened.human.list(pid=pid))
            capabilities_before_consumption = to_jsonable(
                reopened.store.list_capabilities(subject=pid)
            )
            decisions_before_consumption = to_jsonable(
                reopened.store.list_data_flow_decisions(pid=pid)
            )
            unconsumed_status, unconsumed = reopened_request(
                'POST',
                f'/api/human-requests/{request_id}/respond',
                {'approved': True, 'answer': 'still too early', 'auto_run': False},
            )
            assert unconsumed_status == 409
            assert 'not been released' in unconsumed['error']['message']
            assert to_jsonable(reopened.human.get(request_id)) == parent_before_consumption
            assert to_jsonable(reopened.process.get(pid)) == process_before_consumption
            assert to_jsonable(reopened.human.list(pid=pid)) == requests_before_consumption
            assert to_jsonable(reopened.store.list_capabilities(subject=pid)) == capabilities_before_consumption
            assert to_jsonable(reopened.store.list_data_flow_decisions(pid=pid)) == decisions_before_consumption

            released_status, released = reopened_request('GET', '/api/snapshot')
            assert released_status == 200
            parent = next(
                item
                for item in released['human_requests']
                if item['request_id'] == request_id
            )
            assert parent['payload']['question'] == 'GUI_REOPEN_SECRET_SENTINEL'
            assert [
                item.request_id
                for item in reopened.human.list()
                if item.payload.get('type') == 'data_release_approval'
            ] == [release_id]

            answered_status, answered = reopened_request(
                'POST',
                f'/api/human-requests/{request_id}/respond',
                {'approved': True, 'answer': 'after reopen', 'auto_run': False},
            )
            assert answered_status == 200
            assert answered['request']['status'] == 'approved'
            assert 'answer' not in answered['request']['decision']
            assert answered['request']['payload']['release_required'] is True
            decision_release = next(
                item
                for item in reopened.human.list(pid=pid)
                if item.payload.get('type') == 'data_release_approval'
                and item.request_id != release_id
            )
            assert decision_release.status == HumanRequestStatus.PENDING
            reopened.human.approve(
                decision_release.request_id,
                {'approved': True, 'source': 'test.gui'},
            )
            decision_status, decision_snapshot = reopened_request(
                'GET',
                '/api/snapshot',
            )
            assert decision_status == 200
            decision_parent = next(
                item
                for item in decision_snapshot['human_requests']
                if item['request_id'] == request_id
            )
            assert decision_parent['decision']['answer'] == 'after reopen'
            assert reopened.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            reopened_server.shutdown()
            reopened_thread.join(timeout=5)
            reopened_server.service.shutdown()
            reopened_server.server_close()
            reopened.close()

    def test_gui_visible_release_is_invalidated_by_sink_registry_generation(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='invalidate stale GUI release',
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_STALE_RELEASE_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
            immutable=False,
        )
        human = runtime.config.runtime.default_human
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f'human:{human}:{runtime.config.runtime.terminal_channel}',
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {
                'type': 'question',
                'question': 'GUI_STALE_RELEASE_SECRET_SENTINEL',
            },
            source_oids=[source.oid],
        )

        initial_status, initial = self.request('GET', '/api/snapshot')
        assert initial_status == 200
        assert 'GUI_STALE_RELEASE_SECRET_SENTINEL' not in json.dumps(initial, sort_keys=True)
        first_release = next(
            item
            for item in runtime.human.list()
            if item.payload.get('type') == 'data_release_approval'
        )
        approval_status, _ = self.request(
            'POST',
            f'/api/human-requests/{first_release.request_id}/respond',
            {'approved': True, 'auto_run': False},
        )
        assert approval_status == 200
        visible_status, visible = self.request('GET', '/api/snapshot')
        assert visible_status == 200
        visible_parent = next(
            item for item in visible['human_requests'] if item['request_id'] == request_id
        )
        assert visible_parent['payload']['question'] == 'GUI_STALE_RELEASE_SECRET_SENTINEL'

        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern='human:gui-release-generation-bump:terminal',
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        parent_before_denial = to_jsonable(runtime.human.get(request_id))
        process_before_denial = to_jsonable(runtime.process.get(pid))
        requests_before_denial = to_jsonable(runtime.human.list(pid=pid))
        capabilities_before_denial = to_jsonable(
            runtime.store.list_capabilities(subject=pid)
        )
        decisions_before_denial = to_jsonable(
            runtime.store.list_data_flow_decisions(pid=pid)
        )
        stale_status, stale = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'stale release', 'auto_run': False},
        )
        assert stale_status == 409
        assert 'not been released' in stale['error']['message']
        assert to_jsonable(runtime.human.get(request_id)) == parent_before_denial
        assert to_jsonable(runtime.process.get(pid)) == process_before_denial
        assert to_jsonable(runtime.human.list(pid=pid)) == requests_before_denial
        assert to_jsonable(runtime.store.list_capabilities(subject=pid)) == capabilities_before_denial
        assert to_jsonable(runtime.store.list_data_flow_decisions(pid=pid)) == decisions_before_denial

        redacted_status, redacted = self.request('GET', '/api/snapshot')
        assert redacted_status == 200
        assert 'GUI_STALE_RELEASE_SECRET_SENTINEL' not in json.dumps(redacted, sort_keys=True)
        redacted_parent = next(
            item for item in redacted['human_requests'] if item['request_id'] == request_id
        )
        assert redacted_parent['payload']['release_required'] is True
        second_release = next(
            item
            for item in runtime.human.list()
            if item.payload.get('type') == 'data_release_approval'
            and item.request_id != first_release.request_id
        )
        assert second_release.status == HumanRequestStatus.PENDING

        renewed_status, renewed = self.request(
            'POST',
            f'/api/human-requests/{second_release.request_id}/respond',
            {'approved': True, 'auto_run': False},
        )
        assert renewed_status == 200, renewed
        renewed_snapshot_status, renewed_snapshot = self.request('GET', '/api/snapshot')
        assert renewed_snapshot_status == 200
        renewed_parent = next(
            item
            for item in renewed_snapshot['human_requests']
            if item['request_id'] == request_id
        )
        assert renewed_parent['payload']['question'] == 'GUI_STALE_RELEASE_SECRET_SENTINEL'

        runtime.memory.update_object(
            pid,
            source,
            ObjectPatch(payload={'value': 'source changed after GUI release'}),
        )
        parent_before_source_denial = to_jsonable(runtime.human.get(request_id))
        process_before_source_denial = to_jsonable(runtime.process.get(pid))
        source_stale_status, source_stale = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'source is stale', 'auto_run': False},
        )
        assert source_stale_status == 409
        assert 'not been released' in source_stale['error']['message']
        assert to_jsonable(runtime.human.get(request_id)) == parent_before_source_denial
        assert to_jsonable(runtime.process.get(pid)) == process_before_source_denial

        source_redacted_status, source_redacted = self.request('GET', '/api/snapshot')
        assert source_redacted_status == 200
        assert 'GUI_STALE_RELEASE_SECRET_SENTINEL' not in json.dumps(
            source_redacted,
            sort_keys=True,
        )
        source_redacted_parent = next(
            item
            for item in source_redacted['human_requests']
            if item['request_id'] == request_id
        )
        assert source_redacted_parent['payload']['release_required'] is True

    @pytest.mark.parametrize(
        ('sensitivity', 'initial_max', 'downgraded_max'),
        [
            ('secret', 'secret', 'normal'),
            ('normal', 'normal', 'public'),
        ],
    )
    def test_gui_projection_revalidates_trusted_sink_clearance(
        self,
        sensitivity: str,
        initial_max: str,
        downgraded_max: str,
    ) -> None:
        runtime = self.server.service.runtime
        sentinel = f'GUI_TRUSTED_DOWNGRADE_{sensitivity.upper()}_SENTINEL'
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='revalidate trusted GUI projection clearance',
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': sentinel},
            metadata=ObjectMetadata(sensitivity=sensitivity),
        )
        human = runtime.config.runtime.default_human
        pattern = f'human:{human}:{runtime.config.runtime.terminal_channel}'

        def register(max_sensitivity: str) -> None:
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=pattern,
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity=max_sensitivity,
                ),
                actor='test.host',
                replace=True,
                require_capability=False,
            )

        register(initial_max)
        request_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': sentinel},
            source_oids=[source.oid],
        )

        initial_status, initial = self.request('GET', '/api/snapshot')
        assert initial_status == 200
        initial_parent = next(
            item for item in initial['human_requests'] if item['request_id'] == request_id
        )
        assert initial_parent['payload']['question'] == sentinel

        register(downgraded_max)
        decisions_before = {
            item.decision_id for item in runtime.store.list_data_flow_decisions(pid=pid)
        }
        downgraded_status, downgraded = self.request('GET', '/api/snapshot')

        assert downgraded_status == 200
        assert sentinel not in json.dumps(downgraded, sort_keys=True)
        downgraded_parent = next(
            item
            for item in downgraded['human_requests']
            if item['request_id'] == request_id
        )
        assert downgraded_parent['payload']['release_required'] is True
        denials = [
            item
            for item in runtime.store.list_data_flow_decisions(pid=pid, outcome='deny')
            if item.decision_id not in decisions_before
            and item.sink == f'human:{human}:gui'
        ]
        assert len(denials) == 1
        denial = denials[0]
        assert denial.labels.sensitivity.value == sensitivity
        assert f'exceeds Sink maximum {downgraded_max}' in denial.reason
        assert any(
            record.action == 'data_flow.egress'
            and record.target == f'human:{human}:gui'
            and record.decision.get('decision_id') == denial.decision_id
            and record.decision.get('outcome') == 'deny'
            for record in runtime.audit.trace()
        )
        assert any(
            event.type == EventType.DATA_FLOW_DECISION
            and event.payload.get('decision_id') == denial.decision_id
            and event.payload.get('outcome') == 'deny'
            for event in runtime.events.list(target=f'data_flow_sink:human:{human}:gui')
        )

        register(initial_max)
        restored_status, restored = self.request('GET', '/api/snapshot')
        assert restored_status == 200, restored
        restored_parent = next(
            item for item in restored['human_requests'] if item['request_id'] == request_id
        )
        assert restored_parent['payload']['question'] == sentinel
        repeated_status, repeated = self.request('GET', '/api/snapshot')
        assert repeated_status == 200, repeated
        repeated_parent = next(
            item for item in repeated['human_requests'] if item['request_id'] == request_id
        )
        assert repeated_parent['payload']['question'] == sentinel

    def test_gui_response_guard_and_decision_share_binding_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='serialize GUI presentation guard with Human decision',
        )
        human = runtime.config.runtime.default_human
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_ATOMIC_GUARD_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=(
                    f'human:{human}:'
                    f'{runtime.config.runtime.terminal_channel}'
                ),
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'GUI_ATOMIC_GUARD_SECRET_SENTINEL'},
            source_oids=[source.oid],
        )
        withheld = self.server.service.human_request_view(runtime.human.get(request_id))
        release_id = withheld['release_request_id']
        runtime.human.approve(release_id, {'approved': True, 'source': 'test.gui'})
        visible = self.server.service.human_request_view(runtime.human.get(request_id))
        assert visible['payload']['question'] == 'GUI_ATOMIC_GUARD_SECRET_SENTINEL'

        original_guard = runtime.human.is_request_withheld_for_presentation
        guard_checked = threading.Event()
        allow_decision = threading.Event()
        blocked_once = False

        def guarded(request: HumanRequest | str, *, presentation: str) -> bool:
            nonlocal blocked_once
            result = original_guard(request, presentation=presentation)
            selected_id = request.request_id if isinstance(request, HumanRequest) else request
            if selected_id == request_id and not blocked_once:
                blocked_once = True
                guard_checked.set()
                assert allow_decision.wait(timeout=5)
            return result

        monkeypatch.setattr(runtime.human, 'is_request_withheld_for_presentation', guarded)
        response_box: list[tuple[int, Any]] = []
        response_thread = threading.Thread(
            target=lambda: response_box.append(
                self.request(
                    'POST',
                    f'/api/human-requests/{request_id}/respond',
                    {'approved': True, 'answer': 'atomic answer', 'auto_run': False},
                )
            ),
            daemon=True,
        )
        response_thread.start()
        assert guard_checked.wait(timeout=5)

        mutation_started = threading.Event()
        mutation_done = threading.Event()
        status_seen_after_mutation: list[HumanRequestStatus] = []

        def mutate_registry() -> None:
            mutation_started.set()
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern='human:gui-atomic-generation-bump:terminal',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='secret',
                ),
                actor='test.host',
                require_capability=False,
            )
            status_seen_after_mutation.append(runtime.human.get(request_id).status)
            mutation_done.set()

        mutation_thread = threading.Thread(target=mutate_registry, daemon=True)
        mutation_thread.start()
        assert mutation_started.wait(timeout=5)
        assert mutation_done.wait(timeout=0.1) is False

        allow_decision.set()
        response_thread.join(timeout=5)
        mutation_thread.join(timeout=5)

        assert response_thread.is_alive() is False
        assert mutation_thread.is_alive() is False
        assert response_box[0][0] == 200
        assert status_seen_after_mutation == [HumanRequestStatus.APPROVED]
        assert runtime.human.get(request_id).decision['answer'] == 'atomic answer'

    def test_gui_release_binds_the_returned_decision_view(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='bind GUI release to the complete returned view',
        )
        human = runtime.config.runtime.default_human
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_COMPLETE_VIEW_SECRET'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=(
                    f'human:{human}:'
                    f'{runtime.config.runtime.terminal_channel}'
                ),
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'GUI_COMPLETE_VIEW_SECRET'},
            source_oids=[source.oid],
        )

        initial_status, _initial = self.request('GET', '/api/snapshot')
        assert initial_status == 200
        first_release = next(
            item
            for item in runtime.human.list(pid=pid)
            if item.payload.get('type') == 'data_release_approval'
        )
        runtime.human.approve(
            first_release.request_id,
            {'approved': True, 'source': 'test.gui'},
        )
        visible_status, visible = self.request('GET', '/api/snapshot')
        assert visible_status == 200
        visible_parent = next(
            item for item in visible['human_requests'] if item['request_id'] == request_id
        )
        assert visible_parent['payload']['question'] == 'GUI_COMPLETE_VIEW_SECRET'
        decisions_before_response = len(
            runtime.store.list_data_flow_decisions(pid=pid)
        )

        answered_status, answered = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {
                'approved': True,
                'answer': 'GUI_DECISION_SECRET_SENTINEL',
                'auto_run': False,
            },
        )

        assert answered_status == 200
        assert 'GUI_DECISION_SECRET_SENTINEL' not in json.dumps(
            answered['request'],
            sort_keys=True,
        )
        assert answered['request']['payload']['release_required'] is True
        assert len(runtime.store.list_data_flow_decisions(pid=pid)) > decisions_before_response
        second_release = next(
            item
            for item in runtime.human.list(pid=pid)
            if item.payload.get('type') == 'data_release_approval'
            and item.request_id != first_release.request_id
        )
        assert second_release.status == HumanRequestStatus.PENDING

        runtime.human.approve(
            second_release.request_id,
            {'approved': True, 'source': 'test.gui'},
        )
        released_status, released = self.request('GET', '/api/snapshot')
        assert released_status == 200
        released_parent = next(
            item for item in released['human_requests'] if item['request_id'] == request_id
        )
        assert released_parent['decision']['answer'] == 'GUI_DECISION_SECRET_SENTINEL'

    def test_snapshot_source_bounds_process_and_registry_reads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self.server.service
        runtime = service.runtime
        for index in range(17):
            runtime.process.spawn(image='base-agent:v0', goal=f'source-bound-{index}')
        runtime.config = replace(
            runtime.config,
            gui=replace(runtime.config.gui, snapshot_collection_max_items=16),
            mcp=replace(
                runtime.config.mcp,
                list_limit=1,
                server_page_limit=17,
            ),
        )
        service._static_snapshot_dirty = True
        seen: dict[str, list[int | None]] = {}

        def spy_limit(owner: object, attribute: str, label: str) -> None:
            original = getattr(owner, attribute)

            def wrapped(*args: object, **kwargs: object) -> object:
                seen.setdefault(label, []).append(kwargs.get('limit'))
                return original(*args, **kwargs)

            monkeypatch.setattr(owner, attribute, wrapped)

        spy_limit(runtime.process, 'list', 'processes')
        spy_limit(runtime.human, 'list', 'human_requests')
        spy_limit(runtime.tools, 'list', 'tools')
        spy_limit(runtime.image_registry, 'list_images', 'images')
        spy_limit(runtime.skills, 'discover_skills_window', 'skills')
        spy_limit(runtime.jsonrpc, 'list_endpoints_window', 'jsonrpc_endpoints')
        spy_limit(runtime.mcp, 'list_servers_window', 'mcp_servers')
        spy_limit(runtime.modules, 'loaded_module_summaries', 'modules')
        spy_limit(service, '_llm_profile_summaries', 'llm_profiles')

        snapshot = service.snapshot()

        assert len(snapshot['processes']) == 16
        assert len(snapshot['tools']) == 16
        assert seen == {
            'processes': [17],
            'human_requests': [17],
            'tools': [17],
            'images': [17],
            'skills': [17],
            'jsonrpc_endpoints': [17],
            'mcp_servers': [17],
            'modules': [17],
            'llm_profiles': [17],
        }
        assert snapshot['_truncated']['processes']['source_limited'] is True
        assert snapshot['_truncated']['processes']['omitted_is_lower_bound'] is True
        assert snapshot['_truncated']['tools']['source_limited'] is True

    def test_snapshot_reports_truncation_at_stricter_jsonrpc_source_limit(self) -> None:
        service = self.server.service
        runtime = service.runtime
        runtime.config = replace(
            runtime.config,
            jsonrpc=replace(runtime.config.jsonrpc, list_limit=2),
        )
        runtime.jsonrpc.config = runtime.config
        for index in range(3):
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _gui_jsonrpc_manifest(f'gui-source-limit-{index}'),
                actor='test',
                require_capability=False,
            )
        service._static_snapshot_dirty = True

        snapshot = service.snapshot()

        assert len(snapshot['jsonrpc_endpoints']) == 2
        assert snapshot['_truncated']['jsonrpc_endpoints'] == {
            'kind': 'array',
            'returned': 2,
            'omitted': 1,
            'omitted_is_lower_bound': True,
            'source_limited': True,
        }

    def test_snapshot_batches_process_activity_rating_and_resource_queries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = self.server.service
        runtime = service.runtime
        pids = [runtime.process.spawn(image='base-agent:v0', goal=f'batch-{index}') for index in range(3)]
        for index, pid in enumerate(pids):
            runtime.messages.post(sender='gui-test', recipient_pid=pid, body=f'message-{index}')
        runtime.ratings.upsert(pids[0], score=5, comment='batched')

        calls = {'activity': 0, 'remaining': 0, 'ratings': 0}
        original_activity = runtime.store.get_process_activity_summaries
        original_remaining = runtime.resources.remaining_budgets
        original_ratings = runtime.ratings.get_many
        original_list_llm_calls = runtime.store.list_llm_calls

        def activity(*args: object, **kwargs: object) -> object:
            calls['activity'] += 1
            return original_activity(*args, **kwargs)

        def remaining(*args: object, **kwargs: object) -> object:
            calls['remaining'] += 1
            return original_remaining(*args, **kwargs)

        def ratings(*args: object, **kwargs: object) -> object:
            calls['ratings'] += 1
            return original_ratings(*args, **kwargs)

        def list_llm_calls(pid: str | None = None, limit: int | None = None) -> object:
            assert pid is None, 'snapshot must not load LLM call rows once per process'
            return original_list_llm_calls(pid=pid, limit=limit)

        def unexpected_single_process_query(*_args: object, **_kwargs: object) -> object:
            raise AssertionError('snapshot used a per-process manager query')

        monkeypatch.setattr(runtime.store, 'get_process_activity_summaries', activity)
        monkeypatch.setattr(runtime.resources, 'remaining_budgets', remaining)
        monkeypatch.setattr(runtime.ratings, 'get_many', ratings)
        monkeypatch.setattr(runtime.store, 'list_llm_calls', list_llm_calls)
        monkeypatch.setattr(runtime.messages, 'list', unexpected_single_process_query)
        monkeypatch.setattr(runtime.resources, 'remaining_budget', unexpected_single_process_query)
        monkeypatch.setattr(runtime.ratings, 'get', unexpected_single_process_query)

        snapshot = service.snapshot()

        assert calls == {'activity': 1, 'remaining': 1, 'ratings': 1}
        by_pid = {process['pid']: process for process in snapshot['processes']}
        assert set(by_pid) == set(pids)
        assert all(process['unread_message_count'] == 1 for process in by_pid.values())
        assert by_pid[pids[0]]['rating']['score'] == 5

    def test_process_summary_uses_cumulative_resource_totals_beyond_recent_call_window(
        self,
    ) -> None:
        service = self.server.service
        runtime = service.runtime
        runtime.config = replace(
            runtime.config,
            gui=replace(runtime.config.gui, snapshot_process_llm_call_limit=2),
        )
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='show cumulative long-running task usage',
        )
        runtime.resources.charge(
            pid,
            ResourceUsage(
                llm_calls=32,
                llm_prompt_tokens=891_111,
                llm_completion_tokens=22_133,
                llm_total_tokens=913_244,
            ),
            source='test.gui.cumulative_llm_usage',
        )

        snapshot = service.snapshot()
        summary = next(item for item in snapshot['processes'] if item['pid'] == pid)

        assert summary['llm_call_count'] == 32
        assert summary['token_total'] == 913_244

    def test_snapshot_selects_recent_process_messages_at_the_source(self) -> None:
        service = self.server.service
        runtime = service.runtime
        runtime.config = replace(
            runtime.config,
            gui=replace(runtime.config.gui, snapshot_process_message_limit=2),
        )
        pid = runtime.process.spawn(image='base-agent:v0', goal='recent message window')
        for index in range(5):
            runtime.messages.post(sender='gui-test', recipient_pid=pid, body=f'message-{index}')

        snapshot = service.snapshot()
        process = next(item for item in snapshot['processes'] if item['pid'] == pid)

        assert process['unread_message_count'] == 5
        assert [message['body'] for message in process['messages']] == ['message-3', 'message-4']

    def test_llm_profile_endpoints_persist_user_profiles_and_reject_secrets(self, monkeypatch) -> None:
        monkeypatch.setenv('KIMI_API_KEY', 'secret')

        status, profiles = self.request('GET', '/api/llm-profiles')
        assert status == 200
        assert any(profile['profile_id'] == 'default' and profile['source'] == 'config' for profile in profiles)

        status, created = self.request(
            'POST',
            '/api/llm-profiles',
            {
                'profile_id': 'kimi-k2.7-code',
                'model': 'kimi-k2.7-code',
                'base_url': 'https://kimi.example/v1',
                'api_key_env': 'KIMI_API_KEY',
                'api_mode': 'chat',
                'timeout_s': 17.5,
                'max_retries': 4,
                'store': False,
                'temperature': 0.1,
                'reasoning_effort': 'high',
                'verbosity': 'low',
                'safety_identifier_env': 'OPENAI_SAFETY_IDENTIFIER',
                'prompt_cache_retention': 'in-memory',
                'responses_previous_response_id': True,
                'parallel_tool_calls': False,
                'auto_wait_on_empty_tool_calls': True,
                'fallback_json_actions': True,
                'max_tokens': 2048,
                'context_window_tokens': 100000,
                'allow_custom_base_url': False,
            },
        )
        assert status == 200
        assert created['profile_id'] == 'kimi-k2.7-code'
        assert created['source'] == 'user'
        assert created['editable'] is True
        assert created['api_key_env_present'] is True
        assert created['timeout_s'] == 17.5
        assert created['max_retries'] == 4
        assert created['store'] is False
        assert created['reasoning_effort'] == 'high'
        assert created['verbosity'] == 'low'
        assert created['safety_identifier_env'] == 'OPENAI_SAFETY_IDENTIFIER'
        assert created['prompt_cache_retention'] == 'in_memory'
        assert created['responses_previous_response_id'] is True
        assert created['parallel_tool_calls'] is False
        assert created['auto_wait_on_empty_tool_calls'] is True
        assert created['fallback_json_actions'] is True
        assert created['allow_custom_base_url'] is False

        omitted_fields = {
            'kind': 'openai_compatible',
            'model': 'kimi-k2.7-code',
            'base_url': 'https://kimi.example/v1',
            'api_key_env': 'KIMI_API_KEY',
            'api_mode': 'chat',
            'timeout_s': 17.5,
            'max_retries': 4,
            'store': False,
            'reasoning_effort': 'high',
            'verbosity': 'low',
            'safety_identifier_env': 'OPENAI_SAFETY_IDENTIFIER',
            'prompt_cache_retention': 'in_memory',
            'responses_previous_response_id': True,
            'parallel_tool_calls': False,
            'auto_wait_on_empty_tool_calls': True,
            'fallback_json_actions': True,
            'temperature': 0.1,
        }
        updates = {
            'max_tokens': 4096,
            'context_window_tokens': 200000,
            'allow_custom_base_url': True,
        }
        before_update = self.server.service.runtime.llms.profile('kimi-k2.7-code')
        assert all(getattr(before_update, field) != value for field, value in updates.items())
        status, updated = self.request(
            'PUT',
            '/api/llm-profiles/kimi-k2.7-code',
            updates,
        )
        assert status == 200
        assert updated['max_tokens'] == 4096
        assert updated['context_window_tokens'] == 200000
        assert updated['allow_custom_base_url'] is True
        assert updated['reasoning_effort'] == 'high'
        assert updated['verbosity'] == 'low'
        assert updated['safety_identifier_env'] == 'OPENAI_SAFETY_IDENTIFIER'
        assert updated['prompt_cache_retention'] == 'in_memory'
        assert updated['responses_previous_response_id'] is True
        updated_profile = self.server.service.runtime.llms.profile('kimi-k2.7-code')
        for field, expected in {**omitted_fields, **updates}.items():
            assert getattr(updated_profile, field) == expected, field
        persisted_profile = json.loads(self.llm_profiles_file.read_text(encoding='utf-8'))['profiles']['kimi-k2.7-code']
        for field, expected in {**omitted_fields, **updates}.items():
            if field != 'kind':
                assert persisted_profile[field] == expected, field
        assert 'secret' not in self.llm_profiles_file.read_text(encoding='utf-8')

        status, rejected = self.request(
            'POST',
            '/api/llm-profiles',
            {'profile_id': 'bad-secret', 'model': 'bad', 'api_key_env': 'BAD_API_KEY', 'api_key': 'secret'},
        )
        assert status == 400
        assert 'API keys are not accepted' in rejected['error']['message']

        self.server.service.shutdown()
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

        self.server = create_gui_http_server(
            db='local',
            port=0,
            token='test-token',
            auto_run=False,
            llm_profiles_file=self.llm_profiles_file,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address
        status, profiles = self.request('GET', '/api/llm-profiles')
        assert status == 200
        assert any(
            profile['profile_id'] == 'kimi-k2.7-code'
            and profile['max_tokens'] == 4096
            and profile['context_window_tokens'] == 200000
            and profile['timeout_s'] == 17.5
            and profile['max_retries'] == 4
            and profile['store'] is False
            and profile['reasoning_effort'] == 'high'
            and profile['verbosity'] == 'low'
            and profile['parallel_tool_calls'] is False
            and profile['auto_wait_on_empty_tool_calls'] is True
            and profile['fallback_json_actions'] is True
            for profile in profiles
        )
        reopened_profile = self.server.service.runtime.llms.profile('kimi-k2.7-code')
        for field, expected in {**omitted_fields, **updates}.items():
            assert getattr(reopened_profile, field) == expected, field
        reopened_persisted = json.loads(self.llm_profiles_file.read_text(encoding='utf-8'))['profiles']['kimi-k2.7-code']
        for field, expected in {**omitted_fields, **updates}.items():
            if field != 'kind':
                assert reopened_persisted[field] == expected, field

    def test_llm_profile_spawn_exec_validation_and_delete_in_use(self) -> None:
        status, body = self.request('POST', '/api/processes', {'goal': 'bad profile', 'auto_run': False, 'llm_profile': 'missing'})
        assert status == 400
        assert 'unknown LLM profile' in body['error']['message']

        status, _profile = self.request(
            'POST',
            '/api/llm-profiles',
            {'profile_id': 'glm-5.2', 'model': 'glm-5.2', 'api_key_env': 'GLM_API_KEY'},
        )
        assert status == 200
        status, spawned = self.request('POST', '/api/processes', {'goal': 'profile', 'auto_run': False, 'llm_profile': 'glm-5.2'})
        assert status == 200
        pid = spawned['pid']
        status, body = self.request('DELETE', '/api/llm-profiles/glm-5.2')
        assert status == 409
        assert pid in body['error']['pids']

        status, bad_exec = self.request(
            'POST',
            f'/api/processes/{pid}/exec',
            {'image': 'base-agent:v0', 'goal': 'new', 'llm_profile': 'missing', 'confirmed': True},
        )
        assert status == 400
        assert 'unknown LLM profile' in bad_exec['error']['message']

        self.server.service.runtime.process.exit(pid, message='done')
        status, deleted = self.request('DELETE', '/api/llm-profiles/glm-5.2')
        assert status == 409
        assert deleted['error']['profile_id'] == 'glm-5.2'

    def test_process_spawn_rejects_non_string_llm_profile_before_mutation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        status, _profile = self.request(
            'POST',
            '/api/llm-profiles',
            {'profile_id': '7', 'model': 'numeric-profile', 'api_key_env': 'NUMERIC_PROFILE_API_KEY'},
        )
        assert status == 200
        runtime = self.server.service.runtime
        original_spawn = runtime.process.spawn
        calls: list[dict[str, Any]] = []

        def tracked_spawn(*args: Any, **kwargs: Any):
            calls.append(dict(kwargs))
            return original_spawn(*args, **kwargs)

        monkeypatch.setattr(runtime.process, 'spawn', tracked_spawn)

        status, body = self.request(
            'POST',
            '/api/processes',
            {'goal': 'must not spawn', 'llm_profile': 7, 'auto_run': False},
        )

        assert status == 400
        assert 'llm_profile must be a JSON string or null' in body['error']['message']
        assert calls == []

    def test_process_exec_rejects_non_string_llm_profile_before_mutation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        status, _profile = self.request(
            'POST',
            '/api/llm-profiles',
            {'profile_id': 'True', 'model': 'boolean-profile', 'api_key_env': 'BOOLEAN_PROFILE_API_KEY'},
        )
        assert status == 200
        _status, spawned = self.request(
            'POST',
            '/api/processes',
            {'goal': 'exec profile target', 'auto_run': False},
        )
        pid = spawned['pid']
        runtime = self.server.service.runtime
        before = runtime.process.get(pid)
        original_exec = runtime.exec_process
        calls: list[dict[str, Any]] = []

        def tracked_exec(*args: Any, **kwargs: Any):
            calls.append(dict(kwargs))
            return original_exec(*args, **kwargs)

        monkeypatch.setattr(runtime, 'exec_process', tracked_exec)

        status, body = self.request(
            'POST',
            f'/api/processes/{pid}/exec',
            {
                'image': 'base-agent:v0',
                'goal': 'must not exec',
                'llm_profile': True,
                'confirmed': True,
                'auto_run': False,
            },
        )

        assert status == 400
        assert 'llm_profile must be a JSON string or null' in body['error']['message']
        assert calls == []
        after = runtime.process.get(pid)
        assert after.llm_profile_id == before.llm_profile_id
        assert after.image_id == before.image_id
        assert after.revision == before.revision

    def test_process_spawn_authority_manifest_keeps_workspace_access_request_only(self) -> None:
        runtime = self.server.service.runtime
        subtree = 'agent_outputs/gui_authority'
        status, spawned = self.request(
            'POST',
            '/api/processes',
            {
                'image': 'coding-agent:v0',
                'goal': 'edit one scoped result',
                'working_directory': subtree,
                'auto_run': False,
                'authority_manifest': {
                    'authorized_capabilities': [
                        {
                            'resource': 'human:owner',
                            'rights': ['write'],
                            'delegable': False,
                        }
                    ],
                    'approval_policy': {
                        'requestable_capabilities': [
                            {
                                'resource': f'filesystem:workspace:{subtree}/*',
                                'rights': ['read', 'write'],
                                'delegable': False,
                            }
                        ]
                    },
                },
            },
        )

        assert status == 200
        pid = spawned['pid']
        inside = f'filesystem:workspace:{subtree}/result.txt'
        outside = 'filesystem:workspace:agent_outputs/outside.txt'
        assert runtime.capability.check(pid, 'human:owner', CapabilityRight.WRITE)
        assert not runtime.capability.check(pid, inside, CapabilityRight.READ)
        assert not runtime.capability.check(pid, inside, CapabilityRight.WRITE)

        with pytest.raises(CapabilityDenied):
            runtime.shell.run(pid, ['python', '-m', 'pytest', '-q'])
        assert runtime.human.pending() == []

        denied = runtime.tools.call(
            pid,
            'request_permission',
            {'resource': outside, 'rights': ['write'], 'reason': 'outside launch scope'},
        )
        assert not denied.ok
        assert runtime.human.pending() == []

        with pytest.raises(HumanResponseRequired):
            runtime.tools.call(
                pid,
                'request_permission',
                {'resource': inside, 'rights': ['write'], 'reason': 'create scoped result'},
            )
        assert [request.pid for request in runtime.human.pending()] == [pid]
        assert not runtime.capability.check(pid, inside, CapabilityRight.WRITE)

    def test_process_spawn_reviewed_shell_policy_requires_exact_command_approval(self) -> None:
        runtime = self.server.service.runtime
        status, spawned = self.request(
            'POST',
            '/api/processes',
            {
                'image': 'coding-agent:v0',
                'goal': 'run reviewed tests',
                'auto_run': False,
                'authority_manifest': {
                    'authorized_capabilities': [
                        {
                            'resource': 'human:owner',
                            'rights': ['write'],
                            'delegable': False,
                        },
                        {
                            'resource': 'shell:*',
                            'rights': ['execute'],
                            'delegable': False,
                            'constraints': {
                                'shell_policy_level': 'allowlist_auto_else_ask',
                            },
                        },
                    ],
                    'approval_policy': {'requestable_capabilities': []},
                },
            },
        )

        assert status == 200
        pid = spawned['pid']
        policy = next(
            capability
            for capability in runtime.capability.capabilities_for(pid)
            if capability.resource == 'shell:*'
        )
        assert policy.constraints == {
            'shell_policy_level': 'allowlist_auto_else_ask'
        }

        argv = ['python', '-m', 'pytest', '-q']
        with pytest.raises(HumanApprovalRequired):
            runtime.shell.run(pid, argv)

        pending = runtime.human.pending()
        assert len(pending) == 1
        assert pending[0].pid == pid
        assert pending[0].payload['context']['argv'] == argv
        assert any(
            record.action == 'human.query' and record.actor == pid
            for record in runtime.audit.trace(actor=pid)
        )
        assert any(
            event.type == EventType.HUMAN_QUERY and event.source == pid
            for event in runtime.events.list()
        )

    def test_process_rating_endpoint_updates_snapshot_and_audit(self) -> None:
        status, spawned = self.request('POST', '/api/processes', {'goal': 'rate agent', 'auto_run': False})
        assert status == 200
        pid = spawned['pid']

        status, empty = self.request('GET', f'/api/processes/{pid}/rating')
        assert status == 200
        assert empty is None

        status, rating = self.request('POST', f'/api/processes/{pid}/rating', {'score': 5, 'comment': 'strong result'})
        assert status == 200
        assert rating['pid'] == pid
        assert rating['score'] == 5
        assert rating['comment'] == 'strong result'
        assert rating['rater'] == DEFAULT_CONFIG.runtime.default_human
        assert rating['source'] == 'gui'

        status, updated = self.request('POST', f'/api/processes/{pid}/rating', {'score': 3, 'comment': 'missed detail'})
        assert status == 200
        assert updated['rating_id'] == rating['rating_id']
        assert updated['score'] == 3

        status, snapshot = self.request('GET', '/api/snapshot')
        assert status == 200
        process = next(item for item in snapshot['processes'] if item['pid'] == pid)
        assert process['rating']['score'] == 3
        assert process['rating']['comment'] == 'missed detail'
        assert any(
            record.action == 'agent.rating.upsert'
            and record.target == f'process:{pid}'
            for record in self.server.service.runtime.audit.trace()
        )

    def test_process_rating_endpoint_rejects_invalid_requests(self) -> None:
        status, spawned = self.request('POST', '/api/processes', {'goal': 'bad rating', 'auto_run': False})
        assert status == 200
        pid = spawned['pid']

        status, body = self.request('POST', f'/api/processes/{pid}/rating', {'score': 0})
        assert status == 400
        assert 'between 1 and 5' in body['error']['message']

        status, body = self.request('GET', '/api/processes/missing/rating')
        assert status == 404
        assert 'process not found' in body['error']['message']

    def test_encoded_route_segments_are_decoded(self) -> None:
        status, inspected = self.request('GET', '/api/images/base-agent%3Av0')

        assert status == 200
        assert inspected['image']['image_id'] == 'base-agent:v0'

    def test_process_spawn_accepts_initial_working_directory(self) -> None:
        status, spawned = self.request(
            'POST',
            '/api/processes',
            {'goal': 'cwd target', 'working_directory': 'src\\app', 'auto_run': False},
        )

        assert status == 200
        assert spawned['process']['working_directory'] == 'src/app'

    def test_cors_is_limited_to_local_gui_origins(self) -> None:
        status, headers, _body = self.request_raw(
            'OPTIONS',
            '/api/health',
            extra_headers={'Origin': 'http://127.0.0.1:5173'},
        )
        assert status == 204
        assert headers['access-control-allow-origin'] == 'http://127.0.0.1:5173'

        status, headers, _body = self.request_raw(
            'OPTIONS',
            '/api/health',
            extra_headers={'Origin': 'https://example.test'},
        )
        assert status == 204
        assert 'access-control-allow-origin' not in headers

        status, headers, _body = self.request_raw(
            'OPTIONS',
            '/api/health',
            extra_headers={'Origin': 'agent-libos://app'},
        )
        assert status == 204
        assert headers['access-control-allow-origin'] == 'agent-libos://app'

        status, headers, _body = self.request_raw(
            'OPTIONS',
            '/api/health',
            extra_headers={'Origin': 'agent-libos://untrusted'},
        )
        assert status == 204
        assert 'access-control-allow-origin' not in headers

        status, headers, _body = self.request_raw(
            'OPTIONS',
            '/api/health',
            extra_headers={'Origin': 'null'},
        )
        assert status == 204
        assert 'access-control-allow-origin' not in headers

    def test_sse_replays_snapshot_event(self) -> None:
        request = urllib.request.Request(f'http://{self.host}:{self.port}/api/events/stream?cursor=0', headers={'Authorization': 'Bearer test-token'})
        with urllib.request.urlopen(
            request,
            timeout=_GUI_TEST_HTTP_TIMEOUT_S,
        ) as response:
            assert response.status == 200
            assert response.headers['Cache-Control'] == 'no-store'
            assert response.headers['X-Content-Type-Options'] == 'nosniff'
            assert response.headers['Referrer-Policy'] == 'no-referrer'
            frame_lines: list[str] = []
            while len(frame_lines) < 3:
                line = response.readline().decode('utf-8').strip()
                if line:
                    frame_lines.append(line)
            assert frame_lines[0].startswith('id: ')
            assert frame_lines[1] == 'event: snapshot'
            assert frame_lines[2].startswith('data: ')

    def test_sse_broadcaster_invalidates_evicted_and_restarted_cursors(self) -> None:
        broadcaster = GuiEventBroadcaster(max_events=2)
        broadcaster.publish('snapshot', {'version': 1})
        broadcaster.publish('snapshot', {'version': 2})
        broadcaster.publish('snapshot', {'version': 3})

        evicted = broadcaster.replay_after(0)

        assert [event.event for event in evicted] == ['event.invalidated', 'snapshot', 'snapshot']
        assert evicted[0].seq == 1
        assert evicted[0].data == {
            'invalidated': True,
            'reason': 'sse_cursor_not_replayable',
            'requested_cursor': 0,
            'reset_cursor': 1,
            'oldest_available': 2,
            'latest_available': 3,
        }
        assert [event.seq for event in evicted[1:]] == [2, 3]

        restarted = broadcaster.replay_after(99)

        assert [event.event for event in restarted] == ['event.invalidated', 'snapshot', 'snapshot']
        assert restarted[0].seq == 0
        assert restarted[0].data['requested_cursor'] == 99
        assert restarted[0].data['reset_cursor'] == 0

    def test_gui_delta_deduplication_is_bounded(self) -> None:
        seen = _BoundedSeenKeys(2)

        assert seen.add_if_new('first') is True
        assert seen.add_if_new('second') is True
        assert seen.add_if_new('second') is False
        assert seen.add_if_new('third') is True
        assert len(seen) == 2
        assert seen.add_if_new('first') is True
        assert len(seen) == 2

    def test_snapshot_audit_window_contains_latest_records(self) -> None:
        for index in range(205):
            self.server.service.runtime.audit.record(
                actor='test',
                action=f'audit.window.{index}',
                target='process:audit-window',
            )

        status, snapshot = self.request('GET', '/api/snapshot')
        actions = [record['action'] for record in snapshot['audit']]

        assert status == 200
        assert 'audit.window.204' in actions
        assert 'audit.window.0' not in actions

    def test_snapshot_truncates_model_amplified_event_payloads(self) -> None:
        huge = 'x' * (self.server.service.runtime.config.gui.snapshot_string_max_chars + 100)
        self.server.service.runtime.events.emit(
            EventType.EXTERNAL_WRITE,
            source='gui-test',
            target='gui-test',
            payload={'blob': huge},
        )

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        serialized = json.dumps(snapshot)
        assert huge not in serialized
        event = snapshot['events'][-1]
        assert isinstance(event['payload']['blob'], str)
        assert len(event['payload']['blob']) == self.server.service.runtime.config.gui.snapshot_string_max_chars
        truncation = {
            path: meta
            for path, meta in snapshot['_truncated'].items()
            if path.endswith('.payload.blob')
        }
        assert list(truncation.values())[0]['kind'] == 'string'
        assert list(truncation.values())[0]['chars'] == len(huge)

    def test_snapshot_array_truncation_uses_metadata_not_sentinel_items(self) -> None:
        self.server.service.runtime.config = replace(
            self.server.service.runtime.config,
            gui=replace(
                self.server.service.runtime.config.gui,
                snapshot_collection_max_items=20,
                snapshot_event_limit=25,
            ),
        )
        for index in range(25):
            self.server.service.runtime.events.emit(
                EventType.EXTERNAL_WRITE,
                source='gui-test',
                target='gui-test',
                payload={'index': index},
            )

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        assert len(snapshot['events']) == 20
        assert all('event_id' in event for event in snapshot['events'])
        assert not any(event.get('truncated') is True for event in snapshot['events'])
        assert snapshot['_truncated']['events']['kind'] == 'array'
        assert snapshot['_truncated']['events']['omitted'] == 5

    def test_process_events_endpoint_is_store_bounded(self) -> None:
        pid = self.server.service.runtime.process.spawn(goal='bounded event api')
        self.server.service.runtime.config = replace(
            self.server.service.runtime.config,
            gui=replace(self.server.service.runtime.config.gui, snapshot_event_limit=3),
        )
        emitted = [
            self.server.service.runtime.events.emit(
                EventType.EXTERNAL_WRITE,
                source='gui-test',
                target=pid,
                payload={'index': index},
            )
            for index in range(5)
        ]

        status, events = self.request('GET', f'/api/processes/{pid}/events?limit=2')
        previous_status, previous = self.request(
            'GET',
            f'/api/processes/{pid}/events?limit=2&before={events[0]["event_id"]}',
        )
        invalid_status, invalid = self.request('GET', f'/api/processes/{pid}/events?limit=4')

        assert status == 200
        assert [event['event_id'] for event in events] == [emitted[-2].event_id, emitted[-1].event_id]
        assert previous_status == 200
        assert [event['event_id'] for event in previous] == [emitted[-4].event_id, emitted[-3].event_id]
        assert invalid_status == 400
        assert 'at most 3' in invalid['error']['message']

    def test_oversized_snapshot_sse_payload_uses_explicit_truncated_event(self) -> None:
        event_name, payload = _sse_payload_data(
            'snapshot',
            {'snapshot': {'events': [{'payload': 'x' * 100}]}},
            max_bytes=50,
            string_limit=200,
            collection_limit=200,
        )

        assert event_name == 'snapshot_truncated'
        assert payload['invalidated'] is True
        assert payload['event'] == 'snapshot'

    def test_strict_json_bool_rejects_string_false(self) -> None:
        status, body = self.request('POST', '/api/processes', {'goal': 'strict bool', 'auto_run': 'false'})

        assert status == 400
        assert 'auto_run must be a JSON boolean' in body['error']['message']

    def test_process_audit_filters_before_limit(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'audit target', 'auto_run': False})
        pid = spawned['pid']
        self.server.service.runtime.audit.record(
            actor=pid,
            action='process.audit.target',
            target=f'process:{pid}',
        )
        for index in range(205):
            self.server.service.runtime.audit.record(
                actor='noise',
                action=f'process.audit.noise.{index}',
                target='process:noise',
            )

        status, records = self.request('GET', f'/api/processes/{pid}/audit?limit=1')

        assert status == 200
        assert [record['action'] for record in records] == ['process.audit.target']

    def test_process_audit_before_cursor_returns_gap_free_older_page(self) -> None:
        _status, spawned = self.request(
            'POST',
            '/api/processes',
            {'goal': 'audit cursor target', 'auto_run': False},
        )
        pid = spawned['pid']
        emitted = [
            self.server.service.runtime.audit.record(
                actor=pid,
                action=f'process.audit.cursor.{index}',
                target=f'process:{pid}',
            )
            for index in range(4)
        ]

        first_status, first = self.request(
            'GET',
            f'/api/processes/{pid}/audit?limit=2',
        )
        second_status, second = self.request(
            'GET',
            f'/api/processes/{pid}/audit?limit=2&before={first[0]["record_id"]}',
        )

        expected = [record.record_id for record in emitted]
        actual = [record['record_id'] for record in second + first]
        assert first_status == second_status == 200
        assert actual == expected

    def test_process_audit_default_limit_uses_active_config_and_pages_gap_free(
        self,
        tmp_path: Path,
    ) -> None:
        target = str(tmp_path / "gui-audit-limit.sqlite")
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(local_store_target=target),
            gui=replace(DEFAULT_CONFIG.gui, snapshot_audit_limit=7),
        )
        server = create_gui_http_server(
            config=config,
            port=0,
            token="configured-audit-token",
            auto_run=False,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            pid = "pid_configured_audit"
            emitted = [
                server.service.runtime.audit.record(
                    actor=pid,
                    action=f"process.audit.configured.{index:02d}",
                    target=f"process:{pid}",
                )
                for index in range(15)
            ]

            first_status, first = _request_to_server(
                server,
                "GET",
                f"/api/processes/{pid}/audit",
                token="configured-audit-token",
            )
            second_status, second = _request_to_server(
                server,
                "GET",
                f"/api/processes/{pid}/audit?before={first[0]['record_id']}",
                token="configured-audit-token",
            )
            third_status, third = _request_to_server(
                server,
                "GET",
                f"/api/processes/{pid}/audit?before={second[0]['record_id']}",
                token="configured-audit-token",
            )

            assert first_status == second_status == third_status == 200
            assert [len(first), len(second), len(third)] == [7, 7, 1]
            expected = [record.record_id for record in emitted]
            actual = [record["record_id"] for record in third + second + first]
            assert actual == expected
            assert len(actual) == len(set(actual))
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.service.shutdown()
            server.server_close()

    def test_high_risk_exec_requires_confirmation(self) -> None:
        status, _profile = self.request(
            'POST',
            '/api/llm-profiles',
            {'profile_id': 'gui-exec', 'model': 'gui-exec-model', 'api_key_env': 'GUI_EXEC_API_KEY'},
        )
        assert status == 200
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'goal', 'auto_run': False})
        pid = spawned['pid']
        status, denied = self.request(
            'POST',
            f'/api/processes/{pid}/exec',
            {'image': 'base-agent:v0', 'goal': 'new', 'llm_profile': 'gui-exec'},
        )
        assert status == 409
        assert denied['error']['confirmation_required']
        assert denied['error']['preview']['llm_profile'] == 'gui-exec'
        status, string_confirmed = self.request('POST', f'/api/processes/{pid}/exec', {'image': 'base-agent:v0', 'goal': 'new', 'confirmed': 'true'})
        assert status == 400
        assert 'confirmed must be a JSON boolean' in string_confirmed['error']['message']
        status, allowed = self.request(
            'POST',
            f'/api/processes/{pid}/exec',
            {
                'image': 'base-agent:v0',
                'goal': 'new',
                'confirmed': True,
                'auto_run': False,
                'llm_profile': 'gui-exec',
            },
        )
        assert status == 200
        assert allowed['process']['image_id'] == 'base-agent:v0'
        assert allowed['process']['llm_profile_id'] == 'gui-exec'

    def test_destructive_process_signal_requires_confirmation(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'signal target', 'auto_run': False})
        pid = spawned['pid']

        status, paused = self.request(
            'POST',
            f'/api/processes/{pid}/signal',
            {'signal': ProcessSignal.PAUSE.value},
        )
        assert status == 200
        assert paused['status'] == ProcessStatus.PAUSED.value
        assert paused['wait_state']['kind'] == 'paused'
        assert paused['outcome'] is None
        paused_generation = paused['state_generation']

        status, denied = self.request('POST', f'/api/processes/{pid}/signal', {'signal': ProcessSignal.TERMINATE.value})
        assert status == 409
        assert denied['error']['confirmation_required']
        assert denied['error']['preview']['signal'] == ProcessSignal.TERMINATE.value

        status, string_confirmed = self.request(
            'POST',
            f'/api/processes/{pid}/signal',
            {'signal': ProcessSignal.TERMINATE.value, 'confirmed': 'true'},
        )
        assert status == 400
        assert 'confirmed must be a JSON boolean' in string_confirmed['error']['message']

        status, allowed = self.request(
            'POST',
            f'/api/processes/{pid}/signal',
            {'signal': ProcessSignal.TERMINATE.value, 'confirmed': True},
        )
        assert status == 200
        assert allowed['status'] == ProcessStatus.KILLED.value
        assert allowed['wait_state'] is None
        assert allowed['outcome']['schema_version'] == 1
        assert allowed['outcome']['kind'] == 'killed'
        assert allowed['state_generation'] > paused_generation

    def test_checkpoint_inspect_typed_state_survives_sqlite_restart_across_boundaries(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = str(tmp_path / 'typed-checkpoint.sqlite')
        runtime = Runtime.open(db_path)
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='checkpoint parent')
            runtime.capability.grant(
                parent,
                'process:spawn',
                [CapabilityRight.WRITE],
                issued_by='test',
            )
            child = runtime.spawn_child_process(parent, 'checkpoint child')
            with pytest.raises(ProcessWaitRequired):
                runtime.process.wait(parent, child)
            waiting = runtime.process.get(parent)
            expected_wait = process_wait_state_to_mapping(waiting.wait_state)
            waiting_generation = waiting.state_generation
            waiting_checkpoint = runtime.checkpoint.create(
                parent,
                'typed waiting snapshot',
                actor=parent,
            )

            runtime.process.signal_child(
                parent,
                child,
                ProcessSignal.TERMINATE,
                reason='typed terminal snapshot',
            )
            terminal = runtime.process.get(child)
            expected_outcome = process_outcome_to_mapping(terminal.outcome)
            terminal_generation = terminal.state_generation
            terminal_checkpoint = runtime.checkpoint.create(
                parent,
                'typed terminal snapshot',
                actor=parent,
            )
        finally:
            runtime.close()

        reopened = Runtime.open(db_path)
        try:
            manager_waiting = reopened.checkpoint.inspect(waiting_checkpoint, actor=parent)
            waiting_row = next(
                row for row in manager_waiting['processes'] if row['pid'] == parent
            )
            assert waiting_row['wait_state'] == expected_wait
            assert waiting_row['outcome'] is None
            assert waiting_row['state_generation'] == waiting_generation

            manager_terminal = reopened.checkpoint.inspect(terminal_checkpoint, actor=parent)
            terminal_row = next(
                row for row in manager_terminal['processes'] if row['pid'] == child
            )
            assert terminal_row['wait_state'] is None
            assert terminal_row['outcome'] == expected_outcome
            assert terminal_row['state_generation'] == terminal_generation

            tool_waiting = reopened.tools.call(
                parent,
                'inspect_checkpoint',
                {'checkpoint_id': waiting_checkpoint},
            )
            assert tool_waiting.ok, tool_waiting.error
            tool_waiting_row = next(
                row for row in tool_waiting.payload['processes'] if row['pid'] == parent
            )
            assert tool_waiting_row['wait_state'] == expected_wait
            assert tool_waiting_row['outcome'] is None
            assert tool_waiting_row['state_generation'] == waiting_generation

            tool_terminal = reopened.tools.call(
                parent,
                'inspect_checkpoint',
                {'checkpoint_id': terminal_checkpoint},
            )
            assert tool_terminal.ok, tool_terminal.error
            tool_terminal_row = next(
                row for row in tool_terminal.payload['processes'] if row['pid'] == child
            )
            assert tool_terminal_row['wait_state'] is None
            assert tool_terminal_row['outcome'] == expected_outcome
            assert tool_terminal_row['state_generation'] == terminal_generation

            syscall_terminal = asyncio.run(
                LibOSSyscallSession(reopened, parent).handle(
                    'checkpoint.inspect',
                    {'checkpoint_id': terminal_checkpoint},
                )
            )
            syscall_terminal_row = next(
                row for row in syscall_terminal['processes'] if row['pid'] == child
            )
            assert syscall_terminal_row['outcome'] == expected_outcome
            assert syscall_terminal_row['state_generation'] == terminal_generation
        finally:
            reopened.close()

        cli_waiting = checkpoint_cli_json(
            ['--db', db_path, 'checkpoint', 'inspect', waiting_checkpoint]
        )
        cli_waiting_row = next(
            row for row in cli_waiting['processes'] if row['pid'] == parent
        )
        assert cli_waiting_row['wait_state'] == expected_wait
        assert cli_waiting_row['state_generation'] == waiting_generation
        cli_terminal = checkpoint_cli_json(
            ['--db', db_path, 'checkpoint', 'inspect', terminal_checkpoint]
        )
        cli_terminal_row = next(
            row for row in cli_terminal['processes'] if row['pid'] == child
        )
        assert cli_terminal_row['outcome'] == expected_outcome
        assert cli_terminal_row['state_generation'] == terminal_generation

        typed_server = create_gui_http_server(
            db=db_path,
            port=0,
            token='typed-checkpoint-token',
            auto_run=False,
            llm_profiles_file=tmp_path / 'typed-llm-profiles.json',
        )
        typed_thread = threading.Thread(target=typed_server.serve_forever, daemon=True)
        typed_thread.start()
        try:
            host, port = typed_server.server_address
            def gui_inspect(checkpoint_id: str) -> dict[str, Any]:
                conn = http.client.HTTPConnection(
                    host,
                    port,
                    timeout=_GUI_TEST_HTTP_TIMEOUT_S,
                )
                conn.request(
                    'GET',
                    f'/api/checkpoints/{checkpoint_id}',
                    headers={'Authorization': 'Bearer typed-checkpoint-token'},
                )
                response = conn.getresponse()
                payload = json.loads(response.read().decode('utf-8'))
                conn.close()
                assert response.status == 200
                return payload

            gui_waiting = gui_inspect(waiting_checkpoint)
            gui_waiting_row = next(
                row for row in gui_waiting['processes'] if row['pid'] == parent
            )
            assert gui_waiting_row['wait_state'] == expected_wait
            assert gui_waiting_row['state_generation'] == waiting_generation

            gui_terminal = gui_inspect(terminal_checkpoint)
            gui_terminal_row = next(
                row for row in gui_terminal['processes'] if row['pid'] == child
            )
            assert gui_terminal_row['wait_state'] is None
            assert gui_terminal_row['outcome'] == expected_outcome
            assert gui_terminal_row['state_generation'] == terminal_generation
        finally:
            typed_server.shutdown()
            typed_thread.join(timeout=5)
            typed_server.service.shutdown()
            typed_server.server_close()

    def test_invalid_process_signal_is_a_bad_request_without_mutation(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'invalid signal target', 'auto_run': False})
        pid = spawned['pid']

        status, body = self.request('POST', f'/api/processes/{pid}/signal', {'signal': 'not-a-signal'})

        assert status == 400
        assert 'unknown process signal' in body['error']['message']
        assert self.server.service.runtime.process.get(pid).status == ProcessStatus.RUNNABLE

    def test_missing_required_mutation_fields_are_bad_requests(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'required fields', 'auto_run': False})
        pid = spawned['pid']

        exec_status, exec_body = self.request(
            'POST',
            f'/api/processes/{pid}/exec',
            {'goal': 'missing image', 'confirmed': True, 'auto_run': False},
        )
        cd_status, cd_body = self.request('POST', f'/api/processes/{pid}/cd', {})
        checkpoint_status, checkpoint_body = self.request('POST', '/api/checkpoints/create', {})

        assert exec_status == 400
        assert 'image must be a non-empty JSON string' in exec_body['error']['message']
        assert cd_status == 400
        assert 'path must be a non-empty JSON string' in cd_body['error']['message']
        assert checkpoint_status == 400
        assert 'pid must be a non-empty JSON string' in checkpoint_body['error']['message']
        assert self.server.service.runtime.process.get(pid).image_id == 'base-agent:v0'

    @pytest.mark.parametrize("field", ["image", "working_directory"])
    @pytest.mark.parametrize("invalid", [None, 7, [], {}, ""])
    def test_process_spawn_rejects_invalid_optional_string_fields_without_spawning(
        self,
        field: str,
        invalid: Any,
    ) -> None:
        runtime = self.server.service.runtime
        before = {process.pid for process in runtime.process.list()}

        status, body = self.request(
            "POST",
            "/api/processes",
            {"goal": "must not spawn", field: invalid, "auto_run": False},
        )

        assert status == 400
        assert f"{field} must be a non-empty JSON string" in body["error"]["message"]
        assert {process.pid for process in runtime.process.list()} == before

    def test_process_spawn_accepts_valid_optional_string_fields(self) -> None:
        status, spawned = self.request(
            "POST",
            "/api/processes",
            {
                "goal": "valid spawn strings",
                "image": "base-agent:v0",
                "working_directory": ".",
                "auto_run": False,
            },
        )

        assert status == 200
        assert spawned["process"]["image_id"] == "base-agent:v0"
        assert spawned["process"]["working_directory"] == "."

    def test_process_exec_rejects_non_object_args_before_mutation_and_accepts_object(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _status, spawned = self.request(
            "POST",
            "/api/processes",
            {"goal": "exec args target", "auto_run": False},
        )
        pid = spawned["pid"]
        runtime = self.server.service.runtime
        original_exec = runtime.exec_process
        calls: list[dict[str, Any]] = []

        def tracked_exec(*args: Any, **kwargs: Any):
            calls.append(dict(kwargs))
            return original_exec(*args, **kwargs)

        monkeypatch.setattr(runtime, "exec_process", tracked_exec)
        for invalid in (None, [], "{}", 7):
            status, body = self.request(
                "POST",
                f"/api/processes/{pid}/exec",
                {
                    "image": "base-agent:v0",
                    "args": invalid,
                    "confirmed": True,
                    "auto_run": False,
                },
            )
            assert status == 400
            assert "process exec args must be a JSON object" in body["error"]["message"]
        assert calls == []

        status, _body = self.request(
            "POST",
            f"/api/processes/{pid}/exec",
            {
                "image": "base-agent:v0",
                "args": {"mode": "review"},
                "confirmed": True,
                "auto_run": False,
            },
        )
        assert status == 200
        assert calls == [{"args": {"mode": "review"}, "goal": None, "preserve_memory": True, "preserve_capabilities": False, "llm_profile_id": None}]

    def test_process_exec_rejects_invalid_goal_before_mutation_and_accepts_object_goal(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _status, spawned = self.request(
            "POST",
            "/api/processes",
            {"goal": "exec goal target", "auto_run": False},
        )
        pid = spawned["pid"]
        runtime = self.server.service.runtime
        original_exec = runtime.exec_process
        calls: list[dict[str, Any]] = []

        def tracked_exec(*args: Any, **kwargs: Any):
            calls.append(dict(kwargs))
            return original_exec(*args, **kwargs)

        monkeypatch.setattr(runtime, "exec_process", tracked_exec)
        for invalid in ([], 7, False):
            status, body = self.request(
                "POST",
                f"/api/processes/{pid}/exec",
                {
                    "image": "base-agent:v0",
                    "goal": invalid,
                    "confirmed": True,
                    "auto_run": False,
                },
            )
            assert status == 400
            assert "goal must be a JSON string, object, or null" in body["error"]["message"]
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert calls == []

        goal = {"task": "review", "target": "README.md"}
        status, _body = self.request(
            "POST",
            f"/api/processes/{pid}/exec",
            {
                "image": "base-agent:v0",
                "goal": goal,
                "confirmed": True,
                "auto_run": False,
            },
        )
        assert status == 200
        assert calls == [
            {
                "args": {},
                "goal": goal,
                "preserve_memory": True,
                "preserve_capabilities": False,
                "llm_profile_id": None,
            }
        ]

    def test_process_exec_rejects_invalid_image_before_confirmation_audit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _status, spawned = self.request(
            "POST",
            "/api/processes",
            {"goal": "invalid exec image target", "auto_run": False},
        )
        pid = spawned["pid"]
        runtime = self.server.service.runtime
        audit_calls: list[str] = []
        original_record = runtime.audit.record

        def tracked_record(*args: Any, **kwargs: Any):
            audit_calls.append(str(kwargs.get("action") or ""))
            return original_record(*args, **kwargs)

        monkeypatch.setattr(runtime.audit, "record", tracked_record)
        for payload in ({}, {"image": None}, {"image": 7}, {"image": []}, {"image": ""}):
            status, body = self.request(
                "POST",
                f"/api/processes/{pid}/exec",
                payload,
            )
            assert status == 400
            assert "image must be a non-empty JSON string" in body["error"]["message"]
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert "gui.confirmation_required" not in audit_calls

    def test_process_exit_rejects_non_string_message_before_mutation_and_accepts_nullable_string(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = self.server.service.runtime
        original_exit = runtime.process.exit
        calls: list[tuple[str, str | None]] = []

        def tracked_exit(pid: str, *, failed: bool = False, message: str | None = None):
            calls.append((pid, message))
            return original_exit(pid, failed=failed, message=message)

        monkeypatch.setattr(runtime.process, "exit", tracked_exit)
        for invalid in ([], {}, 7, False):
            _status, spawned = self.request(
                "POST",
                "/api/processes",
                {"goal": "exit message target", "auto_run": False},
            )
            pid = spawned["pid"]
            status, body = self.request(
                "POST",
                f"/api/processes/{pid}/exit",
                {"message": invalid, "confirmed": True},
            )
            assert status == 400
            assert "message must be a JSON string or null" in body["error"]["message"]
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert calls == []

        for message in (None, "completed from GUI"):
            _status, spawned = self.request(
                "POST",
                "/api/processes",
                {"goal": "valid exit message", "auto_run": False},
            )
            pid = spawned["pid"]
            status, _body = self.request(
                "POST",
                f"/api/processes/{pid}/exit",
                {"message": message, "confirmed": True},
            )
            assert status == 200
            assert runtime.process.get(pid).status == ProcessStatus.EXITED
        assert [message for _pid, message in calls] == [None, "completed from GUI"]

    def test_actor_is_rejected_on_routes_that_do_not_apply_actor_authority(self) -> None:
        service = self.server.service
        runtime = service.runtime
        _status, spawned = self.request(
            'POST',
            '/api/processes',
            {'goal': 'actor contract target', 'auto_run': False},
        )
        pid = spawned['pid']
        request_id = 'hreq_actor_contract'
        now = utc_now()
        runtime.store.insert_human_request(
            HumanRequest(
                request_id=request_id,
                pid=pid,
                human=runtime.config.runtime.default_human,
                payload={'type': 'question', 'question': 'must remain pending'},
                status=HumanRequestStatus.PENDING,
                decision=None,
                blocking=True,
                created_at=now,
                updated_at=now,
            )
        )
        service.save_user_llm_profile(
            'actor-contract-profile',
            {'model': 'actor-contract-model', 'api_key_env': 'ACTOR_CONTRACT_API_KEY'},
        )

        def observable_state() -> dict[str, Any]:
            return {
                'processes': to_jsonable(runtime.process.list()),
                'messages': to_jsonable(runtime.messages.list(pid, include_acked=True)),
                'rating': to_jsonable(runtime.ratings.get(pid)),
                'human_requests': to_jsonable(runtime.human.list()),
                'object_tasks': to_jsonable(runtime.object_tasks.list()),
                'capabilities': to_jsonable(runtime.store.list_capabilities()),
                'external_effects': to_jsonable(runtime.store.list_external_effects()),
                'jsonrpc_endpoints': runtime.jsonrpc.list_endpoints(require_capability=False),
                'mcp_servers': runtime.mcp.list_servers(require_capability=False),
                'llm_profiles': service._llm_profile_summaries(),
                'llm_profiles_file': self.llm_profiles_file.read_text(encoding='utf-8'),
                'audit': to_jsonable(runtime.audit.trace()),
                'events': to_jsonable(runtime.events.list()),
                'gui_events': to_jsonable(service.broadcaster.replay_after(0)),
                'scheduler': service.scheduler.status(),
                'service_closing': service._closing,
                'service_closed': service.closed,
            }

        before = observable_state()
        unsupported_actor_routes = [
            ('process.spawn', 'POST', '/api/processes', {'goal': 'must not spawn', 'actor': pid}),
            ('workflow.run', 'POST', '/api/workflows/run', {'tool': 'get_working_directory', 'args': {}, 'actor': pid}),
            ('object_task.start', 'POST', '/api/object-tasks/start', {'pid': pid, 'tool': 'get_working_directory', 'args': {}, 'actor': pid}),
            ('object_task.cancel', 'POST', '/api/object-tasks/missing/cancel', {'pid': pid, 'actor': pid}),
            ('object_task.wait', 'POST', '/api/object-tasks/missing/wait', {'pid': pid, 'actor': pid}),
            ('object_task.watch_owner', 'POST', '/api/object-tasks/missing/watch-owner', {'pid': pid, 'enabled': True, 'actor': pid}),
            ('scheduler.auto', 'POST', '/api/scheduler/auto', {'enabled': True, 'actor': pid}),
            ('scheduler.pause', 'POST', '/api/scheduler/pause', {'actor': pid}),
            ('process.rating', 'POST', f'/api/processes/{pid}/rating', {'score': 5, 'actor': pid}),
            ('process.run', 'POST', f'/api/processes/{pid}/run', {'max_quanta': 1, 'actor': pid}),
            ('process.step', 'POST', f'/api/processes/{pid}/step', {'actor': pid}),
            ('process.pause', 'POST', f'/api/processes/{pid}/pause', {'reason': 'must not pause', 'actor': pid}),
            ('process.resume', 'POST', f'/api/processes/{pid}/resume', {'actor': pid}),
            ('process.signal', 'POST', f'/api/processes/{pid}/signal', {'signal': ProcessSignal.PAUSE.value, 'actor': pid}),
            ('process.message', 'POST', f'/api/processes/{pid}/message', {'body': 'must not deliver', 'actor': pid}),
            ('process.interrupt', 'POST', f'/api/processes/{pid}/interrupt', {'body': 'must not interrupt', 'actor': pid}),
            ('process.cd', 'POST', f'/api/processes/{pid}/cd', {'path': '.', 'actor': pid}),
            ('process.exec', 'POST', f'/api/processes/{pid}/exec', {'image': 'coding-agent:v0', 'confirmed': True, 'actor': pid}),
            ('process.exit', 'POST', f'/api/processes/{pid}/exit', {'confirmed': True, 'actor': pid}),
            ('human.respond', 'POST', f'/api/human-requests/{request_id}/respond', {'approved': True, 'answer': 'must not answer', 'actor': pid}),
            ('capability.explain', 'POST', '/api/capabilities/explain', {'subject': pid, 'resource': 'object:actor-contract', 'right': 'read', 'actor': pid}),
            ('llm_profile.create', 'POST', '/api/llm-profiles', {'profile_id': 'must-not-create', 'model': 'unused', 'api_key_env': 'UNUSED_API_KEY', 'actor': pid}),
            ('llm_profile.update', 'PUT', '/api/llm-profiles/actor-contract-profile', {'model': 'must-not-update', 'actor': pid}),
            ('llm_profile.delete', 'DELETE', '/api/llm-profiles/actor-contract-profile', {'actor': pid}),
            ('jsonrpc.call', 'POST', '/api/jsonrpc/missing/call', {'pid': pid, 'method_id': 'read', 'confirmed': True, 'actor': pid}),
            ('mcp.call', 'POST', '/api/mcp/missing/call', {'pid': pid, 'tool_id': 'read', 'confirmed': True, 'actor': pid}),
            ('shutdown', 'POST', '/api/shutdown', {'actor': pid}),
        ]
        for route_name, method, path, payload in unsupported_actor_routes:
            status, body = self.request(method, path, payload)
            assert status == 400, route_name
            assert 'does not accept actor' in body['error']['message'], route_name
            assert observable_state() == before, route_name

        status, invalid_actor = self.request(
            'POST',
            '/api/checkpoints/create',
            {'pid': pid, 'actor': None},
        )
        assert status == 400
        assert 'actor must be a non-empty JSON string' in invalid_actor['error']['message']
        process = runtime.process.get(pid)
        assert process.status == ProcessStatus.RUNNABLE
        assert process.image_id == 'base-agent:v0'
        assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        assert observable_state() == before

    def test_high_risk_image_commit_requires_confirmation(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'commit source', 'auto_run': False})
        pid = spawned['pid']
        status, created = self.request('POST', '/api/checkpoints/create', {'pid': pid, 'reason': 'commit'})
        assert status == 200
        status, denied = self.request('POST', '/api/images/commit', {'checkpoint_id': created['checkpoint_id'], 'image_id': 'gui-committed:v0', 'name': 'gui-committed'})
        assert status == 409
        assert denied['error']['confirmation_required']
        status, forbidden = self.request('POST', '/api/images/commit', {'checkpoint_id': created['checkpoint_id'], 'image_id': 'gui-committed:v0', 'name': 'gui-committed', 'actor': pid, 'confirmed': True})
        assert status == 403
        assert 'lacks write' in forbidden['error']['message']
        status, committed = self.request('POST', '/api/images/commit', {'checkpoint_id': created['checkpoint_id'], 'image_id': 'gui-committed:v0', 'name': 'gui-committed', 'confirmed': True})
        assert status == 200
        assert committed['image_id'] == 'gui-committed:v0'
        status, inspected = self.request('GET', '/api/images/gui-committed:v0')
        assert status == 200
        assert inspected['image']['boot']['kind'] == 'checkpoint_commit'

    def test_checkpoint_actor_mode_enforces_restore_capability(self) -> None:
        _status, owner = self.request('POST', '/api/processes', {'goal': 'checkpoint owner', 'auto_run': False})
        _status, other = self.request('POST', '/api/processes', {'goal': 'unprivileged actor', 'auto_run': False})
        status, created = self.request('POST', '/api/checkpoints/create', {'pid': owner['pid'], 'reason': 'admin checkpoint'})
        assert status == 200
        status, body = self.request(
            'POST',
            f"/api/checkpoints/{created['checkpoint_id']}/restore",
            {'actor': other['pid'], 'confirmed': True},
        )
        assert status == 403
        assert 'checkpoint' in body['error']['message']

    def test_capability_actor_mode_enforces_process_authority(self) -> None:
        runtime = self.server.service.runtime
        _status, actor = self.request('POST', '/api/processes', {'goal': 'capability actor', 'auto_run': False})
        _status, subject = self.request('POST', '/api/processes', {'goal': 'capability subject', 'auto_run': False})

        status, denied = self.request(
            'POST',
            '/api/capabilities/grant',
            {
                'subject': subject['pid'],
                'resource': 'object:gui-actor-grant',
                'rights': ['read'],
                'actor': actor['pid'],
                'confirmed': True,
            },
        )
        assert status == 403
        assert 'lacks grant/admin authority' in denied['error']['message']

        status, spoofed = self.request(
            'POST',
            '/api/capabilities/grant',
            {
                'subject': subject['pid'],
                'resource': 'object:gui-spoofed-human-grant',
                'rights': ['read'],
                'actor': DEFAULT_CONFIG.runtime.default_human_actor,
                'confirmed': True,
            },
        )
        assert status == 403
        assert 'lacks grant/admin authority' in spoofed['error']['message']

        status, admin_granted = self.request(
            'POST',
            '/api/capabilities/grant',
            {
                'subject': subject['pid'],
                'resource': 'object:gui-admin-grant',
                'rights': ['read'],
                'confirmed': True,
            },
        )
        assert status == 200
        assert admin_granted['subject'] == subject['pid']

        runtime.capability.grant(actor['pid'], 'object:gui-actor-grant', [CapabilityRight.READ], issued_by='test')
        runtime.capability.grant(actor['pid'], 'object:gui-actor-grant', [CapabilityRight.GRANT], issued_by='test')
        status, granted = self.request(
            'POST',
            '/api/capabilities/grant',
            {
                'subject': subject['pid'],
                'resource': 'object:gui-actor-grant',
                'rights': ['read'],
                'actor': actor['pid'],
                'confirmed': True,
            },
        )
        assert status == 200
        assert granted['subject'] == subject['pid']
        assert granted['parent_cap_id']

        runtime.capability.grant(actor['pid'], 'object:gui-delegate', [CapabilityRight.READ], issued_by='test', delegable=True)
        status, mismatched_parent = self.request(
            'POST',
            '/api/capabilities/delegate',
            {
                'parent': subject['pid'],
                'child': actor['pid'],
                'resource': 'object:gui-delegate',
                'rights': ['read'],
                'actor': actor['pid'],
                'confirmed': True,
            },
        )
        assert status == 403
        assert 'actor-mode delegation' in mismatched_parent['error']['message']

        status, delegated = self.request(
            'POST',
            '/api/capabilities/delegate',
            {
                'parent': actor['pid'],
                'child': subject['pid'],
                'resource': 'object:gui-delegate',
                'rights': ['read'],
                'actor': actor['pid'],
                'confirmed': True,
            },
        )
        assert status == 200
        assert delegated['subject'] == subject['pid']

        cap = runtime.capability.grant(subject['pid'], 'object:gui-revoke', [CapabilityRight.READ], issued_by='test')
        status, revoke_denied = self.request(
            'POST',
            f"/api/capabilities/{cap.cap_id}/revoke",
            {'actor': actor['pid'], 'confirmed': True},
        )
        assert status == 403
        assert 'lacks revoke/admin authority' in revoke_denied['error']['message']

        runtime.capability.grant(actor['pid'], 'object:gui-revoke', [CapabilityRight.REVOKE], issued_by='test')
        status, revoked = self.request(
            'POST',
            f"/api/capabilities/{cap.cap_id}/revoke",
            {'actor': actor['pid'], 'confirmed': True},
        )
        assert status == 200
        assert revoked['status'] == 'revoked'

    def test_image_register_accepts_package_files_and_rejects_host_file_path(self) -> None:
        files = _gui_image_package_files()
        status, denied = self.request('POST', '/api/images/register', {'files': files, 'source': 'gui-package-agent'})
        assert status == 409
        assert denied['error']['confirmation_required']
        status, string_confirmed = self.request('POST', '/api/images/register', {'files': files, 'source': 'gui-package-agent', 'confirmed': 'true'})
        assert status == 400
        assert 'confirmed must be a JSON boolean' in string_confirmed['error']['message']
        status, path_rejected = self.request('POST', '/api/images/register', {'path': 'image-package', 'confirmed': True})
        assert status == 400
        assert 'package files' in path_rejected['error']['message']
        status, registered = self.request('POST', '/api/images/register', {'files': files, 'source': 'gui-package-agent', 'confirmed': True})
        assert status == 200
        assert registered['image_id'] == 'gui-package-agent:v0'
        assert registered['boot']['kind'] == 'image_package'
        status, listed = self.request('GET', '/api/images')
        assert status == 200
        assert 'gui-package-agent:v0' in {item['image_id'] for item in listed}

    def test_scheduler_requests_are_serialized(self) -> None:
        first_status, first = self.request('POST', '/api/processes', {'goal': 'goal', 'auto_run': False})
        assert first_status == 200
        pid = first['pid']
        self.server.service.scheduler.running = True
        status, duplicate = self.request('POST', f'/api/processes/{pid}/run', {'max_quanta': 1})
        assert status == 200
        assert duplicate['running']
        self.server.service.scheduler.running = False

    def test_scheduler_background_releases_runtime_lock_between_quanta(self) -> None:
        calls: list[tuple[int | None, bool]] = []

        def fake_run_until_idle(
            *,
            max_quanta: int | None = None,
            process_human_queue: bool = True,
            cancel_inflight_on_budget_exhaustion: bool = True,
        ) -> list[dict[str, int]]:
            assert cancel_inflight_on_budget_exhaustion is False
            calls.append((max_quanta, process_human_queue))
            return [{'call': len(calls)}] if len(calls) == 1 else []

        self.server.service.runtime.run_until_idle = fake_run_until_idle

        status = self.server.service.scheduler.start(max_quanta=3, reason='test-batch')
        assert status['running']
        thread = self.server.service.scheduler._thread
        assert thread is not None
        thread.join(timeout=2)

        assert calls == [(1, False), (1, False)]
        assert self.server.service.scheduler.status()['last_result'] == [{'call': 1}]

    def test_scheduler_step_internal_error_is_safe_and_correlation_stable(
        self,
    ) -> None:
        runtime = self.server.service.runtime
        secret = "N5q8Vm2Lc7Xp4Rw9Kd6Hz3Ta"
        path = "/Users/private/gui-runtime.sqlite"
        sql = "SELECT * FROM gui_credentials"
        failure = RuntimeError(
            f"driver failed at {path}; opaque={secret}; SQL={sql}"
        )
        pid = runtime.process.spawn(image="base-agent:v0", goal="GUI scheduler failure")

        async def fail_step(_pid: str) -> None:
            raise failure

        runtime.arun_process_once = fail_step

        status, body = self.request(
            "POST",
            f"/api/processes/{pid}/step",
            {},
        )

        assert status == 500
        error = body["error"]
        assert error["code"] == "internal_error"
        assert error["error_type"] == "RuntimeError"
        assert error["correlation_id"].startswith("corr_")
        assert error["correlation_id"] in error["message"]

        health_status, health = self.request("GET", "/api/health")
        assert health_status == 200
        assert health["scheduler"]["last_error"] == error["message"]

        outward = dumps({"body": body, "health": health})
        assert secret not in outward
        assert path not in outward
        assert sql not in outward

        audit = next(
            record
            for record in runtime.audit.trace()
            if record.action == "gui.request_internal_error"
        )
        assert audit.correlation_id == error["correlation_id"]
        internal = dumps(audit.decision)
        assert secret not in internal
        assert path not in internal
        assert sql not in internal
        observation = audit.decision["internal_error"]
        assert observation["correlation_id"] == error["correlation_id"]
        assert observation["exception_text"]["bytes"] > 0
        assert len(observation["exception_text"]["sha256"]) == 64

    def test_llm_provider_failure_is_text_free_across_gui_surfaces(self) -> None:
        runtime = self.server.service.runtime
        secret = "GUI_PROVIDER_ERROR_SENTINEL_7Xp4Rw9Kd6Hz3Ta"
        private_path = "/Users/private/provider-credentials.json"

        class FailingProviderClient:
            def complete_action(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError(
                    f"provider failed at {private_path}; opaque={secret}"
                )

        runtime.llm.client = FailingProviderClient()
        status, spawned = self.request(
            "POST",
            "/api/processes",
            {"goal": "GUI provider failure", "auto_run": False},
        )
        assert status == 200
        pid = spawned["pid"]

        status, step = self.request(
            "POST",
            f"/api/processes/{pid}/step",
            {},
        )
        assert status == 200
        assert step["result"]["ok"] is False
        correlation_id = step["result"]["error_details"]["correlation_id"]
        assert correlation_id in step["result"]["error"]

        snapshot_status, snapshot = self.request("GET", "/api/snapshot")
        audit_status, audit = self.request("GET", f"/api/processes/{pid}/audit")
        calls_status, calls = self.request(
            "GET",
            f"/api/processes/{pid}/llm-calls",
        )
        assert snapshot_status == audit_status == calls_status == 200

        failed_audit = next(
            record for record in audit if record["action"] == "llm.action_failed"
        )
        assert failed_audit["correlation_id"] == correlation_id
        assert failed_audit["decision"]["error_details"] == step["result"]["error_details"]
        assert calls["items"][0]["error"] == step["result"]["error"]
        assert "observability" not in calls["items"][0]

        process = runtime.process.get(pid)
        assert process.outcome is not None
        assert process.outcome.result_oid is not None
        result_object = runtime.store.get_object(process.outcome.result_oid)
        assert result_object is not None
        assert correlation_id in result_object.payload["message"]

        outward = dumps(
            to_jsonable(
                {
                    "step": step,
                    "snapshot": snapshot,
                    "audit": audit,
                    "calls": calls,
                    "result_object": result_object,
                }
            )
        )
        assert secret not in outward
        assert private_path not in outward

    def test_scheduler_background_internal_error_is_safe_in_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = self.server.service.runtime
        secret = "C6w9Rp3Nk8Xm5Vq2Ld7Hs4Ta"
        path = "/Users/private/gui-background.sqlite"
        failure = RuntimeError(
            f"provider failed at {path}; opaque={secret}; SQL=SELECT private_value"
        )

        def fail_background(**_kwargs: Any) -> None:
            raise failure

        runtime.run_until_idle = fail_background

        original_publish = self.server.service.publish_scheduler_status

        def publish_after_background_settles() -> None:
            thread = self.server.service.scheduler._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2)
            original_publish()

        monkeypatch.setattr(
            self.server.service,
            "publish_scheduler_status",
            publish_after_background_settles,
        )

        status = self.server.service.scheduler.start(max_quanta=1, reason="failure-test")
        assert status["running"]
        thread = self.server.service.scheduler._thread
        assert thread is not None
        thread.join(timeout=2)

        scheduler = self.server.service.scheduler.status()
        assert scheduler["running"] is False
        assert scheduler["last_error"] is not None
        assert "internal_error" in scheduler["last_error"]
        assert "correlation_id=" in scheduler["last_error"]
        encoded = dumps(scheduler)
        assert secret not in encoded
        assert path not in encoded

        audit = next(
            record
            for record in runtime.audit.trace()
            if record.action == "gui.scheduler_background_internal_error"
        )
        assert audit.correlation_id in scheduler["last_error"]
        internal = dumps(audit.decision)
        assert secret not in internal
        assert path not in internal

    def test_scheduler_background_completes_slow_inflight_quantum_at_batch_boundary(self) -> None:
        runtime = self.server.service.runtime
        runtime.scheduler.poll_interval_s = 0.001
        runtime.scheduler.drain_window_s = 0.001
        pid = runtime.process.spawn(image='base-agent:v0', goal='slow GUI quantum')
        completed = threading.Event()

        async def slow_quantum(selected_pid: str) -> dict[str, str]:
            assert selected_pid == pid
            await asyncio.sleep(0.03)
            runtime.process.pause(selected_pid, 'slow quantum completed')
            completed.set()
            return {'pid': selected_pid, 'status': 'completed'}

        runtime.arun_process_once = slow_quantum

        status = self.server.service.scheduler.start(max_quanta=1, reason='slow-batch')
        assert status['running']
        thread = self.server.service.scheduler._thread
        assert thread is not None
        thread.join(timeout=2)

        assert completed.is_set()
        assert runtime.process.get(pid).status == ProcessStatus.PAUSED
        cancellations = [
            record
            for record in runtime.audit.trace()
            if record.action == 'scheduler.process_task_cancelled'
            and record.target == f'process:{pid}'
        ]
        assert cancellations == []
        assert self.server.service.scheduler.status()['last_result'] == [
            {'pid': pid, 'status': 'completed'}
        ]

    def test_health_uses_fast_path_when_runtime_lock_is_busy(self) -> None:
        self.server.service.runtime_lock.acquire()
        try:
            status, health = self.request('GET', '/api/health')
        finally:
            self.server.service.runtime_lock.release()

        assert status == 200
        assert health['runtime_busy'] is True
        assert health['process_count'] is None

    def test_gui_shutdown_waits_for_runtime_users_and_can_retry_after_timeout(self) -> None:
        runtime = Runtime.open('local')
        service = GuiRuntimeService(runtime=runtime, auto_run=False, token='lifecycle-test')
        entered = threading.Event()
        release = threading.Event()
        worker_done = threading.Event()

        def runtime_user() -> None:
            with service.runtime_user():
                entered.set()
                release.wait(timeout=2.0)
            worker_done.set()

        worker = threading.Thread(target=runtime_user)
        worker.start()
        assert entered.wait(timeout=2.0)
        try:
            assert service.shutdown(timeout_s=0.01) is False
            assert not service._closed
            release.set()
            assert worker_done.wait(timeout=2.0)
            assert service.shutdown(timeout_s=1.0) is True
            assert service._closed
            assert runtime.process.list() == []
        finally:
            release.set()
            worker.join(timeout=2.0)
            service.shutdown(timeout_s=1.0)
            runtime.close()

    def test_owned_runtime_partial_shutdown_never_reopens_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = GuiRuntimeService(db='local', auto_run=False, token='partial-shutdown')
        original_shutdown = service.runtime.shutdown
        calls = 0

        def fail_once(*, actor: str, reason: str) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {'ok': False, 'object_tasks_stopped': False}
            return original_shutdown(actor=actor, reason=reason)

        monkeypatch.setattr(service.runtime, 'shutdown', fail_once)
        try:
            assert service.shutdown(timeout_s=1.0) is False
            assert service._closing is True
            assert service._closed is False
            with pytest.raises(GuiServerError, match='shutting down'):
                with service.runtime_user():
                    pass

            assert service.shutdown(timeout_s=1.0) is True
            assert service._closed is True
            assert calls == 2
        finally:
            service.shutdown(timeout_s=1.0)

    def test_process_run_targets_selected_process(self) -> None:
        _first_status, first = self.request('POST', '/api/processes', {'goal': 'first', 'auto_run': False})
        _second_status, second = self.request('POST', '/api/processes', {'goal': 'second', 'auto_run': False})
        seen: list[str] = []
        seen_event = threading.Event()

        async def fake_quantum(pid: str) -> dict[str, str]:
            seen.append(pid)
            self.server.service.runtime.process.pause(pid, 'fake quantum completed')
            seen_event.set()
            return {'pid': pid}
        self.server.service.runtime.arun_process_once = fake_quantum
        status, body = self.request('POST', f"/api/processes/{second['pid']}/run", {'max_quanta': 1})
        assert seen_event.wait(timeout=2.0)
        assert status == 200
        assert body['reason'] == f"run:{second['pid']}"
        assert seen == [second['pid']]
        records = self.server.service.runtime.audit.trace()
        assert not any((record.target == f"process:{first['pid']}" and record.action == 'scheduler.run_quantum' for record in records))
        assert any((record.target == f"process:{second['pid']}" and record.action == 'scheduler.run_quantum' for record in records))

    def test_process_step_returns_and_publishes_final_scheduler_status(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'step once', 'auto_run': False})
        pid = spawned['pid']

        async def fake_quantum(selected_pid: str) -> dict[str, str]:
            assert selected_pid == pid
            return {'pid': selected_pid, 'status': 'completed'}

        self.server.service.runtime.arun_process_once = fake_quantum
        before = self.server.service.broadcaster.replay_after(0)[-1].seq

        status, body = self.request('POST', f'/api/processes/{pid}/step', {})

        assert status == 200
        assert body['started'] is True
        assert body['scheduler']['running'] is False
        snapshots = [
            event.data['snapshot']
            for event in self.server.service.broadcaster.replay_after(before)
            if event.event == 'snapshot'
        ]
        assert snapshots
        assert snapshots[-1]['scheduler']['running'] is False

    def test_workflow_run_endpoint_returns_result_and_snapshot_process(self) -> None:
        status, result = self.request('POST', '/api/workflows/run', {'tool': 'get_working_directory', 'args': {}})

        assert status == 200
        assert result['ok'] is True
        assert result['tool'] == 'get_working_directory'
        assert result['status'] == 'exited'
        assert result['result_oid'] is not None
        status, snapshot = self.request('GET', '/api/snapshot')
        assert status == 200
        processes = {process['pid']: process for process in snapshot['processes']}
        assert processes[result['pid']]['status'] == 'exited'

    @pytest.mark.parametrize("field", ["image", "working_directory"])
    @pytest.mark.parametrize("invalid", [None, 7, [], {}, ""])
    def test_workflow_rejects_invalid_optional_string_fields_before_launch(
        self,
        field: str,
        invalid: Any,
    ) -> None:
        runtime = self.server.service.runtime
        before = {process.pid for process in runtime.process.list()}

        status, body = self.request(
            "POST",
            "/api/workflows/run",
            {
                "tool": "get_working_directory",
                "args": {},
                field: invalid,
                "confirmed": True,
            },
        )

        assert status == 400
        assert f"{field} must be a non-empty JSON string" in body["error"]["message"]
        assert {process.pid for process in runtime.process.list()} == before

    def test_workflow_accepts_valid_optional_string_fields_with_confirmation(self) -> None:
        status, result = self.request(
            "POST",
            "/api/workflows/run",
            {
                "tool": "get_working_directory",
                "args": {},
                "image": "base-agent:v0",
                "working_directory": ".",
                "confirmed": True,
            },
        )

        assert status == 200
        assert result["ok"] is True
        assert result["status"] == "exited"

    @pytest.mark.parametrize("invalid", [None, 7, False, [], {}, ""])
    def test_workflow_rejects_non_string_tool_before_confirmation_or_launch(
        self,
        invalid: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = self.server.service.runtime
        audit_calls: list[str] = []
        workflow_calls: list[str] = []
        original_record = runtime.audit.record
        original_run_workflow = runtime.run_workflow

        def tracked_record(*args: Any, **kwargs: Any):
            audit_calls.append(str(kwargs.get("action") or ""))
            return original_record(*args, **kwargs)

        def tracked_run_workflow(tool: str, *args: Any, **kwargs: Any):
            workflow_calls.append(tool)
            return original_run_workflow(tool, *args, **kwargs)

        monkeypatch.setattr(runtime.audit, "record", tracked_record)
        monkeypatch.setattr(runtime, "run_workflow", tracked_run_workflow)

        status, body = self.request(
            "POST",
            "/api/workflows/run",
            {"tool": invalid, "args": {}},
        )

        assert status == 400
        assert "tool must be a non-empty JSON string" in body["error"]["message"]
        assert "gui.confirmation_required" not in audit_calls
        assert workflow_calls == []

    def test_side_effect_workflow_requires_confirmation(self) -> None:
        status, denied = self.request('POST', '/api/workflows/run', {'tool': 'ask_human', 'args': {'question': 'Continue?'}})

        assert status == 409
        assert denied['error']['confirmation_required']
        assert denied['error']['action'] == 'workflow.run'
        assert denied['error']['preview']['tool'] == 'ask_human'

    def test_unknown_workflow_tool_requires_confirmation_fail_closed(self) -> None:
        status, denied = self.request('POST', '/api/workflows/run', {'tool': 'missing_workflow_tool', 'args': {}})

        assert status == 409
        assert denied['error']['confirmation_required']
        assert denied['error']['action'] == 'workflow.run'

    def test_object_task_endpoint_runs_task_and_exposes_snapshot(self) -> None:
        status, spawned = self.request('POST', '/api/processes', {'goal': 'object task', 'auto_run': False})
        assert status == 200
        pid = spawned['pid']
        self.server.service.runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
        owner = self.server.service.runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'name': 'owner'},
            metadata=ObjectMetadata(title='owner'),
            immutable=False,
        )

        status, started = self.request(
            'POST',
            '/api/object-tasks/start',
            {
                'pid': pid,
                'owner_oid': owner.oid,
                'tool': 'get_working_directory',
                'args': {},
                'owner_watch': True,
                'watch_events': ['updated'],
                'watch_channel': 'owner-watch',
            },
        )
        assert status == 200
        assert started['owner_watch']['enabled'] is True
        assert started['owner_watch']['events'] == ['updated']
        assert started['owner_watch']['channel'] == 'owner-watch'
        status, waited = self.request('POST', f"/api/object-tasks/{started['task_id']}/wait", {'pid': pid, 'timeout_s': 2})
        assert status == 200
        assert waited['status'] == 'succeeded'
        assert waited['result_oid'] is not None
        status, snapshot = self.request('GET', '/api/snapshot')
        assert status == 200
        assert any(
            task['task_id'] == started['task_id']
            and task['status'] == 'succeeded'
            and task['owner_watch']['enabled'] is True
            for task in snapshot['object_tasks']
        )

    def test_object_task_watch_owner_endpoint_updates_existing_task(self) -> None:
        status, spawned = self.request('POST', '/api/processes', {'goal': 'object task watch', 'auto_run': False})
        assert status == 200
        pid = spawned['pid']
        self.server.service.runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
        owner = self.server.service.runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'name': 'owner'},
            metadata=ObjectMetadata(title='owner'),
            immutable=False,
        )
        status, started = self.request(
            'POST',
            '/api/object-tasks/start',
            {'pid': pid, 'owner_oid': owner.oid, 'tool': 'receive_process_messages', 'args': {'channel': 'owner-watch'}},
        )
        assert status == 200
        status, waited = self.request('POST', f"/api/object-tasks/{started['task_id']}/wait", {'pid': pid, 'timeout_s': 2})
        assert status == 200
        assert waited['status'] == 'waiting_message'

        status, watched = self.request(
            'POST',
            f"/api/object-tasks/{started['task_id']}/watch-owner",
            {
                'pid': pid,
                'enabled': True,
                'watch_events': ['updated'],
                'watch_channel': 'owner-watch',
                'watch_kind': 'interrupt',
            },
        )

        assert status == 200
        assert watched['owner_watch']['enabled'] is True
        assert watched['owner_watch']['events'] == ['updated']
        assert watched['owner_watch']['channel'] == 'owner-watch'
        assert watched['owner_watch']['kind'] == 'interrupt'

    def test_object_task_start_rejects_invalid_watch_kind_as_bad_request(self) -> None:
        status, spawned = self.request('POST', '/api/processes', {'goal': 'bad watch kind', 'auto_run': False})
        assert status == 200
        pid = spawned['pid']
        owner = self.server.service.runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'name': 'owner'},
            metadata=ObjectMetadata(title='owner'),
            immutable=False,
        )

        status, body = self.request(
            'POST',
            '/api/object-tasks/start',
            {
                'pid': pid,
                'owner_oid': owner.oid,
                'tool': 'get_working_directory',
                'args': {},
                'owner_watch': True,
                'watch_kind': 'bad-kind',
            },
        )

        assert status == 400
        assert 'owner watch kind' in body['error']['message']

    def test_object_task_wait_uses_bounded_timeout(self) -> None:
        seen: list[float | None] = []

        def fake_wait(task_id: str, *, actor_pid: str | None = None, timeout: float | None = None) -> dict[str, object]:
            seen.append(timeout)
            return {'task_id': task_id, 'actor_pid': actor_pid, 'timeout': timeout, 'status': 'running'}

        self.server.service.runtime.object_tasks.wait = fake_wait  # type: ignore[method-assign]

        status, body = self.request('POST', '/api/object-tasks/task-1/wait', {'pid': 'pid-1'})
        assert status == 200
        assert body['timeout'] == DEFAULT_CONFIG.gui.object_task_wait_default_timeout_s
        assert seen == [DEFAULT_CONFIG.gui.object_task_wait_default_timeout_s]

        status, body = self.request('POST', '/api/object-tasks/task-1/wait', {'timeout_s': 'nan'})
        assert status == 400
        assert 'finite' in body['error']['message']

        status, body = self.request(
            'POST',
            '/api/object-tasks/task-1/wait',
            {'timeout_s': DEFAULT_CONFIG.gui.object_task_wait_max_timeout_s + 1},
        )
        assert status == 400
        assert 'at most' in body['error']['message']

    def test_injected_runtime_config_controls_spawn_and_wait_defaults(self) -> None:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(default_image_id='gui-base:v0', coding_image_id='gui-coding:v0'),
            gui=replace(DEFAULT_CONFIG.gui, object_task_wait_default_timeout_s=0.25, object_task_wait_max_timeout_s=0.5),
        )
        runtime = Runtime.open(config=config)
        server = create_gui_http_server(runtime=runtime, port=0, token='custom-token', auto_run=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        seen: list[float | None] = []

        def fake_wait(task_id: str, *, actor_pid: str | None = None, timeout: float | None = None) -> dict[str, object]:
            seen.append(timeout)
            return {'task_id': task_id, 'actor_pid': actor_pid, 'timeout': timeout, 'status': 'running'}

        server.service.runtime.object_tasks.wait = fake_wait  # type: ignore[method-assign]
        thread.start()
        try:
            status, spawned = _request_to_server(server, 'POST', '/api/processes', {'goal': 'custom', 'auto_run': False}, token='custom-token')
            assert status == 200
            assert spawned['process']['image_id'] == 'gui-base:v0'

            status, body = _request_to_server(server, 'POST', '/api/object-tasks/task-1/wait', {'pid': spawned['pid']}, token='custom-token')
            assert status == 200
            assert body['timeout'] == 0.25
            assert seen == [0.25]

            status, body = _request_to_server(server, 'POST', '/api/object-tasks/task-1/wait', {'timeout_s': 0.75}, token='custom-token')
            assert status == 400
            assert '0.5 seconds' in body['error']['message']
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.service.shutdown()
            server.server_close()

    def test_config_argument_controls_gui_runtime_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        target = str(tmp_path / 'gui-memory.sqlite')
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(
                local_store_target=target,
                default_image_id='configured-gui-base:v0',
                coding_image_id='configured-gui-coding:v0',
            ),
            gui=replace(DEFAULT_CONFIG.gui, object_task_wait_default_timeout_s=0.2, object_task_wait_max_timeout_s=0.4),
        )
        server = create_gui_http_server(config=config, port=0, token='custom-token', auto_run=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        seen: list[float | None] = []

        def fake_wait(task_id: str, *, actor_pid: str | None = None, timeout: float | None = None) -> dict[str, object]:
            seen.append(timeout)
            return {'task_id': task_id, 'actor_pid': actor_pid, 'timeout': timeout, 'status': 'running'}

        server.service.runtime.object_tasks.wait = fake_wait  # type: ignore[method-assign]
        thread.start()
        try:
            assert server.service.db == target
            assert server.service.runtime.store.path == target

            status, spawned = _request_to_server(server, 'POST', '/api/processes', {'goal': 'custom', 'auto_run': False}, token='custom-token')
            assert status == 200
            assert spawned['process']['image_id'] == 'configured-gui-base:v0'

            status, body = _request_to_server(server, 'POST', '/api/object-tasks/task-1/wait', {'pid': spawned['pid']}, token='custom-token')
            assert status == 200
            assert body['timeout'] == 0.2
            assert seen == [0.2]
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.service.shutdown()
            server.server_close()

    def test_gui_runtime_service_redacts_postgres_dsn_in_status_payloads(self) -> None:
        dsn = 'postgresql://agent:secret@localhost:5432/agent_libos'
        runtime = Runtime.open('local')
        server = create_gui_http_server(runtime=runtime, db=dsn, port=0, token='custom-token', auto_run=False)
        try:
            redacted = 'postgresql://***@localhost:5432/agent_libos'
            assert server.service.db == redacted
            assert server.service.health()['db'] == redacted
            assert server.service.snapshot()['db'] == redacted
        finally:
            server.service.shutdown()
            server.server_close()
            runtime.close()

    def test_gui_runtime_service_uses_configured_postgres_dsn_when_db_is_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(
                store_backend='postgres',
                store_dsn='postgresql://agent:secret@localhost:5432/agent_libos',
            )
        )
        calls: dict[str, object] = {}
        original_open = Runtime.open

        def fake_open(target: object = None, **kwargs: object) -> Runtime:
            calls['target'] = target
            calls['config'] = kwargs.get('config')
            return original_open('local')

        monkeypatch.setattr(Runtime, 'open', staticmethod(fake_open))
        server = create_gui_http_server(config=config, port=0, token='custom-token', auto_run=False)
        try:
            redacted = 'postgresql://***@localhost:5432/agent_libos'

            assert calls['target'] is None
            assert server.service.db == redacted
            assert server.service.health()['db'] == redacted
            assert server.service.snapshot()['db'] == redacted
        finally:
            server.service.shutdown()
            server.server_close()

    def test_injected_runtime_config_controls_request_body_limit(self) -> None:
        runtime = Runtime.open(config=AgentLibOSConfig(gui=GuiDefaults(request_body_max_bytes=8)))
        server = create_gui_http_server(runtime=runtime, port=0, token='custom-token', auto_run=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, body = _request_to_server(server, 'POST', '/api/scheduler/auto', {'enabled': True}, token='custom-token')
            assert status == 413
            assert '8 bytes' in body['error']['message']
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.service.shutdown()
            server.server_close()

    def test_jsonrpc_register_rejects_host_file_path(self) -> None:
        status, body = self.request('POST', '/api/jsonrpc/register', {'path': 'secrets.yaml', 'confirmed': True})
        assert status == 400
        assert 'manifest_text' in body['error']['message']

    def test_jsonrpc_register_requires_manifest_text(self) -> None:
        status, body = self.request('POST', '/api/jsonrpc/register', {'confirmed': True})
        assert status == 400
        assert 'manifest_text' in body['error']['message']

    def test_jsonrpc_register_actor_mode_requires_endpoint_write_capability(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'jsonrpc actor', 'auto_run': False})
        pid = spawned['pid']
        manifest = _gui_jsonrpc_manifest('gui-actor-jsonrpc')

        status, denied = self.request(
            'POST',
            '/api/jsonrpc/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )

        assert status == 403
        assert 'jsonrpc_endpoint:gui-actor-jsonrpc' in denied['error']['message']

        self.server.service.runtime.capability.grant(
            pid,
            'jsonrpc_endpoint:gui-actor-jsonrpc',
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        status, registered = self.request(
            'POST',
            '/api/jsonrpc/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )

        assert status == 200
        assert registered['endpoint_id'] == 'gui-actor-jsonrpc'

    def test_mcp_register_rejects_host_file_path(self) -> None:
        status, body = self.request('POST', '/api/mcp/register', {'path': 'secrets.yaml', 'confirmed': True})
        assert status == 400
        assert 'manifest_text' in body['error']['message']

    def test_mcp_register_rejects_non_string_source_before_mutation(self) -> None:
        status, body = self.request(
            'POST',
            '/api/mcp/register',
            {
                'manifest_text': _gui_mcp_manifest('gui-invalid-source'),
                'source': 7,
                'confirmed': True,
            },
        )

        assert status == 400
        assert body['error']['message'] == 'source must be a JSON string or null'
        assert self.server.service.runtime.store.get_mcp_server('gui-invalid-source') is None

    def test_mcp_register_replace_and_unregister_are_explicit_confirmed_operations(self) -> None:
        manifest = _gui_mcp_manifest('gui-replace-mcp')
        replacement = manifest.replace('timeout_s: 5', 'timeout_s: 6')

        first_status, first = self.request(
            'POST',
            '/api/mcp/register',
            {'manifest_text': manifest, 'confirmed': True},
        )
        duplicate_status, _duplicate = self.request(
            'POST',
            '/api/mcp/register',
            {'manifest_text': replacement, 'confirmed': True},
        )
        replace_status, replaced = self.request(
            'POST',
            '/api/mcp/register',
            {'manifest_text': replacement, 'replace': True, 'confirmed': True},
        )
        confirmation_status, confirmation = self.request(
            'POST',
            '/api/mcp/gui-replace-mcp/unregister',
            {},
        )
        unregister_status, unregistered = self.request(
            'POST',
            '/api/mcp/gui-replace-mcp/unregister',
            {'confirmed': True},
        )

        assert first_status == 200
        assert first['timeout_s'] == 5
        assert duplicate_status == 400
        assert replace_status == 200
        assert replaced['timeout_s'] == 6
        assert confirmation_status == 409
        assert confirmation['error']['action'] == 'mcp.unregister'
        assert unregister_status == 200
        assert unregistered == {'server_id': 'gui-replace-mcp', 'deleted': True}
        assert self.server.service.runtime.store.get_mcp_server('gui-replace-mcp') is None

    def test_mcp_unregister_actor_mode_requires_exact_server_admin_capability(self) -> None:
        runtime = self.server.service.runtime
        _status, spawned = self.request(
            'POST',
            '/api/processes',
            {'goal': 'mcp unregister actor', 'auto_run': False},
        )
        pid = spawned['pid']
        runtime.mcp.register_server_from_yaml_text(
            _gui_mcp_manifest('gui-unregister-actor'),
            actor='test',
            require_capability=False,
        )

        typo_status, typo = self.request(
            'POST',
            '/api/mcp/gui-unregister-actor/unregister',
            {'actor_pid': pid, 'confirmed': True},
        )

        denied_status, denied = self.request(
            'POST',
            '/api/mcp/gui-unregister-actor/unregister',
            {'actor': pid, 'confirmed': True},
        )

        assert typo_status == 400
        assert typo['error']['code'] == 'unknown_request_field'
        assert denied_status == 403
        assert 'mcp_server:gui-unregister-actor' in denied['error']['message']
        assert runtime.store.get_mcp_server('gui-unregister-actor') is not None

        runtime.capability.grant(
            pid,
            'mcp_server:gui-unregister-actor',
            [CapabilityRight.ADMIN],
            issued_by='test',
        )
        allowed_status, allowed = self.request(
            'POST',
            '/api/mcp/gui-unregister-actor/unregister',
            {'actor': pid, 'confirmed': True},
        )

        assert allowed_status == 200
        assert allowed == {'server_id': 'gui-unregister-actor', 'deleted': True}
        assert runtime.store.get_mcp_server('gui-unregister-actor') is None

    def test_mcp_tools_get_is_cache_only_and_live_refresh_requires_post(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed: list[dict[str, object]] = []

        def record_list_tools(
            server_id: str,
            *,
            actor: str,
            require_capability: bool,
            refresh: bool,
        ) -> dict[str, object]:
            observed.append(
                {
                    'server_id': server_id,
                    'actor': actor,
                    'require_capability': require_capability,
                    'refresh': refresh,
                }
            )
            return {
                'server_id': server_id,
                'schema_version': 2,
                'transport': 'stdio',
                'protocol_mode': 'auto',
                'tools': [],
                'refreshed': refresh,
                'response_bytes': 0,
            }

        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            'list_tools',
            record_list_tools,
        )
        cursor = self.server.service.broadcaster.replay_after(0)[-1].seq

        cached_status, cached = self.request(
            'GET',
            '/api/mcp/gui-modern-mcp/tools?refresh=false',
        )
        for query in (
            'refresh=true',
            'refresh=treu',
            'refresh=',
            'refresh=false&refresh=false',
            'unknown=false',
        ):
            status, _body = self.request(
                'GET',
                f'/api/mcp/gui-modern-mcp/tools?{query}',
            )
            assert status == 400
        typo_status, typo = self.request(
            'POST',
            '/api/mcp/gui-modern-mcp/tools/refresh',
            {'actor_pid': 'pid-modern-reader'},
        )
        refresh_status, refreshed = self.request(
            'POST',
            '/api/mcp/gui-modern-mcp/tools/refresh',
            {'actor': 'pid-modern-reader'},
        )
        published_reasons = [
            event.data['reason']
            for event in self.server.service.broadcaster.replay_after(cursor)
            if event.event == 'snapshot'
        ]

        assert cached_status == 200
        assert cached['refreshed'] is False
        assert typo_status == 400
        assert typo['error']['code'] == 'unknown_request_field'
        assert refresh_status == 200
        assert refreshed['refreshed'] is True
        assert published_reasons == ['mcp.tools.refresh']
        assert observed == [
            {
                'server_id': 'gui-modern-mcp',
                'actor': 'gui',
                'require_capability': False,
                'refresh': False,
            },
            {
                'server_id': 'gui-modern-mcp',
                'actor': 'pid-modern-reader',
                'require_capability': True,
                'refresh': True,
            },
        ]

    def test_mcp_v3_resources_are_post_only_strict_and_use_host_actor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed: list[dict[str, Any]] = []

        def list_resources(
            server_id: str,
            *,
            cursor: str | None,
            actor: str,
        ) -> dict[str, Any]:
            observed.append({"server_id": server_id, "cursor": cursor, "actor": actor})
            return {
                "items": [{"resource_id": "logical-doc", "name": "Document"}],
                "next_cursor": "opaque-next",
                "cache_hint": None,
            }

        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            "list_resources",
            list_resources,
            raising=False,
        )

        get_status, _get = self.request(
            "GET", "/api/mcp/modern/resources/list"
        )
        unknown_status, unknown = self.request(
            "POST",
            "/api/mcp/modern/resources/list",
            {"cursor": "c1", "url": "https://hidden.invalid"},
        )
        query_status, query_error = self.request(
            "POST",
            "/api/mcp/modern/resources/list?url=https%3A%2F%2Fhidden.invalid",
            {},
        )
        status, page = self.request(
            "POST", "/api/mcp/modern/resources/list", {"cursor": "c1"}
        )

        assert get_status == 404
        assert unknown_status == 400
        assert unknown["error"]["code"] == "unknown_request_field"
        assert query_status == 400
        assert query_error["error"]["code"] == "unknown_query_parameter"
        assert status == 200
        assert page["next_cursor"] == "opaque-next"
        assert observed == [{"server_id": "modern", "cursor": "c1", "actor": "gui"}]

    def test_mcp_v3_resource_read_rejects_non_string_variables_pre_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0

        def read_resource(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return {"kind": "complete", "value": None}

        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            "read_resource",
            read_resource,
            raising=False,
        )

        status, body = self.request(
            "POST",
            "/api/mcp/modern/resources/read",
            {"resource_id": "doc", "variables": {"page": 1}},
        )

        assert status == 400
        assert "string values" in body["error"]["message"]
        assert calls == 0

    def test_mcp_v3_prompt_preview_confirmation_and_async_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed: list[bool] = []

        async def get_prompt(
            server_id: str,
            prompt_id: str,
            *,
            arguments: dict[str, str] | None,
            confirmed: bool,
            expected_preview_sha256: str | None,
            actor: str,
        ) -> dict[str, Any]:
            assert (server_id, prompt_id, arguments, actor) == (
                "modern",
                "review",
                {"topic": "MCP"},
                "gui",
            )
            if confirmed:
                assert expected_preview_sha256 == "a" * 64
            else:
                assert expected_preview_sha256 is None
            observed.append(confirmed)
            return {
                "kind": "complete",
                "preview_sha256": "a" * 64,
                "value": {
                    "prompt_id": "review",
                    "messages": [],
                    "user_confirmation_required": True,
                },
            }

        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            "get_prompt",
            get_prompt,
            raising=False,
        )

        preview_status, _preview = self.request(
            "POST",
            "/api/mcp/modern/prompts/get",
            {"prompt_id": "review", "arguments": {"topic": "MCP"}},
        )
        unbound_status, unbound = self.request(
            "POST",
            "/api/mcp/modern/prompts/get",
            {
                "prompt_id": "review",
                "arguments": {"topic": "MCP"},
                "confirmed": True,
            },
        )
        confirmed_status, _confirmed = self.request(
            "POST",
            "/api/mcp/modern/prompts/get",
            {
                "prompt_id": "review",
                "arguments": {"topic": "MCP"},
                "confirmed": True,
                "expected_preview_sha256": "a" * 64,
            },
        )

        assert preview_status == 200
        assert unbound_status == 400
        assert unbound["error"]["code"] == "mcp_prompt_preview_binding_required"
        assert confirmed_status == 200
        assert observed == [False, True]

    def test_mcp_v3_completion_context_is_a_strict_string_map_pre_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed: list[dict[str, Any]] = []

        def complete_prompt(
            server_id: str,
            reference_type: str,
            reference_id: str,
            argument: dict[str, str],
            *,
            context: dict[str, str] | None,
            actor: str,
        ) -> dict[str, Any]:
            observed.append(
                {
                    "server_id": server_id,
                    "reference_type": reference_type,
                    "reference_id": reference_id,
                    "argument": argument,
                    "context": context,
                    "actor": actor,
                }
            )
            return {
                "kind": "complete",
                "value": {"values": ["one"], "has_more": False},
            }

        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            "complete_prompt",
            complete_prompt,
            raising=False,
        )
        invalid_status, _invalid = self.request(
            "POST",
            "/api/mcp/modern/completion",
            {
                "reference_type": "ref/prompt",
                "reference_id": "review",
                "argument": {"name": "topic", "value": "MCP"},
                "context": {"count": 1},
            },
        )
        blank_key_status, _blank_key = self.request(
            "POST",
            "/api/mcp/modern/completion",
            {
                "reference_type": "ref/prompt",
                "reference_id": "review",
                "argument": {"name": "topic", "value": "MCP"},
                "context": {"   ": "hidden"},
            },
        )
        status, _result = self.request(
            "POST",
            "/api/mcp/modern/completion",
            {
                "reference_type": "ref/prompt",
                "reference_id": "review",
                "argument": {"name": "topic", "value": "MCP"},
                "context": {"tenant": "local"},
            },
        )

        assert invalid_status == 400
        assert blank_key_status == 400
        assert status == 200
        assert observed == [{
            "server_id": "modern",
            "reference_type": "ref/prompt",
            "reference_id": "review",
            "argument": {"name": "topic", "value": "MCP"},
            "context": {"tenant": "local"},
            "actor": "gui",
        }]

    def test_mcp_v3_gui_host_resource_and_prompt_mrtr_round_trip(self) -> None:
        import contextlib
        import time

        mcp_types = pytest.importorskip("mcp.types")
        from agent_libos.mcp import (
            InMemoryMcpCredentialBroker,
            McpPromptSpec,
            McpResourceSpec,
            McpSdkV2SessionProvider,
            McpServerManifestV3,
        )
        from agent_libos.models import McpHttpTransportSpec

        runtime = self.server.service.runtime
        manifest = McpServerManifestV3(
            schema_version=3,
            server_id="gui-host-mrtr",
            transport="streamable_http",
            http=McpHttpTransportSpec(url="http://127.0.0.1:8765/mcp"),
            timeout_s=2.0,
            max_request_bytes=16_384,
            max_response_bytes=16_384,
            protocol_mode=McpProtocolMode.REVISION_2026_07_28,
            resources=(
                McpResourceSpec(
                    resource_id="document",
                    remote_uri="opaque://provider/document",
                ),
            ),
            prompts=(
                McpPromptSpec(
                    prompt_id="review",
                    mcp_name="provider.review",
                    argument_names=("subject",),
                ),
            ),
        )
        runtime.mcp.register_server(
            manifest,
            actor="runtime",
            require_capability=False,
        )
        broker = InMemoryMcpCredentialBroker()
        runtime._mcp_continuation_manager.broker = broker
        initial_calls: list[tuple[str, Any]] = []

        class Session:
            protocol_version = "2026-07-28"

            async def read_resource(self, selector: str, **kwargs: Any) -> Any:
                initial_calls.append(("resource", (selector, kwargs)))
                return mcp_types.InputRequiredResult(
                    inputRequests={
                        "resource-confirm": mcp_types.ElicitRequest(
                            params=mcp_types.ElicitRequestFormParams(
                                message="Approve the untrusted Resource?",
                                requestedSchema={
                                    "type": "object",
                                    "properties": {
                                        "approved": {"type": "boolean"}
                                    },
                                    "required": ["approved"],
                                },
                            )
                        )
                    },
                    requestState="resource-state",
                )

            async def get_prompt(
                self,
                name: str,
                arguments: dict[str, str],
                **kwargs: Any,
            ) -> Any:
                initial_calls.append(("prompt", (name, arguments, kwargs)))
                return mcp_types.InputRequiredResult(
                    inputRequests={
                        "prompt-confirm": mcp_types.ElicitRequest(
                            params=mcp_types.ElicitRequestFormParams(
                                message="Approve the untrusted Prompt?",
                                requestedSchema={
                                    "type": "object",
                                    "properties": {
                                        "approved": {"type": "boolean"}
                                    },
                                    "required": ["approved"],
                                },
                            )
                        )
                    },
                    requestState="prompt-state",
                )

        @contextlib.asynccontextmanager
        async def session_factory(_server: Any, *, deadline: float) -> Any:
            assert deadline > time.monotonic()
            yield Session()

        sdk_provider = McpSdkV2SessionProvider(
            session_factory,
            result_adapter=runtime._mcp_v3_tool_provider.result_adapter,
        )
        runtime.mcp._modern_client.resource_provider = sdk_provider  # noqa: SLF001
        runtime.mcp._modern_client.prompt_provider = sdk_provider  # noqa: SLF001
        continuation_calls: list[tuple[str, Any]] = []

        class ContinuationProvider:
            async def continue_resource(
                self,
                server: Any,
                resource_name: str,
                logical_id: str,
                input_responses: dict[str, Any],
                request_state: str | None,
                *,
                deadline: float,
            ) -> dict[str, Any]:
                assert deadline > time.monotonic()
                continuation_calls.append(
                    (
                        "resource",
                        (
                            server.server_id,
                            resource_name,
                            logical_id,
                            input_responses,
                            request_state,
                        ),
                    )
                )
                return {
                    "resultType": "complete",
                    "resource_id": logical_id,
                    "contents": [
                        {
                            "kind": "text",
                            "text": "approved resource",
                            "annotations": None,
                            "metadata": {},
                        }
                    ],
                    "provenance": "untrusted_mcp_resource",
                }

            async def continue_prompt(
                self,
                server: Any,
                prompt_name: str,
                logical_id: str,
                arguments: dict[str, str],
                input_responses: dict[str, Any],
                request_state: str | None,
                *,
                deadline: float,
            ) -> dict[str, Any]:
                assert deadline > time.monotonic()
                continuation_calls.append(
                    (
                        "prompt",
                        (
                            server.server_id,
                            prompt_name,
                            logical_id,
                            arguments,
                            input_responses,
                            request_state,
                        ),
                    )
                )
                return {
                    "resultType": "complete",
                    "prompt_id": logical_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "kind": "text",
                                "text": "approved prompt",
                                "annotations": None,
                                "metadata": {},
                            },
                            "provenance": "untrusted_mcp_prompt",
                        }
                    ],
                    "description": None,
                    "user_confirmation_required": True,
                }

        runtime.mcp._modern_continuation_provider = ContinuationProvider()  # noqa: SLF001

        def respond(pending: dict[str, Any]) -> dict[str, Any]:
            assert pending["kind"] == "input_required"
            status, result = self.request(
                "POST",
                f"/api/mcp/continuations/{pending['continuation_id']}/respond",
                {
                    "expected_revision": pending["revision"],
                    "responses": {
                        "input-1": {
                            "action": "accept",
                            "content": {"approved": True},
                        }
                    },
                    "human_request_id": pending["human_request_id"],
                    "human_expected_revision": pending["human_revision"],
                    "human_preview_sha256": pending["human_preview_sha256"],
                    "confirmed": True,
                },
            )
            assert status == 200
            return result

        try:
            resource_status, resource_pending = self.request(
                "POST",
                "/api/mcp/gui-host-mrtr/resources/read",
                {"resource_id": "document"},
            )
            assert resource_status == 200, resource_pending
            resource_complete = respond(resource_pending)

            prompt_status, prompt_pending = self.request(
                "POST",
                "/api/mcp/gui-host-mrtr/prompts/get",
                {"prompt_id": "review", "arguments": {"subject": "release"}},
            )
            assert prompt_status == 200, prompt_pending
            prompt_complete = respond(prompt_pending)
        finally:
            broker.close()

        assert resource_complete["value"]["resource_id"] == "document"
        assert (
            resource_complete["value"]["provenance"]
            == "untrusted_mcp_resource"
        )
        assert prompt_complete["value"]["prompt_id"] == "review"
        assert prompt_complete["value"]["user_confirmation_required"] is True
        assert (
            prompt_complete["value"]["messages"][0]["provenance"]
            == "untrusted_mcp_prompt"
        )
        assert len(initial_calls) == 2
        assert continuation_calls == [
            (
                "resource",
                (
                    "gui-host-mrtr",
                    "opaque://provider/document",
                    "document",
                    {
                        "resource-confirm": {
                            "action": "accept",
                            "content": {"approved": True},
                        }
                    },
                    "resource-state",
                ),
            ),
            (
                "prompt",
                (
                    "gui-host-mrtr",
                    "provider.review",
                    "review",
                    {"subject": "release"},
                    {
                        "prompt-confirm": {
                            "action": "accept",
                            "content": {"approved": True},
                        }
                    },
                    "prompt-state",
                ),
            ),
        ]

    def test_mcp_v3_continuation_and_task_answers_require_human_receipts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

        def respond(*args: Any, **kwargs: Any) -> dict[str, Any]:
            observed.append(("respond", args, kwargs))
            return {"kind": "complete", "value": None}

        def update(*args: Any, **kwargs: Any) -> dict[str, Any]:
            observed.append(("update", args, kwargs))
            return {
                "kind": "remote_task",
                "task_ref": "task-local",
                "revision": 8,
                "status": "working",
                "input_requests": [],
            }

        mcp = self.server.service.runtime.mcp
        monkeypatch.setattr(mcp, "respond_continuation", respond, raising=False)
        monkeypatch.setattr(mcp, "update_remote_task", update, raising=False)
        missing_status, _missing = self.request(
            "POST",
            "/api/mcp/continuations/continuation-local/respond",
            {"expected_revision": 3, "responses": {"field": "yes"}, "confirmed": True},
        )
        malformed_status, _malformed = self.request(
            "POST",
            "/api/mcp/remote-tasks/task-local/update",
            {
                "expected_revision": 7,
                "responses": {"field": "yes"},
                "human_request_id": "human-local",
                "human_expected_revision": 4,
                "human_preview_sha256": "C" * 64,
                "confirmed": True,
            },
        )
        receipt = {
            "human_request_id": "human-local",
            "human_expected_revision": 4,
            "human_preview_sha256": "c" * 64,
        }
        continuation_status, _continuation = self.request(
            "POST",
            "/api/mcp/continuations/continuation-local/respond",
            {
                "expected_revision": 3,
                "responses": {"field": "yes"},
                **receipt,
                "confirmed": True,
            },
        )
        task_status, _task = self.request(
            "POST",
            "/api/mcp/remote-tasks/task-local/update",
            {
                "expected_revision": 7,
                "responses": {"field": "yes"},
                **receipt,
                "confirmed": True,
            },
        )

        assert missing_status == 400
        assert malformed_status == 400
        assert continuation_status == 200
        assert task_status == 200
        assert [item[0] for item in observed] == ["respond", "update"]
        assert observed[0][1] == ("continuation-local",)
        assert observed[0][2] == {
            "expected_revision": 3,
            "responses": {"field": "yes"},
            "human_request_id": "human-local",
            "human_expected_revision": 4,
            "human_preview_sha256": "c" * 64,
            "actor": "gui",
        }
        assert observed[1][1] == ("task-local",)
        assert observed[1][2] == {
            "expected_revision": 7,
            "responses": {"field": "yes"},
            "human_request_id": "human-local",
            "human_expected_revision": 4,
            "human_preview_sha256": "c" * 64,
            "actor": "gui",
        }

    def test_mcp_v3_continuation_inspect_and_task_reobserve_restore_durable_views(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        continuation = {
            "kind": "input_required",
            "continuation_id": "continuation-reopened",
            "revision": 12,
            "respondable": True,
            "input_requests": [],
            "human_request_id": "human-reopened",
            "human_revision": 4,
            "human_preview_sha256": "a" * 64,
        }
        task = {
            "kind": "remote_task",
            "task_ref": "task-reopened",
            "revision": 19,
            "status": "working",
            "input_requests": [],
        }

        def inspect(*args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(("inspect", args, kwargs))
            return continuation

        def reobserve(*args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(("get", args, kwargs))
            return task

        mcp = self.server.service.runtime.mcp
        monkeypatch.setattr(mcp, "get_continuation", inspect, raising=False)
        monkeypatch.setattr(mcp, "get_remote_task", reobserve, raising=False)

        inspect_status, inspect_body = self.request(
            "POST",
            "/api/mcp/continuations/continuation-reopened/inspect",
            {},
        )
        unknown_status, _unknown = self.request(
            "POST",
            "/api/mcp/continuations/continuation-reopened/inspect",
            {"raw_request_state": "must-not-be-accepted"},
        )
        absent_status, _absent = self.request(
            "POST", "/api/mcp/remote-tasks/task-reopened/get", {}
        )
        null_status, _null = self.request(
            "POST",
            "/api/mcp/remote-tasks/task-reopened/get",
            {"expected_revision": None},
        )
        cas_status, _cas = self.request(
            "POST",
            "/api/mcp/remote-tasks/task-reopened/get",
            {"expected_revision": 19},
        )
        invalid_status, _invalid = self.request(
            "POST",
            "/api/mcp/remote-tasks/task-reopened/get",
            {"expected_revision": True},
        )
        update_without_cas_status, _update_without_cas = self.request(
            "POST",
            "/api/mcp/remote-tasks/task-reopened/update",
            {
                "responses": {},
                "human_request_id": "human-reopened",
                "human_expected_revision": 4,
                "human_preview_sha256": "a" * 64,
                "confirmed": True,
            },
        )
        cancel_without_cas_status, _cancel_without_cas = self.request(
            "POST",
            "/api/mcp/remote-tasks/task-reopened/cancel",
            {"confirmed": True},
        )

        assert inspect_status == 200
        assert inspect_body == continuation
        assert unknown_status == 400
        assert (
            absent_status,
            null_status,
            cas_status,
            invalid_status,
            update_without_cas_status,
            cancel_without_cas_status,
        ) == (
            200,
            200,
            200,
            400,
            400,
            400,
        )
        assert calls == [
            (
                "inspect",
                ("continuation-reopened",),
                {"actor": "gui"},
            ),
            (
                "get",
                ("task-reopened",),
                {"expected_revision": None, "actor": "gui"},
            ),
            (
                "get",
                ("task-reopened",),
                {"expected_revision": None, "actor": "gui"},
            ),
            (
                "get",
                ("task-reopened",),
                {"expected_revision": 19, "actor": "gui"},
            ),
        ]

    def test_mcp_v3_durable_reload_errors_and_secret_extras_fail_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mcp = self.server.service.runtime.mcp

        def expired(*_args: Any, **_kwargs: Any) -> None:
            raise ValidationError("MCP continuation expired")

        monkeypatch.setattr(mcp, "get_continuation", expired, raising=False)
        expired_status, expired_body = self.request(
            "POST", "/api/mcp/continuations/expired-local/inspect", {}
        )

        def missing(*_args: Any, **_kwargs: Any) -> None:
            raise ValidationError("MCP continuation was not found")

        monkeypatch.setattr(mcp, "get_continuation", missing, raising=False)
        missing_status, missing_body = self.request(
            "POST", "/api/mcp/continuations/missing-local/inspect", {}
        )

        def leaking(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "kind": "input_required",
                "continuation_id": "continuation-local",
                "revision": 1,
                "respondable": True,
                "input_requests": [],
                "human_request_id": "human-local",
                "human_revision": 1,
                "human_preview_sha256": "b" * 64,
                "remote_task_id": "PRIVATE-REMOTE-TASK-ID",
            }

        monkeypatch.setattr(mcp, "get_continuation", leaking, raising=False)
        leak_status, leak_body = self.request(
            "POST", "/api/mcp/continuations/continuation-local/inspect", {}
        )

        assert expired_status == 400
        assert missing_status == 400
        assert "expired" in expired_body["error"]["message"]
        assert "not found" in missing_body["error"]["message"]
        assert leak_status == 502
        assert leak_body["error"]["code"] == "mcp_private_projection_rejected"
        assert "PRIVATE-REMOTE-TASK-ID" not in dumps(leak_body)

    def test_mcp_v3_missing_runtime_surface_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            "list_resource_templates",
            None,
            raising=False,
        )

        status, body = self.request(
            "POST", "/api/mcp/modern/resource-templates/list", {}
        )

        assert status == 501
        assert body["error"] == {
            "message": "MCP client surface is unavailable in this Runtime",
            "code": "mcp_surface_unavailable",
            "operation": "list_resource_templates",
        }

    def test_mcp_oauth_callback_and_remote_task_projection_do_not_reflect_secrets(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        callback_secret = "http://127.0.0.1/callback?code=PRIVATE-CODE&state=PRIVATE-STATE"

        def reject_callback(*_args: Any, **_kwargs: Any) -> None:
            raise ValidationError(f"invalid callback {callback_secret}")

        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            "auth_complete",
            reject_callback,
            raising=False,
        )
        callback_status, callback = self.request(
            "POST",
            "/api/mcp/auth/challenges/challenge-local/callback",
            {"callback_url": callback_secret},
        )

        def leak_oauth_token(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "profile_id": "profile-local",
                "status": "authorized",
                "token": "PRIVATE-OAUTH-TOKEN",
            }

        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            "auth_status",
            leak_oauth_token,
            raising=False,
        )
        oauth_status, oauth = self.request(
            "GET",
            "/api/mcp/auth/profile-local/status",
        )

        def leak_remote_id(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "kind": "remote_task",
                "task_ref": "local-task",
                "remote_task_id": "PRIVATE-REMOTE-ID",
                "status": "working",
            }

        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            "get_remote_task",
            leak_remote_id,
            raising=False,
        )
        task_status, task = self.request(
            "POST",
            "/api/mcp/remote-tasks/local-task/get",
            {"expected_revision": 0},
        )

        assert callback_status == 400
        assert callback["error"] == {
            "message": "MCP OAuth callback was rejected",
            "code": "mcp_oauth_callback_rejected",
        }
        assert "PRIVATE-CODE" not in dumps(callback)
        assert "PRIVATE-STATE" not in dumps(callback)
        assert oauth_status == 502
        assert oauth["error"]["code"] == "mcp_private_projection_rejected"
        assert "PRIVATE-OAUTH-TOKEN" not in dumps(oauth)
        assert task_status == 502
        assert task["error"]["code"] == "mcp_private_projection_rejected"
        assert "PRIVATE-REMOTE-ID" not in dumps(task)

    def test_mcp_oauth_profile_admin_scrubs_transient_secret_on_every_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        secret = "GUI-OAUTH-CLIENT-SECRET-SENTINEL"
        profile = _gui_mcp_oauth_profile()
        mcp = self.server.service.runtime.mcp
        calls: list[tuple[str, bytes | None, str]] = []
        cache_at_response: list[str] = []
        original_write_json = GuiRequestHandler._write_json

        def observe_cache(
            handler: GuiRequestHandler,
            value: Any,
            *,
            status: int = 200,
        ) -> None:
            cache_at_response.append(
                repr(getattr(handler, "_cached_json_body", {}))
            )
            original_write_json(handler, value, status=status)

        def add_profile(
            selected: Any,
            *,
            client_secret: bytes | None,
            actor: str,
        ) -> dict[str, Any]:
            calls.append((selected.profile_id, client_secret, actor))
            return {
                "profile_id": selected.profile_id,
                "status": "authorization_required",
                "scopes": [],
            }

        monkeypatch.setattr(GuiRequestHandler, "_write_json", observe_cache)
        monkeypatch.setattr(mcp, "add_oauth_profile", add_profile, raising=False)

        invalid_status, invalid = self.request(
            "POST",
            "/api/mcp/auth/profiles",
            {
                "profile": {**profile, "registration_mode": "dcr"},
                "client_secret": secret,
                "replace": False,
                "confirmed": True,
            },
        )
        unknown_status, unknown = self.request(
            "POST",
            "/api/mcp/auth/profiles",
            {
                "profile": profile,
                "client_secret": secret,
                "replace": False,
                "confirmed": True,
                "unexpected": True,
            },
        )
        confirmation_status, confirmation = self.request(
            "POST",
            "/api/mcp/auth/profiles",
            {
                "profile": profile,
                "client_secret": secret,
                "replace": True,
                "confirmed": False,
            },
        )

        def reject_profile(
            _selected: Any,
            *,
            client_secret: bytes | None,
            actor: str,
        ) -> None:
            assert client_secret == secret.encode("utf-8")
            assert actor == "gui"
            raise ValidationError(f"broker rejected {secret}")

        monkeypatch.setattr(
            mcp,
            "add_oauth_profile",
            reject_profile,
            raising=False,
        )
        rejected_status, rejected = self.request(
            "POST",
            "/api/mcp/auth/profiles",
            {
                "profile": profile,
                "client_secret": secret,
                "replace": False,
                "confirmed": True,
            },
        )
        monkeypatch.setattr(mcp, "add_oauth_profile", add_profile, raising=False)
        added_status, added = self.request(
            "POST",
            "/api/mcp/auth/profiles",
            {
                "profile": profile,
                "client_secret": secret,
                "replace": False,
                "confirmed": True,
            },
        )

        assert (invalid_status, unknown_status, confirmation_status) == (
            400,
            400,
            409,
        )
        assert rejected_status == 400
        assert rejected["error"] == {
            "message": "MCP OAuth profile change was rejected",
            "code": "mcp_oauth_profile_change_rejected",
        }
        assert added_status == 200
        assert added == {
            "profile_id": "profile-local",
            "status": "authorization_required",
            "scopes": [],
        }
        assert calls == [("profile-local", secret.encode("utf-8"), "gui")]
        assert invalid["error"]["code"] == "invalid_mcp_oauth_profile"
        assert unknown["error"]["code"] == "unknown_request_field"
        assert confirmation["error"]["action"] == "mcp.auth.profile.replace"

        public_state = dumps(
            {
                "responses": [invalid, unknown, confirmation, rejected, added],
                "audit": [
                    to_jsonable(item)
                    for item in self.server.service.runtime.audit.trace()
                ],
                "events": [
                    to_jsonable(item)
                    for item in self.server.service.broadcaster.replay_after(0)
                ],
                "handler_cache_at_response": cache_at_response,
            }
        )
        assert secret not in public_state
        assert cache_at_response

    def test_mcp_oauth_profile_admin_routes_use_exact_gui_host_facades(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        profile = _gui_mcp_oauth_profile()
        status = {
            "profile_id": "profile-local",
            "status": "authorization_required",
            "scopes": ["resource.read"],
        }
        calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

        def operation(name: str, result: Any):
            def selected(*args: Any, **kwargs: Any) -> Any:
                calls.append((name, args, kwargs))
                return result

            return selected

        mcp = self.server.service.runtime.mcp
        monkeypatch.setattr(
            mcp,
            "list_oauth_profiles",
            operation("list", (status,)),
            raising=False,
        )
        monkeypatch.setattr(
            mcp,
            "replace_oauth_profile",
            operation("replace", status),
            raising=False,
        )
        monkeypatch.setattr(
            mcp,
            "remove_oauth_profile",
            operation("remove", {**status, "status": "revoked"}),
            raising=False,
        )

        list_status, listed = self.request("GET", "/api/mcp/auth/profiles")
        replace_status, replaced = self.request(
            "POST",
            "/api/mcp/auth/profiles",
            {
                "profile": profile,
                "client_secret": "one-time-secret",
                "replace": True,
                "confirmed": True,
            },
        )
        remove_status, removed = self.request(
            "POST",
            "/api/mcp/auth/profiles/profile-local/remove",
            {"confirmed": True},
        )
        null_secret_status, _null_secret = self.request(
            "POST",
            "/api/mcp/auth/profiles",
            {
                "profile": profile,
                "client_secret": None,
                "replace": True,
                "confirmed": True,
            },
        )

        assert (list_status, replace_status, remove_status) == (200, 200, 200)
        assert listed == [status]
        assert replaced == status
        assert removed["status"] == "revoked"
        assert null_secret_status == 400
        assert [item[0] for item in calls] == ["list", "replace", "remove"]
        assert calls[0][1] == ()
        assert calls[0][2] == {"actor": "gui"}
        selected_profile = calls[1][1][0]
        assert selected_profile.profile_id == "profile-local"
        assert selected_profile.server_id == "modern"
        assert calls[1][2] == {
            "client_secret": b"one-time-secret",
            "actor": "gui",
        }
        assert calls[2][1] == ("profile-local",)
        assert calls[2][2] == {"actor": "gui"}

    def test_mcp_subscription_routes_never_poll_or_reconnect_implicitly(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

        def operation(name: str, result: Any):
            def selected(*args: Any, **kwargs: Any) -> Any:
                calls.append((name, args, kwargs))
                return result
            return selected

        active = {
            "subscription_id": "sub-local",
            "server_id": "modern",
            "status": "active",
            "requested_filters": ["resources/updated"],
            "acknowledged_filters": ["resources/updated"],
        }
        mcp = self.server.service.runtime.mcp
        monkeypatch.setattr(mcp, "start_subscription", operation("start", active), raising=False)
        monkeypatch.setattr(mcp, "subscription_status", operation("status", active), raising=False)
        monkeypatch.setattr(mcp, "subscription_events", operation("events", []), raising=False)

        start_status, _start = self.request(
            "POST",
            "/api/mcp/modern/subscriptions/start",
            {"filters": ["resources/updated"], "confirmed": True},
        )
        status_status, _status = self.request(
            "POST", "/api/mcp/subscriptions/sub-local/status", {}
        )
        events_status, events = self.request(
            "POST",
            "/api/mcp/subscriptions/sub-local/events",
            {"after": 0, "limit": 10},
        )

        assert (start_status, status_status, events_status) == (200, 200, 200)
        assert events == []
        assert [item[0] for item in calls] == ["start", "status", "events"]
        assert calls[0][2]["actor"] == "gui"
        assert calls[0][2]["filters"] == ("resources/updated",)

    def test_mcp_subscription_event_route_preserves_single_reader_cursor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cursor = 0

        def consume_events(
            subscription_id: str,
            *,
            after: int,
            limit: int,
            actor: str,
        ) -> tuple[McpSubscriptionEvent, ...]:
            nonlocal cursor
            assert subscription_id == "subscription-local"
            assert limit == 1
            assert actor == "gui"
            if after != cursor:
                raise ValidationError(
                    "MCP subscription event cursor is stale or has multiple readers"
                )
            if cursor != 0:
                return ()
            cursor = 1
            return (
                McpSubscriptionEvent(
                    sequence=1,
                    event_type="resourcesListChanged",
                    payload={"changed": True},
                    received_at="2026-08-11T00:00:00Z",
                ),
            )

        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            "subscription_events",
            consume_events,
            raising=False,
        )
        first_status, first = self.request(
            "POST",
            "/api/mcp/subscriptions/subscription-local/events",
            {"after": 0, "limit": 1},
        )
        stale_status, stale = self.request(
            "POST",
            "/api/mcp/subscriptions/subscription-local/events",
            {"after": 0, "limit": 1},
        )
        next_status, next_events = self.request(
            "POST",
            "/api/mcp/subscriptions/subscription-local/events",
            {"after": 1, "limit": 1},
        )

        assert first_status == 200
        assert first[0]["sequence"] == 1
        assert stale_status == 400
        assert "multiple readers" in stale["error"]["message"]
        assert (next_status, next_events) == (200, [])

    def test_mcp_register_actor_mode_requires_server_write_capability(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'mcp actor', 'auto_run': False})
        pid = spawned['pid']
        manifest = _gui_mcp_manifest('gui-actor-mcp')

        status, denied = self.request(
            'POST',
            '/api/mcp/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )

        assert status == 403
        assert 'mcp_server:gui-actor-mcp' in denied['error']['message']

        self.server.service.runtime.capability.grant(
            pid,
            'mcp_server:gui-actor-mcp',
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        spawn_status, spawn_denied = self.request(
            'POST',
            '/api/mcp/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )

        assert spawn_status == 403
        assert 'process:spawn' in spawn_denied['error']['message']

        self.server.service.runtime.capability.grant(
            pid,
            'process:spawn',
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        stdio_status, stdio_denied = self.request(
            'POST',
            '/api/mcp/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )

        assert stdio_status == 403
        assert 'mcp_stdio' in stdio_denied['error']['message']

        self.server.service.runtime.capability.grant(
            pid,
            self.server.service.runtime.mcp.stdio_resource_for_argv(MCP_TEST_STDIO_COMMAND, ['-m', 'demo_mcp']),
            [CapabilityRight.EXECUTE],
            issued_by='test',
        )
        register_status, registered = self.request(
            'POST',
            '/api/mcp/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )
        tools_status, tools = self.request('GET', '/api/mcp/gui-actor-mcp/tools')

        assert register_status == 200
        assert tools_status == 200
        assert registered['server_id'] == 'gui-actor-mcp'
        assert tools['tools'][0]['tool_id'] == 'echo'
        assert tools['tools'][0]['resource'] == 'mcp:gui-actor-mcp:echo'

    def test_mcp_discover_is_an_unconfirmed_external_read_projects_connection_and_publishes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed: list[dict[str, object]] = []
        discovery = McpDiscoveryResult(
            server_id='gui-modern-mcp',
            connection=McpConnectionInfo(
                protocol_mode=McpProtocolMode.AUTO,
                protocol_era=McpProtocolEra.MODERN,
                protocol_revision='2026-07-28',
                sessionless=True,
                fallback_used=False,
                server_name='gui-fixture',
                server_version='2.0.0',
                capabilities=('tools',),
                unsupported_capabilities=('resources',),
            ),
            request_bytes=43,
            response_bytes=101,
            duration_s=0.02,
        )
        expected = {
            'server_id': 'gui-modern-mcp',
            'connection': {
                'protocol_mode': 'auto',
                'protocol_era': 'modern',
                'protocol_revision': '2026-07-28',
                'sessionless': True,
                'fallback_used': False,
                'server_name': 'gui-fixture',
                'server_version': '2.0.0',
                'capabilities': ['tools'],
                'unsupported_capabilities': ['resources'],
            },
            'request_bytes': 43,
            'response_bytes': 101,
            'duration_s': 0.02,
            'receipts': [],
        }

        def record_discover(
            server_id: str,
            *,
            actor: str,
            require_capability: bool,
        ) -> McpDiscoveryResult:
            observed.append(
                {
                    'server_id': server_id,
                    'actor': actor,
                    'require_capability': require_capability,
                }
            )
            return discovery

        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            'discover',
            record_discover,
            raising=False,
        )
        cursor = self.server.service.broadcaster.replay_after(0)[-1].seq

        host_status, host_result = self.request(
            'POST',
            '/api/mcp/gui-modern-mcp/discover',
            {},
        )
        actor_status, actor_result = self.request(
            'POST',
            '/api/mcp/gui-modern-mcp/discover',
            {'actor': 'pid-modern-reader'},
        )
        published_reasons = [
            event.data['reason']
            for event in self.server.service.broadcaster.replay_after(cursor)
            if event.event == 'snapshot'
        ]

        assert host_status == 200
        assert actor_status == 200
        assert host_result == expected
        assert actor_result['connection']['protocol_revision'] == '2026-07-28'
        assert observed == [
            {
                'server_id': 'gui-modern-mcp',
                'actor': 'gui',
                'require_capability': False,
            },
            {
                'server_id': 'gui-modern-mcp',
                'actor': 'pid-modern-reader',
                'require_capability': True,
            },
        ]
        assert published_reasons == ['mcp.discover', 'mcp.discover']

    def test_mcp_discover_does_not_publish_failed_operation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_discover(*_args: object, **_kwargs: object) -> McpDiscoveryResult:
            raise ValidationError('injected MCP discovery failure')

        monkeypatch.setattr(
            self.server.service.runtime.mcp,
            'discover',
            fail_discover,
            raising=False,
        )
        cursor = self.server.service.broadcaster.replay_after(0)[-1].seq

        status, body = self.request(
            'POST',
            '/api/mcp/gui-modern-mcp/discover',
            {},
        )
        published_reasons = [
            event.data['reason']
            for event in self.server.service.broadcaster.replay_after(cursor)
            if event.event == 'snapshot'
        ]

        assert status == 400
        assert body['error']['message'] == 'injected MCP discovery failure'
        assert published_reasons == []

    def test_mcp_call_preserves_invalid_arguments_for_primitive_validation(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'mcp invalid args', 'auto_run': False})
        pid = spawned['pid']
        manifest = _gui_mcp_manifest('gui-invalid-args-mcp')
        self.server.service.runtime.mcp.register_server_from_yaml_text(
            manifest,
            actor='test',
            require_capability=False,
        )
        self.server.service.runtime.capability.grant(
            pid,
            'mcp:gui-invalid-args-mcp:echo',
            [CapabilityRight.READ],
            issued_by='test',
        )

        status, body = self.request(
            'POST',
            '/api/mcp/gui-invalid-args-mcp/call',
            {'pid': pid, 'tool_id': 'echo', 'arguments': [], 'confirmed': True},
        )

        assert status == 400
        assert body['error']['message'] == 'MCP tool arguments must be a strict JSON object'

    def test_mcp_provider_exception_secret_is_absent_from_gui_response(self) -> None:
        secret = "GUI_MCP_HOST_EXCEPTION_SECRET_SENTINEL"

        class FailingProvider:
            def list_tools(self, server: Any, **_kwargs: Any) -> McpToolListResult:
                return McpToolListResult(
                    server_id=server.server_id,
                    tools=[
                        McpProviderTool(
                            name="demo.echo",
                            description="Echo",
                            input_schema={},
                        )
                    ],
                    # The provider contract reports at least the canonical
                    # serialized tool-list size. This security fixture is not
                    # testing byte accounting, so use the manifest ceiling.
                    response_bytes=server.max_response_bytes,
                    duration_s=0.01,
                )

            def call_tool(
                self,
                _server: Any,
                _tool: Any,
                _arguments: dict[str, Any],
                **_kwargs: Any,
            ) -> Any:
                raise RuntimeError(secret)

            def classify_external_effect(
                self,
                _operation: str,
                _context: dict[str, Any],
                _result: Any,
            ) -> ExternalEffectClassification:
                return ExternalEffectClassification(
                    rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                    rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                    state_mutation=False,
                    information_flow=True,
                )

        runtime = self.server.service.runtime
        runtime.mcp.provider = FailingProvider()
        _status, spawned = self.request(
            'POST',
            '/api/processes',
            {'goal': 'MCP GUI provider failure', 'auto_run': False},
        )
        pid = spawned['pid']
        runtime.mcp.register_server_from_yaml_text(
            _gui_mcp_manifest('gui-provider-failure'),
            actor='test',
            require_capability=False,
        )
        runtime.capability.grant(
            pid,
            'mcp:gui-provider-failure:echo',
            [CapabilityRight.READ],
            issued_by='test',
        )
        runtime.capability.grant(
            pid,
            'process:spawn',
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        runtime.capability.grant(
            pid,
            runtime.mcp.stdio_resource_for_argv(MCP_TEST_STDIO_COMMAND, ['-m', 'demo_mcp']),
            [CapabilityRight.EXECUTE],
            issued_by='test',
        )

        status, body = self.request(
            'POST',
            '/api/mcp/gui-provider-failure/call',
            {
                'pid': pid,
                'tool_id': 'echo',
                'arguments': {'text': 'hello'},
                'confirmed': True,
            },
        )

        assert status == 200
        assert body['ok'] is False
        assert set(body['error']) == {
            'code',
            'error_type',
            'correlation_id',
            'retryable',
            'automatic_retry_disabled',
        }
        assert body['error']['retryable'] is False
        assert body['error']['automatic_retry_disabled'] is True
        assert secret not in json.dumps(body, sort_keys=True)

    def test_mcp_v3_tool_exception_redacts_active_header_secret_from_gui_and_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agent_libos.mcp import McpServerManifestV3
        from agent_libos.models import (
            McpHeaderSpec,
            McpHttpTransportSpec,
            McpToolSpec,
            ResourceBudget,
        )

        secret = "GUI_MODERN_MCP_HEADER_SECRET_SENTINEL"
        env_name = "AGENT_LIBOS_MCP_GUI_MODERN_SECRET"
        monkeypatch.setenv(env_name, secret)

        class FailingModernToolProvider:
            mcp_manifest_schema_version = 3
            mcp_protocol_revision = "2026-07-28"

            async def call_tool(
                self,
                _manifest: Any,
                _tool_id: str,
                _arguments: dict[str, Any],
                *,
                deadline: float,
                sensitive_values: tuple[str, ...] = (),
            ) -> Any:
                assert deadline > 0
                assert secret in sensitive_values
                raise ValidationError(
                    f"modern Tool Provider reflected {sensitive_values[0]}"
                )

        runtime = self.server.service.runtime
        server_id = "gui-modern-provider-failure"
        runtime.mcp.register_server(
            McpServerManifestV3(
                schema_version=3,
                server_id=server_id,
                transport="streamable_http",
                http=McpHttpTransportSpec(
                    url="http://127.0.0.1:8765/mcp",
                    headers={
                        "Authorization": McpHeaderSpec(
                            env=env_name,
                            prefix="Bearer ",
                        )
                    },
                ),
                timeout_s=2.0,
                max_request_bytes=16_384,
                max_response_bytes=16_384,
                protocol_mode=McpProtocolMode.REVISION_2026_07_28,
                tools=(
                    McpToolSpec(
                        tool_id="echo",
                        mcp_name="provider.echo",
                        right="read",
                        rollback_class="no_rollback_required",
                        rollback_status="not_required",
                        state_mutation=False,
                        information_flow=True,
                        input_schema={
                            "type": "object",
                            "additionalProperties": False,
                        },
                    ),
                ),
            ),
            actor="runtime",
            require_capability=False,
        )
        runtime.mcp._modern_tool_provider = FailingModernToolProvider()  # noqa: SLF001
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject reflected modern MCP credentials in GUI",
            resource_budget=ResourceBudget(max_mcp_bytes=256_000),
        )
        runtime.capability.grant(
            pid,
            f"mcp:{server_id}:echo",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.capability.grant(
            pid,
            f"mcp_server:{server_id}",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )

        status, body = self.request(
            "POST",
            f"/api/mcp/{server_id}/call",
            {
                "pid": pid,
                "tool_id": "echo",
                "arguments": {},
                "confirmed": True,
            },
        )

        assert status == 400
        public = json.dumps(body, sort_keys=True)
        assert "MCP provider operation failed" in public
        assert secret not in public
        evidence = dumps(
            {
                "audit": [to_jsonable(row) for row in runtime.audit.trace()],
                "effects": [
                    to_jsonable(row)
                    for row in runtime.store.list_external_effects(pid=pid)
                ],
            }
        )
        assert secret not in evidence

    def test_skill_register_without_actor_rejects_host_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'gui-host-path-skill', allowed_tools=['echo'])

            status, denied = self.request(
                'POST',
                '/api/skills/register',
                {'path': str(skill_dir), 'confirmed': True},
            )

            assert status == 400
            assert 'requires an actor' in denied['error']['message']

    def test_skill_activate_requires_and_forwards_discovered_package_hash(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = self.server.service.runtime
        package_sha256 = "b" * 64
        calls: list[dict[str, Any]] = []

        def activate_skill(
            pid: str,
            skill_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            calls.append({"pid": pid, "skill_id": skill_id, **kwargs})
            return {"pid": pid, "skill_id": skill_id}

        monkeypatch.setattr(runtime.skills, "activate_skill", activate_skill)

        status, missing = self.request(
            "POST",
            "/api/skills/gui-cas-skill/activate",
            {"pid": "pid_1", "confirmed": True},
        )
        assert status == 400
        assert "expected_package_sha256" in missing["error"]["message"]

        status, invalid = self.request(
            "POST",
            "/api/skills/gui-cas-skill/activate",
            {
                "pid": "pid_1",
                "expected_package_sha256": "B" * 64,
                "confirmed": True,
            },
        )
        assert status == 400
        assert "lowercase SHA-256" in invalid["error"]["message"]

        status, confirmation = self.request(
            "POST",
            "/api/skills/gui-cas-skill/activate",
            {
                "pid": "pid_1",
                "expected_package_sha256": package_sha256,
            },
        )
        assert status == 409
        assert confirmation["error"]["preview"]["expected_package_sha256"] == package_sha256

        status, activated = self.request(
            "POST",
            "/api/skills/gui-cas-skill/activate",
            {
                "pid": "pid_1",
                "actor": "pid_1",
                "expected_package_sha256": package_sha256,
                "confirmed": True,
            },
        )
        assert status == 200
        assert activated == {"pid": "pid_1", "skill_id": "gui-cas-skill"}
        assert calls == [
            {
                "pid": "pid_1",
                "skill_id": "gui-cas-skill",
                "actor": "pid_1",
                "require_capability": True,
                "expected_package_sha256": package_sha256,
            }
        ]

    def test_skill_register_actor_mode_requires_skill_write_capability(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir).resolve()
            skill_dir = write_skill_package(root, 'gui-actor-skill', allowed_tools=['echo'])
            relative_skill = skill_dir.relative_to(Path.cwd().resolve()).as_posix()
            skill_md = f'{relative_skill}/SKILL.md'
            _status, spawned = self.request('POST', '/api/processes', {'goal': 'skill actor', 'auto_run': False})
            pid = spawned['pid']

            status, denied = self.request(
                'POST',
                '/api/skills/register',
                {'path': relative_skill, 'actor': pid, 'confirmed': True},
            )

            assert status == 403
            assert 'filesystem:workspace' in denied['error']['message']

            self.server.service.runtime.filesystem.grant_path(
                pid,
                skill_md,
                [CapabilityRight.READ],
                issued_by='test',
            )
            status, denied = self.request(
                'POST',
                '/api/skills/register',
                {'path': relative_skill, 'actor': pid, 'confirmed': True},
            )

            assert status == 409
            assert denied['error']['type'] == 'HumanApprovalRequired'
            assert denied['error']['request_id']

            self.server.service.runtime.capability.grant(
                pid,
                'skill:gui-actor-skill',
                [CapabilityRight.WRITE],
                issued_by='test',
            )
            status, registered = self.request(
                'POST',
                '/api/skills/register',
                {'path': relative_skill, 'actor': pid, 'confirmed': True},
            )

            assert status == 200
            assert registered['skill_id'] == 'gui-actor-skill'

    def test_human_request_respond_rejects_non_pending_request(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='gui human conflict',
            authority_manifest={
                'authorized_capabilities': [
                    {'resource': DEFAULT_CONFIG.runtime.default_human_resource, 'rights': ['write']},
                ],
            },
        )
        request_id = runtime.human.ask(pid, 'Approve once?', blocking=True)
        status, approved = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'yes', 'auto_run': False},
        )
        status_again, conflict = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'again', 'auto_run': False},
        )

        assert status == 200
        assert approved['request']['status'] == 'approved'
        assert status_again == 409
        assert 'not pending' in conflict['error']['message']

    def test_permission_response_requires_explicit_valid_policy(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='typed gui permission')
        resource = runtime.filesystem.resource_for('agent_outputs/typed-gui.txt')
        request_id = runtime.human.query_authority_request(
            pid=pid,
            human=DEFAULT_CONFIG.runtime.default_human,
            request={
                'type': 'permission_request',
                'question': 'Allow write?',
                'requested_permission': {
                    'subject': pid,
                    'resource': resource,
                    'rights': ['write'],
                    'constraints': {},
                },
            },
            blocking=True,
            authority_origin='permission_policy',
        )
        presented = self.server.service.human_request_view(
            runtime.human.get(request_id)
        )
        assert presented['request_id'] == request_id
        presentation_effect = next(
            effect
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider_metadata.get('context', {}).get('request_id')
            == request_id
            and effect.provider_metadata.get('context', {}).get('purpose')
            == 'gui_presentation'
        )
        assert (
            presentation_effect.provider_metadata['context']['request_kind']
            == 'approval'
        )

        missing_status, missing = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'auto_run': False},
        )
        invalid_status, invalid = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'decision': {'policy': 'sometimes'}, 'auto_run': False},
        )

        assert missing_status == 400
        assert 'policy' in missing['error']['message']
        assert invalid_status == 400
        assert 'policy' in invalid['error']['message']
        assert runtime.human.get(request_id).status.value == 'pending'

        approved_status, approved = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {
                'approved': True,
                'decision': {'policy': CapabilityManager.ASK_EACH_TIME},
                'auto_run': False,
            },
        )
        assert approved_status == 200
        assert approved['request']['decision']['policy'] == CapabilityManager.ASK_EACH_TIME
        assert runtime.capability.permission_policy(pid, resource, CapabilityRight.WRITE) == CapabilityManager.ASK_EACH_TIME

    def test_question_response_requires_string_answer_before_commit(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='typed gui question')
        request_id = runtime.human.query(
            pid=pid,
            human=DEFAULT_CONFIG.runtime.default_human,
            request={'type': 'question', 'question': 'Which region?'},
            blocking=True,
        )

        missing_status, missing = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'auto_run': False},
        )
        wrong_status, wrong = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 42, 'auto_run': False},
        )
        empty_status, empty = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': '   ', 'auto_run': False},
        )

        assert missing_status == 400
        assert 'answer' in missing['error']['message']
        assert wrong_status == 400
        assert 'answer' in wrong['error']['message']
        assert empty_status == 400
        assert 'answer' in empty['error']['message']
        assert runtime.human.get(request_id).status.value == 'pending'

        accepted_status, accepted = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'eu-west', 'auto_run': False},
        )
        assert accepted_status == 200
        assert accepted['request']['decision']['answer'] == 'eu-west'

    def test_human_request_delta_is_emitted_for_each_changed_version(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='gui human delta',
            authority_manifest={
                'authorized_capabilities': [
                    {'resource': DEFAULT_CONFIG.runtime.default_human_resource, 'rights': ['write']},
                ],
            },
        )
        cursor = self.server.service.broadcaster.replay_after(0)[-1].seq
        request_id = runtime.human.ask(pid, 'Emit both versions?', blocking=True)

        self.server.service.publish_runtime_changes('human.pending')
        pending_events = self.server.service.broadcaster.replay_after(cursor)
        pending_updates = [
            event
            for event in pending_events
            if event.event == 'human_request.updated' and event.data['request_id'] == request_id
        ]
        assert len(pending_updates) == 1
        assert pending_updates[0].data['status'] == 'pending'
        cursor = pending_events[-1].seq

        runtime.human.approve(request_id, {'approved': True, 'answer': 'yes', 'source': 'test'})
        self.server.service.publish_runtime_changes('human.approved')
        approved_events = self.server.service.broadcaster.replay_after(cursor)
        approved_updates = [
            event
            for event in approved_events
            if event.event == 'human_request.updated' and event.data['request_id'] == request_id
        ]
        assert len(approved_updates) == 1
        assert approved_updates[0].data['status'] == 'approved'
        cursor = approved_events[-1].seq

        self.server.service.publish_runtime_changes('human.unchanged')
        unchanged = self.server.service.broadcaster.replay_after(cursor)
        assert not any(
            event.event == 'human_request.updated' and event.data['request_id'] == request_id
            for event in unchanged
        )

    def test_permission_response_without_approved_uses_explicit_deny_policy(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='gui human default reject')
        request_id = runtime.human.query_authority_request(
            pid=pid,
            human=DEFAULT_CONFIG.runtime.default_human,
            request={
                'type': 'permission_request',
                'question': 'Allow object read?',
                'requested_permission': {
                    'subject': pid,
                    'resource': 'object:gui-default-reject',
                    'rights': ['read'],
                },
            },
            blocking=True,
            authority_origin='permission_policy',
        )

        status, rejected = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {
                'decision': {'policy': CapabilityManager.ALWAYS_DENY},
                'auto_run': False,
            },
        )

        assert status == 200
        assert rejected['request']['status'] == 'rejected'
        assert (
            runtime.capability.permission_policy(pid, 'object:gui-default-reject', CapabilityRight.READ)
            == 'always_deny'
        )

    def test_invalid_max_quanta_is_rejected(self) -> None:
        before_count = len(self.server.service.runtime.process.list())
        status, body = self.request('POST', '/api/processes', {'goal': 'goal', 'max_quanta': 1.5})
        assert status == 400
        assert 'max_quanta' in body['error']['message']
        assert len(self.server.service.runtime.process.list()) == before_count

        status, body = self.request('POST', '/api/processes', {'goal': 'goal', 'max_quanta': 0})
        assert status == 400
        assert 'max_quanta' in body['error']['message']
        assert len(self.server.service.runtime.process.list()) == before_count

    def test_process_resume_validates_body_before_mutating_process(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='resume validation')
        runtime.process.pause(pid, 'hold for invalid resume body')
        assert runtime.process.get(pid).status == ProcessStatus.PAUSED

        status, body = self.request_json_text('POST', f'/api/processes/{pid}/resume', '[]')

        assert status == 400
        assert 'JSON object' in body['error']['message']
        assert runtime.process.get(pid).status == ProcessStatus.PAUSED

    @pytest.mark.parametrize(
        'raw',
        [
            b'{"goal":"\xff"}',
            ('{"goal":' + ('[' * 257) + '0' + (']' * 257) + '}').encode('utf-8'),
            b'{"goal":"constant", "max_quanta":NaN}',
            ('{"goal":' + ('[' * 2_000) + '0' + (']' * 2_000) + '}').encode('utf-8'),
        ],
        ids=['invalid-unicode', 'excessive-depth', 'invalid-constant', 'parser-recursion'],
    )
    def test_json_parser_failures_are_stable_bad_requests(self, raw: bytes) -> None:
        status, body = self.request_json_bytes('POST', '/api/processes', raw)

        assert status == 400
        assert body['ok'] is False
        assert 'invalid JSON body' in body['error']['message']

    def test_request_body_size_is_bounded(self) -> None:
        self.server.service.runtime.config = replace(
            self.server.service.runtime.config,
            gui=replace(self.server.service.runtime.config.gui, request_body_max_bytes=1024),
        )
        status, body = self.request('POST', '/api/processes', {'goal': 'x' * 1100})
        assert status == 413
        assert 'exceeds' in body['error']['message']

    def test_shutdown_endpoint_stops_http_server(self) -> None:
        try:
            status, body = self.request('POST', '/api/shutdown', {})
            assert status == 200
            assert body['status'] == 'stopped'
        except ConnectionResetError:
            pass
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        self.server.service.shutdown()

    def test_shutdown_endpoint_reports_incomplete_teardown_and_remains_retryable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_shutdown = self.server.service.shutdown
        monkeypatch.setattr(self.server.service, 'shutdown', lambda timeout_s=None: False)

        status, body = self.request('POST', '/api/shutdown', {})

        assert status == 503
        assert body['ok'] is False
        assert body['error']['retryable'] is True
        assert self.thread.is_alive()

        monkeypatch.setattr(self.server.service, 'shutdown', original_shutdown)
        status, body = self.request('POST', '/api/shutdown', {})
        assert status == 200
        assert body == {'ok': True, 'status': 'stopped'}
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()

    def test_serve_teardown_retries_and_fails_visibly_if_runtime_never_closes(self) -> None:
        class FakeService:
            def __init__(self, results: list[bool | Exception]):
                self.results = iter(results)
                self.calls = 0

            def shutdown(self) -> bool:
                self.calls += 1
                result = next(self.results)
                if isinstance(result, Exception):
                    raise result
                return result

        retrying = FakeService([False, True])
        _shutdown_gui_service_before_exit(retrying)
        assert retrying.calls == 2

        exception_retry = FakeService([RuntimeError('first teardown attempt failed'), True])
        _shutdown_gui_service_before_exit(exception_retry)
        assert exception_retry.calls == 2

        incomplete = FakeService([False, False])
        with pytest.raises(RuntimeError, match='teardown remained incomplete'):
            _shutdown_gui_service_before_exit(incomplete)
        assert incomplete.calls == 2


def _request_to_server(
    server: Any,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    token: str,
) -> tuple[int, Any]:
    host, port = server.server_address
    conn = http.client.HTTPConnection(
        host,
        port,
        timeout=_GUI_TEST_HTTP_TIMEOUT_S,
    )
    headers = {'Authorization': f'Bearer {token}'}
    payload = None
    if body is not None:
        payload = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    decoded = json.loads(data.decode('utf-8')) if data else None
    return response.status, decoded


def _gui_image_package_files() -> dict[str, str]:
    return {
        "IMAGE.yaml": """
image_id: gui-package-agent:v0
name: gui-package-agent
version: v0
prompt: prompt.md
default_tools:
  - human_output
""".lstrip(),
        "prompt.md": "Registered from GUI package files.\n",
    }


def _gui_jsonrpc_manifest(endpoint_id: str) -> str:
    return f"""
schema_version: 1
endpoint_id: {endpoint_id}
url: https://api.example.test/jsonrpc
methods:
  - method_id: echo
    rpc_method: echo
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
""".lstrip()


def _gui_mcp_manifest(server_id: str) -> str:
    return f"""
schema_version: 1
server_id: {server_id}
transport: stdio
stdio:
  command: {MCP_TEST_STDIO_COMMAND_YAML}
  args: ["-m", "demo_mcp"]
tools:
  - tool_id: echo
    mcp_name: demo.echo
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
timeout_s: 5
max_request_bytes: 65536
max_response_bytes: 1048576
""".lstrip()


def _gui_mcp_oauth_profile() -> dict[str, Any]:
    return {
        "profile_id": "profile-local",
        "server_id": "modern",
        "resource_uri": "https://resource.example/mcp",
        "expected_issuer": "https://issuer.example",
        "redirect_uri": "http://127.0.0.1/callback",
        "client_id": "gui-client",
        "registration_mode": "preregistered",
        "token_endpoint_auth_method": "client_secret_basic",
        "allowed_scopes": ["resource.read"],
        "default_scopes": ["resource.read"],
        "allowed_endpoint_origins": ["https://issuer.example"],
        "allow_loopback_http": True,
        "protocol_revision": "2026-07-28",
        "transport": "streamable_http",
    }
