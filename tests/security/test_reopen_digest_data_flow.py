from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_libos import Runtime
from agent_libos.llm.reopen_digest import REOPEN_DIGEST_HEADING
from agent_libos.models import (
    CapabilityRight, DataFlowContext, DataLabels, DataSink, EventType, ObjectMetadata, ObjectType, SinkTrustRule,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.substrate import LocalResourceProviderSubstrate
from tests.support.fakes import RecordingActionClient


def _allow_secret_sink(runtime: Runtime, pattern: str, identity: str | None = None) -> None:
    runtime.data_flow.register_sink_trust(
        SinkTrustRule(pattern=pattern, trust_level="trusted", max_sensitivity="secret", identity_sha256=identity),
        actor="test.host", require_capability=False,
    )


@pytest.mark.parametrize("operation", [
    "read_text", "read_directory", "write_text", "write_directory", "read_then_delete",
    "write_text_source", "write_directory_source",
])
@pytest.mark.parametrize("trusted_llm", [False, True], ids=["denied", "trusted"])
def test_reopen_digest_enforces_historical_metadata_labels(
    tmp_path: Path, operation: str, trusted_llm: bool,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    path = "secret_customer_plan"
    is_directory = "directory" in operation
    explicit_source = operation.endswith("_source")
    if is_directory and not explicit_source:
        (root / path).mkdir()
        (root / path / "child").write_text("private body", encoding="utf-8")
    elif not explicit_source:
        (root / path).write_text("private body", encoding="utf-8")
    database = tmp_path / "runtime.sqlite"
    runtime = Runtime.open(database, substrate=LocalResourceProviderSubstrate(root))
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="continue ordinary work")
        runtime.activate_skill(pid, "agent-libos-runtime-session")
        runtime.activate_skill(pid, "agent-libos-workspace-navigation")
        runtime.capability.grant(pid, "clock:now", [CapabilityRight.READ], issued_by="test")
        runtime.filesystem.grant_directory(
            pid, ".", [CapabilityRight.READ, CapabilityRight.WRITE, CapabilityRight.DELETE], issued_by="test",
        )
        runtime.llm.client = RecordingActionClient([{"action": "get_current_time", "timezone": "UTC"}])
        assert runtime.run_process_once(pid)["ok"]
        # For listings only the child is classified; the directory binding
        # itself is normal. Writes cover both existing classified destinations
        # with normal content and new destinations with an explicit source.
        source_oids = []
        if explicit_source:
            source = runtime.memory.create_object(
                pid, ObjectType.EVIDENCE, {"path": path}, metadata=ObjectMetadata(sensitivity="secret"),
            )
            source_oids.append(source.oid)
        else:
            bound = path + "/child" if operation == "read_directory" else path
            runtime.data_flow.bind_written_file(
                pid=pid, normalized_path=bound, content=b"private body",
                context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
            )
        _allow_secret_sink(runtime, "filesystem:workspace:*")
        with runtime.data_flow.activate(DataFlowContext()):
            if operation.startswith("read"):
                result = runtime.llm.dispatch(pid, {
                    "action": "read_directory" if is_directory else "read_text_file", "path": path,
                })
                assert result["ok"], result
                if operation == "read_then_delete":
                    runtime.filesystem.delete_file(pid, path)
                    assert runtime.data_flow.file_context(path).labels.sensitivity.value == "normal"
            elif is_directory:
                runtime.filesystem.write_directory(pid, path, source_oids=source_oids)
            else:
                runtime.filesystem.write_text(pid, path, "normal replacement", source_oids=source_oids)
        events = [event for event in runtime.events.list()
                  if event.source == pid and event.payload.get("path") == path
                  and event.type in {EventType.EXTERNAL_READ, EventType.EXTERNAL_WRITE}]
        assert events[-1].payload["data_labels"]["sensitivity"] == "secret"
        # Previously consumed events are outside the next recent-event window.
        process = runtime.process.get(pid)
        runtime.store.update_process(replace(process, event_cursor=runtime.events.list(target=pid)[-1].event_id))
    finally:
        runtime.close()
    reopened = Runtime.open(database, substrate=LocalResourceProviderSubstrate(root))
    try:
        client = RecordingActionClient([{"action": "get_current_time", "timezone": "UTC"}])
        reopened.llm.client = client
        if trusted_llm:
            _allow_secret_sink(reopened, "llm:default", reopened.llms.profile_identity_sha256("default"))
        outcome = reopened.run_process_once(pid)
        decisions = [decision for decision in reopened.store.list_data_flow_decisions(pid=pid)
                     if str(decision.sink) == "llm:default" and decision.labels.sensitivity.value == "secret"]
        assert decisions
        if trusted_llm:
            assert outcome["ok"], outcome
            assert path in client.user_prompts[0]
            assert REOPEN_DIGEST_HEADING in client.user_prompts[0]
            assert "private body" not in client.user_prompts[0]
            assert decisions[-1].outcome.value == "allow"
        else:
            assert not outcome["ok"]
            assert not client.user_prompts
            assert decisions[-1].outcome.value == "deny"
            assert any(record.action == "data_flow.egress" and record.decision.get("outcome") == "deny"
                       for record in reopened.audit.trace(actor=pid))
            assert any(event.type == EventType.DATA_FLOW_DECISION and event.payload.get("outcome") == "deny"
                       for event in reopened.events.list(target="data_flow_sink:llm:default"))
    finally:
        reopened.close()


@pytest.mark.parametrize("legacy_payload", [
    {"adapter": "filesystem", "path": "SECRET_DELETED_PATH"},
    {"adapter": "shell", "argv": ["echo", "SECRET_ARGUMENT"], "returncode": 0},
    {"adapter": "git", "operation": "SECRET_OPERATION"},
    {"skill_id": "SECRET_UNLOADED_SKILL"},
])
def test_digest_omits_legacy_metadata_without_label_provenance(legacy_payload: dict) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="legacy history")
        if "skill_id" in legacy_payload:
            event_type = EventType.SKILL_LOADED
        else:
            event_type = EventType.EXTERNAL_READ if legacy_payload["adapter"] == "git" else EventType.EXTERNAL_WRITE
        runtime.events.emit(event_type, source=pid, payload=legacy_payload)
        runtime.events.emit(EventType.RUNTIME_SHUTDOWN, source="runtime", payload={})
        context = SimpleNamespace(object_manifest=[{"disposition": "omitted", "reason": "missing"}])
        digest, flow = runtime.llm._reopen_activity_digest(pid, context, DataFlowContext())
        assert digest is None
        assert flow == DataFlowContext()
    finally:
        runtime.close()


def test_digest_rejects_incomplete_trusted_labels() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="validate history labels")
        runtime.events.emit(EventType.EXTERNAL_READ, source=pid, payload={
            "adapter": "filesystem", "path": "private", "data_labels": {"sensitivity": "normal"},
        })
        runtime.events.emit(EventType.RUNTIME_SHUTDOWN, source="runtime", payload={})
        context = SimpleNamespace(object_manifest=[{"disposition": "omitted", "reason": "missing"}])
        with pytest.raises(ValidationError, match="malformed trusted data_labels"):
            runtime.llm._reopen_activity_digest(pid, context, DataFlowContext())
    finally:
        runtime.close()


@pytest.mark.parametrize("payload", [
    {"adapter": "llm"},
    {"adapter": "jsonrpc"},
    {"adapter": "shell", "error_type": "SubprocessTimeoutExpired"},
    {"adapter": "filesystem", "operation": "failed"},
])
@pytest.mark.parametrize("visible_activity", [False, True])
def test_digest_ignores_labels_of_events_without_rendered_activity(
    payload: dict, visible_activity: bool,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="ordinary work")
        if visible_activity:
            runtime.events.emit(EventType.EXTERNAL_READ, source=pid, payload={
                "adapter": "filesystem", "path": "ordinary.txt", "data_labels": DataLabels().to_dict(),
            })
        runtime.events.emit(EventType.EXTERNAL_WRITE, source=pid, payload={
            **payload, "data_labels": DataLabels(sensitivity="secret").to_dict(),
        })
        runtime.events.emit(EventType.RUNTIME_SHUTDOWN, source="runtime", payload={})
        context = SimpleNamespace(object_manifest=[{"disposition": "omitted", "reason": "missing"}])
        digest, flow = runtime.llm._reopen_activity_digest(pid, context, DataFlowContext())
        assert flow.labels.sensitivity.value == "normal"
        if visible_activity:
            assert digest is not None and "ordinary.txt" in digest
        else:
            assert digest is None
    finally:
        runtime.close()


def test_directory_state_event_keeps_labels_when_binding_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    directory = "private_directory"
    (root / directory).mkdir()
    runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(root))
    try:
        reader = runtime.process.spawn(goal="inspect working directory")
        deleter = runtime.process.spawn(goal="remove directory")
        for pid in (reader, deleter):
            runtime.filesystem.grant_directory(
                pid, directory, [CapabilityRight.READ, CapabilityRight.DELETE], issued_by="test",
            )
        runtime.data_flow.bind_written_file(
            pid=reader, normalized_path=directory, content=b"directory",
            context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
        )
        _allow_secret_sink(runtime, "filesystem:workspace:*")
        provider = runtime.filesystem.provider
        original_state = provider.state

        def state_then_delete(path):
            state = original_state(path)
            monkeypatch.setattr(provider, "state", original_state)
            runtime.filesystem.delete_directory(deleter, directory)
            return state

        # Re-entrant provider mutation exercises the captured label snapshot,
        # even when normal cross-thread writes are serialized by the path lock.
        monkeypatch.setattr(provider, "state", state_then_delete)
        with runtime.data_flow.activate(DataFlowContext()):
            assert runtime.filesystem.validate_directory(reader, directory) == directory
        assert runtime.data_flow.file_context(directory).labels.sensitivity.value == "normal"
        runtime.events.emit(EventType.RUNTIME_SHUTDOWN, source="runtime", payload={})
        context = SimpleNamespace(object_manifest=[{"disposition": "omitted", "reason": "missing"}])
        digest, flow = runtime.llm._reopen_activity_digest(reader, context, DataFlowContext())
        assert digest is not None and directory in digest
        assert flow.labels.sensitivity.value == "secret"
        with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
            runtime.data_flow.authorize_egress(
                pid=reader, sink=DataSink("llm:default"), context=flow,
                payload=digest, operation="test.reopen_digest",
            )
        assert any(record.action == "data_flow.egress" and record.decision.get("outcome") == "deny"
                   for record in runtime.audit.trace(actor=reader))
        assert any(event.type == EventType.DATA_FLOW_DECISION and event.payload.get("outcome") == "deny"
                   for event in runtime.events.list(target="data_flow_sink:llm:default"))
    finally:
        runtime.close()


def test_shell_event_retains_explicit_source_labels_after_reopen(tmp_path: Path) -> None:
    from tests.security.test_shell_primitive import RecordingShellSubstrate, ResolvingShellProvider

    root = tmp_path / "workspace"
    root.mkdir()
    database = tmp_path / "shell.sqlite"
    provider = ResolvingShellProvider()
    runtime = Runtime.open(database, substrate=RecordingShellSubstrate(str(root), provider))
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="continue ordinary work")
        runtime.activate_skill(pid, "agent-libos-runtime-session")
        runtime.capability.grant(pid, "clock:now", [CapabilityRight.READ], issued_by="test")
        runtime.llm.client = RecordingActionClient([{"action": "get_current_time", "timezone": "UTC"}])
        assert runtime.run_process_once(pid)["ok"]
        runtime.shell.grant_policy(pid, runtime.config.shell.always_allow_level, issued_by="test")
        _allow_secret_sink(runtime, "shell:*", runtime.shell.executable_data_sink("shell", "echo", cwd=".").identity_sha256)
        source = runtime.memory.create_object(
            pid, ObjectType.EVIDENCE, {"value": "SECRET_ARGUMENT"}, metadata=ObjectMetadata(sensitivity="secret"),
        )
        with runtime.data_flow.activate(DataFlowContext()):
            runtime.shell.run(pid, ["echo", "SECRET_ARGUMENT"], source_oids=[source.oid])
        event = next(event for event in reversed(runtime.events.list())
                     if event.source == pid and event.payload.get("adapter") == "shell")
        assert event.payload["data_labels"]["sensitivity"] == "secret"
        process = runtime.process.get(pid)
        runtime.store.update_process(replace(process, event_cursor=runtime.events.list(target=pid)[-1].event_id))
    finally:
        runtime.close()
    reopened = Runtime.open(database, substrate=RecordingShellSubstrate(str(root), ResolvingShellProvider()))
    try:
        client = RecordingActionClient([{"action": "get_current_time", "timezone": "UTC"}])
        reopened.llm.client = client
        outcome = reopened.run_process_once(pid)
        assert not outcome["ok"]
        assert not client.user_prompts
        assert any(decision.outcome.value == "deny" and decision.labels.sensitivity.value == "secret"
                   for decision in reopened.store.list_data_flow_decisions(pid=pid)
                   if str(decision.sink) == "llm:default")
    finally:
        reopened.close()
