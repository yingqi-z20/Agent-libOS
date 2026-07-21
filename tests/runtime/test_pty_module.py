from __future__ import annotations

import os
import hashlib
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import psutil
import agent_libos.sdk.protected_operations as protected_operations

from agent_libos import Runtime
from agent_libos.capability.rules import AUTHORITY_RULES_KEY
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    DataFlowContext,
    DataSink,
    EventType,
    ExternalEffectClassification,
    ObjectMetadata,
    SinkTrustLevel,
    SinkTrustRule,
)
from agent_libos.models import ExternalEffectRollbackClass, ExternalEffectRollbackStatus, HumanRequestStatus, ObjectType, ResourceBudget
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, ValidationError
from agent_libos.primitives.git_command_policy import harden_read_only_git_argv
from agent_libos.substrate import LocalResourceProviderSubstrate, ProviderEffectNotStarted, SubprocessLimits
from modules.pty.pty_module import LocalPtyProvider, _PtyRuntimeSession


class TestPtyModule:
    def test_secret_pty_spawn_and_write_require_trusted_spawn_sink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = ResolvingPtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="labeled PTY egress")
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {"secret": "pty-data-flow-sentinel"},
                    metadata=ObjectMetadata(sensitivity="secret"),
                )
                adapter = _pty_adapter(runtime)

                with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
                    adapter.create(
                        pid,
                        ["git", "status"],
                        cwd=".",
                        startup_timeout_s=0,
                        source_oids=[source.oid],
                    )
                assert provider.resolver_calls == []
                assert provider.spawned == []

                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern="pty:spawn:*",
                        trust_level=SinkTrustLevel.TRUSTED,
                        max_sensitivity="secret",
                        identity_sha256=adapter.shell_policy.executable_data_sink(
                            "pty:spawn",
                            "git",
                            cwd=".",
                        ).identity_sha256,
                    ),
                    actor="test.host",
                    require_capability=False,
                )
                created = adapter.create(
                    pid,
                    ["git", "status"],
                    cwd=".",
                    startup_timeout_s=0,
                    source_oids=[source.oid],
                )
                written = adapter.write(
                    pid,
                    created.session_oid,
                    "pty-data-flow-sentinel\n",
                )

                assert len(provider.spawned) == 1
                assert provider.resolver_calls == [
                    harden_read_only_git_argv(["git", "status"])
                ]
                assert written.bytes_written == len("pty-data-flow-sentinel\n".encode())
                assert provider.sessions[0].writes == ["pty-data-flow-sentinel\n"]
            finally:
                runtime.close()

    def test_secret_pty_resize_and_close_deny_before_session_control(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="gate labeled PTY control operations",
                )
                adapter = _pty_adapter(runtime)
                created = adapter.create(
                    pid,
                    ["git", "status"],
                    cwd=".",
                    startup_timeout_s=0,
                )
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {"secret": "pty-control-data-flow-sentinel"},
                    metadata=ObjectMetadata(sensitivity="secret"),
                )
                secret_context = runtime.data_flow.context_from_source_oids(
                    pid,
                    [source.oid],
                )
                session = adapter._sessions[created.session_oid]

                with runtime.data_flow.activate(secret_context):
                    with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
                        adapter.resize(
                            pid,
                            created.session_oid,
                            cols=111,
                            rows=37,
                        )
                    with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
                        adapter.close(pid, created.session_oid)

                assert provider.sessions[0].size == (80, 24)
                assert not provider.sessions[0].closed
                assert not session.stop_event.is_set()
                assert created.session_oid in adapter._sessions
                assert runtime.store.get_object(created.session_oid) is not None
                sink_identity = f"pty:session:{session.session_id}"
                denied = runtime.store.list_data_flow_decisions(
                    pid=pid,
                    outcome="deny",
                )
                assert len(denied) == 2
                assert all(item.sink == sink_identity for item in denied)
                assert all(
                    item.labels.sensitivity.value == "secret"
                    for item in denied
                )
                assert len([
                    record
                    for record in runtime.audit.trace()
                    if record.action == "data_flow.egress"
                    and record.target == sink_identity
                    and record.decision.get("outcome") == "deny"
                ]) == 2
                assert len([
                    event
                    for event in runtime.events.list(
                        target=f"data_flow_sink:{sink_identity}",
                    )
                    if event.type == EventType.DATA_FLOW_DECISION
                    and event.payload.get("outcome") == "deny"
                ]) == 2

                resized = adapter.resize(
                    pid,
                    created.session_oid,
                    cols=100,
                    rows=30,
                )
                closed = adapter.close(pid, created.session_oid)

                assert (resized.cols, resized.rows) == (100, 30)
                assert closed.closed
                assert provider.sessions[0].size == (100, 30)
                assert provider.sessions[0].closed
            finally:
                runtime.close()

    def test_pty_close_prepare_denial_keeps_background_reader_alive(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="keep PTY alive after rejected close",
                )
                adapter = _pty_adapter(runtime)
                created = adapter.create(
                    pid,
                    ["git", "status"],
                    cwd=".",
                    startup_timeout_s=0,
                )
                session = adapter._sessions[created.session_oid]
                assert session.reader_thread is not None
                assert session.reader_thread.is_alive()
                original_authorize = runtime.data_flow.authorize_egress

                def reject_prepare_revalidation(**kwargs: Any) -> Any:
                    if kwargs.get("expected_registry_generation") is not None:
                        raise CapabilityDenied("sink changed before dispatch")
                    return original_authorize(**kwargs)

                monkeypatch.setattr(
                    runtime.data_flow,
                    "authorize_egress",
                    reject_prepare_revalidation,
                )

                with pytest.raises(CapabilityDenied, match="sink changed before dispatch"):
                    adapter.close(pid, created.session_oid)

                assert created.session_oid in adapter._sessions
                assert runtime.store.get_object(created.session_oid) is not None
                assert provider.sessions[0].closed is False
                assert session.closed is False
                assert session.closing is False
                assert session.close_complete.is_set()
                assert session.stop_event.is_set() is False
                assert session.reader_thread.is_alive()

                provider.sessions[0].outputs.append("reader-still-live\n")
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with session.lock:
                        if "reader-still-live\n" in "".join(session.buffer):
                            break
                    time.sleep(0.01)
                with session.lock:
                    assert "reader-still-live\n" in "".join(session.buffer)
            finally:
                runtime.close()

    @pytest.mark.parametrize("write_outcome", ["success", "ambiguous"])
    def test_pty_write_raises_session_data_flow_high_water(
        self,
        write_outcome: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal=f"PTY {write_outcome} write high-water",
                )
                adapter = _pty_adapter(runtime)
                created = adapter.create(
                    pid,
                    ["git", "status"],
                    cwd=".",
                    startup_timeout_s=0,
                )
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {"secret": "pty-write-data-flow-sentinel"},
                    metadata=ObjectMetadata(sensitivity="secret"),
                )
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern="pty:spawn:*",
                        trust_level=SinkTrustLevel.TRUSTED,
                        max_sensitivity="secret",
                        identity_sha256=adapter._sessions[
                            created.session_oid
                        ].data_sink.identity_sha256,
                    ),
                    actor="test.host",
                    require_capability=False,
                )
                if write_outcome == "ambiguous":
                    original_write = provider.sessions[0].write

                    def write_then_lose_outcome(text: str) -> int:
                        original_write(text)
                        raise TimeoutError("PTY write outcome is unknown")

                    monkeypatch.setattr(
                        provider.sessions[0],
                        "write",
                        write_then_lose_outcome,
                    )
                secret_context = runtime.data_flow.context_from_source_oids(
                    pid,
                    [source.oid],
                )

                with runtime.data_flow.activate(secret_context):
                    if write_outcome == "ambiguous":
                        with pytest.raises(TimeoutError, match="outcome is unknown"):
                            adapter.write(
                                pid,
                                created.session_oid,
                                "pty-write-data-flow-sentinel\n",
                            )
                    else:
                        adapter.write(
                            pid,
                            created.session_oid,
                            "pty-write-data-flow-sentinel\n",
                        )

                session = adapter._sessions[created.session_oid]
                assert session.data_flow_context.labels.sensitivity.value == "secret"
                assert source.oid in {
                    item.oid for item in session.data_flow_context.source_refs
                }
                stored_session = runtime.store.get_object(created.session_oid)
                assert stored_session is not None
                assert stored_session.metadata.sensitivity == "secret"

                with runtime.data_flow.activate(DataFlowContext()):
                    adapter.read(
                        pid,
                        created.session_oid,
                        timeout_s=0.2,
                    )
                    returned = runtime.data_flow.current_context()
                    assert returned.labels.sensitivity.value == "secret"
                    assert source.oid in {item.oid for item in returned.source_refs}
            finally:
                runtime.close()

    @pytest.mark.parametrize("resize_outcome", ["success", "ambiguous"])
    def test_pty_resize_raises_session_data_flow_high_water(
        self,
        resize_outcome: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal=f"PTY {resize_outcome} resize high-water",
                )
                adapter = _pty_adapter(runtime)
                created = adapter.create(
                    pid,
                    ["git", "status"],
                    cwd=".",
                    startup_timeout_s=0,
                )
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {"secret": "pty-resize-data-flow-sentinel"},
                    metadata=ObjectMetadata(sensitivity="secret"),
                )
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern="pty:spawn:*",
                        trust_level=SinkTrustLevel.TRUSTED,
                        max_sensitivity="secret",
                        identity_sha256=adapter._sessions[
                            created.session_oid
                        ].data_sink.identity_sha256,
                    ),
                    actor="test.host",
                    require_capability=False,
                )
                if resize_outcome == "ambiguous":
                    original_resize = provider.sessions[0].resize

                    def resize_then_lose_outcome(cols: int, rows: int) -> None:
                        original_resize(cols, rows)
                        raise TimeoutError("PTY resize outcome is unknown")

                    monkeypatch.setattr(
                        provider.sessions[0],
                        "resize",
                        resize_then_lose_outcome,
                    )
                secret_context = runtime.data_flow.context_from_source_oids(
                    pid,
                    [source.oid],
                )

                with runtime.data_flow.activate(secret_context):
                    if resize_outcome == "ambiguous":
                        with pytest.raises(TimeoutError, match="outcome is unknown"):
                            adapter.resize(
                                pid,
                                created.session_oid,
                                cols=100,
                                rows=30,
                            )
                    else:
                        adapter.resize(
                            pid,
                            created.session_oid,
                            cols=100,
                            rows=30,
                        )

                session = adapter._sessions[created.session_oid]
                assert provider.sessions[0].size == (100, 30)
                assert session.data_flow_context.labels.sensitivity.value == "secret"
                assert source.oid in {
                    item.oid for item in session.data_flow_context.source_refs
                }
                stored_session = runtime.store.get_object(created.session_oid)
                assert stored_session is not None
                assert stored_session.metadata.sensitivity == "secret"
            finally:
                runtime.close()

    def test_pty_ambiguous_close_raises_session_data_flow_high_water(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="PTY ambiguous close high-water",
                )
                adapter = _pty_adapter(runtime)
                created = adapter.create(
                    pid,
                    ["git", "status"],
                    cwd=".",
                    startup_timeout_s=0,
                )
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {"secret": "pty-close-data-flow-sentinel"},
                    metadata=ObjectMetadata(sensitivity="secret"),
                )
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern="pty:spawn:*",
                        trust_level=SinkTrustLevel.TRUSTED,
                        max_sensitivity="secret",
                        identity_sha256=adapter._sessions[
                            created.session_oid
                        ].data_sink.identity_sha256,
                    ),
                    actor="test.host",
                    require_capability=False,
                )
                original_close = provider.sessions[0].close

                def close_then_lose_outcome(
                    *,
                    force: bool = True,
                    timeout_s: float = 2.0,
                ) -> int | None:
                    original_close(force=force, timeout_s=timeout_s)
                    raise TimeoutError("PTY close outcome is unknown")

                monkeypatch.setattr(
                    provider.sessions[0],
                    "close",
                    close_then_lose_outcome,
                )
                secret_context = runtime.data_flow.context_from_source_oids(
                    pid,
                    [source.oid],
                )

                with runtime.data_flow.activate(secret_context):
                    with pytest.raises(TimeoutError, match="outcome is unknown"):
                        adapter.close(pid, created.session_oid)

                session = adapter._sessions[created.session_oid]
                assert session.close_outcome_unknown
                assert session.data_flow_context.labels.sensitivity.value == "secret"
                assert source.oid in {
                    item.oid for item in session.data_flow_context.source_refs
                }
                stored_session = runtime.store.get_object(created.session_oid)
                assert stored_session is not None
                assert stored_session.metadata.sensitivity == "secret"
            finally:
                runtime.close()

    @pytest.mark.parametrize("second_operation", ["write", "resize", "close"])
    def test_pty_control_waits_for_session_high_water_publication(
        self,
        second_operation: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            first_release = threading.Event()
            first_thread: threading.Thread | None = None
            second_thread: threading.Thread | None = None
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal=f"serialize PTY high-water before {second_operation}",
                )
                adapter = _pty_adapter(runtime)
                created = adapter.create(
                    pid,
                    ["git", "status"],
                    cwd=".",
                    startup_timeout_s=0,
                )
                session = adapter._sessions[created.session_oid]
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {"secret": "pty-serialized-high-water"},
                    metadata=ObjectMetadata(sensitivity="secret"),
                )
                secret_context = runtime.data_flow.context_from_source_oids(
                    pid,
                    [source.oid],
                )
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern="pty:spawn:*",
                        trust_level=SinkTrustLevel.CONDITIONAL,
                        max_sensitivity="secret",
                        identity_sha256=session.data_sink.identity_sha256,
                    ),
                    actor="test.host",
                    require_capability=False,
                )
                secret_text = "PTY_SERIALIZED_SECRET\n"
                with runtime.data_flow.activate(secret_context):
                    with pytest.raises(HumanApprovalRequired):
                        adapter.write(pid, created.session_oid, secret_text)
                runtime.human.drain_terminal_queue(auto_approve=True)

                first_provider_return_pending = threading.Event()
                second_provider_called = threading.Event()
                original_write = provider.sessions[0].write
                original_resize = provider.sessions[0].resize
                original_close = provider.sessions[0].close

                def coordinated_write(text: str) -> int:
                    if text == secret_text:
                        result = original_write(text)
                        first_provider_return_pending.set()
                        if not first_release.wait(timeout=10):
                            raise TimeoutError("secret PTY write was not released")
                        return result
                    second_provider_called.set()
                    return original_write(text)

                def tracked_resize(cols: int, rows: int) -> None:
                    second_provider_called.set()
                    original_resize(cols, rows)

                def tracked_close(
                    *,
                    force: bool = True,
                    timeout_s: float = 2.0,
                ) -> int | None:
                    second_provider_called.set()
                    return original_close(force=force, timeout_s=timeout_s)

                monkeypatch.setattr(provider.sessions[0], "write", coordinated_write)
                if second_operation == "resize":
                    monkeypatch.setattr(provider.sessions[0], "resize", tracked_resize)
                elif second_operation == "close":
                    monkeypatch.setattr(provider.sessions[0], "close", tracked_close)

                first_errors: list[BaseException] = []
                second_errors: list[BaseException] = []
                second_started = threading.Event()

                def run_secret_write() -> None:
                    try:
                        with runtime.data_flow.activate(secret_context):
                            adapter.write(pid, created.session_oid, secret_text)
                    except BaseException as exc:
                        first_errors.append(exc)

                def run_second_control() -> None:
                    second_started.set()
                    try:
                        if second_operation == "write":
                            adapter.write(pid, created.session_oid, "normal\n")
                        elif second_operation == "resize":
                            adapter.resize(
                                pid,
                                created.session_oid,
                                cols=100,
                                rows=30,
                            )
                        else:
                            adapter.close(pid, created.session_oid)
                    except BaseException as exc:
                        second_errors.append(exc)

                first_thread = threading.Thread(target=run_secret_write, daemon=True)
                second_thread = threading.Thread(target=run_second_control, daemon=True)
                first_thread.start()
                assert first_provider_return_pending.wait(timeout=10)
                second_thread.start()
                assert second_started.wait(timeout=10)
                try:
                    assert not second_provider_called.wait(timeout=0.1)
                finally:
                    first_release.set()
                    first_thread.join(timeout=10)
                    second_thread.join(timeout=10)

                assert not first_thread.is_alive()
                assert not second_thread.is_alive()
                assert first_errors == []
                assert len(second_errors) == 1
                assert isinstance(second_errors[0], HumanApprovalRequired)
                assert not second_provider_called.is_set()
                assert session.data_flow_context.labels.sensitivity.value == "secret"
                assert source.oid in {
                    item.oid for item in session.data_flow_context.source_refs
                }
                assert created.session_oid in adapter._sessions
                assert provider.sessions[0].closed is False
            finally:
                first_release.set()
                if first_thread is not None and first_thread.is_alive():
                    first_thread.join(timeout=2)
                if second_thread is not None and second_thread.is_alive():
                    second_thread.join(timeout=2)
                runtime.close()

    def test_pty_read_waits_for_secret_write_high_water_publication(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            write_release = threading.Event()
            write_thread: threading.Thread | None = None
            read_thread: threading.Thread | None = None
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="serialize PTY read with secret write high-water",
                )
                adapter = _pty_adapter(runtime)
                created = adapter.create(
                    pid,
                    ["git", "status"],
                    cwd=".",
                    startup_timeout_s=0,
                )
                session = adapter._sessions[created.session_oid]
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {"secret": "pty-read-write-high-water"},
                    metadata=ObjectMetadata(sensitivity="secret"),
                )
                secret_context = runtime.data_flow.context_from_source_oids(
                    pid,
                    [source.oid],
                )
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern="pty:spawn:*",
                        trust_level=SinkTrustLevel.TRUSTED,
                        max_sensitivity="secret",
                        identity_sha256=session.data_sink.identity_sha256,
                    ),
                    actor="test.host",
                    require_capability=False,
                )

                secret_text = "PTY_READ_WRITE_SECRET\n"
                write_output_produced = threading.Event()
                original_write = provider.sessions[0].write

                def coordinated_write(text: str) -> int:
                    result = original_write(text)
                    write_output_produced.set()
                    if not write_release.wait(timeout=10):
                        raise TimeoutError("secret PTY write was not released")
                    return result

                monkeypatch.setattr(provider.sessions[0], "write", coordinated_write)
                write_errors: list[BaseException] = []

                def run_secret_write() -> None:
                    try:
                        with runtime.data_flow.activate(secret_context):
                            adapter.write(pid, created.session_oid, secret_text)
                    except BaseException as exc:
                        write_errors.append(exc)

                write_thread = threading.Thread(target=run_secret_write, daemon=True)
                write_thread.start()
                assert write_output_produced.wait(timeout=10)

                expected_output = f"echo:{secret_text}"
                deadline = time.monotonic() + 5
                buffered_output = ""
                while time.monotonic() < deadline:
                    with session.lock:
                        buffered_output = "".join(session.buffer)
                    if expected_output in buffered_output:
                        break
                    time.sleep(0.01)
                assert expected_output in buffered_output

                read_polled = threading.Event()
                original_buffer_is_empty = adapter._buffer_is_empty

                def tracked_buffer_is_empty(selected_session: _PtyRuntimeSession) -> bool:
                    result = original_buffer_is_empty(selected_session)
                    if selected_session is session:
                        read_polled.set()
                    return result

                monkeypatch.setattr(adapter, "_buffer_is_empty", tracked_buffer_is_empty)
                read_done = threading.Event()
                read_errors: list[BaseException] = []
                read_results: list[Any] = []
                read_contexts: list[DataFlowContext] = []

                def read_output() -> None:
                    try:
                        with runtime.data_flow.activate(DataFlowContext()):
                            read_results.append(
                                adapter.read(
                                    pid,
                                    created.session_oid,
                                    timeout_s=0.5,
                                )
                            )
                            read_contexts.append(runtime.data_flow.current_context())
                    except BaseException as exc:
                        read_errors.append(exc)
                    finally:
                        read_done.set()

                read_thread = threading.Thread(target=read_output, daemon=True)
                read_thread.start()
                assert read_polled.wait(timeout=10)
                try:
                    # The output is already buffered, but its secret write has
                    # not published the session high-water yet. A read must
                    # wait for that publication before returning the bytes.
                    assert not read_done.wait(timeout=0.1)
                finally:
                    write_release.set()
                    write_thread.join(timeout=10)
                    read_thread.join(timeout=10)

                assert not write_thread.is_alive()
                assert not read_thread.is_alive()
                assert write_errors == []
                assert read_errors == []
                assert len(read_results) == 1
                assert expected_output in read_results[0].output
                assert len(read_contexts) == 1
                assert read_contexts[0].labels.sensitivity.value == "secret"
                assert source.oid in {ref.oid for ref in read_contexts[0].source_refs}
                assert session.data_flow_context.labels.sensitivity.value == "secret"
            finally:
                write_release.set()
                if write_thread is not None and write_thread.is_alive():
                    write_thread.join(timeout=2)
                if read_thread is not None and read_thread.is_alive():
                    read_thread.join(timeout=2)
                runtime.close()

    def test_secret_pty_spawn_uses_provider_resolved_executable_identity(self) -> None:
        if os.name == "nt":
            pytest.skip("relative executable PTY identity uses POSIX test script")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            commands = root / "commands"
            commands.mkdir()
            executable = commands / "trusted-tool"
            marker = commands / "ran.txt"
            executable.write_text(
                "#!/bin/sh\nprintf ran > ran.txt\nprintf 'ready\\n'\nsleep 2\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            runtime = _open_pty_runtime(temp_dir, LocalPtyProvider(root))
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="resolved PTY executable Sink")
                runtime.shell.grant_policy(
                    pid,
                    runtime.config.shell.always_allow_level,
                    issued_by="test",
                )
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {"secret": "resolved-pty-executable-sentinel"},
                    metadata=ObjectMetadata(sensitivity="secret"),
                )
                adapter = _pty_adapter(runtime)
                # This test covers executable Sink identity, not the independent
                # psutil-backed resource monitor. Disable that monitor so a
                # restricted macOS worker cannot close the PTY before the test
                # script records that dispatch occurred.
                adapter.resources = None
                old_process_cwd_identity = (Path.cwd() / "trusted-tool").resolve().as_posix()
                actual_identity = executable.resolve().as_posix()
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern=f"pty:spawn:{old_process_cwd_identity}",
                        trust_level=SinkTrustLevel.TRUSTED,
                        max_sensitivity="secret",
                        identity_sha256=hashlib.sha256(
                            executable.read_bytes()
                        ).hexdigest(),
                    ),
                    actor="test.host",
                    require_capability=False,
                )

                with pytest.raises(CapabilityDenied, match="data-flow denied egress"):
                    adapter.create(
                        pid,
                        ["./trusted-tool"],
                        cwd="commands",
                        startup_timeout_s=0.1,
                        source_oids=[source.oid],
                    )
                assert not marker.exists()
                assert runtime.store.list_external_effects(pid=pid) == []
                denied = runtime.store.list_data_flow_decisions(pid=pid, outcome="deny")
                assert len(denied) == 1
                assert denied[0].sink == f"pty:spawn:{actual_identity}"
                assert any(
                    record.action == "data_flow.egress"
                    and record.target == f"pty:spawn:{actual_identity}"
                    and record.decision.get("outcome") == "deny"
                    for record in runtime.audit.trace()
                )
                assert any(
                    event.type == EventType.DATA_FLOW_DECISION
                    and event.payload.get("outcome") == "deny"
                    for event in runtime.events.list(
                        target=f"data_flow_sink:pty:spawn:{actual_identity}"
                    )
                )

                runtime.data_flow.unregister_sink_trust(
                    f"pty:spawn:{old_process_cwd_identity}",
                    actor="test.host",
                    require_capability=False,
                )
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern=f"pty:spawn:{actual_identity}",
                        trust_level=SinkTrustLevel.TRUSTED,
                        max_sensitivity="secret",
                        identity_sha256=hashlib.sha256(
                            executable.read_bytes()
                        ).hexdigest(),
                    ),
                    actor="test.host",
                    require_capability=False,
                )
                created = adapter.create(
                    pid,
                    ["./trusted-tool"],
                    cwd="commands",
                    startup_timeout_s=0.2,
                    source_oids=[source.oid],
                )

                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not marker.exists():
                    time.sleep(0.01)
                assert marker.read_text(encoding="utf-8") == "ran"
                adapter.close(pid, created.session_oid)
            finally:
                runtime.close()

    def test_pty_executable_snapshot_preserves_sibling_resource_access(self) -> None:
        if os.name == "nt":
            pytest.skip("relative executable PTY identity uses POSIX test script")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            commands = root / "commands"
            commands.mkdir()
            executable = commands / "read-sibling"
            observed = commands / "observed.txt"
            (commands / "asset.txt").write_text(
                "pty sibling payload",
                encoding="utf-8",
            )
            executable.write_text(
                '#!/bin/sh\ncat "$(dirname "$0")/asset.txt" > observed.txt\nsleep 2\n',
                encoding="utf-8",
            )
            executable.chmod(0o755)
            runtime = _open_pty_runtime(temp_dir, LocalPtyProvider(root))
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="run PTY workspace script with a sibling asset",
                )
                runtime.shell.grant_policy(
                    pid,
                    runtime.config.shell.always_allow_level,
                    issued_by="test",
                )
                adapter = _pty_adapter(runtime)
                adapter.resources = None

                created = adapter.create(
                    pid,
                    ["./read-sibling"],
                    cwd="commands",
                    startup_timeout_s=0.2,
                )

                deadline = time.monotonic() + 1.0
                observed_content: str | None = None
                while time.monotonic() < deadline:
                    if observed.exists():
                        observed_content = observed.read_text(encoding="utf-8")
                        if observed_content == "pty sibling payload":
                            break
                    time.sleep(0.01)
                assert observed_content == "pty sibling payload"
                adapter.close(pid, created.session_oid)
            finally:
                runtime.close()

    def test_replaced_pty_executable_loses_secret_sink_trust_before_spawn(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if os.name == "nt":
            pytest.skip("relative executable PTY identity uses POSIX test script")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            commands = root / "commands"
            commands.mkdir()
            executable = commands / "trusted-tool"
            stolen = commands / "stolen.txt"
            executable.write_text(
                "#!/bin/sh\nprintf 'trusted\\n'\nsleep 2\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            provider = LocalPtyProvider(root)
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="PTY executable replacement PoC",
                )
                runtime.shell.grant_policy(
                    pid,
                    runtime.config.shell.always_allow_level,
                    issued_by="test",
                )
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {"secret": "PTY_EXECUTABLE_REPLACEMENT_SECRET"},
                    metadata=ObjectMetadata(sensitivity="secret"),
                )
                executable_identity = executable.resolve().as_posix()
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern=f"pty:spawn:{executable_identity}",
                        trust_level=SinkTrustLevel.TRUSTED,
                        max_sensitivity="secret",
                        identity_sha256=hashlib.sha256(
                            executable.read_bytes()
                        ).hexdigest(),
                    ),
                    actor="test.host",
                    require_capability=False,
                )
                original_resolve = provider.resolve_argv

                def replace_after_resolve(
                    argv: list[str],
                    *,
                    cwd: str | None = None,
                ) -> list[str]:
                    resolved = original_resolve(argv, cwd=cwd)
                    executable.write_text(
                        "#!/bin/sh\nprintf '%s' \"$1\" > stolen.txt\nsleep 2\n",
                        encoding="utf-8",
                    )
                    return resolved

                monkeypatch.setattr(provider, "resolve_argv", replace_after_resolve)

                adapter = _pty_adapter(runtime)
                with pytest.raises(CapabilityDenied, match="Sink identity changed"):
                    adapter.create(
                        pid,
                        ["./trusted-tool", "PTY_EXECUTABLE_REPLACEMENT_SECRET"],
                        cwd="commands",
                        startup_timeout_s=0.1,
                        source_oids=[source.oid],
                    )

                assert not stolen.exists()
                assert adapter._sessions == {}
                denied = runtime.store.list_data_flow_decisions(pid=pid, outcome="deny")
                assert len(denied) == 1
                assert denied[0].sink == f"pty:spawn:{executable_identity}"
                assert denied[0].reason == "Sink identity changed before provider dispatch"
            finally:
                runtime.close()

    def test_final_dispatch_race_executes_authorized_pty_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if os.name == "nt":
            pytest.skip("relative executable PTY identity uses POSIX test script")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            commands = root / "commands"
            commands.mkdir()
            executable = commands / "trusted-tool"
            trusted = commands / "trusted.txt"
            stolen = commands / "stolen.txt"
            executable.write_text(
                "#!/bin/sh\nprintf trusted > trusted.txt\nprintf 'ready\\n'\nsleep 2\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            provider = LocalPtyProvider(root)
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="close PTY executable dispatch TOCTOU",
                )
                runtime.shell.grant_policy(
                    pid,
                    runtime.config.shell.always_allow_level,
                    issued_by="test",
                )
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {"secret": "FINAL_DISPATCH_PTY_SECRET"},
                    metadata=ObjectMetadata(sensitivity="secret"),
                )
                executable_identity = executable.resolve().as_posix()
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern=f"pty:spawn:{executable_identity}",
                        trust_level=SinkTrustLevel.TRUSTED,
                        max_sensitivity="secret",
                        identity_sha256=hashlib.sha256(
                            executable.read_bytes()
                        ).hexdigest(),
                    ),
                    actor="test.host",
                    require_capability=False,
                )
                original_mark_dispatched = (
                    protected_operations.mark_external_effect_dispatched
                )
                dispatch_count = 0

                def replace_after_final_validation(store: Any, effect_id: str) -> Any:
                    nonlocal dispatch_count
                    result = original_mark_dispatched(store, effect_id)
                    dispatch_count += 1
                    if dispatch_count == 2:
                        executable.write_text(
                            "#!/bin/sh\nprintf '%s' \"$1\" > stolen.txt\nprintf 'replacement\\n'\nsleep 2\n",
                            encoding="utf-8",
                        )
                        executable.chmod(0o755)
                    return result

                monkeypatch.setattr(
                    protected_operations,
                    "mark_external_effect_dispatched",
                    replace_after_final_validation,
                )
                adapter = _pty_adapter(runtime)
                adapter.resources = None

                created = adapter.create(
                    pid,
                    ["./trusted-tool", "FINAL_DISPATCH_PTY_SECRET"],
                    cwd="commands",
                    startup_timeout_s=0.2,
                    source_oids=[source.oid],
                )

                assert dispatch_count >= 2
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not trusted.exists():
                    time.sleep(0.01)
                assert trusted.read_text(encoding="utf-8") == "trusted"
                assert not stolen.exists()
                adapter.close(pid, created.session_oid)
            finally:
                runtime.close()

    def test_pty_resolver_cannot_switch_authorized_executable_sink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = SwitchingPtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="reject PTY resolver Sink switch",
                )
                adapter = _pty_adapter(runtime)

                with pytest.raises(
                    ValidationError,
                    match="provider executable resolver changed the authorized Sink identity",
                ):
                    adapter.create(
                        pid,
                        ["git", "status"],
                        cwd=".",
                        startup_timeout_s=0,
                    )

                assert provider.resolver_calls == [
                    harden_read_only_git_argv(["git", "status"])
                ]
                assert provider.spawned == []
                effects = runtime.store.list_external_effects(pid=pid)
                assert len(effects) == 1
                assert effects[0].operation == "spawn"
                assert effects[0].transaction_state == "unknown"
                assert effects[0].information_flow is True
                assert effects[0].provider_metadata["outcome"] == "unknown_after_provider_success"
                assert effects[0].provider_metadata["provider_phases"] == [
                    {
                        "name": "resolve_argv",
                        "state_mutation": False,
                        "information_flow": True,
                    }
                ]
            finally:
                runtime.close()

    def test_loaded_module_registers_object_bound_pty_tools_and_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                loaded = runtime.modules.inspect_module("agent-libos-pty:v0")
                assert loaded["status"] == "loaded"
                assert "pty_create" in loaded["registered"]["tools"]
                assert "pty-agent:v0" in runtime.images

                pid = runtime.process.spawn(image="pty-agent:v0", goal="use pty")
                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": ["git", "status"], "startup_timeout_s": 0.05, "max_output_chars": 100},
                )

                assert created.ok, created.error
                session_oid = created.payload["session_oid"]
                obj = runtime.store.get_object(session_oid)
                assert obj is not None
                assert obj.type == ObjectType.EXTERNAL_REF
                assert obj.payload["kind"] == "pty_session"
                assert obj.payload["argv"] == ["git", "status"]
                assert session_oid in [handle.oid for handle in runtime.process.get(pid).memory_view.roots]
                assert created.payload["output"] == "ready\n"

                written = runtime.tools.call(pid, "pty_write", {"session_oid": session_oid, "text": "hello\n"})
                assert written.ok, written.error
                assert written.payload["bytes_written"] == len("hello\n".encode("utf-8"))

                read = runtime.tools.call(pid, "pty_read", {"session_oid": session_oid, "timeout_s": 0.5})
                assert read.ok, read.error
                assert "echo:hello" in read.payload["output"]

                resized = runtime.tools.call(pid, "pty_resize", {"session_oid": session_oid, "cols": 100, "rows": 30})
                assert resized.ok, resized.error
                assert provider.sessions[0].size == (100, 30)

                listed = runtime.tools.call(pid, "pty_list", {})
                assert listed.ok, listed.error
                assert [entry["session_oid"] for entry in listed.payload["sessions"]] == [session_oid]

                closed = runtime.tools.call(pid, "pty_close", {"session_oid": session_oid})
                assert closed.ok, closed.error
                assert provider.sessions[0].closed
                assert runtime.store.get_object(session_oid) is None
                assert "primitive.pty.spawn" in [record.action for record in runtime.audit.trace()]
                assert any(effect.provider == "pty" for effect in runtime.store.list_external_effects())
            finally:
                runtime.close()

    def test_pty_create_tool_explicit_cwd_uses_directory_read_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            explicit_cwd = Path(temp_dir) / "work"
            explicit_cwd.mkdir()
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty explicit cwd")
                runtime.capability.issue_trusted(
                    pid,
                    runtime.filesystem.directory_resource_for_path("work"),
                    [CapabilityRight.READ],
                    issued_by="test.pty.cwd",
                )

                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": ["git", "status"], "cwd": "work", "startup_timeout_s": 0},
                )

                assert created.ok, created.error
                assert provider.spawned[0]["cwd"] == "work"
            finally:
                runtime.close()

    def test_process_exit_releases_pty_session_object_and_closes_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty raii exit")
                created = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]

                runtime.process.exit(pid)

                assert provider.sessions[0].closed
                assert runtime.store.get_object(session_oid) is None
            finally:
                runtime.close()

    def test_direct_object_release_closes_pty_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="direct release")
                created = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]

                assert runtime.memory.delete_object_trusted("test", session_oid, reason="direct_release")

                assert provider.sessions[0].closed
                assert runtime.store.get_object(session_oid) is None
            finally:
                runtime.close()

    def test_direct_object_release_unknown_close_keeps_object_and_blocks_blind_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(close_failures=1)
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="direct release retry")
                created = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]

                with pytest.raises(RuntimeError, match="simulated close failure"):
                    runtime.memory.delete_object_trusted("test", session_oid, reason="direct_release_failure")

                assert provider.sessions[0].closed is False
                assert runtime.store.get_object(session_oid) is not None
                assert session_oid in _pty_adapter(runtime)._sessions

                with pytest.raises(ValidationError, match="unresolved prior close outcome"):
                    runtime.memory.delete_object_trusted("test", session_oid, reason="direct_release_retry")
                assert provider.sessions[0].closed is False
                assert runtime.store.get_object(session_oid) is not None
            finally:
                runtime.close()

    def test_direct_object_release_db_failure_preserves_close_effect_and_retries_relational_delete(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="direct release db retry")
                created = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]
                original_record = runtime.audit.record

                def fail_delete_audit(*args: Any, **kwargs: Any) -> Any:
                    if kwargs.get("action") == "memory.delete_object":
                        raise RuntimeError("injected object delete audit failure")
                    return original_record(*args, **kwargs)

                monkeypatch.setattr(runtime.audit, "record", fail_delete_audit)

                with pytest.raises(RuntimeError, match="injected object delete audit failure"):
                    runtime.memory.delete_object_trusted("test", session_oid, reason="direct_release_db_failure")

                assert provider.sessions[0].closed
                assert runtime.store.get_object(session_oid) is not None
                close_effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == "close" and effect.target == f"pty:{session_oid}"
                ]
                assert len(close_effects) == 1
                assert close_effects[0].effect_state == "finalized"

                monkeypatch.setattr(runtime.audit, "record", original_record)
                assert runtime.memory.delete_object_trusted("test", session_oid, reason="direct_release_db_retry")
                assert runtime.store.get_object(session_oid) is None
                assert [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == "close" and effect.target == f"pty:{session_oid}"
                ] == close_effects
            finally:
                runtime.close()

    def test_pty_create_requires_shell_policy_before_provider_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                runtime.register_image(
                    AgentImage(
                        image_id="pty-no-policy:v0",
                        name="pty-no-policy",
                        default_tools=["pty_create"],
                    ),
                    actor="cli",
                )
                pid = runtime.process.spawn(image="pty-no-policy:v0", goal="no shell policy")

                result = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"]})

                assert not result.ok
                assert "lacks shell execute policy" in (result.error or "")
                assert provider.spawned == []
            finally:
                runtime.close()

    def test_pty_create_rejects_launcher_wrapped_git_before_provider_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="reject wrapped Git")
                runtime.shell.grant_policy(
                    pid,
                    runtime.config.shell.always_allow_level,
                    issued_by="test",
                )

                with pytest.raises(ValidationError, match="typed git_"):
                    _pty_adapter(runtime).create(
                        pid,
                        ["env", "git", "branch", "wrapper-created"],
                        cwd=".",
                        startup_timeout_s=0,
                    )

                assert provider.spawned == []
            finally:
                runtime.close()

    def test_pty_create_post_spawn_failure_closes_handle_and_removes_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="post spawn failure")
                adapter = _pty_adapter(runtime)

                def fail_start_reader(*_args: Any, **_kwargs: Any) -> None:
                    raise RuntimeError("reader setup failed")

                monkeypatch.setattr(adapter, "_start_reader", fail_start_reader)

                result = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})

                assert not result.ok
                assert provider.sessions[0].closed
                assert adapter._sessions == {}
                assert [
                    obj
                    for obj in runtime.store.list_objects()
                    if isinstance(obj.payload, dict) and obj.payload.get("kind") == "pty_session"
                ] == []
            finally:
                runtime.close()

    def test_pty_provider_certified_pre_effect_failure_restores_one_time_use(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = PreEffectFailurePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty pre-effect failure")
                argv = ["git", "status"]
                capability = _grant_exact_pty_once(runtime, pid, argv)

                with pytest.raises(ProviderEffectNotStarted, match="before PTY spawn"):
                    _pty_adapter(runtime).create(pid, argv, cwd=".", startup_timeout_s=0)

                assert runtime.store.get_capability(capability.cap_id).uses_remaining == 1
                assert runtime.store.list_external_effects(pid=pid) == []
                assert _pty_adapter(runtime)._pending_session_creates == 0
            finally:
                runtime.close()

    def test_pty_sdk_enter_failure_releases_pending_session_capacity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = _open_pty_runtime(temp_dir, FakePtyProvider(initial_outputs=[]))
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty enter failure")
                adapter = _pty_adapter(runtime)

                class FailingContext:
                    def __enter__(self):
                        raise RuntimeError("protected enter failed")

                    def __exit__(self, *_args: Any) -> bool:
                        return False

                monkeypatch.setattr(
                    runtime.protected_operations,
                    "start",
                    lambda *_args, **_kwargs: FailingContext(),
                )

                with pytest.raises(RuntimeError, match="protected enter failed"):
                    adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0)

                assert adapter._pending_session_creates == 0
                assert adapter._pending_session_creates_by_process == {}
            finally:
                runtime.close()

    def test_pty_ambiguous_spawn_failure_commits_once_and_records_unknown_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = AmbiguousFailurePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty ambiguous spawn failure")
                argv = ["git", "status"]
                capability = _grant_exact_pty_once(runtime, pid, argv)

                with pytest.raises(TimeoutError, match="spawn outcome is unknown"):
                    _pty_adapter(runtime).create(pid, argv, cwd=".", startup_timeout_s=0)

                assert runtime.store.get_capability(capability.cap_id).uses_remaining == 0
                effects = runtime.store.list_external_effects(pid=pid)
                assert len(effects) == 1
                assert effects[0].provider == "pty"
                assert effects[0].operation == "spawn"
                assert effects[0].rollback_class == ExternalEffectRollbackClass.UNKNOWN
                assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
                assert effects[0].provider_metadata["outcome"] == "unknown_after_provider_exception"
                assert _pty_adapter(runtime)._pending_session_creates == 0
            finally:
                runtime.close()

    def test_pty_session_object_failure_records_unknown_spawn_effect_and_consumes_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty object creation failure")
                argv = ["git", "status"]
                capability = _grant_exact_pty_once(runtime, pid, argv)
                adapter = _pty_adapter(runtime)

                def fail_session_object(*_args: Any, **_kwargs: Any) -> tuple[str, str, str]:
                    raise RuntimeError("session object creation failed")

                monkeypatch.setattr(adapter, "_create_session_object", fail_session_object)

                with pytest.raises(RuntimeError, match="session object creation failed"):
                    adapter.create(pid, argv, cwd=".", startup_timeout_s=0)

                assert runtime.store.get_capability(capability.cap_id).uses_remaining == 0
                assert provider.sessions[0].closed
                assert adapter._pending_session_creates == 0
                assert [
                    obj
                    for obj in runtime.store.list_objects()
                    if isinstance(obj.payload, dict) and obj.payload.get("kind") == "pty_session"
                ] == []
                effects = runtime.store.list_external_effects(pid=pid)
                assert len(effects) == 1
                effect = effects[0]
                assert effect.provider == "pty"
                assert effect.operation == "spawn"
                assert effect.rollback_class == ExternalEffectRollbackClass.UNKNOWN
                assert effect.rollback_status == ExternalEffectRollbackStatus.UNKNOWN
                assert effect.provider_metadata["outcome"] == "unknown_after_provider_success"
                assert effect.provider_metadata["failure_phase"] == "session_object_creation"
                assert effect.provider_metadata["cleanup"]["attempted"] is True
                assert effect.provider_metadata["cleanup"]["succeeded"] is True
            finally:
                runtime.close()

    def test_pty_write_resize_and_close_each_record_external_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty mutation effects")
                adapter = _pty_adapter(runtime)
                created = adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0)

                adapter.write(pid, created.session_oid, "hello\n")
                adapter.resize(pid, created.session_oid, cols=100, rows=30)
                adapter.close(pid, created.session_oid)

                effects = runtime.store.list_external_effects(pid=pid)
                by_operation = {effect.operation: effect for effect in effects}
                assert set(by_operation) == {"spawn", "ingest", "write", "resize", "close"}
                for operation in ("write", "resize", "close"):
                    effect = by_operation[operation]
                    assert effect.target == f"pty:{created.session_oid}"
                    assert effect.record_id is not None
                    assert effect.event_id is not None
                    assert effect.rollback_class == ExternalEffectRollbackClass.IRREVERSIBLE
                    assert effect.rollback_status == ExternalEffectRollbackStatus.NOT_SUPPORTED
                assert by_operation["write"].provider_metadata["bytes_written"] == len("hello\n".encode("utf-8"))
                assert by_operation["resize"].provider_metadata["cols"] == 100
                assert by_operation["resize"].provider_metadata["rows"] == 30
                assert by_operation["close"].provider_metadata["reason"] == "pty_close"
            finally:
                runtime.close()

    def test_pty_background_reader_uses_runtime_internal_ingest_operation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = _open_pty_runtime(temp_dir, FakePtyProvider(initial_outputs=["ready\n"]))
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty ingest operation")
                adapter = _pty_adapter(runtime)
                created = adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0.05)

                adapter.close(pid, created.session_oid)

                ingest = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == "ingest"
                    and effect.target == f"pty:{created.session_oid}"
                ]
                assert len(ingest) == 1
                assert ingest[0].effect_state == "finalized"
                assert ingest[0].information_flow is True
            finally:
                runtime.close()

    def test_pty_write_event_failure_leaves_durable_pending_effect_intent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty write pending intent")
                adapter = _pty_adapter(runtime)
                created = adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0)
                finite = _grant_pty_object_once(
                    runtime,
                    pid,
                    created.session_oid,
                    CapabilityRight.WRITE,
                )
                original_emit = adapter.events.emit

                def fail_write_event(event_type: Any, **kwargs: Any) -> Any:
                    payload = kwargs.get("payload") or {}
                    if payload.get("operation") == "write":
                        raise RuntimeError("write event sink failed")
                    return original_emit(event_type, **kwargs)

                monkeypatch.setattr(adapter.events, "emit", fail_write_event)

                with pytest.raises(RuntimeError, match="write event sink failed"):
                    adapter.write(pid, created.session_oid, "hello\n")

                assert provider.sessions[0].writes == ["hello\n"]
                effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == "write"
                ]
                assert len(effects) == 1
                pending = effects[0]
                assert pending.effect_state == "pending"
                assert pending.record_id is None
                assert pending.event_id is None
                assert pending.rollback_class == ExternalEffectRollbackClass.UNKNOWN
                assert pending.rollback_status == ExternalEffectRollbackStatus.UNKNOWN
                assert pending.target == f"pty:{created.session_oid}"
                assert pending.provider_metadata["effect_state"] == "pending"
                assert runtime.store.get_capability(finite.cap_id).uses_remaining == 0
            finally:
                runtime.close()

    @pytest.mark.parametrize("operation", ["write", "resize", "close"])
    def test_pty_one_time_mutation_provider_not_started_restores_use_and_abandons_intent(
        self,
        operation: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal=f"pty {operation} PENS")
                adapter = _pty_adapter(runtime)
                created = adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0)
                right = CapabilityRight.DELETE if operation == "close" else CapabilityRight.WRITE
                finite = _grant_pty_object_once(runtime, pid, created.session_oid, right)
                handle = provider.sessions[0]
                original = getattr(handle, operation)

                def fail_not_started(*_args: Any, **_kwargs: Any) -> Any:
                    raise ProviderEffectNotStarted(f"{operation} did not start")

                monkeypatch.setattr(handle, operation, fail_not_started)

                with pytest.raises(ProviderEffectNotStarted, match="did not start"):
                    _invoke_pty_mutation(adapter, pid, created.session_oid, operation)

                assert runtime.store.get_capability(finite.cap_id).uses_remaining == 1
                assert [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == operation
                ] == []

                monkeypatch.setattr(handle, operation, original)
                _invoke_pty_mutation(adapter, pid, created.session_oid, operation)
                assert runtime.store.get_capability(finite.cap_id).uses_remaining == 0
                effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == operation
                ]
                assert len(effects) == 1
                assert effects[0].effect_state == "finalized"
            finally:
                runtime.close()

    @pytest.mark.parametrize("operation", ["write", "resize", "close"])
    def test_pty_one_time_mutation_ambiguous_failure_consumes_use_and_finalizes_unknown(
        self,
        operation: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal=f"pty {operation} ambiguous")
                adapter = _pty_adapter(runtime)
                created = adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0)
                right = CapabilityRight.DELETE if operation == "close" else CapabilityRight.WRITE
                finite = _grant_pty_object_once(runtime, pid, created.session_oid, right)
                handle = provider.sessions[0]
                original = getattr(handle, operation)

                def fail_ambiguous(*_args: Any, **_kwargs: Any) -> Any:
                    raise RuntimeError(f"{operation} outcome unknown")

                monkeypatch.setattr(handle, operation, fail_ambiguous)

                with pytest.raises(RuntimeError, match="outcome unknown"):
                    _invoke_pty_mutation(adapter, pid, created.session_oid, operation)

                assert runtime.store.get_capability(finite.cap_id).uses_remaining == 0
                effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == operation
                ]
                assert len(effects) == 1
                assert effects[0].effect_state == "finalized"
                assert effects[0].transaction_state == "unknown"
                assert effects[0].provider_metadata["outcome"] == "unknown_after_provider_exception"
                assert effects[0].rollback_class == ExternalEffectRollbackClass.UNKNOWN

                monkeypatch.setattr(handle, operation, original)
            finally:
                runtime.close()

    def test_pty_list_consumes_finite_read_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                owner = runtime.process.spawn(image="pty-agent:v0", goal="pty list owner")
                observer = runtime.process.spawn(image="pty-agent:v0", goal="pty list observer")
                adapter = _pty_adapter(runtime)
                created = adapter.create(owner, ["git", "status"], cwd=".", startup_timeout_s=0)
                finite = _grant_pty_object_once(
                    runtime,
                    observer,
                    created.session_oid,
                    CapabilityRight.READ,
                )

                assert [entry.session_oid for entry in adapter.list(observer)] == [created.session_oid]
                assert runtime.store.get_capability(finite.cap_id).uses_remaining == 0
                assert adapter.list(observer) == []
            finally:
                runtime.close()

    def test_pty_list_tool_aggregates_returned_session_data_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = ResolvingPtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image='pty-agent:v0',
                    goal='preserve PTY list metadata labels',
                )
                runtime.shell.grant_policy(
                    pid,
                    runtime.config.shell.always_allow_level,
                    issued_by='test',
                )
                adapter = _pty_adapter(runtime)
                ordinary = adapter.create(
                    pid,
                    ['git', 'status'],
                    cwd='.',
                    startup_timeout_s=0,
                )
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {'secret': 'PTY_LIST_SECRET_ARGV_SENTINEL'},
                    metadata=ObjectMetadata(sensitivity='secret'),
                )
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern='pty:spawn:*',
                        trust_level=SinkTrustLevel.TRUSTED,
                        max_sensitivity='secret',
                        identity_sha256=adapter.shell_policy.executable_data_sink(
                            'pty:spawn',
                            'echo',
                            cwd='.',
                        ).identity_sha256,
                    ),
                    actor='test.host',
                    require_capability=False,
                )
                secret = adapter.create(
                    pid,
                    ['echo', 'PTY_LIST_SECRET_ARGV_SENTINEL'],
                    cwd='.',
                    startup_timeout_s=0,
                    source_oids=[source.oid],
                )

                with runtime.data_flow.activate(DataFlowContext()):
                    listed = runtime.tools.call(pid, 'pty_list', {})

                assert listed.ok, listed.error
                assert {
                    item['session_oid'] for item in listed.payload['sessions']
                } == {ordinary.session_oid, secret.session_oid}
                assert any(
                    'PTY_LIST_SECRET_ARGV_SENTINEL' in item['argv']
                    for item in listed.payload['sessions']
                )
                assert listed.result_handle is not None
                carrier = runtime.store.get_object(listed.result_handle.oid)
                assert carrier is not None
                assert carrier.metadata.sensitivity == 'secret'
                assert source.oid in carrier.provenance.parent_oids

                returned_context = runtime.data_flow.context_from_source_oids(
                    pid,
                    [listed.result_handle.oid],
                    include_current=False,
                )
                sink = DataSink(identity='human:owner:pty-list-regression')
                with pytest.raises(CapabilityDenied, match='data-flow denied egress'):
                    runtime.data_flow.authorize_egress(
                        pid=pid,
                        sink=sink,
                        context=returned_context,
                        payload=listed.payload,
                        operation='test.pty_list.forward',
                    )
                denial = runtime.store.list_data_flow_decisions(pid=pid, outcome='deny')[-1]
                assert denial.sink == sink.identity
                assert denial.labels.sensitivity.value == 'secret'
                assert listed.result_handle.oid in {
                    item.oid for item in denial.source_refs
                }
                assert any(
                    record.action == 'data_flow.egress'
                    and record.target == sink.identity
                    and record.decision.get('decision_id') == denial.decision_id
                    for record in runtime.audit.trace()
                )
                assert any(
                    event.type == EventType.DATA_FLOW_DECISION
                    and event.payload.get('decision_id') == denial.decision_id
                    for event in runtime.events.list(
                        target=f'data_flow_sink:{sink.identity}',
                    )
                )
            finally:
                runtime.close()

    def test_pty_auto_exit_finalizes_close_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty auto exit")
                adapter = _pty_adapter(runtime)
                created = adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0)
                session = adapter._sessions[created.session_oid]
                session.stop_event.set()
                adapter._join_session_workers(session, timeout_s=1.0)
                provider.sessions[0].alive = False

                adapter._mark_session_exited(session, resource="shell:git")

                close_effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == "close"
                ]
                assert len(close_effects) == 1
                assert close_effects[0].effect_state == "finalized"
                assert close_effects[0].information_flow is True
                assert close_effects[0].provider_metadata["reason"] == "process_exit"
            finally:
                runtime.close()

    def test_pty_auto_exit_event_failure_leaves_pending_close_intent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty auto exit sink failure")
                adapter = _pty_adapter(runtime)
                created = adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0)
                session = adapter._sessions[created.session_oid]
                session.stop_event.set()
                adapter._join_session_workers(session, timeout_s=1.0)
                provider.sessions[0].alive = False
                original_emit = adapter.events.emit

                def fail_exit_event(event_type: Any, **kwargs: Any) -> Any:
                    if (kwargs.get("payload") or {}).get("operation") == "exit":
                        raise RuntimeError("exit event sink failed")
                    return original_emit(event_type, **kwargs)

                monkeypatch.setattr(adapter.events, "emit", fail_exit_event)

                with pytest.raises(RuntimeError, match="exit event sink failed"):
                    adapter._mark_session_exited(session, resource="shell:git")

                close_effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == "close"
                ]
                assert len(close_effects) == 1
                assert close_effects[0].effect_state == "pending"
                assert close_effects[0].information_flow is True
            finally:
                runtime.close()

    def test_pty_auto_exit_close_pens_after_exit_code_finalizes_partial_effect(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty auto exit close PENS")
                adapter = _pty_adapter(runtime)
                created = adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0)
                session = adapter._sessions[created.session_oid]
                session.stop_event.set()
                adapter._join_session_workers(session, timeout_s=1.0)
                provider.sessions[0].alive = False
                original_close = provider.sessions[0].close

                def fail_close_not_started(*_args: Any, **_kwargs: Any) -> Any:
                    raise ProviderEffectNotStarted("auto close did not start")

                monkeypatch.setattr(provider.sessions[0], "close", fail_close_not_started)

                adapter._mark_session_exited(session, resource="shell:git")

                close_effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == "close"
                ]
                assert len(close_effects) == 1
                assert close_effects[0].effect_state == "finalized"
                assert close_effects[0].transaction_state == "committed"
                assert close_effects[0].provider_metadata["outcome"] == "partial_not_started_after_prior_provider_effect"
                assert close_effects[0].information_flow is True

                monkeypatch.setattr(provider.sessions[0], "close", original_close)
            finally:
                runtime.close()

    def test_pty_auto_exit_exit_code_pens_abandons_close_intent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty auto exit read PENS")
                adapter = _pty_adapter(runtime)
                created = adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0)
                session = adapter._sessions[created.session_oid]
                session.stop_event.set()
                adapter._join_session_workers(session, timeout_s=1.0)
                provider.sessions[0].alive = False
                original_exit_code = provider.sessions[0].exit_code

                def fail_exit_code_not_started() -> int | None:
                    raise ProviderEffectNotStarted("exit-code read did not start")

                monkeypatch.setattr(provider.sessions[0], "exit_code", fail_exit_code_not_started)

                adapter._mark_session_exited(session, resource="shell:git")

                assert [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == "close"
                ] == []
                assert session.closed is False
                assert session.closing is False

                monkeypatch.setattr(provider.sessions[0], "exit_code", original_exit_code)
            finally:
                runtime.close()

    def test_pty_auto_exit_exit_code_ambiguous_failure_finalizes_unknown(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty auto exit read unknown")
                adapter = _pty_adapter(runtime)
                created = adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0)
                session = adapter._sessions[created.session_oid]
                session.stop_event.set()
                adapter._join_session_workers(session, timeout_s=1.0)
                provider.sessions[0].alive = False
                original_exit_code = provider.sessions[0].exit_code

                def fail_exit_code_ambiguously() -> int | None:
                    raise RuntimeError("exit-code outcome unknown")

                monkeypatch.setattr(provider.sessions[0], "exit_code", fail_exit_code_ambiguously)

                adapter._mark_session_exited(session, resource="shell:git")

                close_effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == "close"
                ]
                assert len(close_effects) == 1
                assert close_effects[0].effect_state == "finalized"
                assert close_effects[0].transaction_state == "unknown"
                assert close_effects[0].provider_metadata["outcome"] == "unknown_after_provider_exception"
                assert close_effects[0].rollback_class == ExternalEffectRollbackClass.UNKNOWN
                assert session.closed is False
                assert session.closing is False

                monkeypatch.setattr(provider.sessions[0], "exit_code", original_exit_code)
            finally:
                runtime.close()

    @pytest.mark.parametrize(
        "provider_mode",
        ["unsupported-operation", "classifier-exception"],
    )
    def test_pty_mutation_classifier_failure_records_unknown_effects(
        self,
        provider_mode: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = (
                SpawnOnlyClassifierPtyProvider(initial_outputs=[])
                if provider_mode == "unsupported-operation"
                else ClassifierFailurePtyProvider(initial_outputs=[])
            )
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty classifier fallback")
                adapter = _pty_adapter(runtime)
                created = adapter.create(pid, ["git", "status"], cwd=".", startup_timeout_s=0)

                adapter.write(pid, created.session_oid, "hello\n")
                adapter.resize(pid, created.session_oid, cols=100, rows=30)
                adapter.close(pid, created.session_oid)

                effects = {
                    effect.operation: effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation in {"write", "resize", "close"}
                }
                assert set(effects) == {"write", "resize", "close"}
                for effect in effects.values():
                    assert effect.rollback_class == ExternalEffectRollbackClass.UNKNOWN
                    assert effect.rollback_status == ExternalEffectRollbackStatus.UNKNOWN
                    assert effect.provider_metadata["classification_fallback"] == "post_effect_failure"
                    assert "classification_error_type" in effect.provider_metadata
                    assert "classification_error" not in effect.provider_metadata
            finally:
                runtime.close()

    def test_pty_classifier_failure_keeps_started_session_and_records_conservative_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = ClassifierFailurePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty classifier failure")

                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": ["git", "status"], "startup_timeout_s": 0},
                )

                assert created.ok, created.error
                assert provider.sessions[0].closed is False
                effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.operation == "spawn"
                ]
                assert len(effects) == 1
                assert effects[0].rollback_class == ExternalEffectRollbackClass.UNKNOWN
                assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
                assert effects[0].provider_metadata["classification_fallback"] == "post_effect_failure"
            finally:
                runtime.close()

    def test_high_risk_pty_spawn_requests_human_with_continuous_session_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="ask for risky pty")

                with pytest.raises(HumanApprovalRequired):
                    runtime.tools.call(pid, "pty_create", {"argv": ["python", "-c", "print(1)"]})

                pending = runtime.human.pending()[0]
                assert pending.status == HumanRequestStatus.PENDING
                assert pending.payload["context"]["operation"] == "pty.spawn"
                assert pending.payload["context"]["continuous_session"] is True
                assert pending.payload["context"]["argv"] == ["python", "-c", "print(1)"]
                rule = pending.payload["requested_once_capability"]["constraints"]["authority_rules"][0]
                assert rule["operation"] == "pty.spawn"
                assert rule["conditions"]["continuous_session"] is True
                assert "timeout_s" not in rule["conditions"]
                assert provider.spawned == []
            finally:
                runtime.close()

    def test_pty_spawn_honors_exact_authority_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="exact pty authority")
                argv = ["python3", "-c", "print(1)"]
                runtime.capability.issue_trusted(
                    pid,
                    "shell:python3",
                    [CapabilityRight.EXECUTE],
                    issued_by="test",
                    constraints={
                        AUTHORITY_RULES_KEY: [
                            {
                                "rule_id": "test.pty.spawn.exact",
                                "operation": "pty.spawn",
                                "effect": "allow",
                                "risk": "high",
                                "conditions": {
                                    "argv": argv,
                                    "match": "exact",
                                    "cwd": ".",
                                    "continuous_session": True,
                                },
                            }
                        ]
                    },
                )

                created = runtime.tools.call(pid, "pty_create", {"argv": argv, "startup_timeout_s": 0})

                assert created.ok, created.error
                assert provider.spawned[0]["argv"] == argv
            finally:
                runtime.close()

    def test_pty_spawn_does_not_reuse_shell_run_timeout_scoped_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                runtime.register_image(
                    AgentImage(
                        image_id="pty-direct-shell-run-only:v0",
                        name="pty-direct-shell-run-only",
                        default_tools=["pty_create"],
                    ),
                    actor="cli",
                )
                pid = runtime.process.spawn(image="pty-direct-shell-run-only:v0", goal="timeout-scoped pty")
                argv = ["sh", "-c", "sleep 3600"]
                runtime.capability.issue_trusted(
                    pid,
                    "shell:sh",
                    [CapabilityRight.EXECUTE],
                    issued_by="test",
                    constraints={
                        "authority_rules": [
                            {
                                "rule_id": "test.shell.run.short.only",
                                "operation": "shell.run",
                                "effect": "allow",
                                "risk": "high",
                                "conditions": {"argv": argv, "match": "exact", "timeout_max_s": 0.001},
                            }
                        ]
                    },
                )

                result = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": argv, "startup_timeout_s": 0, "max_output_chars": 1},
                )

                assert not result.ok
                assert provider.spawned == []
            finally:
                runtime.close()

    def test_pty_spawn_honors_bare_shell_deny_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="bare deny pty")
                runtime.capability.issue_trusted(
                    pid,
                    "shell:git",
                    [CapabilityRight.EXECUTE],
                    issued_by="test",
                    effect="deny",
                )

                result = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})

                assert not result.ok
                assert "explicit command capability denied" in (result.error or "")
                assert provider.spawned == []
            finally:
                runtime.close()

    def test_shell_once_approval_does_not_authorize_pty_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="shell approval is not pty approval")
                argv = ["python", "-c", "print(1)"]

                with pytest.raises(HumanApprovalRequired):
                    runtime.shell.run(pid, argv)
                runtime.human.drain_terminal_queue(auto_approve=True)
                shell_approval_caps = [
                    cap
                    for cap in runtime.capability.capabilities_for(pid)
                    if cap.resource == runtime.shell.resource_for(argv)
                    and cap.uses_remaining == 1
                    and any(rule.get("operation") == "shell.run" for rule in cap.constraints.get(AUTHORITY_RULES_KEY, []))
                ]
                assert shell_approval_caps

                with pytest.raises(HumanApprovalRequired):
                    runtime.tools.call(pid, "pty_create", {"argv": argv, "startup_timeout_s": 0})

                assert provider.spawned == []
                pending = runtime.human.pending()[0]
                assert pending.payload["context"]["operation"] == "pty.spawn"
                assert pending.payload["context"]["continuous_session"] is True
                refreshed = {cap.cap_id: cap for cap in runtime.capability.capabilities_for(pid)}
                assert refreshed[shell_approval_caps[0].cap_id].uses_remaining == 1
            finally:
                runtime.close()

    def test_other_process_without_object_capability_cannot_operate_pty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                owner = runtime.process.spawn(image="pty-agent:v0", goal="owner")
                other = runtime.process.spawn(image="pty-agent:v0", goal="other")
                created = runtime.tools.call(owner, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]

                read = runtime.tools.call(other, "pty_read", {"session_oid": session_oid})
                write = runtime.tools.call(other, "pty_write", {"session_oid": session_oid, "text": "x"})
                closed = runtime.tools.call(other, "pty_close", {"session_oid": session_oid})

                assert not read.ok
                assert not write.ok
                assert not closed.ok
                assert provider.sessions[0].closed is False
            finally:
                runtime.close()

    def test_delegated_object_write_cannot_drive_existing_pty_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                owner = runtime.process.spawn(image="pty-agent:v0", goal="owner")
                other = runtime.process.spawn(image="pty-agent:v0", goal="delegated writer")
                created = runtime.tools.call(owner, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]
                runtime.capability.grant(other, f"object:{session_oid}", [CapabilityRight.WRITE], issued_by="test")

                write = runtime.tools.call(other, "pty_write", {"session_oid": session_oid, "text": "whoami\n"})

                assert not write.ok
                assert "cannot write to PTY session owned by" in (write.error or "")
                assert provider.sessions[0].writes == []
            finally:
                runtime.close()

    def test_pty_spawn_rejects_workspace_path_hijack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_tool = root / "git"
            fake_tool.write_text("#!/bin/sh\necho hijacked\n", encoding="utf-8")
            fake_tool.chmod(0o755)
            monkeypatch.setenv("PATH", str(root))
            runtime = _open_pty_runtime(temp_dir, LocalPtyProvider(root))
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="path hijack")
                runtime.shell.grant_policy(pid, runtime.config.shell.always_allow_level, issued_by="test")

                with pytest.raises(FileNotFoundError, match="safe PATH"):
                    _pty_adapter(runtime).create(
                        pid,
                        ["git", "status"],
                        cwd=".",
                        startup_timeout_s=0,
                        max_output_chars=1,
                    )

            finally:
                runtime.close()

    def test_pty_provider_without_limits_support_fails_closed_when_budgeted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = NoLimitsPtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="pty limits required",
                    resource_budget=ResourceBudget(max_subprocess_wall_seconds=1.0),
                )

                result = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})

                assert not result.ok
                assert "SubprocessLimits" in (result.error or "")
                assert provider.spawned == []
            finally:
                runtime.close()

    def test_pty_provider_that_supports_limits_receives_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="pty passes limits",
                    resource_budget=ResourceBudget(max_subprocess_wall_seconds=1.0),
                )

                created = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})

                assert created.ok, created.error
                assert isinstance(provider.spawned[0]["limits"], SubprocessLimits)
            finally:
                runtime.close()

    def test_windows_local_pty_provider_fails_closed_for_budgeted_spawn(self) -> None:
        if os.name != "nt":
            pytest.skip("Windows PTY limit enforcement is platform-specific")
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalPtyProvider(temp_dir)
            with pytest.raises(ValidationError, match="SubprocessLimits"):
                provider.spawn([os.fspath(Path(temp_dir) / "unused.exe")], limits=SubprocessLimits(wall_seconds=1.0))

    def test_posix_pty_exit_cleanup_kills_background_descendant(self) -> None:
        if os.name == "nt":
            pytest.skip("POSIX process-group cleanup is platform-specific")
        marker = f"PTY_EXIT_CLEANUP_{time.monotonic_ns()}"
        child_script = (
            "import pathlib, sys, time; "
            "time.sleep(0.5); "
            "pathlib.Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            sentinel = Path(temp_dir) / "sentinel.txt"
            parent_script = (
                "import subprocess, sys; "
                "subprocess.Popen("
                f"[sys.executable, '-c', {child_script!r}, {str(sentinel)!r}, {marker!r}], "
                "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
                "print('parent-exit')"
            )
            runtime = _open_pty_runtime(temp_dir, LocalPtyProvider(temp_dir))
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty exit cleanup")
                argv = ["python3", "-c", parent_script]
                runtime.capability.issue_trusted(
                    pid,
                    "shell:python3",
                    [CapabilityRight.EXECUTE],
                    issued_by="test",
                    constraints={
                        AUTHORITY_RULES_KEY: [
                            {
                                "rule_id": "test.pty.exit-cleanup.spawn",
                                "operation": "pty.spawn",
                                "effect": "allow",
                                "risk": "high",
                                "conditions": {
                                    "argv": argv,
                                    "match": "exact",
                                    "cwd": ".",
                                    "continuous_session": True,
                                },
                                "description": "Allow the exact PTY cleanup regression spawn.",
                            }
                        ]
                    },
                )
                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": argv, "startup_timeout_s": 0.2},
                )
                assert created.ok, created.error

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and not sentinel.exists():
                    time.sleep(0.05)
                assert not sentinel.exists()
            finally:
                runtime.close()

    def test_pty_exit_cleanup_closes_provider_forcefully(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[], session_alive=False)
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty force exit cleanup")
                created = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and not provider.sessions[0].close_forces:
                    time.sleep(0.01)

                assert provider.sessions[0].close_forces == [True]
            finally:
                runtime.close()

    def test_pty_input_limit_fails_closed_before_provider_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(
                temp_dir,
                provider,
                settings={"input_max_chars": 3, "input_hard_limit_chars": 10},
            )
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="input limit")
                created = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error

                written = runtime.tools.call(
                    pid,
                    "pty_write",
                    {"session_oid": created.payload["session_oid"], "text": "abcd"},
                )

                assert not written.ok
                assert "configured limit" in (written.error or "")
                assert provider.sessions[0].writes == []
            finally:
                runtime.close()

    def test_pty_session_limit_fails_closed_before_provider_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider, settings={"max_sessions_per_process": 1})
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="session limit")
                first = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                second = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})

                assert first.ok, first.error
                assert not second.ok
                assert "per-process limit" in (second.error or "")
                assert len(provider.spawned) == 1
            finally:
                runtime.close()

    def test_exited_pty_session_does_not_consume_process_session_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[], session_alive=False)
            runtime = _open_pty_runtime(temp_dir, provider, settings={"max_sessions_per_process": 1})
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="exited pty capacity")
                first = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert first.ok, first.error
                first_oid = first.payload["session_oid"]

                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    session = _pty_adapter(runtime)._sessions.get(first_oid)
                    if session is None or session.closed:
                        break
                    time.sleep(0.01)

                assert first_oid not in _pty_adapter(runtime)._sessions
                second = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})

                assert second.ok, second.error
                assert len(provider.spawned) == 2
            finally:
                runtime.close()

    def test_concurrent_pty_create_reserves_capacity_before_provider_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(spawn_delay_s=0.1)
            runtime = _open_pty_runtime(temp_dir, provider, settings={"max_sessions_per_process": 1})
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="concurrent session limit")
                adapter = _pty_adapter(runtime)
                cwd = runtime.process.working_directory(pid)
                barrier = threading.Barrier(2)
                created: list[Any] = []
                errors: list[Exception] = []

                def create_session() -> None:
                    barrier.wait(timeout=2)
                    try:
                        created.append(adapter.create(pid, ["git", "status"], cwd=cwd, startup_timeout_s=0))
                    except Exception as exc:
                        errors.append(exc)

                threads = [threading.Thread(target=create_session) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=2)

                assert len(created) == 1
                assert len(errors) == 1
                assert "per-process limit" in str(errors[0])
                assert len(provider.spawned) == 1
            finally:
                runtime.close()

    def test_pty_close_unknown_keeps_session_registered_and_blocks_blind_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(close_failures=1)
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="retry close")
                created = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]

                failed = runtime.tools.call(pid, "pty_close", {"session_oid": session_oid})

                assert not failed.ok
                assert runtime.store.get_object(session_oid) is not None
                assert [entry.session_oid for entry in _pty_adapter(runtime).list(pid)] == [session_oid]
                assert provider.sessions[0].closed is False

                retried = runtime.tools.call(pid, "pty_close", {"session_oid": session_oid})

                assert not retried.ok
                assert "unresolved prior close outcome" in (retried.error or "")
                assert provider.sessions[0].closed is False
                assert runtime.store.get_object(session_oid) is not None
            finally:
                runtime.close()

    def test_pty_wall_time_budget_uses_monotonic_session_elapsed_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[], session_pid=os.getpid())
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="pty wall budget",
                    resource_budget=ResourceBudget(max_subprocess_wall_seconds=10.0),
                )
                argv = ["pty-test-command"]
                _grant_exact_pty_once(runtime, pid, argv)
                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": argv, "startup_timeout_s": 0},
                )
                assert created.ok, created.error
                adapter = _pty_adapter(runtime)
                session = adapter._sessions[created.payload["session_oid"]]
                session.started_monotonic = time.monotonic() - 11.0

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and (
                    runtime.process.get(pid).status.value != "killed"
                    or not provider.sessions[0].closed
                ):
                    time.sleep(0.01)

                process = runtime.process.get(pid)
                assert process.status.value == "killed"
                assert process.resource_usage.subprocess_wall_seconds > 0
                assert provider.sessions[0].closed
            finally:
                runtime.close()

    def test_pty_wall_overage_is_charged_before_sampler_access_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[], session_pid=None)
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="pty wall budget before sampler failure",
                    resource_budget=ResourceBudget(max_subprocess_wall_seconds=10.0),
                )
                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": ["git", "status"], "startup_timeout_s": 0},
                )
                assert created.ok, created.error
                adapter = _pty_adapter(runtime)
                session = adapter._sessions[created.payload["session_oid"]]
                session.handle.pid = os.getpid()
                session.started_monotonic = time.monotonic() - 11.0

                def deny_process_access(process_pid: int) -> Any:
                    raise psutil.AccessDenied(pid=process_pid)

                monkeypatch.setattr("modules.pty.pty_module.psutil.Process", deny_process_access)

                adapter._sample_and_charge(session, "shell:git")

                process = runtime.process.get(pid)
                assert process.status.value == "killed"
                assert process.resource_usage.subprocess_wall_seconds > 0
                assert provider.sessions[0].closed
                actions = [record.action for record in runtime.audit.trace()]
                assert "primitive.pty.resource_limit_exceeded" in actions
                assert "primitive.pty.resource_monitor_denied" not in actions
            finally:
                runtime.close()

    def test_pty_resource_monitor_is_independent_from_blocked_output_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = BlockingReadPtyProvider()
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="pty blocked reader budget",
                    resource_budget=ResourceBudget(max_subprocess_wall_seconds=10.0),
                )
                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": ["git", "status"], "startup_timeout_s": 0},
                )
                assert created.ok, created.error
                assert provider.sessions[0].read_started.wait(timeout=1.0)
                session = _pty_adapter(runtime)._sessions[
                    created.payload["session_oid"]
                ]
                session.started_monotonic = time.monotonic() - 11.0

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and (
                    runtime.process.get(pid).status.value != "killed"
                    or not provider.sessions[0].closed
                ):
                    time.sleep(0.01)

                assert runtime.process.get(pid).status.value == "killed"
                assert runtime.process.get(pid).resource_usage.subprocess_wall_seconds > 0
                assert provider.sessions[0].closed
                assert provider.sessions[0].read_returned.wait(timeout=1.0)
            finally:
                provider.release_read.set()
                runtime.close()

    def test_pty_cpu_accounting_keeps_exited_child_contribution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = _open_pty_runtime(temp_dir, FakePtyProvider(initial_outputs=[]))
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty cumulative cpu")
                handle = FakePtySession(cols=80, rows=24, outputs=[], pid=4242)
                session = _PtyRuntimeSession(
                    session_oid="pty_cpu_accounting",
                    session_id="pty_cpu_accounting",
                    owner_pid=pid,
                    argv=["git", "status"],
                    cwd=".",
                    backend=handle.backend,
                    handle=handle,
                    cols=80,
                    rows=24,
                    started_at="test",
                    started_monotonic=time.monotonic(),
                    buffer_max_chars=100,
                    data_sink=DataSink(identity="pty:spawn:/usr/bin/git"),
                    data_flow_context=DataFlowContext(),
                )
                child = SequencedPsutilProcess(pid=4243, cpu_values=[0.5], children=[])
                root = SequencedPsutilProcess(
                    pid=4242,
                    cpu_values=[0.1, 0.3],
                    children=[[child], []],
                )
                monkeypatch.setattr("modules.pty.pty_module.psutil.Process", lambda _pid: root)

                _pty_adapter(runtime)._sample_and_charge(session, "shell:git")
                _pty_adapter(runtime)._sample_and_charge(session, "shell:git")

                usage = runtime.process.get(pid).resource_usage
                assert usage.subprocess_cpu_seconds == pytest.approx(0.8)
            finally:
                runtime.close()

    def test_concurrent_pty_close_waits_and_finalizes_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = CoordinatedClosePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            provider.release_close.clear()
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="concurrent pty close")
                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": ["git", "status"], "startup_timeout_s": 0},
                )
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]
                adapter = _pty_adapter(runtime)
                results: list[Any] = []
                errors: list[BaseException] = []

                def close_session() -> None:
                    try:
                        results.append(adapter.close(pid, session_oid, timeout_s=1.0))
                    except BaseException as exc:
                        errors.append(exc)

                first = threading.Thread(target=close_session, name="test-pty-close-first")
                second = threading.Thread(target=close_session, name="test-pty-close-second")
                first.start()
                assert provider.sessions[0].close_started.wait(timeout=1.0)
                second.start()
                time.sleep(0.05)
                provider.release_close.set()
                first.join(timeout=2.0)
                second.join(timeout=2.0)

                assert not first.is_alive()
                assert not second.is_alive()
                assert errors == []
                assert len(results) == 2
                assert provider.sessions[0].close_calls == 1
                assert adapter._sessions == {}
                assert runtime.store.get_object(session_oid) is None
            finally:
                provider.release_close.set()
                runtime.close()

    def test_pty_resource_monitor_access_denied_closes_session_fail_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[], session_pid=os.getpid())
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="pty monitor denied")
                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": ["git", "status"], "startup_timeout_s": 0},
                )
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]

                def deny_process_access(process_pid: int) -> Any:
                    raise psutil.AccessDenied(pid=process_pid)

                monkeypatch.setattr("modules.pty.pty_module.psutil.Process", deny_process_access)
                adapter = _pty_adapter(runtime)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and session_oid in adapter._sessions:
                    time.sleep(0.01)

                # Session removal happens inside the close settlement hook, while
                # the monitor-denied audit is written immediately afterwards in
                # the worker's finally block.  Wait for the worker boundary so
                # this assertion cannot observe that intentional intermediate
                # state under slower CI scheduling.
                assert adapter._wait_for_worker_threads(timeout_s=2.0)
                assert provider.sessions[0].closed
                assert session_oid not in adapter._sessions
                assert runtime.store.get_object(session_oid) is None
                assert "primitive.pty.resource_monitor_denied" in [
                    record.action for record in runtime.audit.trace()
                ]
            finally:
                runtime.close()

    def test_shutdown_waits_for_removed_pty_reader_before_closing_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[], session_pid=os.getpid())
            runtime = _open_pty_runtime(temp_dir, provider)
            release_blocked_audit = threading.Event()
            audit_blocked = threading.Event()
            close_called = threading.Event()
            shutdown_thread: threading.Thread | None = None
            try:
                original_record = runtime.audit.record

                def record_spy(*args: Any, **kwargs: Any) -> Any:
                    action = kwargs.get("action")
                    if action is None and len(args) >= 2:
                        action = args[1]
                    if action == "primitive.pty.resource_limit_exceeded":
                        audit_blocked.set()
                        release_blocked_audit.wait(timeout=2.0)
                    return original_record(*args, **kwargs)

                original_close = runtime.store.close

                def close_spy() -> None:
                    close_called.set()
                    original_close()

                monkeypatch.setattr(runtime.audit, "record", record_spy)
                monkeypatch.setattr(runtime.store, "close", close_spy)

                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="pty shutdown waits for reader",
                    resource_budget=ResourceBudget(max_subprocess_wall_seconds=10.0),
                )
                created = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error
                session = _pty_adapter(runtime)._sessions[
                    created.payload["session_oid"]
                ]
                session.started_monotonic = time.monotonic() - 11.0
                assert audit_blocked.wait(timeout=2.0)
                assert provider.sessions[0].closed
                assert not _pty_adapter(runtime)._sessions

                shutdown_result: dict[str, Any] = {}

                def shutdown_runtime() -> None:
                    try:
                        shutdown_result["value"] = runtime.shutdown(actor="test", reason="blocked_pty_reader")
                    except Exception as exc:
                        shutdown_result["error"] = exc

                shutdown_thread = threading.Thread(target=shutdown_runtime, name="test-pty-shutdown")
                shutdown_thread.start()
                time.sleep(0.1)

                assert shutdown_thread.is_alive()
                assert not close_called.is_set()

                release_blocked_audit.set()
                shutdown_thread.join(timeout=2.0)

                assert not shutdown_thread.is_alive()
                assert "error" not in shutdown_result
                assert shutdown_result["value"]["ok"] is True
                assert close_called.is_set()
            finally:
                release_blocked_audit.set()
                if shutdown_thread is not None and shutdown_thread.is_alive():
                    shutdown_thread.join(timeout=2.0)
                if not runtime.lifecycle.closed:
                    runtime.close()

    def test_recovery_handoff_abandons_pty_workers_without_durable_writes_and_reopens(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "pty-recovery-handoff.sqlite"
            provider = FakePtyProvider(
                initial_outputs=[],
                session_pid=os.getpid(),
            )
            runtime = _open_pty_runtime(
                temp_dir,
                provider,
                target=database,
            )
            adapter = _pty_adapter(runtime)
            monitor_started = threading.Event()
            finalizer_calls: list[str] = []
            close_snapshots: list[tuple[Any, ...]] = []
            reopened: Runtime | None = None
            released = False

            def quiet_monitor(
                session: _PtyRuntimeSession,
                _resource: str,
            ) -> None:
                monitor_started.set()
                session.stop_event.wait(timeout=1.0)

            def ordinary_finalizer() -> None:
                finalizer_calls.append("ordinary")

            monkeypatch.setattr(adapter, "_sample_and_charge", quiet_monitor)
            runtime.bind_shutdown_finalizer(ordinary_finalizer)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="recovery handoff closes PTY transients",
                )
                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": ["git", "status"], "startup_timeout_s": 0},
                )
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]
                session = adapter._sessions[session_oid]
                assert provider.sessions[0].read_started.wait(timeout=1.0)
                assert monitor_started.wait(timeout=1.0)
                assert session.reader_thread is not None
                assert session.reader_thread.is_alive()
                assert session.monitor_thread is not None
                assert session.monitor_thread.is_alive()

                with runtime.lifecycle.admit():
                    runtime.lifecycle.mark_recovery_required(
                        publication_id="pty-recovery-handoff",
                    )
                before = _durable_pty_snapshot(runtime)
                original_close = runtime.store.close

                def observed_close() -> None:
                    close_snapshots.append(_durable_pty_snapshot(runtime))
                    original_close()

                monkeypatch.setattr(runtime.store, "close", observed_close)

                outcome = runtime.release_recovery_diagnostics()
                released = outcome["ok"] is True

                assert released
                assert outcome["recovery_diagnostics_released"] is True
                assert provider.sessions[0].closed
                assert not session.reader_thread.is_alive()
                assert not session.monitor_thread.is_alive()
                assert adapter._sessions == {}
                assert close_snapshots == [before]
                assert finalizer_calls == []

                reopened = _open_pty_runtime(
                    temp_dir,
                    FakePtyProvider(initial_outputs=[]),
                    target=database,
                )
                assert reopened.lifecycle.state == "open"
                assert reopened.store.get_object(session_oid) is None
            finally:
                if not released:
                    runtime.store.close()
                if reopened is not None:
                    reopened.close()

    def test_pty_recovery_cleanup_rejects_direct_call_without_lifecycle_lease(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="reject direct PTY recovery cleanup",
                )
                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": ["git", "status"], "startup_timeout_s": 0},
                )
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]
                adapter = _pty_adapter(runtime)
                session = adapter._sessions[session_oid]
                assert provider.sessions[0].read_started.wait(timeout=1.0)
                before = _durable_pty_snapshot(runtime)

                with pytest.raises(
                    RuntimeError,
                    match="recovery cleanup lease",
                ):
                    adapter.release_recovery_diagnostics()

                assert adapter._sessions[session_oid] is session
                assert not session.recovery_abandoning
                assert not session.stop_event.is_set()
                assert not provider.sessions[0].closed
                assert _durable_pty_snapshot(runtime) == before
            finally:
                runtime.close()

    def test_recovery_handoff_keeps_failed_pty_cleanup_retryable_and_store_open(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "pty-recovery-handoff-retry.sqlite"
            provider = FakePtyProvider(initial_outputs=[], close_failures=1)
            runtime = _open_pty_runtime(
                temp_dir,
                provider,
                target=database,
            )
            adapter = _pty_adapter(runtime)
            close_snapshots: list[tuple[Any, ...]] = []
            released = False
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="retry failed recovery PTY cleanup",
                )
                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": ["git", "status"], "startup_timeout_s": 0},
                )
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]
                session = adapter._sessions[session_oid]
                assert provider.sessions[0].read_started.wait(timeout=1.0)

                with runtime.lifecycle.admit():
                    runtime.lifecycle.mark_recovery_required(
                        publication_id="pty-recovery-handoff-retry",
                    )
                before = _durable_pty_snapshot(runtime)
                original_close = runtime.store.close

                def observed_close() -> None:
                    close_snapshots.append(_durable_pty_snapshot(runtime))
                    original_close()

                monkeypatch.setattr(runtime.store, "close", observed_close)

                first = runtime.release_recovery_diagnostics()

                assert first["ok"] is False
                assert runtime.lifecycle.state == "close_failed"
                assert runtime.store.get_object(session_oid) is not None
                assert adapter._sessions[session_oid] is session
                assert not provider.sessions[0].closed
                assert close_snapshots == []
                assert _durable_pty_snapshot(runtime) == before

                second = runtime.release_recovery_diagnostics()
                released = second["ok"] is True

                assert released
                assert provider.sessions[0].closed
                assert adapter._sessions == {}
                assert close_snapshots == [before]
            finally:
                if not released:
                    runtime.store.close()

    def test_recovery_handoff_base_exception_keeps_pty_cleanup_retryable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "pty-recovery-handoff-interrupt.sqlite"
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(
                temp_dir,
                provider,
                target=database,
            )
            adapter = _pty_adapter(runtime)
            close_called = threading.Event()
            released = False
            try:
                pid = runtime.process.spawn(
                    image="pty-agent:v0",
                    goal="retry interrupted recovery PTY cleanup",
                )
                created = runtime.tools.call(
                    pid,
                    "pty_create",
                    {"argv": ["git", "status"], "startup_timeout_s": 0},
                )
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]
                session = adapter._sessions[session_oid]
                assert provider.sessions[0].read_started.wait(timeout=1.0)
                original_session_close = session.handle.close
                close_attempts = 0

                def interrupt_first_close(
                    *,
                    force: bool = True,
                    timeout_s: float = 2.0,
                ) -> int | None:
                    nonlocal close_attempts
                    close_attempts += 1
                    if close_attempts == 1:
                        raise KeyboardInterrupt("interrupt recovery PTY close")
                    return original_session_close(
                        force=force,
                        timeout_s=timeout_s,
                    )

                monkeypatch.setattr(session.handle, "close", interrupt_first_close)
                original_store_close = runtime.store.close

                def observed_store_close() -> None:
                    close_called.set()
                    original_store_close()

                monkeypatch.setattr(runtime.store, "close", observed_store_close)
                with runtime.lifecycle.admit():
                    runtime.lifecycle.mark_recovery_required(
                        publication_id="pty-recovery-handoff-interrupt",
                    )
                before = _durable_pty_snapshot(runtime)

                with pytest.raises(
                    KeyboardInterrupt,
                    match="interrupt recovery PTY close",
                ):
                    runtime.release_recovery_diagnostics()

                assert runtime.lifecycle.state == "close_failed"
                assert adapter._sessions[session_oid] is session
                assert not close_called.is_set()
                assert _durable_pty_snapshot(runtime) == before

                retry = runtime.release_recovery_diagnostics()
                released = retry["ok"] is True

                assert released
                assert provider.sessions[0].closed
                assert adapter._sessions == {}
                assert close_called.is_set()
            finally:
                if not released:
                    runtime.store.close()

    def test_pty_reader_drops_old_output_when_buffer_limit_is_reached(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakePtyProvider(initial_outputs=[])
            runtime = _open_pty_runtime(temp_dir, provider, settings={"buffer_max_chars": 6})
            try:
                pid = runtime.process.spawn(image="pty-agent:v0", goal="buffer limit")
                created = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]

                provider.sessions[0].outputs.extend(["abcdef", "ghij"])
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    entries = _pty_adapter(runtime).list(pid)
                    if entries and entries[0].dropped_chars:
                        break
                    time.sleep(0.01)

                read = runtime.tools.call(pid, "pty_read", {"session_oid": session_oid})
                assert read.ok, read.error
                assert read.payload["output"] == "ghij"
                assert read.payload["dropped_chars"] == 6
            finally:
                runtime.close()


def _open_pty_runtime(
    root: str,
    provider: "FakePtyProvider",
    *,
    settings: dict[str, Any] | None = None,
    target: str | Path = "local",
) -> Runtime:
    substrate = LocalResourceProviderSubstrate(root)
    substrate.pty = provider
    if settings is not None:
        substrate.pty_settings = settings
    manifest = _module_manifest()
    source_sha = next(
        line.split(":", 1)[1].strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.startswith("sha256:")
    )
    manifest_sha = hashlib.sha256(manifest.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    runtime = Runtime.open(
        target,
        substrate=substrate,
        config=AgentLibOSConfig(),
        module_manifests=(str(manifest),),
        trusted_modules=(f"agent-libos-pty:v0:{manifest_sha}:{source_sha}",),
    )
    # This contract harness is the Host launch policy for the bundled image:
    # it explicitly authorizes the declarations instead of enabling the
    # removed 0.2 image-auto-grant compatibility mode.
    original_spawn = runtime.process.spawn

    def host_authorized_spawn(*args: Any, **kwargs: Any) -> str:
        selected_image = kwargs.get("image", args[0] if args else None)
        if (
            selected_image == "pty-agent:v0"
            and "capabilities" not in kwargs
            and "authority_manifest" not in kwargs
        ):
            kwargs["capabilities"] = list(
                runtime.get_image("pty-agent:v0").required_capabilities
            )
        return original_spawn(*args, **kwargs)

    runtime.process.spawn = host_authorized_spawn
    return runtime


def _module_manifest() -> Path:
    manifest = Path("modules/pty/module.yaml").resolve()
    shutil.rmtree(manifest.parent / "__pycache__", ignore_errors=True)
    return manifest


def _pty_adapter(runtime: Runtime) -> Any:
    return runtime.module_state.get("_agent_libos_pty_adapter")


def _durable_pty_snapshot(runtime: Runtime) -> tuple[Any, ...]:
    return (
        tuple(runtime.store.list_objects()),
        tuple(runtime.store.list_audit()),
        tuple(runtime.store.list_events()),
        tuple(runtime.store.list_external_effects()),
        tuple(runtime.store.list_operations()),
    )


def _grant_exact_pty_once(runtime: Runtime, pid: str, argv: list[str]) -> Any:
    return runtime.capability.issue_trusted(
        pid,
        runtime.shell.resource_for(argv),
        [CapabilityRight.EXECUTE],
        issued_by="test",
        constraints={
            AUTHORITY_RULES_KEY: [
                {
                    "rule_id": "test.pty.once.exact",
                    "operation": "pty.spawn",
                    "effect": "allow",
                    "risk": "medium",
                    "conditions": {
                        "argv": list(argv),
                        "match": "exact",
                        "cwd": ".",
                        "continuous_session": True,
                    },
                }
            ]
        },
        uses_remaining=1,
    )


def _grant_pty_object_once(
    runtime: Runtime,
    pid: str,
    session_oid: str,
    right: CapabilityRight,
) -> Any:
    return runtime.capability.issue_trusted(
        pid,
        f"object:{session_oid}",
        [right],
        issued_by="test.pty.object.once",
        uses_remaining=1,
    )


def _invoke_pty_mutation(adapter: Any, pid: str, session_oid: str, operation: str) -> Any:
    if operation == "write":
        return adapter.write(pid, session_oid, "hello\n")
    if operation == "resize":
        return adapter.resize(pid, session_oid, cols=100, rows=30)
    if operation == "close":
        return adapter.close(pid, session_oid)
    raise AssertionError(f"unsupported PTY mutation operation: {operation}")


class FakePtyProvider:
    supports_subprocess_limits = True

    def __init__(
        self,
        *,
        initial_outputs: list[str] | None = None,
        close_failures: int = 0,
        session_pid: int | None = None,
        session_alive: bool = True,
        spawn_delay_s: float = 0.0,
    ) -> None:
        self.initial_outputs = list(["ready\n"] if initial_outputs is None else initial_outputs)
        self.close_failures = close_failures
        self.session_pid = session_pid
        self.session_alive = session_alive
        self.spawn_delay_s = spawn_delay_s
        self.spawned: list[dict[str, Any]] = []
        self.sessions: list[FakePtySession] = []

    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        cols: int = 80,
        rows: int = 24,
        limits: SubprocessLimits | None = None,
    ) -> "FakePtySession":
        if self.spawn_delay_s > 0:
            time.sleep(self.spawn_delay_s)
        self.spawned.append({"argv": list(argv), "cwd": cwd, "cols": cols, "rows": rows, "limits": limits})
        session = FakePtySession(
            cols=cols,
            rows=rows,
            outputs=list(self.initial_outputs),
            close_failures=self.close_failures,
            pid=self.session_pid,
            alive=self.session_alive,
        )
        self.sessions.append(session)
        return session

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
            rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
            state_mutation=True,
            information_flow=True,
            metadata={"operation": operation, "backend": "fake-pty"},
        )


class ResolvingPtyProvider(FakePtyProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.resolver_calls: list[list[str]] = []

    def resolve_argv(self, argv: list[str], *, cwd: str | None = None) -> list[str]:
        self.resolver_calls.append(list(argv))
        return list(argv)


class NoLimitsPtyProvider(FakePtyProvider):
    supports_subprocess_limits = False


class SwitchingPtyProvider(ResolvingPtyProvider):
    def resolve_argv(self, argv: list[str], *, cwd: str | None = None) -> list[str]:
        self.resolver_calls.append(list(argv))
        return [str(Path(tempfile.gettempdir()) / "switched-pty-executable"), *argv[1:]]


class PreEffectFailurePtyProvider(FakePtyProvider):
    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        cols: int = 80,
        rows: int = 24,
        limits: SubprocessLimits | None = None,
    ) -> "FakePtySession":
        raise ProviderEffectNotStarted("provider failed before PTY spawn")


class AmbiguousFailurePtyProvider(FakePtyProvider):
    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        cols: int = 80,
        rows: int = 24,
        limits: SubprocessLimits | None = None,
    ) -> "FakePtySession":
        self.spawned.append({"argv": list(argv), "cwd": cwd, "cols": cols, "rows": rows, "limits": limits})
        raise TimeoutError("spawn outcome is unknown")


class SpawnOnlyClassifierPtyProvider(FakePtyProvider):
    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if operation != "spawn":
            raise ValueError(f"unsupported PTY classifier operation: {operation}")
        return super().classify_external_effect(operation, context, result)


class ClassifierFailurePtyProvider(FakePtyProvider):
    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        raise RuntimeError("simulated PTY classifier failure")


class BlockingReadPtyProvider(FakePtyProvider):
    def __init__(self) -> None:
        super().__init__(initial_outputs=[])
        self.release_read = threading.Event()

    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        cols: int = 80,
        rows: int = 24,
        limits: SubprocessLimits | None = None,
    ) -> "BlockingReadPtySession":
        self.spawned.append({"argv": list(argv), "cwd": cwd, "cols": cols, "rows": rows, "limits": limits})
        session = BlockingReadPtySession(
            cols=cols,
            rows=rows,
            outputs=[],
            pid=os.getpid(),
            release_read=self.release_read,
        )
        self.sessions.append(session)
        return session


class CoordinatedClosePtyProvider(FakePtyProvider):
    def __init__(self, *, initial_outputs: list[str] | None = None) -> None:
        super().__init__(initial_outputs=initial_outputs)
        self.release_close = threading.Event()

    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        cols: int = 80,
        rows: int = 24,
        limits: SubprocessLimits | None = None,
    ) -> "CoordinatedClosePtySession":
        self.spawned.append({"argv": list(argv), "cwd": cwd, "cols": cols, "rows": rows, "limits": limits})
        session = CoordinatedClosePtySession(
            cols=cols,
            rows=rows,
            outputs=list(self.initial_outputs),
            release_close=self.release_close,
        )
        self.sessions.append(session)
        return session


class FakePtySession:
    backend = "fake-pty"

    def __init__(
        self,
        *,
        cols: int,
        rows: int,
        outputs: list[str],
        close_failures: int = 0,
        pid: int | None = None,
        alive: bool = True,
    ) -> None:
        self.outputs = outputs
        self.writes: list[str] = []
        self.closed = False
        self.alive = alive
        self.size = (cols, rows)
        self.close_failures = close_failures
        self.pid = pid
        self.close_forces: list[bool] = []
        self.read_started = threading.Event()

    def read(self, *, timeout_s: float = 0.0) -> str:
        self.read_started.set()
        if self.outputs:
            return self.outputs.pop(0)
        if timeout_s > 0:
            time.sleep(min(timeout_s, 0.01))
        return ""

    def write(self, text: str) -> int:
        self.writes.append(text)
        self.outputs.append(f"echo:{text}")
        return len(text.encode("utf-8"))

    def resize(self, cols: int, rows: int) -> None:
        self.size = (cols, rows)

    def is_alive(self) -> bool:
        return self.alive and not self.closed

    def exit_code(self) -> int | None:
        return 0 if self.closed or not self.alive else None

    def close(self, *, force: bool = True, timeout_s: float = 2.0) -> int | None:
        self.close_forces.append(force)
        if self.close_failures > 0:
            self.close_failures -= 1
            raise RuntimeError("simulated close failure")
        self.closed = True
        self.alive = False
        return 0


class BlockingReadPtySession(FakePtySession):
    def __init__(self, *, release_read: threading.Event, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.release_read = release_read
        self.read_started = threading.Event()
        self.read_returned = threading.Event()

    def read(self, *, timeout_s: float = 0.0) -> str:
        self.read_started.set()
        self.release_read.wait(timeout=5.0)
        self.read_returned.set()
        return ""

    def close(self, *, force: bool = True, timeout_s: float = 2.0) -> int | None:
        self.release_read.set()
        return super().close(force=force, timeout_s=timeout_s)


class CoordinatedClosePtySession(FakePtySession):
    def __init__(self, *, release_close: threading.Event, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.release_close = release_close
        self.close_started = threading.Event()
        self.close_calls = 0

    def close(self, *, force: bool = True, timeout_s: float = 2.0) -> int | None:
        self.close_calls += 1
        self.close_started.set()
        if not self.release_close.wait(timeout=max(1.0, timeout_s)):
            raise TimeoutError("coordinated close was not released")
        return super().close(force=force, timeout_s=timeout_s)


class SequencedPsutilProcess:
    def __init__(
        self,
        *,
        pid: int,
        cpu_values: list[float],
        children: list[list["SequencedPsutilProcess"]],
        rss: int = 1,
    ) -> None:
        self.pid = pid
        self._cpu_values = list(cpu_values)
        self._children = list(children)
        self._cpu_index = 0
        self._children_index = 0
        self._rss = rss

    def create_time(self) -> float:
        return float(self.pid)

    def children(self, *, recursive: bool) -> list["SequencedPsutilProcess"]:
        assert recursive
        index = min(self._children_index, len(self._children) - 1)
        self._children_index += 1
        return list(self._children[index])

    def cpu_times(self) -> Any:
        index = min(self._cpu_index, len(self._cpu_values) - 1)
        self._cpu_index += 1
        return type("CpuTimes", (), {"user": self._cpu_values[index], "system": 0.0})()

    def memory_info(self) -> Any:
        return type("MemoryInfo", (), {"rss": self._rss})()
