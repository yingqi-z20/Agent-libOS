from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.capability.rules import AUTHORITY_RULES_KEY
from agent_libos.models import AgentImage, CapabilityRight, ExternalEffectClassification
from agent_libos.models import ExternalEffectRollbackClass, ExternalEffectRollbackStatus, HumanRequestStatus, ObjectType, ResourceBudget
from agent_libos.models.exceptions import HumanApprovalRequired
from agent_libos.substrate import LocalResourceProviderSubstrate, SubprocessLimits


class TestPtyModule:
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
                    session = _pty_adapter(runtime)._sessions[first_oid]
                    if session.closed:
                        break
                    time.sleep(0.01)

                assert _pty_adapter(runtime)._sessions[first_oid].closed
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

    def test_pty_close_failure_keeps_session_registered_for_retry(self) -> None:
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

                assert retried.ok, retried.error
                assert provider.sessions[0].closed
                assert runtime.store.get_object(session_oid) is None
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
                    resource_budget=ResourceBudget(max_subprocess_wall_seconds=0.001),
                )
                created = runtime.tools.call(pid, "pty_create", {"argv": ["git", "status"], "startup_timeout_s": 0})
                assert created.ok, created.error

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and runtime.process.get(pid).status.value != "killed":
                    time.sleep(0.01)

                process = runtime.process.get(pid)
                assert process.status.value == "killed"
                assert process.resource_usage.subprocess_wall_seconds > 0
                assert provider.sessions[0].closed
            finally:
                runtime.close()

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
) -> Runtime:
    substrate = LocalResourceProviderSubstrate(root)
    substrate.pty = provider
    if settings is not None:
        substrate.pty_settings = settings
    manifest = _module_manifest()
    source_sha = hashlib.sha256((manifest.parent / "pty_module.py").read_bytes()).hexdigest()
    return Runtime.open(
        "local",
        substrate=substrate,
        module_manifests=(str(manifest),),
        trusted_modules=(f"agent-libos-pty:v0:{source_sha}",),
    )


def _module_manifest() -> Path:
    return Path("modules/pty/module.yaml").resolve()


def _pty_adapter(runtime: Runtime) -> Any:
    return getattr(runtime, "_agent_libos_pty_adapter")


class FakePtyProvider:
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

    def read(self, *, timeout_s: float = 0.0) -> str:
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
        if self.close_failures > 0:
            self.close_failures -= 1
            raise RuntimeError("simulated close failure")
        self.closed = True
        self.alive = False
        return 0
