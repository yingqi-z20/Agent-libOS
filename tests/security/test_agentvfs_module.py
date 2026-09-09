from __future__ import annotations

import hashlib
import json
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agent_libos import ObjectMetadata, ObjectType, Runtime
from agent_libos.config import AgentLibOSConfig
from agent_libos.models import ObjectPatch
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime import RuntimeAssemblyCleanupRequired
from agent_libos.substrate import LocalResourceProviderSubstrate
from tests.support.public_errors import assert_public_error_message

COMMIT_A = "a" * 64
COMMIT_B = "b" * 64


class FakeAgentVfsControl:
    """In-test agentvfs control daemon speaking the newline JSONL protocol."""

    def __init__(self) -> None:
        self._dir = tempfile.TemporaryDirectory(prefix="avfs-fake-")
        self.path = str(Path(self._dir.name) / "control.sock")
        self.received: list[str] = []
        self.failures: dict[str, str] = {}
        self.line_responses: dict[str, dict[str, Any]] = {}
        self._responses: dict[str, dict[str, Any]] = {
            "status": {
                "ok": True,
                "version": "cas-minimal-0.1",
                "branch": "main",
                "commit": COMMIT_A,
                "telemetry_drops_total": 0,
            },
            "checkpoint": {"ok": True, "commit": COMMIT_B},
            "rollback": {"ok": True, "rolled_back_to": COMMIT_B},
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._thread: threading.Thread | None = None

    def start(self) -> FakeAgentVfsControl:
        self._listener.bind(self.path)
        self._listener.listen(8)
        self._listener.settimeout(0.1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    data = conn.recv(65536)
                except OSError:
                    continue
                line = bytes(data).decode(errors="replace").strip()
                verb = line.split(" ", 1)[0]
                with self._lock:
                    self.received.append(line)
                    error = self.failures.get(verb)
                    payload = (
                        {"ok": False, "error": error}
                        if error
                        else self.line_responses.get(line)
                        or self._responses.get(
                            verb, {"ok": False, "error": f"unknown verb {verb!r}"}
                        )
                    )
                try:
                    conn.sendall((json.dumps(payload) + "\n").encode())
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._listener.close()
        self._dir.cleanup()


def _module_manifest() -> Path:
    manifest = Path("modules/agentvfs/module.yaml").resolve()
    shutil.rmtree(manifest.parent / "__pycache__", ignore_errors=True)
    return manifest


def _trust_key(manifest: Path) -> str:
    source_sha = next(
        line.split(":", 1)[1].strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.startswith("sha256:")
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return f"agent-libos-agentvfs:v0:{manifest_sha}:{source_sha}"


def _write_session(xdg_root: Path, workspace: str, socket_path: str, *, status: str) -> None:
    session_dir = xdg_root / "agentvfs" / workspace
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "name": workspace,
                "status": status,
                "socket": socket_path,
                "mount": str(xdg_root / workspace / "mount"),
                "store": str(xdg_root / workspace / "store"),
            }
        ),
        encoding="utf-8",
    )


def _close_runtime(runtime: Runtime) -> None:
    deadline = time.monotonic() + 15.0
    result = runtime.close()
    while not result.get("ok") and time.monotonic() < deadline:
        time.sleep(0.01)
        result = runtime.close()
    assert result.get("ok"), result


def _flatten(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return "\n".join(_flatten(exc) for exc in error.exceptions)
    return str(error)


class TestAgentVfsModule:

    def setup_method(self) -> None:
        self._workspace = "libos-module-test"
        self._fake = FakeAgentVfsControl().start()
        self._xdg = tempfile.TemporaryDirectory(prefix="avfs-xdg-")
        _write_session(
            Path(self._xdg.name), self._workspace, self._fake.path, status="started"
        )
        self._runtimes: list[Runtime] = []
        self._workspaces: list[tempfile.TemporaryDirectory[str]] = []

    def teardown_method(self) -> None:
        for runtime in self._runtimes:
            _close_runtime(runtime)
        while self._workspaces:
            self._workspaces.pop().cleanup()
        self._xdg.cleanup()
        self._fake.stop()

    def _open_runtime(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        binding: Any = "default",
        session_status: str = "started",
    ) -> Runtime:
        if session_status != "started":
            _write_session(
                Path(self._xdg.name),
                self._workspace,
                self._fake.path,
                status=session_status,
            )
        monkeypatch.setenv("XDG_RUNTIME_DIR", self._xdg.name)
        workspace = tempfile.TemporaryDirectory(prefix="avfs-ws-")
        self._workspaces.append(workspace)
        substrate = LocalResourceProviderSubstrate(workspace.name)
        if binding != "default":
            substrate.agentvfs = binding
        else:
            substrate.agentvfs = {"workspace": self._workspace}
        runtime = Runtime.open(
            "local",
            substrate=substrate,
            config=AgentLibOSConfig(),
            module_manifests=(str(_module_manifest()),),
            trusted_modules=(_trust_key(_module_manifest()),),
        )
        self._runtimes.append(runtime)
        return runtime

    def _spawn(self, runtime: Runtime, rights: list[str] | None = None) -> str:
        capabilities = (
            None
            if rights is None
            else [{"resource": f"agentvfs:{self._workspace}", "rights": rights}]
        )
        return runtime.process.spawn(
            image="agentvfs-agent:v0",
            goal="agentvfs module contract",
            capabilities=capabilities,
        )

    def test_tools_require_capability_before_any_socket_traffic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._open_runtime(monkeypatch)
        assert runtime.get_image("agentvfs-agent:v0").default_tools is not None
        pid = self._spawn(runtime)

        for name, args in (
            ("agentvfs_status", {}),
            ("agentvfs_checkpoint", {"label": "c1"}),
            ("agentvfs_rollback", {"target": "c1"}),
        ):
            denied = runtime.tools.call(pid, name, args)
            assert not denied.ok, (name, denied.error)
            assert_public_error_message(
                denied.error,
                code="permission_denied",
                error_type="CapabilityDenied",
                forbidden=("agentvfs",),
            )

        assert self._fake.received == []

    def test_status_and_checkpoint_succeed_with_granted_capability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._open_runtime(monkeypatch)
        pid = self._spawn(runtime, ["read", "write"])

        status = runtime.tools.call(pid, "agentvfs_status", {})
        assert status.ok, status.error
        assert status.payload["workspace"] == self._workspace
        assert status.payload["commit"] == COMMIT_A
        assert status.payload["branch"] == "main"

        checkpoint = runtime.tools.call(pid, "agentvfs_checkpoint", {"label": "c1"})
        assert checkpoint.ok, checkpoint.error
        assert checkpoint.payload["commit"] == COMMIT_B
        assert checkpoint.payload["label"] == "c1"

        assert self._fake.received == ["status", "checkpoint c1"]
        audit_actions = [record.action for record in runtime.audit.trace()]
        assert "module.agentvfs.status" in audit_actions
        assert "module.agentvfs.checkpoint" in audit_actions
        event_types = [
            event.event_type if hasattr(event, "event_type") else event.type
            for event in runtime.store.list_events()
        ]
        assert "external_read" in event_types
        assert "external_write" in event_types

    def test_rollback_requires_the_stronger_admin_right(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._open_runtime(monkeypatch)
        pid = self._spawn(runtime, ["read", "write"])

        denied = runtime.tools.call(pid, "agentvfs_rollback", {"target": "c1"})
        assert not denied.ok, denied.error
        assert_public_error_message(
            denied.error,
            code="permission_denied",
            error_type="CapabilityDenied",
        )
        assert self._fake.received == []

        runtime.capability.grant(
            subject=pid,
            resource=f"agentvfs:{self._workspace}",
            rights=["admin"],
            issued_by="test",
        )
        rolled = runtime.tools.call(pid, "agentvfs_rollback", {"target": "c1"})
        assert rolled.ok, rolled.error
        assert rolled.payload["rolled_back_to"] == COMMIT_B
        assert self._fake.received == ["rollback c1"]

    def test_daemon_error_is_surfaced_as_tool_failure_without_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._open_runtime(monkeypatch)
        pid = self._spawn(runtime, ["read", "write"])

        self._fake.failures["checkpoint"] = "daemon exploded"
        failed = runtime.tools.call(pid, "agentvfs_checkpoint", {"label": "c1"})
        assert not failed.ok
        assert_public_error_message(
            failed.error,
            code="execution_error",
            error_type="AgentVfsControlError",
            forbidden=("daemon exploded",),
        )

        self._fake.failures.pop("checkpoint")
        recovered = runtime.tools.call(pid, "agentvfs_status", {})
        assert recovered.ok, recovered.error

    def test_unbound_module_fails_closed_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._open_runtime(monkeypatch, binding=None)
        pid = self._spawn(runtime, ["read", "write", "admin"])

        for name, args in (
            ("agentvfs_status", {}),
            ("agentvfs_checkpoint", {"label": "c1"}),
            ("agentvfs_rollback", {"target": "c1"}),
        ):
            unbound = runtime.tools.call(pid, name, args)
            assert not unbound.ok, (name, unbound.error)
            assert_public_error_message(
                unbound.error,
                code="execution_error",
                error_type="ToolExecutionError",
            )
        assert self._fake.received == []

    def test_startup_fails_closed_when_bound_workspace_is_not_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        error: BaseException | None = None
        try:
            self._open_runtime(monkeypatch, session_status="stale")
        except BaseException as exc:
            error = exc
        if error is None:
            raise AssertionError("Runtime.open should fail when the bound workspace is stale")
        for handle in RuntimeAssemblyCleanupRequired.extract(error):
            handle.release()
        message = _flatten(error)
        assert "not running" in message

    def test_paired_checkpoint_records_agentvfs_commit_in_libos_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._open_runtime(monkeypatch)
        pid = self._spawn(runtime, ["read", "write"])

        created = runtime.tools.call(
            pid,
            "agentvfs_checkpoint",
            {"label": "c1", "pair_libos": True, "reason": "paired contract"},
        )
        assert created.ok, created.error
        checkpoint_id = created.payload["libos_checkpoint_id"]
        assert checkpoint_id.startswith("ckpt_")
        assert created.payload["commit"] == COMMIT_B

        inspected = runtime.checkpoint.inspect(checkpoint_id, actor=pid)
        metadata = inspected["checkpoint"]["metadata"]
        assert metadata["agentvfs_commit"] == COMMIT_B
        assert metadata["agentvfs_label"] == "c1"
        assert metadata["agentvfs_workspace"] == self._workspace
        assert self._fake.received == ["checkpoint c1"]

    def test_paired_checkpoint_requires_libos_authority_before_any_traffic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._open_runtime(monkeypatch)
        pid = self._spawn(runtime, ["read", "write"])
        process_checkpoint_cap = next(
            capability
            for capability in runtime.store.list_capabilities(subject=pid)
            if capability.resource == f"checkpoint:process:{pid}"
        )
        runtime.capability.revoke(
            process_checkpoint_cap.cap_id,
            revoked_by="test",
            require_authority=False,
        )

        denied = runtime.tools.call(
            pid, "agentvfs_checkpoint", {"label": "c1", "pair_libos": True}
        )
        assert not denied.ok, denied.error
        assert_public_error_message(
            denied.error,
            code="permission_denied",
            error_type="CapabilityDenied",
        )
        assert self._fake.received == []
        assert runtime.checkpoint.list(pid=pid) == []
        audit_actions = [record.action for record in runtime.audit.trace()]
        assert "module.agentvfs.pair_checkpoint.denied" in audit_actions

    def test_paired_rollback_reports_pending_host_restore_and_id_is_restorable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._open_runtime(monkeypatch)
        pid = self._spawn(runtime, ["read", "write", "admin"])

        created = runtime.tools.call(
            pid, "agentvfs_checkpoint", {"label": "c1", "pair_libos": True}
        )
        assert created.ok, created.error
        checkpoint_id = created.payload["libos_checkpoint_id"]

        rolled = runtime.tools.call(
            pid,
            "agentvfs_rollback",
            {"target": "c1", "pair_libos": True, "libos_checkpoint_id": checkpoint_id},
        )
        assert rolled.ok, rolled.error
        assert rolled.payload["rolled_back_to"] == COMMIT_B
        assert rolled.payload["paired_libos_checkpoint_id"] == checkpoint_id
        assert rolled.payload["libos_restore"] == "pending_host_restore"
        assert rolled.payload["libos_restore_hint"]
        assert self._fake.received == ["checkpoint c1", "rollback c1"]

        restored = runtime.checkpoint.restore(
            "test", checkpoint_id, require_capability=False
        )
        assert restored["status"] == "restored"
        audit_actions = [record.action for record in runtime.audit.trace()]
        assert "module.agentvfs.libos_restore_pending" in audit_actions

    def test_paired_rollback_restores_both_planes_when_host_grants_checkpoint_admin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._open_runtime(monkeypatch)
        pid = self._spawn(runtime, ["read", "write", "admin"])
        note = runtime.memory.create_object(
            pid,
            ObjectType.PLAN,
            {"version": 1},
            ObjectMetadata(title="paired note"),
            immutable=False,
            name="agentvfs.paired.note",
        )

        created = runtime.tools.call(
            pid, "agentvfs_checkpoint", {"label": "c1", "pair_libos": True}
        )
        assert created.ok, created.error
        checkpoint_id = created.payload["libos_checkpoint_id"]

        runtime.memory.update_object(
            pid, note, ObjectPatch(payload={"version": 2})
        )
        assert (
            runtime.memory.get_object_by_name(pid, "agentvfs.paired.note").payload[
                "version"
            ]
            == 2
        )

        runtime.capability.grant(
            subject=pid,
            resource=f"checkpoint:{checkpoint_id}",
            rights=["admin"],
            issued_by="test",
        )
        rolled = runtime.tools.call(
            pid,
            "agentvfs_rollback",
            {"target": "c1", "pair_libos": True, "libos_checkpoint_id": checkpoint_id},
        )
        assert rolled.ok, rolled.error
        assert rolled.payload["libos_restore"] == "restored"
        assert not rolled.payload["libos_restore_hint"]
        assert (
            runtime.memory.get_object_by_name(pid, "agentvfs.paired.note").payload
            == {"version": 1}
        )

    def test_paired_rollback_rejects_commit_mismatch_after_filesystem_rollback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._open_runtime(monkeypatch)
        pid = self._spawn(runtime, ["read", "write", "admin"])
        created = runtime.tools.call(
            pid, "agentvfs_checkpoint", {"label": "c1", "pair_libos": True}
        )
        assert created.ok, created.error
        checkpoint_id = created.payload["libos_checkpoint_id"]

        self._fake.line_responses["rollback c2"] = {
            "ok": True,
            "rolled_back_to": COMMIT_A,
        }
        mismatch = runtime.tools.call(
            pid,
            "agentvfs_rollback",
            {"target": "c2", "pair_libos": True, "libos_checkpoint_id": checkpoint_id},
        )
        assert not mismatch.ok, mismatch.error
        assert_public_error_message(
            mismatch.error,
            code="validation_error",
            error_type="ValidationError",
        )
        # The filesystem rollback stands; the daemon saw the mismatched target.
        assert self._fake.received == ["checkpoint c1", "rollback c2"]

    def test_paired_rollback_requires_checkpoint_id_argument(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = self._open_runtime(monkeypatch)
        pid = self._spawn(runtime, ["read", "write", "admin"])

        missing = runtime.tools.call(
            pid, "agentvfs_rollback", {"target": "c1", "pair_libos": True}
        )
        assert not missing.ok, missing.error
        assert self._fake.received == []
