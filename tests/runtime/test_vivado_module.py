from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.modules.loader import ModuleLoader
from agent_libos.models import CapabilityRight, ExternalEffectClassification
from agent_libos.models import ExternalEffectRollbackClass, ExternalEffectRollbackStatus, ObjectType
from agent_libos.substrate import LocalResourceProviderSubstrate


@pytest.fixture(autouse=True)
def _clear_vivado_module_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("AGENT_LIBOS_VIVADO_"):
            monkeypatch.delenv(key, raising=False)


class TestVivadoModule:
    def test_manifest_verifies_current_source_hash(self) -> None:
        verified = ModuleLoader().verify(_module_manifest())

        assert verified["module_id"] == "agent-libos-vivado:v0"
        assert verified["source_sha256"] == hashlib.sha256((_module_manifest().parent / "vivado_module.py").read_bytes()).hexdigest()
        assert "vivado_sync_push" in verified["provides"]["tools"]

    def test_loaded_module_registers_tools_image_and_session_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakeVivadoProvider()
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                loaded = runtime.modules.inspect_module("agent-libos-vivado:v0")
                assert loaded["status"] == "loaded"
                assert "vivado_session_create" in loaded["registered"]["tools"]
                assert "vivado-agent:v0" in runtime.images

                pid = runtime.process.spawn(image="vivado-agent:v0", goal="drive vivado")
                health = runtime.tools.call(pid, "vivado_health", {})
                assert health.ok, health.error
                assert health.payload["ok"] is True

                created = runtime.tools.call(pid, "vivado_session_create", {"project": "demo"})
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]
                obj = runtime.store.get_object(session_oid)
                assert obj is not None
                assert obj.type == ObjectType.EXTERNAL_REF
                assert obj.payload["kind"] == "vivado_session"
                assert obj.payload["project"] == "demo"

                written = runtime.tools.call(
                    pid,
                    "vivado_session_send_stdin",
                    {"session_oid": session_oid, "text": "open_project demo.xpr\n"},
                )
                assert written.ok, written.error
                assert provider.stdin[created.payload["session_id"]] == ["open_project demo.xpr\n"]

                read = runtime.tools.call(pid, "vivado_session_read_output", {"session_oid": session_oid, "timeout_ms": 1})
                assert read.ok, read.error
                assert read.payload["cursor"] == 1
                assert "Vivado ready" in read.payload["output"]
                assert read.payload["heartbeat_sent"] is True
                assert any(effect.operation == "heartbeat" for effect in runtime.store.list_external_effects())

                status = runtime.tools.call(pid, "vivado_session_status", {"session_oid": session_oid})
                assert status.ok, status.error
                assert status.payload["status"] == "running"

                heartbeat = runtime.tools.call(pid, "vivado_session_heartbeat", {"session_oid": session_oid})
                assert heartbeat.ok, heartbeat.error
                assert provider.heartbeats[created.payload["session_id"]] >= 2

                closed = runtime.tools.call(pid, "vivado_session_close", {"session_oid": session_oid})
                assert closed.ok, closed.error
                assert runtime.store.get_object(session_oid) is None
                assert created.payload["session_id"] in provider.deleted_sessions
                assert any(effect.provider == "vivado" for effect in runtime.store.list_external_effects())
            finally:
                runtime.close()

    def test_startup_reads_vivado_settings_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_LIBOS_VIVADO_SESSION_NAME_PREFIX", "env_session")
        monkeypatch.setenv("AGENT_LIBOS_VIVADO_OUTPUT_TIMEOUT_MS", "7")
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakeVivadoProvider()
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="vivado-agent:v0", goal="env config")
                created = runtime.tools.call(pid, "vivado_session_create", {"project": "demo"})
                assert created.ok, created.error
                assert created.payload["name"] == "env_session:1"

                read = runtime.tools.call(pid, "vivado_session_read_output", {"session_oid": created.payload["session_oid"]})

                assert read.ok, read.error
                assert provider.output_timeouts == [7]
            finally:
                runtime.close()

    def test_session_create_requires_vivado_capability_before_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakeVivadoProvider()
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                runtime.register_image(
                    {
                        "image_id": "vivado-no-cap:v0",
                        "name": "vivado-no-cap",
                        "default_tools": ["vivado_session_create"],
                    },
                    actor="test",
                )
                pid = runtime.process.spawn(image="vivado-no-cap:v0", goal="missing cap")

                result = runtime.tools.call(pid, "vivado_session_create", {"project": "demo"})

                assert not result.ok
                assert "lacks execute" in (result.error or "")
                assert provider.created_sessions == []
            finally:
                runtime.close()

    def test_sync_pull_requires_local_read_before_provider_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            (project / "old.txt").write_text("old\n", encoding="utf-8")
            provider = FakeVivadoProvider()
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="vivado-agent:v0", goal="pull requires read")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.WRITE], issued_by="test")

                result = runtime.tools.call(pid, "vivado_sync_pull", {"project": "demo", "local_root": "project"})

                assert not result.ok
                assert "lacks read" in (result.error or "")
                assert provider.pull_plan_calls == 0
            finally:
                runtime.close()

    def test_sync_push_and_pull_with_default_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            (project / "src").mkdir(parents=True)
            (project / "src" / "top.tcl").write_text("puts hello\n", encoding="utf-8")
            (project / ".git").mkdir()
            (project / ".git" / "ignored").write_text("secret\n", encoding="utf-8")
            provider = FakeVivadoProvider()
            provider.remote_files["out/result.txt"] = b"ok\n"
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="vivado-agent:v0", goal="sync")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.READ], issued_by="test")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.WRITE], issued_by="test")

                pushed = runtime.tools.call(pid, "vivado_sync_push", {"project": "demo", "local_root": "project"})
                assert pushed.ok, pushed.error
                assert pushed.payload["uploaded_files"] == ["src/top.tcl"]
                assert ".git/ignored" not in provider.uploaded
                assert provider.commits == [{"project": "demo", "sync_id": pushed.payload["sync_id"], "force": False}]

                pulled = runtime.tools.call(
                    pid,
                    "vivado_sync_pull",
                    {
                        "project": "demo",
                        "local_root": "project",
                        "include_globs": ["out/**"],
                    },
                )
                assert pulled.ok, pulled.error
                assert (project / "out" / "result.txt").read_text(encoding="utf-8") == "ok\n"
                assert pulled.payload["downloaded_files"] == ["out/result.txt"]
            finally:
                runtime.close()

    def test_pull_rejects_reserved_sync_symlink_before_provider_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            outside = root / "outside"
            project.mkdir()
            outside.mkdir()
            try:
                os.symlink(outside, project / ".vivado-server-sync", target_is_directory=True)
            except (OSError, NotImplementedError):
                pytest.skip("symlink creation is not available in this environment")
            provider = FakeVivadoProvider()
            provider.remote_files["out/result.txt"] = b"ok\n"
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="vivado-agent:v0", goal="reserved symlink pull")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.READ], issued_by="test")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.WRITE], issued_by="test")

                result = runtime.tools.call(
                    pid,
                    "vivado_sync_pull",
                    {"project": "demo", "local_root": "project", "include_globs": ["out/**"]},
                )

                assert not result.ok
                assert "reserved sync directory" in (result.error or "")
                assert provider.pull_plan_calls == 0
                assert provider.download_calls == 0
            finally:
                runtime.close()

    def test_push_rejects_symlink_before_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            (project / "target.txt").write_text("target\n", encoding="utf-8")
            try:
                os.symlink(project / "target.txt", project / "link.txt")
            except (OSError, NotImplementedError):
                pytest.skip("symlink creation is not available in this environment")
            provider = FakeVivadoProvider()
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="vivado-agent:v0", goal="symlink push")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.READ], issued_by="test")

                result = runtime.tools.call(pid, "vivado_sync_push", {"project": "demo", "local_root": "project"})

                assert not result.ok
                assert "symlinks" in (result.error or "")
                assert provider.uploaded == {}
            finally:
                runtime.close()

    def test_pull_checksum_failure_leaves_existing_target_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            (project / "out").mkdir(parents=True)
            target = project / "out" / "result.txt"
            target.write_text("old\n", encoding="utf-8")
            provider = FakeVivadoProvider()
            provider.remote_files["out/result.txt"] = b"new\n"
            provider.bad_download_sha = True
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="vivado-agent:v0", goal="bad pull")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.READ], issued_by="test")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.WRITE], issued_by="test")

                result = runtime.tools.call(
                    pid,
                    "vivado_sync_pull",
                    {"project": "demo", "local_root": "project", "include_globs": ["out/**"]},
                )

                assert not result.ok
                assert "sha256 mismatch" in (result.error or "")
                assert target.read_text(encoding="utf-8") == "old\n"
            finally:
                runtime.close()

    def test_pull_rejects_oversized_plan_entry_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            provider = FakeVivadoProvider()
            provider.remote_files["out/big.bin"] = b"12345"
            runtime = _open_vivado_runtime(temp_dir, provider, settings={"max_file_bytes": 4})
            try:
                pid = runtime.process.spawn(image="vivado-agent:v0", goal="big pull")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.READ], issued_by="test")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.WRITE], issued_by="test")

                result = runtime.tools.call(
                    pid,
                    "vivado_sync_pull",
                    {"project": "demo", "local_root": "project", "include_globs": ["out/**"]},
                )

                assert not result.ok
                assert "max_file_bytes=4" in (result.error or "")
                assert provider.download_calls == 0
            finally:
                runtime.close()

    def test_pull_delete_extra_requires_delete_permission_before_provider_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            (project / "old.txt").write_text("old\n", encoding="utf-8")
            provider = FakeVivadoProvider()
            provider.pull_delete_files = ["old.txt"]
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="vivado-agent:v0", goal="delete extra")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.READ], issued_by="test")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.WRITE], issued_by="test")

                result = runtime.tools.call(
                    pid,
                    "vivado_sync_pull",
                    {"project": "demo", "local_root": "project", "delete_extra": True},
                )

                assert not result.ok
                assert "lacks delete" in (result.error or "")
                assert provider.pull_plan_calls == 0
                assert (project / "old.txt").exists()
            finally:
                runtime.close()

    def test_push_consumes_one_time_project_capability_before_failed_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            (project / "top.tcl").write_text("puts hello\n", encoding="utf-8")
            provider = FakeVivadoProvider()
            provider.fail_upload = True
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                runtime.register_image(
                    {
                        "image_id": "vivado-sync-once:v0",
                        "name": "vivado-sync-once",
                        "default_tools": ["vivado_sync_push"],
                    },
                    actor="test",
                )
                pid = runtime.process.spawn(image="vivado-sync-once:v0", goal="push once")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.READ], issued_by="test")
                runtime.capability.grant_once(pid, "vivado:project:demo", [CapabilityRight.WRITE], issued_by="test")

                result = runtime.tools.call(pid, "vivado_sync_push", {"project": "demo", "local_root": "project"})

                assert not result.ok
                assert provider.push_plan_calls == 1
                assert provider.aborted == ["sync-1"]
                assert not runtime.capability.check(pid, "vivado:project:demo", CapabilityRight.WRITE)
            finally:
                runtime.close()

    def test_pull_consumes_one_time_write_capability_before_failed_download_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            project.mkdir()
            provider = FakeVivadoProvider()
            provider.remote_files["out/result.txt"] = b"new\n"
            provider.bad_download_sha = True
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                runtime.register_image(
                    {
                        "image_id": "vivado-pull-once:v0",
                        "name": "vivado-pull-once",
                        "default_tools": ["vivado_sync_pull"],
                    },
                    actor="test",
                )
                pid = runtime.process.spawn(image="vivado-pull-once:v0", goal="pull once")
                runtime.capability.grant(pid, "vivado:project:demo", [CapabilityRight.READ], issued_by="test")
                runtime.filesystem.grant_directory(pid, "project", [CapabilityRight.READ], issued_by="test")
                write_resource = runtime.filesystem.directory_resource_for("project")
                runtime.capability.grant_once(pid, write_resource, [CapabilityRight.WRITE], issued_by="test")

                result = runtime.tools.call(
                    pid,
                    "vivado_sync_pull",
                    {"project": "demo", "local_root": "project", "include_globs": ["out/**"]},
                )

                assert not result.ok
                assert "sha256 mismatch" in (result.error or "")
                assert not runtime.capability.check(pid, write_resource, CapabilityRight.WRITE)
            finally:
                runtime.close()

    def test_other_process_with_delegated_object_write_cannot_drive_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakeVivadoProvider()
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                owner = runtime.process.spawn(image="vivado-agent:v0", goal="owner")
                other = runtime.process.spawn(image="vivado-agent:v0", goal="other")
                created = runtime.tools.call(owner, "vivado_session_create", {"project": "demo"})
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]
                runtime.capability.grant(other, f"object:{session_oid}", [CapabilityRight.WRITE], issued_by="test")

                result = runtime.tools.call(
                    other,
                    "vivado_session_send_stdin",
                    {"session_oid": session_oid, "text": "launch_runs impl_1\n"},
                )

                assert not result.ok
                assert "owned by" in (result.error or "")
                assert provider.stdin[created.payload["session_id"]] == []
            finally:
                runtime.close()

    def test_direct_object_release_closes_remote_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = FakeVivadoProvider()
            runtime = _open_vivado_runtime(temp_dir, provider)
            try:
                pid = runtime.process.spawn(image="vivado-agent:v0", goal="release")
                created = runtime.tools.call(pid, "vivado_session_create", {"project": "demo"})
                assert created.ok, created.error
                session_oid = created.payload["session_oid"]

                assert runtime.memory.delete_object_trusted("test", session_oid, reason="direct_release")

                assert created.payload["session_id"] in provider.deleted_sessions
                assert runtime.store.get_object(session_oid) is None
            finally:
                runtime.close()


def _open_vivado_runtime(root: str, provider: "FakeVivadoProvider", *, settings: dict[str, Any] | None = None) -> Runtime:
    substrate = LocalResourceProviderSubstrate(root)
    substrate.vivado = provider
    if settings is not None:
        substrate.vivado_settings = settings
    manifest = _module_manifest()
    source_sha = hashlib.sha256((manifest.parent / "vivado_module.py").read_bytes()).hexdigest()
    return Runtime.open(
        "local",
        substrate=substrate,
        module_manifests=(str(manifest),),
        trusted_modules=(f"agent-libos-vivado:v0:{source_sha}",),
    )


def _module_manifest() -> Path:
    return Path("modules/vivado/module.yaml").resolve()


class FakeVivadoProvider:
    def __init__(self) -> None:
        self.created_sessions: list[dict[str, Any]] = []
        self.stdin: dict[str, list[str]] = {}
        self.heartbeats: dict[str, int] = {}
        self.deleted_sessions: list[str] = []
        self.remote_files: dict[str, bytes] = {}
        self.uploaded: dict[str, bytes] = {}
        self.commits: list[dict[str, Any]] = []
        self.aborted: list[str] = []
        self.fail_upload = False
        self.bad_download_sha = False
        self.pull_delete_files: list[str] = []
        self.pull_delete_dirs: list[str] = []
        self.pull_plan_calls = 0
        self.push_plan_calls = 0
        self.download_calls = 0
        self.output_timeouts: list[int] = []
        self._session_counter = 0
        self._sync_counter = 0

    def health(self, *, timeout_s: float) -> dict[str, Any]:
        return {"status": "ok"}

    def create_push_plan(
        self,
        project: str,
        entries: list[dict[str, Any]],
        *,
        delete_extra: bool,
        include_globs: list[str],
        exclude_globs: list[str],
        timeout_s: float,
    ) -> dict[str, Any]:
        self.push_plan_calls += 1
        self._sync_counter += 1
        upload_files = [
            {
                "path": entry["path"],
                "size_bytes": entry["size_bytes"],
                "mtime_unix_ms": entry["mtime_unix_ms"],
                "sha256": entry["sha256"],
            }
            for entry in entries
            if entry.get("kind") == "file"
        ]
        return {
            "sync_id": f"sync-{self._sync_counter}",
            "upload_files": upload_files,
            "create_dirs": [entry["path"] for entry in entries if entry.get("kind") == "dir"],
            "delete_files": [],
            "delete_dirs": [],
            "expires_at": "2026-06-27T12:34:56Z",
        }

    def upload_push_file(
        self,
        project: str,
        sync_id: str,
        sync_path: str,
        source_path: Path,
        *,
        size_bytes: int,
        sha256: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        if self.fail_upload:
            raise RuntimeError("upload failed")
        raw = source_path.read_bytes()
        assert len(raw) == size_bytes
        assert hashlib.sha256(raw).hexdigest() == sha256
        self.uploaded[sync_path] = raw
        return {"path": sync_path, "size_bytes": len(raw), "sha256": sha256}

    def commit_push(self, project: str, sync_id: str, *, force: bool, timeout_s: float) -> dict[str, Any]:
        self.commits.append({"project": project, "sync_id": sync_id, "force": force})
        self.remote_files.update(self.uploaded)
        return {
            "sync_id": sync_id,
            "status": "committed",
            "uploaded_files": sorted(self.uploaded),
            "created_dirs": ["src"] if "src/top.tcl" in self.uploaded else [],
            "deleted_files": [],
            "deleted_dirs": [],
        }

    def abort_push(self, project: str, sync_id: str, *, timeout_s: float) -> dict[str, Any]:
        self.aborted.append(sync_id)
        return {"sync_id": sync_id, "status": "aborted"}

    def create_pull_plan(
        self,
        project: str,
        entries: list[dict[str, Any]],
        *,
        delete_extra: bool,
        include_globs: list[str],
        exclude_globs: list[str],
        timeout_s: float,
    ) -> dict[str, Any]:
        self.pull_plan_calls += 1
        selected = [
            path
            for path in sorted(self.remote_files)
            if not include_globs or any(path.startswith(glob[:-3]) for glob in include_globs if glob.endswith("/**"))
        ]
        return {
            "download_files": [
                {
                    "path": path,
                    "size_bytes": len(self.remote_files[path]),
                    "mtime_unix_ms": 1_700_000_000_000,
                    "sha256": hashlib.sha256(self.remote_files[path]).hexdigest(),
                }
                for path in selected
            ],
            "create_dirs": sorted({path.rsplit("/", 1)[0] for path in selected if "/" in path}),
            "delete_files": list(self.pull_delete_files),
            "delete_dirs": list(self.pull_delete_dirs),
        }

    def download_file(self, project: str, sync_path: str, target_path: Path, *, timeout_s: float) -> dict[str, Any]:
        self.download_calls += 1
        raw = self.remote_files[sync_path]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        if self.bad_download_sha:
            sha = "0" * 64
        return {
            "path": sync_path,
            "size_bytes": len(raw),
            "mtime_unix_ms": 1_700_000_000_000,
            "sha256": sha,
            "bytes_written": len(raw),
            "actual_sha256": hashlib.sha256(raw).hexdigest(),
        }

    def create_session(self, project: str, args: list[str], *, timeout_s: float) -> dict[str, Any]:
        self._session_counter += 1
        session_id = f"sess-{self._session_counter}"
        self.created_sessions.append({"project": project, "args": list(args), "session_id": session_id})
        self.stdin[session_id] = []
        self.heartbeats[session_id] = 0
        return {
            "session_id": session_id,
            "project": project,
            "status": "running",
            "started_at": "2026-06-27T12:00:00Z",
            "last_heartbeat_at": "2026-06-27T12:00:00Z",
            "exit_code": None,
        }

    def send_stdin(self, session_id: str, text: str, *, timeout_s: float) -> dict[str, Any]:
        self.stdin[session_id].append(text)
        return {"session_id": session_id, "status": "running"}

    def read_output(self, session_id: str, *, cursor: int, timeout_ms: int, timeout_s: float) -> dict[str, Any]:
        self.output_timeouts.append(timeout_ms)
        return {
            "cursor": cursor + 1,
            "chunks": [{"seq": cursor, "timestamp": "2026-06-27T12:00:01Z", "text": "Vivado ready\n"}],
            "status": "running",
            "overrun": False,
        }

    def heartbeat(self, session_id: str, *, timeout_s: float) -> dict[str, Any]:
        self.heartbeats[session_id] += 1
        return {"session_id": session_id, "status": "running", "last_heartbeat_at": "2026-06-27T12:00:02Z"}

    def get_session(self, session_id: str, *, timeout_s: float) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "project": "demo",
            "status": "running",
            "started_at": "2026-06-27T12:00:00Z",
            "last_heartbeat_at": "2026-06-27T12:00:02Z",
            "exit_code": None,
        }

    def delete_session(self, session_id: str, *, timeout_s: float) -> dict[str, Any]:
        self.deleted_sessions.append(session_id)
        return {"session_id": session_id, "status": "terminated"}

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if operation in {"health", "pull_plan", "download_file", "read_output", "session_status"}:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={"operation": operation},
            )
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
            rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
            state_mutation=True,
            information_flow=True,
            metadata={"operation": operation},
        )
