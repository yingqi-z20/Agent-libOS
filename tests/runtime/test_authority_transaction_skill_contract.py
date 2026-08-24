from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import (
    CapabilityEffect,
    CapabilityRight,
    EventType,
    ValidationResult,
)
from agent_libos.models.exceptions import CapabilityDenied
from agent_libos.runtime.runtime import Runtime
from agent_libos.substrate import LocalResourceProviderSubstrate, SubprocessLimits
from agent_libos.tools.sandbox import SandboxBackend
from tests.support.skills import write_skill_package


PERSISTENT_BACKENDS = [
    "sqlite-file",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


def _release_fenced_runtime_or_close(runtime: Runtime) -> None:
    reason = runtime.lifecycle.shutdown_reason
    if (
        runtime.lifecycle.state == "close_failed"
        and isinstance(reason, str)
        and reason.startswith("runtime.recovery_required:")
    ):
        result = runtime.release_recovery_diagnostics()
        assert result["ok"] is True, result
        assert result["recovery_diagnostics_released"] is True
        return
    runtime.close()


_JIT_SOURCE_V1 = (
    "export async function run(args: unknown, libos: unknown) "
    "{ return { version: 1 }; }\n"
)
_JIT_SOURCE_V2 = _JIT_SOURCE_V1.replace("version: 1", "version: 2")


class _PassingSandbox(SandboxBackend):
    def static_check(self, source_code: str) -> ValidationResult:
        return ValidationResult(ok=True)

    async def arun_source(
        self,
        source_code: str,
        args: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        return dict(args)

    def run_tests(
        self,
        source_code: str,
        tests: list[dict[str, Any]],
        timeout: float | None = None,
        *,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
    ) -> ValidationResult:
        return ValidationResult(ok=True, metadata={})


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("policy_change", ["revoke", "deny"])
def test_skill_unload_reauthorizes_at_the_mutation_barrier(
    kind: str,
    policy_change: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_id = f"unload-barrier-{policy_change}"
    skill_dir = write_skill_package(tmp_path, skill_id, allowed_tools=["echo"])
    with _persistent_target(kind, tmp_path, suffix=policy_change) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            runtime.skills.register_skill_from_path(
                skill_dir,
                actor="test.host",
                require_capability=False,
            )
            actor = runtime.process.spawn(goal=f"Skill unload {policy_change} barrier")
            runtime.skills.activate_skill(
                actor,
                skill_id,
                actor=actor,
                require_capability=False,
            )
            authority = runtime.capability.issue_trusted(
                actor,
                f"skill:{skill_id}",
                [CapabilityRight.EXECUTE],
                issued_by="test.host",
            )
            barrier = Barrier(2)
            original_require = runtime.skills._require_skill_right

            def pause_after_preflight(*args: object, **kwargs: object):
                decisions = original_require(*args, **kwargs)
                barrier.wait(timeout=10)
                barrier.wait(timeout=10)
                return decisions

            monkeypatch.setattr(
                runtime.skills,
                "_require_skill_right",
                pause_after_preflight,
            )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    runtime.skills.unload_skill,
                    actor,
                    skill_id,
                    actor=actor,
                )
                barrier.wait(timeout=10)
                if policy_change == "revoke":
                    runtime.capability.revoke(
                        authority.cap_id,
                        revoked_by="test.defender",
                        require_authority=False,
                    )
                else:
                    runtime.capability.issue_trusted(
                        actor,
                        f"skill:{skill_id}",
                        [CapabilityRight.EXECUTE],
                        issued_by="test.defender",
                        effect=CapabilityEffect.DENY,
                    )
                barrier.wait(timeout=10)
                with pytest.raises(CapabilityDenied, match="authority changed"):
                    future.result(timeout=10)

            process = runtime.process.get(actor)
            assert skill_id in process.loaded_skills
            assert "echo" in process.tool_table
            assert not [
                event
                for event in runtime.events.list()
                if event.type == EventType.SKILL_UNLOADED
                and event.target == actor
                and event.payload.get("skill_id") == skill_id
            ]
            assert not [
                record
                for record in runtime.audit.trace()
                if record.action == "skill.unload"
                and record.target == f"process:{actor}"
                and record.decision.get("skill_id") == skill_id
            ]
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("failure", ["event", "audit", "settlement"])
def test_skill_unload_failure_rolls_back_business_evidence_and_authority(
    kind: str,
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_id = f"unload-atomic-{failure}"
    skill_dir = write_skill_package(tmp_path, skill_id, allowed_tools=["echo"])
    with _persistent_target(kind, tmp_path, suffix=failure) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            runtime.skills.register_skill_from_path(
                skill_dir,
                actor="test.host",
                require_capability=False,
            )
            actor = runtime.process.spawn(goal=f"Skill unload {failure} rollback")
            runtime.skills.activate_skill(
                actor,
                skill_id,
                actor=actor,
                require_capability=False,
            )
            authority = runtime.capability.grant_once(
                actor,
                f"skill:{skill_id}",
                [CapabilityRight.EXECUTE],
                issued_by="test.host",
            )
            before_event_ids = {event.event_id for event in runtime.events.list()}
            before_audit_ids = {record.record_id for record in runtime.audit.trace()}
            before_reservations = runtime.store.select_table_rows(
                "capability_use_reservations",
                order_by="reservation_id",
            )

            if failure == "event":
                original_emit = runtime.events.emit

                def fail_after_event(event_type: EventType, *args: object, **kwargs: object):
                    result = original_emit(event_type, *args, **kwargs)
                    if event_type == EventType.SKILL_UNLOADED:
                        raise RuntimeError("injected Skill unload event failure")
                    return result

                monkeypatch.setattr(runtime.events, "emit", fail_after_event)
            elif failure == "audit":
                original_record = runtime.audit.record

                def fail_after_audit(*args: object, **kwargs: object):
                    result = original_record(*args, **kwargs)
                    if kwargs.get("action") == "skill.unload":
                        raise RuntimeError("injected Skill unload audit failure")
                    return result

                monkeypatch.setattr(runtime.audit, "record", fail_after_audit)
            else:
                original_commit = runtime.capability.commit_reserved_use

                def fail_after_settlement(*args: object, **kwargs: object):
                    original_commit(*args, **kwargs)
                    raise RuntimeError("injected Skill unload settlement failure")

                monkeypatch.setattr(
                    runtime.capability,
                    "commit_reserved_use",
                    fail_after_settlement,
                )

            with pytest.raises(RuntimeError, match=f"Skill unload {failure} failure"):
                runtime.skills.unload_skill(actor, skill_id, actor=actor)

            process = runtime.process.get(actor)
            assert skill_id in process.loaded_skills
            assert "echo" in process.tool_table
            persisted = runtime.store.get_capability(authority.cap_id)
            assert persisted is not None
            assert persisted.active
            assert persisted.uses_remaining == 1
            assert runtime.store.select_table_rows(
                "capability_use_reservations",
                order_by="reservation_id",
            ) == before_reservations
            assert not [
                event
                for event in runtime.events.list()
                if event.event_id not in before_event_ids
                and event.type == EventType.SKILL_UNLOADED
            ]
            assert not [
                record
                for record in runtime.audit.trace()
                if record.record_id not in before_audit_ids
                and record.action == "skill.unload"
            ]
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_skill_activation_settlement_failure_restores_exact_jit_registry_state(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_id = "activation-jit-settlement"
    tool_name = "activation_jit_settlement_tool"
    skill_dir = _write_jit_skill(
        tmp_path,
        skill_id=skill_id,
        tool_name=tool_name,
        source=_JIT_SOURCE_V1,
    )
    with _persistent_target(kind, tmp_path, suffix="jit-settlement") as (target, config):
        runtime = Runtime.open(target, config=config)
        runtime.tools.sandbox = _PassingSandbox()
        try:
            runtime.skills.register_skill_from_path(
                skill_dir,
                actor="test.host",
                require_capability=False,
            )
            actor = runtime.process.spawn(goal="Skill activation JIT settlement rollback")
            initial = runtime.skills.activate_skill(
                actor,
                skill_id,
                actor=actor,
                require_capability=False,
            )
            old_tool_id = initial["jit_tool_ids"][tool_name]
            old_handle = runtime.tools.loaded_tool_handle(old_tool_id)
            assert old_handle is not None

            _write_jit_skill(
                tmp_path,
                skill_id=skill_id,
                tool_name=tool_name,
                source=_JIT_SOURCE_V2,
            )
            runtime.skills.register_skill_from_path(
                skill_dir,
                actor="test.host",
                replace=True,
                require_capability=False,
            )
            authority = runtime.capability.grant_once(
                actor,
                f"skill:{skill_id}",
                [CapabilityRight.EXECUTE],
                issued_by="test.host",
            )
            before_candidates = runtime.store.select_table_rows("tool_candidates")
            before_reservations = runtime.store.select_table_rows(
                "capability_use_reservations",
                order_by="reservation_id",
            )
            before_event_ids = {event.event_id for event in runtime.events.list()}
            before_audit_ids = {record.record_id for record in runtime.audit.trace()}
            original_commit = runtime.capability.commit_reserved_use

            def fail_after_settlement(*args: object, **kwargs: object):
                original_commit(*args, **kwargs)
                raise KeyboardInterrupt("injected Skill activation JIT settlement interruption")

            monkeypatch.setattr(
                runtime.capability,
                "commit_reserved_use",
                fail_after_settlement,
            )

            with pytest.raises(
                KeyboardInterrupt,
                match="Skill activation JIT settlement interruption",
            ):
                runtime.skills.activate_skill(actor, skill_id, actor=actor)

            process = runtime.process.get(actor)
            assert process.loaded_skills[skill_id]["jit_tool_ids"][tool_name] == old_tool_id
            assert process.tool_table[tool_name] == old_tool_id
            assert runtime.tools.loaded_tool_handle(old_tool_id) is old_handle
            matching_handles = [
                handle.tool_id
                for handle in runtime.tools.loaded_tool_handles()
                if handle.name == tool_name
            ]
            assert matching_handles == [old_tool_id]
            assert [
                str(row["tool_id"])
                for row in runtime.store.list_tools()
                if row["name"] == tool_name
            ] == [old_tool_id]
            assert runtime.store.select_table_rows("tool_candidates") == before_candidates
            assert runtime.store.select_table_rows(
                "capability_use_reservations",
                order_by="reservation_id",
            ) == before_reservations
            assert not [
                event
                for event in runtime.events.list()
                if event.event_id not in before_event_ids
                and event.type == EventType.SKILL_LOADED
            ]
            assert not [
                record
                for record in runtime.audit.trace()
                if record.record_id not in before_audit_ids
                and record.action == "skill.activate"
            ]
            persisted = runtime.store.get_capability(authority.cap_id)
            assert persisted is not None
            assert persisted.active
            assert persisted.uses_remaining == 1
            old_call = runtime.tools.call(actor, tool_name, {"probe": "old"})
            assert old_call.ok
            assert old_call.payload == {"probe": "old"}
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_workspace_skill_shared_authority_settlement_failure_restores_jit_state(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_id = "workspace-shared-jit-settlement"
    tool_name = "workspace_shared_jit_settlement_tool"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill_dir = _write_jit_skill(
        workspace,
        skill_id=skill_id,
        tool_name=tool_name,
        source=_JIT_SOURCE_V1,
    )
    with _persistent_target(kind, tmp_path, suffix="workspace-shared-jit") as (target, config):
        runtime = Runtime.open(
            target,
            config=config,
            substrate=LocalResourceProviderSubstrate(workspace),
        )
        runtime.tools.sandbox = _PassingSandbox()
        try:
            runtime.skills.register_skill_from_path(
                skill_dir,
                actor="test.host",
                require_capability=False,
            )
            registered_before = runtime.store.get_skill(skill_id)
            assert registered_before is not None
            actor = runtime.process.spawn(goal="Workspace shared Skill JIT rollback")
            initial = runtime.skills.activate_skill(
                actor,
                skill_id,
                actor=actor,
                require_capability=False,
            )
            old_tool_id = initial["jit_tool_ids"][tool_name]
            old_handle = runtime.tools.loaded_tool_handle(old_tool_id)
            assert old_handle is not None

            _write_jit_skill(
                workspace,
                skill_id=skill_id,
                tool_name=tool_name,
                source=_JIT_SOURCE_V2,
            )
            runtime.filesystem.grant_path_list(
                actor,
                read_files=(
                    f"{skill_id}/SKILL.md",
                    f"{skill_id}/references/agent-libos/jit-tools.json",
                    f"{skill_id}/scripts/contract.ts",
                ),
                issued_by="test.host",
            )
            authority = runtime.capability.grant_once(
                actor,
                f"skill:{skill_id}",
                [CapabilityRight.WRITE, CapabilityRight.EXECUTE],
                issued_by="test.host",
            )
            before_candidates = runtime.store.select_table_rows("tool_candidates")
            before_reservations = runtime.store.select_table_rows(
                "capability_use_reservations",
                order_by="reservation_id",
            )
            before_event_ids = {event.event_id for event in runtime.events.list()}
            before_audit_ids = {record.record_id for record in runtime.audit.trace()}
            original_commit = runtime.capability.commit_reserved_use
            settlement_calls = 0

            def fail_after_settlement(*args: object, **kwargs: object):
                nonlocal settlement_calls
                result = original_commit(*args, **kwargs)
                settlement_calls += 1
                raise RuntimeError("injected workspace shared authority settlement failure")

            monkeypatch.setattr(
                runtime.capability,
                "commit_reserved_use",
                fail_after_settlement,
            )

            with pytest.raises(
                RuntimeError,
                match="workspace shared authority settlement failure",
            ):
                runtime.skills.activate_skill_from_workspace_path(
                    actor,
                    skill_id,
                    replace=True,
                )

            assert settlement_calls == 1
            assert runtime.store.get_skill(skill_id) == registered_before
            process = runtime.process.get(actor)
            assert process.loaded_skills[skill_id]["jit_tool_ids"][tool_name] == old_tool_id
            assert process.tool_table[tool_name] == old_tool_id
            assert runtime.tools.loaded_tool_handle(old_tool_id) is old_handle
            assert [
                handle.tool_id
                for handle in runtime.tools.loaded_tool_handles()
                if handle.name == tool_name
            ] == [old_tool_id]
            assert [
                str(row["tool_id"])
                for row in runtime.store.list_tools()
                if row["name"] == tool_name
            ] == [old_tool_id]
            assert runtime.store.select_table_rows("tool_candidates") == before_candidates
            assert runtime.store.select_table_rows(
                "capability_use_reservations",
                order_by="reservation_id",
            ) == before_reservations
            assert not [
                event
                for event in runtime.events.list()
                if event.event_id not in before_event_ids
                and event.type in {EventType.SKILL_REGISTERED, EventType.SKILL_LOADED}
            ]
            assert not [
                record
                for record in runtime.audit.trace()
                if record.record_id not in before_audit_ids
                and record.action in {"skill.register", "skill.activate"}
            ]
            persisted = runtime.store.get_capability(authority.cap_id)
            assert persisted is not None
            assert persisted.active
            assert persisted.uses_remaining == 1
            old_call = runtime.tools.call(actor, tool_name, {"probe": "old"})
            assert old_call.ok
            assert old_call.payload == {"probe": "old"}

            monkeypatch.setattr(
                runtime.capability,
                "commit_reserved_use",
                original_commit,
            )
            replacement = runtime.skills.activate_skill_from_workspace_path(
                actor,
                skill_id,
                replace=True,
            )
            new_tool_id = replacement["jit_tool_ids"][tool_name]
            assert new_tool_id != old_tool_id
            assert runtime.tools.loaded_tool_handle(old_tool_id) is None
            assert [
                handle.tool_id
                for handle in runtime.tools.loaded_tool_handles()
                if handle.name == tool_name
            ] == [new_tool_id]
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_skill_activation_commit_fence_rolls_back_durable_and_jit_state(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_id = "activation-jit-commit-fence"
    tool_name = "activation_jit_commit_fence_tool"
    skill_dir = _write_jit_skill(
        tmp_path,
        skill_id=skill_id,
        tool_name=tool_name,
        source=_JIT_SOURCE_V1,
    )
    with _persistent_target(kind, tmp_path, suffix="jit-commit-fence") as (target, config):
        runtime = Runtime.open(target, config=config)
        runtime.tools.sandbox = _PassingSandbox()
        release_commit = threading.Event()
        worker: threading.Thread | None = None
        fencer: threading.Thread | None = None
        original_guard = runtime.store._admission_commit_guard
        assert original_guard is not None
        try:
            runtime.skills.register_skill_from_path(
                skill_dir,
                actor="test.host",
                require_capability=False,
            )
            actor = runtime.process.spawn(goal="Skill activation recovery fence rollback")
            initial = runtime.skills.activate_skill(
                actor,
                skill_id,
                actor=actor,
                require_capability=False,
            )
            old_tool_id = initial["jit_tool_ids"][tool_name]
            old_handle = runtime.tools.loaded_tool_handle(old_tool_id)
            assert old_handle is not None

            _write_jit_skill(
                tmp_path,
                skill_id=skill_id,
                tool_name=tool_name,
                source=_JIT_SOURCE_V2,
            )
            runtime.skills.register_skill_from_path(
                skill_dir,
                actor="test.host",
                replace=True,
                require_capability=False,
            )
            authority = runtime.capability.grant_once(
                actor,
                f"skill:{skill_id}",
                [CapabilityRight.EXECUTE],
                issued_by="test.host",
            )
            before_tables = {
                table: runtime.store.select_table_rows(table)
                for table in (
                    "audit_records",
                    "capabilities",
                    "capability_use_reservations",
                    "events",
                    "processes",
                    "tool_candidates",
                    "tools",
                )
            }
            before_objects = runtime.store.list_objects()

            original_publish = runtime.tools.registry.publish_jit
            arm_commit_barrier = threading.Event()
            commit_waiting = threading.Event()
            fence_finished = threading.Event()
            worker_thread_id: int | None = None

            def publish_and_arm(handle: object, source: str) -> None:
                original_publish(handle, source)
                if getattr(handle, "name", None) == tool_name:
                    arm_commit_barrier.set()

            monkeypatch.setattr(runtime.tools.registry, "publish_jit", publish_and_arm)

            @contextlib.contextmanager
            def blocked_commit_guard() -> Iterator[None]:
                if (
                    arm_commit_barrier.is_set()
                    and threading.get_ident() == worker_thread_id
                ):
                    commit_waiting.set()
                    assert release_commit.wait(timeout=10)
                with original_guard():
                    yield

            monkeypatch.setattr(
                runtime.store,
                "_admission_commit_guard",
                blocked_commit_guard,
            )
            worker_errors: list[BaseException] = []
            fence_errors: list[BaseException] = []

            def activate() -> None:
                nonlocal worker_thread_id
                worker_thread_id = threading.get_ident()
                try:
                    runtime.skills.activate_skill(actor, skill_id, actor=actor)
                except BaseException as exc:
                    worker_errors.append(exc)

            def fence() -> None:
                try:
                    with runtime.lifecycle.admit():
                        runtime.lifecycle.mark_recovery_required(
                            publication_id="skill-activation-commit-fence",
                        )
                except BaseException as exc:
                    fence_errors.append(exc)
                finally:
                    fence_finished.set()

            worker = threading.Thread(target=activate)
            worker.start()
            assert commit_waiting.wait(timeout=10)
            fencer = threading.Thread(target=fence)
            fencer.start()
            assert fence_finished.wait(timeout=10)
            assert fence_errors == []
            release_commit.set()
            worker.join(timeout=15)
            fencer.join(timeout=15)
            assert not worker.is_alive()
            assert not fencer.is_alive()

            assert len(worker_errors) == 1
            assert isinstance(worker_errors[0], RuntimeError)
            assert "not accepting operations: state=close_failed" in str(worker_errors[0])
            assert {
                table: runtime.store.select_table_rows(table)
                for table in before_tables
            } == before_tables
            assert runtime.store.list_objects() == before_objects
            process = runtime.store.get_process(actor)
            assert process is not None
            assert process.loaded_skills[skill_id]["jit_tool_ids"][tool_name] == old_tool_id
            assert process.tool_table[tool_name] == old_tool_id
            assert runtime.tools.loaded_tool_handle(old_tool_id) is old_handle
            assert [
                handle.tool_id
                for handle in runtime.tools.loaded_tool_handles()
                if handle.name == tool_name
            ] == [old_tool_id]
            persisted = runtime.store.get_capability(authority.cap_id)
            assert persisted is not None
            assert persisted.active
            assert persisted.uses_remaining == 1
        finally:
            release_commit.set()
            if worker is not None and worker.is_alive():
                worker.join(timeout=15)
            if fencer is not None and fencer.is_alive():
                fencer.join(timeout=15)
            monkeypatch.setattr(
                runtime.store,
                "_admission_commit_guard",
                original_guard,
            )
            _release_fenced_runtime_or_close(runtime)

        reopened = Runtime.open(target, config=config)
        reopened.tools.sandbox = _PassingSandbox()
        try:
            process = reopened.process.get(actor)
            assert process.loaded_skills[skill_id]["jit_tool_ids"][tool_name] == old_tool_id
            assert reopened.tools.loaded_tool_handle(old_tool_id) is not None
            old_call = reopened.tools.call(actor, tool_name, {"probe": "old"})
            assert old_call.ok
            assert old_call.payload == {"probe": "old"}
        finally:
            reopened.close()


def _write_jit_skill(
    root: Path,
    *,
    skill_id: str,
    tool_name: str,
    source: str,
) -> Path:
    return write_skill_package(
        root,
        skill_id,
        jit_tools=[
            {
                "name": tool_name,
                "description": "Authority settlement contract JIT tool.",
                "source_path": "scripts/contract.ts",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "tests": [],
            }
        ],
        scripts={"scripts/contract.ts": source},
    )


@contextlib.contextmanager
def _persistent_target(
    kind: str,
    tmp_path: Path,
    *,
    suffix: str,
) -> Iterator[tuple[str | Path, AgentLibOSConfig]]:
    if kind == "sqlite-file":
        yield tmp_path / f"authority-skill-{suffix}.sqlite", AgentLibOSConfig()
        return
    if kind == "postgres":
        with _postgres_schema_dsn() as dsn:
            yield dsn, AgentLibOSConfig(
                runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
            )
        return
    raise AssertionError(f"unknown persistent backend: {kind}")


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_authority_skill_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield _dsn_with_search_path(dsn, schema)
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


def _dsn_with_search_path(dsn: str, schema: str) -> str:
    parsed = urlsplit(dsn)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "options"
    ]
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )
