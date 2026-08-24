from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG
from agent_libos.mcp.continuations import McpContinuationManager
from agent_libos.mcp.human import HumanObjectManagerMcpBridge
from agent_libos.mcp.oauth import InMemoryMcpCredentialBroker
from agent_libos.mcp.subscriptions import McpSubscriptionManager
from agent_libos.mcp.tasks import McpRemoteTaskManager
from agent_libos.models import CapabilityRight
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.human import HumanRequestStatus
from agent_libos.runtime.lifecycle import RuntimeLifecycle
from agent_libos.runtime.runtime import Runtime
from agent_libos.runtime.task_runs import TaskRunManager
from agent_libos.substrate.local import LocalResourceProviderSubstrate

from tests.unit.test_mcp_v3_continuations import (
    _Boundary as _ContinuationBoundary,
    _binding as _continuation_binding,
    _input_required,
)
from tests.unit.test_mcp_v3_tasks import (
    _TASKS_DIGEST,
    _Boundary as _TaskBoundary,
    _binding as _task_binding,
    _sha,
    _task_result,
)


_TASK_CLOCK = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=5)


class _SimulatedProcessDeath(BaseException):
    """Bypass manager exception cleanup exactly as an abrupt process exit does."""


class _FaultBroker(InMemoryMcpCredentialBroker):
    def __init__(self) -> None:
        super().__init__()
        self.crash_before_put = False
        self.fail_delete_refs: set[str] = set()
        self.live_refs: set[str] = set()
        self.on_delete: Any | None = None

    def put_secret_at(
        self,
        secret_ref: str,
        namespace: str,
        value: bytes,
        *,
        expires_at: str | None,
    ) -> None:
        if self.crash_before_put:
            raise _SimulatedProcessDeath("after Human, before broker write")
        super().put_secret_at(
            secret_ref,
            namespace,
            value,
            expires_at=expires_at,
        )
        self.live_refs.add(secret_ref)

    def delete_secret(self, secret_ref: str) -> None:
        if self.on_delete is not None:
            self.on_delete(secret_ref)
        if secret_ref in self.fail_delete_refs:
            raise RuntimeError("injected credential broker delete failure")
        super().delete_secret(secret_ref)
        self.live_refs.discard(secret_ref)


class _CommitFaultRepository:
    def __init__(
        self,
        delegate: Any,
        *,
        crash_before_commit: bool = False,
        crash_after_commit: bool = False,
        crash_after_terminal_commit: bool = False,
    ) -> None:
        self.delegate = delegate
        self.crash_before_commit = crash_before_commit
        self.crash_after_commit = crash_after_commit
        self.crash_after_terminal_commit = crash_after_terminal_commit

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def commit(self, *args: Any, **kwargs: Any) -> bool:
        if self.crash_before_commit:
            raise _SimulatedProcessDeath("after broker, before main insert")
        committed = self.delegate.commit(*args, **kwargs)
        if committed and self.crash_after_commit:
            raise _SimulatedProcessDeath("after atomic main commit")
        return committed

    def commit_terminal(self, *args: Any, **kwargs: Any) -> bool:
        committed = self.delegate.commit_terminal(*args, **kwargs)
        if committed and self.crash_after_terminal_commit:
            raise _SimulatedProcessDeath("after atomic terminal retirement")
        return committed


class _CasCrashRepository:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.crashed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def compare_and_swap(self, *args: Any, **kwargs: Any) -> bool:
        committed = self.delegate.compare_and_swap(*args, **kwargs)
        if committed and not self.crashed:
            self.crashed = True
            raise _SimulatedProcessDeath("after dispatch claim commit")
        return committed


class _DispatchCrashBoundary(_TaskBoundary):
    async def update_remote_task(self, **kwargs: Any) -> dict[str, Any]:
        self.update_calls.append(kwargs)
        raise _SimulatedProcessDeath("update dispatched; outcome unknown")

    async def cancel_remote_task(self, **kwargs: Any) -> dict[str, Any]:
        self.cancel_calls.append(kwargs)
        raise _SimulatedProcessDeath("cancel dispatched; outcome unknown")


class _ContinuationCancelBoundary(_ContinuationBoundary):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_calls = 0

    async def cancel_continuation(self, **_kwargs: Any) -> None:
        self.cancel_calls += 1


def _config(*, terminal_records: int = 256) -> AgentLibOSConfig:
    return AgentLibOSConfig(
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            tasks_extension_enabled=True,
            tasks_extension_spec_sha256=_TASKS_DIGEST,
            continuation_terminal_records=terminal_records,
            remote_task_terminal_records=terminal_records,
        )
    )


def _substrate(root: Path, broker: _FaultBroker) -> Any:
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    substrate = LocalResourceProviderSubstrate(workspace)
    substrate.mcp_credential_broker = broker
    return substrate


def _open_runtime(
    database: Path,
    root: Path,
    broker: _FaultBroker,
    *,
    terminal_records: int = 256,
) -> Runtime:
    return Runtime.open(
        database,
        substrate=_substrate(root, broker),
        config=_config(terminal_records=terminal_records),
    )


def _spawn_human_owner(runtime: Runtime) -> str:
    return runtime.process.spawn(
        image="base-agent:v0",
        goal="exercise MCP crash recovery",
        authority_manifest={
            "authorized_capabilities": [
                {
                    "resource": "human:owner",
                    "rights": [CapabilityRight.WRITE.value],
                }
            ]
        },
    )


def _task_input_required(
    task_id: str,
    *,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    selected_created = created_at or _timestamp(_TASK_CLOCK - timedelta(seconds=10))
    selected_updated = updated_at or _timestamp(_TASK_CLOCK - timedelta(seconds=1))
    return _task_result(
        status="input_required",
        task_id=task_id,
        resultType="task",
        createdAt=selected_created,
        lastUpdatedAt=selected_updated,
        ttlMs=3_600_000,
        pollIntervalMs=0,
        inputRequests={
            "remote-input-key": {
                "method": "elicitation/create",
                "params": {
                    "mode": "form",
                    "message": "Approve the remote Task?",
                    "requestedSchema": {
                        "type": "object",
                        "properties": {"approved": {"type": "boolean"}},
                        "required": ["approved"],
                    },
                },
            }
        },
    )


def _working_task(
    task_id: str,
    *,
    created_at: str,
    updated_at: str,
) -> dict[str, Any]:
    return _task_result(
        task_id=task_id,
        createdAt=created_at,
        lastUpdatedAt=updated_at,
        ttlMs=3_600_000,
        pollIntervalMs=0,
    )


def _completed_task(
    task_id: str,
    *,
    created_at: str,
    updated_at: str,
) -> dict[str, Any]:
    return _task_result(
        status="completed",
        task_id=task_id,
        createdAt=created_at,
        lastUpdatedAt=updated_at,
        ttlMs=3_600_000,
        pollIntervalMs=0,
        result={"ok": True},
    )


def _preparation_refs(preparation: Any) -> set[str]:
    return {
        reference
        for reference in (preparation.broker_ref, preparation.result_ref)
        if reference is not None
    }


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _assert_not_answerable(runtime: Runtime, request_id: str) -> None:
    assert runtime.human.get(request_id).status is not HumanRequestStatus.PENDING


def _assert_store_files_exclude(database: Path, *secrets: bytes) -> None:
    candidates = [database, *database.parent.glob(f"{database.name}-*")]
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = candidate.read_bytes()
        for secret in secrets:
            assert secret not in payload, (candidate, secret)


@pytest.mark.parametrize("operation", ["continuation", "remote_task"])
@pytest.mark.parametrize(
    "kill_point",
    ["after_human_before_broker", "after_broker_before_main_insert"],
)
def test_initial_side_effect_preparation_is_aborted_on_sqlite_reopen(
    tmp_path: Path,
    operation: str,
    kill_point: str,
) -> None:
    database = tmp_path / f"initial-{operation}-{kill_point}.sqlite"
    broker = _FaultBroker()
    initial = _open_runtime(database, tmp_path, broker)
    preparation: Any
    try:
        owner_id = _spawn_human_owner(initial)
        if operation == "continuation":
            manager = initial._mcp_continuation_manager
            invoke = lambda: manager.capture_input_required(
                _continuation_binding(owner_id=owner_id),
                _input_required(state="CONTINUATION-KILLPOINT-STATE"),
                expires_at=None,
            )
        else:
            manager = initial._mcp_remote_task_manager
            assert manager is not None
            manager._now = lambda: _TASK_CLOCK  # noqa: SLF001
            invoke = lambda: manager.capture_task(
                _task_binding(owner_id=owner_id),
                _task_input_required("REMOTE-TASK-KILLPOINT-BEARER"),
            )
        if kill_point == "after_human_before_broker":
            broker.crash_before_put = True
        else:
            manager.side_effects = _CommitFaultRepository(
                initial.uow.mcp_side_effects,
                crash_before_commit=True,
            )

        with pytest.raises(_SimulatedProcessDeath):
            invoke()

        preparations = initial.uow.mcp_side_effects.list()
        assert len(preparations) == 1
        preparation = preparations[0]
        assert preparation.operation_kind == operation
        assert preparation.status == "prepared"
        assert preparation.human_request_id is not None
        assert (
            initial.human.get(preparation.human_request_id).status
            is HumanRequestStatus.PENDING
        )
        refs = _preparation_refs(preparation)
        if kill_point == "after_human_before_broker":
            assert refs.isdisjoint(broker.live_refs)
        else:
            assert refs <= broker.live_refs
        repository = (
            initial.uow.mcp_continuations
            if operation == "continuation"
            else initial.uow.mcp_remote_tasks
        )
        assert repository.get(preparation.operation_id) is None
    finally:
        broker.crash_before_put = False
        initial.close()

    reopened = _open_runtime(database, tmp_path, broker)
    try:
        recovered = (
            reopened.recovered_mcp_continuations
            if operation == "continuation"
            else reopened.recovered_mcp_remote_tasks
        )
        assert recovered == 1
        assert reopened.uow.mcp_side_effects.list() == ()
        repository = (
            reopened.uow.mcp_continuations
            if operation == "continuation"
            else reopened.uow.mcp_remote_tasks
        )
        assert repository.get(preparation.operation_id) is None
        assert preparation.human_request_id is not None
        _assert_not_answerable(reopened, preparation.human_request_id)
        assert _preparation_refs(preparation).isdisjoint(broker.live_refs)
    finally:
        reopened.close()

    _assert_store_files_exclude(
        database,
        b"CONTINUATION-KILLPOINT-STATE",
        b"REMOTE-TASK-KILLPOINT-BEARER",
    )


@pytest.mark.parametrize("failure_mode", ["post_commit_crash", "broker_delete_failure"])
def test_followup_commit_failure_retires_only_old_refs_on_sqlite_reopen(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    database = tmp_path / "continuation-followup-commit.sqlite"
    broker = _FaultBroker()
    boundary = _ContinuationBoundary(
        _input_required(state="FOLLOWUP-KILLPOINT-STATE", message="One more value")
    )
    initial = _open_runtime(database, tmp_path, broker)
    try:
        owner_id = _spawn_human_owner(initial)
        manager = initial._mcp_continuation_manager
        manager.boundary = boundary
        first = manager.capture_input_required(
            _continuation_binding(owner_id=owner_id),
            _input_required(state="INITIAL-KILLPOINT-STATE"),
            expires_at=None,
        )
        first_record = initial.uow.mcp_continuations.get(first.continuation_id)
        assert first_record is not None and first_record.broker_ref is not None
        bridge = manager.human_requests
        assert isinstance(bridge, HumanObjectManagerMcpBridge)
        bridge.settle_answer(
            first.human_request_id,
            {"input-1": {"action": "decline"}},
            expected_revision=first.human_revision,
            preview_sha256=first.human_preview_sha256,
        )
        if failure_mode == "post_commit_crash":
            manager.side_effects = _CommitFaultRepository(
                initial.uow.mcp_side_effects,
                crash_after_commit=True,
            )
            with pytest.raises(
                _SimulatedProcessDeath,
                match="after atomic main commit",
            ):
                asyncio.run(
                    manager.respond(
                        first.continuation_id,
                        expected_revision=first.revision,
                        binding=_continuation_binding(owner_id=owner_id),
                        human_request_id=first.human_request_id,
                        human_expected_revision=first.human_revision,
                        human_preview_sha256=first.human_preview_sha256,
                        deadline=100.0,
                    )
                )
        else:
            broker.fail_delete_refs.add(first_record.broker_ref)
            public = asyncio.run(
                manager.respond(
                    first.continuation_id,
                    expected_revision=first.revision,
                    binding=_continuation_binding(owner_id=owner_id),
                    human_request_id=first.human_request_id,
                    human_expected_revision=first.human_revision,
                    human_preview_sha256=first.human_preview_sha256,
                    deadline=100.0,
                )
            )
            assert public.continuation_id == first.continuation_id

        committed = initial.uow.mcp_continuations.get(first.continuation_id)
        assert committed is not None
        assert committed.status == "input_required"
        assert committed.revision == first.revision + 2
        assert committed.broker_ref is not None
        assert committed.broker_ref != first_record.broker_ref
        retirements = initial.uow.mcp_side_effects.list(status="cleaning")
        assert len(retirements) == 1
        assert retirements[0].metadata["retire_refs"] == (
            first_record.broker_ref,
        )
        assert first_record.broker_ref in broker.live_refs
        assert committed.broker_ref in broker.live_refs
        assert committed.human_request_id != first_record.human_request_id
        assert (
            initial.human.get(committed.human_request_id).status
            is HumanRequestStatus.PENDING
        )
    finally:
        broker.fail_delete_refs.clear()
        initial.close()

    reopened = _open_runtime(database, tmp_path, broker)
    try:
        assert reopened.recovered_mcp_continuations == 1
        assert reopened.uow.mcp_side_effects.list() == ()
        recovered = reopened.uow.mcp_continuations.get(first.continuation_id)
        assert recovered == committed
        assert first_record.broker_ref not in broker.live_refs
        assert committed.broker_ref in broker.live_refs
        assert broker.get_secret(committed.broker_ref)
        _assert_not_answerable(reopened, first_record.human_request_id)
        assert (
            reopened.human.get(committed.human_request_id).status
            is HumanRequestStatus.PENDING
        )
        assert recovered.human_request_id == committed.human_request_id
        assert len(boundary.calls) == 1
    finally:
        reopened.close()

    _assert_store_files_exclude(
        database,
        b"INITIAL-KILLPOINT-STATE",
        b"FOLLOWUP-KILLPOINT-STATE",
    )


def test_terminal_transition_precommit_crash_preserves_active_refs_and_human(
    tmp_path: Path,
) -> None:
    database = tmp_path / "task-terminal-precommit.sqlite"
    broker = _FaultBroker()
    boundary = _TaskBoundary()
    initial = _open_runtime(database, tmp_path, broker)
    try:
        owner_id = _spawn_human_owner(initial)
        manager = initial._mcp_remote_task_manager
        assert manager is not None
        manager.boundary = boundary
        manager._now = lambda: _TASK_CLOCK  # noqa: SLF001
        binding = _task_binding(owner_id=owner_id)
        pending = manager.capture_task(
            binding,
            _task_input_required("REMOTE-TERMINAL-PRECOMMIT"),
        )
        before = initial.uow.mcp_remote_tasks.get(pending.task_ref)
        assert before is not None
        old_refs = {
            reference
            for reference in (before.broker_ref, before.result_ref)
            if reference is not None
        }
        boundary.get_results.append(
            _completed_task(
                "REMOTE-TERMINAL-PRECOMMIT",
                created_at=_timestamp(_TASK_CLOCK - timedelta(seconds=10)),
                updated_at=_timestamp(_TASK_CLOCK),
            )
        )
        manager._now = lambda: _TASK_CLOCK + timedelta(seconds=1)  # noqa: SLF001
        manager.side_effects = _CommitFaultRepository(
            initial.uow.mcp_side_effects,
            crash_before_commit=True,
        )

        with pytest.raises(_SimulatedProcessDeath, match="before main insert"):
            asyncio.run(
                manager.get(
                    pending.task_ref,
                    expected_revision=pending.revision,
                    binding=binding,
                    deadline=100.0,
                )
            )

        assert initial.uow.mcp_remote_tasks.get(pending.task_ref) == before
        preparations = initial.uow.mcp_side_effects.list(status="prepared")
        assert len(preparations) == 1
        preparation = preparations[0]
        assert set(preparation.metadata["retire_refs"]) == old_refs
        assert preparation.metadata["retire_human_request_id"] == pending.human_request_id
        assert preparation.result_ref is not None
        assert preparation.result_ref in broker.live_refs
        assert old_refs <= broker.live_refs
        assert (
            initial.human.get(pending.human_request_id).status
            is HumanRequestStatus.PENDING
        )
    finally:
        initial.close()

    reopened = _open_runtime(database, tmp_path, broker)
    try:
        assert reopened.recovered_mcp_remote_tasks == 1
        assert reopened.uow.mcp_side_effects.list() == ()
        recovered = reopened.uow.mcp_remote_tasks.get(pending.task_ref)
        assert recovered == before
        assert old_refs <= broker.live_refs
        assert preparation.result_ref not in broker.live_refs
        assert (
            reopened.human.get(pending.human_request_id).status
            is HumanRequestStatus.PENDING
        )
        assert recovered.human_request_id == pending.human_request_id
        assert len(boundary.get_calls) == 1
    finally:
        reopened.close()

    _assert_store_files_exclude(database, b"REMOTE-TERMINAL-PRECOMMIT")


def test_continuation_terminal_cap_prune_crash_reopens_exact_retirement(
    tmp_path: Path,
) -> None:
    database = tmp_path / "continuation-terminal-prune.sqlite"
    broker = _FaultBroker()
    boundary = _ContinuationBoundary(
        {"resultType": "complete", "round": "a"},
        {"resultType": "complete", "round": "b"},
    )
    initial = _open_runtime(database, tmp_path, broker, terminal_records=1)
    try:
        owner_id = _spawn_human_owner(initial)
        manager = initial._mcp_continuation_manager
        manager.boundary = boundary
        bridge = manager.human_requests
        assert isinstance(bridge, HumanObjectManagerMcpBridge)

        binding_a = _continuation_binding(
            owner_id=owner_id,
            effect_id="terminal-continuation-a",
        )
        pending_a = manager.capture_input_required(
            binding_a,
            _input_required(state="TERMINAL-CONTINUATION-A"),
            expires_at=None,
        )
        bridge.settle_answer(
            pending_a.human_request_id,
            {"input-1": {"action": "decline"}},
            expected_revision=pending_a.human_revision,
            preview_sha256=pending_a.human_preview_sha256,
        )
        asyncio.run(
            manager.respond(
                pending_a.continuation_id,
                expected_revision=pending_a.revision,
                binding=binding_a,
                human_request_id=pending_a.human_request_id,
                human_expected_revision=pending_a.human_revision,
                human_preview_sha256=pending_a.human_preview_sha256,
                deadline=100.0,
            )
        )
        retained_a = initial.uow.mcp_continuations.get(pending_a.continuation_id)
        assert retained_a is not None and retained_a.status == "complete"

        binding_b = _continuation_binding(
            owner_id=owner_id,
            effect_id="terminal-continuation-b",
        )
        pending_b = manager.capture_input_required(
            binding_b,
            _input_required(state="TERMINAL-CONTINUATION-B"),
            expires_at=None,
        )
        bridge.settle_answer(
            pending_b.human_request_id,
            {"input-1": {"action": "decline"}},
            expected_revision=pending_b.human_revision,
            preview_sha256=pending_b.human_preview_sha256,
        )
        manager.side_effects = _CommitFaultRepository(
            initial.uow.mcp_side_effects,
            crash_after_terminal_commit=True,
        )

        with pytest.raises(
            _SimulatedProcessDeath,
            match="after atomic terminal retirement",
        ):
            asyncio.run(
                manager.respond(
                    pending_b.continuation_id,
                    expected_revision=pending_b.revision,
                    binding=binding_b,
                    human_request_id=pending_b.human_request_id,
                    human_expected_revision=pending_b.human_revision,
                    human_preview_sha256=pending_b.human_preview_sha256,
                    deadline=101.0,
                )
            )

        assert initial.uow.mcp_continuations.get(pending_a.continuation_id) is None
        retained_b = initial.uow.mcp_continuations.get(pending_b.continuation_id)
        assert retained_b is not None and retained_b.status == "complete"
        retirements = initial.uow.mcp_side_effects.list(status="cleaning")
        assert len(retirements) == 1
        assert retirements[0].operation_id == pending_a.continuation_id
        assert len(boundary.calls) == 2
    finally:
        initial.close()

    reopened = _open_runtime(database, tmp_path, broker, terminal_records=1)
    try:
        assert reopened.recovered_mcp_continuations == 1
        assert reopened.uow.mcp_side_effects.list() == ()
        assert reopened.uow.mcp_continuations.get(pending_a.continuation_id) is None
        assert reopened.uow.mcp_continuations.get(pending_b.continuation_id) == retained_b
        _assert_not_answerable(reopened, pending_a.human_request_id)
        _assert_not_answerable(reopened, pending_b.human_request_id)
        assert len(boundary.calls) == 2
    finally:
        reopened.close()

    _assert_store_files_exclude(
        database,
        b"TERMINAL-CONTINUATION-A",
        b"TERMINAL-CONTINUATION-B",
    )


def test_continuation_dispatch_claim_crash_reopens_attention_without_replay(
    tmp_path: Path,
) -> None:
    database = tmp_path / "continuation-dispatch-claim.sqlite"
    broker = _FaultBroker()
    boundary = _ContinuationCancelBoundary()
    initial = _open_runtime(database, tmp_path, broker)
    try:
        owner_id = _spawn_human_owner(initial)
        manager = initial._mcp_continuation_manager
        manager.boundary = boundary
        binding = _continuation_binding(owner_id=owner_id)
        pending = manager.capture_input_required(
            binding,
            _input_required(state="CONTINUATION-DISPATCH-STATE"),
            expires_at=None,
        )
        before = initial.uow.mcp_continuations.get(pending.continuation_id)
        assert before is not None and before.broker_ref is not None
        manager.repository = _CasCrashRepository(initial.uow.mcp_continuations)

        with pytest.raises(_SimulatedProcessDeath, match="dispatch claim"):
            asyncio.run(
                manager.cancel(
                    pending.continuation_id,
                    expected_revision=pending.revision,
                    binding=binding,
                    deadline=100.0,
                )
            )

        interrupted = initial.uow.mcp_continuations.get(pending.continuation_id)
        assert interrupted is not None and interrupted.status == "dispatching"
        assert interrupted.broker_ref == before.broker_ref
        assert before.broker_ref in broker.live_refs
        assert (
            initial.human.get(pending.human_request_id).status
            is HumanRequestStatus.PENDING
        )
        assert boundary.cancel_calls == 0
    finally:
        initial.close()

    reopened = _open_runtime(database, tmp_path, broker)
    try:
        assert reopened.recovered_mcp_continuations == 1
        recovered = reopened.uow.mcp_continuations.get(pending.continuation_id)
        assert recovered is not None and recovered.status == "needs_attention"
        assert recovered.broker_ref is None
        assert recovered.metadata["automatic_retry_disabled"] is True
        assert recovered.metadata["dispatch_state"] == "unknown"
        assert before.broker_ref not in broker.live_refs
        assert reopened.uow.mcp_side_effects.list() == ()
        _assert_not_answerable(reopened, pending.human_request_id)
        assert boundary.cancel_calls == 0
    finally:
        reopened.close()

    _assert_store_files_exclude(database, b"CONTINUATION-DISPATCH-STATE")


def test_continuation_expiry_broker_failure_reopens_exact_terminal_state(
    tmp_path: Path,
) -> None:
    database = tmp_path / "continuation-expiry-retirement.sqlite"
    broker = _FaultBroker()
    boundary = _ContinuationCancelBoundary()
    initial = _open_runtime(database, tmp_path, broker)
    try:
        owner_id = _spawn_human_owner(initial)
        manager = initial._mcp_continuation_manager
        manager.boundary = boundary
        pending = manager.capture_input_required(
            _continuation_binding(owner_id=owner_id),
            _input_required(state="CONTINUATION-EXPIRY-STATE"),
            expires_at=None,
        )
        before = initial.uow.mcp_continuations.get(pending.continuation_id)
        assert before is not None and before.broker_ref is not None
        manager._now = lambda: (  # noqa: SLF001
            datetime.fromisoformat(before.expires_at) + timedelta(seconds=1)
        )
        broker.fail_delete_refs.add(before.broker_ref)

        with pytest.raises(ValidationError, match="broker cleanup failed"):
            manager._reconcile_expired(limit=500)  # noqa: SLF001

        committed = initial.uow.mcp_continuations.get(pending.continuation_id)
        assert committed is not None and committed.status == "expired"
        assert committed.broker_ref is None
        retirements = initial.uow.mcp_side_effects.list(status="cleaning")
        assert len(retirements) == 1
        assert retirements[0].operation_id == pending.continuation_id
        assert before.broker_ref in broker.live_refs
        assert (
            initial.human.get(pending.human_request_id).status
            is HumanRequestStatus.PENDING
        )
        assert boundary.cancel_calls == 0
    finally:
        broker.fail_delete_refs.clear()
        initial.close()

    reopened = _open_runtime(database, tmp_path, broker)
    try:
        assert reopened.recovered_mcp_continuations == 1
        assert reopened.uow.mcp_continuations.get(pending.continuation_id) == committed
        assert reopened.uow.mcp_side_effects.list() == ()
        assert before.broker_ref not in broker.live_refs
        _assert_not_answerable(reopened, pending.human_request_id)
        assert boundary.cancel_calls == 0
    finally:
        reopened.close()

    _assert_store_files_exclude(database, b"CONTINUATION-EXPIRY-STATE")


def test_terminal_retirement_broker_failure_reopens_without_orphan_or_replay(
    tmp_path: Path,
) -> None:
    database = tmp_path / "task-terminal-retirement.sqlite"
    broker = _FaultBroker()
    boundary = _TaskBoundary()
    initial = _open_runtime(database, tmp_path, broker, terminal_records=1)
    try:
        owner_id = _spawn_human_owner(initial)
        manager = initial._mcp_remote_task_manager
        assert manager is not None
        manager.boundary = boundary
        clock = [_TASK_CLOCK]
        manager._now = lambda: clock[0]  # noqa: SLF001

        task_a_clock = clock[0]
        binding_a = _task_binding(
            owner_id=owner_id,
            origin_request_sha256=_sha("terminal-a-request"),
            origin_effect_id="terminal-a-effect",
        )
        task_a = manager.capture_task(
            binding_a,
            _working_task(
                "REMOTE-TERMINAL-A",
                created_at=_timestamp(task_a_clock - timedelta(seconds=10)),
                updated_at=_timestamp(task_a_clock - timedelta(seconds=1)),
            ),
        )
        boundary.get_results.append(
            _completed_task(
                "REMOTE-TERMINAL-A",
                created_at=_timestamp(task_a_clock - timedelta(seconds=10)),
                updated_at=_timestamp(task_a_clock),
            )
        )
        clock[0] += timedelta(seconds=1)
        asyncio.run(
            manager.get(
                task_a.task_ref,
                expected_revision=task_a.revision,
                binding=binding_a,
                deadline=100.0,
            )
        )
        retained_a = initial.uow.mcp_remote_tasks.get(task_a.task_ref)
        assert retained_a is not None and retained_a.result_ref is not None

        clock[0] += timedelta(minutes=1)
        task_b_clock = clock[0]
        binding_b = _task_binding(
            owner_id=owner_id,
            origin_request_sha256=_sha("terminal-b-request"),
            origin_effect_id="terminal-b-effect",
        )
        task_b = manager.capture_task(
            binding_b,
            _working_task(
                "REMOTE-TERMINAL-B",
                created_at=_timestamp(task_b_clock - timedelta(seconds=10)),
                updated_at=_timestamp(task_b_clock - timedelta(seconds=1)),
            ),
        )
        boundary.get_results.append(
            _completed_task(
                "REMOTE-TERMINAL-B",
                created_at=_timestamp(task_b_clock - timedelta(seconds=10)),
                updated_at=_timestamp(task_b_clock),
            )
        )
        clock[0] += timedelta(seconds=1)
        broker.fail_delete_refs.add(retained_a.result_ref)

        with pytest.raises(ValidationError, match="broker cleanup failed"):
            asyncio.run(
                manager.get(
                    task_b.task_ref,
                    expected_revision=task_b.revision,
                    binding=binding_b,
                    deadline=101.0,
                )
            )

        assert initial.uow.mcp_remote_tasks.get(task_a.task_ref) is None
        retained_b = initial.uow.mcp_remote_tasks.get(task_b.task_ref)
        assert retained_b is not None and retained_b.status == "completed"
        retirements = initial.uow.mcp_side_effects.list(status="cleaning")
        assert len(retirements) == 1
        assert retirements[0].operation_id == task_a.task_ref
        assert retained_a.result_ref in broker.live_refs
    finally:
        broker.fail_delete_refs.clear()
        initial.close()

    reopened = _open_runtime(database, tmp_path, broker, terminal_records=1)
    try:
        assert reopened.recovered_mcp_remote_tasks == 1
        assert reopened.uow.mcp_side_effects.list() == ()
        assert reopened.uow.mcp_remote_tasks.get(task_a.task_ref) is None
        assert reopened.uow.mcp_remote_tasks.get(task_b.task_ref) == retained_b
        assert retained_a.result_ref not in broker.live_refs
        assert retained_b.result_ref in broker.live_refs
        assert len(boundary.get_calls) == 2
    finally:
        reopened.close()

    _assert_store_files_exclude(
        database,
        b"REMOTE-TERMINAL-A",
        b"REMOTE-TERMINAL-B",
    )


def test_expiry_commit_broker_failure_reopens_needs_attention_and_cleans_human(
    tmp_path: Path,
) -> None:
    database = tmp_path / "task-expiry-retirement.sqlite"
    broker = _FaultBroker()
    boundary = _TaskBoundary()
    initial = _open_runtime(database, tmp_path, broker)
    try:
        owner_id = _spawn_human_owner(initial)
        manager = initial._mcp_remote_task_manager
        assert manager is not None
        manager.boundary = boundary
        manager._now = lambda: _TASK_CLOCK  # noqa: SLF001
        pending = manager.capture_task(
            _task_binding(owner_id=owner_id),
            _task_input_required("REMOTE-EXPIRY-BEARER"),
        )
        before = initial.uow.mcp_remote_tasks.get(pending.task_ref)
        assert before is not None
        refs = {
            reference
            for reference in (before.broker_ref, before.result_ref)
            if reference is not None
        }
        assert refs <= broker.live_refs
        manager._now = lambda: (  # noqa: SLF001
            datetime.fromisoformat(before.expires_at) + timedelta(seconds=1)
        )
        broker.fail_delete_refs.update(refs)

        with pytest.raises(ValidationError, match="broker cleanup failed"):
            manager._reconcile_expired(limit=500)  # noqa: SLF001

        committed = initial.uow.mcp_remote_tasks.get(pending.task_ref)
        assert committed is not None
        assert committed.status == "needs_attention"
        assert committed.broker_ref is None
        assert committed.result_ref is None
        retirements = initial.uow.mcp_side_effects.list(status="cleaning")
        assert len(retirements) == 1
        assert retirements[0].operation_id == pending.task_ref
        assert (
            initial.human.get(pending.human_request_id).status
            is HumanRequestStatus.PENDING
        )
        assert boundary.get_calls == boundary.update_calls == boundary.cancel_calls == []
    finally:
        broker.fail_delete_refs.clear()
        initial.close()

    reopened = _open_runtime(database, tmp_path, broker)
    try:
        assert reopened.recovered_mcp_remote_tasks == 1
        recovered = reopened.uow.mcp_remote_tasks.get(pending.task_ref)
        assert recovered == committed
        assert recovered.metadata["reason_code"] == "expired"
        assert reopened.uow.mcp_side_effects.list() == ()
        assert refs.isdisjoint(broker.live_refs)
        _assert_not_answerable(reopened, pending.human_request_id)
        assert boundary.get_calls == boundary.update_calls == boundary.cancel_calls == []
    finally:
        reopened.close()

    _assert_store_files_exclude(database, b"REMOTE-EXPIRY-BEARER")


@pytest.mark.parametrize("operation", ["update", "cancel"])
def test_interrupted_task_mutation_reopens_needs_attention_without_replay(
    tmp_path: Path,
    operation: str,
) -> None:
    database = tmp_path / f"task-{operation}-dispatch.sqlite"
    broker = _FaultBroker()
    boundary = _DispatchCrashBoundary()
    initial = _open_runtime(database, tmp_path, broker)
    try:
        owner_id = _spawn_human_owner(initial)
        manager = initial._mcp_remote_task_manager
        assert manager is not None
        manager.boundary = boundary
        manager._now = lambda: _TASK_CLOCK  # noqa: SLF001
        binding = _task_binding(owner_id=owner_id)
        pending = manager.capture_task(
            binding,
            _task_input_required(f"REMOTE-{operation.upper()}-BEARER"),
        )
        before = initial.uow.mcp_remote_tasks.get(pending.task_ref)
        assert before is not None
        refs = {
            reference
            for reference in (before.broker_ref, before.result_ref)
            if reference is not None
        }
        assert refs <= broker.live_refs
        if operation == "update":
            bridge = manager.human_requests
            assert isinstance(bridge, HumanObjectManagerMcpBridge)
            bridge.settle_answer(
                pending.human_request_id,
                {
                    "input-1": {
                        "action": "accept",
                        "content": {"approved": True},
                    }
                },
                expected_revision=pending.human_revision,
                preview_sha256=pending.human_preview_sha256,
            )
            invoke = lambda: asyncio.run(
                manager.update(
                    pending.task_ref,
                    expected_revision=pending.revision,
                    binding=binding,
                    human_request_id=pending.human_request_id,
                    human_expected_revision=pending.human_revision,
                    human_preview_sha256=pending.human_preview_sha256,
                    deadline=100.0,
                )
            )
        else:
            invoke = lambda: asyncio.run(
                manager.cancel(
                    pending.task_ref,
                    expected_revision=pending.revision,
                    binding=binding,
                    deadline=100.0,
                )
            )

        with pytest.raises(_SimulatedProcessDeath):
            invoke()
        interrupted = initial.uow.mcp_remote_tasks.get(pending.task_ref)
        assert interrupted is not None
        assert interrupted.status == f"{operation}_dispatching"
        assert initial.uow.mcp_side_effects.list() == ()
    finally:
        initial.close()

    reopened = _open_runtime(database, tmp_path, broker)
    try:
        assert reopened.recovered_mcp_remote_tasks == 1
        recovered = reopened.uow.mcp_remote_tasks.get(pending.task_ref)
        assert recovered is not None
        assert recovered.status == "needs_attention"
        assert recovered.metadata["automatic_retry_disabled"] is True
        assert recovered.metadata["dispatch_state"] == "unknown"
        assert recovered.broker_ref is None
        assert recovered.result_ref is None
        assert refs.isdisjoint(broker.live_refs)
        assert reopened.uow.mcp_side_effects.list() == ()
        _assert_not_answerable(reopened, pending.human_request_id)

        manager = reopened._mcp_remote_task_manager
        assert manager is not None
        manager.boundary = boundary
        with pytest.raises(ValidationError, match="current state"):
            asyncio.run(
                manager.get(
                    pending.task_ref,
                    expected_revision=recovered.revision,
                    binding=binding,
                    deadline=101.0,
                )
            )
        calls = boundary.update_calls if operation == "update" else boundary.cancel_calls
        assert len(calls) == 1
    finally:
        reopened.close()

    _assert_store_files_exclude(
        database,
        f"REMOTE-{operation.upper()}-BEARER".encode("ascii"),
    )


def test_real_sqlite_preparation_recovery_follows_taskrun_preflight_in_one_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "recovery-order.sqlite"
    broker = _FaultBroker()
    initial = _open_runtime(database, tmp_path, broker)
    try:
        owner_id = _spawn_human_owner(initial)
        broker.crash_before_put = True
        with pytest.raises(_SimulatedProcessDeath):
            initial._mcp_continuation_manager.capture_input_required(
                _continuation_binding(owner_id=owner_id),
                _input_required(state="RECOVERY-ORDER-PRIVATE-STATE"),
                expires_at=None,
            )
        preparation = initial.uow.mcp_side_effects.list()[0]
    finally:
        broker.crash_before_put = False
        initial.close()

    order: list[str] = []
    lease_tokens: list[int] = []
    active: list[tuple[int, RuntimeLifecycle]] = []
    next_token = 0
    original_lease = RuntimeLifecycle.recovery_lease
    original_preflight = TaskRunManager.validate_recoverable_payloads

    @contextmanager
    def tracked_lease(lifecycle: RuntimeLifecycle) -> Iterator[None]:
        nonlocal next_token
        next_token += 1
        token = next_token
        with original_lease(lifecycle):
            active.append((token, lifecycle))
            try:
                yield
            finally:
                assert active.pop() == (token, lifecycle)

    def observe(name: str) -> None:
        assert active, f"{name} ran outside the Runtime recovery lease"
        token, lifecycle = active[-1]
        lifecycle.require_recovery_lease()
        order.append(name)
        lease_tokens.append(token)

    def tracked_preflight(manager: TaskRunManager) -> None:
        observe("taskrun-preflight")
        original_preflight(manager)

    monkeypatch.setattr(RuntimeLifecycle, "recovery_lease", tracked_lease)
    monkeypatch.setattr(
        TaskRunManager,
        "validate_recoverable_payloads",
        tracked_preflight,
    )
    broker.on_delete = lambda _secret_ref: observe("mcp-preparation-cleanup")

    reopened = _open_runtime(database, tmp_path, broker)
    try:
        assert order == ["taskrun-preflight", "mcp-preparation-cleanup"]
        assert len(set(lease_tokens)) == 1
        assert reopened.recovered_mcp_continuations == 1
        assert reopened.uow.mcp_side_effects.list() == ()
        assert preparation.human_request_id is not None
        _assert_not_answerable(reopened, preparation.human_request_id)
    finally:
        broker.on_delete = None
        reopened.close()

    _assert_store_files_exclude(database, b"RECOVERY-ORDER-PRIVATE-STATE")
