from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from typing import Any

import pytest

import agent_libos.storage.sqlite as sqlite_backend
from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG, SkillDefaults
from agent_libos.models import (
    CapabilityRight,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ResourceBudget,
    ResourceUsage,
)
from agent_libos.models.exceptions import (
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.builder import RuntimeBuilder
from agent_libos.runtime.event_bus import EventBus
from agent_libos.runtime.lifecycle import RuntimeLifecycle
from agent_libos.runtime.syscall_router import SyscallRouter
from agent_libos.sdk import (
    ProtectedOperationContract,
    ProtectedOperationInvocation,
    ProtectedOperationProtocolError,
    ProviderPhase,
    ResourcePolicy,
)
from agent_libos.skills.schema import SkillPackage
from agent_libos.storage import SQLiteStore
from agent_libos.storage.sql import SQLRuntimeStore
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.utils.ids import new_id, utc_now


def test_authority_manifest_bind_rolls_back_when_required_audit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    pid = "pid_manifest_atomic_bind"
    try:
        before_audit = runtime.store.list_audit()
        before_events = runtime.store.list_events()
        monkeypatch.setattr(
            runtime.audit,
            "record",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected authority audit failure")
            ),
        )

        with pytest.raises(RuntimeError, match="authority audit failure"):
            runtime.authority_manifests.prepare_launch(
                pid=pid,
                image_id=DEFAULT_CONFIG.runtime.default_image_id,
                goal_ref=None,
                supplied={},
            )

        assert runtime.store.get_authority_manifest_for_process(pid) is None
        assert runtime.store.list_audit() == before_audit
        assert runtime.store.list_events() == before_events
    finally:
        runtime.close()


def test_authority_manifest_compile_rolls_back_a_late_issue_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    pid = "pid_manifest_atomic_compile"
    try:
        manifest = runtime.authority_manifests.prepare_launch(
            pid=pid,
            image_id=DEFAULT_CONFIG.runtime.default_image_id,
            goal_ref=None,
            supplied={
                "authorized_capabilities": [
                    {"resource": "object:atomic-one", "rights": ["read"]},
                    {"resource": "object:atomic-two", "rights": ["write"]},
                ]
            },
        )
        original_issue = runtime.capability.issue_trusted
        issue_count = 0
        before_audit = runtime.store.list_audit()
        before_events = runtime.store.list_events()

        def fail_second_issue(*args: Any, **kwargs: Any) -> Any:
            nonlocal issue_count
            issue_count += 1
            if issue_count == 2:
                raise RuntimeError("injected second capability issue failure")
            return original_issue(*args, **kwargs)

        monkeypatch.setattr(runtime.capability, "issue_trusted", fail_second_issue)
        with pytest.raises(RuntimeError, match="second capability issue failure"):
            runtime.authority_manifests.compile_root_capabilities(manifest)

        assert runtime.store.list_capabilities(pid) == []
        assert runtime.store.list_audit() == before_audit
        assert runtime.store.list_events() == before_events
    finally:
        runtime.close()


def test_rating_update_and_audit_failure_share_one_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="atomic rating")
        before = runtime.ratings.upsert(pid, score=2, comment="before")
        monkeypatch.setattr(
            runtime.audit,
            "record",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected rating audit failure")
            ),
        )

        with pytest.raises(RuntimeError, match="rating audit failure"):
            runtime.ratings.upsert(pid, score=5, comment="must roll back")

        after = runtime.ratings.get(pid)
        assert after == before
    finally:
        runtime.close()


def test_syscall_router_restores_exact_route_when_audit_fails() -> None:
    class ToggleAudit:
        error: BaseException | None = None

        def record(self, **_kwargs: Any) -> None:
            if self.error is not None:
                raise self.error

    audit = ToggleAudit()
    router = SyscallRouter(audit)  # type: ignore[arg-type]
    handler = lambda _session, _args: "ok"

    audit.error = KeyboardInterrupt("register audit interrupted")
    with pytest.raises(KeyboardInterrupt, match="register audit interrupted"):
        router.register("module.echo", handler, registered_by="module.test")
    assert router.get("module.echo") is None

    audit.error = None
    registered = router.register(
        "module.echo",
        handler,
        registered_by="module.test",
    )
    audit.error = RuntimeError("unregister audit failed")
    with pytest.raises(RuntimeError, match="unregister audit failed"):
        router.unregister("module.echo", registered_by="module.test")
    assert router.get("module.echo") is registered


def test_audit_explicit_pages_are_bounded_and_gap_free() -> None:
    store = SQLiteStore(":memory:")
    audit = AuditManager(store, query_limit=2)  # type: ignore[arg-type]
    try:
        records = [
            audit.record(actor="test", action=f"page.{index}")
            for index in range(5)
        ]

        newest = audit.trace(limit=2)
        older = audit.trace(limit=2, before_record_id=newest[0].record_id)
        oldest = audit.trace(limit=2, before_record_id=older[0].record_id)

        assert [item.record_id for item in [*oldest, *older, *newest]] == [
            item.record_id for item in records
        ]
        with pytest.raises(ValidationError, match="no greater than 2"):
            audit.trace(limit=3)
    finally:
        store.close()


def test_large_discrete_resource_values_remain_exact() -> None:
    runtime = Runtime.open("local")
    limit = 2**53
    try:
        pid = runtime.process.spawn(
            goal="exact integer resource accounting",
            resource_budget=ResourceBudget(max_external_read_bytes=limit),
        )

        with pytest.raises(ResourceLimitExceeded):
            runtime.resources.preflight(
                pid,
                ResourceUsage(external_read_bytes=limit + 1),
                source="test",
            )

        runtime.resources.charge(
            pid,
            ResourceUsage(external_read_bytes=limit),
            source="test",
        )
        assert runtime.process.get(pid).resource_usage.external_read_bytes == limit
        assert runtime.resources.remaining_budget(pid).max_external_read_bytes == 0
    finally:
        runtime.close()


def test_large_discrete_child_reservation_keeps_adjacent_capacity() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(
            goal="exact parent reservation",
            resource_budget=ResourceBudget(max_external_read_bytes=2**53 + 2),
        )
        runtime.process.spawn_child(
            parent,
            goal="exact child reservation",
            resource_budget=ResourceBudget(max_external_read_bytes=2**53 + 1),
        )

        assert runtime.resources.remaining_budget(parent).max_external_read_bytes == 1
    finally:
        runtime.close()


def test_skill_storage_search_uses_unicode_casefold_and_bounded_catalog() -> None:
    config = AgentLibOSConfig(
        skills=replace(SkillDefaults(), catalog_scan_limit=2)
    )
    store = SQLiteStore(":memory:", config=config)
    try:
        packages = (
            SkillPackage(
                skill_id="unicode-street",
                name="Straße",
                description="Exact Unicode name match.",
                instructions="test",
            ),
            SkillPackage(
                skill_id="description-match",
                name="Other",
                description="A strasse description match.",
                instructions="test",
            ),
        )
        for package in packages:
            store.upsert_skill(
                package,
                source_type="runtime",
                source="test",
                package_sha256="a" * 64,
                registered_by="test",
                created_at=utc_now(),
            )

        result = store.list_skills(text="STRASSE", limit=1)
        assert [package.skill_id for package, _metadata in result] == [
            "unicode-street"
        ]

        store.upsert_skill(
            SkillPackage(
                skill_id="catalog-overflow",
                name="Overflow",
                description="Does not matter because the catalog is over bound.",
                instructions="test",
            ),
            source_type="runtime",
            source="test",
            package_sha256="b" * 64,
            registered_by="test",
            created_at=utc_now(),
        )
        with pytest.raises(ValidationError, match="catalog_scan_limit=2"):
            store.list_skills(text="STRASSE", limit=1)
    finally:
        store.close()


@pytest.mark.platform_darwin
@pytest.mark.platform_linux
def test_failed_sqlite_constructor_retains_lease_until_retry(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sqlite_backend.fcntl is None or not hasattr(sqlite_backend.os, "O_NOFOLLOW"):
        pytest.skip("secure file lease is not used on this platform")

    database = tmp_path / "failed-constructor-owner.sqlite"
    real_connect = sqlite_backend.sqlite3.connect
    connect_count = 0

    class FailFirstCloseConnection(sqlite3.Connection):
        close_attempts = 0

        def close(self) -> None:
            self.close_attempts += 1
            if self.close_attempts == 1:
                raise OSError("injected live connection close failure")
            super().close()

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal connect_count
        connect_count += 1
        if connect_count == 1:
            kwargs["factory"] = FailFirstCloseConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite_backend.sqlite3, "connect", connect)
    original_init = SQLRuntimeStore._init_store
    monkeypatch.setattr(
        SQLRuntimeStore,
        "_init_store",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected store initialization failure")
        ),
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        SQLiteStore(database)
    assert {type(error) for error in caught.value.exceptions} == {
        RuntimeError,
        OSError,
    }
    assert len(SQLiteStore._failed_owners) == 1
    failed_owner = next(iter(SQLiteStore._failed_owners.values()))
    assert failed_owner._sqlite_connection_reports_closed() is False
    assert failed_owner._lease_handle is not None

    monkeypatch.setattr(SQLRuntimeStore, "_init_store", original_init)
    reopened = SQLiteStore(database)
    try:
        assert SQLiteStore._failed_owners == {}
        assert failed_owner._runtime_ownership_released() is True
    finally:
        reopened.close()


def test_sync_builder_rejection_does_not_rebind_config_or_close_shared_substrate(
    tmp_path: Any,
) -> None:
    class TrackingSubstrate(LocalResourceProviderSubstrate):
        shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    store = SQLiteStore(":memory:")
    substrate = TrackingSubstrate(tmp_path)
    first_config = AgentLibOSConfig(
        gui=replace(DEFAULT_CONFIG.gui, snapshot_audit_limit=7)
    )
    second_config = AgentLibOSConfig(
        gui=replace(DEFAULT_CONFIG.gui, snapshot_audit_limit=9)
    )
    runtime = RuntimeBuilder.configured(
        Runtime,
        config=first_config,
        substrate=substrate,
    ).from_store(store)
    try:
        with pytest.raises(
            RuntimeError,
            match="not ready|admission commit guard is already bound",
        ):
            RuntimeBuilder.configured(
                Runtime,
                config=second_config,
                substrate=substrate,
            ).from_store(store)

        assert store.config is runtime.config
        assert store.config.gui.snapshot_audit_limit == 7
        assert substrate.shutdown_calls == 0
    finally:
        runtime.close()
    assert substrate.shutdown_calls == 1


def test_recovery_lease_copied_to_detached_task_expires_with_parent_scope() -> None:
    async def exercise() -> None:
        store = SQLiteStore(":memory:")
        audit = AuditManager(store)  # type: ignore[arg-type]
        lifecycle = RuntimeLifecycle(
            store=store,
            audit=audit,
            events=EventBus(store),  # type: ignore[arg-type]
            substrate=object(),
        )
        lifecycle.begin_recovery()
        release_child = asyncio.Event()

        async def detached() -> None:
            await release_child.wait()
            with pytest.raises(RuntimeError, match="active startup recovery lease"):
                lifecycle.require_recovery_lease()

        try:
            with lifecycle.recovery_lease():
                task = asyncio.create_task(detached())
                lifecycle.require_recovery_lease()
            release_child.set()
            await task
        finally:
            store.close()

    asyncio.run(exercise())


def test_sync_failed_assembly_cleanup_drives_async_only_component() -> None:
    calls: list[str] = []

    class AsyncOnlyComponent:
        async def aclose(self) -> None:
            calls.append("aclose")

    store = SQLiteStore(":memory:")
    lifecycle = RuntimeLifecycle(
        store=store,
        audit=AuditManager(store),  # type: ignore[arg-type]
        events=EventBus(store),  # type: ignore[arg-type]
        substrate=object(),
    )
    errors: list[dict[str, str]] = []
    caught: list[BaseException] = []
    try:
        assert lifecycle._stop_failed_assembly_sync_component(
            "async_only",
            AsyncOnlyComponent(),
            errors,
            caught,
        )
        assert calls == ["aclose"]
        assert errors == []
        assert caught == []
    finally:
        store.close()


class _ProtectedProvider:
    def classify_external_effect(
        self,
        _operation: str,
        _context: dict[str, Any],
        _result: Any,
    ) -> ExternalEffectClassification:
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.UNKNOWN,
            rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
            state_mutation=False,
            information_flow=False,
        )


def _protected_phase_fixture(
    runtime: Runtime,
    *,
    state_mutation: bool,
    information_flow: bool,
) -> tuple[str, Any, ProtectedOperationContract, ProtectedOperationInvocation]:
    pid = runtime.process.spawn(goal="protected phase ceiling")
    capability = runtime.capability.issue_trusted(
        pid,
        "test:phase-ceiling",
        [CapabilityRight.EXECUTE],
        issued_by="test",
        uses_remaining=1,
    )
    decision = runtime.capability.require(
        pid,
        "test:phase-ceiling",
        CapabilityRight.EXECUTE,
        consume=False,
    )
    contract = ProtectedOperationContract(
        name=f"primitive.test.phase_{new_id('contract')}",
        provider="test",
        operation="phase",
        evidence_roles=("audit", "event", "effect"),
        resource_policy=ResourcePolicy.NONE,
        state_mutation=state_mutation,
        information_flow=information_flow,
    )
    runtime.protected_operations.register_contract(contract)
    invocation = ProtectedOperationInvocation(
        pid=pid,
        actor=pid,
        target="test:phase-ceiling",
        decisions=(decision,),
    )
    return pid, capability, contract, invocation


@pytest.mark.parametrize(
    "phase",
    [
        ProviderPhase("mutating", state_mutation=True, commits_authority=False),
        ProviderPhase("informational", information_flow=True, commits_authority=False),
    ],
)
def test_provider_phase_cannot_exceed_registered_contract_ceiling(
    phase: ProviderPhase,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid, capability, contract, invocation = _protected_phase_fixture(
            runtime,
            state_mutation=False,
            information_flow=False,
        )
        calls: list[str] = []
        with pytest.raises(ProtectedOperationProtocolError, match="contract ceiling"):
            with runtime.protected_operations.start(
                contract,
                invocation,
                provider=_ProtectedProvider(),
            ) as operation:
                operation.call(phase, lambda: calls.append("provider"))

        assert calls == []
        assert runtime.store.get_capability(capability.cap_id).uses_remaining == 1
        assert runtime.store.list_external_effects(pid=pid) == []
    finally:
        runtime.close()


def test_failed_active_phase_is_classified_at_monotonic_effect_ceiling() -> None:
    runtime = Runtime.open("local")
    try:
        pid, _capability, contract, invocation = _protected_phase_fixture(
            runtime,
            state_mutation=True,
            information_flow=True,
        )
        with pytest.raises(RuntimeError, match="provider failed after start"):
            with runtime.protected_operations.start(
                contract,
                invocation,
                provider=_ProtectedProvider(),
            ) as operation:
                operation.call(
                    ProviderPhase(
                        "effectful",
                        state_mutation=True,
                        information_flow=True,
                    ),
                    lambda: (_ for _ in ()).throw(
                        RuntimeError("provider failed after start")
                    ),
                )

        effect = runtime.store.list_external_effects(pid=pid)[0]
        assert effect.transaction_state == "unknown"
        assert effect.state_mutation is True
        assert effect.information_flow is True
    finally:
        runtime.close()


def test_runtime_scoped_run_validates_and_forwards_exact_pid_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="scoped run")
        scheduler_scopes: list[frozenset[str] | None] = []
        human_scopes: list[frozenset[str] | None] = []

        def run_scheduler(*_args: Any, **kwargs: Any) -> list[Any]:
            scheduler_scopes.append(kwargs.get("pids"))
            return []

        def drain_human(**kwargs: Any) -> list[Any]:
            human_scopes.append(kwargs.get("pids"))
            return []

        monkeypatch.setattr(runtime.scheduler, "run_until_idle", run_scheduler)
        monkeypatch.setattr(runtime.human, "drain_terminal_queue", drain_human)

        assert runtime.run_until_idle(max_quanta=1, pids=[pid]) == []
        assert scheduler_scopes == [frozenset({pid})]
        assert human_scopes == [frozenset({pid})]

        with pytest.raises(ValidationError, match="must not be empty"):
            runtime.run_until_idle(max_quanta=1, pids=[])
        with pytest.raises(ValidationError, match="duplicates"):
            runtime.run_until_idle(max_quanta=1, pids=[pid, pid])
    finally:
        runtime.close()
