from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    AgentImage,
    CapabilityEffect,
    CapabilityRight,
    CapabilitySpec,
    CapabilityStatus,
    EventType,
    JsonRpcEndpointSpec,
    JsonRpcMethodSpec,
    LLMCallRecord,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.runtime.runtime import Runtime
from agent_libos.skills.schema import SkillPackage
from agent_libos.storage import StoreCloseClaimOutcome, open_store
from agent_libos.storage.factory import open_store_for_migration
from agent_libos.storage.postgres import PostgresStore
from agent_libos.storage.sql import STORE_SCHEMA_VERSION
from agent_libos.utils.ids import utc_now


class _ScriptedActionClient:
    def __init__(self) -> None:
        self.actions = [{"action": "get_current_time", "timezone": "UTC"}]

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        action = self.actions.pop(0)
        name = str(action["action"])
        args = {key: value for key, value in action.items() if key != "action"}
        return LLMCompletion(content="", tool_calls=[{"id": "pg_call", "name": name, "arguments": json.dumps(args)}])


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_test_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield _dsn_with_search_path(dsn, schema)
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


def _dsn_with_search_path(dsn: str, schema: str) -> str:
    parsed = urlsplit(dsn)
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != "options"]
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


@pytest.mark.postgres
class TestPostgresStore:
    def test_skill_description_search_uses_postgres_supported_json_access(self) -> None:
        with _postgres_schema_dsn() as dsn:
            store = PostgresStore(dsn)
            try:
                store.upsert_skill(
                    SkillPackage(
                        skill_id="pg-search-contract",
                        name="pg-search-contract",
                        description="Find the cobalt description beacon.",
                        instructions="Keep the ultraviolet instruction needle private.",
                    ),
                    source_type="runtime",
                    source="postgres-test",
                    package_sha256="a" * 64,
                    registered_by="test.host",
                    created_at=utc_now(),
                )

                visible = store.list_skills(text="cobalt description beacon", limit=10)
                hidden = store.list_skills(text="ultraviolet instruction needle", limit=10)

                assert [package.skill_id for package, _metadata in visible] == [
                    "pg-search-contract"
                ]
                assert hidden == []
            finally:
                store.close()

    def test_offline_migration_open_does_not_initialize_empty_schema(self) -> None:
        import psycopg

        with _postgres_schema_dsn() as dsn:
            with pytest.raises(
                ValidationError,
                match="requires an existing initialized Agent libOS store",
            ):
                open_store_for_migration(dsn)

            with psycopg.connect(dsn, autocommit=True) as conn:
                rows = conn.execute(
                    """
                    SELECT relation.relname
                    FROM pg_catalog.pg_class AS relation
                    JOIN pg_catalog.pg_namespace AS namespace
                      ON namespace.oid = relation.relnamespace
                    WHERE namespace.nspname = current_schema()
                      AND relation.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
                    """
                ).fetchall()
            assert rows == []

    def test_fresh_schema_rejects_a_second_schema_marker(self) -> None:
        from psycopg.errors import CheckViolation

        with _postgres_schema_dsn() as dsn:
            store = PostgresStore(dsn)
            try:
                with pytest.raises(CheckViolation):
                    store.conn.execute(
                        "INSERT INTO runtime_schema (singleton, schema_version) "
                        "VALUES (?, ?)",
                        (2, STORE_SCHEMA_VERSION),
                    )
                row = store.conn.execute(
                    "SELECT singleton, schema_version FROM runtime_schema"
                ).fetchone()
                assert row == {
                    "singleton": 1,
                    "schema_version": STORE_SCHEMA_VERSION,
                }
            finally:
                store.close()

    def test_admission_guard_handoff_closes_session_lease_and_allows_reopen(
        self,
    ) -> None:
        with _postgres_schema_dsn() as dsn:
            store = PostgresStore(dsn)

            @contextlib.contextmanager
            def expected_guard() -> Iterator[None]:
                yield

            store.bind_admission_commit_guard(expected_guard)
            assert (
                store.probe_admission_guard_close(expected_guard)
                is StoreCloseClaimOutcome.READY
            )
            assert (
                store.claim_admission_guard_close(expected_guard)
                is StoreCloseClaimOutcome.READY
            )
            outcome = store.release_admission_guard_and_close(expected_guard)

            assert outcome.guard_matched is True
            assert outcome.ownership_released is True
            assert outcome.warnings == ()
            assert store._admission_commit_guard is None
            assert store._runtime_lease_acquired is False
            assert store._runtime_lease_key is None

            reopened = PostgresStore(dsn)
            reopened.close()

    def test_interrupted_initialization_releases_session_advisory_lease(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _postgres_schema_dsn() as dsn:
            primary_error = KeyboardInterrupt(
                "injected PostgreSQL initialization interrupt"
            )

            def fail_initialization(*_args: object, **_kwargs: object) -> None:
                raise primary_error

            with monkeypatch.context() as scoped:
                scoped.setattr(PostgresStore, "_init_store", fail_initialization)
                with pytest.raises(KeyboardInterrupt) as caught:
                    PostgresStore(dsn)

            assert caught.value is primary_error
            reopened = PostgresStore(dsn)
            try:
                assert reopened.list_processes() == []
            finally:
                reopened.close()

    @pytest.mark.parametrize("policy_change", ["revoke", "deny"])
    def test_authority_transaction_orders_policy_change_before_registry_mutation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        policy_change: str,
    ) -> None:
        with _postgres_schema_dsn() as dsn:
            config = AgentLibOSConfig(
                runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
            )
            runtime = Runtime(open_store(dsn, config=config), config=config)
            try:
                actor = runtime.process.spawn(goal=f"postgres authority {policy_change}")
                endpoint_id = f"pg-authority-{policy_change}"
                resource = runtime.jsonrpc.endpoint_resource(endpoint_id)
                authority = runtime.capability.grant(
                    actor,
                    resource,
                    [CapabilityRight.WRITE],
                    issued_by="test.host",
                )
                endpoint = JsonRpcEndpointSpec(
                    schema_version=1,
                    endpoint_id=endpoint_id,
                    url="https://api.example.test/jsonrpc",
                    headers={},
                    methods=[
                        JsonRpcMethodSpec(
                            method_id="probe",
                            rpc_method="probe.read",
                            right="read",
                            rollback_class="no_rollback_required",
                            state_mutation=False,
                            information_flow=True,
                        )
                    ],
                    timeout_s=1.0,
                    max_request_bytes=1024,
                    max_response_bytes=2048,
                )
                original_require = runtime.capability.require

                def change_policy_after_preflight(*args: object, **kwargs: object):
                    decision = original_require(*args, **kwargs)
                    if policy_change == "revoke":
                        runtime.capability.revoke(
                            authority.cap_id,
                            revoked_by="test.host",
                            require_authority=False,
                        )
                    else:
                        runtime.capability.issue_trusted(
                            actor,
                            resource,
                            [CapabilityRight.WRITE],
                            issued_by="test.host",
                            effect=CapabilityEffect.DENY,
                        )
                    return decision

                monkeypatch.setattr(
                    runtime.capability,
                    "require",
                    change_policy_after_preflight,
                )

                with pytest.raises(CapabilityDenied, match="authority changed"):
                    runtime.jsonrpc.register_endpoint(endpoint, actor=actor)

                assert runtime.store.get_jsonrpc_endpoint(endpoint_id) is None
            finally:
                runtime.close()

    @pytest.mark.parametrize("policy_change", ["deny", "revoke"])
    def test_grant_transfer_recomputes_requested_rights_inside_authority_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
        policy_change: str,
    ) -> None:
        with _postgres_schema_dsn() as dsn:
            config = AgentLibOSConfig(
                runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
            )
            runtime = Runtime(open_store(dsn, config=config), config=config)
            try:
                actor = runtime.process.spawn(goal=f"postgres grant transfer {policy_change} actor")
                child = runtime.process.spawn(goal=f"postgres grant transfer {policy_change} child")
                resource = f"object:postgres-grant-transfer-{policy_change}"
                parent = runtime.capability.issue_trusted(
                    actor,
                    resource,
                    [CapabilityRight.READ],
                    issued_by="test.host",
                )
                grant_once = runtime.capability.grant_once(
                    actor,
                    resource,
                    [CapabilityRight.GRANT],
                    issued_by="test.host",
                )
                barrier = Barrier(2)
                original_require = runtime.capability._require_issue_authority

                def pause_after_preflight(who: str, spec: CapabilitySpec):
                    decision = original_require(who, spec)
                    barrier.wait(timeout=5)
                    barrier.wait(timeout=5)
                    return decision

                monkeypatch.setattr(
                    runtime.capability,
                    "_require_issue_authority",
                    pause_after_preflight,
                )
                before_reservations = runtime.store.select_table_rows(
                    "capability_use_reservations",
                    order_by="reservation_id",
                )

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        runtime.capability.issue,
                        actor,
                        child,
                        CapabilitySpec(
                            resource=resource,
                            rights={CapabilityRight.READ.value},
                        ),
                    )
                    barrier.wait(timeout=5)
                    if policy_change == "deny":
                        runtime.capability.issue_trusted(
                            actor,
                            resource,
                            [CapabilityRight.READ],
                            issued_by="test.defender",
                            effect=CapabilityEffect.DENY,
                        )
                    else:
                        runtime.capability.revoke(
                            parent.cap_id,
                            revoked_by="test.defender",
                            require_authority=False,
                        )
                    barrier.wait(timeout=5)
                    with pytest.raises(CapabilityDenied):
                        future.result(timeout=5)

                latest_grant = runtime.store.get_capability(grant_once.cap_id)
                assert latest_grant is not None
                assert latest_grant.status == CapabilityStatus.ACTIVE
                assert latest_grant.uses_remaining == 1
                assert runtime.store.select_table_rows(
                    "capability_use_reservations",
                    order_by="reservation_id",
                ) == before_reservations
                assert not runtime.capability.check(
                    child,
                    resource,
                    CapabilityRight.READ,
                )
            finally:
                runtime.close()

    def test_one_shot_authority_has_one_concurrent_winner_without_stranded_reservation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _postgres_schema_dsn() as dsn:
            config = AgentLibOSConfig(
                runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
            )
            runtime = Runtime(open_store(dsn, config=config), config=config)
            try:
                actor = runtime.process.spawn(goal="postgres one-shot authority actor")
                children = [
                    runtime.process.spawn(goal="postgres one-shot child one"),
                    runtime.process.spawn(goal="postgres one-shot child two"),
                ]
                resource = "object:postgres-one-shot-authority"
                authority = runtime.capability.grant_once(
                    actor,
                    resource,
                    [CapabilityRight.ADMIN],
                    issued_by="test.host",
                )
                barrier = Barrier(2)
                original_require = runtime.capability._require_issue_authority

                def synchronize_preflights(who: str, spec: CapabilitySpec):
                    decision = original_require(who, spec)
                    barrier.wait(timeout=5)
                    return decision

                monkeypatch.setattr(
                    runtime.capability,
                    "_require_issue_authority",
                    synchronize_preflights,
                )

                def issue(child: str):
                    try:
                        return runtime.capability.issue(
                            actor,
                            child,
                            CapabilitySpec(
                                resource=resource,
                                rights={CapabilityRight.READ.value},
                            ),
                        )
                    except CapabilityDenied as exc:
                        return exc

                with ThreadPoolExecutor(max_workers=2) as executor:
                    outcomes = list(executor.map(issue, children))

                assert sum(not isinstance(outcome, CapabilityDenied) for outcome in outcomes) == 1
                assert sum(isinstance(outcome, CapabilityDenied) for outcome in outcomes) == 1
                latest = runtime.store.get_capability(authority.cap_id)
                assert latest is not None
                assert latest.status == CapabilityStatus.REVOKED
                assert latest.uses_remaining == 0
                rows = runtime.store.select_table_rows(
                    "capability_use_reservations",
                    "cap_id = ?",
                    (authority.cap_id,),
                    order_by="reservation_id",
                )
                assert [row["status"] for row in rows] == ["committed"]
            finally:
                runtime.close()

    def test_checkpoint_restore_audit_failure_rolls_back_main_state_and_authority(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _postgres_schema_dsn() as dsn:
            config = AgentLibOSConfig(
                runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
            )
            runtime = Runtime(open_store(dsn, config=config), config=config)
            try:
                owner = runtime.process.spawn(goal="postgres restore owner")
                controller = runtime.process.spawn(goal="postgres restore controller")
                checkpoint_id = runtime.checkpoint.create(
                    owner,
                    "postgres authority atomicity",
                    actor=owner,
                )
                current_owner = runtime.process.get(owner)
                runtime.store.patch_process(
                    owner,
                    {"status_message": "current state"},
                    expected_revision=current_owner.revision,
                )
                authority = runtime.capability.grant_once(
                    controller,
                    f"checkpoint:{checkpoint_id}",
                    [CapabilityRight.ADMIN],
                    issued_by="test.host",
                )
                before_event_ids = {event.event_id for event in runtime.events.list()}
                before_audit_ids = {record.record_id for record in runtime.audit.trace()}
                before_reservations = runtime.store.select_table_rows(
                    "capability_use_reservations",
                    order_by="reservation_id",
                )
                original_record = runtime.audit.record

                def fail_after_restore_audit(*args, **kwargs):
                    result = original_record(*args, **kwargs)
                    if kwargs.get("action") == "checkpoint.restore":
                        raise RuntimeError("injected postgres restore audit failure")
                    return result

                monkeypatch.setattr(runtime.audit, "record", fail_after_restore_audit)

                with pytest.raises(RuntimeError, match="postgres restore audit failure"):
                    runtime.checkpoint.restore(controller, checkpoint_id)

                assert runtime.process.get(owner).status_message == "current state"
                latest = runtime.store.get_capability(authority.cap_id)
                assert latest is not None
                assert latest.status == CapabilityStatus.ACTIVE
                assert latest.uses_remaining == 1
                assert runtime.store.select_table_rows(
                    "capability_use_reservations",
                    order_by="reservation_id",
                ) == before_reservations
                assert not [
                    event
                    for event in runtime.events.list()
                    if event.event_id not in before_event_ids
                    and event.type == EventType.ROLLBACK
                ]
                assert not [
                    record
                    for record in runtime.audit.trace()
                    if record.record_id not in before_audit_ids
                    and record.action == "checkpoint.restore"
                ]
            finally:
                runtime.close()

    def test_checkpoint_restore_settlement_failure_rolls_back_composite_unit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _postgres_schema_dsn() as dsn:
            config = AgentLibOSConfig(
                runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
            )
            runtime = Runtime(open_store(dsn, config=config), config=config)
            image_id = "postgres-checkpoint-settlement:v0"
            try:
                runtime.register_image(
                    AgentImage(
                        image_id=image_id,
                        name=image_id,
                        system_prompt="captured",
                    ),
                    actor="test.host",
                )
                owner = runtime.process.spawn(image=image_id, goal="postgres settlement owner")
                controller = runtime.process.spawn(goal="postgres settlement controller")
                checkpoint_id = runtime.checkpoint.create(
                    owner,
                    "postgres settlement atomicity",
                    actor=owner,
                )
                current_owner = runtime.process.get(owner)
                runtime.store.patch_process(
                    owner,
                    {"status_message": "current state"},
                    expected_revision=current_owner.revision,
                )
                runtime.register_image(
                    AgentImage(
                        image_id=image_id,
                        name=image_id,
                        system_prompt="current",
                    ),
                    actor="test.host",
                    replace=True,
                )
                authorities = [
                    runtime.capability.grant_once(
                        controller,
                        f"checkpoint:{checkpoint_id}",
                        [CapabilityRight.ADMIN],
                        issued_by="test.host",
                    ),
                    runtime.capability.grant_once(
                        controller,
                        f"image:{image_id}",
                        [CapabilityRight.ADMIN],
                        issued_by="test.host",
                    ),
                ]
                before_reservations = runtime.store.select_table_rows(
                    "capability_use_reservations",
                    order_by="reservation_id",
                )
                original_commit = runtime.capability.commit_reserved_use
                commit_calls = 0

                def fail_after_second_settlement(*args, **kwargs):
                    nonlocal commit_calls
                    result = original_commit(*args, **kwargs)
                    commit_calls += 1
                    if commit_calls == 2:
                        raise RuntimeError("injected postgres settlement failure")
                    return result

                monkeypatch.setattr(
                    runtime.capability,
                    "commit_reserved_use",
                    fail_after_second_settlement,
                )

                with pytest.raises(RuntimeError, match="postgres settlement failure"):
                    runtime.checkpoint.restore(controller, checkpoint_id)

                assert commit_calls == 2
                assert runtime.process.get(owner).status_message == "current state"
                assert runtime.get_image(image_id).system_prompt == "current"
                for authority in authorities:
                    latest = runtime.store.get_capability(authority.cap_id)
                    assert latest is not None
                    assert latest.status == CapabilityStatus.ACTIVE
                    assert latest.uses_remaining == 1
                assert runtime.store.select_table_rows(
                    "capability_use_reservations",
                    order_by="reservation_id",
                ) == before_reservations
            finally:
                runtime.close()

    def test_postgres_runtime_store_smoke(self) -> None:
        with _postgres_schema_dsn() as dsn:
            config = AgentLibOSConfig(runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn))
            store = open_store(dsn, config=config)
            runtime = Runtime(store, config=config, llm_client=_ScriptedActionClient())
            try:
                pid = runtime.process.spawn(goal="postgres store smoke")
                runtime.activate_skill(pid, "agent-libos-runtime-session")
                assert "get_current_time" in runtime.process.get(pid).model_tool_table
                runtime.capability.grant(pid, "filesystem:workspace:*", [CapabilityRight.READ], issued_by="test")
                runtime.capability.grant(pid, "clock:now", [CapabilityRight.READ], issued_by="test")
                runtime.messages.post(sender="human:owner", recipient_pid=pid, subject="hello", body="postgres")

                endpoint = JsonRpcEndpointSpec(
                    schema_version=1,
                    endpoint_id="pg-demo",
                    url="https://api.example.test/jsonrpc",
                    headers={},
                    methods=[
                        JsonRpcMethodSpec(
                            method_id="echo",
                            rpc_method="demo.echo",
                            right="read",
                            rollback_class="no_rollback_required",
                            state_mutation=False,
                            information_flow=True,
                        )
                    ],
                    timeout_s=1.0,
                    max_request_bytes=1024,
                    max_response_bytes=2048,
                )
                runtime.store.upsert_jsonrpc_endpoint(endpoint, registered_by="test", created_at=utc_now())
                assert runtime.store.get_jsonrpc_endpoint("pg-demo")[0].method_by_id("echo") is not None

                runtime.store.insert_llm_call(
                    LLMCallRecord(
                        call_id="llm_pg_smoke",
                        pid=pid,
                        image_id="base-agent:v0",
                        purpose="test",
                        status="ok",
                        messages=[{"role": "user", "content": "postgres"}],
                        response_content="ok",
                        created_at=utc_now(),
                    )
                )
                assert runtime.store.list_llm_calls(pid=pid)[0].call_id == "llm_pg_smoke"

                result = runtime.run_process_until_idle(pid, max_quanta=1)
                assert result[0]["action"]["action"] == "get_current_time"

                checkpoint_id = runtime.checkpoint.create(pid, "postgres smoke", require_capability=False)
                restored = runtime.checkpoint.restore("test", checkpoint_id, require_capability=False)
                forked = runtime.checkpoint.fork_from_checkpoint("test", checkpoint_id, require_capability=False)

                assert restored["status"] == "restored"
                assert runtime.process.get(forked["fork_root_pid"]) is not None
                assert any(record.action == "checkpoint.restore" for record in runtime.audit.trace())
                assert runtime.events.list()
            finally:
                runtime.close()
