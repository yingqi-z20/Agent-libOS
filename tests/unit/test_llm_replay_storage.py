from __future__ import annotations

from dataclasses import replace

import pytest

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import UnsupportedStoreVersion, ValidationError
from agent_libos.models.llm_replay import LLMReplayHead, LLMReplayTurn
from agent_libos.storage import SQLiteStore, UnitOfWork
from agent_libos.utils.serde import dumps


def _turn(turn_id: str = "turn-1", *, pid: str = "pid-1", run_id: str = "run-1") -> LLMReplayTurn:
    return LLMReplayTurn.from_payload(
        turn_id=turn_id, pid=pid, run_id=run_id,
        provider_fingerprint="provider-1", model="gpt-6-astra", context_generation="initial",
        source_labels={"sensitivity": "public"}, created_at="2026-09-07T00:00:00Z",
        payload={"groups": [{"output_items": [{"type": "reasoning", "encrypted_content": "PRIVATE_CIPHERTEXT"}]}]},
    )


def _head(turn: LLMReplayTurn, revision: int = 1) -> LLMReplayHead:
    return LLMReplayHead(pid=turn.pid, turn_id=turn.turn_id, revision=revision, updated_at=turn.created_at)


def test_replay_is_private_persistent_and_exact_insert_retry_is_idempotent(tmp_path) -> None:
    path = tmp_path / "runtime.sqlite"
    turn = _turn()
    store = SQLiteStore(path)
    try:
        repository = UnitOfWork(store).processes
        repository.insert_llm_replay_turn(turn)
        repository.insert_llm_replay_turn(turn)
        assert repository.compare_and_set_llm_replay_head(_head(turn), expected_revision=None)
        assert repository.get_llm_replay_turn(turn.turn_id) == turn
        assert "PRIVATE_CIPHERTEXT" not in repr(turn)
        assert "PRIVATE_CIPHERTEXT" not in dumps(turn)
        with pytest.raises(ValidationError, match="unsupported runtime store table"):
            store.validate_table_identifier("llm_replay_turns")
    finally:
        store.close()
    reopened = SQLiteStore(path)
    try:
        assert reopened.get_llm_replay_head(turn.pid) == _head(turn)
        assert reopened.get_llm_replay_turn(turn.turn_id) == turn
    finally:
        reopened.close()


def test_replay_heads_compare_and_swap_and_reject_cross_process_or_missing_turn() -> None:
    store = SQLiteStore(":memory:")
    try:
        first, second = _turn(), _turn("turn-2")
        for turn in (first, second):
            store.insert_llm_replay_turn(turn)
        assert store.compare_and_set_llm_replay_head(_head(first), expected_revision=None)
        assert not store.compare_and_set_llm_replay_head(_head(second), expected_revision=None)
        assert store.compare_and_set_llm_replay_head(_head(second, 2), expected_revision=1)
        assert not store.compare_and_set_llm_replay_head(_head(first, 2), expected_revision=1)
        with pytest.raises(ValidationError, match="another process"):
            store.compare_and_set_llm_replay_head(replace(_head(first), pid="other"), expected_revision=None)
        with pytest.raises(ValidationError, match="missing turn"):
            store.compare_and_set_llm_replay_head(replace(_head(first), turn_id="missing"), expected_revision=None)
    finally:
        store.close()


def test_replay_body_cannot_be_replaced_corrupted_or_resurrected_after_purge() -> None:
    store = SQLiteStore(":memory:")
    turn = _turn()
    try:
        store.insert_llm_replay_turn(turn)
        with pytest.raises(ValidationError, match="conflicting immutable"):
            store.insert_llm_replay_turn(replace(turn, model="other-model"))
        assert store.compare_and_set_llm_replay_head(_head(turn), expected_revision=None)
        assert store.purge_llm_replay(run_id=turn.run_id) == 1
        tombstone = store.get_llm_replay_turn(turn.turn_id)
        assert tombstone is not None and tombstone.payload is None and tombstone.purged_at
        assert tombstone.payload_sha256 == turn.payload_sha256
        assert tombstone.payload_bytes == turn.payload_bytes
        assert store.get_llm_replay_head(turn.pid) is None
        with pytest.raises(ValidationError, match="purged turn"):
            store.compare_and_set_llm_replay_head(_head(turn), expected_revision=None)
        with pytest.raises(ValidationError, match="conflicting immutable"):
            store.insert_llm_replay_turn(turn)
        assert store.purge_llm_replay(pid=turn.pid) == 0
    finally:
        store.close()


def test_replay_read_checks_payload_digest_and_clear_preserves_checkpoint_turn() -> None:
    store = SQLiteStore(":memory:")
    turn = _turn()
    try:
        store.insert_llm_replay_turn(turn)
        store.compare_and_set_llm_replay_head(_head(turn), expected_revision=None)
        store.clear_llm_replay_head(turn.pid)
        assert store.get_llm_replay_head(turn.pid) is None
        assert store.get_llm_replay_turn(turn.turn_id) == turn
        with store.transaction() as cursor:
            cursor.execute("UPDATE llm_replay_turns SET payload_json = '{}' WHERE turn_id = ?", (turn.turn_id,))
        with pytest.raises(ValidationError, match="LLM replay turn"):
            store.get_llm_replay_turn(turn.turn_id)
    finally:
        store.close()


def test_replay_revalidates_mutable_payload_and_enforces_host_retention_limit() -> None:
    turn = _turn()
    disabled = SQLiteStore(":memory:", config=replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False)))
    limited = SQLiteStore(":memory:", config=replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, responses_replay_max_bytes=8)))
    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(ValidationError, match="persist_full_io"):
            disabled.insert_llm_replay_turn(turn)
        with pytest.raises(ValidationError, match="byte limit"):
            limited.insert_llm_replay_turn(turn)
        assert turn.payload is not None
        turn.payload["tampered"] = True
        with pytest.raises(ValueError, match="integrity mismatch"):
            store.insert_llm_replay_turn(turn)
    finally:
        for item in (disabled, limited, store):
            item.close()


def test_task_run_purge_erases_private_replay_and_other_runs_survive() -> None:
    store = SQLiteStore(":memory:")
    first, second = _turn(), _turn("turn-2", pid="pid-2", run_id="run-2")
    try:
        store.insert_llm_replay_turn(first)
        store.insert_llm_replay_turn(second)
        store.purge_task_run_payloads("run-1", purged_at="2026-09-07T01:00:00Z")
        assert store.get_llm_replay_turn(first.turn_id).payload is None
        assert store.get_llm_replay_turn(second.turn_id) == second
    finally:
        store.close()


@pytest.mark.parametrize("version", [4, 5, 6, 7, 8])
def test_replay_schema_preserves_all_historical_sqlite_catalog_ratchets(version: int) -> None:
    assert SQLiteStore._canonical_full_schema_catalog(version)


def test_replay_missing_index_is_rejected_without_schema_repair(tmp_path) -> None:
    import sqlite3

    path = tmp_path / "runtime.sqlite"
    store = SQLiteStore(path)
    store.close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX idx_llm_replay_turns_pid")
    with pytest.raises(UnsupportedStoreVersion, match="v8 replay index"):
        SQLiteStore(path)


@pytest.mark.parametrize("invalid_result", [None, 1, "false"])
def test_replay_cas_facade_rolls_back_non_boolean_backend_results(invalid_result) -> None:
    from agent_libos.storage import ProcessRepository
    from tests.unit.test_storage_repositories import _InvalidCasBackend

    class InvalidReplayBackend(_InvalidCasBackend):
        compare_and_set_llm_replay_head = _InvalidCasBackend.complete_execution

    backend = InvalidReplayBackend(invalid_result)
    repository = ProcessRepository(backend)
    with pytest.raises(ValidationError, match="non-boolean"):
        repository.compare_and_set_llm_replay_head(_head(_turn()), expected_revision=None)
    assert backend.rolled_back
    assert not backend.mutation_visible


def test_replay_recovery_pid_pages_merge_deduplicate_and_use_pending_index() -> None:
    from agent_libos.models import DataFlowContext

    store = SQLiteStore(":memory:")
    try:
        for pid in ("a", "c"):
            turn = _turn(f"turn-{pid}", pid=pid)
            store.insert_llm_replay_turn(turn)
            store.compare_and_set_llm_replay_head(_head(turn), expected_revision=None)
        for pid, status in (("b", "pending"), ("c", "pending"), ("d", "completed")):
            store.upsert_llm_pending_action(pid, {
                "wait_type": "event", "status": status,
                "data_flow_context": DataFlowContext().to_dict(),
            })
        assert store.list_llm_replay_recovery_pids(limit=2) == ["a", "b"]
        assert store.list_llm_replay_recovery_pids(after_pid="b", limit=2) == ["c"]
        with pytest.raises(ValidationError, match="bounds"):
            store.list_llm_replay_recovery_pids(limit=0)
        plan = store.conn.execute(
            "EXPLAIN QUERY PLAN SELECT pid FROM llm_pending_actions "
            "WHERE status = 'pending' AND pid > ? ORDER BY pid LIMIT ?", ("a", 2),
        ).fetchall()
        assert any("idx_llm_pending_replay_recovery" in str(row["detail"]) for row in plan)
    finally:
        store.close()


def test_recovery_preserves_only_existing_read_grants_and_all_restrictive_rows(tmp_path) -> None:
    from agent_libos.models import Capability, CapabilityEffect, CapabilityStatus
    from tests.unit.test_storage_recovery import _runtime_object

    path = tmp_path / "recovery.sqlite"
    store = SQLiteStore(path)
    store.insert_object(_runtime_object("source", {"secret": "in memory only"}))
    capabilities = []
    for name, subject, rights, effect, status in (
        ("allow", "owner", {"read", "write"}, CapabilityEffect.ALLOW, CapabilityStatus.ACTIVE),
        ("deny", "owner", {"read", "delete"}, CapabilityEffect.DENY, CapabilityStatus.ACTIVE),
        ("ask", "owner", {"read", "materialize"}, CapabilityEffect.ASK, CapabilityStatus.ACTIVE),
        ("other", "other-owner", {"read"}, CapabilityEffect.ALLOW, CapabilityStatus.ACTIVE),
        ("write", "owner", {"write"}, CapabilityEffect.ALLOW, CapabilityStatus.ACTIVE),
        ("revoked", "owner", {"read"}, CapabilityEffect.ALLOW, CapabilityStatus.REVOKED),
    ):
        capability = Capability(
            cap_id=name, subject=subject, resource="object:source", rights=rights,
            constraints={"test_restriction": "preserved"}, issued_by="host", issued_at="2026-09-07T00:00:00Z",
            effect=effect, status=status, delegable=True,
        )
        store.insert_capability(capability)
        capabilities.append(capability)
    store.close()
    reopened = SQLiteStore(path)
    try:
        def retained(oids):
            assert oids == ("source",)
            assert reopened._transaction_depth > 0
            return {("owner", "source"): frozenset({"allow", "write", "revoked"})}

        result = reopened.recover_missing_runtime_object_payloads(
            require_recovery_lease=lambda: None, retained_read_capabilities=retained,
        )
        assert result.total_count == 1
        for original in capabilities:
            current = reopened.get_capability(original.cap_id)
            assert current is not None
            if original.cap_id in {"allow", "deny", "ask"}:
                assert current.status == CapabilityStatus.ACTIVE
                assert current.rights == {"read"}
                assert not current.delegable
                assert current.effect == original.effect
                assert current.constraints == original.constraints
            else:
                assert current.status == CapabilityStatus.REVOKED
                assert current.rights == original.rights
        assert reopened.get_persisted_object_state("source").lifecycle_state.value == "released"
    finally:
        reopened.close()
