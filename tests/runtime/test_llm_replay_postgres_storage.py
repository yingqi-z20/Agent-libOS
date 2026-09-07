from __future__ import annotations

from dataclasses import replace

import pytest

from agent_libos.models.exceptions import ValidationError
from agent_libos.storage import PostgresStore, UnitOfWork
from tests.runtime.test_semantic_v5_postgres_migration import _postgres_schema_dsn
from tests.unit.test_llm_replay_storage import _head, _turn


pytestmark = pytest.mark.postgres


def test_postgres_private_replay_immutable_cas_restart_and_purge() -> None:
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        first, second = _turn(), _turn("turn-2")
        try:
            repository = UnitOfWork(store).processes
            repository.insert_llm_replay_turn(first)
            repository.insert_llm_replay_turn(first)
            repository.insert_llm_replay_turn(second)
            assert repository.compare_and_set_llm_replay_head(_head(first), expected_revision=None)
            assert repository.compare_and_set_llm_replay_head(_head(second, 2), expected_revision=1)
            assert not repository.compare_and_set_llm_replay_head(_head(first, 2), expected_revision=1)
            with pytest.raises(ValidationError, match="another process"):
                repository.compare_and_set_llm_replay_head(replace(_head(first), pid="other"), expected_revision=None)
            with pytest.raises(ValidationError, match="conflicting immutable"):
                repository.insert_llm_replay_turn(replace(first, model="other-model"))
        finally:
            store.close()
        store = PostgresStore(dsn)
        try:
            assert store.get_llm_replay_head(first.pid) == _head(second, 2)
            assert store.get_llm_replay_turn(first.turn_id) == first
            with pytest.raises(RuntimeError, match="rollback replay"):
                with store.transaction():
                    store.purge_llm_replay(pid=first.pid)
                    raise RuntimeError("rollback replay")
            assert store.get_llm_replay_turn(first.turn_id) == first
            store.purge_task_run_payloads(first.run_id, purged_at="2026-09-07T01:00:00Z")
            assert store.get_llm_replay_head(first.pid) is None
            assert store.get_llm_replay_turn(first.turn_id).payload is None
            with pytest.raises(ValidationError, match="conflicting immutable"):
                store.insert_llm_replay_turn(first)
        finally:
            store.close()
