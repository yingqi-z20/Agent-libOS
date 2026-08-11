from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from agent_libos.mcp.side_effects import (
    commit_mcp_preparation,
    commit_mcp_preparation_deferred,
    commit_terminal_mcp_preparation_deferred,
    finalize_mcp_preparation,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage.mcp_v7 import (
    McpRemoteTaskRecord,
    McpSideEffectPreparationRecord,
)


_H0 = "0" * 64
_H1 = "1" * 64
_H2 = "2" * 64
_T0 = "2030-01-01T00:00:00Z"
_T1 = "2030-01-01T00:00:01Z"
_T2 = "2030-01-01T00:00:02Z"
_EXPIRY = "2030-01-02T00:00:00Z"


class _Repository:
    def __init__(self, preparation: McpSideEffectPreparationRecord) -> None:
        self.rows = {preparation.preparation_id: preparation}
        self.commit_calls = 0
        self.terminal_commit_calls = 0
        self.corrupt_readback = False

    def get(self, preparation_id: str) -> McpSideEffectPreparationRecord | None:
        return self.rows.get(preparation_id)

    def commit(
        self,
        preparation_id: str,
        *,
        expected_revision: int,
        replacement: McpRemoteTaskRecord,
    ) -> bool:
        current = self.rows.get(preparation_id)
        if current is None or current.revision != expected_revision:
            return False
        self.commit_calls += 1
        metadata: dict[str, Any] = {
            "automatic_retry_disabled": True,
            "cleanup_mode": "retire",
            "retire_refs": tuple(current.metadata.get("retire_refs", ())),
        }
        for key in (
            "retire_human_request_id",
            "retire_human_preview_sha256",
        ):
            if key in current.metadata:
                metadata[key] = current.metadata[key]
        revision = current.revision + (2 if self.corrupt_readback else 1)
        self.rows[preparation_id] = replace(
            current,
            status="cleaning",
            revision=revision,
            metadata=metadata,
            updated_at=max(current.updated_at, replacement.updated_at),
        )
        return True

    def commit_terminal(
        self,
        preparation_id: str,
        *,
        expected_revision: int,
    ) -> bool:
        current = self.rows.get(preparation_id)
        if current is None or current.revision != expected_revision:
            return False
        self.terminal_commit_calls += 1
        metadata = dict(current.metadata)
        metadata["cleanup_mode"] = "retire"
        self.rows[preparation_id] = replace(
            current,
            status="cleaning",
            revision=current.revision + 1,
            metadata=metadata,
        )
        return True

    def delete(self, preparation_id: str, *, expected_revision: int) -> bool:
        current = self.rows.get(preparation_id)
        if current is None or current.revision != expected_revision:
            return False
        del self.rows[preparation_id]
        return True


class _Broker:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def available(self) -> bool:
        return True

    def delete_secret(self, secret_ref: str) -> None:
        self.deleted.append(secret_ref)


class _Human:
    def __init__(self) -> None:
        self.cancelled: list[tuple[str, str, str]] = []

    def cancel_question(
        self,
        request_id: str,
        *,
        preview_sha256: str,
        reason: str,
    ) -> None:
        self.cancelled.append((request_id, preview_sha256, reason))


def _preparation(*, terminal: bool = False) -> McpSideEffectPreparationRecord:
    return McpSideEffectPreparationRecord(
        preparation_id="preparation-terminal" if terminal else "preparation-normal",
        operation_kind="remote_task",
        operation_id="task-local",
        operation_revision=1 if terminal else 0,
        server_id="server-v3",
        server_spec_sha256=_H0,
        server_generation=3,
        owner_id="owner",
        auth_principal_sha256=_H1,
        auth_scope_sha256=_H2,
        human_request_id=None,
        human_preview_sha256=None,
        broker_ref=None if terminal else "new-remote-ref",
        broker_value_sha256=None if terminal else _H0,
        result_ref=None if terminal else "new-state-ref",
        result_sha256=None if terminal else _H1,
        status="prepared",
        revision=0,
        expires_at=_EXPIRY,
        metadata={
            "automatic_retry_disabled": True,
            "cleanup_mode": "abort",
            "retire_refs": ("old-remote-ref", "old-state-ref"),
            "retire_human_request_id": "human-old",
            "retire_human_preview_sha256": _H2,
        },
        created_at=_T0,
        updated_at=_T1,
    )


def _replacement() -> McpRemoteTaskRecord:
    return McpRemoteTaskRecord(
        task_ref="task-local",
        server_id="server-v3",
        server_spec_sha256=_H0,
        server_generation=3,
        owner_id="owner",
        auth_principal_sha256=_H1,
        auth_scope_sha256=_H2,
        origin_request_sha256=_H0,
        origin_effect_id="initial-effect",
        human_request_id=None,
        broker_ref="new-remote-ref",
        remote_id_sha256=_H0,
        status="working",
        revision=1,
        expires_at=_EXPIRY,
        poll_interval_ms=250,
        status_message_sha256=None,
        result_ref="new-state-ref",
        result_sha256=_H1,
        metadata={"automatic_retry_disabled": True},
        created_at=_T0,
        updated_at=_T2,
    )


def test_deferred_commit_leaves_exact_retirement_without_external_cleanup() -> None:
    preparation = _preparation()
    repository = _Repository(preparation)
    broker, human = _Broker(), _Human()

    retirement = commit_mcp_preparation_deferred(
        repository,
        preparation,
        _replacement(),
    )

    assert repository.commit_calls == 1
    assert repository.get(preparation.preparation_id) == retirement
    assert retirement.status == "cleaning"
    assert retirement.metadata["cleanup_mode"] == "retire"
    assert broker.deleted == []
    assert human.cancelled == []

    finalize_mcp_preparation(
        repository,
        retirement,
        broker=broker,
        human_requests=human,
    )
    finalize_mcp_preparation(
        repository,
        retirement,
        broker=broker,
        human_requests=human,
    )

    assert broker.deleted == ["old-remote-ref", "old-state-ref"]
    assert human.cancelled == [
        ("human-old", _H2, "mcp_human_binding_retired")
    ]
    assert repository.get(preparation.preparation_id) is None


def test_existing_commit_helper_composes_deferred_commit_and_finalize() -> None:
    preparation = _preparation()
    repository = _Repository(preparation)
    broker, human = _Broker(), _Human()

    commit_mcp_preparation(
        repository,
        preparation,
        _replacement(),
        broker=broker,
        human_requests=human,
    )

    assert repository.commit_calls == 1
    assert repository.get(preparation.preparation_id) is None
    assert broker.deleted == ["old-remote-ref", "old-state-ref"]


def test_deferred_commit_rejects_non_exact_retirement_readback() -> None:
    preparation = _preparation()
    repository = _Repository(preparation)
    repository.corrupt_readback = True

    with pytest.raises(ValidationError, match="retirement receipt changed"):
        commit_mcp_preparation_deferred(
            repository,
            preparation,
            _replacement(),
        )


def test_terminal_deferred_commit_uses_same_finalize_receipt() -> None:
    preparation = _preparation(terminal=True)
    repository = _Repository(preparation)
    broker, human = _Broker(), _Human()

    retirement = commit_terminal_mcp_preparation_deferred(
        repository,
        preparation,
    )

    assert repository.terminal_commit_calls == 1
    assert broker.deleted == []
    assert human.cancelled == []
    finalize_mcp_preparation(
        repository,
        retirement,
        broker=broker,
        human_requests=human,
    )
    assert broker.deleted == ["old-remote-ref", "old-state-ref"]
    assert human.cancelled == [
        ("human-old", _H2, "mcp_human_binding_retired")
    ]


def test_finalize_rejects_a_prepared_abort_receipt() -> None:
    preparation = _preparation()
    repository = _Repository(preparation)

    with pytest.raises(ValidationError, match="retirement state is invalid"):
        finalize_mcp_preparation(
            repository,
            preparation,
            broker=_Broker(),
            human_requests=_Human(),
        )
