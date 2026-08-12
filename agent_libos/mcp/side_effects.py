"""Crash-atomic ownership records for MCP Human and broker side effects."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Protocol, runtime_checkable

from agent_libos.mcp.human import McpHumanRequestBridge
from agent_libos.mcp.providers import McpCredentialBroker
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.storage.mcp_v7 import (
    McpContinuationRecord,
    McpRemoteTaskRecord,
    McpSideEffectPreparationRecord,
)
from agent_libos.utils.ids import new_id


@runtime_checkable
class McpSideEffectRepository(Protocol):
    def insert(
        self,
        record: McpSideEffectPreparationRecord,
    ) -> McpSideEffectPreparationRecord: ...

    def get(
        self,
        preparation_id: str,
    ) -> McpSideEffectPreparationRecord | None: ...

    def list(self, **filters: object) -> tuple[McpSideEffectPreparationRecord, ...]: ...

    def compare_and_swap(
        self,
        preparation_id: str,
        *,
        expected_revision: int,
        replacement: McpSideEffectPreparationRecord,
    ) -> bool: ...

    def delete(self, preparation_id: str, *, expected_revision: int) -> bool: ...

    def commit(
        self,
        preparation_id: str,
        *,
        expected_revision: int,
        replacement: McpContinuationRecord | McpRemoteTaskRecord,
    ) -> bool: ...

    def commit_terminal(
        self,
        preparation_id: str,
        *,
        expected_revision: int,
    ) -> bool: ...


def prepare_mcp_side_effects(
    *,
    repository: McpSideEffectRepository,
    broker: McpCredentialBroker,
    operation_kind: str,
    operation_id: str,
    operation_revision: int | None,
    server_id: str,
    server_spec_sha256: str,
    server_generation: int,
    owner_id: str,
    auth_principal_sha256: str,
    auth_scope_sha256: str,
    human_request_id: str | None,
    human_preview_sha256: str | None,
    broker_namespace: str | None,
    broker_value_sha256: str | None,
    result_namespace: str | None,
    result_sha256: str | None,
    expires_at: str,
    created_at: str,
    retire_refs: tuple[str, ...] = (),
    retire_human_request_id: str | None = None,
    retire_human_preview_sha256: str | None = None,
) -> McpSideEffectPreparationRecord:
    """Persist ownership before any Human or secret value is created."""

    if (broker_namespace is not None or result_namespace is not None) and not broker.available():
        raise ValidationError("MCP credential broker is unavailable")
    if (retire_human_request_id is None) != (
        retire_human_preview_sha256 is None
    ):
        raise ValidationError("MCP retired Human binding is incomplete")
    broker_ref = (
        _reserve_secret_ref(broker, broker_namespace)
        if broker_namespace is not None
        else None
    )
    result_ref = (
        _reserve_secret_ref(broker, result_namespace)
        if result_namespace is not None
        else None
    )
    metadata: dict[str, object] = {
        "automatic_retry_disabled": True,
        "cleanup_mode": "abort",
        "retire_refs": retire_refs,
    }
    if retire_human_request_id is not None:
        metadata["retire_human_request_id"] = retire_human_request_id
        metadata["retire_human_preview_sha256"] = retire_human_preview_sha256
    record = McpSideEffectPreparationRecord(
        preparation_id=new_id("mcpprep"),
        operation_kind=operation_kind,
        operation_id=operation_id,
        operation_revision=operation_revision,
        server_id=server_id,
        server_spec_sha256=server_spec_sha256,
        server_generation=server_generation,
        owner_id=owner_id,
        auth_principal_sha256=auth_principal_sha256,
        auth_scope_sha256=auth_scope_sha256,
        human_request_id=human_request_id,
        human_preview_sha256=human_preview_sha256,
        broker_ref=broker_ref,
        broker_value_sha256=broker_value_sha256,
        result_ref=result_ref,
        result_sha256=result_sha256,
        status="prepared",
        revision=0,
        expires_at=expires_at,
        metadata=metadata,
        created_at=created_at,
        updated_at=created_at,
    )
    persisted = repository.insert(record)
    if persisted != record:
        raise ValidationError("MCP side-effect preparation changed during insert")
    return record


def write_mcp_prepared_secrets(
    preparation: McpSideEffectPreparationRecord,
    *,
    broker: McpCredentialBroker,
    broker_namespace: str | None,
    broker_value: bytes | None,
    result_namespace: str | None,
    result_value: bytes | None,
) -> None:
    _write_prepared_secret(
        broker,
        reference=preparation.broker_ref,
        namespace=broker_namespace,
        value=broker_value,
        expected_sha256=preparation.broker_value_sha256,
        expires_at=preparation.expires_at,
    )
    _write_prepared_secret(
        broker,
        reference=preparation.result_ref,
        namespace=result_namespace,
        value=result_value,
        expected_sha256=preparation.result_sha256,
        expires_at=preparation.expires_at,
    )


def commit_mcp_preparation(
    repository: McpSideEffectRepository,
    preparation: McpSideEffectPreparationRecord,
    replacement: McpContinuationRecord | McpRemoteTaskRecord,
    *,
    broker: McpCredentialBroker,
    human_requests: McpHumanRequestBridge,
) -> None:
    retirement = commit_mcp_preparation_deferred(
        repository,
        preparation,
        replacement,
    )
    finalize_mcp_preparation(
        repository,
        retirement,
        broker=broker,
        human_requests=human_requests,
    )


def commit_mcp_preparation_deferred(
    repository: McpSideEffectRepository,
    preparation: McpSideEffectPreparationRecord,
    replacement: McpContinuationRecord | McpRemoteTaskRecord,
) -> McpSideEffectPreparationRecord:
    """Atomically commit one operation row while deferring external cleanup.

    Callers may run this helper inside a wider RuntimeStore transaction.  It
    deliberately performs no broker or Human mutation: the returned durable
    ``cleaning/retire`` row is the exact post-commit cleanup receipt.
    """

    if not repository.commit(
        preparation.preparation_id,
        expected_revision=preparation.revision,
        replacement=replacement,
    ):
        raise ValidationError("MCP side-effect preparation commit conflict")
    retirement = repository.get(preparation.preparation_id)
    return _require_deferred_retirement(
        preparation,
        retirement,
        updated_at=max(preparation.updated_at, replacement.updated_at),
        terminal=False,
    )


def commit_terminal_mcp_preparation(
    repository: McpSideEffectRepository,
    preparation: McpSideEffectPreparationRecord,
    *,
    broker: McpCredentialBroker,
    human_requests: McpHumanRequestBridge,
) -> None:
    retirement = commit_terminal_mcp_preparation_deferred(
        repository,
        preparation,
    )
    finalize_mcp_preparation(
        repository,
        retirement,
        broker=broker,
        human_requests=human_requests,
    )


def commit_terminal_mcp_preparation_deferred(
    repository: McpSideEffectRepository,
    preparation: McpSideEffectPreparationRecord,
) -> McpSideEffectPreparationRecord:
    """Atomically delete one terminal operation while deferring cleanup."""

    if not repository.commit_terminal(
        preparation.preparation_id,
        expected_revision=preparation.revision,
    ):
        raise ValidationError("MCP terminal side-effect preparation conflict")
    retirement = repository.get(preparation.preparation_id)
    return _require_deferred_retirement(
        preparation,
        retirement,
        updated_at=preparation.updated_at,
        terminal=True,
    )


def finalize_mcp_preparation(
    repository: McpSideEffectRepository,
    retirement: McpSideEffectPreparationRecord,
    *,
    broker: McpCredentialBroker,
    human_requests: McpHumanRequestBridge,
) -> None:
    """Idempotently finish one committed ``cleaning/retire`` receipt."""

    _require_retirement_shape(retirement)
    current = repository.get(retirement.preparation_id)
    if current is None:
        return
    if current != retirement:
        raise ValidationError("MCP side-effect retirement readback changed")
    _cleanup_committed_retirement(
        repository,
        current,
        broker=broker,
        human_requests=human_requests,
    )


def cleanup_mcp_preparation(
    repository: McpSideEffectRepository,
    preparation: McpSideEffectPreparationRecord,
    *,
    broker: McpCredentialBroker,
    human_requests: McpHumanRequestBridge,
    updated_at: str,
    reason: str,
) -> None:
    current = repository.get(preparation.preparation_id)
    if current is None:
        return
    if current.status == "prepared":
        abort_metadata = dict(current.metadata)
        abort_metadata["automatic_retry_disabled"] = True
        abort_metadata["cleanup_mode"] = "abort"
        claimed = replace(
            current,
            status="cleaning",
            revision=current.revision + 1,
            metadata=abort_metadata,
            updated_at=max(current.updated_at, updated_at),
        )
        if not repository.compare_and_swap(
            current.preparation_id,
            expected_revision=current.revision,
            replacement=claimed,
        ):
            raise ValidationError("MCP side-effect cleanup claim conflict")
        current = claimed
    if current.status != "cleaning":
        raise ValidationError("MCP side-effect cleanup state is invalid")
    cleanup_mode = current.metadata.get("cleanup_mode")
    if cleanup_mode == "abort":
        _cancel_prepared_human(human_requests, current, reason=reason)
        _delete_prepared_secret(broker, current.broker_ref)
        _delete_prepared_secret(broker, current.result_ref)
    elif cleanup_mode == "retire":
        retire_refs = current.metadata.get("retire_refs")
        if not isinstance(retire_refs, (list, tuple)):
            raise ValidationError("MCP side-effect retire refs are invalid")
        for reference in retire_refs:
            _delete_prepared_secret(broker, reference)
        _cancel_retired_human(human_requests, current)
    else:
        raise ValidationError("MCP side-effect cleanup mode is invalid")
    if not repository.delete(
        current.preparation_id,
        expected_revision=current.revision,
    ):
        raise ValidationError("MCP side-effect cleanup delete conflict")


def _cleanup_committed_retirement(
    repository: McpSideEffectRepository,
    preparation: McpSideEffectPreparationRecord,
    *,
    broker: McpCredentialBroker,
    human_requests: McpHumanRequestBridge,
) -> None:
    _require_retirement_shape(preparation)
    retire_refs = preparation.metadata.get("retire_refs")
    assert isinstance(retire_refs, (list, tuple))
    for reference in retire_refs:
        _delete_prepared_secret(broker, reference)
    _cancel_retired_human(human_requests, preparation)
    if not repository.delete(
        preparation.preparation_id,
        expected_revision=preparation.revision,
    ):
        if repository.get(preparation.preparation_id) is None:
            return
        raise ValidationError("MCP side-effect retirement delete conflict")


def _require_deferred_retirement(
    preparation: McpSideEffectPreparationRecord,
    retirement: McpSideEffectPreparationRecord | None,
    *,
    updated_at: str,
    terminal: bool,
) -> McpSideEffectPreparationRecord:
    if retirement is None:
        raise ValidationError("MCP side-effect retirement receipt is unavailable")
    if terminal:
        metadata = dict(preparation.metadata)
        metadata["cleanup_mode"] = "retire"
    else:
        metadata = {
            "automatic_retry_disabled": True,
            "cleanup_mode": "retire",
            "retire_refs": tuple(preparation.metadata.get("retire_refs", ())),
        }
        for key in (
            "retire_human_request_id",
            "retire_human_preview_sha256",
        ):
            if key in preparation.metadata:
                metadata[key] = preparation.metadata[key]
    expected = replace(
        preparation,
        status="cleaning",
        revision=preparation.revision + 1,
        metadata=metadata,
        updated_at=updated_at,
    )
    if retirement != expected:
        raise ValidationError("MCP side-effect retirement receipt changed")
    _require_retirement_shape(retirement)
    return retirement


def _require_retirement_shape(
    retirement: McpSideEffectPreparationRecord,
) -> None:
    if not isinstance(retirement, McpSideEffectPreparationRecord):
        raise ValidationError("MCP side-effect retirement receipt is invalid")
    if (
        retirement.status != "cleaning"
        or retirement.metadata.get("cleanup_mode") != "retire"
    ):
        raise ValidationError("MCP side-effect retirement state is invalid")
    retire_refs = retirement.metadata.get("retire_refs")
    if not isinstance(retire_refs, (list, tuple)) or len(retire_refs) > 2:
        raise ValidationError("MCP side-effect retirement refs are invalid")
    if any(type(reference) is not str or not reference for reference in retire_refs):
        raise ValidationError("MCP side-effect retirement ref is invalid")


def reconcile_mcp_preparations(
    repository: McpSideEffectRepository,
    *,
    operation_kind: str,
    broker: McpCredentialBroker,
    human_requests: McpHumanRequestBridge,
    updated_at: str,
) -> int:
    changed = 0
    for status in ("prepared", "cleaning"):
        while True:
            batch = repository.list(
                operation_kind=operation_kind,
                status=status,
                limit=500,
            )
            if not batch:
                break
            for preparation in batch:
                cleanup_mcp_preparation(
                    repository,
                    preparation,
                    broker=broker,
                    human_requests=human_requests,
                    updated_at=updated_at,
                    reason="runtime_restart",
                )
                changed += 1
    return changed


def _reserve_secret_ref(
    broker: McpCredentialBroker,
    namespace: str,
) -> str:
    reference = broker.reserve_secret_ref(namespace)
    if type(reference) is not str or not reference:
        raise ValidationError("MCP credential broker reference is invalid")
    return reference


def _write_prepared_secret(
    broker: McpCredentialBroker,
    *,
    reference: str | None,
    namespace: str | None,
    value: bytes | None,
    expected_sha256: str | None,
    expires_at: str,
) -> None:
    if reference is None:
        if namespace is not None or value is not None or expected_sha256 is not None:
            raise ValidationError("MCP prepared secret binding is incomplete")
        return
    if namespace is None or value is None or expected_sha256 is None:
        raise ValidationError("MCP prepared secret binding is incomplete")
    if sha256(value).hexdigest() != expected_sha256:
        raise ValidationError("MCP prepared secret digest changed")
    try:
        broker.put_secret_at(
            reference,
            namespace,
            value,
            expires_at=expires_at,
        )
    except Exception as exc:
        raise ValidationError("MCP credential broker write failed") from exc


def _cancel_prepared_human(
    human_requests: McpHumanRequestBridge,
    preparation: McpSideEffectPreparationRecord,
    *,
    reason: str,
) -> None:
    if preparation.human_request_id is None:
        return
    if preparation.human_preview_sha256 is None:
        raise ValidationError("MCP prepared Human binding is incomplete")
    try:
        human_requests.cancel_question(
            preparation.human_request_id,
            preview_sha256=preparation.human_preview_sha256,
            reason=reason,
        )
    except NotFound:
        return


def _cancel_retired_human(
    human_requests: McpHumanRequestBridge,
    preparation: McpSideEffectPreparationRecord,
) -> None:
    request_id = preparation.metadata.get("retire_human_request_id")
    preview_sha256 = preparation.metadata.get("retire_human_preview_sha256")
    if request_id is None and preview_sha256 is None:
        return
    if type(request_id) is not str or type(preview_sha256) is not str:
        raise ValidationError("MCP retired Human binding is invalid")
    try:
        human_requests.cancel_question(
            request_id,
            preview_sha256=preview_sha256,
            reason="mcp_human_binding_retired",
        )
    except NotFound:
        return


def _delete_prepared_secret(
    broker: McpCredentialBroker,
    reference: str | None,
) -> None:
    if reference is None:
        return
    if not broker.available():
        raise ValidationError("MCP credential broker is unavailable")
    try:
        broker.delete_secret(reference)
    except Exception as exc:
        raise ValidationError("MCP credential broker cleanup failed") from exc

__all__ = [
    "McpSideEffectRepository",
    "cleanup_mcp_preparation",
    "commit_mcp_preparation",
    "commit_mcp_preparation_deferred",
    "commit_terminal_mcp_preparation",
    "commit_terminal_mcp_preparation_deferred",
    "finalize_mcp_preparation",
    "prepare_mcp_side_effects",
    "reconcile_mcp_preparations",
    "write_mcp_prepared_secrets",
]
