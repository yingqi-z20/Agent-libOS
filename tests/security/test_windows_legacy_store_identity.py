from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from agent_libos.models import CapabilityRight
from agent_libos.models.exceptions import CapabilityDenied
from agent_libos.runtime.runtime import Runtime
from agent_libos.runtime.windows_store_identity import (
    LegacyWindowsStoreIdentityError,
    _validate_persisted_filesystem_identities,
)
from agent_libos.storage import SQLiteStore, UnitOfWork
from agent_libos.storage.contracts import (
    PersistedCapabilityResourceIdentity,
    PersistedCheckpointCapabilityInventory,
    PersistedFileLabelPathIdentity,
)
from agent_libos.substrate import LocalFilesystemProvider


class _PagedAuthority:
    def __init__(
        self,
        *,
        capabilities: tuple[PersistedCapabilityResourceIdentity, ...] = (),
        bindings: tuple[PersistedFileLabelPathIdentity, ...] = (),
    ) -> None:
        self.capabilities = capabilities
        self.bindings = bindings
        self.capability_cursors: list[str | None] = []
        self.binding_cursors: list[str | None] = []

    def query_active_capability_resource_identities(
        self,
        *,
        after_cap_id: str | None,
        limit: int,
    ) -> list[PersistedCapabilityResourceIdentity]:
        self.capability_cursors.append(after_cap_id)
        return _page(self.capabilities, after_cap_id, limit, "capability_id")

    def query_live_file_label_path_identities(
        self,
        *,
        after_binding_id: str | None,
        limit: int,
    ) -> list[PersistedFileLabelPathIdentity]:
        self.binding_cursors.append(after_binding_id)
        return _page(self.bindings, after_binding_id, limit, "binding_id")


class _PagedCheckpoints:
    def __init__(
        self,
        inventories: tuple[PersistedCheckpointCapabilityInventory, ...] = (),
    ) -> None:
        self.inventories = inventories
        self.cursors: list[str | None] = []

    def query_checkpoint_capability_inventories(
        self,
        *,
        after_checkpoint_id: str | None,
        limit: int,
    ) -> list[PersistedCheckpointCapabilityInventory]:
        self.cursors.append(after_checkpoint_id)
        return _page(
            self.inventories,
            after_checkpoint_id,
            limit,
            "checkpoint_id",
        )


class _CanonicalProvider:
    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.aliases = aliases or {}

    def resolve(self, path: object) -> SimpleNamespace:
        selected = str(path).replace("\\", "/")
        return SimpleNamespace(relative=self.aliases.get(selected, selected))


class _SensitiveFailureProvider:
    def __init__(self, message: str) -> None:
        self.message = message

    def resolve(self, path: object) -> SimpleNamespace:
        del path
        raise CapabilityDenied(self.message)


class _FailingAuthority(_PagedAuthority):
    def __init__(self, inventory: str, message: str) -> None:
        super().__init__()
        self.inventory = inventory
        self.message = message

    def query_active_capability_resource_identities(
        self,
        *,
        after_cap_id: str | None,
        limit: int,
    ) -> list[PersistedCapabilityResourceIdentity]:
        if self.inventory == "capability":
            raise CapabilityDenied(self.message)
        return super().query_active_capability_resource_identities(
            after_cap_id=after_cap_id,
            limit=limit,
        )

    def query_live_file_label_path_identities(
        self,
        *,
        after_binding_id: str | None,
        limit: int,
    ) -> list[PersistedFileLabelPathIdentity]:
        if self.inventory == "binding":
            raise CapabilityDenied(self.message)
        return super().query_live_file_label_path_identities(
            after_binding_id=after_binding_id,
            limit=limit,
        )


class _FailingCheckpoints(_PagedCheckpoints):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def query_checkpoint_capability_inventories(
        self,
        *,
        after_checkpoint_id: str | None,
        limit: int,
    ) -> list[PersistedCheckpointCapabilityInventory]:
        del after_checkpoint_id, limit
        raise CapabilityDenied(self.message)


class _FilesystemCanonicalizer:
    namespace = "workspace"

    def __init__(self, provider: object | None = None) -> None:
        self.provider = provider or _CanonicalProvider()

    @staticmethod
    def resource_for(path: str) -> str:
        encoded = "/".join(quote(part, safe="-._~") for part in path.split("/"))
        return "filesystem:workspace:" if encoded in {"", "."} else f"filesystem:workspace:{encoded}"

    @classmethod
    def directory_resource_for(cls, path: str) -> str:
        selected = path.rstrip("/")
        return (
            "filesystem:workspace:*"
            if selected in {"", "."}
            else f"{cls.resource_for(selected)}/*"
        )


def _page(
    records: tuple[object, ...],
    after: str | None,
    limit: int,
    field: str,
) -> list[object]:
    return [
        item
        for item in records
        if after is None or getattr(item, field) > after
    ][:limit]


def _capability(capability_id: str, resource: str) -> PersistedCapabilityResourceIdentity:
    return PersistedCapabilityResourceIdentity(
        capability_id=capability_id,
        resource=resource,
    )


def test_windows_legacy_identity_validation_pages_all_live_and_restorable_rows() -> None:
    authority = _PagedAuthority(
        capabilities=(
            _capability("cap_01", "object:one"),
            _capability("cap_02", "filesystem:workspace:Report.txt"),
            _capability("cap_03", "filesystem:workspace:reports/*"),
        ),
        bindings=(
            PersistedFileLabelPathIdentity("binding_01", "Report.txt"),
            PersistedFileLabelPathIdentity("binding_02", "reports/secret.txt"),
            PersistedFileLabelPathIdentity("binding_03", "third.txt"),
        ),
    )
    checkpoints = _PagedCheckpoints(
        (
            PersistedCheckpointCapabilityInventory(
                "ckpt_01",
                (_capability("cap_snapshot_01", "filesystem:workspace:Report.txt"),),
            ),
            PersistedCheckpointCapabilityInventory(
                "ckpt_02",
                (_capability("cap_snapshot_02", "object:two"),),
            ),
            PersistedCheckpointCapabilityInventory("ckpt_03", ()),
        )
    )

    summary = _validate_persisted_filesystem_identities(
        authority=authority,
        checkpoints=checkpoints,
        filesystem=_FilesystemCanonicalizer(),
        page_size=2,
    )

    assert summary.platform_checked is True
    assert summary.active_capabilities == 3
    assert summary.live_file_labels == 3
    assert summary.checkpoints == 3
    assert summary.checkpoint_capabilities == 2
    assert authority.capability_cursors == [None, "cap_02"]
    assert authority.binding_cursors == [None, "binding_02"]
    assert checkpoints.cursors == [None, "ckpt_02"]


@pytest.mark.parametrize("inventory", ["capability", "binding", "checkpoint"])
def test_windows_legacy_identity_validation_rejects_aliases_with_opaque_ids_only(
    inventory: str,
) -> None:
    alias = "reports/SECRET.txt"
    canonical = "reports/Secret.txt"
    authority = _PagedAuthority()
    checkpoints = _PagedCheckpoints()
    if inventory == "capability":
        authority.capabilities = (
            _capability("cap_legacy_alias", f"filesystem:workspace:{alias}"),
        )
    elif inventory == "binding":
        authority.bindings = (
            PersistedFileLabelPathIdentity("binding_legacy_alias", alias),
        )
    else:
        checkpoints.inventories = (
            PersistedCheckpointCapabilityInventory(
                "ckpt_legacy_alias",
                (_capability("cap_checkpoint_alias", f"filesystem:workspace:{alias}"),),
            ),
        )

    with pytest.raises(LegacyWindowsStoreIdentityError) as raised:
        _validate_persisted_filesystem_identities(
            authority=authority,
            checkpoints=checkpoints,
            filesystem=_FilesystemCanonicalizer(
                _CanonicalProvider({alias: canonical})
            ),
            page_size=2,
        )

    diagnostic = str(raised.value)
    assert alias not in diagnostic
    assert canonical not in diagnostic
    assert "legacy_alias" in diagnostic


@pytest.mark.parametrize(
    "failure",
    (
        "capability-page",
        "capability-parse",
        "capability-provider",
        "binding-page",
        "binding-provider",
        "checkpoint-page",
        "checkpoint-provider",
    ),
)
def test_windows_legacy_identity_errors_discard_sensitive_exception_context(
    failure: str,
) -> None:
    alias = "private/ALIAS-secret.txt"
    canonical = "private/Canonical-secret.txt"
    sensitive_message = f"cannot resolve {alias} as {canonical}"
    authority: _PagedAuthority = _PagedAuthority()
    checkpoints: _PagedCheckpoints = _PagedCheckpoints()
    provider: object = _CanonicalProvider()

    if failure == "capability-page":
        authority = _FailingAuthority("capability", sensitive_message)
    elif failure == "capability-parse":
        authority.capabilities = (
            _capability(
                "cap_sensitive_parse",
                f"filesystem:workspace:{alias}*invalid",
            ),
        )
    elif failure == "capability-provider":
        authority.capabilities = (
            _capability(
                "cap_sensitive_provider",
                f"filesystem:workspace:{alias}",
            ),
        )
        provider = _SensitiveFailureProvider(sensitive_message)
    elif failure == "binding-page":
        authority = _FailingAuthority("binding", sensitive_message)
    elif failure == "binding-provider":
        authority.bindings = (
            PersistedFileLabelPathIdentity("binding_sensitive_provider", alias),
        )
        provider = _SensitiveFailureProvider(sensitive_message)
    elif failure == "checkpoint-page":
        checkpoints = _FailingCheckpoints(sensitive_message)
    else:
        checkpoints.inventories = (
            PersistedCheckpointCapabilityInventory(
                "ckpt_sensitive_provider",
                (
                    _capability(
                        "cap_sensitive_checkpoint",
                        f"filesystem:workspace:{alias}",
                    ),
                ),
            ),
        )
        provider = _SensitiveFailureProvider(sensitive_message)

    with pytest.raises(LegacyWindowsStoreIdentityError) as raised:
        _validate_persisted_filesystem_identities(
            authority=authority,
            checkpoints=checkpoints,
            filesystem=_FilesystemCanonicalizer(provider),
            page_size=2,
        )

    error = raised.value
    formatted = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert error.__context__ is None
    assert error.__cause__ is None
    assert alias not in formatted
    assert canonical not in formatted


def test_windows_legacy_identity_sql_pages_exclude_inactive_history() -> None:
    store = SQLiteStore(":memory:")
    try:
        with store.transaction() as cursor:
            cursor.executemany(
                """
                INSERT INTO capabilities (
                    cap_id, subject, resource, rights_json, constraints_json,
                    issued_by, issued_at, expires_at, delegable, revocable,
                    effect, issuer_cap_id, parent_cap_id, delegation_depth,
                    max_delegation_depth, uses_remaining, status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ("cap_active", "pid", "filesystem:workspace:live.txt", '["read"]', "{}", "test", "2026-01-01", None, 0, 1, "allow", None, None, 0, None, None, "active", "{}"),
                    ("cap_revoked", "pid", "filesystem:workspace:legacy.txt", '["read"]', "{}", "test", "2026-01-01", None, 0, 1, "allow", None, None, 0, None, None, "revoked", "{}"),
                ),
            )
            cursor.executemany(
                """
                INSERT INTO file_label_bindings (
                    binding_id, normalized_path, content_sha256, labels_json,
                    source_refs_json, generation, tombstoned, active,
                    created_by, created_at, superseded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ("binding_active", "live.txt", "a" * 64, "{}", "[]", 1, 0, 1, "test", "2026-01-01", None),
                    ("binding_inactive", "old.txt", "b" * 64, "{}", "[]", 1, 0, 0, "test", "2026-01-01", "2026-01-02"),
                    ("binding_tombstone", "gone.txt", None, "{}", "[]", 1, 1, 0, "test", "2026-01-01", "2026-01-02"),
                ),
            )

        uow = UnitOfWork(store)
        capabilities = uow.authority.query_active_capability_resource_identities(
            after_cap_id=None,
            limit=1,
        )
        labels = uow.authority.query_live_file_label_path_identities(
            after_binding_id=None,
            limit=1,
        )

        assert [item.capability_id for item in capabilities] == ["cap_active"]
        assert [item.binding_id for item in labels] == ["binding_active"]
    finally:
        store.close()


def test_checkpoint_capability_inventory_includes_every_restorable_row() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="persist checkpoint capability inventory")
        capability = runtime.capability.grant(
            pid,
            "filesystem:workspace:Report.txt",
            [CapabilityRight.READ],
            issued_by="test",
        )
        checkpoint_id = runtime.checkpoint.create(
            pid,
            "inventory",
            actor=pid,
            require_capability=False,
        )
        checkpoint_rows = runtime.store._query(
            "SELECT snapshot_json FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        )
        snapshot = json.loads(checkpoint_rows[0]["snapshot_json"])
        snapshot_capability = next(
            row
            for row in snapshot["rows"]["capabilities"]
            if row["cap_id"] == capability.cap_id
        )
        # Restore can reactivate a retained non-ACTIVE row when its current
        # capability is absent, so the inventory must not filter on snapshot
        # status. Simulate that exact durable legacy condition directly.
        snapshot_capability["status"] = "revoked"
        with runtime.store.transaction() as cursor:
            cursor.execute(
                "UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?",
                (json.dumps(snapshot, sort_keys=True), checkpoint_id),
            )
            cursor.execute(
                "DELETE FROM capabilities WHERE cap_id = ?",
                (capability.cap_id,),
            )

        inventories = runtime.uow.snapshots.query_checkpoint_capability_inventories(
            after_checkpoint_id=None,
            limit=1,
        )

        selected = next(item for item in inventories if item.checkpoint_id == checkpoint_id)
        assert (capability.cap_id, capability.resource) in {
            (item.capability_id, item.resource) for item in selected.capabilities
        }
    finally:
        runtime.close()


@pytest.mark.platform_windows
def test_windows_native_legacy_alias_capability_is_rejected_without_path_disclosure(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Win32 path identity regression")
    root = tmp_path / "workspace"
    root.mkdir()
    stored = root / "Secret.txt"
    stored.write_text("secret", encoding="utf-8")
    provider = LocalFilesystemProvider(root)
    filesystem = _FilesystemCanonicalizer(provider)
    legacy_spelling = "secret.txt"

    with pytest.raises(LegacyWindowsStoreIdentityError) as raised:
        _validate_persisted_filesystem_identities(
            authority=_PagedAuthority(
                capabilities=(
                    _capability(
                        "cap_native_alias",
                        f"filesystem:workspace:{legacy_spelling}",
                    ),
                )
            ),
            checkpoints=_PagedCheckpoints(),
            filesystem=filesystem,
            page_size=10,
        )

    diagnostic = str(raised.value)
    assert legacy_spelling not in diagnostic
    assert "Secret.txt" not in diagnostic
    assert "cap_native_alias" in diagnostic
