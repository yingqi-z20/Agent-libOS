from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import unquote

from agent_libos.capability.resources import ResourceAuthority
from agent_libos.models import ResourceScope
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.storage.contracts import (
    PersistedCapabilityResourceIdentity,
    PersistedCheckpointCapabilityInventory,
    PersistedFileLabelPathIdentity,
)


WINDOWS_STORE_IDENTITY_VALIDATION_PAGE_SIZE = 100
_OPAQUE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")


class _ResolvedPath(Protocol):
    relative: str


class _FilesystemProvider(Protocol):
    def resolve(self, path: Any) -> _ResolvedPath: ...


class FilesystemIdentityCanonicalizer(Protocol):
    """Narrow read-only filesystem surface used by startup validation."""

    namespace: str
    provider: _FilesystemProvider

    def resource_for(self, path: str) -> str: ...

    def directory_resource_for(self, path: str) -> str: ...


class AuthorityIdentityReader(Protocol):
    def query_active_capability_resource_identities(
        self,
        *,
        after_cap_id: str | None,
        limit: int,
    ) -> list[PersistedCapabilityResourceIdentity]: ...

    def query_live_file_label_path_identities(
        self,
        *,
        after_binding_id: str | None,
        limit: int,
    ) -> list[PersistedFileLabelPathIdentity]: ...


class CheckpointIdentityReader(Protocol):
    def query_checkpoint_capability_inventories(
        self,
        *,
        after_checkpoint_id: str | None,
        limit: int,
    ) -> list[PersistedCheckpointCapabilityInventory]: ...


@dataclass(frozen=True, slots=True)
class WindowsStoreIdentityValidationSummary:
    """Read-only inventory counts from one startup validation pass."""

    platform_checked: bool
    active_capabilities: int = 0
    live_file_labels: int = 0
    checkpoints: int = 0
    checkpoint_capabilities: int = 0


class LegacyWindowsStoreIdentityError(ValidationError):
    """Persisted authority could name one Windows identity more than once."""


def validate_legacy_windows_store_identities(
    *,
    authority: AuthorityIdentityReader,
    checkpoints: CheckpointIdentityReader,
    filesystem: FilesystemIdentityCanonicalizer,
    page_size: int = WINDOWS_STORE_IDENTITY_VALIDATION_PAGE_SIZE,
) -> WindowsStoreIdentityValidationSummary:
    """Fail closed when live/restorable Windows path identities are non-canonical.

    This function is deliberately idempotent and read-only. Runtime assembly
    calls it once before durable recovery effects and again under the recovery
    lease after reconciliation, because effect settlement can reactivate a
    capability reservation. Non-Windows Hosts have no Win32 alias identity to
    validate and return without touching the Store.
    """

    if os.name != "nt":
        return WindowsStoreIdentityValidationSummary(platform_checked=False)
    return _validate_persisted_filesystem_identities(
        authority=authority,
        checkpoints=checkpoints,
        filesystem=filesystem,
        page_size=page_size,
    )


def _validate_persisted_filesystem_identities(
    *,
    authority: AuthorityIdentityReader,
    checkpoints: CheckpointIdentityReader,
    filesystem: FilesystemIdentityCanonicalizer,
    page_size: int,
) -> WindowsStoreIdentityValidationSummary:
    if type(page_size) is not int or page_size <= 0:
        raise ValidationError(
            "Windows persisted identity validation page_size must be a positive integer"
        )

    active_capabilities = _validate_active_capabilities(
        authority=authority,
        filesystem=filesystem,
        page_size=page_size,
    )
    live_file_labels = _validate_live_file_labels(
        authority=authority,
        filesystem=filesystem,
        page_size=page_size,
    )
    checkpoint_count, checkpoint_capabilities = _validate_checkpoints(
        checkpoints=checkpoints,
        filesystem=filesystem,
        page_size=page_size,
    )
    return WindowsStoreIdentityValidationSummary(
        platform_checked=True,
        active_capabilities=active_capabilities,
        live_file_labels=live_file_labels,
        checkpoints=checkpoint_count,
        checkpoint_capabilities=checkpoint_capabilities,
    )


def _validate_active_capabilities(
    *,
    authority: AuthorityIdentityReader,
    filesystem: FilesystemIdentityCanonicalizer,
    page_size: int,
) -> int:
    after_cap_id: str | None = None
    scanned = 0
    while True:
        page_failed = False
        try:
            page = authority.query_active_capability_resource_identities(
                after_cap_id=after_cap_id,
                limit=page_size,
            )
        except Exception:
            page_failed = True
            page = []
        if page_failed:
            raise LegacyWindowsStoreIdentityError(
                "cannot validate persisted Windows capability identities"
            )
        _require_stable_page(
            page,
            after=after_cap_id,
            limit=page_size,
            identifier=lambda item: item.capability_id,
            inventory="capability",
        )
        for item in page:
            _require_canonical_capability_resource(
                item,
                filesystem=filesystem,
                checkpoint_id=None,
            )
        scanned += len(page)
        if len(page) < page_size:
            return scanned
        after_cap_id = page[-1].capability_id


def _validate_live_file_labels(
    *,
    authority: AuthorityIdentityReader,
    filesystem: FilesystemIdentityCanonicalizer,
    page_size: int,
) -> int:
    after_binding_id: str | None = None
    scanned = 0
    while True:
        page_failed = False
        try:
            page = authority.query_live_file_label_path_identities(
                after_binding_id=after_binding_id,
                limit=page_size,
            )
        except Exception:
            page_failed = True
            page = []
        if page_failed:
            raise LegacyWindowsStoreIdentityError(
                "cannot validate persisted Windows file-label identities"
            )
        _require_stable_page(
            page,
            after=after_binding_id,
            limit=page_size,
            identifier=lambda item: item.binding_id,
            inventory="file-label binding",
        )
        for item in page:
            resolution_failed = False
            try:
                canonical = filesystem.provider.resolve(
                    item.normalized_path
                ).relative
            except Exception:
                resolution_failed = True
                canonical = ""
            if resolution_failed:
                _raise_noncanonical_identity(
                    "binding_id",
                    item.binding_id,
                )
            if canonical != item.normalized_path:
                _raise_noncanonical_identity(
                    "binding_id",
                    item.binding_id,
                )
        scanned += len(page)
        if len(page) < page_size:
            return scanned
        after_binding_id = page[-1].binding_id


def _validate_checkpoints(
    *,
    checkpoints: CheckpointIdentityReader,
    filesystem: FilesystemIdentityCanonicalizer,
    page_size: int,
) -> tuple[int, int]:
    after_checkpoint_id: str | None = None
    checkpoint_count = 0
    capability_count = 0
    while True:
        page_failed = False
        try:
            page = checkpoints.query_checkpoint_capability_inventories(
                after_checkpoint_id=after_checkpoint_id,
                limit=page_size,
            )
        except Exception:
            page_failed = True
            page = []
        if page_failed:
            raise LegacyWindowsStoreIdentityError(
                "cannot validate persisted Windows checkpoint identities"
            )
        _require_stable_page(
            page,
            after=after_checkpoint_id,
            limit=page_size,
            identifier=lambda item: item.checkpoint_id,
            inventory="checkpoint",
        )
        for inventory in page:
            for capability in inventory.capabilities:
                _require_canonical_capability_resource(
                    capability,
                    filesystem=filesystem,
                    checkpoint_id=inventory.checkpoint_id,
                )
            capability_count += len(inventory.capabilities)
        checkpoint_count += len(page)
        if len(page) < page_size:
            return checkpoint_count, capability_count
        after_checkpoint_id = page[-1].checkpoint_id


def _require_canonical_capability_resource(
    capability: PersistedCapabilityResourceIdentity,
    *,
    filesystem: FilesystemIdentityCanonicalizer,
    checkpoint_id: str | None,
) -> None:
    resource = capability.resource
    parse_failed = False
    try:
        pattern = ResourceAuthority().parse(resource)
    except CapabilityDenied:
        parse_failed = True
        pattern = None
    if parse_failed:
        if resource.strip().startswith("filesystem:"):
            _raise_capability_identity(capability.capability_id, checkpoint_id)
        return
    assert pattern is not None
    if pattern.kind != "filesystem":
        return
    if pattern.scope is ResourceScope.PREFIX and not pattern.body:
        if resource != "filesystem:*":
            _raise_capability_identity(capability.capability_id, checkpoint_id)
        return

    namespace, separator, encoded_path = pattern.body.partition(":")
    if namespace != filesystem.namespace:
        return
    if pattern.scope is ResourceScope.PREFIX:
        expected = f"filesystem:{filesystem.namespace}:*"
        if resource != expected or pattern.body != filesystem.namespace:
            _raise_capability_identity(capability.capability_id, checkpoint_id)
        return
    if not separator:
        _raise_capability_identity(capability.capability_id, checkpoint_id)

    resolution_failed = False
    try:
        decoded_path = unquote(encoded_path, encoding="utf-8", errors="strict")
        canonical_path = filesystem.provider.resolve(decoded_path).relative
        if pattern.scope is ResourceScope.EXACT:
            expected = filesystem.resource_for(canonical_path)
        elif pattern.scope is ResourceScope.SUBTREE:
            expected = filesystem.directory_resource_for(canonical_path)
        else:
            resolution_failed = True
            expected = ""
    except Exception:
        resolution_failed = True
        expected = ""
    if resolution_failed:
        _raise_capability_identity(capability.capability_id, checkpoint_id)
    if resource != expected:
        _raise_capability_identity(capability.capability_id, checkpoint_id)


def _require_stable_page(
    page: list[Any],
    *,
    after: str | None,
    limit: int,
    identifier: Any,
    inventory: str,
) -> None:
    if not isinstance(page, list) or len(page) > limit:
        raise LegacyWindowsStoreIdentityError(
            f"persisted Windows {inventory} inventory returned an invalid page"
        )
    previous = after
    for item in page:
        selected = identifier(item)
        if (
            type(selected) is not str
            or not selected
            or (previous is not None and selected <= previous)
        ):
            raise LegacyWindowsStoreIdentityError(
                f"persisted Windows {inventory} inventory is not cursor-stable"
            )
        previous = selected


def _raise_capability_identity(
    capability_id: str,
    checkpoint_id: str | None,
) -> None:
    fields = []
    if checkpoint_id is not None:
        fields.append(f"checkpoint_id={_opaque_identifier(checkpoint_id)}")
    fields.append(f"capability_id={_opaque_identifier(capability_id)}")
    raise LegacyWindowsStoreIdentityError(
        "persisted Windows filesystem capability identity is non-canonical: "
        + ", ".join(fields)
    )


def _raise_noncanonical_identity(field: str, value: str) -> None:
    raise LegacyWindowsStoreIdentityError(
        "persisted Windows filesystem path identity is non-canonical: "
        f"{field}={_opaque_identifier(value)}"
    )


def _opaque_identifier(value: Any) -> str:
    text = value if type(value) is str else ""
    if _OPAQUE_ID.fullmatch(text):
        return text
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return f"invalid_{digest[:16]}"
