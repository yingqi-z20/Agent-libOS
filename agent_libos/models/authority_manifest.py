from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION = 2
PERMITTED_EFFECTS_POLICY_PROVENANCE_KEY = (
    "agent_libos_permitted_effects_policy_schema_version"
)


def encode_permitted_effects_policy(value: list[str] | None) -> dict[str, Any]:
    """Encode the v2 effect ceiling without collapsing ``None`` and ``[]``."""

    if value is not None and (
        not isinstance(value, list)
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError("permitted effects policy must be null or a list of strings")
    return {
        "schema_version": PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION,
        "effects": None if value is None else list(value),
    }


def upcast_permitted_effects_policy(value: Any) -> list[str] | None:
    """Decode v2 policy state and preserve the legacy v1 empty-list meaning.

    Version 1 stored the list directly and used an empty list for unrestricted
    compatibility mode.  Version 2 is tagged so ``None`` remains unrestricted
    while an explicit empty list can mean deny all.
    """

    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError("legacy permitted effects policy must contain strings")
        return list(value) if value else None
    if not isinstance(value, Mapping):
        raise ValueError("permitted effects policy must be a versioned object")
    if set(value) != {"schema_version", "effects"}:
        raise ValueError("permitted effects policy has unsupported fields")
    schema_version = value.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported permitted effects policy schema_version: "
            f"{schema_version!r}"
        )
    effects = value.get("effects")
    if effects is None:
        return None
    if not isinstance(effects, list) or not all(
        isinstance(item, str) for item in effects
    ):
        raise ValueError("permitted effects policy effects must be null or a list of strings")
    return list(effects)


@dataclass(frozen=True)
class TaskAuthorityManifest:
    """Host-authored launch authority contract for one process.

    Image capability declarations are copied into ``required_capabilities`` for
    comparison only.  Authority is issued exclusively from
    ``authorized_capabilities``.
    """

    manifest_id: str
    pid: str
    image_id: str
    goal_ref: str | None
    authorized_capabilities: list[dict[str, Any]] = field(default_factory=list)
    required_capabilities: list[dict[str, Any]] = field(default_factory=list)
    # None is unrestricted compatibility mode; [] is an explicit deny-all
    # ceiling. Persistence uses a tagged policy envelope so they cannot be
    # collapsed by JSON defaults or legacy upcasting.
    permitted_effects: list[str] | None = None
    resource_budget: dict[str, Any] = field(default_factory=dict)
    approval_policy: dict[str, Any] = field(default_factory=dict)
    data_flow_policy: dict[str, Any] = field(default_factory=dict)
    expires_at: str | None = None
    issued_by: str = "runtime.bootstrap"
    parent_manifest_id: str | None = None
    manifest_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    permitted_effects_policy_schema_version: int = (
        PERMITTED_EFFECTS_POLICY_SCHEMA_VERSION
    )

    @property
    def missing_required_capabilities(self) -> list[dict[str, Any]]:
        authorized = {
            (
                str(item.get("resource") or ""),
                tuple(sorted(str(right) for right in item.get("rights", []))),
            )
            for item in self.authorized_capabilities
        }
        return [
            dict(item)
            for item in self.required_capabilities
            if (
                str(item.get("resource") or ""),
                tuple(sorted(str(right) for right in item.get("rights", []))),
            )
            not in authorized
        ]
