from __future__ import annotations

import hashlib
import json
from typing import Any

from agent_libos.models.exceptions import ValidationError
from agent_libos.ports import ProtectedEffectPort
from agent_libos.utils.serde import dumps, to_jsonable


APPROVAL_BINDING_KEY = "approval_binding"
APPROVAL_BINDING_SCHEMA_VERSION = 1
SEMANTIC_APPROVAL_BINDING_SCHEMA_VERSION = 2

_LEGACY_BINDING_FIELDS = frozenset(
    {"effect_id", "canonical_args_hash", "target_state_version"}
)

_TRANSIENT_KEYS = frozenset(
    {
        "sandbox_profile",
        "capability_ids",
        "selected_capability_id",
        "matched_capability_ids",
        "request_id",
        "human_request_id",
        "policy",
        "policy_level",
        "policy_reason",
        "grant_scope",
        "working_directory",
        "workspace_root",
        "high_risk",
        "matched_rule",
        "rpc_method",
        "mcp_name",
        "transport",
        "risk",
        "rule_id",
        "rule_effect",
        "reason",
    }
)


def canonical_effect_payload(context: dict[str, Any]) -> dict[str, Any]:
    """Return stable, payload-safe facts bound by a human approval."""

    return {
        str(key): to_jsonable(value)
        for key, value in sorted(context.items())
        if str(key) not in _TRANSIENT_KEYS
        and not str(key).endswith(("_preview", "_observation"))
    }


def canonical_effect_hash(context: dict[str, Any]) -> str:
    return hashlib.sha256(dumps(canonical_effect_payload(context)).encode("utf-8")).hexdigest()


def normalize_approval_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("approval binding must be an object")
    if is_semantic_approval_binding(value):
        # Keep the capability package independent of semantic policy
        # implementation details while sharing its strict public wire model.
        # The lazy import also avoids making the basic legacy Human binding
        # depend on semantic runtime construction.
        from agent_libos.models.semantic import SemanticApprovalBindingV2

        try:
            return SemanticApprovalBindingV2.from_dict(value).to_dict()
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
    unknown = sorted(set(value) - _LEGACY_BINDING_FIELDS)
    missing = sorted(_LEGACY_BINDING_FIELDS - set(value))
    if missing:
        raise ValidationError(
            "approval binding is missing fields: " + ", ".join(missing)
        )
    if unknown:
        raise ValidationError(
            "approval binding contains unknown fields: " + ", ".join(unknown)
        )
    if type(value.get("effect_id")) is not str:
        raise ValidationError("approval binding effect_id must be a string")
    if type(value.get("canonical_args_hash")) is not str:
        raise ValidationError(
            "approval binding canonical_args_hash must be a string"
        )
    target_state_version = value.get("target_state_version")
    if target_state_version is not None and (
        isinstance(target_state_version, bool)
        or not isinstance(target_state_version, (str, int))
    ):
        raise ValidationError(
            "approval binding target_state_version must be a string, integer, or null"
        )
    effect_id = str(value.get("effect_id") or "").strip()
    args_hash = str(value.get("canonical_args_hash") or "").strip().lower()
    if not effect_id.startswith("eff_"):
        raise ValidationError("approval binding requires a planned external effect id")
    if len(args_hash) != 64 or any(character not in "0123456789abcdef" for character in args_hash):
        raise ValidationError("approval binding requires a canonical SHA-256 argument hash")
    return {
        "effect_id": effect_id,
        "canonical_args_hash": args_hash,
        "target_state_version": value.get("target_state_version"),
    }


def is_semantic_approval_binding(value: Any) -> bool:
    """Identify BindingV2 without accepting lookalike or boolean versions."""

    return (
        isinstance(value, dict)
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == SEMANTIC_APPROVAL_BINDING_SCHEMA_VERSION
    )


def approval_binding_sha256(value: Any) -> str:
    """Digest the canonical strict binding without retaining effect payloads."""

    return hashlib.sha256(
        json.dumps(
            normalize_approval_binding(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def current_approval_effect_binding(
    store: ProtectedEffectPort,
    operation_id: str | None,
) -> dict[str, Any] | None:
    """Resolve a reserved one-shot approval from an explicit operation chain."""

    seen_operations: set[str] = set()
    bindings: dict[str, dict[str, Any]] = {}
    while operation_id is not None and operation_id not in seen_operations:
        seen_operations.add(operation_id)
        links = store.list_operation_evidence(
            operation_ids=[operation_id],
            evidence_types=["capability_reservation"],
        )
        for link in links:
            reservation = store.get_capability_use_reservation(link.evidence_id)
            cap_id = reservation.get("cap_id") if isinstance(reservation, dict) else None
            capability = store.get_capability(str(cap_id)) if cap_id else None
            raw_binding = (
                capability.constraints.get(APPROVAL_BINDING_KEY)
                if capability is not None
                else None
            )
            if raw_binding is None:
                continue
            binding = normalize_approval_binding(raw_binding)
            existing = bindings.get(binding["effect_id"])
            if existing is not None and existing != binding:
                raise ValidationError(
                    "external effect dispatch has conflicting approval bindings"
                )
            bindings[binding["effect_id"]] = binding
        operation = store.get_operation(operation_id)
        operation_id = operation.parent_operation_id if operation is not None else None
    if len(bindings) > 1:
        raise ValidationError("external effect dispatch has multiple approval bindings")
    return next(iter(bindings.values()), None)
