from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agent_libos.capability.effect_binding import (
    APPROVAL_BINDING_KEY,
    canonical_effect_hash,
    normalize_approval_binding,
)
from agent_libos.capability.evaluator import (
    DATA_RELEASE_BINDING_KEY,
    KNOWN_CONSTRAINT_KEYS,
    CapabilityEvaluator,
)
from agent_libos.capability.rules import AUTHORITY_RULES_KEY, AuthorityRuleCodec
from agent_libos.models import CapabilityEffect, CapabilityRight, HumanRequest
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.serde import to_jsonable


_ACTION_RE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")
_PAYLOAD_FIELDS = frozenset(
    {
        "type",
        "question",
        "requested_once_capability",
        "context",
        "effect_binding",
        # These fields are written only by the Host Human/DataFlow boundary.
        # They do not shape authority and remain excluded from the effect hash.
        "_agent_libos_authority_request_origin",
        "_agent_libos_data_flow_context",
        "_agent_libos_data_release_for_request_id",
        "_agent_libos_data_release_request_id",
        "_agent_libos_data_release_request_ids",
        "_agent_libos_data_release_presentation",
        "_agent_libos_data_release_visible",
        "_agent_libos_data_release_terminal_committed_request_id",
    }
)
_REQUIRED_PAYLOAD_FIELDS = frozenset(
    {
        "type",
        "requested_once_capability",
        "context",
        "effect_binding",
    }
)
_ONCE_FIELDS = frozenset(
    {"subject", "resource", "rights", "constraints", "expires_at", "delegable"}
)
_REQUIRED_ONCE_FIELDS = frozenset(
    {"subject", "resource", "rights", "constraints"}
)
_REMOTE_RESOURCE_FIELDS = {
    "jsonrpc.call": ("jsonrpc", "endpoint_id", "method_id"),
    "mcp.call": ("mcp", "server_id", "tool_id"),
}
_FORBIDDEN_EXACT_CONSTRAINTS = frozenset(
    {
        "shell_policy_level",
        "inherited_from",
        DATA_RELEASE_BINDING_KEY,
    }
)
_GIT_STRING_CONSTRAINTS = frozenset(
    {
        "git_remote",
        "git_url_fingerprint",
        "git_expected_state_token",
        "git_old_oid",
    }
)


@dataclass(frozen=True, slots=True)
class HostHumanApprovalRequest:
    """Strict Host-decoded Human external-operation approval.

    A Human may deliberately approve a bounded capability scope (for example
    one directory subtree).  Such a request is still origin-, subject-,
    operation-, argument-, and effect-bound, but it is not eligible for the
    Phase 4 exact-resource machine path.

    This is an in-memory authority view.  It is never a persistence or API
    model, and deliberately retains no derived permit/deny instruction.
    """

    action_id: str
    resource: str
    right: str
    context: dict[str, Any]
    capability: dict[str, Any]
    constraints: dict[str, Any]
    binding: dict[str, Any]
    expires_at: str | None

    @property
    def rights(self) -> tuple[str, ...]:
        return (self.right,)


@dataclass(frozen=True, slots=True)
class ExactSemanticApprovalRequest(HostHumanApprovalRequest):
    """Host approval whose resource is exact and machine-policy eligible."""


def decode_host_human_approval_request(
    request: HumanRequest,
) -> HostHumanApprovalRequest:
    """Decode a Host-composed Human approval, including bounded scopes.

    This decoder is for canonical Human presentation and settlement fences.
    It deliberately accepts a scoped resource while retaining every other
    structural and binding check used by the exact machine decoder.  Machine
    deny/allow selection must continue to call
    :func:`decode_exact_semantic_approval_request`.
    """

    return _decode_semantic_approval_request(
        request,
        exact_resource=False,
        result_type=HostHumanApprovalRequest,
    )


def decode_exact_semantic_approval_request(
    request: HumanRequest,
) -> ExactSemanticApprovalRequest:
    """Decode one exact external-operation request or fail closed.

    Unknown fields, multiple or duplicate rights, delegable grants, wildcard
    resources, malformed expiry values, and any binding drift are rejected.
    Unsupported or high-risk *well-formed* operations remain valid here so the
    policy layer can route them to Human instead of inventing a machine deny.
    """

    return _decode_semantic_approval_request(
        request,
        exact_resource=True,
        result_type=ExactSemanticApprovalRequest,
    )


def _decode_semantic_approval_request(
    request: HumanRequest,
    *,
    exact_resource: bool,
    result_type: type[HostHumanApprovalRequest],
) -> HostHumanApprovalRequest:
    if not isinstance(request, HumanRequest):
        raise TypeError("semantic exact request decoder requires HumanRequest")
    try:
        payload = _external_payload(request.payload)
        capability = _once_capability(payload, request.pid)
        action_id, resource, right, context = _operation_identity(
            payload,
            request=request,
            capability=capability,
            exact_resource=exact_resource,
        )
        constraints = _constraints(
            capability,
            action_id=action_id,
        )
        binding = _effect_binding(
            payload,
            constraints=constraints,
            context=context,
        )
        expires_at = _expiry(capability)
    except ValidationError:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationError("semantic exact request is malformed") from exc
    return result_type(
        action_id=action_id,
        resource=resource,
        right=right,
        context=context,
        capability=capability,
        constraints=constraints,
        binding=binding,
        expires_at=expires_at,
    )


def semantic_effect_identity(request: HumanRequest) -> str:
    """Return a raw effect id only for one strictly decoded Host request.

    Malformed external input is bound by its canonical payload digest.  This
    keeps attacker-controlled effect identifiers out of append-only Human and
    semantic evidence while giving the policy broker and Human CAS boundary
    one shared identity function.
    """

    try:
        return decode_exact_semantic_approval_request(request).binding["effect_id"]
    except ValidationError:
        encoded = json.dumps(
            to_jsonable(request.payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"unbound:{hashlib.sha256(encoded).hexdigest()}"


def _external_payload(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "semantic external approval payload")
    _exact_fields(
        payload,
        allowed=_PAYLOAD_FIELDS,
        required=_REQUIRED_PAYLOAD_FIELDS,
        label="semantic external approval payload",
    )
    if payload.get("type") != "external_operation_approval":
        raise ValidationError(
            "semantic exact request must be an external operation approval"
        )
    if payload.get("_agent_libos_authority_request_origin") != "external_operation":
        raise ValidationError(
            "semantic exact request is missing Host external-operation origin"
        )
    if "question" in payload and (
        type(payload["question"]) is not str or not payload["question"].strip()
    ):
        raise ValidationError("semantic external approval question is malformed")
    raw_flow = payload.get("_agent_libos_data_flow_context")
    if raw_flow is not None and not isinstance(raw_flow, Mapping):
        raise ValidationError("semantic external approval DataFlow context is malformed")
    return payload


def _once_capability(payload: Mapping[str, Any], pid: str) -> dict[str, Any]:
    capability = _mapping(
        payload.get("requested_once_capability"),
        "semantic requested one-shot capability",
    )
    _exact_fields(
        capability,
        allowed=_ONCE_FIELDS,
        required=_REQUIRED_ONCE_FIELDS,
        label="semantic requested one-shot capability",
    )
    if type(capability.get("subject")) is not str or capability["subject"] != pid:
        raise ValidationError("semantic one-shot subject must match request process")
    if "delegable" in capability and capability["delegable"] is not False:
        raise ValidationError("semantic one-shot capability must be nondelegable")
    return capability


def _operation_identity(
    payload: Mapping[str, Any],
    *,
    request: HumanRequest,
    capability: Mapping[str, Any],
    exact_resource: bool,
) -> tuple[str, str, str, dict[str, Any]]:
    context = _mapping(payload.get("context"), "semantic operation context")
    action_id = context.get("authority_operation")
    if (
        type(action_id) is not str
        or len(action_id) > 512
        or _ACTION_RE.fullmatch(action_id) is None
    ):
        raise ValidationError("semantic authority operation is malformed")
    resource = _approval_resource(
        capability.get("resource"),
        exact=exact_resource,
    )
    right = _single_right(capability.get("rights"))
    if context.get("pid") != request.pid:
        raise ValidationError("semantic operation subject does not match request process")
    if context.get("right") != right:
        raise ValidationError("semantic operation right does not match capability")
    if _expected_resource(action_id, context) != resource:
        raise ValidationError("semantic operation resource does not match capability")
    return action_id, resource, right, context


def _approval_resource(value: Any, *, exact: bool) -> str:
    maximum = 2_048 if exact else 65_536
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > maximum
        or "\x00" in value
        or (exact and "*" in value)
    ):
        requirement = "exact" if exact else "a bounded non-empty string"
        raise ValidationError(
            f"semantic one-shot resource must be {requirement}"
        )
    return value


def _single_right(value: Any) -> str:
    if type(value) is not list or len(value) != 1 or type(value[0]) is not str:
        raise ValidationError(
            "semantic one-shot capability requires exactly one unique right"
        )
    try:
        return CapabilityRight(value[0]).value
    except ValueError as exc:
        raise ValidationError("semantic one-shot capability right is unknown") from exc


def _expected_resource(action_id: str, context: Mapping[str, Any]) -> str | None:
    remote = _REMOTE_RESOURCE_FIELDS.get(action_id)
    if remote is None:
        value = context.get("resource")
        return value if type(value) is str else None
    prefix, owner_field, member_field = remote
    owner = context.get(owner_field)
    member = context.get(member_field)
    if (
        type(owner) is not str
        or not owner
        or type(member) is not str
        or not member
    ):
        return None
    return f"{prefix}:{owner}:{member}"


def _constraints(
    capability: Mapping[str, Any],
    *,
    action_id: str,
) -> dict[str, Any]:
    constraints = _mapping(
        capability.get("constraints"),
        "semantic one-shot constraints",
    )
    unknown = set(constraints) - KNOWN_CONSTRAINT_KEYS
    if unknown:
        raise ValidationError(
            "semantic one-shot constraints contain unknown fields: "
            + ", ".join(sorted(unknown))
        )
    if APPROVAL_BINDING_KEY not in constraints:
        raise ValidationError("semantic one-shot approval binding is missing")
    forbidden = _FORBIDDEN_EXACT_CONSTRAINTS.intersection(constraints)
    if forbidden:
        raise ValidationError(
            "semantic exact request contains authority-shaping constraints: "
            + ", ".join(sorted(forbidden))
        )
    null_fields = [key for key, item in constraints.items() if item is None]
    if null_fields:
        raise ValidationError(
            "semantic one-shot constraints contain null fields: "
            + ", ".join(sorted(null_fields))
        )
    _authority_rules(constraints, action_id=action_id)
    _git_constraints(constraints, action_id=action_id)
    return constraints


def _authority_rules(
    constraints: Mapping[str, Any],
    *,
    action_id: str,
) -> None:
    if AUTHORITY_RULES_KEY not in constraints:
        return
    codec = AuthorityRuleCodec()
    rules = codec.coerce_many(constraints[AUTHORITY_RULES_KEY])
    if not rules:
        raise ValidationError("semantic exact authority rules must not be empty")
    if len({rule.rule_id for rule in rules}) != len(rules):
        raise ValidationError("semantic exact authority rule ids must be unique")
    evaluator = CapabilityEvaluator(codec)
    for rule in rules:
        if rule.operation != action_id or rule.effect is not CapabilityEffect.ALLOW:
            raise ValidationError(
                "semantic exact authority rule does not match the requested operation"
            )
        invalid = (
            evaluator.unknown_authority_rule_conditions(rule)
            or evaluator.malformed_authority_rule_conditions(rule)
        )
        if invalid:
            raise ValidationError(
                "semantic exact authority rule has invalid conditions: "
                + ", ".join(invalid)
            )


def _git_constraints(
    constraints: Mapping[str, Any],
    *,
    action_id: str,
) -> None:
    present = _GIT_STRING_CONSTRAINTS.intersection(constraints)
    has_refs = "git_allowed_refs" in constraints
    if (present or has_refs) and not action_id.startswith("git."):
        raise ValidationError("semantic Git constraints require a Git operation")
    for key in present:
        value = constraints[key]
        if type(value) is not str or not value:
            raise ValidationError(f"semantic constraint {key} must be a non-empty string")
    if has_refs:
        refs = constraints["git_allowed_refs"]
        if (
            type(refs) is not list
            or not refs
            or any(type(ref) is not str or not ref for ref in refs)
            or len(refs) != len(set(refs))
        ):
            raise ValidationError(
                "semantic constraint git_allowed_refs must contain unique refs"
            )


def _effect_binding(
    payload: Mapping[str, Any],
    *,
    constraints: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        top = normalize_approval_binding(dict(payload.get("effect_binding")))
        nested = normalize_approval_binding(
            dict(constraints.get(APPROVAL_BINDING_KEY))
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValidationError("semantic approval binding is malformed") from exc
    if top != nested:
        raise ValidationError("semantic approval bindings do not match")
    if top["canonical_args_hash"] != canonical_effect_hash(dict(context)):
        raise ValidationError("semantic approval binding arguments changed")
    if top["target_state_version"] != context.get("target_state_version"):
        raise ValidationError("semantic approval target state changed")
    return top


def _expiry(capability: Mapping[str, Any]) -> str | None:
    if "expires_at" not in capability or capability["expires_at"] is None:
        return None
    value = capability["expires_at"]
    if type(value) is not str or not value or value != value.strip():
        raise ValidationError(
            "semantic one-shot expires_at must be an ISO-8601 datetime"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(
            "semantic one-shot expires_at must be an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError("semantic one-shot expires_at must include a timezone")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    if any(type(key) is not str for key in value):
        raise ValidationError(f"{label} must use string fields")
    return dict(value)


def _exact_fields(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValidationError(
            f"{label} contains unknown fields: " + ", ".join(sorted(unknown))
        )
    if missing:
        raise ValidationError(
            f"{label} is missing fields: " + ", ".join(sorted(missing))
        )


__all__ = [
    "ExactSemanticApprovalRequest",
    "HostHumanApprovalRequest",
    "decode_host_human_approval_request",
    "decode_exact_semantic_approval_request",
    "semantic_effect_identity",
]
