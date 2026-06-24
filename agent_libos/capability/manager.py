from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable

from agent_libos.capability.profiles import SandboxProfileBuilder
from agent_libos.capability.resources import ResourceAuthority
from agent_libos.capability.rules import AUTHORITY_RULES_KEY, AuthorityRuleCodec
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    AuthorityRisk,
    CapabilityLease,
    Capability,
    CapabilityDecision,
    CapabilityEffect,
    CapabilityRight,
    CapabilitySpec,
    CapabilityStatus,
    EventType,
    ObjectHandle,
    OperationContext,
    DelegationPolicy,
    ResourcePattern,
    SandboxProfile,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.storage import SQLiteStore
from agent_libos.utils.ids import new_id, utc_now


@dataclass(frozen=True)
class _IssueAuthority:
    mutation_decision: CapabilityDecision | None = None
    transfer_parent: Capability | None = None


class CapabilityManager:
    """Capability directory, authorization engine, and delegation helper."""

    POLICY_KEY = "permission_policy"
    ALWAYS_ALLOW = "always_allow"
    ALWAYS_DENY = "always_deny"
    ASK_EACH_TIME = "ask_each_time"
    ALLOW_ONCE = "allow_once"
    MISSING = "missing"
    POLICY_VALUES = {ALWAYS_ALLOW, ALWAYS_DENY, ASK_EACH_TIME, ALLOW_ONCE}

    _KNOWN_CONSTRAINT_KEYS = {
        "shell_policy_level",
        "inherited_from",
        AUTHORITY_RULES_KEY,
    }

    def __init__(self, store: SQLiteStore, audit: AuditManager, events: EventBus, config: AgentLibOSConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.store = store
        self.audit = audit
        self.events = events
        self.resources = ResourceAuthority()
        self.rule_codec = AuthorityRuleCodec()
        self.profiles = SandboxProfileBuilder()

    def issue(
        self,
        actor: str,
        subject: str,
        spec: CapabilitySpec | dict[str, Any],
        *,
        issuer_cap_id: str | None = None,
        require_authority: bool = True,
    ) -> Capability:
        selected = self._coerce_spec(spec)
        if require_authority:
            issue_authority = self._require_issue_authority(actor, selected)
            issuer_cap_id = (
                issue_authority.mutation_decision.selected_capability_id
                if issue_authority.mutation_decision is not None
                else None
            )
        else:
            issue_authority = _IssueAuthority()
        transfer_parent = issue_authority.transfer_parent
        delegation_depth = transfer_parent.delegation_depth + 1 if transfer_parent is not None else 0
        max_delegation_depth = (
            self._delegated_max_delegation_depth(transfer_parent, selected)
            if transfer_parent is not None and selected.delegable
            else self._initial_max_delegation_depth(selected)
        )
        expires_at = selected.expires_at
        if transfer_parent is not None and expires_at is None:
            expires_at = transfer_parent.expires_at
        self._consume_mutation_authority(
            issue_authority.mutation_decision,
            used_by=actor,
            reason="one-time issue authority consumed",
        )
        cap = self._insert_capability(
            subject=subject,
            resource=selected.resource,
            rights=selected.rights,
            effect=selected.effect,
            constraints=selected.constraints,
            metadata=selected.metadata,
            issued_by=actor,
            issuer_cap_id=issuer_cap_id,
            parent_cap_id=transfer_parent.cap_id if transfer_parent is not None else None,
            delegation_depth=delegation_depth,
            max_delegation_depth=max_delegation_depth,
            expires_at=expires_at,
            uses_remaining=selected.uses_remaining,
            delegable=selected.delegable,
            revocable=selected.revocable,
        )
        self.audit.record(
            actor=actor,
            action="capability.issue",
            target=f"{subject}:{cap.resource}",
            capability_refs=[cap.cap_id] + ([issuer_cap_id] if issuer_cap_id else []),
            decision={
                "effect": cap.effect.value,
                "rights": sorted(cap.rights),
                "uses_remaining": cap.uses_remaining,
                "delegable": cap.delegable,
            },
        )
        return cap

    def issue_trusted(
        self,
        subject: str,
        resource: str,
        rights: Iterable[str | CapabilityRight],
        *,
        issued_by: str,
        effect: str | CapabilityEffect = CapabilityEffect.ALLOW,
        constraints: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None,
        uses_remaining: int | None = None,
        delegable: bool = False,
        revocable: bool = True,
    ) -> Capability:
        return self.issue(
            actor=issued_by,
            subject=subject,
            spec=CapabilitySpec(
                resource=resource,
                rights=self._normalize_rights(rights),
                effect=CapabilityEffect(effect),
                constraints=dict(constraints or {}),
                metadata=dict(metadata or {}),
                expires_at=expires_at,
                uses_remaining=uses_remaining,
                delegable=delegable,
                revocable=revocable,
            ),
            require_authority=False,
        )

    def grant(
        self,
        subject: str,
        resource: str,
        rights: Iterable[str | CapabilityRight],
        issued_by: str = "system",
        constraints: dict | None = None,
        expires_at: str | None = None,
        delegable: bool = False,
        revocable: bool = True,
    ) -> Capability:
        # Trusted runtime paths keep this compact bootstrap helper, while all
        # records still flow through issue() for canonical spec conversion.
        effect, uses_remaining = self._effect_from_policy_constraint(constraints or {})
        clean_constraints = {
            key: value
            for key, value in dict(constraints or {}).items()
            if key != self.POLICY_KEY
        }
        return self.issue_trusted(
            subject=subject,
            resource=resource,
            rights=rights,
            issued_by=issued_by,
            effect=effect,
            constraints=clean_constraints,
            expires_at=expires_at,
            uses_remaining=uses_remaining,
            delegable=delegable,
            revocable=revocable,
        )

    def delegate(
        self,
        parent: str,
        child: str,
        spec: CapabilitySpec | dict[str, Any],
        *,
        actor: str | None = None,
    ) -> Capability:
        selected = self._coerce_spec(spec)
        parent_cap = self._find_delegation_parent(parent, selected)
        self._validate_delegation_parent(parent_cap, selected)
        child_max_depth = self._delegated_max_delegation_depth(parent_cap, selected)
        cap = self._insert_capability(
            subject=child,
            resource=selected.resource,
            rights=selected.rights,
            effect=selected.effect,
            constraints=selected.constraints,
            metadata={**selected.metadata, "delegated_from": parent},
            issued_by=actor or parent,
            issuer_cap_id=parent_cap.cap_id,
            parent_cap_id=parent_cap.cap_id,
            delegation_depth=parent_cap.delegation_depth + 1,
            max_delegation_depth=child_max_depth,
            expires_at=selected.expires_at or parent_cap.expires_at,
            uses_remaining=selected.uses_remaining,
            delegable=selected.delegable,
            revocable=selected.revocable,
        )
        self.audit.record(
            actor=actor or parent,
            action="capability.delegate",
            target=f"{parent}->{child}:{cap.resource}",
            capability_refs=[parent_cap.cap_id, cap.cap_id],
            decision={"rights": sorted(cap.rights), "effect": cap.effect.value},
        )
        return cap

    def validate_delegation(self, parent: str, spec: CapabilitySpec | dict[str, Any]) -> Capability:
        selected = self._coerce_spec(spec)
        parent_cap = self._find_delegation_parent(parent, selected)
        self._validate_delegation_parent(parent_cap, selected)
        return parent_cap

    def inherit(
        self,
        parent: str,
        child: str,
        resource: str,
        rights: Iterable[str | CapabilityRight],
        issued_by: str,
        constraints: dict | None = None,
    ) -> Capability:
        return self.delegate(
            parent,
            child,
            CapabilitySpec(
                resource=resource,
                rights=self._normalize_rights(rights),
                effect=CapabilityEffect.ALLOW,
                constraints=dict(constraints or {}),
                metadata={"issued_by": issued_by},
                delegable=False,
            ),
            actor=issued_by,
        )

    def authorize(
        self,
        subject: str,
        resource: str,
        right: str | CapabilityRight,
        context: OperationContext | dict[str, Any] | None = None,
        *,
        audit: bool = False,
    ) -> CapabilityDecision:
        requested_right = str(right)
        selected_context = self._context_dict(context)
        matches = self._matching_capabilities(subject, resource, requested_right, include_ask=True)
        return self._decision_from_matches(
            subject=subject,
            resource=resource,
            requested_right=requested_right,
            matches=matches,
            selected_context=selected_context,
            audit=audit,
        )

    def authorize_matching_capabilities(
        self,
        subject: str,
        resource: str,
        right: str | CapabilityRight,
        capabilities: Iterable[Capability],
        context: OperationContext | dict[str, Any] | None = None,
        *,
        audit: bool = False,
    ) -> CapabilityDecision:
        requested_right = str(right)
        selected_context = self._context_dict(context)
        matches = [
            cap
            for cap in capabilities
            if cap.active
            and not self._is_expired(cap)
            and self._parent_chain_active(cap)
            and self._resource_matches(cap.resource, resource)
            and requested_right in cap.rights
        ]
        return self._decision_from_matches(
            subject=subject,
            resource=resource,
            requested_right=requested_right,
            matches=matches,
            selected_context=selected_context,
            audit=audit,
        )

    def _decision_from_matches(
        self,
        *,
        subject: str,
        resource: str,
        requested_right: str,
        matches: list[Capability],
        selected_context: dict[str, Any],
        audit: bool,
    ) -> CapabilityDecision:
        matched_ids = [cap.cap_id for cap in matches]
        failed_constraints: list[tuple[Capability, dict[str, Any]]] = []
        for cap in matches:
            constraint_results = self._evaluate_constraints(cap, selected_context)
            constraint_effect = self._constraint_effect(constraint_results)
            if constraint_effect == CapabilityEffect.DENY:
                decision = CapabilityDecision(
                    subject=subject,
                    resource=resource,
                    right=requested_right,
                    allowed=False,
                    effect=CapabilityEffect.DENY,
                    reason=f"capability constraints denied {requested_right} on {resource}",
                    matched_capability_ids=matched_ids,
                    selected_capability_id=cap.cap_id,
                    issuer_chain=self._issuer_chain(cap),
                    constraint_results=constraint_results,
                    context=selected_context,
                )
                return self._record_decision(decision, audit=audit)
            if not all(bool(item.get("ok")) for item in constraint_results.values()):
                restrictive_constraint_failed = (
                    cap.effect in {CapabilityEffect.DENY, CapabilityEffect.ASK}
                    and not self._constraint_failure_is_scoped_miss(constraint_results)
                )
                if restrictive_constraint_failed:
                    # Restrictive policy records must fail closed. The one
                    # exception is an AuthorityRule deny/ask whose rule simply
                    # does not match this operation; that is a scoped miss, not
                    # a malformed restriction.
                    decision = CapabilityDecision(
                        subject=subject,
                        resource=resource,
                        right=requested_right,
                        allowed=False,
                        effect=CapabilityEffect.DENY,
                        reason=f"capability constraints denied {requested_right} on {resource}",
                        matched_capability_ids=matched_ids,
                        selected_capability_id=cap.cap_id,
                        issuer_chain=self._issuer_chain(cap),
                        constraint_results=constraint_results,
                        context=selected_context,
                    )
                    return self._record_decision(decision, audit=audit)
                failed_constraints.append((cap, constraint_results))
                continue
            if cap.effect == CapabilityEffect.DENY:
                # Unconstrained deny still dominates all matching grants. A
                # deny carrying AuthorityRule constraints is scoped: it only
                # dominates when those rules match the current operation
                # context, which lets policy express "deny git push" without
                # denying every `shell:git` operation.
                decision = CapabilityDecision(
                    subject=subject,
                    resource=resource,
                    right=requested_right,
                    allowed=False,
                    effect=CapabilityEffect.DENY,
                    reason=f"{subject} denied {requested_right} on {resource}",
                    matched_capability_ids=matched_ids,
                    selected_capability_id=cap.cap_id,
                    issuer_chain=self._issuer_chain(cap),
                    constraint_results=constraint_results,
                    context=selected_context,
                )
                return self._record_decision(decision, audit=audit)
            if cap.effect == CapabilityEffect.ASK or constraint_effect == CapabilityEffect.ASK:
                decision = CapabilityDecision(
                    subject=subject,
                    resource=resource,
                    right=requested_right,
                    allowed=False,
                    effect=CapabilityEffect.ASK,
                    reason=f"{subject} requires human approval for {requested_right} on {resource}",
                    matched_capability_ids=matched_ids,
                    selected_capability_id=cap.cap_id,
                    issuer_chain=self._issuer_chain(cap),
                    constraint_results=constraint_results,
                    context=selected_context,
                )
                return self._record_decision(decision, audit=audit)
            if cap.effect == CapabilityEffect.ALLOW:
                decision = CapabilityDecision(
                    subject=subject,
                    resource=resource,
                    right=requested_right,
                    allowed=True,
                    effect=CapabilityEffect.ALLOW,
                    reason="capability allowed operation",
                    matched_capability_ids=matched_ids,
                    selected_capability_id=cap.cap_id,
                    consume_capability_id=cap.cap_id if cap.uses_remaining is not None else None,
                    issuer_chain=self._issuer_chain(cap),
                    constraint_results=constraint_results,
                    context=selected_context,
                )
                return self._record_decision(decision, audit=audit)
        if failed_constraints:
            cap, constraint_results = failed_constraints[0]
            decision = CapabilityDecision(
                subject=subject,
                resource=resource,
                right=requested_right,
                allowed=False,
                effect=None,
                reason=f"capability constraints rejected {requested_right} on {resource}",
                matched_capability_ids=matched_ids,
                selected_capability_id=cap.cap_id,
                issuer_chain=self._issuer_chain(cap),
                constraint_results=constraint_results,
                context=selected_context,
            )
            return self._record_decision(decision, audit=audit)
        decision = CapabilityDecision(
            subject=subject,
            resource=resource,
            right=requested_right,
            allowed=False,
            effect=None,
            reason=f"{subject} lacks {requested_right} on {resource}",
            matched_capability_ids=matched_ids,
            context=selected_context,
        )
        return self._record_decision(decision, audit=audit)

    def require(
        self,
        subject: str,
        resource: str,
        right: str | CapabilityRight,
        context: OperationContext | dict[str, Any] | None = None,
    ) -> CapabilityDecision:
        decision = self.authorize(subject, resource, right, context, audit=True)
        if decision.allowed:
            return decision
        raise CapabilityDenied(decision.reason)

    def check(
        self,
        subject: str,
        resource: str,
        right: str | CapabilityRight,
        context: OperationContext | dict[str, Any] | None = None,
    ) -> bool:
        return self.authorize(subject, resource, right, context).allowed

    def permission_policy(
        self,
        subject: str,
        resource: str,
        right: str | CapabilityRight,
        context: OperationContext | dict[str, Any] | None = None,
    ) -> str:
        return self.authorize(subject, resource, right, context).policy

    def set_permission_policy(
        self,
        subject: str,
        resource: str,
        rights: Iterable[str | CapabilityRight],
        policy: str,
        issued_by: str | None = None,
        constraints: dict | None = None,
    ) -> Capability:
        if policy not in self.POLICY_VALUES:
            raise ValueError(f"unknown permission policy: {policy}")
        effect, uses_remaining = self._effect_from_policy(policy)
        cap = self.issue_trusted(
            subject=subject,
            resource=resource,
            rights=rights,
            issued_by=issued_by or self.config.runtime.default_human_actor,
            effect=effect,
            constraints=dict(constraints or {}),
            uses_remaining=uses_remaining,
        )
        self.audit.record(
            actor=issued_by or self.config.runtime.default_human_actor,
            action="capability.permission_policy",
            target=f"{subject}:{resource}",
            capability_refs=[cap.cap_id],
            decision={"policy": policy, "effect": effect.value, "rights": sorted(cap.rights)},
        )
        return cap

    def grant_once(
        self,
        subject: str,
        resource: str,
        rights: Iterable[str | CapabilityRight],
        issued_by: str | None = None,
        constraints: dict | None = None,
    ) -> Capability:
        return self.issue_trusted(
            subject=subject,
            resource=resource,
            rights=rights,
            issued_by=issued_by or self.config.runtime.default_human_actor,
            effect=CapabilityEffect.ALLOW,
            constraints=dict(constraints or {}),
            uses_remaining=1,
        )

    def consume_use(self, cap_id: str, *, used_by: str, reason: str = "capability use consumed", count: int = 1) -> Capability:
        if count < 1:
            raise ValidationError("capability consume count must be >= 1")
        cap = self.store.get_capability(cap_id)
        if cap is None:
            raise NotFound(f"capability not found: {cap_id}")
        if cap.uses_remaining is None:
            return cap
        updated = self.store.consume_capability_uses(cap_id, count)
        if updated is None:
            raise CapabilityDenied(f"capability use exhausted: {cap_id}")
        self.audit.record(
            actor=used_by,
            action="capability.consume",
            target=cap.resource,
            capability_refs=[cap_id],
            decision={"uses_remaining": updated.uses_remaining, "count": count, "reason": reason},
        )
        if updated.revoked:
            self.events.emit(
                EventType.CAPABILITY_REVOKED,
                source=used_by,
                target=cap.subject,
                payload={"capability_id": cap_id, "reason": reason},
            )
        return updated

    def _restore_reserved_use(
        self,
        cap_id: str,
        *,
        restored_by: str,
        reason: str = "reserved capability use restored",
        count: int = 1,
    ) -> Capability:
        """Restore a use consumed as an effect reservation before provider I/O.

        This is intentionally not part of the public capability workflow:
        regular callers should grant or consume capabilities, while primitives
        use this only when a provider raises before reporting a committed effect.
        """
        if count < 1:
            raise ValidationError("capability restore count must be >= 1")
        cap = self.store.get_capability(cap_id)
        if cap is None:
            raise NotFound(f"capability not found: {cap_id}")
        if cap.uses_remaining is None:
            return cap
        updated = self.store.restore_reserved_capability_uses(cap_id, count)
        if updated is None:
            raise CapabilityDenied(f"reserved capability use cannot be restored: {cap_id}")
        self.audit.record(
            actor=restored_by,
            action="capability.restore_reserved_use",
            target=cap.resource,
            capability_refs=[cap_id],
            decision={"uses_remaining": updated.uses_remaining, "count": count, "reason": reason},
        )
        return updated

    def consume_allow_once(self, subject: str, resource: str, right: str | CapabilityRight, used_by: str) -> None:
        decision = self.authorize(subject, resource, right)
        if decision.consume_capability_id is not None:
            self.consume_use(decision.consume_capability_id, used_by=used_by, reason="one-time permission consumed")

    def revoke(
        self,
        cap_id: str,
        revoked_by: str = "system",
        reason: str | None = None,
        *,
        require_authority: bool = True,
    ) -> Capability:
        cap = self.store.get_capability(cap_id)
        if cap is None:
            raise NotFound(f"capability not found: {cap_id}")
        if not cap.revocable:
            raise CapabilityDenied(f"capability is not revocable: {cap_id}")
        if require_authority:
            authority_decision = self._require_revoke_authority(revoked_by, cap)
        else:
            authority_decision = None
        revoked = replace(cap, status=CapabilityStatus.REVOKED)
        self.store.update_capability(revoked)
        self._consume_mutation_authority(authority_decision, used_by=revoked_by, reason="one-time revoke authority consumed")
        self.events.emit(
            EventType.CAPABILITY_REVOKED,
            source=revoked_by,
            target=cap.subject,
            payload={"capability_id": cap_id, "reason": reason},
        )
        self.audit.record(
            actor=revoked_by,
            action="capability.revoke",
            target=cap.resource,
            capability_refs=[cap_id],
            decision={"revoked": True, "reason": reason, "subject": cap.subject},
        )
        return revoked

    def disable_subject_capability(
        self,
        cap_id: str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> Capability:
        cap = self.store.get_capability(cap_id)
        if cap is None:
            raise NotFound(f"capability not found: {cap_id}")
        updated = replace(cap, status=CapabilityStatus.DISABLED)
        self.store.update_capability(updated)
        self.audit.record(
            actor=actor,
            action="capability.disable",
            target=cap.resource,
            capability_refs=[cap_id],
            decision={"reason": reason, "subject": cap.subject},
        )
        return updated

    def revoke_resource_trusted(
        self,
        resource: str,
        *,
        revoked_by: str,
        reason: str | None = None,
    ) -> list[Capability]:
        revoked: list[Capability] = []
        for cap in self.store.list_capabilities():
            if cap.resource != resource or not cap.active:
                continue
            updated = replace(cap, status=CapabilityStatus.REVOKED)
            self.store.update_capability(updated)
            revoked.append(updated)
            self.events.emit(
                EventType.CAPABILITY_REVOKED,
                source=revoked_by,
                target=cap.subject,
                payload={"capability_id": cap.cap_id, "reason": reason},
            )
        if revoked:
            self.audit.record(
                actor=revoked_by,
                action="capability.revoke_resource",
                target=resource,
                capability_refs=[cap.cap_id for cap in revoked],
                decision={"revoked": len(revoked), "reason": reason},
            )
        return revoked

    def authorize_handle(self, subject: str, handle: ObjectHandle, right: str | CapabilityRight) -> CapabilityDecision:
        requested = str(right)
        resource = f"object:{handle.oid}"
        if requested not in handle.rights:
            raise CapabilityDenied(f"object handle lacks {requested}: {handle.oid}")
        cap = self.store.get_capability(handle.capability_id)
        if cap is None or cap.revoked or not cap.active or self._is_expired(cap) or not self._parent_chain_active(cap):
            raise CapabilityDenied(f"invalid object capability: {handle.capability_id}")
        if cap.subject != subject:
            raise CapabilityDenied(f"capability subject mismatch: {cap.subject} != {subject}")
        if cap.resource != resource:
            raise CapabilityDenied(f"object handle resource mismatch: {cap.resource} != {resource}")
        if requested not in cap.rights:
            raise CapabilityDenied(f"object capability lacks {requested}: {handle.oid}")
        global_decision = self.authorize(subject, resource, requested)
        if not global_decision.allowed:
            return global_decision
        # A handle is authority only through the capability it names. A separate
        # broad grant may make the same operation legal, but it must not make a
        # forged or stale handle valid for Object Memory APIs.
        return self.authorize_matching_capabilities(subject, resource, requested, [cap])

    def assert_handle(self, subject: str, handle: ObjectHandle, right: str | CapabilityRight) -> None:
        requested = str(right)
        decision = self.authorize_handle(subject, handle, requested)
        if not decision.allowed:
            raise CapabilityDenied(f"capability lacks {requested}: {handle.oid}")
        if decision.consume_capability_id is not None:
            self.consume_use(
                decision.consume_capability_id,
                used_by="object_memory",
                reason="one-time object handle permission consumed",
            )

    def handle_for_object(
        self,
        subject: str,
        oid: str,
        rights: Iterable[str | CapabilityRight],
        issued_by: str = "system",
        expires_at: str | None = None,
        uses_remaining: int | None = None,
    ) -> ObjectHandle:
        normalized = self._normalize_rights(rights)
        cap = self.issue_trusted(
            subject=subject,
            resource=f"object:{oid}",
            rights=normalized,
            issued_by=issued_by,
            expires_at=expires_at,
            uses_remaining=uses_remaining,
            delegable=False,
        )
        return ObjectHandle(oid=oid, rights=normalized, capability_id=cap.cap_id, expires_at=expires_at)

    def capabilities_for(self, subject: str) -> list[Capability]:
        return self.store.list_capabilities(subject=subject)

    def matching_capabilities(
        self,
        subject: str,
        resource: str,
        right: str | CapabilityRight,
        *,
        include_ask: bool = False,
    ) -> list[Capability]:
        """Return active Capability records that match a canonical request.

        Primitive-specific policy layers sometimes need to inspect matched
        records without treating a broad policy capability as final authority.
        They should still reuse this matcher so typed resource semantics,
        expiry handling, wildcard rules, and deny precedence stay centralized.
        """

        return self._matching_capabilities(subject, resource, right, include_ask=include_ask)

    def list_subject(self, subject: str, *, include_inactive: bool = False, limit: int | None = None) -> list[Capability]:
        caps = self.capabilities_for(subject)
        if not include_inactive:
            caps = [cap for cap in caps if cap.active and not self._is_expired(cap) and self._parent_chain_active(cap)]
        return caps[: (limit or self.config.capability.list_limit)]

    def inspect(self, cap_id: str) -> dict[str, Any]:
        cap = self.store.get_capability(cap_id)
        if cap is None:
            raise NotFound(f"capability not found: {cap_id}")
        return self._capability_json(cap)

    def explain_decision(
        self,
        subject: str,
        resource: str,
        right: str | CapabilityRight,
        context: OperationContext | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        decision = self.authorize(subject, resource, right, context)
        return self._decision_json(decision)

    def sandbox_profile_for_decision(
        self,
        decision: CapabilityDecision,
        *,
        operation: str,
        risk: str = "medium",
        rule_id: str | None = None,
        restrictions: dict[str, Any] | None = None,
    ) -> SandboxProfile:
        """Create a primitive-facing profile from a finalized decision.

        The profile is derived from the same decision that authorized the
        operation. Primitive code can therefore audit and enforce sandbox
        restrictions without inventing a second, weaker authority model.
        """

        return SandboxProfile(
            operation=operation,
            resource=decision.resource,
            effect=decision.effect or CapabilityEffect.DENY,
            risk=self._coerce_risk(risk),
            rule_id=rule_id,
            restrictions=dict(restrictions or {}),
        )

    def constraints_satisfied(self, cap: Capability, context: OperationContext | dict[str, Any] | None = None) -> bool:
        selected_context = self._context_dict(context)
        results = self._evaluate_constraints(cap, selected_context)
        return all(bool(item.get("ok")) for item in results.values()) and self._constraint_effect(results) != CapabilityEffect.DENY

    def spec(
        self,
        resource: str,
        rights: Iterable[str | CapabilityRight],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"resource": resource, "rights": [str(right) for right in rights], **kwargs}

    def tool_execute(self, tool: str, rights: Iterable[str | CapabilityRight] | None = None, **kwargs: Any) -> dict[str, Any]:
        resource = tool if tool.startswith("tool:") else f"tool:{tool}"
        return self.spec(resource, rights or [CapabilityRight.EXECUTE], **kwargs)

    def project_read(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return self.spec(f"project:{name}", [CapabilityRight.READ], **kwargs)

    def object_access(self, oid: str, rights: Iterable[str | CapabilityRight], **kwargs: Any) -> dict[str, Any]:
        return self.spec(f"object:{oid}", rights, **kwargs)

    def parse_resource_pattern(self, resource: str, *, requested: bool = False) -> ResourcePattern:
        return self.resources.parse(resource, requested=requested)

    def _insert_capability(
        self,
        *,
        subject: str,
        resource: str,
        rights: set[str],
        effect: CapabilityEffect,
        constraints: dict[str, Any],
        metadata: dict[str, Any],
        issued_by: str,
        issuer_cap_id: str | None,
        parent_cap_id: str | None,
        delegation_depth: int,
        max_delegation_depth: int | None,
        expires_at: str | None,
        uses_remaining: int | None,
        delegable: bool,
        revocable: bool,
    ) -> Capability:
        if not subject:
            raise ValidationError("capability subject must be non-empty")
        self.parse_resource_pattern(resource)
        normalized_rights = self._normalize_rights(rights)
        self._validate_constraints(constraints)
        if uses_remaining is not None and uses_remaining < 1:
            raise ValidationError("uses_remaining must be >= 1 when set")
        if max_delegation_depth is not None and max_delegation_depth < delegation_depth:
            raise ValidationError("max_delegation_depth cannot be less than delegation_depth")
        normalized_expires_at = self._normalize_expires_at(expires_at)
        cap = Capability(
            cap_id=new_id("cap"),
            subject=subject,
            resource=self._canonical_resource(resource),
            rights=normalized_rights,
            constraints=dict(constraints),
            issued_by=issued_by,
            issued_at=utc_now(),
            expires_at=normalized_expires_at,
            delegable=delegable,
            revocable=revocable,
            effect=effect,
            issuer_cap_id=issuer_cap_id,
            parent_cap_id=parent_cap_id,
            delegation_depth=delegation_depth,
            max_delegation_depth=max_delegation_depth,
            uses_remaining=uses_remaining,
            status=CapabilityStatus.ACTIVE,
            metadata=dict(metadata),
        )
        self.store.insert_capability(cap)
        self._attach_to_process(subject, cap.cap_id)
        self.events.emit(
            EventType.CAPABILITY_GRANTED,
            source=issued_by,
            target=subject,
            payload={
                "capability_id": cap.cap_id,
                "resource": cap.resource,
                "rights": sorted(cap.rights),
                "effect": cap.effect.value,
                "uses_remaining": cap.uses_remaining,
            },
        )
        return cap

    def _consume_mutation_authority(self, decision: CapabilityDecision | None, *, used_by: str, reason: str) -> None:
        if decision is None or decision.consume_capability_id is None:
            return
        self.consume_use(decision.consume_capability_id, used_by=used_by, reason=reason)

    def claim_decision_use(self, decision: CapabilityDecision, *, used_by: str, reason: str) -> None:
        if decision.consume_capability_id is None:
            return
        self.consume_use(decision.consume_capability_id, used_by=used_by, reason=reason)

    def _require_issue_authority(self, actor: str, spec: CapabilitySpec) -> _IssueAuthority:
        if self._is_trusted_issuer(actor):
            return _IssueAuthority()
        admin = self.authorize(actor, spec.resource, CapabilityRight.ADMIN)
        if admin.allowed:
            return _IssueAuthority(mutation_decision=admin)
        grant = self.authorize(actor, spec.resource, CapabilityRight.GRANT)
        if grant.allowed:
            transfer_parent = self._find_transfer_parent(actor, spec)
            self._validate_transfer_parent(transfer_parent, spec)
            return _IssueAuthority(mutation_decision=grant, transfer_parent=transfer_parent)
        raise CapabilityDenied(f"{actor} lacks grant/admin authority to issue {sorted(spec.rights)} on {spec.resource}")

    def _require_revoke_authority(self, actor: str, cap: Capability) -> CapabilityDecision | None:
        if self._is_trusted_issuer(actor) or actor == cap.issued_by:
            return None
        if actor == cap.subject:
            if cap.effect == CapabilityEffect.ALLOW:
                return None
            raise CapabilityDenied(f"{actor} cannot self-revoke restrictive capability {cap.cap_id}")
        revoke = self.authorize(actor, cap.resource, CapabilityRight.REVOKE)
        if revoke.allowed:
            return revoke
        admin = self.authorize(actor, cap.resource, CapabilityRight.ADMIN)
        if admin.allowed:
            return admin
        raise CapabilityDenied(f"{actor} lacks revoke/admin authority for capability {cap.cap_id}")

    def _find_delegation_parent(self, parent: str, spec: CapabilitySpec) -> Capability:
        candidates = [
            cap
            for cap in self.capabilities_for(parent)
            if cap.active
            and not self._is_expired(cap)
            and self._parent_chain_active(cap)
            and cap.effect == CapabilityEffect.ALLOW
            and cap.delegable
            and self._resource_covers(cap.resource, spec.resource)
            and spec.rights.issubset(cap.rights)
        ]
        if not candidates:
            raise CapabilityDenied(f"{parent} cannot delegate {sorted(spec.rights)} on {spec.resource}")
        candidates.sort(key=lambda cap: (len(cap.resource), cap.issued_at), reverse=True)
        return candidates[0]

    def _find_transfer_parent(self, actor: str, spec: CapabilitySpec) -> Capability:
        candidates = [
            cap
            for cap in self.capabilities_for(actor)
            if cap.active
            and not self._is_expired(cap)
            and self._parent_chain_active(cap)
            and cap.effect == CapabilityEffect.ALLOW
            and self._resource_covers(cap.resource, spec.resource)
            and spec.rights.issubset(cap.rights)
        ]
        if not candidates:
            raise CapabilityDenied(
                f"{actor} cannot grant {sorted(spec.rights)} on {spec.resource} without already holding those rights"
            )
        candidates.sort(key=lambda cap: (len(cap.resource), cap.issued_at, cap.cap_id), reverse=True)
        return candidates[0]

    def _validate_delegation_parent(self, parent_cap: Capability, selected: CapabilitySpec) -> None:
        if parent_cap.uses_remaining is not None:
            raise CapabilityDenied("finite-use capabilities cannot be delegated")
        if selected.delegable and not parent_cap.delegable:
            raise CapabilityDenied(f"parent capability is not delegable: {parent_cap.cap_id}")
        self._require_temporal_attenuation(parent_cap, selected, action="delegated")
        parent_max_depth = self._capability_max_delegation_depth(parent_cap)
        if parent_cap.delegation_depth >= parent_max_depth:
            raise CapabilityDenied("delegation depth exhausted")
        if selected.max_delegation_depth is not None and selected.max_delegation_depth > parent_max_depth:
            raise CapabilityDenied("delegated capability cannot increase parent delegation depth")
        child_max_depth = selected.max_delegation_depth if selected.max_delegation_depth is not None else parent_max_depth
        if selected.delegable and parent_cap.delegation_depth + 1 >= child_max_depth:
            raise CapabilityDenied("delegated capability cannot be delegable after depth exhaustion")
        self._require_constraint_attenuation(parent_cap, selected)

    def _validate_transfer_parent(self, parent_cap: Capability, selected: CapabilitySpec) -> None:
        if selected.effect != CapabilityEffect.ALLOW:
            raise CapabilityDenied("grant authority can only transfer allow capabilities")
        if parent_cap.uses_remaining is not None:
            raise CapabilityDenied("finite-use capabilities cannot be granted onward")
        if selected.delegable and not parent_cap.delegable:
            raise CapabilityDenied(f"transferred capability cannot be delegable from non-delegable parent: {parent_cap.cap_id}")
        self._require_temporal_attenuation(parent_cap, selected, action="transferred")
        parent_max_depth = self._capability_max_delegation_depth(parent_cap)
        if selected.max_delegation_depth is not None and selected.max_delegation_depth > parent_max_depth:
            raise CapabilityDenied("transferred capability cannot increase parent delegation depth")
        if selected.delegable and parent_cap.delegation_depth + 1 >= parent_max_depth:
            raise CapabilityDenied("transferred capability cannot be delegable after depth exhaustion")
        self._require_constraint_attenuation(parent_cap, selected)

    def _require_temporal_attenuation(self, parent_cap: Capability, selected: CapabilitySpec, *, action: str) -> None:
        if parent_cap.expires_at is not None and selected.expires_at is not None and selected.expires_at > parent_cap.expires_at:
            raise CapabilityDenied(f"{action} capability cannot outlive parent capability")

    def _initial_max_delegation_depth(self, selected: CapabilitySpec) -> int | None:
        if selected.max_delegation_depth is not None:
            return int(selected.max_delegation_depth)
        if selected.delegable:
            return self.config.capability.default_delegation_depth
        return None

    def _delegated_max_delegation_depth(self, parent_cap: Capability, selected: CapabilitySpec) -> int:
        parent_max_depth = self._capability_max_delegation_depth(parent_cap)
        return int(selected.max_delegation_depth) if selected.max_delegation_depth is not None else parent_max_depth

    def _capability_max_delegation_depth(self, cap: Capability) -> int:
        if cap.max_delegation_depth is not None:
            return int(cap.max_delegation_depth)
        return self.config.capability.default_delegation_depth

    def _matching_capabilities(
        self,
        subject: str,
        resource: str,
        right: str | CapabilityRight,
        *,
        include_ask: bool = False,
    ) -> list[Capability]:
        requested_right = str(right)
        self.parse_resource_pattern(resource, requested=True)
        matches: list[Capability] = []
        for cap in self.store.list_capabilities(subject=subject):
            if not cap.active or self._is_expired(cap):
                continue
            if not self._parent_chain_active(cap):
                continue
            if not include_ask and cap.effect == CapabilityEffect.ASK:
                continue
            if not self._resource_matches(cap.resource, resource):
                continue
            if requested_right not in cap.rights:
                continue
            matches.append(cap)
        matches.sort(key=lambda cap: cap.cap_id)
        matches.sort(key=lambda cap: cap.issued_at, reverse=True)
        matches.sort(key=lambda cap: len(cap.resource), reverse=True)
        matches.sort(key=lambda cap: 0 if cap.effect == CapabilityEffect.DENY else 1)
        return matches

    def _resource_matches(self, granted: str, requested: str) -> bool:
        return self.resources.matches(granted, requested)

    def _resource_covers(self, granted: str, requested_pattern: str) -> bool:
        return self.resources.covers(granted, requested_pattern)

    def _coerce_spec(self, spec: CapabilitySpec | dict[str, Any]) -> CapabilitySpec:
        if isinstance(spec, CapabilitySpec):
            data = {
                "resource": spec.resource,
                "rights": list(spec.rights),
                "effect": spec.effect,
                "rules": list(spec.rules),
                "lease": spec.lease,
                "delegation": spec.delegation,
                "constraints": dict(spec.constraints),
                "metadata": dict(spec.metadata),
                "expires_at": spec.expires_at,
                "uses_remaining": spec.uses_remaining,
                "delegable": spec.delegable,
                "revocable": spec.revocable,
                "max_delegation_depth": spec.max_delegation_depth,
            }
        elif isinstance(spec, dict):
            data = dict(spec)
        else:
            raise ValidationError("capability spec must be a mapping")

        constraints = dict(data.get("constraints") or {})
        effect = data.get("effect", CapabilityEffect.ALLOW)
        policy = data.get("policy")
        if policy is None:
            policy = data.get(self.POLICY_KEY)
        if policy is None:
            policy = constraints.pop(self.POLICY_KEY, None)
        else:
            constraints.pop(self.POLICY_KEY, None)
        uses_remaining = data.get("uses_remaining")
        expires_at = self._normalize_expires_at(data.get("expires_at"))
        if policy is not None:
            effect, uses_remaining = self._effect_from_policy(str(policy))
        lease = data.get("lease")
        if lease is not None:
            selected_lease = self._coerce_lease(lease)
            expires_at = selected_lease.expires_at
            uses_remaining = selected_lease.uses_remaining
        delegation = self._coerce_delegation(data.get("delegation"))
        rules = data.get("rules") or []
        if rules:
            constraints[AUTHORITY_RULES_KEY] = [self.rule_codec.to_json(rule) for rule in list(rules)]
        return CapabilitySpec(
            resource=str(data["resource"]),
            rights=self._normalize_rights(data.get("rights", [CapabilityRight.READ.value])),
            effect=CapabilityEffect(effect),
            rules=[self.rule_codec.coerce(rule) for rule in list(rules)],
            lease=CapabilityLease(expires_at=expires_at, uses_remaining=uses_remaining),
            delegation=delegation,
            constraints=constraints,
            metadata=dict(data.get("metadata") or {}),
            expires_at=expires_at,
            uses_remaining=uses_remaining,
            delegable=delegation.delegable if delegation is not None else bool(data.get("delegable", False)),
            revocable=delegation.revocable if delegation is not None else bool(data.get("revocable", True)),
            max_delegation_depth=delegation.max_delegation_depth if delegation is not None else data.get("max_delegation_depth"),
        )

    def _coerce_lease(self, value: CapabilityLease | dict[str, Any]) -> CapabilityLease:
        if isinstance(value, CapabilityLease):
            return value
        if not isinstance(value, dict):
            raise ValidationError("capability lease must be a mapping")
        uses_remaining = value.get("uses_remaining")
        if uses_remaining is not None:
            uses_remaining = int(uses_remaining)
        expires_at = self._normalize_expires_at(value.get("expires_at"))
        return CapabilityLease(
            expires_at=expires_at,
            uses_remaining=uses_remaining,
        )

    def _coerce_delegation(self, value: DelegationPolicy | dict[str, Any] | None) -> DelegationPolicy | None:
        if value is None:
            return None
        if isinstance(value, DelegationPolicy):
            return value
        if not isinstance(value, dict):
            raise ValidationError("capability delegation policy must be a mapping")
        max_depth = value.get("max_delegation_depth")
        if max_depth is not None and int(max_depth) < 0:
            raise ValidationError("max_delegation_depth must be >= 0")
        return DelegationPolicy(
            delegable=bool(value.get("delegable", False)),
            revocable=bool(value.get("revocable", True)),
            max_delegation_depth=int(max_depth) if max_depth is not None else None,
        )

    def _normalize_rights(self, rights: Iterable[str | CapabilityRight]) -> set[str]:
        try:
            normalized = {CapabilityRight(str(right)).value for right in rights}
        except ValueError as exc:
            raise ValidationError(f"unknown capability right: {exc}") from exc
        if not normalized:
            raise ValidationError("capability must include at least one right")
        if len(normalized) > self.config.capability.max_rights_per_capability:
            raise ValidationError("capability rights exceed configured limit")
        return normalized

    def _canonical_resource(self, resource: str) -> str:
        return self.resources.canonical(resource)

    def _effect_from_policy_constraint(self, constraints: dict[str, Any]) -> tuple[CapabilityEffect, int | None]:
        return self._effect_from_policy(str(constraints.get(self.POLICY_KEY, self.ALWAYS_ALLOW)))

    def _effect_from_policy(self, policy: str) -> tuple[CapabilityEffect, int | None]:
        if policy == self.ALWAYS_ALLOW:
            return CapabilityEffect.ALLOW, None
        if policy == self.ALWAYS_DENY:
            return CapabilityEffect.DENY, None
        if policy == self.ASK_EACH_TIME:
            return CapabilityEffect.ASK, None
        if policy == self.ALLOW_ONCE:
            return CapabilityEffect.ALLOW, 1
        raise ValueError(f"unknown permission policy: {policy}")

    def _coerce_risk(self, value: str | AuthorityRisk) -> AuthorityRisk:
        try:
            return AuthorityRisk(str(value))
        except ValueError as exc:
            raise ValidationError(f"unknown authority risk: {value}") from exc

    def _validate_constraints(self, constraints: dict[str, Any]) -> None:
        try:
            size = len(json.dumps(constraints, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        except TypeError as exc:
            raise ValidationError("capability constraints must be JSON-serializable") from exc
        if size > self.config.capability.max_constraints_bytes:
            raise ValidationError("capability constraints exceed configured byte limit")
        if AUTHORITY_RULES_KEY in constraints:
            self.rule_codec.coerce_many(constraints[AUTHORITY_RULES_KEY])

    def _evaluate_constraints(self, cap: Capability, context: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for key, value in cap.constraints.items():
            if key not in self._KNOWN_CONSTRAINT_KEYS:
                results[key] = {"ok": False, "reason": "unknown constraint key"}
                continue
            if key == AUTHORITY_RULES_KEY:
                try:
                    rules = self.rule_codec.coerce_many(value)
                except ValidationError as exc:
                    results[key] = {"ok": False, "reason": str(exc)}
                    continue
                results[key] = self._evaluate_authority_rules(rules, context)
                continue
            results[key] = {"ok": True, "value": value}
        return results

    def _constraint_effect(self, constraint_results: dict[str, Any]) -> CapabilityEffect | None:
        effects = {
            str(result.get("effect"))
            for result in constraint_results.values()
            if result.get("effect") is not None
        }
        if CapabilityEffect.DENY.value in effects:
            return CapabilityEffect.DENY
        if CapabilityEffect.ASK.value in effects:
            return CapabilityEffect.ASK
        if CapabilityEffect.ALLOW.value in effects:
            return CapabilityEffect.ALLOW
        return None

    def _constraint_failure_is_scoped_miss(self, constraint_results: dict[str, Any]) -> bool:
        failed = {
            key: result
            for key, result in constraint_results.items()
            if not bool(result.get("ok"))
        }
        return (
            set(failed) == {AUTHORITY_RULES_KEY}
            and failed[AUTHORITY_RULES_KEY].get("reason") == "no authority rule matched operation context"
        )

    def _evaluate_authority_rules(self, rules: list[Any], context: dict[str, Any]) -> dict[str, Any]:
        operation = str(context.get("authority_operation") or context.get("operation") or "")
        if not operation:
            return {"ok": False, "reason": "authority rule requires operation context"}
        matched = [rule for rule in rules if rule.operation == operation and self._authority_rule_matches(rule, context)]
        if not matched:
            return {
                "ok": False,
                "reason": "no authority rule matched operation context",
                "operation": operation,
                "rule_ids": [rule.rule_id for rule in rules],
            }
        deny = next((rule for rule in matched if rule.effect == CapabilityEffect.DENY), None)
        if deny is not None:
            return {
                "ok": False,
                "effect": CapabilityEffect.DENY.value,
                "rule_id": deny.rule_id,
                "risk": deny.risk.value,
                "operation": operation,
                "reason": "authority rule denied operation",
            }
        ask = next((rule for rule in matched if rule.effect == CapabilityEffect.ASK), None)
        if ask is not None:
            return {
                "ok": True,
                "effect": CapabilityEffect.ASK.value,
                "rule_id": ask.rule_id,
                "risk": ask.risk.value,
                "operation": operation,
            }
        allow = matched[0]
        return {
            "ok": True,
            "effect": CapabilityEffect.ALLOW.value,
            "rule_id": allow.rule_id,
            "risk": allow.risk.value,
            "operation": operation,
        }

    def _authority_rule_matches(self, rule: Any, context: dict[str, Any]) -> bool:
        conditions = dict(rule.conditions or {})
        allowed_conditions = {
            "argv",
            "argv_sha256",
            "match",
            "regex_token",
            "cwd",
            "path",
            "resource",
            "right",
            "endpoint_id",
            "method_id",
            "params_sha256",
            "content_sha256",
            "timeout_s",
            "timeout_max_s",
            "network",
            "filesystem_intent",
            "continuous_session",
            "operation",
            "authority_operation",
            "recursive",
            "missing_ok",
            "overwrite",
            "parents",
            "exist_ok",
        }
        if any(key not in allowed_conditions for key in conditions):
            return False
        if "operation" in conditions and str(context.get("operation")) != str(conditions["operation"]):
            return False
        if "authority_operation" in conditions and str(context.get("authority_operation")) != str(conditions["authority_operation"]):
            return False
        if "argv" in conditions and not self._argv_condition_matches(conditions, context):
            return False
        regex = conditions.get("regex_token")
        if isinstance(regex, str):
            import re

            try:
                pattern = re.compile(regex)
            except re.error:
                return False
            argv = context.get("argv")
            if not isinstance(argv, list) or not any(pattern.fullmatch(str(token)) for token in argv):
                return False
        for key in [
            "argv_sha256",
            "cwd",
            "path",
            "resource",
            "right",
            "endpoint_id",
            "method_id",
            "params_sha256",
            "content_sha256",
            "network",
            "filesystem_intent",
            "continuous_session",
            "recursive",
            "missing_ok",
            "overwrite",
            "parents",
            "exist_ok",
        ]:
            if key in conditions and context.get(key) != conditions[key]:
                return False
        if "timeout_s" in conditions:
            try:
                if float(context.get("timeout_s")) != float(conditions["timeout_s"]):
                    return False
            except (TypeError, ValueError):
                return False
        if "timeout_max_s" in conditions:
            try:
                if float(context.get("timeout_s")) > float(conditions["timeout_max_s"]):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _argv_condition_matches(self, conditions: dict[str, Any], context: dict[str, Any]) -> bool:
        expected = conditions.get("argv")
        actual = context.get("argv")
        if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
            return False
        if not isinstance(actual, list) or not all(isinstance(item, str) for item in actual):
            return False
        match = str(conditions.get("match", "exact"))
        if match == "exact":
            return actual == expected
        if match == "prefix":
            return len(actual) >= len(expected) and actual[: len(expected)] == expected
        return False

    def _require_constraint_attenuation(self, parent_cap: Capability, spec: CapabilitySpec) -> None:
        for key, value in parent_cap.constraints.items():
            if spec.constraints.get(key) != value:
                raise CapabilityDenied(f"delegated capability cannot drop parent constraint: {key}")
        for key, value in spec.constraints.items():
            if key in parent_cap.constraints:
                continue
            if key not in self._KNOWN_CONSTRAINT_KEYS:
                raise CapabilityDenied(f"delegated capability uses unknown constraint: {key}")
            if key == self.config.shell.policy_capability_key:
                raise CapabilityDenied(f"delegated constraint is not covered by parent: {key}")

    def _context_dict(self, context: OperationContext | dict[str, Any] | None) -> dict[str, Any]:
        if context is None:
            return {}
        if isinstance(context, OperationContext):
            return {
                "primitive": context.primitive,
                "operation": context.operation,
                **context.metadata,
            }
        return dict(context)

    def _is_expired(self, cap: Capability) -> bool:
        if cap.expires_at is None:
            return False
        try:
            return self._expires_at_datetime(cap.expires_at) <= datetime.now(timezone.utc)
        except ValidationError:
            return True

    def _parent_chain_active(self, cap: Capability) -> bool:
        parent_id = cap.parent_cap_id
        seen = {cap.cap_id}
        while parent_id is not None:
            if parent_id in seen:
                return False
            parent = self.store.get_capability(parent_id)
            if parent is None or not parent.active or self._is_expired(parent):
                return False
            seen.add(parent_id)
            parent_id = parent.parent_cap_id
        return True

    def _normalize_expires_at(self, value: Any) -> str | None:
        if value is None:
            return None
        dt = self._expires_at_datetime(value)
        return dt.astimezone(timezone.utc).isoformat()

    def _expires_at_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            raw = value.strip()
            if not raw:
                raise ValidationError("capability expires_at must be a non-empty ISO timestamp")
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValidationError("capability expires_at must be an ISO timestamp") from exc
        else:
            raise ValidationError("capability expires_at must be an ISO timestamp")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _is_trusted_issuer(self, actor: str) -> bool:
        if actor in self.config.capability.trusted_issuers:
            return True
        return any(actor.startswith(prefix) for prefix in self.config.capability.trusted_issuer_prefixes)

    def _issuer_chain(self, cap: Capability) -> list[str]:
        chain: list[str] = []
        current: Capability | None = cap
        seen: set[str] = set()
        while current is not None and current.cap_id not in seen:
            seen.add(current.cap_id)
            chain.append(current.cap_id)
            parent_id = current.parent_cap_id or current.issuer_cap_id
            current = self.store.get_capability(parent_id) if parent_id else None
        return chain

    def _record_decision(self, decision: CapabilityDecision, *, audit: bool) -> CapabilityDecision:
        if audit:
            self.audit.record(
                actor=decision.subject,
                action="capability.authorize",
                target=decision.resource,
                capability_refs=decision.matched_capability_ids,
                decision=self._decision_json(decision),
            )
        return decision

    def _decision_json(self, decision: CapabilityDecision) -> dict[str, Any]:
        return {
            "subject": decision.subject,
            "resource": decision.resource,
            "right": decision.right,
            "allowed": decision.allowed,
            "effect": decision.effect.value if decision.effect else None,
            "policy": decision.policy,
            "reason": decision.reason,
            "matched_capability_ids": decision.matched_capability_ids,
            "selected_capability_id": decision.selected_capability_id,
            "consume_capability_id": decision.consume_capability_id,
            "human_request_id": decision.human_request_id,
            "issuer_chain": decision.issuer_chain,
            "constraint_results": decision.constraint_results,
            "context": self._preview_context(decision.context),
        }

    def _preview_context(self, context: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)
        limit = self.config.capability.decision_explain_preview_chars
        if len(text) <= limit:
            return context
        return {"preview": text[:limit], "truncated": True}

    def _capability_json(self, cap: Capability) -> dict[str, Any]:
        rules = []
        if AUTHORITY_RULES_KEY in cap.constraints:
            rules = [self.rule_codec.to_json(rule) for rule in self.rule_codec.coerce_many(cap.constraints[AUTHORITY_RULES_KEY])]
        return {
            "cap_id": cap.cap_id,
            "subject": cap.subject,
            "resource": cap.resource,
            "rights": sorted(cap.rights),
            "effect": cap.effect.value,
            "issuer": cap.issued_by,
            "issuer_cap_id": cap.issuer_cap_id,
            "parent_cap_id": cap.parent_cap_id,
            "delegation_depth": cap.delegation_depth,
            "max_delegation_depth": cap.max_delegation_depth,
            "issued_at": cap.issued_at,
            "expires_at": cap.expires_at,
            "uses_remaining": cap.uses_remaining,
            "status": cap.status.value,
            "delegable": cap.delegable,
            "revocable": cap.revocable,
            "lease": {"expires_at": cap.expires_at, "uses_remaining": cap.uses_remaining},
            "delegation": {
                "delegable": cap.delegable,
                "revocable": cap.revocable,
                "depth": cap.delegation_depth,
                "max_depth": cap.max_delegation_depth,
            },
            "rules": rules,
            "constraints": cap.constraints,
            "metadata": cap.metadata,
        }

    def _attach_to_process(self, subject: str, cap_id: str) -> None:
        process = self.store.get_process(subject)
        if process is None:
            return
        if cap_id not in process.capabilities:
            process.capabilities.append(cap_id)
            process.updated_at = utc_now()
            self.store.update_process(process)
