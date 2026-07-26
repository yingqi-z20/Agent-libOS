from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable

from agent_libos.capability.admission import (
    CapabilityAdmissionPort,
    install_instance_admission_guards,
)
from agent_libos.capability.effect_binding import APPROVAL_BINDING_KEY as APPROVAL_BINDING_CONSTRAINT_KEY
from agent_libos.capability.evaluator import (
    DATA_RELEASE_BINDING_KEY as DATA_RELEASE_BINDING_CONSTRAINT_KEY,
    KNOWN_CONSTRAINT_KEYS,
    CapabilityEvaluator,
)
from agent_libos.capability.lease import CapabilityLeaseService
from agent_libos.capability.mutation import CapabilityDraft, CapabilityMutationService
from agent_libos.capability.profiles import SandboxProfileBuilder
from agent_libos.capability.resources import ResourceAuthority
from agent_libos.capability.rules import AUTHORITY_RULES_KEY, AuthorityRuleCodec
from agent_libos.capability.transaction import AuthorityTransaction
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    AuthorityRisk,
    CapabilityLease,
    Capability,
    CapabilityDecision,
    CapabilityEffect,
    CapabilityRight,
    CapabilitySpec,
    ObjectHandle,
    OperationContext,
    DelegationPolicy,
    ResourcePattern,
    SandboxProfile,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.ports import AuditPort, CapabilityStorePort, EventPort, OperationPort
from agent_libos.utils.ids import utc_now


CAPABILITY_MANAGER_MUTATION_PUBLIC_METHODS = frozenset(
    {
        "assert_handle",
        "authority_transaction",
        "claim_decision_use",
        "commit_reserved_use",
        "consume_allow_once",
        "consume_use",
        "delegate",
        "derive_authority",
        "disable_subject_capability",
        "finalize_exec_revocations",
        "grant",
        "grant_once",
        "handle_for_object",
        "inherit",
        "issue",
        "issue_trusted",
        "require",
        "reserve_decision_use",
        "reserve_use",
        "restore_reserved_use",
        "revoke",
        "revoke_resource_trusted",
        "selected_authority_transaction",
        "set_permission_policy",
        "stage_exec_revocation",
        "transition_allowed_rights",
    }
)

CAPABILITY_MANAGER_MIXED_PUBLIC_METHODS = frozenset(
    {
        "authorize",
        "authorize_matching_capabilities",
        "decision_from_matches",
        "reauthorize_decision",
        "reauthorize_selected_decision",
    }
)

CAPABILITY_MANAGER_READ_ONLY_PUBLIC_METHODS = frozenset(
    {
        "authorize_handle",
        "capabilities_for",
        "check",
        "constraints_satisfied",
        "explain_decision",
        "inspect",
        "inspect_for_presentation",
        "is_expired",
        "list_for_presentation",
        "list_subject",
        "matching_capabilities",
        "object_access",
        "parse_resource_pattern",
        "presentation_page",
        "permission_policy",
        "project_read",
        "resources_overlap",
        "sandbox_profile_for_decision",
        "spec",
        "spec_covers",
        "tool_execute",
        "validate_delegation",
    }
)


@dataclass(frozen=True)
class _IssueAuthority:
    mutation_decision: CapabilityDecision | None = None
    transfer_parent: Capability | None = None


@dataclass(frozen=True)
class _CapabilityPresentationPage:
    capabilities: list[dict[str, Any]]
    has_more: bool
    next_cursor: str | None


class CapabilityManager:
    """Capability directory, authorization engine, and delegation helper."""

    POLICY_KEY = "permission_policy"
    ALWAYS_ALLOW = "always_allow"
    ALWAYS_DENY = "always_deny"
    ASK_EACH_TIME = "ask_each_time"
    ALLOW_ONCE = "allow_once"
    MISSING = "missing"
    APPROVAL_BINDING_KEY = APPROVAL_BINDING_CONSTRAINT_KEY
    DATA_RELEASE_BINDING_KEY = DATA_RELEASE_BINDING_CONSTRAINT_KEY
    POLICY_VALUES = {ALWAYS_ALLOW, ALWAYS_DENY, ASK_EACH_TIME, ALLOW_ONCE}

    _KNOWN_CONSTRAINT_KEYS = KNOWN_CONSTRAINT_KEYS

    def __init__(
        self,
        store: CapabilityStorePort,
        audit: AuditPort,
        events: EventPort,
        config: AgentLibOSConfig | None = None,
        *,
        operations: OperationPort,
        admission: CapabilityAdmissionPort | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.store = store
        self.audit = audit
        self.events = events
        self.operations = operations
        self.resources = ResourceAuthority()
        self.rule_codec = AuthorityRuleCodec()
        self.profiles = SandboxProfileBuilder()
        self.evaluator = CapabilityEvaluator(self.rule_codec, config=self.config)
        self._admission = admission
        self.leases = CapabilityLeaseService(
            store,
            audit,
            events,
            self.operations,
            admission=admission,
        )
        self.mutations = CapabilityMutationService(
            store,
            audit,
            events,
            self.leases,
            admission=admission,
        )
        install_instance_admission_guards(
            self,
            admission=admission,
            mutation_methods=CAPABILITY_MANAGER_MUTATION_PUBLIC_METHODS,
            mixed_audit_methods=CAPABILITY_MANAGER_MIXED_PUBLIC_METHODS,
        )

    def authority_transaction(
        self,
        decisions: Iterable[CapabilityDecision | None],
        *,
        actor: str,
        operation: str,
    ) -> AuthorityTransaction:
        """Return the sole mutation boundary for cached authority decisions."""

        return AuthorityTransaction(
            self.store,
            decisions,
            actor=actor,
            operation=operation,
            reauthorize=self.reauthorize_decision,
            reserve=self._reserve_current_decision_use,
            commit=self.commit_reserved_use,
            admission=self._admission,
        )

    def selected_authority_transaction(
        self,
        decisions: Iterable[CapabilityDecision | None],
        *,
        actor: str,
        operation: str,
    ) -> AuthorityTransaction:
        """Bind a mutation to the exact grants selected during preflight.

        This stricter boundary is required when a capability identifier is part
        of the authority-bearing input, such as an Object Memory handle.  A
        different broad grant may still make the same request generally legal,
        but it must not revive a stale, revoked, expired, or exhausted named
        grant.
        """

        return AuthorityTransaction(
            self.store,
            decisions,
            actor=actor,
            operation=operation,
            reauthorize=self.reauthorize_selected_decision,
            reserve=self._reserve_current_decision_use,
            commit=self.commit_reserved_use,
            admission=self._admission,
        )

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
        if not require_authority:
            draft = self._prepare_issue_draft(
                actor=actor,
                subject=subject,
                selected=selected,
                transfer_parent=None,
                issuer_cap_id=issuer_cap_id,
            )
            return self.mutations.issue(
                draft,
                actor=actor,
                authority_decision=None,
                attach_to_process=self._attach_to_process,
            )

        # This first pass is advisory only.  AuthorityTransaction reauthorizes
        # the selected ADMIN/GRANT decision after entering the mutation UoW.
        # A GRANT also transfers the requested rights, so its allow/deny and
        # parent-attenuation checks must be recomputed inside that same UoW;
        # rechecking GRANT alone would allow a concurrent READ/WRITE deny to be
        # missed while still publishing the child capability.
        issue_authority = self._require_issue_authority(actor, selected)
        authority_decision = issue_authority.mutation_decision
        assert authority_decision is not None
        with self.authority_transaction(
            [authority_decision],
            actor=actor,
            operation="capability issue",
        ) as current_decisions:
            current_decision = current_decisions[0]
            transfer_parent: Capability | None = None
            if issue_authority.transfer_parent is not None:
                transfer_parent = self._find_transfer_parent(actor, selected)
                self._validate_transfer_parent(transfer_parent, selected)
            draft = self._prepare_issue_draft(
                actor=actor,
                subject=subject,
                selected=selected,
                transfer_parent=transfer_parent,
                issuer_cap_id=current_decision.selected_capability_id,
            )
            return self.mutations.issue(
                draft,
                actor=actor,
                authority_decision=None,
                attach_to_process=self._attach_to_process,
            )

    def _prepare_issue_draft(
        self,
        *,
        actor: str,
        subject: str,
        selected: CapabilitySpec,
        transfer_parent: Capability | None,
        issuer_cap_id: str | None,
    ) -> CapabilityDraft:
        delegation_depth = transfer_parent.delegation_depth + 1 if transfer_parent is not None else 0
        max_delegation_depth = (
            self._delegated_max_delegation_depth(transfer_parent, selected)
            if transfer_parent is not None and selected.delegable
            else self._initial_max_delegation_depth(selected)
        )
        expires_at = selected.expires_at
        if transfer_parent is not None and expires_at is None:
            expires_at = transfer_parent.expires_at
        return self._capability_draft(
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
        max_delegation_depth: int | None = None,
    ) -> Capability:
        """Issue authority from a trusted embedding-host code path.

        This is an explicit host bypass, not an actor-name authentication
        mechanism. Process, model, Skill, and JIT surfaces must use ``issue``
        or a primitive that performs the corresponding authority checks.
        """
        selected_constraints = self._optional_spec_mapping(
            constraints,
            field="constraints",
        )
        selected_metadata = self._optional_spec_mapping(
            metadata,
            field="metadata",
        )
        return self.issue(
            actor=issued_by,
            subject=subject,
            spec=CapabilitySpec(
                resource=resource,
                rights=self._normalize_rights(rights),
                effect=CapabilityEffect(effect),
                constraints=selected_constraints,
                metadata=selected_metadata,
                expires_at=expires_at,
                uses_remaining=uses_remaining,
                delegable=delegable,
                revocable=revocable,
                max_delegation_depth=max_delegation_depth,
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
        metadata: dict[str, Any] | None = None,
    ) -> Capability:
        # Trusted runtime paths keep this compact bootstrap helper, while all
        # records still flow through issue() for canonical spec conversion.
        selected_constraints = self._optional_spec_mapping(
            constraints,
            field="constraints",
        )
        effect, uses_remaining = self._effect_from_policy_constraint(
            selected_constraints
        )
        clean_constraints = {
            key: value
            for key, value in selected_constraints.items()
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
            metadata=metadata,
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
        # A delegated capability is not visible until its process attachment,
        # grant event, and audit evidence can all commit together.  In
        # particular, an audit sink failure must not leave the caller with an
        # active child capability after delegate() reports an error.
        with self.store.transaction():
            cap = self._delegate_selected(parent, child, selected, actor=actor)
        return cap

    def _delegate_selected(
        self,
        parent: str,
        child: str,
        selected: CapabilitySpec,
        *,
        actor: str | None,
    ) -> Capability:
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
        self.mutations.record_delegation(
            cap,
            parent_cap=parent_cap,
            parent_subject=parent,
            child_subject=child,
            actor=actor or parent,
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
                constraints=self._optional_spec_mapping(
                    constraints,
                    field="constraints",
                ),
                metadata={"issued_by": issued_by},
                delegable=False,
            ),
            actor=issued_by,
        )

    def spec_covers(
        self,
        parent: Capability | CapabilitySpec | dict[str, Any],
        requested: CapabilitySpec | dict[str, Any],
    ) -> bool:
        """Return whether a declared authority spec safely covers another spec.

        This is the public, side-effect-free coverage primitive used by launch
        manifests and transition planners.  It deliberately applies the same
        resource, rights, constraint, expiry, and finite-use attenuation rules
        as real capability delegation.
        """

        if isinstance(parent, Capability):
            parent_resource = parent.resource
            parent_rights = set(parent.rights)
            parent_effect = parent.effect
            parent_constraints = dict(parent.constraints)
            parent_expires_at = parent.expires_at
            parent_uses = parent.uses_remaining
            parent_delegable = parent.delegable
            parent_max_depth = (
                self._capability_max_delegation_depth(parent)
                if parent.delegable or parent.max_delegation_depth is not None
                else None
            )
        else:
            selected_parent = self._coerce_spec(parent)
            parent_resource = selected_parent.resource
            parent_rights = set(selected_parent.rights)
            parent_effect = selected_parent.effect
            parent_constraints = dict(selected_parent.constraints)
            parent_expires_at = selected_parent.expires_at
            parent_uses = selected_parent.uses_remaining
            parent_delegable = selected_parent.delegable
            parent_max_depth = (
                int(selected_parent.max_delegation_depth)
                if selected_parent.max_delegation_depth is not None
                else (self.config.capability.default_delegation_depth if selected_parent.delegable else None)
            )
        selected = self._coerce_spec(requested)
        # Real delegation and transfer authority is rooted only in an ALLOW
        # capability.  A DENY/ASK declaration is a restrictive boundary, not
        # a ceiling from which affirmative authority may be derived.
        if not self._spec_root_covers(
            parent_effect,
            parent_resource,
            selected,
        ):
            return False
        if not selected.rights.issubset(parent_rights):
            return False
        if any(selected.constraints.get(key) != value for key, value in parent_constraints.items()):
            return False
        if parent_expires_at is not None and (
            selected.expires_at is None or selected.expires_at > parent_expires_at
        ):
            return False
        if parent_uses is not None and selected.uses_remaining != parent_uses:
            return False
        if selected.delegable and not parent_delegable:
            return False
        if (
            parent_max_depth is not None
            and selected.max_delegation_depth is not None
            and int(selected.max_delegation_depth) > parent_max_depth
        ):
            return False
        return True

    def _spec_root_covers(
        self,
        parent_effect: CapabilityEffect,
        parent_resource: str,
        selected: CapabilitySpec,
    ) -> bool:
        return (
            parent_effect == CapabilityEffect.ALLOW
            and self._resource_covers(parent_resource, selected.resource)
        )

    def derive_authority(
        self,
        *,
        source_subject: str,
        target_subject: str,
        requested_specs: Iterable[CapabilitySpec | dict[str, Any]],
        transition_kind: str,
        ceiling_specs: Iterable[CapabilitySpec | dict[str, Any]] | None = None,
        actor: str | None = None,
    ) -> list[Capability]:
        """Derive child authority through one audited transition entry point."""

        ceiling = list(ceiling_specs or [])
        selected_specs = [self._coerce_spec(requested) for requested in requested_specs]
        derived: list[Capability] = []
        transition_actor = actor or f"authority_transition:{transition_kind}"
        # Validate the complete transition before publishing its first grant,
        # then keep all delegated rows and evidence under one outer
        # transaction.  The store lock also prevents a parent capability from
        # changing between validation and insertion.
        with self.store.transaction():
            for selected in selected_specs:
                if ceiling and not any(self.spec_covers(limit, selected) for limit in ceiling):
                    raise CapabilityDenied(
                        f"{transition_kind} authority exceeds transition ceiling: "
                        f"{selected.resource} rights={sorted(selected.rights)}"
                    )
                parent_cap = self._find_delegation_parent(source_subject, selected)
                self._validate_delegation_parent(parent_cap, selected)
            for selected in selected_specs:
                derived.append(
                    self._delegate_selected(
                        source_subject,
                        target_subject,
                        selected,
                        actor=transition_actor,
                    )
                )
            self.audit.record(
                actor=actor or source_subject,
                action="capability.derive_authority",
                target=f"{source_subject}->{target_subject}",
                capability_refs=[cap.cap_id for cap in derived],
                decision={
                    "transition_kind": transition_kind,
                    "derived": len(derived),
                    "ceiling_applied": bool(ceiling),
                },
            )
        return derived

    def is_expired(self, capability: Capability) -> bool:
        """Public expiry predicate for authority transition planners."""

        return self._is_expired(capability)

    def resources_overlap(self, left: str, right: str) -> bool:
        """Return whether two canonical resource patterns may select one target."""

        try:
            left_pattern = self.parse_resource_pattern(left)
            right_pattern = self.parse_resource_pattern(right)
        except CapabilityDenied:
            return left == right
        if left_pattern.kind != right_pattern.kind:
            return False
        if self._resource_matches(left, right) or self._resource_matches(right, left):
            return True
        left_has_wildcard = left.endswith(":*") or left.endswith("/*")
        right_has_wildcard = right.endswith(":*") or right.endswith("/*")
        if left_has_wildcard or right_has_wildcard:
            return left_pattern.body.startswith(right_pattern.body) or right_pattern.body.startswith(left_pattern.body)
        return False

    def transition_allowed_rights(
        self,
        capability: Capability,
        *,
        transition_kind: str,
        duplicates_authority: bool,
    ) -> list[str]:
        """Apply the common policy ceiling for checkpoint/image/process transitions."""

        if not capability.active or self.is_expired(capability):
            return []
        if duplicates_authority and capability.uses_remaining is not None:
            return []
        if capability.effect != CapabilityEffect.ALLOW:
            return sorted(capability.rights)
        restrictive = [
            candidate
            for candidate in self.capabilities_for(capability.subject)
            if candidate.active
            and not self.is_expired(candidate)
            and candidate.effect in {CapabilityEffect.DENY, CapabilityEffect.ASK}
        ]
        allowed: list[str] = []
        for right in sorted(capability.rights):
            if any(
                right in candidate.rights
                and self.resources_overlap(capability.resource, candidate.resource)
                for candidate in restrictive
            ):
                continue
            decision = self.authorize(
                capability.subject,
                capability.resource,
                right,
                {"primitive": "authority_transition", "operation": transition_kind},
            )
            if decision.allowed and decision.selected_capability_id == capability.cap_id:
                allowed.append(right)
        self.audit.record(
            actor=capability.subject,
            action="capability.transition_filter",
            target=f"capability:{capability.cap_id}",
            capability_refs=[capability.cap_id],
            decision={
                "transition_kind": transition_kind,
                "duplicates_authority": duplicates_authority,
                "allowed_rights": allowed,
            },
        )
        return allowed

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
            if cap.subject == subject
            and cap.active
            and not self._is_expired(cap)
            and self._parent_chain_active(cap)
            and self._resource_matches(cap.resource, resource)
            and requested_right in cap.rights
        ]
        matches = self._sort_matching_capabilities(matches)
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
        decision = self.evaluator.decide(
            subject=subject,
            resource=resource,
            requested_right=requested_right,
            matches=matches,
            context=selected_context,
        )
        selected = next(
            (cap for cap in matches if cap.cap_id == decision.selected_capability_id),
            None,
        )
        if selected is not None:
            decision = replace(decision, issuer_chain=self._issuer_chain(selected))
        return self._record_decision(decision, audit=audit)

    def decision_from_matches(
        self,
        *,
        subject: str,
        resource: str,
        requested_right: str,
        matches: list[Capability],
        selected_context: dict[str, Any],
        audit: bool,
    ) -> CapabilityDecision:
        """Evaluate preselected grants through the public capability façade."""

        return self._decision_from_matches(
            subject=subject,
            resource=resource,
            requested_right=requested_right,
            matches=matches,
            selected_context=selected_context,
            audit=audit,
        )

    def require(
        self,
        subject: str,
        resource: str,
        right: str | CapabilityRight,
        context: OperationContext | dict[str, Any] | None = None,
        *,
        consume: bool = True,
        used_by: str | None = None,
        reason: str = "one-time required capability consumed",
    ) -> CapabilityDecision:
        """Require authority and atomically claim finite-use grants by default.

        Callers that cross a fallible effect boundary and need compensating
        rollback must opt out with ``consume=False`` and use the reservation
        API.  Making consumption the default prevents a forgotten follow-up
        claim from silently turning a one-shot grant into reusable authority.
        """
        decision = self.authorize(subject, resource, right, context, audit=True)
        if not decision.allowed:
            raise CapabilityDenied(decision.reason)
        if consume:
            with self.authority_transaction(
                [decision],
                actor=used_by or subject,
                operation=reason,
            ) as current:
                return replace(current[0], consume_capability_id=None)
        return decision

    def reauthorize_decision(
        self,
        decision: CapabilityDecision,
        *,
        audit: bool = False,
    ) -> CapabilityDecision:
        """Re-evaluate a cached decision against the current authority state.

        Protected operations use this immediately before reserving authority
        and preparing an external-effect intent.  Replaying the exact request
        context is important: constraints and scoped deny rules must retain
        the same semantics as the original authorization.
        """

        current = self.authorize(
            decision.subject,
            decision.resource,
            decision.right,
            dict(decision.context),
            audit=audit,
        )
        if not current.allowed:
            raise CapabilityDenied(
                "capability authority changed before protected dispatch: "
                f"{current.reason}"
            )
        return current

    def reauthorize_selected_decision(
        self,
        decision: CapabilityDecision,
        *,
        audit: bool = False,
    ) -> CapabilityDecision:
        """Reauthorize only the grant named by a cached allowed decision.

        Global evaluation still runs first so a newly committed DENY or ASK
        dominates.  The subsequent evaluation is restricted to the original
        selected capability, preventing an unrelated grant from laundering a
        stale named authority decision.
        """

        selected_capability_id = decision.selected_capability_id
        if not decision.allowed or selected_capability_id is None:
            raise CapabilityDenied(
                "cannot reauthorize a denied or unbound capability decision"
            )
        if (
            decision.consume_capability_id is not None
            and decision.consume_capability_id != selected_capability_id
        ):
            raise CapabilityDenied(
                "cached finite-use decision is not bound to its selected capability"
            )

        global_decision = self.authorize(
            decision.subject,
            decision.resource,
            decision.right,
            dict(decision.context),
            audit=audit,
        )
        if not global_decision.allowed:
            raise CapabilityDenied(
                "capability authority changed before protected dispatch: "
                f"{global_decision.reason}"
            )

        selected = self.store.get_capability(selected_capability_id)
        if selected is None:
            raise CapabilityDenied(
                "selected capability no longer exists before protected dispatch"
            )
        if (
            selected.metadata.get("object_handle") is True
            and selected.resource != decision.resource
        ):
            raise CapabilityDenied(
                "named object capability no longer matches its exact resource"
            )
        current = self.authorize_matching_capabilities(
            decision.subject,
            decision.resource,
            decision.right,
            [selected],
            dict(decision.context),
            audit=audit,
        )
        if not current.allowed or current.selected_capability_id != selected_capability_id:
            raise CapabilityDenied(
                "selected capability authority changed before protected dispatch"
            )
        return current

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
        expires_at: str | None = None,
    ) -> Capability:
        if policy not in self.POLICY_VALUES:
            raise ValueError(f"unknown permission policy: {policy}")
        effect, uses_remaining = self._effect_from_policy(policy)
        with self.store.transaction():
            cap = self.issue_trusted(
                subject=subject,
                resource=resource,
                rights=rights,
                issued_by=issued_by or self.config.runtime.default_human_actor,
                effect=effect,
                constraints=constraints,
                expires_at=expires_at,
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
        expires_at: str | None = None,
    ) -> Capability:
        return self.issue_trusted(
            subject=subject,
            resource=resource,
            rights=rights,
            issued_by=issued_by or self.config.runtime.default_human_actor,
            effect=CapabilityEffect.ALLOW,
            constraints=constraints,
            expires_at=expires_at,
            uses_remaining=1,
        )

    def consume_use(self, cap_id: str, *, used_by: str, reason: str = "capability use consumed", count: int = 1) -> Capability:
        return self.leases.consume(cap_id, used_by=used_by, reason=reason, count=count)

    def reserve_use(
        self,
        cap_id: str,
        *,
        reserved_by: str,
        reason: str = "capability use reserved",
        count: int = 1,
    ) -> str:
        """Atomically reserve a finite capability use before a fallible effect.

        The returned reservation id is the only value accepted by the restore
        path. Explicit revoke/disable invalidates outstanding reservations, so a
        late failure cleanup cannot reactivate authority that was revoked while
        the provider call was in flight.
        """
        return self.leases.reserve(
            cap_id,
            reserved_by=reserved_by,
            reason=reason,
            count=count,
        )

    def reserve_decision_use(self, decision: CapabilityDecision | None, *, used_by: str, reason: str) -> str | None:
        if decision is None:
            return None
        if not decision.allowed:
            raise CapabilityDenied("cannot reserve a denied capability decision")
        with self.store.transaction():
            current = self.reauthorize_decision(decision)
            return self._reserve_current_decision_use(
                current,
                used_by=used_by,
                reason=reason,
            )

    def _reserve_current_decision_use(
        self,
        decision: CapabilityDecision | None,
        *,
        used_by: str,
        reason: str,
    ) -> str | None:
        return self.leases.reserve_decision(
            decision,
            used_by=used_by,
            reason=reason,
        )

    def commit_reserved_use(self, reservation_id: str | None, *, committed_by: str, reason: str) -> bool:
        return self.leases.commit(reservation_id, committed_by=committed_by, reason=reason)

    def restore_reserved_use(
        self,
        reservation_id: str | None,
        *,
        restored_by: str,
        reason: str = "reserved capability use restored",
    ) -> Capability | None:
        """Restore only the exact still-live reservation created for this effect."""
        return self.leases.restore(
            reservation_id,
            restored_by=restored_by,
            reason=reason,
        )

    def consume_allow_once(self, subject: str, resource: str, right: str | CapabilityRight, used_by: str) -> None:
        decision = self.authorize(subject, resource, right)
        if decision.consume_capability_id is not None:
            self.claim_decision_use(
                decision,
                used_by=used_by,
                reason="one-time permission consumed",
            )

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
        if authority_decision is None:
            return self.mutations.revoke(
                cap_id,
                revoked_by=revoked_by,
                reason=reason,
                authority_decision=None,
            )
        with self.authority_transaction(
            [authority_decision],
            actor=revoked_by,
            operation="capability revoke",
        ):
            return self.mutations.revoke(
                cap_id,
                revoked_by=revoked_by,
                reason=reason,
                authority_decision=None,
            )

    def stage_exec_revocation(
        self,
        cap_id: str,
        *,
        rollback_token: str,
    ) -> Capability:
        """Internal exec publication transition; never grants authority."""

        return self.mutations.stage_exec_revocation(
            cap_id,
            rollback_token=rollback_token,
        )

    def finalize_exec_revocations(
        self,
        subject: str,
        *,
        rollback_token: str,
    ) -> list[Capability]:
        return self.mutations.finalize_exec_revocations(
            subject,
            rollback_token=rollback_token,
        )

    def disable_subject_capability(
        self,
        cap_id: str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> Capability:
        return self.mutations.disable(cap_id, actor=actor, reason=reason)

    def revoke_resource_trusted(
        self,
        resource: str,
        *,
        revoked_by: str,
        reason: str | None = None,
    ) -> list[Capability]:
        return self.mutations.revoke_resource(
            resource,
            revoked_by=revoked_by,
            reason=reason,
        )

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
        transaction = AuthorityTransaction(
            self.store,
            [decision],
            actor="object_memory",
            operation="object handle permission",
            reauthorize=lambda _decision: self._reauthorize_handle_decision(
                subject,
                handle,
                requested,
            ),
            reserve=self._reserve_current_decision_use,
            commit=self.commit_reserved_use,
            admission=self._admission,
        )
        with transaction:
            pass

    def _reauthorize_handle_decision(
        self,
        subject: str,
        handle: ObjectHandle,
        requested: str,
    ) -> CapabilityDecision:
        try:
            current = self.authorize_handle(subject, handle, requested)
        except CapabilityDenied as exc:
            raise CapabilityDenied(
                "object handle authority changed before permission claim"
            ) from exc
        if not current.allowed:
            raise CapabilityDenied(
                "object handle authority changed before permission claim"
            )
        if current.selected_capability_id != handle.capability_id:
            raise CapabilityDenied(
                "object handle authority no longer matches its named capability"
            )
        return current

    def handle_for_object(
        self,
        subject: str,
        oid: str,
        rights: Iterable[str | CapabilityRight],
        issued_by: str = "system",
        expires_at: str | None = None,
        uses_remaining: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ObjectHandle:
        normalized = self._normalize_rights(rights)
        selected_metadata = self._optional_spec_mapping(
            metadata,
            field="metadata",
        )
        cap = self.issue_trusted(
            subject=subject,
            resource=f"object:{oid}",
            rights=normalized,
            issued_by=issued_by,
            expires_at=expires_at,
            uses_remaining=uses_remaining,
            delegable=False,
            metadata={**selected_metadata, "object_handle": True},
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

    def list_for_presentation(
        self,
        *,
        subject: str | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[Capability]:
        """Return an exact, hard-bounded capability presentation page.

        Authority checks intentionally retain the complete subject set through
        :meth:`capabilities_for`. Presentation enumeration instead uses a
        stable SQL keyset and keeps only one bounded page in memory while
        applying expiry and parent-chain validity checks.
        """

        selected_limit = self.config.capability.list_limit if limit is None else limit
        if (
            type(selected_limit) is not int
            or selected_limit <= 0
            or selected_limit > self.config.capability.list_limit
        ):
            raise ValidationError(
                "capability list limit must be a positive integer no greater than "
                f"capability.list_limit={self.config.capability.list_limit}"
            )
        if include_inactive:
            return self.store.list_capabilities(
                subject=subject,
                limit=selected_limit,
            )

        selected: list[Capability] = []
        after_cap_id: str | None = None
        page_limit = min(
            self.config.capability.list_limit,
            selected_limit + 1,
        )
        while len(selected) < selected_limit:
            page = self.store.query_capabilities(
                subject=subject,
                active_only=True,
                after_cap_id=after_cap_id,
                limit=page_limit,
            )
            if not page:
                break
            after_cap_id = str(page[-1].cap_id)
            for capability in page:
                if self._is_expired(capability) or not self._parent_chain_active(
                    capability
                ):
                    continue
                selected.append(capability)
                if len(selected) == selected_limit:
                    break
            if len(page) < page_limit:
                break
        return selected

    def list_subject(
        self,
        subject: str,
        *,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[Capability]:
        return self.list_for_presentation(
            subject=subject,
            include_inactive=include_inactive,
            limit=limit,
        )

    def inspect(self, cap_id: str) -> dict[str, Any]:
        cap = self.store.get_capability(cap_id)
        if cap is None:
            raise NotFound(f"capability not found: {cap_id}")
        return self._capability_json(cap)

    def inspect_for_presentation(
        self,
        cap_id: str,
        *,
        max_bytes: int,
    ) -> dict[str, Any]:
        """Return one deterministic capability view within a caller budget."""

        self._validate_presentation_byte_limit(max_bytes)
        cap = self.store.get_capability(cap_id)
        if cap is None:
            raise NotFound(f"capability not found: {cap_id}")
        return self._bounded_capability_json(cap, max_bytes=max_bytes)

    def presentation_page(
        self,
        *,
        subject: str | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
        after_cap_id: str | None = None,
        max_bytes: int,
    ) -> _CapabilityPresentationPage:
        """Build a resumable count- and byte-bounded capability page.

        The storage query intentionally fetches one row at a time.  This keeps
        large per-record constraints or metadata from being decoded in bulk
        before the presentation byte budget can stop the scan.
        """

        selected_limit = self.config.capability.list_limit if limit is None else limit
        if (
            type(selected_limit) is not int
            or selected_limit <= 0
            or selected_limit > self.config.capability.list_limit
        ):
            raise ValidationError(
                "capability list limit must be a positive integer no greater than "
                f"capability.list_limit={self.config.capability.list_limit}"
            )
        if after_cap_id is not None and (
            type(after_cap_id) is not str or not after_cap_id
        ):
            raise ValidationError(
                "capability page cursor must be a non-empty string when set"
            )
        self._validate_presentation_byte_limit(max_bytes)

        selected: list[dict[str, Any]] = []
        scan_cursor = after_cap_id
        while True:
            records = self.store.query_capabilities(
                subject=subject,
                active_only=not include_inactive,
                after_cap_id=scan_cursor,
                limit=1,
            )
            if not records:
                return _CapabilityPresentationPage(
                    capabilities=selected,
                    has_more=False,
                    next_cursor=None,
                )
            capability = records[0]
            capability_id = str(capability.cap_id)
            if not include_inactive and (
                self._is_expired(capability)
                or not self._parent_chain_active(capability)
            ):
                scan_cursor = capability_id
                continue
            if len(selected) >= selected_limit:
                return _CapabilityPresentationPage(
                    capabilities=selected,
                    has_more=True,
                    next_cursor=scan_cursor,
                )

            empty_envelope_size = self._presentation_json_size(
                {
                    "capabilities": selected,
                    "has_more": True,
                    "next_cursor": capability_id,
                }
            )
            available = max_bytes - empty_envelope_size
            if available <= 0:
                if selected:
                    return _CapabilityPresentationPage(
                        capabilities=selected,
                        has_more=True,
                        next_cursor=scan_cursor,
                    )
                raise ValidationError(
                    "capability presentation byte limit is too small for a page envelope"
                )
            try:
                projection = self._bounded_capability_json(
                    capability,
                    max_bytes=available,
                )
            except ValidationError:
                if selected:
                    return _CapabilityPresentationPage(
                        capabilities=selected,
                        has_more=True,
                        next_cursor=scan_cursor,
                    )
                raise
            candidate = [*selected, projection]
            candidate_size = self._presentation_json_size(
                {
                    "capabilities": candidate,
                    "has_more": True,
                    "next_cursor": capability_id,
                }
            )
            if candidate_size > max_bytes:
                if selected:
                    return _CapabilityPresentationPage(
                        capabilities=selected,
                        has_more=True,
                        next_cursor=scan_cursor,
                    )
                raise ValidationError(
                    "capability presentation cannot fit one record within the byte limit"
                )
            selected.append(projection)
            scan_cursor = capability_id

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
        draft = self._capability_draft(
            subject=subject,
            resource=resource,
            rights=rights,
            effect=effect,
            constraints=constraints,
            metadata=metadata,
            issued_by=issued_by,
            issuer_cap_id=issuer_cap_id,
            parent_cap_id=parent_cap_id,
            delegation_depth=delegation_depth,
            max_delegation_depth=max_delegation_depth,
            expires_at=expires_at,
            uses_remaining=uses_remaining,
            delegable=delegable,
            revocable=revocable,
        )
        return self.mutations.publish(draft, attach_to_process=self._attach_to_process)

    def _capability_draft(
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
    ) -> CapabilityDraft:
        if not subject:
            raise ValidationError("capability subject must be non-empty")
        self.parse_resource_pattern(resource)
        normalized_rights = self._normalize_rights(rights)
        self._validate_constraints(constraints)
        uses_remaining = self._positive_uses_remaining(
            uses_remaining,
            label="uses_remaining",
        )
        delegable = self._capability_boolean(delegable, field="delegable")
        revocable = self._capability_boolean(revocable, field="revocable")
        max_delegation_depth = self._max_delegation_depth(
            max_delegation_depth
        )
        if max_delegation_depth is not None and max_delegation_depth < delegation_depth:
            raise ValidationError("max_delegation_depth cannot be less than delegation_depth")
        normalized_expires_at = self._normalize_expires_at(expires_at)
        return CapabilityDraft(
            subject=subject,
            resource=self._canonical_resource(resource),
            rights=normalized_rights,
            constraints=dict(constraints),
            issued_by=issued_by,
            expires_at=normalized_expires_at,
            delegable=delegable,
            revocable=revocable,
            effect=effect,
            issuer_cap_id=issuer_cap_id,
            parent_cap_id=parent_cap_id,
            delegation_depth=delegation_depth,
            max_delegation_depth=max_delegation_depth,
            uses_remaining=uses_remaining,
            metadata=dict(metadata),
        )

    def claim_decision_use(self, decision: CapabilityDecision, *, used_by: str, reason: str) -> None:
        if not decision.allowed:
            raise CapabilityDenied("cannot claim a denied capability decision")
        # A cached allow is advisory.  Re-evaluate the exact request and reserve
        # its current finite authority under the same store transaction that
        # settles the claim, so a committed DENY/revoke/expiry cannot race a
        # stale direct decrement.
        with self.authority_transaction(
            [decision],
            actor=used_by,
            operation=reason,
        ):
            pass

    def _require_issue_authority(self, actor: str, spec: CapabilitySpec) -> _IssueAuthority:
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
        if actor == cap.issued_by:
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
        self._require_no_restrictive_parent_boundary(parent, spec, action="delegate")
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
        self._require_no_restrictive_parent_boundary(actor, spec, action="grant")
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
        first_error: CapabilityDenied | None = None
        for candidate in candidates:
            try:
                self._validate_transfer_parent(candidate, spec)
            except CapabilityDenied as exc:
                if first_error is None:
                    first_error = exc
                continue
            return candidate
        assert first_error is not None
        raise first_error

    def _require_no_restrictive_parent_boundary(self, subject: str, spec: CapabilitySpec, *, action: str) -> None:
        for cap in self.capabilities_for(subject):
            if not cap.active or self._is_expired(cap):
                continue
            if not self._parent_chain_active(cap):
                continue
            if not spec.rights.intersection(cap.rights):
                continue
            if not self._resource_patterns_intersect(cap.resource, spec.resource):
                continue
            self._require_parent_authority_rules_well_formed(cap, subject=subject, spec=spec, action=action)
            if cap.effect not in {CapabilityEffect.DENY, CapabilityEffect.ASK}:
                continue
            raise CapabilityDenied(
                f"{subject} cannot {action} {sorted(spec.rights)} on {spec.resource}; "
                f"restrictive capability {cap.cap_id} also covers that authority"
            )

    def _require_parent_authority_rules_well_formed(
        self,
        cap: Capability,
        *,
        subject: str,
        spec: CapabilitySpec,
        action: str,
    ) -> None:
        raw_rules = cap.constraints.get(AUTHORITY_RULES_KEY)
        if raw_rules is None:
            return
        try:
            rules = self.rule_codec.coerce_many(raw_rules)
        except Exception as exc:
            raise CapabilityDenied(
                f"{subject} cannot {action} {sorted(spec.rights)} on {spec.resource}; "
                f"capability {cap.cap_id} has malformed authority rules"
            ) from exc
        for rule in rules:
            unknown_conditions = self._unknown_authority_rule_conditions(rule)
            if unknown_conditions:
                raise CapabilityDenied(
                    f"{subject} cannot {action} {sorted(spec.rights)} on {spec.resource}; "
                    f"capability {cap.cap_id} has unknown authority rule conditions"
                )
            malformed_conditions = self._malformed_authority_rule_conditions(rule)
            if malformed_conditions:
                raise CapabilityDenied(
                    f"{subject} cannot {action} {sorted(spec.rights)} on {spec.resource}; "
                    f"capability {cap.cap_id} has malformed authority rule conditions"
                )

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
        if (
            selected.expires_at is not None
            and self._expires_at_datetime(selected.expires_at)
            <= datetime.now(timezone.utc)
        ):
            raise ValidationError(
                f"{action} capability expiry must be in the future"
            )
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
        return self._sort_matching_capabilities(matches)

    def _sort_matching_capabilities(self, capabilities: Iterable[Capability]) -> list[Capability]:
        return self.evaluator.sort_matching_capabilities(capabilities)

    def _resource_matches(self, granted: str, requested: str) -> bool:
        return self.resources.matches(granted, requested)

    def _resource_covers(self, granted: str, requested_pattern: str) -> bool:
        return self.resources.covers(granted, requested_pattern)

    def _resource_patterns_intersect(self, left: str, right: str) -> bool:
        return self._resource_covers(left, right) or self._resource_covers(right, left)

    def _coerce_spec(self, spec: CapabilitySpec | dict[str, Any]) -> CapabilitySpec:
        data, structured = self._capability_spec_data(spec)
        resource = data.get("resource")
        if type(resource) is not str or not resource:
            raise ValidationError("capability resource must be a non-empty string")
        raw_rights = self._spec_rights(data, structured=structured)
        constraints = self._spec_mapping(data, field="constraints")
        metadata = self._spec_mapping(data, field="metadata")
        self._validate_spec_policy_sources(data, constraints)
        rules = data.get("rules", [])
        if type(rules) not in {list, tuple}:
            raise ValidationError("capability rules must be a list")
        effect = data.get("effect", CapabilityEffect.ALLOW)
        policy = self._pop_spec_policy(data, constraints)
        uses_remaining = self._positive_uses_remaining(
            data.get("uses_remaining"),
            label="capability uses_remaining",
        )
        expires_at = self._normalize_expires_at(data.get("expires_at"))
        if policy is not None:
            effect, uses_remaining = self._effect_from_policy(str(policy))
        lease = data.get("lease")
        if lease is not None:
            selected_lease = self._coerce_lease(lease)
            expires_at = selected_lease.expires_at
            uses_remaining = selected_lease.uses_remaining
        normalized_effect = CapabilityEffect(effect)
        if (
            normalized_effect in {CapabilityEffect.ASK, CapabilityEffect.DENY}
            and uses_remaining is not None
        ):
            raise ValidationError(
                "uses_remaining is supported only for allow capabilities"
            )
        # Validate every authority-shaping scalar before selecting either the
        # legacy top-level fields or the structured delegation policy.
        delegable = self._capability_boolean(
            data.get("delegable", False),
            field="delegable",
        )
        revocable = self._capability_boolean(
            data.get("revocable", True),
            field="revocable",
        )
        max_delegation_depth = self._max_delegation_depth(
            data.get("max_delegation_depth")
        )
        delegation = self._coerce_delegation(data.get("delegation"))
        if rules:
            constraints[AUTHORITY_RULES_KEY] = [
                self.rule_codec.to_json(rule)
                for rule in rules
            ]
        return CapabilitySpec(
            resource=resource,
            rights=self._normalize_rights(raw_rights),
            effect=normalized_effect,
            rules=[self.rule_codec.coerce(rule) for rule in rules],
            lease=CapabilityLease(expires_at=expires_at, uses_remaining=uses_remaining),
            delegation=delegation,
            constraints=constraints,
            metadata=metadata,
            expires_at=expires_at,
            uses_remaining=uses_remaining,
            delegable=delegation.delegable if delegation is not None else delegable,
            revocable=delegation.revocable if delegation is not None else revocable,
            max_delegation_depth=(
                delegation.max_delegation_depth
                if delegation is not None
                else max_delegation_depth
            ),
        )

    def _capability_spec_data(
        self,
        spec: CapabilitySpec | dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        if isinstance(spec, CapabilitySpec):
            self._validate_typed_spec_conflicts(spec)
            return {
                "resource": spec.resource,
                "rights": spec.rights,
                "effect": spec.effect,
                "rules": spec.rules,
                "lease": spec.lease,
                "delegation": spec.delegation,
                "constraints": spec.constraints,
                "metadata": spec.metadata,
                "expires_at": spec.expires_at,
                "uses_remaining": spec.uses_remaining,
                "delegable": spec.delegable,
                "revocable": spec.revocable,
                "max_delegation_depth": spec.max_delegation_depth,
            }, True
        if type(spec) is not dict:
            raise ValidationError("capability spec must be a mapping")
        data = dict(spec)
        self._validate_mapping_spec_fields(data)
        return data, False

    def _validate_typed_spec_conflicts(self, spec: CapabilitySpec) -> None:
        if spec.lease is not None:
            lease = self._coerce_lease(spec.lease)
            if spec.expires_at is not None and (
                self._normalize_expires_at(spec.expires_at) != lease.expires_at
            ):
                raise ValidationError(
                    "capability spec contains conflicting lease fields"
                )
            if spec.uses_remaining is not None and (
                self._positive_uses_remaining(
                    spec.uses_remaining,
                    label="capability uses_remaining",
                )
                != lease.uses_remaining
            ):
                raise ValidationError(
                    "capability spec contains conflicting lease fields"
                )
        if spec.delegation is not None:
            delegation = self._coerce_delegation(spec.delegation)
            assert delegation is not None
            if (
                (spec.delegable is not False and spec.delegable != delegation.delegable)
                or (spec.revocable is not True and spec.revocable != delegation.revocable)
                or (
                    spec.max_delegation_depth is not None
                    and spec.max_delegation_depth
                    != delegation.max_delegation_depth
                )
            ):
                raise ValidationError(
                    "capability spec contains conflicting delegation fields"
                )

    @classmethod
    def _validate_mapping_spec_fields(cls, data: dict[str, Any]) -> None:
        known = {
            "resource",
            "rights",
            "effect",
            "rules",
            "lease",
            "delegation",
            "constraints",
            "metadata",
            "expires_at",
            "uses_remaining",
            "delegable",
            "revocable",
            "max_delegation_depth",
            "policy",
            cls.POLICY_KEY,
        }
        if set(data) - known:
            raise ValidationError("capability spec contains unknown fields")
        if "lease" in data and {"expires_at", "uses_remaining"}.intersection(data):
            raise ValidationError("capability spec contains conflicting lease fields")
        delegation_fields = {
            "delegable",
            "revocable",
            "max_delegation_depth",
        }
        if "delegation" in data and delegation_fields.intersection(data):
            raise ValidationError(
                "capability spec contains conflicting delegation fields"
            )
        if "policy" in data and cls.POLICY_KEY in data:
            raise ValidationError("capability spec contains conflicting policy fields")

    @staticmethod
    def _spec_rights(
        data: dict[str, Any],
        *,
        structured: bool,
    ) -> Any:
        raw_rights = data.get("rights", [CapabilityRight.READ.value])
        if structured and type(raw_rights) not in {list, tuple, set, frozenset}:
            raise ValidationError(
                "CapabilitySpec rights must be a declared rights collection"
            )
        if not structured and type(raw_rights) is not list:
            raise ValidationError("capability mapping rights must be a list")
        return raw_rights

    @classmethod
    def _validate_spec_policy_sources(
        cls,
        data: dict[str, Any],
        constraints: dict[str, Any],
    ) -> None:
        if (
            cls.POLICY_KEY in constraints
            and ("policy" in data or cls.POLICY_KEY in data)
        ):
            raise ValidationError(
                "capability spec contains conflicting policy fields"
            )

    @classmethod
    def _pop_spec_policy(
        cls,
        data: dict[str, Any],
        constraints: dict[str, Any],
    ) -> Any:
        policy = data.get("policy")
        if policy is None:
            policy = data.get(cls.POLICY_KEY)
        if policy is None:
            return constraints.pop(cls.POLICY_KEY, None)
        else:
            constraints.pop(cls.POLICY_KEY, None)
            return policy

    @staticmethod
    def _spec_mapping(
        data: dict[str, Any],
        *,
        field: str,
    ) -> dict[str, Any]:
        if field not in data:
            return {}
        value = data[field]
        if type(value) is not dict:
            raise ValidationError(f"capability {field} must be an object")
        if any(type(key) is not str for key in value):
            raise ValidationError(
                f"capability {field} keys must be strings"
            )
        return dict(value)

    @classmethod
    def _optional_spec_mapping(
        cls,
        value: Any,
        *,
        field: str,
    ) -> dict[str, Any]:
        if value is None:
            return {}
        return cls._spec_mapping({field: value}, field=field)

    def _coerce_lease(self, value: CapabilityLease | dict[str, Any]) -> CapabilityLease:
        if isinstance(value, CapabilityLease):
            data = {
                "expires_at": value.expires_at,
                "uses_remaining": value.uses_remaining,
            }
        elif type(value) is dict:
            data = dict(value)
            if set(data) - {"expires_at", "uses_remaining"}:
                raise ValidationError("capability lease contains unknown fields")
        else:
            raise ValidationError("capability lease must be a mapping")
        uses_remaining = self._positive_uses_remaining(
            data.get("uses_remaining"),
            label="capability lease uses_remaining",
        )
        expires_at = self._normalize_expires_at(data.get("expires_at"))
        return CapabilityLease(
            expires_at=expires_at,
            uses_remaining=uses_remaining,
        )

    @staticmethod
    def _positive_uses_remaining(value: Any, *, label: str) -> int | None:
        if value is None:
            return None
        if type(value) is not int or value < 1:
            raise ValidationError(f"{label} must be a positive integer")
        return value

    def _coerce_delegation(self, value: DelegationPolicy | dict[str, Any] | None) -> DelegationPolicy | None:
        if value is None:
            return None
        if isinstance(value, DelegationPolicy):
            data = {
                "delegable": value.delegable,
                "revocable": value.revocable,
                "max_delegation_depth": value.max_delegation_depth,
            }
        elif type(value) is dict:
            data = dict(value)
            if set(data) - {
                "delegable",
                "revocable",
                "max_delegation_depth",
            }:
                raise ValidationError(
                    "capability delegation policy contains unknown fields"
                )
        else:
            raise ValidationError("capability delegation policy must be a mapping")
        return DelegationPolicy(
            delegable=self._capability_boolean(
                data.get("delegable", False),
                field="delegable",
            ),
            revocable=self._capability_boolean(
                data.get("revocable", True),
                field="revocable",
            ),
            max_delegation_depth=self._max_delegation_depth(
                data.get("max_delegation_depth")
            ),
        )

    @staticmethod
    def _capability_boolean(value: Any, *, field: str) -> bool:
        if type(value) is not bool:
            raise ValidationError(
                f"capability delegation {field} must be a boolean"
            )
        return value

    @staticmethod
    def _max_delegation_depth(value: Any) -> int | None:
        if value is None:
            return None
        if type(value) is not int or value < 0:
            raise ValidationError(
                "max_delegation_depth must be a non-negative integer"
            )
        return value

    def _normalize_rights(self, rights: Iterable[str | CapabilityRight]) -> set[str]:
        if type(rights) not in {list, tuple, set, frozenset}:
            raise ValidationError(
                "capability rights must be a declared rights collection"
            )
        if any(not isinstance(right, str) for right in rights):
            raise ValidationError(
                "capability rights entries must be strings"
            )
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
            rules = self.rule_codec.coerce_many(constraints[AUTHORITY_RULES_KEY])
            for rule in rules:
                unknown = self._unknown_authority_rule_conditions(rule)
                if unknown:
                    raise ValidationError(
                        "capability authority rule contains unknown conditions: "
                        + ", ".join(unknown)
                    )
                malformed = self._malformed_authority_rule_conditions(rule)
                if malformed:
                    raise ValidationError(
                        "capability authority rule contains malformed conditions: "
                        + ", ".join(malformed)
                    )

    def _evaluate_constraints(self, cap: Capability, context: dict[str, Any]) -> dict[str, Any]:
        return self.evaluator.evaluate_constraints(cap, context)

    def _constraint_effect(self, constraint_results: dict[str, Any]) -> CapabilityEffect | None:
        return self.evaluator.constraint_effect(constraint_results)

    def _constraint_failure_is_scoped_miss(self, constraint_results: dict[str, Any]) -> bool:
        return self.evaluator.constraint_failure_is_scoped_miss(constraint_results)

    def _evaluate_authority_rules(self, rules: list[Any], context: dict[str, Any]) -> dict[str, Any]:
        return self.evaluator.evaluate_authority_rules(rules, context)

    def _unknown_authority_rule_conditions(self, rule: Any) -> list[str]:
        return self.evaluator.unknown_authority_rule_conditions(rule)

    def _malformed_authority_rule_conditions(self, rule: Any) -> list[str]:
        return self.evaluator.malformed_authority_rule_conditions(rule)

    def _authority_rule_matches(self, rule: Any, context: dict[str, Any]) -> bool:
        return self.evaluator.authority_rule_matches(rule, context)

    @staticmethod
    def _finite_nonnegative_timeout(value: Any) -> float | None:
        return CapabilityEvaluator.finite_nonnegative_timeout(value)

    def _argv_condition_matches(self, conditions: dict[str, Any], context: dict[str, Any]) -> bool:
        return self.evaluator.argv_condition_matches(conditions, context)

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
        return self.evaluator.context_dict(context)

    def _is_expired(self, cap: Capability) -> bool:
        return self.evaluator.is_expired(cap)

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
        return self.evaluator.expires_at_datetime(value)

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
            self.operations.expect("decision")
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

    def _bounded_capability_json(
        self,
        cap: Capability,
        *,
        max_bytes: int,
    ) -> dict[str, Any]:
        self._validate_presentation_byte_limit(max_bytes)
        projected = self._capability_json(cap)
        if self._presentation_json_size(projected) <= max_bytes:
            return projected

        projected["metadata_projection"] = self._omitted_projection(
            cap.metadata
        )
        projected["metadata"] = {}
        if self._presentation_json_size(projected) <= max_bytes:
            return projected

        projected["constraints_projection"] = self._omitted_projection(
            cap.constraints
        )
        projected["constraints"] = {}
        projected["rules"] = []
        if self._presentation_json_size(projected) <= max_bytes:
            return projected
        raise ValidationError(
            "capability presentation byte limit is too small for the authority identity"
        )

    @staticmethod
    def _validate_presentation_byte_limit(max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValidationError(
                "capability presentation byte limit must be a positive integer"
            )

    @staticmethod
    def _presentation_json_bytes(value: Any) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def _presentation_json_size(cls, value: Any) -> int:
        return len(cls._presentation_json_bytes(value))

    @classmethod
    def _omitted_projection(cls, value: Any) -> dict[str, Any]:
        payload = cls._presentation_json_bytes(value)
        return {
            "omitted": True,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def _attach_to_process(self, subject: str, cap_id: str) -> None:
        process = self.store.get_process(subject)
        if process is None:
            return
        self.store.append_process_capability_ids(subject, [cap_id])
