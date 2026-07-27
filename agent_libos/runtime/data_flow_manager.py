from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import math
import threading
from typing import Any, Callable, Iterable, Iterator, Mapping, TypeVar

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    AgentObject,
    CapabilityDecision,
    CapabilityRight,
    DataFlowContext,
    DataFlowDecision,
    DataFlowDirection,
    DataFlowOutcome,
    DataIntegrity,
    DataLabels,
    DataReleaseBinding,
    DataSensitivity,
    DataSink,
    DataSourceRef,
    EventType,
    FileLabelBinding,
    ObjectLifecycleState,
    ObjectRight,
    SinkTrustLevel,
    SinkTrustRule,
    SinkTrustSpec,
    integrity_rank,
    sensitivity_rank,
    sink_pattern_matches,
)
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, ValidationError
from agent_libos.ports import DataReleaseApprovalPort
from agent_libos.ports.blocking_work import run_blocking_once
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps, to_jsonable


_EMPTY_CONTEXT = DataFlowContext()
_TARGET_STATE_VERSION_UNRESOLVED = object()
_ThreadResultT = TypeVar("_ThreadResultT")


@dataclass
class _DirectoryLabelWatch:
    normalized_path: str
    changed: bool = False
    context: DataFlowContext = field(default_factory=DataFlowContext)


class DataFlowDenied(CapabilityDenied):
    """A denial carrying only the trusted, payload-free evidence to persist."""

    def __init__(self, decision: DataFlowDecision, sink: DataSink) -> None:
        self.data_flow_decision = decision
        self.data_flow_sink = sink
        super().__init__(
            f"data-flow denied egress to {sink.identity}: {decision.reason} "
            f"(decision_id={decision.decision_id})"
        )


class DataFlowManager:
    """Host-owned source-to-sink policy and evidence boundary.

    Callers may identify source Objects, but never supply trusted labels.  The
    manager resolves labels and versions from Object Memory, resolves Sink
    trust only from the durable Host registry, and returns an optional exact
    one-shot release capability decision for the protected-operation SDK to
    reserve atomically with ordinary authority.
    """

    REGISTRY_RESOURCE = "data_flow_sink_registry:*"
    RELEASE_RESOURCE_PREFIX = "data_release:"
    RELEASE_BINDING_KEY = "data_release_binding"
    FILE_BINDING_SOURCE_REF_PREFIX = "file_binding:"

    def __init__(
        self,
        store: Any,
        capabilities: Any,
        audit: Any,
        events: Any,
        authority_manifests: Any,
        objects: Any,
        *,
        memory: Any,
        config: AgentLibOSConfig | None = None,
        blocking_work_supervisor: Any | None = None,
    ) -> None:
        self.store = store
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.config = config or DEFAULT_CONFIG
        self.objects = objects
        # A process may host more than one Runtime sequentially or concurrently.
        # Keep ambient taint scoped to this manager so a source reference from a
        # closed Runtime can never leak into another store through ContextVar
        # inheritance.
        self._current_context: ContextVar[DataFlowContext] = ContextVar(
            f"agent_libos_data_flow_context_{id(self)}",
            default=_EMPTY_CONTEXT,
        )
        self._recovered_source_snapshot_access: ContextVar[bool] = ContextVar(
            f"agent_libos_recovered_source_snapshot_access_{id(self)}",
            default=False,
        )
        self.memory = memory
        self.human: DataReleaseApprovalPort | None = None
        self.authority_manifests = authority_manifests
        self._blocking_work_supervisor = blocking_work_supervisor
        self._directory_label_watch_lock = threading.Lock()
        self._directory_label_watches: dict[int, _DirectoryLabelWatch] = {}

    def bind_human(self, human: DataReleaseApprovalPort) -> None:
        """Complete the intentional DataFlow/Human construction cycle once."""

        if self.human is not None and self.human is not human:
            raise RuntimeError("DataFlowManager Human service is already bound")
        self.human = human

    def bootstrap_configured_rules(self) -> None:
        """Reconcile Host configuration with bootstrap-owned durable rules."""

        configured_rules = tuple(
            self._coerce_rule(configured)
            for configured in tuple(self.config.data_flow.sink_rules)
        )
        configured_patterns = {rule.pattern for rule in configured_rules}
        for active in tuple(self.store.list_sink_trust(active_only=True)):
            if (
                active.created_by == "runtime.bootstrap"
                and active.pattern not in configured_patterns
            ):
                self.unregister_sink_trust(
                    active.pattern,
                    actor="runtime.bootstrap",
                    require_capability=False,
                )

        for rule in configured_rules:
            active = self.store.inspect_sink_trust(rule.pattern)
            if active is not None and hmac.compare_digest(active.spec_hash, rule.spec_hash()):
                continue
            self.register_sink_trust(
                rule,
                actor="runtime.bootstrap",
                replace=active is not None,
                require_capability=False,
            )

    def register_sink_trust(
        self,
        spec: SinkTrustRule | Mapping[str, Any],
        *,
        actor: str,
        replace: bool = False,
        require_capability: bool = True,
    ) -> SinkTrustSpec:
        rule = self._coerce_rule(spec)
        authority = None
        if require_capability:
            authority = self.capabilities.require(
                actor,
                self.config.data_flow.registry_resource,
                CapabilityRight.ADMIN,
                {
                    "operation": "register_sink_trust",
                    "pattern": rule.pattern,
                    "replace": replace,
                    "spec_hash": rule.spec_hash(),
                },
                consume=False,
            )
        with self.capabilities.authority_transaction(
            [authority],
            actor=actor,
            operation="Sink trust register",
        ) as current_decisions:
            current_authority = current_decisions[0] if current_decisions else None
            active_rules = tuple(self.store.list_sink_trust(active_only=True))
            replacing_existing = any(item.pattern == rule.pattern for item in active_rules)
            if (
                not replacing_existing
                and len(active_rules) >= self.config.data_flow.registry_list_limit
            ):
                raise ValidationError(
                    "active Sink trust registry reached configured limit="
                    f"{self.config.data_flow.registry_list_limit}"
                )
            rule_base = rule.pattern[:-1] if rule.pattern.endswith("*") else rule.pattern
            for existing in active_rules:
                if replace and existing.pattern == rule.pattern:
                    continue
                existing_base = (
                    existing.pattern[:-1]
                    if existing.pattern.endswith("*")
                    else existing.pattern
                )
                if existing_base == rule_base:
                    raise ValidationError(
                        "equal-priority overlapping Sink trust patterns are forbidden: "
                        f"{existing.pattern!r} and {rule.pattern!r}"
                    )
            generation = int(self.store.get_sink_trust_generation()) + 1
            record = SinkTrustSpec(
                trust_id=new_id("sinktrust"),
                pattern=rule.pattern,
                trust_level=rule.trust_level,
                max_sensitivity=rule.max_sensitivity,
                tenants=rule.tenants,
                principals=rule.principals,
                identity_sha256=rule.identity_sha256,
                generation=generation,
                created_by=actor,
                created_at=utc_now(),
            )
            self.store.register_sink_trust(record, replace=replace)
            self.events.emit(
                EventType.SINK_TRUST_REGISTERED,
                source=actor,
                target=rule.pattern,
                payload={
                    "trust_id": record.trust_id,
                    "pattern": rule.pattern,
                    "trust_level": rule.trust_level.value,
                    "max_sensitivity": rule.max_sensitivity.value,
                    "spec_hash": record.spec_hash,
                    "generation": generation,
                    "replaced": replace,
                },
            )
            self.audit.record(
                actor=actor,
                action="data_flow.sink_trust.register",
                target=rule.pattern,
                capability_refs=(
                    [current_authority.selected_capability_id]
                    if current_authority is not None and current_authority.selected_capability_id
                    else []
                ),
                decision={
                    "trust_id": record.trust_id,
                    "trust_level": rule.trust_level.value,
                    "max_sensitivity": rule.max_sensitivity.value,
                    "tenants": list(rule.tenants),
                    "principals": list(rule.principals),
                    "identity_sha256": rule.identity_sha256,
                    "spec_hash": record.spec_hash,
                    "generation": generation,
                    "replaced": replace,
                },
            )
        return record

    def unregister_sink_trust(
        self,
        pattern: str,
        *,
        actor: str,
        require_capability: bool = True,
    ) -> SinkTrustSpec:
        selected = str(pattern).strip()
        authority = None
        if require_capability:
            authority = self.capabilities.require(
                actor,
                self.config.data_flow.registry_resource,
                CapabilityRight.ADMIN,
                {"operation": "unregister_sink_trust", "pattern": selected},
                consume=False,
            )
        with self.capabilities.authority_transaction(
            [authority],
            actor=actor,
            operation="Sink trust unregister",
        ):
            active = self.store.inspect_sink_trust(selected)
            if active is None:
                raise ValidationError(f"active Sink trust record not found: {selected}")
            generation = int(self.store.get_sink_trust_generation()) + 1
            now = utc_now()
            removed = self.store.unregister_sink_trust(
                selected,
                generation=generation,
                deactivated_at=now,
            )
            if not removed:
                raise ValidationError(f"active Sink trust record changed concurrently: {selected}")
            self.events.emit(
                EventType.SINK_TRUST_UNREGISTERED,
                source=actor,
                target=selected,
                payload={
                    "trust_id": active.trust_id,
                    "spec_hash": active.spec_hash,
                    "generation": generation,
                },
            )
            self.audit.record(
                actor=actor,
                action="data_flow.sink_trust.unregister",
                target=selected,
                decision={
                    "trust_id": active.trust_id,
                    "spec_hash": active.spec_hash,
                    "generation": generation,
                },
            )
        return active

    def inspect_sink_trust(self, pattern: str) -> SinkTrustSpec | None:
        return self.store.inspect_sink_trust(pattern)

    def list_sink_trust(
        self,
        *,
        active_only: bool = True,
        generation: int | None = None,
    ) -> tuple[SinkTrustSpec, ...]:
        return tuple(
            self.store.list_sink_trust(active_only=active_only, generation=generation)
        )

    def resolve_sink_trust(self, sink: DataSink) -> SinkTrustSpec | None:
        matches = [
            item
            for item in self.store.list_sink_trust(active_only=True)
            if sink_pattern_matches(item.pattern, sink.registry_identity)
        ]
        if not matches:
            return None
        matches.sort(key=lambda item: (len(item.pattern.rstrip("*")), item.generation), reverse=True)
        longest = len(matches[0].pattern.rstrip("*"))
        tied = [item for item in matches if len(item.pattern.rstrip("*")) == longest]
        if len(tied) > 1:
            raise ValidationError(
                f"conflicting equal-priority Sink trust records for {sink.registry_identity}"
            )
        return matches[0]

    def current_context(self) -> DataFlowContext:
        return self._current_context.get()

    @contextmanager
    def activate(self, context: DataFlowContext) -> Iterator[DataFlowContext]:
        token = self.push(context)
        try:
            yield context
        finally:
            self.reset(token)

    def push(self, context: DataFlowContext) -> Token[DataFlowContext]:
        if not isinstance(context, DataFlowContext):
            raise ValidationError("trusted data-flow context must use DataFlowContext")
        return self._current_context.set(context)

    def reset(self, token: Token[DataFlowContext]) -> None:
        self._current_context.reset(token)

    @contextmanager
    def recovered_source_snapshot_access(self) -> Iterator[None]:
        """Allow one Host-owned durable action to resume after store reopen."""

        token = self._recovered_source_snapshot_access.set(True)
        try:
            yield
        finally:
            self._recovered_source_snapshot_access.reset(token)

    def context_from_source_oids(
        self,
        pid: str,
        source_oids: Iterable[str] | None,
        *,
        include_current: bool = True,
    ) -> DataFlowContext:
        contexts: list[DataFlowContext] = [self.current_context()] if include_current else []
        selected = tuple(dict.fromkeys(str(item) for item in (source_oids or ())))
        for oid in selected:
            contexts.append(self._context_for_object(pid, oid))
        return DataFlowContext.aggregate(contexts)

    def context_from_trusted_source_oids(
        self,
        source_oids: Iterable[str] | None,
    ) -> DataFlowContext:
        """Resolve exact Object labels for Host-managed internal propagation.

        Unlike :meth:`context_from_source_oids`, this does not interpret a
        process as the reader.  It is reserved for runtime-owned handoffs such
        as ObjectTask notifications, where the Host must preserve the labels
        of an Object even when the receiving process is deliberately not
        granted authority to read that Object.
        """

        if isinstance(source_oids, (str, bytes)):
            raise ValidationError("trusted data-flow source_oids must be a collection")
        selected = tuple(
            dict.fromkeys(str(item or "").strip() for item in (source_oids or ()))
        )
        if any(not oid for oid in selected):
            raise ValidationError("trusted data-flow source_oids cannot contain empty Object ids")
        contexts = [
            self._context_for_object("runtime", oid, require_read=False)
            for oid in selected
        ]
        return DataFlowContext.aggregate(contexts)

    def context_from_object_snapshot(self, obj: AgentObject) -> DataFlowContext:
        """Freeze lineage for an already-authorized exact Object snapshot.

        This Host-only handoff deliberately performs no second authorization or
        Object-store lookup.  Callers must have obtained ``obj`` from an
        authorized Object Memory read and must pass the exact payload snapshot
        whose bytes they are about to move across another primitive boundary.
        """

        if not isinstance(obj, AgentObject):
            raise ValidationError("trusted data-flow Object snapshot must use AgentObject")
        if obj.lifecycle_state is not ObjectLifecycleState.LIVE:
            raise ValidationError("trusted data-flow Object snapshot must be live")
        return DataFlowContext(
            labels=DataLabels.from_object_metadata(obj.metadata),
            source_refs=(
                DataSourceRef(
                    obj.oid,
                    obj.version,
                    self._object_payload_sha256(obj.payload),
                ),
            ),
        )

    def context_from_materialization(self, pid: str, materialized: Any) -> DataFlowContext:
        contexts: list[DataFlowContext] = []
        for entry in tuple(getattr(materialized, "object_manifest", ()) or ()):
            if not isinstance(entry, Mapping) or entry.get("disposition") != "included":
                continue
            oid = str(entry.get("oid") or "")
            if not oid:
                raise ValidationError("materialized context contains an invalid Object reference")
            context = self._context_for_object(pid, oid, require_read=False)
            expected_version = entry.get("version")
            if expected_version != context.source_refs[0].version:
                raise ValidationError(f"materialized source version changed: {oid}")
            contexts.append(context)
        aggregated = DataFlowContext.aggregate(contexts)
        return DataFlowContext(
            labels=aggregated.labels,
            source_refs=aggregated.source_refs,
            materialization_id=str(getattr(materialized, "materialization_id", "") or "") or None,
        )

    def authorize_egress(
        self,
        *,
        pid: str,
        sink: DataSink,
        context: DataFlowContext | None,
        payload: Any,
        operation: str,
        target_state_version: str | int | None = None,
        request_release: bool = True,
        expected_registry_generation: int | None = None,
        current_target_state_version: object = _TARGET_STATE_VERSION_UNRESOLVED,
        allow_recovered_source_snapshots: bool = False,
        reserved_release_decision: CapabilityDecision | None = None,
        reserved_release_id: str | None = None,
        minimum_integrity: DataIntegrity | str = DataIntegrity.UNTRUSTED,
    ) -> tuple[DataFlowDecision, CapabilityDecision | None]:
        if (reserved_release_decision is None) != (reserved_release_id is None):
            raise ValidationError(
                "reserved data release revalidation requires both the decision and reservation id"
            )
        selected_context = context or self.current_context()
        payload_hash, payload_bytes = self._payload_digest(payload)
        generation = int(self.store.get_sink_trust_generation())
        if expected_registry_generation is not None and generation != expected_registry_generation:
            return self._deny(
                pid,
                sink,
                selected_context,
                payload_hash,
                generation,
                "Sink trust registry generation changed before dispatch",
            )
        source_error = self._validate_source_refs(
            selected_context.source_refs,
            allow_recovered_source_snapshots=allow_recovered_source_snapshots,
        )
        if source_error is not None:
            return self._deny(
                pid,
                sink,
                selected_context,
                payload_hash,
                generation,
                source_error,
            )
        trust = self.resolve_sink_trust(sink)
        policy_error = self._clearance_error(
            sink,
            selected_context.labels,
            trust,
            minimum_integrity=minimum_integrity,
        )
        if policy_error is not None:
            return self._deny(
                pid,
                sink,
                selected_context,
                payload_hash,
                generation,
                policy_error,
                trust=trust,
            )
        if current_target_state_version is not _TARGET_STATE_VERSION_UNRESOLVED and (
            type(current_target_state_version) is not type(target_state_version)
            or current_target_state_version != target_state_version
        ):
            return self._deny(
                pid,
                sink,
                selected_context,
                payload_hash,
                generation,
                "data-flow target state version changed before dispatch",
                trust=trust,
            )

        trust_level = trust.trust_level if trust is not None else SinkTrustLevel.UNTRUSTED
        needs_release = (
            trust_level is SinkTrustLevel.CONDITIONAL
            and sensitivity_rank(selected_context.labels.sensitivity)
            > sensitivity_rank(DataSensitivity.NORMAL)
        )
        release_decision: CapabilityDecision | None = None
        binding: DataReleaseBinding | None = None
        if needs_release:
            assert trust is not None
            binding = self.release_binding(
                pid=pid,
                sink=sink,
                trust=trust,
                context=selected_context,
                payload_hash=payload_hash,
                operation=operation,
                target_state_version=target_state_version,
            )
            release_decision = self._matching_release_decision(pid, binding)
            if release_decision is None and reserved_release_decision is not None:
                assert reserved_release_id is not None
                release_decision = self._matching_reserved_release_decision(
                    pid,
                    binding,
                    reserved_release_decision,
                    reserved_release_id,
                )
            if release_decision is None:
                decision = self._record_decision(
                    pid=pid,
                    sink=sink,
                    context=selected_context,
                    payload_hash=payload_hash,
                    generation=generation,
                    outcome=DataFlowOutcome.RELEASE_REQUIRED,
                    reason="conditional Sink requires an exact one-shot data release",
                    trust=trust,
                )
                if request_release:
                    request_id = self._request_release(
                        pid,
                        sink,
                        selected_context,
                        binding,
                        payload_bytes=payload_bytes,
                    )
                    raise HumanApprovalRequired(
                        request_id=request_id,
                        message=f"{pid} is waiting for a one-shot data release to {sink.identity}",
                    )
                raise CapabilityDenied(decision.reason)

        decision = self._record_decision(
            pid=pid,
            sink=sink,
            context=selected_context,
            payload_hash=payload_hash,
            generation=generation,
            outcome=DataFlowOutcome.ALLOW,
            reason=(
                "exact one-shot release matched conditional Sink"
                if release_decision is not None
                else "Sink clearance allows data sensitivity and identity"
            ),
            trust=trust,
            release_capability_id=(
                release_decision.selected_capability_id if release_decision is not None else None
            ),
        )
        return decision, release_decision

    def precheck_egress_clearance(
        self,
        *,
        pid: str,
        sink: DataSink,
        context: DataFlowContext | None,
        payload: Any,
        minimum_integrity: DataIntegrity | str = DataIntegrity.UNTRUSTED,
    ) -> DataFlowDecision:
        """Reject impossible egress before provider/profile resolution.

        A conditional high-sensitivity Sink is reported as release-required
        but does not create the Human request yet; the exact request is created
        later from the final provider payload by ``authorize_egress``.
        """

        selected_context = context or self.current_context()
        payload_hash, _ = self._payload_digest(payload)
        generation = int(self.store.get_sink_trust_generation())
        source_error = self._validate_source_refs(selected_context.source_refs)
        if source_error is not None:
            self._deny(
                pid,
                sink,
                selected_context,
                payload_hash,
                generation,
                source_error,
            )
        trust = self.resolve_sink_trust(sink)
        policy_error = self._clearance_error(
            sink,
            selected_context.labels,
            trust,
            minimum_integrity=minimum_integrity,
        )
        if policy_error is not None:
            self._deny(
                pid,
                sink,
                selected_context,
                payload_hash,
                generation,
                policy_error,
                trust=trust,
            )
        needs_release = bool(
            trust is not None
            and trust.trust_level is SinkTrustLevel.CONDITIONAL
            and sensitivity_rank(selected_context.labels.sensitivity)
            > sensitivity_rank(DataSensitivity.NORMAL)
        )
        return self._record_decision(
            pid=pid,
            sink=sink,
            context=selected_context,
            payload_hash=payload_hash,
            generation=generation,
            outcome=(
                DataFlowOutcome.RELEASE_REQUIRED if needs_release else DataFlowOutcome.ALLOW
            ),
            reason=(
                "conditional Sink requires exact final-payload release"
                if needs_release
                else "Sink clearance precheck passed"
            ),
            trust=trust,
        )

    def classify_egress_snapshot(
        self,
        *,
        sink: DataSink,
        context: DataFlowContext,
        allow_recovered_source_snapshots: bool = False,
        minimum_integrity: DataIntegrity | str = DataIntegrity.UNTRUSTED,
    ) -> DataFlowOutcome:
        """Classify current source and Sink state without emitting evidence.

        This read-only form is for Host guards that must not consume a release
        or mutate the decision ledger. Provider dispatch must still use
        :meth:`authorize_egress` through the protected-operation SDK.
        """

        with self.store.locked():
            if self._validate_source_refs(
                context.source_refs,
                allow_recovered_source_snapshots=allow_recovered_source_snapshots,
            ) is not None:
                return DataFlowOutcome.DENY
            try:
                trust = self.resolve_sink_trust(sink)
            except Exception:
                return DataFlowOutcome.DENY
            if self._clearance_error(
                sink,
                context.labels,
                trust,
                minimum_integrity=minimum_integrity,
            ) is not None:
                return DataFlowOutcome.DENY
            if (
                trust is not None
                and trust.trust_level is SinkTrustLevel.CONDITIONAL
                and sensitivity_rank(context.labels.sensitivity)
                > sensitivity_rank(DataSensitivity.NORMAL)
            ):
                return DataFlowOutcome.RELEASE_REQUIRED
            return DataFlowOutcome.ALLOW

    def reject_sink_identity_change(
        self,
        *,
        pid: str,
        sink: DataSink,
        context: DataFlowContext,
        payload: Any,
        reason: str = "Sink identity changed before provider dispatch",
    ) -> None:
        """Persist a payload-free denial for a late Host Sink identity change."""

        payload_hash, _payload_bytes = self._payload_digest(payload)
        generation = int(self.store.get_sink_trust_generation())
        try:
            trust = self.resolve_sink_trust(sink)
        except ValidationError:
            trust = None
        self._deny(
            pid,
            sink,
            context,
            payload_hash,
            generation,
            reason,
            trust=trust,
        )

    def release_binding(
        self,
        *,
        pid: str,
        sink: DataSink,
        trust: SinkTrustSpec,
        context: DataFlowContext,
        payload_hash: str,
        operation: str,
        target_state_version: str | int | None,
    ) -> DataReleaseBinding:
        manifest_hash = self._manifest_hash(pid)
        return DataReleaseBinding(
            sink=sink.identity,
            sink_identity_sha256=sink.registry_identity_sha256,
            trust_id=trust.trust_id,
            trust_hash=trust.spec_hash,
            registry_generation=int(self.store.get_sink_trust_generation()),
            manifest_hash=manifest_hash,
            labels_hash=context.labels.labels_hash(),
            source_refs_hash=context.source_refs_hash(),
            payload_hash=payload_hash,
            operation=operation,
            target_state_version=target_state_version,
        )

    def is_release_binding_current(
        self,
        *,
        pid: str,
        sink: DataSink,
        context: DataFlowContext,
        payload_hash: str,
        operation: str,
        target_state_version: str | int | None,
        binding: DataReleaseBinding | Mapping[str, Any],
        allow_recovered_source_snapshots: bool = False,
    ) -> bool:
        """Check an exact release binding against current Host-owned state.

        The check is deliberately read-only and payload-free: it validates the
        stored source references, current Sink clearance and registry record,
        authority manifest, and all remaining binding fields without recording
        a decision, consuming authority, or creating another Human request.
        """

        try:
            expected = DataReleaseBinding.normalize(binding)
        except (TypeError, ValueError):
            return False
        if self._validate_source_refs(
            context.source_refs,
            allow_recovered_source_snapshots=allow_recovered_source_snapshots,
        ) is not None:
            return False
        try:
            trust = self.resolve_sink_trust(sink)
        except Exception:
            return False
        if trust is None or self._clearance_error(sink, context.labels, trust) is not None:
            return False
        try:
            current = self.release_binding(
                pid=pid,
                sink=sink,
                trust=trust,
                context=context,
                payload_hash=payload_hash,
                operation=operation,
                target_state_version=target_state_version,
            ).to_dict()
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(dumps(expected), dumps(current))

    def bind_written_file(
        self,
        *,
        pid: str,
        normalized_path: str,
        content: bytes,
        context: DataFlowContext,
    ) -> FileLabelBinding:
        previous_generation = self.store.get_file_label_binding_generation(normalized_path)
        previous = self.store.get_file_label_binding(normalized_path)
        selected_context = (
            DataFlowContext.aggregate(
                [
                    DataFlowContext(
                        labels=previous.labels,
                        source_refs=previous.source_refs,
                    ),
                    context,
                ]
            )
            if previous is not None
            else context
        )
        binding = FileLabelBinding(
            binding_id=new_id("filelabel"),
            normalized_path=normalized_path,
            content_sha256=hashlib.sha256(content).hexdigest(),
            labels=selected_context.labels,
            source_refs=selected_context.source_refs,
            generation=previous_generation + 1,
            tombstoned=False,
            active=True,
            created_by=pid,
            created_at=utc_now(),
        )
        self.store.upsert_file_label_binding(binding)
        self._record_file_label_mutation(
            normalized_path,
            previous=previous,
            current=binding,
        )
        return binding

    def bind_written_file_digest(
        self,
        *,
        pid: str,
        normalized_path: str,
        content_sha256: str,
        context: DataFlowContext,
    ) -> FileLabelBinding:
        """Bind labels when a protected Host provider supplies an exact digest."""

        if (
            not isinstance(content_sha256, str)
            or len(content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in content_sha256)
        ):
            raise ValidationError("file content digest must be a lowercase SHA-256")
        previous_generation = self.store.get_file_label_binding_generation(normalized_path)
        previous = self.store.get_file_label_binding(normalized_path)
        selected_context = (
            DataFlowContext.aggregate(
                [
                    DataFlowContext(labels=previous.labels, source_refs=previous.source_refs),
                    context,
                ]
            )
            if previous is not None
            else context
        )
        binding = FileLabelBinding(
            binding_id=new_id("filelabel"),
            normalized_path=normalized_path,
            content_sha256=content_sha256,
            labels=selected_context.labels,
            source_refs=selected_context.source_refs,
            generation=previous_generation + 1,
            tombstoned=False,
            active=True,
            created_by=pid,
            created_at=utc_now(),
        )
        self.store.upsert_file_label_binding(binding)
        self._record_file_label_mutation(
            normalized_path,
            previous=previous,
            current=binding,
        )
        return binding

    def observe_ingress(self, context: DataFlowContext) -> DataFlowContext:
        """Conservatively add a trusted inbound source to the active tool flow."""

        combined = DataFlowContext.aggregate([self.current_context(), context])
        self._current_context.set(combined)
        return combined

    async def run_sync_in_worker(
        self,
        function: Callable[..., _ThreadResultT],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _ThreadResultT:
        """Run a sync primitive without losing worker ContextVar mutations."""

        def invoke() -> tuple[bool, _ThreadResultT | BaseException, DataFlowContext]:
            try:
                return True, function(*args, **kwargs), self.current_context()
            except BaseException as exc:
                return False, exc, self.current_context()

        supervisor = self._blocking_work_supervisor
        if supervisor is None:
            succeeded, value, returned_context = await run_blocking_once(invoke)
        else:
            succeeded, value, returned_context = await supervisor.run(invoke)
        self.observe_ingress(returned_context)
        if not succeeded:
            assert isinstance(value, BaseException)
            raise value
        return value

    def observe_unclassified_ingress(
        self,
        request_context: DataFlowContext,
        *,
        origin: str,
    ) -> DataFlowContext:
        return self.observe_ingress(
            self.unclassified_ingress_context(request_context, origin=origin)
        )

    @staticmethod
    def unclassified_ingress_context(
        request_context: DataFlowContext,
        *,
        origin: str,
    ) -> DataFlowContext:
        external = DataFlowContext(
            labels=DataLabels(
                sensitivity=DataSensitivity.NORMAL,
                trust_level="untrusted",
                integrity="untrusted",
                origin=origin,
            )
        )
        return DataFlowContext.aggregate((request_context, external))

    def file_context(self, normalized_path: str) -> DataFlowContext:
        context, _state_version = self.file_snapshot(normalized_path)
        return context

    def file_snapshot(
        self,
        normalized_path: str,
    ) -> tuple[DataFlowContext, str]:
        """Capture one active file-label binding and its exact generation."""

        context, state_version, _binding = self.file_deletion_snapshot(
            normalized_path
        )
        return context, state_version

    def file_deletion_snapshot(
        self,
        normalized_path: str,
    ) -> tuple[DataFlowContext, str, dict[str, tuple[str, int]]]:
        """Capture one label and the exact binding eligible for tombstoning."""

        binding = self.store.get_file_label_binding(normalized_path)
        if binding is None or binding.tombstoned:
            context = self.external_file_context()
        else:
            context = self._file_binding_context(binding)
        return (
            context,
            self._file_binding_state_version(normalized_path, binding),
            (
                {normalized_path: (binding.binding_id, binding.generation)}
                if binding is not None and not binding.tombstoned
                else {}
            ),
        )

    def directory_label_snapshot(
        self,
        normalized_path: str,
        child_paths: Iterable[str] = (),
    ) -> tuple[dict[str, DataFlowContext], str]:
        """Capture only the directory and bounded returned-child bindings.

        Directory listing callers already cap provider output.  Looking up the
        exact bounded path set preserves label fidelity without materializing an
        arbitrarily large label subtree before applying that cap.
        """

        if isinstance(child_paths, (str, bytes)):
            raise ValidationError("directory label child_paths must be a collection")
        selected_paths = tuple(
            dict.fromkeys(
                [str(normalized_path), *(str(path) for path in child_paths)]
            )
        )
        bindings = [
            binding
            for path in selected_paths
            if (
                binding := self.store.get_file_label_binding(path)
            ) is not None
            and not binding.tombstoned
        ]
        return (
            {
                item.normalized_path: self._file_binding_context(item)
                for item in bindings
            },
            self._file_tree_state_version(bindings),
        )

    @contextmanager
    def watch_directory_labels(
        self,
        normalized_path: str,
    ) -> Iterator[_DirectoryLabelWatch]:
        """Track bounded label mutations that overlap one active listing."""

        watch = _DirectoryLabelWatch(str(normalized_path).rstrip("/"))
        identity = id(watch)
        with self._directory_label_watch_lock:
            self._directory_label_watches[identity] = watch
        try:
            yield watch
        finally:
            with self._directory_label_watch_lock:
                self._directory_label_watches.pop(identity, None)

    def directory_label_state_version(
        self,
        normalized_path: str,
        child_paths: Iterable[str] = (),
    ) -> str:
        _contexts, state_version = self.directory_label_snapshot(
            normalized_path,
            child_paths,
        )
        return state_version

    @staticmethod
    def external_file_context() -> DataFlowContext:
        return DataFlowContext(
            labels=DataLabels(
                sensitivity=DataSensitivity.NORMAL,
                trust_level="untrusted",
                integrity="untrusted",
                origin="external-filesystem",
            )
        )

    def file_state_version(self, normalized_path: str) -> str:
        binding = self.store.get_file_label_binding(normalized_path)
        return self._file_binding_state_version(normalized_path, binding)

    @staticmethod
    def _file_binding_state_version(
        normalized_path: str,
        binding: FileLabelBinding | None,
    ) -> str:
        material = {
            "path": normalized_path,
            "binding_id": binding.binding_id if binding is not None else None,
            "generation": binding.generation if binding is not None else None,
            "content_sha256": binding.content_sha256 if binding is not None else None,
        }
        return hashlib.sha256(dumps(material).encode("utf-8")).hexdigest()

    def file_tree_context(self, normalized_path: str) -> DataFlowContext:
        """Aggregate every active label at or below a filesystem path."""

        context, _state_version = self.file_tree_snapshot(normalized_path)
        return context

    def file_tree_snapshot(
        self,
        normalized_path: str,
    ) -> tuple[DataFlowContext, str]:
        """Read subtree labels and their state fingerprint from one snapshot."""

        context, state_version, _bindings = self.file_tree_deletion_snapshot(
            normalized_path
        )
        return context, state_version

    def file_tree_deletion_snapshot(
        self,
        normalized_path: str,
    ) -> tuple[DataFlowContext, str, dict[str, tuple[str, int]]]:
        """Capture subtree labels and exact bindings eligible for delete settlement."""

        selected = str(normalized_path).rstrip("/")
        bindings = self.store.list_file_label_bindings_for_tree(selected)
        contexts = [
            self._file_binding_context(item)
            for item in bindings
        ]
        if not any(item.normalized_path == selected for item in bindings):
            contexts.append(
                DataFlowContext(
                    labels=DataLabels(
                        sensitivity=DataSensitivity.NORMAL,
                        trust_level="untrusted",
                        integrity="untrusted",
                        origin="external-filesystem",
                    )
                )
            )
        return (
            DataFlowContext.aggregate(contexts),
            self._file_tree_state_version(bindings),
            {
                item.normalized_path: (item.binding_id, item.generation)
                for item in bindings
            },
        )

    def _file_binding_context(self, binding: FileLabelBinding) -> DataFlowContext:
        """Project a durable active file binding without reviving Object refs."""

        if binding.content_sha256 is None or binding.tombstoned or not binding.active:
            raise ValidationError(
                f"active file label binding is invalid: {binding.binding_id}"
            )
        return DataFlowContext(
            labels=binding.labels,
            source_refs=(
                DataSourceRef(
                    oid=f"{self.FILE_BINDING_SOURCE_REF_PREFIX}{binding.binding_id}",
                    version=binding.generation,
                    content_sha256=binding.content_sha256,
                ),
            ),
        )

    def provenance_sources(
        self,
        context: DataFlowContext,
        *,
        exclude_oids: Iterable[str] = (),
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Split flow refs into Object parents and durable non-Object sources.

        File binding refs are valid data-flow sources but are not Object OIDs.
        Preserve them as provenance ``source_refs`` and recursively recover the
        immutable binding's Object ancestry for provenance consumers.
        """

        if not isinstance(context, DataFlowContext):
            raise ValidationError("provenance context must use DataFlowContext")
        if isinstance(exclude_oids, (str, bytes)):
            raise ValidationError("provenance exclude_oids must be a collection")
        excluded = {
            str(oid).strip()
            for oid in exclude_oids
            if str(oid).strip()
        }
        parent_oids: dict[str, None] = {}
        durable_refs: dict[str, None] = {}
        visited: set[tuple[str, int, str]] = set()

        def collect(ref: DataSourceRef) -> None:
            key = (ref.oid, ref.version, ref.content_sha256)
            if key in visited:
                return
            visited.add(key)
            if not ref.oid.startswith(self.FILE_BINDING_SOURCE_REF_PREFIX):
                if ref.oid not in excluded:
                    parent_oids.setdefault(ref.oid, None)
                return
            durable_refs.setdefault(ref.oid, None)
            binding_id = ref.oid.removeprefix(self.FILE_BINDING_SOURCE_REF_PREFIX)
            binding = (
                self.store.get_file_label_binding_by_id(binding_id)
                if binding_id
                else None
            )
            if (
                binding is None
                or binding.tombstoned
                or binding.generation != ref.version
                or binding.content_sha256 is None
                or not hmac.compare_digest(
                    binding.content_sha256,
                    ref.content_sha256,
                )
            ):
                return
            for source_ref in binding.source_refs:
                collect(source_ref)

        for source_ref in context.source_refs:
            collect(source_ref)
        return tuple(parent_oids), tuple(durable_refs)

    def file_tree_state_version(self, normalized_path: str) -> str:
        """Return a stable version for all active labels in a path subtree."""

        return self._file_tree_state_version(
            self.store.list_file_label_bindings_for_tree(normalized_path)
        )

    @staticmethod
    def _file_tree_state_version(bindings: Iterable[FileLabelBinding]) -> str:
        material = [
            {
                "path": item.normalized_path,
                "binding_id": item.binding_id,
                "generation": item.generation,
                "content_sha256": item.content_sha256,
            }
            for item in bindings
        ]
        return hashlib.sha256(dumps(material).encode("utf-8")).hexdigest()

    def tombstone_file(
        self,
        *,
        pid: str,
        normalized_path: str,
        expected_binding_id: str | None = None,
        expected_generation: int | None = None,
    ) -> None:
        previous = self.store.get_file_label_binding(normalized_path)
        self.store.tombstone_file_label_binding(
            normalized_path,
            binding_id=new_id("filelabel"),
            created_by=pid,
            created_at=utc_now(),
            expected_binding_id=expected_binding_id,
            expected_generation=expected_generation,
        )
        self._record_file_label_mutation(
            normalized_path,
            previous=previous,
            current=None,
        )

    def _record_file_label_mutation(
        self,
        normalized_path: str,
        *,
        previous: FileLabelBinding | None,
        current: FileLabelBinding | None,
    ) -> None:
        contexts = [
            self._file_binding_context(binding)
            for binding in (previous, current)
            if binding is not None and not binding.tombstoned and binding.active
        ]
        if not contexts:
            contexts = [self.external_file_context()]
        mutation_labels = DataFlowContext.aggregate(contexts).labels
        selected_path = str(normalized_path).rstrip("/")
        with self._directory_label_watch_lock:
            for watch in self._directory_label_watches.values():
                prefix = f"{watch.normalized_path}/" if watch.normalized_path else ""
                if watch.normalized_path and not (
                    selected_path == watch.normalized_path
                    or selected_path.startswith(prefix)
                ):
                    continue
                combined = DataFlowContext.aggregate(
                    (
                        watch.context,
                        DataFlowContext(labels=mutation_labels),
                    )
                )
                # Mutation taint needs labels only; exact current bindings below
                # retain bounded durable refs for returned entries.
                watch.context = DataFlowContext(labels=combined.labels)
                watch.changed = True

    def tombstone_path_tree(
        self,
        *,
        pid: str,
        expected_bindings: Mapping[str, tuple[str, int]],
    ) -> None:
        expected = dict(expected_bindings)
        with self.store.transaction():
            for path in sorted(expected):
                binding_id, generation = expected[path]
                self.tombstone_file(
                    pid=pid,
                    normalized_path=path,
                    expected_binding_id=binding_id,
                    expected_generation=generation,
                )

    def _context_for_object(
        self,
        pid: str,
        oid: str,
        *,
        require_read: bool = True,
    ) -> DataFlowContext:
        if not oid:
            raise ValidationError("data-flow source Objects require a non-empty Object id")
        if require_read:
            self.capabilities.require(
                pid,
                f"object:{oid}",
                ObjectRight.READ,
                {"operation": "data_flow_source", "oid": oid},
                consume=False,
            )
        obj = self.objects.get_object(oid)
        if obj is None:
            raise ValidationError(f"data-flow source Object not found: {oid}")
        return self.context_from_object_snapshot(obj)

    def _validate_source_refs(
        self,
        refs: Iterable[DataSourceRef],
        *,
        allow_recovered_source_snapshots: bool = False,
    ) -> str | None:
        allow_recovered_source_snapshots = (
            allow_recovered_source_snapshots
            or self._recovered_source_snapshot_access.get()
        )
        for ref in refs:
            if ref.oid.startswith(self.FILE_BINDING_SOURCE_REF_PREFIX):
                binding_id = ref.oid.removeprefix(
                    self.FILE_BINDING_SOURCE_REF_PREFIX
                )
                if not binding_id:
                    return "data-flow source file binding reference is malformed"
                binding = self.store.get_file_label_binding_by_id(binding_id)
                if binding is None:
                    return (
                        "data-flow source file binding is unavailable: "
                        f"{binding_id}"
                    )
                if (
                    binding.tombstoned
                    or binding.generation != ref.version
                    or binding.content_sha256 is None
                    or not hmac.compare_digest(
                        binding.content_sha256,
                        ref.content_sha256,
                    )
                ):
                    return (
                        "data-flow source file binding changed before dispatch: "
                        f"{binding_id}"
                    )
                continue
            obj = self.objects.get_object(ref.oid)
            if obj is None:
                state = self.objects.get_persisted_object_state(ref.oid)
                if state is None:
                    return f"data-flow source Object disappeared: {ref.oid}"
                if (
                    state.lifecycle_state is not ObjectLifecycleState.RELEASED
                    or not allow_recovered_source_snapshots
                    or not state.recovered_after_reopen
                ):
                    return f"data-flow source Object is no longer live: {ref.oid}"
                # Object payloads are intentionally runtime-local. A durable
                # pending Human/LLM action therefore reopens with only the
                # Host-written source snapshot and an explicitly recovery-marked
                # released Object row. Ordinary same-runtime checks do not opt
                # into this narrow resume path.
                if state.version != ref.version:
                    return f"data-flow source Object changed before dispatch: {ref.oid}"
                if state.payload_present:
                    return f"data-flow source Object payload is unavailable: {ref.oid}"
                continue
            actual_hash = self._object_payload_sha256(obj.payload)
            if obj.version != ref.version or not hmac.compare_digest(actual_hash, ref.content_sha256):
                return f"data-flow source Object changed before dispatch: {ref.oid}"
        return None

    @classmethod
    def _object_payload_sha256(cls, payload: Any) -> str:
        """Hash a stable finite JSON projection, including legacy NaN rows."""

        projected = cls._finite_json_projection(to_jsonable(payload))
        return hashlib.sha256(dumps(projected).encode("utf-8")).hexdigest()

    @classmethod
    def _finite_json_projection(cls, value: Any) -> Any:
        if type(value) is float and not math.isfinite(value):
            if math.isnan(value):
                label = "NaN"
            else:
                label = "Infinity" if value > 0 else "-Infinity"
            return {"_non_finite_number": label}
        if isinstance(value, dict):
            return {
                key: cls._finite_json_projection(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._finite_json_projection(item) for item in value]
        return value

    def _clearance_error(
        self,
        sink: DataSink,
        labels: DataLabels,
        trust: SinkTrustSpec | None,
        *,
        minimum_integrity: DataIntegrity | str = DataIntegrity.UNTRUSTED,
    ) -> str | None:
        selected_minimum_integrity = DataIntegrity(minimum_integrity)
        if integrity_rank(labels.integrity) < integrity_rank(
            selected_minimum_integrity
        ):
            return (
                f"data integrity {labels.integrity.value} is below operation minimum "
                f"{selected_minimum_integrity.value}"
            )
        if labels.is_mixed_identity:
            return "mixed tenant/principal data must be reclassified by the Host"
        max_sensitivity = (
            trust.max_sensitivity
            if trust is not None
            else DataSensitivity(self.config.data_flow.default_max_sensitivity)
        )
        trust_level = trust.trust_level if trust is not None else SinkTrustLevel.UNTRUSTED
        if trust_level is SinkTrustLevel.UNTRUSTED:
            max_sensitivity = min(
                (max_sensitivity, DataSensitivity.NORMAL),
                key=sensitivity_rank,
            )
        if sensitivity_rank(labels.sensitivity) > sensitivity_rank(max_sensitivity):
            return (
                f"data sensitivity {labels.sensitivity.value} exceeds Sink maximum "
                f"{max_sensitivity.value}"
            )
        if trust is not None and trust.identity_sha256 is not None:
            if sink.registry_identity_sha256 is None or not hmac.compare_digest(
                sink.registry_identity_sha256,
                trust.identity_sha256,
            ):
                return "Sink configuration identity hash does not match Host trust record"
        if labels.tenant is not None and (
            trust is None or labels.tenant not in set(trust.tenants)
        ):
            return f"tenant {labels.tenant!r} is outside Sink clearance"
        if labels.principal is not None and (
            trust is None or labels.principal not in set(trust.principals)
        ):
            return f"principal {labels.principal!r} is outside Sink clearance"
        return None

    def _matching_release_decision(
        self,
        pid: str,
        binding: DataReleaseBinding,
    ) -> CapabilityDecision | None:
        resource = f"{self.RELEASE_RESOURCE_PREFIX}{binding.sink}"
        context = {self.RELEASE_BINDING_KEY: binding.to_dict()}
        decision = self.capabilities.authorize(
            pid,
            resource,
            CapabilityRight.APPROVE,
            context,
            audit=True,
        )
        if not decision.allowed or decision.selected_capability_id is None:
            return None
        cap = self.store.get_capability(decision.selected_capability_id)
        if cap is None or cap.uses_remaining != 1:
            return None
        raw_binding = cap.constraints.get(self.RELEASE_BINDING_KEY)
        try:
            normalized = DataReleaseBinding.normalize(raw_binding)
        except (TypeError, ValueError):
            return None
        if not hmac.compare_digest(dumps(normalized), dumps(binding.to_dict())):
            return None
        if decision.consume_capability_id is None:
            return None
        return decision

    def _matching_reserved_release_decision(
        self,
        pid: str,
        binding: DataReleaseBinding,
        decision: CapabilityDecision,
        reservation_id: str,
    ) -> CapabilityDecision | None:
        resource = f"{self.RELEASE_RESOURCE_PREFIX}{binding.sink}"
        cap_id = decision.consume_capability_id
        if (
            not decision.allowed
            or decision.subject != pid
            or decision.resource != resource
            or decision.right != CapabilityRight.APPROVE.value
            or cap_id is None
            or decision.selected_capability_id != cap_id
        ):
            return None
        reservation = self.store.get_capability_use_reservation(reservation_id)
        reservation_count = (
            reservation.get("count") if reservation is not None else None
        )
        if (
            reservation is None
            or reservation.get("status") not in {"reserved", "committed"}
            or str(reservation.get("cap_id") or "") != str(cap_id)
            or type(reservation_count) is not int
            or reservation_count != 1
        ):
            return None
        raw_binding = decision.context.get(self.RELEASE_BINDING_KEY)
        try:
            normalized = DataReleaseBinding.normalize(raw_binding)
        except (TypeError, ValueError):
            return None
        if not hmac.compare_digest(dumps(normalized), dumps(binding.to_dict())):
            return None
        return decision

    def _request_release(
        self,
        pid: str,
        sink: DataSink,
        context: DataFlowContext,
        binding: DataReleaseBinding,
        *,
        payload_bytes: int,
    ) -> str:
        if self.human is None:
            raise CapabilityDenied(
                f"{pid} requires a one-shot release for {sink.identity}, but no Human manager is bound"
            )
        payload = {
            "type": "data_release_approval",
            "question": f"Release this labeled payload to {sink.identity}?",
            "requested_once_capability": {
                "subject": pid,
                "resource": f"{self.RELEASE_RESOURCE_PREFIX}{sink.identity}",
                "rights": [CapabilityRight.APPROVE.value],
                "constraints": {self.RELEASE_BINDING_KEY: binding.to_dict()},
            },
            "context": {
                "sink": sink.identity,
                "sink_identity_sha256": sink.identity_sha256,
                "sensitivity": context.labels.sensitivity.value,
                "tenant": context.labels.tenant,
                "principal": context.labels.principal,
                "payload_bytes": payload_bytes,
                "payload_sha256": binding.payload_hash,
                "labels_sha256": binding.labels_hash,
                "source_refs_sha256": binding.source_refs_hash,
                "source_count": len(context.source_refs),
                "trust_id": binding.trust_id,
                "trust_sha256": binding.trust_hash,
                "registry_generation": binding.registry_generation,
                "manifest_sha256": binding.manifest_hash,
                "operation": binding.operation,
            },
        }
        return str(
            self.human.request_data_release(
                pid=pid,
                human=self.config.runtime.default_human,
                request=payload,
                blocking=True,
            )
        )

    def _record_decision(
        self,
        *,
        pid: str,
        sink: DataSink,
        context: DataFlowContext,
        payload_hash: str,
        generation: int,
        outcome: DataFlowOutcome,
        reason: str,
        trust: SinkTrustSpec | None,
        release_capability_id: str | None = None,
    ) -> DataFlowDecision:
        decision = DataFlowDecision(
            decision_id=new_id("dfd"),
            pid=pid,
            sink=sink.identity,
            direction=DataFlowDirection.EGRESS,
            outcome=outcome,
            reason=reason,
            labels=context.labels,
            source_refs=context.source_refs,
            payload_hash=payload_hash,
            registry_generation=generation,
            created_at=utc_now(),
            trust_id=trust.trust_id if trust is not None else None,
            trust_hash=trust.spec_hash if trust is not None else None,
            release_capability_id=release_capability_id,
        )
        self._persist_decision(decision=decision, sink=sink)
        return decision

    def persist_denied_decision(
        self,
        *,
        decision: DataFlowDecision,
        sink: DataSink,
    ) -> DataFlowDecision:
        """Persist an exact denial after a surrounding transaction rolled back.

        The protected-operation SDK calls this only for the structured denial
        raised by this manager.  Keeping the original decision id lets the
        caller-facing error, Audit, Event, and append-only decision row refer
        to the same payload-free evidence.
        """

        if decision.outcome is not DataFlowOutcome.DENY:
            raise ValidationError("only data-flow denials may be re-persisted")
        return self._persist_decision(decision=decision, sink=sink)

    def _persist_decision(
        self,
        *,
        decision: DataFlowDecision,
        sink: DataSink,
    ) -> DataFlowDecision:
        if decision.sink != sink.identity:
            raise ValidationError("data-flow decision Sink does not match evidence Sink")
        existing = self.store.get_data_flow_decision(decision.decision_id)
        if existing is not None:
            if existing != decision:
                raise ValidationError(
                    f"data-flow decision id conflict: {decision.decision_id}"
                )
            return existing
        context = DataFlowContext(
            labels=decision.labels,
            source_refs=decision.source_refs,
        )
        evidence = {
            "decision_id": decision.decision_id,
            "direction": decision.direction.value,
            "outcome": decision.outcome.value,
            "reason": decision.reason,
            "sink": sink.identity,
            "sink_identity_sha256": sink.identity_sha256,
            "sink_trust_identity": sink.registry_identity,
            "sink_trust_identity_sha256": sink.registry_identity_sha256,
            "labels": decision.labels.to_dict(),
            "labels_sha256": decision.labels.labels_hash(),
            "source_refs": [item.to_dict() for item in decision.source_refs],
            "source_refs_sha256": context.source_refs_hash(),
            "payload_sha256": decision.payload_hash,
            "trust_id": decision.trust_id,
            "trust_sha256": decision.trust_hash,
            "registry_generation": decision.registry_generation,
            "release_capability_id": decision.release_capability_id,
        }
        with self.store.transaction():
            existing = self.store.get_data_flow_decision(decision.decision_id)
            if existing is not None:
                if existing != decision:
                    raise ValidationError(
                        f"data-flow decision id conflict: {decision.decision_id}"
                    )
                return existing
            self.store.insert_data_flow_decision(decision)
            self.events.emit(
                EventType.DATA_FLOW_DECISION,
                source=decision.pid,
                target=f"data_flow_sink:{sink.identity}",
                payload=evidence,
            )
            self.audit.record(
                actor=decision.pid,
                action="data_flow.egress",
                target=sink.identity,
                input_refs=[item.oid for item in decision.source_refs],
                capability_refs=(
                    [decision.release_capability_id]
                    if decision.release_capability_id
                    else []
                ),
                decision=evidence,
            )
        return decision

    def _deny(
        self,
        pid: str,
        sink: DataSink,
        context: DataFlowContext,
        payload_hash: str,
        generation: int,
        reason: str,
        *,
        trust: SinkTrustSpec | None = None,
    ) -> tuple[DataFlowDecision, None]:
        decision = self._record_decision(
            pid=pid,
            sink=sink,
            context=context,
            payload_hash=payload_hash,
            generation=generation,
            outcome=DataFlowOutcome.DENY,
            reason=reason,
            trust=trust,
        )
        raise DataFlowDenied(decision, sink)

    def _manifest_hash(self, pid: str) -> str:
        manifest = self.authority_manifests.get_for_process(pid)
        if manifest is not None and getattr(manifest, "manifest_hash", None):
            return str(manifest.manifest_hash)
        return hashlib.sha256(f"no-authority-manifest:{pid}".encode("utf-8")).hexdigest()

    @staticmethod
    def _payload_digest(payload: Any) -> tuple[str, int]:
        if isinstance(payload, bytes):
            raw = payload
        elif isinstance(payload, bytearray):
            raw = bytes(payload)
        else:
            raw = dumps(to_jsonable(payload)).encode("utf-8")
        return hashlib.sha256(raw).hexdigest(), len(raw)

    @staticmethod
    def _coerce_rule(value: SinkTrustRule | Mapping[str, Any] | Any) -> SinkTrustRule:
        if isinstance(value, SinkTrustRule):
            return value
        if not isinstance(value, Mapping):
            try:
                value = asdict(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError("Sink trust spec must be a SinkTrustRule or object") from exc
        try:
            return SinkTrustRule(**dict(value))
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
