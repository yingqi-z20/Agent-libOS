from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import StrEnum
import inspect
import sys
from typing import Any, Awaitable, Callable, Iterable, Mapping, TypeVar

from agent_libos.models import (
    AuditRecord,
    CapabilityDecision,
    DataFlowContext,
    DataFlowDecision,
    DataFlowDirection,
    DataIntegrity,
    DataSink,
    Event,
    EventPriority,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRecoveryQuery,
    ExternalEffectRecoverySummary,
    ExternalEffectRecord,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    OperationKind,
    ResourceUsage,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.evidence.external_effects import (
    abandon_external_effect_intent,
    classify_external_effect,
    iter_external_effect_recovery,
    mark_external_effect_dispatched,
    prepare_external_effect_intent,
    record_external_effect,
    require_external_effect_classifier,
)
from agent_libos.substrate import ProviderEffectNotStarted
from agent_libos.ports import EffectAuthorityPort, ProtectedEffectPort
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import to_jsonable


class AuthorityMode(StrEnum):
    CAPABILITY = "capability"
    RUNTIME_INTERNAL = "runtime_internal"


class PostProviderFailureMode(StrEnum):
    PROPAGATE = "propagate"
    PRESERVE_RESULT = "preserve_result"


class ResourcePolicy(StrEnum):
    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


_EnumT = TypeVar("_EnumT")


def _validated_enum(label: str, value: Any, enum_type: type[_EnumT]) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"protected operation {label} must be a valid {enum_type.__name__}"
        ) from exc


def _require_bool(label: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"protected operation {label} must be boolean")


def _require_bool_fields(value: Any, labels: tuple[str, ...]) -> None:
    for label in labels:
        _require_bool(label, getattr(value, label))


def _validate_evidence_roles(value: Any) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(not isinstance(role, str) for role in value)
        or set(value) != {"audit", "effect", "event"}
    ):
        raise ValueError(
            "protected operation evidence_roles must be a duplicate-free tuple "
            "containing audit, event, and effect"
        )


def _normalize_contract_enums(contract: Any) -> None:
    for label, enum_type in (
        ("resource_policy", ResourcePolicy),
        ("authority_mode", AuthorityMode),
        ("data_flow_direction", DataFlowDirection),
        ("minimum_egress_integrity", DataIntegrity),
        ("post_provider_failure_mode", PostProviderFailureMode),
        ("classifier_failure_rollback_class", ExternalEffectRollbackClass),
        ("classifier_failure_rollback_status", ExternalEffectRollbackStatus),
    ):
        object.__setattr__(
            contract,
            label,
            _validated_enum(label, getattr(contract, label), enum_type),
        )


def _normalize_optional_policy_name(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} policy names must be non-empty strings")
    return value.strip()


def _validate_contract_data_flow(contract: Any) -> None:
    if (
        contract.data_flow_direction is not DataFlowDirection.NONE
        and not contract.information_flow
    ):
        raise ValueError("data-flow directions require information_flow=True")
    if (
        contract.minimum_egress_integrity is not DataIntegrity.UNTRUSTED
        and contract.data_flow_direction
        not in {DataFlowDirection.EGRESS, DataFlowDirection.BIDIRECTIONAL}
    ):
        raise ValueError(
            "minimum egress integrity requires an egress data-flow direction"
        )


@dataclass(frozen=True)
class ProtectedOperationContract:
    name: str
    provider: str
    operation: str
    evidence_roles: tuple[str, ...]
    resource_policy: ResourcePolicy
    authority_mode: AuthorityMode = AuthorityMode.CAPABILITY
    state_mutation: bool = False
    information_flow: bool = False
    data_flow_direction: DataFlowDirection = DataFlowDirection.NONE
    minimum_egress_integrity: DataIntegrity = DataIntegrity.UNTRUSTED
    post_provider_failure_mode: PostProviderFailureMode = PostProviderFailureMode.PROPAGATE
    internal_reason: str | None = None
    require_classifier: bool = True
    preflight_classifier: bool = False
    classifier_failure_rollback_class: ExternalEffectRollbackClass = ExternalEffectRollbackClass.UNKNOWN
    classifier_failure_rollback_status: ExternalEffectRollbackStatus = ExternalEffectRollbackStatus.UNKNOWN
    classifier_failure_label: str = "post_effect_failure"
    prepared_recovery: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.provider or not self.operation:
            raise ValueError("protected operation contract names must be non-empty")
        _validate_evidence_roles(self.evidence_roles)
        _require_bool_fields(
            self,
            (
                "state_mutation",
                "information_flow",
                "require_classifier",
                "preflight_classifier",
            ),
        )
        _normalize_contract_enums(self)
        if self.authority_mode == AuthorityMode.RUNTIME_INTERNAL and not str(
            self.internal_reason or ""
        ).strip():
            raise ValueError("runtime-internal protected operations require an explicit reason")
        object.__setattr__(
            self,
            "prepared_recovery",
            _normalize_optional_policy_name(
                self.prepared_recovery,
                label="prepared recovery",
            ),
        )
        _validate_contract_data_flow(self)


@dataclass(frozen=True)
class ProviderPhase:
    name: str
    state_mutation: bool = False
    information_flow: bool = False
    commits_authority: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("provider phase name must be non-empty")
        _require_bool_fields(
            self,
            ("state_mutation", "information_flow", "commits_authority"),
        )


@dataclass(frozen=True)
class ResourceSettlement:
    usage: ResourceUsage | Mapping[str, Any]
    source: str
    context: Mapping[str, Any] = field(default_factory=dict)
    allow_overage: bool = True
    kill_on_exceed: bool = True
    charge_reserved_maximum: bool = False

    def __post_init__(self) -> None:
        _require_bool_fields(
            self,
            ("allow_overage", "kill_on_exceed", "charge_reserved_maximum"),
        )


@dataclass(frozen=True)
class ProviderEffectNotStartedResult:
    """A structured provider result backed by a not-started certificate.

    Some primitives preserve their public result type instead of propagating
    :class:`ProviderEffectNotStarted`.  Returning this marker from the provider
    callable lets the SDK settle the phase as not-started without recording the
    current phase as an observed mutation.
    """

    error: ProviderEffectNotStarted
    result: Any
    outcome: str = "partial_not_started_after_prior_provider_effect"

    def __post_init__(self) -> None:
        if not isinstance(self.error, ProviderEffectNotStarted):
            raise TypeError("not-started result requires ProviderEffectNotStarted")
        if not self.outcome:
            raise ValueError("not-started result outcome must be non-empty")


@dataclass(frozen=True)
class ProtectedOperationEvidence:
    event_type: EventType | str
    event_source: str
    audit_action: str
    audit_actor: str
    event_target: str | None = None
    event_payload: Mapping[str, Any] = field(default_factory=dict)
    event_priority: EventPriority | str = EventPriority.NORMAL
    audit_target: str | None = None
    audit_decision: Mapping[str, Any] = field(default_factory=dict)
    input_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()
    capability_refs: tuple[str, ...] = ()
    correlation_id: str | None = None
    parent_record_id: str | None = None
    effect_metadata: Mapping[str, Any] = field(default_factory=dict)
    provider_receipt: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_type",
            _validated_enum("event_type", self.event_type, EventType),
        )
        object.__setattr__(
            self,
            "event_priority",
            _validated_enum("event_priority", self.event_priority, EventPriority),
        )


Hook = Callable[[], None]
FailureEvidenceFactory = Callable[[BaseException, str], ProtectedOperationEvidence]
PreparedRecoveryHandler = Callable[[Any], None]
FailureResourceFactory = Callable[[BaseException, str], ResourceSettlement | None]
FailureSettlementHandler = Callable[[BaseException, str], None]
AuthorityRevalidator = Callable[[], Iterable[CapabilityDecision]]
DataSinkRevalidator = Callable[[], DataSink]
TargetStateVersionResolver = Callable[[], str | int | None]
_DATA_FLOW_PAYLOAD_UNSET = object()


@dataclass(frozen=True)
class ProviderRegistryBinding:
    """Immutable provider registry identity captured before authorization."""

    registry_spec_sha256: str
    registry_generation: int

    @classmethod
    def from_context(cls, value: Mapping[str, Any]) -> "ProviderRegistryBinding":
        if not isinstance(value, Mapping):
            raise ValueError("provider registry binding context must be a mapping")
        return cls(
            registry_spec_sha256=value.get("registry_spec_sha256"),
            registry_generation=value.get("registry_generation"),
        )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.registry_spec_sha256, str)
            or len(self.registry_spec_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.registry_spec_sha256
            )
        ):
            raise ValueError("provider registry spec digest must be a lowercase SHA-256")
        if (
            isinstance(self.registry_generation, bool)
            or not isinstance(self.registry_generation, int)
            or self.registry_generation < 0
        ):
            raise ValueError("provider registry generation must be a non-negative integer")


ProviderRegistryBindingResolver = Callable[[], ProviderRegistryBinding]
ProviderRegistryPhaseGuard = Callable[[], AbstractContextManager[Any]]


@dataclass(frozen=True)
class ProtectedOperationInvocation:
    pid: str
    actor: str
    target: str | None
    decisions: tuple[CapabilityDecision, ...] = ()
    canonical_args: Mapping[str, Any] = field(default_factory=dict)
    observation: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    preflight_usage: ResourceUsage | Mapping[str, Any] | None = None
    reservation_usage: ResourceUsage | Mapping[str, Any] | None = None
    resource_source: str | None = None
    resource_context: Mapping[str, Any] = field(default_factory=dict)
    prepare: Hook | None = None
    authority_revalidator: AuthorityRevalidator | None = None
    provider_registry_binding: ProviderRegistryBinding | None = None
    provider_registry_binding_resolver: ProviderRegistryBindingResolver | None = None
    provider_registry_phase_guard: ProviderRegistryPhaseGuard | None = None
    restore_not_started: Hook | None = None
    failure_evidence: FailureEvidenceFactory | None = None
    failure_resource: ResourceSettlement | FailureResourceFactory | None = None
    failure_settlement: FailureSettlementHandler | None = None
    data_sink: DataSink | None = None
    data_sink_revalidator: DataSinkRevalidator | None = None
    data_flow_context: DataFlowContext | None = None
    data_flow_ingress_context: DataFlowContext | None = None
    data_flow_payload: Any = field(default=_DATA_FLOW_PAYLOAD_UNSET, repr=False)
    data_flow_operation: str | None = None
    data_flow_target_state_version: str | int | None = None
    data_flow_target_state_version_resolver: TargetStateVersionResolver | None = None
    data_flow_allow_recovered_source_snapshots: bool = False
    additional_data_sinks: tuple[DataSink, ...] = ()

    def __post_init__(self) -> None:
        _require_bool(
            "data_flow_allow_recovered_source_snapshots",
            self.data_flow_allow_recovered_source_snapshots,
        )


class ProtectedOperationProtocolError(ValidationError):
    pass


def _call_synchronous_hook(
    label: str,
    hook: Callable[..., Any],
    *args: Any,
) -> Any:
    """Run a transaction-local hook without silently dropping async work."""

    result = hook(*args)
    deferred = inspect.isawaitable(result)
    if inspect.isgenerator(result):
        result.close()
        deferred = True
    elif inspect.isasyncgen(result):
        deferred = True
    if deferred:
        if inspect.iscoroutine(result):
            result.close()
        raise ProtectedOperationProtocolError(
            f"protected operation {label} hook must complete synchronously"
        )
    return result


@dataclass(frozen=True)
class _ActiveBoundary:
    sdk_identity: int
    contract_name: str
    phase_name: str
    effect_id: str


_CURRENT_BOUNDARY: ContextVar[_ActiveBoundary | None] = ContextVar(
    "agent_libos_protected_provider_boundary",
    default=None,
)

T = TypeVar("T")


class ProtectedOperationSDK:
    """One fail-closed lifecycle for trusted provider operations."""

    def __init__(
        self,
        *,
        effects: ProtectedEffectPort,
        authority_policy: EffectAuthorityPort,
        capabilities: Any,
        audit: Any,
        events: Any,
        resources: Any | None,
        operations: Any,
        require_recovery_lease: Callable[[], None],
        data_flow: Any | None = None,
    ) -> None:
        self.effects = effects
        self.authority_policy = authority_policy
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.resources = resources
        self.operations = operations
        self._require_recovery_lease = require_recovery_lease
        self.data_flow = data_flow
        self._contracts: dict[str, ProtectedOperationContract] = {}
        self._prepared_recovery_handlers: dict[str, PreparedRecoveryHandler] = {}
        self._identity = id(self)

    def register_contract(self, contract: ProtectedOperationContract) -> ProtectedOperationContract:
        existing = self._contracts.get(contract.name)
        if existing is not None and existing != contract:
            raise ValidationError(f"protected operation contract conflict: {contract.name}")
        self._contracts[contract.name] = contract
        return contract

    def contracts(self) -> tuple[ProtectedOperationContract, ...]:
        return tuple(self._contracts[name] for name in sorted(self._contracts))

    def register_prepared_recovery(
        self,
        name: str,
        handler: PreparedRecoveryHandler,
    ) -> None:
        """Register one trusted, transaction-local prepared-state repair."""

        selected = str(name).strip()
        if not selected or not callable(handler):
            raise ValidationError("prepared recovery requires a name and callable handler")
        if inspect.iscoroutinefunction(handler):
            raise ValidationError(
                "prepared recovery requires a synchronous handler"
            )
        existing = self._prepared_recovery_handlers.get(selected)
        if existing is not None and existing != handler:
            raise ValidationError(f"prepared recovery handler conflict: {selected}")
        self._prepared_recovery_handlers[selected] = handler

    def recover_prepared(self, *, page_size: int = 500) -> ExternalEffectRecoverySummary:
        """Restore local prepare state that never reached a provider phase.

        A handler failure propagates and leaves the intent and its reservations
        unchanged because the handler, restoration, and abandonment share one
        RuntimeStore transaction.  Startup therefore fails closed instead of
        silently dropping local state that still needs repair.
        """

        self._require_recovery_lease()

        recovered_sample: list[str] = []
        recovered_total = 0
        query = ExternalEffectRecoveryQuery(
            transaction_states=("prepared",),
            limit=page_size,
        )
        for effect in iter_external_effect_recovery(self.effects, query):
            if not self._recover_prepared_effect(effect):
                continue
            recovered_total += 1
            if len(recovered_sample) < page_size:
                recovered_sample.append(effect.effect_id)
        return ExternalEffectRecoverySummary(
            total_count=recovered_total,
            sample_effect_ids=tuple(recovered_sample),
        )

    def _recover_prepared_effect(self, effect: ExternalEffectRecord) -> bool:
        recovery = self._prepared_recovery_metadata(effect)
        if recovery is None:
            return False
        reservation_ids, contract_name, actor, handler = recovery
        expected_reservation_reason = (
            f"protected operation reserved authority for {contract_name}"
        )
        with self.effects.transaction():
            operation_id = self._prepared_operation_binding(
                effect,
                contract_name=contract_name,
                actor=actor,
            )
            for reservation_id in reservation_ids:
                capability_id = self._prepared_reservation_capability(
                    effect,
                    reservation_id=reservation_id,
                    actor=actor,
                    expected_reason=expected_reservation_reason,
                )
                self._validate_prepared_reservation_evidence(
                    effect,
                    operation_id=operation_id,
                    reservation_id=reservation_id,
                    capability_id=capability_id,
                    contract_name=contract_name,
                    actor=actor,
                )
            if handler is not None:
                handler_result = _call_synchronous_hook(
                    "prepared recovery",
                    handler,
                    effect,
                )
                if handler_result is not None:
                    raise ValidationError(
                        "prepared recovery handler must return None"
                    )
            for reservation_id in reversed(reservation_ids):
                restored = self.capabilities.restore_reserved_use(
                    reservation_id,
                    restored_by="runtime.recovery",
                    reason=(
                        "protected operation recovered before provider dispatch: "
                        f"{contract_name}"
                    ),
                )
                if restored is None:
                    raise ValidationError(
                        "prepared protected operation reservation could not be restored: "
                        f"{effect.effect_id}"
                    )
            abandon_external_effect_intent(
                self.effects,
                effect.effect_id,
                operations=self.operations,
            )
        return True

    def _prepared_recovery_metadata(
        self,
        effect: ExternalEffectRecord,
    ) -> tuple[
        tuple[str, ...],
        str,
        str,
        PreparedRecoveryHandler | None,
    ] | None:
        protected = effect.provider_metadata.get("protected_operation")
        if not isinstance(protected, Mapping):
            return None
        raw_reservations = protected.get("reservation_ids")
        if not isinstance(raw_reservations, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in raw_reservations
        ):
            raise ValidationError(
                f"prepared protected operation has invalid reservation links: {effect.effect_id}"
            )
        reservation_ids = tuple(raw_reservations)
        if len(set(reservation_ids)) != len(reservation_ids):
            raise ValidationError(
                f"prepared protected operation has duplicate reservation links: {effect.effect_id}"
            )
        contract_name = protected.get("contract_name")
        actor = protected.get("actor")
        if not isinstance(contract_name, str) or not contract_name:
            raise ValidationError(
                f"prepared protected operation has invalid contract identity: {effect.effect_id}"
            )
        if not isinstance(actor, str) or not actor:
            raise ValidationError(
                f"prepared protected operation has invalid actor identity: {effect.effect_id}"
            )
        handler = self._prepared_recovery_handler(effect.effect_id, protected)
        return reservation_ids, contract_name, actor, handler

    def _prepared_reservation_capability(
        self,
        effect: ExternalEffectRecord,
        *,
        reservation_id: str,
        actor: str,
        expected_reason: str,
    ) -> str:
        reservation = self.effects.get_capability_use_reservation(reservation_id)
        if reservation is None:
            raise ValidationError(
                "prepared protected operation reservation is missing: "
                f"{effect.effect_id}"
            )
        if reservation.get("status") != "reserved":
            raise ValidationError(
                "prepared protected operation reservation is not live: "
                f"{effect.effect_id}"
            )
        reservation_count = reservation.get("count")
        if type(reservation_count) is not int or reservation_count != 1:
            raise ValidationError(
                "prepared protected operation reservation count is not exactly one: "
                f"{effect.effect_id}"
            )
        capability_id = reservation.get("cap_id")
        if not isinstance(capability_id, str) or not capability_id:
            raise ValidationError(
                "prepared protected operation reservation capability is invalid: "
                f"{effect.effect_id}"
            )
        if (
            reservation.get("reserved_by") != actor
            or reservation.get("reason") != expected_reason
        ):
            raise ValidationError(
                "prepared protected operation reservation binding mismatch: "
                f"{effect.effect_id}"
            )
        return capability_id

    def _prepared_operation_binding(
        self,
        effect: ExternalEffectRecord,
        *,
        contract_name: str,
        actor: str,
    ) -> str:
        links = self.effects.list_operation_evidence(
            evidence_types=("external_effect",),
            evidence_id=effect.effect_id,
            limit=2,
        )
        if len(links) != 1:
            raise ValidationError(
                "prepared protected operation effect binding is missing or ambiguous: "
                f"{effect.effect_id}"
            )
        link = links[0]
        metadata = link.metadata
        if (
            link.role != "effect"
            or not isinstance(metadata, Mapping)
            or metadata.get("effect_state") != "pending"
            or metadata.get("provider") != effect.provider
            or metadata.get("operation") != effect.operation
        ):
            raise ValidationError(
                "prepared protected operation effect binding is invalid: "
                f"{effect.effect_id}"
            )
        operation = self.effects.get_operation(link.operation_id)
        if (
            operation is None
            or operation.name != contract_name
            or operation.actor != actor
            or operation.pid != effect.pid
        ):
            raise ValidationError(
                "prepared protected operation identity binding mismatch: "
                f"{effect.effect_id}"
            )
        return operation.operation_id

    def _validate_prepared_reservation_evidence(
        self,
        effect: ExternalEffectRecord,
        *,
        operation_id: str,
        reservation_id: str,
        capability_id: str,
        contract_name: str,
        actor: str,
    ) -> None:
        links = self.effects.list_operation_evidence(
            operation_ids=(operation_id,),
            evidence_types=("capability_reservation",),
            evidence_id=reservation_id,
            limit=3,
        )
        if len(links) != 2:
            raise ValidationError(
                "prepared protected operation reservation evidence is missing or ambiguous: "
                f"{effect.effect_id}"
            )
        by_role = {link.role: link for link in links}
        if set(by_role) != {"reservation", "effect_reservation"}:
            raise ValidationError(
                "prepared protected operation reservation evidence is invalid: "
                f"{effect.effect_id}"
            )
        reservation_metadata = by_role["reservation"].metadata
        effect_metadata = by_role["effect_reservation"].metadata
        if (
            not isinstance(reservation_metadata, Mapping)
            or reservation_metadata.get("capability_id") != capability_id
            or reservation_metadata.get("status") != "reserved"
            or type(reservation_metadata.get("count")) is not int
            or reservation_metadata.get("count") != 1
            or not isinstance(effect_metadata, Mapping)
            or effect_metadata.get("effect_id") != effect.effect_id
            or effect_metadata.get("capability_id") != capability_id
            or type(effect_metadata.get("count")) is not int
            or effect_metadata.get("count") != 1
            or effect_metadata.get("contract_name") != contract_name
            or effect_metadata.get("actor") != actor
        ):
            raise ValidationError(
                "prepared protected operation reservation evidence mismatch: "
                f"{effect.effect_id}"
            )

    def _prepared_recovery_handler(
        self,
        effect_id: str,
        protected: Mapping[str, Any],
    ) -> PreparedRecoveryHandler | None:
        recovery_name = protected.get("prepared_recovery")
        if recovery_name is None:
            return None
        if not isinstance(recovery_name, str) or not recovery_name:
            raise ValidationError(
                f"prepared protected operation has invalid recovery policy: {effect_id}"
            )
        handler = self._prepared_recovery_handlers.get(recovery_name)
        if handler is None:
            raise ValidationError(
                f"prepared recovery handler is not registered: {recovery_name}"
            )
        return handler

    def current_boundary(self) -> tuple[str, str, str] | None:
        current = _CURRENT_BOUNDARY.get()
        if current is None or current.sdk_identity != self._identity:
            return None
        return current.contract_name, current.phase_name, current.effect_id

    def activate_boundary(
        self,
        *,
        contract_name: str,
        phase_name: str,
        effect_id: str,
    ) -> Token[_ActiveBoundary | None]:
        """Enter this SDK instance's provider boundary for one phase."""

        return _CURRENT_BOUNDARY.set(
            _ActiveBoundary(
                sdk_identity=self._identity,
                contract_name=contract_name,
                phase_name=phase_name,
                effect_id=effect_id,
            )
        )

    def start(
        self,
        contract: ProtectedOperationContract | str,
        invocation: ProtectedOperationInvocation,
        *,
        provider: Any,
    ) -> "ProtectedOperation":
        name = contract if isinstance(contract, str) else contract.name
        registered = self._contracts.get(name)
        if registered is None:
            raise ValidationError(f"protected operation contract is not registered: {name}")
        if not isinstance(contract, str) and registered != contract:
            raise ValidationError(f"protected operation contract does not match registry: {name}")
        recovery_name = registered.prepared_recovery
        if (
            recovery_name is not None
            and recovery_name not in self._prepared_recovery_handlers
        ):
            raise ValidationError(
                "protected operation prepared recovery handler is not registered: "
                f"{recovery_name}"
            )
        return ProtectedOperation(self, registered, invocation, provider)


class ProtectedOperation:
    def __init__(
        self,
        sdk: ProtectedOperationSDK,
        contract: ProtectedOperationContract,
        invocation: ProtectedOperationInvocation,
        provider: Any,
    ) -> None:
        self.sdk = sdk
        self.contract = contract
        self.invocation = invocation
        self.provider = provider
        self.effect_id: str | None = None
        self._reservation_ids: list[str] = []
        self._reservation_ids_by_capability: dict[str, str] = {}
        self._reservations_committed = False
        self._dispatched = False
        self._terminal = False
        self._completed_phases: list[ProviderPhase] = []
        self._authority_decisions: tuple[CapabilityDecision, ...] = ()
        self._failure_settlement_run = False
        self._data_flow_decision: DataFlowDecision | None = None
        self._data_flow_release_decision: CapabilityDecision | None = None
        self._data_flow_release_reservation_id: str | None = None
        self._additional_data_flow_decisions: tuple[DataFlowDecision, ...] = ()
        self._additional_data_flow_release_decisions: tuple[
            CapabilityDecision | None, ...
        ] = ()
        self._additional_data_flow_release_reservation_ids: tuple[
            str | None, ...
        ] = ()
        self._data_flow_registry_generation: int | None = None
        self._data_flow_ingress_observed = False
        self._operation_cm: Any | None = None
        self._resource_reservation_id: str | None = None

    @property
    def terminal(self) -> bool:
        return self._terminal

    def __enter__(self) -> "ProtectedOperation":
        current = self.sdk.operations.current()
        if (
            current is None
            or current.name != self.contract.name
            or current.pid != self.invocation.pid
            or current.actor != self.invocation.actor
        ):
            self._operation_cm = self.sdk.operations.scope(
                kind=OperationKind.PRIMITIVE,
                name=self.contract.name,
                actor=self.invocation.actor,
                pid=self.invocation.pid,
                expected_roles=(),
            )
            self._operation_cm.__enter__()
        try:
            self._prepare()
        except BaseException:
            if self._operation_cm is not None:
                operation_cm = self._operation_cm
                self._operation_cm = None
                operation_cm.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        protocol_error: BaseException | None = None
        settlement_error: BaseException | None = None
        try:
            if not self._terminal:
                if not self._dispatched:
                    self._abort_not_started("provider_not_dispatched")
                    if exc is None:
                        protocol_error = ProtectedOperationProtocolError(
                            f"protected operation exited without provider phase: {self.contract.name}"
                        )
                elif exc is None:
                    protocol_error = ProtectedOperationProtocolError(
                        f"protected operation exited without complete(): {self.contract.name}"
                    )
                    self._finalize_unknown(protocol_error, "protocol_incomplete")
                else:
                    self._finalize_unknown(exc, "caller_failed_after_provider")
        except BaseException as error:
            settlement_error = error

        selected_type = exc_type
        selected_exc = exc
        selected_tb = tb
        if settlement_error is not None:
            selected_type = type(settlement_error)
            selected_exc = settlement_error
            selected_tb = settlement_error.__traceback__
        elif protocol_error is not None:
            selected_type = type(protocol_error)
            selected_exc = protocol_error
            selected_tb = protocol_error.__traceback__
        operation_exit_error: BaseException | None = None
        if self._operation_cm is not None:
            operation_cm = self._operation_cm
            self._operation_cm = None
            try:
                operation_cm.__exit__(selected_type, selected_exc, selected_tb)
            except BaseException as error:
                operation_exit_error = error
        if settlement_error is not None:
            raise settlement_error.with_traceback(settlement_error.__traceback__)
        if protocol_error is not None:
            raise protocol_error
        if operation_exit_error is not None:
            raise operation_exit_error.with_traceback(operation_exit_error.__traceback__)
        return False

    def call(self, phase: ProviderPhase, function: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        self._require_active()
        with self._provider_registry_phase_scope():
            self._dispatch_or_abort_not_started(phase)
            token = self._activate_boundary(phase)
            try:
                result = function(*args, **kwargs)
            except ProviderEffectNotStarted as error:
                self._handle_not_started(error, phase)
                raise
            except BaseException as error:
                self._observe_data_flow_ingress(phase)
                self._expect_settlement_evidence()
                self._commit_reservations_best_effort()
                self._finalize_unknown(error, phase.name, active_phase=phase)
                raise
            finally:
                _CURRENT_BOUNDARY.reset(token)
        if isinstance(result, ProviderEffectNotStartedResult):
            self._handle_not_started(result.error, phase, outcome=result.outcome)
            return result  # type: ignore[return-value]
        self._observe_data_flow_ingress(phase)
        self._record_completed_phase(phase)
        self._expect_settlement_evidence()
        self._completed_phases.append(phase)
        if phase.commits_authority:
            try:
                self._commit_reservations()
            except BaseException as error:
                self._finalize_unknown(error, "capability_commit")
                raise
        return result

    async def acall(
        self,
        phase: ProviderPhase,
        function: Callable[..., Awaitable[T]],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        self._require_active()
        if self.invocation.provider_registry_phase_guard is not None:
            raise ValidationError(
                "registry-guarded provider phases must use synchronous call()"
            )
        self._dispatch_or_abort_not_started(phase)
        token = self._activate_boundary(phase)
        try:
            result = await function(*args, **kwargs)
        except ProviderEffectNotStarted as error:
            self._handle_not_started(error, phase)
            raise
        except BaseException as error:
            self._observe_data_flow_ingress(phase)
            self._expect_settlement_evidence()
            self._commit_reservations_best_effort()
            self._finalize_unknown(error, phase.name, active_phase=phase)
            raise
        finally:
            _CURRENT_BOUNDARY.reset(token)
        if isinstance(result, ProviderEffectNotStartedResult):
            self._handle_not_started(result.error, phase, outcome=result.outcome)
            return result  # type: ignore[return-value]
        self._observe_data_flow_ingress(phase)
        self._record_completed_phase(phase)
        self._expect_settlement_evidence()
        self._completed_phases.append(phase)
        if phase.commits_authority:
            try:
                self._commit_reservations()
            except BaseException as error:
                self._finalize_unknown(error, "capability_commit")
                raise
        return result

    def complete(
        self,
        result: T,
        evidence: ProtectedOperationEvidence,
        *,
        classification_context: Mapping[str, Any] | None = None,
        classification_result: Any | None = None,
        classification_override: ExternalEffectClassification | None = None,
        settle_success: Hook | None = None,
        resource: ResourceSettlement | None = None,
    ) -> T:
        self._require_active()
        if not self._dispatched:
            raise ProtectedOperationProtocolError(
                f"protected operation completed without provider dispatch: {self.contract.name}"
            )
        self._validate_resource_settlement(resource)
        try:
            classification = self._classification_with_phase_floor(
                classification_override
                if classification_override is not None
                else self._classification(
                    classification_context,
                    result if classification_result is None else classification_result,
                )
            )
            effect_metadata, flattened_metadata = self._safe_effect_metadata(
                evidence,
                classification_keys=classification.metadata.keys(),
            )
            classified_receipt = classification.metadata.get("provider_receipt")
            provider_receipt = (
                dict(evidence.provider_receipt)
                if evidence.provider_receipt
                else dict(classified_receipt)
                if isinstance(classified_receipt, Mapping)
                else {}
            )
            with self.sdk.effects.transaction():
                if settle_success is not None:
                    _call_synchronous_hook(
                        "success settlement",
                        settle_success,
                    )
                event, audit_record = self._persist_evidence(evidence)
                record_external_effect(
                    self.sdk.effects,
                    pid=self.invocation.pid,
                    provider=self.contract.provider,
                    operation=self.contract.operation,
                    target=self.invocation.target,
                    classification=classification,
                    audit_record=audit_record,
                    event=event,
                    metadata={
                        "context": dict(self.invocation.observation),
                        "protected_operation": self._protected_operation_evidence(),
                        "provider_phases": self._phase_metadata(),
                        "data_flow": self._data_flow_evidence(),
                        "result": effect_metadata,
                        **flattened_metadata,
                        "provider_receipt": provider_receipt,
                    },
                    intent_effect_id=self.effect_id,
                    operations=self.sdk.operations,
                )
            self._terminal = True
        except BaseException as error:
            try:
                self._run_failure_settlement(error, "completion_settlement")
                if self._resource_reservation_id is not None:
                    self._settle_resource_reservation_unknown(
                        error=error,
                        phase="completion_settlement",
                    )
            except BaseException as settlement_error:
                self._terminal = True
                raise settlement_error from error
            self._terminal = True
            if self.contract.post_provider_failure_mode == PostProviderFailureMode.PRESERVE_RESULT:
                return result
            raise

        self._charge_resource(resource)
        return result

    def _prepare(self) -> None:
        self._validate_authority()
        self._validate_provider_registry_binding_contract()
        self.sdk.authority_policy.assert_effect(
            self.invocation.pid,
            f"{self.contract.provider}.{self.contract.operation}",
        )
        self._preflight_data_flow()
        if self.contract.require_classifier:
            require_external_effect_classifier(self.provider, self.contract.operation)
        if self.contract.preflight_classifier:
            # Capability/manifest gates run before inspecting provider-specific
            # operation support. The second classification after completion can
            # still fail independently and uses the conservative ceiling.
            classify_external_effect(
                self.provider,
                self.contract.operation,
                dict(self.invocation.observation),
                {"preflight": True},
            )
        if self.invocation.failure_resource is not None:
            if self.contract.resource_policy == ResourcePolicy.NONE:
                raise ValidationError(
                    f"protected operation contract forbids failure resource settlement: {self.contract.name}"
                )
            if self.sdk.resources is None:
                raise ValidationError(
                    "protected operation failure resource settlement requires ResourceManager"
                )
        if self.invocation.reservation_usage is not None:
            if self.contract.resource_policy == ResourcePolicy.NONE:
                raise ValidationError(
                    f"protected operation contract forbids resource reservation: {self.contract.name}"
                )
            if self.sdk.resources is None:
                raise ValidationError("protected operation resource reservation requires ResourceManager")
        elif self.invocation.preflight_usage is not None:
            if self.contract.resource_policy == ResourcePolicy.NONE:
                raise ValidationError(
                    f"protected operation contract forbids resource preflight: {self.contract.name}"
                )
            if self.sdk.resources is None:
                raise ValidationError("protected operation resource preflight requires ResourceManager")
            source = self.invocation.resource_source or self.contract.name
            self.sdk.resources.preflight(
                self.invocation.pid,
                self.invocation.preflight_usage,
                source=source,
                context=dict(self.invocation.resource_context),
            )
        elif self.contract.resource_policy == ResourcePolicy.REQUIRED:
            raise ValidationError(
                f"protected operation requires resource preflight: {self.contract.name}"
            )
        observation = to_jsonable(dict(self.invocation.observation))
        canonical_args = to_jsonable(dict(self.invocation.canonical_args))
        if not isinstance(observation, dict) or not isinstance(canonical_args, dict):
            raise ValidationError("protected operation contexts must serialize to objects")
        try:
            with self.sdk.effects.transaction():
                if self.invocation.prepare is not None:
                    _call_synchronous_hook(
                        "prepare",
                        self.invocation.prepare,
                    )
                self._revalidate_authority()
                self._revalidate_data_flow()
                self._reserve_decisions()
                effect = prepare_external_effect_intent(
                    self.sdk.effects,
                    pid=self.invocation.pid,
                    provider=self.contract.provider,
                    operation=self.contract.operation,
                    target=self.invocation.target,
                    state_mutation=self.contract.state_mutation,
                    information_flow=self.contract.information_flow,
                    metadata={
                        "context": observation,
                        "protected_operation": self._protected_operation_evidence(),
                        "data_flow": self._data_flow_evidence(),
                    },
                    idempotency_key=self.invocation.idempotency_key,
                    canonical_args=canonical_args,
                    operations=self.sdk.operations,
                    authority_policy=self.sdk.authority_policy,
                )
                self.effect_id = effect.effect_id
                self._bind_prepared_reservations()
                if self.invocation.reservation_usage is not None:
                    assert self.sdk.resources is not None
                    self._resource_reservation_id = self.sdk.resources.reserve_usage(
                        self.invocation.pid,
                        self.invocation.reservation_usage,
                        source=self.invocation.resource_source or self.contract.name,
                        context=dict(self.invocation.resource_context),
                        reserved_by=effect.effect_id,
                    )
        except BaseException as error:
            self._persist_rolled_back_data_flow_denial(error)
            raise

    def _bind_prepared_reservations(self) -> None:
        assert self.effect_id is not None
        for capability_id, reservation_id in self._reservation_ids_by_capability.items():
            linked = self.sdk.operations.link_evidence(
                "capability_reservation",
                reservation_id,
                "effect_reservation",
                metadata={
                    "effect_id": self.effect_id,
                    "capability_id": capability_id,
                    "count": 1,
                    "contract_name": self.contract.name,
                    "actor": self.invocation.actor,
                },
            )
            if linked is None:
                raise ValidationError(
                    "protected operation could not bind its finite-use reservation"
                )

    def _validate_authority(self) -> None:
        if self.contract.authority_mode == AuthorityMode.RUNTIME_INTERNAL:
            return
        if not self.invocation.decisions:
            raise CapabilityDenied(
                f"protected provider operation requires an explicit capability decision: {self.contract.name}"
            )
        for decision in self.invocation.decisions:
            if not decision.allowed:
                raise CapabilityDenied(
                    f"protected provider operation received a denied capability decision: {decision.reason}"
                )
            if decision.subject != self.invocation.pid:
                raise CapabilityDenied(
                    "protected provider operation capability subject does not match the acting process"
                )

    def _reserve_decisions(self) -> None:
        decisions = [*self._authority_decisions]
        release_decisions = (
            self._data_flow_release_decision,
            *self._additional_data_flow_release_decisions,
        )
        decisions.extend(
            decision for decision in release_decisions if decision is not None
        )
        for decision in decisions:
            cap_id = decision.consume_capability_id
            if cap_id is None:
                continue
            capability_id = str(cap_id)
            reservation_id = self._reservation_ids_by_capability.get(capability_id)
            if reservation_id is None:
                reservation_id = self.sdk.capabilities.reserve_decision_use(
                    decision,
                    used_by=self.invocation.actor,
                    reason=f"protected operation reserved authority for {self.contract.name}",
                )
            if (
                reservation_id is not None
                and capability_id not in self._reservation_ids_by_capability
            ):
                self._reservation_ids_by_capability[capability_id] = reservation_id
                self._reservation_ids.append(reservation_id)
        release_reservation_ids = tuple(
            None
            if decision is None or decision.consume_capability_id is None
            else self._reservation_ids_by_capability.get(
                str(decision.consume_capability_id)
            )
            for decision in release_decisions
        )
        self._data_flow_release_reservation_id = release_reservation_ids[0]
        self._additional_data_flow_release_reservation_ids = (
            release_reservation_ids[1:]
        )

    def _revalidate_authority(self) -> None:
        if self.contract.authority_mode == AuthorityMode.RUNTIME_INTERNAL:
            self._authority_decisions = ()
            return
        if self.invocation.authority_revalidator is None:
            current = tuple(
                self.sdk.capabilities.reauthorize_decision(decision)
                for decision in self.invocation.decisions
            )
        else:
            current = tuple(self.invocation.authority_revalidator())
        if len(current) != len(self.invocation.decisions):
            raise CapabilityDenied(
                "protected operation authority revalidation changed the decision set"
            )
        for original, decision in zip(self.invocation.decisions, current, strict=True):
            self._validate_reauthorized_decision(original, decision)
        self._authority_decisions = current

    def _revalidate_dispatch_authority(self) -> None:
        if self.contract.authority_mode == AuthorityMode.RUNTIME_INTERNAL:
            return
        current: list[CapabilityDecision] = []
        for prepared in self._authority_decisions:
            reserved_capability_id = prepared.consume_capability_id
            if reserved_capability_id is None:
                decision = self.sdk.capabilities.reauthorize_decision(prepared)
                if decision.consume_capability_id is not None:
                    raise CapabilityDenied(
                        "protected operation authority changed to unreserved finite use "
                        "before protected dispatch"
                    )
            else:
                capability_id = str(reserved_capability_id)
                reservation_id = self._reservation_ids_by_capability.get(capability_id)
                if reservation_id is None:
                    raise CapabilityDenied(
                        "protected operation finite authority reservation disappeared "
                        "before protected dispatch"
                    )
                if not self._reservations_committed:
                    reservation = self.sdk.effects.get_capability_use_reservation(
                        reservation_id
                    )
                    reservation_count = (
                        reservation.get("count")
                        if reservation is not None
                        else None
                    )
                    if (
                        reservation is None
                        or reservation.get("status") != "reserved"
                        or str(reservation.get("cap_id") or "") != capability_id
                        or type(reservation_count) is not int
                        or reservation_count != 1
                    ):
                        raise CapabilityDenied(
                            "protected operation finite authority reservation changed "
                            "before protected dispatch"
                        )
                decision = prepared
            self._validate_reauthorized_decision(prepared, decision)
            current.append(decision)
        self._authority_decisions = tuple(current)

    def _validate_provider_registry_binding_contract(self) -> None:
        captured = self.invocation.provider_registry_binding
        resolver = self.invocation.provider_registry_binding_resolver
        guard = self.invocation.provider_registry_phase_guard
        supplied = (captured is not None, resolver is not None, guard is not None)
        if any(supplied) and not all(supplied):
            raise ValidationError(
                "protected operation provider registry binding, resolver, and phase guard must be supplied together"
            )
        if captured is not None and not isinstance(captured, ProviderRegistryBinding):
            raise ValidationError(
                "protected operation provider registry binding has an invalid type"
            )
        if resolver is not None and not callable(resolver):
            raise ValidationError(
                "protected operation provider registry binding resolver must be callable"
            )
        if guard is not None and not callable(guard):
            raise ValidationError(
                "protected operation provider registry phase guard must be callable"
            )

    def _provider_registry_phase_scope(self) -> AbstractContextManager[Any]:
        guard = self.invocation.provider_registry_phase_guard
        return nullcontext() if guard is None else guard()

    def _revalidate_provider_registry_binding(self) -> None:
        captured = self.invocation.provider_registry_binding
        resolver = self.invocation.provider_registry_binding_resolver
        if captured is None:
            return
        if resolver is None:
            raise ProtectedOperationProtocolError(
                "protected operation provider registry binding resolver disappeared"
            )
        current = resolver()
        if not isinstance(current, ProviderRegistryBinding):
            raise ValidationError(
                "protected operation provider registry binding resolver returned an invalid type"
            )
        if current != captured:
            raise CapabilityDenied(
                "provider registry binding changed before protected dispatch"
            )

    def _validate_reauthorized_decision(
        self,
        original: CapabilityDecision,
        decision: CapabilityDecision,
    ) -> None:
        if (
            decision.subject,
            decision.resource,
            decision.right,
        ) != (
            original.subject,
            original.resource,
            original.right,
        ):
            raise CapabilityDenied(
                "protected operation authority revalidation changed the requested authority"
            )
        if not decision.allowed:
            raise CapabilityDenied(
                "protected operation authority changed before dispatch: "
                f"{decision.reason}"
            )
        if decision.subject != self.invocation.pid:
            raise CapabilityDenied(
                "protected operation revalidated capability subject does not match the acting process"
            )

    def _persist_rolled_back_data_flow_denial(self, error: BaseException) -> None:
        decision = getattr(error, "data_flow_decision", None)
        sink = getattr(error, "data_flow_sink", None)
        manager = self.sdk.data_flow
        if (
            manager is None
            or not isinstance(decision, DataFlowDecision)
            or not isinstance(sink, DataSink)
        ):
            return
        manager.persist_denied_decision(decision=decision, sink=sink)

    def _preflight_data_flow(self) -> None:
        direction = self.contract.data_flow_direction
        manager = self.sdk.data_flow
        has_ingress = direction in {
            DataFlowDirection.INGRESS,
            DataFlowDirection.BIDIRECTIONAL,
        }
        ingress_context = self.invocation.data_flow_ingress_context
        if has_ingress:
            if manager is None:
                raise ValidationError(
                    f"ingress protected operation requires DataFlowManager: {self.contract.name}"
                )
            if not isinstance(ingress_context, DataFlowContext):
                raise ValidationError(
                    "ingress protected operation requires a trusted "
                    f"DataFlowContext: {self.contract.name}"
                )
        elif ingress_context is not None:
            raise ValidationError(
                f"non-ingress protected operation declares data-flow ingress state: {self.contract.name}"
            )

        has_egress = direction in {
            DataFlowDirection.EGRESS,
            DataFlowDirection.BIDIRECTIONAL,
        }
        if not has_egress:
            if (
                self.invocation.data_sink is not None
                or self.invocation.additional_data_sinks
                or self.invocation.data_sink_revalidator is not None
                or self.invocation.data_flow_context is not None
                or self.invocation.data_flow_payload is not _DATA_FLOW_PAYLOAD_UNSET
                or self.invocation.data_flow_operation is not None
                or self.invocation.data_flow_target_state_version is not None
                or self.invocation.data_flow_target_state_version_resolver is not None
                or self.invocation.data_flow_allow_recovered_source_snapshots
            ):
                raise ValidationError(
                    f"non-egress protected operation declares data-flow egress state: {self.contract.name}"
                )
            return
        if manager is None:
            raise ValidationError(
                f"egress protected operation requires DataFlowManager: {self.contract.name}"
            )
        sink = self.invocation.data_sink
        if not isinstance(sink, DataSink):
            raise ValidationError(
                f"egress protected operation requires a concrete DataSink: {self.contract.name}"
            )
        additional_sinks = self.invocation.additional_data_sinks
        if not isinstance(additional_sinks, tuple) or any(
            not isinstance(item, DataSink) for item in additional_sinks
        ):
            raise ValidationError(
                "additional egress data Sinks must be a tuple of concrete DataSink values"
            )
        sinks = (sink, *additional_sinks)
        if len({item.identity for item in sinks}) != len(sinks):
            raise ValidationError("egress data Sink identities must be unique")
        context = self.invocation.data_flow_context
        if not isinstance(context, DataFlowContext):
            raise ValidationError(
                f"egress protected operation requires a trusted DataFlowContext: {self.contract.name}"
            )
        if self.invocation.data_flow_payload is _DATA_FLOW_PAYLOAD_UNSET:
            raise ValidationError(
                f"egress protected operation requires an explicit payload descriptor: {self.contract.name}"
            )
        operation = str(self.invocation.data_flow_operation or "").strip()
        if not operation:
            raise ValidationError(
                f"egress protected operation requires an operation descriptor: {self.contract.name}"
            )
        payload = self.invocation.data_flow_payload
        decisions: list[DataFlowDecision] = []
        releases: list[CapabilityDecision | None] = []
        registry_generation: int | None = None
        for selected_sink in sinks:
            authorization: dict[str, Any] = {
                "pid": self.invocation.pid,
                "sink": selected_sink,
                "context": context,
                "payload": payload,
                "operation": operation,
                "target_state_version": self.invocation.data_flow_target_state_version,
                "request_release": True,
                "minimum_integrity": self.contract.minimum_egress_integrity,
                "allow_recovered_source_snapshots": (
                    self.invocation.data_flow_allow_recovered_source_snapshots
                ),
            }
            if registry_generation is not None:
                authorization["expected_registry_generation"] = registry_generation
            decision, release = manager.authorize_egress(**authorization)
            if registry_generation is None:
                registry_generation = decision.registry_generation
            decisions.append(decision)
            releases.append(release)
        self._data_flow_decision = decisions[0]
        self._data_flow_release_decision = releases[0]
        self._additional_data_flow_decisions = tuple(decisions[1:])
        self._additional_data_flow_release_decisions = tuple(releases[1:])
        self._data_flow_registry_generation = registry_generation

    def _observe_data_flow_ingress(self, phase: ProviderPhase) -> None:
        if self._data_flow_ingress_observed or not phase.information_flow:
            return
        if self.contract.data_flow_direction not in {
            DataFlowDirection.INGRESS,
            DataFlowDirection.BIDIRECTIONAL,
        }:
            return
        manager = self.sdk.data_flow
        context = self.invocation.data_flow_ingress_context
        assert manager is not None and isinstance(context, DataFlowContext)
        manager.observe_ingress(context)
        self._data_flow_ingress_observed = True

    def _revalidate_data_sink_identity(self) -> None:
        resolver = self.invocation.data_sink_revalidator
        if resolver is None:
            return
        expected = self.invocation.data_sink
        assert isinstance(expected, DataSink)
        context = self.invocation.data_flow_context
        assert isinstance(context, DataFlowContext)
        payload = self.invocation.data_flow_payload
        assert payload is not _DATA_FLOW_PAYLOAD_UNSET
        try:
            current = resolver()
        except (OSError, ValidationError) as error:
            manager = self.sdk.data_flow
            assert manager is not None
            manager.reject_sink_identity_change(
                pid=self.invocation.pid,
                sink=expected,
                context=context,
                payload=payload,
                reason=(
                    "Sink identity could not be revalidated before provider dispatch "
                    f"({type(error).__name__})"
                ),
            )
            raise AssertionError("data-flow Sink rejection must raise") from error
        if not isinstance(current, DataSink):
            raise ValidationError("data Sink revalidator must return DataSink")
        if current == expected:
            return
        manager = self.sdk.data_flow
        assert manager is not None
        manager.reject_sink_identity_change(
            pid=self.invocation.pid,
            sink=current,
            context=context,
            payload=payload,
        )
        raise AssertionError("data-flow Sink rejection must raise")

    def _revalidate_data_flow(self, *, use_reserved_release: bool = False) -> None:
        direction = self.contract.data_flow_direction
        if direction not in {DataFlowDirection.EGRESS, DataFlowDirection.BIDIRECTIONAL}:
            return
        manager = self.sdk.data_flow
        sink = self.invocation.data_sink
        assert manager is not None and isinstance(sink, DataSink)
        sinks = (sink, *self.invocation.additional_data_sinks)
        context = self.invocation.data_flow_context
        assert isinstance(context, DataFlowContext)
        payload = self.invocation.data_flow_payload
        assert payload is not _DATA_FLOW_PAYLOAD_UNSET
        operation = str(self.invocation.data_flow_operation or "").strip()
        assert operation
        authorization: dict[str, Any] = {
            "pid": self.invocation.pid,
            "context": context,
            "payload": payload,
            "operation": operation,
            "target_state_version": self.invocation.data_flow_target_state_version,
            "request_release": False,
            "minimum_integrity": self.contract.minimum_egress_integrity,
            "expected_registry_generation": self._data_flow_registry_generation,
            "allow_recovered_source_snapshots": (
                self.invocation.data_flow_allow_recovered_source_snapshots
            ),
        }
        resolver = self.invocation.data_flow_target_state_version_resolver
        if resolver is not None:
            authorization["current_target_state_version"] = resolver()
        prepared_releases = (
            self._data_flow_release_decision,
            *self._additional_data_flow_release_decisions,
        )
        reservation_ids = (
            self._data_flow_release_reservation_id,
            *self._additional_data_flow_release_reservation_ids,
        )
        if len(prepared_releases) != len(sinks):
            raise ProtectedOperationProtocolError(
                "prepared data-flow release set does not match egress Sinks"
            )
        if use_reserved_release and len(reservation_ids) != len(sinks):
            raise ProtectedOperationProtocolError(
                "reserved data-flow release set does not match egress Sinks"
            )
        decisions: list[DataFlowDecision] = []
        releases: list[CapabilityDecision | None] = []
        for index, selected_sink in enumerate(sinks):
            selected_authorization = {**authorization, "sink": selected_sink}
            prepared_release = prepared_releases[index]
            if use_reserved_release and prepared_release is not None:
                reservation_id = reservation_ids[index]
                if reservation_id is None:
                    raise CapabilityDenied(
                        "data release reservation disappeared before protected dispatch"
                    )
                selected_authorization.update(
                    reserved_release_decision=prepared_release,
                    reserved_release_id=reservation_id,
                )
            decision, release = manager.authorize_egress(**selected_authorization)
            if (prepared_release is None) != (release is None):
                raise CapabilityDenied(
                    "data release authority changed before protected dispatch"
                )
            if release is not None and prepared_release is not None:
                if release.selected_capability_id != prepared_release.selected_capability_id:
                    raise CapabilityDenied(
                        "data release capability changed before protected dispatch"
                    )
            decisions.append(decision)
            releases.append(release)
        self._data_flow_decision = decisions[0]
        self._data_flow_release_decision = releases[0]
        self._additional_data_flow_decisions = tuple(decisions[1:])
        self._additional_data_flow_release_decisions = tuple(releases[1:])

    def _protected_operation_evidence(self) -> dict[str, Any]:
        return {
            "contract_name": self.contract.name,
            "actor": self.invocation.actor,
            "minimum_egress_integrity": (
                self.contract.minimum_egress_integrity.value
            ),
            "reservation_ids": list(self._reservation_ids),
            "prepared_recovery": self.contract.prepared_recovery,
        }

    def _data_flow_evidence(self) -> dict[str, Any] | None:
        decision = self._data_flow_decision
        if decision is None:
            return None
        sink = self.invocation.data_sink
        assert isinstance(sink, DataSink)
        evidence = self._data_flow_decision_evidence(decision, sink)
        if self._additional_data_flow_decisions:
            evidence["additional_egresses"] = [
                self._data_flow_decision_evidence(selected_decision, selected_sink)
                for selected_decision, selected_sink in zip(
                    self._additional_data_flow_decisions,
                    self.invocation.additional_data_sinks,
                    strict=True,
                )
            ]
        return evidence

    def _data_flow_decision_evidence(
        self,
        decision: DataFlowDecision,
        sink: DataSink,
    ) -> dict[str, Any]:
        return {
            "decision_id": decision.decision_id,
            "sink": decision.sink,
            "sink_identity_sha256": sink.identity_sha256,
            "sink_trust_identity": sink.registry_identity,
            "sink_trust_identity_sha256": sink.registry_identity_sha256,
            "direction": decision.direction.value,
            "outcome": decision.outcome.value,
            "reason": decision.reason,
            "labels": decision.labels.to_dict(),
            "labels_sha256": decision.labels.labels_hash(),
            "source_refs": [item.to_dict() for item in decision.source_refs],
            "source_refs_sha256": DataFlowContext(
                labels=decision.labels,
                source_refs=decision.source_refs,
            ).source_refs_hash(),
            "payload_sha256": decision.payload_hash,
            "trust_id": decision.trust_id,
            "trust_sha256": decision.trust_hash,
            "registry_generation": decision.registry_generation,
            "release_capability_id": decision.release_capability_id,
            "minimum_egress_integrity": (
                self.contract.minimum_egress_integrity.value
            ),
        }

    def _dispatch(self, phase: ProviderPhase) -> None:
        if self.effect_id is None:
            raise ProtectedOperationProtocolError("protected operation has no prepared effect")
        try:
            with self.sdk.effects.transaction():
                self._revalidate_dispatch_authority()
                self._revalidate_provider_registry_binding()
                self._revalidate_data_sink_identity()
                self._revalidate_data_flow(use_reserved_release=True)
                # Preserve a more specific raced authority/data-flow denial,
                # while still rejecting an over-ceiling phase before dispatch
                # is persisted or any provider code can run.
                self._validate_phase_against_contract(phase)
                mark_external_effect_dispatched(self.sdk.effects, self.effect_id)
                current = self.sdk.effects.get_external_effect(self.effect_id)
                if current is None:
                    raise ProtectedOperationProtocolError(
                        "protected operation effect disappeared during dispatch"
                    )
                metadata = {
                    **dict(current.provider_metadata),
                    "active_provider_phase": {
                        "name": phase.name,
                        "state_mutation": phase.state_mutation,
                        "information_flow": phase.information_flow,
                        "commits_authority": phase.commits_authority,
                    },
                }
                if not self.sdk.effects.transition_external_effect(
                    self.effect_id,
                    expected_states=("dispatched",),
                    transaction_state="dispatched",
                    provider_metadata=metadata,
                    updated_at=self._now(),
                ):
                    raise ProtectedOperationProtocolError(
                        "protected operation phase dispatch cannot be persisted: "
                        f"{self.contract.name}:{phase.name}"
                    )
        except BaseException as error:
            self._persist_rolled_back_data_flow_denial(error)
            raise
        self._dispatched = True

    def _dispatch_or_abort_not_started(self, phase: ProviderPhase) -> None:
        try:
            self._dispatch(phase)
        except BaseException:
            if not any(
                item.state_mutation
                or item.information_flow
                or item.commits_authority
                for item in self._completed_phases
            ):
                self._abort_not_started(
                    "provider_dispatch_rejected_after_non_effectful_phases"
                )
            raise

    def _record_completed_phase(self, phase: ProviderPhase) -> None:
        if self.effect_id is None:
            raise ProtectedOperationProtocolError("protected operation has no effect for phase completion")
        current = self.sdk.effects.get_external_effect(self.effect_id)
        if current is None:
            raise ProtectedOperationProtocolError("protected operation effect disappeared after provider call")
        completed = list(current.provider_metadata.get("completed_provider_phases") or [])
        completed.append(
            {
                "name": phase.name,
                "state_mutation": phase.state_mutation,
                "information_flow": phase.information_flow,
                "commits_authority": phase.commits_authority,
            }
        )
        metadata = {
            **dict(current.provider_metadata),
            "active_provider_phase": None,
            "completed_provider_phases": completed,
            "observed_state_mutation": any(bool(item.get("state_mutation")) for item in completed),
            "observed_information_flow": any(bool(item.get("information_flow")) for item in completed),
        }
        if not self.sdk.effects.transition_external_effect(
            self.effect_id,
            expected_states=("dispatched",),
            transaction_state="dispatched",
            provider_metadata=metadata,
            updated_at=self._now(),
        ):
            error = ProtectedOperationProtocolError(
                f"protected operation phase completion cannot be persisted: {self.contract.name}:{phase.name}"
            )
            self._expect_settlement_evidence()
            self._commit_reservations_best_effort()
            self._finalize_unknown(
                error,
                f"{phase.name}_phase_evidence",
                active_phase=phase,
            )
            raise error

    def _activate_boundary(self, phase: ProviderPhase) -> Token[_ActiveBoundary | None]:
        return self.sdk.activate_boundary(
            contract_name=self.contract.name,
            phase_name=phase.name,
            effect_id=str(self.effect_id),
        )

    def _commit_reservations(self) -> None:
        if self._reservations_committed:
            return
        with self.sdk.effects.transaction():
            for reservation_id in self._reservation_ids:
                self.sdk.capabilities.commit_reserved_use(
                    reservation_id,
                    committed_by=self.invocation.actor,
                    reason=f"protected operation crossed provider boundary: {self.contract.name}",
                )
        self._reservations_committed = True

    def _expect_settlement_evidence(self) -> None:
        self.sdk.operations.expect("audit", "event")

    def _commit_reservations_best_effort(self) -> None:
        try:
            self._commit_reservations()
        except Exception:
            pass

    def _restore_reservations(self) -> None:
        for reservation_id in reversed(self._reservation_ids):
            self.sdk.capabilities.restore_reserved_use(
                reservation_id,
                restored_by=self.invocation.actor,
                reason=f"protected operation certified not started: {self.contract.name}",
            )

    def _abort_not_started(self, reason: str) -> None:
        if self._terminal:
            return
        with self.sdk.effects.transaction():
            if self.invocation.restore_not_started is not None:
                _call_synchronous_hook(
                    "not-started restoration",
                    self.invocation.restore_not_started,
                )
            self._restore_reservations()
            self._release_resource_reservation(reason)
            abandon_external_effect_intent(
                self.sdk.effects,
                self.effect_id,
                operations=self.sdk.operations,
            )
        self._terminal = True

    def _handle_not_started(
        self,
        error: ProviderEffectNotStarted,
        phase: ProviderPhase,
        *,
        outcome: str = "partial_not_started_after_prior_provider_effect",
    ) -> None:
        effectful_phases = [
            item
            for item in self._completed_phases
            if item.state_mutation or item.information_flow or item.commits_authority
        ]
        if not effectful_phases:
            self._abort_not_started("provider_certified_not_started")
            return
        self._commit_reservations_best_effort()
        evidence = self._failure_evidence(error, phase.name)
        state_mutation = any(item.state_mutation for item in effectful_phases)
        information_flow = any(item.information_flow for item in effectful_phases)
        classification = ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.UNKNOWN,
            rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
            state_mutation=state_mutation,
            information_flow=information_flow,
            metadata={
                "outcome": outcome,
                "phase": phase.name,
                "error_type": type(error).__name__,
            },
        )
        self._settle_failure(
            classification,
            evidence,
            error=error,
            phase=phase.name,
        )

    def _finalize_unknown(
        self,
        error: BaseException,
        phase: str,
        *,
        active_phase: ProviderPhase | None = None,
    ) -> None:
        if self._terminal or self.effect_id is None:
            return
        evidence = self._failure_evidence(error, phase)
        outcome = (
            "unknown_after_provider_success"
            if phase in {"caller_failed_after_provider", "protocol_incomplete"}
            else "unknown_after_provider_exception"
        )
        phase_mutation, phase_flow = self._effect_ceiling(active_phase=active_phase)
        classification = ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.UNKNOWN,
            rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
            state_mutation=phase_mutation,
            information_flow=phase_flow,
            metadata={
                "outcome": outcome,
                "phase": phase,
                "error_type": type(error).__name__,
            },
        )
        self._settle_failure(
            classification,
            evidence,
            error=error,
            phase=phase,
        )

    def _settle_failure(
        self,
        classification: ExternalEffectClassification,
        evidence: ProtectedOperationEvidence,
        *,
        error: BaseException,
        phase: str,
    ) -> None:
        settlement_error: BaseException | None = None
        try:
            self._run_failure_settlement(error, phase)
        except BaseException as failure:
            settlement_error = failure
        settled = False
        try:
            effect_metadata, flattened_metadata = self._safe_effect_metadata(
                evidence,
                classification_keys=classification.metadata.keys(),
            )
            with self.sdk.effects.transaction():
                event, audit_record = self._persist_evidence(evidence)
                record_external_effect(
                    self.sdk.effects,
                    pid=self.invocation.pid,
                    provider=self.contract.provider,
                    operation=self.contract.operation,
                    target=self.invocation.target,
                    classification=classification,
                    audit_record=audit_record,
                    event=event,
                    metadata={
                        "context": dict(self.invocation.observation),
                        "provider_phases": self._phase_metadata(),
                        "data_flow": self._data_flow_evidence(),
                        "result": effect_metadata,
                        **flattened_metadata,
                        "error_type": classification.metadata.get("error_type"),
                    },
                    intent_effect_id=self.effect_id,
                    operations=self.sdk.operations,
                )
            settled = True
        except Exception:
            # The prepared/dispatched intent is the durable unknown evidence.
            pass
        self._terminal = True
        if settled:
            failure_resource = self._failure_resource_settlement(error, phase)
            if self._resource_reservation_id is not None and failure_resource is None:
                self._settle_resource_reservation_unknown(error=error, phase=phase)
            else:
                self._charge_resource(failure_resource)
        if settlement_error is not None:
            raise settlement_error from error

    def _run_failure_settlement(self, error: BaseException, phase: str) -> None:
        handler = self.invocation.failure_settlement
        if handler is None or self._failure_settlement_run or not self._dispatched:
            return
        self._failure_settlement_run = True
        with self.sdk.effects.transaction():
            _call_synchronous_hook(
                "failure settlement",
                handler,
                error,
                phase,
            )

    def _failure_resource_settlement(
        self,
        error: BaseException,
        phase: str,
    ) -> ResourceSettlement | None:
        selected = self.invocation.failure_resource
        settlement = selected(error, phase) if callable(selected) else selected
        if settlement is not None:
            self._validate_resource_settlement(settlement, required=False)
            return settlement
        if self.contract.resource_policy != ResourcePolicy.REQUIRED:
            return None
        if self.invocation.reservation_usage is not None:
            return None
        usage = self.invocation.preflight_usage
        if usage is None:
            raise ProtectedOperationProtocolError(
                f"protected operation requires failure resource settlement: {self.contract.name}"
            )
        return ResourceSettlement(
            usage=usage,
            source=self.invocation.resource_source or self.contract.name,
            context={
                **dict(self.invocation.resource_context),
                "failure_phase": phase,
                "error_type": type(error).__name__,
            },
        )

    def _validate_resource_settlement(
        self,
        resource: ResourceSettlement | None,
        *,
        required: bool = True,
    ) -> None:
        if (
            resource is None
            and required
            and self.contract.resource_policy == ResourcePolicy.REQUIRED
        ):
            raise ProtectedOperationProtocolError(
                f"protected operation requires resource settlement: {self.contract.name}"
            )
        if resource is not None and self.contract.resource_policy == ResourcePolicy.NONE:
            raise ProtectedOperationProtocolError(
                f"protected operation contract forbids resource settlement: {self.contract.name}"
            )
        if resource is not None and self.sdk.resources is None:
            raise ValidationError("protected operation resource settlement requires ResourceManager")

    def _charge_resource(self, resource: ResourceSettlement | None) -> None:
        if resource is None:
            return
        self.sdk.operations.expect("resource_charge")
        assert self.sdk.resources is not None
        if self._resource_reservation_id is not None:
            self.sdk.resources.settle_usage_reservation(
                self._resource_reservation_id,
                actual_usage=None if resource.charge_reserved_maximum else resource.usage,
                charge_maximum=resource.charge_reserved_maximum,
                source=resource.source,
                context=dict(resource.context),
            )
            return
        self.sdk.resources.charge(
            self.invocation.pid,
            resource.usage,
            source=resource.source,
            context=dict(resource.context),
            allow_overage=resource.allow_overage,
            kill_on_exceed=resource.kill_on_exceed,
        )

    def _release_resource_reservation(self, reason: str) -> None:
        if self._resource_reservation_id is None:
            return
        assert self.sdk.resources is not None
        self.sdk.resources.settle_usage_reservation(
            self._resource_reservation_id,
            release=True,
            source=self.invocation.resource_source or self.contract.name,
            context={**dict(self.invocation.resource_context), "release_reason": reason},
        )

    def _settle_resource_reservation_unknown(
        self,
        *,
        error: BaseException,
        phase: str,
    ) -> None:
        assert self._resource_reservation_id is not None
        assert self.sdk.resources is not None
        self.sdk.operations.expect("resource_charge")
        self.sdk.resources.settle_usage_reservation(
            self._resource_reservation_id,
            charge_maximum=True,
            source=self.invocation.resource_source or self.contract.name,
            context={
                **dict(self.invocation.resource_context),
                "failure_phase": phase,
                "error_type": type(error).__name__,
                "settlement": "fail_closed_maximum",
            },
        )

    def _phase_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "name": phase.name,
                "state_mutation": phase.state_mutation,
                "information_flow": phase.information_flow,
                "commits_authority": phase.commits_authority,
            }
            for phase in self._completed_phases
        ]

    @staticmethod
    def _now() -> str:
        return utc_now()

    def _safe_effect_metadata(
        self,
        evidence: ProtectedOperationEvidence,
        *,
        classification_keys: Iterable[str] = (),
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        metadata = dict(evidence.effect_metadata)
        # These keys are owned by the SDK/effect ledger. Extension evidence may
        # add safe domain fields but cannot replace canonical lifecycle data.
        reserved = {
            "context",
            "effect_state",
            "provider_receipt",
            "result",
            "transaction_state",
            *classification_keys,
        }
        flattened = {key: value for key, value in metadata.items() if key not in reserved}
        return metadata, flattened

    def _failure_evidence(self, error: BaseException, phase: str) -> ProtectedOperationEvidence:
        if self.invocation.failure_evidence is not None:
            return self.invocation.failure_evidence(error, phase)
        event_type = (
            EventType.EXTERNAL_WRITE if self.contract.state_mutation else EventType.EXTERNAL_READ
        )
        return ProtectedOperationEvidence(
            event_type=event_type,
            event_source=self.invocation.actor,
            event_target=self.invocation.target,
            event_payload={
                "provider": self.contract.provider,
                "operation": self.contract.operation,
                "outcome": "unknown",
                "phase": phase,
                "error_type": type(error).__name__,
            },
            audit_action=f"{self.contract.name}.failed",
            audit_actor=self.invocation.actor,
            audit_target=self.invocation.target,
            audit_decision={
                "provider": self.contract.provider,
                "operation": self.contract.operation,
                "effect_outcome": "unknown",
                "phase": phase,
                "error_type": type(error).__name__,
            },
        )

    def _persist_evidence(
        self,
        evidence: ProtectedOperationEvidence,
    ) -> tuple[Event, AuditRecord]:
        event = self.sdk.events.emit(
            evidence.event_type,
            source=evidence.event_source,
            target=evidence.event_target,
            payload=dict(evidence.event_payload),
            priority=evidence.event_priority,
            correlation_id=evidence.correlation_id,
            causality=(
                {"audit_parent_record_id": evidence.parent_record_id}
                if evidence.parent_record_id is not None
                else None
            ),
        )
        audit_record = self.sdk.audit.record(
            actor=evidence.audit_actor,
            action=evidence.audit_action,
            target=evidence.audit_target,
            input_refs=list(evidence.input_refs),
            output_refs=list(evidence.output_refs),
            capability_refs=list(evidence.capability_refs),
            decision=dict(evidence.audit_decision),
            correlation_id=evidence.correlation_id,
            parent_record_id=evidence.parent_record_id,
        )
        return event, audit_record

    def _classification(
        self,
        context: Mapping[str, Any] | None,
        result: Any,
    ) -> ExternalEffectClassification:
        try:
            classification = classify_external_effect(
                self.provider,
                self.contract.operation,
                dict(context or self.invocation.observation),
                result,
            )
        except Exception as error:
            classification = ExternalEffectClassification(
                rollback_class=self.contract.classifier_failure_rollback_class,
                rollback_status=self.contract.classifier_failure_rollback_status,
                state_mutation=self.contract.state_mutation,
                information_flow=self.contract.information_flow,
                metadata={
                    "classification_fallback": self.contract.classifier_failure_label,
                    "classification_error_type": type(error).__name__,
                },
            )
        return classification

    def _classification_with_phase_floor(
        self,
        classification: ExternalEffectClassification,
    ) -> ExternalEffectClassification:
        """Preserve effects observed in successfully completed provider phases.

        A contract is a maximum capability/effect declaration, not proof that
        every successful path exercised that effect.  Explicit completion may
        therefore narrow the contract, but neither an override nor a provider
        classifier may erase a phase that actually completed.
        """

        phase_mutation = any(item.state_mutation for item in self._completed_phases)
        phase_flow = any(item.information_flow for item in self._completed_phases)
        return ExternalEffectClassification(
            rollback_class=classification.rollback_class,
            rollback_status=classification.rollback_status,
            state_mutation=bool(classification.state_mutation or phase_mutation),
            information_flow=bool(classification.information_flow or phase_flow),
            metadata=dict(classification.metadata),
        )

    def _validate_phase_against_contract(self, phase: ProviderPhase) -> None:
        """Require every provider phase to stay within its registered ceiling."""

        exceeded: list[str] = []
        if phase.state_mutation and not self.contract.state_mutation:
            exceeded.append("state_mutation")
        if phase.information_flow and not self.contract.information_flow:
            exceeded.append("information_flow")
        if exceeded:
            raise ProtectedOperationProtocolError(
                "provider phase exceeds protected operation contract ceiling "
                f"for {self.contract.name}:{phase.name}: {', '.join(exceeded)}"
            )

    def _effect_ceiling(
        self,
        *,
        active_phase: ProviderPhase | None = None,
    ) -> tuple[bool, bool]:
        """Return the conservative ceiling for ambiguous failure settlement."""

        phases = [*self._completed_phases]
        if active_phase is not None:
            phases.append(active_phase)
        return (
            bool(
                self.contract.state_mutation
                or any(item.state_mutation for item in phases)
            ),
            bool(
                self.contract.information_flow
                or any(item.information_flow for item in phases)
            ),
        )

    def _require_active(self) -> None:
        if self._terminal:
            raise ProtectedOperationProtocolError(
                f"protected operation is already terminal: {self.contract.name}"
            )
        if self.effect_id is None:
            raise ProtectedOperationProtocolError(
                f"protected operation has not been entered: {self.contract.name}"
            )


__all__ = [
    "AuthorityMode",
    "PostProviderFailureMode",
    "ProtectedOperation",
    "ProtectedOperationContract",
    "ProtectedOperationEvidence",
    "ProtectedOperationInvocation",
    "ProtectedOperationProtocolError",
    "ProtectedOperationSDK",
    "ProviderEffectNotStartedResult",
    "ProviderPhase",
    "ResourcePolicy",
    "ResourceSettlement",
]
