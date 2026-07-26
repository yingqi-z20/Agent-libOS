from __future__ import annotations

from contextvars import Token
from typing import Any, Iterable, Mapping, Protocol

from agent_libos.models import (
    DataFlowContext,
    DataFlowDecision,
    DataFlowOutcome,
    DataIntegrity,
    DataReleaseBinding,
    DataSink,
    SinkTrustSpec,
)


class DataFlowPort(Protocol):
    """Execution-time data-flow context without a Runtime dependency."""

    def current_context(self) -> DataFlowContext:
        ...

    def push(self, context: DataFlowContext) -> Token[DataFlowContext]:
        ...

    def reset(self, token: Token[DataFlowContext]) -> None:
        ...

    def observe_ingress(self, context: DataFlowContext) -> DataFlowContext:
        ...

    def provenance_sources(
        self,
        context: DataFlowContext,
        *,
        exclude_oids: Iterable[str] = (),
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        ...


class DataReleaseApprovalPort(Protocol):
    """Trusted Human request surface completing the DataFlow/Human cycle."""

    def request_data_release(
        self,
        *,
        pid: str,
        human: str,
        request: dict[str, Any],
        blocking: bool = True,
    ) -> str:
        ...


class HumanDataFlowPort(Protocol):
    """Data-flow decisions required by the Human presentation boundary."""

    RELEASE_BINDING_KEY: str

    def classify_egress_snapshot(
        self,
        *,
        sink: DataSink,
        context: DataFlowContext,
        allow_recovered_source_snapshots: bool = False,
        minimum_integrity: DataIntegrity | str = DataIntegrity.UNTRUSTED,
    ) -> DataFlowOutcome:
        ...

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
        ...

    def resolve_sink_trust(self, sink: DataSink) -> SinkTrustSpec | None:
        ...

    def context_from_source_oids(
        self,
        pid: str,
        source_oids: Iterable[str] | None,
        *,
        include_current: bool = True,
    ) -> DataFlowContext:
        ...

    def precheck_egress_clearance(
        self,
        *,
        pid: str,
        sink: DataSink,
        context: DataFlowContext | None,
        payload: Any,
        minimum_integrity: DataIntegrity | str = DataIntegrity.UNTRUSTED,
    ) -> DataFlowDecision:
        ...

    def observe_ingress(self, context: DataFlowContext) -> DataFlowContext:
        ...
