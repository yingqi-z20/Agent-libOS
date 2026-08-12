from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from agent_libos.models.human import HumanRequest
from agent_libos.models.semantic import (
    SemanticApprovalCandidate,
    SemanticAssessment,
    SemanticAssessmentRequest,
)

if TYPE_CHECKING:
    from agent_libos.capability import Capability
    from agent_libos.models.semantic import (
        MachinePolicySettlementV1,
        SemanticApprovalBindingV2,
    )

class SemanticAssessmentPort(Protocol):
    """Host-owned synchronous classifier surface; implementations return no permit."""

    def assess(self, request: SemanticAssessmentRequest) -> SemanticAssessment:
        ...


class SemanticAutoApprovalSettlementPort(Protocol):
    """Host-only exact-once authority settlement surface.

    Implementations must run inside the caller's shared UnitOfWork transaction
    and revalidate every live Host predicate before delegating to the common
    Human terminalization kernel.  The classifier never receives this port.
    """

    def settle_exact_once(
        self,
        *,
        request_id: str,
        expected_revision: int,
        job_id: str,
        assessment_id: str,
        assessment: SemanticAssessment,
        candidate: SemanticApprovalCandidate,
        semantic_terminalizer: Callable[
            [
                "Capability",
                "SemanticApprovalBindingV2",
                "MachinePolicySettlementV1",
            ],
            bool,
        ],
    ) -> tuple[HumanRequest, dict[str, Any]]:
        ...


class SemanticDenySettlementPort(Protocol):
    """Host-only deterministic-deny terminalization surface."""

    def settle_deny(
        self,
        *,
        request_id: str,
        expected_revision: int,
        decision: Any,
        semantic_terminalizer: Callable[[], bool],
    ) -> tuple[HumanRequest, dict[str, Any]]:
        ...


class SemanticFlowPort(Protocol):
    """Payload-free Phase 2 flow evidence and bounded query surface."""

    def capture_root_goal(self, **facts: Any) -> Any:
        ...

    def capture_provider_ingress(self, **facts: Any) -> Any:
        ...

    def capture_derived_entity(self, **facts: Any) -> Any:
        ...

    def capture_activity(self, **facts: Any) -> Any:
        ...

    def capture_file_version(self, **facts: Any) -> Any:
        ...

    def capture_git_snapshot(self, **facts: Any) -> Any:
        ...

    def append_assessment_findings(self, **facts: Any) -> Any:
        ...

    def approval_eligibility(self, **facts: Any) -> Any:
        ...

    def coverage(self, entity_id: str, **filters: Any) -> Any:
        ...

    def memory_gate(self, entity_id: str, **facts: Any) -> Any:
        ...

    def flow_status(self) -> dict[str, Any]:
        ...

    def query_flow_entities(self, **filters: Any) -> dict[str, Any]:
        ...

    def query_flow_activities(self, **filters: Any) -> dict[str, Any]:
        ...

    def query_flow_edges(self, **filters: Any) -> dict[str, Any]:
        ...

    def query_flow_lineage(self, node_id: str, **filters: Any) -> dict[str, Any]:
        ...


__all__ = [
    "SemanticAssessmentPort",
    "SemanticAutoApprovalSettlementPort",
    "SemanticDenySettlementPort",
    "SemanticFlowPort",
]
