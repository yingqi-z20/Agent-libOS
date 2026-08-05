from __future__ import annotations

from typing import Protocol

from agent_libos.models.semantic import SemanticAssessment, SemanticAssessmentRequest


class SemanticAssessmentPort(Protocol):
    """Host-owned synchronous classifier surface; implementations return no permit."""

    def assess(self, request: SemanticAssessmentRequest) -> SemanticAssessment:
        ...


__all__ = ["SemanticAssessmentPort"]
