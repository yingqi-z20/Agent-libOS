from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.config import SemanticDefaults
from agent_libos.models.exceptions import ValidationError
from agent_libos.semantic.service import SemanticManager
from agent_libos.storage import SemanticHumanOutcomeLinkRecord


_DIGEST = "a" * 64
_NOW = "2026-08-10T00:00:00+00:00"


@dataclass(frozen=True)
class _Assessment:
    assessment_id: str = "assessment-1"
    job_id: str = "job-1"
    request_id: str = "request-1"
    pid: str = "pid-1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "job_id": self.job_id,
            "request_id": self.request_id,
            "pid": self.pid,
            # This immutable snapshot was completed before the Human response.
            "human_outcome": "pending",
        }


@dataclass(frozen=True)
class _Settlement:
    settlement_id: str = "settlement-1"
    assessment_id: str = "assessment-1"
    job_id: str = "job-1"
    request_id: str = "request-1"
    pid: str = "pid-1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "settlement_id": self.settlement_id,
            "assessment_id": self.assessment_id,
            "job_id": self.job_id,
            "request_id": self.request_id,
            "pid": self.pid,
        }


def _link(*, request_id: str = "request-1") -> SemanticHumanOutcomeLinkRecord:
    return SemanticHumanOutcomeLinkRecord(
        link_id="link-1",
        request_id=request_id,
        request_revision=1,
        pid="pid-1",
        assessment_id="assessment-1",
        job_id="job-1",
        settlement_id="settlement-1",
        outcome="approved",
        source="machine_policy",
        decision_sha256=_DIGEST,
        created_at=_NOW,
    )


class _Repository:
    def __init__(self, *, link: SemanticHumanOutcomeLinkRecord | None = None) -> None:
        self.assessment = _Assessment()
        self.settlement = _Settlement()
        self.link = link or _link()
        self.assessment_joins: list[tuple[str, ...]] = []
        self.settlement_joins: list[tuple[str, ...]] = []

    def query_semantic_assessments(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(records=(self.assessment,), next_cursor=None)

    def get_semantic_assessment(self, assessment_id: str) -> _Assessment | None:
        return self.assessment if assessment_id == self.assessment.assessment_id else None

    def semantic_human_outcome_links_for_assessments(
        self, assessment_ids: tuple[str, ...]
    ) -> dict[str, SemanticHumanOutcomeLinkRecord]:
        self.assessment_joins.append(assessment_ids)
        return {self.assessment.assessment_id: self.link}

    def query_semantic_machine_settlements(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(records=(self.settlement,), next_cursor=None)

    def semantic_human_outcome_links_for_settlements(
        self, settlement_ids: tuple[str, ...]
    ) -> dict[str, SemanticHumanOutcomeLinkRecord]:
        self.settlement_joins.append(settlement_ids)
        return {self.settlement.settlement_id: self.link}


def _manager(repository: _Repository) -> SemanticManager:
    return SemanticManager(
        repository,
        config=SemanticDefaults(mode="off", adapter="deterministic"),
    )


def test_human_outcome_link_is_batch_joined_into_assessment_and_settlement_views() -> None:
    repository = _Repository()
    manager = _manager(repository)

    assessment_page = manager.query_assessments(limit=10)
    assessment = manager.get_assessment("assessment-1")
    settlement_page = manager.query_machine_settlements(limit=10)

    assert assessment_page["items"][0]["human_outcome"] == "approved"
    assert assessment is not None
    assert assessment["human_outcome"] == "approved"
    settlement = settlement_page["items"][0]
    assert settlement["human_outcome"] == "approved"
    assert settlement["human_outcome_source"] == "machine_policy"
    assert settlement["human_outcome_request_revision"] == 1
    assert settlement["human_outcome_decision_sha256"] == _DIGEST
    assert settlement["human_outcome_created_at"] == repository.link.created_at
    assert repository.assessment_joins == [
        ("assessment-1",),
        ("assessment-1",),
    ]
    assert repository.settlement_joins == [("settlement-1",)]


def test_human_outcome_join_binding_mismatch_fails_closed_without_echo() -> None:
    sentinel = "RAW_SECRET_REQUEST_SENTINEL"
    manager = _manager(_Repository(link=_link(request_id=sentinel)))

    with pytest.raises(ValidationError) as raised:
        manager.query_assessments(limit=10)

    assert sentinel not in str(raised.value)
