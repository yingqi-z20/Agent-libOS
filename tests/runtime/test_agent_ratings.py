from __future__ import annotations

from dataclasses import replace

import pytest

from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.runtime.runtime import Runtime


class TestAgentRatings:
    def test_agent_rating_persists_updates_and_audits(self, tmp_path) -> None:
        db = tmp_path / "ratings.sqlite"
        runtime = Runtime.open(db)
        try:
            pid = runtime.process.spawn(goal="rate this agent")
            first = runtime.ratings.upsert(pid, score=4, comment="useful")
            second = runtime.ratings.upsert(pid, score=5, comment="excellent")

            assert second.rating_id == first.rating_id
            assert second.created_at == first.created_at
            assert second.updated_at >= first.updated_at
            assert second.score == 5
            assert second.comment == "excellent"
            assert second.rater == DEFAULT_CONFIG.runtime.default_human
            assert second.source == "gui"
            assert any(
                record.action == "agent.rating.upsert"
                and record.actor == f"human:{DEFAULT_CONFIG.runtime.default_human}"
                and record.target == f"process:{pid}"
                and record.decision
                and record.decision["score"] == 5
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

        reopened = Runtime.open(db)
        try:
            rating = reopened.ratings.get(pid)
            assert rating is not None
            assert rating.rating_id == first.rating_id
            assert rating.score == 5
            assert rating.comment == "excellent"
        finally:
            reopened.close()

    def test_agent_rating_rejects_invalid_process_score_and_comment(self) -> None:
        config = AgentLibOSConfig(gui=replace(DEFAULT_CONFIG.gui, agent_rating_comment_max_chars=4))
        runtime = Runtime.open("local", config=config)
        try:
            pid = runtime.process.spawn(goal="validate rating")
            with pytest.raises(NotFound, match="process not found"):
                runtime.ratings.upsert("missing", score=3)
            with pytest.raises(ValidationError, match="between 1 and 5"):
                runtime.ratings.upsert(pid, score=0)
            with pytest.raises(ValidationError, match="integer"):
                runtime.ratings.upsert(pid, score=True)
            with pytest.raises(ValidationError, match="at most 4 characters"):
                runtime.ratings.upsert(pid, score=3, comment="too long")
        finally:
            runtime.close()
