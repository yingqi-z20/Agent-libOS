from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import AgentRating
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.storage import ProcessRepository
from agent_libos.utils.ids import new_id, utc_now


class AgentRatingManager:
    def __init__(self, store: ProcessRepository, audit: AuditManager, *, config: AgentLibOSConfig | None = None):
        self.store = store
        self.audit = audit
        self.config = config or DEFAULT_CONFIG

    def get(self, pid: str, *, rater: str | None = None, source: str = "gui") -> AgentRating | None:
        self._require_process(pid)
        return self.store.get_agent_rating(pid, rater or self.config.runtime.default_human, source)

    def get_many(
        self,
        pids: Iterable[str],
        *,
        rater: str | None = None,
        source: str = "gui",
    ) -> dict[str, AgentRating]:
        selected = sorted({str(pid) for pid in pids if str(pid)})
        if not selected:
            return {}
        return self.store.get_agent_ratings_for_processes(
            selected,
            rater=rater or self.config.runtime.default_human,
            source=source,
        )

    def upsert(
        self,
        pid: str,
        *,
        score: int,
        comment: str = "",
        rater: str | None = None,
        source: str = "gui",
        metadata: dict[str, Any] | None = None,
    ) -> AgentRating:
        self._require_process(pid)
        selected_rater = rater or self.config.runtime.default_human
        selected_comment = self._validate_comment(comment)
        selected_score = self._validate_score(score)
        now = utc_now()
        rating = AgentRating(
            rating_id=new_id("rating"),
            pid=pid,
            score=selected_score,
            comment=selected_comment,
            rater=selected_rater,
            source=source,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        # The upsert, exact same-key readback performed by the repository, and
        # required audit evidence share one commit/serialization boundary.
        with self.store.transaction():
            saved = self.store.upsert_agent_rating(rating)
            self.audit.record(
                actor=f"human:{selected_rater}",
                action="agent.rating.upsert",
                target=f"process:{pid}",
                decision={
                    "rating_id": saved.rating_id,
                    "score": saved.score,
                    "source": saved.source,
                    "comment_chars": len(saved.comment),
                },
            )
        return saved

    def _require_process(self, pid: str) -> None:
        if self.store.get_process(pid) is None:
            raise NotFound(f"process not found: {pid}")

    def _validate_score(self, score: int) -> int:
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValidationError("agent rating score must be an integer from 1 to 5")
        if score < 1 or score > 5:
            raise ValidationError("agent rating score must be between 1 and 5")
        return score

    def _validate_comment(self, comment: str) -> str:
        if not isinstance(comment, str):
            raise ValidationError("agent rating comment must be a string")
        limit = self.config.gui.agent_rating_comment_max_chars
        if len(comment) > limit:
            raise ValidationError(f"agent rating comment must be at most {limit} characters")
        return comment
