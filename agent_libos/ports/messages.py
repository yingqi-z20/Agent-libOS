from __future__ import annotations

from typing import Any, Iterable, Protocol

from agent_libos.models import ProcessMessage, ProcessMessageKind, ProcessWaitState


class ProcessMessagePort(Protocol):
    """Narrow process-message sink used by Human delivery."""

    def post(
        self,
        *,
        sender: str,
        recipient_pid: str,
        kind: ProcessMessageKind | str = ProcessMessageKind.NORMAL,
        channel: str = "default",
        correlation_id: str | None = None,
        reply_to: str | None = None,
        subject: str = "",
        body: str = "",
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source_oids: Iterable[str] | None = None,
    ) -> ProcessMessage:
        ...

    def unread(
        self,
        pid: str,
        *,
        kind: ProcessMessageKind | str | None = None,
        sender: str | None = None,
        channel: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        message_ids: list[str] | None = None,
    ) -> list[ProcessMessage]:
        ...

    def notice(
        self,
        pid: str,
        *,
        kind: ProcessMessageKind | str,
        phase: str,
        source: str = "runtime",
        instruction: str | None = None,
    ) -> dict[str, Any] | None:
        ...


class CheckpointMessagePort(Protocol):
    """Message-wait reconciliation needed while restoring a checkpoint."""

    def has_matching_unread_wait(
        self,
        pid: str,
        wait_state: ProcessWaitState | None,
    ) -> bool:
        ...
