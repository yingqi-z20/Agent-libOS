from __future__ import annotations

import hashlib
from typing import Any

from agent_libos.models import ProcessMessage
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.serde import dumps


def task_run_message_evidence_projection(
    message: ProcessMessage,
) -> dict[str, Any] | None:
    """Return a content-free evidence projection for a Run-bound message.

    Task Run message bodies remain readable only in stores that terminal
    cleanup can remove. Events and audit records are append-only, so they must
    receive hashes at admission time instead of plaintext that would outlive a
    ``purge_on_terminal`` Run.
    """

    run_id = message.metadata.get("task_run_id")
    if run_id is None:
        return None
    if not isinstance(run_id, str) or not run_id:
        raise ValidationError("TaskRun process message evidence has an invalid run id")
    return {
        "task_run_id": run_id,
        "message_id": message.message_id,
        "sender": message.sender,
        "recipient_pid": message.recipient_pid,
        "kind": message.kind.value,
        "channel": message.channel,
        "correlation_id": message.correlation_id,
        "reply_to": message.reply_to,
        "subject_sha256": _text_sha256(message.subject),
        "body_sha256": _text_sha256(message.body),
        "payload_sha256": _canonical_sha256(message.payload),
        "data_labels": message.metadata.get("data_labels"),
    }


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


__all__ = ["task_run_message_evidence_projection"]
