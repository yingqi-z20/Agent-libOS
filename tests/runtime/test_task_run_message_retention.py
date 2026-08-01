from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime, TaskRunSpecV1
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import EventType, TaskRunStatus
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.serde import dumps


_SUBJECT_CANARY = "TASK_RUN_MESSAGE_SUBJECT_PRIVATE_SENTINEL"
_BODY_CANARY = "TASK_RUN_MESSAGE_BODY_PRIVATE_SENTINEL"
_PAYLOAD_CANARY = "TASK_RUN_MESSAGE_PAYLOAD_PRIVATE_SENTINEL"
_FOLLOW_UP_CANARY = "TASK_RUN_FOLLOW_UP_PRIVATE_SENTINEL"


def _config():
    return replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def _logical_sqlite_dump(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        return "\n".join(connection.iterdump())


def test_run_recipient_messages_are_host_bound_and_terminally_purged(
    tmp_path: Path,
) -> None:
    database = tmp_path / "messages.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(goal="receive private input", display_title="Messages"),
            client_request_id="create-message-retention",
        )
        assert created.root_pid is not None

        posted = runtime.human.send_process_message(
            created.root_pid,
            _BODY_CANARY,
            subject=_SUBJECT_CANARY,
            payload={"private": _PAYLOAD_CANARY},
        )
        assert posted.metadata["task_run_id"] == created.run_id
        assert posted.subject == _SUBJECT_CANARY
        assert posted.body == _BODY_CANARY
        assert posted.payload["private"] == _PAYLOAD_CANARY
        assert runtime.store.list_process_messages(created.root_pid) == [posted]

        posted_event = next(
            event
            for event in runtime.events.list(target=created.root_pid)
            if event.type is EventType.PROCESS_MESSAGE_POSTED
            and event.payload.get("message_id") == posted.message_id
        )
        message_audits = [
            record
            for record in runtime.audit.trace(target=f"process:{created.root_pid}")
            if record.action in {"process.message.post", "human.message"}
            and record.decision is not None
            and record.decision.get("message_id") == posted.message_id
        ]
        assert {record.action for record in message_audits} == {
            "process.message.post",
            "human.message",
        }
        projections = [
            posted_event.payload,
            *(record.decision for record in message_audits if record.decision is not None),
        ]
        for projection in projections:
            assert "subject" not in projection
            assert "body" not in projection
            assert "payload" not in projection
            assert projection["task_run_id"] == created.run_id
            assert projection["subject_sha256"] == _text_sha256(posted.subject)
            assert projection["body_sha256"] == _text_sha256(posted.body)
            assert projection["payload_sha256"] == _canonical_sha256(posted.payload)
            serialized = dumps(projection)
            assert _SUBJECT_CANARY not in serialized
            assert _BODY_CANARY not in serialized
            assert _PAYLOAD_CANARY not in serialized

        terminal = runtime.task_runs.cancel(
            created.run_id,
            expected_revision=created.revision,
            command_id="cancel-message-retention",
        )
        assert terminal.status is TaskRunStatus.CANCELLED
        assert runtime.store.list_process_messages(created.root_pid) == []
    finally:
        runtime.close()

    logical_store = _logical_sqlite_dump(database)
    assert _SUBJECT_CANARY not in logical_store
    assert _BODY_CANARY not in logical_store
    assert _PAYLOAD_CANARY not in logical_store


def test_follow_up_content_survives_reopen_then_leaves_no_terminal_canary(
    tmp_path: Path,
) -> None:
    database = tmp_path / "follow-up.sqlite"
    runtime = Runtime.open(database, config=_config())
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(goal="accept a later requirement", display_title="Follow-up"),
            client_request_id="create-follow-up-retention",
        )
        assert created.root_pid is not None
        followed = runtime.task_runs.follow_up(
            created.run_id,
            {"instruction": _FOLLOW_UP_CANARY},
            expected_revision=created.revision,
            command_id="append-private-follow-up",
        )
        messages = runtime.store.list_process_messages(created.root_pid)
        assert len(messages) == 1
        assert messages[0].metadata["task_run_id"] == created.run_id
        assert messages[0].payload["run_id"] == created.run_id
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=_config())
    try:
        current = reopened.task_runs.get(followed.run_id)
        assert current.root_pid is not None
        messages = reopened.store.list_process_messages(current.root_pid)
        assert len(messages) == 1
        assert messages[0].metadata["task_run_id"] == followed.run_id

        prompt_context = reopened.task_runs.prompt_context_for_pid(current.root_pid)
        assert prompt_context is not None
        assert any(
            _FOLLOW_UP_CANARY in requirement["content_text"]
            for requirement in prompt_context["requirements"]
        )

        terminal = reopened.task_runs.cancel(
            followed.run_id,
            expected_revision=current.revision,
            command_id="cancel-follow-up-retention",
        )
        assert terminal.status is TaskRunStatus.CANCELLED
        assert reopened.store.list_process_messages(current.root_pid) == []
    finally:
        reopened.close()

    assert _FOLLOW_UP_CANARY not in _logical_sqlite_dump(database)


def test_ordinary_message_keeps_existing_plaintext_subject_evidence() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="ordinary message evidence")
        posted = runtime.human.send_process_message(
            pid,
            "ordinary body",
            subject="ordinary subject",
            payload={"ordinary": "payload"},
        )
        event = next(
            item
            for item in runtime.events.list(target=pid)
            if item.type is EventType.PROCESS_MESSAGE_POSTED
            and item.payload.get("message_id") == posted.message_id
        )
        assert event.payload["subject"] == "ordinary subject"
        assert "subject_sha256" not in event.payload

        audits = [
            record
            for record in runtime.audit.trace(target=f"process:{pid}")
            if record.action in {"process.message.post", "human.message"}
            and record.decision is not None
            and record.decision.get("message_id") == posted.message_id
        ]
        assert {record.action for record in audits} == {
            "process.message.post",
            "human.message",
        }
        assert all(record.decision["subject"] == "ordinary subject" for record in audits)
        assert all("subject_sha256" not in record.decision for record in audits)
    finally:
        runtime.close()


def test_ordinary_recipient_rejects_forged_task_run_message_binding() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="ordinary message recipient")
        with pytest.raises(ValidationError, match="reserved"):
            runtime.messages.post(
                sender="host",
                recipient_pid=pid,
                metadata={"task_run_id": "run_forged"},
            )
        assert runtime.store.list_process_messages(pid) == []
    finally:
        runtime.close()
