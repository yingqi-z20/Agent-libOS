"""After a Runtime reopen the prompt names what the process already did.

Tool-result payloads are released on reopen, so the model used to see only
opaque omitted identifiers and re-read every file it had already read or
written.  The digest is rebuilt from durable events and carries paths, argv,
return codes, Skill ids, and counts, never tool output.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_libos import Runtime
from agent_libos.llm.reopen_digest import (
    REOPEN_DIGEST_HEADING,
    collect_pre_reopen_events,
    context_lost_earlier_results,
    render_reopen_activity_digest,
)
from agent_libos.models import CapabilityRight, Event, EventPriority, EventType
from agent_libos.substrate import LocalResourceProviderSubstrate
from tests.support.fakes import RecordingActionClient


def _event(
    index: int,
    event_type: EventType,
    payload: dict,
    *,
    source: str = "pid_a",
    target: str | None = "filesystem:workspace:x",
) -> Event:
    return Event(
        event_id=f"evt_{index:04d}",
        type=event_type,
        source=source,
        target=target,
        payload=payload,
        priority=EventPriority.NORMAL,
        created_at=f"2026-09-07T12:00:{index:02d}.000000+00:00",
    )


def _fake_store(events: list[Event]):
    ordered = sorted(events, key=lambda event: (event.created_at, event.event_id))

    def list_events(*, limit: int, before_event_id: str | None = None) -> list[Event]:
        selected = ordered
        if before_event_id is not None:
            cursor = next(event for event in ordered if event.event_id == before_event_id)
            selected = [
                event
                for event in ordered
                if (event.created_at, event.event_id) < (cursor.created_at, cursor.event_id)
            ]
        return selected[-limit:]

    return list_events


def test_manifest_signature_selects_only_lost_results() -> None:
    assert context_lost_earlier_results(
        [{"oid": "a", "disposition": "omitted", "reason": "capability_denied"}]
    )
    assert context_lost_earlier_results(
        [{"oid": "a", "disposition": "omitted", "reason": "missing"}]
    )
    assert not context_lost_earlier_results(
        [
            {"oid": "a", "disposition": "omitted", "reason": "superseded"},
            {"oid": "b", "disposition": "omitted", "reason": "token_budget"},
            {"oid": "c", "disposition": "included", "reason": "selected"},
        ]
    )
    assert not context_lost_earlier_results(None)


def test_collection_pages_backwards_and_keeps_only_pre_shutdown_process_events() -> None:
    events = [
        _event(1, EventType.EXTERNAL_READ, {"adapter": "filesystem", "path": "AGENTS.md"}),
        _event(2, EventType.EXTERNAL_READ, {"adapter": "filesystem", "path": "other.md"}, source="pid_b"),
        _event(3, EventType.EXTERNAL_WRITE, {"adapter": "filesystem", "path": "src/a.py", "bytes_written": 10}),
        _event(4, EventType.RESOURCE_CHARGED, {"units": 1}, target="pid_a"),
        _event(5, EventType.RUNTIME_SHUTDOWN, {}, source="runtime", target=None),
        _event(6, EventType.EXTERNAL_READ, {"adapter": "filesystem", "path": "after.md"}),
    ]

    selected = collect_pre_reopen_events(
        _fake_store(events), "pid_a", scan_limit=100, page_size=2
    )

    assert [event.event_id for event in selected] == ["evt_0001", "evt_0003"]
    assert collect_pre_reopen_events(
        _fake_store(events[:4]), "pid_a", scan_limit=100, page_size=2
    ) == [], "without a recorded shutdown nothing was lost"
    partial = collect_pre_reopen_events(
        _fake_store(events), "pid_a", scan_limit=4, page_size=2
    )
    assert [event.event_id for event in partial] == ["evt_0003"], (
        "a bounded scan yields a partial digest, never events after the shutdown"
    )


def test_rendering_is_bounded_and_payload_free() -> None:
    events = [
        _event(1, EventType.SKILL_LOADED, {"skill_id": "agent-libos-workspace-editing", "tool_names": ["write_text_file"]}),
        _event(2, EventType.EXTERNAL_READ, {"adapter": "filesystem", "path": ".", "operation": "read_directory"}),
        _event(3, EventType.EXTERNAL_READ, {"adapter": "filesystem", "path": "AGENTS.md"}),
        _event(4, EventType.EXTERNAL_READ, {"adapter": "filesystem", "path": "AGENTS.md"}),
        _event(5, EventType.EXTERNAL_WRITE, {"adapter": "filesystem", "path": "src/a.py", "bytes_written": 120, "created": False}),
        _event(6, EventType.EXTERNAL_WRITE, {"adapter": "filesystem", "path": "src/a.py", "bytes_written": 130, "created": False}),
        _event(7, EventType.EXTERNAL_WRITE, {"adapter": "shell", "argv": ["python", "-m", "unittest"], "returncode": 1, "operation": "run"}),
        _event(8, EventType.EXTERNAL_WRITE, {"adapter": "shell", "argv": ["python", "-m", "unittest"], "returncode": 0, "operation": "run"}),
        _event(9, EventType.EXTERNAL_READ, {"adapter": "git", "operation": "status"}),
        _event(10, EventType.EXTERNAL_WRITE, {"adapter": "llm", "profile_id": "default", "status": "ok"}),
        _event(11, EventType.CHECKPOINT_CREATED, {"checkpoint_id": "ckpt_1", "reason": "SECRET_REASON"}),
    ]

    digest = render_reopen_activity_digest(events)

    assert digest.startswith(REOPEN_DIGEST_HEADING)
    assert "- files read: AGENTS.md" in digest
    assert "- directories listed: ." in digest
    assert "- files written: src/a.py (130 B x2)" in digest
    assert "- commands run: python -m unittest -> returncode 0 x2" in digest
    assert "- git inspections: status x1" in digest
    assert "- skills activated: agent-libos-workspace-editing" in digest
    assert "- checkpoints created: 1" in digest
    assert "SECRET_REASON" not in digest
    assert "llm" not in digest.split(REOPEN_DIGEST_HEADING, 1)[1].split("- guidance")[0]
    assert "Re-read only the files you must edit" in digest
    assert render_reopen_activity_digest([]) == ""

    many = [
        _event(index, EventType.EXTERNAL_READ, {"adapter": "filesystem", "path": f"pkg/module_{index:03d}.py"})
        for index in range(1, 60)
    ]
    bounded = render_reopen_activity_digest(many)
    assert "+19 more" in bounded
    assert len(bounded) < 6_000


def test_prompt_after_reopen_names_earlier_reads_writes_and_commands() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "workspace"
        root.mkdir()
        (root / "AGENTS.md").write_text("follow me\n", encoding="utf-8")
        db = str(Path(tmp) / "runtime.sqlite")
        runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
        # Results dispatched through the model path become MemoryView roots,
        # which is what the reopen releases; direct broker calls would not.
        runtime.llm.client = RecordingActionClient(
            [
                {"action": "read_text_file", "path": "AGENTS.md"},
                {"action": "write_text_file", "path": "notes.md", "content": "PRIVATE_NOTE_BODY\n"},
            ]
        )
        try:
            pid = runtime.process.spawn(image="coding-agent:v0", goal="survive a reopen")
            runtime.activate_skill(pid, "agent-libos-workspace-navigation")
            runtime.activate_skill(pid, "agent-libos-workspace-editing")
            runtime.filesystem.grant_directory(
                pid, ".", [CapabilityRight.READ, CapabilityRight.WRITE], issued_by="test"
            )
            first = runtime.run_process_once(pid)
            assert first["ok"] is True, first
            second = runtime.run_process_once(pid)
            assert second["ok"] is True, second
        finally:
            runtime.close()

        reopened = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
        client = RecordingActionClient([{"action": "get_current_time", "timezone": "UTC"}])
        reopened.llm.client = client
        try:
            reopened.activate_skill(pid, "agent-libos-runtime-session")
            reopened.capability.grant(pid, "clock:now", [CapabilityRight.READ], issued_by="test")

            reopened.run_process_once(pid)

            prompt = client.user_prompts[0]
            assert "capability_denied=" in prompt or "missing=" in prompt, prompt[-2000:]
            assert REOPEN_DIGEST_HEADING in prompt
            digest = prompt.split(REOPEN_DIGEST_HEADING, 1)[1].split("\n\n", 1)[0]
            assert "AGENTS.md" in digest
            assert "notes.md (" in digest
            assert "skills activated: agent-libos-workspace-navigation" in digest
            assert "PRIVATE_NOTE_BODY" not in prompt, "the digest never carries payloads"
            assert "follow me" not in prompt
        finally:
            reopened.close()


def test_prompt_without_lost_results_has_no_digest() -> None:
    client = RecordingActionClient([{"action": "get_current_time", "timezone": "UTC"}])
    runtime = Runtime.open("local")
    runtime.llm.client = client
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="fresh start")
        runtime.activate_skill(pid, "agent-libos-runtime-session")
        runtime.capability.grant(pid, "clock:now", [CapabilityRight.READ], issued_by="test")

        runtime.run_process_once(pid)

        assert REOPEN_DIGEST_HEADING not in client.user_prompts[0]
    finally:
        runtime.close()
