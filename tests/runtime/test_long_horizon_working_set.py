"""Runtime behavior that keeps a long task observable and bounded.

These tests cover the ``working_set`` feedback window and supersession stubs,
the admission-derived materialization budget, the early surfacing of queued
ordinary process input, the batch-truncation event, and JSON-string container
argument repair.  They are deterministic and token-free.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.llm.context_management import estimate_request_input_tokens
from agent_libos.llm.prompt import FEEDBACK_STUB_RECORD_TYPE, build_system_prompt
from agent_libos.memory.object_memory import (
    _feedback_stub_summary,
    _observation_supersession_key,
)
from agent_libos.models import CapabilityRight, EventType, ObjectType, ProcessMessageKind
from agent_libos.tools.broker import _normalize_scalar_string
from tests.support.fakes import RecordingActionClient


def _tool_result(runtime: Runtime, pid: str, tool_name: str, result: dict[str, Any]):
    if tool_name == "read_text_file":
        result = {"encoding": "utf-8", "truncated": False, **result}
    elif tool_name == "read_directory":
        result = {"truncated": False, **result}
    return runtime.memory.create_object(
        pid,
        ObjectType.TOOL_RESULT,
        {"tool_name": tool_name, "result": result},
    )


class _MultiCallClient:
    def __init__(self, batches: list[list[dict[str, Any]]]) -> None:
        self.batches = list(batches)
        self.user_prompts: list[str] = []

    def complete_action(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMCompletion:
        self.user_prompts.append(str(messages[-1]["content"]))
        batch = self.batches.pop(0)
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": f"call_{len(self.user_prompts)}_{index}",
                    "name": str(action["action"]),
                    "arguments": json.dumps(
                        {key: value for key, value in action.items() if key != "action"}
                    ),
                }
                for index, action in enumerate(batch)
            ],
        )


def test_working_set_supersedes_repeated_observations_and_stubs_older_feedback() -> None:
    config = replace(
        DEFAULT_CONFIG,
        memory=replace(
            DEFAULT_CONFIG.memory,
            working_set_recent_feedback=2,
            working_set_verbatim_feedback_tokens=0,
        ),
    )
    runtime = Runtime.open("local", config=config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="bounded feedback")
        goal = runtime.process.get(pid).memory_view.roots[0]
        stale_read = _tool_result(
            runtime, pid, "read_text_file", {"path": "src/a.py", "content": "OLD_CONTENT_A"}
        )
        listing = _tool_result(
            runtime,
            pid,
            "read_directory",
            {"path": ".", "entries": [{"name": "LISTED_ENTRY.py"}, {"name": "other.py"}]},
        )
        follow_up = _tool_result(
            runtime,
            pid,
            "read_process_messages",
            {"messages": [{"subject": "Customer follow-up", "body": "FOLLOW_UP_CONSTRAINT"}]},
        )
        fresh_read = _tool_result(
            runtime, pid, "read_text_file", {"path": "src/a.py", "content": "NEW_CONTENT_A"}
        )
        tests = _tool_result(
            runtime,
            pid,
            "run_shell_command",
            {"argv": ["python", "-m", "unittest"], "returncode": 1, "stdout": "FAILED_OUTPUT"},
        )
        view = runtime.memory.create_view(
            pid, [goal, stale_read, listing, follow_up, fresh_read, tests]
        )

        context = runtime.memory.materialize_context(
            pid, view, policy="working_set", budget_tokens=100_000, charge_resources=False
        )

        text = context.text
        assert "NEW_CONTENT_A" in text
        assert "FAILED_OUTPUT" in text
        assert "FOLLOW_UP_CONSTRAINT" in text, "human input results are never stubbed"
        assert "OLD_CONTENT_A" not in text, "a fresher read of the same path supersedes"
        assert "LISTED_ENTRY.py" not in text, "feedback outside the window is stubbed"
        assert FEEDBACK_STUB_RECORD_TYPE in text
        assert '"stub_reason":"older_feedback"' in text
        assert '"entries_count":2' in text
        assert context.object_refs == [
            goal.oid, listing.oid, follow_up.oid, fresh_read.oid, tests.oid
        ], "stubs are included in root order; only the superseded copy is omitted"
        assert context.omitted_objects == [stale_read.oid]
        reasons = {entry["oid"]: entry["reason"] for entry in context.object_manifest}
        assert reasons[stale_read.oid] == "superseded"
        transforms = {entry["oid"]: entry["transform"] for entry in context.object_manifest}
        assert transforms[listing.oid] == "compacted"
        assert transforms[follow_up.oid] == "tool_result_projection_v1"
        assert transforms[fresh_read.oid] == "tool_result_projection_v1"
        assert transforms[tests.oid] == "tool_result_projection_v1"
        stub_tokens = sum(
            entry["tokens"] for entry in context.object_manifest if entry["transform"] == "compacted"
        )
        assert stub_tokens < 200
    finally:
        runtime.close()


def test_working_set_stubs_apply_only_to_that_policy_and_keep_stable_prefix() -> None:
    config = replace(
        DEFAULT_CONFIG,
        memory=replace(
            DEFAULT_CONFIG.memory,
            working_set_recent_feedback=1,
            working_set_verbatim_feedback_tokens=0,
        ),
    )
    runtime = Runtime.open("local", config=config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="prefix stability")
        goal = runtime.process.get(pid).memory_view.roots[0]
        first = _tool_result(runtime, pid, "git_status", {"changed_paths": ["a.py"]})
        before = runtime.memory.materialize_context(
            pid,
            runtime.memory.create_view(pid, [goal, first]),
            policy="working_set",
            budget_tokens=100_000,
            charge_resources=False,
        )
        second = _tool_result(runtime, pid, "write_text_file", {"path": "a.py", "bytes_written": 12})
        after = runtime.memory.materialize_context(
            pid,
            runtime.memory.create_view(pid, [goal, first, second]),
            policy="working_set",
            budget_tokens=100_000,
            charge_resources=False,
        )
        # The first result left the window, so its bytes change exactly once and
        # everything before it (goal) stays a byte-identical prefix.
        goal_prefix = before.text.split("\n\n", 1)[0]
        assert after.text.startswith(goal_prefix)
        assert FEEDBACK_STUB_RECORD_TYPE in after.text
        assert FEEDBACK_STUB_RECORD_TYPE not in before.text

        plan_first = runtime.memory.materialize_context(
            pid,
            runtime.memory.create_view(pid, [goal, first, second]),
            policy="plan_first",
            budget_tokens=100_000,
            charge_resources=False,
        )
        assert FEEDBACK_STUB_RECORD_TYPE not in plan_first.text
    finally:
        runtime.close()


def test_working_set_token_window_keeps_a_multi_file_orientation_verbatim() -> None:
    """Thirteen file reads must all stay in view when they fit the token window."""

    config = replace(
        DEFAULT_CONFIG,
        memory=replace(
            DEFAULT_CONFIG.memory,
            working_set_recent_feedback=1,
            working_set_verbatim_feedback_tokens=20_000,
        ),
    )
    runtime = Runtime.open("local", config=config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="orientation")
        goal = runtime.process.get(pid).memory_view.roots[0]
        reads = [
            _tool_result(
                runtime, pid, "read_text_file", {"path": f"src/module_{index}.py", "content": f"MODULE_{index} " * 40}
            )
            for index in range(13)
        ]
        context = runtime.memory.materialize_context(
            pid,
            runtime.memory.create_view(pid, [goal, *reads]),
            policy="working_set",
            budget_tokens=100_000,
            charge_resources=False,
        )
        assert FEEDBACK_STUB_RECORD_TYPE not in context.text
        assert all(f"MODULE_{index} " in context.text for index in range(13))

        tight = replace(
            config,
            memory=replace(config.memory, working_set_verbatim_feedback_tokens=2_000),
        )
        tight_runtime = Runtime.open("local", config=tight)
        try:
            tight_pid = tight_runtime.process.spawn(image="base-agent:v0", goal="orientation")
            tight_goal = tight_runtime.process.get(tight_pid).memory_view.roots[0]
            tight_reads = [
                _tool_result(
                    tight_runtime,
                    tight_pid,
                    "read_text_file",
                    {"path": f"src/module_{index}.py", "content": f"MODULE_{index} " * 40},
                )
                for index in range(13)
            ]
            bounded = tight_runtime.memory.materialize_context(
                tight_pid,
                tight_runtime.memory.create_view(tight_pid, [tight_goal, *tight_reads]),
                policy="working_set",
                budget_tokens=100_000,
                charge_resources=False,
            )
            assert "MODULE_12 " in bounded.text, "the newest read is always verbatim"
            assert "MODULE_0 " not in bounded.text, "the oldest read is stubbed beyond the window"
            assert FEEDBACK_STUB_RECORD_TYPE in bounded.text
            assert len(bounded.object_refs) == 14, "stubs are included, not omitted"
        finally:
            tight_runtime.close()
    finally:
        runtime.close()


def test_supersession_keys_cover_observations_but_never_actions_or_failures() -> None:
    assert _observation_supersession_key(
        {"tool_name": "read_text_file", "result": {
            "path": "src/a.py", "content": "x", "encoding": "utf-8", "truncated": False,
        }}
    ) == _observation_supersession_key(
        {"tool_name": "read_text_file", "result": {
            "path": "src/a.py", "content": "y", "encoding": "utf-8", "truncated": False,
        }}
    )
    assert _observation_supersession_key(
        {"tool_name": "read_text_file", "result": {
            "path": "src/a.py", "encoding": "utf-8", "truncated": False,
        }}
    ) != _observation_supersession_key(
        {"tool_name": "read_text_file", "result": {
            "path": "src/b.py", "encoding": "utf-8", "truncated": False,
        }}
    )
    # A shell run is evidence, never superseded: a passing suite must not hide
    # the earlier failing run that proves the defect was reproduced first.
    assert _observation_supersession_key(
        {"tool_name": "run_shell_command", "result": {"argv": ["python", "-m", "unittest"], "returncode": 0}}
    ) is None
    assert _observation_supersession_key(
        {"tool_name": "write_text_file", "result": {"path": "src/a.py"}}
    ) is None
    assert _observation_supersession_key(
        {"tool_name": "read_text_file", "ok": False, "error": {"message": "missing"}}
    ) is None
    assert _observation_supersession_key(
        {"tool_name": "process_exit", "result": {"status": "completion_review_required"}}
    ) == ("process_exit", "completion_review")
    assert _observation_supersession_key(
        {"tool_name": "process_exit", "result": {"status": "exited"}}
    ) is None


def test_feedback_stub_summary_describes_without_copying_content() -> None:
    summary = _feedback_stub_summary(
        {
            "tool_name": "run_shell_command",
            "result": {
                "argv": ["python", "-m", "unittest", "discover", "-s", "tests", "-q"],
                "returncode": 1,
                "stdout": "SECRET " * 500,
                "stderr": "",
            },
        }
    )
    assert summary["tool_name"] == "run_shell_command"
    assert summary["returncode"] == 1
    assert summary["argv"][0] == "python"
    assert summary["stdout_chars"] == len("SECRET " * 500)
    assert "SECRET" not in json.dumps(summary)

    failure = _feedback_stub_summary(
        {"tool_name": "read_text_file", "ok": False, "error": {"message": "file not found: x.py"}}
    )
    assert failure["ok"] is False
    assert failure["error"].startswith("file not found")


def test_materialization_budget_follows_per_call_admission_headroom() -> None:
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(
            DEFAULT_CONFIG.llm,
            max_input_tokens_per_call=60_000,
            max_total_tokens_per_call=76_384,
            context_window_tokens=76_384,
        ),
    )
    runtime = Runtime.open("local", config=config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="headroom")
        process = runtime.process.get(pid)
        image = runtime.images[process.image_id]
        assert process.resource_budget.max_context_materialization_tokens > 60_000

        budget = runtime.llm._materialization_budget_tokens(
            pid, image=image, process=process, skills=[], openai_tools=[]
        )
        overhead = estimate_request_input_tokens(
            [
                {"role": "system", "content": build_system_prompt(image)},
                {"role": "user", "content": ""},
            ],
            [],
        )
        assert budget == 60_000 - overhead - config.llm_context.materialization_headroom_tokens
        assert budget < process.resource_budget.max_context_materialization_tokens

        heavy_skills = [{"instructions": "x" * 400_000, "allowed_tools": []}]
        floored = runtime.llm._materialization_budget_tokens(
            pid, image=image, process=process, skills=heavy_skills, openai_tools=[]
        )
        assert floored == config.llm_context.materialization_budget_floor_tokens
    finally:
        runtime.close()


def test_materialization_budget_never_exceeds_the_process_window() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="window ceiling")
        process = runtime.process.get(pid)
        process.resource_budget = replace(
            process.resource_budget, max_context_materialization_tokens=20_000
        )
        runtime.store.update_process(process)
        process = runtime.process.get(pid)
        image = runtime.images[process.image_id]

        budget = runtime.llm._materialization_budget_tokens(
            pid, image=image, process=process, skills=[], openai_tools=[]
        )

        assert budget == 20_000
    finally:
        runtime.close()


def test_queued_normal_message_directive_appears_before_first_tool_selection() -> None:
    client = RecordingActionClient([{"action": "get_current_time", "timezone": "UTC"}])
    runtime = Runtime.open("local")
    runtime.llm.client = client
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="react to queued input")
        runtime.activate_skill(pid, "agent-libos-runtime-session")
        runtime.capability.grant(pid, "clock:now", [CapabilityRight.READ], issued_by="test")
        runtime.messages.post(
            sender="human:test",
            recipient_pid=pid,
            kind=ProcessMessageKind.NORMAL,
            subject="Customer follow-up",
            body="SECRET_FOLLOW_UP_BODY",
        )

        result = runtime.run_process_once(pid)

        prompt = client.user_prompts[0]
        assert "Pending explicit process input (mandatory control action)" in prompt
        assert "SECRET_FOLLOW_UP_BODY" not in prompt
        # Ordinary input informs the prompt early but does not pre-empt dispatch.
        assert result["action"]["action"] == "get_current_time"
        assert result["result"]["message_notice"]["phase"] == "after_tool_call"
        assert len(runtime.messages.unread(pid, kind=ProcessMessageKind.NORMAL)) == 1
    finally:
        runtime.close()


def test_truncated_batch_reports_unexecuted_calls_to_the_model() -> None:
    client = _MultiCallClient(
        [
            [
                {"action": "read_memory_object", "name": "no-such-ledger"},
                {"action": "get_current_time", "timezone": "UTC"},
            ],
            [{"action": "get_current_time", "timezone": "UTC"}],
        ]
    )
    runtime = Runtime.open("local")
    runtime.llm.client = client
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="observe truncation")
        runtime.activate_skill(pid, "agent-libos-object-memory")
        runtime.activate_skill(pid, "agent-libos-runtime-session")
        runtime.capability.grant(pid, "clock:now", [CapabilityRight.READ], issued_by="test")

        first = runtime.run_process_once(pid)

        assert first["ok"] is True
        assert first["stop_reason"] == "tool_failed"
        assert first["executed_count"] == 1
        assert first["results"][0]["ok"] is False
        truncated = [
            event
            for event in runtime.events.list(target=pid, limit=200)
            if event.type == EventType.TOOL_BATCH_TRUNCATED
        ]
        assert len(truncated) == 1
        payload = truncated[0].payload
        assert payload["stop_reason"] == "tool_failed"
        assert payload["unexecuted_actions"] == ["get_current_time"]
        assert payload["requested_count"] == 2 and payload["executed_count"] == 1

        second = runtime.run_process_once(pid)

        assert second["ok"] is True
        assert "tool_batch_truncated" in client.user_prompts[1]
        assert "get_current_time" in client.user_prompts[1]
    finally:
        runtime.close()


def test_json_string_container_arguments_are_decoded_only_when_schema_forbids_strings() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="argument repair")

        normalized = runtime.tools.normalize_model_action(
            pid,
            {"action": "process_exit", "payload": json.dumps({"summary": "done", "count": 2})},
        )

        assert normalized["payload"] == {"summary": "done", "count": 2}
        records = [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "llm.tool_arguments_normalized"
        ]
        assert records and "payload" in records[-1].decision["normalized_fields"]

        untouched = runtime.tools.normalize_model_action(
            pid,
            {"action": "process_exit", "message": json.dumps({"not": "decoded"})},
        )
        assert untouched["message"] == json.dumps({"not": "decoded"})

        malformed = runtime.tools.normalize_model_action(
            pid,
            {"action": "process_exit", "payload": "{not json"},
        )
        assert malformed["payload"] == "{not json"
    finally:
        runtime.close()


def test_queued_input_directive_survives_an_event_backlog() -> None:
    """A busy quantum's bookkeeping events must not hide the read directive.

    Each tool result emits several events.  With the render window at its
    default the notice raised at quantum start would previously fall outside
    the oldest-first page and the directive only appeared quanta later.
    """

    client = RecordingActionClient([{"action": "get_current_time", "timezone": "UTC"}])
    runtime = Runtime.open("local")
    runtime.llm.client = client
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="backlog")
        runtime.activate_skill(pid, "agent-libos-runtime-session")
        runtime.capability.grant(pid, "clock:now", [CapabilityRight.READ], issued_by="test")
        for index in range(30):
            _tool_result(runtime, pid, "read_text_file", {"path": f"src/file_{index}.py", "content": "x"})
        backlog = runtime.events.list(target=pid, limit=500)
        assert len(backlog) > runtime.config.llm_context.recent_event_limit
        runtime.messages.post(
            sender="human:test",
            recipient_pid=pid,
            kind=ProcessMessageKind.NORMAL,
            subject="Customer follow-up",
            body="SECRET_BACKLOG_BODY",
        )

        runtime.run_process_once(pid)

        prompt = client.user_prompts[0]
        assert "Pending explicit process input (mandatory control action)" in prompt
        assert "SECRET_BACKLOG_BODY" not in prompt
        # The whole backlog was scanned and acknowledged in one quantum.
        process = runtime.process.get(pid)
        remaining = runtime.events.list(target=pid, limit=500, after_event_id=process.event_cursor)
        backlog_ids = {event.event_id for event in backlog}
        assert all(event.event_id not in backlog_ids for event in remaining)
    finally:
        runtime.close()


def test_event_projection_keeps_newest_visible_events_and_counts_bookkeeping() -> None:
    from agent_libos.llm.event_projection import project_prompt_events
    from agent_libos.models import Event, EventPriority

    def event(index: int, event_type: EventType, payload: dict[str, Any]) -> Event:
        return Event(
            event_id=f"evt_{index:04d}",
            type=event_type,
            source="test",
            target="pid_x",
            payload=payload,
            priority=EventPriority.NORMAL,
            created_at=f"2026-09-07T10:00:{index % 60:02d}+00:00",
        )

    events = [
        event(0, EventType.OBJECT_CREATED, {"type": "tool_result", "oid": "obj_a"}),
        event(1, EventType.CAPABILITY_GRANTED, {"resource": "object:obj_a", "rights": ["read"]}),
        *(event(10 + index, EventType.EXTERNAL_WRITE, {"path": f"f{index}"}) for index in range(5)),
        event(20, EventType.CHECKPOINT_CREATED, {"checkpoint_id": "ckpt_1"}),
        event(21, EventType.CAPABILITY_GRANTED, {"resource": "filesystem:workspace:*", "rights": ["read"]}),
    ]

    batch = project_prompt_events(events, max_visible=3)

    visible_types = [record["type"] for record in batch.visible_records]
    assert visible_types == ["external_write", "checkpoint_created", "capability_granted"]
    assert batch.represented_through_event_id == "evt_0021"
    assert batch.omitted_counts["tool_result_object_created"] == 1
    assert batch.omitted_counts["object_capability_granted"] == 1
    assert batch.omitted_counts["recent_window"] == 4
    assert batch.summary["input_event_count"] == len(events)
    assert batch.summary["represented_event_count"] == 3
    assert batch.summary["omitted_event_count"] == len(events) - 3


def test_discovery_supersession_keys_follow_the_surfaced_skills() -> None:
    """Two discovery queries issued in one response must both stay visible.

    Keying every ``discover_skills`` result on the bare tool name let the
    second query's result hide the first, so the model repeated the query and
    read the omission as evidence of unfinished earlier work.
    """

    tool_query = {
        "tool_name": "discover_skills",
        "result": {
            "skills": [
                {"skill_id": "agent-libos-git-inspection"},
                {"skill_id": "agent-libos-workspace-editing"},
            ],
            "has_more": True,
        },
    }
    memory_query = {
        "tool_name": "discover_skills",
        "result": {"skills": [{"skill_id": "agent-libos-object-memory"}], "has_more": False},
    }
    repeat = {
        "tool_name": "discover_skills",
        "result": {
            "skills": [
                {"skill_id": "agent-libos-workspace-editing"},
                {"skill_id": "agent-libos-git-inspection"},
            ],
            "has_more": True,
        },
    }

    assert _observation_supersession_key(tool_query) != _observation_supersession_key(memory_query)
    assert _observation_supersession_key(tool_query) == _observation_supersession_key(repeat)
    assert _observation_supersession_key(
        {"tool_name": "discover_skills", "result": {"skills": "not-a-list"}}
    ) is None


@pytest.mark.parametrize("value", ["null", "None", " null ", "NULL"])
def test_schema_accepted_strings_remain_literal_memory_values(value: str) -> None:
    assert _normalize_scalar_string(value, {"string", "null"}) == value
    assert _normalize_scalar_string(value, {"string"}) == value
    assert _normalize_scalar_string("null", {"integer", "null"}) is None
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="preserve literal data")
        runtime.activate_skill(pid, "agent-libos-object-memory")
        action = runtime.tools.normalize_model_action(pid, {
            "action": "create_memory_object", "name": "literal", "type": "artifact",
            "payload": value,
        })
        assert runtime.llm.dispatch(pid, action)["ok"]
        result = runtime.llm.dispatch(pid, {"action": "read_memory_object", "name": "literal"})
        assert result["payload"]["payload"] == value
        assert result["payload"]["payload_type"] == "string"
        assert runtime.llm.dispatch(pid, {
            "action": "create_memory_object", "name": "entries", "type": "artifact",
            "payload": [], "immutable": False,
        })["ok"]
        append = runtime.tools.normalize_model_action(pid, {
            "action": "append_memory_object", "name": "entries", "entry": value,
        })
        assert runtime.llm.dispatch(pid, append)["ok"]
        result = runtime.llm.dispatch(pid, {"action": "read_memory_object", "name": "entries"})
        assert result["payload"]["payload"] == [value]
        nullable = runtime.tools.normalize_model_action(pid, {
            "action": "process_exit", "message": value,
        })
        assert nullable["message"] == value
    finally:
        runtime.close()


@pytest.mark.parametrize("paged", [False, True], ids=["subtrees", "byte-pages"])
def test_working_set_keeps_independent_memory_selections(paged: bool) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="read both selections")
        runtime.activate_skill(pid, "agent-libos-object-memory")
        assert runtime.llm.dispatch(pid, {
            "action": "create_memory_object", "name": "document", "type": "artifact",
            "payload": {"a": "FIRST_SUBTREE_PAYLOAD", "b": "SECOND_SUBTREE_PAYLOAD"},
        })["ok"]
        first_action = {"action": "read_memory_object", "name": "document"}
        first_action.update({"max_payload_chars": 30} if paged else {"json_pointer": "/a"})
        first = runtime.llm.dispatch(pid, first_action)
        assert first["ok"]
        second_action = {"action": "read_memory_object", "name": "document"}
        second_action.update({
            "cursor": first["payload"]["next_cursor"],
            "expected_sha256": first["payload"]["sha256"], "max_payload_chars": 30,
        } if paged else {"json_pointer": "/b"})
        second = runtime.llm.dispatch(pid, second_action)
        assert second["ok"]
        context = runtime.memory.materialize_context(
            pid, runtime.process.get(pid).memory_view, policy="working_set",
            budget_tokens=100_000, charge_resources=False,
        )
        assert {first["result_oid"], second["result_oid"]} <= set(context.object_refs)
        assert not any(entry["reason"] == "superseded" for entry in context.object_manifest)
        repeated = runtime.llm.dispatch(pid, second_action)
        context = runtime.memory.materialize_context(
            pid, runtime.process.get(pid).memory_view, policy="working_set",
            budget_tokens=100_000, charge_resources=False,
        )
        assert first["result_oid"] in context.object_refs
        assert repeated["result_oid"] in context.object_refs
        assert second["result_oid"] in context.omitted_objects
    finally:
        runtime.close()


def test_working_set_keeps_capability_pages_and_supersedes_identical_repeats() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="inspect all permissions")
        first = runtime.tools.call(pid, "list_capabilities", {"limit": 1})
        assert first.ok and first.payload["has_more"]
        second = runtime.tools.call(pid, "list_capabilities", {
            "limit": 1, "after_cap_id": first.payload["next_cursor"],
        })
        assert second.ok and second.payload["capabilities"]
        assert first.payload["capabilities"] != second.payload["capabilities"]
        handles = [first.result_handle, second.result_handle]
        context = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, handles), policy="working_set",
            budget_tokens=100_000, charge_resources=False,
        )
        assert context.object_refs == [handle.oid for handle in handles]
        assert not context.omitted_objects

        # Repeating the same observed page remains compactable. Copy its actual
        # result because listing itself creates result capabilities, which can
        # change the next live capability page between calls.
        repeated = runtime.memory.create_object(
            pid, ObjectType.TOOL_RESULT,
            runtime.memory.get_object(pid, first.result_handle).payload,
        )
        context = runtime.memory.materialize_context(
            pid, runtime.memory.create_view(pid, [*handles, repeated]),
            policy="working_set", budget_tokens=100_000, charge_resources=False,
        )
        assert context.object_refs == [second.result_handle.oid, repeated.oid]
        assert context.omitted_objects == [first.result_handle.oid]
        assert next(
            entry for entry in context.object_manifest if entry["oid"] == first.result_handle.oid
        )["reason"] == "superseded"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("tool_name", "first", "second"),
    [
        ("list_memory_namespace", {"namespace": "project", "objects": [{"name": "a"}]},
         {"namespace": "project", "objects": [{"name": "b"}]}),
        ("list_checkpoints", {"checkpoints": [{"pid": "first"}]},
         {"checkpoints": [{"pid": "second"}]}),
        ("list_child_processes", {"children": [{"pid": "exited"}, {"pid": "running"}]},
         {"children": [{"pid": "running"}]}),
        ("list_object_tasks", {"tasks": [{"owner_oid": "first"}]},
         {"tasks": [{"owner_oid": "second"}]}),
        ("list_jsonrpc_endpoints", {"endpoints": [{"endpoint_id": "first"}]},
         {"endpoints": [{"endpoint_id": "second"}]}),
        ("list_mcp_servers", {"servers": [{"server_id": "first"}]},
         {"servers": [{"server_id": "second"}]}),
        ("list_mcp_tools", {"server_id": "server", "tools": [{"name": "cached"}]},
         {"server_id": "server", "tools": [{"name": "live"}]}),
        ("list_mcp_resources", {"server_id": "server", "kind": "resource", "items": []},
         {"server_id": "server", "kind": "template", "items": []}),
        ("list_mcp_resources", {"server_id": "server", "items": [{"resource_id": "first"}]},
         {"server_id": "server", "items": [{"resource_id": "second"}]}),
    ],
)
def test_listing_supersession_requires_identical_observed_results(
    tool_name: str, first: dict[str, Any], second: dict[str, Any],
) -> None:
    key = _observation_supersession_key({"tool_name": tool_name, "result": first})
    assert key is not None
    assert key != _observation_supersession_key({"tool_name": tool_name, "result": second})
    assert key == _observation_supersession_key({
        "tool_name": tool_name, "result": dict(reversed(list(first.items()))),
    })


@pytest.mark.parametrize(
    ("tool_name", "fields", "extent_field"),
    [
        ("read_text_file", {"encoding": "utf-8"}, "bytes_read"),
        ("read_directory", {}, "count"),
    ],
)
def test_filesystem_supersession_keeps_independent_prefixes(
    tool_name: str, fields: dict[str, Any], extent_field: str,
) -> None:
    def key(**changes: Any):
        return _observation_supersession_key({
            "tool_name": tool_name,
            "result": {"path": "target", **fields, **changes},
        })

    complete = key(truncated=False, **{extent_field: 100})
    assert complete is not None
    assert complete == key(truncated=False, **{extent_field: 200})
    prefix = key(truncated=True, **{extent_field: 10})
    assert prefix is not None and prefix != complete
    assert prefix != key(truncated=True, **{extent_field: 20})
    assert key() is None, "legacy results without extent provenance are not replaced"
    assert key(truncated=True) is None
    if tool_name == "read_text_file":
        assert complete != key(truncated=False, encoding="latin-1")
