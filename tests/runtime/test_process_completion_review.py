from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.llm.context_memory import LLM_CONTEXT_ENRICHMENT_RESOURCE
from agent_libos.models import CapabilityRight, ObjectType, ProcessStatus
from agent_libos.tools.builtin.process import _bounded_json_value


def _completion_evidence(
    review: dict[str, object],
    *,
    tool_name: str = "list_capabilities",
) -> dict[str, object]:
    goal = review["goal"]
    assert isinstance(goal, dict)
    goal_oid = goal["oid"]
    message_ids = review["acknowledged_human_message_ids"]
    assert isinstance(message_ids, list)
    return {
        "goal_oid": goal_oid,
        "reviewed_message_ids": list(message_ids),
        "acceptance_checks": [
            {
                "requirement": "complete the cumulative process goal",
                "source_refs": [goal_oid, *message_ids],
                "status": "completed",
                "evidence_tool_calls": [tool_name],
                "evidence_summary": "The cited successful tool call supplies test evidence.",
            }
        ],
        "final_verification": [tool_name],
    }


def _effective_authority(
    runtime: Runtime,
    pid: str,
    *,
    exclude_object_oids: set[str] | None = None,
) -> dict[str, frozenset[str]]:
    excluded = {f"object:{oid}" for oid in (exclude_object_oids or set())}
    rights_by_resource: dict[str, set[str]] = {}
    for capability in runtime.store.list_capabilities(subject=pid):
        if capability.resource in excluded:
            continue
        rights_by_resource.setdefault(capability.resource, set()).update(
            capability.rights
        )
    return {
        resource: frozenset(rights)
        for resource, rights in rights_by_resource.items()
    }


def _start_review(runtime: Runtime, pid: str) -> tuple[dict[str, object], str]:
    inspected = runtime.llm.dispatch(pid, {"action": "list_capabilities"})
    assert inspected["ok"] is True
    first = runtime.llm.dispatch(
        pid,
        {"action": "process_exit", "payload": {"summary": "not final yet"}},
    )
    assert first["ok"] is True
    assert first["payload"]["status"] == "completion_review_required"
    review = first["payload"]["completion_review"]
    assert isinstance(review, dict)
    result_oid = first["result_oid"]
    assert isinstance(result_oid, str)
    return review, result_oid


def test_coding_exit_review_is_nonterminal_published_and_audited(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "completion-review.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="inspect one thing, report it, and then exit",
        )
        inspected = runtime.llm.dispatch(pid, {"action": "list_capabilities"})
        assert inspected["ok"] is True
        authority_before = _effective_authority(runtime, pid)

        first = runtime.llm.dispatch(
            pid,
            {"action": "process_exit", "payload": {"summary": "not final yet"}},
        )
        assert first["ok"] is True
        assert first["payload"]["status"] == "completion_review_required"
        review = first["payload"]["completion_review"]
        review_result_oid = first["result_oid"]

        process = runtime.process.get(pid)
        assert process.status == ProcessStatus.RUNNABLE
        assert process.memory_view is not None
        assert review_result_oid in {handle.oid for handle in process.memory_view.roots}
        assert review["schema_version"] == 2
        assert review["goal"]["source"] == "object_memory"
        assert review["goal"]["version"] == 1
        assert len(review["goal"]["payload_sha256"]) == 64
        assert "payload" not in review["goal"]
        assert "fallback" not in review["goal"]
        assert review["goal"]["reference"] == {
            "kind": "object_memory",
            "namespace": review["goal"]["reference"]["namespace"],
            "name": review["goal"]["reference"]["name"],
            "activate_skill": "agent-libos-object-memory",
            "tool": "read_memory_object",
            "arguments": {
                "namespace": review["goal"]["reference"]["namespace"],
                "name": review["goal"]["reference"]["name"],
                "max_payload_chars": (
                    runtime.config.tools.memory_payload_hard_limit_chars
                ),
            },
        }
        assert review["observed_successful_tool_calls"] == ["list_capabilities"]
        # Publishing the immutable review ToolResult necessarily creates an
        # exact-object handle. Apart from that newly produced evidence object,
        # the review may only mint redundant handles for resources the process
        # could already read; it must not expand effective ambient authority.
        assert _effective_authority(
            runtime,
            pid,
            exclude_object_oids={review_result_oid},
        ) == authority_before
        required = [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "process.exit_review_required"
        ]
        assert len(required) == 1
        assert required[0].decision["authority_changed"] is False

        completed = runtime.llm.dispatch(
            pid,
            {
                "action": "process_exit",
                "review_token": review["review_token"],
                "completion_evidence": _completion_evidence(review),
                "payload": {"summary": "finished after cumulative review"},
            },
        )

        assert completed["ok"] is True
        assert completed["payload"].get("status") == "exited", completed
        assert runtime.process.get(pid).status == ProcessStatus.EXITED
        passed = [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "process.exit_review_passed"
        ]
        assert len(passed) == 1
        assert passed[0].decision["authority_changed"] is False
        assert passed[0].decision["acceptance_check_count"] == 1
    finally:
        runtime.close()


def test_final_exit_linearizes_against_concurrent_human_followup(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runtime = Runtime.open(tmp_path / "completion-review-race.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="finish only if no newer human follow-up exists",
        )
        review, _review_result_oid = _start_review(runtime, pid)
        exit_entered = Event()
        post_attempted = Event()
        post_errors: list[BaseException] = []
        original_exit = runtime.process.exit

        def delayed_exit(*args: Any, **kwargs: Any) -> Any:
            exit_entered.set()
            assert post_attempted.wait(2)
            return original_exit(*args, **kwargs)

        monkeypatch.setattr(runtime.process, "exit", delayed_exit)

        def post_followup() -> None:
            assert exit_entered.wait(2)
            post_attempted.set()
            try:
                runtime.human.send_process_message(
                    pid,
                    "This message must either precede review or lose to terminal exit.",
                    subject="Concurrent follow-up",
                )
            except BaseException as exc:  # captured for the main test thread
                post_errors.append(exc)

        poster = Thread(target=post_followup, daemon=True)
        poster.start()
        completed = runtime.llm.dispatch(
            pid,
            {
                "action": "process_exit",
                "review_token": review["review_token"],
                "completion_evidence": _completion_evidence(review),
            },
        )
        poster.join(timeout=2)

        assert poster.is_alive() is False
        assert completed["ok"] is True
        assert completed["payload"]["status"] == "exited"
        assert len(post_errors) == 1
        assert "terminal process" in str(post_errors[0])
        assert runtime.store.list_process_messages(pid) == []
    finally:
        runtime.close()


def test_exit_review_maps_explicit_unobserved_tools_to_builtin_skills(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "completion-review-skill-hints.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal=(
                "Fix the issue, inspect git status, create checkpoint evidence, "
                "and then finish."
            ),
        )

        review, _review_result_oid = _start_review(runtime, pid)
        hints = {
            item["tool"]: item["activate_skill"]
            for item in review["explicit_unobserved_tool_hints"]
        }

        assert hints["git_status"] == "agent-libos-git-inspection"
        assert hints["create_checkpoint"] == "agent-libos-checkpoints"
    finally:
        runtime.close()


def test_exit_review_rejects_stale_token_after_human_followup(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "completion-review-message.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="complete the original requirement",
        )
        review, _review_result_oid = _start_review(runtime, pid)
        message = runtime.human.send_process_message(
            pid,
            "Also verify the added acceptance criterion.",
            subject="Additional criterion",
        )

        stale = runtime.llm.dispatch(
            pid,
            {
                "action": "process_exit",
                "review_token": review["review_token"],
                "completion_evidence": _completion_evidence(review),
            },
        )

        assert stale["payload"]["status"] == "completion_review_required"
        stale_review = stale["payload"]["completion_review"]
        assert stale_review["unread_human_message_ids"] == [message.message_id]
        assert message.body not in str(stale_review)
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE

        runtime.activate_skill(pid, "agent-libos-child-processes")
        read = runtime.llm.dispatch(pid, {"action": "read_process_messages"})
        assert read["ok"] is True
        refreshed = runtime.llm.dispatch(pid, {"action": "process_exit"})
        refreshed_review = refreshed["payload"]["completion_review"]

        assert refreshed_review["review_token"] != review["review_token"]
        assert refreshed_review["acknowledged_human_message_ids"] == [
            message.message_id
        ]
        assert message.body not in json.dumps(refreshed_review, ensure_ascii=False)
        assert refreshed_review["acknowledged_human_message_reference"] == {
            "kind": "process_message_ids",
            "activate_skill": "agent-libos-child-processes",
            "tool": "read_process_messages",
            "fixed_arguments": {
                "include_acked": True,
                "ack": False,
            },
            "copy_arguments": {
                "message_ids": "acknowledged_human_message_ids",
                "limit": "acknowledged_human_message_count",
            },
        }
    finally:
        runtime.close()


def test_final_exit_rejects_missing_sources_and_unobserved_tool_claims(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "completion-review-evidence.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="complete the original requirement and follow-up",
        )
        message = runtime.human.send_process_message(
            pid,
            "Also complete this acknowledged follow-up.",
        )
        runtime.activate_skill(pid, "agent-libos-child-processes")
        runtime.llm.dispatch(pid, {"action": "read_process_messages"})
        review, _review_result_oid = _start_review(runtime, pid)
        evidence = _completion_evidence(review)
        acceptance_check = evidence["acceptance_checks"][0]
        assert isinstance(acceptance_check, dict)
        acceptance_check["source_refs"] = [review["goal"]["oid"]]
        acceptance_check["evidence_tool_calls"] = ["git_diff"]
        evidence["final_verification"] = ["git_diff"]

        rejected = runtime.llm.dispatch(
            pid,
            {
                "action": "process_exit",
                "review_token": review["review_token"],
                "completion_evidence": evidence,
            },
        )

        assert rejected["ok"] is True
        assert rejected["payload"]["status"] == "completion_review_required"
        errors = rejected["payload"]["completion_review"]["validation_errors"]
        assert "acceptance checks do not cover every expected_source_ref" in errors
        assert any("cites unobserved tools" in error for error in errors)
        assert message.message_id in review["acknowledged_human_message_ids"]
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
    finally:
        runtime.close()


def test_exit_review_references_messages_without_reinlining_bodies(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "completion-review-message-bounds.sqlite")
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="complete every acknowledged follow-up",
        )
        messages = [
            runtime.human.send_process_message(
                pid,
                f"FOLLOW_UP_BODY_SENTINEL_{index}_" + ("x" * 4_000),
                subject=f"Requirement {index}",
            )
            for index in range(9)
        ]
        runtime.activate_skill(pid, "agent-libos-child-processes")
        read = runtime.llm.dispatch(
            pid,
            {"action": "read_process_messages", "limit": len(messages)},
        )
        assert read["ok"] is True

        review, _review_result_oid = _start_review(runtime, pid)

        expected_ids = [message.message_id for message in messages]
        assert review["acknowledged_human_message_ids"] == expected_ids
        assert review["acknowledged_human_message_count"] == 9
        assert len(review["acknowledged_human_messages_sha256"]) == 64
        assert review["acknowledged_human_message_reference"] == {
            "kind": "process_message_ids",
            "activate_skill": "agent-libos-child-processes",
            "tool": "read_process_messages",
            "fixed_arguments": {
                "include_acked": True,
                "ack": False,
            },
            "copy_arguments": {
                "message_ids": "acknowledged_human_message_ids",
                "limit": "acknowledged_human_message_count",
            },
        }
        rendered_review = json.dumps(review, ensure_ascii=False)
        assert "FOLLOW_UP_BODY_SENTINEL" not in rendered_review
        assert len(rendered_review) < 12_000

        repeated = runtime.llm.dispatch(pid, {"action": "process_exit"})
        repeated_review = repeated["payload"]["completion_review"]
        assert repeated_review["review_token"] == review["review_token"]
        assert (
            repeated_review["acknowledged_human_messages_sha256"]
            == review["acknowledged_human_messages_sha256"]
        )
        assert (
            repeated_review["observed_successful_tool_calls"]
            == review["observed_successful_tool_calls"]
        )
    finally:
        runtime.close()


def test_exit_review_references_live_goal_without_reinlining_payload(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "completion-review-goal-reference.sqlite")
    try:
        goal = "LIVE_GOAL_BODY_SENTINEL_" + ("x" * 31_000)
        pid = runtime.process.spawn(image="coding-agent:v0", goal=goal)

        review, _review_result_oid = _start_review(runtime, pid)
        rendered_review = json.dumps(review, ensure_ascii=False)

        assert review["goal"]["source"] == "object_memory"
        assert "payload" not in review["goal"]
        assert "fallback" not in review["goal"]
        assert "LIVE_GOAL_BODY_SENTINEL" not in rendered_review
        assert len(rendered_review) < 8_000
    finally:
        runtime.close()


def test_exit_review_fails_closed_when_human_message_set_exceeds_bound(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        tools=replace(
            DEFAULT_CONFIG.tools,
            message_read_limit=1,
            message_read_hard_limit=1,
            message_filter_ids_hard_limit=1,
        ),
    )
    runtime = Runtime.open(
        tmp_path / "completion-review-message-overflow.sqlite",
        config=config,
    )
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="do not silently omit human follow-ups",
        )
        runtime.human.send_process_message(pid, "first follow-up")
        runtime.human.send_process_message(pid, "second follow-up")

        result = runtime.llm.dispatch(pid, {"action": "process_exit"})

        assert result["ok"] is False
        assert result["payload"]["error"]["code"] == "validation_error"
        assert result["payload"]["error"]["details"] == {
            "human_message_count": 2,
            "review_message_limit": 1,
        }
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
    finally:
        runtime.close()


def test_exit_review_survives_runtime_reopen(tmp_path: Path) -> None:
    database = tmp_path / "completion-review-reopen.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="retain the final review across restart",
        )
        process = runtime.process.get(pid)
        assert process.goal_oid is not None
        goal_oid = process.goal_oid
        sibling = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {
                "text": (
                    f"[{goal_oid}] "
                    "UNRELATED_SIBLING_CONTEXT_SENTINEL"
                )
            },
        )
        process = runtime.process.get(pid)
        assert process.memory_view is not None
        process.memory_view.roots.insert(0, sibling)
        runtime.store.update_process(process)
        runtime.llm.client = _SingleActionClient("list_capabilities", {})
        first_quantum = runtime.run_process_once(pid)
        assert first_quantum["action"]["action"] == "list_capabilities"
        review, review_result_oid = _start_review(runtime, pid)
    finally:
        runtime.close()

    reopened = Runtime.open(database)
    try:
        process = reopened.process.get(pid)
        assert process.memory_view is not None
        assert review_result_oid in {handle.oid for handle in process.memory_view.roots}

        refreshed = reopened.llm.dispatch(pid, {"action": "process_exit"})
        refreshed_review = refreshed["payload"]["completion_review"]
        assert refreshed_review["goal"]["source"] == "persisted_initial_llm_context"
        assert refreshed_review["goal"]["reference"]["kind"] == "retained_llm_evidence"
        assert refreshed_review["goal"]["fallback"] == {
            "text": "retain the final review across restart"
        }
        assert (
            refreshed_review["goal"]["payload_sha256"]
            == review["goal"]["payload_sha256"]
        )
        assert "UNRELATED_SIBLING_CONTEXT_SENTINEL" not in str(
            refreshed_review["goal"]["fallback"]
        )
        assert refreshed_review["review_token"] == review["review_token"]

        completed = reopened.llm.dispatch(
            pid,
            {
                "action": "process_exit",
                "review_token": refreshed_review["review_token"],
                "completion_evidence": _completion_evidence(refreshed_review),
            },
        )

        assert completed["payload"].get("status") == "exited", completed
        assert reopened.process.get(pid).status == ProcessStatus.EXITED
    finally:
        reopened.close()


def test_exit_review_recovers_goal_from_persistent_context_prompt(
    tmp_path: Path,
) -> None:
    database = tmp_path / "completion-review-persistent-context.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="recover this goal from the persistent context prompt",
        )
        runtime.capability.grant(
            pid,
            LLM_CONTEXT_ENRICHMENT_RESOURCE,
            [CapabilityRight.EXECUTE],
            issued_by="completion-review-test",
        )
        runtime.llm.client = _SingleActionClient("list_capabilities", {})
        runtime.run_process_once(pid)
    finally:
        runtime.close()

    reopened = Runtime.open(database)
    try:
        review_result = reopened.llm.dispatch(pid, {"action": "process_exit"})
        review = review_result["payload"]["completion_review"]

        assert review["goal"]["source"] == "persisted_initial_llm_context"
        assert "recover this goal from the persistent context prompt" in str(
            review["goal"]["fallback"]
        )
    finally:
        reopened.close()


def test_non_gated_image_keeps_single_phase_process_exit(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "single-phase-exit.sqlite")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="exit without the coding image review gate",
        )

        exited = runtime.llm.dispatch(
            pid,
            {"action": "process_exit", "payload": {"done": True}},
        )

        assert exited["ok"] is True
        assert exited["payload"]["status"] == "exited"
        assert exited["payload"]["completion_review"] is None
        assert runtime.process.get(pid).status == ProcessStatus.EXITED
    finally:
        runtime.close()


def test_exit_review_fails_closed_after_reopen_without_full_io_retention(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False),
    )
    database = tmp_path / "completion-review-redacted-reopen.sqlite"
    runtime = Runtime.open(database, config=config)
    try:
        pid = runtime.process.spawn(
            image="coding-agent:v0",
            goal="this goal must not be copied into durable LLM evidence",
        )
        runtime.llm.client = _SingleActionClient("list_capabilities", {})
        first = runtime.run_process_once(pid)
        assert first["action"]["action"] == "list_capabilities"
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=config)
    try:
        blocked = reopened.llm.dispatch(pid, {"action": "process_exit"})

        assert blocked["ok"] is False
        assert "goal payload is unavailable after Runtime reopen" in blocked["error"]
        assert runtime_goal_not_in_payload(blocked)
        assert reopened.process.get(pid).status == ProcessStatus.RUNNABLE
    finally:
        reopened.close()


def runtime_goal_not_in_payload(result: dict[str, Any]) -> bool:
    return "this goal must not be copied" not in json.dumps(result, default=str)


def test_bounded_goal_fallback_truncates_decoded_values_not_serialized_json() -> None:
    value = {
        "goal": "quoted \\\"requirement\\\" with slash \\\\ and unicode \u76ee\u6807 " * 200,
        "nested": [{"requirement": "x" * 800}, {"keep": True}],
    }

    bounded = _bounded_json_value(value, 512)
    rendered = json.dumps(
        bounded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert len(rendered) <= 512
    assert json.loads(rendered) == bounded
    assert bounded["truncated"] is True
    assert isinstance(bounded["preview"], dict)
    assert bounded["preview"]["goal"].endswith("\u2026")


class _SingleActionClient:
    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.name = name
        self.arguments = dict(arguments)

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "completion_review_seed",
                    "name": self.name,
                    "arguments": json.dumps(self.arguments),
                }
            ],
            raw=SimpleNamespace(id="completion_review_seed_raw"),
            api="chat",
            model="fake",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
