from __future__ import annotations

import json
from dataclasses import replace

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.llm.prompt import RETAINED_GOAL_CONTEXT_BINDING_KEY
from agent_libos.models import (
    CapabilityRight,
    EventType,
    ObjectType,
    PROMPT_MODE_LIBOS_DEFAULT,
    ViewMode,
)
from tests.support.fakes import RecordingActionClient


IMAGE_ID = "executor-cache-semantics:v0"
_V2_CONFIG = replace(
    DEFAULT_CONFIG,
    llm=replace(DEFAULT_CONFIG.llm, prompt_layout="cache_optimized_v2"),
)


def _register_cumulative_image(runtime: Runtime) -> None:
    runtime.register_image(
        AgentImage(
            image_id=IMAGE_ID,
            name="executor-cache-semantics",
            system_prompt="Complete the entire durable goal.",
            prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
            default_tools=["echo", "process_exit"],
            context_policy="recency_first",
            metadata={"completion_gate": "cumulative_review"},
        ),
        actor="test",
    )


def test_live_materialized_goal_is_not_replayed_as_retained_context() -> None:
    goal = "CACHE_STABLE_LIVE_GOAL_SENTINEL"
    runtime = Runtime.open("local")
    try:
        _register_cumulative_image(runtime)
        client = RecordingActionClient(
            [
                {"action": "echo", "milestone": "phase one"},
                {"action": "process_exit", "payload": {"done": True}},
            ]
        )
        runtime.llm.client = client
        pid = runtime.process.spawn(image=IMAGE_ID, goal=goal)

        runtime.run_process_once(pid)
        runtime.run_process_once(pid)

        second_prompt = client.user_prompts[1]
        assert second_prompt.count(goal) == 1
        assert "Retained original goal contract" not in second_prompt
    finally:
        runtime.close()


def test_missing_materialized_goal_is_recovered_from_retained_context() -> None:
    goal = "CACHE_RECOVERED_MISSING_GOAL_SENTINEL"
    runtime = Runtime.open("local", config=_V2_CONFIG)
    try:
        _register_cumulative_image(runtime)
        client = RecordingActionClient(
            [
                {"action": "echo", "milestone": "phase one"},
                {"action": "process_exit", "payload": {"done": True}},
            ]
        )
        runtime.llm.client = client
        pid = runtime.process.spawn(image=IMAGE_ID, goal=goal)

        runtime.run_process_once(pid)
        off_goal = runtime.memory.create_object(
            pid,
            ObjectType.OBSERVATION,
            {"summary": "the live goal was compacted out of this view"},
        )
        process = runtime.process.get(pid)
        goal_oid = process.goal_oid
        process.memory_view = runtime.memory.create_view(
            pid,
            [off_goal],
            mode=ViewMode.READ_ONLY,
        )
        runtime.store.update_process(process)

        runtime.run_process_once(pid)

        second_prompt = client.user_prompts[1]
        assert second_prompt.count(goal) == 1
        assert "Retained original goal contract" in second_prompt
        request_record = [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "llm.request"
        ][-1]
        assert goal_oid is not None
        assert goal_oid in request_record.input_refs
        first_call = runtime.store.list_llm_calls(pid=pid)[0]
        assert first_call.request_options[RETAINED_GOAL_CONTEXT_BINDING_KEY][
            "goal_oid"
        ] == goal_oid
    finally:
        runtime.close()


def test_exec_goal_does_not_recover_prior_v2_goal_without_matching_binding() -> None:
    old_goal = "CACHE_OLD_EXEC_GOAL_MUST_NOT_REPLAY"
    new_goal = "CACHE_NEW_EXEC_GOAL_IS_CURRENT"
    runtime = Runtime.open("local", config=_V2_CONFIG)
    try:
        _register_cumulative_image(runtime)
        client = RecordingActionClient(
            [
                {"action": "echo", "milestone": "old generation"},
                {"action": "echo", "milestone": "new generation"},
            ]
        )
        runtime.llm.client = client
        pid = runtime.process.spawn(image=IMAGE_ID, goal=old_goal)
        runtime.capability.grant(
            pid,
            f"image:{IMAGE_ID}",
            [CapabilityRight.READ],
            issued_by="test",
        )

        runtime.run_process_once(pid)
        runtime.exec_process(pid, IMAGE_ID, goal=new_goal, preserve_memory=True)
        unrelated = runtime.memory.create_object(
            pid,
            ObjectType.OBSERVATION,
            {"summary": "the replacement goal is not in this explicit view"},
        )
        process = runtime.process.get(pid)
        process.memory_view = runtime.memory.create_view(
            pid,
            [unrelated],
            mode=ViewMode.READ_ONLY,
        )
        runtime.store.update_process(process)

        runtime.run_process_once(pid)

        second_prompt = client.user_prompts[1]
        assert old_goal not in second_prompt
        assert "Retained original goal contract" not in second_prompt
    finally:
        runtime.close()


def test_source_only_event_projection_represents_full_batch_before_advancing_cursor(
) -> None:
    sentinel = "RAW_PROCESS_SIGNAL_PAYLOAD_MUST_NOT_REACH_MODEL"
    runtime = Runtime.open("local", config=_V2_CONFIG)
    try:
        _register_cumulative_image(runtime)
        client = RecordingActionClient(
            [{"action": "process_exit", "payload": {"done": True}}]
        )
        runtime.llm.client = client
        pid = runtime.process.spawn(image=IMAGE_ID, goal="project event batch")
        markers = [
            runtime.events.emit(
                EventType.PROCESS_SIGNAL,
                source="test",
                target=pid,
                payload={"signal": f"probe-{index}", "raw": sentinel},
            )
            for index in range(12)
        ]
        fetched = runtime.events.list(
            target=pid,
            limit=runtime.config.llm_context.recent_event_limit,
        )
        assert fetched

        result = runtime.run_process_once(pid)

        assert result["ok"], result
        prompt = client.user_prompts[0]
        assert sentinel not in prompt
        assert "event_projection_summary" in prompt
        assert all(event.event_id not in prompt for event in markers)
        assert runtime.process.get(pid).event_cursor == fetched[-1].event_id
        request = [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "llm.request"
        ][-1]
        assert request.decision["event_projection"]["input_event_count"] == len(
            fetched
        )
        assert request.decision["event_projection"][
            "represented_through_event_id"
        ] == fetched[-1].event_id
    finally:
        runtime.close()


def test_provider_failure_does_not_acknowledge_projected_event_batch() -> None:
    runtime = Runtime.open("local")
    try:
        _register_cumulative_image(runtime)
        runtime.llm.client = _FailingClient()
        pid = runtime.process.spawn(image=IMAGE_ID, goal="preserve failed event batch")
        runtime.events.emit(
            EventType.PROCESS_SIGNAL,
            source="test",
            target=pid,
            payload={"signal": "retry-me"},
        )

        result = runtime.run_process_once(pid)

        assert not result["ok"]
        assert runtime.process.get(pid).event_cursor is None
    finally:
        runtime.close()


def test_effective_provider_options_override_configured_cache_telemetry() -> None:
    runtime = Runtime.open("local")
    try:
        _register_cumulative_image(runtime)
        runtime.llm.client = _EffectiveOptionsClient()
        pid = runtime.process.spawn(image=IMAGE_ID, goal="record effective options")

        result = runtime.run_process_once(pid)

        assert result["ok"], result
        options = runtime.store.list_llm_calls(pid)[0].request_options
        assert options["openai_prompt_cache_key_sent"] is False
        assert options["openai_prompt_cache_options_sent"] is False
        assert options["openai_prompt_cache_retention"] is None
        assert options["openai_safety_identifier_sent"] is False
        assert options["openai_compatibility_removed_options"] == [
            "prompt_cache_key",
            "prompt_cache_retention",
        ]
        assert options["openai_prompt_cache_mode"] == "provider_default"
        assert options["openai_prompt_cache_breakpoint_count"] == 0
        assert options["openai_prompt_cache_downgrade_reason"] == (
            "provider_rejected_cache_options"
        )
    finally:
        runtime.close()


def test_prompt_projection_telemetry_records_only_sizes_tokens_and_hashes() -> None:
    goal = "PROMPT_PLAINTEXT_MUST_NOT_BE_COPIED_TO_TELEMETRY"
    runtime = Runtime.open("local", config=_V2_CONFIG)
    try:
        _register_cumulative_image(runtime)
        runtime.llm.client = RecordingActionClient(
            [{"action": "echo", "milestone": "measure projection"}]
        )
        pid = runtime.process.spawn(image=IMAGE_ID, goal=goal)

        result = runtime.run_process_once(pid)

        assert result["ok"], result
        options = runtime.store.list_llm_calls(pid)[0].request_options
        projection = options["prompt_projection"]
        assert projection["schema_version"] == 1
        assert projection["layout"] == "cache_optimized_v2"
        assert projection["stable_prefix_message_count"] == 2
        assert len(projection["stable_prefix_sha256"]) == 64
        assert projection["segments"]
        assert all(
            set(segment)
            == {
                "ordinal",
                "part",
                "role",
                "bytes",
                "estimated_tokens",
                "sha256",
                "stable_prefix",
            }
            for segment in projection["segments"]
        )
        assert set(projection["tools"]) == {
            "bytes",
            "estimated_tokens",
            "sha256",
        }
        assert goal not in json.dumps(projection, sort_keys=True)
        assert options["llm_latency_ms"] >= 0
    finally:
        runtime.close()


class _FailingClient:
    def complete_action(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic provider failure")


class _EffectiveOptionsClient:
    def complete_action(self, *_args: object, **_kwargs: object) -> LLMCompletion:
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "effective_options_exit",
                    "name": "process_exit",
                    "arguments": json.dumps({"payload": {"done": True}}),
                }
            ],
            api="chat",
            provider_request_options={
                "prompt_cache_key_sent": False,
                "prompt_cache_options_sent": False,
                "prompt_cache_retention": None,
                "safety_identifier_sent": False,
            },
            compatibility_removed_options=[
                "prompt_cache_retention",
                "prompt_cache_key",
            ],
        )
