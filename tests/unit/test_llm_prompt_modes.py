from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.llm.event_projection import project_prompt_events
from agent_libos.llm.prompt import (
    build_system_prompt,
    build_user_prompt,
    recover_initial_goal_context,
)
from agent_libos.models import (
    CapabilityRight,
    Event,
    EventPriority,
    EventType,
    JIT_TOOL_EXPOSURE_DIRECT,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    JIT_TOOL_EXPOSURES,
    PROMPT_MODE_IMAGE_ONLY,
    PROMPT_MODE_LIBOS_DEFAULT,
    PROMPT_MODE_MINIMAL_RUNTIME,
    MaterializedContext,
    ObjectMetadata,
    ObjectType,
)
from agent_libos.models.exceptions import HumanApprovalRequired, ValidationError
from agent_libos.tools.base import SyncAgentTool, ToolContext
from tests.support.skills import write_skill_package


class _CaptureToolIdentityArgs(BaseModel):
    value: str


class _CaptureToolIdentity(SyncAgentTool[_CaptureToolIdentityArgs]):
    name = "capture_tool_identity"
    description = "Capture the Host-bound model tool-call identity."
    args_schema = _CaptureToolIdentityArgs

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def run(
        self,
        args: _CaptureToolIdentityArgs,
        ctx: ToolContext,
    ) -> dict[str, str]:
        self.seen.append(dict(ctx.metadata))
        return {"value": args.value}


class _WaitForToolIdentityApproval(_CaptureToolIdentity):
    name = "wait_for_tool_identity_approval"
    description = "Wait once, then capture the resumed model tool-call identity."

    def run(
        self,
        args: _CaptureToolIdentityArgs,
        ctx: ToolContext,
    ) -> dict[str, str]:
        self.seen.append(dict(ctx.metadata))
        if "human_resume_request_id" not in ctx.metadata:
            request_id = ctx.runtime.human.query(
                pid=ctx.pid,
                human=ctx.runtime.config.runtime.default_human,
                request={
                    "type": "external_operation_approval",
                    "question": "Approve the identity resume probe",
                    "context": {"operation": "identity_resume_probe"},
                },
                blocking=True,
            )
            raise HumanApprovalRequired(request_id, "identity resume probe")
        return {"value": args.value}


class TestLLMPromptModes:

    def test_llm_tool_call_identity_is_bound_into_tool_context(self) -> None:
        runtime = Runtime.open("local")
        try:
            capture = _CaptureToolIdentity()
            runtime.tools.register_tool(capture, registered_by="test", ephemeral=True)
            runtime.register_image(
                AgentImage(
                    image_id="tool-call-identity:v0",
                    name="tool-call-identity",
                    system_prompt="Capture exact provider tool-call identity.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["capture_tool_identity", "process_exit"],
                ),
                actor="test",
            )
            client = ScriptedTranscriptClient(
                [
                    [
                        _tool_call(
                            "provider-tool-call-7",
                            "capture_tool_identity",
                            {"value": "one"},
                        )
                    ],
                    [
                        _tool_call(
                            "provider-exit-8",
                            "process_exit",
                            {"payload": {"done": True}},
                        )
                    ],
                ]
            )
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="tool-call-identity:v0",
                goal="capture the native tool-call identity",
            )

            first = runtime.run_process_once(pid)
            completed = runtime.run_process_once(pid)

            assert first["ok"], first
            assert completed["ok"], completed
            assert len(capture.seen) == 1
            assert capture.seen[0]["llm_transcript_output_key"].startswith("llmcall_")
            assert capture.seen[0]["llm_tool_call_id"] == "provider-tool-call-7"
            assert capture.seen[0]["llm_tool_name"] == "capture_tool_identity"
        finally:
            runtime.close()

    def test_llm_tool_call_identity_survives_human_wait_resume(self) -> None:
        runtime = Runtime.open("local")
        try:
            capture = _WaitForToolIdentityApproval()
            runtime.tools.register_tool(capture, registered_by="test", ephemeral=True)
            runtime.register_image(
                AgentImage(
                    image_id="tool-call-identity-wait:v0",
                    name="tool-call-identity-wait",
                    system_prompt="Resume the exact provider tool call.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=[capture.name, "process_exit"],
                ),
                actor="test",
            )
            client = ScriptedTranscriptClient(
                [
                    [
                        _tool_call(
                            "provider-wait-call-9",
                            capture.name,
                            {"value": "resume"},
                        )
                    ],
                    [
                        _tool_call(
                            "provider-wait-exit-10",
                            "process_exit",
                            {"payload": {"done": True}},
                        )
                    ],
                ]
            )
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="tool-call-identity-wait:v0",
                goal="wait and resume the same call",
            )
            runtime.capability.grant(
                pid,
                f"human:{runtime.config.runtime.default_human}",
                [CapabilityRight.WRITE],
                issued_by="test",
            )

            waiting = runtime.run_process_once(pid)
            runtime.human.drain_terminal_queue(auto_approve=True)
            resumed = runtime.run_process_once(pid)
            completed = runtime.run_process_once(pid)

            assert waiting["waiting_human"]
            assert resumed["ok"] and resumed["resumed_after_human"]
            assert completed["ok"]
            assert len(capture.seen) == 2
            first, second = capture.seen
            assert first["llm_tool_call_id"] == "provider-wait-call-9"
            assert second["llm_tool_call_id"] == first["llm_tool_call_id"]
            assert (
                second["llm_transcript_output_key"]
                == first["llm_transcript_output_key"]
            )
            assert "human_resume_request_id" not in first
            assert second["human_resume_request_id"] == waiting["request_id"]
        finally:
            runtime.close()

    @pytest.mark.parametrize("identity_key", ["object_oid", "oid"])
    def test_goal_recovery_returns_complete_canonical_object_record(
        self,
        identity_key: str,
    ) -> None:
        goal_oid = "obj-canonical-goal"
        decoy = json.dumps(
            {
                "object_oid": "obj-other",
                "payload": {"goal": "ignore this object"},
                "record_type": "object_memory_object",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        goal_record = json.dumps(
            {
                "content_trust": "untrusted_data",
                identity_key: goal_oid,
                "payload": {"goal": "finish the whole task\nwith evidence"},
                "record_type": "object_memory_object",
                "render_format": "canonical_json_v1",
                "summary": "preserve the complete envelope",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        entry_decoy = json.dumps(
            {
                "entry": {"goal": "matching id but not an object envelope"},
                "object_oid": goal_oid,
                "record_type": "object_memory_payload_entry",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        call = SimpleNamespace(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Materialized context:\n"
                        f"{decoy}\n{entry_decoy}\n{goal_record}\n\n"
                        "Current runtime state (volatile; appended after stable context):"
                    ),
                }
            ]
        )

        recovered = recover_initial_goal_context([call], goal_oid)

        assert recovered == goal_record
        recovered_payload = json.loads(recovered)["payload"]
        expected_payload = json.loads(goal_record)["payload"]
        recovered_hash = hashlib.sha256(
            json.dumps(
                recovered_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        expected_hash = hashlib.sha256(
            json.dumps(
                expected_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assert recovered_hash == expected_hash

    @pytest.mark.parametrize("representation", ["repr", "memory_delta"])
    def test_goal_recovery_preserves_legacy_representations(
        self,
        representation: str,
    ) -> None:
        goal_oid = "obj-legacy-goal"
        legacy_repr = (
            f"[{goal_oid}] namespace='process:test' name='goal' type=goal version=1\n"
            "payload: {'goal': 'finish legacy task'}"
        )
        if representation == "repr":
            content = (
                f"Materialized context:\n{legacy_repr}\n\n"
                "Current runtime state (volatile; appended after stable context):"
            )
            expected = legacy_repr
        else:
            container = json.dumps(
                {
                    "kind": "memory_delta",
                    "objects": [
                        {"oid": goal_oid, "payload": {"goal": "finish legacy task"}}
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            content = f"prefix\n---\n{container}\n---\nsuffix"
            expected = json.dumps(
                {"oid": goal_oid, "payload": {"goal": "finish legacy task"}},
                ensure_ascii=False,
                sort_keys=True,
            )
        call = SimpleNamespace(
            messages=[{"role": "user", "content": content}]
        )

        assert recover_initial_goal_context([call], goal_oid) == expected

    def test_agent_image_jit_tool_exposure_defaults_to_direct(self) -> None:
        image = AgentImage(image_id="default-jit-exposure:v0", name="default-jit-exposure")

        assert image.jit_tool_exposure == JIT_TOOL_EXPOSURE_DIRECT
        assert JIT_TOOL_EXPOSURES == {JIT_TOOL_EXPOSURE_DIRECT, JIT_TOOL_EXPOSURE_MULTIPLEXED}

    def test_image_only_system_prompt_is_exact_image_prompt(self) -> None:
        image = AgentImage(
            image_id="mini-compatible:v0",
            name="mini-compatible",
            system_prompt="  Use only the bash tool.\n",
            prompt_mode=PROMPT_MODE_IMAGE_ONLY,
        )

        prompt = build_system_prompt(image)

        assert prompt == "  Use only the bash tool.\n"
        assert "Agent libOS" not in prompt
        assert "fallback JSON action" not in prompt

    def test_minimal_runtime_prompt_does_not_inject_libos_planner_protocol(self) -> None:
        image = AgentImage(
            image_id="minimal:v0",
            name="minimal",
            system_prompt="Image-owned behavior.",
            prompt_mode=PROMPT_MODE_MINIMAL_RUNTIME,
        )

        prompt = build_system_prompt(image)

        assert "Image-owned behavior." in prompt
        assert "Available tools are supplied through the model tool schema" in prompt
        assert "You are the execution planner running inside Agent libOS" not in prompt
        assert "fallback JSON action" not in prompt
        assert prompt.index("Available tools are supplied") < prompt.index(
            "Image-owned behavior."
        )

    def test_libos_default_prompt_keeps_existing_runtime_envelope(self) -> None:
        image = AgentImage(
            image_id="native:v0",
            name="native",
            system_prompt="Native image.",
            prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
        )

        prompt = build_system_prompt(image)

        assert "You are the execution planner running inside Agent libOS" in prompt
        assert "Native image." in prompt
        assert "fallback JSON action" not in prompt
        assert "Cumulative completion contract:" in prompt
        assert prompt.index("You are the execution planner") < prompt.index(
            "You may write ordinary assistant text"
        )
        assert prompt.index("You may write ordinary assistant text") < prompt.index(
            "Cumulative completion contract:"
        )
        assert prompt.index("Cumulative completion contract:") < prompt.index(
            "Current AgentImage: native:v0"
        )

    def test_image_only_runtime_quantum_does_not_inject_runtime_user_instructions(self) -> None:
        runtime = Runtime.open("local")
        try:
            runtime.register_image(
                AgentImage(
                    image_id="mini-compatible:v0",
                    name="mini-compatible",
                    system_prompt="Use only model-supplied tool schemas.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["process_exit"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            client = PromptRecordingClient()
            runtime.llm.client = client
            pid = runtime.process.spawn(image="mini-compatible:v0", goal="fix the repository")

            result = runtime.run_next_process_once()

            assert result["ok"], result
            assert client.system_prompts == ["Use only model-supplied tool schemas."]
            assert len(client.user_prompts) == 1
            user_prompt = client.user_prompts[0]
            assert user_prompt == "fix the repository"
            assert "object_memory_object" not in user_prompt
            assert "content_trust" not in user_prompt
            assert "Available tools:" not in user_prompt
            assert "input_schema" not in user_prompt
            assert "output_schema" not in user_prompt
            assert client.tool_batches[0][0]["type"] == "function"
            assert "Capabilities:" not in user_prompt
            assert "Choose the next single runtime action" not in user_prompt
            assert "Cumulative completion contract:" not in user_prompt
        finally:
            runtime.close()

    def test_image_only_runtime_quantum_replays_native_tool_transcript(self) -> None:
        runtime = Runtime.open("local")
        try:
            runtime.register_image(
                AgentImage(
                    image_id="transparent-transcript:v0",
                    name="transparent-transcript",
                    system_prompt="Exact upstream agent prompt.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["echo", "process_exit"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            client = TranscriptRecordingClient()
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="transparent-transcript:v0",
                goal="perform one upstream tool turn",
            )

            first = runtime.run_process_once(pid)
            second = runtime.run_process_once(pid)

            assert first["action"]["action"] == "echo"
            assert second["action"]["action"] == "process_exit"
            assert client.message_batches[0] == [
                {"role": "system", "content": "Exact upstream agent prompt."},
                {"role": "user", "content": "perform one upstream tool turn"},
            ]
            replay = client.message_batches[1]
            assert [message["role"] for message in replay] == [
                "system",
                "user",
                "assistant",
                "tool",
            ]
            assert replay[2]["tool_calls"] == [
                {
                    "id": "transparent_echo",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"value": "upstream-result"}',
                    },
                }
            ]
            assert replay[3]["tool_call_id"] == "transparent_echo"
            assert replay[3]["name"] == "echo"
            assert "upstream-result" in replay[3]["content"]
            serialized = json.dumps(replay, sort_keys=True)
            assert "object_memory_object" not in serialized
            assert "content_trust" not in serialized
            assert "Agent libOS" not in serialized
        finally:
            runtime.close()

    def test_image_only_fails_before_provider_when_full_io_is_disabled(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False),
        )
        runtime = Runtime.open("local", config=config)
        try:
            runtime.register_image(
                AgentImage(
                    image_id="transparent-no-retention:v0",
                    name="transparent-no-retention",
                    system_prompt="Exact upstream agent prompt.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["process_exit"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            client = TranscriptRecordingClient()
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="transparent-no-retention:v0",
                goal="must not reach the provider",
            )

            result = runtime.run_process_once(pid)

            assert not result["ok"]
            assert result["reason"] == "image_only_full_io_required"
            assert client.message_batches == []
            assert any(
                record.action == "llm.action_failed"
                and record.decision.get("reason") == "image_only_full_io_required"
                for record in runtime.audit.trace(actor=pid)
            )
        finally:
            runtime.close()

    def test_image_only_structured_goal_uses_canonical_json(self) -> None:
        runtime = Runtime.open("local")
        try:
            runtime.register_image(
                AgentImage(
                    image_id="transparent-structured-goal:v0",
                    name="transparent-structured-goal",
                    system_prompt="Exact structured-goal prompt.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["process_exit"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            client = ScriptedTranscriptClient(
                [[_tool_call("structured_exit", "process_exit", {"payload": {"done": True}})]]
            )
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="transparent-structured-goal:v0",
                goal={"zeta": [2, 1], "alpha": {"done": False}},
            )

            result = runtime.run_process_once(pid)

            assert result["ok"], result
            assert client.message_batches[0][1] == {
                "role": "user",
                "content": '{"alpha": {"done": false}, "zeta": [2, 1]}',
            }
        finally:
            runtime.close()

    def test_image_only_transcript_survives_runtime_reopen(self, tmp_path: Any) -> None:
        database = tmp_path / "transparent-transcript.sqlite"
        runtime = Runtime.open(database)
        try:
            runtime.register_image(
                AgentImage(
                    image_id="transparent-reopen:v0",
                    name="transparent-reopen",
                    system_prompt="Exact reopen prompt.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["echo", "process_exit"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            runtime.llm.client = ScriptedTranscriptClient(
                [[_tool_call("reopen_echo", "echo", {"value": "persisted"})]]
            )
            pid = runtime.process.spawn(
                image="transparent-reopen:v0",
                goal="continue after reopening",
            )

            first = runtime.run_process_once(pid)

            assert first["ok"], first
        finally:
            runtime.close()

        reopened = Runtime.open(database)
        try:
            client = ScriptedTranscriptClient(
                [[_tool_call("reopen_exit", "process_exit", {"payload": {"done": True}})]]
            )
            reopened.llm.client = client

            second = reopened.run_process_once(pid)

            assert second["ok"], second
            replay = client.message_batches[0]
            assert [message["role"] for message in replay] == [
                "system",
                "user",
                "assistant",
                "tool",
            ]
            assert replay[1]["content"] == "continue after reopening"
            assert replay[2]["tool_calls"][0]["id"] == "reopen_echo"
            assert replay[3]["tool_call_id"] == "reopen_echo"
            assert "persisted" in replay[3]["content"]
            assert "result_oid" not in replay[3]["content"]
            assert "tool_id" not in replay[3]["content"]
        finally:
            reopened.close()

    def test_image_only_action_repair_is_not_retained_in_transcript(self) -> None:
        runtime = Runtime.open("local")
        try:
            runtime.register_image(
                AgentImage(
                    image_id="transparent-repair:v0",
                    name="transparent-repair",
                    system_prompt="Exact repair prompt.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["echo", "process_exit"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            client = ScriptedTranscriptClient(
                [
                    [_tool_call("repair_bad", "not_a_visible_tool", {})],
                    [_tool_call("repair_echo", "echo", {"value": "repaired"})],
                    [_tool_call("repair_exit", "process_exit", {"payload": {"done": True}})],
                ]
            )
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="transparent-repair:v0",
                goal="repair without transcript pollution",
            )

            first = runtime.run_process_once(pid)
            second = runtime.run_process_once(pid)

            assert first["ok"] and second["ok"]
            assert "could not be dispatched" in client.message_batches[1][-1]["content"]
            replay = client.message_batches[2]
            assert [message["role"] for message in replay] == [
                "system",
                "user",
                "assistant",
                "tool",
            ]
            assert all(
                "could not be dispatched" not in str(message.get("content", ""))
                for message in replay
            )
            assert replay[2]["tool_calls"][0]["id"] == "repair_echo"
        finally:
            runtime.close()

    def test_image_only_wait_resume_completes_the_native_tool_pair(self) -> None:
        runtime = Runtime.open("local")
        try:
            runtime.register_image(
                AgentImage(
                    image_id="transparent-wait:v0",
                    name="transparent-wait",
                    system_prompt="Exact wait prompt.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["receive_process_messages", "process_exit"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            client = ScriptedTranscriptClient(
                [
                    [_tool_call("wait_receive", "receive_process_messages", {})],
                    [_tool_call("wait_exit", "process_exit", {"payload": {"done": True}})],
                ]
            )
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="transparent-wait:v0",
                goal="wait for one message",
            )

            waiting = runtime.run_process_once(pid)
            runtime.messages.post(
                sender="human:test",
                recipient_pid=pid,
                subject="resume transparent transcript",
                payload={"ready": True},
            )
            resumed = runtime.run_process_once(pid)
            completed = runtime.run_process_once(pid)

            assert waiting["waiting_message"]
            assert resumed["ok"] and resumed["resumed_after_message"]
            assert completed["ok"]
            assert len(client.message_batches) == 2
            replay = client.message_batches[1]
            assert [message["role"] for message in replay] == [
                "system",
                "user",
                "assistant",
                "tool",
            ]
            assert replay[2]["tool_calls"][0]["id"] == "wait_receive"
            assert replay[3]["tool_call_id"] == "wait_receive"
            assert "resume transparent transcript" in replay[3]["content"]
        finally:
            runtime.close()

    def test_image_only_tool_failure_is_replayed_as_model_projection(self) -> None:
        runtime = Runtime.open("local")
        try:
            runtime.register_image(
                AgentImage(
                    image_id="transparent-tool-failure:v0",
                    name="transparent-tool-failure",
                    system_prompt="Exact failure prompt.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["read_memory_object", "process_exit"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            client = ScriptedTranscriptClient(
                [
                    [_tool_call("missing_read", "read_memory_object", {"name": "missing"})],
                    [_tool_call("failure_exit", "process_exit", {"payload": {"done": True}})],
                ]
            )
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="transparent-tool-failure:v0",
                goal="observe one tool failure",
            )

            failed_tool = runtime.run_process_once(pid)
            completed = runtime.run_process_once(pid)

            assert failed_tool["ok"] and not failed_tool["result"]["ok"]
            assert completed["ok"]
            tool_message = client.message_batches[1][-1]
            assert tool_message["role"] == "tool"
            assert tool_message["tool_call_id"] == "missing_read"
            projection = json.loads(tool_message["content"])
            assert projection["ok"] is False
            assert "error" in projection
            assert "result_oid" not in projection
            assert "tool_id" not in projection
        finally:
            runtime.close()

    def test_image_only_exec_starts_a_new_transcript_anchor(self) -> None:
        runtime = Runtime.open("local")
        try:
            for image_id, prompt in (
                ("transparent-before-exec:v0", "Before exec."),
                ("transparent-after-exec:v0", "After exec."),
            ):
                runtime.register_image(
                    AgentImage(
                        image_id=image_id,
                        name=image_id.split(":", 1)[0],
                        system_prompt=prompt,
                        prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                        default_tools=["echo", "process_exit"],
                        context_policy="recency_first",
                    ),
                    actor="test",
                )
            client = ScriptedTranscriptClient(
                [
                    [_tool_call("before_exec_echo", "echo", {"value": "old"})],
                    [_tool_call("after_exec_exit", "process_exit", {"payload": {"done": True}})],
                ]
            )
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="transparent-before-exec:v0",
                goal="old goal",
            )
            runtime.capability.grant(
                pid,
                "image:transparent-after-exec:v0",
                [CapabilityRight.READ],
                issued_by="test",
            )

            first = runtime.run_process_once(pid)
            runtime.exec_process(
                pid,
                "transparent-after-exec:v0",
                goal="new goal",
                preserve_memory=False,
            )
            second = runtime.run_process_once(pid)

            assert first["ok"] and second["ok"]
            assert client.message_batches[1] == [
                {"role": "system", "content": "After exec."},
                {"role": "user", "content": "new goal"},
            ]
        finally:
            runtime.close()

    def test_image_only_parallel_stop_persists_non_effect_cancellations(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, parallel_tool_calls=True),
        )
        runtime = Runtime.open("local", config=config)
        try:
            runtime.register_image(
                AgentImage(
                    image_id="transparent-parallel:v0",
                    name="transparent-parallel",
                    system_prompt="Exact parallel prompt.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["process_exit", "echo"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            runtime.llm.client = ScriptedTranscriptClient(
                [[
                    _tool_call("parallel_exit", "process_exit", {"payload": {"done": True}}),
                    _tool_call("parallel_skipped", "echo", {"value": "must-not-run"}),
                ]]
            )
            pid = runtime.process.spawn(
                image="transparent-parallel:v0",
                goal="stop a parallel batch",
            )

            result = runtime.run_process_once(pid)

            assert result["ok"] and result["executed_count"] == 1
            call = runtime.store.get_latest_llm_call(
                pid=pid,
                purpose="action_selection",
            )
            assert call is not None
            marker = call.request_options["image_only_transcript"]
            rows = runtime.store.list_llm_tool_outputs(
                pid=pid,
                response_id=marker["output_key"],
            )
            assert {row["call_id"] for row in rows} == {
                "parallel_exit",
                "parallel_skipped",
            }
            skipped = next(row for row in rows if row["call_id"] == "parallel_skipped")
            envelope = json.loads(skipped["output_text"])
            cancellation = json.loads(envelope["content"])
            assert envelope["synthetic"] is True
            assert cancellation == {
                "cancelled": True,
                "effect_started": False,
                "ok": False,
                "reason": "process_terminal",
            }
            assert not any(
                record.action == "tool.call"
                and record.decision.get("tool") == "echo"
                for record in runtime.audit.trace(actor=pid)
            )
        finally:
            runtime.close()

    def test_image_only_historical_sensitive_tool_output_gates_next_egress(self) -> None:
        runtime = Runtime.open("local")
        try:
            runtime.register_image(
                AgentImage(
                    image_id="transparent-ifc:v0",
                    name="transparent-ifc",
                    system_prompt="Exact IFC prompt.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["read_memory_object", "process_exit"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            client = ScriptedTranscriptClient(
                [[_tool_call("classified_read", "read_memory_object", {"name": "classified"})]]
            )
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="transparent-ifc:v0",
                goal="read classified data once",
            )
            runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {"secret": "TRANSCRIPT_SECRET_SENTINEL"},
                metadata=ObjectMetadata(sensitivity="secret"),
                name="classified",
            )

            first = runtime.run_process_once(pid)
            second = runtime.run_process_once(pid)

            assert first["ok"], first
            assert not second["ok"]
            assert "data-flow denied egress" in second["error"]
            assert len(client.message_batches) == 1
            result_oid = first["result"]["result_oid"]
            request = [
                record
                for record in runtime.audit.trace(actor=pid)
                if record.action == "llm.request"
            ][-1]
            assert runtime.process.get(pid).goal_oid in request.input_refs
            assert result_oid in request.input_refs
            decisions = runtime.store.list_data_flow_decisions(pid=pid, outcome="deny")
            assert decisions[-1].labels.sensitivity.value == "secret"
        finally:
            runtime.close()

    def test_runtime_prompt_keeps_append_only_context_ahead_of_volatile_state(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="stable goal")
            process = runtime.process.get(pid)
            capability = runtime.capability.grant(
                pid,
                "filesystem:workspace:report.txt",
                [CapabilityRight.READ],
                issued_by="test",
            )
            first_process = replace(
                process,
                pid="pid-first",
                state_generation=1,
                status_message="first quantum",
            )
            second_process = replace(
                process,
                pid="pid-second",
                state_generation=99,
                status_message="second quantum",
            )
            first_context = MaterializedContext(
                text="stable context line one\nstable context line two",
                object_refs=["obj-one"],
                token_count=10,
                omitted_objects=[],
                policy_used="recency_first",
            )
            second_context = MaterializedContext(
                text=(
                    "stable context line one\nstable context line two\n"
                    "new append-only entry"
                ),
                object_refs=["obj-one", "obj-two"],
                token_count=20,
                omitted_objects=["obj-three"],
                policy_used="recency_first",
            )
            event = Event(
                event_id="evt-second",
                type=EventType.PROCESS_SIGNAL,
                source="human:owner",
                target="pid-second",
                payload={"signal": "resume"},
                priority=EventPriority.NORMAL,
                created_at="2026-01-01T00:00:00+00:00",
            )
            available_first = [
                {
                    "skill_id": "agent-libos-workspace-navigation",
                    "description": "Inspect workspace files.",
                    "active": False,
                }
            ]
            available_second = [{**available_first[0], "active": True}]
            tools = [
                {
                    "name": "read_text_file",
                    "spec_json": json.dumps(
                        {
                            "description": "Read text.",
                            "input_schema": {"type": "object"},
                            "output_schema": {"type": "object"},
                            "policy": {"effect": "read"},
                        }
                    ),
                }
            ]

            first = build_user_prompt(
                first_process,
                first_context,
                [],
                [],
                tools,
                available_skills=available_first,
                original_goal_context="stable goal",
            )
            second = build_user_prompt(
                second_process,
                second_context,
                [event],
                [capability],
                tools,
                available_skills=available_second,
                original_goal_context="stable goal",
            )

            context_end = first.index(first_context.text) + len(first_context.text)
            common_prefix = 0
            for left, right in zip(first, second):
                if left != right:
                    break
                common_prefix += 1
            assert common_prefix >= context_end
            assert "input_schema" not in first
            assert "output_schema" not in first
            assert "read_text_file" not in first
            assert "state_generation" not in second
            assert process.image_id not in second
            assert "tool_table" not in second
            first_catalog = first.split("Retained original goal", 1)[0]
            second_catalog = second.split("Retained original goal", 1)[0]
            assert '"active"' not in first_catalog
            assert '"active"' not in second_catalog
            assert first_catalog == second_catalog
        finally:
            runtime.close()

    def test_projected_message_notice_mapping_preserves_mandatory_read_directive(self) -> None:
        pid = "pid-message-projection"
        process = SimpleNamespace(
            pid=pid,
            parent_pid=None,
            working_directory=".",
            goal_oid="obj-goal",
            checkpoint_head=None,
            status_message=None,
            model_tool_table=["read_process_messages"],
        )
        sentinel = "RAW_MESSAGE_METADATA_SENTINEL"
        notice = Event(
            event_id="evt-notice-sensitive",
            type=EventType.PROCESS_MESSAGE_NOTICE,
            source="runtime",
            target=pid,
            payload={
                "kind": "interrupt",
                "count": 2,
                "message_ids": [sentinel],
                "correlation_ids": [sentinel],
                "instruction": sentinel,
            },
            priority=EventPriority.HIGH,
            created_at="2026-01-01T00:00:00+00:00",
        )
        projected = project_prompt_events([notice])
        context = MaterializedContext(
            text="stable goal context",
            object_refs=[],
            token_count=4,
            omitted_objects=[],
            policy_used="recency_first",
        )

        prompt = build_user_prompt(
            process,
            context,
            projected.visible_records,
            [],
            [],
        )

        assert "Pending explicit process input (mandatory control action):" in prompt
        assert "read_process_messages" in prompt
        assert '"count":2' in prompt
        assert '"kind":"interrupt"' in prompt
        assert sentinel not in prompt
        assert "message_ids" not in prompt
        assert "correlation_ids" not in prompt

    def test_fallback_json_opt_in_adds_only_compatibility_input_schemas(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="compatibility")
            process = runtime.process.get(pid)
            context = MaterializedContext(
                text="compatibility goal context",
                object_refs=[],
                token_count=4,
                omitted_objects=[],
                policy_used="recency_first",
            )
            tools = [
                {
                    "name": "process_exit",
                    "spec_json": json.dumps(
                        {
                            "description": "Exit.",
                            "input_schema": {"type": "object"},
                            "output_schema": {"type": "object"},
                            "policy": {"effect": "write"},
                        }
                    ),
                }
            ]

            prompt = build_user_prompt(
                process,
                context,
                [],
                [],
                tools,
                fallback_json_actions=True,
            )

            assert "Compatibility JSON action protocol" in prompt
            assert '"input_schema"' in prompt
            assert '"output_schema"' not in prompt
            assert '"policy"' not in prompt
        finally:
            runtime.close()

    def test_loaded_skill_uses_native_markdown_and_compact_metadata(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="use a Skill")
            process = runtime.process.get(pid)
            context = MaterializedContext(
                text="Skill projection context",
                object_refs=[],
                token_count=3,
                omitted_objects=[],
                policy_used="recency_first",
            )
            instructions = (
                "# Native Skill heading\n\n"
                "Use `alpha_tool` directly.\n\n"
                "- Preserve this Markdown list."
            )
            skill = {
                "skill_id": "projection-skill",
                "name": "SHOULD_NOT_RENDER_NAME",
                "version": "SHOULD_NOT_RENDER_VERSION",
                "description": "SHOULD_NOT_RENDER_DESCRIPTION",
                "instructions": instructions,
                "allowed_tools": ["zeta_tool", "alpha_tool"],
                "actions": [
                    {
                        "name": "project",
                        "use_cases": ["zeta case", "alpha case"],
                        "input_schema": {"INPUT_SCHEMA_MARKER": True},
                        "output_schema": {"OUTPUT_SCHEMA_MARKER": True},
                        "examples": [{"EXAMPLE_MARKER": True}],
                        "failure_modes": ["FAILURE_MODE_MARKER"],
                    }
                ],
                "jit_tools": [
                    {
                        "name": "jit_projection",
                        "description": "JIT_DESCRIPTION_MARKER",
                        "input_schema": {"JIT_SCHEMA_MARKER": True},
                        "tests": [{"JIT_TEST_MARKER": True}],
                        "source_sha256": "JIT_HASH_MARKER",
                    }
                ],
                "required_capabilities": [
                    {
                        "resource": "filesystem:workspace:report.md",
                        "rights": ["write", "read"],
                    }
                ],
                "resources": [
                    {
                        "path": "references/guide.md",
                        "kind": "text",
                        "size_bytes": 42,
                        "sha256": "RESOURCE_HASH_MARKER",
                    }
                ],
                "package_sha256": "PACKAGE_HASH_MARKER",
            }

            prompt = build_user_prompt(
                process,
                context,
                [],
                [],
                [],
                skills=[skill],
            )

            assert f"## projection-skill\n\n{instructions}" in prompt
            assert '"instructions"' not in prompt
            assert "\\n\\nUse `alpha_tool`" not in prompt
            assert '"allowed_tools":["alpha_tool","zeta_tool"]' in prompt
            assert '"name":"project"' in prompt
            assert '"use_cases":["alpha case","zeta case"]' in prompt
            assert '"jit_tools":["jit_projection"]' in prompt
            assert '"kind":"text"' in prompt
            assert '"path":"references/guide.md"' in prompt
            assert '"size_bytes":42' in prompt
            assert '"rights":["read","write"]' in prompt
            for omitted in (
                "SHOULD_NOT_RENDER_NAME",
                "SHOULD_NOT_RENDER_VERSION",
                "SHOULD_NOT_RENDER_DESCRIPTION",
                "INPUT_SCHEMA_MARKER",
                "OUTPUT_SCHEMA_MARKER",
                "EXAMPLE_MARKER",
                "FAILURE_MODE_MARKER",
                "JIT_DESCRIPTION_MARKER",
                "JIT_SCHEMA_MARKER",
                "JIT_TEST_MARKER",
                "JIT_HASH_MARKER",
                "RESOURCE_HASH_MARKER",
                "PACKAGE_HASH_MARKER",
            ):
                assert omitted not in prompt

            without_skills = build_user_prompt(
                process,
                context,
                [],
                [],
                [],
                skills=[],
            )
            assert "Loaded skills:" not in without_skills
        finally:
            runtime.close()

    def test_prompt_catalog_and_authority_lists_have_deterministic_order(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="stable ordering")
            process = runtime.process.get(pid)
            capabilities = [
                runtime.capability.grant(
                    pid,
                    resource,
                    rights,
                    issued_by="test",
                )
                for resource, rights in (
                    ("filesystem:workspace:zeta", [CapabilityRight.WRITE]),
                    (
                        "filesystem:workspace:alpha",
                        [CapabilityRight.WRITE, CapabilityRight.READ],
                    ),
                )
            ]
            context = MaterializedContext(
                text="Deterministic projection context",
                object_refs=[],
                token_count=3,
                omitted_objects=[],
                policy_used="recency_first",
            )
            tools = [
                {
                    "name": name,
                    "spec_json": json.dumps(
                        {
                            "description": f"{name} description",
                            "input_schema": {"type": "object"},
                        }
                    ),
                }
                for name in ("zeta_tool", "alpha_tool")
            ]
            skills = [
                {
                    "skill_id": name,
                    "instructions": f"Instructions for {name}.",
                }
                for name in ("zeta-skill", "alpha-skill")
            ]
            available_skills = [
                {"skill_id": name, "description": f"Discover {name}."}
                for name in ("zeta-available", "alpha-available")
            ]
            requestable = [
                {"resource": resource, "rights": rights}
                for resource, rights in (
                    ("filesystem:workspace:zeta", ["write"]),
                    ("filesystem:workspace:alpha", ["write", "read"]),
                )
            ]

            first = build_user_prompt(
                process,
                context,
                [],
                capabilities,
                tools,
                skills=skills,
                available_skills=available_skills,
                requestable_capabilities=requestable,
                fallback_json_actions=True,
            )
            second = build_user_prompt(
                process,
                context,
                [],
                list(reversed(capabilities)),
                list(reversed(tools)),
                skills=list(reversed(skills)),
                available_skills=list(reversed(available_skills)),
                requestable_capabilities=list(reversed(requestable)),
                fallback_json_actions=True,
            )

            assert first == second
        finally:
            runtime.close()

    def test_prompt_renders_every_event_supplied_by_context_projection(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="handle events")
            process = runtime.process.get(pid)
            context = MaterializedContext(
                text="Event projection context",
                object_refs=[],
                token_count=3,
                omitted_objects=[],
                policy_used="recency_first",
            )
            events = [
                Event(
                    event_id=f"actionable-event-{index:02d}",
                    type=EventType.PROCESS_SIGNAL,
                    source="human:owner",
                    target=pid,
                    payload={"index": index},
                    priority=EventPriority.NORMAL,
                    created_at=f"2026-01-01T00:00:{index:02d}+00:00",
                )
                for index in range(12)
            ]

            prompt = build_user_prompt(
                process,
                context,
                events,
                [],
                [],
            )

            assert "Recent events:" in prompt
            for event in events:
                assert prompt.count(event.event_id) == 1
        finally:
            runtime.close()

    def test_image_only_runtime_quantum_does_not_project_loaded_skill_instructions(
        self,
        tmp_path: Any,
    ) -> None:
        skill_dir = write_skill_package(
            tmp_path,
            "image-only-reviewer",
            body="Always preserve the IMAGE_ONLY_SKILL_MARKER constraint.\n",
        )
        runtime = Runtime.open("local")
        try:
            runtime.register_image(
                AgentImage(
                    image_id="image-only-with-skill:v0",
                    name="image-only-with-skill",
                    system_prompt="Use only model-supplied tool schemas.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["process_exit"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            pid = runtime.process.spawn(
                image="image-only-with-skill:v0",
                goal="review the repository",
            )
            runtime.skills.register_skill_from_path(
                skill_dir,
                actor="test",
                require_capability=False,
            )
            runtime.capability.grant(
                pid,
                "skill:image-only-reviewer",
                [CapabilityRight.EXECUTE],
                issued_by="test",
            )
            runtime.skills.activate_skill(pid, "image-only-reviewer", actor=pid)
            client = PromptRecordingClient()
            runtime.llm.client = client

            result = runtime.run_next_process_once()

            assert result["ok"], result
            assert client.user_prompts[0] == "review the repository"
            assert "IMAGE_ONLY_SKILL_MARKER" not in client.user_prompts[0]
            assert "Loaded skills:" not in client.user_prompts[0]
            assert "Available tools:" not in client.user_prompts[0]
            assert "Capabilities:" not in client.user_prompts[0]
            assert "Choose the next single runtime action" not in client.user_prompts[0]
        finally:
            runtime.close()

    def test_image_only_ignores_fallback_prompt_opt_in(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, fallback_json_actions=True),
        )
        runtime = Runtime.open("local", config=config)
        try:
            runtime.register_image(
                AgentImage(
                    image_id="image-only-fallback:v0",
                    name="image-only-fallback",
                    system_prompt="Exact image-owned system prompt.",
                    prompt_mode=PROMPT_MODE_IMAGE_ONLY,
                    default_tools=["process_exit"],
                    context_policy="recency_first",
                ),
                actor="test",
            )
            client = PromptRecordingClient()
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="image-only-fallback:v0",
                goal="finish through compatibility mode",
            )

            result = runtime.run_next_process_once()

            assert result["ok"], result
            assert client.system_prompts == ["Exact image-owned system prompt."]
            assert client.user_prompts[0] == "finish through compatibility mode"
            call = runtime.store.get_latest_llm_call(
                pid=pid,
                purpose="action_selection",
            )
            assert call is not None
            assert call.request_options["fallback_json_actions_enabled"] is False
        finally:
            runtime.close()

    def test_unknown_prompt_mode_fails_closed_at_image_registration(self) -> None:
        runtime = Runtime.open("local")
        try:
            with pytest.raises(ValidationError, match="unknown prompt_mode"):
                runtime.register_image(
                    {
                        "image_id": "bad-prompt-mode:v0",
                        "name": "bad-prompt-mode",
                        "prompt_mode": "ambient_runtime",
                    },
                    actor="test",
                )
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        "prompt_mode",
        [PROMPT_MODE_LIBOS_DEFAULT, PROMPT_MODE_MINIMAL_RUNTIME],
    )
    def test_cumulative_completion_contract_survives_runtime_reopen(
        self,
        tmp_path: Any,
        prompt_mode: str,
    ) -> None:
        database = tmp_path / f"completion-contract-{prompt_mode}.sqlite"
        image_id = f"completion-contract-{prompt_mode}:v0"
        runtime = Runtime.open(database)
        try:
            runtime.register_image(
                AgentImage(
                    image_id=image_id,
                    name=f"completion-contract-{prompt_mode}",
                    system_prompt="Complete the whole durable task.",
                    prompt_mode=prompt_mode,
                    default_tools=["echo", "process_exit"],
                    context_policy="recency_first",
                    metadata={"completion_gate": "cumulative_review"},
                ),
                actor="test",
            )
            pid = runtime.process.spawn(
                image=image_id,
                goal="finish original requirement and final evidence step",
            )
            runtime.llm.client = PromptRecordingClient(
                tool_name="echo",
                arguments={"milestone": "phase one"},
            )

            first = runtime.run_process_once(pid)

            assert first["action"]["action"] == "echo"
        finally:
            runtime.close()

        reopened = Runtime.open(database)
        try:
            client = PromptRecordingClient()
            reopened.llm.client = client

            completed = reopened.run_process_once(pid)

            assert completed["action"]["action"] == "process_exit"
            user_prompt = client.user_prompts[0]
            system_prompt = client.system_prompts[0]
            assert "Cumulative completion contract:" in system_prompt
            assert "original process goal remains authoritative" in system_prompt
            assert "does not erase" in system_prompt
            assert "unmentioned requirements" in system_prompt
            assert "A passing test proves" in system_prompt
            assert "whole" in system_prompt
            assert "goal is complete" in system_prompt
            assert "Cumulative completion contract:" not in user_prompt
            assert "identity anchor only; not an Object name or read capability" in user_prompt
            assert "cumulative-review image uses nonterminal process_exit" in user_prompt
            assert "Retained original goal contract" in user_prompt
            assert "finish original requirement and final evidence step" in user_prompt
            goal_oid = reopened.process.get(pid).goal_oid
            request_record = [
                record
                for record in reopened.audit.trace(actor=pid)
                if record.action == "llm.request"
            ][-1]
            assert goal_oid is not None
            assert goal_oid in request_record.input_refs
        finally:
            reopened.close()


class PromptRecordingClient:
    def __init__(
        self,
        *,
        tool_name: str = "process_exit",
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []
        self.tool_batches: list[list[dict[str, Any]]] = []
        self.tool_name = tool_name
        self.arguments = (
            {"payload": {"done": True}}
            if arguments is None
            else dict(arguments)
        )

    def complete_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMCompletion:
        self.system_prompts.append(str(messages[0]["content"]))
        self.user_prompts.append(str(messages[-1]["content"]))
        self.tool_batches.append(tools)
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "prompt_mode_exit",
                    "name": self.tool_name,
                    "arguments": json.dumps(self.arguments),
                }
            ],
            raw=SimpleNamespace(id="prompt_mode_raw"),
            api="chat",
            model="fake",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


class TranscriptRecordingClient:
    def __init__(self) -> None:
        self.message_batches: list[list[dict[str, Any]]] = []

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        del tools
        self.message_batches.append([dict(message) for message in messages])
        if len(self.message_batches) == 1:
            return LLMCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "transparent_echo",
                        "name": "echo",
                        "arguments": '{"value": "upstream-result"}',
                    }
                ],
                raw=SimpleNamespace(id="transparent_chat_1"),
                api="chat",
                response_id="transparent_chat_1",
                model="fake",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "transparent_exit",
                    "name": "process_exit",
                    "arguments": '{"payload": {"done": true}}',
                }
            ],
            raw=SimpleNamespace(id="transparent_chat_2"),
            api="chat",
            response_id="transparent_chat_2",
            model="fake",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


def _tool_call(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": name,
        "arguments": json.dumps(arguments, sort_keys=True),
    }


class ScriptedTranscriptClient:
    def __init__(self, tool_call_batches: list[list[dict[str, Any]]]) -> None:
        self._tool_call_batches = [list(batch) for batch in tool_call_batches]
        self.message_batches: list[list[dict[str, Any]]] = []

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        del tools
        self.message_batches.append(json.loads(json.dumps(messages)))
        tool_calls = self._tool_call_batches.pop(0)
        ordinal = len(self.message_batches)
        return LLMCompletion(
            content="",
            tool_calls=tool_calls,
            raw=SimpleNamespace(id=f"scripted_chat_{ordinal}"),
            api="chat",
            response_id=f"scripted_chat_{ordinal}",
            model="fake",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
