from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

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
)
from agent_libos.models.exceptions import ValidationError
from tests.support.skills import write_skill_package


class TestLLMPromptModes:

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
            system_prompt="Use only the bash tool.",
            prompt_mode=PROMPT_MODE_IMAGE_ONLY,
        )

        prompt = build_system_prompt(image)

        assert prompt == "Use only the bash tool."
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
            assert "fix the repository" in user_prompt
            assert "Available tools:" not in user_prompt
            assert "input_schema" not in user_prompt
            assert "output_schema" not in user_prompt
            assert client.tool_batches[0][0]["type"] == "function"
            assert "Capabilities:" not in user_prompt
            assert "Choose the next single runtime action" not in user_prompt
            assert "Cumulative completion contract:" not in user_prompt
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

    def test_image_only_runtime_quantum_preserves_loaded_skill_instructions(
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
            assert "IMAGE_ONLY_SKILL_MARKER" in client.user_prompts[0]
            assert "Loaded skills:" in client.user_prompts[0]
            assert "Available tools:" not in client.user_prompts[0]
            assert "Capabilities:" not in client.user_prompts[0]
            assert "Choose the next single runtime action" not in client.user_prompts[0]
        finally:
            runtime.close()

    def test_image_only_fallback_opt_in_keeps_exact_system_prompt(self) -> None:
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
            runtime.process.spawn(
                image="image-only-fallback:v0",
                goal="finish through compatibility mode",
            )

            result = runtime.run_next_process_once()

            assert result["ok"], result
            assert client.system_prompts == ["Exact image-owned system prompt."]
            assert client.user_prompts[0].startswith(
                "Compatibility JSON action protocol"
            )
            assert "finish through compatibility mode" in client.user_prompts[0]
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
