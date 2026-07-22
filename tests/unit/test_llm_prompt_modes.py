from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.llm.client import LLMCompletion
from agent_libos.llm.prompt import build_system_prompt
from agent_libos.models import (
    CapabilityRight,
    JIT_TOOL_EXPOSURE_DIRECT,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    JIT_TOOL_EXPOSURES,
    PROMPT_MODE_IMAGE_ONLY,
    PROMPT_MODE_LIBOS_DEFAULT,
    PROMPT_MODE_MINIMAL_RUNTIME,
)
from agent_libos.models.exceptions import ValidationError
from tests.support.skills import write_skill_package


class TestLLMPromptModes:

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
        assert "fallback JSON action" in prompt

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
            assert "Capabilities:" not in user_prompt
            assert "Choose the next single runtime action" not in user_prompt
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


class PromptRecordingClient:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def complete_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMCompletion:
        self.system_prompts.append(str(messages[0]["content"]))
        self.user_prompts.append(str(messages[-1]["content"]))
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "prompt_mode_exit",
                    "name": "process_exit",
                    "arguments": json.dumps({"payload": {"done": True}}),
                }
            ],
            raw=SimpleNamespace(id="prompt_mode_raw"),
            api="chat",
            model="fake",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
