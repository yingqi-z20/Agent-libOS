from __future__ import annotations

import json
from typing import Any

from agent_libos.llm.client import LLMCompletion


class RecordingActionClient:
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self.actions = list(actions)
        self.user_prompts: list[str] = []
        self.tool_batches: list[list[dict[str, Any]]] = []

    def complete_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMCompletion:
        self.user_prompts.append(str(messages[-1]["content"]))
        self.tool_batches.append(tools)
        action = self.actions.pop(0)
        name = str(action["action"])
        args = {key: value for key, value in action.items() if key != "action"}
        return LLMCompletion(
            content="",
            tool_calls=[{"id": f"test_tool_call_{len(self.user_prompts)}", "name": name, "arguments": json.dumps(args)}],
        )
