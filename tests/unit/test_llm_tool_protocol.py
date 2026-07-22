from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.llm.actions import split_action
from agent_libos.llm.tool_protocol import tool_call_to_action
from agent_libos.tools.base import ToolContext
from agent_libos.tools.builtin.git import (
    GitBranchTool,
    GitStashTool,
    GitTagTool,
    GitWorktreeTool,
)


class _RecordingGit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def record(**kwargs: Any) -> dict[str, Any]:
            self.calls.append((name, kwargs))
            return {"ok": True}

        return record


class TestToolProtocol:

    def test_tool_name_wins_over_action_argument(self) -> None:
        action = tool_call_to_action({'name': 'read_directory', 'arguments': '{"action": "delete_directory", "path": "."}'})
        assert action == {'action': 'read_directory', 'path': '.'}

    def test_empty_tool_name_can_use_fallback_action_argument(self) -> None:
        action = tool_call_to_action({'name': '', 'arguments': '{"action": "read_directory", "path": "."}'})
        assert action == {'action': 'read_directory', 'path': '.'}

    def test_empty_tool_name_without_fallback_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            tool_call_to_action({'name': '', 'arguments': '{"path": "."}'})

    @pytest.mark.parametrize('arguments', ([], 0, False))
    def test_falsey_non_object_arguments_are_rejected(self, arguments: object) -> None:
        with pytest.raises(ValueError):
            tool_call_to_action({'name': 'read_directory', 'arguments': arguments})

    def test_none_or_empty_arguments_default_to_empty_object(self) -> None:
        assert tool_call_to_action({'name': 'get_current_time', 'arguments': None}) == {'action': 'get_current_time'}
        assert tool_call_to_action({'name': 'get_current_time', 'arguments': ''}) == {'action': 'get_current_time'}

    @pytest.mark.parametrize(
        ("tool_type", "operation", "extra_args"),
        [
            (GitBranchTool, "create", {"name": "topic"}),
            (GitTagTool, "create", {"name": "v1"}),
            (GitStashTool, "push", {}),
            (GitWorktreeTool, "create", {}),
        ],
    )
    def test_git_operation_survives_tool_envelope_and_executes(
        self,
        tool_type: type[GitBranchTool | GitTagTool | GitStashTool | GitWorktreeTool],
        operation: str,
        extra_args: dict[str, Any],
    ) -> None:
        tool = tool_type()
        state_token = "0" * 64
        llm_args = {
            "operation": operation,
            "expected_state_token": state_token,
            **extra_args,
        }
        schema = tool.spec().input_schema

        assert "operation" in schema["properties"]
        assert "action" not in schema["properties"]

        action = tool_call_to_action(
            {"name": tool.name, "arguments": json.dumps(llm_args)}
        )
        name, args = split_action(action)
        git = _RecordingGit()
        runtime = SimpleNamespace(git=git)
        result = tool.invoke(
            args,
            ToolContext(
                trace_id="trace",
                call_id="call",
                pid="pid_test",
                runtime=runtime,
            ),
        )

        assert name == tool.name
        assert result.ok
        assert len(git.calls) == 1
        method_name, runtime_args = git.calls[0]
        assert method_name == tool.method_name
        assert runtime_args["pid"] == "pid_test"
        assert runtime_args["action"] == operation
        assert runtime_args["expected_state_token"] == state_token
        assert "operation" not in runtime_args
        assert all(runtime_args[key] == value for key, value in extra_args.items())

        # Direct tool callers using the previous argument name remain compatible.
        legacy_args = {
            "action": operation,
            "expected_state_token": state_token,
            **extra_args,
        }
        legacy_result = tool.invoke(
            legacy_args,
            ToolContext(
                trace_id="trace",
                call_id="legacy-call",
                pid="pid_test",
                runtime=runtime,
            ),
        )
        assert legacy_result.ok
        assert git.calls[-1][1]["action"] == operation
        assert "operation" not in git.calls[-1][1]
