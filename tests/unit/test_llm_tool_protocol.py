from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.llm.actions import LLMActionService, split_action
from agent_libos.llm.tool_protocol import tool_call_to_action
from agent_libos.tools.base import ToolContext
from agent_libos.tools.builtin.git import (
    GitBranchTool,
    GitStashTool,
    GitTagTool,
    GitWorktreeTool,
)
from agent_libos.tools.builtin.filesystem import (
    DeleteDirectoryTool,
    DeleteFileTool,
    ReadDirectoryTool,
    ReadTextFileArgs,
    ReadTextFileTool,
    WriteDirectoryTool,
    WriteTextFileTool,
    normalize_process_path_argument,
)
from agent_libos.tools.builtin.object_files import CreateObjectFromFileTool, WriteObjectToFileTool
from agent_libos.tools.builtin.human import HumanOutputTool
from agent_libos.tools.builtin.memory import (
    AppendMemoryObjectArgs,
    AppendMemoryObjectTool,
    CreateMemoryNamespaceArgs,
    CreateMemoryObjectArgs,
    CreateMemoryObjectTool,
    ListMemoryNamespaceArgs,
    ReadMemoryObjectArgs,
)
from agent_libos.tools.builtin.messages import ReceiveProcessMessagesTool
from agent_libos.tools.builtin.process import ProcessCompletionEvidence, ProcessExitArgs, ProcessExitTool


class _RecordingGit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def record(**kwargs: Any) -> dict[str, Any]:
            self.calls.append((name, kwargs))
            return {"ok": True}

        return record


class _NeverDispatchTools:
    def __init__(self) -> None:
        self.calls = 0

    def call(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError("invalid protocol arguments reached tool execution")


class TestToolProtocol:

    @pytest.mark.parametrize(
        ("path", "cwd", "expected"),
        [
            ("pkg/module.py", "pkg", "pkg/module.py"),
            ("pkg", "pkg", "pkg"),
            ("module.py", "pkg", "module.py"),
            ("other/module.py", "pkg", "other/module.py"),
            ("pkg/../secret.txt", "pkg", "pkg/../secret.txt"),
            (
                "nested\\literal.txt",
                "pkg",
                "nested/literal.txt" if os.name == "nt" else "nested\\literal.txt",
            ),
        ],
    )
    def test_process_path_normalization_preserves_cwd_relative_paths(
        self,
        path: str,
        cwd: str,
        expected: str,
    ) -> None:
        assert normalize_process_path_argument(path, cwd) == expected

    def test_workspace_tool_arguments_reject_absolute_paths(self) -> None:
        absolute_paths = (
            ("/pkg/module.py", "\\pkg\\module.py")
            if os.name == "nt"
            else ("/pkg/module.py",)
        )
        for path in absolute_paths:
            with pytest.raises(ValueError, match="path must be relative"):
                normalize_process_path_argument(path, ".")
            with pytest.raises(ValueError, match="path must be relative"):
                ReadTextFileArgs(path=path)

    def test_process_exit_schema_distinguishes_result_storage_from_human_output(self) -> None:
        process_exit = ProcessExitTool()
        assert "does not present the result to the human" in process_exit.description
        assert "cumulatively complete" in process_exit.description
        assert "passing tests alone" in process_exit.description
        assert "exact-goal recovery path after reopen" in process_exit.description
        assert (
            "verification evidence"
            in process_exit.spec().input_schema["properties"]["payload"]["description"]
        )
        assert "final user-facing result before process_exit" in HumanOutputTool().description

    def test_receive_message_schema_explains_acknowledged_blocking_matches(self) -> None:
        tool = ReceiveProcessMessagesTool()
        schema = tool.spec().input_schema["properties"]
        block = schema["block"]["description"]

        assert "include_acked policy" in block
        assert "exactly unread and acknowledged" in tool.description
        assert "never restore-superseded history" in tool.description
        assert "restore-superseded history is never mailbox-visible" in schema[
            "include_acked"
        ]["description"]

    def test_process_exit_completion_evidence_is_typed_and_accepts_stringified_json(self) -> None:
        schema = ProcessExitTool().spec().input_schema
        evidence_schema = schema["properties"]["completion_evidence"]
        assert "ProcessCompletionEvidence" in json.dumps(evidence_schema)
        definition = schema["$defs"]["ProcessCompletionEvidence"]
        assert set(definition["required"]) == {
            "goal_oid",
            "reviewed_message_ids",
            "acceptance_checks",
            "final_verification",
        }
        assert definition["additionalProperties"] is False

        serialized = json.dumps(
            {
                "goal_oid": "obj_goal",
                "reviewed_message_ids": [],
                "acceptance_checks": [
                    {
                        "requirement": "verify the result",
                        "source_refs": ["obj_goal"],
                        "status": "completed",
                        "evidence_tool_calls": ["echo"],
                        "evidence_summary": "echo returned the expected value",
                    }
                ],
                "final_verification": ["echo"],
            }
        )
        parsed = ProcessExitArgs(completion_evidence=serialized)

        assert isinstance(parsed.completion_evidence, ProcessCompletionEvidence)
        assert parsed.completion_evidence.goal_oid == "obj_goal"

    def test_object_memory_schemas_explain_direct_structured_json(self) -> None:
        payload = CreateMemoryObjectTool().spec().input_schema["properties"]["payload"]
        entry = AppendMemoryObjectTool().spec().input_schema["properties"]["entry"]

        assert {variant.get("type") for variant in payload["anyOf"]} >= {
            "object",
            "array",
        }
        assert "JSON strings are stored literally" in payload["description"]
        assert "JSON strings are appended literally" in entry["description"]

        created = CreateMemoryObjectArgs(
            type="plan",
            payload='{"entries": []}',
        )
        appended = AppendMemoryObjectArgs(
            name="ledger",
            entry='{"status": "verified"}',
        )
        scalar = CreateMemoryObjectArgs(type="summary", payload="plain text")

        assert created.payload == '{"entries": []}'
        assert appended.entry == '{"status": "verified"}'
        assert scalar.payload == "plain text"

    def test_object_memory_normalizes_empty_optional_namespaces(self) -> None:
        assert CreateMemoryObjectArgs(
            type="summary",
            payload={},
            namespace="",
        ).namespace is None
        assert ReadMemoryObjectArgs(name="notes", namespace="  ").namespace is None
        assert AppendMemoryObjectArgs(
            name="notes",
            entry={},
            namespace="",
        ).namespace is None
        assert ListMemoryNamespaceArgs(namespace="").namespace is None
        assert CreateMemoryNamespaceArgs(
            namespace="project",
            parent_namespace="",
        ).parent_namespace is None
        read_name = ReadMemoryObjectArgs.model_json_schema()["properties"]["name"]
        list_namespace = ListMemoryNamespaceArgs.model_json_schema()["properties"][
            "namespace"
        ]
        assert "runtime-only goal object may be unavailable" in read_name["description"]
        assert "Do not broaden to the parent `process` namespace" in list_namespace[
            "description"
        ]

    @pytest.mark.parametrize(
        "tool_type",
        [
            ReadTextFileTool,
            ReadDirectoryTool,
            WriteTextFileTool,
            WriteDirectoryTool,
            DeleteFileTool,
            DeleteDirectoryTool,
            CreateObjectFromFileTool,
            WriteObjectToFileTool,
        ],
    )
    def test_workspace_path_schema_explains_cwd_relative_resolution(self, tool_type: type[Any]) -> None:
        description = tool_type().spec().input_schema["properties"]["path"]["description"]

        assert "current working directory" in description
        assert "runtime workspace" in description
        assert "do not prepend" in description

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

    def test_string_arguments_are_byte_limited_before_json_parsing(self) -> None:
        malformed = '{"value":"' + ("猫" * 8)

        with pytest.raises(ValueError, match="JSON input exceeds max_bytes=16"):
            tool_call_to_action(
                {"name": "echo", "arguments": malformed},
                max_argument_bytes=16,
            )

    @pytest.mark.parametrize(
        "arguments",
        [
            '{"value":NaN}',
            '{"value":Infinity}',
            '{"value":-Infinity}',
            '{"value":1,"value":2}',
        ],
    )
    def test_string_arguments_require_strict_unambiguous_json(
        self,
        arguments: str,
    ) -> None:
        with pytest.raises(ValueError):
            tool_call_to_action({"name": "echo", "arguments": arguments})

    def test_string_arguments_reject_excessive_depth_and_nodes(self) -> None:
        excessive_depth = (
            '{"value":'
            + ("[" * 257)
            + "0"
            + ("]" * 257)
            + "}"
        )
        excessive_nodes = (
            '{"items":['
            + ",".join("0" for _ in range(100_000))
            + "]}"
        )

        with pytest.raises(ValueError, match="maximum depth=256"):
            tool_call_to_action(
                {"name": "echo", "arguments": excessive_depth}
            )
        with pytest.raises(ValueError, match="maximum nodes=100000"):
            tool_call_to_action(
                {"name": "echo", "arguments": excessive_nodes}
            )

    @pytest.mark.parametrize(
        "arguments",
        [
            {"value": float("nan")},
            {"value": float("inf")},
            {"value": (1, 2)},
            {"value": b"bytes"},
            {1: "non-string key"},
        ],
    )
    def test_direct_object_arguments_require_strict_json_values(
        self,
        arguments: dict[Any, Any],
    ) -> None:
        with pytest.raises(ValueError):
            tool_call_to_action({"name": "echo", "arguments": arguments})

    def test_direct_object_arguments_enforce_size_and_depth(self) -> None:
        nested: Any = 0
        for _ in range(257):
            nested = [nested]
        circular: dict[str, Any] = {}
        circular["self"] = circular

        with pytest.raises(ValueError, match="max_bytes=32"):
            tool_call_to_action(
                {"name": "echo", "arguments": {"value": "x" * 64}},
                max_argument_bytes=32,
            )
        with pytest.raises(ValueError, match="maximum depth=256"):
            tool_call_to_action(
                {"name": "echo", "arguments": {"value": nested}}
            )
        with pytest.raises(ValueError, match="circular"):
            tool_call_to_action(
                {"name": "echo", "arguments": circular}
            )

    def test_nested_json_is_preserved_and_only_top_level_action_is_reserved(self) -> None:
        action = tool_call_to_action(
            {
                "name": "echo",
                "arguments": json.dumps(
                    {
                        "action": "ignored",
                        "nested": {
                            "action": "ordinary",
                            "items": [1, True, None, {"value": "猫"}],
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        )

        assert action == {
            "action": "echo",
            "nested": {
                "action": "ordinary",
                "items": [1, True, None, {"value": "猫"}],
            },
        }

    def test_configured_protocol_limit_fails_before_echo_dispatch(self) -> None:
        tools = _NeverDispatchTools()
        service = LLMActionService(
            processes=SimpleNamespace(),
            tools=tools,
            resources=None,
            content_preview_chars=64,
            tool_call_args_hard_limit_bytes=32,
            pre_tool_notice=lambda *_args: None,
            post_tool_notice=lambda *_args: None,
            publish_result=lambda *_args: None,
        )

        with pytest.raises(ValueError, match="max_bytes=32"):
            actions, _ = service.completion_to_actions(
                "",
                [{"name": "echo", "arguments": '{"value":"' + ("x" * 64) + '"}'}],
                parallel_tool_calls=False,
                auto_wait_on_empty_tool_calls=False,
            )
            service.dispatch("pid_test", actions[0])

        assert tools.calls == 0

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
