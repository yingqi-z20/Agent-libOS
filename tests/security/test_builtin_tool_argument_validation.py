from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent_libos.tools.base import BaseAgentTool, ToolContext, ToolErrorCode
from agent_libos.tools.builtin.filesystem import (
    DeleteDirectoryTool,
    DeleteFileTool,
    ReadDirectoryTool,
    ReadTextFileTool,
    WriteDirectoryTool,
    WriteTextFileTool,
)
from agent_libos.tools.builtin.git import GitCommitTool
from agent_libos.tools.builtin.process import ExecProcessTool, ForkChildProcessTool


_STATE_TOKEN = "0" * 64


@pytest.mark.parametrize(
    ("tool", "invalid_args", "valid_args", "field"),
    [
        (
            DeleteDirectoryTool(),
            {"path": "target", "recursive": 1},
            {"path": "target", "recursive": True},
            "recursive",
        ),
        (
            ExecProcessTool(),
            {"image": "base-agent:v0", "preserve_capabilities": 1},
            {"image": "base-agent:v0", "preserve_capabilities": True},
            "preserve_capabilities",
        ),
        (
            ForkChildProcessTool(),
            {"goal": "child", "include_parent_roots": 1},
            {"goal": "child", "include_parent_roots": True},
            "include_parent_roots",
        ),
        (
            GitCommitTool(),
            {
                "message": "commit",
                "expected_state_token": _STATE_TOKEN,
                "amend": 1,
            },
            {
                "message": "commit",
                "expected_state_token": _STATE_TOKEN,
                "amend": True,
            },
            "amend",
        ),
    ],
    ids=(
        "recursive-delete",
        "preserve-capabilities",
        "include-parent-roots",
        "git-amend",
    ),
)
@pytest.mark.parametrize("encoding", ["mapping", "json"])
def test_security_control_arguments_reject_type_coercion_before_execution(
    tool: BaseAgentTool,
    invalid_args: Mapping[str, object],
    valid_args: Mapping[str, object],
    field: str,
    encoding: str,
) -> None:
    execute = AsyncMock(side_effect=AssertionError("invalid arguments reached execution"))
    tool.execute = execute  # type: ignore[method-assign]
    raw_invalid: Mapping[str, object] | str = (
        invalid_args if encoding == "mapping" else json.dumps(invalid_args)
    )

    result = tool.invoke(
        raw_invalid,
        ToolContext(trace_id="trace", call_id="call", pid="pid_test"),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.VALIDATION_ERROR
    execute.assert_not_awaited()

    raw_valid: Mapping[str, object] | str = (
        valid_args if encoding == "mapping" else json.dumps(valid_args)
    )
    parsed = tool.parse_args(raw_valid)
    assert getattr(parsed, field) is True


@pytest.mark.parametrize(
    ('tool', 'valid_args'),
    [
        (ReadTextFileTool(), {'path': 'target.txt'}),
        (ReadDirectoryTool(), {'path': 'target'}),
        (WriteTextFileTool(), {'path': 'target.txt', 'content': 'content'}),
        (WriteDirectoryTool(), {'path': 'target'}),
        (DeleteFileTool(), {'path': 'target.txt'}),
        (DeleteDirectoryTool(), {'path': 'target'}),
    ],
    ids=(
        'read-text',
        'read-directory',
        'write-text',
        'write-directory',
        'delete-file',
        'delete-directory',
    ),
)
@pytest.mark.parametrize('encoding', ['mapping', 'json'])
def test_workspace_filesystem_tools_reject_unknown_arguments_before_execution(
    tool: BaseAgentTool,
    valid_args: Mapping[str, object],
    encoding: str,
) -> None:
    execute = AsyncMock(side_effect=AssertionError('invalid arguments reached execution'))
    tool.execute = execute  # type: ignore[method-assign]
    invalid_args = {**valid_args, 'unknown_argument': True}
    raw_invalid: Mapping[str, object] | str = (
        invalid_args if encoding == 'mapping' else json.dumps(invalid_args)
    )

    result = tool.invoke(
        raw_invalid,
        ToolContext(trace_id='trace', call_id='call', pid='pid_test'),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.VALIDATION_ERROR
    execute.assert_not_awaited()


@pytest.mark.parametrize(
    ('tool', 'valid_args'),
    [
        (ReadTextFileTool(), {'path': 'target.txt'}),
        (ReadDirectoryTool(), {'path': 'target'}),
        (WriteTextFileTool(), {'path': 'target.txt', 'content': 'content'}),
        (WriteDirectoryTool(), {'path': 'target'}),
        (DeleteFileTool(), {'path': 'target.txt'}),
        (DeleteDirectoryTool(), {'path': 'target'}),
    ],
    ids=(
        'read-text',
        'read-directory',
        'write-text',
        'write-directory',
        'delete-file',
        'delete-directory',
    ),
)
@pytest.mark.parametrize('encoding', ['mapping', 'json'])
def test_workspace_filesystem_tools_reject_absolute_paths_before_execution(
    tool: BaseAgentTool,
    valid_args: Mapping[str, object],
    encoding: str,
    tmp_path: Path,
) -> None:
    execute = AsyncMock(side_effect=AssertionError('invalid arguments reached execution'))
    tool.execute = execute  # type: ignore[method-assign]
    invalid_args = {**valid_args, 'path': f'{tmp_path}/target'}
    raw_invalid: Mapping[str, object] | str = (
        invalid_args if encoding == 'mapping' else json.dumps(invalid_args)
    )

    result = tool.invoke(
        raw_invalid,
        ToolContext(trace_id='trace', call_id='call', pid='pid_test'),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.VALIDATION_ERROR
    execute.assert_not_awaited()
