from __future__ import annotations

from dataclasses import replace

import pytest
from jsonschema import ValidationError as JsonSchemaValidationError

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.tools.base import BaseAgentTool
from agent_libos.tools.builtin.memory import ReadMemoryObjectTool
from agent_libos.tools.builtin.object_tasks import WaitObjectTaskTool


def _config_with_object_tool_bounds(
    *,
    memory_default: int,
    memory_maximum: int,
    wait_maximum: float,
) -> AgentLibOSConfig:
    return replace(
        DEFAULT_CONFIG,
        tools=replace(
            DEFAULT_CONFIG.tools,
            memory_payload_chars=memory_default,
            memory_payload_hard_limit_chars=memory_maximum,
            max_sleep_seconds=wait_maximum,
        ),
    )


@pytest.mark.parametrize(
    "config",
    (
        DEFAULT_CONFIG,
        _config_with_object_tool_bounds(
            memory_default=20_000,
            memory_maximum=300_000,
            wait_maximum=120.0,
        ),
        _config_with_object_tool_bounds(
            memory_default=1_000,
            memory_maximum=2_000,
            wait_maximum=10.0,
        ),
    ),
)
def test_object_tool_specs_and_parsers_share_active_bounds(
    config: AgentLibOSConfig,
) -> None:
    memory_tool = ReadMemoryObjectTool()
    memory_property = memory_tool.spec(config=config).input_schema["properties"][
        "max_payload_chars"
    ]
    assert memory_property["default"] == config.tools.memory_payload_chars
    assert (
        memory_property["maximum"]
        == config.tools.memory_payload_hard_limit_chars
    )
    assert (
        memory_tool.parse_args(
            {
                "name": "item",
                "max_payload_chars": config.tools.memory_payload_hard_limit_chars,
            },
            config=config,
        ).max_payload_chars
        == config.tools.memory_payload_hard_limit_chars
    )

    wait_tool = WaitObjectTaskTool()
    wait_property = wait_tool.spec(config=config).input_schema["properties"][
        "timeout_s"
    ]
    wait_number_branch = next(
        branch
        for branch in wait_property["anyOf"]
        if branch.get("type") == "number"
    )
    assert wait_property["maximum"] == config.tools.max_sleep_seconds
    assert wait_number_branch["maximum"] == config.tools.max_sleep_seconds
    assert (
        wait_tool.parse_args(
            {"task_id": "task-1", "timeout_s": config.tools.max_sleep_seconds},
            config=config,
        ).timeout_s
        == config.tools.max_sleep_seconds
    )


@pytest.mark.parametrize(
    ("tool", "args", "config"),
    (
        (
            ReadMemoryObjectTool(),
            {"name": "item", "max_payload_chars": 2_001},
            _config_with_object_tool_bounds(
                memory_default=1_000,
                memory_maximum=2_000,
                wait_maximum=10.0,
            ),
        ),
        (
            WaitObjectTaskTool(),
            {"task_id": "task-1", "timeout_s": 10.1},
            _config_with_object_tool_bounds(
                memory_default=1_000,
                memory_maximum=2_000,
                wait_maximum=10.0,
            ),
        ),
        (
            ReadMemoryObjectTool(),
            {"name": "item", "max_payload_chars": 0},
            DEFAULT_CONFIG,
        ),
        (
            WaitObjectTaskTool(),
            {"task_id": "task-1", "timeout_s": -0.1},
            DEFAULT_CONFIG,
        ),
    ),
)
def test_object_tool_parsers_reject_values_outside_active_bounds(
    tool: BaseAgentTool,
    args: dict[str, object],
    config: AgentLibOSConfig,
) -> None:
    with pytest.raises(JsonSchemaValidationError):
        tool.parse_args(args, config=config)
