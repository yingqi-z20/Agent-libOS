from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, ValidationError as PydanticValidationError

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.tools.base import BaseAgentTool, ToolContext, ToolErrorCode, ToolPolicy
from agent_libos.tools.builtin.clock import SleepTool
from agent_libos.tools.builtin.capabilities import (
    DelegateCapabilityArgs,
    DelegateCapabilityTool,
    ListCapabilitiesArgs,
    ListCapabilitiesTool,
)
from agent_libos.tools.builtin.filesystem import ReadDirectoryTool, ReadTextFileTool
from agent_libos.tools.builtin.git import (
    GitBlameTool,
    GitDiffTool,
    GitListPullRequestsTool,
    GitListRefsTool,
    GitLogTool,
    GitShowTool,
    GitStatusTool,
)
from agent_libos.tools.builtin.shell import RunShellCommandTool


class _EmptyArgs(BaseModel):
    pass


class _RuntimePolicyTimeoutTool(BaseAgentTool[_EmptyArgs]):
    name = "runtime_policy_timeout"
    description = "Exercise active ToolPolicy timeout selection."
    args_schema = _EmptyArgs
    policy = ToolPolicy(
        side_effects=False,
        idempotent=True,
        timeout_s=DEFAULT_CONFIG.tools.standard_timeout_s,
    )

    async def execute(
        self,
        args: _EmptyArgs,
        ctx: ToolContext,
    ) -> dict[str, bool]:
        await asyncio.sleep(0.05)
        return {"completed": True}


class _ValueArgs(BaseModel):
    value: int


class _LegacyParseArgsTool(BaseAgentTool[_ValueArgs]):
    name = "legacy_parse_args"
    description = "Exercise the pre-config parse_args override ABI."
    args_schema = _ValueArgs

    def __init__(self) -> None:
        self.parse_calls = 0

    def parse_args(self, raw_args: object) -> _ValueArgs:  # type: ignore[override]
        self.parse_calls += 1
        return self.args_schema.model_validate(raw_args)

    async def execute(
        self,
        args: _ValueArgs,
        ctx: ToolContext,
    ) -> dict[str, int]:
        return {"value": args.value}


class _TypeErrorParseArgsTool(_LegacyParseArgsTool):
    name = "type_error_parse_args"

    def parse_args(self, raw_args: object) -> _ValueArgs:  # type: ignore[override]
        self.parse_calls += 1
        raise TypeError("parser-internal sentinel")


class _RuntimeCapabilityListArgsTool(BaseAgentTool[ListCapabilitiesArgs]):
    name = "list_capabilities"
    description = "Exercise active capability-list parser configuration."
    args_schema = ListCapabilitiesArgs

    async def execute(
        self,
        args: ListCapabilitiesArgs,
        ctx: ToolContext,
    ) -> dict[str, int | None]:
        return {"limit": args.limit}


def _context(config=DEFAULT_CONFIG) -> ToolContext:
    return ToolContext(
        trace_id="trace-runtime-contract",
        call_id="call-runtime-contract",
        pid="pid-runtime-contract",
        runtime=SimpleNamespace(config=config),
    )


def test_ainvoke_uses_active_runtime_policy_timeout() -> None:
    config = replace(
        DEFAULT_CONFIG,
        tools=replace(DEFAULT_CONFIG.tools, standard_timeout_s=0.001),
    )

    result = _RuntimePolicyTimeoutTool().invoke({}, _context(config))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.TIMEOUT


def test_ainvoke_supports_legacy_parse_args_override_without_config_keyword() -> None:
    tool = _LegacyParseArgsTool()

    result = tool.invoke({"value": 7}, _context())

    assert result.ok is True
    assert result.data == {"value": 7}
    assert tool.parse_calls == 1


def test_ainvoke_does_not_retry_type_error_raised_inside_legacy_parser() -> None:
    tool = _TypeErrorParseArgsTool()

    result = tool.invoke({"value": 7}, _context())

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.VALIDATION_ERROR
    assert tool.parse_calls == 1
    assert "parser-internal sentinel" not in result.model_dump_json()


def test_base_parser_receives_active_config_without_legacy_fallback() -> None:
    config = replace(
        DEFAULT_CONFIG,
        capability=replace(DEFAULT_CONFIG.capability, list_limit=150),
    )

    result = _RuntimeCapabilityListArgsTool().invoke(
        {"limit": 125},
        _context(config),
    )

    assert result.ok is True
    assert result.data == {"limit": 125}


def test_capability_list_schema_and_runtime_publish_the_same_lower_bounds() -> None:
    properties = ListCapabilitiesTool().spec().input_schema["properties"]
    limit_branches = properties["limit"]["anyOf"]
    cursor_branches = properties["after_cap_id"]["anyOf"]

    assert next(
        branch for branch in limit_branches if branch.get("type") == "integer"
    )["minimum"] == 1
    assert next(
        branch for branch in cursor_branches if branch.get("type") == "string"
    )["minLength"] == 1
    assert ListCapabilitiesArgs.model_validate(
        {"limit": 1, "after_cap_id": "cap-1"},
        strict=True,
    ).limit == 1
    for invalid in ({"limit": 0}, {"after_cap_id": ""}):
        with pytest.raises(PydanticValidationError):
            ListCapabilitiesArgs.model_validate(invalid, strict=True)


def test_capability_delegate_schema_and_runtime_publish_the_same_use_floor() -> None:
    properties = DelegateCapabilityTool().spec().input_schema["properties"]
    uses_branches = properties["uses_remaining"]["anyOf"]

    assert next(
        branch for branch in uses_branches if branch.get("type") == "integer"
    )["minimum"] == 1
    valid = {
        "child_pid": "child",
        "resource": "object:item",
        "rights": ["read"],
        "uses_remaining": 1,
    }
    assert DelegateCapabilityArgs.model_validate(valid, strict=True).uses_remaining == 1
    with pytest.raises(PydanticValidationError):
        DelegateCapabilityArgs.model_validate(
            {**valid, "uses_remaining": 0},
            strict=True,
        )


def test_runtime_schema_and_parser_accept_the_same_raised_bounds() -> None:
    config = replace(
        DEFAULT_CONFIG,
        tools=replace(
            DEFAULT_CONFIG.tools,
            max_sleep_seconds=120.0,
            filesystem_read_hard_limit_bytes=(
                DEFAULT_CONFIG.tools.filesystem_read_hard_limit_bytes + 100
            ),
            directory_entry_hard_limit=(
                DEFAULT_CONFIG.tools.directory_entry_hard_limit + 100
            ),
        ),
        shell=replace(DEFAULT_CONFIG.shell, timeout_hard_limit_s=600.0),
    )

    assert SleepTool().spec(config=config).input_schema["properties"]["seconds"][
        "maximum"
    ] == 120.0
    assert SleepTool().parse_args({"seconds": 90.0}, config=config).seconds == 90.0
    assert (
        RunShellCommandTool()
        .parse_args(
            {"argv": ["echo", "ok"], "timeout_s": 400.0},
            config=config,
        )
        .timeout_s
        == 400.0
    )
    assert (
        ReadTextFileTool()
        .parse_args(
            {
                "path": "README.md",
                "max_bytes": DEFAULT_CONFIG.tools.filesystem_read_hard_limit_bytes
                + 1,
            },
            config=config,
        )
        .max_bytes
        == DEFAULT_CONFIG.tools.filesystem_read_hard_limit_bytes + 1
    )
    assert (
        ReadDirectoryTool()
        .parse_args(
            {
                "path": ".",
                "limit": DEFAULT_CONFIG.tools.directory_entry_hard_limit + 1,
            },
            config=config,
        )
        .limit
        == DEFAULT_CONFIG.tools.directory_entry_hard_limit + 1
    )


@pytest.mark.parametrize(
    ("tool", "args"),
    (
        (SleepTool(), {"seconds": 61.0}),
        (
            RunShellCommandTool(),
            {"argv": ["echo", "ok"], "timeout_s": 301.0},
        ),
        (
            ReadTextFileTool(),
            {
                "path": "README.md",
                "max_bytes": DEFAULT_CONFIG.tools.filesystem_read_hard_limit_bytes
                + 1,
            },
        ),
        (
            ReadDirectoryTool(),
            {
                "path": ".",
                "limit": DEFAULT_CONFIG.tools.directory_entry_hard_limit + 1,
            },
        ),
    ),
)
def test_runtime_schema_parser_rejects_values_above_active_default_bounds(
    tool: BaseAgentTool,
    args: dict[str, object],
) -> None:
    with pytest.raises(JsonSchemaValidationError):
        tool.parse_args(args, config=DEFAULT_CONFIG)


def test_git_read_tools_publish_and_parse_active_defaults_and_hard_limits() -> None:
    config = replace(
        DEFAULT_CONFIG,
        git=replace(
            DEFAULT_CONFIG.git,
            status_entry_limit=1,
            status_entry_hard_limit=1,
            log_entry_limit=2,
            log_entry_hard_limit=2,
            output_max_bytes=3,
            output_hard_limit_bytes=8,
            patch_max_bytes=4,
            patch_hard_limit_bytes=8,
        ),
    )
    cases = (
        (GitStatusTool(), "limit", 1, 1),
        (GitLogTool(), "limit", 2, 2),
        (GitDiffTool(), "max_bytes", 4, 8),
        (GitShowTool(), "max_bytes", 4, 8),
        (GitBlameTool(), "max_bytes", 3, 8),
        (GitListRefsTool(), "limit", 1, 1),
        (GitListPullRequestsTool(), "limit", 1, 1),
    )

    for tool, field, expected_default, expected_maximum in cases:
        prop = tool.spec(config=config).input_schema["properties"][field]
        assert prop["default"] == expected_default
        assert prop["maximum"] == expected_maximum

    context = _context(config)
    refs = GitListRefsTool()
    raw = refs._raw_args_with_runtime_defaults({}, context)
    assert refs.parse_args(raw, config=config).limit == 1
    pull_requests = GitListPullRequestsTool()
    raw = pull_requests._raw_args_with_runtime_defaults({}, context)
    assert pull_requests.parse_args(raw, config=config).limit == 1

    with pytest.raises(JsonSchemaValidationError):
        refs.parse_args({"limit": 2}, config=config)
