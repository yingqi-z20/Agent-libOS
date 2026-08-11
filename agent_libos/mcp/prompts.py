"""Safe Host-only projections for MCP Prompts and Completion.

Prompt results are always marked as untrusted user context and require an
explicit user confirmation.  Nothing in this module exposes a model tool or
offers a path to a system/developer message.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agent_libos.mcp.app_policy import reject_mcp_app_text
from agent_libos.mcp.resources import (
    McpArtifactWriter,
    sanitize_provider_json,
    sanitize_resource_contents,
    sdk_content_block,
    validate_sdk_content_block,
)
from agent_libos.mcp.types import (
    McpCompletionResult,
    McpPrompt,
    McpPromptArgument,
    McpPromptMessage,
    McpPromptResult,
    McpResourceContents,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.redaction import redact_sensitive_text


def sdk_prompt(
    value: Any,
    *,
    sensitive_values: Iterable[str] = (),
) -> McpPrompt:
    arguments_value = getattr(value, "arguments", None) or ()
    if type(arguments_value) not in {tuple, list}:
        raise ValidationError("MCP prompt arguments must be a list")
    arguments: list[McpPromptArgument] = []
    seen: set[str] = set()
    for item in arguments_value:
        name = getattr(item, "name", None)
        if type(name) is not str:
            raise ValidationError("MCP prompt argument names must be text")
        if not name or name in seen:
            raise ValidationError("MCP prompt argument names must be unique and non-empty")
        if redact_sensitive_text(name, sensitive_values=sensitive_values) != name:
            raise ValidationError("MCP prompt argument name reflected an operation secret")
        seen.add(name)
        required = getattr(item, "required", False)
        if required is None:
            required = False
        if type(required) is not bool:
            raise ValidationError("MCP prompt argument required flag must be boolean")
        arguments.append(
            McpPromptArgument(
                name=name,
                title=_optional_text(getattr(item, "title", None), sensitive_values),
                description=_optional_text(
                    getattr(item, "description", None), sensitive_values
                ),
                required=required,
            )
        )
    metadata = sanitize_provider_json(
        getattr(value, "meta", None) or {}, sensitive_values=sensitive_values
    )
    if not isinstance(metadata, dict):  # pragma: no cover - call shape above
        raise ValidationError("MCP prompt metadata must be an object")
    return McpPrompt(
        # SDK adapter uses the remote name internally.  McpModernClient maps
        # this to the manifest's logical prompt_id before returning it.
        prompt_id=str(getattr(value, "name", "")),
        name=redact_sensitive_text(
            str(getattr(value, "name", "")), sensitive_values=sensitive_values
        ),
        title=_optional_text(getattr(value, "title", None), sensitive_values),
        description=_optional_text(
            getattr(value, "description", None), sensitive_values
        ),
        arguments=tuple(arguments),
        # Remote icon URLs are data, never browser fetch instructions.
        icons=(),
        metadata=metadata,
    )


def sdk_prompt_result(
    value: Any,
    *,
    server_id: str,
    logical_id: str,
    artifact_writer: McpArtifactWriter | None,
    sensitive_values: Iterable[str] = (),
    maximum_messages: int = 128,
    maximum_content_blocks: int = 256,
) -> McpPromptResult:
    messages_value = getattr(value, "messages", None)
    if type(messages_value) is not list:
        raise ValidationError("MCP prompts/get messages must be a list")
    if type(maximum_messages) is not int or maximum_messages <= 0:
        raise ValidationError("MCP maximum prompt message count is invalid")
    if len(messages_value) > maximum_messages:
        raise ValidationError("MCP prompt exceeded maximum message count")
    if type(maximum_content_blocks) is not int or maximum_content_blocks <= 0:
        raise ValidationError("MCP maximum content block count is invalid")
    if len(messages_value) > maximum_content_blocks:
        raise ValidationError("MCP prompt exceeded maximum content block count")
    for item in messages_value:
        role = getattr(item, "role", None)
        if role not in {"user", "assistant"}:
            raise ValidationError("MCP prompt message role must be user or assistant")
        validate_sdk_content_block(
            getattr(item, "content", None),
            artifact_writer=artifact_writer,
            sensitive_values=sensitive_values,
        )
    messages: list[McpPromptMessage] = []
    for item in messages_value:
        role = getattr(item, "role", None)
        if role not in {"user", "assistant"}:
            # In particular, provider-controlled system/developer roles are
            # never accepted and cannot be normalized into a privileged role.
            raise ValidationError("MCP prompt message role must be user or assistant")
        content = sdk_content_block(
            getattr(item, "content", None),
            server_id=server_id,
            logical_id=logical_id,
            artifact_writer=artifact_writer,
            sensitive_values=sensitive_values,
        )
        messages.append(McpPromptMessage(role=role, content=content))
    return McpPromptResult(
        prompt_id=logical_id,
        messages=tuple(messages),
        description=_optional_text(getattr(value, "description", None), sensitive_values),
        user_confirmation_required=True,
    )


def sanitize_prompt_result(
    value: McpPromptResult,
    *,
    server_id: str,
    logical_id: str,
    sensitive_values: Iterable[str],
    maximum_messages: int = 128,
    maximum_content_blocks: int = 256,
) -> McpPromptResult:
    if type(maximum_messages) is not int or maximum_messages <= 0:
        raise ValidationError("MCP maximum prompt message count is invalid")
    if len(value.messages) > maximum_messages:
        raise ValidationError("MCP prompt exceeded maximum message count")
    if type(maximum_content_blocks) is not int or maximum_content_blocks <= 0:
        raise ValidationError("MCP maximum content block count is invalid")
    if len(value.messages) > maximum_content_blocks:
        raise ValidationError("MCP prompt exceeded maximum content block count")
    messages: list[McpPromptMessage] = []
    for message in value.messages:
        if message.role not in {"user", "assistant"}:
            raise ValidationError("MCP prompt message role must be user or assistant")
        projected = sanitize_resource_contents(
            McpResourceContents(resource_id=logical_id, contents=(message.content,)),
            server_id=server_id,
            logical_id=logical_id,
            sensitive_values=sensitive_values,
            maximum_content_blocks=maximum_content_blocks,
        )
        content = projected.contents[0]
        messages.append(
            McpPromptMessage(
                role=message.role,
                content=content,
                provenance="untrusted_mcp_prompt",
            )
        )
    return McpPromptResult(
        prompt_id=logical_id,
        messages=tuple(messages),
        description=_optional_text(value.description, sensitive_values),
        # A custom provider cannot waive Host/user confirmation.
        user_confirmation_required=True,
    )


def sdk_completion_result(
    value: Any,
    *,
    sensitive_values: Iterable[str] = (),
    maximum_values: int = 100,
) -> McpCompletionResult:
    completion = getattr(value, "completion", None)
    if completion is None:
        raise ValidationError("MCP completion result is missing completion")
    values = getattr(completion, "values", None)
    if type(values) is not list or any(type(item) is not str for item in values):
        raise ValidationError("MCP completion values must be strings")
    if type(maximum_values) is not int or maximum_values <= 0:
        raise ValidationError("MCP maximum completion value count is invalid")
    if len(values) > maximum_values:
        raise ValidationError("MCP completion exceeded maximum value count")
    total = getattr(completion, "total", None)
    if total is not None and (type(total) is not int or total < 0):
        raise ValidationError("MCP completion total must be a non-negative integer")
    has_more = getattr(completion, "has_more", getattr(completion, "hasMore", False))
    if has_more is None:
        has_more = False
    if type(has_more) is not bool:
        raise ValidationError("MCP completion hasMore must be boolean")
    return McpCompletionResult(
        values=_project_completion_values(values, sensitive_values),
        total=total,
        has_more=has_more,
    )


def sanitize_completion_result(
    value: McpCompletionResult,
    *,
    sensitive_values: Iterable[str],
    maximum_values: int = 100,
) -> McpCompletionResult:
    """Validate and project a custom Provider Completion like the SDK path."""

    if type(value.values) is not tuple or any(
        type(item) is not str for item in value.values
    ):
        raise ValidationError("MCP completion values must be a tuple of strings")
    if type(maximum_values) is not int or maximum_values <= 0:
        raise ValidationError("MCP maximum completion value count is invalid")
    if len(value.values) > maximum_values:
        raise ValidationError("MCP completion exceeded maximum value count")
    if value.total is not None and (
        type(value.total) is not int or value.total < 0
    ):
        raise ValidationError("MCP completion total must be a non-negative integer")
    if type(value.has_more) is not bool:
        raise ValidationError("MCP completion hasMore must be boolean")
    return McpCompletionResult(
        values=_project_completion_values(value.values, sensitive_values),
        total=value.total,
        has_more=value.has_more,
    )


def _project_completion_values(
    values: Iterable[str], sensitive_values: Iterable[str]
) -> tuple[str, ...]:
    projected: list[str] = []
    for item in values:
        # Suggestions are untrusted text, not navigation/render instructions.
        # Apply Apps policy before redaction so a reflected operation secret
        # cannot hide which fail-closed rule was enforced.
        reject_mcp_app_text(item)
        projected.append(
            redact_sensitive_text(item, sensitive_values=sensitive_values)
        )
    return tuple(projected)


def _optional_text(value: Any, sensitive_values: Iterable[str]) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValidationError("MCP prompt text field must be a string")
    return redact_sensitive_text(value, sensitive_values=sensitive_values)


__all__ = [
    "sanitize_completion_result",
    "sanitize_prompt_result",
    "sdk_completion_result",
    "sdk_prompt",
    "sdk_prompt_result",
]
