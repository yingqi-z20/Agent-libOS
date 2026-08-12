#!/usr/bin/env python3
"""Exercise the Manifest-v3 Host client without opening a transport.

This deterministic adapter demonstrates the public Resources, Resource
Templates, Prompts, and Completion contract.  Production transports must be
supplied by a governed session factory; the example intentionally does not
teach a direct-SDK shortcut around Runtime policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agent_libos.mcp.client import McpClientBinding, McpModernClient
from agent_libos.mcp.manifest import parse_mcp_v3_manifest_yaml_text
from agent_libos.mcp.types import (
    JsonValue,
    McpComplete,
    McpCompletionResult,
    McpPage,
    McpPrompt,
    McpPromptArgument,
    McpPromptMessage,
    McpPromptResult,
    McpResource,
    McpResourceContents,
    McpResourceTemplate,
    McpTextContent,
)
from agent_libos.models.mcp import McpServerSpec
from agent_libos.utils.serde import to_jsonable


EXAMPLE_ROOT = Path(__file__).resolve().parent


class DeterministicModernProvider:
    """No-I/O provider used only to exercise the bounded Host contract."""

    async def list_resources(
        self,
        server: McpServerSpec,
        cursor: str | None,
        *,
        deadline: float,
    ) -> McpPage[McpResource]:
        del server, deadline
        if cursor is not None:
            return McpPage(items=())
        return McpPage(
            items=(
                McpResource(
                    resource_id="demo://status",
                    name="demo-status",
                    mime_type="application/json",
                ),
            )
        )

    async def list_resource_templates(
        self,
        server: McpServerSpec,
        cursor: str | None,
        *,
        deadline: float,
    ) -> McpPage[McpResourceTemplate]:
        del server, deadline
        if cursor is not None:
            return McpPage(items=())
        return McpPage(
            items=(
                McpResourceTemplate(
                    template_id="demo://greeting/{name}",
                    name="demo-greeting",
                    mime_type="text/plain",
                ),
            )
        )

    async def read_resource(
        self,
        server: McpServerSpec,
        resource_name: str,
        variables: Mapping[str, str] | None,
        *,
        deadline: float,
    ) -> McpComplete[McpResourceContents]:
        del server, variables, deadline
        payload = (
            '{"ok":true,"source":"modern-contract"}'
            if resource_name == "demo://status"
            else resource_name.removeprefix("demo://greeting/")
        )
        return McpComplete(
            value=McpResourceContents(
                resource_id=resource_name,
                contents=(McpTextContent(text=payload),),
            )
        )

    async def list_prompts(
        self,
        server: McpServerSpec,
        cursor: str | None,
        *,
        deadline: float,
    ) -> McpPage[McpPrompt]:
        del server, deadline
        if cursor is not None:
            return McpPage(items=())
        return McpPage(
            items=(
                McpPrompt(
                    prompt_id="demo.review",
                    name="demo-review",
                    arguments=(McpPromptArgument(name="subject", required=True),),
                ),
            )
        )

    async def get_prompt(
        self,
        server: McpServerSpec,
        prompt_name: str,
        arguments: Mapping[str, str],
        *,
        deadline: float,
    ) -> McpComplete[McpPromptResult]:
        del server, deadline
        subject = arguments.get("subject", "unspecified")
        return McpComplete(
            value=McpPromptResult(
                prompt_id=prompt_name,
                messages=(
                    McpPromptMessage(
                        role="user",
                        content=McpTextContent(
                            text=f"Review {subject} for correctness and evidence."
                        ),
                    ),
                ),
                user_confirmation_required=True,
            )
        )

    async def complete(
        self,
        server: McpServerSpec,
        reference: Mapping[str, JsonValue],
        argument: Mapping[str, str],
        context: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
    ) -> McpComplete[McpCompletionResult]:
        del server, reference, context, deadline
        prefix = argument["value"]
        return McpComplete(
            value=McpCompletionResult(values=(f"{prefix}-one", f"{prefix}-two"))
        )


def main() -> int:
    manifest = parse_mcp_v3_manifest_yaml_text(
        (EXAMPLE_ROOT / "http-v3.yaml").read_text(encoding="utf-8")
    )
    binding = McpClientBinding(
        manifest=manifest,
        registry_generation=1,
        owner_id="example-host",
    )
    provider = DeterministicModernProvider()
    client = McpModernClient(
        lambda server_id: binding
        if server_id == manifest.server_id
        else (_raise_unknown(server_id)),
        resource_provider=provider,
        prompt_provider=provider,
    )

    result: dict[str, Any] = {
        "resources": client.list_resources(manifest.server_id),
        "resource_templates": client.list_resource_templates(manifest.server_id),
        "status": client.read_resource(manifest.server_id, "status"),
        "greeting": client.read_resource(
            manifest.server_id,
            "greeting",
            {"name": "agent-libos"},
        ),
        "prompts": client.list_prompts(manifest.server_id),
        "prompt": client.get_prompt(
            manifest.server_id,
            "review",
            {"subject": "the MCP contract"},
        ),
        "completion": client.complete_prompt(
            manifest.server_id,
            "prompt",
            "review",
            {"name": "subject", "value": "contract"},
        ),
    }
    print(
        json.dumps(
            {"schema_version": 1, "result": to_jsonable(result)},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _raise_unknown(server_id: str) -> McpClientBinding:
    raise KeyError(f"unknown example server: {server_id}")


if __name__ == "__main__":
    raise SystemExit(main())
