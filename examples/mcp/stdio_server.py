#!/usr/bin/env python3
"""Deterministic MCP 2026-07-28 demo over stdio.

The process has no credentials, reads no ambient files, and performs no
network I/O.  It is intentionally small enough to use in local smoke tests.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import Completion, PromptReference


server = MCPServer(
    "agent-libos-stdio-demo",
    version="1.0.0",
    instructions="Deterministic local demo; all returned data is static or echoed input.",
)


@server.tool(name="demo.echo", description="Echo one string without side effects.")
def echo(text: str) -> dict[str, str]:
    return {"echo": text, "transport": "stdio"}


@server.resource(
    "demo://status",
    name="demo-status",
    description="Static status for the deterministic stdio demo.",
    mime_type="application/json",
)
def status() -> str:
    return '{"ok":true,"transport":"stdio"}'


@server.resource(
    "demo://greeting/{name}",
    name="demo-greeting",
    description="Deterministic greeting resource template.",
    mime_type="text/plain",
)
def greeting(name: str) -> str:
    return f"hello {name} from stdio"


@server.prompt(name="demo.review", description="Build a deterministic review prompt.")
def review(subject: str) -> str:
    return f"Review {subject} for correctness and list concrete evidence."


@server.completion()
async def complete_review(
    reference: object,
    argument: object,
    context: object,
) -> Completion:
    del context
    if (
        not isinstance(reference, PromptReference)
        or reference.name != "demo.review"
        or getattr(argument, "name", None) != "subject"
        or type(getattr(argument, "value", None)) is not str
    ):
        return Completion(values=[], total=0, has_more=False)
    prefix = argument.value
    return Completion(
        values=[f"{prefix}-correctness", f"{prefix}-evidence"],
        total=2,
        has_more=False,
    )


if __name__ == "__main__":
    server.run("stdio")
