#!/usr/bin/env python3
"""Deterministic loopback-only MCP 2026-07-28 Streamable HTTP demo."""

from __future__ import annotations

import argparse

from mcp.server.mcpserver import MCPServer
from mcp.types import Completion, PromptReference


server = MCPServer(
    "agent-libos-http-demo",
    version="1.0.0",
    instructions="Deterministic loopback demo; all returned data is static or echoed input.",
)


@server.tool(name="demo.echo", description="Echo one string without side effects.")
def echo(text: str) -> dict[str, str]:
    return {"echo": text, "transport": "streamable_http"}


@server.resource(
    "demo://status",
    name="demo-status",
    description="Static status for the deterministic HTTP demo.",
    mime_type="application/json",
)
def status() -> str:
    return '{"ok":true,"transport":"streamable_http"}'


@server.resource(
    "demo://greeting/{name}",
    name="demo-greeting",
    description="Deterministic greeting resource template.",
    mime_type="text/plain",
)
def greeting(name: str) -> str:
    return f"hello {name} from streamable_http"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    server.run(
        "streamable-http",
        host="127.0.0.1",
        port=args.port,
        streamable_http_path="/mcp",
        stateless_http=True,
    )


if __name__ == "__main__":
    main()
