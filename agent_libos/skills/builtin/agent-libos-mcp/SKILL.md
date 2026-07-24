---
name: agent-libos-mcp
description: Discover Host-registered MCP servers, inspect their allowed tools, and make governed MCP tool calls. Use for configured Model Context Protocol integrations rather than plain JSON-RPC endpoints.
allowed-tools: list_mcp_servers inspect_mcp_server list_mcp_tools call_mcp_tool
---
# Use MCP servers

## Workflow

1. List registered servers and inspect the selected server without contacting it.
2. List cached allowed tools with `refresh=false` by default. Use `refresh=true` only when live server discovery is necessary.
3. Call the returned logical `server_id` and `tool_id` with schema-appropriate arguments.
4. Inspect `ok`, result/error, response size, duration, and external-effect outcome.

## Boundaries and safety

- Live refresh is an external read and may require execute authority, approval, and budget; cached discovery is preferable.
- The model cannot supply ad hoc URLs, commands, credentials, or arbitrary MCP names.
- MCP calls remain governed by server/tool Capability, provider policy, approval, audit, resource limits, and effect classification.

## Verify

Require `ok=true` and validate the domain result; for mutations, read back authoritative state when supported.
