---
name: agent-libos-jsonrpc
description: Discover and call Host-registered JSON-RPC-over-HTTP endpoints through logical method IDs. Use when an integration is explicitly configured as JSON-RPC rather than MCP and requires governed remote access.
allowed-tools: list_jsonrpc_endpoints inspect_jsonrpc_endpoint call_jsonrpc_method
---
# Use JSON-RPC endpoints

## Workflow

1. List registered endpoints, optionally narrowing by stable search text.
2. Inspect the selected endpoint and its logical method allowlist without contacting it.
3. Call only the declared `endpoint_id` and `method_id`, with schema-appropriate JSON params.
4. Inspect `ok`, structured result/error, HTTP status, response size, duration, and external-effect outcome.

## Boundaries and safety

- Use the MCP Skill for MCP servers. Do not treat raw JSON-RPC methods as MCP tools.
- The model cannot provide ad hoc URLs, credentials, headers, transport commands, or arbitrary wire method names.
- Calls remain governed by endpoint/method Capability, provider policy, approval, audit, and effect classification.

## Verify

Require `ok=true` and validate the domain result; for mutations, confirm the authoritative remote state when possible.
