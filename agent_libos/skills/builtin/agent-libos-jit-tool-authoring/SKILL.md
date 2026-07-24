---
name: agent-libos-jit-tool-authoring
description: Propose, validate, and register a process-local Deno/TypeScript tool when a reusable bounded operation is missing from the visible Agent libOS tools. Use for narrow computation or composite logic with an explicit JSON contract; never use JIT authoring to bypass Capability, approval, provider, filesystem, shell, or network controls.
allowed-tools: propose_jit_tool validate_jit_tool register_jit_tool
---
# Author a JIT tool

Prefer an existing tool. Author JIT only for reusable bounded logic with a stable JSON contract; proposals and registrations are durable state. Governed `libos.syscall` calls mean JIT is not inherently pure.

## Tool guide

### `propose_jit_tool`

This checks basic name/schema/source/test sizes and creates a process-owned `proposed` candidate. It does not compile or run source.

Use `[A-Za-z0-9_-]{1,64}`, avoid process tool names, and never use reserved multiplexed name `run_jit_tool`. Describe behavior, not implementation.

Use portable schemas: normally a root object, explicit bounded `properties`, every property in `required`, and `additionalProperties:false`. Model optional values as required nullable unions. Describe the exact JSON return; forbid `undefined`, functions, instances, streams, and cycles. Strict provider projection makes implicit optional properties unreliable.

Export `function run(args, libos)` or `async function run(args, libos)` with those names. Imports, re-exports, references, import-equals, dynamic/package loading are rejected. A `Deno` global may exist, but has no ambient read/write/net/env/run/FFI permission; do not depend on it. The supported libOS boundary is:

```ts
export async function run(args: { timezone: string }, libos: any) {
  const observed = await libos.syscall("clock.now", {timezone: args.timezone});
  return {iso8601: String(observed.iso8601), timezone: String(observed.timezone)};
}
```

`clock.now` is a canonical runtime route; its argument is `timezone` and it requires the normal `clock:now` read authority. Use only canonical syscall routes and argument contracts explicitly supplied by trusted image/Skill/Host documentation. The model has no syscall-list or syscall-schema introspection operation: do not derive a route from a model tool name, probe likely names, or treat an `unknown syscall` error as discovery. If the exact route and contract are unavailable, keep the JIT pure, use the existing governed tool, or stop for Host guidance.

Each primitive still enforces Capability, approval, provider, data-flow, budget, event, and audit rules; visibility is not authority. Proposal has no capability-request field.

Tests are JSON objects of this form:

```json
{
  "args":{"timezone":"UTC"},
  "expected":{"iso8601":"2026-01-02T03:04:05+00:00","timezone":"UTC"},
  "syscalls":[
    {
      "name":"clock.now",
      "args":{"timezone":"UTC"},
      "result":{"iso8601":"2026-01-02T03:04:05+00:00","unix_seconds":1767323045,"timezone":"UTC"}
    }
  ]
}
```

`expected` is exact JSON equality. Ordered mocks have `name`, optional exact `args`, and `result`/`payload` or `ok:false,error`. Mismatches reject at the call and leftovers fail validation. Because source can catch thrown mock rejection, never catch unexpected/order errors to pass; catch only intentional denial represented in `expected`.

Tests do not exercise declared schemas. Empty tests prove only the Deno validation path. Cover boundaries, branches, syscall order/args, and intentional denials.

Example:

```json
{
  "name":"normalize_records",
  "description":"Normalize bounded record labels and return counts.",
  "source_code":"export function run(args, libos) { ... }",
  "input_schema":{"type":"object","properties":{"records":{"type":"array","maxItems":100}},"required":["records"],"additionalProperties":false},
  "output_schema":{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"],"additionalProperties":false},
  "tests":[{"args":{"records":[]},"expected":{"count":0},"syscalls":[]}]
}
```

Preserve returned `candidate_id`; it belongs to the current process and cannot be guessed/transferred.

### `validate_jit_tool`

Call with exactly the returned candidate ID:

```json
{"candidate_id":"tcand_..."}
```

Validation performs static checks, invokes Deno, runs tests, and records bounded diagnostics. Inspect `ok`, `errors`, `warnings`, and `logs`; only `ok:true` is eligible. It does not rewrite source/schemas/tests, but persists `validated`/`rejected` status, evidence, timestamps, and audit, so it is non-idempotent.

A rejected candidate cannot be edited here. Correct it in a new proposal; do not revalidate unchanged rejection.

### `register_jit_tool`

Register only after an explicit successful validation:

```json
{"candidate_id":"tcand_..."}
```

Registration auto-validates a non-validated candidate, but explicit validation preserves diagnostics. It targets current PID with `ephemeral_process` scope and exposes no target, approver, scope, replace, lookup, unregister, or idempotency key. It rejects missing/foreign/already-registered candidates and any process binding collision.

Success returns `tool_id`, `name`, `scope`. `ephemeral_process` is ownership scope: committed bindings can rehydrate, while checkpoint forks get new identities. Registration grants no Capability, and every JIT is conservatively side-effecting/non-idempotent.

## Recommended workflow

1. Confirm no visible built-in, registered, or loaded-Skill tool already covers the operation. Keep one-off reasoning inline instead of creating a tool.
2. Define a bounded contract with explicit required/nullability and denial representation.
3. Implement self-contained TypeScript; use constant minimal syscalls after pure validation.
4. Test normal/boundary/branch/syscall/denial behavior; schemas remain separate.
5. Call `propose_jit_tool` once and preserve the exact process-owned candidate ID.
6. Validate once. On failure create a corrected candidate; on success retain evidence.
7. Call `register_jit_tool` once. Verify returned name and scope and preserve `tool_id` as the registration identity.
8. Preserve the exact registered name and input contract. In direct mode the next model request can expose the JIT as its own native schema and it is called by name. In multiplexed mode only the generic `run_jit_tool` schema is exposed: there is no injected JIT catalog or per-tool schema, so call `{"tool_name":"normalize_records","arguments":{...}}` from the retained contract—never use `name`. If that contract is no longer in context, stop instead of guessing or attempting nonexistent schema lookup.
9. Domain-check a small valid call; require a harmless invalid input to reject before execution. For syscalls, exercise safe declared denial.

## Failure and recovery

- Invalid name/schema/source/test: correct and repropose; do not weaken contracts/tests.
- Import, dynamic-code, Deno permission, unknown syscall, or unavailable Deno failure: remove the unsupported dependency or use the proper existing libOS primitive. Do not fall back to shell/network/filesystem access.
- `ok:false`: preserve errors/warnings/logs, create a new candidate, and validate that ID. There is no model-facing candidate edit or discard operation.
- Syscall authority/provider/data-flow/budget denial: preserve it; JIT cannot change it. Request only exact authority when intended.
- Collision or owner mismatch: never assume replacement semantics. Choose a new stable name only if it represents a genuinely distinct contract; otherwise stop for Host/operator cleanup or coordination.
- Unknown lifecycle settlement: do not repeat; no lookup/idempotency key exists. Report known identity and seek authorized Host inspection.
- Valid test but runtime schema rejection: tests and schemas are separate. Fix the declared contract in a new proposal.
- Output-schema failure after syscalls: assume preceding effects may have occurred; schema rejection does not roll them back. Verify state through an independent authoritative read before deciding any recovery.
- Unknown tool-call settlement is potentially effectful. Replay only with authoritative primitive evidence of safety.

## Completion evidence

Require exact candidate ID, `ok:true` validation, matching registration/tool ID, a domain-valid smoke result, and pre-execution rejection of invalid input. Syscall tools also need expected denial evidence and mutation read-back.

Report contract, candidate/tool IDs, exposure mode, warnings, smoke result, and unverified effects. Never claim registration granted authority, tests proved schemas, or scope means immediate disappearance.
