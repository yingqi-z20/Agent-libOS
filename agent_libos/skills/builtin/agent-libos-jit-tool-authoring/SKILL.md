---
name: agent-libos-jit-tool-authoring
description: Propose, validate, and register a process-local Deno/TypeScript JIT tool. Use for unavailable reusable deterministic computations, bounded transformations, or repeated workflows that may use controlled libOS syscalls.
allowed-tools: propose_jit_tool validate_jit_tool register_jit_tool
---
# Author JIT tools

## Workflow

1. Prefer an existing built-in or registered tool when it already models the operation.
2. Propose minimal TypeScript exporting `run(args, libos)`; both parameter identifiers must be exactly `args` and `libos`. Treat `args` as untrusted and validate its shape, types, required fields, and size.
3. Keep source import-free: no static/dynamic imports, re-exports, TypeScript references, or import-equals. Access Agent libOS only through `libos.syscall()`.
4. Add deterministic tests for success, malformed/edge input, and denied/error paths. For syscall tools, provide an ordered `syscalls` mock array with each expected name, optional exact args and ok/error, and result/payload.
5. Validate and address every error; treat warnings and bounded logs as evidence, not success by themselves.
6. Register only a validated candidate, then call the resulting process-local tool with a small verification case.

## Boundaries and safety

- JIT code runs in the no-ambient-permission, cached-only Deno sandbox.
- Registration changes the process tool table but does not grant syscalls, capabilities, network access, or approval bypasses.
- Return compact JSON-compatible values and fail closed without hiding permission failures, validation errors, or partial results.

## Verify

Require successful validation/tests and confirm the registered tool ID, schema behavior, authority denials, and expected output.
