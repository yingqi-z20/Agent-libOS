---
name: agent-libos-tool-protocol-diagnostics
description: Echo arbitrary JSON-compatible arguments unchanged. Use only for deterministic tool-protocol plumbing tests, schema serialization checks, or runtime diagnostics—not for application work.
allowed-tools: echo
---
# Diagnose tool protocol plumbing

## Workflow

1. Send the smallest payload that isolates argument serialization, extra-field handling, or result transport.
2. Compare the returned object structurally with the supplied arguments.
3. Stop after the protocol property is confirmed or disproved.

## Boundaries and safety

- Echo performs no domain operation, persistence, validation, permission grant, or external communication.
- Never include credentials, secrets, or unnecessarily large data in a diagnostic payload.
- Do not use a successful echo as evidence that another tool or provider works.

## Verify

Require an exact JSON-level match for the fields under test, then use the real target tool for end-to-end validation.
