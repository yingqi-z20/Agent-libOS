---
name: agent-libos-authority-basics
description: Inspect current Capability authority and request a human permission decision. Use when an operation is denied, approval requirements are unclear, or exact resource rights must be established before work continues.
allowed-tools: list_capabilities inspect_capability request_permission
---
# Work with authority

## Workflow

1. List active capabilities before assuming authority is missing; include inactive entries only when diagnosing expiry or revocation.
2. Inspect the exact capability when its constraints, lifetime, uses, or effect affect the next action.
3. If authority is absent, request the narrowest resource and rights needed, with a concrete reason.
4. Inspect the returned status/policy. Continue only when it permits the exact operation; stop and report a rejected or deny decision. An ask-per-use policy may still require approval on the original call.

## Boundaries and safety

- Capability inspection never grants authority. Tool visibility is also not proof of authority.
- Use `request_permission` for an authority-policy decision; use `ask_human` for ordinary product or intent questions.
- Never broaden a resource, request admin, or request more rights merely for convenience.

## Verify

If an allow capability was created, inspect its resource, rights, effect, constraints, and remaining uses. Otherwise preserve the denial and do not retry as if authority had been granted.
