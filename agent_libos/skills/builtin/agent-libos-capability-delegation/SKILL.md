---
name: agent-libos-capability-delegation
description: Delegate attenuated Capability authority to a direct child and revoke capabilities. Use when coordinating child processes that need explicit bounded access, or when previously issued authority must be withdrawn.
allowed-tools: delegate_capability revoke_capability
---
# Delegate capabilities

## Workflow

1. Create or identify the direct child before delegation.
2. Derive the smallest resource, rights, constraints, expiry, and use count from the child's concrete task.
3. Delegate only authority the parent holds and is permitted to delegate.
4. Revoke no-longer-needed authority and record a concise reason.

## Boundaries and safety

- Delegation works only to a direct child and can attenuate, never amplify, parent authority.
- Prefer spawn/fork inheritance for authority known at child creation; use delegation for later, explicit additions.
- Do not make delegated authority itself delegable unless the child must create a further bounded worker.

## Verify

Inspect the returned capability and confirm the subject, resource, effect, rights, constraints, expiry, uses, and delegability are exact.
