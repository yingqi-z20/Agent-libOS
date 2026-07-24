---
name: agent-libos-git-remotes
description: Fetch, pull, or push through existing validated Git remotes. Use when synchronizing refs with a Host-configured remote after inspecting local/remote intent and obtaining a fresh repository state token.
allowed-tools: git_fetch git_pull git_push
---
# Synchronize Git remotes

## Workflow

1. Use Git inspection to list configured remotes and review local status/refs.
2. Fetch first when remote state must be observed independently of integration.
3. Pull with fast-forward-only by default; choose merge or rebase only when the integration policy is explicit.
4. Push explicit local and remote refs. Use guarded force only with the exact expected remote lease OID.

## Boundaries and safety

- Tools accept registered remote names, never ad hoc URLs, credentials, arbitrary refspecs, or transport commands.
- Remote operations are external effects governed by capabilities, provider policy, approval, audit, and CAS state.
- Remote deletion and lease-based force updates are destructive; never infer consent from ordinary push authority.

## Verify

Re-read refs/status after success and, where supported, confirm the authoritative remote ref equals the intended OID.
