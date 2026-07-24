---
name: agent-libos-git-pull-requests
description: Create, list, inspect, review, merge, or close repository-local simulated pull requests. Use for the built-in evidence-preserving PR workflow, not for GitHub or another external forge.
allowed-tools: git_create_pull_request git_list_pull_requests git_inspect_pull_request git_review_pull_request git_merge_pull_request git_close_pull_request
---
# Work with simulated pull requests

## Workflow

1. Inspect exact base/head refs and repository state, then create a PR with a fresh state token.
2. List to find candidates and inspect one PR to verify snapshot refs, reviews, and current state.
3. Record a precise comment, approval, or requested-changes decision after reviewing its immutable evidence.
4. Merge with the intended strategy or close without deleting evidence; use the freshest state token.

## Boundaries and safety

- This is the repository-local simulated PR system. It does not contact GitHub, GitLab, or another forge.
- Approval records do not override Capability, merge strategy, CAS, or destructive-operation controls.
- On stale state or changed snapshots, re-inspect rather than replaying a review or merge.

## Verify

Inspect the PR after mutation and verify its status, reviews, snapshot refs, resulting commit/ref, and retained evidence.
