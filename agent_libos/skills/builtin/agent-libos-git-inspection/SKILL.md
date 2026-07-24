---
name: agent-libos-git-inspection
description: Inspect the fixed runtime Git repository, worktrees, status, diffs, history, refs, remotes, and blame. Use to understand repository state and obtain a fresh state token before any Git mutation.
allowed-tools: git_repository_info git_status git_diff git_log git_show git_blame git_list_refs git_list_remotes git_list_worktrees
---
# Inspect Git state

## Workflow

1. Identify the managed worktree and read repository info/status before planning a mutation.
2. After the last workspace edit and test, use the narrowest diff scope and restrict paths. For `worktree`/`staged`, set `base` and `head` to null (never empty strings); for `range`, provide both exact refs/OIDs.
3. Inspect exact validated refs/OIDs for history and commits. Use byte-preserving `path_b64` tokens returned by reads when a path is not safe UTF-8.
4. Carry the newest returned `state_token` only into the immediately considered mutation.

## Boundaries and safety

- Tools operate on the fixed runtime repository; they do not accept arbitrary repositories, argv, patterns, or remote URLs.
- Results are bounded. Treat truncation as incomplete evidence and narrow the query.
- A state token is a compare-and-swap observation, not authority or approval.
- Any later workspace mutation makes prior status/diff evidence stale; inspect again before claims or checkpointing.

## Verify

Before mutation, confirm the selected worktree, literal paths, relevant diff/status, exact ref/OID, and fresh state token all describe the intended state.
