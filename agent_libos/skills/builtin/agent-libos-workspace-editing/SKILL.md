---
name: agent-libos-workspace-editing
description: Write, create, replace, or delete ordinary workspace text files and directories with explicit mutation and verification semantics. Use after resolving cwd, target, and baseline; not for binary/Object transfer, Git metadata, range edits, or shell commands.
allowed-tools: write_text_file write_directory delete_file delete_directory
---
# Edit the workspace

Generic filesystem mutations are whole-target operations. Establish the current cwd and target with workspace navigation before the first mutation, and verify resulting state afterward with a fresh read. Tool inputs are cwd-relative; returned paths are workspace-root-relative identities.

## Tool guide

### `write_text_file`

- Input: cwd-relative `path`, complete string `content`, optional `encoding`, and `overwrite` (default `true`).
- It writes the entire encoded file, not a patch or range. Missing parent directories are created automatically. With `overwrite=false`, an existing file fails instead of being replaced; use that setting for creation when unexpected existence matters.
- Output: root-relative `path`, encoded `bytes_written`, and `created`. `created=false` means an existing file was replaced; it does not mean no change. The result does not echo content or encoding, so verify both separately.

### `write_directory`

- Input: cwd-relative `path`, `parents` (default `true`), and `exist_ok` (default `true`).
- Output: root-relative `path` and `created`. `created=false` with success means the directory already existed and was accepted.
- Set both flags explicitly. Use `parents=false` when missing ancestors should reveal a mistaken path, and `exist_ok=false` when preexistence must fail. Although `write_text_file` can create parents, use this tool first when parent-chain creation must be reviewed independently.

### `delete_file`

- Input: cwd-relative `path` and `missing_ok` (default `false`).
- Output: root-relative `path`, `kind`, `deleted`, and `recursive=false`.
- With `missing_ok=true`, success with `kind=missing` and `deleted=false` means it was already absent, not deleted by this call.

### `delete_directory`

- Input: cwd-relative `path`, `recursive` (default `false`), and `missing_ok` (default `false`).
- Output: root-relative `path`, `kind`, `deleted`, and the effective `recursive` flag.
- Non-recursive deletion fails on a non-empty directory. Recursive deletion is broad and can partially mutate before a late protected entry or provider failure is detected; it is never evidence of atomic rollback.

## Recommended workflow

1. Activate workspace navigation. Record cwd, root-relative target identity, current kind, relevant content/tree, and the user-approved scope. Preserve requested pre-change evidence before writing.
2. For a new file, prefer `write_text_file` with `overwrite=false`. For replacement, compare the complete intended content against the current file, then call it with the intended encoding and explicit overwrite choice. Do not pass a diff fragment as `content`.
3. For directories, call `write_directory` with explicit `parents` and `exist_ok`. Avoid silently accepting an unexpected existing path.
4. Before `delete_file` or `delete_directory`, re-observe the exact target immediately before deletion. Keep `missing_ok=false` unless idempotent cleanup explicitly accepts prior absence. Keep `recursive=false` unless the complete subtree and deletion scope were reviewed.
5. After the last mutation, use a separately activated read tool: re-read written text with the same encoding, list a created/deleted directory's parent, and verify type and canonical path. In a Git worktree, use Git status/diff through the Git Skills to detect unintended tracked changes.

## Failure and recovery

- File/directory writes require filesystem `write`; deletion separately requires filesystem `delete`. Tool visibility, write authority, or a checkpoint does not imply delete authority. Exact approval may be required.
- Inputs resolve from the current process cwd. If cwd changes, recompute every input; do not feed a root-relative result back unchanged from a non-root cwd.
- Workspace containment and generic `.git` protection are enforced. A recursive delete is also rejected when protected Git metadata appears below the target. Do not bypass this with a broader ancestor, shell command, symlink, or alternate path; use the owning Git primitive.
- `delete_directory` can never delete the workspace root, including with `recursive=true`. The default local provider refuses a symlink/junction as the requested target or traversal route and never follows descendant links during recursive deletion; a recursive delete does remove descendant symlink/junction directory entries themselves. It also refuses read, overwrite, or file deletion of a regular file with multiple hard-link names. These are hard stops, not prompts to broaden the target or retry through Shell.
- A failed side-effecting call is not proof that nothing changed. Local filesystem mutations are not transactionally rolled back, and recursive deletion can remove earlier children before failing on a later child. On timeout, provider error, unknown settlement, or post-effect classification failure, inspect the exact target and parent before deciding whether any retry is safe.
- The complete content is subject to the global tool-argument size limit before dispatch. A preflight size rejection performs no write; use the appropriate Object/file-transfer boundary rather than silently splitting one intended whole-file replacement.
- On exists/not-found/wrong-kind errors, observe state and correct the plan. Never recover automatically by enabling overwrite, `parents`, recursion, or `missing_ok`; each changes the accepted mutation scope.
- `bytes_written` proves the accepted encoded byte count, not that another actor did not modify the file afterward. An earlier read, listing, diff, or checkpoint becomes stale after any later mutation.

## Completion evidence

A completed edit requires fresh evidence after the final mutation:

- intended cwd-relative input and returned root-relative target identity agree;
- `created`, `deleted`, `kind`, `recursive`, and `bytes_written` have the expected meanings for that operation;
- a written file re-reads completely with the intended encoding and exact content, not merely the expected byte count;
- a directory creation or deletion is confirmed from its parent listing, with no relevant truncation;
- Git status/diff or another scope-appropriate check shows no unrelated changes when repository state matters;
- tests or validations affected by the change ran after the final write.

If settlement remains unknown or verification disagrees, report the observed state and stop. Do not claim rollback, successful deletion, or exact content from a success flag alone.
