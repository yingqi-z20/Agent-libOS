---
name: agent-libos-workspace-editing
description: Create, overwrite, and delete workspace text files or directories. Use for deliberate filesystem mutations after the target, desired content, authority, and verification method are known.
allowed-tools: write_text_file write_directory delete_file delete_directory
---
# Edit the workspace

## Workflow

1. Inspect the target and current working directory with the navigation Skill before changing content.
2. If the goal explicitly requests baseline or reproduction evidence, stop before writing: activate the command-execution Skill and run the documented command first. A later passing run cannot replace this baseline.
3. Create parent directories explicitly when needed, then make the smallest targeted write.
4. Use non-overwrite or non-recursive modes when replacement or recursive deletion is not required.
5. Re-read the changed file or containing directory to verify the intended state.

## Boundaries and safety

- Paths are relative to the process working directory and must remain inside the runtime workspace.
- Writes and deletes are governed side effects and may require human confirmation even when tools are visible.
- Never recursively delete a broad, unresolved, environment-derived, or guessed path.
- A write invalidates earlier tests, diffs, checkpoints, and final reports. Do not mutate after final human output; if a correction is necessary, reverify and send a corrected report.

## Verify

Check the resolved path, created/deleted flag, byte count, and resulting content or directory entry before proceeding.
