---
name: agent-libos-workspace-navigation
description: Inspect and navigate workspace text files and directories. Use when locating code or configuration, reading bounded text, or changing the process working directory without modifying workspace content.
allowed-tools: get_working_directory set_working_directory read_directory read_text_file
---
# Navigate the workspace

## Workflow

1. Read the process working directory before constructing relative paths when location is uncertain.
2. Change it only to a verified workspace directory that should become the base for filesystem and shell calls.
3. List the narrowest directory needed, then read specific text files with an appropriate byte bound.
4. If a read is truncated, narrow the target or use another bounded inspection method; do not assume missing content.

## Boundaries and safety

- Paths are relative to the current process directory; do not prepend that directory or the runtime workspace root.
- Listing and text reads remain subject to path containment and exact filesystem-read Capability checks.
- Use Object-file transfer when bytes must move into durable Object Memory without being returned to the model.

## Verify

Confirm returned resolved paths and truncation flags before drawing conclusions or activating an editing Skill.
