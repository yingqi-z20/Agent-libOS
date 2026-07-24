---
name: agent-libos-command-execution
description: Execute an allowlisted host command through the governed libOS shell boundary. Use for builds, tests, searches, or utilities that require a process and cannot be completed by a narrower first-class tool.
allowed-tools: run_shell_command
---
# Execute commands

## Workflow

1. Prefer a first-class filesystem, Git, or provider tool when one exactly models the operation.
   Never pass a `git` argv to this tool, including read-only `git status` or `git diff`; activate the matching `agent-libos-git-*` Skill and use its typed tool.
2. If the goal requests reproduction or baseline evidence, run its documented command before the first edit. Never reintroduce a known defect merely to manufacture missed baseline evidence; report the gap instead.
3. Pass an explicit argv array; each argument is one item and no shell string expansion occurs.
4. Set bounded timeout/output limits. Interpret return code with stderr and truncation; rerun narrowly if required output was truncated.

## Boundaries and safety

- Commands run from the process working directory under Host allow/ask policy, Capability checks, approval, audit, and resource limits.
- Do not smuggle pipelines, redirection, credentials, ad hoc network endpoints, or destructive broad paths into argv.
- A zero return code is necessary but may not be sufficient; use domain-specific verification.

## Verify

Confirm the invoked argv, return code, non-truncated relevant output, and expected workspace state. Any later workspace write makes prior test evidence stale; rerun after the last edit.
