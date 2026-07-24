---
name: agent-libos-human-collaboration
description: Ask the human operator a blocking question or deliver a user-facing message. Use when progress requires a genuine intent decision or when an interactive image requires an explicit final status/result.
allowed-tools: ask_human human_output
---
# Collaborate with the human

## Workflow

1. Use `ask_human` only when a material choice cannot be derived safely from available state.
2. Ask one concise question and include only context needed to decide; the runtime suspends and resumes the same call.
3. Use `human_output` for informational delivery, including the final result when required by the image.

## Boundaries and safety

- Use `request_permission`, not `ask_human`, when the desired outcome is Capability policy or approval.
- `human_output` does not request an answer and must not be treated as permission.
- Avoid duplicate questions after a wait; the runtime owns request correlation and resume validation.

## Verify

For a question, act on the returned answer. For output, confirm delivery before exiting the process when the image requires it.
