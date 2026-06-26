# mini-swe-agent Image

`images/mini-swe-agent/` is a package-only AgentImage that follows the
mini-swe-agent `mini.yaml` tool-call shape: the model sees a single `bash`
tool with a required `command` string and an optional `submit` boolean.

```bash
uv run agent-libos images validate images/mini-swe-agent
uv run agent-libos images register images/mini-swe-agent
```

The package uses `prompt_mode: image_only`, `jit_tool_exposure: direct`, and
`default_tools: []`. At boot, the image package registers one process-local JIT
tool named `bash`; it does not expose `process_exit`, Object Memory, or other
builtin tools to the model. If `submit` is `true` and the shell command exits
successfully, the JIT wrapper calls the internal `process.exit` syscall with
the command output as the submitted payload after collecting the tool result.

The wrapper runs:

```text
bash -lc "exec 2>&1; <command>"
```

with a 30 second timeout and a 10000 character observation window.

Known differences from upstream mini-swe-agent remain:

- Agent libOS supplies the task through its existing process goal/Object Memory
  context, not mini-swe-agent's exact `instance_template` user message.
- The local shell is mediated by Agent libOS providers, policies, resource
  budgets, cwd checks, and environment allowlists.
- The package targets the OpenAI tool-call `mini.yaml` interface, not the
  `mswea_bash_command` fenced-code-block format from `default.yaml`.
