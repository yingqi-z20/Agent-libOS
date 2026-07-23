# mini-swe-agent Image

`images/mini-swe-agent/` is a package-only AgentImage that follows the
mini-swe-agent `mini.yaml` tool-call shape: the model sees a single `bash`
tool with a required `command` string and an optional `submit` boolean.

```bash
uv run agent-libos --db .agent_libos.sqlite images validate images/mini-swe-agent
uv run agent-libos --db .agent_libos.sqlite images register images/mini-swe-agent
```

Package validation and registration can inspect the package without Deno, but
booting this image validates and registers its process-local TypeScript JIT
tool. A usable `deno` executable is therefore required to spawn/boot and run the
image; missing Deno makes boot fail validation rather than silently omitting the
`bash` tool.

The CLI loads the project-root `config.yaml` when it is present. In this
checkout that configuration selects `.agent_libos.sqlite`, so omitting `--db`
persists the registration there. A checkout without project configuration
falls back to `DEFAULT_CONFIG`, whose `local` store is in memory and therefore
does not survive a later CLI invocation. Use an explicit `--db` when the
artifact must survive later invocations. Relative database paths are still
resolved from the CLI process's current working directory; use an absolute
`--db` path when invocations from different directories must select the same
store independently of local configuration.

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

with a 30 second shell timeout and a 10000 character observation window. The
package declares a 35 second outer JIT sandbox timeout so the shell timeout has
time to return its structured observation. This per-tool timeout is capped by
the Host's `tools.deno_timeout_hard_limit_s` (60 seconds by default); it does
not raise the 5 second default used by other JIT tools.

The package declares required capabilities for workspace filesystem read/write
and shell execute authority. Those declarations are metadata checked by normal
process bootstrap rules; they do not grant live authority by themselves. A host
or benchmark runner must still grant the spawned process the filesystem and
shell authority it should have for the task.

Observations longer than the window return `output_head`, `output_tail`, and
`elided_chars` instead of a full `output` field. Timed-out or permission-denied
commands return a non-zero observation with `exception_info`; the agent prompt
treats an unrecoverable permission, dependency, or timeout condition as a
blocker that can be submitted explicitly.

Known differences from upstream mini-swe-agent remain:

- Agent libOS supplies the task through its existing process goal/Object Memory
  context, not mini-swe-agent's exact `instance_template` user message.
- The local shell is mediated by Agent libOS providers, policies, resource
  budgets, cwd checks, and environment allowlists.
- The package targets the OpenAI tool-call `mini.yaml` interface, not the
  `mswea_bash_command` fenced-code-block format from `default.yaml`.
