# mini-swe-agent Image

## Package overview

`images/mini-swe-agent/` is a package-only AgentImage inspired by the single
`bash` tool-call shape historically associated with mini-swe-agent's
`mini.yaml`. The model sees one `bash` tool with a required `command` string and
an optional `submit` boolean. This package does not claim drop-in or versioned
compatibility with an unpinned upstream revision.

## Validate and register

```bash
uv run agent-libos --db user images validate images/mini-swe-agent
uv run agent-libos --db user images register images/mini-swe-agent
```

Package validation and registration can inspect the package without Deno, but
booting this image validates and registers its process-local TypeScript JIT
tool. A usable Host-configured `deno` executable is therefore required to
spawn/boot the image; missing Deno makes boot fail validation rather than
silently omitting the `bash` tool. Executing the tool separately requires the
Host Shell provider to resolve and permit a usable `bash` executable because
the mediated command is `bash -lc ...`; Deno itself does not supply Bash.

## Prompt mode and transcript persistence

This image uses `prompt_mode: image_only`. Every LLM turn therefore requires
`llm.persist_full_io: true`, the default, so the Runtime can retain and validate
the lossless native assistant/tool transcript. With
`llm.persist_full_io: false`, boot may register the image, but execution fails
closed before LLM provider dispatch rather than falling back to a lossy prompt.

## CLI store selection

The CLI loads the project-root `config.yaml` when it is present. In this
checkout that configuration selects `user`, so omitting `--db` persists the
registration in `~/.agent-libos/runtime/agent-libos.sqlite`. A checkout without
project configuration falls back to the same `user` target in `DEFAULT_CONFIG`.
Use explicit `--db local` only when state should disappear on close. Relative
custom database paths are resolved from the CLI process's current working
directory and are rejected inside the effective workspace; use `user` or an
absolute external path when invocations must select the same store independently
of local configuration.

## Tool surface

The package uses `prompt_mode: image_only`, `jit_tool_exposure: direct`, and
`default_tools: []`. At boot, the image package registers one process-local JIT
tool named `bash`; it does not expose `process_exit`, Object Memory, or other
builtin tools to the model. If `submit` is `true` and the shell command exits
successfully, the JIT wrapper calls the internal `process.exit` syscall with
the same bounded structured observation returned by the tool. It never copies
an oversized raw shell result into the final process payload.

## Shell mediation boundary

The wrapper runs:

```text
bash -lc "exec 2>&1; <command>"
```

That call crosses the Runtime boundary once through `shell.run`. Before launch,
Shell checks `shell:<command>` execute authority, command policy,
data-flow, cwd/environment constraints, and resource budgets. Once `bash` (or a
child it starts) is running, its direct filesystem, network, child-process, and
other I/O does not re-enter typed filesystem or network Capability checks. The
Shell provider is not an operating-system sandbox, and the filesystem entry in
`required_capabilities` is only a declaration/diagnostic—it does not confine
the Bash subprocess. A Host executing untrusted code must put the Shell
provider or workspace behind an explicit container, WASM, VM, service-provider,
or comparable filesystem/network isolation boundary.

## Bounds and contract tests

The wrapper uses a 30 second shell timeout, a 32,768-Unicode-code-point command limit, and a
10,000-code-point observation window. The JSON Schema validator and TypeScript
wrapper use the same unit, so astral characters such as emoji count once and
head/tail elision never splits a UTF-16 surrogate pair. The package declares a 35 second outer JIT
sandbox timeout so the shell timeout has time to return its structured
observation. This per-tool timeout is capped by the Host's
`tools.deno_timeout_hard_limit_s` (60 seconds by default); it does not raise the
5 second default used by other JIT tools. Package boot also runs six bundled
JIT contract tests covering ordinary success, successful submission, a denied
shell syscall, a failed final submission, and propagation of upstream shell
truncation, plus the rule that a non-zero command cannot submit, before
registering the tool.

## Required capability declarations

The package declares required capabilities for workspace filesystem read/write
and shell execute authority. Bootstrap validates and records those declarations
in the task authority manifest, and exposes any unsatisfied entries through
`missing_required_capabilities` diagnostics and audit. They are declarations,
not launch gates or grants: missing entries do not prevent spawn/boot, and the
declarations do not create live authority. A Host or benchmark runner must still
grant the spawned process the filesystem and shell authority it should have for
the task.

## Observation and submission semantics

Every command observation propagates `stdout_truncated` and
`stderr_truncated` from the shell primitive. `output_incomplete` is true when
either stream was truncated upstream or the wrapper elides captured output.
The shell primitive can truncate a stream before the wrapper receives it.
Observations whose captured output is longer than the wrapper window return
`output_head`, `output_tail`, and `elided_chars` instead of a full `output`
field. `elided_chars` counts only Unicode code points the wrapper removes from the
already captured strings; it cannot quantify bytes omitted earlier by the shell
primitive, so the truncation flags remain authoritative for completeness.
Timed-out or permission-denied commands return a non-zero observation with
`exception_info`. The wrapper calls `process.exit` only after a `submit: true`
command itself returns zero. A permission, dependency, or timeout blocker can
therefore be submitted only if the process can still execute one authorized,
short report command successfully. If no such command can run—for example Bash
or Shell authority is unavailable, or the subprocess budget is exhausted—this
single-tool image cannot self-submit; the Host must externally terminate,
cancel, or judge the process rather than expecting another tool call to exit.
Oversized commands are rejected by both the JSON schema and the JIT wrapper. If
the shell command succeeds but the final process-exit syscall fails, the tool
returns a non-zero observation that retains the same bounded command-output and
truncation evidence and reports the submission failure; it does not claim
completion.

## Compatibility scope

The compatibility scope is intentionally only the local single-tool idea. Key
Agent libOS-specific behavior includes:

- Agent libOS sends the process goal directly as the first user message. A
  string is unchanged and a structured goal is canonical JSON; the Runtime
  does not wrap it in Object Memory or an `instance_template`.
- Later turns use an OpenAI-style native assistant/tool transcript, while its
  durable ledger and all authority enforcement remain Runtime-owned.
- The local shell is mediated by Agent libOS providers, policies, resource
  budgets, cwd checks, and environment allowlists before launch; these controls
  do not mediate the authorized native child's own direct I/O.
- The package uses tool calls rather than a fenced-code-block command protocol.
  Upstream paths, schemas, prompts, and defaults can change independently; use
  this repository's `IMAGE.yaml`, prompt, JIT schema, and tests as the only
  normative contract for this image.
