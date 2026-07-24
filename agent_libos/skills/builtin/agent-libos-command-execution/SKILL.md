---
name: agent-libos-command-execution
description: Run one policy-governed, non-interactive host command through the argv-only Shell boundary. Use for builds, tests, searches, or utilities lacking a narrower tool; not for general Git, TTY/streaming work, ad hoc remotes, or attempts to evade policy.
allowed-tools: run_shell_command
---
# Execute a command

This is a governed native-process boundary, not a command-string convenience API and not an operating-system sandbox. Prefer a narrower typed tool whenever one models the intended operation.

## Tool guide

### `run_shell_command`

- Input `argv`: a non-empty array of strings. `argv[0]` must be nonblank and no token may contain NUL. The provider executes an argv vector with `shell=false`; the primitive does not rebuild a shell command string.
- Input `timeout_s`: a finite number greater than zero and no greater than the configured hard limit. The normal default is runtime-configured.
- Inputs `max_stdout_chars` and `max_stderr_chars`: nonnegative character limits within their schema hard bounds. They can further reduce returned output, but cannot recover text already capped by the primitive or local provider.
- There is no call-level cwd, environment map, stdin, TTY, streaming, or interactive prompt. Select cwd with workspace navigation before the call. The subprocess receives a constrained Host-owned environment and safe `PATH`.
- Output contains the logical `argv`, integer `returncode`, decoded UTF-8 `stdout` and `stderr`, and independent truncation flags. A completed subprocess with nonzero `returncode` is still a successful tool dispatch; decide command success from `returncode` and domain evidence, not only the outer tool `ok`.

Output has multiple bounds. The default local provider monitors byte-backed hard capture limits and decodes UTF-8 with replacement for invalid sequences. If a stream exceeds a provider hard limit, the provider normally kills the process and raises a resource-limit failure; no successful tool output or truncation flag is then available, and effects remain unknown. Only after a provider returns successfully can its own truncation flag, the Shell primitive's configured character cap, and finally the call's requested character cap describe retained prefixes. Output is not byte-preserving, and raising a call limit cannot restore omitted text or turn a provider-limit failure into a successful truncated result.

## Recommended workflow

1. Prefer the owning filesystem, Git, Object, provider, test, or workflow tool. Use `run_shell_command` only when its exact executable and argv are necessary.
2. Resolve the intended process cwd first. Pass each argument as a separate token exactly as the executable expects. Without an explicitly invoked interpreter, pipes, redirection, wildcard expansion, quoting, `$()`, backticks, separators, and environment expansion are literal tokens, not shell syntax.
3. An explicit interpreter such as `bash`, `sh`, or Python can interpret a script argument, but that is a high-risk, policy-visible native execution choice—not a workaround. Use it only when the task truly requires it and approval/policy permits the exact argv. Keep script text bounded and auditable.
4. Choose a realistic timeout and small output caps; prefer native filters, selectors, or quiet mode over flooding model context. Create an output file only when that exact artifact and command-side mutation are part of the user-authorized goal. A child process writes it directly, outside the typed filesystem Capability, approval, and file-lineage boundary, so verify the artifact afterward with workspace or Git tools and never use this as a sensitive-data workaround.
5. After the call, inspect `returncode`, both streams, and both truncation flags. Parse only complete records. If a decisive summary might be beyond a truncated prefix, rerun a narrower read-only query or inspect an authorized output artifact.
6. Verify effects through the owning domain after the final command. A zero exit code says only that the program reported success.

For ordinary Git intent, activate an `agent-libos-git-*` Skill. Direct Git is a legacy exception only for a user-explicit exact read at the runtime workspace root: `git status`, `git status --short`, `git branch --show-current`, `git rev-parse --show-toplevel`, `git diff`, or `git diff --stat`. Extra flags, paths, mutations, remotes, executable paths, and wrapped Git argv are rejected. Do not translate broader Git intent into Shell.

## Failure and recovery

- Shell policy is not merely an allowlist. Host configuration classifies argv as deny, ask, or allow; ask approval is bound to exact argv, cwd, and timeout. Capability or approval cannot override a Host deny, raw-Git guard, containment, budget, or argv policy.
- Before human approval or provider dispatch, data-flow policy also authorizes the executable sink against the command argv and source-Object labels. Shell authority can therefore still be denied for sensitivity or sink-trust reasons. Do not remove lineage, encode the payload, or wrap the executable to evade that decision; use a Host-configured trusted sink or stop.
- Before policy evaluation, cwd and path-like tokens must stay inside the workspace; `file:` URLs and escape syntax are rejected. Other URL-like arguments are not filesystem paths, so authorization of a network-capable program still carries its native network risk.
- An authorized subprocess runs as the host user. These controls are not an OS filesystem/network sandbox and do not mediate a child's direct I/O, even when typed libOS primitives would deny it. Use only trusted executables; hostile-code isolation requires a stronger boundary.
- On timeout, resource-limit, cancellation, or ordinary provider failure after dispatch, output may be unavailable and external effect is conservatively unknown. Finite authority is consumed. Inspect state with a domain read before any retry; replay only a proven-idempotent read whose first dispatch outcome is harmless.
- Even when stream caps pass, the normalized result has a global persistence bound. A side-effecting Shell result can be replaced by `result_omitted` evidence after execution; treat command outcome/effects as unverified and inspect state instead of replaying.
- On truncation, narrowing the command is preferred. Increasing the call cap helps only when that final tool-level cap caused truncation; it cannot undo provider byte limits or configured primitive character limits.
- A denial is a stop condition. Do not wrap, split, rename, path-qualify, or move the command into an interpreter to evade classification.

## Completion evidence

Record the exact requested argv, intended cwd, timeout, return code, and both truncation flags. Completion additionally requires:

- decisive output is complete, or conclusions are explicitly limited to the returned prefix;
- invalid UTF-8 replacement or byte truncation cannot affect the claimed result;
- the authoritative domain state, artifact, or test result was checked after the command;
- no interactive prompt, hidden environment dependency, or unreviewed remote access was required;
- an unknown-effect failure was reconciled rather than blindly retried.

Stop when policy denies the action, required evidence cannot fit safely, or native execution would exceed the requested trust boundary.
