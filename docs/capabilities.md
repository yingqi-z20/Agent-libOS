# Capabilities

Agent libOS uses capabilities as the runtime authority subsystem. A visible
tool, activated Skill, JIT tool, child-process handle, object name, path string,
image id, checkpoint id, or JSON-RPC endpoint id is not enough to perform a
protected operation. Protected effects are authorized at primitive use by
process identity, typed resource pattern, right, effect, constraints, human
approval, and audit. This lets self-evolving agents change their action surface
without implicitly changing what resources they can affect.

## Capability Record

Capability records are structured authority statements:

- `subject`: process or runtime actor that holds the authority.
- `resource`: canonical typed resource pattern.
- `rights`: operation rights such as `read`, `write`, or `execute`.
- `effect`: `allow`, `deny`, or `ask`.
- `issuer`: actor that issued the record.
- `issuer_cap_id` and `parent_cap_id`: lineage for grant/delegation decisions.
- `delegation_depth`: attenuation depth from a parent capability.
- `issued_at`, `expires_at`, `uses_remaining`, and `status`.
- `rules`, `lease`, `delegation`, `constraints`, and `metadata`.

One-shot authority is not encoded as a policy string. It is an `allow`
capability with `uses_remaining=1`; successful primitive use consumes it and
revokes the capability when the count reaches zero. If one-shot Object Memory
authority is resolved through a namespace/name lookup, any handle minted from
that lookup remains one-shot; name lookup cannot turn temporary authority into
a persistent object handle.

`deny` records dominate matching allows. To create an exception, revoke the
broad deny and issue narrower allow/deny records explicitly. The runtime does
not implement hidden override precedence that could accidentally reopen a
blocked resource.

## Rights

The common rights are:

- `read`: inspect or materialize a resource.
- `write`: create or modify a resource.
- `delete`: remove a resource.
- `execute`: run or load a resource.
- `materialize`: include object content in a prompt or tool result.
- `link`: create Object Memory links.
- `diff`: compare object or checkpoint state.
- `grant`: issue authority over a covered resource.
- `revoke`: revoke authority over a covered resource.
- `approve`: approve a human/request resource.
- `admin`: perform destructive or policy-changing operations.

The exact right set depends on the primitive. Unknown or unsupported rights do
not create primitive behavior by themselves. Capability records reject unknown
rights, including `*`; use explicit rights instead of all-rights wildcards.

## Resource Matching

Resources are typed and canonicalized before authorization. Matching is not a
raw `startswith` or suffix check.

Important resource conventions include:

- `filesystem:workspace:<path>` for exact workspace files and directories.
- `filesystem:workspace:<dir>/*` for a directory subtree.
- `filesystem:workspace:*` for the whole workspace namespace.
- `shell:<executable>` for direct command authority; `shell:*` for shell
  policy records when paired with `shell_policy_level`.
- `human:<name>` for human output, questions, and approvals.
- `object:<oid>` for Object Memory content.
- `object_namespace:<namespace>` for Object Memory namespace listing and name
  lookup.
- `process:<pid>` and `process:*` for process operations.
- `message:<pid>` and `message:*` for message queue operations.
- `image:<image_id>` and `image:*` for image registration.
- `skill:<skill_id>` and `skill:*` for Skill operations.
- `skill_source:workspace:<relative_path>` and
  `skill_source:global:<source_id>` for Skill source authority.
- `skill_trust:*` or `skill_trust:<sha256>` for global Skill trust.
- `checkpoint:process:<pid>`, `checkpoint:<checkpoint_id>`, and
  `checkpoint:*` for checkpoint operations.
- `jsonrpc_endpoint:<endpoint_id>` and `jsonrpc_endpoint:*` for JSON-RPC
  endpoint registry metadata.
- `jsonrpc:<endpoint_id>:<method_id>`, `jsonrpc:<endpoint_id>:*`, and
  `jsonrpc:*` for JSON-RPC method invocation.

Wildcard syntax is terminal only. `kind:*` is a typed prefix pattern and
`kind:body/*` is a subtree pattern. Bare global `*` is rejected; authority must
stay inside an explicit resource kind. Prefix collisions are rejected: a grant
for `filesystem:workspace:src/*` covers `src/main.py`, not `src2/main.py`.

Requested resources are produced by primitives after their own normalization.
For example, the filesystem primitive resolves cwd-relative paths, enforces
workspace containment, then asks for the canonical logical resource. Shell
authority uses normalized executable identity and argv token policy, not a shell
string.

## Authorization API

The manager entry point is:

```python
authorize(subject, resource, right, context) -> CapabilityDecision
```

`require(...)` wraps `authorize(...)` and raises on denial. Primitive code should
pass operation context such as path, argv, byte counts, hashes, risk labels,
process lineage, or provider details. The decision records:

- matched capability ids,
- selected capability id,
- issuer chain,
- effect and derived human-facing policy,
- constraint evaluation results,
- one-shot consumption id when applicable,
- human approval request id when a primitive asks the human,
- operation context preview.

Unknown constraint keys fail closed. Primitive-specific evaluators can define
new constraint keys, but the default manager does not silently ignore unknown
policy language.

## Authority Rules And Profiles

Capabilities can carry deterministic `AuthorityRule` entries. A rule has:

- `operation`, such as `filesystem.read`, `shell.run`, `jsonrpc.call`, or
  `deno.syscall`;
- `effect`: `allow`, `ask`, or `deny`;
- `risk`: `harmless`, `low`, `medium`, `high`, or `destructive`;
- structured `conditions`, such as argv tokens, match mode, path/cwd intent,
  network intent, and filesystem intent.

Rules are not LLM judgments. They are local, deterministic policy facts. Unknown
rule shapes or unknown constraint keys fail closed. The primitive converts the
final capability decision into a sandbox profile and records that profile in
approval context, audit, and external-effect metadata where applicable.

## Issue, Delegate, Revoke

All authority mutation goes through explicit operations:

- `issue(actor, subject, spec)`: `actor` must be trusted, hold covering
  `admin` authority, or hold both covering `grant` authority and covering
  `allow` capabilities for every right being transferred. `grant` is not a
  capability-minting right: it can only transfer rights the actor already has,
  cannot create `deny`/`ask` policy records, and cannot transfer finite-use
  capabilities onward.
- `delegate(parent, child, spec)`: `parent` must hold a covering delegable
  `allow` capability. Delegation can only attenuate resource, rights, expiry,
  constraints, and delegation depth. Finite-use capabilities are consumed by
  direct use and cannot be delegated. Delegated records keep a parent link, so a
  later parent revocation or expiry stops the child record from authorizing.
  Child records cannot drop parent constraints such as `shell_policy_level`.
- `revoke(actor, cap_id)`: allowed for trusted issuers, the original issuer,
  the holder relinquishing its own capability, or an actor with covering
  `revoke`/`admin` authority.

Runtime bootstrap, image bootstrap, human approval, admin CLI, checkpoint
restore/fork, and tests use explicit trusted issuer paths and emit audit
records. Trusted issuers are exact configured actor names, not broad prefixes.
Ordinary AgentProcess, Skill, and JIT tool execution has no implicit signing
authority.

Fork and spawn inherit authority only through delegation/attenuation. Exec
switches the image/tool table and may shrink capabilities, but it never grants
the target image's declared `required_capabilities`.

Image-package boot is also not an external authority grant. Its
`workspace.grants` entries apply only to the package workspace seed after it is
materialized into that process's private directory under
`agent_outputs/image_workspaces/`; they cannot name arbitrary host or workspace
paths.

## Permission Policy And Human Approval

Human-facing policy names are still used at prompts and CLI boundaries:

- `always_allow` maps to `effect=allow`.
- `always_deny` maps to `effect=deny`.
- `ask_each_time` maps to `effect=ask`.
- `allow_once` maps to `effect=allow, uses_remaining=1`.

Model-facing `request_permission` is not a raw grant API. It first requires the
caller to hold `human:<name>` write authority, then creates a blocking human
request. Before a request enters the human queue, the runtime canonicalizes the
resource, normalizes rights, classifies risk, records resource scope, attaches
any deterministic constraints, and shows the selected lease shape. Ordinary
model requests cannot ask for broad high-risk authority such as
`capability:*` with privileged rights (`admin`, `grant`, `revoke`, `write`,
`execute`, or `delete`), `shell:*` execute, or root/global filesystem write such
as `filesystem:/:*` or `filesystem:*`. Workspace-level write
(`filesystem:workspace:*`) can be approved by the human. Admin CLI and
bootstrap paths can still issue broader policy explicitly, with audit.

When `ask_each_time` applies, the primitive creates a human approval request and
waits inside the operation. The caller eventually receives either the final
payload or a final denial error; there is no exposed pending/retry syscall
protocol.

Approval context includes path, resource, overwrite risk, byte count, SHA-256,
target state, argv, risk, rule id, sandbox profile, and escaped previews when
available.

## Tool, Skill, And JIT Boundary

The process tool table controls LLM-facing tool visibility. Capabilities
control primitive effects.

`ToolPolicy` is declaration metadata only. Fields such as
`declared_permissions` and `declared_confirmation_required` can help a GUI or a
human reviewer understand a tool, but the broker does not convert them into
grants or confirmations. Real authorization still happens in the primitive that
touches the resource.

For example, a process can see `write_text_file` and still fail to write
`src/app.py` if it lacks write authority for
`filesystem:workspace:src/app.py` or a covering subtree grant.

Loading a Skill can add instructions, existing tools, and Deno/TypeScript JIT
tools to one process table. It does not grant filesystem, shell, human, object,
process, image, checkpoint, JSON-RPC, or Skill source authority. JIT syscalls
bypass the LLM-facing tool table, but they still enter the same primitive
authorization path as built-in tools.

Default images expose `list_capabilities` and `inspect_capability` so a process
can understand its own authority. `delegate_capability` and `revoke_capability`
are registered static tools but are not included in default image tool tables.

## CLI And Syscalls

The CLI supports:

```bash
uv run agent-libos capabilities list [--subject <pid>] [--include-inactive]
uv run agent-libos capabilities inspect <capability_id>
uv run agent-libos capabilities grant <subject> <resource> --rights read write
uv run agent-libos capabilities delegate <parent> <child> <resource> --rights read
uv run agent-libos capabilities revoke <capability_id> [--reason "..."]
uv run agent-libos capabilities explain <subject> <resource> <right>
```

Without `--actor-pid`, CLI commands run as audited admin operations. With
`--actor-pid`, the command is executed as that process and strict capability
checks apply.

Deno/TypeScript JIT tools can use the syscall names:

- `capability.list`
- `capability.inspect`
- `capability.request_permission`
- `capability.delegate`
- `capability.revoke`

Syscalls do not consult the process tool table. They are authorized by pid,
capability records, primitive rules, human approval, and audit.

## JSON-RPC Authority

Remote JSON-RPC calls are pre-registered endpoint resources. Agents cannot pass
URLs or secrets at call time.

Registry inspection and mutation use endpoint resources:

```text
jsonrpc_endpoint:demo-weather
jsonrpc_endpoint:*
```

Method calls use method resources:

```text
jsonrpc:demo-weather:forecast
jsonrpc:demo-weather:*
jsonrpc:*
```

The required right comes from the endpoint method spec: `read`, `write`, or
`execute`. Granting `jsonrpc_endpoint:* read` allows endpoint discovery, not
method invocation. Granting `jsonrpc:demo-weather:forecast read` allows that
specific remote method, subject to primitive validation, human approval,
runtime DNS policy, provider classification, audit, and external-effect
recording. Agent-facing inspect paths do not expose endpoint URLs or header
prefix/suffix values; those are host registry details.

## Filesystem Authority

Filesystem capabilities can target exact files, directory subtrees, or the
whole workspace. Relative paths resolve from the caller process working
directory. The filesystem primitive enforces workspace containment before host
provider calls. Path strings that escape the workspace are rejected even if a
model can produce them.

Read, write, and delete are separate rights. Granting read over a directory does
not grant write or delete.

The default local filesystem provider performs no-follow traversal inside the
workspace root for existing files. Existing file read, write, and delete reject
symlink or junction traversal and reject regular files with multiple hard links
(`st_nlink > 1`). Directory listings report child symlinks as symlink entries
without following them.

## Shell Authority

Shell execution is argv-only. The model-facing tool and syscall accept token
arrays, not shell command strings. Pipes, redirects, wildcard expansion, and
command chaining must be requested explicitly through an interpreter executable
such as `bash`, `sh`, `cmd`, or `powershell`, where policy matching can inspect
the interpreter token.

Arguments beginning with a `file:` URL are rejected before the provider runs.
For bare executables, the local provider resolves argv[0] on a safe host PATH
and refuses a resolution inside the workspace or selected process cwd. Shell
subprocesses receive a constrained environment with `HOME` and `USERPROFILE`
pointing at the workspace root instead of the host user's real home.

Shell command risk is classified by argv-token rules before the provider runs:

- `harmless`: read-only status/version/inspection commands such as
  `git status --short` or `python --version`.
- `low`: read-only project inspection such as `git diff`.
- `medium`: project code execution such as `pytest`, pytest collection,
  `npm test`, or `uv run ...`.
- `high`: package managers, network-capable tools, script interpreters,
  `python -m compileall`, service startup, and other commands likely to change
  host state or cross a boundary.
- `destructive`: delete/move/permission/system-control operations. These are
  denied by the built-in rule set even under broad shell policy.

The built-in shell policy levels then decide how to handle the classified rule:

- `always_deny`: reject every shell command.
- `allowlist_auto_else_ask`: allow `allow` rules and ask for `ask` rules.
- `blocklist_ask_else_auto`: use the same deterministic risk rules but is
  intended for broader local operation.
- `always_allow`: allow non-destructive commands while still reporting risk.

Rules match tokenized argv, not arbitrary substrings. Bare executable names do
not match path-qualified executables by accident. The local provider executes
the argv vector directly with `shell=False`; the primitive never rebuilds a shell
command string from model input. As defense in depth, non-`always_allow`
automatic allow rules downgrade to human approval when argv tokens contain shell
metasyntax such as command substitution, backticks, separators, newlines,
redirection/process substitution, brace expansion, or comments.
Direct command capabilities such as `shell:git` or `shell:git:*` can allow a
specific normalized executable, but a bare direct command grant is intentionally
limited: without `authority_rules`, it only covers argv that the deterministic
classifier already marks `allow`. Medium/high shell side effects need explicit
rules, per-use human approval, or a human/admin-issued shell policy. A bare
`shell:* allow` capability is not treated as direct command authority;
whole-shell authority must be represented as a policy capability carrying
`shell_policy_level`, so the primitive can still apply the four-level shell
policy semantics. Broad `deny` and `ask` records remain conservative
constraints.

Scoped denies are supported with `AuthorityRule` constraints. An unconstrained
deny still dominates all matching grants; a constrained deny dominates only when
its rule matches the current operation context, so policy can allow read-only
`git` inspection while denying `git push`.

Block-list checks also scan nested executable-looking argv tokens such as
`bash`, `powershell`, `python`, or `curl`.
