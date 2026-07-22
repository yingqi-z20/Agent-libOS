# Skills

Agent libOS Skills use the standard Agent Skills package shape: a directory
with a required `SKILL.md` file, optional `scripts/`, `references/`, and
`assets/` resources, and progressive disclosure from catalog metadata to full
instructions and bundled resources.

Skills are not a permission mechanism. Activating a Skill changes only one
process's prompt context and tool visibility. They are part of the
self-evolving action surface, not the authority subsystem. Filesystem, shell,
Object Memory, JSON-RPC, process, checkpoint, and human effects still go
through primitives and Capability.

## Package Shape

```text
skills/review-helper/
  SKILL.md
  scripts/
    count_lines.ts
  references/
    agent-libos/
      actions.json
      required-capabilities.json
      jit-tools.json
    workflow.md
  assets/
```

`SKILL.md` must start with YAML frontmatter:

```yaml
---
name: review-helper
description: Focused code-review workflow helpers.
license: Apache-2.0
compatibility: agent-libos>=0.1.0
allowed-tools:
  - read_text_file
  - read_directory
metadata:
  agent-libos.version: v0
  agent-libos.actions: references/agent-libos/actions.json
  agent-libos.required-capabilities: references/agent-libos/required-capabilities.json
  agent-libos.jit-tools: references/agent-libos/jit-tools.json
---

# Review Helper

Prefer small, evidence-backed findings with file and line references.
```

Supported frontmatter fields are `name`, `description`, `license`,
`compatibility`, `allowed-tools`, and `metadata`. The `name` must be lowercase
letters, digits, and hyphens, and must match the package directory name.
`metadata` values must be strings.

Agent libOS reserves the `agent-libos.*` metadata namespace. Complex extension
data lives in `references/agent-libos/*.json`; metadata only points at those
relative files.

Package validation is bounded by `AgentLibOSConfig.skills`: `SKILL.md` is read
with `skill_md_max_bytes` for process-driven workspace registration and the
hard host limit is `skill_md_hard_limit_bytes`; bundled resources are limited
by `resource_read_max_bytes`, `package_max_bytes`, and `max_package_files`.
Every bounded workspace read is fail-closed: if `SKILL.md`, an explicitly
referenced metadata file, a JIT source, or an optionally discovered resource
reports `truncated: true`, registration rejects the package rather than hashing
or parsing the prefix. JIT source files are subject to the resource byte limit
and the decoded source is then subject to `max_jit_source_chars`. Package text
is decoded as UTF-8 from those snapshotted bytes; the general-purpose
`tools.default_text_encoding` setting does not change Skill hashes or parsing.
Packages whose prompt instructions exceed `max_prompt_instruction_chars` are
rejected during parsing/registration; an oversized package is not accepted by
clipping its instructions. Loaded snapshots are validated against the same
bound before use. Skill JIT sources use `max_jit_source_chars`; tool, action,
JIT, and required-capability counts use their corresponding `max_*` settings.
The package SHA-256 binds the normalized Skill metadata, the prompt
instructions, JIT source hashes, declared resource metadata, and the actual
bundled resource bytes. A package snapshot whose stored resource content no
longer matches its declared size/SHA is rejected.

## Progressive Disclosure

Discovery returns catalog fields such as `name`, `description`, source, package
hash, and high-level tool/action names. Activation materializes the full
`SKILL.md` body into the process prompt and records the exact package snapshot
on the process. Bundled resources are read explicitly with `read_skill_resource`
from that activation snapshot, so later registry replacement affects only new
activations.

This prevents a Skill from keeping ambient read authority to the workspace path
where it was registered.

## LibOS Extensions

`allowed-tools` adds existing static tools to the process tool table during
activation. The tools remain wrappers over primitives; visibility does not imply
resource authority.

`references/agent-libos/jit-tools.json` declares TypeScript JIT tools. Each
entry references a `scripts/*.ts` source file:

```json
[
  {
    "name": "count_lines",
    "description": "Count lines in a text file.",
    "source_path": "scripts/count_lines.ts",
    "input_schema": {"type": "object"},
    "output_schema": {"type": "object"},
    "timeout_s": 10,
    "tests": []
  }
]
```

JIT sources are snapshotted at registration. At activation, bundled JIT tools
are validated and registered through the same ToolBroker path as proposed JIT
tools, including sandbox resource limits and metrics. They can only access libOS
through `libos.syscall()`.

`timeout_s` is optional. When present it sets this JIT tool's outer Deno
execution window; when omitted, `tools.deno_timeout_s` applies. The value must
be a finite, non-boolean number greater than zero and no greater than
`tools.deno_timeout_hard_limit_s`. Process
subprocess wall-time budgets can still terminate execution sooner.

The bundled `swe-agent` editor uses a 1 MiB bounded filesystem read. It refuses
to write when that read reports `truncated: true`; otherwise a partial prefix
could overwrite and destroy the unseen suffix. Large files require an editor or
file workflow that preserves the complete source rather than retrying
`swe_edit` with a partial payload.

`actions.json` and `required-capabilities.json` are advisory prompt metadata.
They do not create capabilities.

## Sources And Trust

Host-side catalog discovery searches exactly the roots in
`skills.workspace_dirs`, followed by `skills.global_dirs`, with equivalent
paths de-duplicated. Defaults include `skills/`, `.agent_libos/skills/`, and
the compatible `.agents/skills/` and `.claude/skills/` roots. Overriding
`workspace_dirs` replaces that workspace-root set; there are no additional
implicit catalog roots.

Workspace Skills are registered through the filesystem primitive when an
AgentProcess is the actor. The process must be able to read `SKILL.md` and any
referenced metadata or script resources. Registration also snapshots additional
bundled files under `scripts/`, `references/`, and `assets/` when the process
already has the corresponding directory and file read authority; it does not
grant or prompt for ambient package-directory reads just to discover optional
resources.

Global Skills are read only from configured global Skill directories. With the
default `skills.global_requires_trust: true`, their full package SHA-256 must
be trusted before registration, either through configured
`trusted_global_package_sha256` values or a durable trust record:

```bash
uv run agent-libos --db .agent_libos.sqlite skills trust ~/.agent-libos/skills/review-helper
uv run agent-libos --db .agent_libos.sqlite skills register ~/.agent-libos/skills/review-helper --source-type global
```

Setting `skills.global_requires_trust: false` disables only this hash-trust
gate. Directory containment, package validation, Skill capabilities, JIT
sandboxing, and primitive authority checks still apply, but every otherwise
valid package under a configured global directory can then be registered
without prior hash pinning. This materially widens the trusted prompt/tool
control plane: changing a file in such a directory can change instructions,
tool visibility, and bundled JIT source accepted at the next registration.
Disable the gate only when those directories and every writer to them are part
of the Host's trusted computing base.

Admin CLI registration can read and snapshot a workspace package directly:

```bash
uv run agent-libos --db .agent_libos.sqlite skills validate skills/swe-agent
uv run agent-libos --db .agent_libos.sqlite skills register skills/swe-agent
uv run agent-libos --db .agent_libos.sqlite skills activate <pid> swe-agent
```

With `--actor-pid`, the CLI enforces that process's filesystem and Skill
capabilities.

Capability requirements:

- Discovering registered Skills as a process requires `skill:*` `read`.
- Inspecting a registered Skill as a process requires `skill:<name>` `read`.
- Registering or replacing a Skill requires `skill:<name>` `write`.
- Activating or unloading a Skill requires `skill:<name>` `execute`.
- Activating or unloading a Skill for a different process also requires
  `process:<pid>` `admin`.
- Trusting or untrusting global Skill package hashes requires
  `skill_trust:*` `admin`.

The host catalog shown to admin callers uses the configured and de-duplicated
workspace/global roots described above; it does not add a second implicit root
list. Process-driven workspace registration remains path-based and must pass
filesystem authority for each package file it snapshots.

## Activation And Unload

Registration, trust changes, activation, and unload reauthorize their complete
allow/deny decisions after entering the operation's `AuthorityTransaction`,
then reserve any finite-use Skill/process-admin grants before the durable
mutation. Business rows, event/audit evidence, and finite-use settlement share
that transaction. A failure before commit therefore restores the exact
reservation token; an explicit revoke still wins over late cleanup.

Process-mediated `discover` and `inspect` are read-only catalog operations.
They reserve the selected finite read decision before constructing the result,
restore it if result construction fails, and commit it after a successful
read; they do not create a business mutation or event/audit publication and do
not use the mutation `AuthorityTransaction` path.

`activate_skill` is atomic across the process tool table, loaded Skill metadata,
process-local JIT rows, executable handles, and name aliases. The runtime
validates the package, existing tool references, duplicate tool/JIT names,
TypeScript source limits, Deno static checks and tests through ToolBroker, and
static tool shadowing before publishing the new activation. A failed activation
discards its unpublished candidates and executable aliases, including when
authority settlement fails. Reactivation retires only the superseded JIT ids
after the replacement and its authority settlement commit.

`unload_skill` removes tool visibility and prompt instructions contributed by
that Skill, along with the loaded Skill's process-local JIT tool and candidate
rows and executable aliases. It does not revoke capabilities, delete audit
history, or roll back external side effects. Activation records the full-tool
and model-projection bindings that existed independently before each Skill
claimed an alias. Unload first selects a still-loaded Skill source, otherwise
restores that recorded base binding, and removes the alias only when no source
remains. Thus unloading a Skill cannot erase an image/manual base tool or a
static tool shared by another loaded Skill. A JIT registration is retired only
after no remaining loaded Skill references its tool id. Checkpoint/image remap
paths preserve these provenance ids together with the ordinary loaded-Skill
tool ids. On an older persisted row that lacks provenance fields, unload first
reconstructs static base bindings from the image declaration and the latest
explicit `process.tools.configure/project` audit evidence; it never infers an
ephemeral JIT id as a base source.

Skill discovery accepts only positive integer limits up to
`SkillDefaults.discover_limit`; boolean, zero/negative, and above-config values
are rejected before a result set can become unbounded.

## Process Semantics

- Image `default_skills` activate at spawn and exec time; failure fails image
  boot instead of starting with a partial default Skill set.
- Fork inherits activated Skills and corresponding tool visibility.
- Spawn-child starts without parent-activated Skills.
- Exec resets activated Skills to the target image defaults.
- No image or Skill default grants external resource capabilities.

## SWE-Agent Style Skill

The workspace includes `skills/swe-agent`, named and registerable as
`swe-agent`. It
reproduces the useful SWE-Agent Agent Computer Interface shape inside Agent
libOS:

- `swe_view` for directory listings and bounded file windows,
- `swe_grep` for concise repository search through `rg`,
- `swe_edit` for exact-text, line-range, or create-if-missing edits,
- `swe_run` for test and diagnostic commands,
- `swe_submit` for final structured process exit.

The Skill carries workflow instructions for localizing before editing, keeping
actions small, treating repository output as untrusted, running focused tests,
and submitting with summary, tests, and residual risk. It does not grant
filesystem or shell authority.
