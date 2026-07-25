# Skills

Agent libOS Skills use the standard Agent Skills package shape: a directory
with a required `SKILL.md` file, optional `scripts/`, `references/`, and
`assets/` resources, and progressive disclosure from catalog metadata to full
instructions and bundled resources. Immutable built-in tool Skills ship inside
the `agent_libos` package; workspace, global, and runtime Skills continue to use
the registered catalog.

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
allowed-tools: read_text_file read_directory
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
`compatibility`, `allowed-tools`, and `metadata`. The `name` must be 1–64
characters, start with a lowercase letter or digit, contain only lowercase
letters, digits, and hyphens, and match the package directory name. A configured
`skills.name_max_chars` below 64 imposes the lower limit.
`metadata` values must be strings.
Per the Agent Skills specification, `allowed-tools` is a space-separated YAML
string. Agent libOS still accepts the historical YAML-list spelling for
registered third-party packages, but generated and repository-shipped packages
use the canonical scalar form.

Agent libOS reserves the `agent-libos-` Skill-name prefix and the
`agent-libos.*` metadata namespace. Registered packages cannot claim the name
of, replace, or shadow an immutable built-in Skill. Complex extension data
lives in `references/agent-libos/*.json`; metadata only points at those
relative files.

Built-in tool packages intentionally contain only `SKILL.md`. Their frontmatter
is limited to `name`, `description`, and `allowed-tools`; each body is at most
16 KiB, the complete `SKILL.md` is at most 24 KiB (leaving 8 KiB for
frontmatter), and each package owns at most nine static tools. The expanded body
budget is loaded only after activation and is reserved for exact tool selection,
parameter combinations, result interpretation, recovery, and completion-evidence
guidance. The default runtime prompt ceiling is 16,384 characters, which is not
lower than the 16 KiB body byte cap, so every valid built-in body is fully
model-visible when activated. `Runtime.open()` validates the complete immutable
built-in catalog against the configured ceiling. A Host ceiling below any
shipped body therefore prevents that Runtime from opening; this is not deferred
until the oversized Skill is selected or activated, and bodies are never
silently clipped. Packages contain no scripts,
bundled resources, JIT definitions, actions, or Capability declarations.
Unlike registered packages, package-owned built-ins fail validation if
`allowed-tools` uses the legacy YAML-list spelling.

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
longer matches its declared size/SHA is rejected. SHA-256 here is an unkeyed
content fingerprint, not a digital signature, MAC, publisher identity, or proof
that the instructions are safe. Global trust records are exact hash pins whose
authenticity depends on how the Host obtained and approved that hash.

## Progressive Disclosure

Fresh shipped images put no Skill metadata or body in the model prompt. The
model starts with the common Skill lifecycle bootstrap and uses
`discover_skills` only when task-specific guidance or a domain schema is
needed. `text` and `limit` apply to every visible Skill in one uniformly bounded
page. Discovery treats two to four concrete metadata terms as an intent query,
requires each informative term, and relevance-ranks the matches. `next_step`
directs the model to activate a plausible exact id or refine a zero-result
query; an unchanged query is never pagination. `has_more` only reports more
matches for the same query, while `visibility_limited` reports that catalog
authority prevented searching all configured sources. There is no cursor.

Every model-facing discovery item has the same schema: identity, description,
declared high-level bindings and requirements, package hash, and `active`.
Source type, registration provenance, and the Host's immutable built-in
implementation are intentionally absent. Discovery returns metadata only; the
full `SKILL.md` body and domain tool schemas appear after activation.

For an immutable packaged Skill, discovery omits the package unless every
declared static tool has an exact image-authorized process binding. Registered
workspace/global/runtime discovery is deliberately metadata-only: entries are
searched when the actor has the configured catalog `read` authority, without
rerunning activation-time loadability checks. A discovered registered entry can
therefore fail activation if, for example, a declared static tool is no longer
registered, a JIT definition fails current validation, or its tool name now
collides with the process table. Activation performs those checks atomically and
publishes no partial Skill on failure. These trust and applicability differences
are enforcement details, not different LLM protocols.

Activation materializes the full body into the process prompt and records the
exact package snapshot on the process. Bundled resources are read explicitly
with `read_skill_resource` from that activation snapshot, so later registry
replacement affects only new registered-Skill activations.

The model-facing lifecycle tools use distinct outer envelopes: discovery
returns its `skills` page and paging metadata directly, while successful
activation and unload return `{"result": {...}}`; resource reads return
`{"resource": {...}}`. The activation result contains `pid`, `skill_id`,
`name`, `version`, `tool_names`, `tool_ids`, `jit_tool_ids`,
`instructions_hash`, and `package_sha256`. The unload result contains `pid`,
`skill_id`, and `removed_tools`.

A successful registered-Skill activation requires `skill:<name>` `execute` and
binds the snapshot to the process. Reading a bundled resource from that loaded
snapshot does not perform a second `skill:<name>` `read` check. The response
has the outer shape `{"resource": {...}}`. The nested resource always contains
both `content` and `content_base64`: for `kind="text"`, `content` is the text and
`content_base64` is null; for `kind="base64"`, the inverse holds. It also
contains `skill_id`, `path`, `size_bytes`, and `sha256`. The `read` right below
governs registered catalog discovery and inspection, not each loaded resource
read.

In multiplexed JIT mode, prompt projection omits individual JIT schemas and the
JIT catalog/source resource entries. This is a discovery boundary, not a second
resource-access boundary: `read_skill_resource` still accepts any exact path in
the already loaded immutable snapshot. A path retained from trusted package
instructions or prior authorized evidence can therefore be read, including an
otherwise omitted JIT-contract resource. The tool does not list paths, and
callers must not guess or probe hidden filenames.

This prevents a Skill from keeping ambient read authority to the workspace path
where it was registered.

## LibOS Extensions

For a registered Skill, `allowed-tools` adds existing static tools to the full
process and model tool tables during activation. The tools remain wrappers over
primitives; visibility does not imply resource authority.

For an immutable packaged Skill, the Host internally uses a trusted projection:
activation atomically copies every named binding from the existing full process
tool table into the model projection. It does not accept a partial intersection,
resolve a missing tool from the registry, add to the full table, create JIT
code, or grant Capability. This provenance distinction remains in durable Host
state and audit evidence, but the model receives the same activation result
shape as for every other Skill.

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
They do not create capabilities, but their package schemas are validated and
their bytes participate in the package hash. An action entry accepts exactly
`name`, `use_cases`, `input_schema`, `output_schema`,
`required_capabilities`, `side_effects`, `failure_modes`, and `examples`.
A required-capability entry accepts exactly `resource`, `rights`, and optional
`constraints`. It has a non-empty canonicalizable `resource`, a non-empty
`rights` array of Capability right strings, and an optional mapping
`constraints`; unknown keys are rejected so a misspelled security field cannot
silently become advisory metadata:

```json
{
  "name": "summarize_file",
  "use_cases": ["Summarize a workspace document"],
  "input_schema": {"type": "object"},
  "output_schema": {"type": "object"},
  "required_capabilities": [
    {"resource": "filesystem:workspace:*", "rights": ["read"]}
  ],
  "side_effects": [],
  "failure_modes": ["file is unavailable"],
  "examples": []
}
```

JIT test entries use `args`, optional `expected`, and an ordered `syscalls`
array. Each syscall fixture supplies `name`, optional exact `args`, optional
`ok`/`error`, and either `result` or `payload`; unexpected, out-of-order, or
unconsumed expected syscalls fail validation.

## Sources And Trust

Built-in packages are loaded read-only from `agent_libos` package resources and
kept separate from durable registry rows and configurable catalog roots. This
source provenance is available to Host administration and audit, not to the
model-facing discovery or lifecycle result schemas. Their package snapshots
still bind loaded prompt instructions and tool ownership across runtime reopen,
fork, checkpoint, and context compaction.

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

With `--actor-pid`, process-mode commands enforce that process's applicable
filesystem, registry, Skill, target-process, source, and trust capabilities.
Static `skills validate` accepts a Host filesystem path and therefore rejects
`--actor-pid`; use actor-mode `skills register` when package reads must cross
the process workspace filesystem boundary.

Capability requirements:

- Discovering applicable built-in Skills requires no Skill Capability;
  discovering registered Skills as a process requires `skill:*` `read`.
- Inspecting a registered Skill as a process requires `skill:<name>` `read`.
- Registering or replacing a Skill requires `skill:<name>` `write`.
- Activating or unloading a registered Skill requires `skill:<name>` `execute`.
- Same-process activation or unload of a catalog-verified built-in Skill needs
  no Skill Capability because it only changes prompt and schema visibility.
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

Host-private loaded records retain the provenance needed to distinguish a
registered activation from an immutable packaged projection. That field is not
rendered into prompts or model tool results. An immutable projection must
preserve the full process tool table and Capability set byte-for-byte while
updating only model visibility and the loaded prompt snapshot. Permission-free
unload is accepted only when the runtime validates its trusted catalog and
snapshot provenance. A forged source, hash, ID, or persisted activation kind
fails closed.

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

- An image with `metadata.tool_projection: skills` begins with
  `discover_skills`, `activate_skill`, `read_skill_resource`, `unload_skill`,
  and `process_exit`; every bootstrap tool must be authorized by
  `default_tools`, or image validation fails.
- An explicitly configured image `default_skills` set activates at spawn and
  exec time; failure fails image boot instead of starting partially. Shipped
  general-purpose images leave this set empty so all Skills load on demand.
- Fork and checkpoint restore inherit loaded Skill snapshots, activation kind,
  provenance, and corresponding model-tool visibility.
- Spawn-child starts without parent-activated Skills.
- Exec resets activated Skills to the target image defaults.
- No image or Skill default grants external resource capabilities.

## SWE-Agent Style Skill

The workspace includes `skills/swe-agent`, named and registerable as
`swe-agent`. It reproduces the useful SWE-Agent Agent Computer Interface shape
inside Agent libOS:

- `swe_view` for directory listings and bounded file windows,
- `swe_grep` for concise repository search through `rg`; `max_results` bounds
  both its returned match lines and its file list,
- `swe_edit` for exact-text, line-range, or create-if-missing edits,
- `swe_run` for test and diagnostic commands,
- `swe_submit` for preparing a structured final payload that must then be
  passed to the built-in `process_exit` tool.

The Skill carries workflow instructions for localizing before editing, keeping
actions small, treating repository output as untrusted, running focused tests,
and preparing a summary, tests, and residual-risk payload. `swe_submit` does
not itself exit, so image completion review remains enforced by the separate
`process_exit` call. `swe_edit` treats line numbers and any supplied exact-text
occurrence as strict one-based coordinates. A line range requires both
endpoints, invalid or out-of-range values fail instead of selecting a nearby
line/match, and an omitted occurrence selects the first exact-text match. For
source files with LF or CRLF separators,
a line-range edit normalizes inserted text to the first separator encountered;
if neither separator exists it uses LF. `swe_run` always returns `returncode`;
empty output is described as successful only for return code zero, so callers
must treat a nonzero code as failure even when both streams are empty.
`swe_grep` propagates the shell capture's truncation flags. If stdout was
truncated, `matches_incomplete` is true, `omitted_matches` is null because the
total is unknowable, and `observed_omitted_matches` counts only captured matches
excluded by `max_results`; `output_incomplete` also covers truncated stderr.
The Skill does not grant filesystem or shell authority.
