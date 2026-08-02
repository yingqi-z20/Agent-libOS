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
compatibility: agent-libos==1.3.0
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
characters, consist of one or more non-empty lowercase-alphanumeric segments
separated by single hyphens, and match the package directory name. Leading,
trailing, or repeated hyphens are invalid. A configured
`skills.name_max_chars` below 64 imposes the lower limit.
`metadata` values must be strings.
`compatibility` is optional and, when present, is validated only as a non-empty
string. The Runtime treats it as opaque publisher metadata: it stores and
projects the value and includes it in the package hash, but does not parse
Semantic Versioning, compare it with the running Agent libOS version, or reject
registration or activation because an expression does not match. The Host or
package-release workflow is responsible for interpreting and verifying this
declaration before approving a package hash.
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
Resource traversal is independently bounded by `max_package_directories` and
`max_package_depth`, whose defaults are 256 aggregate directories and 32
levels, respectively. Missing optional resource roots do not consume the
directory budget. Detecting a package that exceeds either topology bound
rejects the complete package; traversal never returns a partial package.
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
The exact fields, defaults, and cross-field validation rules are listed in the
[configuration reference](configuration.md#configuration-reference).

## Progressive Disclosure

Fresh shipped images put no Skill metadata or body in the model prompt. The
model starts with the common Skill lifecycle bootstrap and uses
`discover_skills` only when task-specific guidance or a domain schema is
needed. `text` and `limit` apply to every visible Skill in one uniformly bounded
page. Discovery case-folds and de-duplicates Unicode word terms from the query,
and drops one-character and common low-information terms when at least one
informative term remains. A one-term query requires that term to match Skill
id, name, or description. A query with two or more selected terms requires at
least two distinct term matches; it does not require every term to match.
An exact full id or name match returns only exact matches. Otherwise results
combine a normalized-phrase bonus with the number and location of matched
terms; id/name hits weigh more than description-only hits. The combined score
sorts descending, with deterministic case-folded name and id tie-breaking.
`next_step` is `activate_skill` when a returned match is not current,
`use_loaded_skill` when every returned match has the same loaded/catalog hash,
or `refine_search` for a zero-result query; an unchanged query is never
pagination. `has_more` only
reports more matches for the same query, while `visibility_limited` reports
that catalog authority prevented searching all configured sources. There is no
cursor.

Every model-facing discovery item has the same schema: identity, description,
declared high-level bindings and requirements, package hash, and `active`.
Source type, registration provenance, and the Host's immutable built-in
implementation are intentionally absent. Discovery returns metadata only; the
full `SKILL.md` body and domain tool schemas appear after activation.
`active=true` binds the trusted loaded snapshot to the discovery row's exact
`package_sha256`; an older immutable snapshot remains loaded but is reported
inactive after catalog replacement. Activation requires that discovered hash
as `expected_package_sha256` and fails before publication if it is stale.

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
entry references a `scripts/*.ts` source file. Its `name` follows the OpenAI
tool-name syntax `[A-Za-z0-9_-]{1,64}` (hyphens and underscores are both
valid, but dots and slashes are not):

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

Every JIT test entry must contain `expected`; omitting it rejects the test
before executing candidate source. `args` is optional and defaults to `{}`.
`syscalls` is also optional and defaults to an empty ordered list. Each syscall
fixture names the expected syscall; when `args` is present it is compared
exactly. Unexpected, out-of-order, or unconsumed expected syscalls fail
validation.

For a successful fixture (`ok` omitted or true), the injected return value is
`result` when that key is present, otherwise `payload`, otherwise JSON `null`.
Thus `result` deliberately wins if legacy metadata contains both keys. An
`ok: false` fixture instead injects the same text-free `syscall_error` boundary
seen by live JIT code; `result` and `payload` are not returned, and any legacy
`error` text is deliberately ignored so package-authored diagnostics cannot be
exposed to candidate code. These precedence and default rules preserve
compatibility with already registered package snapshots; new packages should
use `result` and omit the legacy aliases.

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
implicit catalog roots. Host catalog enumeration and registered-row search are
both bounded by `skills.catalog_scan_limit` (1000 by default). Detecting entry
or row N+1 rejects discovery instead of returning a result that could be
mistaken for a complete search.

Host-path loading, including admin validation/registration and Host catalog
inspection, takes an identity-stable package snapshot. It rejects symlink or
reparse-point components, hard-linked package files, non-regular files,
non-directory resource topology, and platforms on which a stable file or
directory identity cannot be established. Files must remain linked at the
same path with the same identity and metadata throughout their bounded read;
directories are checked before and after enumeration. A concurrent rename,
replacement, mutation, growth, or other identity uncertainty therefore fails
the whole load rather than hashing or registering a mixed snapshot. These are
Host-path rules; process-driven workspace registration below additionally goes
through the filesystem primitive and its authority and path-safety contract.

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
uv run agent-libos --db .agent_libos.sqlite skills discover --text swe-agent
uv run agent-libos --db .agent_libos.sqlite skills activate <pid> swe-agent --expected-package-sha256 <package_sha256-from-discover>
```

Activation requires the exact lowercase package hash from discovery and fails
without replacing the loaded snapshot if the registered content changed in
between. Rediscover and review the new metadata before using a new hash.

With `--actor-pid`, process-mode commands enforce that process's applicable
filesystem, registry, Skill, target-process, source, and trust capabilities.
Static `skills validate` accepts a Host filesystem path and therefore rejects
`--actor-pid`; use actor-mode `skills register` when package reads must cross
the process workspace filesystem boundary.

Capability requirements (resource strings below name their default values;
Hosts can replace the wildcard registry/trust roots through the referenced
configuration keys):

- Discovering applicable built-in Skills requires no Skill Capability;
  discovering registered Skills as a process requires `read` on
  `skills.registry_resource` (default `skill:*`).
- Inspecting a registered Skill as a process requires `skill:<name>` `read`.
- Registering or replacing a Skill requires `skill:<name>` `write`.
- Activating or unloading a registered Skill requires `skill:<name>` `execute`.
- Same-process activation or unload of a catalog-verified built-in Skill needs
  no Skill Capability because it only changes prompt and schema visibility.
- Activating or unloading a Skill for a different process also requires
  `process:<pid>` `admin`.
- Trusting or untrusting global Skill package hashes requires `admin` on
  `skills.trust_resource` (default `skill_trust:*`).

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

Process-mediated `discover` and `inspect` do not mutate the Skill catalog and
do not publish Skill lifecycle events or Skill-operation audit rows. They do,
however, use and durably settle authority. They reserve the selected finite
`read` decision before constructing the result, restore it if result
construction fails, and commit it after a successful read. Committing a
limited-use Capability writes the normal capability-use audit evidence and can
emit `CAPABILITY_REVOKED` when that use exhausts the grant. These reads do not
use the mutation `AuthorityTransaction` path.

`activate_skill` is atomic across the process tool table, loaded Skill metadata,
process-local JIT rows, executable handles, and name aliases. The runtime
validates the package, existing tool references, duplicate tool/JIT names,
TypeScript source limits, Deno static checks and tests through ToolBroker, and
static tool shadowing before publishing the new activation. A failed activation
discards its unpublished candidates and executable aliases, including when
authority settlement fails. Reactivation retires only the superseded JIT ids
after the replacement and its authority settlement commit.

For Durable Task Runs, v1 can certify this binding change only when
`activate_skill` is the single action in a non-parallel validated model
response. Its request binding is checked on both sides of the Provider call,
and the dispatch target is the exact pre-action `tool_id`. The
committed result plus exact post-action tool and loaded-Skill bindings become
part of the safe resume evidence. Other actions that change Image, tool,
provider, or Skill bindings are not certified by v1; they fail closed into
`needs_attention` before a Run may continue or reopen, with no automatic
replay.

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
inside Agent libOS. The shipped package pins `compatibility` to
`agent-libos==1.3.0`: its JIT manifest uses extension fields from this release,
so older parsers must not be promised compatibility merely because the
frontmatter itself can be read. This exact pin is a publisher/Host-facing
declaration, not a Runtime-enforced version gate.

- `swe_view` for directory listings and bounded file windows,
- `swe_grep` for concise repository search through `rg`; it requests a
  NUL-delimited filename on every match, so its `files` list is reliable for a
  single-file search and paths containing colons, while `max_results` bounds
  both its returned match lines and its file list. Calling it also requires the
  Host Shell provider to resolve `rg` from its safe PATH and authorize it under
  Shell Capability and policy; Skill activation validates the JIT package but
  does not prove that this external executable is installed or callable,
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
For a file, `swe_view` reads from byte zero and then selects a bounded line
window from the decoded prefix. Its returned `start_line` is the effective
start after clamping the request into `1..max(total_lines, 1)`, not an echo of
an out-of-range request. `total_lines` counts only logical lines in the
observed prefix, and `lines_below` counts only observed-prefix lines below the
returned window. When `truncated_by_bytes` is true, neither field describes or
estimates the unobserved suffix; conclusions about the complete file require a
complete-file-safe read or a narrower source.
`swe_run` returns the bounded prefixes retained by the shell primitive and
propagates `stdout_truncated` and `stderr_truncated`. If either flag is true,
the corresponding stream is incomplete even though `returncode` remains the
command's exit status; the absent suffix must not be treated as evidence that
a diagnostic or match did not occur.
`swe_grep` propagates the shell capture's truncation flags. If stdout was
truncated, `matches_incomplete` is true, `omitted_matches` is null because the
total is unknowable, and an incomplete NUL-framed tail is excluded from both
`matches` and `files`. `observed_omitted_matches` counts only complete captured
matches excluded by `max_results`; `output_incomplete` also covers truncated
stderr or malformed/incomplete stdout framing.
The Skill does not grant filesystem or shell authority. `swe_view` and
`swe_edit` use typed filesystem primitives, but `swe_run` and `swe_grep`
authorize the outer native-process launch through Shell Capability, policy,
data-flow, cwd/environment checks, and resource controls. Once an authorized
native child runs, its direct filesystem, network, child-process, or other I/O
does not re-enter the corresponding typed libOS primitives; Shell is not an
operating-system sandbox. A Host that runs untrusted native code must supply a
stronger container, WASM, VM, service-provider, or comparable isolation
boundary.
