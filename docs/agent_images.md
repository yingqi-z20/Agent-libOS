# AgentImage Authoring

This guide covers user-defined directory Image packages. An `AgentImage` defines
how a process boots: its system prompt and prompt composition mode, exact default
tool table, default Skills, optional packaged Deno/TypeScript JIT tools, optional
private workspace seed, declared Capability requirements, required startup
modules, and optional Host-configured LLM profile id.

An Image changes process behavior and visible actions; it does not grant general
resource authority. Read [Capabilities](capabilities.md),
[Task Authority Manifests](task_authority_manifest.md), and
[Tools and JIT](tools_and_jit.md) before packaging an effectful agent.

## In this guide

- [Distinguish the Image kinds](#image-kinds)
- [Create a minimal package](#minimal-package)
- [Validate, register, and boot the package](#validate-before-registration)
- [Review the allowed package layout](#package-layout)
- [Choose manifest fields](#manifest-fields)
- [Choose a prompt mode](#prompt-modes)
- [Control the tool table and model projection](#tool-table-and-model-projection)
- [Declare capabilities and required modules](#capability-and-module-declarations)
- [Scope the private workspace seed](#private-workspace-seed)
- [Package JIT tools](#packaged-jit-tools)
- [Select an LLM profile and keep secrets out](#llm-profile-and-secrets)
- [Create an Image from a checkpoint](#checkpoint-derived-images)
- [Review the author checklist](#author-checklist)
- Return to the [documentation home](index.md).

## Image kinds

Agent libOS can boot three related forms:

- built-in Images registered by the Runtime;
- immutable directory-package Images described here; and
- checkpoint-derived Images created by `images commit`.

A package captures only its validated manifest, prompt, declared resources,
workspace seed, and JIT files. A checkpoint-derived Image captures reconstructable
Runtime state from one checkpoint root. Neither form captures or rolls back
external provider state.

## Minimal package

Create a directory with an `IMAGE.yaml` and UTF-8 prompt:

```text
images/custom-review-agent/
  IMAGE.yaml
  prompt.md
```

```yaml
# images/custom-review-agent/IMAGE.yaml
image_id: custom-review-agent:v0
name: custom-review-agent
version: v0
prompt: prompt.md
prompt_mode: libos_default
default_tools:
  - human_output
  - process_exit
required_capabilities:
  - resource: filesystem:workspace:*
    rights: [read]
```

The three required manifest fields are `image_id`, `name`, and `prompt`.
`version` defaults to `v0`; `prompt_mode` defaults to `image_only`; omitted list
and mapping fields are empty. The manifest is closed: unknown fields fail
validation rather than being ignored.

The example's `required_capabilities` entry is a declaration. It reports what
the Image expects but grants no filesystem authority. The Host must authorize a
narrow, task-appropriate path when launching the process.

## Validate before registration

Static validation does not register or boot the Image:

```bash
uv run agent-libos --db user \
  images validate images/custom-review-agent
```

When validation succeeds, register the immutable package artifact and inspect
the result:

```bash
uv run agent-libos --db user \
  images register images/custom-review-agent
uv run agent-libos --db user \
  images inspect custom-review-agent:v0
```

Like the Skills CLI, these commands accept `--actor-pid`; omitting it runs
them in admin CLI mode with capability enforcement disabled.

Registration rejects an existing image id unless the Host explicitly supplies
`--replace`. Validation and registration read the package but do not invoke an
LLM. Boot-time validation may additionally require Deno and exact startup
modules when the package declares them.

To exercise a registered Image, spawn it in the same persistent database. The
spawn output supplies the pid used for subsequent narrow grants and execution:

```bash
uv run agent-libos --db user \
  spawn --image custom-review-agent:v0 --goal "Review the selected file"
```

Do not run the scheduler until the Host has reviewed the task's exact authority,
model profile, token budget, and output Sink. `run`, `llm-once`, and `exec --run`
may call a real provider and spend tokens.

## Package layout

A complete package may use these locations:

```text
images/custom-review-agent/
  IMAGE.yaml
  prompt.md
  tools/
    jit-tools.json
    scripts/
      summarize.ts
  resources/
    rubric.md
  workspace/
    seed.txt
```

Only declared package content is persisted:

- `IMAGE.yaml` and its referenced prompt are allowed at the package root;
- `resources/` contains inert package resources;
- `workspace/` contains the optional materialized workspace seed; and
- `tools/` is included only when `jit_tools` references a `tools/*.json`
  catalog whose `source_path` entries stay under `tools/scripts/*.ts`.

Unrelated top-level files, absolute/traversing paths, `.git` metadata, links,
special files, duplicate normalized paths, missing references, invalid UTF-8,
and configured file/count/byte-limit violations fail validation. Do not place
credentials, `.env`, local caches, or private notes beside the manifest.

## Manifest fields

| Field | Purpose |
| --- | --- |
| `image_id` | Required stable id, conventionally `<name>:<version>` |
| `name` | Required display/name identity |
| `prompt` | Required package-relative UTF-8 prompt path |
| `version` | Image version string; defaults to `v0` |
| `prompt_mode` | `image_only`, `minimal_runtime`, or `libos_default` |
| `jit_tool_exposure` | `direct` or `multiplexed`; defaults to `direct` |
| `planner` | Image-defined planner metadata plus validated context-management settings |
| `action_schema` | Optional structured action metadata |
| `default_skills` | Skill ids requested during boot under the Skill lifecycle |
| `default_tools` | Exact initial static process tool table |
| `context_policy` | Image context-selection policy; defaults to `plan_first` |
| `safety_profile` | Image safety-profile id; defaults to `default` |
| `llm_profile` | Host-configured profile id, never an API key |
| `required_capabilities` | Requirement declarations, never grants |
| `required_modules` | Exact startup module id/source-hash prerequisites |
| `metadata` | Closed manifest's extensible metadata mapping |
| `signature` | Optional package field; it does not replace Host source trust or artifact validation |
| `jit_tools` | Package-relative `tools/*.json` JIT catalog |
| `workspace` | Optional private workspace seed, cwd, and package-scoped grants |

Exact size/count limits live in `agent_libos.config.DEFAULT_CONFIG`; see
the [configuration guide](configuration.md) for semantics and the
[generated field reference](configuration_reference.md) for exhaustive paths,
types, defaults, and units. A field being syntactically valid does not mean a
Host should enable it for every task.

## Prompt modes

`prompt_mode` and Host `llm.prompt_layout` are independent.

### `image_only`

This is the custom-package default. For an ordinary non-TaskRun process, the
Image prompt is the exact system message, a string goal is the unchanged first
user message, a structured goal is canonical JSON, and later turns use the
durable native tool transcript. The Runtime does not inject its ordinary Object
Memory, Capability, Skill, planner, or fallback-protocol sections.

Lossless transcript replay requires `llm.persist_full_io: true`; execution fails
before provider dispatch when it is disabled. Durable Task Run supervision is
an explicit exception: the Run supplies its Host-authored durable user contract
while preserving the Image system prompt. `image_only` cannot use prompt-mode
context management because that would add Runtime-authored model input.

### `minimal_runtime`

Includes the cumulative completion contract, factual Runtime state, activated
Skills, recovered-goal context, and optional fallback JSON guidance, while
omitting the complete Agent libOS base/action-planner envelope.

### `libos_default`

Adds the built-in Agent libOS planner envelope to `minimal_runtime`. Built-in
general-purpose Images use this mode.

See [Runtime Model: Images and Tool Tables](runtime_model.md#images-and-tool-tables)
for the exact ordinary-process/TaskRun and prompt-layout combinations.

## Tool table and model projection

`default_tools` is exact. The Runtime does not silently add `process_exit`,
Object Memory, Human output, or any other builtin. List every static tool the
Image must bind. A process can call only a binding in its complete tool table;
the model can select only tools in its narrower model projection. Both paths
still enter primitive authority checks.

Without `metadata.tool_projection: skills`, the static bindings are also the
initial model projection. With that metadata setting, the Image must provide the
required Skill bootstrap tools; immutable built-in Skills can project only their
already-authorized owned subset after activation. A separately registered Skill
may expand tables only through its own trust and Skill-authority checks.

## Capability and module declarations

### `required_capabilities`

Each entry uses the Task Authority Capability-spec shape. Requirements are copied
into launch evidence and reported as satisfied or missing. They are not launch
grants, not a sandbox, and not a substitute for the Host's
`authorized_capabilities`.

Root spawn never grants them automatically. Exec and checkpoint/package boot do
not grant them either. The Host should authorize only the specific resources and
rights required by the actual task.

### `required_modules`

Each entry contains:

```yaml
required_modules:
  - module_id: example-module:v0
    source_sha256: "<64-character source hash from modules verify>"
```

Registration normalizes hexadecimal case, but boot requires that exact module id
and source digest to have been trusted and loaded before spawn/exec. An Image
never loads a Python module automatically. See [Runtime Modules](modules.md).

## Private workspace seed

An Image package may materialize a private per-process copy of `workspace/`:

```yaml
workspace:
  source: workspace
  working_directory: .
  grants:
    - path: .
      rights: [read, write]
      recursive: true
      delegable: false
```

This is the narrow trusted-package bootstrap exception to the ordinary rule that
Image metadata does not grant authority. `workspace.grants` may issue only
filesystem `read`, `write`, and `delete` rights for the materialized private
copy. Paths must stay inside `workspace.source`; grants do not expose or
authorize the Host package source directory or the caller's general workspace.
They are nondelegable unless an entry explicitly opts in, and any child
derivation still follows ordinary attenuation.

Do not confuse package workspace grants with `required_capabilities`: the former
apply only to the Runtime-owned private copy, while the latter are unmet external
requirements/declarations.

## Packaged JIT tools

Set `jit_tools` to a JSON catalog under `tools/`:

```yaml
jit_tools: tools/jit-tools.json
jit_tool_exposure: direct
```

Each JSON entry supplies a name, description, `source_path` under
`tools/scripts/`, input/output JSON Schemas, optional bounded contract tests,
metadata, and optional positive `timeout_s`. Package validation and registration
reject imports and unsafe source and validate the schema and bounded test-case
shapes without requiring Deno. Process boot then uses the ToolBroker validation
path, including the declared contract tests, before a package JIT tool becomes
visible; boot fails closed when a usable Deno executable is unavailable.

`direct` exposes each JIT tool schema individually. `multiplexed` exposes one
stable `run_jit_tool` schema; the Image prompt or Skill instructions must then
describe the available JIT catalog. The multiplexer name is reserved and cannot
also be a packaged JIT tool.

Deno availability, the per-tool timeout, the Host hard limit, and process
resource budgets remain independent. A JIT syscall still needs the same
Capability, Task Authority, policy, data-flow, and budget checks as its Python
primitive. See [Tools and JIT](tools_and_jit.md).

## LLM profile and secrets

`llm_profile` is an id resolved from Host configuration. The package never
contains an endpoint credential. Keep `api_key_env` in Host configuration and
the secret in the named environment variable. A non-default profile does not
inherit the default profile's ambient endpoint/model policy; see
[Configuration: effective LLM profile precedence](configuration.md#effective-llm-profile-precedence).

## Checkpoint-derived Images

To create an Image from a checkpoint rather than a directory package:

```bash
uv run agent-libos --db user \
  images commit <checkpoint_id> stateful-agent:v0 --name stateful-agent
```

The artifact can capture reconstructable Object Memory, loaded Skills,
process-local JIT tools, exact tool visibility, cwd, and module prerequisites.
External capabilities become requirement declarations; filesystem/provider
state and external side effects are not packaged or rolled back. See
[Checkpoints: Commit To Image](checkpoints.md#commit-to-image).

## Author checklist

Before registering or distributing a package:

- keep `IMAGE.yaml` closed and every path package-relative;
- include only the exact prompt, resources, workspace seed, and JIT files needed;
- keep secrets and local metadata outside the package;
- choose a prompt mode deliberately and test its TaskRun behavior if applicable;
- list the exact static tools needed, including lifecycle/output tools;
- treat Capability requirements as declarations and review launch authority
  separately;
- scope `workspace.grants` only to the private materialized seed;
- pin required Runtime Modules by the verified source digest;
- run `images validate` for package/static checks, then boot in a controlled
  Runtime with Deno to exercise packaged JIT contract tests; and
- register under a new immutable image id unless replacement has been explicitly
  reviewed.

For a deliberately specialized one-tool package, see
[`mini-swe-agent` Image](mini_swe_agent_image.md).
