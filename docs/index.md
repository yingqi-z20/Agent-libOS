# Agent libOS Documentation

This is the documentation home for the current checkout. Choose a path by what
you need to accomplish; subsystem reference pages are linked again in the
complete inventory below.

The project is experimental. Current behavior is defined by the code and the
current-contract documents beside it. Release readiness is conditional on the
exact source tree and its required CI evidence; a prose page is not a release
receipt.

## Start here

| Goal | Read first | Then use |
| --- | --- | --- |
| Run the token-free local demo | [README quick start](../README.md#3-to-5-minute-quick-start) | [Troubleshooting](troubleshooting.md) |
| Use the CLI | [CLI guide](cli.md) | [Machine-generated exhaustive CLI reference](cli_reference.md) |
| Configure the Runtime | [Configuration guide](configuration.md) | [Machine-generated exhaustive field reference](configuration_reference.md) |
| Embed the Runtime in Python | [Python API](python_api.md) | [Runtime model](runtime_model.md), [providers](providers.md) |
| Use the desktop console | [GUI first-task path](gui.md#first-task-user-path) | [GUI workspace reference](gui.md#current-workspace), [development and API boundary](gui.md#development) |
| Operate a persistent store | [Storage](storage.md) | [Configuration](configuration.md), [release/support status](support_matrix.md) |
| Understand the security model | [Threat model](threat_model.md) | [Capabilities](capabilities.md), [data flow](data_flow.md), [invariants](invariants.md) |
| Author an image, Skill, or module | [AgentImages](agent_images.md), [Skills](skills.md), [Runtime Modules](modules.md) | [Tools and JIT](tools_and_jit.md) |
| Add a protected provider operation | [Protected Operation SDK](protected_operation_sdk.md) | [Provider substrate](providers.md), [architecture](architecture.md) |
| Evaluate or contribute | [Development guide](development.md) | [Benchmark contract](benchmark.md), [contribution guide](../CONTRIBUTING.md) |
| Prepare a release | [Release status](release_status.md) | [Release runbook](releasing.md), [artifact anonymity](artifact_anonymity.md) |

If an unfamiliar project term blocks the path, use the [glossary](glossary.md).
If a command fails, start with [troubleshooting](troubleshooting.md) before
changing authority, retention, or provider settings.

## User and operator path

1. Run the [3 to 5 minute deterministic demo](../README.md#3-to-5-minute-quick-start).
2. Choose the [CLI guide](cli.md), [Python API](python_api.md), or
   [GUI first-task path](gui.md#first-task-user-path).
3. Read [configuration precedence and security-sensitive settings](configuration.md)
   before adding a persistent database, model profile, remote provider, or
   trusted module. Use the [generated field reference](configuration_reference.md)
   for the exhaustive path, type, default, and unit inventory.
4. Use [troubleshooting](troubleshooting.md) for installation, Deno, database,
   LLM, GUI, and benchmark failures.
5. Consult the [support matrix](support_matrix.md) before treating an OS,
   provider, or packaged desktop path as validated.

For long-running supervised work, continue with [Durable Task Runs](durable_task_runs.md).
For process state and context, use [Object Memory](object_memory.md),
[checkpoints](checkpoints.md), and the [Runtime model](runtime_model.md).

## Security reviewer path

Read these in order:

1. [Architecture](architecture.md) for the layer and composition boundaries.
2. [Threat model](threat_model.md) for assets, attackers, TCB, assumptions,
   guarantees, and non-goals.
3. [Capabilities](capabilities.md) and [Task Authority Manifests](task_authority_manifest.md)
   for operation authority.
4. [Data flow and trusted Sinks](data_flow.md) for information-flow authority.
5. [Protected Operation SDK](protected_operation_sdk.md), [events](events.md),
   and [Explainable Operations](explainable_operations.md) for effect and evidence
   settlement.
6. [Runtime invariants](invariants.md) and the [support matrix](support_matrix.md)
   for checked claims and remaining environment gates.

The optional semantic plane is described in
[Semantic Approval and Data Identification](semantic_shadow.md). Its historical
research proposal is not a current contract.

## Extension and integration path

- [AgentImage authoring](agent_images.md) defines package structure, prompt modes,
  declared requirements, workspace seeds, validation, registration, and boot.
- [Skills](skills.md) covers progressive-disclosure instruction/tool packages.
- [Tools and Deno/TypeScript JIT](tools_and_jit.md) covers ToolBroker, syscalls,
  validation, sandboxing, and visibility.
- [Runtime Modules](modules.md) covers trusted Python startup extensions.
- [Provider substrate](providers.md) and the
  [Protected Operation SDK](protected_operation_sdk.md) cover Host provider
  extension boundaries.
- [JSON-RPC](jsonrpc.md) and [MCP](mcp.md) cover registered remote clients.
- [Typed Git](git.md) covers the repository-bound provider and primitive.

None of these extension mechanisms grants process resource authority merely by
making an action visible. Use the [glossary](glossary.md) when the distinctions
between visibility, authority, effect, operation, and evidence are unclear.

## Contributor and release path

- [Development](development.md): setup, deterministic lanes, optional real
  integrations, documentation rules, and dependency changes.
- [Benchmark](benchmark.md): runtime-safety and practical-workflow contracts,
  runners, outputs, metrics, and publication rules.
- [Release status](release_status.md): current scope and required gates; not a
  CI receipt.
- [Release runbook](releasing.md): explicitly authorized artifact, tag, and
  publication workflow.
- [Artifact anonymity](artifact_anonymity.md): contributor-only identity/secret
  review for research or source artifacts.
- [Support matrix](support_matrix.md): declared support, per-change evidence,
  and environment gates.

## Complete current-contract inventory

### Concepts and runtime

- [Architecture](architecture.md)
- [Runtime model](runtime_model.md)
- [Glossary](glossary.md)
- [Durable Task Runs](durable_task_runs.md)
- [Object Memory](object_memory.md)
- [Checkpoints](checkpoints.md)
- [Runtime events](events.md)
- [Explainable Operations](explainable_operations.md)

### Authority, information flow, and evidence

- [Threat model](threat_model.md)
- [Capabilities](capabilities.md)
- [Task Authority Manifests](task_authority_manifest.md)
- [Data flow and trusted Sinks](data_flow.md)
- [Semantic approval and data identification](semantic_shadow.md)
- [Evidence payload retention](evidence_payload_retention.md)
- [Runtime invariants](invariants.md)

### Interfaces and operations

- [CLI guide](cli.md)
- [Machine-generated exhaustive CLI reference](cli_reference.md)
- [Python API](python_api.md)
- [Electron GUI](gui.md)
- [GUI API schema subset](gui_api_schema.json)
- [Configuration guide](configuration.md)
- [Machine-generated exhaustive configuration field reference](configuration_reference.md)
- [Storage](storage.md)
- [Troubleshooting](troubleshooting.md)
- [Support matrix](support_matrix.md)

### Extensibility and providers

- [AgentImages](agent_images.md)
- [`mini-swe-agent` image](mini_swe_agent_image.md)
- [Skills](skills.md)
- [Tools and Deno/TypeScript JIT](tools_and_jit.md)
- [Runtime Modules](modules.md)
- [Protected Operation SDK](protected_operation_sdk.md)
- [Provider substrate](providers.md)
- [Typed Git](git.md)
- [JSON-RPC](jsonrpc.md)
- [MCP client](mcp.md)

### Evaluation and maintenance

- [Benchmark contract](benchmark.md)
- [Development guide](development.md)
- [Release status](release_status.md)
- [Release runbook](releasing.md)
- [Artifact anonymity checklist](artifact_anonymity.md)
- [Research thesis](paper_thesis.md), which is framing rather than an API,
  security, support, or release contract

## Historical documents

The following files are retained for traceability and old-link migration. They
must not be used to infer current commands, interfaces, security guarantees, or
release status:

- [Historical design archive notice](../agent_libos_design_doc.md)
- [Historical paper roadmap notice](../plan.md)
- [Historical prelaunch review notice](prelaunch_hardening_report.md)
- [Commit-bound semantic permission/data-flow research](semantic_permission_and_dataflow_research.md)

Each historical document identifies its baseline or retirement status and
points back to maintained references. Git history is the source for a genuinely
commit-bound historical audit.

## Contract and version rules

- The code and current documents in one checkout belong together. Repository
  links are relative for that reason.
- The generated package-index README rewrites repository links to the immutable
  `v<version>` tag. A published version must never link its API or security
  contract to mutable `main`.
- The installed version's generated `agent-libos ... --help` output is the
  exhaustive CLI parser reference. The checked-in generated CLI reference must
  match it; [cli.md](cli.md) is the narrative workflow and safety guide.
- `agent_libos.config.DEFAULT_CONFIG` is authoritative for exact defaults;
  the [generated configuration field reference](configuration_reference.md)
  exhaustively inventories declared paths, types, defaults, and units, while
  [configuration.md](configuration.md) defines loading, precedence, validation,
  security handling, and subsystem semantics.
- Product and protocol/schema version numbers are independent. Use the
  [version map](glossary.md#version-map) before interpreting an unqualified
  `v1`, `v2`, `v3`, or `v7`.
- [Release status](release_status.md) describes the validation contract, not an
  observed pass. A pass claim requires the immutable evidence listed there.
