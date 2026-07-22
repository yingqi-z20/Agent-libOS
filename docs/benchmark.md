# Runtime-Safety Benchmark

The M1 benchmark harness is a deterministic runtime-safety workload for
Agent libOS. It is designed to compare agent runtime boundaries against simpler
wrappers while avoiding default token spend. The suite now includes a
self-evolution subset for the paper theme: capability-controlled changes through
Skills, Deno/TypeScript JIT tools, image registration/exec/checkpoint commit,
child processes, checkpoints, Object Memory, and registered remote resources.

The task schema is defined in
[benchmarks/runtime_safety/schema.md](../benchmarks/runtime_safety/schema.md).

## Task Suite

The checked-in suite contains 32 schema-v1 YAML tasks under
`benchmarks/runtime_safety/tasks/`. They cover at least these classes:

- secret read attempts,
- forbidden filesystem writes,
- forbidden filesystem deletes,
- shell bypass and exfiltration attempts,
- data-label exfiltration attempts in which ordinary write authority exists
  but the destination Sink has insufficient clearance,
- object authority leakage,
- process authority leakage,
- self-evolution attempts involving Skills, JIT tools, image
  registration/exec/checkpoint commit, child processes, checkpoint fork, and
  JSON-RPC visibility;
- typed Git worktree containment, executable repository configuration,
  unauthorized remote use, and patch data-label lineage.

Each task declares the required workload and oracle fields, and may also
declare optional setup, authority, and policy inputs:

- `schema_version: 1`,
- a goal,
- a fixture workspace,
- attack class,
- allowed effects,
- forbidden effects,
- success oracle,
- safety oracle,
- optional initial capabilities and policy,
- deterministic `mock_actions`.

Fixtures live under `benchmarks/runtime_safety/fixtures/`. Runner workspaces are
copied to temporary output directories so checked-in fixtures are not mutated.

## Runners

Supported runner names and interventions are:

| Runner | Intervention |
| --- | --- |
| `direct_tool_wrapper` | Direct wrapper baseline on an isolated fixture copy. Supported filesystem and Object operations execute; unsupported boundaries are simulated. |
| `confirmation_wrapper` | Direct-wrapper variant that asks before every modeled side effect other than filesystem/Object reads, using the task's configured default decision. |
| `sandbox_only` | Static tool-category boundary: allows only fixture-contained filesystem CRUD, wrapper-local Object read/create/append, and process exit; all other modeled action categories are denied. |
| `agent_libos_full` | Full Agent libOS runtime boundary and evidence pipeline. |
| `no_primitive_approval` | Bypasses final capability `ASK` decisions, rule-driven Shell prompts backed by allowed policy authority, and mandatory Git approval bindings; missing authority, explicit deny, capability constraints, required Git rights, and data-flow checks remain active. |
| `no_audit_linkage` | Audit-linkage **observer** ablation described below. |
| `no_namespace_isolation` | Grants the target read/materialize access to all setup-seeded Objects and their namespaces; namespace enforcement is not disabled globally. |
| `no_fork_attenuation` | Benchmark-only child compiler copies every active current parent grant to each child instead of deriving attenuated authority. |

Wrapper and sandbox runners are baselines, not trusted security boundaries.
The direct and confirmation wrappers perform filesystem reads/writes/deletes
against each runner's copied fixture and perform Object reads/writes against
wrapper-local in-memory state. Shell execution and provider/self-evolution
actions the wrappers do not implement are simulated. `sandbox_only` instead
denies action categories outside its static filesystem/Object boundary. Every
normalized semantic effect is recorded with a distinct `performed`, `simulated`,
or `denied` outcome; denied actions without a modeled effect are listed in
`result.metadata.sandbox_denied_actions`.

`confirmation_wrapper` also records each prompt as a performed
`human.request(request_kind=approval)` effect. Because that prompt is the
runner's own intervention, it is treated as an allowed baseline effect unless
the task explicitly forbids it; tasks whose completion specifically depends on
approval must still list it in `expected_effects`.

Agent libOS runners execute through the runtime, using process capabilities,
primitive checks, human policy, audit records, and persisted LLM calls.
LLM action selection is also a persisted `external.provider_call` effect, so
the checked-in tasks explicitly allow `llm/complete`; this is not an implicit
oracle exception. A task that omits the entry reports the provider call as an
unknown effect.

`no_audit_linkage` does not pretend that the runtime stopped producing audit
rows. Its precise intervention is at the benchmark observer: audit rows are not
passed to normalized-effect reconstruction, audit completeness is reported as
zero, and the Explain summary is withheld. Persisted external-effect rows and
explicit runtime-result denials remain available because they are independent
evidence channels. An action that has no such evidence becomes missing/invalid;
the ablation never reconstructs it from the hidden audit log. Operational call
counters may still be measured internally, but they do not supply the safety
oracle. The exact intervention strings are emitted in run provenance and result
metadata so downstream tables cannot silently reinterpret the runner name.

## Running

Default exploratory smoke:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --limit 3 --output .benchmark_runs/m1-smoke
uv run python experiments/collect_metrics.py .benchmark_runs/m1-smoke
```

For a CI or release gate, require every selected task's declared success and
safety oracles to pass:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --limit 3 --require-all-passed --output .benchmark_runs/m1-gate
```

Without `--require-all-passed`, an oracle failure is preserved in the output
but does not by itself change the command's exit status; this default supports
counterfactual baseline collection. Runner failures and structurally invalid
evidence always return nonzero. The flag changes only the final exit gate, not
the generated results or metrics.

All rate-bearing runners (the observer-only audit ablation is excluded):

```bash
uv run python experiments/run_benchmark.py \
  --suite benchmarks/runtime_safety \
  --runner direct_tool_wrapper \
  --runner confirmation_wrapper \
  --runner sandbox_only \
  --runner agent_libos_full \
  --runner no_primitive_approval \
  --runner no_namespace_isolation \
  --runner no_fork_attenuation \
  --output .benchmark_runs/m1
uv run python experiments/collect_metrics.py .benchmark_runs/m1
```

`--runner all` additionally selects `no_audit_linkage`. That runner
intentionally withholds audit evidence, so tasks whose effects have no
independent external-effect or explicit-denial evidence become invalid. A
full current matrix therefore writes the diagnostic artifact and then returns
nonzero; standalone metric recollection returns 2 for the same invalid rows.
Use `--runner all` to study evidence loss, not as a green rate-comparison gate.

Select tasks:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --task fs_secret_read_001 --output .benchmark_runs/one
```

Select attack classes:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --attack-class shell_policy_bypass --output .benchmark_runs/shell
```

The `data_label_exfiltration` class deliberately gives the target process the
ordinary capability needed by the requested primitive. The full runtime must
still deny the action at the independent data-flow gate, while baseline
wrappers expose the counterfactual action. Its setup seeds a labeled Object in
the target LLM context and pins the benchmark LLM profile as a trusted Sink;
the attempted filesystem write remains an untrusted Sink.

Tasks are loaded in lexicographic filename order, filters preserve that order,
and `--limit` is applied last. Both `--limit` and `--max-quanta` require a
positive integer; invalid zero or negative values are rejected instead of
silently changing the selected workload.

## Real LLM Mode

The default mode is `--llm mock`. It uses task `mock_actions` and does not spend
tokens.

Real LLM mode is explicit and must be scoped to one task:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --llm real --limit 1 --output .benchmark_runs/real-smoke
```

Real mode must select exactly one task after filtering and supports only the
Agent libOS runner family (`agent_libos_full` and its four named ablation
runners). Wrapper runners, including a selection made with `--runner all`, are
rejected. Real mode uses `LLMClient.from_env()` and runtime `llm_calls`
persistence.

## Outputs

`run_benchmark.py` writes:

- `metadata.json`: selected suite, tasks, runners, LLM mode, process id, and
  CLI-run provenance.
- `results.jsonl`: one `BenchmarkResult` row per task/runner.
- `effects.jsonl`: one `EffectRecord` row per normalized observed, denied,
  simulated, not-started, or unknown effect.
- `summary.json`: result/effect/ok/safety counts, runner and task ids observed
  in the written result rows, plus `runner_failures` and `invalid_runs` counts.
  It is a descriptive summary, not the intended-workload manifest.
- `metrics.json`: aggregate metrics.
- `metrics.csv`: stable CSV metrics columns.

Agent libOS runner directories also include per-task runtime store databases
under the output directory.

An expected task or safety failure is represented in the result fields and does
not make the benchmark command itself fail. A benchmark infrastructure failure
(for example, runner setup raising unexpectedly) is marked with
`metadata.runner_failed`, is still written to the output files, and causes the
command to exit nonzero. The console summary caps the failure preview at 20
rows; complete per-run diagnostics remain in `results.jsonl`.

## Result Fields

`results.jsonl` rows include:

- `task_id`
- `runner`
- `attack_class`
- `ok`
- `task_success`
- `safety_passed`
- `unknown_effects`
- `forbidden_performed`
- `approval_count`
- `tool_calls`
- `primitive_calls`
- `llm_tokens`
- `wall_time_s`
- `audit_records`
- `audit_completeness`
- `valid`
- `invalid_reasons`
- `errors`
- `workspace`
- `metadata`, including `metadata.self_evolution_counts` for per-run
  self-evolution attempts.

`task_success` means that every declared `success_oracle` check passed. In an
attack/refusal scenario, an `expected_effects` check may establish only that the
planned attempt reached an explicit terminal outcome; it is not a general
utility score or proof that the free-form natural-language goal was completed.
`safety_passed` is evaluated separately.

Agent-libOS runner result metadata also includes an `explainability` object with
`operation_count`, `causal_root_count`, `evidence_complete_root_count`, and
`unknown_outcome_count` for operations created after task setup. These are
diagnostic provenance counts, not additional rate columns or a safety score.
The existing `audit_completeness` metric keeps its historical benchmark
definition and is not reinterpreted as semantic explanation quality.

Every generated `effects.jsonl` row has `effect_id`, `task_id`, `runner`,
`type`, `performed`, `denied`, `simulated`, `outcome`, and `evidence`.
Outcomes are `performed`, `denied`, `not_started`, `simulated`, or `unknown`.
`outcome` is authoritative for scoring; `performed`, `denied`, and `simulated`
are legacy compatibility flags. Definite outcomes must pass the collector's
flag-consistency checks. For `outcome: unknown`, those flags may preserve a
partial observation but cannot resolve the uncertainty and must not be scored;
the unknown outcome itself invalidates the run. The row also
contains nullable type-specific fields `path`, `argv`, `namespace`, `name`,
`skill_id`, `tool`, `image`, `checkpoint`, `resource`, `operation`, `endpoint`,
`method`, and `provider`, plus `error`, `classification`, and the `metadata`
mapping. A valid scored effect has classification `allowed` or `forbidden`.
See the [schema-v1 reference](../benchmarks/runtime_safety/schema.md#run-output-evidence)
for the full task and output contract.

Agent libOS runners use persisted runtime `external_effects` as primary provider
evidence and correlated audit records for internal runtime mutations. An exact
primitive denial may use `runtime_result_denial`. A successful/error tool result
without matching effect/audit evidence is `outcome: unknown`,
`evidence: missing`; `result.ok` alone never proves that an effect did or did not
happen. Direct/confirmation-wrapper filesystem and wrapper-local Object
operations are genuine baseline observations and use `outcome: performed`.
Unsupported Shell, provider, and self-evolution actions use
`outcome: simulated`; `sandbox_only` rejects those categories with
`outcome: denied`. Denied, not-started, and simulated attempts do not count as
performed unauthorized effects.

Protected provider operations—including filesystem, clock, Shell, Human, PTY,
live JSON-RPC/MCP, LLM, and typed Git provider phases—prepare an
external-effect row with `effect_state: pending` before dispatch and an unknown
outcome. A normal classification CASes that same `effect_id` to `finalized`; if
a post-provider sink crashes first, the benchmark still imports the intent as
`outcome: unknown` with runtime external-effect evidence. The run is invalidated
rather than silently scored as safe.

## Metrics

Stable metric columns are:

- `runner`
- `tasks`
- `task_success_rate`
- `safety_pass_rate`
- `unauthorized_side_effect_rate`
- `false_denial_rate`
- `approval_count`
- `tool_calls`
- `primitive_calls`
- `llm_tokens`
- `wall_time_s`
- `audit_completeness`
- `skill_activations`
- `jit_registrations`
- `image_commits`
- `image_registrations`
- `image_execs`
- `child_processes`
- `checkpoint_forks`
- `remote_calls`
- `unauthorized_side_effect_numerator`
- `unauthorized_side_effect_denominator`
- `false_denial_numerator`
- `false_denial_denominator`
- `valid`
- `invalid_reason_count`
- `unknown_classifications`
- `unknown_outcomes`
- `simulated_effects`
- `invalid_reasons`

The rate denominators are explicit in every row:

- `unauthorized_side_effect_rate` is forbidden performed effects divided by
  definitely performed effects. Denied, not-started, simulated, and unknown
  outcomes are excluded from that denominator. Exact counts are reported in
  the corresponding `unauthorized_side_effect_*` fields.
- `false_denial_rate` is allowed denied attempts divided only by allowed effect
  attempts with definite `performed` or `denied` outcomes. Forbidden, unknown,
  simulated, and not-started records are not part of this denominator. Exact
  counts are reported in the corresponding `false_denial_*` fields.

Metric rows are fail-closed. Duplicate/missing result task keys or effect ids,
orphan effects, invalid numeric/count fields, unknown classifications/outcomes,
missing evidence, inconsistent outcome flags, or runner infrastructure failure
set `valid: false`. Raw counts and invalid reasons remain available, but all
rate fields (including task/safety/audit rates) become `null`, and the benchmark
CLI exits nonzero. Invalid evidence is never silently folded into a favorable
rate.

Both entrypoints preserve that automation contract: `run_benchmark.py` exits
nonzero after writing an invalid run, and a later standalone
`collect_metrics.py <run-dir>` recomputation returns exit code 2 when
`valid: false`.

`metadata.json` is also a completion manifest. `run_benchmark.py` writes it
before runner execution; its non-empty, unique `tasks` and `runners` lists
therefore define the intended task×runner Cartesian product. Metrics are valid
only when every declared pair has exactly one result and no result appears
outside that matrix. Consequently an interrupted CLI run, truncated copy, or
missing runner cannot be reported as a favorable partial sample.

`write_run_outputs(...)` also supports direct programmatic/test callers. If no
metadata file exists, that helper writes a self-describing fallback derived
from the rows it was given. Such a post-hoc file cannot prove that an upstream
caller supplied every task it originally intended; callers that need
completion checking must write the intended metadata before execution, as the
benchmark CLI does.

CLI-created metadata also includes `provenance.schema_version: 1` with:

- Git commit, dirty state, and a hash of tracked changes plus untracked file
  content;
- each selected task-file hash and each selected fixture-tree hash;
- the serialized `DEFAULT_CONFIG` hash plus LLM mode and quantum bound;
- selected runner intervention text and ablation/runner/oracle/metrics support-source hash;
- Python implementation/version, OS release/architecture, dependency versions,
  deterministic-Deno mode, and only a boolean for real-LLM credential presence.

No credential value, hostname, or executable path is recorded. These fields let
an artifact consumer distinguish code/workload/config/environment snapshots;
the metrics collector currently checks matrix completeness, while release or
paper packaging should additionally recompute and compare provenance hashes.
Programmatic `write_run_outputs(...)` fallback metadata remains intentionally
post-hoc and is not a provenance attestation.

Do not mix the benchmark's counting layers when reporting results: `tasks` is
the number of result rows, the rate denominators above count different qualified
subsets of normalized effect records, and `tool_calls` / `primitive_calls` are
runner-reported execution trace counts. `metrics.json` records these units in
`count_units`.

Aggregation is per runner over all result and effect rows in the intended
matrix. `task_success_rate` and `safety_pass_rate` are micro-averages over
result rows; the two effect rates are ratios of the pooled qualified effect
records, not averages of per-task percentages. Execution counters and
`wall_time_s` are sums, while `audit_completeness` is the arithmetic mean over
result rows. A task with more effects can therefore contribute more weight to
an effect rate than a task with fewer effects.

Within one valid artifact, runner rows are comparable because metadata binds
them to the same task set. Across artifacts, compare rates only after matching
the task and fixture hashes, output/effect schema, runner intervention and
source hash, configuration hash, LLM mode, quantum bound, and relevant
environment provenance. The collector validates one artifact's completeness;
it does not assert cross-artifact comparability.

## Recovery scale gates

Two SQLite reopen benchmarks supplement the task×runner workload with
structural scale checks. They emit one machine-readable JSON object to stdout
and to `--output`; elapsed times are diagnostic fields, never pass/fail
thresholds.

The [external-effect recovery benchmark](../benchmarks/external_effect_recovery/README.md)
has two named profiles:

- `ci`: 100,000 total rows, 1,000 pending rows, page size 500. The per-change
  release workflow runs this profile.
- `million`: 1,000,000 total rows, 10,000 pending rows, page size 500. The
  scheduled/manual scale workflow runs this profile.

```bash
uv run python experiments/run_external_effect_recovery_scale.py \
  --profile ci \
  --output .benchmark_runs/external-effect-recovery-ci.json
```

The [runtime-publication reopen benchmark](../benchmarks/runtime_publication_recovery/README.md)
currently has one named profile, `ci`: 10,000 terminal publications, 1,001
unreconciled rows, and page size 500. Both the per-change release workflow and
the scheduled/manual scale workflow run that same 10k profile; there is no
named one-million-publication profile.

```bash
uv run python experiments/run_publication_reconciliation_scale.py \
  --profile ci \
  --output .benchmark_runs/publication-reconciliation-ci.json
```

Both entrypoints accept explicit size and page overrides for focused tests or
manual experiments. Such a custom invocation is not evidence that a named
profile or checked-in workflow was exercised.

## Historical Deterministic Validation Snapshot

The recorded run for clean, pre-consolidation source snapshot
`c03a4ec764e02bd4df59e2769edeb1278d5ea545` is
`.benchmark_runs/release-c03a4ec`. Its provenance says `dirty: false`,
`llm_mode: mock`, and `real_llm_credentials_present: false`. The run is valid
and reports 28/28 task success, 28/28 safety pass, 122 normalized effects,
zero unauthorized performed effects out of 97 definitely performed effects,
zero unknown outcomes/classifications, and zero allowed denials out of 97
allowed performed-or-denied attempts (`false_denial_rate = 0/97 = 0%`). It
also records 74 tool calls, 91 primitive calls, and 76 remote calls. The
`llm_tokens: 144` value is deterministic usage accounting from the planned
mock client, not a real provider request or token spend.

The artifact's `metadata.json` SHA-256 is
`7ef7b0054f1e4fbd2bcb9b33e803016e62010254a122dffa8c692f0837ba6b54`; its
recollected `metrics.json` SHA-256 is
`f6b3b0aa5e2a403c3ed0a7c848dcbccffa7faabe5eda7edf6cfe26ebccde53b6`.
This artifact does not validate the current working tree. History consolidation
was not a new benchmark run and did not by itself prove content identity. The
artifact is ignored and must be copied separately when publishing evidence.

This population includes explicitly declared LLM provider effects as well as
Human approval and the authorized attenuated child spawn; compare rates only
with snapshots using the same workload and effect model. The two 97-element
denominators are qualified effect populations, not the 122-record total or the
28 task rows. This supersedes the older 27-task `0/22` snapshot and the
incompatible `3/43 = 7.0%` wording whose denominator counted unrelated
normalized records.

The older cross-runner smoke over eight runners and three selected tasks is
historical and is not part of this commit-bound artifact. The 28-task run is an
implementation-validation snapshot, not a claim that the harness is a complete
paper evaluation. Consult [release_status.md](release_status.md) for current
readiness and [support_matrix.md](support_matrix.md) for environment gates;
historical snapshot provenance remains in this document.

## Practical workflow evidence levels

[`benchmarks/practical_agent_workflows/`](../benchmarks/practical_agent_workflows/README.md)
is the first mainline replacement for the branch-only practical evaluation.
Run it with:

```bash
uv run python experiments/run_practical_evaluation.py \
  --output .benchmark_runs/practical/report.json
```

The report keeps four counting layers separate: scenarios, semantic effects,
runtime tool calls, and explicit operations. `native-live` scenarios must map
every semantic effect to a real ToolBroker call, a stateful provider before/
after oracle, a persisted external effect, and an Explain-resolvable operation.
The native connector provider writes the actual semantic class and target into
its provider receipt, and the runner requires an exact per-effect match rather
than accepting equal counts as evidence of correspondence.
There is no fallback branch: absent native evidence fails the scenario and
`modeled_fallback` remains zero. Unsupported or research-only scenarios belong
to the separately counted `modeled` suite and never enter a native denominator.

The report has its own `schema_version: 1` contract. It includes one result per
scenario, evidence-level scenario and semantic-effect counts, native ToolBroker
call and operation totals, evidence ids, complete errors, and the three strict
gate fields `native_live_ok`, `modeled_suite_ok`, and `modeled_fallback`.
The CLI returns 0 only when both suites pass and fallback remains zero; it
writes and prints a completed failing report before returning 1. See the
[practical suite reference](../benchmarks/practical_agent_workflows/README.md)
and its
[JSON Schema](../benchmarks/practical_agent_workflows/report.schema.json) for
field units, versioning, exit codes, and cross-run comparability limits.

The initial connector provider covers stateful mail, CRM, and calendar writes
through registered JSON-RPC methods. It is deterministic test infrastructure,
not a new core primitive or a claim of production connector coverage. The
`eva` branch's 5-track x 8-family x 2-variant scenario design is rebuilt as 80
strictly `modeled` scenarios. Their utility/security oracles validate design
coverage only: they have no native actions, tool calls, operations, or runtime
coverage credit. This preserves the useful catalog without migrating the old
fallback runner or its ambiguous `modeled+live-runtime` label.

The runtime-safety benchmark remains suitable for deterministic smoke and early
evaluation. The practical suite establishes evidence-level accounting but is
not yet a complete paper evaluation; broader adversarial hosted-provider work
and an evaluation of explanation usefulness remain future work. The checked-in
Git tasks use deterministic local repositories and are evidence for the typed
Runtime boundary, not real GitHub/GitLab interoperability.
