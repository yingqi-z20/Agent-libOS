# External-effect recovery scale benchmark

This benchmark populates a large file-backed SQLite history, closes the writer,
reopens the persisted runtime store, first probes the typed
`ExternalEffectRecoveryQuery`, and then assembles a real `Runtime` over that
store so `ProtectedOperationSDK.recover_prepared()` processes the backlog. It
fails on structural regressions: query work must remain proportional to pending
pages, raw rows may include only one look-ahead row per non-final page, initial
and resumed queries must use the matching recovery index, the handler must
leave no pending prepared rows, and Runtime diagnostics may retain at most one
page of effect IDs. A guarded legacy-list method also makes the benchmark fail
if startup falls back to loading complete external-effect history. Connection
tracing covers the read-only preflight connection, the main connection from its
creation through full schema initialization, and the complete Runtime handler
window, including statements issued through `_query`, `_execute`, the
connection, or a direct cursor. The trace is installed by the connection
factory, before an outer `connect` wrapper can issue prefix SQL. Exact
per-shape ledgers default-deny unexpected
SELECTs and DML; only the reviewed tuple-keyset page, effect-primary-key,
set-based stale-operation index, and identity-fenced prepared-effect deletion
shapes are recognized. The gate also verifies the exact recovery-index columns
and search constraints and uses SQL aggregation to prove every expected seeded
identity reaches its expected final classification without materializing the
history in Python.

Elapsed seed and recovery times are recorded for diagnostics only. They are
never pass/fail thresholds, so slow or noisy CI hosts do not create brittle
results.

## Profiles and automation

The entrypoint has two named profiles:

| Profile | Total history | Pending prepared rows | Page size | Automation |
| --- | ---: | ---: | ---: | --- |
| `ci` | 100,000 | 1,000 | 500 | Per-change release workflow |
| `million` | 1,000,000 | 10,000 | 500 | Scheduled/manual scale workflow |

Run the 100k-record CI profile:

```console
uv run python experiments/run_external_effect_recovery_scale.py --profile ci
```

Run the one-million-record manual/nightly profile:

```console
uv run python experiments/run_external_effect_recovery_scale.py --profile million
```

`--total-records`, `--pending-records`, and `--page-size` override the selected
profile. Values must be integers with `total-records > 0`,
`0 <= pending-records <= total-records`, and `page-size > 0`. A custom size is
not a named profile and should be reported with its explicit parameters.

## Output

By default the command writes
`.benchmark_runs/external-effect-recovery-scale.json`; use `--output PATH` to
select another file. The same JSON object is printed to stdout. It has
`schema_version: 3`, the selected population/page sizes, expected and observed
page/query/row/work-unit counts, startup and handler statement counts,
convergence counts, required index and query plans, timing fields, and
`timing_is_informational_only: true`. CLI artifacts also require an
`artifact_metadata` object with nested `schema_version: 1`:

- `benchmark_id`, an opaque `run_id`, and UTC `started_at`/`completed_at`
  identify one completed invocation;
- `invocation.selected_profile` records the CLI selection,
  `explicit_overrides` records exactly which size flags were supplied, and
  `effective_parameters` records the values actually passed to the runner;
- `invocation.classification` is `named-profile` only when no override flag was
  supplied. `named_profile_evidence` additionally requires the effective
  parameters to equal the selected profile defaults;
- `provenance.git` records commit, dirty state, and the working-tree digest
  when the guarded Git inspection is available. Benchmark source-file hashes
  and non-secret Python/OS/package versions are recorded alongside it.

The run id and hashes support identity and comparison; they are not a digital
signature or an attestation against a party that can rewrite the artifact. A
structural-contract failure raises and the command exits nonzero instead of
writing a favorable result.

See the repository [benchmark guide](../../docs/benchmark.md#recovery-scale-gates)
for how this gate relates to the runtime-safety and practical-workflow suites.
