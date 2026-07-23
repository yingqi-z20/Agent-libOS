# Practical Workflow Evaluation

This suite separates runtime-backed evidence from design-only modeled evidence.
It is deterministic and token-free. The default catalog currently contains
three `native-live` connector scenarios and 80 `modeled` scenarios.

Run the default catalog with:

```bash
uv run python experiments/run_practical_evaluation.py \
  --output .benchmark_runs/practical/report.json
```

The command always writes the same JSON document to stdout. When `--output` is
provided, it also creates parent directories and writes that document, followed
by a newline, to the selected path.

## Evidence levels

- `native-live` requires one real ToolBroker action per declared semantic
  effect, a provider-state before/after oracle, one committed persisted external
  effect with an exactly matching provider receipt, and at least one
  Explain-resolvable operation link. Missing or mismatched evidence fails the
  scenario; there is no modeled fallback.
- `modeled` validates catalog structure and the exact declared utility,
  forbidden-effect, and design-only provenance claims. Utility requirements
  must name effects actually declared as modeled, the security oracle's
  forbidden list must exactly equal the effects declared denied, and the row
  must explicitly disclaim runtime evidence. It contributes no tool calls,
  external-effect ids, operations, or runtime coverage.

The stateful mail, CRM, and calendar provider used by `native-live` scenarios is
local deterministic evaluation infrastructure. It is not evidence of production
connector interoperability.

## Report schema v1

The machine-readable contract is
[`report.schema.json`](report.schema.json). `schema_version` is the report
schema version and is independent of the runtime-safety task/output schema.

Top-level fields are:

| Field | Meaning and unit |
| --- | --- |
| `schema_version` | Integer report version; currently `1`. |
| `results` | One result per selected scenario, in evaluation order. |
| `scenario_counts` | Scenario counts keyed by `native-live` and `modeled`. |
| `semantic_effect_counts` | Declared semantic-effect counts keyed by evidence level. These are not runtime effect-row counts. |
| `native_tool_calls` | Sum of ToolBroker calls attempted by native-live results. |
| `native_operations` | Sum of distinct operation ids resolved within each native-live result. |
| `modeled_fallback` | Native scenarios reclassified as modeled; schema v1 fixes this field to exactly `0`. |
| `native_live_ok` | `true` iff every selected native-live result passed its native evidence oracle. |
| `modeled_suite_ok` | `true` iff every selected modeled result passed its modeled oracle. |

Each `results[]` row contains:

- `scenario_id`: stable scenario identity;
- `evidence_level`: `native-live` or `modeled`;
- `ok`: that scenario's evidence-level-specific oracle result;
- `semantic_effects`, `tool_calls`, and `operations`: counts in their distinct
  units;
- `external_effect_ids` and `operation_ids`: runtime evidence identities; both
  are empty for modeled rows;
- `errors`: complete scenario diagnostics. An empty list means the row passed.

For programmatic calls with a custom scenario list, an empty evidence-level
partition has an `all(...)` result of `true`; consumers must inspect
`scenario_counts` when the presence of both evidence levels is required. The
default CLI catalog includes both partitions.

Schema-v1 consumers should ignore unknown additive fields. Removing a field,
changing its type or unit, changing the meaning of an evidence level, or
changing the pass/fail semantics requires a new `schema_version`. The checked-in
JSON Schema and this document must change with any report-contract change.

## Exit status

The practical CLI is a strict gate by default; it has no analogue of the
runtime-safety benchmark's optional `--require-all-passed` flag.

- `0`: the completed report has `native_live_ok: true`,
  `modeled_suite_ok: true`, and `modeled_fallback: 0`;
- `1` after a complete report is emitted: that report violates at least one of
  those three gate conditions; an uncaught setup, runtime, or output I/O error
  may also terminate Python with 1 before a complete report exists, so callers
  must check for the report before interpreting the code as an oracle failure;
- `2`: command-line argument parsing failed;
- other nonzero termination: the process did not complete the documented gate;
  a complete schema-valid report is required before interpreting any exit code
  as evaluation evidence.

## Comparing reports

Do not combine the four counting layers. Scenario rates use scenario rows,
semantic-effect totals use declarations, and tool/operation totals exist only
for native-live evidence. Modeled rows must never enter a native-runtime
denominator.

The JSON Schema enforces field types, the zero-fallback rule, per-row evidence
shape, and consistency between each boolean gate and failed rows. The CLI also
runs `validate_practical_report(...)` before writing output. That semantic
validator rejects duplicate scenario/evidence identities, mismatched aggregate
counts, incorrect gates, and passing native rows that lack the declared
one-tool-call/one-effect/operation evidence. Consumers accepting reports from
outside the CLI must run both the checked-in JSON Schema and the semantic
validator; JSON Schema cannot express all cross-row sums and uniqueness rules.

The v1 report records scenario ids and evidence levels but does not embed Git,
configuration, or environment provenance. Cross-run comparison therefore
requires external artifact provenance and an exact match of selected scenario
ids, evidence-level assignments, suite source, Runtime configuration, and
environment. The current `3 + 80` catalog size is a checked-in population, not
a schema guarantee.
