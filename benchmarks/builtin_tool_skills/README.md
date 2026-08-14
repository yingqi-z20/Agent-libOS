# Built-in Tool Skill routing evaluation

This opt-in real-LLM evaluation compares the compact built-in Skill projection
against a paired no-Skills, full-schema baseline. Five held-out intents cover
workspace editing, read-only Git, shell execution, checkpoints, and MCP. Each
intent has a nearby but incorrect boundary and is repeated as exactly three
pairs to reduce single-sample variance. Both arms receive the same neutral task
goal; the goal does not mention Skills, an expected Skill id, or the exact probe
tool. The 15 pairs use a deterministic alternating AB/BA order, with treatment
first in eight pairs and baseline first in seven, so one arm does not always
benefit from running first.

The same opt-in pytest module also runs a complete 26-case activation catalog:
one source-neutral positive intent and at least one adjacent negative boundary
for every distributed built-in Skill. That catalog checks routing and
activation only. The five paired scenarios remain the deeper effect-verified
comparison and must not be described as exhaustive product coverage.

The treatment image is derived from `coding-agent`, has all 101 catalog-owned
tools in its authority ceiling, and starts with Skills-based projection. The
baseline exposes all non-lifecycle tools immediately and omits the four Skill
lifecycle tools, so it has neither a Skill catalog prompt nor activation path.
The baseline is local and deterministic apart from the model call; MCP tests
read only the cached Host registry and never require network access.

The JSON report records per-run and aggregate:

- treatment Skill activation and per-arm correct route selection;
- successful structured probe-tool results and scenario-specific outcome
  oracles (exact file bytes, Git status state, shell argv/exit/output,
  checkpoint readback, or MCP registry metadata shape);
- invalid model tool calls, measured from runtime action-repair evidence;
- a bounded exit-review trace containing only review status, token/evidence
  presence, validation errors, and exact unobserved Tool names;
- initial, authorized-full, and cumulative tool-schema bytes;
- compact catalog metadata bytes and cumulative serialized prompt bytes;
- a clearly labeled `bytes / 4` schema-token estimate and provider-reported
  prompt tokens;
- treatment-minus-baseline deltas for success, routing, invalid calls, schema
  overhead, prompt bytes, and token measurements.
- per-run `pair_id`, global `pair_index`, `pair_position`, explicit `pair_order`,
  `run_evidence_complete`, observed process status, `exited_via_tool`, logical
  LLM calls, provider attempts, input/output/cache tokens, and a hash of the
  identical neutral goal;
- one `metrics.paired.samples` row per pair containing both arms and direct
  treatment-minus-baseline inputs for paired confidence intervals and binary
  tests;
- a redacted `evaluation_provenance` envelope binding the report to one stable
  source tree and one safe provider/model/config identity. It never contains an
  API key or raw endpoint.

The estimate is for comparisons only; provider tokenization remains the source
of truth for billed prompt usage.

The report contract is `schema_version: 3`. With no `--scenario` selector, the
CLI evaluates all five held-out scenarios with the fixed three repetitions and
two arms: 15 pairs and 30 runs. Repeating `--scenario` selects a subset but does
not change the three repetitions. The separate 26-case activation catalog is a
pytest-only routing check; it is not included in this JSON report. The current
treatment authority ceiling contains 101 unique catalog-owned tools; this is an
inventory fact for this source version, not a stable schema field.

The fresh-state read oracles fail closed on incomplete evidence. Git must
return exactly the one untracked `tracked-intent.txt` fixture with normalized
status fields, `truncated: false`, and a 64-hex state token. MCP must return
exactly `servers: []` and `has_more: false`; merely returning a list shape is
not sufficient.

Run it only with explicit real-LLM credentials and confirmation:

```bash
.venv/bin/python experiments/run_builtin_tool_skill_evaluation.py \
  --confirm-real-llm \
  --require-all-correct \
  --require-publication-gate \
  --output .benchmark_runs/builtin-tool-skills/report.json
```

`--require-all-correct` fails unless every treatment and baseline run returns a
successful probe result, passes its observable-state oracle, chooses the
correct route, and exits. Merely dispatching the expected tool and then calling
`process_exit` is not sufficient. The flag is not enabled by default: without
it, a successfully written schema-v3 report exits 0 even when one or more
runs fail those correctness checks. CI and release gates must therefore pass
an explicit gate; provider/setup exceptions and artifact-write failures still
terminate nonzero independently of the flag.

`--require-publication-gate` validates whether the artifact is complete enough
to report, independently of whether the observed outcomes are favorable. It
requires the full 15-pair/30-run schema-v3 matrix, exact 8/7 counterbalance,
complete and decidable paired inputs and oracles, a clean and unchanged Git
identity, stable model/config identity, nonempty model and credential presence,
and provider-attempt evidence for every logical LLM call. Use it together with
`--require-all-correct` when correctness is also a release criterion. A selected
scenario subset and historical schema-v1/v2 reports can still be read and
summarized, but cannot pass the publication gate.

Publication readiness does not require a favorable terminal state: a bounded
run that exhausts its quanta may validly report `runnable` with complete oracle
and telemetry evidence. The validator recomputes `completed` from the observed
status, successful exit-tool receipt, and oracle result. An exit-tool receipt
counts only when the Tool result is
successful and its payload says `status: exited` and `terminal_committed: true`;
a successful completion-review response is nonterminal. Cache-write tokens are
provider-optional: they are a nonnegative integer only when every logical call
reports them, and otherwise are explicitly `null`; they are therefore excluded
from paired continuous deltas. All other provider-attempt, input/output,
cache-read, and cache-metric reported-call counts must cover every logical call.

The exit-review trace deliberately omits review tokens, prompts, completion
evidence payloads, goal text, provider responses, and raw Tool results. It is
diagnostic evidence for a failed terminal sequence, not a copy of model I/O.

The output path is reserved before paid work begins and the completed JSON is
published with an atomic replace. A failed rerun leaves a small non-favorable
failure marker at the requested path and retains the previous complete report
beside it for recovery.

Preview the fixed 15-pair plan without reading credentials or making provider
calls:

```bash
.venv/bin/python experiments/run_builtin_tool_skill_evaluation.py --dry-run
```

The corresponding pytest is marked `real_llm`, so ordinary CI collects but
skips it. To run it explicitly:

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/test_builtin_tool_skill_evaluation.py \
  --run-real-llm
```
