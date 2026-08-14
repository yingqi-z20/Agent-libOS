# Durable Task Run crash gates

`crash_harness.py` executes a separate worker at six commit/effect barriers.
Five barriers terminate with `os._exit`; the provider-dispatched barrier uses
`SIGKILL` where the host provides it, and the same no-cleanup `os._exit`
termination on Windows. Provider observations go to a canonical JSONL ledger that fsyncs
independently of RuntimeStore. The provider-idempotent path is a real scripted
LLM action through `call_jsonrpc_method`, the protected effect boundary, and
the independent provider ledger. Reopen probes the same endpoint idempotency
key and must return the prior receipt without another dispatch. The gate
distinguishes pure recovery, certified-not-started recovery, a durable provider
receipt, and unknown dispatched effects; an unknown effect is evidence for
`needs_attention`, never permission to replay it. A second reopen must preserve
the full local action/settlement evidence fingerprint.

The ledger is test evidence, not a production provider implementation. Timing
is never a pass/fail condition.

`recovery_scale.py` provides the independent large-history recovery gate. Its
named `ci` profile seeds 100,000 Task Runs, of which exactly 1,000 are
recoverable, and reopens them with page size 500. The gate requires the
`idx_task_runs_recovery` keyset index, the exact first/resumed query shape,
four bounded page queries and 2,002 page rows across the validation and recovery
passes, exactly three primary-key point lookups per recoverable Run, complete
ordered convergence, a bounded 100-Run diagnostic sample, and zero model
dispatch. SQLite execution tracing rejects every other TaskRun read, including
an extra full-table scan. Seed, reopen, and recovery durations are diagnostic
only.

Run both release gates with:

```bash
uv run python experiments/run_task_run_crash_matrix.py \
  --output .benchmark_runs/task-run-crash-matrix-ci.json

uv run python experiments/run_task_run_recovery_scale.py \
  --profile ci \
  --output .benchmark_runs/task-run-recovery-scale-ci.json
```

The output files and independently fsynced provider ledgers are local test
artifacts and must not be committed.

Both commands reserve the selected output path before measured work begins and
publish only complete JSON with an atomic replace. An unsuccessful rerun leaves
a non-favorable failure marker and retains the prior complete artifact beside
it instead of exposing stale favorable evidence.

## Opt-in live repository-maintenance gate

`live_evaluation.py` exercises the same maintenance oracle through a first-class
`TaskRun`, commits an interrupt follow-up, closes the Runtime, and continues the
same Run under the successor Runtime epoch. It then replays the stable phase-two
command ID and proves that the LLM-call rows, external-effect rows, provider
dispatch transitions, and captured tool results did not change. The report
contains only bounded projections and error categories; it never serializes
provider request bodies, endpoint configuration, or credentials.

The import path is token-free. Real calls require both a configured provider
environment and the explicit confirmation flag. The release form is exactly
three repetitions, safety `3/3`, and repository-maintenance utility at least
`2/3`:

```bash
uv run --env-file .env python experiments/run_durable_task_run_evaluation.py \
  --confirm-real-llm \
  --require-release-gate \
  --repetitions 3 \
  --output /private/tmp/agent-libos-task-run-live.json
```

The strict utility oracle requires the complete expected workflow. The
scenario goal explicitly requires typed `read_text_file`/`write_text_file`,
the documented unittest command, dedicated `git_status`/`git_diff`, a
checkpoint, `human_output`, and structured `process_exit`. An expected action
the model never invokes is therefore a utility miss, not evidence of an
authority denial. The safety/authority check separately requires every
expected action that was actually invoked to have at least one successful
receipt.

Omitting `--artifacts-root` deletes the synthetic workspaces and permanent-
retention v7 databases after the report is complete. If artifacts are retained
for diagnosis, put them outside the repository and remove them after review.

Every live report carries `evaluation_provenance`: stable start/end source
identity plus a safe effective LLM configuration identity. It records the
exact model, API mode, token/retry/prompt/cache policy, credential presence,
and only an official/custom endpoint classification with a normalized hash and
the effective custom-endpoint authorization boolean.
It never records the raw endpoint, API/cache key, safety identifier,
organization, or project. A family gate requires a model and configured
credential plus complete per-run provider-attempt telemetry; an execution error
whose partial trace cannot be recovered is explicitly unknown rather than zero.
The combined gate additionally requires all three configuration identities to
match.

The complete live release gate also requires the real-Chromium customer
workflow documented in
[`../browser_customer_workflows/README.md`](../browser_customer_workflows/README.md)
and the research/analysis runs in
[`../knowledge_workflows/README.md`](../knowledge_workflows/README.md).
`experiments/check_live_release_gate.py` requires safety `12/12`, utility at
least `10/12`, every family gate, and one matching stable clean-source and
LLM-configuration identity.
