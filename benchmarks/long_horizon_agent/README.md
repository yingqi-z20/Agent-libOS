# Long-horizon Agent evaluation

This opt-in evaluation runs a realistic repository-maintenance task that cannot
pass by emitting a plausible answer. The real model must inspect an unfamiliar
Git workspace, reproduce a test failure, edit production and regression code,
honor an untrusted prompt-injection string in an incident attachment as data,
handle a mid-task customer follow-up, survive a durable Runtime close/reopen,
run the full tests, inspect Git state and diff, create a checkpoint, deliver one
human-facing result, and exit.

The coding image uses a two-phase cumulative completion review. Its first exit
attempt is nonterminal and re-surfaces the original goal, acknowledged human
follow-ups, and observed successful tools. A confirmed exit needs a fresh
review token plus structured evidence covering every source. After a Runtime
reopen, the review reuses already-retained full-I/O LLM evidence; when the Host
disabled full-I/O retention and no checkpoint restored the goal payload, it
fails closed and asks for a restatement instead of guessing.

The report records durable state oracles, ordered tool/Skill use, failed and
invalid tool calls, successful-call rate, provider token usage, schema bytes,
restart survival, and follow-up constraint coverage. The oracle requires a
baseline command before the first edit and fresh test/Git/checkpoint/report/exit
evidence after the last edit. It parses executable regression-test definitions
instead of accepting comment markers, and independently probes the exact-price,
zero-quantity, Decimal-return, and public-signature behaviors. It never copies
`.env` values, model prompts, or provider responses into the report.

Run one paid trial explicitly:

```bash
uv run --env-file .env python experiments/run_long_horizon_evaluation.py \
  --confirm-real-llm \
  --require-all-successful \
  --output .benchmark_runs/long-horizon/report.json
```

Add `--artifacts-root .benchmark_runs/long-horizon/artifacts` to retain the
synthetic workspace and Runtime database when a failed run needs tool-argument
or audit diagnosis. The directory must be absent or empty.
