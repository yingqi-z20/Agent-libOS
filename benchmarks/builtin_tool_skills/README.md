# Built-in Tool Skill routing evaluation

This opt-in real-LLM evaluation compares the compact built-in Skill projection
against a paired no-Skills, full-schema baseline. Five held-out intents cover
workspace editing, read-only Git, shell execution, checkpoints, and MCP. Each
intent has a nearby but incorrect boundary and is repeated as exactly three
pairs to reduce single-sample variance. Both arms receive the same neutral goal;
the goal does not tell the treatment arm which Skill to activate.

The treatment image is derived from `coding-agent`, has all 99 catalog-owned
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
- initial, authorized-full, and cumulative tool-schema bytes;
- compact catalog metadata bytes and cumulative serialized prompt bytes;
- a clearly labeled `bytes / 4` schema-token estimate and provider-reported
  prompt tokens;
- treatment-minus-baseline deltas for success, routing, invalid calls, schema
  overhead, prompt bytes, and token measurements.

The estimate is for comparisons only; provider tokenization remains the source
of truth for billed prompt usage.

Run it only with explicit real-LLM credentials and confirmation:

```bash
.venv/bin/python experiments/run_builtin_tool_skill_evaluation.py \
  --confirm-real-llm \
  --require-all-correct \
  --output .benchmark_runs/builtin-tool-skills/report.json
```

`--require-all-correct` fails unless every treatment and baseline run returns a
successful probe result, passes its observable-state oracle, chooses the
correct route, and exits. Merely dispatching the expected tool and then calling
`process_exit` is not sufficient.

Preview the fixed three-pair plan without reading credentials or making provider
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
