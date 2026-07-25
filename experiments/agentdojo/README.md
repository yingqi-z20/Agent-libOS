# AgentDojo native-semantics pilot

This isolated subproject evaluates Agent libOS with AgentDojo without adding
AgentDojo's broad SDK dependency graph to the repository's root lock.
It has its own frozen lock and declares Python 3.11–3.12; that narrower range is
the harness/dependency contract, not a reduction of the core package's Python
3.11–3.14 support. Run `uv run --frozen pytest -q` from this directory for its
deterministic harness tests. The root `scripts/test_matrix.py` lanes do not
collect this independent environment.

The harness fixes two paired arms:

- `upstream_control`: AgentDojo's native `FunctionsRuntime` and tool loop, using
  Agent libOS's `LLMClient` so model, endpoint, API mode, temperature, and tool
  schema serialization are shared.
- `libos_ambient`: wrappers over the same AgentDojo function contracts,
  executed through the Agent libOS scheduler and `ToolBroker`. All suite tools
  are ambiently available. This is an integration/behavior arm, not a claim
  that Capability, approval, IFC, or protected external effects are enforced.

The default ambient prompt envelope remains `minimal_runtime`, preserving the
configuration selected by the initial pilot. Commit `bac4764` deliberately
redefined `image_only`: it now sends the exact AgentDojo system message, the raw
user goal, and a durable native assistant/tool transcript. The earlier
`image_only` evidence used the removed Object Memory snapshot semantics and is
kept only as a historical baseline. A new four-suite direct acceptance run is
recorded in [the 2026-07-25 initial report](INITIAL_REPORT_2026-07-25.md). Use
`--libos-prompt-mode` to select these modes explicitly.

With `image_only`, both arms begin with system/user messages and continue with
native assistant/tool history; Capability, approval, IFC, audit, and external
effect controls remain outside the model-visible transcript. The
`minimal_runtime` and `libos_default` ablations still use Agent libOS runtime
prompt composition. Provider-request traces preserve these differences. The
trace captures LLMClient input before provider schema normalization; the
verifier separately checks paired tool-name sets and normalized chat schema
maps. Tool ordering can still differ and is reported as an observation.
Trajectory execution is also deterministic rather than counterbalanced:
`upstream_control` runs immediately before `libos_ambient` for each semantic
case and repetition. Pairing therefore aligns case inputs but does not eliminate
order effects from provider drift, caching, load, or other time-varying state.

AgentDojo's injection-task oracle returns `True` when the attack goal succeeds.
The report therefore names this field `attack_success`; it is the targeted ASR
numerator. `security_pass` is its negation. Provider/runtime errors are emitted
as `infra_error` and excluded from favorable denominators.

## Reproduce

From this directory:

```bash
uv sync --frozen
uv run --frozen agent-libos-dojo catalog
uv run --frozen agent-libos-dojo run \
  --output ../../.benchmark_runs/agentdojo/pilot \
  --dry-run
uv run --frozen agent-libos-dojo run \
  --output ../../.benchmark_runs/agentdojo/pilot \
  --confirm-real-llm \
  --fail-on-invalid
uv run --frozen agent-libos-dojo verify \
  --output ../../.benchmark_runs/agentdojo/pilot \
  --require-complete \
  --require-all-valid
```

The default pilot is 24 trajectories: four suites × three case modes (clean
user utility, attacked user utility/security, and injection goal as a direct
user request) × two arms. It uses one repetition, `injecagent`, temperature 0,
at most 16 provider calls per trajectory, at most 4096 output tokens per call,
an observed aggregate token stop at 20M, and the `minimal_runtime` ambient
prompt mode. The token stop is checked between trajectories, so a run can
overshoot by at most the final in-flight trajectory.

Real runs take OpenAI configuration from the explicit `--env-file` path. If an
ambient `OPENAI_*` value is absent from that file or differs from it, the run
fails before creating artifacts or constructing a provider client; an identical
ambient value is harmless. Before creating artifacts, the harness captures one
resolved in-memory client/redaction snapshot. Every trajectory clones that
snapshot, and a changed or missing dotenv file fails closed before the next
trajectory rather than silently mixing provider configurations. The harness
records the effective model name and a SHA-256 fingerprint of the endpoint, but
never the API key or raw endpoint.
Full synthetic AgentDojo messages and injection strings are stored under the
ignored `.benchmark_runs/` tree and should be handled as sensitive evaluation
evidence.

Each run contains:

- `metadata.json`: status, exact versions/configuration, a non-secret effective
  LLM-config digest, Git and lock provenance, and deterministic fingerprints of
  both this harness and the editable root `agent_libos` package (including
  package data and excluding bytecode caches);
- `results.jsonl`: one direction-explicit row per trajectory;
- `metrics.json`: utility, targeted ASR, safe-and-useful, direct injection
  solvability, invalid rate, repeated identical calls, paired arm disagreements,
  and token/time totals;
- `traces/*.json`: full AgentDojo injection and model/tool evidence;
- `runtimes/*/runtime.sqlite`: native Agent libOS evidence for ambient cases;
- `manifest.json`: row/trace counts and artifact hashes.

`agent-libos-dojo verify` recomputes metrics and hashes, checks row/trace and
paired-surface consistency (including realized provider API and compatibility
fallback parity), proves that the hidden terminal carrier stayed off the
provider surface, and scans all ordinary run files for the exact API key
and base URL from the selected dotenv file. Verification rejects symbolic
links, special files, files above 256 MiB, and artifact trees above 2 GiB before
parsing them. Source provenance likewise rejects symlinks rather than binding
only a mutable external target path. Completed runs must contain the
positive, unique planned-case manifest recorded in metadata. Strict verification
with `--require-complete` or `--require-all-valid` additionally requires every
semantic case to contain exactly one control and one ambient trajectory.
`--case-limit` must preserve complete groups of the selected arms, so a paired
run cannot be truncated after only one arm. Repeated suite, arm, mode, or task
selectors are rejected before planning. Use non-strict verification only for
an explicitly selected single-arm diagnostic run. The command exits nonzero on
a failed check.

The model naturally ends with assistant text, whereas the current Agent libOS
scheduler requires an action. `libos_ambient` therefore registers a runtime-only
terminal carrier. It is removed from every provider tool list and excluded from
tool-call metrics; deterministic tests verify both properties.
