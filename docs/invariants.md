# Agent libOS Runtime Invariants

The machine-checked runtime invariant map lives in
`tests/invariants.yaml`. It is the authoritative source for connecting safety
claims to pytest node ids and benchmark attack classes.

Validate it with:

```bash
uv run python scripts/check_test_invariants.py
```

The checker accepts JSON-subset or YAML syntax and fails when a listed pytest
node cannot be collected, an invariant lacks deterministic regression coverage,
an invariant's `benchmark_attack_classes` declaration diverges from the
top-level mapping, or a runtime-safety benchmark task uses an unmapped
`attack_class`.

## Current Invariant Groups

- `tool-visibility-is-not-authority`: visible tools and endpoints do not grant
  protected resource authority.
- `primitive-checks-before-effects`: primitives enforce capability, policy,
  approval, and validation before side effects, including PTY spawn and write
  limits.
- `capability-matching-and-delegation`: typed matching, deny dominance,
  one-shot grants, revocation, grant-as-transfer, parent-linked delegation
  attenuation, and ISO-normalized leases.
- `process-authority-is-explicit`: spawn, fork, exec, and cwd behavior do not
  imply broader authority.
- `object-memory-names-are-not-capabilities`: Object Memory names and
  namespaces do not bypass object capabilities.
- `object-memory-materialization-budget-is-authoritative`: Object Memory
  context materialization uses current payload token estimates.
- `child-memory-merge-lifecycle-is-explicit`: terminal child process memory
  remains mergeable, then is adopted or released by the parent lifecycle.
- `object-memory-lifecycle-is-explicit`: Object Memory ownership, release, and
  RAII cleanup are explicit and revoke stale authority, including Object-bound
  PTY handles.
- `human-approval-is-blocking-and-audited`: human questions and approvals block,
  resume, consume one-shot grants, are decided exactly once from pending state,
  and route through primitives.
- `shell-and-jit-containment`: shell and Deno JIT execution stay policy-bound,
  sandboxed, process-local, and syscall-mediated; PTY creation reuses shell
  authorization and follow-on PTY access uses Object capabilities.
- `command-risk-rules-are-deterministic`: command risk rules separate
  harmless, risky, and destructive shell operations without model judgment.
- `sandbox-profile-derived-from-capability-decision`: primitive sandbox
  profiles are derived from the same capability decision that authorizes the
  operation.
- `audit-query-windows-retain-latest-records`: limited audit queries select the
  latest matching records before returning them chronologically, and process
  audit views filter before applying their limit.
- `tool-observability-redacts-sensitive-payloads`: tool audit/event
  observability stores bounded preview, hash, size, and truncation metadata
  instead of raw sensitive args or results.
- `jit-security-does-not-rely-on-static-blacklist`: JIT safety is enforced by
  Deno no-permission isolation, libOS syscalls, capabilities, human approval,
  and budgets rather than dangerous API regex blacklists.
- `tool-policy-cannot-self-grant-authority`: ToolPolicy declarations cannot
  grant execution, resource authority, or confirmation.
- `tool-result-size-boundary-is-explicit`: tool result payload limits prevent
  unbounded result persistence while preserving committed side effects as
  explicit omitted-success results instead of retryable failures.
- `workflow-entry-uses-toolbroker-authority`: user-facing workflow entrypoints
  run tools through process tool tables, ToolBroker, result objects, and normal
  wait/exit/exec lifecycle semantics.
- `object-task-entry-uses-toolbroker-and-object-authority`: Object-bound
  background tasks run tools through ToolBroker, process tool tables, Object
  capabilities, owner-watch Object Memory primitive notifications, and
  process-message boundaries. Runner processes are host-managed and excluded
  from the LLM scheduler; terminal/cancel reconciliation must not leave active
  pins behind, and owner-watch resumes only replay tools with explicitly safe
  message-receive semantics.
- `llm-call-records-are-bounded-and-redacted`: LLM call persistence stores
  bounded preview, size, hash, and truncation metadata instead of raw prompts,
  tool arguments, reasoning, or provider responses.
- `llm-profile-selection-is-process-local`: host-selected LLM profiles are
  stored as process-local ids, resolved at LLM-call time, inherited by child
  processes, preserved by image-package defaults, isolated from non-default
  ambient provider environment, and fail closed when the id is unknown.
- `resource-budgets-are-hierarchical`: resource usage is charged to the acting
  process and its parent chain, and visibility/capability mechanisms cannot
  mint additional budget.
- `llm-token-usage-is-charged-before-tool-dispatch`: provider-reported LLM token
  usage is settled before any model-selected tool call is dispatched.
- `subprocess-resource-profiles-are-enforced`: shell and Deno subprocess wall,
  CPU, and RSS limits are enforced by providers and audited on exceedance; PTY
  providers also have deterministic fake-provider coverage and real backend
  smoke coverage where available.
- `skill-activation-does-not-grant-authority`: Skills change visibility and
  prompt context without granting resources; API actor mode must still honor
  skill capability or human-approval gates.
- `runtime-modules-load-trusted-code-atomically`: startup Runtime Modules bind
  trust to the current source hash, reject ambiguous manifests and duplicate
  module ids, resolve import strings without executing untrusted package code,
  bound source hashing, and roll back failed registrations so persisted module
  status stays aligned with loaded runtime state.
- `checkpoint-restore-and-fork-are-scoped`: checkpoint restore/fork are scoped,
  capability-controlled, and append-only outside reconstructable state.
- `image-self-evolution-requires-image-authority`: image registration, package
  boot, exec, and checkpoint commit require image authority and do not bake
  external authority.
- `jsonrpc-provider-effects-are-registered-and-classified`: JSON-RPC endpoint
  registration and calls use registered endpoint/method authority and
  classified provider effects.
- `runtime-safety-benchmark-is-deterministic`: benchmark tasks and smoke runs
  remain deterministic and token-free by default.

## Known Test Gaps

- Audit explain is not implemented yet; current tests check audit record
  emission and selected audit counts, not query/explanation completeness.
- The runtime-safety benchmark is an early deterministic workload, not a
  complete paper evaluation suite.
- Context materialization metadata is not complete enough to compute
  included/omitted/summarized/truncated object statistics for every LLM call.
- Real MCP, Git worktree, and mock PR providers are planned but not implemented.
