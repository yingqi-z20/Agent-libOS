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
`attack_class`. It also fails when the invariant ids documented below contain a
duplicate, omit an id from the manifest, or retain an id that the manifest no
longer defines.

## Current Invariant Groups

- `tool-visibility-is-not-authority`: visible tools and endpoints do not grant
  protected resource authority.
- `primitive-checks-before-effects`: primitives enforce capability, policy,
  approval, and validation before side effects, including hidden provider
  metadata gates, filesystem mutation authority before target-state
  observation, descriptor-bound no-follow filesystem state snapshots,
  stale-size-safe bounded reads, PTY spawn cleanup, and write limits.
  Filesystem/clock/shell/PTY provider calls persist a pending `unknown`
  effect intent before the boundary, CAS the same id on final classification,
  and after attempting the call remove it without a final record only when the
  provider certifies `ProviderEffectNotStarted` and every completed earlier
  phase has no mutation or information flow and explicitly does not commit
  authority. Clock sleep/asleep inserts the intent before its first monotonic
  observation, treats elapsed-time measurement as information flow, and permits
  restore/abandon only when that first observation is certified not-started.
- `capability-matching-and-delegation`: typed matching and global precedence
  across every matching capability: `DENY` (including restrictive authority
  rules carried by otherwise-allow capabilities), then an exact matched
  one-shot human approval binding that safely resolves an ASK boundary, then
  `ASK`, then ordinary `ALLOW`. Stale, target-version-mismatched, or
  other-operation approval bindings never participate in that exception;
  one-shot grants, atomic default consumption, exact effect reservations,
  revoke-wins restoration, crash abandonment, grant-as-transfer,
  parent-linked delegation attenuation, restrictive parent boundaries,
  strict authority-rule decoding where only absent or null conditions become
  an empty mapping, malformed authority-rule fail-closed behavior, and
  ISO-normalized leases.
  Delegation publishes its row/process attachment/evidence atomically, and a
  multi-spec authority derivation is prevalidated and committed all-or-nothing.
- `capability-subject-isolation`: preselected capability candidates are filtered
  to the requested process subject, and Human-approved capability
  specifications cannot redirect authority to another process.
- `capability-list-api-pages-are-hard-bounded`: GUI and CLI capability lists
  reject non-positive, coercible, or over-configured limits. Include-inactive
  views push the bound into SQL before decoding; active-only views use a stable
  capability-id keyset, retain one bounded page at a time, and continue through
  expired or invalid parent chains until they collect the exact requested page
  or exhaust the source.
- `authority-mutations-revalidate-inside-one-transaction`: JSON-RPC, MCP,
  DataFlow, Skill registration/activation/unload/trust, capability issue/revoke, and
  checkpoint publication recompute the complete allow/deny decision after
  entering their UnitOfWork. Global Skill publication also rechecks the exact
  source/hash trust row in that boundary. Finite uses are reserved before
  mutation and settled with its evidence; JIT activation retires superseded
  executable handles only after settlement. Unlimited revocation, a newly
  inserted deny, or failed reservation settlement therefore has a documented
  serial order with the write on both SQLite and PostgreSQL.
- `process-authority-is-explicit`: spawn, fork, exec, and cwd behavior do not
  imply broader authority. Cwd selection requires filesystem directory read,
  and explicit child/PTY cwd probes occur only after their higher-level
  authority gates and under a filesystem effect intent. A spawn Authority Rule
  can bind the grant to one `image_id` and `process.spawn_child`; that grant
  cannot authorize a fork or another child image.
- `process-message-label-observation-is-linearizable`: observing durable
  message labels preserves a concurrent ACK or terminal process transition and
  creates at most one durable label carrier for a message.
- `filesystem-label-critical-sections-are-hierarchical`: overlapping path
  trees serialize fairly, including normalized aliases and missing-parent
  creation, while unrelated paths remain concurrent and widening reentrant
  lock upgrades fail.
- `filesystem-host-aliases-share-one-authority-and-label-identity`: host path
  aliases that name the same filesystem entry share capability and data-label
  identity, while genuinely distinct host names remain distinct. Manifest
  schema v2 records the exact Darwin and Linux nodes required for this claim;
  the configured native-platform CI matrix selects those markers separately
  and fails if any selected platform node skips.
- `task-authority-manifest-bounds-launch`: image requirements are declarations;
  Host manifest inputs are closed and strictly typed. Unknown fields, invalid
  scalar types, and malformed provider-effect wildcard forms fail closed before
  launch; valid manifests compile launch grants and bound model requests, child
  transitions, budgets, approval policy, and provider effect classes. GUI
  reviewed-command launch policy is explicitly opt-in, carries the constrained
  `allowlist_auto_else_ask` level, and still requires exact per-use approval for
  non-allowlisted commands; the disabled default denies before prompting.
- `effect-ceilings-distinguish-unrestricted-and-deny-all`: omission of a Task
  Authority effect ceiling remains unrestricted, while an explicit empty
  ceiling is a versioned deny-all value that cannot be downgraded after reopen.
- `effect-transactions-are-idempotent-and-reconcilable`: provider intents bind
  canonical arguments and idempotency keys, approval leases bind exact effects,
  and startup reconciliation queries but never replays providers.
- `authoritative-effect-recovery-is-provider-bound-and-atomic`: Host recovery
  options bind one effect, expected transaction state, and Runtime epoch;
  registered-provider verification is required before a receipt can settle it.
  Certified non-dispatch atomically restores reserved authority, releases the
  matching resource envelope, and appends the effect transition and audit;
  prepared effects accept no other terminal recovery conclusion.
- `external-effect-recovery-is-keyset-bounded`: startup recovery scans only
  nonterminal external effects through bounded indexed keyset pages and
  converges the full backlog without materializing the full effect history.
- `protected-provider-operations-use-sdk`: LLM, filesystem, Git, clock, shell,
  JSON-RPC, MCP, Human, and PTY provider effects share one contract registry and
  one prepare/dispatch/finalize state machine. Static coverage rejects direct
  low-level lifecycle and out-of-phase provider calls, while generic tests
  cover sync/async phases, restart recovery of prepared local state and finite
  authority, not-started restoration, partial effects, unknown outcomes,
  conservative classification, at-most-once settlement, evidence, and resource
  charge order. Required-resource failure paths settle measured partial usage
  or conservatively charge their preflight envelope; static coverage follows
  provider-reaching helpers and session handles at each call site. Every
  egress contract declares a direction and concrete Sink/source/payload/
  operation descriptors. Egress source, trust, target-state, payload, and exact
  release bindings are revalidated before every provider phase, including
  multi-phase state/resolve-to-write transitions.
- `protected-egress-enforces-minimum-integrity`: an egress or bidirectional
  protected-operation contract may declare a minimum source integrity. Lower
  integrity is rejected before provider dispatch, cannot be overridden by a
  sensitivity release, and the Host-declared floor remains in final effect
  evidence. This is an opt-in containment primitive; the permissive default
  preserves existing contracts.
- `provider-usage-reservations-fail-closed`: MCP uses one absolute deadline
  across DNS, executable snapshot, live listing, validation, and call dispatch.
  An exhausted deadline cannot start a provider; known response bytes settle
  exactly, an unknown host failure charges the current phase maximum, and a
  later phase that never started charges zero rather than the full composite
  reservation. Provider exceptions cross public, Tool, syscall, LLM, and
  evidence surfaces only as a code/type/correlation envelope without host
  exception text.
- `provider-results-are-decoded-at-the-host-boundary`: MCP and JSON-RPC provider
  results are detached and validated before runtime field access; malformed or
  unknown post-return failures expose only public envelopes, and unknown
  response bytes settle at the active-stage ceiling.
- `provider-approval-is-bound-to-versioned-spec`: JSON-RPC and MCP approvals
  bind an immutable registry-specification digest and monotonic generation,
  including absent first-registration state, and are revalidated before every
  provider phase and after reopen.
- `data-labels-propagate-conservatively`: derived Object sensitivity, trust,
  and integrity labels merge conservatively; manifests expose metadata only;
  label downgrade requires declassification authority.
- `data-labels-constrain-runtime-mediated-egress`: LLM/Human/JSON-RPC/MCP/file/
  Shell/PTY payload exits require Host Sink clearance in addition to ordinary
  authority. Conditional high data uses an exact one-shot release; source and
  trust changes and revoked reusable authority fail before provider/DNS/state/
  spawn. Same-runtime Object release invalidates ordinary egress; only
  recovery-marked durable Human/LLM actions may resume from their exact stored
  source snapshot. Persisted file labels instead expose an opaque immutable-binding
  source reference, so observed content remains usable after reopen without
  reviving released Object payloads, while a missing binding or mismatched
  generation/content hash fails closed. Labels survive LLM context, successful and failed synchronous
  tool threads (including explicit result carriers), output validation, async primitive worker handoff,
  timeout-managed async tool tasks, Object-derived tool metadata, JIT
  read/create/list/append/error/timeout, exact LLM-release resume, live MCP
  discovery, PTY session writes and public control operations, auto-created
  filesystem parents, atomic child-name/label publication for directory
  listings, recursive and non-recursive directory deletion with
  binding-level CAS that preserves a post-dispatch replacement, and filesystem
  reads bound to one label generation,
  Object/file conversion, ambiguous file writes, process goal/message/result,
  ObjectTask, fork/exec, and reopen paths. Mutable Shell, PTY, and MCP stdio
  executables dispatch through Host-owned content snapshots after final Sink
  validation; a bounded all-or-nothing direct-sibling compatibility view remains
  reachable beside the pinned executable without becoming part of its content
  identity or package attestation. Rejected exact LLM
  releases require explicit Host resume and cannot be regenerated by a model
  parent signal. Ambiguous Human provider outcomes are not automatically replayed
  and install the same explicit Host-resume gate;
  GUI release binds the complete public request view including `decision`, and
  interactive replies apply only to the exact Human request already shown to
  the operator. The CLI also retains an exact conditional-release request ID, so
  bounded pending-request windows cannot starve its prerequisite. Unchanged
  unrestricted GUI views reuse receipts only within the
  same provider session after a linearized current-policy check, and internal
  presentation evidence cannot starve bounded causal snapshot windows.
  Successfully delivered Human outputs are private-digest-bound frozen payloads:
  later mutable-source versions do not hide fixed bytes, while digest mismatch
  or current Sink-clearance failure remains withheld. Current
  file-tree labels use bounded keyset batches and a bytewise-collated exact
  prefix range, so backend locale and wildcard path characters cannot widen a
  subtree query. The
  `data_label_exfiltration` benchmark proves that ordinary write capability
  alone cannot export secret context.
- `builtin-tool-skills-do-not-expand-image-authority`: built-in tool Skills
  reveal only bindings already owned by the image and cannot add callable
  tools, capabilities, or primitive authority.
- `cumulative-completion-review-is-fresh-and-authority-neutral`: coding-agent
  completion review re-surfaces the cumulative goal and acknowledged Human
  follow-ups without expanding effective ambient authority beyond the exact
  immutable review evidence it creates. Final review and process exit
  linearize against concurrent Human follow-ups; message bodies and recovered
  prompt evidence are bounded to the exact reviewed sources. Review tokens
  become stale when Human messages change, and restart recovery fails closed
  when no retained goal evidence is available.
- `object-memory-names-are-not-capabilities`: Object Memory names and
  namespaces do not bypass object capabilities. Successful namespace listing
  consumes every finite namespace/object visibility decision used in the
  returned result in the same transaction as its audit. Namespace discovery
  and non-exact queries use payload-free Object keyset pages and bounded direct
  child-namespace keyset pages under an independent scan ceiling; only
  authorized text candidates or returned list entries load payloads. Reaching
  the ceiling before a complete result fails explicitly and rolls back derived
  handles and finite-use consumption.
- `object-memory-materialization-budget-is-authoritative`: Object Memory
  context materialization is bounded by final rendered object text, not trusted
  metadata token estimates.
- `context-compaction-preserves-authority-and-fails-closed`: context compaction
  uses child-process summarizers without granting external authority, validates
  summaries, preserves process authority, and fails closed on races or invalid
  output. The default source-only path creates no persistent delta context;
  enrichment requires explicit Host configuration or process authority.
  Explicitly authorized persisted LLM-context storage pressure starts
  compaction before the Object Memory hard limit; without the independent
  maintenance authority it cannot elevate the process. Synchronous or resumed
  failure terminates without recursively retrying the same oversized
  generation. A failed compaction job stores one correlated public envelope
  plus a private type/byte-count/SHA-256 observation, never Host, Provider, OS,
  validation, or authority exception text. Interrupt and cancellation signals
  retain their control-flow exception type after the job is durably failed;
  they cannot become a successful tool result.
- `child-memory-merge-lifecycle-is-explicit`: terminal child process memory
  remains mergeable only by a live direct parent. Handle grants, parent roots,
  ownership adoption/release, child-view consumption, and audit evidence commit
  atomically; the consumed terminal view is a durable idempotency marker, so
  concurrent or replayed merges have one winner and no duplicate authority.
- `object-memory-lifecycle-is-explicit`: Object Memory ownership, release, and
  RAII cleanup are explicit and revoke stale authority, including Object-bound
  PTY handles. Lifecycle mutations serialize ownership-lock before store
  transaction and use LIVE, owner, and version conditional writes so release
  races, concurrent updates, and owner ABA fail closed. Trusted delete rolls
  back Object release, capability revocation, audit, and in-memory payload
  together; a multi-Object ownership transfer and its audit are all-or-nothing.
  Forked MemoryView roots are prevalidated and their child grants, finite source
  uses, and audit commit in one ownership-lock/transaction boundary. Mutable
  Object metadata is structurally revalidated and bounded by configured field,
  collection, item, and canonical-byte limits before any Object side effect.
  Namespace metadata is strict, finite, cycle-free bounded JSON under the same
  byte ceiling and is rejected before namespace, capability, audit, event, or
  cache mutation across Host, tool, and syscall entrypoints.
  Link authorization, both Objects' LIVE checks, finite-use consumption, and
  evidence share that same ownership-lock/transaction linearization point.
- `root-initial-goal-recovery-is-exact-and-redacted`: the dedicated launch-goal
  recovery path applies only to an immutable initial goal from a committed root
  spawn when full-I/O retention is enabled. Its bounded envelope
  binds the publication's root launch, initial image, process creation identity,
  goal id, complete Object identity/version, canonical payload size, and hashes.
  Startup rehydrates only a matching active LIVE runtime-memory marker before
  the generic volatile-payload sweep; child/fork goals, exec replacement goals,
  mutable goals, mismatches, and tampering fail closed without provider
  dispatch or Object-row mutation. Ordinary publication reads expose hash-only
  projections. Write-time opt-out, a later policy opt-out, terminal cleanup,
  and launch rollback/compensation remove reversible content through fenced CAS;
  transition into a non-committable launch state cannot commit ahead of that
  redaction.
- `runtime-store-single-active-writer`: a writable persistent runtime store has
  at most one active Runtime per actual database target. SQLite validates
  owner-only, single-link database/lease/journal/WAL/SHM files, holds adjacent
  path and inode-keyed identity leases where supported, and holds an exclusive
  lock on the database actually opened; an existing path that disappears during
  preflight is never recreated. Pre/post identity checks reject ordinary path
  aliases and replacement races, while same-UID mutation of the owning directory
  remains a trusted Host boundary and the live path/parent must not be renamed or
  replaced. PostgreSQL keys advisory leases by exact database/schema.
- `storage-transactions-recover-or-fail-closed`: commit/savepoint finalization
  failure restores SQL and lazily journaled per-Object payload state when the
  transaction is definitely rollbackable. A commit diagnostic after the driver
  stops reporting an active transaction is outcome-uncertain and poisons/closes
  the store without manufacturing a mixed SQL/payload snapshot; rollback failure
  likewise poisons/closes the store. Poison errors expose only stable reason
  codes; driver-authored exception text is retained only as in-memory type,
  byte-count, and SHA-256 diagnostics. Direct payload-cache replacement first
  conditionally updates exactly one live Object row, so missing or released
  identifiers cannot create cache-only payloads or resurrect released state.
- `runtime-domain-storage-uses-exact-typed-facades`: Runtime, process, syscall,
  data-flow, and Tool orchestration use exact typed storage facades; persisted
  Object security projections are validated without materializing runtime-only
  payloads.
- `condition-owned-process-waits-preserve-owner`: generic Human pause/resume
  control fails closed for child/message, Human, and ObjectTask condition waits,
  and a different condition domain cannot replace an active wait. Paused and
  Host-resume gates likewise cannot be overwritten by condition registration.
  An ObjectTask runner may enter `running` only while its durable task is
  `queued` with the exact task-owned `ToolProcessWait`, or while its durable task
  still records the matching Human/child/message wait after that owner has
  released the runner to an unclaimed `runnable` state. It cannot trust a caller
  snapshot, steal an execution lease, or clear a still-owned condition wait or
  pause gate.
  Child, message, and Tool owners may renew only an exactly equal wait through
  revision/generation CAS; Human request ids may change only monotonically by
  addition or decision/removal. The owner may wake through its domain-specific
  fence. Pause/Host-resume waits cannot be minted into a condition wake token;
  ordinary paused and compatibility-suspended states remain resumable only by
  their trusted control paths.
- `process-waits-and-outcomes-are-typed-and-generation-fenced`: every semantic
  process transition atomically persists status, typed wait/outcome, and a
  monotonic state generation. Wakeups compare the exact typed state and
  generation, checkpoint restore reserves a new high-water generation, and
  fork remaps typed PID/Object references. `status_message` is compatibility
  output only and is never a runtime control protocol. Process list/wait
  boundaries expose the tagged values and generation directly. Exec rejects an
  active typed wait before creating its publication because no exec transaction
  owns the child, mailbox, Human, Tool, or Host-resume dependency that would
  otherwise be orphaned. Generic store patches and whole-process updates cannot
  write semantic state fields. Normal orchestration uses one transition service;
  explicitly typed execution/restore repository CAS primitives are the only
  exceptions when the state and its concurrency fence require the same SQL
  commit point. Exec-epoch commit requires the exact non-null admission token
  recorded by the matching applying `process_exec` publication at its final
  pre-commit phase and CASes RUNNING status, generation, owner, and lease. These
  typed boundaries compute the next state generation, preventing a direct-write
  rewind from reviving a stale token.
- `v4-persisted-state-is-strict-and-versioned`: a 1.2.1 store accepts only the
  frozen version-4 physical schema (including Durable Task Run and typed process
  state) and canonical security carriers. Schema-v3, older, incomplete, or
  malformed state is rejected before mutation; no compatibility path is
  represented as a migration. Recovery operates only on valid schema-v4 state.
- `durable-task-run-ledger-is-versioned-and-generation-fenced`: mutable Run
  projections advance through revision CAS under the current monotonic Runtime
  epoch. Stable command identities make retries idempotent, while requirements,
  links, and ledger history append rather than being rewritten. Recovery uses a
  bounded `(created_at, run_id)` keyset index and stale owners cannot claim,
  spawn, or commit work. Pause and interrupt persist a new control generation
  before blocking provider/tool admission, drain only calls admitted by the
  prior generation through local settlement, and never take over a live
  execution lease; interrupt requirements and their ProcessMessage commit in
  the same transaction.
- `durable-task-run-local-resume-is-integrity-bound`: a safe resume point exists
  only after one complete model action and all paired tool results are locally
  committed. Its canonical payload hash binds the Run, process revision,
  context generation, epoch, Image, tool and provider identities, and latest
  effect sequence; missing, malformed, or drifting bindings fail closed.
- `durable-task-run-never-replays-unknown-effects`: recovery distinguishes
  locally pure work, authoritative non-dispatch, already-durable provider
  success, and dispatched/unknown outcomes. Only the first three evidence
  classes may converge automatically; unknown provider truth blocks the Run
  without another downstream dispatch.
- `durable-task-run-payload-retention-is-explicit-and-complete`: durable
  plaintext is disabled by default and requires explicit Host configuration.
  Default terminal finalization hash-reduces Run-owned goal, follow-up,
  transcript, retained model/tool I/O content, and linked terminal Human request
  payload/decision bodies, then deletes resume points, pending continuation
  actions, and messages automatically bound from a Run-member recipient; an
  ordinary sender cannot suppress or forge that binding. Append-only event and
  audit rows receive only separate subject/body/payload hashes, labels, and
  identifiers for those bound messages at admission time, while their readable
  mailbox row remains available until purge; ordinary non-Run message evidence
  is unchanged. Human identity, type, status, timestamps, hashes, and audit
  linkage remain without readable prompt/answer/decision content. It also
  hash-reduces linked external-effect
  provider metadata and receipt bodies; hashes, labels, usage, effect state,
  links, and audit projections remain.
  `permanent` is an explicit Host/admin choice and purge failure prevents
  terminalization.
- `durable-task-run-cancellation-never-fakes-effect-settlement`: pause,
  cancellation, and deadline intent are persisted before new dispatch stops.
  Descendants converge before the root, but dispatched or unknown effects must
  still settle; ambiguity moves the Run to `needs_attention` and can never be
  presented as `cancelled`.
- `checkpoint-restore-never-erases-durable-task-effect-history`: Task Run
  payloads, ledgers, durable messages, and pending actions are outside
  Checkpoint/Image snapshots. Restore refuses active intersections; fork strips
  historical Run bindings and goal markers and cannot attach an unbound child
  below an active Run member. Neither operation rewinds append-only Run links or
  external-effect history, and a terminal purge cannot be reversed through a
  retained checkpoint artifact.
- `human-approval-is-blocking-and-audited`: human questions and approvals block,
  resume, reserve and consume one-shot grants exactly once, are decided exactly
  once from pending state, and route through primitives. Concurrent terminal
  drains serialize request selection through the terminal transition, so only
  the winning worker may install an automatic permission policy or cross the
  human output provider boundary. Permission policy and question-answer types
  are explicit, run-local `ContextVar` policy cannot cross concurrent runs,
  multiple blockers remain waiting, and terminal processes cancel requests.
  Blocking terminal provider I/O runs outside the selection lock so exit/cancel
  never waits for human input, and GUI history bounds never omit pending rows.
  Human output commits its delivered marker and pending intent before the
  provider; event/audit/effect finalization follows provider success. A later
  settlement failure preserves the dispatched pending intent and never replays
  the Human sink. Later GUI presentation verifies the delivered message digest
  and current labels/Sink policy without treating ordinary post-output context
  advancement as a source-freshness failure. Classifier failure uses the
  conservative contract ceiling.
  Terminal prompt reads and automatic-response writes also use structured
  pending intents; they retain only length/hash observations, never raw
  prompt, answer, or provider exception text. Human output provider failures
  likewise retain only the error type.
- `human-authority-and-evidence-commit-atomically`: Human requests, one-shot
  authority reservations, operation links, events, and audit commit or roll
  back as one unit, so an evidence-sink failure cannot publish a partial Human
  authority transition.
- `ambiguous-human-provider-outcomes-require-host-resume`: once a Human
  provider phase may have started but its outcome is unknown, the exact request
  becomes terminal and non-retryable, the effect ledger retains the
  uncertainty, and the process receives a durable Host-resume gate. A model
  parent cannot clear that gate, fallback reconciliation preserves it, and
  additional Human waits cannot downgrade it to an ordinary runnable state.
- `human-response-payloads-are-bounded-before-side-effects`: Human decisions
  and provider answers are bounded by canonical byte size, JSON depth, and node
  count before request, process, or capability decision state can change.
  Direct invalid decisions publish no event or audit. A terminal-provider read
  is protected and evidenced before its result can be validated; an invalid
  answer contributes only a bounded rejection marker to that evidence and
  leaves the pending request retryable.
- `jit-syscall-arguments-are-exactly-typed-and-bounded`: every field consumed
  by a common-contract built-in JIT syscall is validated against its exact JSON
  type and configured hard bounds before primitives, deferred lifecycle state,
  mailbox acknowledgement, destructive/replacement semantics, or success
  audit can occur. Invalid calls remain charged and request-audited, canonical
  aliases share the same contract, and finite integer or floating-point JSON
  durations retain their documented numeric behavior. Capability delegation
  retains its capability-domain validation.
- `builtin-tool-arguments-are-exactly-typed`: model-facing built-in Tool calls
  validate both JSON strings and mappings in Pydantic strict mode before tool
  execution. Numeric or string lookalikes therefore cannot enable recursive
  deletion, preserve authority across exec, inherit parent memory into a child,
  or select destructive Git mutation controls; correctly typed JSON values
  retain their documented behavior.
- `shell-and-jit-containment`: native Shell execution is argv-constrained and
  remains behind Shell policy, capability/effect checks, and finite-use leases;
  destructive built-in denial remains absolute, while matching custom denials
  override built-in ask/allow classifications. Custom argv rules compare the
  full conservative cross-platform executable identity and combine argv,
  regex, cwd, timeout, and direct operation conditions conjunctively against a
  complete primitive operation context; missing required context cannot
  auto-allow execution. It is a Host subprocess boundary, not the Deno JIT
  sandbox. Deno JIT
  candidates are separately sandbox-validated, process-local, cached-only at
  runtime, and mediated through primitive syscalls. JIT lifecycle
  rows/aliases/handles commit atomically, composite failures discard unpublished
  candidates, and cancellation terminates the isolated Deno process group or a
  verified discovered-tree fallback while incomplete cleanup fails closed;
  Host provider-error attribution requires both runner-private syscall-error
  provenance and a per-execution protocol proof, so candidate-authored error
  metadata remains an ordinary sandbox failure;
  a dedicated POSIX death-pipe/process-group supervisor or Windows
  `KILL_ON_JOB_CLOSE` Job Object establishes hard-host-termination containment
  before Deno is released, failing closed if containment setup fails;
  PTY creation reuses shell authorization and follow-on PTY access uses Object
  capabilities. Shell and PTY reserve finite-use authority, restore only on
  certified `ProviderEffectNotStarted`, and record ambiguous failures as
  `unknown`. PTY spawn/write/resize/close use structured pending intents and
  same-id conditional finalization; spawn publication failures retain cleanup
  metadata, fence every failed-create handle behind a Host-only orphan key
  before best-effort public Object deletion, and permit lifecycle close retry
  without making the orphan caller-addressable. Classifier failures finalize
  unknown evidence rather than dropping it. Follow-on finite object rights
  reserve/restore around the host call, and automatic child-exit cleanup records
  a close intent before exit-code observation/close. Object release finalizers
  run outside the SQL transaction so PTY
  close can durably record its intent; `swe_edit` refuses truncated source.
  Auto-allowed direct Git inspection disables optional locks, repository
  fsmonitor, and external diff helpers before the provider boundary. Direct Git
  argv and supported transparent launcher wrappers are limited to six exact
  inspection commands even under an always-allow shell policy. Authorized
  interpreters and native programs retain the documented host-user I/O boundary
  of Local Shell and are not claimed to be mediated as nested Git operations.
- `git-provider-is-pinned-and-non-executable`: the typed Git provider operates
  only on the configured workspace repository or an explicitly trusted managed
  worktree. It rejects parent discovery, untrusted gitfiles, symlinked metadata,
  alternates, and repository configuration that could execute hooks, filters,
  helpers, drivers, editors, signers, or implicit network fetches.
- `git-mutations-require-authority-state-and-evidence`: typed Git writes require
  Git authority, matching filesystem authority, an exact prior state token, and
  protected-operation evidence. Destructive, remote, and ref-rewriting actions
  additionally bind their approval to exact parameters and old object ids;
  pre-dispatch denials are not misclassified as unknown effects.
- `git-model-tool-retryability-is-effect-aware`: model-facing Git failures
  never advertise `unknown_effect`, ambiguous mutation timeouts, or
  post-dispatch mutation staleness as safely retryable. Read-only transient
  failures and mutation staleness with an explicit Host-certified
  not-started/pre-dispatch marker may remain retryable, but callers must
  re-observe state and reconstruct the decision.
- `git-patches-remotes-and-prs-preserve-cas-lineage`: patch Objects carry exact
  content hashes and source labels, remote operations bind the selected remote
  configuration and old refs, and simulated pull requests use common-dir refs
  plus atomically persisted metadata with base/head compare-and-swap checks.
- `command-risk-rules-are-deterministic`: command risk rules separate
  harmless, risky, and destructive shell operations without model judgment.
- `sandbox-profile-derived-from-capability-decision`: primitive sandbox
  profiles are derived from the same capability decision that authorizes the
  operation.
- `audit-query-windows-retain-latest-records`: limited audit queries select the
  latest matching records before returning them chronologically, and process
  audit views filter before applying their limit. Query limits, filters, and
  presentation flags require exact scalar types and reject values above the
  configured source bound before reaching storage.
- `event-query-windows-are-store-bounded`: LLM context and GUI process-event
  reads apply cursor/filter and limit in the store, return the newest bounded
  matching window in order, and do not materialize an unbounded event history.
  Public event queries reject coerced filters, cursors, booleans, and limits
  above that source bound before dispatching a store query.
- `gui-snapshot-reads-are-source-bounded`: top-level snapshot collections fetch
  at most the configured collection window plus one lookahead row before
  assembly. Process unread/interrupt counts, recent messages, bounded LLM
  count/token windows, ratings, ancestor reservations, and hierarchical
  remaining budgets use batch queries rather than one set of queries per
  visible process; GUI-level omissions detected by lookahead remain explicit in
  `_truncated`, while stricter subsystem list maxima remain authoritative.
  Persisted indexed visibility flags exclude internal presentation evidence
  before `LIMIT`; missing or malformed required 0.3 visibility state fails
  closed instead of being repaired during open.
- `tool-observability-redacts-sensitive-payloads`: tool audit/event
  observability redacts known structured payload/credential keys plus recognized
  scalar credential forms, URI/DSN userinfo, and HTTP Cookie headers before it
  stores bounded preview, hash, size, and truncation metadata. The hash covers
  the redacted projection, not the secret-bearing source. This is a
  defense-in-depth, syntax-aware filter rather than a claim that arbitrary
  opaque strings can always be identified as secrets; evidence producers must
  still use the typed sensitive fields or sanitize provider-specific formats.
- `jit-security-does-not-rely-on-static-blacklist`: JIT safety is enforced by
  Deno no-permission isolation, libOS syscalls, capabilities, human approval,
  and budgets rather than dangerous API regex blacklists. Unsupported
  dependency forms and BOM-prefixed dependency bypasses are rejected before
  Deno starts. Timeout, output-limit, resource-monitor, readiness, cancellation,
  and validation failures supervise and clean up the child process, monitor and
  cancellation tasks, file descriptors, and platform Job resources.
- `tool-policy-cannot-self-grant-authority`: ToolPolicy declarations cannot
  grant execution, resource authority, or confirmation.
- `tool-result-size-boundary-is-explicit`: tool result payload limits prevent
  unbounded result persistence while preserving committed side effects as
  explicit omitted-success results instead of retryable failures.
- `workflow-entry-uses-toolbroker-authority`: user-facing workflow entrypoints
  run tools from complete process tables rather than model projections, while
  retaining ToolBroker, result-object, and normal wait/exit/exec lifecycle
  semantics.
- `process-message-waits-are-race-free`: an empty blocking mailbox read and its
  wait registration are atomic with message posting, so a concurrent matching
  post either satisfies the read or wakes the registered process. Message row,
  evidence, terminal recheck, and wake state also roll back together.
- `process-message-mailbox-status-selection-is-exact`: normal mailbox reads
  select `unread`; `include_acked` selects exactly `unread` plus `acked`.
  The SQL status set is applied before ordering, limits, full matching counts,
  and blocking-read readiness, so `superseded_by_restore` history cannot hide a
  visible page or wake a waiter. A trusted storage query with neither `status`
  nor `statuses` remains an explicit unfiltered history query for recovery and
  inspection code; mailbox APIs do not use that form for `include_acked`.
- `process-message-input-remains-mediated-and-actionable`: queued user input
  remains in the durable process-message subsystem rather than being copied
  into proactive context deltas. The prompt directs a capable process to read
  the queue without embedding the message body, and narrowly normalizes
  reversible string encodings emitted by real providers before schema
  validation; neither path bypasses ToolBroker authority or audit.
- `runtime-lifecycle-transitions-are-atomic`: capability issuance, the core
  process exec row/capability/evidence transition, process exit, and
  parent/message waiter transitions do not publish partial authoritative state
  when an in-transaction sink fails. Higher-level image boot uses compensating
  restore rather than claiming one transaction across host/package work.
  Exec capability staging uses an expected-state transition, so a concurrent
  revoke or disable wins and cannot be overwritten or later resurrected.
  Successful exec advances the process execution generation, clears the old
  owner/lease, and returns the replacement image to `RUNNABLE` in the same
  transaction as its publication and evidence. The fenced worker may finish
  exactly one ToolResult handle append only when the committed publication,
  operation binding, prior token fields, current generation, cleared lease,
  row revision, ToolResult Object, and new object-handle capability all match.
  That narrow completion does not publish another MemoryView root or admit a
  repeated, forged, cross-purpose, or otherwise ordinary old-token mutation.
  Terminal signals use the same durable boundary; independent post-commit
  terminal notifier/finalizer failures cannot strand the other cleanup phase.
- `process-tool-table-authority-is-atomic`: complete process tool-table
  authority and the narrower model projection use process-row CAS inside the
  same transaction as their audit and ambient operation-evidence link. Audit,
  link, or other `BaseException` failures leave neither a partial row nor
  evidence behind; durable state remains unchanged after reopen, and
  concurrent configurations commit as complete serialized tables.
- `runtime-publication-compensation-is-retry-safe`: interrupted process launch
  and exec publications carry typed, exact artifact ownership receipts before
  publication-owned effects commit. Each recovery claim is durable before
  compensation starts; the same runtime resumes its lease idempotently, while
  startup under the backend-wide runtime lease takes over an orphaned claimant
  with a new fenced attempt. Cleanup, restore, terminal publication, and linked
  operation convergence then share one store transaction. Failed attempts remain
  retryable until the configured attempt ceiling persists a manual disposition,
  and every later reopen fails closed while that manual record remains.
  Launch-time capability grants and their exact receipts share one database and
  Object-payload unit of work. Committed checkpoint JIT installation records the
  candidate and Tool as separate exact receipts in the same unit of work;
  compensation handles those identities independently and never infers candidate
  ownership by looking up `registered_tool_id` from a Tool receipt.
  Compensation runs receipts in reverse order, rejects unknown handlers, and
  verifies capability/reservation, Tool row/handle/source/alias, candidate
  descriptor, loaded Skill, and workspace convergence before reporting
  `rolled_back`. Global JIT rehydration runs only after publication recovery and
  is never reached while an orphaned or manual publication is unresolved.
- `runtime-publication-startup-recovery-is-keyset-bounded`: launch, exec, and
  checkpoint pending recovery scan exact kind/state/marker keyset pages under a
  hard limit. Launch/exec terminal-operation repair and committed
  checkpoint-restore operation repair scan only durable marker-false rows;
  failed/manual checkpoint restores remain forward-recovery inputs. Orphaned
  `CREATED` processes are found by an indexed anti-join. Every backlog is fully
  processed while returned diagnostic ids remain bounded.
- `runtime-publication-domain-is-closed`: publication kinds and states are
  canonical at repository, backend, physical-schema, and reopen boundaries, so
  an invalid row cannot be silently skipped by recovery.
- `checkpoint-reconciliation-uses-exact-typed-storage-ports`: checkpoint
  restore orchestration is limited to exact publication and operation storage
  ports; architecture checks reject Any-typed, nested-store, reflection,
  raw-SQL, malformed-record, and generic-publication escape paths.
- `checkpoint-payload-delivery-is-attempt-fenced`: restored payload delivery is
  paged and fenced by an exact durable attempt. Acknowledgment is owner-bound,
  reconciliation-complete, read back, and safely compensated before any retry.
- `startup-recovery-entrypoints-require-the-opaque-lifecycle-lease`: every
  mutation-capable recovery entry invoked by Runtime assembly (prepared
  protected effects, provider reconciliation, capability/resource
  reservations, volatile Object payloads, ObjectTasks,
  launch/exec/checkpoint publications, stale operations, and stale process
  executions) validates the lifecycle-owned recovery lease as
  its first action. The lease is valid only while the runtime is `RECOVERING`
  and its private ContextVar value has the lifecycle's opaque identity. Calls
  from an `OPEN` runtime therefore fail before the first durable read, claim,
  callback, compensation, audit, or event write. JIT registry rehydration is
  also mutation-capable startup recovery: it requires the same opaque lease
  before its first process or artifact read, including when called directly
  through the JIT service rather than the broker.
- `jit-rehydration-is-keyset-bounded-and-owner-validated`: startup scans
  the normalized durable ephemeral-binding projection directly through the
  stable `(pid, tool_name)` keyset, without scanning or decoding unrelated
  process rows. Process and Tool mutations maintain exact JIT eligibility in
  that projection transactionally; a binary-collated partial covering index
  keeps both first-page and deep-cursor database work proportional to eligible
  bindings, not all callable history. Every SQL page and exact ephemeral-Tool/
  registered-owner-candidate lookup is hard-capped, and artifacts are fetched
  once per binding page rather than once per process. Candidate ownership and
  durable name are validated before the loaded-registry shortcut, so a
  cross-process alias is pruned even when its Tool id was already restored for
  the owner. Recovery returns exact totals and retains only one page of
  restored/pruned samples. Historical scan and temporary diagnostic memory are
  page-bounded; the final registry remains proportional to active JIT tools. A
  single process with arbitrarily many aliases cannot create an unbounded
  Python record or per-binding SQL query.
- `resource-usage-reservation-recovery-is-lease-gated-and-bounded`: startup
  recovery rejects callers without the opaque recovery lease before the first
  repository read. Active usage reservations are traversed by a status-first,
  hard-bounded `(created_at, reservation_id)` keyset. Ambiguous settlements,
  actual charges, and any resulting overage kill share one transaction, while
  diagnostics retain only one page of IDs plus the exact total.
- `stale-operation-recovery-is-keyset-bounded`: stale running operations are
  recovered through hard-bounded indexed keyset pages and a store-locked
  temporary uncertainty index. Diagnostics remain bounded while descendant
  unknown-effect outcomes are preserved.
- `startup-recovery-diagnostics-are-bounded`: prepared-effect reconciliation,
  provider reconciliation, stale capability-use reservations, provider-usage
  reservations, volatile Object payloads, ObjectTask reconciliation, JIT
  rehydration, stale operations, and stale process executions
  process their complete indexed/keyset backlog but retain only exact totals
  and one bounded sample page. Prepared protected effects restore their linked finite-use
  reservations before the remaining status-indexed capability reservations
  are abandoned. Stale execution state, concurrency high-water, audit, and
  event rows commit together page by page.
- `object-and-object-task-recovery-is-keyset-bounded`: volatile runtime-memory
  Object rows are released under the startup lease through a partial recovery
  index and per-Object CAS transactions before ObjectTask result repair. Active
  tasks, succeeded rows with result references, and retryable notification rows
  use normalized status columns plus stable `(created_at, task_id)` keysets.
  Same-timestamp backlogs larger than SQL bind limits converge without full
  history lists, and Runtime exposes exact totals with one-page samples.
- `checkpoint-restore-publication-program-is-immutable`: a restore plan is
  complete at insert, anchored by an immutable receipt-side digest, and
  validated before a recovery claim or callback. Committed marker-false rows
  also revalidate the exact operation binding and kind/name/actor/PID before
  convergence; plan-only persisted corruption fails startup closed.
  A failed exec preserves unrelated authority even when its issuer happens to
  use an `image:*` name; only exact publication metadata or receipts establish
  rollback ownership. Snapshot-based exec and process-local Tool, candidate,
  and Skill publication share one registry lifecycle lock, so a legitimate
  concurrent mutation commits only after exec reaches a terminal publication
  and cannot be overwritten by compensation from an older snapshot. Snapshot
  restore and a durable `compensation_applied` receipt marker commit together;
  if the later publication/operation terminal transaction fails, recovery sees
  that marker and finishes terminalization without replaying the snapshot over
  mutations admitted after the original exec returned. If online compensation
  fails before that marker exists, the internally issued, exact
  publication/operation-bound recovery signal moves the whole runtime to
  `CLOSE_FAILED` before control leaves ImageBoot, including its service-level
  direct entry point. The store stays available for diagnosis and ordinary
  close remains fail closed; every public mutation admission is rejected
  without writes until an explicit `release_recovery_diagnostics()` handoff
  releases the backend lease and a fresh reopen performs authoritative startup
  recovery. Forged or unbound
  recovery signals cannot suppress ordinary operation terminalization, while a
  damaged association discovered after a genuine durable signal remains
  fail-closed. The recovery fence also advances an admission epoch. The shared
  registry barrier revalidates that epoch after its outermost lock acquisition,
  so a candidate or Skill mutation admitted before the fence cannot wake and
  publish afterward. Capability consume/reserve waiters likewise revalidate
  after acquiring the backend transaction lock, and `AuthorityTransaction`
  revalidates before settlement and UnitOfWork commit. A stale lease therefore
  rolls back business state, finite-use reservations, and evidence instead of
  committing after poison. A recovery fence may supersede an earlier ordinary
  shutdown timeout; ordinary shutdown alone does not revoke already-admitted
  work.
- `scheduler-quantum-ownership-is-serialized`: scheduler and direct pid
  single-step APIs share the same runtime lock, store claim, and resource-charge
  boundary, so one process cannot be re-entered concurrently. Constructor and
  run controls require exact, finite, bounded scalar types before creating a
  worker or dispatching a quantum. Terminal process
  rows are immutable to ordinary writers, and a detached worker's execution
  generation/owner/lease token cannot mutate any process-local field. A bound
  worker token never falls back to Host authority for another PID; intentional
  cross-PID control writes name the target, allowed source statuses, revision,
  and reason. While a `process_exec` publication is active, its exact RUNNING
  generation/owner/lease tuple exclusively owns process-row writes; an ordinary
  tokenless Host patch is rejected before mutation. Trusted pause, cancellation,
  termination, resource-limit kill, and ObjectTask fallback may supersede that
  lease only through a scoped takeover naming the exact PID, revision, state
  generation, lease tuple, intended typed state, and nonce. Optional reason
  Object/capability/view preparation and the single semantic state transition
  must all finish in the same unit of work; a cross-PID write or incomplete
  takeover rolls the transaction back. A synthetic RUNNING row with no lease
  tuple retains the legacy exact control CAS, while a partially populated tuple
  fails closed. The only terminal-row bookkeeping exception is an exact CAS scope
  naming target, terminal source status, revision, execution generation,
  ambient worker token, and reason; there is no general terminal-mutation
  bypass. Exec admission atomically rotates either the runnable Host epoch or
  the exact active worker token before it creates a publication. Successful
  exec clears that internal lease and returns the process to the runnable queue
  in the publication commit transaction. Failed exec compensation also returns
  it to the queue behind a newer generation; it never revives the superseded
  worker token. Snapshot restore itself CASes the caller-observed current row
  revision together with RUNNING status, the state generation derived from the
  publication-bound before snapshot, and the admission generation/owner/lease
  recorded by the publication. If a trusted takeover wins first, compensation
  preserves that winner, records
  `compensation_failed`, resolves the operation as `UNKNOWN`, and fences the
  runtime in `close_failed` instead of reporting a false rollback. Terminal
  commit receipts remain authoritative if acknowledgement is interrupted,
  while a successor claim may legitimately advance the live row.
- `awaitable-quantum-cleanup-is-bounded-and-lifecycle-honest`: one monotonic
  shutdown deadline bounds pending-task cancellation, asynchronous-generator
  finalization, and default-executor cleanup after every awaitable quantum.
  Cancellation-resistant work fails the quantum explicitly and releases its
  process execution lease. A still-running default-executor worker remains a
  scheduler lifecycle fence, so runtime shutdown reports incomplete until the
  worker actually stops and a retry can close shared state safely.
- `runtime-shutdown-is-drained-and-retry-safe`: scheduler work, ObjectTask
  executors, Human/provider blocking jobs, PTY reader/monitor workers, active
  admission leases, and GUI runtime users drain before shared state closes; a
  timed-out shutdown leaves storage open and can be retried. After admission
  drains, ordinary shutdown records attempt evidence before stopping components
  while it still owns the store; that record does not claim later teardown
  succeeded. A stage failure names the dynamic `<stage>_stopped: false` result
  field, and only completed component teardown plus terminal store-ownership
  release yields the shared first-attempt `ok: true` result. A checked public
  mutation inventory is installed under admission. Every public Human control
  method is classified as mutation or read-only; approvals, presentation,
  terminal draining, cancellation, and recovery are rejected at `STOPPING`
  before any durable or in-memory write. Every public CapabilityManager method
  is likewise classified as read, mutation, or audit-sensitive mixed;
  the public lease and mutation subservices are complete guarded ratchets, so
  direct lower-level calls cannot bypass lifecycle fail-close. Runtime-owned blocking work uses a
  drainable supervisor; standalone reusable components use an owned one-call
  executor that is drained even after coroutine cancellation. The architecture
  ratchet permits no raw `asyncio.to_thread` or default-executor dispatches.
  Recovery-diagnostics handoff never reclassifies ordinary user/module
  shutdown callbacks as safe. It runs only explicitly tagged, idempotent
  transient cleanups under a no-commit fence. The PTY cleanup closes live
  handles and joins reader/monitor workers without changing Objects or evidence;
  a partial failure preserves the callback and session for retry and keeps the
  store open. Its only evidence-free provider action is `handle.close()` behind
  the lifecycle's opaque, callback-scoped recovery-cleanup lease. The static
  protected-operation ratchet rejects direct invocation, a late/forged guard,
  or any other provider method on this path.
  The HTTP endpoint acknowledges success only after completed Runtime teardown,
  while process exit fails visibly if bounded retries still fail.
- `gui-local-control-surface-is-origin-bound`: browser CORS accepts loopback
  development origins and exactly `agent-libos://app` for the production
  renderer, while rejecting `null` and every other custom origin.
- `object-task-entry-uses-toolbroker-and-object-authority`: Object-bound
  background tasks run tools through ToolBroker, process tool tables, Object
  capabilities, owner-watch Object Memory primitive notifications, and
  process-message boundaries. Runner processes are host-managed and excluded
  from the LLM scheduler; one-shot owner authority is reserved before runner
  creation and committed with the durable task record. The mapped cross-actor
  regression directly proves finite read consumption for get and finite write
  consumption for cancel. Failed executor handoff
  terminalizes the task and removes the unstarted runner, while failed result
  wiring terminalizes the runner and releases the unpublished result and its
  derived handles. Terminal/cancel reconciliation must not leave active pins
  behind, and owner-watch resumes only replay tools with explicitly safe
  message-receive semantics.
- `image-only-transcript-is-transparent-and-flow-guarded`: `image_only` sends
  the exact Image system prompt, the raw or canonical-JSON process goal, and
  only the cumulative native assistant/tool transcript. Runtime Object Memory,
  Skill, Capability, fallback, repair, and explanatory prompt text cannot enter
  the durable conversation. Call ids and model-facing result projections remain
  paired across parallel stops, waits, and reopen; Image/goal/prompt changes
  start a new anchor. The active full-I/O head is retention-protected and a
  configuration that cannot preserve it fails before provider dispatch.
  Trusted historical labels and goal/result Object references remain outside
  the model transcript for IFC approval and audit, so sensitive tool output
  still denies an uncleared later LLM egress.
- `llm-call-records-opt-out-are-bounded-and-redacted`: when
  `llm.persist_full_io` is false, new LLM call rows store canonical content-free
  summary envelopes containing only schema version, tier, byte count, hash, and
  JSON kind/item count where applicable. No prompt, schema, response, tool-call
  argument, reasoning,
  provider payload, error text, key name, scalar value, or preview is durable
  in the row. The same content-free summary and hash boundary applies to
  related audit records, events, and result Objects; none persist raw provider
  I/O. Pending conditional releases likewise
  persist only hashes and non-sensitive resume metadata before approval;
  same-runtime approval reuses the hash-bound in-memory request, while reopen
  fails closed without provider dispatch.
- `payload-retention-preserves-runtime-evidence-and-recovery`: payload retention
  is explicit, bounded, monotonic, and transactionally audited. It accepts only
  canonical provenance-bound content-free targets and does not erase evidence
  still required by live recovery.
- `llm-responses-state-chain-is-lossless`: the low-level `LLMClient` sends an
  explicitly supplied `previous_response_id` only for an official, stored
  Responses request. Paired tool history is represented natively; unpaired or
  unsupported output becomes bounded plain context and disables the chain. The
  current full-snapshot AgentProcess executor does not enable this client path.
- `llm-async-clients-are-event-loop-scoped`: real async SDK clients and their
  keep-alive pools are request-scoped and cannot cross scheduler event loops.
- `llm-provider-state-is-scope-bound-and-nonreplayable`: the Runtime records
  scope-sensitive provider fingerprints while the AgentProcess executor remains
  stateless even when provider continuation policy is configured. The low-level
  client does not enforce those Runtime fingerprints. Durable waits use
  token-scoped pending/resuming/completed CAS and
  synchronize restored generations; an ABA
  claim, post-claim exception, or interrupted reopen fails closed and is never
  auto-replayed.
- `llm-profile-selection-is-process-local`: host-selected LLM profiles are
  stored as process-local ids, resolved at LLM-call time, inherited by child
  processes, preserved by image-package defaults, isolated from non-default
  ambient provider environment, and fail closed when the id is unknown.
- `automatic-context-management-does-not-grant-authority`: context pressure
  may select an Image-configured tool, but never inserts it into the process
  tool table or bypasses argument validation, Capability, resource, approval,
  event, audit, and durable-wait enforcement. Model-window maintenance is
  dispatched only when persistent context is explicitly enabled and the
  process independently holds `context:maintenance/execute`; default
  source-only pressure leaves the request unchanged and records that maintenance
  was not authorized. An initial failure with unchanged
  context generation is audited and remains invisible while the original model
  request continues; a changed generation ends that quantum so the next one
  rebuilds context. Human/child/message waits remain durable, and a failed
  resumed attempt terminally completes its pending generation before rebuilding
  the ordinary request from the current generation. Prompt-mode pressure
  accounts for its final numeric notice and fails before provider dispatch if
  that exact request exceeds the configured context window. Storage-triggered
  maintenance uses the same authority and durable-wait path but fails the
  process if compaction cannot advance the context generation.
- `resource-budgets-are-hierarchical`: resource usage is charged to the acting
  process and its parent chain, and visibility/capability mechanisms cannot
  mint additional budget. The complete hierarchy, reservations, event, and
  audit commit atomically; overage terminal callbacks run after releasing the
  store lock. Discrete counts/bytes/tokens are integers while runtime and
  subprocess wall/CPU seconds are continuous finite values.
- `llm-token-usage-is-charged-before-tool-dispatch`: provider-reported LLM token
  usage is validated (including type, sign, and component consistency) and
  settled before any model-selected tool call is dispatched.
- `subprocess-resource-profiles-are-enforced`: shell and Deno subprocess wall,
  CPU, and RSS limits are enforced by supporting providers and audited on
  exceedance. POSIX PTY supervision runs independently from output reads,
  accumulates observed CPU by process identity, and fails closed when
  process-tree accounting is denied. The Windows ConPTY backend advertises no
  such support and rejects `SubprocessLimits` before spawn. POSIX cleanup falls
  back to explicit descendant signaling when process-group signaling is denied
  and serializes concurrent close attempts.
- `skill-activation-does-not-grant-authority`: Skills change visibility and
  prompt context without granting resources. Immutable built-in activation
  projects an exact Image-bound subset without changing the complete table;
  registered Skills may expand complete/model tables only through their
  governed activation path. Actor-scoped validation, registration, and
  activation must still honor process authority and Skill capability or
  human-approval gates; actor mode cannot invoke Host-path validation.
  Finite-use Skill permissions are
  reserved before a registry, trust, activation, or unload mutation and are
  committed only with that mutation. Registry/trust/audit state changes are
  transactional; failed activation cannot leave a visible JIT alias, and
  reactivation or unload retires the exact superseded process-local JIT rows.
  Loaded-Skill provenance preserves a base/shared alias until its last actual
  source is unloaded; noncanonical persisted provenance is rejected before
  unload. Discovery rejects non-positive, boolean, and above-config limits.
- `skill-read-cancellation-restores-finite-authority`: Skill discovery and
  inspection restore reserved finite read authority when interrupted before
  publication, including cancellation exceptions outside `Exception`.
  Compensation preserves the original failure; if restoration itself fails,
  authority remains consumed, the failure is annotated, and restart abandons
  the stale reservation without minting another use.
- `skill-package-snapshots-are-identity-stable-and-bounded`: Host Skill
  packages retain and revalidate a no-follow root-to-parent directory chain,
  reject link or identity changes before hashing, and read every regular file
  to EOF through per-file and cumulative byte limits. Workspace package reads
  enforce the same cumulative budget before reading the remaining files, while
  stable Host and workspace snapshots produce the same package hash.
- `workspace-skill-snapshots-reject-truncated-inputs`: workspace Skill
  registration rejects truncated `SKILL.md`, extension metadata, and JIT source
  reads before parsing or hashing them, so a snapshot never authenticates only
  a bounded prefix of a larger file.
- `package-json-manifests-fail-closed-at-parser-limits`: host and workspace
  Skill metadata plus image-package JIT manifests reject oversized JSON
  integers and excessive nesting as typed validation failures instead of
  leaking interpreter parser exceptions.
- `shared-json-codecs-are-finite-and-node-bounded`: shared JSON decoding
  applies fixed integer, nesting, node, and finite-number limits, while shared
  serialization never emits non-standard `NaN` or `Infinity` tokens. Ordinary
  finite JSON remains compatible and memory exhaustion is never normalized as
  an input error.
- `yaml-documents-fail-closed-at-bootstrap-limits`: the shared YAML loader
  applies configuration-independent UTF-8 byte, parser-event, composed-node,
  nesting-depth, alias-count, expanded alias-graph, logical scalar-byte, and
  radix-independent integer-digit limits before Python object construction.
  Recursive aliases, non-string mapping keys, and duplicate keys fail as typed
  validation errors. Duplicate checks apply to pre-merge source pairs so
  standard merge precedence and explicit overrides remain supported. Malformed
  scalar tags are normalized at the construction boundary, while `MemoryError`,
  cancellation, and other system exceptions still propagate. Bootstrap config
  reads only one bounded prefix, uses this same loader, preserves merge
  behavior, and retains its public `ValueError` contract.
- `runtime-modules-load-trusted-code-atomically`: startup Runtime Modules bind
  trust to the current source hash, reject ambiguous manifests and duplicate
  module ids, resolve import strings without executing untrusted package code,
  read manifests and sources to EOF through bounded identity-stable descriptors,
  retain and revalidate a no-follow POSIX root-to-parent descriptor chain (or
  the equivalent Win32 handle chain), guard every package parent directory
  against replacement/reparse traversal,
  reject direct registration-buffer mutation that bypasses `provides`, and roll
  back failed declared or hook-created tool/image/syscall/hook registrations so
  persisted module status stays aligned with loaded runtime state. Module
  failure rows, audits, and warning results contain correlated text-free error
  envelopes and diagnostic hashes rather than exception-authored text. Package
  import interruption also removes its synthetic namespace/importer for every
  `BaseException`, so cancellation cannot leave import state behind. One shared
  runtime registry lock serializes
  the full module lifecycle with official Tool/Image publications, preventing
  a failed snapshot restore from clobbering a concurrent successful load.
- `module-json-manifests-use-bounded-strict-parsing`: Runtime Module JSON
  manifests preflight through the shared strict JSON parser before the
  module-specific duplicate-key pass, so compact depth, node, integer, and
  non-finite-number attacks fail as typed validation errors even when the
  interpreter integer guard is disabled. The existing bounded,
  identity-stable Host read and ordinary finite metadata semantics remain.
- `checkpoint-restore-and-fork-are-scoped`: checkpoint creation atomically
  captures one consistent store snapshot and publishes its row, head, initial
  read capability, event, and audit. Restore/fork are scoped,
  capability-controlled, ownership-based, revoke-wins, and append-only outside
  reconstructable state. Restore allocates revision and execution-generation
  high-water marks, clears owner/lease identity, and never revalidates a stale
  CAS or worker token; fork initializes a new identity rather than cloning
  concurrency tokens. Persistent SQLite-file and PostgreSQL contracts prove
  those high-water marks and fork identities survive reopen, and that a writer
  paused behind restore's transaction cannot commit against the old epoch.
  Borrowed/`EXTERNAL_REF` state is not cloned, JIT
  tool/candidate ids are remapped and atomically published, and finite-use
  snapshot authority is never copied. Restore reauthorizes its composite
  decision set, publishes reconstructable state and core event/audit evidence,
  and settles finite uses in one AuthorityTransaction; a sink or settlement
  failure rolls back the full unit without fallible compensation. Fork revalidates/consumes actor authority
  only in its publication transaction, global Skill/Image rows are not replaced
  by fork. Restore records a versioned `checkpoint_restore` publication and
  exact operation binding in the main-state transaction; post-commit phases
  receive ordered receipts, fence mutation admission on failure, and resume
  under a durable startup lease. Version 2 records Object-payload reconciliation
  before image/JIT/finalizer work; exact version-1 programs and anchors remain
  readable without mutation. Recovery selectively rehydrates unchanged restored
  rows from the hash-bound snapshot before general missing-payload cleanup and
  general JIT rehydration. A terminal delivery handshake prevents a late startup
  failure from consuming that replay, while a newer Object version is never
  overwritten. A second successful reopen is a no-op. Bounded Object-release
  intents use stable module-declared handler ids and idempotency keys; a missing
  handler preserves its work and moves the publication to `manual` rather than
  claiming completion. Fork post-commit failures are reported without claiming
  rollback.
  Restore and fork require canonical data-flow metadata on every captured
  pending-action row, including completed history, and on every captured
  process message. Missing, incomplete, or malformed carriers reject the
  artifact before process-state publication; restore never invents
  conservative labels for legacy rows.
  Post-checkpoint terminal ObjectTask history is retained as
  `superseded_by_restore` with stale live runner/result references cleared;
  the comparison uses the terminal transition time so delayed notification
  bookkeeping cannot supersede a task captured as complete. A normal store
  reopen also changes a persisted success with a missing runtime-only result
  payload to `result_unavailable_after_reopen` instead of leaving a dangling
  `result_oid`. Process-message rows superseded by restore likewise remain
  durable history but are excluded by the backend status set from mailbox
  pages, counts, and blocking readiness; `include_acked` admits only `unread`
  and `acked`. Restore reserves checkpoint admin plus every changed-image
  right as one composite set and settles it only with the main transaction;
  replacing an existing image requires `admin`, while restoring a missing
  image requires `write`.
  Checkpoint create/restore/fork acquire the shared registry lifecycle lock
  before Object Memory ownership and store locks; image/JIT reconciliation
  remains inside that lifecycle boundary, while host release finalizers run
  outside it. Multi-image reconciliation commits cache, image rows, and artifact
  rows as one batch or restores the complete prior cache. Durable finalizer
  handlers are buffered by trusted module entrypoints, so they are reconstructed
  before checkpoint publication recovery rather than waiting for startup hooks.
  Diagnostic inspect/diff/replay use one reservation scope: each finite
  checkpoint/process read claim rolls back on failure and settles exactly once
  on success.
  `checkpoint:*` control capabilities are excluded from new snapshot artifacts
  and are also filtered during restore and fork, so snapshot transitions cannot
  duplicate or resurrect authority over checkpoint artifacts.
- `image-self-evolution-requires-image-authority`: image registration, package
  boot, exec, and checkpoint commit require image authority and do not bake
  external authority. Failed registration/commit removes new artifacts and
  restores replaced manifests; failed package boot/exec removes the exact
  publication-owned capability, receipt, private workspace, and unpublished
  JIT source/candidate state. Image-package workspace grant `recursive` and
  `delegable` values are admitted only as exact booleans (with absent fields
  defaulting to false), before image/artifact publication, private-directory
  creation, capability issuance, or related audit/publication evidence. Host
  package bytes are read through bounded,
  identity-stable Unix or Win32 descriptors while guarded root-to-parent
  component chains (no-follow POSIX descriptors or replacement-blocking Win32
  handles) prevent
  reparse/replacement traversal. Registry callers and
  getters receive isolated deep copies, concurrent same-id registrations are
  serialized and revalidated in the cache/store critical section, and a
  candidate is published to the shared cache only after its durable manifest
  and required event/audit evidence commit. Failed registration or replacement
  therefore remains invisible to unlocked launch, LLM, Skill, and JIT readers, and
  committed-image boot does not overwrite the global Skill registry. New image
  publication requires `write`, replacement requires `admin`, and finite image
  plus checkpoint-read authority settles in the registry mutation transaction.
- `agent-output-is-not-control-channel`: untrusted command output cannot trigger
  lifecycle control actions; submission/exit must use explicit tool or syscall
  arguments.
- `jsonrpc-provider-effects-are-registered-and-classified`: JSON-RPC endpoint
  registration and calls use registered endpoint/method authority, gate calls
  and per-item registry operations before manifest metadata is exposed, and
  classify provider effects. Provider dispatch uses immutable Host-snapshotted
  transport inputs and ignores ambient proxy configuration. A single monotonic
  deadline spans pinned connection, TLS handshake, request dispatch, response
  headers, and the complete bounded body. Its joined per-socket watchdog closes
  only that request socket; a timeout after dispatch never fails over or replays
  a potentially non-idempotent request. Registry
  row/stale-grant/event/audit mutations are
  transactional, including finite registry-authority reservation/settlement.
  A call reserves finite authority and persists pending evidence
  before non-local DNS and transport; DNS observation prevents later transport
  PENS from erasing information flow or restoring the use.
- `mcp-provider-effects-are-registered-and-classified`: MCP server registration
  and tool calls use registered server/tool authority, gate calls before server
  metadata is exposed, and classify provider effects. Validation, discovery,
  and calls share immutable Host-snapshotted transport inputs and do not inherit
  ambient transport environment. Registry mutations are transactional.
  Refreshed tool listing/tool calls atomically reserve their
  deduplicated main, server, process-spawn, and exact stdio authority and persist
  pending evidence before DNS/live-provider boundaries. Local/stdio first-call
  PENS may restore; non-local DNS observation cannot be erased.
- `mcp-protocol-mode-is-explicit-and-registry-fenced`: Manifest v1 omits
  protocol mode and retains its exact canonical registry/approval/Sink identity;
  Manifest v2 requires one of the three release-locked modes. Replacement
  advances the registry generation and every discovery/list/call phase rejects
  a changed spec or generation before further dispatch.
- `mcp-modern-negotiation-never-falls-back-on-ambiguous-failure`: automatic
  modern negotiation may fall back only on the transport-specific recognized
  legacy signal and only before Tool dispatch. Authentication/server errors,
  malformed or oversized replies, modern protocol errors, DNS/TLS failures, and
  ambiguous timeouts never become legacy evidence or a replay path.
- `mcp-unsupported-modern-features-never-gain-runtime-authority`: discovered
  server capabilities, annotations, cache hints, notifications, and unsupported
  reverse requests are untrusted diagnostics. They cannot register Tools,
  grant Capability, alter effect policy, invoke Runtime behavior, or persist a
  negotiated session.
- `mcp-input-required-is-never-automatically-replayed`: modern
  `input_required` is a stable, non-retryable terminal result. Continuation
  state is not persisted or returned; consequential/ambiguous mutation remains
  unknown and a linked Durable Task Run requires Host attention.
- `mcp-v1-identity-and-provider-contract-remain-stable`: adding Manifest v2 and
  the optional modern provider SPI does not add a v1 discovery probe, change v1
  canonical bytes/digests, or add required parameters to the existing
  `McpProvider` signatures.
- `registry-manifests-admit-only-finite-json`: MCP and JSON-RPC mapping and
  YAML registration recursively reject non-finite values in server/endpoint
  metadata, tool/method metadata, and input/parameter schemas before durable
  mutation. Validation exposes stable domain errors, propagates memory and
  control failures, and preserves nested finite JSON.
- `explainable-operations-use-explicit-causality`: protected LLM, Tool,
  syscall, primitive, and runtime boundaries persist typed parent/child rows and
  explicit evidence links. Human/child/message waits reuse their durable
  operation ids. Runtime publications authoritatively reconcile their linked
  operations in the same terminal transaction: `committed` is `succeeded`,
  `rolled_back` is `failed`, and uncertain compensation is `unknown`. Reopen
  performs this reconciliation before interrupting only the remaining orphaned
  running rows, and may correct an earlier terminal outcome after a crash. A
  failed terminal transaction leaves both the publication and its exactly
  linked operation nonterminal for recovery; an unlinked pending signal cannot
  bypass generic operation finalization. Publication planning atomically stores
  both immutable versioned `plan.operation_id` data and the operation's
  publication id/kind/binding metadata; the reverse link must resolve to exactly
  one operation. Recovery never creates a missing association: blank, missing,
  unbound, multiply-bound, identity-mismatched, or already-bound operations fail
  reopen closed without being rewritten. Online spawn/fork/spawn-child process,
  event/audit, publication, and operation terminal writes share one transaction;
  terminal sink failure compensates exactly or fences mutation until reopen.
  Exact prebinding permits a root-spawn pre-return `pid=None` row only so its
  terminal transaction can canonicalize the publication child PID.
  Explanation
  completeness checks declared roles and never fills gaps from pid/time
  proximity. A `process.exec` root spans snapshot, publication, boot, evidence,
  commit, and rollback under the same admission lease. Host output applies
  observability redaction.
- `context-manifests-are-metadata-only`: each LLM context preparation records
  source Object selection/omission reason, version, transform, tokens, hashes,
  final context generation/Object, and compaction metadata without copying
  Object payloads or rendered prompt text.
- `process-terminal-cleanup-is-durable-and-bounded`: a committed exit, failure,
  kill, or cancellation atomically creates one durable cleanup intent. Its
  notifier and process-finalizer phases are independently fenced and
  idempotent; incomplete work is retryable during the same call or after
  reopen without rewriting the terminal outcome. Startup recovery reads only
  hard-bounded `(created_at, pid)` keyset pages and retains at most one page of
  diagnostic samples.
- `internal-exception-text-never-crosses-runtime-boundaries`: unknown internal
  exceptions, post-dispatch validation/authority failures, Provider failures,
  Runtime Module failures, and JIT output/syscall failures expose only stable
  public codes, types, and correlation ids. Only Host input validation proven
  to precede tool/provider implementation dispatch may use a fixed,
  instance-free template. Scheduler status, process/tool failure results,
  action feedback, and GUI responses never retain exception text; durable
  diagnostics bind the same correlation id to only the error type, byte count,
  and SHA-256 fingerprint. Workflow failures, image-boot status and rollback
  evidence, terminal notifier/resource-finalizer warnings, and Runtime assembly
  cleanup handles follow the same boundary. Context-compaction job state follows the same
  envelope and fingerprint boundary, including validation, authority,
  interruption, and cancellation failures. Checkpoint post-commit and recovery
  receipts and invalid ResourceUsage projections likewise never echo exception
  text or unknown caller-supplied field names. A process-local marker distinguishes caught
  exceptions from deliberate structured `ToolResult` failures, and serialized
  fields cannot forge that marker.
- `runtime-safety-benchmark-is-deterministic`: benchmark tasks and smoke runs
  remain task-schema-v1/run-output-schema-v2, deterministic, and token-free by
  default. Effect outcome and evidence are explicit; exact/prefix/glob matching
  cannot broaden implicitly; unknown/invalid/orphan/runner-failure output nulls rate fields;
  false-denial and unauthorized-effect denominators use only their documented
  qualified effect subsets.
- `practical-native-evidence-has-no-modeled-fallback`: native practical rows
  require real tool, provider-state, external-effect, and operation evidence;
  modeled rows stay in a separate denominator.

- `skill-discovery-catalogs-are-bounded-and-source-consistent`: Host and persisted Skill catalogs share Unicode matching and fail closed at the configured scan ceiling.
- `runtime-registration-mutations-are-audit-atomic`: Tool and syscall route bindings roll back atomically when required audit recording fails.
- `rating-mutations-are-audit-atomic`: rating updates and required audit records commit or roll back in one transaction.

## Known Test Gaps

- The runtime-safety benchmark is an early deterministic workload, not a
  complete paper evaluation suite.
- Explainability tests verify provenance completeness and deterministic
  summaries, but do not yet measure whether operators understand explanations
  better in a user study.
- Per-change Python CI exercises 3.11 and 3.14 on Ubuntu and the complete
  deterministic `all` lane on native Windows Python 3.11 with Deno and the
  optional PTY dependency installed. A separate configured Python 3.11 matrix
  runs the required host-filesystem-identity nodes on Ubuntu and macOS 14 with
  `--fail-on-skip`. Python 3.12/3.13 are inside the declared package range but
  are not separate root-runtime jobs, and broader macOS runtime behavior remains
  environment-gated. Windows CI does not create guarantees the implementation
  does not provide: ConPTY still lacks Job Object/parent-death and wall/CPU/RSS
  supervision, while credential-manager and real remote-authentication paths
  remain explicit environment gates.
- PostgreSQL CI uses PostgreSQL 17 on Ubuntu. Other supported server versions
  and deployment TLS/authentication topologies are not release-gated here.
- JSON-RPC, MCP, and Git remote tests use deterministic loopback or local
  remotes. The complete MCP transport, adapter, and SDK integration files plus
  fixed-upstream applicable Tools-client conformance scenarios without an
  expected-failure baseline run on Ubuntu Python 3.11 and 3.14 from the frozen
  optional extra; real proxy/TLS/DNS policy,
  HTTPS/OpenSSH authentication, and remote MCP deployment identity remain
  environment-gated. GitHub/GitLab API integrations and MCP MRTR/OAuth/listen,
  Resources, Prompts, Tasks, Apps, Roots, Sampling, Logging, OTel product
  support, and server surfaces are not implemented.
- Real LLM credentials and token-spending paths are opt-in. The default matrix
  covers mock/action-selection behavior, not a live request for every supported
  profile or provider deployment.
- GUI CI covers React tests, TypeScript typechecking, production build, and the
  Python HTTP/SSE server on Ubuntu. Accessibility/usability studies, native
  Electron lifecycle, and the production-build custom-protocol BrowserWindow
  smoke remain environment-gated. Installer packaging, signing, and
  notarization are not configured.
- Data-flow tests cover runtime-mediated payloads; trusted modules/providers,
  native child I/O, Sink re-forwarding, and direct RuntimeStore administration
  remain operator trust boundaries rather than containment-test guarantees.
- Practical and recovery benchmarks validate their documented deterministic
  profiles. Hosted-provider workflows, real-model benchmark breadth, and the
  unimplemented million-publication recovery profile remain outside the
  per-change gate. See `docs/support_matrix.md` for the authoritative coverage
  matrix and release-gate procedure.
- Release-artifact CI uses remote Actions pinned to reviewed full commit SHAs,
  builds one canonical wheel/source pair with a frozen release tool group,
  rejects extra or non-regular output, records its exact checksum manifest, and
  makes the Python 3.11–3.14 clean installs consume that same pair with
  hash-bearing root-lock dependency exports. Moving hosted runner images and
  compatibility selectors plus the lack of signed attestation mean this is not
  a bit-for-bit reproducible publication chain; PyPI upload or any external
  release mutation remains separately authorized.
