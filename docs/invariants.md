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
  observation, stale-size-safe bounded reads, PTY spawn cleanup, and write
  limits. Filesystem/clock/shell/PTY provider calls persist a pending `unknown`
  effect intent before the boundary, CAS the same id on final classification,
  and after attempting the call remove it without a final record only when the
  provider certifies `ProviderEffectNotStarted` and no earlier information flow
  occurred. Clock sleep/asleep inserts the intent before its first monotonic
  observation, treats elapsed-time measurement as information flow, and permits
  restore/abandon only when that first observation is certified not-started.
- `capability-matching-and-delegation`: typed matching, deny dominance,
  one-shot grants, atomic default consumption, exact effect reservations,
  revoke-wins restoration, crash abandonment, grant-as-transfer,
  parent-linked delegation attenuation, restrictive parent boundaries,
  malformed authority-rule fail-closed behavior, and ISO-normalized leases.
  Delegation publishes its row/process attachment/evidence atomically, and a
  multi-spec authority derivation is prevalidated and committed all-or-nothing.
- `capability-subject-isolation`: preselected capability candidates are filtered
  to the requested process subject, and Human-approved capability
  specifications cannot redirect authority to another process.
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
  authority gates and under a filesystem effect intent.
- `process-message-label-observation-is-linearizable`: observing durable
  message labels preserves a concurrent ACK or terminal process transition and
  creates at most one durable label carrier for a message.
- `filesystem-label-critical-sections-are-hierarchical`: overlapping path
  trees serialize fairly, including normalized aliases and missing-parent
  creation, while unrelated paths remain concurrent and widening reentrant
  lock upgrades fail.
- `task-authority-manifest-bounds-launch`: image requirements are declarations;
  Host manifests compile launch grants and bound model requests, child
  transitions, budgets, approval policy, and provider effect classes.
- `effect-ceilings-distinguish-unrestricted-and-deny-all`: omission of a Task
  Authority effect ceiling remains unrestricted, while an explicit empty
  ceiling is a versioned deny-all value that cannot be downgraded after reopen.
- `effect-transactions-are-idempotent-and-reconcilable`: provider intents bind
  canonical arguments and idempotency keys, approval leases bind exact effects,
  and startup reconciliation queries but never replays providers.
- `external-effect-recovery-is-keyset-bounded`: startup recovery scans only
  nonterminal external effects through bounded indexed keyset pages and
  converges the full backlog without materializing the full effect history.
- `protected-provider-operations-use-sdk`: LLM, filesystem, clock, shell, JSON-RPC,
  MCP, Human, and PTY provider effects share one contract registry and one
  prepare/dispatch/finalize state machine. Static coverage rejects direct
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
  parent signal. Ambiguous Human provider outcomes are not automatically replayed;
  GUI release binds the complete public request view including `decision`, and
  interactive replies apply only to the exact Human request already shown to
  the operator. The CLI also retains an exact conditional-release request ID, so
  bounded pending-request windows cannot starve its prerequisite. Unchanged
  unrestricted GUI views reuse receipts only within the
  same provider session after a linearized current-policy check, and internal
  presentation evidence cannot starve bounded causal snapshot windows. Current
  file-tree labels use bounded keyset batches and a bytewise-collated exact
  prefix range, so backend locale and wildcard path characters cannot widen a
  subtree query. The
  `data_label_exfiltration` benchmark proves that ordinary write capability
  alone cannot export secret context.
- `model-tool-projection-does-not-change-authority`: lazy model schemas are a
  durable projection of the complete image tool table and cannot add
  capabilities.
- `object-memory-names-are-not-capabilities`: Object Memory names and
  namespaces do not bypass object capabilities. Successful namespace listing
  consumes every finite namespace/object visibility decision used in the
  returned result in the same transaction as its audit.
- `object-memory-materialization-budget-is-authoritative`: Object Memory
  context materialization is bounded by final rendered object text, not trusted
  metadata token estimates.
- `context-compaction-preserves-authority-and-fails-closed`: context compaction
  uses child-process summarizers without granting external authority, validates
  summaries, preserves process authority, and fails closed on races or invalid
  output.
- `child-memory-merge-lifecycle-is-explicit`: terminal child process memory
  remains mergeable, then is adopted or released by the parent lifecycle.
- `object-memory-lifecycle-is-explicit`: Object Memory ownership, release, and
  RAII cleanup are explicit and revoke stale authority, including Object-bound
  PTY handles. Lifecycle mutations serialize ownership-lock before store
  transaction and use LIVE, owner, and version conditional writes so release
  races, concurrent updates, and owner ABA fail closed. Trusted delete rolls
  back Object release, capability revocation, audit, and in-memory payload
  together; a multi-Object ownership transfer and its audit are all-or-nothing.
  Link authorization, both Objects' LIVE checks, finite-use consumption, and
  evidence share that same ownership-lock/transaction linearization point.
- `runtime-store-single-active-writer`: a writable persistent runtime store has
  at most one active Runtime opener across canonical/symlink path aliases.
  SQLite validates a no-follow regular-file lease or uses its kernel exclusive
  fallback; the secure POSIX path keeps database/lease/journal/WAL/SHM files
  owner-only; PostgreSQL keys advisory leases by database/schema.
- `storage-transactions-recover-or-fail-closed`: commit/savepoint finalization
  failure restores SQL and opted-in Object payload state; rollback failure
  poisons/closes the store.
- `runtime-domain-storage-uses-exact-typed-facades`: Runtime, process, syscall,
  data-flow, and Tool orchestration use exact typed storage facades; persisted
  Object security projections are validated without materializing runtime-only
  payloads.
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
- `v3-persisted-state-is-strict-and-versioned`: a 0.3 store accepts only the
  frozen release version-3 physical schema (including typed process state) and
  canonical security carriers. Older, draft-v3, incomplete, or malformed state
  is rejected before mutation; no draft-v3 compatibility path is represented
  as a migration. Active wait cleanup and provider-effect recovery operate only
  on valid 0.3 state.
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
  the Human sink. Classifier failure uses the conservative contract ceiling.
  Terminal prompt reads and automatic-response writes also use structured
  pending intents; they retain only length/hash observations, never raw
  prompt, answer, or provider exception text. Human output provider failures
  likewise retain only the error type.
- `human-authority-and-evidence-commit-atomically`: Human requests, one-shot
  authority reservations, operation links, events, and audit commit or roll
  back as one unit, so an evidence-sink failure cannot publish a partial Human
  authority transition.
- `shell-and-jit-containment`: shell and Deno JIT execution stay policy-bound,
  including shell policy capability effects and finite-use leases, sandboxed,
  process-local, cached-only at runtime, and syscall-mediated. JIT lifecycle
  rows/aliases/handles commit atomically, composite failures discard unpublished
  candidates, and cancellation terminates the isolated Deno process group;
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
  metadata, while classifier failures finalize unknown evidence rather than
  dropping it. Follow-on finite object rights reserve/restore around the host
  call, and automatic child-exit cleanup records a close intent before exit-code
  observation/close. Object release finalizers run outside the SQL transaction so PTY
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
  audit views filter before applying their limit.
- `event-query-windows-are-store-bounded`: LLM context and GUI process-event
  reads apply cursor/filter and limit in the store, return the newest bounded
  matching window in order, and do not materialize an unbounded event history.
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
- `process-message-waits-are-race-free`: an empty blocking mailbox read and its
  wait registration are atomic with message posting, so a concurrent matching
  post either satisfies the read or wakes the registered process. Message row,
  evidence, terminal recheck, and wake state also roll back together.
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
  boundary, so one process cannot be re-entered concurrently. Terminal process
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
- `runtime-shutdown-is-drained-and-retry-safe`: scheduler work, ObjectTask
  executors, Human/provider blocking jobs, PTY reader/monitor workers, active
  admission leases, and GUI runtime users drain before shared state closes; a
  timed-out shutdown leaves storage open and can be retried. A checked public
  mutation inventory is installed under admission. Every public Human control
  method is classified as mutation or read-only; approvals, presentation,
  terminal draining, cancellation, and recovery are rejected at `STOPPING`
  before any durable or in-memory write. All 48 public CapabilityManager
  methods are likewise classified as read, mutation, or audit-sensitive mixed;
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
  development origins and exactly `agent-libos://app` for the packaged
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
- `llm-call-records-opt-out-are-bounded-and-redacted`: when
  `llm.persist_full_io` is false, LLM call persistence stores bounded preview,
  size, hash, and truncation metadata instead of raw prompts, tool arguments,
  reasoning, or provider responses. Pending conditional releases likewise
  persist only hashes and non-sensitive resume metadata before approval;
  same-runtime approval reuses the hash-bound in-memory request, while reopen
  fails closed without provider dispatch.
- `payload-retention-preserves-runtime-evidence-and-recovery`: payload retention
  is explicit, bounded, monotonic, and transactionally audited. It accepts only
  canonical provenance-bound content-free targets and does not erase evidence
  still required by live recovery.
- `llm-responses-state-chain-is-lossless`: OpenAI Responses state chaining
  is opt-in and preserves every representable native tool output, including
  parallel batches and reopened waits. Missing/extra/redacted outputs, changed
  profile/scope/context generation, or local rollback break the chain rather
  than continuing against guessed provider state.
- `llm-async-clients-are-event-loop-scoped`: real async SDK clients and their
  keep-alive pools are request-scoped and cannot cross scheduler event loops.
- `llm-provider-state-is-scope-bound-and-nonreplayable`: provider-chain state
  is bound to process/context plus a credential-keyed model, endpoint, API-mode,
  and tenant fingerprint. When a credential-bound fingerprint cannot be
  derived, requests remain stateless instead of reusing provider state. Durable
  waits use token-scoped pending/resuming/completed CAS and synchronize restored
  generations; an ABA
  claim, post-claim exception, or interrupted reopen fails closed and is never
  auto-replayed.
- `llm-profile-selection-is-process-local`: host-selected LLM profiles are
  stored as process-local ids, resolved at LLM-call time, inherited by child
  processes, preserved by image-package defaults, isolated from non-default
  ambient provider environment, and fail closed when the id is unknown.
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
  CPU, and RSS limits are enforced by providers and audited on exceedance; PTY
  supervision runs independently from output reads, accumulates observed CPU by
  process identity, and fails closed when process-tree accounting is denied.
  Cleanup falls back to explicit descendant signaling when process-group
  signaling is denied and serializes concurrent close attempts.
- `skill-activation-does-not-grant-authority`: Skills change visibility and
  prompt context without granting resources; API actor mode must still honor
  skill capability or human-approval gates. Finite-use Skill permissions are
  reserved before a registry, trust, activation, or unload mutation and are
  committed only with that mutation. Registry/trust/audit state changes are
  transactional; failed activation cannot leave a visible JIT alias, and
  reactivation or unload retires the exact superseded process-local JIT rows.
  Loaded-Skill provenance preserves a base/shared alias until its last actual
  source is unloaded; noncanonical persisted provenance is rejected before
  unload. Discovery rejects non-positive, boolean, and above-config limits.
- `runtime-modules-load-trusted-code-atomically`: startup Runtime Modules bind
  trust to the current source hash, reject ambiguous manifests and duplicate
  module ids, resolve import strings without executing untrusted package code,
  bound source hashing, and roll back failed declared or hook-created
  tool/image/syscall/hook registrations so persisted module status stays
  aligned with loaded runtime state. One shared runtime registry lock serializes
  the full module lifecycle with official Tool/Image publications, preventing
  a failed snapshot restore from clobbering a concurrent successful load.
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
  Legacy minimal flow carriers canonicalize during restore; incomplete active
  pending carriers fail closed through the retryable terminal lifecycle, while
  completed history receives conservative labels without failing its process.
  Post-checkpoint terminal ObjectTask history is retained as
  `superseded_by_restore` with stale live runner/result references cleared;
  the comparison uses the terminal transition time so delayed notification
  bookkeeping cannot supersede a task captured as complete. A normal store
  reopen also changes a persisted success with a missing runtime-only result
  payload to `result_unavailable_after_reopen` instead of leaving a dangling
  `result_oid`. Restore reserves checkpoint admin plus every changed-image
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
  The implementation uses one reservation scope for diagnostic
  inspect/diff/replay; the mapped one-shot regression directly proves inspect
  consumes the selected finite checkpoint/process read exactly once.
- `image-self-evolution-requires-image-authority`: image registration, package
  boot, exec, and checkpoint commit require image authority and do not bake
  external authority. Failed registration/commit removes new artifacts and
  restores replaced manifests; failed package boot/exec removes the exact
  publication-owned capability, receipt, private workspace, and unpublished
  JIT source/candidate state. Registry callers and
  getters receive isolated deep copies, concurrent same-id registrations are
  serialized and revalidated in the cache/store critical section, and
  committed-image boot does not overwrite the global Skill registry. New image
  publication requires `write`, replacement requires `admin`, and finite image
  plus checkpoint-read authority settles in the registry mutation transaction.
- `agent-output-is-not-control-channel`: untrusted command output cannot trigger
  lifecycle control actions; submission/exit must use explicit tool or syscall
  arguments.
- `jsonrpc-provider-effects-are-registered-and-classified`: JSON-RPC endpoint
  registration and calls use registered endpoint/method authority, gate calls
  and per-item registry operations before manifest metadata is exposed, and
  classify provider effects. Registry row/stale-grant/event/audit mutations are
  transactional, including finite registry-authority reservation/settlement.
  A call reserves finite authority and persists pending evidence
  before non-local DNS and transport; DNS observation prevents later transport
  PENS from erasing information flow or restoring the use.
- `mcp-provider-effects-are-registered-and-classified`: MCP server registration
  and tool calls use registered server/tool authority, gate calls before server
  metadata is exposed, and classify provider effects. Registry mutations are
  transactional. Refreshed tool listing/tool calls atomically reserve their
  deduplicated main, server, process-spawn, and exact stdio authority and persist
  pending evidence before DNS/live-provider boundaries. Local/stdio first-call
  PENS may restore; non-local DNS observation cannot be erased.
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
- `runtime-safety-benchmark-is-deterministic`: benchmark tasks and smoke runs
  remain schema-v1, deterministic, and token-free by default. Effect outcome
  and evidence are explicit; exact/prefix/glob matching cannot broaden
  implicitly; unknown/invalid/orphan/runner-failure output nulls rate fields;
  false-denial and unauthorized-effect denominators use only their documented
  qualified effect subsets.
- `practical-native-evidence-has-no-modeled-fallback`: native practical rows
  require real tool, provider-state, external-effect, and operation evidence;
  modeled rows stay in a separate denominator.

## Known Test Gaps

- The runtime-safety benchmark is an early deterministic workload, not a
  complete paper evaluation suite.
- Explainability tests verify provenance completeness and deterministic
  summaries, but do not yet measure whether operators understand explanations
  better in a user study.
- MCP Resources/Prompts remain planned but are not implemented.
