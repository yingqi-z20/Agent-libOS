# Explainable Operations

Explainable Operations is the host-side provenance query layer for protected
runtime work. It answers what ran, why authority allowed or denied it, which
human decision or finite-use reservation participated, whether a provider
effect was finalized or remained uncertain, and which resource/context records
were produced.

The explanation is deterministic. It uses persisted identifiers and typed
links; it does not call an LLM and does not associate records merely because
their timestamps are close.

## Operation Tree

Every record has an `operation_id`, a causal root, and an optional parent:

```text
llm_request
  -> tool_call
     -> syscall
        -> primitive
     -> primitive
  -> runtime operation
```

One LLM action-selection cycle is one logical `llm_request`; provider repair
attempts are separate `llm_call` evidence under that operation. Parallel tools
are sibling operations. A protected host call without an active parent becomes
a root operation.

Human, child-process, and message waits set the affected LLM/tool operations to
`waiting`. Durable pending actions store both ids, and resume re-enters those
same operations even though a new concrete tool-call attempt may receive a new
`call_id`. ObjectTask execution links its background tool operation to the
persisted task-start operation; Human resume reuses the operation linked to the
request.

At runtime reopen, durable runtime publications are reconciled before generic
stale-operation handling. A linked publication is authoritative:
`committed` maps to `succeeded`, `rolled_back` maps to `failed`, and a failed
or manual compensation maps to `unknown`. Publication finalization and this
operation outcome are written in the same store transaction. Recovery is
idempotent and may correct a previously terminal operation written on the
wrong side of a crash window. Generic stale handling then marks a remaining
`running` operation `unknown` when it or a descendant has a pending or unknown
external effect; a stale `running` operation without that uncertainty becomes
`interrupted`, and a durable `waiting` operation remains waiting. If the
publication terminal transaction itself fails, the publication remains
nonterminal and its exactly linked operation remains `running` for recovery
instead of being independently finalized by the generic operation wrapper. An
unlinked or mismatched pending-publication signal cannot suppress ordinary
terminalization.
Launch/exec terminal reconciliation and committed checkpoint-restore operation
repair are index-backed and hard-bounded by keyset pages. Failed/manual
checkpoint restores remain forward-recovery inputs. Online terminal
transactions durably set the reconciliation marker, so
reopen visits only rows still requiring repair; returned diagnostic id lists
are bounded even when recovery processes a larger backlog. A successful
RuntimeStore mutation of a bound operation clears that marker in the same
transaction, so the next reopen revalidates the changed contract without
rescanning settled history.

Checkpoint-restore plans are fully specified at insert and anchored by a
receipt-side digest. Generic Host RuntimeStore writes cannot mutate their plan,
receipt transcript, recovery lease, or operation marker; the storage-owned
restore writer performs only validated state-machine transitions. Recovery
checks the anchor and ordered causal transcript before phase/finalizer replay
or committed-operation repair.

The exact link is created during publication planning, in the same store
transaction that inserts the publication: `plan.operation_id` points to the
operation, while a normalized uniquely indexed column records the reverse
publication id and operation metadata records its id, kind, and versioned
durable binding marker. The normalized value and metadata must agree. The
plan-side operation id and binding version become immutable after binding, and
the reverse association must resolve to exactly one operation through the typed
repository lookup. Reconciliation never creates this association for an
unbound row. A blank or missing operation id, a missing operation, a fully
matching but unbound operation, multiple reverse bindings, a changed
kind/name/actor/PID, or an operation already bound to another publication fails
reopen closed without rewriting the operation. As with the rest of RuntimeStore
evidence, this is an application integrity contract rather than protection from
a database administrator who bypasses RuntimeStore and edits an already-settled
row directly.

Online `spawn`, `fork`, and `spawn_child` commit their process transition,
event/audit evidence, publication receipt, and successful operation outcome in
one terminal transaction. A sink failure rolls all of it back before exact
compensation; rolled-back/failed receipts are likewise atomic with failed or
unknown operation outcomes. If that terminal sink also fails, the publication
and operation stay nonterminal, mutation admission is fenced until reopen, and
retry cannot leave a duplicate process. For root `process.spawn`, a pre-return
crash may initially leave the operation PID unset; exact prebinding authorizes
the terminal transaction to canonicalize it to the publication's child PID.

An uncertain pending provider
effect makes the affected primitive/tool and enclosing LLM outcome `unknown`;
this is distinct from missing evidence. The same rule applies after settlement
has finalized an effect with `transaction_state=unknown`; bookkeeping
finalization does not turn an uncertain provider outcome into an ordinary
failure.

## Evidence

`operation_evidence` explicitly relates operations to these roles:

- invocation and result;
- capability decision and finite-use reservation;
- Human approval or wait;
- provider effect;
- resource charge;
- context manifest;
- event and audit.

Data-flow decisions are append-only evidence as well. Their linked audit/event
records carry the decision id, Sink/trust hashes, registry generation, source
references, label summary, and release id without copying the payload. A
pre-provider data-flow denial therefore remains explainable even though no
external-effect intent exists and no ordinary finite-use capability was
consumed.

Audit and event managers attach every record emitted inside an active operation.
Capability, Human, external-effect, ToolBroker, LLM, ObjectTask, and context
code additionally link their own durable identifiers. A uniqueness constraint
on `(operation_id, evidence_type, evidence_id, role)` makes repeated attachment
idempotent. Evidence pagination groups all roles for one evidence identity
before applying the page limit, so an audit carrying both `audit` and
`decision` roles is never split into duplicate timeline rows.

Each operation carries `expected_roles`. Authorization adds `decision` only
when a real capability decision is made. Crossing a provider boundary adds
`effect`, `event`, and `audit`. Therefore an operation denied before the
provider is not incorrectly reported as missing effect evidence.

The [Protected Operation SDK](protected_operation_sdk.md) declares these roles
from the registered contract and links the prepared/finalized effect to the
same primitive operation. Multi-phase DNS, validation, transport, and cleanup
steps do not rely on timestamp correlation or create competing root causes.

`evidence_complete` means all declared roles have at least one explicit link.
It is provenance completeness, not a security or semantic-quality score.
`missing_evidence` names the operation and role; `uncertainties` separately
reports waiting, interruption, unknown outcome, or a pending provider effect.

## Context Materialization Manifest

Each LLM action-selection records one metadata-only manifest containing:

- process/view id, policy, effective token budget, and context generation;
- optional final LLM-context Object id/version, final rendered token count, and
  final rendered SHA-256 (the default source-only path records null Object
  identity because it does not create one);
- each source-view candidate's Object id/version/type;
- included/omitted disposition and reason (`selected`, `filter_mismatch`,
  `capability_denied`, `token_budget`, or `missing`), plus the optional appended
  final context entry with reason `llm_context`;
- per-entry security `labels` (`sensitivity`, `trust_level`, `integrity`,
  `origin`, `tenant`, `principal`, and `declassification_authority`) when the
  source was readable;
- transformation (`verbatim`, `compacted`, or `truncated`) and per-entry
  rendered token/hash metadata; and
- final-context compaction mode, timestamp, and transformation.

The default source-only path records the caller-selected source entries without
an appended final-context entry. When persistent context enrichment is
explicitly enabled, the source entries and final-context entry have different
roles. Source entries describe the Object snapshots considered by
`memory.materialize_context`; their per-entry token/hash values describe those
rendered source chunks. The Runtime then updates or creates the process's
append-only LLM-context Object and appends
a separate included entry whose reason is `llm_context`. Top-level
`context_oid`, `context_version`, `rendered_tokens`, and `rendered_sha256`
identify that final Object snapshot and the LLM-context text prepared for the
provider request. Before source materialization, the Runtime removes the
process's existing LLM-context Object from the view, so the appended final entry
does not duplicate an earlier source-candidate entry for that Object.

Labels are metadata, not payload copies. An omitted `missing` or
`capability_denied` entry can have `labels=null` because the materializer could
not safely read the Object metadata. The `llm_context` reason is distinct from
the source-selection reasons and is not evidence that every source candidate
was included.

Built-in LLM context producers write only the documented metadata projection;
they do not copy Object payloads, rendered prompt text, Human answers, or
provider responses. Explain applies the same allowlist projection. The lower
level persisted model remains an open list of mappings for compatibility, so
this is a producer/Explain guarantee rather than a database-level closed-schema
guarantee for arbitrary trusted Host writers. Direct
`memory.materialize_context` calls return the same source-candidate selection
metadata in memory, but do not append the final `llm_context` entry or create a
durable manifest row. Durable rows are created only when the final LLM context
is prepared for an LLM action-selection.

## Host Interfaces

CLI:

```bash
agent-libos --db <store> explain process <pid>
agent-libos --db <store> explain operation <operation_id>
agent-libos --db <store> explain call <call_id>
agent-libos --db <store> explain effect <effect_id>
agent-libos --db <store> explain request <request_id>
agent-libos --db <store> explain audit <record_id>
agent-libos --db <store> explain event <event_id>
agent-libos --db <store> explain reservation <reservation_id>
agent-libos --db <store> explain context <materialization_id>
```

HTTP:

- `GET /api/operations?pid=...&limit=...&cursor=...`
- `GET /api/operations/{operation_id}?evidence_limit=...&cursor=...`
- `GET /api/operations/resolve?kind=...&id=...`

The process list returns causal roots; operation detail expands the selected
root into all explicitly linked descendants.

These two pagination boundaries have different resource semantics. The process
root list applies its cursor and limit in the RuntimeStore with a bounded
keyset query. Operation-detail `evidence_limit` and `cursor`, however, are
presentation bounds on the returned evidence page: the current implementation
first materializes the complete causal operation tree and its evidence links,
then groups, sorts, and slices that in memory. They do not make detail lookup
store-bounded. Very large causal trees are therefore a Host resource boundary;
deployments must control their size and must not treat the response page limit
as a database-work or memory cap.

No match returns `404`. An evidence id that maps to multiple causal roots
returns explicit candidates (`409` over HTTP); the runtime never chooses by
time or similarity.

The Electron GUI provides an Explain tab with outcome/completeness summary,
causal tree, filtered evidence timeline, and pagination. Audit, event, LLM, and
Human timeline entries can open their linked operation. Snapshot/SSE changes
remount the panel against current evidence.

## Visibility and Redaction

Explain is available only to host CLI and the authenticated local GUI API. It
is not a model tool or process syscall.

Responses preserve routing ids, statuses, rights, targets, hashes, counts, and
rollback classification. They redact typed payload-like fields, credentials,
Human content, LLM raw I/O, Object payloads, raw command arguments and
environments, stdout/stderr, and provider metadata. Nested projections also
apply the shared syntax-aware filter for known credential keys, scalar
credential forms, URI/DSN userinfo, and HTTP Cookie headers. This filter is
defense in depth: it cannot infer that every arbitrary value under an unknown
field name is sensitive, so evidence producers must use typed sensitive fields
or explicitly sanitize provider-specific secret formats. Explain rendering
leaves the original audit and external-effect records unchanged; it does not
rewrite their source fields while producing a redacted projection.

Unlinked rows are not backfilled or heuristically reconstructed. The frozen
schema-v7 RuntimeStore contract requires the explanation tables and explicit
links; an older or incomplete store is rejected before mutation.
