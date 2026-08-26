# Evidence and LLM Payload Retention

Agent libOS keeps evidence rows and their causal identity. Payload retention is
a separate, explicit maintenance operation: it reduces selected terminal
provider payloads without deleting `llm_calls`, `external_effects`, audit/event
links, provider identities, effect identities, idempotency keys, timestamps, or
canonical argument hashes.

The implementation lives in
`agent_libos.evidence.payload_retention.PayloadRetentionMaintenance`. Its
default `PayloadRetentionPolicy()` is disabled. Runtime startup never runs
retention implicitly.

Durable Task Run retention is a separate lifecycle contract. Its default
`purge_on_terminal` policy hash-reduces Run-owned goal, follow-up, resume, and
persisted completion material while the Run is `finalizing`. The same
transaction invokes a dedicated terminal reducer for linked LLM request,
response, tool-call, and tool-output bodies and deletes pending continuation
actions. It also deletes durable messages automatically bound from their
Run-member recipient; ordinary callers cannot suppress, override, or forge
that binding. Run-linked Human request prompt, response, and decision bodies
are replaced by content-free hash projections; request id, type, status,
timestamps, audit linkage, and content digests remain. The transaction also
reduces every linked, terminal external effect's provider metadata and provider
receipt body through the canonical
`full -> summary -> hash_only` transitions. It does not run the age-based
`PayloadRetentionMaintenance` scan or delete Human, LLM, or effect rows. Effect
identity, state, classification, canonical-argument hash, original payload
digest, receipt digest, and causal links remain, but readable receipt content
does not. Nonterminal effects are never reduced. `permanent` skips this
automatic Run-terminal cleanup, although ordinary evidence-retention policy
remains independently applicable; a Host/admin may later invoke the same
audited purge explicitly for a terminal permanent Run.
See [Durable Task Runs](durable_task_runs.md).

## Monotonic tiers

Payloads move in one direction:

```text
full -> summary -> hash_only
```

- `full` is the value originally written by the provider boundary.
- `summary` is a content-free envelope. It contains only schema version, JSON
  kind, byte count, top-level item count where applicable, and the SHA-256 of
  the original canonical JSON. It contains no preview, keys, scalar values, or
  model-generated paraphrase.
- `hash_only` retains only the envelope schema/tier and original SHA-256.

The reducer preserves the original field hashes across both transitions and
stores an aggregate LLM payload hash. A maintenance pass cannot upgrade a row
or skip directly from `full` to `hash_only`.

External-effect tier and original payload digest are stored as dedicated
record-level provenance columns. At `full`, the digest column is necessarily
`null`: the first `full -> summary` reduction computes the original canonical
payload digest and records it alongside the new tier, and later reductions
preserve it. Provider metadata and receipts never establish their own retention
tier merely by resembling an internal envelope. New effect inserts and
provider finalization accept only `full`; the retention CAS is the only store
operation that can advance the tier or populate the digest column.

For LLM calls the content-bearing payload fields are messages, visible tools,
response content, tool calls, reasoning, the bounded provider-response
projection stored under `raw_response`, and the durable public/domain error
field. The `raw_response` name does not mean a byte-for-byte SDK object:
sensitive/opaque values are hashed and oversized structures are represented by
omission metadata and a digest before this retention state machine sees them.
Each present value is replaced by its
tier-appropriate content-free envelope; absent optional reasoning,
raw-response, or error fields remain `null`. Provider-invocation exception text
has a stricter boundary: it is normalized to a text-free public error before
any durable or model-facing sink, regardless of retention tier or
`llm.persist_full_io`.

A `ProviderTraceV1` is one possible value of the reasoning field, so reduction
applies to the whole trace—including every attempt's readable reasoning,
output, and tool arguments—in one transition. A content-free attempt-count,
coverage, status, and limit summary may remain in request options; it is not a
source from which the removed content can be reconstructed.

The entire `observability` mapping is also replaced; retention does not preserve
selected observability keys or field previews. Its replacement contains only
the retention schema version, tier, aggregate retained-payload hash, and a
SHA-256 of the original observability mapping. Thus prior observability content
such as trace details remains verifiable only by digest, not readable after
retention.

Call identity, process/image, purpose, provider/model/request/response ids,
request options, usage, status, and timestamps remain intact. For external
effects the provider metadata and provider receipt are reduced; ledger identity
and classification remain intact.

The LLM marker remains the authoritative payload provenance. Storage also
persists its current tier as a checked indexing projection. Inserts derive that
projection from the marker, retention compare-and-swap updates advance both in
one statement, and row decoding fails closed if they disagree. This permits an
exact cross-backend eligibility index without parsing provider-controlled JSON
inside SQL.

## Rows that are never reduced

External effects are eligible only when both conditions hold:

- `effect_state == finalized`; and
- `transaction_state` is `committed`, `failed`, or `compensated`.

`pending`, `prepared`, `authorized`, `approved`, `dispatched`, and `unknown`
effects are never reduced. The service checks this before mutation, and the
backend compare-and-swap must repeat the state check in the update statement so
a reconciliation race cannot trim a newly nonterminal row.

LLM calls are eligible only after `status` is `ok` or `error` and
`completed_at` is durable. The following terminal records remain protected
runtime dependencies until their executable semantics have a separate durable
projection:

- the unique latest successful call for a `(pid, purpose)` stream, when it is an
  `image_only` `action_selection` candidate carrying a schema-v2 marker. Runtime
  replay additionally requires exact paired tool outputs or, for an empty-call
  response, a separate successful action-validation marker; retention protects
  an unvalidated candidate conservatively rather than deleting possible
  recovery evidence;
- the unique latest error in an exact
  `image_only_request:<anchor-fingerprint>` stream when its schema-v1 marker
  carries the two-message system/goal request needed to retry a first provider
  failure after reopen;
- the unique latest OpenAI Responses call for a `(pid, purpose)` stream, when
  that call is eligible to supply low-level provider-side continuation state;
  the current AgentProcess executor does not create an eligible provider-chain
  head, but the storage predicate remains fail-closed for compatible rows; and
- a tool-call record carrying the `process_exit` payload used to recover a
  missing context-compressor result.

The second rule also protects a legacy, truncated tool-call observation whose
contents cannot be proven safe to discard. This is intentionally conservative:
retaining an old payload is preferable to breaking a live/pending continuation
or silently losing recovery state.

The head decision exactly matches each runtime lookup stream. Responses
continuation and a first-attempt fingerprinted request stream use the greatest
`(created_at, call_id)` among all rows with the same `(pid, purpose)`, without
filtering status first; a newer pending or failed Responses call therefore
supersedes an older successful call. The image-only transcript instead uses the
greatest successful `action_selection` row, so neither a legacy newer
same-purpose error nor a current `image_only_error` row shadows its last complete
head. Image-only provider errors instead use `image_only_error`. All of those
comparisons use the backend's bytewise keyset collation.
When a successful action-validated transcript becomes canonical, a content-free
tombstone is attempted afterward. If it becomes the latest row in that request
stream, the older full request can advance through the configured retention
tiers. A tombstone write failure is audited without discarding the canonical
successful transcript or paired tool outputs, and the old request remains
protected. The same conservative protection applies if the Host terminalizes
the process before any success.

A row without a process id cannot be selected by these runtime lookups and is
therefore not a continuation/transcript/request head. The bounded storage query
classifies heads with an indexed correlated seek; it does not issue one lookup
per candidate. The retention CAS repeats the "a newer call exists" fence before
reducing any guarded row classified as non-head.

## Explicit maintenance API

Runtime configuration is the Host-owned policy source. Retention is disabled
unless it is explicitly enabled with a summary age; the hash-only age and page
limits are optional:

```yaml
runtime:
  payload_retention_enabled: true
  payload_retention_summary_after_seconds: 2592000
  payload_retention_hash_only_after_seconds: 7776000
  payload_retention_page_size: 100
  payload_retention_page_hard_limit: 1000
```

`RuntimeBuilder` derives one immutable policy from those exact settings and
exposes lifecycle-gated maintenance at `runtime.payload_retention`. A caller
submits one dataset per request:

```python
from agent_libos.evidence.payload_retention import (
    PayloadRetentionKind,
    PayloadRetentionRequest,
)

preview = runtime.payload_retention.run(
    PayloadRetentionRequest(
        kind=PayloadRetentionKind.LLM_CALL,
        dry_run=True,
    )
)
```

The CLI performs one page at a time and defaults to preview. Mutation requires
both enabled configuration and the explicit `--apply` flag:

```console
agent-libos --config config.yaml --db user payload-retention llm_call
agent-libos --config config.yaml --db user payload-retention llm_call --apply
agent-libos --config config.yaml --db user payload-retention external_effect --apply
```

Use the returned `next_cursor.created_at` and `next_cursor.record_id` as
`--after-created-at` and `--after-record-id` for the next page. An optional
`--limit` bounds the page within the policy hard limit. `--actor` overrides the
default `host.retention` actor identity, and `--correlation-id` attaches a
Host-supplied correlation id; both are recorded on the audited maintenance
summary. Startup only
constructs the maintenance service; it never invokes a scan or mutation.

Every call is bounded. A page is ordered by `(created_at, record_id)` and may
return an opaque keyset cursor for the next call. A request cannot exceed the
policy hard limit. `dry_run=True` performs the same eligibility decisions but
does not update a payload.

Retention ages are measured from terminal evidence, not from the page-order
cursor. For an LLM call the age anchor is its durable `completed_at`. For an
external effect it is `updated_at` (falling back to `created_at` only when the
former is absent), which normally represents the finalization update. Rows with
a missing, invalid, naive, or future terminal timestamp are not advanced. The
storage scan may use the older `created_at` as a coarse indexed cutoff, but the
maintenance service always repeats the exact terminal-age decision above.

The SQL adapters push the coarse age cutoff, terminal-state predicate, and
`full`/`summary` tier predicate into the keyset query. Consequently durable
nonterminal and `hash_only` history does not consume a page and is not scanned
again on every maintenance run. The service repeats all terminal, runtime
dependency, timestamp, and monotonic-tier checks before planning a mutation;
the optimization therefore narrows storage candidates without weakening the
retention safety rules.

Every disabled, dry-run, and applied request writes one
`evidence.payload_retention.maintenance` audit summary. The summary contains
only counts, policy ages, an SHA-256 of the candidate-id set, and an SHA-256 of
the next cursor. Candidate ids and source payloads are not copied into audit.
Applied updates and that audit row run under the same store transaction, so an
audit failure rolls back the batch.

## `persist_full_io` compatibility

`llm.persist_full_io=true` remains the write-time choice for full LLM I/O.
New `persist_full_io=false` rows are written directly at the `summary` tier:
every content-bearing field and durable error field is replaced by the
canonical content-free envelope above, and `observability` contains only the
aggregate retention marker. No prompt, schema, response, tool-call argument,
reasoning, provider payload, error text, key name, scalar value, or preview
remains readable in those content-bearing or observability fields. The
identity, policy, accounting, status, and timestamp fields listed above remain
durable and may themselves contain ordinary field names and scalar values.
Provider-invocation exception text is never durable even when
`persist_full_io=true`; the setting authorizes retention of successful
request/response content, not raw exception messages.

An `image_only` process is not an exception that writes a redacted transcript.
It requires a lossless native transcript head and fails before provider dispatch
when `persist_full_io=false`, so no new action-selection call row is produced by
that attempted quantum. Runtime-owned prompt modes can use the opt-out normally.

The same setting controls a separate root-spawn startup aid. A committed root
spawn stores a size-bounded, integrity-bound recovery envelope for its immutable
initial goal in the internal publication receipt only when
`persist_full_io=true`; ordinary
publication reads expose a hash-only projection. `persist_full_io=false` writes
only hashes. Startup may use a still-full envelope only for the exact matching
live nonterminal root goal, and terminal root cleanup redacts it. This envelope
is not an `llm_calls` row and is not scanned or aged by
`PayloadRetentionMaintenance`; backups and direct database access can still see
it until that explicit lifecycle redaction occurs. Failed-launch rollback and
startup recovery claim a non-committable publication state only in the same
outer transaction that reduces a full envelope to hashes; redaction failure
rolls the state transition back for a fenced retry.

Opt-out rows written by earlier releases can contain bounded observation
previews. The retention service recognizes those legacy rows as the summary
tier and can normalize them to the canonical content-free summary envelope;
normalization also replaces the complete legacy `observability` mapping with
the retention marker described above. It never restores missing content. A
legacy truncated tool-call preview remains protected when runtime-dependency
safety cannot be proven.

Retention is therefore an additional staged minimization policy, not an
override that can weaken `persist_full_io=false` or reconstruct redacted data.
It does not promise a maximum deletion deadline for protected runtime
dependencies such as a first failed image-only retry anchor that never reaches
a successful transcript. Operators should normally run and inspect a dry-run
page before applying it.

## Backend contract

SQLite and PostgreSQL adapters implement the typed
`PayloadRetentionStore` protocol:

- keyset scans for LLM calls and external effects accept `older_than`, `after`,
  and `limit` and return at most the requested limit;
- each LLM page identifies every continuation-, transcript-, or image-only
  retry-request-capable candidate that is the actual latest row under its
  runtime rule: latest-any for Responses/request streams and latest-successful
  for an image-only transcript stream;
- update methods compare the expected record-level tier and aggregate payload
  hash;
- the external-effect update also compares `effect_state` and
  `transaction_state` and only accepts terminal values; and
- all mutations use the caller's transaction, allowing the audit row and batch
  to commit atomically.

Both adapters create partial composite indexes over the terminal, non-
`hash_only` population. Their leading `(created_at, stable_id)` keys satisfy
the keyset order. Resumed pages express the lower bound as the SQL row-value
comparison `(created_at, stable_id) > (?, ?)`, allowing the adapter to seek to
the cursor instead of filtering the already-visited index prefix. The remaining
predicate columns make the candidate lookup covering. SQLite selects the matching partial index explicitly;
PostgreSQL receives the same query after its dialect removes that SQLite-only
planner hint. The bounded candidate lookup then joins at most `limit + 1`
primary-key rows to materialize complete records. No implementation may satisfy
this contract by loading all historical rows into Python.

LLM head classification uses
`idx_llm_calls_provider_chain_head(pid, purpose, created_at, call_id)` for one
correlated seek per bounded candidate. The same index backs the atomic update
fence that rejects reducing a Responses continuation, `image_only` transcript
head, or image-only retry-request anchor if no newer call exists. The historical
index name covers all three kinds.
