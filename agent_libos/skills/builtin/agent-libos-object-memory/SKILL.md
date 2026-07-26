---
name: agent-libos-object-memory
description: "Create, discover, read, and append governed runtime-local Object Memory state. Use for named JSON that must survive prompt or compaction changes within one live Runtime; use files or captured checkpoints for reopen durability."
allowed-tools: create_memory_namespace list_memory_namespace create_memory_object read_memory_object append_memory_object
---
# Manage runtime-local Object Memory

Payloads survive prompt/compaction changes only in the live Runtime. Metadata can outlive missing content after reopen; export or explicitly capture required payloads.

Names are namespace-local lookup, not authority: trimmed, nonempty, not `.`/`..`, with no `/` or `\`. Omitted namespace means process scope.

Tool schema visibility comes from Skill projection. Within these tools,
namespace/Object capabilities govern discovery and operations; prompt visibility
separately uses MemoryView roots/filters, materialize rights, and budget. Listing
is not a view query. Only `create_memory_object` auto-adds its handle as a root.

## Tool guide

### `create_memory_namespace`

Creates one slash-delimited `namespace`; `parent_namespace` defaults to its path parent. Create parents first. Top-level needs no parent right; child needs existing parent write. Duplicate fails, not ensure/upsert. `metadata` grants nothing. Output is `namespace`, `parent_namespace`, `created=true`.

### `list_memory_namespace`

Lists readable Objects and direct readable child namespaces. Never broaden an omitted/exact scope. Bounded `limit` is shared: Objects consume it first, then children. No cursor/total/completeness exists; a full page or missing name does not prove absence.

Object and direct-child namespace discovery use bounded keyset pages under a separate configured scan ceiling. If that ceiling is reached before the requested result is complete, listing fails explicitly; narrow the namespace or limit instead of treating the failed scan as evidence of absence.

Objects expose OID/namespace/name/type/version; children expose namespace/parent. Namespace and entry read rights are enforced and finite grants may be consumed. Payload, metadata, labels, immutability, provenance, and view membership are absent.

### `create_memory_object`

Creates typed JSON and attaches its handle to the caller's MemoryView. Required: `type`, direct JSON `payload`. Optional: name, namespace, metadata, parents, `immutable` (true).

- Valid types are `task`, `goal`, `plan`, `step`, `constraint`, `message`, `human_decision`, `human_request`, `tool_result`, `observation`, `error_trace`, `code_patch`, `test_result`, `evidence`, `claim`, `summary`, `skill`, `tool_spec`, `tool_candidate`, `tool_artifact`, `checkpoint`, `process_state`, `external_ref`, and `artifact`. Type classifies; it does not impose a payload schema.
- Pass containers directly, not encoded strings. A JSON-looking string is still
  a string and is stored literally; use an actual object/array value when a
  container is intended. Payload is bounded. Choose a stable name for later
  read/append/transfer/task use.
- Set `immutable=false` at creation when append/writable ownership is needed. Use a `plan` with `{"entries":[]}` as a ledger.
- `metadata` accepts only `title`, `summary`, `tags`, `mime_type`, `sensitivity`, `retention_policy`, `trust_level`, `integrity`, `tenant`, and `principal`. Fields and collection items are strictly typed and bounded by configured character, item-count, and canonical-byte limits. Origin/token estimate are derived; models cannot declassify or elevate trust/integrity.
- `parent_oids` are readable source Objects, never paths/call IDs. They supplement observed sources; invalid parents reject. Runtime controls labels/provenance.
- Needs namespace write plus parent read. Output is OID/namespace/name/type, not version, metadata, labels, provenance, or mutability.
- Object publication, its capability/audit evidence, and caller MemoryView root
  attachment share one transaction. A view-attachment failure rolls the create
  back instead of leaving an unreported named Object.

### `read_memory_object`

Reads exact name as canonical JSON. `json_pointer` selects RFC 6901 subtree (`""` whole; `~0` for `~`, `~1` for `/`). `max_payload_chars` is a bounded character count.

If it fits at cursor 0, representation is `json_value` with typed `payload`. Otherwise it is `canonical_json_page` with exact text `preview` and no payload. Cursor is a UTF-8 byte offset: use only `next_cursor`, and pass first-page `sha256` as `expected_sha256` on continuation.

Output includes identity/version/pointer, type/shape, `serialized_bytes`, `sha256`, page offsets, `truncated`, `omitted_bytes`, and `next_cursor`. Concatenate previews under one digest, then parse; a fragment is not JSON evidence.

### `append_memory_object`

Appends direct JSON `entry`; JSON-looking strings remain literal strings.
`list_field` defaults to `entries`. On object payload, a missing field is
created (typos matter) and non-list fails. Root lists ignore the field and
return null. Scalars/immutable Objects fail.

Needs namespace read and Object read/write; inherits flow labels/provenance and size limits. Output is identity, new version, `appended=true`, field, and length—not content. No caller CAS exists; internal conditional update can conflict. Version is not append count.

## Recommended workflow

1. Call `list_memory_namespace` on the exact scope to discover names, OIDs, types, and versions. Treat a limited result as partial.
2. Create missing namespaces parent-first. Do not retry duplicate creation as an ensure operation.
3. Create stable direct JSON with accurate type/parents and explicit immutability. Ledger cumulative requirements, decisions, evidence, and blockers.
4. Read before acting. Select a subtree or collect every page with cursor plus constant digest; never reason from a fragment.
5. Before append, verify exact namespace/name, payload shape, target list field, and whether the entry is already present. Append once, then retain returned version/length.
6. Before exit, re-read ledgers and map every requirement to evidence/blocker; arrange reopen durability separately.

## Failure and recovery

- Namespace/Object duplicate: list and read the exact existing name. Reuse only after verification; otherwise choose a deliberate new name. Blind retry cannot upsert.
- Missing/unreadable parent: correct the OID or authority. Never drop true provenance merely to make creation pass.
- Too large: split named chunks or make parent-linked summaries; never hide loss or encode JSON to evade limits.
- Paged read: follow cursor with pointer/digest. On mismatch restart at zero; correct invalid pointers/cursors instead of guessing.
- Append conflict/ambiguity: re-read relevant content/version and retry only if entry is absent. Blind retry duplicates data.
- Wrong field: preserve the mistaken append and correct the design; do not pretend it did not occur.
- Permission/label denial: obtain appropriate authority or change the authorized design. Names and Skill activation grant nothing; never route through another namespace/process to evade policy.
- Runtime reopen with missing payload: recover from an authoritative file or captured checkpoint/image. Do not infer content from surviving metadata, version, or OID.

There is no wait, rollback, arbitrary update, delete, rename, view-management, or cross-runtime restore operation in this Skill. Stop with an explicit blocker when state cannot be safely reconciled.

## Completion evidence

For namespace creation, retain returned namespace/parent and `created=true`. For listing, retain exact scope and limit and label completeness as unknown when a page is full.

For creation, retain returned identity/type plus submitted immutability/parents; do not claim unreturned labels. For reads, retain identity/version, pointer, representation, digest, byte spans, and completion. Large values need every contiguous page under one digest.

For append, retain OID, namespace/name, new version, list field, length, and an authorized read-back showing the intended entry exactly once when duplication matters. Returned `appended=true` proves that append committed, but not the entire resulting content.

For long tasks, map every original/later requirement to evidence or blocker. Live state meets no reopen requirement until exported/captured and verified.
