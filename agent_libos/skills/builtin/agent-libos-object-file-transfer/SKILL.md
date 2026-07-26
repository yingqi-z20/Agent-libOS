---
name: agent-libos-object-file-transfer
description: "Import governed workspace text into Object Memory or export Object text to a file. Use for content boundary transfer, not inspection, editing, binary data, or byte-exact copies."
allowed-tools: create_object_from_file write_object_to_file
---
# Transfer text between workspace files and Object Memory

Both input paths are relative to the caller's working directory and must remain inside the workspace. Returned paths are canonical workspace-relative evidence; if cwd is not the workspace root, do not reuse one as though it were cwd-relative.

Imported payloads survive prompt/compaction changes only in the live Runtime; ordinary payloads vanish on reopen unless explicitly captured. Exported files are workspace effects that restore does not undo.

## Tool guide

### `create_object_from_file`

Creates one named immutable Object from a governed text file. Required inputs are `name` and cwd-relative `path`; optional inputs are `namespace`, `encoding`, `max_bytes`, `allow_truncated`, and `object_type`.

- Omitted `namespace` means the process namespace. It must already be writable, and `name` must be unused there. Creation never overwrites or upserts.
- `encoding` defaults to the runtime text encoding, normally UTF-8. A wrong codec or undecodable file fails. This tool decodes text; it is not a binary reader.
- `max_bytes` bounds the read. Its exposed default/maximum is the lower of the Object-file settings and the filesystem primitive read hard limit (1 MiB under default configuration), even if the separate Object-file absolute limit is higher. With `allow_truncated=false`, overflow fails before creation; true stores a decoded prefix, possibly omitting an incomplete final multibyte character.
- A second gate covers the full Object payload. False rejects overflow; true
  shortens `content`, marks `truncated=true`, and adds `stored_bytes`, so source
  `bytes_read` may exceed stored content bytes.
- `object_type` defaults to `artifact` and must be a valid Agent libOS Object type, not a MIME type or free-form schema. It classifies; it does not validate payload shape.
- Stored fields include `kind="workspace_text_file"`, `source_path`, `encoding`,
  `content`, `bytes_read`, and `truncated`; second-gate truncation also stores
  `stored_bytes`. The immutable Object is not attached to the MemoryView, cannot
  be appended, and does not expose text on success.
- Success returns `oid`, namespace/name/type, `source_path`, `bytes_read`, and `truncated`—no content, hash, stored length, provenance, or labels. `bytes_read` is not equality evidence after payload fitting.
- Import needs exact-file read and namespace write. Read/grant consumption may precede duplicate-name, type, or size failure; the boundaries are not one transaction.

### `write_object_to_file`

Writes text from one named Object to a governed file. Inputs are exact `name`, optional `namespace`, cwd-relative `path`, optional `encoding`, and `overwrite` (schema default true).

- Resolve by namespace-local name, not OID. Omitted `namespace` means the process namespace; name lookup grants no authority.
- Payload must be a string or an object whose top-level `content` is a string. Lists, scalars, null, and arbitrary objects are rejected rather than JSON-serialized.
- Export `encoding` is independent. The tool does not reuse an import's recorded encoding; equal Unicode text can produce different bytes.
- Prefer `overwrite=false` for new destinations and uncertain retries. Use true only for an intended replacement. Missing parent directories may be created; returned `created` describes only the final file.
- Export needs namespace/Object read, exact-file write, and allowed label egress. Object read or finite grant consumption may precede a later write/approval failure.
- The Host revalidates the exact source Object snapshot immediately before
  egress. If its version/content changes after lookup, export fails before the
  file write rather than writing a stale snapshot.
- Success returns `oid`, `namespace`, `name`, canonical `path`, `bytes_written`, and `created`. It returns no text, digest, fsync evidence, or comparison with an original source.

Activation grants no authority. For binary/byte-exact transfer, use an explicitly visible Host byte tool and its Skill; if none exists, stop.

## Recommended workflow

For import:

1. Decide whether the product is an Object reference. If the model must read text now, use workspace-navigation instead.
2. Confirm cwd, relative path, encoding, completeness, writable namespace, unused name, and Object type. Do not guess or silently accept truncation.
3. Call `create_object_from_file` once with `allow_truncated=false` by default and a justified `max_bytes`. If the known input exceeds the hard limit, use an accepted chunking/domain method.
4. Record OID and namespace/name. Require `truncated=false` for completeness. If
   partial import was intentional, activate object-memory and read `stored_bytes`
   plus `content`; the bridge result alone exposes neither. Use object-memory to
   inspect any imported content.

For export:

1. With object-memory, confirm exact namespace/name and a string or top-level string `content`. A result OID is not a name.
2. Choose path and encoding explicitly. Inspect destination existence and normally pass `overwrite=false`.
3. Call `write_object_to_file` once. If equality matters, perform authorized read-back; bridge metadata cannot prove it.
4. Treat the file as a persistent effect. Arrange cleanup only when separately authorized; restore is not cleanup.

A round trip can preserve stored Unicode text while changing encoded bytes. It cannot satisfy a byte-exact or binary-copy criterion.

## Failure and recovery

- Duplicate name: do not blind-retry. Inspect the exact existing Object; reuse only if verified, otherwise choose a deliberate new name or stop.
- Ambiguous import: the file may have been read and an Object may exist. Reconcile the exact namespace/name before another non-idempotent creation.
- Truncation/size: raise `max_bytes` only within bounds or use approved chunking/summary. Never report a prefix as complete.
- Decode failure: establish the real codec; do not cycle encodings merely until one stops erroring, because that can silently corrupt text.
- Unsupported export payload: read it and explicitly create an authorized text projection. Do not claim arbitrary JSON serialization.
- Existing destination: with `overwrite=false`, inspect and compare. Finish without writing if it is already correct; otherwise obtain clear replacement intent.
- Ambiguous export: inspect the exact destination before retrying and retain `overwrite=false` unless replacement was intended. Timeout is not evidence that no file was written.
- Source snapshot conflict: re-read the exact Object version/content and retry
  only after deciding that the new snapshot is the intended export. The failed
  stale-snapshot attempt does not create or replace the destination.
- Authority/approval/label denial: correct authority or destination; never reroute to evade policy.
- Runtime reopen: recover imported content from the authoritative file or a checkpoint/image that captured it; do not infer vanished payload from durable OID metadata.

There is no wait, resume, rollback, or cross-boundary transaction tool here. If inspection cannot distinguish whether a non-idempotent transfer occurred, stop with an explicit unknown-state blocker.

## Completion evidence

For complete import, retain `oid`, namespace/name, `source_path`, encoding/type, `bytes_read`, and `truncated=false`. This proves bounded text import, not byte identity, prompt visibility, hash equality, or reopen durability. Record the limit and label every `truncated=true` result partial.

For export, retain Object identity, canonical destination `path`, chosen `encoding`, `bytes_written`, `created`, and whether overwrite was authorized. This proves a governed text write, not content equality or durable-storage flush.

When exact content is an acceptance criterion, add independent authorized read-back evidence: compared identities, method, and matching text/digest. When reopen durability matters, the verified workspace file or separately verified checkpoint/image is the durable artifact—not the runtime-local Object alone.
