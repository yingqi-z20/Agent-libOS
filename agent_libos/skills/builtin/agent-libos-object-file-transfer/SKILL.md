---
name: agent-libos-object-file-transfer
description: Transfer text between the governed workspace filesystem and durable Object Memory. Use when file contents must become a lineage-tracked object, or an existing text object must be materialized as a file without returning its contents.
allowed-tools: create_object_from_file write_object_to_file
---
# Transfer files and Objects

## Workflow

1. Use `create_object_from_file` to import a bounded text file into a named immutable object; reject truncation unless partial content is explicitly acceptable.
2. Use `write_object_to_file` when a named object's payload is text or contains a text `content` field.
3. On import, verify the OID, source path, bytes read, and `truncated`; on export, verify the destination path, bytes written, and `created`.

## Boundaries and safety

- Paths are relative to the process working directory and remain workspace-contained.
- Import requires filesystem read plus object write; export requires object read plus filesystem write.
- Use ordinary read/write text tools when content is already in model context and no durable boundary transfer is needed.

## Verify

Inspect the destination namespace or filesystem location without assuming omitted transferred content was complete.
