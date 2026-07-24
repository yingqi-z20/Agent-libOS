---
name: agent-libos-object-memory
description: Create namespaces and typed durable Object Memory, list or read visible named objects, and append to mutable objects. Use for structured working state, lineage-aware artifacts, or results that must survive prompt changes.
allowed-tools: create_memory_namespace list_memory_namespace create_memory_object read_memory_object append_memory_object
---
# Use Object Memory

## Workflow

1. Reuse the process namespace unless a durable directory-like scope improves organization.
2. For a long task, create a mutable acceptance ledger before external effects: preserve the original goal, explicit deliverables, verification steps, and open items.
3. Merge acknowledged human follow-ups into the ledger as deltas. They change only stated requirements unless they explicitly replace or cancel prior ones; record message IDs when available.
4. List only the current/exact namespace; never broaden to parent `process`. Omit `limit` or keep it within the runtime maximum (normally 50). Runtime-only goal objects may be released after reopen; recover the goal through nonterminal completion review, not a guessed memory read.
5. Create typed objects with direct JSON payloads. Use `parent_oids` only for confirmed Object Memory OIDs returned by memory tools, never file/tool-result IDs. For an appendable ledger, use `immutable: false` and payload `{"entries": []}`, then append to that exact name/list field.

## Boundaries and safety

- A visible name never grants object authority; reads, writes, and links remain Capability checked.
- Model-created metadata cannot elevate trust, integrity, or declassification authority.
- Object Memory is not a workspace file. Use the transfer Skill only when crossing that boundary.

## Verify

Before final output or exit, re-read the ledger and confirm every cumulative item has evidence or an explicit blocker. Also confirm namespace, name, type, version, truncation, and lineage inputs.
