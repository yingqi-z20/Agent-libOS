# Changelog

This changelog starts with the currently maintained release line. It does not
reconstruct older release notes or infer publication dates from Git history.
Git history remains the record for earlier development snapshots.

## Unreleased

Changes intended for the next published version must be summarized here before
release. Do not treat an entry in this section as shipped behavior.

- Define shared field and result contracts for model-facing tool semantics,
  generate matching Skill guidance, and gate schema/argument/result drift in CI.
  All core tools receive Chat/Responses and MCP schema checks; explicit semantic
  coverage includes nullable targets, literal JSON, exit results, and file CAS.
  Contracts add no authority or aliases and retain existing Skill size limits.
- Preserve `payload: null` for complete Object Memory reads in v2 result replay,
  while omitting the empty payload placeholder on paginated reads.
- Render v2 Object Memory targets with the API's nullable current-namespace
  selector, preserving literal foreign namespaces and permission/delegation
  resources. Retained goal bindings use the same Host namespace as the prompt.
- Add an optional Host logical LLM-call deadline across transport retries,
  compatibility fallback, and SDK cleanup, with terminal attempt evidence and
  recoverable timeout handling. Existing I/O timeouts and disabled defaults
  remain compatible; evaluation provenance binds an enabled deadline.
- Align coding and Skill guidance on conditional file writes and concise
  same-response final reporting and confirmed exit, preserving cumulative
  review and fresh Human-input checks.
- Keep the original schema's optional fields and open-object constraints when
  OpenAI strict conversion cannot complete. Discard partial conversion edits,
  and make the nullable process-result identifier explicit in its tool guide.
- Explain nullable checkpoint caller selection consistently with strict tool
  schemas, and accept the compact creation receipt without requiring a Host id.
- Preserve safe failure codes, types, and recovery hints for v2 Memory and
  process-exit results that previously disappeared behind success projections.
- Dispatch every tool call in a multi-call model response as one ordered batch
  even when `llm.parallel_tool_calls` is off. Providers that ignore the request
  option previously had all but one call silently discarded, which cost a
  quantum per dropped call and let the model believe unexecuted calls had run.
  Sequential Responses replay still repairs to exactly one call, and a Durable
  TaskRun repairs a batch that contains `activate_skill` instead of failing.
- Make declared tool names first-class Skill discovery keys. A query that names
  the exact tools a goal requires (`run_shell_command git_status human_output`)
  now resolves every owning built-in Skill in one `discover_skills` call, and a
  single-tool-name query returns only its owner.
- When the model selects a tool the image binds but no active Skill projects,
  the action repair names the owning built-in Skill and its package hash so the
  next response can activate it directly.
- Project `read_process_messages` and `receive_process_messages` from spawn for
  Skills-projection images that bind them, so mandatory queued-input handling
  no longer needs a discover/activate/read round trip. Visibility only; the
  message primitives keep their own authority and evidence checks.
- Explain omitted materialized-context Objects by reason in the legacy prompt
  layout and tell the model to re-observe instead of guessing what an omitted
  result said or assuming the work never happened.
- Guide the coding image to discover by exact tool names in one query and to
  batch independent activations; document the changed prompt, projection, and
  discovery contracts.
- Partition `.env.example` out of the sdist so the release-contract checks pass
  with the tracked template.
- Bound long-task prompt growth by projection: under the `working_set` policy
  the newest `memory.working_set_recent_feedback` tool results and any older
  ones that fit `memory.working_set_verbatim_feedback_tokens` render verbatim,
  feedback beyond that renders as a compact `object_memory_feedback_stub`
  (tool, target, outcome, sizes), and a repeated observation of the same target
  supersedes its stale copy (omitted with reason `superseded`). Human input
  results are never compacted and the durable Objects are unchanged.
- Derive each quantum's source materialization budget from the real per-call
  admission headroom (`llm.max_input_tokens_per_call` minus the system prompt,
  loaded Skill bodies, tool schemas, and `llm_context.materialization_headroom_tokens`,
  floored at `llm_context.materialization_budget_floor_tokens`). A long task now
  omits stale feedback instead of being killed at budget admission.
- Notice unread normal process messages before tool selection as well as after
  a tool call, so a Human follow-up queued while the Runtime was closed reaches
  the first prompt after reopen instead of one tool call later. The mandatory
  read directive is now driven by the mailbox itself, not by the bounded recent
  event window, and the executor scans up to
  `llm_context.recent_event_scan_limit` post-cursor events per quantum while
  rendering only the newest `recent_event_limit` visible ones; tool-result
  Object creation and per-Object grant rows are counted, not rendered.
- Emit `tool_batch_truncated` when later calls of a multi-call response did not
  run (failed call, queued input, wait, exec) so the model never assumes they
  did.
- Decode a JSON-encoded object/array tool argument when every schema variant
  forbids strings (for example `process_exit.completion_evidence`), with the
  same `llm.tool_arguments_normalized` audit as scalar repairs.
- Slim the legacy prompt Capabilities table: fold per-Object read grants into
  one `object:materialized` row with a count and drop constant or delegation-only
  fields unless a visible tool consumes them.
- Teach the coding image and the runtime-session and human-collaboration Skills
  that the final `human_output` and the confirmed `process_exit` may share one
  response with the exit last, removing one serial LLM call from every
  completion.
- Add the optional `skills.compact_tool_guides_after_use` Host knob (off by
  default) that omits a loaded Skill's `### tool` guide subsection once that
  tool has succeeded for the process.
- Add the `ledgerctl_rounding_consumers` long-horizon scenario (three consumer
  modules with divergent rounding defects, a misleading incident, a delete
  injection, a CHANGELOG follow-up, hidden per-consumer behavior probes and
  receipt-order oracles), a `--scenario` CLI selector, per-run wall-clock,
  latency, prompt-section, batch, and stop-reason diagnostics, and the
  payload-free `experiments/inspect_long_horizon_run.py` inspector.
- Preserve the literal text `"null"`/`"None"` when a tool argument's schema
  accepts strings, including nullable fields and direct JSON values. Callers
  must send JSON `null` for a null value; unambiguous non-string repairs remain
  audited as `llm.tool_arguments_normalized`.
- Let model-facing tool failures carry identifier-shaped diagnostic codes from
  the tool boundary (`git_error_code`, `hint`, `current_package_sha256`) while
  exception text stays hashed; `git_diff` now names
  `worktree_scope_requires_null_base_and_head` and its siblings instead of an
  opaque correlation id the model retried seven times, a stale Skill hash
  reports the current hash for a one-call re-pin, a compare-and-swap write
  conflict and an unappendable Object payload name their cause, and every
  `LibOSError` accepts an optional identifier `details` bag.
- Accept the main worktree's reported identity digest wherever a Git tool takes
  `worktree_id`; `git_status` returns that digest, and a model that copied it
  into `git_diff` was rejected with `invalid_path` twice per run.
- Key `discover_skills` supersession on the surfaced Skill set so two different
  discovery queries in one response both stay visible, never supersede
  `run_shell_command` results (a passing test run hid the earlier failing run
  and one model reverted its fix to reproduce the failure again), and drop the
  re-observation cue when every omitted Object is merely `superseded`; that cue
  made models infer unfinished earlier work at the start of a task.
- Add a payload-free `Durable activity before the last Runtime reopen` prompt
  digest (files read and written, commands with return codes, Git inspections,
  Skills, checkpoints) rebuilt from durable events when a reopen released the
  earlier tool results, bounded by `llm_context.reopen_digest_event_scan_limit`.
  External-effect metadata retains its observation-time data labels, and the
  digest contributes those labels to LLM egress authorization. Legacy metadata
  without label provenance is omitted, except for already-visible Skill IDs.

## 1.5.3

`1.5.3` is the current release version aligned across the Python project,
package lockfiles, GUI package, MCP client identity, desktop metadata, and
release workflows.

- Upgrade the official OpenAI default to `gpt-6-astra` through the Responses
  API, with bounded encrypted reasoning replay across tool loops, conversation
  turns, restart, and local checkpoints. Replay state remains Host-private and
  is excluded from public projections and AgentImages.
- Introduce RuntimeStore schema v8 and the explicit offline SQLite/PostgreSQL
  v7-to-v8 migration, with source-authority revalidation, payload purge, and
  whole-turn context compaction.
- Normalize reasoning-token accounting and add Host-selectable prompt-cache
  candidates. The release defaults remain `legacy_v1` and `provider_default`
  until the existing paired real-LLM release gate passes.
- Align the minimum OpenAI SDK dependency and release metadata with the
  validated `2.52.0` version.

Publication still follows the separately authorized, receipt-bound process in
[docs/releasing.md](docs/releasing.md).

## 1.5.2

`1.5.2` was the preceding stabilization release version aligned across the Python
project, package lockfiles, GUI package, MCP client identity, desktop metadata,
and release workflows. It builds on the 1.5.1 Runtime authority and data-flow
semantics with further filesystem-authorization hardening, storage and recovery
safety, generated-reference tooling, and a comprehensive documentation accuracy
and completeness pass. As with every version entry here, publication still
requires the separately authorized, receipt-bound process in
[docs/releasing.md](docs/releasing.md); this entry alone does not claim a tag,
package-index upload, signed desktop distribution, or completed external-provider
gate.

- Hardened filesystem authorization against path rebinding between
  authorization and use.
- Implemented runtime safety and recovery enhancements across the storage
  factory and backend boundaries, checkpoint restore and snapshot remapping,
  startup recovery, Git command policy, and Windows store identity handling.
- Introduced the generated CLI reference, the generated configuration field
  reference, and the version-pinned PyPI README generator, extracting the CLI
  parser build for generated-reference use.
- Fixed an ObjectTask wait race that could return a stale snapshot from
  before the worker published its waiting notification, terminal result
  collection in the ask-file-then-show example script, and a CI readiness
  race in the GUI end-to-end suite.
- Fixed documentation accuracy and completeness across the guide set. The
  generated CLI reference now reports the effective destination default for
  store-constant flags, and `task-run list --status` help and rejection
  messages enumerate the closed status set.
- Completed a comprehensive documentation review and remediation across the
  guide set: corrected drifted inventories (MCP Resource surface, image
  bindings, checkpoint images), documented the CLI exit-code contract and exec
  image-package authority grant, fixed the storage migration backup example,
  replaced stale version tokens, pinned the PyPI README quick-start clone to
  the release tag, added help text for required CLI options, and added a
  glossary version-map tripwire regression test.

## 1.5.1

`1.5.1` was the preceding stabilization release version aligned across the
Python project, package lockfiles, GUI package, MCP client identity, desktop
metadata, and release workflows. It preserves the Runtime authority and
data-flow semantics while hardening first-run behavior, offline migration
reconciliation, terminal-owner cleanup, live-evaluation evidence recomputation,
and release artifact validation.

The source distribution now uses an explicit include/exclude partition. Its
checker rejects ordinary source files outside that partition and validates the
exact core, PostgreSQL, PTY, and MCP `Requires-Dist`/`Provides-Extra` metadata
in both the wheel and source archive. See
[docs/release_status.md](docs/release_status.md) for the implemented scope and
remaining environment gates.

As with every version entry here, publication still requires the separately
authorized, receipt-bound process in [docs/releasing.md](docs/releasing.md).
This entry alone does not claim a tag, package-index upload, signed desktop
distribution, or completed external-provider gate.

## 1.5.0

`1.5.0` was the preceding aligned release version across the Python project,
package lockfiles, GUI package, and release-artifact workflow.

This release preserves Manifest v1/v2 governed Tools compatibility and adds
the exact-`2026-07-28` Manifest v3 client for governed Tools, Resources,
Resource Templates, Prompts, Completion, Host-owned OAuth, MRTR, pinned remote
Tasks, and bounded subscriptions. It also introduces the explicit RuntimeStore
schema-v6-to-v7 migration and tightens Tool argument, deadline, redaction,
retry-classification, registry-search, artifact, and cross-SDK conformance
contracts.

The tag, GitHub Release, and downloadable Python artifacts must remain bound to
the exact source commit, CI receipt, and checksums recorded by the separately
authorized process in [docs/releasing.md](docs/releasing.md). This entry does
not claim a PyPI upload, signed desktop distribution, or completion of the
environment gates listed in the release status.

## 1.4.2 — prior release candidate

`1.4.2` was the previously aligned release-candidate version. This historical
entry does not claim that a tag, package-index upload, or GitHub release exists.
