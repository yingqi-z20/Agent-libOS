---
name: agent-libos-agent-images
description: Register AgentImage packages, publish checkpoint-derived images, or replace the current process image. Use when creating reusable runtime configurations or deliberately changing process code, tools, prompt identity, memory, or authority.
allowed-tools: load_image_package commit_checkpoint_to_image exec_process
---
# Manage AgentImages

AgentImages are execution contracts. Package load and checkpoint commit mutate the registry; only exec replaces this process. **Publication is not boot**, and does not prove the image can boot here.

Normally publish a new, versioned `image_id` with `replace=false`. An image ID is a replaceable registry key, not immutable content. Artifact/package hashes identify captured bytes; they prove neither provenance, trust, safety, nor execution success.

## Tool guide

| Intent | Tool | Current process changes? |
|---|---|---|
| Register a reviewed workspace package | `load_image_package` | No |
| Publish reconstructable checkpoint state | `commit_checkpoint_to_image` | No |
| Adopt another image and tool table | `exec_process` | Yes; PID stays the same |

### `load_image_package`

Pass `path` to the package directory or exact `IMAGE.yaml`; relative paths resolve from current process cwd. Set `replace` explicitly (normally `false`). First inspect the manifest, prompts/mode, default Skills/tools, module/capability requirements, JIT sources, packaged `workspace/`, and grants with a filesystem Skill. Reject unexplained untrusted content.

The call validates and registers the image/artifact. It does not exec, rebuild current tools, change memory/authority, materialize a workspace, or test boot. Later boot may create a private workspace, set cwd, install Skills/tools, register package JIT, and grant declared package-workspace access. External requirements are not grants.

Check the complete output: `image_id`, `name`, `version`, `source_path`, `replaced`, `default_tools`, `package_jit_tools`, `boot_kind`, `artifact_id`, `package_sha256`, and both requirement counts. A package hash is content identity only; a manifest signature field is metadata unless separately authenticated.

Reading needs filesystem authority; a new ID needs exact `image:<image_id>` write, and replacement needs exact admin. A durable but unbootable/quarantined row still counts as an existing ID: `replace=false` cannot overwrite it with write authority, and deliberate replacement requires `replace=true` plus exact admin. Replacement changes future resolution. A process already bearing that ID may resolve new prompt/policy while retaining its bound tools. Prefer a fresh ID and explicit exec.

### `commit_checkpoint_to_image`

Use a checkpoint previously inspected with the checkpoint Skill; this Skill cannot inspect one. Supply exact `checkpoint_id`, new `image_id`, `name`, explicit `version`, `replace` (normally `false`), and intentional `metadata`.

The call publishes a checkpoint-derived artifact and registry entry. It does not restore/resume the checkpoint, change the caller, create a process, exec this process, or test boot.

The root-process artifact captures reconstructable Object roots/payloads, namespaces/internal Object authority, static tool names/bindings, process-local JIT state/source, loaded Skills, cwd string, and module/capability requirements. Static implementation code is not embedded: boot resolves each non-JIT name against the current Host `ToolBroker`, so a same-name implementation may drift or disappear. Required Module hashes mitigate this for module-supplied tools but do not pin every Host/built-in tool. It does not clone children, files/shell, provider registrations/sessions, MCP/JSON-RPC state, UI state, budgets/usage, or external effects. Source-row metadata may remain, but boot does not replay its lease, status, waits, mailbox/history, event cursor, or outcomes. Package files separately when boot needs them.

Boot remaps internal Object/namespace capabilities. External capabilities become requirements only. Hash-pinned modules fail closed if absent/mismatched; global Skill trust and provider registries are not packaged. Before boot, the Host must also confirm that every captured Object's data-flow labels, including tenant boundaries, are admissible for the target process; boot rejects incompatible metadata rather than relabeling it.

Drift hazard: commit gets prompt/mode, planner/action settings, context/safety policy, JIT exposure, default Skills, and LLM profile from the **current registry entry** for the captured `source_image_id`; these are not all checkpoint-frozen. Replacing/removing that ID after checkpoint creation can yield a changed contract or fallbacks. Confirm it, do not replace it, commit promptly, and use a fresh target. Output cannot prove inherited-contract equality.

New targets need exact image-write; replacement needs exact image-admin; source access needs checkpoint- or process-read. Verify all identity/version/replacement fields, `boot_kind`, artifact ID/hash, and requirement counts. The hash neither freezes the registry key nor proves boot.

### `exec_process`

Use an exact trusted/just-published `image`; explicitly provide audit `args`, intended `goal`, `preserve_memory`, and `preserve_capabilities`.

`args` is audit-only structured context. It is recorded in exec evidence, not passed to target boot or prompt, and cannot set goal, tools, memory, cwd, or authority. Put work intent in `goal` and configuration in the image.

Exec **must be the only tool call in its tool-call batch**. Never send sibling calls with it. Success changes the active prompt/tool/Skill contract, so all follow-up belongs to a new model step under the post-exec contract.

Exec retains PID and budget/usage. It has no LLM-profile argument, so the process profile field remains. Resolve every typed child/mailbox/Human/Tool/Host wait first; exec requires runnable state or the exact active lease. Switching images needs exact target-image read; external requirements do not grant themselves. A finite or one-shot image-read grant is settled before boot publication and is not refunded merely because a later boot step fails and the prior process snapshot is restored.

Choose `goal` deliberately. Omitting it retains the old `goal_oid`; supplying one creates a replacement goal Object. Avoid carrying an old goal into a materially different contract.

Choose both flags rather than relying on defaults:

- `preserve_memory=true` keeps the MemoryView; checkpoint boot may merge baked roots. Use only for data valid under the target.
- `preserve_memory=false` replaces the **view**, not owned Objects: it is not a clean slate. Objects and Object/own-namespace authority remain; absent a new goal, old `goal_oid` may sit outside the view; checkpoint roots are installed. Stable named baked Objects may reuse same-PID live Objects instead of overwriting their payload.
- `preserve_capabilities=false` removes non-Object external authority, but keeps Object/own-namespace authority. Boot can add internal/package-workspace authority, never external requirements.
- `preserve_capabilities=true` retains current authority; use only when each capability remains necessary.

Success returns `pid`, `old_image`, `new_image`, `status`, `goal_oid`, both flags, and `active_tools`. `active_tools` is the full bound tool table, not necessarily the model-visible schema projection; a listed tool may still require Skill activation.

## Recommended workflow

1. Distinguish publication-only from replacing this process.
2. Inspect all package references, or the checkpoint plus unchanged source image.
3. Choose a fresh versioned ID and `replace=false`; record the expected contract.
4. Publish, then compare every returned identity/hash, boot kind, tool summary, and requirement count. Stop on mismatch.
5. For publication-only, report “registered, not boot-tested” and stop.
6. Before exec, resolve waits; checkpoint valuable current work; confirm target read, modules, external authority, and providers.
7. Set explicit goal/flags; memory false is view replacement. Put only audit rationale in `args`.
8. Call exec alone. Continue in a new step, activate needed Skills, verify visible schemas, and inspect task-critical cwd, memory, authority, modules, workspace, and providers.

## Failure and recovery

For load/commit validation or authority errors, fix the reviewed input or authority. Never force progress with `replace=true`, a weakened manifest, or guessed IDs. A reported failure is not evidence that the requested entry exists. If transport timeout makes a non-idempotent mutation ambiguous, stop: this Skill's mutation-only publication tools cannot inspect the registry. Hand the returned IDs and transport evidence to a Host/operator with registry/artifact inspection, and retry only after that path proves the original mutation absent.

Exec preflight can reject an unknown image, missing exact read authority, active typed wait, unavailable/mismatched module or artifact, malformed image, or incompatible process state. Resolve the named condition before reconsidering exec.

If boot fails and the runtime reports successful rollback, the prior contract/snapshot was restored; do not claim the target is active. Any settled one-shot target-image read remains consumed, so a deliberate later attempt needs newly valid authority. If the error reports recovery required, ambiguous rollback, incomplete compensation, or a recovery fence, stop all mutation. Do not retry exec, guess an image, or use either contract's tool assumptions. The Host must reopen/reconcile the durable publication and confirm a terminal result. Images/checkpoints do not undo provider effects, so separately reconcile remote actions.

Never “roll back” by replacing a registry ID with old bytes. That changes future resolution and may affect other processes; it does not restore process state. Use checkpoint recovery, or later exec a separately published image after the current publication has a known outcome.

## Completion evidence

For package publication, report the exact ID/version, `replaced`, source path, boot/artifact IDs, package hash, default/JIT tool names, requirement counts, and “registered, not boot-tested” unless boot actually succeeded.

For checkpoint publication, report the exact ID/version, `replaced`, checkpoint boot kind, artifact ID/hash, and requirement counts. Disclose that external state/authority was not cloned and behavioral fields came from the source image's registry entry at commit time. Call the artifact/hash immutable, never the registry ID.

For exec, require returned `pid` to match, `old_image` to be expected, `new_image` to equal the requested ID, status and both flags to match the intended transition, and `goal_oid` to reflect the goal choice. Treat `active_tools` only as a bound-table inventory. Completion also requires post-exec confirmation of task-critical visible schemas, cwd, memory, authority, modules, workspace, and providers.

Never claim completion from a planned call, hash alone, registry entry alone, or ambiguous/recovery-required result. Preserve returned IDs and hashes so later Host/runtime evidence can be correlated.
