# Runtime Events

Agent libOS 1.4.1 persists a closed catalog of `EventType` values. Events
are durable observations for operators, the GUI, context materialization, and
Explain evidence. They are not an authority source, a task queue, or a
replacement for the process, operation, capability, Human-request, or external-
effect state machines.

## Envelope

Every stored `Event` has these fields:

| Field | Contract |
| --- | --- |
| `event_id` | Opaque primary key. Ordinary `emit()` generates it; `emit_once()` requires the Host producer to supply a stable semantic id. Do not infer time or sequence from the id. |
| `type` | One of the 46 values in the catalog below. Unknown values are rejected before insertion. |
| `source` | Producer-selected string identity. It may be a PID, Runtime/component name, capability issuer, Human identity, or resource; it is not a foreign key. |
| `target` | Producer-selected string identity or `null` for a broadcast observation. It is not a grant or proof of visibility. A target-filtered query returns both exact-target events and `target=null` broadcasts. |
| `payload` | JSON object owned by the event producer. The catalog lists current discriminators and common keys; failure/recovery variants may add diagnostic fields. Consumers must tolerate additional keys and must not treat an event payload as an authority decision unless the owning domain contract says so. |
| `priority` | `low`, `normal`, `high`, or `critical`. Priority is display/attention metadata, not durable-delivery ordering or admission authority. Current producers use `normal` unless the catalog says otherwise. |
| `created_at` | Runtime-assigned UTC timestamp used as the first query-order key. It is not a provider-effect timestamp or a causal clock. |
| `correlation_id` | Optional grouping identity, commonly shared with safe public error/audit diagnostics. It is neither unique nor a foreign key. |
| `causality` | Producer-owned JSON object. Current uses include `audit_parent_record_id` and terminal-state identity. It is not a validated or complete causal graph. |

Payloads are evidence, not automatically payload-free. For example, Shell
events contain bounded command arguments, while data-flow decisions contain
labels and hashes rather than the released payload. Apply retention, access,
and backup controls appropriate to potentially sensitive audit data.

## Event type catalog

The rows are in `EventType` declaration order. “Payload” names the current
stable discriminator/common fields rather than defining a closed JSON Schema.

| Event type | Producer and meaning | Source → target | Payload | Non-normal priority |
| --- | --- | --- | --- | --- |
| `runtime_shutdown` | Runtime lifecycle records an admitted shutdown attempt reaching its evidence phase | shutdown actor → `runtime` | `reason` | — |
| `task_run_created` | `TaskRunManager` atomically publishes a new durable Run together with its root process and initial requirement | `runtime.task_runs` → Run id | `schema_version`, `run_id`, `root_pid`, `revision` | — |
| `process_created` | `ProcessManager` publishes a root or spawned child | `runtime` or parent PID → new PID | root: `pid`, `image`, `goal_oid`, `working_directory`, `llm_profile_id`; child: `parent`, `child`, `image`, `goal_oid`, `working_directory`, `status`, `llm_profile_id` | — |
| `process_forked` | `ProcessManager` publishes a fork, or `CheckpointManager` publishes a checkpoint fork | parent/source actor → child/fork-root PID | direct fork: `parent`, `child`, `mode`, `working_directory`, `llm_profile_id`; checkpoint fork: `checkpoint_id`, `source_pid`, `fork_root_pid` | — |
| `process_exec` | `ProcessManager` commits an image replacement | PID → same PID | `old_image`, `new_image`, `preserve_memory`, `preserve_capabilities`, `goal_oid`, `working_directory`, `llm_profile_id` | — |
| `process_exited` | `ProcessManager` records a terminal outcome; terminal recovery uses `emit_once()` | PID → parent PID or `null` | `pid`, `status`, `result_oid`; recovery/signal finalization may add `reason` | — |
| `process_message_posted` | `ProcessMessageManager` commits a durable mailbox message | sender identity → recipient PID | ordinary message: `message_id`, `kind`, `channel`, `correlation_id`, `reply_to`, `subject`, `sender`, `data_labels`; Task Run-bound message: IDs/labels plus `task_run_id`, `subject_sha256`, `body_sha256`, `payload_sha256`, with no readable content | `high` for interrupt messages |
| `process_message_notice` | `ProcessMessageManager` publishes a tool-boundary notice | notice source → recipient PID | `phase`, `kind`, `count`, `message_ids`, `channels`, `correlation_ids`, `instruction`; the model projection reduces this to the read-pending control, count, and kind | `high` for interrupt notices |
| `process_message_acked` | `ProcessMessageManager` commits acknowledgement | recipient PID → same PID | `message_ids`, `count` | — |
| `process_signal` | Process/Human control or stale-execution recovery records a state-changing signal | actor/component → PID | ordinary: `signal`, `payload`; recovery: `pid`, `reason` | — |
| `object_created` | `ObjectMemoryManager` publishes an Object descriptor | PID → same PID | `oid`, `namespace`, `name`, `qualified_name`, `type`, `data_labels` | — |
| `object_updated` | `ObjectMemoryManager` publishes a new Object version | PID → same PID | `oid`, `namespace`, `name`, `qualified_name`, `version`, `data_labels` | — |
| `object_linked` | `ObjectMemoryManager` publishes a typed Object link | actor/PID → same actor/PID | `src`, `relation`, `dst` | — |
| `object_task_started` | `ObjectTaskManager` publishes a durable task and runner | creator PID → owner OID | `task_id`, `runner_pid`, `tool` | — |
| `object_task_running` | `ObjectTaskStateService` starts Host-managed runner execution | creator PID → owner OID | `task_id`, `runner_pid`, `tool` | — |
| `object_task_waiting` | `ObjectTaskStateService` records an external wait | `object_task` → owner OID | `task_id`, `status`, `wait`, `tool` | `high` |
| `object_task_completed` | `ObjectTaskStateService` records successful task completion | creator PID → owner OID | `task_id`, `result_oid`, `tool` | — |
| `object_task_failed` | `ObjectTaskStateService` records failed task completion | `object_task` → owner OID | `task_id`, `tool`, `error` | `high` |
| `object_task_cancelled` | `ObjectTaskStateService` records cancellation | cancelling actor → owner OID | `task_id`, `reason` | — |
| `object_task_notification_undelivered` | Notification service records a terminal recipient | `object_task` → recipient PID | `task_id`, `status`, `reason` | — |
| `object_task_owner_change_notified` | `ObjectTaskManager` delivers an owner-watch change | actor PID → owner OID | `task_id`, `runner_pid`, `message_id`, `event` | — |
| `object_task_owner_change_undelivered` | `ObjectTaskManager` cannot deliver an owner-watch change | actor PID → owner OID | `task_id`, `runner_pid`, `event`, `error` | `high` |
| `human_query` | `HumanObjectManager` commits a Human request | PID → `human:<id>` | `request_id`, `request_type`, bounded `request` observation, `blocking` | — |
| `human_response` | `HumanObjectManager` records a decision or protected terminal read | responder/Human identity → PID or Human resource | decision evidence includes `request_id`, `status`; protected reads include `purpose`, `operation`, `chars`; producer-specific failure diagnostics may include `provider_outcome`, `outcome`, `phase`, or `error_type` | — |
| `semantic_policy_response` | Host machine policy atomically terminalizes an eligible Human request | `policy:semantic:<epoch>` → PID | `request_id`, expected `revision`, `status`, `settlement_id`, `epoch_id`, `policy_sha256`, exact binding digest, reason code, and optional one-use `capability_id`; never contains Human-authored response content | — |
| `image_registered` | `ImageRegistry` commits registration/replacement | actor → `image:<id>` | `image_id`, `name`, `version`, `replaced`, `source`, `boot_kind` | — |
| `image_committed` | `ImageRegistry` commits a checkpoint-derived image | actor → `image:<id>` | `image_id`, `checkpoint_id`, `artifact_id`, `artifact_sha256`, `artifact_bytes` | — |
| `skill_registered` | `SkillManager` commits a package registration | actor → `skill:<id>` | `skill_id`, `version`, `source_type` | — |
| `skill_loaded` | `SkillManager` activates a Skill for a process | actor → PID | `skill_id`, `tool_names` | — |
| `skill_unloaded` | `SkillManager` removes a process activation | actor → PID | `skill_id`, `removed_tools` | — |
| `skill_trusted` | `SkillManager` commits Host trust for a package hash | actor → trust resource | `source_type`, `source` | — |
| `module_loaded` | trusted module registry publishes startup load | `runtime` → `module:<id>` | `module_id`, `registered` summary | — |
| `tool_called` | `ToolExecutionService` admits a tool call far enough to record invocation | PID → tool resource | `call_id`, bounded/sanitized `args` observation | — |
| `tool_completed` | `ToolExecutionService` commits a successful result Object | tool resource → PID | `call_id`, `result_oid` | — |
| `tool_failed` | Tool executor or LLM action layer records denial, validation, lifecycle, or execution failure | tool resource → PID | `call_id` where available, bounded `error`, `policy_decision`; variants may include `result_oid`, `tool_name`, `request_id`, `policy_reason` | — |
| `capability_granted` | capability mutation/lease service commits a grant or restores a reservation | issuer/restoring actor → subject PID | grant: `capability_id`, `resource`, `rights`, `effect`, `uses_remaining`; restore: `capability_id`, `reason`, `reservation_id`, `uses_remaining` | — |
| `capability_revoked` | capability mutation/finite-use lease/exec shrink commits revocation | revoking or consuming actor → subject PID | `capability_id`, `reason`; finite-use reservation/exhaustion variants may add `reservation_id` | — |
| `checkpoint_created` | `CheckpointManager` commits a scoped checkpoint | actor → root PID | `checkpoint_id`, `reason`, `subtree_pids` | — |
| `rollback` | `CheckpointManager` commits the main destructive restore transaction | actor → restored root PID | `checkpoint_id`, `restored_pids`, `superseded_object_tasks`, `external_effects_since_checkpoint`, `external_effect_summary`, `restore_external_policy`, `main_state_committed` | — |
| `external_read` | Protected Operation SDK records a provider observation; filesystem/Git/clock/JSON-RPC/MCP producers select it for non-mutation | invoking actor/PID → protected resource/Sink | provider-specific bounded observation that may include `adapter`/`provider` and `operation`; uncertain-failure diagnostics are producer-specific and may include `outcome`, `phase`, or `error_type` | — |
| `external_write` | Protected Operation SDK or Host registry mutation records a provider/effectful operation | invoking actor/PID → protected resource/Sink | provider-specific bounded observation that may include `adapter`/`provider` and `operation`; uncertain-failure diagnostics are producer-specific and may include `outcome`, `phase`, or `error_type` | — |
| `human_output` | protected Human terminal/output delivery | PID → Human resource/channel | terminal output: `request_id`, `channel`, `chars`; terminal operations also include `purpose`, `operation`; producer-specific uncertain variants may include `provider_outcome`, `outcome`, `phase`, or `error_type` | — |
| `resource_charged` | `ResourceManager` commits usage accounting | charge source → PID | `pid`, `usage`, `context` | — |
| `resource_limit_exceeded` | `ResourceManager` commits enforcement and affected-process summary | `resource_manager` → PID | `pid`, `owner_pid`, `reason`, `killed_pids`, `limit` | `critical` |
| `sink_trust_registered` | `DataFlowManager` commits Host Sink trust | actor → Sink pattern | `trust_id`, `pattern`, `trust_level`, `max_sensitivity`, `spec_hash`, `generation`, `replaced` | — |
| `sink_trust_unregistered` | `DataFlowManager` removes Host Sink trust | actor → selected trust identity | `trust_id`, `spec_hash`, `generation` | — |
| `data_flow_decision` | `DataFlowManager` persists an allow/conditional/deny decision that actually reached the data-flow gate | PID → `data_flow_sink:<identity>` | `decision_id`, `direction`, `outcome`, `reason`, Sink identities/hashes, labels/hash, source refs/hash, payload hash, trust id/hash, registry generation, release capability id | — |

`low` is a valid envelope value but no built-in producer currently selects it.
The absence of an event type is also meaningful: there is no generic
`process_woke` or `process_state_changed` event. Read the typed process row and
the owning domain evidence instead of reconstructing lifecycle state from the
catalog.

## Ordering and cursors

Persistent event queries order rows by `(created_at, event_id)`. Without a
limit, `EventBus.list()` returns the complete matching sequence in ascending
order. With a limit and no `after_event_id`, it selects the newest window and
returns that window in ascending order; with `after_event_id`, it returns the
next ascending window. `before_event_id` and `after_event_id` are mutually
exclusive, and an unknown cursor returns an empty page.

This composite order is deterministic for persisted rows, but it is not causal
order. Concurrent producers can obtain the same timestamp, the event-id
tie-breaker is opaque, and a provider effect may precede the transaction that
records its event. Follow the operation, publication, process-generation, or
external-effect state machine when semantic ordering matters. Persist and pass
the returned event id as a cursor; do not synthesize a cursor or compare ids.

The public query limit is positive and cannot exceed the import-time
`DEFAULT_CONFIG.gui.event_buffer_limit` (currently `1000`). It is a fixed
`EventBus` query ceiling, not the active Runtime's configurable GUI SSE buffer
size. GUI snapshot filtering can hide
`HUMAN_OUTPUT` events with `purpose=gui_presentation` and matching Human-GUI
`DATA_FLOW_DECISION` rows; that filter does not delete the durable event.

## Idempotent publication

`emit()` allocates a fresh event id and therefore is not an idempotency API.
`emit_once()` accepts a non-empty producer-chosen id of at most 256 non-NUL
characters.
An exact retry with the same type, source, target, payload, priority,
correlation id, and causality returns the original row, including its original
`created_at`. Reusing the id with any different semantic field raises a
`ValidationError`. The caller, not `EventBus`, is responsible for deriving a
stable id from the semantic transition; current terminal recovery derives one
from event type, PID, and state generation.

Insertion and any active Explain-operation evidence link share one store
transaction. Consequently a failed evidence link leaves no orphan event, and
concurrent exact `emit_once()` attempts converge to one row or an integrity
error. `emit_once()` does not deduplicate distinct ids and is not a substitute
for a provider idempotency key or an external-effect intent.

## Atomicity and causal boundaries

`EventBus` guarantees atomicity only for its own insert and the operation link
it creates. When a producer calls it inside a wider `UnitOfWork`, the nested
transaction joins that outer commit, so the producer may deliberately couple
an event with process, capability, message, Human, checkpoint, audit, or effect
state. When it is called after a domain commit, the event is intentionally a
separate diagnostic phase. The catalog does not upgrade every event into a
claim that all related rows committed atomically.

In particular:

- a process status change has no automatic generic event; use the lifecycle
  matrix in [Runtime Model](runtime_model.md) to identify its owning evidence;
- an authority/visibility rejection that occurs before data-flow evaluation
  has no `DATA_FLOW_DECISION`; see [Data Flow](data_flow.md);
- an external read/write event says that protected-operation evidence was
  persisted, not that the SQL transaction and provider effect were one atomic
  system. Consult the external-effect intent and transition rows for
  `prepared`/`dispatched`/`committed`/`unknown` recovery state;
- `correlation_id` and `causality` aid joining evidence but do not impose a
  foreign-key, uniqueness, delivery, happens-before, or graph-completeness
  guarantee; and
- application APIs append events, but a Host or database administrator remains
  outside the tamper-resistance boundary. Independent integrity requires an
  external signed or append-only evidence system.

See [Architecture](architecture.md) for component ownership and
[Runtime Storage](storage.md) for transaction, recovery, retention, backup,
and active-writer boundaries.
