from __future__ import annotations

from agent_libos.capability.manager import CAPABILITY_MANAGER_MUTATION_PUBLIC_METHODS
from agent_libos.ports import ExplainBoundaryDescriptor


def boundary(
    component: str,
    method: str,
    kind: str,
    name: str,
    actor_arg: str = "",
    pid_arg: str = "",
    *,
    result_pid: bool = False,
    preflight_method: str = "",
    lifecycle_lock_attr: str = "",
) -> ExplainBoundaryDescriptor:
    return ExplainBoundaryDescriptor(
        component=component,
        method=method,
        kind=kind,
        name=name,
        actor_arg=actor_arg,
        pid_arg=pid_arg,
        result_pid=result_pid,
        preflight_method=preflight_method,
        lifecycle_lock_attr=lifecycle_lock_attr,
    )


PRIMITIVE_BOUNDARIES = (
    boundary("filesystem", "read_text", "primitive", "primitive.filesystem.read_text", "pid", "pid"),
    boundary("filesystem", "read_bytes", "primitive", "primitive.filesystem.read_bytes", "pid", "pid"),
    boundary("filesystem", "write_text", "primitive", "primitive.filesystem.write_text", "pid", "pid"),
    boundary("filesystem", "read_directory", "primitive", "primitive.filesystem.read_directory", "pid", "pid"),
    boundary("filesystem", "write_directory", "primitive", "primitive.filesystem.write_directory", "pid", "pid"),
    boundary("filesystem", "delete_file", "primitive", "primitive.filesystem.delete_file", "pid", "pid"),
    boundary("filesystem", "delete_directory", "primitive", "primitive.filesystem.delete_directory", "pid", "pid"),
    boundary("shell", "run", "primitive", "primitive.shell.run", "pid", "pid"),
    boundary("git", "repository_info", "primitive", "runtime.git.repository_info", "pid", "pid"),
    boundary("git", "status", "primitive", "runtime.git.status", "pid", "pid"),
    boundary("git", "diff", "primitive", "runtime.git.diff", "pid", "pid"),
    boundary("git", "log", "primitive", "runtime.git.log", "pid", "pid"),
    boundary("git", "show", "primitive", "runtime.git.show", "pid", "pid"),
    boundary("git", "blame", "primitive", "runtime.git.blame", "pid", "pid"),
    boundary("git", "list_refs", "primitive", "runtime.git.list_refs", "pid", "pid"),
    boundary("git", "list_remotes", "primitive", "runtime.git.list_remotes", "pid", "pid"),
    boundary("git", "list_worktrees", "primitive", "runtime.git.list_worktrees", "pid", "pid"),
    boundary("git", "stage", "primitive", "runtime.git.stage", "pid", "pid"),
    boundary("git", "unstage", "primitive", "runtime.git.unstage", "pid", "pid"),
    boundary("git", "commit", "primitive", "runtime.git.commit", "pid", "pid"),
    boundary("git", "restore", "primitive", "runtime.git.restore", "pid", "pid"),
    boundary("git", "branch", "primitive", "runtime.git.branch", "pid", "pid"),
    boundary("git", "switch", "primitive", "runtime.git.switch", "pid", "pid"),
    boundary("git", "tag", "primitive", "runtime.git.tag", "pid", "pid"),
    boundary("git", "integrate", "primitive", "runtime.git.integrate", "pid", "pid"),
    boundary("git", "stash", "primitive", "runtime.git.stash", "pid", "pid"),
    boundary("git", "reset", "primitive", "runtime.git.reset", "pid", "pid"),
    boundary("git", "clean", "primitive", "runtime.git.clean", "pid", "pid"),
    boundary("git", "worktree", "primitive", "runtime.git.worktree", "pid", "pid"),
    boundary("git", "create_patch", "primitive", "runtime.git.create_patch", "pid", "pid"),
    boundary("git", "apply_patch", "primitive", "runtime.git.apply_patch", "pid", "pid"),
    boundary("git", "fetch", "primitive", "runtime.git.fetch", "pid", "pid"),
    boundary("git", "pull", "primitive", "runtime.git.pull", "pid", "pid"),
    boundary("git", "push", "primitive", "runtime.git.push", "pid", "pid"),
    boundary("git", "create_pull_request", "primitive", "runtime.git.create_pull_request", "pid", "pid"),
    boundary("git", "list_pull_requests", "primitive", "runtime.git.list_pull_requests", "pid", "pid"),
    boundary("git", "inspect_pull_request", "primitive", "runtime.git.inspect_pull_request", "pid", "pid"),
    boundary("git", "review_pull_request", "primitive", "runtime.git.review_pull_request", "pid", "pid"),
    boundary("git", "merge_pull_request", "primitive", "runtime.git.merge_pull_request", "pid", "pid"),
    boundary("git", "close_pull_request", "primitive", "runtime.git.close_pull_request", "pid", "pid"),
    boundary("jsonrpc", "call", "primitive", "primitive.jsonrpc.call", "pid", "pid"),
    boundary("mcp", "list_tools", "primitive", "primitive.mcp.list_tools", "actor", "actor"),
    boundary("mcp", "call_tool", "primitive", "primitive.mcp.call", "pid", "pid"),
    boundary("clock", "now", "primitive", "primitive.clock.now", "pid", "pid"),
    boundary("clock", "sleep", "primitive", "primitive.clock.sleep", "pid", "pid"),
)

PROCESS_BOUNDARIES = (
    boundary(
        "process", "spawn", "runtime", "process.spawn",
        result_pid=True, preflight_method="preflight_spawn",
    ),
    boundary(
        "process", "fork", "runtime", "process.fork", "parent", "parent",
        preflight_method="preflight_fork",
    ),
    boundary(
        "process", "spawn_child", "runtime", "process.spawn_child", "parent", "parent",
        preflight_method="preflight_spawn_child",
    ),
    boundary("process", "set_working_directory", "runtime", "process.chdir", "pid", "pid"),
    boundary("process", "signal_child", "runtime", "process.signal_child", "pid", "pid"),
    boundary("process", "signal", "runtime", "process.signal", "target", "target"),
    boundary("process", "wait", "runtime", "process.wait", "pid", "pid"),
    boundary("process", "pause", "runtime", "process.pause", "pid", "pid"),
    boundary("process", "resume", "runtime", "process.resume", "pid", "pid"),
    boundary("process", "cancel", "runtime", "process.cancel", "pid", "pid"),
    boundary("process", "exit", "runtime", "process.exit", "pid", "pid"),
)

IMAGE_BOOT_BOUNDARIES = (
    boundary(
        "image_boot",
        "exec",
        "runtime",
        "process.exec",
        "pid",
        "pid",
        preflight_method="preflight_exec",
    ),
)

MEMORY_BOUNDARIES = (
    boundary("memory", "create_object", "runtime", "memory.create_object", "pid", "pid"),
    boundary("memory", "update_object", "runtime", "memory.update_object", "pid", "pid"),
    boundary("memory", "append_object_by_name", "runtime", "memory.append_object", "pid", "pid"),
    boundary("memory", "link_objects", "runtime", "memory.link_objects", "pid", "pid"),
    boundary("memory", "create_view", "runtime", "memory.create_view", "pid", "pid"),
    boundary("memory", "fork_view", "runtime", "memory.fork_view", "parent_pid", "parent_pid"),
    boundary("memory", "merge_view", "runtime", "memory.merge_view", "parent_pid", "parent_pid"),
    boundary("memory", "snapshot_view", "runtime", "memory.snapshot_view", "pid", "pid"),
    boundary("memory", "materialize_context", "runtime", "memory.materialize_context", "pid", "pid"),
)

CHECKPOINT_BOUNDARIES = (
    boundary("checkpoint", "create", "runtime", "checkpoint.create", "actor", "pid"),
    boundary(
        "checkpoint", "inspect", "runtime", "checkpoint.inspect", "actor", "actor",
        preflight_method="preflight_checkpoint",
    ),
    boundary(
        "checkpoint", "diff", "runtime", "checkpoint.diff", "actor", "actor",
        preflight_method="preflight_checkpoint",
    ),
    boundary(
        "checkpoint", "restore", "runtime", "checkpoint.restore", "actor", "actor",
        preflight_method="preflight_checkpoint",
    ),
    boundary(
        "checkpoint", "fork_from_checkpoint", "runtime", "checkpoint.fork", "actor", "actor",
        preflight_method="preflight_checkpoint",
    ),
    boundary(
        "checkpoint", "replay_to_event", "runtime", "checkpoint.replay", "actor", "actor",
        preflight_method="preflight_checkpoint",
    ),
)

HUMAN_BOUNDARIES = (
    boundary("human", "query", "runtime", "human.query", "pid", "pid"),
    boundary("human", "request_permission", "runtime", "human.request_permission", "pid", "pid"),
    boundary("human", "ask", "runtime", "human.ask", "pid", "pid"),
    boundary("human", "output", "runtime", "human.output", "pid", "pid"),
    boundary("human", "interrupt", "runtime", "human.interrupt", "pid", "pid"),
    boundary("human", "send_process_message", "runtime", "human.process_message", "human", "recipient_pid"),
)

MESSAGE_BOUNDARIES = (
    boundary("messages", "post", "runtime", "process.message.post", "sender", "recipient_pid"),
    boundary("messages", "send_from_process", "runtime", "process.message.send", "sender_pid", "sender_pid"),
    boundary("messages", "receive", "runtime", "process.message.receive", "pid", "pid"),
    boundary("messages", "ack", "runtime", "process.message.ack", "pid", "pid"),
)

OBJECT_TASK_BOUNDARIES = (
    boundary("object_tasks", "start", "runtime", "object_task.start", "pid", "pid"),
    boundary("object_tasks", "watch_owner", "runtime", "object_task.watch", "actor_pid", "actor_pid"),
    boundary("object_tasks", "cancel", "runtime", "object_task.cancel", "actor_pid", "actor_pid"),
    boundary("object_tasks", "wait", "runtime", "object_task.wait", "actor_pid", "actor_pid"),
)

AUTHORITY_BOUNDARIES = (
    boundary("authority_manifests", "prepare_launch", "runtime", "authority_manifest.bind", "issued_by", "pid"),
    boundary("capability", "issue", "runtime", "capability.issue", "actor", "subject"),
    boundary("capability", "delegate", "runtime", "capability.delegate", "parent", "parent"),
    boundary("capability", "derive_authority", "runtime", "capability.derive_authority", "source_subject", "source_subject"),
    boundary("capability", "revoke", "runtime", "capability.revoke", "revoked_by", "revoked_by"),
)

EXTENSION_BOUNDARIES = (
    boundary("skills", "register_skill_package", "runtime", "skill.register", "actor", "actor"),
    boundary(
        "skills",
        "activate_skill",
        "runtime",
        "skill.activate",
        "pid",
        "pid",
        lifecycle_lock_attr="_lifecycle_lock",
    ),
    boundary("skills", "unload_skill", "runtime", "skill.unload", "pid", "pid"),
    boundary("skills", "trust_skill_source", "runtime", "skill.trust", "actor", "actor"),
    boundary("skills", "untrust_skill_source", "runtime", "skill.untrust", "actor", "actor"),
    boundary("image_registry", "register", "runtime", "image.register", "actor", "actor"),
    boundary("image_registry", "register_from_package_files", "runtime", "image.register_package", "actor", "actor"),
    boundary(
        "image_registry", "commit_from_checkpoint", "runtime", "image.commit", "actor", "actor",
        preflight_method="preflight_checkpoint_commit",
    ),
    boundary("jsonrpc", "register_endpoint", "runtime", "jsonrpc.register", "actor", "actor"),
    boundary("jsonrpc", "unregister_endpoint", "runtime", "jsonrpc.unregister", "actor", "actor"),
    boundary("mcp", "register_server", "runtime", "mcp.register", "actor", "actor"),
    boundary("mcp", "unregister_server", "runtime", "mcp.unregister", "actor", "actor"),
)

EXPLAIN_BOUNDARY_DESCRIPTORS = (
    *PRIMITIVE_BOUNDARIES,
    *PROCESS_BOUNDARIES,
    *IMAGE_BOOT_BOUNDARIES,
    *MEMORY_BOUNDARIES,
    *CHECKPOINT_BOUNDARIES,
    *HUMAN_BOUNDARIES,
    *MESSAGE_BOUNDARIES,
    *OBJECT_TASK_BOUNDARIES,
    *AUTHORITY_BOUNDARIES,
    *EXTENSION_BOUNDARIES,
)

# HumanObjectManager is also a direct Host control surface (CLI/GUI and
# recovery code call it without going through a Tool).  Classify every public
# method here so a newly added Human method cannot silently escape lifecycle
# admission.  Methods in this set may persist state, publish protected
# evidence, call a provider, or update Host data-flow/presentation state.
HUMAN_CONTROL_MUTATION_ADMISSION_BOUNDARIES = (
    ("human", "request_data_release", "control.human.request_data_release"),
    ("human", "answer_for_request", "control.human.answer_for_request"),
    ("human", "approve", "control.human.approve"),
    ("human", "approve_for_presentation", "control.human.approve_for_presentation"),
    ("human", "reject", "control.human.reject"),
    ("human", "reject_for_presentation", "control.human.reject_for_presentation"),
    ("human", "list_for_presentation", "control.human.list_for_presentation"),
    (
        "human",
        "list_for_presentation_window",
        "control.human.list_for_presentation_window",
    ),
    ("human", "present_request_view", "control.human.present_request_view"),
    ("human", "cancel_pending_for_process", "control.human.cancel_pending_for_process"),
    ("human", "process_next_terminal", "control.human.process_next_terminal"),
    ("human", "aprocess_next_terminal", "control.human.aprocess_next_terminal"),
    ("human", "drain_terminal_queue", "control.human.drain_terminal_queue"),
    ("human", "adrain_terminal_queue", "control.human.adrain_terminal_queue"),
    ("human", "present_terminal_request", "control.human.present_terminal_request"),
    ("human", "recover_prepared_output", "control.human.recover_prepared_output"),
)

# These are the only public HumanObjectManager methods intentionally excluded
# from mutation admission.  The shutdown contract test compares this allowlist
# plus the mutation inventory with the concrete class, turning the
# classification into a static ratchet.
HUMAN_READ_ONLY_PUBLIC_METHODS = frozenset(
    {
        "format_terminal_request",
        "get",
        "is_request_withheld_for_presentation",
        "list",
        "pending",
        "public_request_payload",
        "public_request_view",
    }
)

_CAPABILITY_EXPLAIN_MUTATION_METHODS = frozenset(
    {"delegate", "derive_authority", "issue", "revoke"}
)

CAPABILITY_CONTROL_MUTATION_ADMISSION_BOUNDARIES = tuple(
    (
        "capability",
        method,
        f"control.capability.{method}",
    )
    for method in sorted(
        CAPABILITY_MANAGER_MUTATION_PUBLIC_METHODS
        - _CAPABILITY_EXPLAIN_MUTATION_METHODS
    )
)


# Runtime is the public composition-root facade.  Guard the facade as well as
# the delegated service so one lease spans validation, provider work, and all
# follow-up evidence.  Nested guards inherit the active lifecycle lease and do
# not increment the active-lease count a second time.
RUNTIME_PUBLIC_MUTATION_METHODS = frozenset(
    {
        "activate_skill",
        "add_handle_to_process_view",
        "arun_next_process_once",
        "arun_process_once",
        "arun_process_until_idle",
        "arun_until_idle",
        "arun_workflow",
        "exec_process",
        "fork_child_process",
        "register_image",
        "register_sink_trust",
        "register_skill_from_path",
        "resolve_process_working_directory",
        "run_next_process_once",
        "run_process_once",
        "run_process_until_idle",
        "run_until_idle",
        "run_workflow",
        "set_process_working_directory",
        "spawn_child_process",
        "trust_skill_source",
        "unload_skill",
        "unregister_sink_trust",
    }
)

RUNTIME_CONTROL_MUTATION_ADMISSION_BOUNDARIES = tuple(
    ("runtime", method, f"control.runtime.{method}")
    for method in sorted(RUNTIME_PUBLIC_MUTATION_METHODS)
)

RUNTIME_READ_ONLY_PUBLIC_METHODS = frozenset(
    {
        "current_human_run_context",
        "discover_skills",
        "get_image",
        "inspect_sink_trust",
        "inspect_skill",
        "list_sink_trust",
    }
)

# These methods deliberately manage host lifecycle or caller-local ContextVar
# state instead of entering the ordinary operation-admission gate.
RUNTIME_LIFECYCLE_PUBLIC_METHODS = frozenset(
    {
        "arelease_recovery_diagnostics",
        "ashutdown",
        "bind_shutdown_finalizer",
        "close",
        "release_recovery_diagnostics",
        "shutdown",
    }
)
RUNTIME_CONTEXT_PUBLIC_METHODS = frozenset({"human_run_context"})


SCHEDULER_PUBLIC_MUTATION_METHODS = frozenset(
    {
        "arun_once",
        "arun_pid_once",
        "arun_pid_until_idle",
        "arun_until_idle",
        "run_once",
        "run_pid_once",
        "run_pid_until_idle",
        "run_until_idle",
    }
)

SCHEDULER_CONTROL_MUTATION_ADMISSION_BOUNDARIES = tuple(
    ("scheduler", method, f"control.scheduler.{method}")
    for method in sorted(SCHEDULER_PUBLIC_MUTATION_METHODS)
)

SCHEDULER_READ_ONLY_PUBLIC_METHODS = frozenset(
    {"active_pids", "is_active_quantum", "next_runnable", "runnable_pids"}
)
SCHEDULER_COORDINATION_PUBLIC_METHODS = frozenset({"quiescent_state"})
SCHEDULER_LIFECYCLE_PUBLIC_METHODS = frozenset({"shutdown"})


# DataFlow has several methods that look like queries but deliberately append
# decisions or capability evidence.  Keep them in the mutation inventory;
# ambient-flow ContextVar helpers are classified separately so reset/finally
# cleanup remains possible after a recovery fence.
DATA_FLOW_PUBLIC_MUTATION_METHODS = frozenset(
    {
        "authorize_egress",
        "bind_written_file",
        "bind_written_file_digest",
        "bootstrap_configured_rules",
        "context_from_source_oids",
        "persist_denied_decision",
        "precheck_egress_clearance",
        "register_sink_trust",
        "reject_sink_identity_change",
        "run_sync_in_worker",
        "tombstone_file",
        "tombstone_path_tree",
        "unregister_sink_trust",
    }
)

DATA_FLOW_CONTROL_MUTATION_ADMISSION_BOUNDARIES = tuple(
    ("data_flow", method, f"control.data_flow.{method}")
    for method in sorted(DATA_FLOW_PUBLIC_MUTATION_METHODS)
)

DATA_FLOW_READ_ONLY_PUBLIC_METHODS = frozenset(
    {
        "classify_egress_snapshot",
        "context_from_materialization",
        "context_from_trusted_source_oids",
        "current_context",
        "directory_label_snapshot",
        "directory_label_state_version",
        "external_file_context",
        "file_context",
        "file_deletion_snapshot",
        "file_snapshot",
        "file_state_version",
        "file_tree_context",
        "file_tree_deletion_snapshot",
        "file_tree_snapshot",
        "file_tree_state_version",
        "inspect_sink_trust",
        "is_release_binding_current",
        "list_sink_trust",
        "provenance_sources",
        "release_binding",
        "resolve_sink_trust",
        "unclassified_ingress_context",
    }
)

DATA_FLOW_CONTEXT_PUBLIC_METHODS = frozenset(
    {
        "activate",
        "observe_ingress",
        "observe_unclassified_ingress",
        "push",
        "recovered_source_snapshot_access",
        "reset",
    }
)
DATA_FLOW_COMPOSITION_PUBLIC_METHODS = frozenset({"bind_human"})


_OBJECT_TASK_EXPLAIN_MUTATION_METHODS = frozenset(
    descriptor.method for descriptor in OBJECT_TASK_BOUNDARIES
)
OBJECT_TASK_CONTROL_MUTATION_ADMISSION_BOUNDARIES = (
    ("object_tasks", "get", "control.object_tasks.get"),
    ("object_tasks", "has_active_for_owner", "control.object_tasks.has_active_for_owner"),
    ("object_tasks", "list", "control.object_tasks.list"),
    ("object_tasks", "notify_owner_changed", "control.object_tasks.notify_owner_changed"),
    ("object_tasks", "notify_process_message", "control.object_tasks.notify_process_message"),
    ("object_tasks", "notify_process_terminal", "control.object_tasks.notify_process_terminal"),
    ("object_tasks", "shutdown", "control.object_tasks.shutdown"),
)
OBJECT_TASK_PUBLIC_MUTATION_METHODS = (
    _OBJECT_TASK_EXPLAIN_MUTATION_METHODS
    | frozenset(
        method
        for _component, method, _name in OBJECT_TASK_CONTROL_MUTATION_ADMISSION_BOUNDARIES
    )
)
OBJECT_TASK_READ_ONLY_PUBLIC_METHODS = frozenset({"is_runner_pid"})
OBJECT_TASK_RECOVERY_PUBLIC_METHODS = frozenset({"recover"})
OBJECT_TASK_LIFECYCLE_PUBLIC_METHODS = frozenset({"start_worker"})

# Public Host/control mutations that do not open their own explainable
# operation.  They still require the same lifecycle admission lease.  Keeping
# this list beside the explainable boundary inventory makes admission coverage
# machine-checkable instead of relying on a representative shutdown test.
CONTROL_MUTATION_ADMISSION_BOUNDARIES = (
    *HUMAN_CONTROL_MUTATION_ADMISSION_BOUNDARIES,
    *CAPABILITY_CONTROL_MUTATION_ADMISSION_BOUNDARIES,
    *RUNTIME_CONTROL_MUTATION_ADMISSION_BOUNDARIES,
    *SCHEDULER_CONTROL_MUTATION_ADMISSION_BOUNDARIES,
    *DATA_FLOW_CONTROL_MUTATION_ADMISSION_BOUNDARIES,
    *OBJECT_TASK_CONTROL_MUTATION_ADMISSION_BOUNDARIES,
    (
        "process",
        "add_handle_to_process_view",
        "control.process.add_handle_to_process_view",
    ),
    (
        "checkpoint",
        "reconcile_terminal_restore_publications",
        "control.checkpoint.reconcile_terminal_restore_publications",
    ),
    ("tools", "register_tool", "control.tools.register_tool"),
    ("tools", "unregister_tool", "control.tools.unregister_tool"),
    ("tools", "discard_tool_registration", "control.tools.discard_tool_registration"),
    ("tools", "configure_process_tools", "control.tools.configure_process_tools"),
    ("tools", "grant_execute", "control.tools.grant_execute"),
    ("tools", "propose", "control.tools.propose"),
    ("tools", "validate", "control.tools.validate"),
    ("tools", "register", "control.tools.register"),
    ("tools", "discard_candidate", "control.tools.discard_candidate"),
    ("tools", "call", "control.tools.call"),
    ("tools", "acall", "control.tools.acall"),
    ("tools", "configure_model_tool_projection", "control.tools.configure_model_tool_projection"),
    ("tools", "restore_loaded_jit_state", "control.tools.restore_loaded_jit_state"),
    ("tools", "forget_loaded_jit", "control.tools.forget_loaded_jit"),
    ("tools", "install_committed_jit", "control.tools.install_committed_jit"),
    ("tools", "rehydrate_registered_jit_tools", "control.tools.rehydrate_registered_jit_tools"),
    ("modules", "load_core_module", "control.modules.load_core_module"),
    ("modules", "load_startup_modules", "control.modules.load_startup_modules"),
    ("modules", "load_module_manifest", "control.modules.load_module_manifest"),
    ("modules", "run_startup_hooks", "control.modules.run_startup_hooks"),
)

HUMAN_PUBLIC_MUTATION_METHODS = frozenset(
    descriptor.method
    for descriptor in HUMAN_BOUNDARIES
) | frozenset(
    method
    for component, method, _name in HUMAN_CONTROL_MUTATION_ADMISSION_BOUNDARIES
    if component == "human"
)

PUBLIC_MUTATION_ADMISSION_BOUNDARY_NAMES = frozenset(
    descriptor.name for descriptor in EXPLAIN_BOUNDARY_DESCRIPTORS
) | frozenset(name for _component, _method, name in CONTROL_MUTATION_ADMISSION_BOUNDARIES)


def validate_explain_descriptors() -> None:
    names = [descriptor.name for descriptor in EXPLAIN_BOUNDARY_DESCRIPTORS]
    if len(names) != len(set(names)):
        raise ValueError("duplicate explain boundary descriptor")
    control_names = [name for _component, _method, name in CONTROL_MUTATION_ADMISSION_BOUNDARIES]
    control_targets = [
        (component, method)
        for component, method, _name in CONTROL_MUTATION_ADMISSION_BOUNDARIES
    ]
    if len(control_names) != len(set(control_names)):
        raise ValueError("duplicate control mutation admission boundary name")
    if len(control_targets) != len(set(control_targets)):
        raise ValueError("duplicate control mutation admission boundary target")
    if set(names) & set(control_names):
        raise ValueError("explain and control admission boundary names overlap")


validate_explain_descriptors()


__all__ = [
    "CAPABILITY_CONTROL_MUTATION_ADMISSION_BOUNDARIES",
    "CONTROL_MUTATION_ADMISSION_BOUNDARIES",
    "DATA_FLOW_COMPOSITION_PUBLIC_METHODS",
    "DATA_FLOW_CONTEXT_PUBLIC_METHODS",
    "DATA_FLOW_CONTROL_MUTATION_ADMISSION_BOUNDARIES",
    "DATA_FLOW_PUBLIC_MUTATION_METHODS",
    "DATA_FLOW_READ_ONLY_PUBLIC_METHODS",
    "EXPLAIN_BOUNDARY_DESCRIPTORS",
    "HUMAN_CONTROL_MUTATION_ADMISSION_BOUNDARIES",
    "HUMAN_PUBLIC_MUTATION_METHODS",
    "HUMAN_READ_ONLY_PUBLIC_METHODS",
    "OBJECT_TASK_CONTROL_MUTATION_ADMISSION_BOUNDARIES",
    "OBJECT_TASK_LIFECYCLE_PUBLIC_METHODS",
    "OBJECT_TASK_PUBLIC_MUTATION_METHODS",
    "OBJECT_TASK_READ_ONLY_PUBLIC_METHODS",
    "OBJECT_TASK_RECOVERY_PUBLIC_METHODS",
    "PUBLIC_MUTATION_ADMISSION_BOUNDARY_NAMES",
    "RUNTIME_CONTEXT_PUBLIC_METHODS",
    "RUNTIME_CONTROL_MUTATION_ADMISSION_BOUNDARIES",
    "RUNTIME_LIFECYCLE_PUBLIC_METHODS",
    "RUNTIME_PUBLIC_MUTATION_METHODS",
    "RUNTIME_READ_ONLY_PUBLIC_METHODS",
    "SCHEDULER_CONTROL_MUTATION_ADMISSION_BOUNDARIES",
    "SCHEDULER_COORDINATION_PUBLIC_METHODS",
    "SCHEDULER_LIFECYCLE_PUBLIC_METHODS",
    "SCHEDULER_PUBLIC_MUTATION_METHODS",
    "SCHEDULER_READ_ONLY_PUBLIC_METHODS",
]
