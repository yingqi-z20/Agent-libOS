from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Mapping, Protocol

from agent_libos.utils.serde import dumps, loads


class TaskRunDispatchDeferred(RuntimeError):
    """A persisted TaskRun control generation refused a new dispatch.

    This is an internal scheduler boundary, not a failed model action.  The
    executor converts it to a skipped quantum after any already-started call
    has committed its local settlement.
    """


class TaskRunLLMHook(Protocol):
    """Narrow boundary from validated local LLM transcripts to Task Runs.

    The executor calls this hook only after the provider completion is present
    in the local LLM-call ledger and every normalized action in the completion
    has passed dispatch validation.  Implementations must treat ``call_id`` as
    the transcript authority.  Provider-side response state is deliberately
    excluded from this recovery contract.
    """

    def record_validated_transcript(
        self,
        *,
        pid: str,
        call_id: str,
        action_manifest: Mapping[str, Any],
        context_generation: str,
    ) -> None:
        ...

    def record_completed_transcript(
        self,
        *,
        pid: str,
        call_id: str,
        outcome_manifest: Mapping[str, Any],
        context_generation: str,
    ) -> None:
        ...

    def stage_completed_transcript(
        self,
        *,
        pid: str,
        call_id: str,
        outcome_manifest: Mapping[str, Any],
        context_generation: str,
    ) -> None:
        ...

    def pending_validated_action_for_pid(
        self,
        pid: str,
    ) -> Mapping[str, Any] | None:
        ...

    def expected_tool_id_for_pending_action(
        self,
        pid: str,
        action: Mapping[str, Any],
    ) -> str | None:
        """Return the integrity-bound tool identity for a durable dispatch."""

        ...

    def request_binding_hash_for_pid(self, pid: str) -> str | None:
        """Return the current durable Image/tool/provider request binding."""

        ...

    def settlement_binding_hash_for_pid(self, pid: str) -> str | None:
        """Return the binding for an already-admitted call's settlement."""

        ...

    def dispatch_scope_for_pid(
        self,
        pid: str,
        kind: str,
    ) -> AbstractContextManager[None]:
        """Atomically admit one provider/tool call against Run control state."""

        ...

    def defer_unstarted_action_for_pid(self, pid: str) -> None:
        """Rewind a claimed action when its tool call was never admitted."""

        ...

    def mark_request_scope_drift_for_pid(self, pid: str) -> None:
        """Fail closed when a Provider response outlives its request binding."""

        ...

    def prompt_context_for_pid(self, pid: str) -> Mapping[str, Any] | None:
        ...

    def requirement_binding_for_prompt(
        self,
        pid: str,
        *,
        context_generation: str,
    ) -> Mapping[str, Any] | None:
        """Return the Host-authored requirement identities frozen for a prompt."""

        ...


_PROMPT_CONTEXT_MAX_BYTES = 16 * 1024 * 1024
_PROMPT_CONTEXT_MAX_TRANSCRIPT_MESSAGES = 10_000
_PROMPT_CONTEXT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "context_generation",
        "goal_text",
        "requirements",
        "transcript_messages",
        "compressed_summary",
        "data_labels",
    }
)
_REQUIREMENT_KEYS = frozenset(
    {"requirement_id", "kind", "content_text", "status"}
)
_VALIDATED_ACTION_KEYS = frozenset(
    {
        "schema_version",
        "call_id",
        "actions",
        "parallel_tool_calls",
        "host_auto_wait",
        "tool_call_count",
        "data_labels",
        "previous_response_id_used",
    }
)


def validated_action_manifest(
    actions: list[dict[str, Any]],
    *,
    call_id: str,
    parallel_tool_calls: bool,
    host_auto_wait: bool,
    tool_call_count: int,
    data_labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the versioned, provider-independent resume-point projection."""

    manifest = {
        "schema_version": 1,
        "call_id": call_id,
        "actions": [dict(action) for action in actions],
        "parallel_tool_calls": parallel_tool_calls,
        "host_auto_wait": host_auto_wait,
        "tool_call_count": tool_call_count,
        "data_labels": dict(data_labels),
        # The full-snapshot executor rebuilds correctness-critical context
        # locally.  A Provider response id may remain useful observability, but
        # it is never part of the durable TaskRun resume contract.
        "previous_response_id_used": False,
    }
    return normalize_validated_action_manifest(manifest)


def normalize_validated_action_manifest(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a complete local action bundle before any recovered dispatch."""

    _require_exact_keys(
        value,
        _VALIDATED_ACTION_KEYS,
        "TaskRun validated action manifest",
    )
    _require_schema_one(value, "TaskRun validated action manifest")
    _require_nonempty_text(value.get("call_id"), "TaskRun validated action call_id")
    _require_boolean_fields(
        value,
        ("parallel_tool_calls", "host_auto_wait"),
        "TaskRun validated action",
    )
    tool_call_count = _require_nonnegative_integer(
        value.get("tool_call_count"),
        "TaskRun validated action tool_call_count",
    )
    _require_mapping(value.get("data_labels"), "TaskRun validated action data_labels")
    if value.get("previous_response_id_used") is not False:
        raise ValueError(
            "provider previous_response_id cannot be the durable action source"
        )
    actions = _validated_actions(value.get("actions"))
    parallel = bool(value["parallel_tool_calls"])
    host_auto_wait = bool(value["host_auto_wait"])
    if host_auto_wait and (len(actions) != 1 or tool_call_count != 0):
        raise ValueError("TaskRun host auto-wait manifest is invalid")
    if parallel and len(actions) > 1 and tool_call_count < len(actions):
        raise ValueError("TaskRun parallel action manifest lost tool-call evidence")
    return _detached_object(value, "validated action manifest")


def completed_outcome_manifest(
    *,
    state: str,
    paired_outputs_persisted: bool,
    data_labels: Mapping[str, Any],
    result: Mapping[str, Any] | None = None,
    durable_wait: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical local outcome projection for one transcript head."""

    if state not in {"completed", "waiting"}:
        raise ValueError("TaskRun transcript outcome state is invalid")
    if not isinstance(paired_outputs_persisted, bool):
        raise ValueError("paired_outputs_persisted must be a boolean")
    if state == "waiting" and durable_wait is None:
        raise ValueError("waiting TaskRun transcript requires durable wait evidence")
    if state != "waiting" and durable_wait is not None:
        raise ValueError("non-waiting TaskRun transcript cannot carry a durable wait")
    if durable_wait is not None:
        wait_type = durable_wait.get("wait_type")
        if wait_type not in {"human", "process", "message"}:
            raise ValueError("TaskRun durable wait type is invalid")
    if not isinstance(data_labels, Mapping):
        raise ValueError("TaskRun outcome data_labels must be an object")
    manifest = {
        "schema_version": 1,
        "state": state,
        "paired_outputs_persisted": paired_outputs_persisted,
        "data_labels": dict(data_labels),
        "result": dict(result) if result is not None else None,
        "durable_wait": dict(durable_wait) if durable_wait is not None else None,
        "previous_response_id_used": False,
    }
    selected = loads(dumps(manifest))
    if not isinstance(selected, dict):  # pragma: no cover - fixed local shape
        raise RuntimeError("completed transcript manifest is not an object")
    return selected


def normalize_task_run_prompt_context(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete, local TaskRun prompt/resume projection.

    This boundary rejects partial or Provider-dependent recovery state.  The
    manager remains responsible for integrity-checking its payload and fencing
    the Run before returning this projection.
    """

    _require_exact_keys(value, _PROMPT_CONTEXT_KEYS, "TaskRun prompt context")
    _require_schema_one(value, "TaskRun prompt context")
    for name in ("run_id", "context_generation", "goal_text"):
        _require_nonempty_text(value.get(name), f"TaskRun prompt context {name}")
    summary = value.get("compressed_summary")
    if summary is not None and not isinstance(summary, str):
        raise ValueError("TaskRun compressed summary must be text or null")
    _require_mapping(value.get("data_labels"), "TaskRun prompt context data_labels")
    _validate_prompt_requirements(value.get("requirements"))
    _validate_prompt_transcript(value.get("transcript_messages"))
    encoded = dumps(dict(value)).encode("utf-8")
    if len(encoded) > _PROMPT_CONTEXT_MAX_BYTES:
        raise ValueError("TaskRun prompt context exceeds the hard byte limit")
    return _detached_object(value, "TaskRun prompt context")


def _validate_prompt_requirements(value: Any) -> None:
    if not isinstance(value, list):
        raise ValueError("TaskRun prompt context requirements must be a list")
    seen_requirement_ids: set[str] = set()
    for requirement in value:
        _require_exact_keys(
            requirement,
            _REQUIREMENT_KEYS,
            "TaskRun prompt requirement",
        )
        for name in _REQUIREMENT_KEYS:
            _require_nonempty_text(
                requirement.get(name),
                f"TaskRun prompt requirement {name}",
            )
        requirement_id = str(requirement["requirement_id"])
        if requirement_id in seen_requirement_ids:
            raise ValueError("TaskRun prompt requirements contain duplicate ids")
        seen_requirement_ids.add(requirement_id)


def _validate_prompt_transcript(value: Any) -> None:
    if not isinstance(value, list):
        raise ValueError("TaskRun transcript_messages must be a list")
    if len(value) > _PROMPT_CONTEXT_MAX_TRANSCRIPT_MESSAGES:
        raise ValueError("TaskRun transcript contains too many messages")
    for message in value:
        if not isinstance(message, Mapping):
            raise ValueError("TaskRun transcript message must be an object")
        if message.get("role") not in {"assistant", "tool", "user"}:
            raise ValueError("TaskRun transcript message role is invalid")


def _require_exact_keys(value: Any, keys: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} has an invalid shape")


def _require_schema_one(value: Mapping[str, Any], label: str) -> None:
    schema = value.get("schema_version")
    if type(schema) is not int or schema != 1:
        raise ValueError(f"{label} schema is unsupported")


def _require_nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value


def _require_boolean_fields(
    value: Mapping[str, Any],
    names: tuple[str, ...],
    label: str,
) -> None:
    for name in names:
        if type(value.get(name)) is not bool:
            raise ValueError(f"{label} {name} must be boolean")


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} is invalid")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _validated_actions(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > 256:
        raise ValueError("TaskRun validated action list is invalid")
    if any(not isinstance(action, Mapping) or not action for action in value):
        raise ValueError("TaskRun validated action entry is invalid")
    return value


def _detached_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    selected = loads(dumps(dict(value)))
    if not isinstance(selected, dict):  # pragma: no cover - validated above
        raise RuntimeError(f"{label} is not an object")
    return selected


def task_run_contract_message(context: Mapping[str, Any]) -> str:
    """Render the cumulative durable contract without Provider state."""

    selected = normalize_task_run_prompt_context(context)
    contract = {
        "schema_version": 1,
        "run_id": selected["run_id"],
        "goal": selected["goal_text"],
        "requirements": selected["requirements"],
        "compressed_summary": selected["compressed_summary"],
    }
    return (
        "Durable TaskRun contract (authoritative across Runtime restarts):\n"
        f"{dumps(contract)}\n"
        "Every requirement whose status is pending or in_progress remains "
        "mandatory. Track and verify each requirement_id independently; a "
        "blocked or model-cancelled item is not satisfied or Host-waived. "
        "Provider response state is not an authority for recovery."
    )


__all__ = [
    "TaskRunLLMHook",
    "completed_outcome_manifest",
    "normalize_validated_action_manifest",
    "normalize_task_run_prompt_context",
    "task_run_contract_message",
    "validated_action_manifest",
]
