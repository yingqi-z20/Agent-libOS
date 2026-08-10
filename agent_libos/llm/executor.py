from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, TYPE_CHECKING

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProcessMessageWaitRequired,
    ProcessWaitRequired,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.public_errors import public_error_envelope
from agent_libos.utils.serde import dumps, to_jsonable
from agent_libos.llm.client import (
    LLMClient,
    LLMCompletion,
    LLMError,
    LLMTransientError,
    llm_error_internal_observation,
)
from agent_libos.llm.action_parser import parse_json_action
from agent_libos.llm.context_memory import (
    LLM_CONTEXT_MAINTENANCE_RESOURCE,
    LLMContextMemory,
    LLMContextStoragePressure,
)
from agent_libos.llm.context_management import (
    ContextManagementPolicy,
    ContextPressureAssessment,
    assess_context_pressure,
    context_management_policy,
    context_pressure_prompt,
    estimate_request_input_tokens,
    provider_usage_lower_bound,
)
from agent_libos.llm.event_projection import project_prompt_events
from agent_libos.llm.prompt import (
    build_system_prompt,
    build_user_prompt,
    recover_initial_goal_context,
)
from agent_libos.llm.records import observable_llm_call_fields
from agent_libos.llm.provider_trace import (
    custom_provider_trace,
    is_provider_trace,
    project_provider_raw_response,
    provider_trace_from_error,
    provider_trace_summary,
)
from agent_libos.llm.usage import canonicalize_llm_usage
from agent_libos.llm.tool_protocol import tool_call_to_action
from agent_libos.llm.task_runs import (
    TaskRunDispatchDeferred,
    TaskRunLLMHook,
    completed_outcome_manifest,
    normalize_task_run_prompt_context,
    normalize_validated_action_manifest,
    task_run_contract_message,
    validated_action_manifest,
)
from agent_libos.llm.pending import (
    LLMPendingActionService,
    PENDING_TASK_RUN_TRANSCRIPT_KEY,
    pending_data_flow_metadata,
    pending_metadata,
    pending_resume_token,
    pending_task_run_transcript_call_id,
)
from agent_libos.llm.actions import LLMActionService, auto_wait_message_action
from agent_libos.llm.provider_service import LLMProviderService
from agent_libos.process_transition import ProcessTransitionService
from agent_libos.ports import (
    AuditPort,
    AuthorityManifestPort,
    DataFlowPort,
    EventPort,
    OperationPort,
    ProcessControlPort,
    ProcessMessagePort,
    ResourcePort,
)
from agent_libos.storage import UnitOfWork
from agent_libos.tools.observability import sanitize_for_observability
from agent_libos.models import (
    CapabilityRight,
    DataFlowContext,
    DataLabels,
    DataSink,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    FailedProcessOutcome,
    HumanRequestStatus,
    LLMCallRecord,
    MaterializedContext,
    ObjectHandle,
    ObjectRight,
    ObjectType,
    PROMPT_MODE_IMAGE_ONLY,
    ProcessMessageKind,
    ProcessStatus,
    ResourceUsage,
    ViewMode,
)
from agent_libos.sdk import (
    PostCommitResultObservation,
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProviderPhase,
    ResourceSettlement,
)
from agent_libos.substrate import ProviderEffectNotStarted

if TYPE_CHECKING:
    from agent_libos.capability.manager import CapabilityManager
    from agent_libos.human.manager import HumanObjectManager
    from agent_libos.llm.profiles import LLMProfileRegistry
    from agent_libos.memory.object_memory import ObjectMemoryManager
    from agent_libos.sdk import ProtectedOperationSDK
    from agent_libos.skills.manager import SkillManager
    from agent_libos.tools.broker import ToolBroker


_HOST_AUTO_WAIT_METADATA = {
    "kind": "host_generated_action",
    "schema_version": 1,
    "source": "llm.empty_tool_calls_auto_wait",
    "tool_name": "receive_process_messages",
}

_FULL_SNAPSHOT_RESPONSE_CHAIN_DISABLED_REASON = "full_snapshot_executor_replay"
_TASK_RUN_REQUIREMENT_BINDING_KEY = "task_run_requirement_binding_v1"
_IMAGE_ONLY_FROZEN_ANCHOR_KEY = "image_only_request_anchor"
_IMAGE_ONLY_FROZEN_ANCHOR_SCHEMA_VERSION = 1
_IMAGE_ONLY_REQUEST_KEY = "image_only_request"
_IMAGE_ONLY_REQUEST_PURPOSE_PREFIX = "image_only_request"
_IMAGE_ONLY_ERROR_PURPOSE = "image_only_error"
_IMAGE_ONLY_REQUEST_SCHEMA_VERSION = 1
_IMAGE_ONLY_TRANSCRIPT_KEY = "image_only_transcript"
_IMAGE_ONLY_TRANSCRIPT_SCHEMA_VERSION = 2
_IMAGE_ONLY_EMPTY_HEAD_VALIDATION_KEY = "image_only_empty_head_validation"
_IMAGE_ONLY_EMPTY_HEAD_VALIDATION_PURPOSE_PREFIX = "image_only_empty_validation"
_IMAGE_ONLY_EMPTY_HEAD_VALIDATION_SCHEMA_VERSION = 1
_IMAGE_ONLY_TOOL_OUTPUT_SCHEMA_VERSION = 1


class _LLMProviderChainScopeChanged(ProviderEffectNotStarted):
    """The selected provider-side state no longer matches the dispatch scope."""


class _LLMReleaseApprovalRequired(HumanApprovalRequired):
    """A conditional LLM request whose exact prepared payload must be resumed."""

    def __init__(
        self,
        original: HumanApprovalRequired,
        prepared_request: dict[str, Any],
    ) -> None:
        super().__init__(original.request_id, str(original))
        self.prepared_request = prepared_request


class _LLMReleasePayloadUnavailable(RuntimeError):
    """An opt-out release cannot be resumed after its in-memory payload is lost."""


class _ImageOnlyFullIORequired(ValidationError):
    """Transparent replay is impossible without the durable full-I/O ledger."""


class _ContextManagementHandled(Exception):
    """End this quantum after a host-selected context-management action."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("context management handled this quantum")
        self.result = result


@dataclass(slots=True)
class _LLMCallState:
    pid: str
    process: Any
    call_id: str
    created_at: str
    profile_id: str
    attempt: int
    max_attempts: int
    request_options: dict[str, Any]
    request_messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    flow_context: DataFlowContext
    resolved: Any | None = None
    client: Any | None = None
    sink: DataSink | None = None
    data_flow_chain_fingerprint: str = ""
    source_refs_fingerprint: str = ""
    provider_chain_fingerprint: str | None = None
    previous_response_id: str | None = None
    parallel_tool_calls: bool = False
    auto_wait_on_empty_tool_calls: bool = False
    fallback_json_actions: bool = False
    temperature: float = 0.0
    max_tokens: int = 0
    max_input_tokens_per_call: int = 0
    max_total_tokens_per_call: int = 0
    estimated_input_tokens: int = 0
    resource_envelope: ResourceUsage | None = None
    resource_envelope_sha256: str = ""
    completion_usage: dict[str, int] = field(default_factory=dict)
    invalid_completion_usage_fields: set[str] = field(default_factory=set)
    provider_trace: dict[str, Any] | None = None
    provider_dispatched: bool = False
    # Runtime-internal, transient identity of the durably committed provider
    # result.  It is populated by the protected-operation observer and lets
    # the Host FlowGraph bind MODEL_OUTPUT to the exact PROVIDER_RESULT row.
    semantic_provider_observation: PostCommitResultObservation | None = field(
        default=None,
        repr=False,
    )
    budget_admission_denial_audited: bool = False
    egress_payload: dict[str, Any] = field(default_factory=dict)
    canonical_args: dict[str, Any] = field(default_factory=dict)
    resumed_release: bool = False

    @property
    def prepared(self) -> bool:
        return self.resolved is not None and self.client is not None and self.sink is not None


class LLMProcessExecutor:
    """Runs one model-selected tool action per process quantum."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWork,
        process: ProcessControlPort,
        operations: OperationPort,
        data_flow: DataFlowPort,
        tools: "ToolBroker",
        resources: ResourcePort | None,
        llms: "LLMProfileRegistry",
        memory: "ObjectMemoryManager",
        audit: AuditPort,
        events: EventPort,
        images: Mapping[str, Any],
        messages: ProcessMessagePort,
        human: "HumanObjectManager",
        skills: "SkillManager",
        protected_operations: "ProtectedOperationSDK",
        authority_manifests: AuthorityManifestPort,
        capabilities: "CapabilityManager",
        client: LLMClient | None = None,
        config: AgentLibOSConfig | None = None,
        blocking_work: Any | None = None,
        task_runs: TaskRunLLMHook | None = None,
        host_semantic_result_observer: Callable[..., None] | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self._processes = unit_of_work.processes
        self._objects = unit_of_work.objects
        self._authority = unit_of_work.authority
        self._evidence = unit_of_work.evidence
        self._process = process
        self._process_transitions = getattr(process, "transitions", None)
        if self._process_transitions is None:
            self._process_transitions = ProcessTransitionService(self._processes)
        self._operations = operations
        self._data_flow = data_flow
        self._tools = tools
        self._resources = resources
        self._llms = llms
        self._memory = memory
        self._audit = audit
        self._events = events
        self._images = images
        self._messages = messages
        self._human = human
        self._skills = skills
        self._protected_operations = protected_operations
        self._authority_manifests = authority_manifests
        self._capabilities = capabilities
        self._task_runs = task_runs
        if host_semantic_result_observer is not None and not callable(
            host_semantic_result_observer
        ):
            raise TypeError("semantic model-result observer must be callable")
        self._host_semantic_result_observer = host_semantic_result_observer
        if client is not None:
            self._llms.set_test_client(self.config.llm.default_profile_id, client)
        self.pending = LLMPendingActionService(
            processes=self._processes,
            evidence=self._evidence,
            operations=self._operations,
            data_flow=self._data_flow,
            restore_child_goal=self._restore_pending_compaction_child_goal,
        )
        self.provider = LLMProviderService(blocking_work)
        self.actions = LLMActionService(
            processes=self._processes,
            tools=self._tools,
            resources=self._resources,
            content_preview_chars=self.config.llm.content_preview_chars,
            tool_call_args_hard_limit_bytes=(
                self.config.tools.tool_call_args_hard_limit_bytes
            ),
            pre_tool_notice=self._pre_tool_interrupt_notice,
            post_tool_notice=self._notify_normal_messages,
            publish_result=self._add_to_view,
        )
        self.context_memory = LLMContextMemory(
            self._processes,
            self._objects,
            self._evidence,
            self._memory,
            self._capabilities,
            self._operations,
            self._resources,
            config=self.config,
        )
        self._load_pending_actions()

    @property
    def client(self) -> Any:
        """Compatibility view of the default LLM profile client."""
        return self._llms.default_client

    @client.setter
    def client(self, value: Any) -> None:
        self._llms.set_test_client(self.config.llm.default_profile_id, value)

    def bind_host_semantic_result_observer(
        self,
        observer: Callable[..., None],
    ) -> None:
        """Install the one-shot Host observer for payload-free model lineage."""

        if not callable(observer):
            raise TypeError("semantic model-result observer must be callable")
        if self._host_semantic_result_observer is not None:
            raise RuntimeError("semantic model-result observer is already bound")
        self._host_semantic_result_observer = observer

    def _observe_host_semantic_result(
        self,
        state: _LLMCallState,
        completion: Any,
        record: Any,
    ) -> None:
        observer = self._host_semantic_result_observer
        if observer is None:
            return
        try:
            result = observer(state, completion, record)
            if result is not None:
                raise TypeError("semantic model-result observer must be synchronous")
        except Exception:
            # Semantic lineage is observational and cannot change a completed
            # provider call or its durable LLM record.
            return

    @staticmethod
    def _bind_semantic_provider_observation(
        state: _LLMCallState,
        _result: Any,
        observation: PostCommitResultObservation,
    ) -> None:
        """Bind one committed LLM result to its transient call state."""

        if (
            not isinstance(observation, PostCommitResultObservation)
            or observation.contract_name != "primitive.llm.complete"
            or observation.pid != state.pid
            or not observation.effect_id
            or not observation.result_sha256
        ):
            raise ValidationError("LLM semantic provider observation is malformed")
        state.semantic_provider_observation = observation

    def _requestable_capabilities_for_prompt(
        self,
        pid: str,
    ) -> list[dict[str, Any]]:
        manifest = self._authority_manifests.get_for_process(pid)
        if manifest is None or not isinstance(manifest.approval_policy, Mapping):
            return []
        raw_requestable = manifest.approval_policy.get(
            "requestable_capabilities",
            [],
        )
        if not isinstance(raw_requestable, list):
            return []
        return [
            dict(spec)
            for spec in raw_requestable
            if isinstance(spec, Mapping)
        ]

    def _retained_original_goal_context(
        self,
        *,
        process: Any,
        image: Any,
    ) -> str | None:
        if image.metadata.get("completion_gate") != "cumulative_review":
            return None
        goal_oid = str(process.goal_oid or "")
        if not goal_oid:
            return None
        calls = self._processes.list_llm_calls(
            pid=process.pid,
            limit=self.config.llm.call_record_hard_limit,
        )
        try:
            return recover_initial_goal_context(calls, goal_oid)
        except ValueError:
            # The completion gate owns the fail-closed user-facing recovery
            # path for an oversized or unavailable retained goal.
            return None

    def _include_retained_goal_labels(
        self,
        flow_context: DataFlowContext,
        goal_oid: str | None,
    ) -> DataFlowContext:
        if not goal_oid:
            return flow_context
        metadata = self._objects.get_persisted_object_metadata(goal_oid)
        if metadata is None:
            return flow_context
        return DataFlowContext.aggregate(
            (
                flow_context,
                DataFlowContext(labels=DataLabels.from_object_metadata(metadata)),
            )
        )

    def _task_run_prompt_context(self, pid: str) -> dict[str, Any] | None:
        if self._task_runs is None:
            return None
        selected = self._task_runs.prompt_context_for_pid(pid)
        if selected is None:
            return None
        try:
            return normalize_task_run_prompt_context(selected)
        except (TypeError, ValueError) as exc:
            raise ValidationError("durable TaskRun prompt context is invalid") from exc

    def _task_run_requirement_binding(
        self,
        pid: str,
        context: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Freeze the Host-authored requirement set sent in this request.

        The binding is deliberately kept out of model-visible prompt text.  It
        travels in the local LLM-call options so a release approval/reopen keeps
        the exact requirement set that preceded the Provider call.
        """

        if self._task_runs is None or context is None:
            return None
        selected = self._task_runs.requirement_binding_for_prompt(
            pid,
            context_generation=str(context["context_generation"]),
        )
        if selected is None:
            return None
        try:
            detached = json.loads(dumps(dict(selected)))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "durable TaskRun requirement binding is invalid"
            ) from exc
        if not isinstance(detached, dict):
            raise ValidationError("durable TaskRun requirement binding is invalid")
        return detached

    def _pending_task_run_validated_action(
        self,
        pid: str,
    ) -> dict[str, Any] | None:
        if self._task_runs is None:
            return None
        getter = getattr(
            self._task_runs,
            "pending_validated_action_for_pid",
            None,
        )
        # Legacy/non-Durable hook implementations have no recovery bundle.
        # The real TaskRun manager implements this method; absence must never
        # make an ordinary AgentProcess call the Provider fail.
        if not callable(getter):
            return None
        selected = getter(pid)
        if selected is None:
            return None
        try:
            return normalize_validated_action_manifest(selected)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "durable TaskRun pending action manifest is invalid"
            ) from exc

    def _pending_task_run_action_boundary(
        self,
        pid: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Convert a persisted control fence into a skipped quantum marker."""

        try:
            return self._pending_task_run_validated_action(pid), False
        except TaskRunDispatchDeferred:
            return None, True

    def _task_run_expected_tool_id(
        self,
        pid: str,
        action: Mapping[str, Any],
    ) -> str | None:
        if self._task_runs is None:
            return None
        getter = getattr(
            self._task_runs,
            "expected_tool_id_for_pending_action",
            None,
        )
        if not callable(getter):
            return None
        selected = getter(pid, action)
        if selected is not None and (
            not isinstance(selected, str) or not selected
        ):
            raise ValidationError("durable TaskRun exact tool identity is invalid")
        return selected

    def _task_run_request_binding_hash(self, pid: str) -> str | None:
        if self._task_runs is None:
            return None
        getter = getattr(self._task_runs, "request_binding_hash_for_pid", None)
        if not callable(getter):
            return None
        selected = getter(pid)
        if selected is None:
            return None
        if (
            not isinstance(selected, str)
            or len(selected) != 64
            or any(character not in "0123456789abcdef" for character in selected)
        ):
            raise ValidationError("durable TaskRun request binding hash is invalid")
        return selected

    def _task_run_settlement_binding_hash(self, pid: str) -> str | None:
        if self._task_runs is None:
            return None
        getter = getattr(
            self._task_runs,
            "settlement_binding_hash_for_pid",
            None,
        )
        if not callable(getter):
            return self._task_run_request_binding_hash(pid)
        selected = getter(pid)
        if selected is None:
            return None
        if (
            not isinstance(selected, str)
            or len(selected) != 64
            or any(character not in "0123456789abcdef" for character in selected)
        ):
            raise ValidationError("durable TaskRun settlement binding hash is invalid")
        return selected

    def _task_run_dispatch_scope(self, pid: str, kind: str):
        if self._task_runs is None:
            return nullcontext()
        getter = getattr(self._task_runs, "dispatch_scope_for_pid", None)
        if not callable(getter):
            return nullcontext()
        return getter(pid, kind)

    def _defer_unstarted_task_run_action(self, pid: str) -> None:
        if self._task_runs is None:
            return
        defer = getattr(self._task_runs, "defer_unstarted_action_for_pid", None)
        if callable(defer):
            defer(pid)

    def _mark_task_run_request_scope_drift(self, pid: str) -> None:
        if self._task_runs is None:
            return
        marker = getattr(self._task_runs, "mark_request_scope_drift_for_pid", None)
        if callable(marker):
            marker(pid)

    async def _dispatch_pending_task_run_validated_action(
        self,
        pid: str,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Dispatch one integrity-checked local bundle without another LLM call."""

        actions = [dict(action) for action in manifest["actions"]]
        parallel_tool_calls = bool(manifest["parallel_tool_calls"])
        host_auto_wait = bool(manifest["host_auto_wait"])
        if host_auto_wait:
            if len(actions) != 1:
                raise ValidationError(
                    "durable TaskRun host auto-wait action count changed"
                )
            self.actions.validate_host_auto_wait(pid, actions[0])
        else:
            for action in actions:
                self._validate_dispatchable_action(pid, action)
        if parallel_tool_calls and len(actions) > 1:
            self._preflight_parallel_tool_batch(pid, actions)

        call_id = str(manifest["call_id"])
        tool_call_count = int(manifest["tool_call_count"])
        tool_calls: list[dict[str, Any]] = []
        if tool_call_count:
            selected_actions = actions if parallel_tool_calls else [actions[-1]]
            padding = max(0, tool_call_count - len(selected_actions))
            tool_calls.extend({} for _ in range(padding))
            for index, action in enumerate(selected_actions, start=padding):
                digest = hashlib.sha256(
                    f"{call_id}:{index}".encode("utf-8")
                ).hexdigest()[:24]
                tool_calls.append(
                    {
                        "id": f"taskrun_call_{digest}",
                        "name": str(action.get("action") or ""),
                        "arguments": "{}",
                    }
                )
        completion = LLMCompletion(
            content="",
            tool_calls=tool_calls,
            api="local_task_run_resume",
            response_id=call_id,
            request_id=None,
            usage={},
        )
        labels = DataLabels.from_dict(dict(manifest["data_labels"]))
        flow_token = self._data_flow.push(DataFlowContext(labels=labels))
        try:
            self._audit.record(
                actor=pid,
                action="llm.task_run_validated_action_recovered",
                target=f"llm_call:{call_id}",
                decision={
                    "call_id": call_id,
                    "action_count": len(actions),
                    "parallel_tool_calls": parallel_tool_calls,
                    "provider_called": False,
                },
            )
            return await self._dispatch_completed_llm_action(
                pid=pid,
                completion=completion,
                actions=actions,
                parallel_tool_calls=parallel_tool_calls,
                host_auto_wait=host_auto_wait,
                call_id=call_id,
            )
        finally:
            self._data_flow.reset(flow_token)

    @staticmethod
    def _include_task_run_labels(
        flow_context: DataFlowContext,
        task_context: Mapping[str, Any],
    ) -> DataFlowContext:
        try:
            labels = DataLabels.from_dict(dict(task_context["data_labels"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("durable TaskRun data labels are invalid") from exc
        return DataFlowContext.aggregate(
            (flow_context, DataFlowContext(labels=labels))
        )

    @staticmethod
    def _task_run_messages(
        *,
        system_message: Mapping[str, Any],
        task_context: Mapping[str, Any],
        current_user_message: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            dict(system_message),
            {
                "role": "user",
                "content": task_run_contract_message(task_context),
            },
            *[
                dict(message)
                for message in task_context["transcript_messages"]
            ],
        ]
        if current_user_message is not None:
            messages.append(dict(current_user_message))
        return messages

    def _image_only_transcript_anchor(
        self,
        image: Any,
        process: Any,
    ) -> dict[str, str]:
        system_prompt = str(getattr(image, "system_prompt", "") or "")
        return {
            "image_id": str(image.image_id),
            "goal_oid": str(process.goal_oid or ""),
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            # Checkpoint restore deliberately advances this durable generation
            # without deleting append-only LLM evidence. Binding it into the
            # anchor prevents post-checkpoint transcript rows from leaking
            # into the restored process timeline, while an ordinary Runtime
            # reopen keeps the same generation and resumes normally.
            "llm_context_generation": self._processes.get_llm_context_generation(
                process.pid
            ),
        }

    @staticmethod
    def _image_only_request_purpose(anchor: Mapping[str, str]) -> str:
        fingerprint = hashlib.sha256(
            dumps(to_jsonable(dict(anchor))).encode("utf-8")
        ).hexdigest()[:24]
        return f"{_IMAGE_ONLY_REQUEST_PURPOSE_PREFIX}:{fingerprint}"

    @staticmethod
    def _image_only_empty_head_validation_purpose(transcript_call_id: str) -> str:
        return (
            f"{_IMAGE_ONLY_EMPTY_HEAD_VALIDATION_PURPOSE_PREFIX}:"
            f"{transcript_call_id}"
        )

    @classmethod
    def _frozen_image_only_anchor(
        cls,
        state: _LLMCallState,
        *,
        image: Any,
    ) -> dict[str, str]:
        frozen = state.request_options.get(_IMAGE_ONLY_FROZEN_ANCHOR_KEY)
        if not isinstance(frozen, dict):
            raise ValidationError("image_only request anchor was not frozen")
        if frozen.get("schema_version") != _IMAGE_ONLY_FROZEN_ANCHOR_SCHEMA_VERSION:
            raise ValidationError("image_only frozen request anchor has an unsupported schema")
        expected_static = {
            "image_id": str(image.image_id),
            "goal_oid": str(state.process.goal_oid or ""),
            "system_prompt_sha256": hashlib.sha256(
                str(getattr(image, "system_prompt", "") or "").encode("utf-8")
            ).hexdigest(),
        }
        anchor: dict[str, str] = {}
        for key, expected in expected_static.items():
            value = frozen.get(key)
            if value != expected:
                raise ValidationError(f"image_only frozen request anchor changed {key}")
            anchor[key] = expected
        generation = frozen.get("llm_context_generation")
        if not isinstance(generation, str) or not generation:
            raise ValidationError(
                "image_only frozen request anchor is missing llm_context_generation"
            )
        anchor["llm_context_generation"] = generation
        if frozen.get("purpose") != cls._image_only_request_purpose(anchor):
            raise ValidationError("image_only frozen request anchor purpose changed")
        return anchor

    @classmethod
    def _image_only_marker_for_anchor(
        cls,
        call: LLMCallRecord,
        anchor: Mapping[str, str],
    ) -> dict[str, Any] | None:
        marker = call.request_options.get(_IMAGE_ONLY_TRANSCRIPT_KEY)
        if not isinstance(marker, dict):
            return None
        if marker.get("schema_version") != _IMAGE_ONLY_TRANSCRIPT_SCHEMA_VERSION:
            raise ValidationError("image_only transcript head has an unsupported schema")
        for key, expected in anchor.items():
            value = marker.get(key)
            if not isinstance(value, str):
                raise ValidationError(f"image_only transcript head is missing {key}")
            if value != expected:
                return None
        return marker

    @classmethod
    def _image_only_request_marker_for_anchor(
        cls,
        call: LLMCallRecord,
        anchor: Mapping[str, str],
    ) -> dict[str, Any] | None:
        marker = call.request_options.get(_IMAGE_ONLY_REQUEST_KEY)
        if not isinstance(marker, dict):
            return None
        if marker.get("schema_version") != _IMAGE_ONLY_REQUEST_SCHEMA_VERSION:
            raise ValidationError("image_only request anchor has an unsupported schema")
        for key, expected in anchor.items():
            value = marker.get(key)
            if not isinstance(value, str):
                raise ValidationError(f"image_only request anchor is missing {key}")
            if value != expected:
                return None
        if marker.get("purpose") != cls._image_only_request_purpose(anchor):
            raise ValidationError("image_only request anchor purpose changed")
        return marker

    def _image_only_goal_content(
        self,
        *,
        process: Any,
    ) -> str:
        goal_oid = str(process.goal_oid or "")
        if not goal_oid:
            raise ValidationError(
                "image_only cannot start a transcript without a process goal"
            )
        goal = self._objects.get_object(goal_oid)
        if goal is None or goal.type != ObjectType.GOAL:
            raise ValidationError("image_only process goal payload is unavailable")
        payload = goal.payload
        if (
            isinstance(payload, dict)
            and set(payload) == {"text"}
            and isinstance(payload.get("text"), str)
        ):
            return payload["text"]
        return dumps(to_jsonable(payload))

    def _image_only_source_context(
        self,
        *,
        pid: str,
        oid: str | None,
        materialized_oids: set[str],
        retained_labels: Mapping[str, Any] | None = None,
    ) -> DataFlowContext:
        contexts: list[DataFlowContext] = []
        if retained_labels is not None:
            try:
                contexts.append(
                    DataFlowContext(labels=DataLabels.from_dict(retained_labels))
                )
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "image_only transcript contains invalid data-flow labels"
                ) from exc
        if oid and oid in materialized_oids:
            contexts.append(
                self._data_flow.context_from_source_oids(
                    pid,
                    [oid],
                    include_current=False,
                )
            )
        elif oid:
            metadata = self._objects.get_persisted_object_metadata(oid)
            if metadata is not None:
                contexts.append(
                    DataFlowContext(
                        labels=DataLabels.from_object_metadata(metadata),
                    )
                )
        return DataFlowContext.aggregate(contexts)

    @staticmethod
    def _decode_image_only_tool_output(value: Any) -> dict[str, Any]:
        if not isinstance(value, str):
            raise ValidationError("image_only transcript tool output is unavailable")
        try:
            envelope = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError("image_only transcript tool output is corrupt") from exc
        if (
            not isinstance(envelope, dict)
            or envelope.get("schema_version")
            != _IMAGE_ONLY_TOOL_OUTPUT_SCHEMA_VERSION
            or not isinstance(envelope.get("content"), str)
            or type(envelope.get("synthetic")) is not bool
            or type(envelope.get("ok")) is not bool
            or not isinstance(envelope.get("labels"), dict)
        ):
            raise ValidationError("image_only transcript tool output has an invalid shape")
        result_oid = envelope.get("result_oid")
        if result_oid is not None and (not isinstance(result_oid, str) or not result_oid):
            raise ValidationError("image_only transcript tool output has an invalid result_oid")
        return envelope

    def _latest_image_only_transcript(
        self,
        *,
        pid: str,
        image: Any,
        anchor: Mapping[str, str],
    ) -> tuple[LLMCallRecord | None, dict[str, Any] | None]:
        latest = self._processes.get_latest_successful_llm_call(
            pid=pid,
            purpose="action_selection",
        )
        if latest is None:
            return None, None
        marker = self._image_only_marker_for_anchor(latest, anchor)
        if marker is not None:
            self._assert_image_only_transcript_complete(
                pid=pid,
                image=image,
                latest=latest,
                marker=marker,
            )
            return latest, marker
        if latest.image_id != image.image_id:
            return latest, None
        raw_marker = latest.request_options.get(_IMAGE_ONLY_TRANSCRIPT_KEY)
        if raw_marker is None:
            raise ValidationError(
                "image_only legacy prompt history cannot be resumed transparently"
            )
        if not isinstance(raw_marker, dict):
            raise ValidationError("image_only transcript head marker is invalid")
        # A valid head for another goal, prompt, or restore generation is an
        # anchor boundary. Do not resurrect an older transcript merely because
        # its other fields happen to match the current process again.
        return latest, None

    def _latest_image_only_request(
        self,
        *,
        pid: str,
        image: Any,
        anchor: Mapping[str, str],
    ) -> tuple[LLMCallRecord | None, dict[str, Any] | None]:
        request = self._processes.get_latest_llm_call(
            pid=pid,
            purpose=self._image_only_request_purpose(anchor),
        )
        if request is None or request.status != "error":
            return None, None
        raw_marker = request.request_options.get(_IMAGE_ONLY_REQUEST_KEY)
        if raw_marker is None:
            return None, None
        marker = self._image_only_request_marker_for_anchor(request, anchor)
        if marker is not None:
            return request, marker
        if request.image_id == image.image_id and not isinstance(raw_marker, dict):
            raise ValidationError("image_only request anchor marker is invalid")
        return None, None

    def _assert_image_only_transcript_complete(
        self,
        *,
        pid: str,
        image: Any,
        latest: LLMCallRecord,
        marker: Mapping[str, Any],
    ) -> None:
        messages = self._image_only_canonical_messages(image, latest, marker)
        manifest, assistant_calls, seen_call_ids = self._image_only_replay_manifest(
            latest,
            marker,
        )
        self._append_image_only_assistant_message(messages, latest, assistant_calls)
        if not manifest:
            validation = self._processes.get_latest_llm_call(
                pid=pid,
                purpose=self._image_only_empty_head_validation_purpose(
                    latest.call_id
                ),
            )
            validation_marker = (
                validation.request_options.get(
                    _IMAGE_ONLY_EMPTY_HEAD_VALIDATION_KEY
                )
                if validation is not None
                else None
            )
            if (
                validation is None
                or validation.status != "ok"
                or not isinstance(validation_marker, dict)
                or validation_marker.get("schema_version")
                != _IMAGE_ONLY_EMPTY_HEAD_VALIDATION_SCHEMA_VERSION
                or validation_marker.get("transcript_call_id") != latest.call_id
            ):
                raise ValidationError(
                    "image_only empty transcript head was not action-validated"
                )
        output_key = marker.get("output_key")
        if not isinstance(output_key, str) or not output_key:
            raise ValidationError("image_only transcript head is missing its output key")
        output_rows = self._processes.list_llm_tool_outputs(
            pid=pid,
            response_id=output_key,
        )
        outputs_by_call_id = {
            str(row.get("call_id") or ""): row for row in output_rows
        }
        if set(outputs_by_call_id) != seen_call_ids:
            raise ValidationError("image_only transcript tool outputs are incomplete")
        for item in manifest:
            row = outputs_by_call_id[item["call_id"]]
            if str(row.get("tool_name") or "") != item["name"]:
                raise ValidationError("image_only transcript tool output is not paired")
            self._decode_image_only_tool_output(row.get("output_text"))

    def _new_image_only_transcript(
        self,
        *,
        pid: str,
        image: Any,
        process: Any,
        context: MaterializedContext,
    ) -> tuple[list[dict[str, Any]], DataFlowContext, list[str]]:
        goal_oid = str(process.goal_oid or "") or None
        goal_content = self._image_only_goal_content(
            process=process,
        )
        goal_flow = self._image_only_source_context(
            pid=pid,
            oid=goal_oid,
            materialized_oids=set(context.object_refs),
        )
        return (
            [
                {"role": "system", "content": build_system_prompt(image)},
                {"role": "user", "content": goal_content},
            ],
            goal_flow,
            [goal_oid] if goal_oid else [],
        )

    def _retry_image_only_request(
        self,
        *,
        pid: str,
        image: Any,
        process: Any,
        context: MaterializedContext,
        request: LLMCallRecord,
        marker: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], DataFlowContext, list[str]]:
        if request.status != "error" or not isinstance(request.messages, list):
            raise ValidationError("image_only retry request is incomplete")
        canonical_count = marker.get("canonical_message_count")
        # An error request may contain a snapshot assembled from an earlier
        # successful transcript. It is evidence of an attempted request, not
        # a completed transcript head, so only the original two-message goal
        # request is safe to recover from this row.
        if canonical_count != 2 or len(request.messages) < 2:
            raise ValidationError(
                "image_only complete transcript head is unavailable for retry"
            )
        selected = request.messages[:2]
        if any(not isinstance(message, dict) for message in selected):
            raise ValidationError("image_only retry request contains an invalid message")
        messages = [dict(message) for message in selected]
        if messages[0] != {"role": "system", "content": build_system_prompt(image)}:
            raise ValidationError("image_only retry request system prompt changed unexpectedly")
        if messages[1].get("role") != "user" or not isinstance(
            messages[1].get("content"), str
        ):
            raise ValidationError("image_only retry request goal message is invalid")
        goal_oid = str(process.goal_oid or "") or None
        flow_contexts, input_oids = self._image_only_replay_sources(
            pid=pid,
            goal_oid=goal_oid,
            materialized_oids=set(context.object_refs),
            marker=marker,
        )
        return (
            messages,
            DataFlowContext.aggregate(flow_contexts),
            list(dict.fromkeys(input_oids)),
        )

    @staticmethod
    def _image_only_replay_tool_call(
        *,
        ordinal: int,
        item: Any,
        tool_call: Any,
        seen_call_ids: set[str],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        if not isinstance(item, dict) or not isinstance(tool_call, dict):
            raise ValidationError("image_only transcript contains an invalid tool call")
        call_id = str(item.get("call_id") or "").strip()
        name = str(item.get("name") or "").strip()
        stored_call_id = str(
            tool_call.get("call_id") or tool_call.get("id") or ""
        ).strip()
        if item.get("ordinal") != ordinal:
            raise ValidationError("image_only transcript tool-call manifest is invalid")
        if not call_id or call_id in seen_call_ids or not name:
            raise ValidationError("image_only transcript tool-call manifest is invalid")
        if stored_call_id != call_id:
            raise ValidationError("image_only transcript tool-call manifest is invalid")
        if str(tool_call.get("name") or "").strip() != name:
            raise ValidationError("image_only transcript tool-call manifest is invalid")
        seen_call_ids.add(call_id)
        arguments = tool_call.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = dumps(to_jsonable(arguments))
        return (
            {"call_id": call_id, "name": name},
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            },
        )

    def _image_only_replay_manifest(
        self,
        latest: LLMCallRecord,
        marker: Mapping[str, Any],
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]], set[str]]:
        raw_manifest = marker.get("tool_calls")
        stored_calls = latest.tool_calls
        if not isinstance(raw_manifest, list) or not isinstance(stored_calls, list):
            raise ValidationError("image_only transcript tool-call manifest is unavailable")
        if len(raw_manifest) != len(stored_calls):
            raise ValidationError(
                "image_only transcript tool-call manifest disagrees with the response"
            )
        manifest: list[dict[str, str]] = []
        assistant_calls: list[dict[str, Any]] = []
        seen_call_ids: set[str] = set()
        for ordinal, (item, tool_call) in enumerate(zip(raw_manifest, stored_calls)):
            manifest_item, assistant_call = self._image_only_replay_tool_call(
                ordinal=ordinal,
                item=item,
                tool_call=tool_call,
                seen_call_ids=seen_call_ids,
            )
            manifest.append(manifest_item)
            assistant_calls.append(assistant_call)
        return manifest, assistant_calls, seen_call_ids

    @staticmethod
    def _image_only_canonical_messages(
        image: Any,
        latest: LLMCallRecord,
        marker: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if latest.status != "ok" or not isinstance(latest.messages, list):
            raise ValidationError("image_only transcript head is incomplete")
        canonical_count = marker.get("canonical_message_count")
        if type(canonical_count) is not int:
            raise ValidationError("image_only transcript head has an invalid message boundary")
        if canonical_count < 2 or canonical_count > len(latest.messages):
            raise ValidationError("image_only transcript head has an invalid message boundary")
        selected = latest.messages[:canonical_count]
        if any(not isinstance(message, dict) for message in selected):
            raise ValidationError("image_only transcript head contains an invalid message")
        messages = [dict(message) for message in selected]
        expected_system = {"role": "system", "content": build_system_prompt(image)}
        if messages[0] != expected_system:
            raise ValidationError("image_only transcript system prompt changed unexpectedly")
        goal_message = messages[1]
        if goal_message.get("role") != "user":
            raise ValidationError("image_only transcript goal message is invalid")
        if not isinstance(goal_message.get("content"), str):
            raise ValidationError("image_only transcript goal message is invalid")
        return messages

    @staticmethod
    def _append_image_only_assistant_message(
        messages: list[dict[str, Any]],
        latest: LLMCallRecord,
        assistant_calls: list[dict[str, Any]],
    ) -> None:
        response_content = latest.response_content
        if not isinstance(response_content, str):
            raise ValidationError("image_only transcript assistant content is unavailable")
        if assistant_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": response_content,
                    "tool_calls": assistant_calls,
                }
            )
        elif response_content:
            messages.append({"role": "assistant", "content": response_content})

    def _image_only_replay_sources(
        self,
        *,
        pid: str,
        goal_oid: str | None,
        materialized_oids: set[str],
        marker: Mapping[str, Any],
    ) -> tuple[list[DataFlowContext], list[str]]:
        labels = marker.get("labels")
        if not isinstance(labels, dict):
            raise ValidationError("image_only transcript head is missing data-flow labels")
        flow_contexts = [
            self._image_only_source_context(
                pid=pid,
                oid=goal_oid,
                materialized_oids=materialized_oids,
                retained_labels=labels,
            )
        ]
        raw_input_oids = marker.get("input_oids")
        if not isinstance(raw_input_oids, list):
            raise ValidationError("image_only transcript head has invalid input Objects")
        if any(not isinstance(oid, str) or not oid for oid in raw_input_oids):
            raise ValidationError("image_only transcript head has invalid input Objects")
        input_oids = [*raw_input_oids, *([goal_oid] if goal_oid else [])]
        return flow_contexts, list(dict.fromkeys(input_oids))

    def _append_image_only_tool_outputs(
        self,
        *,
        pid: str,
        messages: list[dict[str, Any]],
        manifest: list[dict[str, str]],
        seen_call_ids: set[str],
        marker: Mapping[str, Any],
        materialized_oids: set[str],
        flow_contexts: list[DataFlowContext],
        input_oids: list[str],
    ) -> None:
        output_key = marker.get("output_key")
        if not isinstance(output_key, str) or not output_key:
            raise ValidationError("image_only transcript head is missing its output key")
        output_rows = self._processes.list_llm_tool_outputs(
            pid=pid,
            response_id=output_key,
        )
        outputs_by_call_id = {
            str(row.get("call_id") or ""): row for row in output_rows
        }
        if set(outputs_by_call_id) != seen_call_ids:
            raise ValidationError("image_only transcript tool outputs are incomplete")
        for item in manifest:
            envelope = self._decode_image_only_tool_output(
                outputs_by_call_id[item["call_id"]].get("output_text")
            )
            result_oid = envelope.get("result_oid")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "name": item["name"],
                    "content": envelope["content"],
                }
            )
            flow_contexts.append(
                self._image_only_source_context(
                    pid=pid,
                    oid=result_oid,
                    materialized_oids=materialized_oids,
                    retained_labels=envelope["labels"],
                )
            )
            if result_oid:
                input_oids.append(result_oid)

    def _replay_image_only_transcript(
        self,
        *,
        pid: str,
        image: Any,
        process: Any,
        context: MaterializedContext,
        latest: LLMCallRecord,
        marker: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], DataFlowContext, list[str]]:
        messages = self._image_only_canonical_messages(image, latest, marker)
        manifest, assistant_calls, seen_call_ids = self._image_only_replay_manifest(
            latest,
            marker,
        )
        self._append_image_only_assistant_message(messages, latest, assistant_calls)
        goal_oid = str(process.goal_oid or "") or None
        materialized_oids = set(context.object_refs)
        flow_contexts, input_oids = self._image_only_replay_sources(
            pid=pid,
            goal_oid=goal_oid,
            materialized_oids=materialized_oids,
            marker=marker,
        )
        self._append_image_only_tool_outputs(
            pid=pid,
            messages=messages,
            manifest=manifest,
            seen_call_ids=seen_call_ids,
            marker=marker,
            materialized_oids=materialized_oids,
            flow_contexts=flow_contexts,
            input_oids=input_oids,
        )
        return (
            messages,
            DataFlowContext.aggregate(flow_contexts),
            list(dict.fromkeys(input_oids)),
        )

    def _image_only_messages_and_flow(
        self,
        *,
        pid: str,
        image: Any,
        process: Any,
        context: MaterializedContext,
        anchor: Mapping[str, str],
    ) -> tuple[list[dict[str, Any]], DataFlowContext, list[str]]:
        latest, marker = self._latest_image_only_transcript(
            pid=pid,
            image=image,
            anchor=anchor,
        )
        if latest is None or marker is None:
            request, request_marker = self._latest_image_only_request(
                pid=pid,
                image=image,
                anchor=anchor,
            )
            if request is not None and request_marker is not None:
                return self._retry_image_only_request(
                    pid=pid,
                    image=image,
                    process=process,
                    context=context,
                    request=request,
                    marker=request_marker,
                )
            return self._new_image_only_transcript(
                pid=pid,
                image=image,
                process=process,
                context=context,
            )
        return self._replay_image_only_transcript(
            pid=pid,
            image=image,
            process=process,
            context=context,
            latest=latest,
            marker=marker,
        )

    def _build_model_messages(
        self,
        *,
        pid: str,
        image: Any,
        process: Any,
        context: Any,
        events: list[Any],
        capabilities: list[Any],
        tools: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        available_skills: list[dict[str, Any]] | None = None,
        original_goal_context: str | None = None,
    ) -> list[dict[str, str]]:
        try:
            fallback_json_actions = (
                self._llms.fallback_json_actions_for_process(pid)
            )
        except ValidationError:
            # Provider resolution owns the durable fail-closed error path for
            # an unknown profile; prompt construction must not escape it.
            fallback_json_actions = False
        return [
            {"role": "system", "content": build_system_prompt(image)},
            {
                "role": "user",
                "content": build_user_prompt(
                    process=process,
                    context=context,
                    events=events,
                    capabilities=capabilities,
                    tools=tools,
                    skills=skills,
                    available_skills=available_skills or [],
                    prompt_mode=image.prompt_mode,
                    requestable_capabilities=(
                        self._requestable_capabilities_for_prompt(pid)
                    ),
                    original_goal_context=original_goal_context,
                    fallback_json_actions=fallback_json_actions,
                ),
            },
        ]

    def _assemble_llm_request(
        self,
        *,
        pid: str,
        image: Any,
        process: Any,
        context: MaterializedContext,
        events: list[Any],
        label_events: list[Any],
        capabilities: list[Any],
        tools: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        available_skills: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        DataFlowContext,
        str | None,
        dict[str, str] | None,
        dict[str, Any] | None,
    ]:
        task_context = self._task_run_prompt_context(pid)
        requirement_binding = self._task_run_requirement_binding(pid, task_context)
        if image.prompt_mode == PROMPT_MODE_IMAGE_ONLY:
            return self._assemble_image_only_llm_request(
                pid=pid,
                image=image,
                process=process,
                context=context,
                task_context=task_context,
                requirement_binding=requirement_binding,
            )
        # The live goal is already part of the authoritative materialized
        # context in the ordinary case.  Replaying the retained copy as well
        # duplicates a potentially large, volatile block on every later
        # quantum and destroys the otherwise stable prompt prefix.  Retained
        # recovery is only for the exceptional case where compaction, a view
        # change, or a constrained materialization omitted the goal Object.
        goal_is_materialized = bool(
            process.goal_oid is not None
            and process.goal_oid in context.object_refs
        )
        original_goal_context = (
            None
            if goal_is_materialized or task_context is not None
            else self._retained_original_goal_context(
                process=process,
                image=image,
            )
        )
        if (
            task_context is None
            and process.goal_oid is not None
            and not goal_is_materialized
            and original_goal_context is None
            # An explicit MemoryView may deliberately omit the goal while the
            # exact live Object remains available to the Runtime.  That is a
            # visibility choice, not lost first-quantum state.  Fail closed
            # only when the payload itself is unavailable and there is no
            # retained prompt evidence to recover from.
            and self._objects.get_object(process.goal_oid) is None
            and self._processes.get_latest_llm_call(
                pid=pid,
                purpose="action_selection",
            )
            is None
        ):
            raise ValidationError(
                "process goal payload is unavailable before the first LLM quantum"
            )
        flow_context = self._data_flow.context_from_materialization(pid, context)
        if original_goal_context is not None:
            flow_context = self._include_retained_goal_labels(
                flow_context,
                process.goal_oid,
            )
        if task_context is not None:
            flow_context = self._include_task_run_labels(
                flow_context,
                task_context,
            )
        event_projection = project_prompt_events(
            events,
            context_object_name=self.context_memory.object_name(pid),
            payload_max_chars=self.config.llm_context.prompt_event_payload_max_chars,
        )
        # Event projection is prompt content independently of the configured
        # context-memory policy. Merge its trusted labels directly into the
        # request flow context so source-only mode cannot omit classified
        # off-view events from provider clearance.
        flow_context = self.context_memory.include_event_labels(
            flow_context,
            label_events,
        )
        messages = self._build_model_messages(
            pid=pid,
            image=image,
            process=process,
            context=context,
            events=event_projection.model_records,
            capabilities=capabilities,
            tools=tools,
            skills=skills,
            available_skills=available_skills,
            original_goal_context=original_goal_context,
        )
        if task_context is not None:
            messages = self._task_run_messages(
                system_message=messages[0],
                task_context=task_context,
                current_user_message=messages[1],
            )
        input_refs = list(context.object_refs)
        if (
            original_goal_context is not None
            and process.goal_oid is not None
            and process.goal_oid not in input_refs
        ):
            input_refs.append(process.goal_oid)
        self._audit.record(
            actor=pid,
            action="llm.request",
            target=f"image:{image.image_id}",
            input_refs=input_refs,
            decision={
                "messages": len(messages),
                "policy": image.context_policy,
                "event_projection": event_projection.summary,
                "task_run": (
                    {
                        "run_id": task_context["run_id"],
                        "context_generation": task_context["context_generation"],
                        "transcript_message_count": len(
                            task_context["transcript_messages"]
                        ),
                    }
                    if task_context is not None
                    else None
                ),
            },
        )
        return (
            messages,
            flow_context,
            event_projection.represented_through_event_id,
            None,
            requirement_binding,
        )

    def _assemble_image_only_llm_request(
        self,
        *,
        pid: str,
        image: Any,
        process: Any,
        context: MaterializedContext,
        task_context: Mapping[str, Any] | None,
        requirement_binding: dict[str, Any] | None,
    ) -> tuple[
        list[dict[str, Any]],
        DataFlowContext,
        None,
        dict[str, str],
        dict[str, Any] | None,
    ]:
        if not self.config.llm.persist_full_io:
            raise _ImageOnlyFullIORequired(
                "image_only requires llm.persist_full_io=true for durable transcript replay"
            )
        anchor = self._image_only_transcript_anchor(image, process)
        task_run_projection: dict[str, Any] | None = None
        if task_context is None:
            messages, flow_context, input_refs = self._image_only_messages_and_flow(
                pid=pid,
                image=image,
                process=process,
                context=context,
                anchor=anchor,
            )
        else:
            messages = self._task_run_messages(
                system_message={
                    "role": "system",
                    "content": build_system_prompt(image),
                },
                task_context=task_context,
            )
            flow_context = self._include_task_run_labels(
                self._data_flow.context_from_materialization(pid, context),
                task_context,
            )
            input_refs = list(context.object_refs)
            task_run_projection = {
                "run_id": task_context["run_id"],
                "context_generation": task_context["context_generation"],
                "transcript_message_count": len(
                    task_context["transcript_messages"]
                ),
            }
        decision: dict[str, Any] = {
            "messages": len(messages),
            "policy": image.context_policy,
            "prompt_mode": PROMPT_MODE_IMAGE_ONLY,
            "event_projection": {"model_visible": False},
        }
        if task_run_projection is not None:
            decision["task_run"] = task_run_projection
        self._audit.record(
            actor=pid,
            action="llm.request",
            target=f"image:{image.image_id}",
            input_refs=input_refs,
            decision=decision,
        )
        return messages, flow_context, None, anchor, requirement_binding

    def run_once(self, pid: str) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun_once(pid))
        raise RuntimeError("Cannot call run_once() inside a running event loop. Use await arun_once(...).")

    async def arun_once(self, pid: str) -> dict[str, Any]:
        process = self._process.get(pid)
        if process.status not in {ProcessStatus.RUNNING, ProcessStatus.RUNNABLE}:
            return await self._arun_once_impl(pid)
        pending = self.pending.get(pid)
        operation_id = (
            str(pending.get("llm_operation_id") or "")
            if pending is not None
            and pending.get("status") in {"pending", "resuming"}
            else ""
        )
        with self._operations.scope(
            kind="llm_request",
            name="llm.action_selection",
            actor=pid,
            pid=pid,
            expected_roles=["context", "invocation", "audit"],
            operation_id=operation_id or None,
            auto_finish=False,
        ) as operation:
            result = await self._arun_once_impl(pid)
            if any(result.get(key) for key in ("waiting_human", "waiting_event", "waiting_message", "pending_action_resuming")):
                self._operations.wait(operation_id=operation.operation_id)
            elif result.get("resource_limit_exceeded"):
                self._operations.finish("denied", operation_id=operation.operation_id)
            elif result.get("ok"):
                descendants = self._evidence.list_operations(
                    root_operation_id=operation.root_operation_id
                )
                outcome = (
                    "unknown"
                    if any(
                        candidate.operation_id != operation.operation_id
                        and candidate.outcome.value == "unknown"
                        for candidate in descendants
                    )
                    else "succeeded"
                )
                self._operations.finish(outcome, operation_id=operation.operation_id)
            elif result.get("skipped"):
                self._operations.finish("interrupted", operation_id=operation.operation_id)
            else:
                self._operations.finish("failed", operation_id=operation.operation_id)
            return result

    async def _arun_once_impl(self, pid: str) -> dict[str, Any]:
        process = self._process.get(pid)
        if process.status not in {ProcessStatus.RUNNING, ProcessStatus.RUNNABLE}:
            return {"ok": False, "skipped": True, "status": process.status.value}
        durable_pending = self._synchronize_pending_action(pid)
        pending_result = await self._resume_pending_quantum_action(pid)
        if pending_result is not None:
            return pending_result
        if durable_pending is not None and durable_pending.get("status") == "resuming":
            return {
                "ok": False,
                "pending_action_resuming": True,
                "wait_type": durable_pending.get("wait_type"),
            }
        pending_task_run_action, task_run_deferred = (
            self._pending_task_run_action_boundary(pid)
        )
        if task_run_deferred:
            return {
                "ok": False,
                "skipped": True,
                "task_run_control_deferred": True,
            }
        if pending_task_run_action is not None:
            return await self._dispatch_pending_task_run_validated_action(
                pid,
                pending_task_run_action,
            )
        image = self._images.get(process.image_id)
        if image is None:
            error = f"agent image not found for process {pid}: {process.image_id}"
            self._process.exit(pid, failed=True, message=error)
            self._audit.record(
                actor=pid,
                action="llm.image_missing",
                target=f"image:{process.image_id}",
                decision={"error": error},
            )
            return {"ok": False, "error": error}
        process = self._ensure_process_memory_view(pid, process)

        self._notify_interrupt_messages(pid)
        source_view = self.context_memory.view_without_context(pid, process.memory_view)
        source_context = self._memory.materialize_context(
            pid,
            source_view,
            policy=image.context_policy,
            budget_tokens=process.resource_budget.max_context_materialization_tokens,
            charge_resources=False,
        )
        label_events = self._events.list(
            target=pid,
            limit=self.config.llm_context.recent_event_limit,
            after_event_id=process.event_cursor,
        )
        events = [
            replace(
                event,
                source=self._tools.redact_model_context(pid, event.source),
                payload=self._tools.redact_model_context(pid, event.payload),
                correlation_id=self._tools.redact_model_context(pid, event.correlation_id),
                causality=self._tools.redact_model_context(pid, event.causality),
            )
            for event in label_events
        ]
        capabilities = self._capabilities.capabilities_for(pid)
        # The prompt-visible tool list must match the process tool table. The
        # broker still owns the real execute check, but showing extra tools
        # teaches the model to choose actions the process cannot call.
        tools = self._tools.model_visible_tools(pid)
        prompt_process = replace(
            process,
            tool_table=self._tools.model_tool_table(pid),
            loaded_skills=self._tools.model_loaded_skills(pid),
        )
        skills = self._skills.prompt_context(pid)
        # Skill metadata is discovered through the same on-demand tool path as
        # every other Skill.  Only activated bodies belong in the prompt.
        available_skills: list[dict[str, Any]] = []
        prepared_context = await self._prepare_llm_context(
            pid=pid,
            image=image,
            process=prompt_process,
            source_context=source_context,
            events=events,
            label_events=label_events,
            capabilities=capabilities,
            tools=tools,
        )
        if isinstance(prepared_context, dict):
            return prepared_context
        context = prepared_context
        try:
            (
                messages,
                flow_context,
                represented_through_event_id,
                image_only_anchor,
                task_run_requirement_binding,
            ) = (
                self._assemble_llm_request(
                    pid=pid,
                    image=image,
                    process=prompt_process,
                    context=context,
                    events=events,
                    label_events=label_events,
                    capabilities=capabilities,
                    tools=tools,
                    skills=skills,
                    available_skills=available_skills,
                )
            )
        except Exception as exc:
            if image.prompt_mode != PROMPT_MODE_IMAGE_ONLY:
                raise
            return self._fail_llm_quantum(pid, exc)
        flow_token = self._data_flow.push(flow_context)
        try:
            openai_tools = self._tools.openai_tool_schemas(pid)
            response_scope_fingerprint = (
                self._current_responses_state_scope_fingerprint(pid)
            )
            (
                completion,
                actions,
                parallel_tool_calls,
                host_auto_wait,
                call_id,
            ) = await self._complete_valid_action_with_control_boundary(
                pid,
                messages,
                openai_tools,
                response_scope_fingerprint=response_scope_fingerprint,
                image_only_anchor=image_only_anchor,
                task_run_requirement_binding=task_run_requirement_binding,
            )
            # Cursor advancement is an acknowledgement that the projected
            # batch reached a successfully recorded provider completion.  A
            # precheck, release, or provider failure leaves the cursor in place
            # so the next quantum cannot silently lose unseen events.
            if represented_through_event_id is not None:
                self._advance_event_cursor(pid, represented_through_event_id)
            return await self._dispatch_completed_llm_action(
                pid=pid,
                completion=completion,
                actions=actions,
                parallel_tool_calls=parallel_tool_calls,
                host_auto_wait=host_auto_wait,
                call_id=call_id,
            )
        except _ContextManagementHandled as handled:
            return handled.result
        except _LLMReleaseApprovalRequired as exc:
            return self._wait_for_llm_release(pid, exc)
        except HumanApprovalRequired as exc:
            self._audit.record(
                actor=pid,
                action="llm.action_waiting_human",
                target=f"human_request:{exc.request_id}",
                decision={"request_id": exc.request_id, "message": str(exc)},
            )
            return {"ok": False, "waiting_human": True, "request_id": exc.request_id}
        except ProcessWaitRequired as exc:
            self._audit.record(
                actor=pid,
                action="llm.action_waiting_child",
                target=f"process:{exc.child_pid}",
                decision={"child_pid": exc.child_pid, "message": str(exc)},
            )
            return {"ok": False, "waiting_event": True, "child_pid": exc.child_pid}
        except ProcessMessageWaitRequired as exc:
            self._audit.record(
                actor=pid,
                action="llm.action_waiting_message",
                target=f"process:{pid}",
                decision={"recipient_pid": exc.recipient_pid, "filters": exc.filters, "message": str(exc)},
            )
            return {"ok": False, "waiting_message": True, "filters": exc.filters}
        except ResourceLimitExceeded as exc:
            self._resources.kill_if_exceeded(pid, reason=str(exc))
            self._audit.record(
                actor=pid,
                action="llm.resource_limit_exceeded",
                target=f"process:{pid}",
                decision={"error": str(exc)},
            )
            return {"ok": False, "resource_limit_exceeded": True, "error": str(exc)}
        except Exception as exc:
            return self._fail_llm_quantum(pid, exc)
        finally:
            self._data_flow.reset(flow_token)

    async def _resume_pending_quantum_action(
        self,
        pid: str,
    ) -> dict[str, Any] | None:
        if self.pending.has_memory(pid, "llm_release"):
            return await self._resume_pending_action_fail_closed(
                pid,
                self._resume_pending_llm_release_action,
            )
        if self.pending.has_memory(pid, "human"):
            return await self._resume_pending_action_fail_closed(
                pid,
                self._resume_pending_human_action,
            )
        if self.pending.has_memory(pid, "child"):
            return await self._resume_pending_action_fail_closed(
                pid,
                self._resume_pending_wait_action,
            )
        if self.pending.has_memory(pid, "message"):
            return await self._resume_pending_action_fail_closed(
                pid,
                self._resume_pending_message_action,
            )
        return None

    async def _prepare_llm_context(
        self,
        *,
        pid: str,
        image: Any,
        process: Any,
        source_context: MaterializedContext,
        events: list[Any],
        label_events: list[Any],
        capabilities: list[Any],
        tools: list[dict[str, Any]],
    ) -> MaterializedContext | dict[str, Any]:
        try:
            return self.context_memory.prepare(
                pid=pid,
                image=image,
                process=process,
                source_context=source_context,
                events=events,
                label_events=label_events,
                capabilities=capabilities,
                tools=tools,
            )
        except LLMContextStoragePressure as pressure:
            return await self._handle_context_storage_pressure(
                pid,
                image=image,
                pressure=pressure,
            )
        except ResourceLimitExceeded as exc:
            self._resources.kill_if_exceeded(pid, reason=str(exc))
            self._audit.record(
                actor=pid,
                action="llm.resource_limit_exceeded",
                target=f"process:{pid}",
                decision={"error": str(exc)},
            )
            return {
                "ok": False,
                "resource_limit_exceeded": True,
                "error": str(exc),
            }

    def _ensure_process_memory_view(self, pid: str, process: Any) -> Any:
        if process.memory_view is not None:
            return process
        memory_view = self._memory.create_view(pid, [], mode=ViewMode.READ_ONLY)
        updated_at = utc_now()
        return self._processes.patch_process(
            pid,
            {"memory_view": memory_view, "updated_at": updated_at},
            expected_revision=process.revision,
        )

    def _fail_llm_quantum(self, pid: str, error: Exception) -> dict[str, Any]:
        if isinstance(error, LLMTransientError):
            public_error, internal_error = self._llm_error_artifacts(error)
            durable_error = public_error["message"]
            self._process.pause_for_host_resume(
                pid,
                f"Retryable LLM provider failure: {durable_error}",
            )
            self._audit.record(
                actor=pid,
                action="llm.action_retryable_failure",
                target=f"process:{pid}",
                decision={
                    "error": durable_error,
                    "error_details": public_error,
                    "internal_error": internal_error,
                    "retryable": True,
                    "error_type": type(error).__name__,
                },
                correlation_id=public_error["correlation_id"],
            )
            return {
                "ok": False,
                "error": durable_error,
                "error_details": public_error,
                "retryable": True,
                "paused": True,
            }
        preserve_domain_text = self._preserve_llm_domain_error_text(error)
        public_error: dict[str, str] | None = None
        internal_error: dict[str, Any] | None = None
        if preserve_domain_text:
            durable_error = str(error)
        else:
            public_error, internal_error = self._llm_error_artifacts(error)
            durable_error = public_error["message"]
        reason = (
            "image_only_full_io_required"
            if isinstance(error, _ImageOnlyFullIORequired)
            else None
        )
        self._process.exit(
            pid,
            failed=True,
            message=f"LLM quantum failed: {durable_error}",
        )
        decision: dict[str, Any] = {"error": durable_error}
        if public_error is not None and internal_error is not None:
            decision.update(
                {
                    "error_details": public_error,
                    "internal_error": internal_error,
                }
            )
        if reason is not None:
            decision["reason"] = reason
        self._audit.record(
            actor=pid,
            action="llm.action_failed",
            target=f"process:{pid}",
            decision=decision,
            correlation_id=(
                public_error["correlation_id"]
                if public_error is not None
                else None
            ),
        )
        result: dict[str, Any] = {"ok": False, "error": durable_error}
        if public_error is not None:
            result["error_details"] = public_error
        if reason is not None:
            result["reason"] = reason
        return result

    async def _handle_context_storage_pressure(
        self,
        pid: str,
        *,
        image: Any,
        pressure: LLMContextStoragePressure,
    ) -> dict[str, Any]:
        policy = context_management_policy(image.planner)
        episode_id = new_id("ctxpressure")
        action, pending_context, common = self._storage_context_attempt_inputs(
            pressure,
            policy=policy,
            episode_id=episode_id,
        )
        prior = self.pending.get(pid) or {}
        prior_metadata = self._context_management_pending_metadata(prior)
        if self._storage_context_attempt_already_recorded(
            prior,
            prior_metadata,
            context_generation=pressure.context_generation,
        ):
            reason = "storage_compaction_attempt_already_recorded"
            return self._fail_storage_context_attempt(
                pid,
                common={
                    **common,
                    "episode_id": prior_metadata.get("episode_id"),
                },
                reason=reason,
                error=RuntimeError(reason),
                include_error=False,
            )
        marker_failure = self._record_storage_context_attempt(
            pid,
            action=action,
            pending_context=pending_context,
            common=common,
        )
        if marker_failure is not None:
            return marker_failure
        try:
            result = await self._dispatch_auto_context_action(
                pid,
                action=action,
                pending_context=pending_context,
            )
        except _ContextManagementHandled as handled:
            return handled.result
        except Exception as exc:
            return self._fail_storage_context_attempt(
                pid,
                common=common,
                reason=type(exc).__name__,
                error=exc,
            )
        return self._finish_storage_context_attempt(
            pid,
            pressure=pressure,
            action=action,
            pending_context=pending_context,
            common=common,
            result=result,
            episode_id=episode_id,
        )

    def _storage_context_attempt_inputs(
        self,
        pressure: LLMContextStoragePressure,
        *,
        policy: ContextManagementPolicy,
        episode_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        assessment = pressure.to_dict()
        action = policy.tool_action()
        if policy.tool_name == "compact_process_context":
            action["force"] = True
            action.setdefault(
                "max_chunks",
                self.config.llm_context.storage_compaction_max_chunks,
            )
            action.setdefault(
                "preserve_recent_entries",
                self.config.llm_context.storage_compaction_preserve_recent_entries,
            )
        common = {
            **assessment,
            "episode_id": episode_id,
            "policy_fingerprint": policy.fingerprint,
            "mode": policy.mode,
            "tool_name": policy.tool_name,
        }
        pending_context = {
            "kind": "context_management_auto",
            "schema_version": 1,
            "source": "runtime_context_management",
            "episode_id": episode_id,
            "policy_fingerprint": policy.fingerprint,
            "mode": policy.mode,
            "tool_name": policy.tool_name,
            "context_generation": pressure.context_generation,
            "assessment": assessment,
        }
        return action, pending_context, common

    @staticmethod
    def _storage_context_attempt_already_recorded(
        prior: dict[str, Any],
        prior_metadata: dict[str, Any],
        *,
        context_generation: str,
    ) -> bool:
        assessment = prior_metadata.get("assessment")
        return (
            prior.get("status") == "completed"
            and isinstance(assessment, dict)
            and assessment.get("trigger") == "storage_payload"
            and prior_metadata.get("context_generation") == context_generation
            and prior_metadata.get("outcome") == "attempted"
        )

    def _record_storage_context_attempt(
        self,
        pid: str,
        *,
        action: dict[str, Any],
        pending_context: dict[str, Any],
        common: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            # The read-only check in LLMContextMemory is only a planning
            # predicate. Revalidate and consume the maintenance authority at
            # the first durable attempt boundary. Keeping the finite-use
            # claim, marker, and attempt evidence in one transaction means a
            # pre-dispatch persistence failure rolls all of them back. Once
            # this transaction commits, the attempt owns that use even when
            # provider/tool outcome is later unknown; replay must reconcile
            # the existing attempt instead of restoring authority.
            with self._capabilities.store.transaction():
                self._capabilities.require(
                    pid,
                    LLM_CONTEXT_MAINTENANCE_RESOURCE,
                    CapabilityRight.EXECUTE,
                    used_by=pid,
                    reason="storage context auto-compaction attempt",
                )
                self._persist_completed_context_management_marker(
                    pid,
                    action=action,
                    metadata={**pending_context, "outcome": "attempted"},
                )
                self._audit.record(
                    actor=pid,
                    action="llm.context_pressure_detected",
                    target=f"process:{pid}",
                    decision=common,
                )
                self._audit.record(
                    actor=pid,
                    action="llm.context_pressure_auto_attempted",
                    target=f"process:{pid}",
                    decision=common,
                )
        except CapabilityDenied:
            # Revocation/exhaustion that wins before the attempt transaction
            # is a clean authorization race, not a compaction failure. End
            # this quantum without dispatch; the next materialization sees
            # the current authority state and will not retrigger maintenance.
            self._audit.record(
                actor=pid,
                action="llm.context_pressure_maintenance_not_authorized",
                target=f"process:{pid}",
                decision={**common, "maintenance_authorized": False},
            )
            return {
                "ok": True,
                "context_compacted": False,
                "context_storage_pressure": True,
                "context_maintenance_not_authorized": True,
                "context_generation": common.get("context_generation"),
            }
        except Exception as exc:
            return self._fail_storage_context_attempt(
                pid,
                common=common,
                reason="attempt_marker_persistence_failed",
                error=exc,
                include_error_type=True,
            )
        return None

    def _finish_storage_context_attempt(
        self,
        pid: str,
        *,
        pressure: LLMContextStoragePressure,
        action: dict[str, Any],
        pending_context: dict[str, Any],
        common: dict[str, Any],
        result: dict[str, Any],
        episode_id: str,
    ) -> dict[str, Any]:
        generation_after = self._processes.get_llm_context_generation(pid)
        if self._context_compaction_succeeded(
            result,
            generation_before=pressure.context_generation,
            generation_after=generation_after,
        ):
            self._persist_completed_context_management_marker(
                pid,
                action=action,
                metadata={
                    **pending_context,
                    "context_generation_after": generation_after,
                    "outcome": "compacted",
                },
            )
            self._audit.record(
                actor=pid,
                action="llm.context_pressure_compacted",
                target=f"process:{pid}",
                decision={
                    **common,
                    "context_generation_after": generation_after,
                },
            )
            return {
                "ok": True,
                "context_compacted": True,
                "context_storage_pressure": True,
                "context_pressure_episode_id": episode_id,
                "context_generation": generation_after,
            }

        reason = self._context_management_failure_reason(
            result,
            generation_before=pressure.context_generation,
            generation_after=generation_after,
        )
        return self._fail_storage_context_attempt(
            pid,
            common=common,
            reason=reason,
            error=RuntimeError(
                f"LLM context storage compaction failed: {reason}"
            ),
            extra={
                "result": sanitize_for_observability(result),
                "context_generation_after": generation_after,
            },
            include_error=False,
        )

    def _fail_storage_context_attempt(
        self,
        pid: str,
        *,
        common: dict[str, Any],
        reason: str,
        error: Exception,
        extra: dict[str, Any] | None = None,
        include_error: bool = True,
        include_error_type: bool = False,
    ) -> dict[str, Any]:
        decision = {
            **common,
            "reason": reason,
            **(extra or {}),
        }
        public_error, internal_error = self._llm_error_artifacts(
            error,
            code="llm_context_management_error",
        )
        if include_error_type:
            decision["error_type"] = type(error).__name__
        if include_error:
            decision["error"] = public_error["message"]
            decision["error_details"] = public_error
            decision["internal_error"] = internal_error
        self._audit.record(
            actor=pid,
            action="llm.context_pressure_failed",
            target=f"process:{pid}",
            decision=decision,
            correlation_id=public_error["correlation_id"],
        )
        return self._fail_llm_quantum(pid, error)

    @staticmethod
    def _llm_error_artifacts(
        error: BaseException,
        *,
        code: str = "llm_error",
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Return one correlation-stable, text-free outward failure view.

        Provider and extension exception text may contain prompts, paths, DSNs,
        or credentials.  ``persist_full_io`` governs successful request/response
        retention; it never authorizes exception text to cross a durable or
        model-facing boundary.
        """

        public_error = public_error_envelope(error, code=code)
        internal_error = llm_error_internal_observation(
            error,
            correlation_id=public_error["correlation_id"],
        )
        return public_error, internal_error

    def _durable_llm_error(self, error: BaseException) -> str:
        if self._preserve_llm_domain_error_text(error):
            return str(error)
        return self._llm_error_artifacts(error)[0]["message"]

    @staticmethod
    def _preserve_llm_domain_error_text(error: BaseException) -> bool:
        """Keep established Host/model protocol errors actionable.

        These classes are minted before provider dispatch or from the model's
        normal action payload. Provider invocation failures are normalized to
        ``LLMError`` by ``LLMProviderService`` and never enter this branch.
        """

        return not isinstance(error, LLMError) and isinstance(
            error,
            (
                CapabilityDenied,
                NotFound,
                ResourceLimitExceeded,
                ValidationError,
                ValueError,
            ),
        )

    def _completion_content_preview(self, content: Any) -> str:
        if not self.config.llm.persist_full_io:
            return ""
        return str(content)[: self.config.llm.content_preview_chars]

    def _advance_event_cursor(self, pid: str, event_id: str) -> None:
        """Advance one worker's cursor without racing cross-process accounting.

        A child resource charge also updates each ancestor's process row.  Keep
        the cursor re-read and field-level CAS in the repository lock domain so
        that unrelated hierarchical accounting cannot invalidate the revision
        between those two operations.  ``patch_process`` still enforces the
        ambient execution-token fence before this worker may mutate the row.
        """

        with self._processes.locked():
            current = self._process.get(pid)
            if current.event_cursor == event_id:
                return
            self._processes.patch_process(
                pid,
                {"event_cursor": event_id, "updated_at": utc_now()},
                expected_revision=current.revision,
            )

    def _completed_action_result(
        self,
        pid: str,
        action: dict[str, Any],
        result: dict[str, Any],
        content_preview: str,
        tool_call_count: int,
        resumed_after_human: bool = False,
        resumed_after_message: bool = False,
        action_source: str | None = None,
    ) -> dict[str, Any]:
        decision = {
            "action": sanitize_for_observability(action),
            "result": sanitize_for_observability(result),
            "content_preview": content_preview,
            "tool_call_count": tool_call_count,
            "resumed_after_human": resumed_after_human,
            "resumed_after_message": resumed_after_message,
        }
        if action_source is not None:
            decision["action_source"] = action_source
        self._audit.record(
            actor=pid,
            action="llm.action",
            target=action.get("action"),
            decision=decision,
        )
        payload = {"ok": True, "action": action, "result": result}
        if resumed_after_human:
            payload["resumed_after_human"] = True
        if resumed_after_message:
            payload["resumed_after_message"] = True
        return payload

    async def _dispatch_completed_llm_action(
        self,
        *,
        pid: str,
        completion: Any,
        actions: list[dict[str, Any]],
        parallel_tool_calls: bool,
        host_auto_wait: bool = False,
        resumed_after_human: bool = False,
        call_id: str,
    ) -> dict[str, Any]:
        if parallel_tool_calls and len(actions) > 1:
            return await self._dispatch_action_batch(
                pid=pid,
                completion=completion,
                actions=actions,
                call_id=call_id,
            )
        try:
            with self._task_run_dispatch_scope(pid, "tool"):
                return await self._dispatch_completed_llm_action_admitted(
                    pid=pid,
                    completion=completion,
                    actions=actions,
                    parallel_tool_calls=parallel_tool_calls,
                    host_auto_wait=host_auto_wait,
                    resumed_after_human=resumed_after_human,
                    call_id=call_id,
                )
        except TaskRunDispatchDeferred:
            # ``pending_validated_action_for_pid`` may already have changed a
            # purely local action from validated to dispatching.  The exact
            # tool admission did not occur, so durably rewind that claim and
            # leave it available for an ordinary pause/resume.  Interrupt
            # control may supersede it after the run scope drains.
            self._defer_unstarted_task_run_action(pid)
            return {
                "ok": False,
                "skipped": True,
                "task_run_control_deferred": True,
            }

    async def _dispatch_completed_llm_action_admitted(
        self,
        *,
        pid: str,
        completion: Any,
        actions: list[dict[str, Any]],
        parallel_tool_calls: bool,
        host_auto_wait: bool = False,
        resumed_after_human: bool = False,
        call_id: str,
    ) -> dict[str, Any]:
        if host_auto_wait and len(actions) != 1:
            raise ValueError("host-generated auto-wait must contain exactly one action")
        if parallel_tool_calls and len(actions) > 1:
            raise RuntimeError("parallel TaskRun actions require per-tool admission")
        action = actions[-1]
        tool_call_context = self._selected_completion_tool_call_context(completion)
        content_preview = self._completion_content_preview(completion.content)
        tool_call_count = len(completion.tool_calls)
        host_pending_metadata = (
            self._host_auto_wait_metadata() if host_auto_wait else None
        )
        try:
            result = await self._adispatch_selected_action(
                pid,
                action,
                host_auto_wait=host_auto_wait,
                context_metadata=self._tool_context_identity_metadata(
                    tool_call_context
                ),
            )
        except HumanApprovalRequired as exc:
            return self._wait_for_human_action(
                pid=pid,
                action=action,
                request_id=exc.request_id,
                message=str(exc),
                content_preview=content_preview,
                tool_call_count=tool_call_count,
                pending_metadata=host_pending_metadata,
                task_run_call_id=call_id,
                **tool_call_context,
            )
        except ProcessWaitRequired as exc:
            return self._wait_for_child_action(
                pid=pid,
                action=exc.resume_action or action,
                child_pid=exc.child_pid,
                message=str(exc),
                content_preview=content_preview,
                tool_call_count=tool_call_count,
                pending_metadata=host_pending_metadata,
                task_run_call_id=call_id,
                **tool_call_context,
            )
        except ProcessMessageWaitRequired as exc:
            return self._wait_for_message_action(
                pid=pid,
                action=action,
                filters=exc.filters,
                message=str(exc),
                content_preview=content_preview,
                tool_call_count=tool_call_count,
                pending_metadata=host_pending_metadata,
                task_run_call_id=call_id,
                **tool_call_context,
            )
        self._persist_response_tool_output(
            pid=pid,
            result=result,
            **tool_call_context,
        )
        completed = self._completed_action_result(
            pid=pid,
            action=action,
            result=result,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
            resumed_after_human=resumed_after_human,
            action_source=(
                "host_empty_tool_calls_auto_wait" if host_auto_wait else None
            ),
        )
        self._record_task_run_completed_transcript(
            pid=pid,
            call_id=call_id,
            state="completed",
            paired_outputs_persisted=True,
            result=completed,
        )
        return completed

    async def _dispatch_action_batch(
        self,
        *,
        pid: str,
        completion: Any,
        actions: list[dict[str, Any]],
        call_id: str,
    ) -> dict[str, Any]:
        completed_actions: list[dict[str, Any]] = []
        completed_results: list[dict[str, Any]] = []
        content_preview = self._completion_content_preview(
            getattr(completion, "content", "")
        )
        tool_call_count = len(getattr(completion, "tool_calls", []) or [])
        stop_reason = "completed"
        stopped_action: dict[str, Any] | None = None
        stopped_result: dict[str, Any] | None = None

        for action_index, action in enumerate(actions):
            tool_call_context = self._completion_tool_call_context(
                completion,
                index=action_index,
            )
            try:
                with self._task_run_dispatch_scope(pid, "tool"):
                    result = await self.adispatch(
                        pid,
                        action,
                        context_metadata=self._tool_context_identity_metadata(
                            tool_call_context
                        ),
                    )
                    if result.get("interrupted_by_message"):
                        stop_reason = "interrupted_by_message"
                        stopped_action = action
                        stopped_result = result
                        self._persist_response_tool_output(
                            pid=pid,
                            result=result,
                            **tool_call_context,
                        )
                        self._persist_unexecuted_parallel_tool_outputs(
                            pid=pid,
                            completion=completion,
                            start_index=action_index + 1,
                            reason=stop_reason,
                        )
                        break
                    self._persist_response_tool_output(
                        pid=pid,
                        result=result,
                        **tool_call_context,
                    )
                    completed_actions.append(action)
                    completed_results.append(result)
            except TaskRunDispatchDeferred:
                if action_index == 0:
                    self._defer_unstarted_task_run_action(pid)
                    return {
                        "ok": False,
                        "skipped": True,
                        "task_run_control_deferred": True,
                    }
                stop_reason = "task_run_control_deferred"
                self._persist_unexecuted_parallel_tool_outputs(
                    pid=pid,
                    completion=completion,
                    start_index=action_index,
                    reason=stop_reason,
                )
                break
            except (
                HumanApprovalRequired,
                ProcessWaitRequired,
                ProcessMessageWaitRequired,
            ) as exc:
                return self._complete_parallel_action_wait(
                    pid=pid,
                    completion=completion,
                    actions=actions,
                    action_index=action_index,
                    action=action,
                    wait=exc,
                    completed_actions=completed_actions,
                    completed_results=completed_results,
                    content_preview=content_preview,
                    tool_call_count=tool_call_count,
                    call_id=call_id,
                    tool_call_context=tool_call_context,
                )
            except ResourceLimitExceeded:
                self._persist_unexecuted_parallel_tool_outputs(
                    pid=pid,
                    completion=completion,
                    start_index=action_index,
                    reason="resource_limit_exceeded",
                )
                self._record_action_batch(
                    pid=pid,
                    actions=actions,
                    completed_actions=completed_actions,
                    completed_results=completed_results,
                    content_preview=content_preview,
                    tool_call_count=tool_call_count,
                    stop_reason="resource_limit_exceeded",
                )
                raise

            if stop_reason == "interrupted_by_message":
                break
            if not result.get("ok"):
                stop_reason = "tool_failed"
                break
            if action.get("action") == "exec_process":
                # A successful exec rotates the process execution generation
                # and replaces the active image/tool contract.  Never dispatch
                # later calls selected under the pre-exec prompt or execution
                # token; the next quantum must rebuild both from the new image.
                stop_reason = "process_exec"
                break
            if result.get("message_notice"):
                stop_reason = "message_notice"
                break
            if self._process_is_terminal(pid):
                stop_reason = "process_terminal"
                break

        return self._settle_parallel_action_batch(
            pid=pid,
            completion=completion,
            actions=actions,
            completed_actions=completed_actions,
            completed_results=completed_results,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
            stop_reason=stop_reason,
            stopped_action=stopped_action,
            stopped_result=stopped_result,
            call_id=call_id,
        )

    def _complete_parallel_action_wait(
        self,
        *,
        pid: str,
        completion: Any,
        actions: list[dict[str, Any]],
        action_index: int,
        action: dict[str, Any],
        wait: HumanApprovalRequired
        | ProcessWaitRequired
        | ProcessMessageWaitRequired,
        completed_actions: list[dict[str, Any]],
        completed_results: list[dict[str, Any]],
        content_preview: str,
        tool_call_count: int,
        call_id: str,
        tool_call_context: dict[str, str | None],
    ) -> dict[str, Any]:
        if isinstance(wait, HumanApprovalRequired):
            stop_reason = "waiting_human"
        elif isinstance(wait, ProcessWaitRequired):
            stop_reason = "waiting_child"
        else:
            stop_reason = "waiting_message"
        self._persist_unexecuted_parallel_tool_outputs(
            pid=pid,
            completion=completion,
            start_index=action_index + 1,
            reason=stop_reason,
        )
        wait_result = self._parallel_batch_wait_result(
            completed_actions,
            completed_results,
        )
        pending_action = action
        if isinstance(wait, HumanApprovalRequired):
            payload = self._wait_for_human_action(
                pid=pid,
                action=action,
                request_id=wait.request_id,
                message=str(wait),
                content_preview=content_preview,
                tool_call_count=tool_call_count,
                task_run_call_id=call_id,
                task_run_wait_result=wait_result,
                **tool_call_context,
            )
        elif isinstance(wait, ProcessWaitRequired):
            pending_action = wait.resume_action or action
            payload = self._wait_for_child_action(
                pid=pid,
                action=pending_action,
                child_pid=wait.child_pid,
                message=str(wait),
                content_preview=content_preview,
                tool_call_count=tool_call_count,
                task_run_call_id=call_id,
                task_run_wait_result=wait_result,
                **tool_call_context,
            )
        else:
            payload = self._wait_for_message_action(
                pid=pid,
                action=action,
                filters=wait.filters,
                message=str(wait),
                content_preview=content_preview,
                tool_call_count=tool_call_count,
                task_run_call_id=call_id,
                task_run_wait_result=wait_result,
                **tool_call_context,
            )
        self._record_action_batch(
            pid=pid,
            actions=actions,
            completed_actions=completed_actions,
            completed_results=completed_results,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
            stop_reason=stop_reason,
            pending_action=pending_action,
        )
        return self._with_parallel_batch_progress(
            payload,
            completed_actions,
            completed_results,
        )

    def _settle_parallel_action_batch(
        self,
        *,
        pid: str,
        completion: Any,
        actions: list[dict[str, Any]],
        completed_actions: list[dict[str, Any]],
        completed_results: list[dict[str, Any]],
        content_preview: str,
        tool_call_count: int,
        stop_reason: str,
        stopped_action: dict[str, Any] | None,
        stopped_result: dict[str, Any] | None,
        call_id: str,
    ) -> dict[str, Any]:
        if stop_reason != "completed" and stop_reason != "interrupted_by_message":
            self._persist_unexecuted_parallel_tool_outputs(
                pid=pid,
                completion=completion,
                start_index=len(completed_actions),
                reason=stop_reason,
            )

        self._record_action_batch(
            pid=pid,
            actions=actions,
            completed_actions=completed_actions,
            completed_results=completed_results,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
            stop_reason=stop_reason,
            stopped_action=stopped_action,
            stopped_result=stopped_result,
        )
        payload: dict[str, Any] = {
            "ok": True,
            "parallel_tool_calls": True,
            "actions": completed_actions,
            "results": completed_results,
            "tool_call_count": tool_call_count,
            "executed_count": len(completed_actions),
            "stop_reason": stop_reason,
        }
        completed = self._complete_action_batch_payload(
            payload=payload,
            completed_actions=completed_actions,
            completed_results=completed_results,
            stopped_action=stopped_action,
            stopped_result=stopped_result,
        )
        self._record_task_run_completed_transcript(
            pid=pid,
            call_id=call_id,
            state="completed",
            paired_outputs_persisted=True,
            result=completed,
        )
        return completed

    @staticmethod
    def _complete_action_batch_payload(
        *,
        payload: dict[str, Any],
        completed_actions: list[dict[str, Any]],
        completed_results: list[dict[str, Any]],
        stopped_action: dict[str, Any] | None,
        stopped_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if completed_actions:
            payload["action"] = completed_actions[-1]
            payload["result"] = completed_results[-1]
        elif stopped_action is not None and stopped_result is not None:
            payload["action"] = stopped_action
            payload["result"] = stopped_result
        if stopped_action is not None and stopped_result is not None:
            payload.update(
                {
                    "stopped_action": stopped_action,
                    "stopped_result": stopped_result,
                }
            )
        return payload

    def _record_action_batch(
        self,
        *,
        pid: str,
        actions: list[dict[str, Any]],
        completed_actions: list[dict[str, Any]],
        completed_results: list[dict[str, Any]],
        content_preview: str,
        tool_call_count: int,
        stop_reason: str,
        pending_action: dict[str, Any] | None = None,
        stopped_action: dict[str, Any] | None = None,
        stopped_result: dict[str, Any] | None = None,
    ) -> None:
        self._audit.record(
            actor=pid,
            action="llm.action_batch",
            target=f"process:{pid}",
            decision={
                "actions": sanitize_for_observability(actions),
                "completed_actions": sanitize_for_observability(completed_actions),
                "completed_results": sanitize_for_observability(completed_results),
                "pending_action": sanitize_for_observability(pending_action) if pending_action else None,
                "stopped_action": sanitize_for_observability(stopped_action) if stopped_action else None,
                "stopped_result": sanitize_for_observability(stopped_result) if stopped_result else None,
                "content_preview": content_preview,
                "tool_call_count": tool_call_count,
                "requested_count": len(actions),
                "executed_count": len(completed_actions),
                "stop_reason": stop_reason,
            },
        )

    @staticmethod
    def _with_parallel_batch_progress(
        payload: dict[str, Any],
        completed_actions: list[dict[str, Any]],
        completed_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload["parallel_tool_calls"] = True
        payload["completed_actions"] = completed_actions
        payload["completed_results"] = completed_results
        payload["executed_count"] = len(completed_actions)
        return payload

    @staticmethod
    def _parallel_batch_wait_result(
        completed_actions: list[dict[str, Any]],
        completed_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "parallel_tool_calls": True,
            "completed_actions": list(completed_actions),
            "completed_results": list(completed_results),
            "executed_count": len(completed_actions),
        }

    def _process_is_terminal(self, pid: str) -> bool:
        return self._process.get(pid).status in {
            ProcessStatus.EXITED,
            ProcessStatus.FAILED,
            ProcessStatus.KILLED,
        }

    def _wait_for_human_action(
        self,
        pid: str,
        action: dict[str, Any],
        request_id: str,
        message: str,
        content_preview: str,
        tool_call_count: int,
        response_id: str | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        pending_metadata: dict[str, Any] | None = None,
        task_run_call_id: str | None = None,
        task_run_wait_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected_pending_metadata = self._pending_metadata_with_task_run_transcript(
            pending_metadata,
            task_run_call_id,
        )
        resume_token = self._persist_pending_action(
            pid,
            wait_type="human",
            request_id=request_id,
            action=action,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
            response_id=response_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            pending_metadata=selected_pending_metadata,
        )
        operation_context = self.pending.get(pid) or {}
        self.pending.remember(pid, "human", {
            "request_id": request_id,
            "resume_token": resume_token,
            "llm_operation_id": operation_context.get("llm_operation_id"),
            "tool_operation_id": operation_context.get("tool_operation_id"),
            "action": dict(action),
            "content_preview": content_preview,
            "tool_call_count": tool_call_count,
            "response_id": response_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "pending_metadata": selected_pending_metadata,
        })
        self._audit.record(
            actor=pid,
            action="llm.action_waiting_human",
            target=f"human_request:{request_id}",
            decision={
                "request_id": request_id,
                "action": sanitize_for_observability(action),
                "message": message,
                "tool_call_count": tool_call_count,
            },
        )
        payload = {"ok": False, "waiting_human": True, "request_id": request_id}
        self._record_task_run_completed_transcript(
            pid=pid,
            call_id=task_run_call_id,
            state="waiting",
            paired_outputs_persisted=True,
            result=task_run_wait_result,
            durable_wait={"wait_type": "human", "request_id": request_id},
        )
        return payload

    def _wait_for_llm_release(
        self,
        pid: str,
        exc: _LLMReleaseApprovalRequired,
    ) -> dict[str, Any]:
        prepared = dict(exc.prepared_request)
        durable_action = (
            prepared
            if self.config.llm.persist_full_io
            else self._redacted_llm_release_action(prepared)
        )
        resume_token = self._persist_pending_action(
            pid,
            wait_type="llm_release",
            request_id=exc.request_id,
            action=durable_action,
            content_preview="",
            tool_call_count=0,
        )
        operation_context = self.pending.get(pid) or {}
        self.pending.remember(pid, "llm_release", {
            "request_id": exc.request_id,
            "resume_token": resume_token,
            "llm_operation_id": operation_context.get("llm_operation_id"),
            "tool_operation_id": operation_context.get("tool_operation_id"),
            "action": prepared,
            "data_flow_context": dict(
                operation_context.get("data_flow_context") or {}
            ),
        })
        self._audit.record(
            actor=pid,
            action="llm.release_waiting_human",
            target=f"human_request:{exc.request_id}",
            decision={
                "request_id": exc.request_id,
                "profile_id": prepared.get("profile_id"),
                "payload_sha256": dict(prepared.get("canonical_args") or {}).get(
                    "payload_sha256"
                ),
                "attempt": prepared.get("attempt"),
            },
        )
        return {
            "ok": False,
            "waiting_human": True,
            "request_id": exc.request_id,
        }

    @classmethod
    def _redacted_llm_release_action(
        cls,
        prepared: dict[str, Any],
    ) -> dict[str, Any]:
        canonical_args = dict(prepared.get("canonical_args") or {})
        request_options = dict(prepared.get("request_options") or {})
        return {
            "kind": "llm_release_request_redacted",
            "schema_version": 1,
            "pid": prepared.get("pid"),
            "call_id": prepared.get("call_id"),
            "profile_id": prepared.get("profile_id"),
            "payload_sha256": canonical_args.get("payload_sha256"),
            "prepared_request_sha256": cls._prepared_llm_release_sha256(prepared),
            "attempt": prepared.get("attempt"),
            "payload_retained": False,
            "context_pressure": dict(
                request_options.get("context_pressure") or {}
            ),
        }

    @staticmethod
    def _prepared_llm_release_sha256(prepared: dict[str, Any]) -> str:
        return hashlib.sha256(
            dumps(to_jsonable(prepared)).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _resolve_pending_llm_release_payload(
        cls,
        *,
        in_memory_action: dict[str, Any],
        durable_action: dict[str, Any],
    ) -> dict[str, Any]:
        durable_kind = str(durable_action.get("kind") or "")
        if durable_kind == "llm_release_request":
            return durable_action
        if durable_kind != "llm_release_request_redacted":
            raise RuntimeError("durable pending LLM release has an invalid payload kind")
        if durable_action.get("schema_version") != 1:
            raise RuntimeError(
                "durable pending LLM release has an unsupported redacted schema"
            )

        expected_sha256 = str(
            durable_action.get("prepared_request_sha256") or ""
        )
        if len(expected_sha256) != 64:
            raise RuntimeError(
                "durable pending LLM release is missing its prepared-request hash"
            )
        if str(in_memory_action.get("kind") or "") != "llm_release_request":
            raise _LLMReleasePayloadUnavailable(
                "prepared LLM release payload is unavailable because full-I/O "
                "retention was disabled and the exact in-memory request was lost"
            )
        actual_sha256 = cls._prepared_llm_release_sha256(in_memory_action)
        if not hmac.compare_digest(actual_sha256, expected_sha256):
            raise RuntimeError(
                "in-memory prepared LLM release does not match its durable hash"
            )
        return in_memory_action

    def _record_llm_release_payload_unavailable(
        self,
        *,
        pid: str,
        request_id: str,
        claimed: dict[str, Any],
        error: RuntimeError,
    ) -> None:
        durable_action = dict(claimed.get("action") or {})
        self._audit.record(
            actor="llm.executor",
            action="llm.release_resume_payload_unavailable",
            target=f"human_request:{request_id}",
            decision={
                "request_id": request_id,
                "profile_id": durable_action.get("profile_id"),
                "payload_sha256": durable_action.get("payload_sha256"),
                "prepared_request_sha256": durable_action.get(
                    "prepared_request_sha256"
                ),
                "error_type": type(error).__name__,
                "persist_full_io": self.config.llm.persist_full_io,
                "replayed": False,
            },
        )

    def _wait_for_child_action(
        self,
        pid: str,
        action: dict[str, Any],
        child_pid: str,
        message: str,
        content_preview: str,
        tool_call_count: int,
        response_id: str | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        pending_metadata: dict[str, Any] | None = None,
        task_run_call_id: str | None = None,
        task_run_wait_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected_pending_metadata = self._pending_metadata_with_task_run_transcript(
            pending_metadata,
            task_run_call_id,
        )
        resume_token = self._persist_pending_action(
            pid,
            wait_type="child",
            child_pid=child_pid,
            action=action,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
            response_id=response_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            pending_metadata=selected_pending_metadata,
        )
        operation_context = self.pending.get(pid) or {}
        self.pending.remember(pid, "child", {
            "child_pid": child_pid,
            "resume_token": resume_token,
            "llm_operation_id": operation_context.get("llm_operation_id"),
            "tool_operation_id": operation_context.get("tool_operation_id"),
            "action": dict(action),
            "content_preview": content_preview,
            "tool_call_count": tool_call_count,
            "response_id": response_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "pending_metadata": selected_pending_metadata,
        })
        self._audit.record(
            actor=pid,
            action="llm.action_waiting_child",
            target=f"process:{child_pid}",
            decision={
                "child_pid": child_pid,
                "action": sanitize_for_observability(action),
                "message": message,
                "tool_call_count": tool_call_count,
            },
        )
        payload = {"ok": False, "waiting_event": True, "child_pid": child_pid}
        self._record_task_run_completed_transcript(
            pid=pid,
            call_id=task_run_call_id,
            state="waiting",
            paired_outputs_persisted=True,
            result=task_run_wait_result,
            durable_wait={"wait_type": "process", "child_pid": child_pid},
        )
        return payload

    def _wait_for_message_action(
        self,
        pid: str,
        action: dict[str, Any],
        filters: dict[str, Any],
        message: str,
        content_preview: str,
        tool_call_count: int,
        response_id: str | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        pending_metadata: dict[str, Any] | None = None,
        task_run_call_id: str | None = None,
        task_run_wait_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected_pending_metadata = self._pending_metadata_with_task_run_transcript(
            pending_metadata,
            task_run_call_id,
        )
        resume_token = self._persist_pending_action(
            pid,
            wait_type="message",
            filters=filters,
            action=action,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
            response_id=response_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            pending_metadata=selected_pending_metadata,
        )
        operation_context = self.pending.get(pid) or {}
        self.pending.remember(pid, "message", {
            "filters": dict(filters),
            "resume_token": resume_token,
            "llm_operation_id": operation_context.get("llm_operation_id"),
            "tool_operation_id": operation_context.get("tool_operation_id"),
            "action": dict(action),
            "content_preview": content_preview,
            "tool_call_count": tool_call_count,
            "response_id": response_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "pending_metadata": selected_pending_metadata,
        })
        self._audit.record(
            actor=pid,
            action="llm.action_waiting_message",
            target=f"process:{pid}",
            decision={
                "filters": filters,
                "action": sanitize_for_observability(action),
                "message": message,
                "tool_call_count": tool_call_count,
            },
        )
        payload = {"ok": False, "waiting_message": True, "filters": filters}
        self._record_task_run_completed_transcript(
            pid=pid,
            call_id=task_run_call_id,
            state="waiting",
            paired_outputs_persisted=True,
            result=task_run_wait_result,
            durable_wait={"wait_type": "message", "filters": dict(filters)},
        )
        return payload

    async def _resume_pending_human_action(self, pid: str) -> dict[str, Any]:
        pending = self.pending.require_memory(pid, "human")
        resume_token = self._pending_resume_token(pending)
        request_id = str(pending["request_id"])
        request = self._human.get(request_id)
        if request.status == HumanRequestStatus.PENDING:
            return {"ok": False, "waiting_human": True, "request_id": request_id}

        claimed = self.pending.claim(pid, resume_token=resume_token)
        if claimed is None:
            self.pending.forget_generation(pid, "human", resume_token)
            return self._pending_action_resuming_result(pid)
        pending = claimed
        action = dict(pending["action"])
        task_run_call_id = self._pending_task_run_call_id(pending)
        context_management_metadata = self._context_management_pending_metadata(pending)
        self.pending.forget_generation(pid, "human", resume_token)
        if request.status == HumanRequestStatus.APPROVED or (
            self._action_name(action) == "request_permission" and request.status == HumanRequestStatus.REJECTED
        ):
            # Re-dispatch the exact same action. The resume request id is scoped
            # to this single tool call, so concurrent tool calls cannot observe
            # another process' human decision.
            try:
                with self._data_flow.recovered_source_snapshot_access():
                    result = await self.adispatch(
                        pid,
                        action,
                        context_metadata={
                            **self._pending_data_flow_metadata(pending),
                            **self._tool_context_identity_metadata(
                                self._pending_tool_call_context(pending)
                            ),
                            "human_resume_request_id": request_id,
                            "operation_id": pending.get("tool_operation_id"),
                            **self._context_management_dispatch_metadata(pending),
                        },
                    )
            except HumanApprovalRequired as exc:
                return self._wait_for_human_action(
                    pid=pid,
                    action=action,
                    request_id=exc.request_id,
                    message=str(exc),
                    content_preview=str(pending.get("content_preview", "")),
                    tool_call_count=int(pending.get("tool_call_count", 0)),
                    pending_metadata=context_management_metadata or None,
                    task_run_call_id=task_run_call_id,
                    **self._pending_tool_call_context(pending),
                )
            except ProcessMessageWaitRequired as exc:
                return self._wait_for_message_action(
                    pid=pid,
                    action=action,
                    filters=exc.filters,
                    message=str(exc),
                    content_preview=str(pending.get("content_preview", "")),
                    tool_call_count=int(pending.get("tool_call_count", 0)),
                    pending_metadata=context_management_metadata or None,
                    task_run_call_id=task_run_call_id,
                    **self._pending_tool_call_context(pending),
                )
            except ProcessWaitRequired as exc:
                return self._wait_for_child_action(
                    pid=pid,
                    action=exc.resume_action or action,
                    child_pid=exc.child_pid,
                    message=str(exc),
                    content_preview=str(pending.get("content_preview", "")),
                    tool_call_count=int(pending.get("tool_call_count", 0)),
                    pending_metadata=context_management_metadata or None,
                    task_run_call_id=task_run_call_id,
                    **self._pending_tool_call_context(pending),
                )
            except Exception as exc:
                if not context_management_metadata:
                    raise
                return await self._recover_context_management_error(
                    pid, pending, context_management_metadata, exc
                )
            self._persist_response_tool_output(
                pid=pid,
                result=result,
                **self._pending_tool_call_context(pending),
            )
            self._clear_pending_action(pid, self._pending_resume_token(pending))
            if context_management_metadata:
                return await self._finish_resumed_context_management(
                    pid,
                    result=result,
                    metadata=context_management_metadata,
                )
            completed = self._completed_action_result(
                pid=pid,
                action=action,
                result=result,
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
                resumed_after_human=True,
            )
            self._record_task_run_completed_transcript(
                pid=pid,
                call_id=task_run_call_id,
                state="completed",
                paired_outputs_persisted=True,
                result=completed,
            )
            return completed

        error = f"human rejected approval request {request_id}"
        # A rejected per-use approval is surfaced as a failed action result, not
        # as a runtime crash, so the process can explain or choose another path.
        self._emit_pending_action_rejected(pid, action, request_id, error)
        result = {"ok": False, "tool_id": None, "result_oid": None, "payload": None, "error": error}
        self._persist_response_tool_output(
            pid=pid,
            result=result,
            **self._pending_tool_call_context(pending),
        )
        self._clear_pending_action(pid, self._pending_resume_token(pending))
        if context_management_metadata:
            return await self._finish_resumed_context_management(
                pid,
                result=result,
                metadata=context_management_metadata,
            )
        completed = self._completed_action_result(
            pid=pid,
            action=action,
            result=result,
            content_preview=str(pending.get("content_preview", "")),
            tool_call_count=int(pending.get("tool_call_count", 0)),
            resumed_after_human=True,
        )
        self._record_task_run_completed_transcript(
            pid=pid,
            call_id=task_run_call_id,
            state="completed",
            paired_outputs_persisted=True,
            result=completed,
        )
        return completed

    async def _resume_pending_llm_release_action(self, pid: str) -> dict[str, Any]:
        pending = self.pending.require_memory(pid, "llm_release")
        resume_token = self._pending_resume_token(pending)
        request_id = str(pending["request_id"])
        request = self._human.get(request_id)
        if request.status == HumanRequestStatus.PENDING:
            durable = self.pending.get(pid) or {}
            try:
                self._resolve_pending_llm_release_payload(
                    in_memory_action=dict(pending.get("action") or {}),
                    durable_action=dict(durable.get("action") or {}),
                )
            except RuntimeError as error:
                claimed = self.pending.claim(
                    pid,
                    resume_token=resume_token,
                )
                if claimed is None:
                    self.pending.forget_generation(
                        pid,
                        "llm_release",
                        resume_token,
                    )
                    return self._pending_action_resuming_result(pid)
                self.pending.forget_generation(
                    pid,
                    "llm_release",
                    resume_token,
                )
                self._record_llm_release_payload_unavailable(
                    pid=pid,
                    request_id=request_id,
                    claimed=claimed,
                    error=error,
                )
                raise
            return {"ok": False, "waiting_human": True, "request_id": request_id}

        claimed = self.pending.claim(
            pid,
            resume_token=resume_token,
        )
        if claimed is None:
            self.pending.forget_generation(
                pid,
                "llm_release",
                resume_token,
            )
            return self._pending_action_resuming_result(pid)
        self.pending.forget_generation(
            pid,
            "llm_release",
            resume_token,
        )

        if request.status != HumanRequestStatus.APPROVED:
            durable_action = dict(claimed.get("action") or {})
            self._audit.record(
                actor=pid,
                action="llm.release_rejected",
                target=f"human_request:{request_id}",
                decision={
                    "request_id": request_id,
                    "profile_id": durable_action.get("profile_id"),
                    "payload_sha256": (
                        durable_action.get("payload_sha256")
                        or dict(durable_action.get("canonical_args") or {}).get(
                            "payload_sha256"
                        )
                    ),
                },
            )
            self._clear_pending_action(pid, self._pending_resume_token(claimed))
            # A rejected conditional release is a terminal decision for this
            # exact model request.  Leaving the process runnable would rebuild
            # the same prompt on the next quantum and immediately ask for a
            # replacement release approval.  Pause after persisting the
            # structured rejection so an explicit Host resume is required to
            # start a genuinely new model turn.
            self._process.pause_for_host_resume(
                pid,
                f"LLM data release rejected: {request_id}",
            )
            return {
                "ok": False,
                "llm_release_rejected": True,
                "request_id": request_id,
            }

        try:
            prepared = self._resolve_pending_llm_release_payload(
                in_memory_action=dict(pending.get("action") or {}),
                durable_action=dict(claimed.get("action") or {}),
            )
        except RuntimeError as error:
            self._record_llm_release_payload_unavailable(
                pid=pid,
                request_id=request_id,
                claimed=claimed,
                error=error,
            )
            raise

        try:
            flow_context = DataFlowContext.from_dict(
                dict(claimed.get("data_flow_context") or {})
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("durable LLM release has invalid data-flow context") from exc
        flow_token = self._data_flow.push(flow_context)
        try:
            try:
                completed_action = await self._complete_valid_action(
                    pid,
                    list(prepared.get("base_messages") or []),
                    list(prepared.get("tools") or []),
                    max_attempts=int(prepared.get("max_attempts") or 0) or None,
                    response_scope_fingerprint=(
                        str(prepared["response_scope_fingerprint"])
                        if prepared.get("response_scope_fingerprint") is not None
                        else None
                    ),
                    _prepared_request=prepared,
                )
            except _LLMReleaseApprovalRequired as exc:
                return self._wait_for_llm_release(pid, exc)
            return await self._dispatch_resumed_llm_release_completion(
                pid,
                claimed,
                completed_action,
            )
        finally:
            self._data_flow.reset(flow_token)

    async def _dispatch_resumed_llm_release_completion(
        self,
        pid: str,
        claimed: dict[str, Any],
        completed_action: tuple[Any, list[dict[str, Any]], bool, bool, str],
    ) -> dict[str, Any]:
        (
            completion,
            actions,
            parallel_tool_calls,
            host_auto_wait,
            call_id,
        ) = completed_action
        self._clear_pending_action(pid, self._pending_resume_token(claimed))
        return await self._dispatch_completed_llm_action(
            pid=pid,
            completion=completion,
            actions=actions,
            parallel_tool_calls=parallel_tool_calls,
            host_auto_wait=host_auto_wait,
            resumed_after_human=True,
            call_id=call_id,
        )

    def _action_name(self, action: dict[str, Any]) -> str:
        return str(action.get("action") or action.get("tool") or action.get("name") or "")

    def _pending_action_resuming_result(self, pid: str) -> dict[str, Any]:
        pending = self.pending.get(pid) or {}
        status = pending.get("status")
        return {
            "ok": False,
            "pending_action_resuming": status == "resuming",
            "pending_action_already_completed": status == "completed",
            "pending_action_generation_changed": status == "pending",
            "wait_type": pending.get("wait_type"),
        }

    async def _resume_pending_action_fail_closed(self, pid: str, resume: Any) -> dict[str, Any]:
        """Never leave a claimed, non-replayable action on a runnable process."""

        initial = self.pending.get(pid) or {}
        initial_token = str(initial.get("resume_token") or "")
        try:
            return await resume(pid)
        except BaseException as exc:
            current = self.pending.get(pid) or {}
            if (
                initial_token
                and str(current.get("resume_token") or "") == initial_token
                and current.get("status") in {"resuming", "completed"}
            ):
                self._fail_interrupted_pending_resume(pid, current, exc)
            raise

    def _fail_interrupted_pending_resume(
        self,
        pid: str,
        pending: dict[str, Any],
        error: BaseException,
    ) -> None:
        public_error, internal_error = self._llm_error_artifacts(
            error,
            code="llm_pending_resume_error",
        )
        message = (
            "durable LLM action resume failed after its non-replayable claim; "
            f"automatic replay is disabled: {public_error['message']}"
        )
        terminal_error: dict[str, Any] | None = None
        process = self._processes.get_process(pid)
        if process is not None and process.status not in {
            ProcessStatus.EXITED,
            ProcessStatus.FAILED,
            ProcessStatus.KILLED,
        }:
            try:
                self._process.exit(pid, failed=True, message=message)
            except Exception as exc:
                terminal_public, terminal_internal = self._llm_error_artifacts(
                    exc,
                    code="llm_pending_resume_terminal_error",
                )
                terminal_error = {
                    "public_error": terminal_public,
                    "internal_error": terminal_internal,
                }
                # Process finalization can span multiple subsystems.  If it
                # fails after the claim, persist the minimum fail-closed state
                # so a direct run_once caller cannot spin on a RUNNABLE row.
                try:
                    with self._processes.transaction():
                        current = self._processes.get_process(pid)
                        if current is not None and current.status not in {
                            ProcessStatus.EXITED,
                            ProcessStatus.FAILED,
                            ProcessStatus.KILLED,
                        }:
                            self._process_transitions.transition(
                                pid,
                                ProcessStatus.FAILED,
                                expected_revision=current.revision,
                                expected_status=current.status,
                                expected_state_generation=current.state_generation,
                                outcome=FailedProcessOutcome(
                                    code="pending_action_resume_interrupted"
                                ),
                                status_message=message,
                            )
                except Exception as fallback_exc:
                    fallback_public, fallback_internal = self._llm_error_artifacts(
                        fallback_exc,
                        code="llm_pending_resume_terminal_error",
                    )
                    terminal_error = {
                        **dict(terminal_error or {}),
                        "fallback": {
                            "public_error": fallback_public,
                            "internal_error": fallback_internal,
                        },
                    }
        try:
            self._audit.record(
                actor="llm.executor",
                action="llm.pending_action_resume_interrupted",
                target=f"process:{pid}",
                decision={
                    "wait_type": pending.get("wait_type"),
                    "status": pending.get("status"),
                    "error_type": type(error).__name__,
                    "error": public_error["message"],
                    "error_details": public_error,
                    "internal_error": internal_error,
                    "terminal_error": terminal_error,
                    "replayed": False,
                },
                correlation_id=public_error["correlation_id"],
            )
        except Exception:
            # Preserve the original post-claim failure.  The durable resuming
            # row and FAILED process state remain the primary evidence.
            pass

    @staticmethod
    def _pending_resume_token(pending: dict[str, Any]) -> str:
        return pending_resume_token(pending)

    @staticmethod
    def _pending_metadata_with_task_run_transcript(
        metadata: Mapping[str, Any] | None,
        call_id: str | None,
    ) -> dict[str, Any]:
        selected = dict(metadata or {})
        if call_id is not None:
            if not isinstance(call_id, str) or not call_id:
                raise RuntimeError("TaskRun pending transcript call id is invalid")
            selected[PENDING_TASK_RUN_TRANSCRIPT_KEY] = call_id
        return selected

    @staticmethod
    def _pending_task_run_call_id(pending: dict[str, Any]) -> str | None:
        return pending_task_run_transcript_call_id(pending)

    @staticmethod
    def _pending_tool_call_context(pending: dict[str, Any]) -> dict[str, str | None]:
        return {
            "response_id": str(pending["response_id"]) if pending.get("response_id") else None,
            "tool_call_id": str(pending["tool_call_id"]) if pending.get("tool_call_id") else None,
            "tool_name": str(pending["tool_name"]) if pending.get("tool_name") else None,
        }

    @staticmethod
    def _host_auto_wait_metadata() -> dict[str, Any]:
        return dict(_HOST_AUTO_WAIT_METADATA)

    @classmethod
    def _host_auto_wait_pending_metadata(
        cls,
        pending: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = pending_metadata(pending)
        if metadata.get("kind") != _HOST_AUTO_WAIT_METADATA["kind"]:
            return {}
        if metadata != _HOST_AUTO_WAIT_METADATA:
            raise RuntimeError(
                "durable host-generated auto-wait metadata is invalid"
            )
        return cls._host_auto_wait_metadata()

    @staticmethod
    def _context_management_pending_metadata(
        pending: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = pending_metadata(pending)
        if (
            metadata.get("kind") != "context_management_auto"
            or metadata.get("schema_version") != 1
            or metadata.get("source") != "runtime_context_management"
        ):
            return {}
        return metadata

    @classmethod
    def _context_management_dispatch_metadata(
        cls,
        pending: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = cls._context_management_pending_metadata(pending)
        if not metadata:
            return {}
        return {
            "context_management_auto": True,
            "context_pressure_episode_id": metadata.get("episode_id"),
            "context_pressure_policy_fingerprint": metadata.get(
                "policy_fingerprint"
            ),
        }

    async def _finish_resumed_context_management(
        self,
        pid: str,
        *,
        result: dict[str, Any],
        metadata: dict[str, Any],
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        generation_before = str(metadata.get("context_generation") or "")
        generation_after = self._processes.get_llm_context_generation(pid)
        common = {
            **dict(metadata.get("assessment") or {}),
            "source": "runtime_context_management",
            "episode_id": metadata.get("episode_id"),
            "policy_fingerprint": metadata.get("policy_fingerprint"),
            "mode": metadata.get("mode"),
            "tool_name": metadata.get("tool_name"),
            "context_generation_after": generation_after,
        }
        if self._context_compaction_succeeded(
            result,
            generation_before=generation_before,
            generation_after=generation_after,
        ):
            self._audit.record(
                actor=pid,
                action="llm.context_pressure_compacted",
                target=f"process:{pid}",
                decision=common,
            )
            storage_pressure = (
                dict(metadata.get("assessment") or {}).get("trigger")
                == "storage_payload"
            )
            return {
                "ok": True,
                "context_compacted": True,
                "context_storage_pressure": storage_pressure,
                "context_pressure_episode_id": metadata.get("episode_id"),
                "context_generation": generation_after,
                "resumed_context_management": True,
            }

        reason = failure_reason or self._context_management_failure_reason(
            result,
            generation_before=generation_before,
            generation_after=generation_after,
        )
        self._audit.record(
            actor=pid,
            action="llm.context_pressure_failed",
            target=f"process:{pid}",
            decision={
                **common,
                "reason": reason,
                "result": sanitize_for_observability(result),
            },
        )
        if dict(metadata.get("assessment") or {}).get("trigger") == "storage_payload":
            return self._fail_llm_quantum(
                pid,
                RuntimeError(f"LLM context storage compaction failed: {reason}"),
            )
        # The failed maintenance action is deliberately not represented as a
        # model tool result. Continue by rebuilding the ordinary Provider
        # request; the completed pending marker deduplicates this episode.
        return await self._arun_once_impl(pid)

    async def _recover_context_management_error(
        self,
        pid: str,
        pending: dict[str, Any],
        metadata: dict[str, Any],
        error: Exception,
    ) -> dict[str, Any]:
        self._clear_pending_action(pid, self._pending_resume_token(pending))
        public_error, internal_error = self._llm_error_artifacts(
            error,
            code="llm_context_management_error",
        )
        return await self._finish_resumed_context_management(
            pid,
            result={
                "ok": False,
                "tool_id": None,
                "result_oid": None,
                "payload": None,
                "error": public_error["message"],
                "error_details": public_error,
                "internal_error": internal_error,
                "error_type": type(error).__name__,
            },
            metadata=metadata,
            failure_reason=type(error).__name__,
        )

    @staticmethod
    def _completion_tool_call_context(completion: Any, *, index: int) -> dict[str, str | None]:
        response_id = str(
            getattr(completion, "_agent_libos_transcript_output_key", None)
            or getattr(completion, "response_id", "")
            or ""
        ) or None
        tool_calls = list(getattr(completion, "tool_calls", []) or [])
        if not tool_calls:
            return {"response_id": response_id, "tool_call_id": None, "tool_name": None}
        try:
            tool_call = tool_calls[index]
        except IndexError:
            return {"response_id": response_id, "tool_call_id": None, "tool_name": None}
        if not isinstance(tool_call, dict):
            return {"response_id": response_id, "tool_call_id": None, "tool_name": None}
        call_id = str(
            tool_call.get("call_id") or tool_call.get("id") or ""
        ).strip() or None
        tool_name = str(tool_call.get("name") or "").strip() or None
        return {"response_id": response_id, "tool_call_id": call_id, "tool_name": tool_name}

    @staticmethod
    def _tool_context_identity_metadata(
        context: Mapping[str, Any],
    ) -> dict[str, str]:
        """Project Host-captured provider identities into trusted ToolContext metadata."""

        projected: dict[str, str] = {}
        for source, target in (
            ("response_id", "llm_transcript_output_key"),
            ("tool_call_id", "llm_tool_call_id"),
            ("tool_name", "llm_tool_name"),
        ):
            value = context.get(source)
            if isinstance(value, str) and value:
                projected[target] = value
        return projected

    def _selected_completion_tool_call_context(self, completion: Any) -> dict[str, str | None]:
        tool_calls = list(getattr(completion, "tool_calls", []) or [])
        for index in range(len(tool_calls) - 1, -1, -1):
            tool_call = tool_calls[index]
            if not isinstance(tool_call, dict):
                continue
            try:
                tool_call_to_action(
                    tool_call,
                    max_argument_bytes=(
                        self.config.tools.tool_call_args_hard_limit_bytes
                    ),
                )
            except Exception:
                continue
            return self._completion_tool_call_context(completion, index=index)
        response_id = str(
            getattr(completion, "_agent_libos_transcript_output_key", None)
            or getattr(completion, "response_id", "")
            or ""
        ) or None
        return {"response_id": response_id, "tool_call_id": None, "tool_name": None}

    @staticmethod
    def _image_only_expected_tool_outputs(
        transcript: Mapping[str, Any],
        response_id: str,
    ) -> dict[str, str]:
        if transcript.get("schema_version") != _IMAGE_ONLY_TRANSCRIPT_SCHEMA_VERSION:
            raise ValidationError("image_only transcript output key changed")
        if transcript.get("output_key") != response_id:
            raise ValidationError("image_only transcript output key changed")
        manifest = transcript.get("tool_calls")
        if not isinstance(manifest, list):
            raise ValidationError("image_only transcript tool-call manifest is unavailable")
        expected = {
            str(item.get("call_id") or ""): str(item.get("name") or "")
            for item in manifest
            if isinstance(item, dict)
        }
        if len(expected) != len(manifest):
            raise ValidationError("image_only transcript tool output is not paired")
        return expected

    def _persist_image_only_tool_output(
        self,
        *,
        pid: str,
        call: LLMCallRecord | None,
        result: dict[str, Any],
        response_id: str,
        tool_call_id: str,
        tool_name: str | None,
        synthetic: bool,
    ) -> bool:
        transcript = (
            call.request_options.get(_IMAGE_ONLY_TRANSCRIPT_KEY)
            if call is not None
            else None
        )
        if not isinstance(transcript, dict):
            return False
        expected = self._image_only_expected_tool_outputs(transcript, response_id)
        if tool_call_id not in expected:
            raise ValidationError("image_only transcript tool output is not paired")
        if expected[tool_call_id] != str(tool_name or ""):
            raise ValidationError("image_only transcript tool output is not paired")
        payload = result.get("payload")
        if payload is None:
            payload = {
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
            }
        content = payload if isinstance(payload, str) else dumps(to_jsonable(payload))
        result_oid = str(result.get("result_oid") or "") or None
        result_flow = (
            self._data_flow.context_from_source_oids(
                pid,
                [result_oid],
                include_current=False,
            )
            if result_oid is not None
            else self._data_flow.current_context()
        )
        envelope = {
            "schema_version": _IMAGE_ONLY_TOOL_OUTPUT_SCHEMA_VERSION,
            "content": content,
            "result_oid": result_oid,
            "ok": bool(result.get("ok")),
            "synthetic": bool(synthetic),
            "labels": result_flow.labels.to_dict(),
        }
        self._processes.upsert_llm_tool_output(
            pid=pid,
            response_id=response_id,
            call_id=tool_call_id,
            tool_name=tool_name,
            output=dumps(envelope),
        )
        try:
            output_rows = self._processes.list_llm_tool_outputs(
                pid=pid,
                response_id=response_id,
            )
        except Exception as exc:
            # The output upsert above is the correctness boundary. A read used
            # only to decide whether retention cleanup may run cannot turn a
            # successfully executed tool into a failed/replayed quantum.
            try:
                self._audit.record(
                    actor=pid,
                    action="llm.image_only_request_anchor_supersede_failed",
                    target=f"llm_call:{call.call_id if call is not None else response_id}",
                    decision={
                        "error_type": type(exc).__name__,
                        "phase": "paired_output_query",
                    },
                )
            except Exception:
                pass
            return True
        if {str(row.get("call_id") or "") for row in output_rows} == set(expected):
            self._try_supersede_image_only_request_anchor(
                pid=pid,
                image_id=call.image_id if call is not None else None,
                transcript_call_id=(
                    call.call_id if call is not None else response_id
                ),
                request_options=call.request_options if call is not None else {},
            )
        return True

    def _persist_response_tool_output(
        self,
        *,
        pid: str,
        result: dict[str, Any],
        response_id: str | None,
        tool_call_id: str | None,
        tool_name: str | None,
        synthetic: bool = False,
    ) -> None:
        if not response_id or not tool_call_id or not self.config.llm.persist_full_io:
            return
        call = self._processes.get_llm_call(response_id)
        if self._persist_image_only_tool_output(
            pid=pid,
            call=call,
            result=result,
            response_id=response_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            synthetic=synthetic,
        ):
            return
        call = self._processes.get_latest_llm_call(
            pid=pid,
            purpose="action_selection",
        )
        if (
            call is None
            or call.api != "responses"
            or call.response_id != response_id
            or call.request_options.get("openai_provider_chain_eligible") is not True
        ):
            return
        manifest = call.request_options.get("openai_response_tool_calls")
        if not isinstance(manifest, list):
            return
        expected_call_ids = [
            str(item.get("call_id") or "").strip()
            for item in manifest
            if isinstance(item, dict)
        ]
        if (
            len(expected_call_ids) != len(manifest)
            or any(not call_id for call_id in expected_call_ids)
            or len(set(expected_call_ids)) != len(expected_call_ids)
            or tool_call_id not in set(expected_call_ids)
        ):
            return
        self._processes.upsert_llm_tool_output(
            pid=pid,
            response_id=response_id,
            call_id=tool_call_id,
            tool_name=tool_name,
            output=dumps(result),
        )

    def _persist_unexecuted_parallel_tool_outputs(
        self,
        *,
        pid: str,
        completion: Any,
        start_index: int,
        reason: str,
    ) -> None:
        tool_calls = list(getattr(completion, "tool_calls", []) or [])
        for index in range(max(0, start_index), len(tool_calls)):
            context = self._completion_tool_call_context(completion, index=index)
            if not context.get("tool_call_id"):
                continue
            self._persist_response_tool_output(
                pid=pid,
                result={
                    "ok": False,
                    "tool_id": None,
                    "result_oid": None,
                    "payload": {
                        "ok": False,
                        "cancelled": True,
                        "effect_started": False,
                        "reason": reason,
                    },
                    "error": None,
                },
                synthetic=True,
                **context,
            )

    async def _resume_pending_wait_action(self, pid: str) -> dict[str, Any]:
        pending = self.pending.require_memory(pid, "child")
        resume_token = self._pending_resume_token(pending)
        child_pid = str(pending["child_pid"])
        child = self._process.get(child_pid)
        if child.status not in {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}:
            return {"ok": False, "waiting_event": True, "child_pid": child_pid}

        claimed = self.pending.claim(pid, resume_token=resume_token)
        if claimed is None:
            self.pending.forget_generation(pid, "child", resume_token)
            return self._pending_action_resuming_result(pid)
        pending = claimed
        action = dict(pending["action"])
        task_run_call_id = self._pending_task_run_call_id(pending)
        context_management_metadata = self._context_management_pending_metadata(
            pending
        )
        self.pending.forget_generation(pid, "child", resume_token)
        try:
            with self._data_flow.recovered_source_snapshot_access():
                result = await self.adispatch(
                    pid,
                    action,
                    context_metadata={
                        **self._pending_data_flow_metadata(pending),
                        **self._tool_context_identity_metadata(
                            self._pending_tool_call_context(pending)
                        ),
                        "pending_child_resume": True,
                        "pending_child_pid": child_pid,
                        "operation_id": pending.get("tool_operation_id"),
                        **self._context_management_dispatch_metadata(pending),
                    },
                )
        except ProcessWaitRequired as exc:
            return self._wait_for_child_action(
                pid=pid,
                action=exc.resume_action or action,
                child_pid=exc.child_pid,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
                pending_metadata=context_management_metadata or None,
                task_run_call_id=task_run_call_id,
                **self._pending_tool_call_context(pending),
            )
        except HumanApprovalRequired as exc:
            return self._wait_for_human_action(
                pid=pid,
                action=action,
                request_id=exc.request_id,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
                pending_metadata=context_management_metadata or None,
                task_run_call_id=task_run_call_id,
                **self._pending_tool_call_context(pending),
            )
        except ProcessMessageWaitRequired as exc:
            return self._wait_for_message_action(
                pid=pid,
                action=action,
                filters=exc.filters,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
                pending_metadata=context_management_metadata or None,
                task_run_call_id=task_run_call_id,
                **self._pending_tool_call_context(pending),
            )
        except Exception as exc:
            if not context_management_metadata:
                raise
            return await self._recover_context_management_error(
                pid, pending, context_management_metadata, exc
            )
        self._persist_response_tool_output(
            pid=pid,
            result=result,
            **self._pending_tool_call_context(pending),
        )
        self._clear_pending_action(pid, self._pending_resume_token(pending))
        if context_management_metadata:
            return await self._finish_resumed_context_management(
                pid,
                result=result,
                metadata=context_management_metadata,
            )
        completed = self._completed_action_result(
            pid=pid,
            action=action,
            result=result,
            content_preview=str(pending.get("content_preview", "")),
            tool_call_count=int(pending.get("tool_call_count", 0)),
            resumed_after_human=False,
        )
        self._record_task_run_completed_transcript(
            pid=pid,
            call_id=task_run_call_id,
            state="completed",
            paired_outputs_persisted=True,
            result=completed,
        )
        return completed

    async def _resume_pending_message_action(self, pid: str) -> dict[str, Any]:
        pending = self.pending.require_memory(pid, "message")
        resume_token = self._pending_resume_token(pending)
        filters = dict(pending.get("filters") or {})
        messages = self._messages.unread(
            pid,
            kind=filters.get("kind"),
            sender=filters.get("sender"),
            channel=filters.get("channel"),
            correlation_id=filters.get("correlation_id"),
            reply_to=filters.get("reply_to"),
            message_ids=filters.get("message_ids"),
        )
        if not messages:
            return {"ok": False, "waiting_message": True, "filters": filters}
        claimed = self.pending.claim(pid, resume_token=resume_token)
        if claimed is None:
            self.pending.forget_generation(pid, "message", resume_token)
            return self._pending_action_resuming_result(pid)
        pending = claimed
        action = dict(pending["action"])
        task_run_call_id = self._pending_task_run_call_id(pending)
        context_management_metadata = self._context_management_pending_metadata(
            pending
        )
        host_auto_wait_metadata = self._host_auto_wait_pending_metadata(pending)
        pending_action_metadata = (
            context_management_metadata or host_auto_wait_metadata
        )
        self.pending.forget_generation(pid, "message", resume_token)
        try:
            with self._data_flow.recovered_source_snapshot_access():
                result = await self._adispatch_selected_action(
                    pid,
                    action,
                    host_auto_wait=bool(host_auto_wait_metadata),
                    context_metadata={
                        **self._pending_data_flow_metadata(pending),
                        **self._tool_context_identity_metadata(
                            self._pending_tool_call_context(pending)
                        ),
                        "operation_id": pending.get("tool_operation_id"),
                        **self._context_management_dispatch_metadata(pending),
                    },
                )
        except ProcessMessageWaitRequired as exc:
            return self._wait_for_message_action(
                pid=pid,
                action=action,
                filters=exc.filters,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
                pending_metadata=pending_action_metadata or None,
                task_run_call_id=task_run_call_id,
                **self._pending_tool_call_context(pending),
            )
        except ProcessWaitRequired as exc:
            return self._wait_for_child_action(
                pid=pid,
                action=exc.resume_action or action,
                child_pid=exc.child_pid,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
                pending_metadata=pending_action_metadata or None,
                task_run_call_id=task_run_call_id,
                **self._pending_tool_call_context(pending),
            )
        except HumanApprovalRequired as exc:
            return self._wait_for_human_action(
                pid=pid,
                action=action,
                request_id=exc.request_id,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
                pending_metadata=pending_action_metadata or None,
                task_run_call_id=task_run_call_id,
                **self._pending_tool_call_context(pending),
            )
        except Exception as exc:
            if not context_management_metadata:
                raise
            return await self._recover_context_management_error(
                pid, pending, context_management_metadata, exc
            )
        self._persist_response_tool_output(
            pid=pid,
            result=result,
            **self._pending_tool_call_context(pending),
        )
        self._clear_pending_action(pid, self._pending_resume_token(pending))
        if context_management_metadata:
            return await self._finish_resumed_context_management(
                pid,
                result=result,
                metadata=context_management_metadata,
            )
        completed = self._completed_action_result(
            pid=pid,
            action=action,
            result=result,
            content_preview=str(pending.get("content_preview", "")),
            tool_call_count=int(pending.get("tool_call_count", 0)),
            resumed_after_message=True,
            action_source=(
                "host_empty_tool_calls_auto_wait"
                if host_auto_wait_metadata
                else None
            ),
        )
        self._record_task_run_completed_transcript(
            pid=pid,
            call_id=task_run_call_id,
            state="completed",
            paired_outputs_persisted=True,
            result=completed,
        )
        return completed

    def _emit_pending_action_rejected(self, pid: str, action: dict[str, Any], request_id: str, error: str) -> None:
        tool_name = str(action.get("action"))
        source = f"tool:{tool_name}"
        try:
            handle = self._tools.resolve(tool_name, pid=pid)
            source = f"tool:{handle.tool_id}"
        except Exception:
            pass
        self._events.emit(
            EventType.TOOL_FAILED,
            source=source,
            target=pid,
            payload={
                "error": error,
                "tool_name": tool_name,
                "request_id": request_id,
                "policy_decision": "deny",
                "policy_reason": "human_rejected_per_use_approval",
            },
        )
        self._audit.record(
            actor=pid,
            action="llm.pending_action_rejected",
            target=tool_name,
            decision={"request_id": request_id, "action": sanitize_for_observability(action), "error": error},
        )

    def _completion_to_action(self, content: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        actions, _ = self.actions.completion_to_actions(
            content,
            tool_calls,
            parallel_tool_calls=False,
            auto_wait_on_empty_tool_calls=False,
            fallback_json_actions=False,
        )
        return actions[0]

    def _completion_to_actions(
        self,
        content: str,
        tool_calls: list[dict[str, Any]],
        *,
        parallel_tool_calls: bool,
        auto_wait_on_empty_tool_calls: bool,
        fallback_json_actions: bool = False,
    ) -> tuple[list[dict[str, Any]], bool]:
        return self.actions.completion_to_actions(
            content,
            tool_calls,
            parallel_tool_calls=parallel_tool_calls,
            auto_wait_on_empty_tool_calls=auto_wait_on_empty_tool_calls,
            fallback_json_actions=fallback_json_actions,
        )

    @staticmethod
    def _auto_wait_message_action() -> dict[str, Any]:
        return auto_wait_message_action()

    async def _complete_valid_action(
        self,
        pid: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_attempts: int | None = None,
        response_scope_fingerprint: str | None = None,
        image_only_anchor: Mapping[str, str] | None = None,
        task_run_requirement_binding: Mapping[str, Any] | None = None,
        _prepared_request: dict[str, Any] | None = None,
    ) -> tuple[Any, list[dict[str, Any]], bool, bool, str]:
        attempt_messages = list(
            (_prepared_request or {}).get("attempt_messages") or messages
        )
        last_error: Exception | None = None
        selected_max_attempts = max_attempts or self.config.llm.action_repair_attempts
        start_attempt = int((_prepared_request or {}).get("attempt") or 1)
        prepared_request = _prepared_request
        for attempt_number in range(start_attempt, selected_max_attempts + 1):
            try:
                (
                    completion,
                    parallel_tool_calls,
                    auto_wait_on_empty_tool_calls,
                    fallback_json_actions,
                    profile_id,
                    call_id,
                ) = await self._complete_action_recorded(
                    pid=pid,
                    messages=attempt_messages,
                    tools=tools,
                    attempt=attempt_number,
                    max_attempts=selected_max_attempts,
                    response_scope_fingerprint=response_scope_fingerprint,
                    image_only_anchor=image_only_anchor,
                    task_run_requirement_binding=task_run_requirement_binding,
                    _prepared_request=prepared_request,
                )
            except _LLMReleaseApprovalRequired as exc:
                exc.prepared_request["base_messages"] = list(messages)
                exc.prepared_request["attempt_messages"] = list(attempt_messages)
                raise
            prepared_request = None
            try:
                raw_actions, auto_wait_used = self._completion_to_actions(
                    completion.content,
                    completion.tool_calls,
                    parallel_tool_calls=parallel_tool_calls,
                    auto_wait_on_empty_tool_calls=auto_wait_on_empty_tool_calls,
                    fallback_json_actions=fallback_json_actions,
                )
                if auto_wait_used:
                    self._audit.record(
                        actor=pid,
                        action="llm.empty_tool_calls_auto_wait",
                        target=f"process:{pid}",
                        decision={
                            "attempt": attempt_number,
                            "llm_profile_id": profile_id,
                            "action": self._auto_wait_message_action(),
                            "content_preview": self._completion_content_preview(
                                completion.content
                            ),
                            "tool_call_count": len(completion.tool_calls),
                        },
                    )
                if auto_wait_used:
                    if len(raw_actions) != 1:
                        raise ValueError(
                            "host-generated auto-wait must contain exactly one action"
                        )
                    actions = [dict(raw_actions[0])]
                    self.actions.validate_host_auto_wait(pid, actions[0])
                else:
                    actions = [
                        self._tools.normalize_model_action(pid, action)
                        for action in raw_actions
                    ]
                    for action in actions:
                        self._validate_dispatchable_action(pid, action)
                if parallel_tool_calls and len(actions) > 1:
                    self._preflight_parallel_tool_batch(pid, actions)
                self._supersede_validated_image_only_empty_head(
                    pid=pid,
                    completion=completion,
                )
                self._publish_and_claim_task_run_validated_action(
                    pid=pid,
                    call_id=call_id,
                    actions=actions,
                    parallel_tool_calls=parallel_tool_calls,
                    host_auto_wait=auto_wait_used,
                    tool_call_count=len(completion.tool_calls),
                )
                return (
                    completion,
                    actions,
                    parallel_tool_calls,
                    auto_wait_used,
                    call_id,
                )
            except ValueError as exc:
                last_error = exc
                self._audit.record(
                    actor=pid,
                    action="llm.action_repair_requested",
                    target=f"process:{pid}",
                    decision={
                        "attempt": attempt_number,
                        "error": self._durable_llm_error(exc),
                        "tool_call_count": len(completion.tool_calls),
                        "tool_calls_preview": self._tool_call_previews(completion.tool_calls),
                        "content_preview": self._completion_content_preview(
                            completion.content
                        ),
                    },
                )
                if attempt_number >= selected_max_attempts:
                    break
                compatibility_hint = (
                    " If native tool calls are unavailable, you may instead use the "
                    "enabled compatibility JSON action protocol."
                    if fallback_json_actions
                    else ""
                )
                attempt_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "The previous model response could not be dispatched: "
                            f"{exc}. Choose "
                            f"{'one or more' if parallel_tool_calls else 'exactly one'} "
                            "available OpenAI tool call by its function name. "
                            f"Available tool names: {self._tools.model_tool_names(pid)}"
                            f"{compatibility_hint}"
                        ),
                    },
                ]
        assert last_error is not None
        raise last_error

    async def _complete_valid_action_with_control_boundary(
        self,
        pid: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        response_scope_fingerprint: str | None,
        image_only_anchor: Mapping[str, str] | None,
        task_run_requirement_binding: Mapping[str, Any] | None,
    ) -> tuple[Any, list[dict[str, Any]], bool, bool, str]:
        """Map a persisted control fence onto the existing handled boundary."""

        try:
            return await self._complete_valid_action(
                pid,
                messages,
                tools,
                response_scope_fingerprint=response_scope_fingerprint,
                image_only_anchor=image_only_anchor,
                task_run_requirement_binding=task_run_requirement_binding,
            )
        except TaskRunDispatchDeferred as exc:
            raise _ContextManagementHandled(
                {
                    "ok": False,
                    "skipped": True,
                    "task_run_control_deferred": True,
                }
            ) from exc

    def _preflight_parallel_tool_batch(self, pid: str, actions: list[dict[str, Any]]) -> None:
        self.actions.preflight_parallel(pid, actions)

    def _record_task_run_validated_transcript(
        self,
        *,
        pid: str,
        call_id: str,
        actions: list[dict[str, Any]],
        parallel_tool_calls: bool,
        host_auto_wait: bool,
        tool_call_count: int,
    ) -> dict[str, Any] | None:
        """Publish a local-only resume point after complete action validation."""

        if self._task_runs is None:
            return None
        record = self._processes.get_llm_call(call_id)
        if record is None or record.status != "ok" or not record.completed_at:
            raise RuntimeError(
                "validated TaskRun transcript is missing from the local LLM ledger"
            )
        try:
            manifest = validated_action_manifest(
                actions,
                call_id=call_id,
                parallel_tool_calls=parallel_tool_calls,
                host_auto_wait=host_auto_wait,
                tool_call_count=tool_call_count,
                data_labels=self._data_flow.current_context().labels.to_dict(),
            )
            self._task_runs.record_validated_transcript(
                pid=pid,
                call_id=call_id,
                action_manifest=manifest,
                context_generation=self._processes.get_llm_context_generation(pid),
            )
        except Exception as exc:
            # A resume-point persistence conflict is a Runtime failure, not a
            # malformed model action.  In particular, never enter action repair
            # and issue another Provider call after the validated completion.
            raise RuntimeError(
                "failed to persist validated TaskRun LLM transcript"
            ) from exc
        return manifest

    def _publish_and_claim_task_run_validated_action(
        self,
        *,
        pid: str,
        call_id: str,
        actions: list[dict[str, Any]],
        parallel_tool_calls: bool,
        host_auto_wait: bool,
        tool_call_count: int,
    ) -> None:
        expected = self._record_task_run_validated_transcript(
            pid=pid,
            call_id=call_id,
            actions=actions,
            parallel_tool_calls=parallel_tool_calls,
            host_auto_wait=host_auto_wait,
            tool_call_count=tool_call_count,
        )
        process = self._processes.get_process(pid)
        is_durable_run = (
            process is not None
            and getattr(process, "task_run_id", None) is not None
        )
        if expected is None or not is_durable_run:
            return
        claimed = self._pending_task_run_validated_action(pid)
        if claimed is None or dumps(claimed) != dumps(expected):
            raise RuntimeError(
                "durable TaskRun action claim changed its validated manifest"
            )

    def _record_task_run_completed_transcript(
        self,
        *,
        pid: str,
        call_id: str | None,
        state: str,
        paired_outputs_persisted: bool,
        result: Mapping[str, Any] | None = None,
        durable_wait: Mapping[str, Any] | None = None,
    ) -> None:
        """Publish a safe point only after a local result or wait is durable."""

        if self._task_runs is None or call_id is None:
            return
        record = self._processes.get_llm_call(call_id)
        if record is None or record.status != "ok" or not record.completed_at:
            raise RuntimeError(
                "completed TaskRun transcript is missing from the local LLM ledger"
            )
        try:
            data_labels = self._task_run_outcome_data_labels(pid, result)
            outcome_manifest = completed_outcome_manifest(
                state=state,
                paired_outputs_persisted=paired_outputs_persisted,
                data_labels=data_labels,
                result=result,
                durable_wait=durable_wait,
            )
            stage_completed = getattr(
                self._task_runs,
                "stage_completed_transcript",
                None,
            )
            if callable(stage_completed):
                stage_completed(
                    pid=pid,
                    call_id=call_id,
                    outcome_manifest=outcome_manifest,
                    context_generation=(
                        self._processes.get_llm_context_generation(pid)
                    ),
                )
            elif self._task_runs.prompt_context_for_pid(pid) is not None:
                raise RuntimeError(
                    "durable TaskRun completed-outcome staging is unavailable"
                )
            self._task_runs.record_completed_transcript(
                pid=pid,
                call_id=call_id,
                outcome_manifest=outcome_manifest,
                context_generation=self._processes.get_llm_context_generation(pid),
            )
        except Exception as exc:
            # Never continue the process after losing the local safe-point
            # commit. The Provider completion must not be replayed merely to
            # make the TaskRun projection look complete.
            raise RuntimeError(
                "failed to persist completed TaskRun LLM transcript"
            ) from exc

    def _task_run_outcome_data_labels(
        self,
        pid: str,
        result: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Bind copied result/wait content to every locally known label."""

        contexts = [self._data_flow.current_context()]
        pending = self.pending.get(pid)
        if pending is not None:
            raw_context = pending.get("data_flow_context")
            if not isinstance(raw_context, dict):
                raise RuntimeError(
                    "durable pending TaskRun action lacks data-flow labels"
                )
            try:
                contexts.append(DataFlowContext.from_dict(raw_context))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "durable pending TaskRun action has invalid data-flow labels"
                ) from exc
        result_oids = self._task_run_result_oids(result)
        for result_oid in result_oids:
            # Settlement is a trusted Runtime projection, not a new process
            # read.  A one-shot object capability may already have been
            # consumed by the Tool (or the process may have exited), so using
            # the public data-flow read path here would incorrectly deny the
            # transcript commit after the effect completed.  Read only the
            # persisted metadata and fail closed if the referenced result is
            # no longer labelable.
            metadata = self._objects.get_persisted_object_metadata(result_oid)
            if metadata is None:
                raise RuntimeError(
                    "TaskRun result Object metadata is unavailable for labeling"
                )
            contexts.append(
                DataFlowContext(
                    labels=DataLabels.from_object_metadata(metadata),
                )
            )
        return DataFlowContext.aggregate(contexts).labels.to_dict()

    @staticmethod
    def _task_run_result_oids(
        result: Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        if result is None:
            return ()
        stack: list[Any] = [result]
        selected: set[str] = set()
        visited = 0
        while stack:
            current = stack.pop()
            visited += 1
            if visited > 4_096:
                raise RuntimeError("TaskRun result projection is too deeply nested")
            if isinstance(current, Mapping):
                result_oid = current.get("result_oid")
                # Tool results can contain the original model arguments as a
                # nested diagnostic projection. Optional ids are sometimes
                # rendered there as string sentinels (for example "None"),
                # and external/domain payloads may reuse the key name. Only a
                # Runtime Object id can contribute data labels; a syntactically
                # valid but missing obj_* id still fails closed in the metadata
                # lookup above.
                if isinstance(result_oid, str) and result_oid.startswith("obj_"):
                    selected.add(result_oid)
                    if len(selected) > 256:
                        raise RuntimeError(
                            "TaskRun result projection contains too many Object ids"
                        )
                stack.extend(current.values())
            elif isinstance(current, (list, tuple)):
                stack.extend(current)
        return tuple(sorted(selected))

    def _validate_dispatchable_action(self, pid: str, action: dict[str, Any]) -> None:
        self.actions.validate(pid, action)

    def _supersede_validated_image_only_empty_head(
        self,
        *,
        pid: str,
        completion: Any,
    ) -> None:
        if list(getattr(completion, "tool_calls", []) or []):
            return
        output_key = str(
            getattr(completion, "_agent_libos_transcript_output_key", "") or ""
        )
        if not output_key:
            return
        call = self._processes.get_llm_call(output_key)
        if call is None:
            return
        transcript = call.request_options.get(_IMAGE_ONLY_TRANSCRIPT_KEY)
        if not isinstance(transcript, dict) or transcript.get("tool_calls") != []:
            return
        # A provider completion is not a usable transcript head until its
        # model action has passed normalization and dispatch validation.  In
        # particular, an empty/invalid completion that enters repair must not
        # discard the only durable first-request retry anchor.
        now = utc_now()
        self._processes.insert_llm_call(
            LLMCallRecord(
                call_id=new_id("llmvalidation"),
                pid=pid,
                image_id=call.image_id,
                purpose=self._image_only_empty_head_validation_purpose(call.call_id),
                status="ok",
                messages=[],
                tools=[],
                request_options={
                    _IMAGE_ONLY_EMPTY_HEAD_VALIDATION_KEY: {
                        "schema_version": (
                            _IMAGE_ONLY_EMPTY_HEAD_VALIDATION_SCHEMA_VERSION
                        ),
                        "transcript_call_id": call.call_id,
                    }
                },
                response_content="",
                tool_calls=[],
                observability={"kind": "image_only_empty_head_validation"},
                created_at=now,
                completed_at=now,
            )
        )
        self._try_supersede_image_only_request_anchor(
            pid=pid,
            image_id=call.image_id,
            transcript_call_id=call.call_id,
            request_options=call.request_options,
        )

    def _tool_call_previews(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        previews: list[dict[str, Any]] = []
        for ordinal, tool_call in enumerate(tool_calls):
            raw_args = tool_call.get("arguments")
            if isinstance(raw_args, str):
                raw_bytes = raw_args.encode("utf-8", errors="replace")
                try:
                    observable_args: Any = json.loads(raw_args)
                except ValueError:
                    observable_args = raw_args
            else:
                raw_text = repr(raw_args)
                raw_bytes = raw_text.encode("utf-8", errors="replace")
                observable_args = raw_args
            observation = sanitize_for_observability(
                observable_args,
                preview_chars=self.config.llm.tool_arguments_preview_chars,
            )
            preview = {
                "ordinal": ordinal,
                "arguments_type": type(raw_args).__name__,
                "arguments_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "arguments_bytes": len(raw_bytes),
            }
            if self.config.llm.persist_full_io:
                preview.update(
                    {
                        "id": tool_call.get("id"),
                        "call_id": tool_call.get("call_id"),
                        "name": tool_call.get("name"),
                        "arguments_preview": observation["preview"],
                        "arguments_truncated": observation["truncated"],
                        "arguments_redacted": observation["redacted"],
                    }
                )
            previews.append(preview)
        return previews

    async def _complete_action_recorded(
        self,
        *,
        pid: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        attempt: int,
        max_attempts: int,
        response_scope_fingerprint: str | None = None,
        image_only_anchor: Mapping[str, str] | None = None,
        task_run_requirement_binding: Mapping[str, Any] | None = None,
        _force_stateless: bool = False,
        _chain_scope_retry: int = 0,
        _prepared_request: dict[str, Any] | None = None,
    ) -> tuple[Any, bool, bool, bool, str, str]:
        state = self._initialize_llm_call_state(
            pid=pid,
            messages=messages,
            tools=tools,
            attempt=attempt,
            max_attempts=max_attempts,
            prepared_request=_prepared_request,
            image_only_anchor=image_only_anchor,
            task_run_requirement_binding=task_run_requirement_binding,
        )
        # Admission is serialized with persisted pause/interrupt generation.
        # Once admitted, the scope stays active through local LLM-call
        # persistence so a concurrent controller can drain it without taking
        # over the process execution lease.
        with self._task_run_dispatch_scope(pid, "provider"):
            try:
                if _prepared_request is None:
                    await self._prepare_fresh_llm_request(
                        state,
                        response_scope_fingerprint=response_scope_fingerprint,
                        force_stateless=_force_stateless,
                    )
                else:
                    self._prepare_resumed_llm_request(state, _prepared_request)
                self._assert_task_run_request_scope_current(
                    state.pid,
                    response_scope_fingerprint,
                )
                completion = await self._invoke_prepared_llm_request(state)
                self._assert_task_run_request_scope_current(
                    state.pid,
                    response_scope_fingerprint,
                    settlement=True,
                )
            except _ContextManagementHandled:
                # No Provider request was attempted, so do not charge or persist a
                # synthetic failed LLM call for host-side maintenance/waiting.
                raise
            except HumanApprovalRequired as exc:
                if not state.prepared:
                    raise
                prepared = self._build_llm_release_request(
                    state,
                    previous=_prepared_request,
                    response_scope_fingerprint=response_scope_fingerprint,
                )
                raise _LLMReleaseApprovalRequired(exc, prepared) from exc
            except _LLMProviderChainScopeChanged:
                if _chain_scope_retry >= 1:
                    raise
                return await self._complete_action_recorded(
                    pid=pid,
                    messages=messages,
                    tools=state.tools,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    response_scope_fingerprint=response_scope_fingerprint,
                    image_only_anchor=image_only_anchor,
                    task_run_requirement_binding=task_run_requirement_binding,
                    _force_stateless=True,
                    _chain_scope_retry=_chain_scope_retry + 1,
                )
            except Exception as exc:
                # Host-side admission failures occur before the protected
                # provider phase and are already represented by resource/audit
                # evidence. Do not turn them into synthetic Provider calls.
                if (
                    isinstance(exc, ResourceLimitExceeded)
                    and not state.provider_dispatched
                ):
                    self._deny_llm_budget_admission(
                        state,
                        reason="resource_envelope_unavailable",
                    )
                if state.provider_dispatched or not isinstance(
                    exc,
                    ResourceLimitExceeded,
                ):
                    self._record_llm_call_error(state, exc)
                raise
            return self._record_llm_call_success(state, completion)

    def _initialize_llm_call_state(
        self,
        *,
        pid: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        attempt: int,
        max_attempts: int,
        prepared_request: dict[str, Any] | None,
        image_only_anchor: Mapping[str, str] | None,
        task_run_requirement_binding: Mapping[str, Any] | None,
    ) -> _LLMCallState:
        process = self._process.get(pid)
        if prepared_request is None:
            profile_id = (
                process.llm_profile_id or self.config.llm.default_profile_id
            )
            request_options: dict[str, Any] = {
                "attempt": attempt,
                "max_attempts": max_attempts,
                "purpose": "action_selection",
                "llm_profile_id": profile_id,
            }
            if image_only_anchor is not None:
                frozen_anchor = dict(image_only_anchor)
                request_options[_IMAGE_ONLY_FROZEN_ANCHOR_KEY] = {
                    "schema_version": _IMAGE_ONLY_FROZEN_ANCHOR_SCHEMA_VERSION,
                    **frozen_anchor,
                    "purpose": self._image_only_request_purpose(frozen_anchor),
                }
            if task_run_requirement_binding is not None:
                request_options[_TASK_RUN_REQUIREMENT_BINDING_KEY] = json.loads(
                    dumps(dict(task_run_requirement_binding))
                )
            return _LLMCallState(
                pid=pid,
                process=process,
                call_id=new_id("llmcall"),
                created_at=utc_now(),
                profile_id=profile_id,
                attempt=attempt,
                max_attempts=max_attempts,
                request_options=request_options,
                request_messages=messages,
                tools=tools,
                flow_context=self._data_flow.current_context(),
            )
        if (
            prepared_request.get("kind") != "llm_release_request"
            or int(prepared_request.get("schema_version") or 0) != 1
            or str(prepared_request.get("pid") or "") != pid
        ):
            raise RuntimeError("invalid durable prepared LLM release request")
        return _LLMCallState(
            pid=pid,
            process=process,
            call_id=str(prepared_request["call_id"]),
            created_at=str(prepared_request["created_at"]),
            profile_id=str(prepared_request["profile_id"]),
            attempt=attempt,
            max_attempts=max_attempts,
            request_options=dict(prepared_request.get("request_options") or {}),
            request_messages=list(prepared_request.get("request_messages") or []),
            tools=list(prepared_request.get("tools") or []),
            flow_context=DataFlowContext.from_dict(
                dict(prepared_request.get("flow_context") or {})
            ),
            resumed_release=True,
        )

    async def _prepare_fresh_llm_request(
        self,
        state: _LLMCallState,
        *,
        response_scope_fingerprint: str | None,
        force_stateless: bool,
    ) -> None:
        profile_snapshot = self._llms.profile_snapshot(state.profile_id)
        precheck_sink = DataSink(
            f"llm:{state.profile_id}",
            profile_snapshot.identity_sha256,
        )
        resolved = self._llms.resolve(
            state.profile_id,
            snapshot=profile_snapshot,
        )
        state.resolved = resolved
        state.client = resolved.client
        state.sink = DataSink(
            f"llm:{resolved.profile_id}",
            resolved.identity_sha256,
        )
        if state.sink != precheck_sink:
            raise _LLMProviderChainScopeChanged(
                "LLM profile Sink changed while assembling the request"
            )
        self._set_llm_provider_scope(state)
        # This executor currently rebuilds and sends the complete cumulative
        # runtime snapshot for every quantum.  Combining that snapshot with a
        # provider-side ``previous_response_id`` replays the same history by
        # two channels, grows requests quadratically, and can duplicate tool
        # outputs.  Keep the low-level LLMClient chaining capability available
        # to callers that send deltas, but keep this full-snapshot executor
        # stateless until it has an explicit delta protocol.
        state.previous_response_id, previous_outputs = None, []
        state.request_messages = self._messages_with_tool_outputs(
            state.request_messages,
            previous_outputs,
        )
        image = self._images.get(state.process.image_id)
        if image is None:
            raise RuntimeError(
                f"agent image disappeared while assembling the LLM request: {state.process.image_id}"
            )
        state.parallel_tool_calls = bool(resolved.parallel_tool_calls)
        state.auto_wait_on_empty_tool_calls = bool(
            resolved.auto_wait_on_empty_tool_calls
        )
        # A compatibility JSON protocol would be runtime-authored prompt text,
        # so it is deliberately unavailable at the transparent boundary.
        state.fallback_json_actions = bool(
            resolved.fallback_json_actions
            and image.prompt_mode != PROMPT_MODE_IMAGE_ONLY
        )
        state.temperature = resolved.temperature
        state.max_tokens = resolved.max_tokens
        self._update_llm_request_options(
            state,
            response_scope_fingerprint=response_scope_fingerprint,
            previous_output_count=len(previous_outputs),
        )
        await self._apply_context_management(state, image=image)
        self._prepare_image_only_request_record(state, image=image)
        state.egress_payload = {
            "messages": state.request_messages,
            "tools": state.tools,
            "profile_id": resolved.profile_id,
            "previous_response_id": state.previous_response_id,
            "parallel_tool_calls": state.parallel_tool_calls,
        }
        # Context pressure and the hard per-call envelope are evaluated only
        # after the complete request has been assembled. Prompt-mode notices
        # therefore participate in the exact payload admitted for dispatch.
        self._prepare_llm_budget_envelope(state)
        self._data_flow.precheck_egress_clearance(
            pid=state.pid,
            sink=precheck_sink,
            context=state.flow_context,
            payload=state.egress_payload,
        )
        state.canonical_args = {
            "call_id": state.call_id,
            "profile_id": resolved.profile_id,
            "sink_identity_sha256": state.sink.identity_sha256,
            "payload_sha256": hashlib.sha256(
                dumps(to_jsonable(state.egress_payload)).encode("utf-8")
            ).hexdigest(),
            "attempt": state.attempt,
            "resource_envelope_sha256": state.resource_envelope_sha256,
        }

    def _set_llm_provider_scope(self, state: _LLMCallState) -> None:
        assert state.sink is not None
        state.data_flow_chain_fingerprint = (
            self._data_flow_provider_chain_fingerprint(
                pid=state.pid,
                sink=state.sink,
                context=state.flow_context,
            )
        )
        state.source_refs_fingerprint = state.flow_context.source_refs_hash()
        state.provider_chain_fingerprint = (
            self._combined_provider_chain_fingerprint(
                state.client,
                state.data_flow_chain_fingerprint,
            )
        )

    def _update_llm_request_options(
        self,
        state: _LLMCallState,
        *,
        response_scope_fingerprint: str | None,
        previous_output_count: int,
    ) -> None:
        resolved = state.resolved
        client = state.client
        assert resolved is not None
        provider_capable = bool(
            self.config.llm.persist_full_io
            and isinstance(client, LLMClient)
            and client.responses_previous_response_id
            and client.store
            and client._use_responses_api()
            and client._use_openai_request_options()
            and state.provider_chain_fingerprint is not None
        )
        response_chain_configured = bool(
            isinstance(client, LLMClient)
            and client.responses_previous_response_id
        )
        state.request_options.update(
            {
                "llm_profile_id": resolved.profile_id,
                "llm_context_generation": self._processes.get_llm_context_generation(
                    state.pid
                ),
                "client_class": type(client).__name__,
                "real_llm_client": isinstance(client, LLMClient),
                "openai_tool_schema": self._tool_schema_observation(state.tools),
                "openai_responses_previous_response_id_configured": (
                    response_chain_configured
                ),
                "openai_responses_previous_response_id_enabled": False,
                "openai_responses_previous_response_id_disabled_reason": (
                    _FULL_SNAPSHOT_RESPONSE_CHAIN_DISABLED_REASON
                    if response_chain_configured
                    else None
                ),
                "openai_provider_chain_capable": provider_capable,
                "openai_provider_chain_eligible": False,
                "openai_previous_response_id": state.previous_response_id,
                "openai_previous_response_tool_output_count": previous_output_count,
                "openai_response_scope_fingerprint": response_scope_fingerprint,
                "openai_provider_chain_fingerprint": state.provider_chain_fingerprint,
                "data_flow_provider_chain_fingerprint": state.data_flow_chain_fingerprint,
                "data_flow_provider_source_refs_sha256": state.source_refs_fingerprint,
                "openai_prompt_cache_key_configured": bool(
                    isinstance(client, LLMClient) and client.prompt_cache_key
                ),
                "openai_prompt_cache_retention_configured": (
                    client.prompt_cache_retention
                    if isinstance(client, LLMClient)
                    else None
                ),
                "openai_prompt_cache_key_sent": None,
                "openai_prompt_cache_options_sent": None,
                "openai_prompt_cache_retention": None,
                "openai_safety_identifier_configured": bool(
                    isinstance(client, LLMClient) and client.safety_identifier
                ),
                "openai_safety_identifier_sent": None,
                "openai_compatibility_removed_options": [],
                "openai_parallel_tool_calls_enabled": state.parallel_tool_calls,
                "agent_libos_auto_wait_on_empty_tool_calls_enabled": (
                    state.auto_wait_on_empty_tool_calls
                ),
                "fallback_json_actions_enabled": state.fallback_json_actions,
                "fallback_json_action_used": False,
            }
        )

    async def _apply_context_management(self, state: _LLMCallState, *, image: Any) -> None:
        resolved = state.resolved
        assert resolved is not None
        policy = context_management_policy(image.planner)
        generation = str(state.request_options["llm_context_generation"])
        latest_call = self._processes.get_latest_llm_call(
            pid=state.pid,
            purpose="action_selection",
        )
        lower_bound = provider_usage_lower_bound(
            latest_call,
            profile_id=resolved.profile_id,
            context_generation=generation,
            previous_response_id=state.previous_response_id,
        )
        assessment = assess_context_pressure(
            messages=state.request_messages,
            tools=state.tools,
            context_window_tokens=resolved.context_window_tokens,
            reserved_output_tokens=resolved.max_tokens,
            threshold_ratio=policy.threshold_ratio,
            profile_id=resolved.profile_id,
            context_generation=generation,
            provider_lower_bound_tokens=lower_bound,
        )
        prior = self._prior_context_pressure_state(
            state.pid,
            latest_call=latest_call,
            policy_fingerprint=policy.fingerprint,
        )
        episode_id = (
            str(prior.get("episode_id") or "")
            if prior.get("active") is True
            else ""
        ) or new_id("ctxpressure")
        attempted = bool(prior.get("auto_attempted")) if prior.get("active") is True else False
        observation = {
            **assessment.to_dict(),
            "source": "runtime_context_management",
            "active": assessment.triggered,
            "episode_id": episode_id,
            "policy_fingerprint": policy.fingerprint,
            "mode": policy.mode,
            "auto_attempted": attempted,
            "action": "none",
        }
        state.request_options["context_pressure"] = observation

        if not assessment.triggered:
            if prior.get("active") is True:
                observation["action"] = "recovered"
                self._audit_context_pressure(
                    state.pid,
                    "llm.context_pressure_recovered",
                    assessment,
                    policy,
                    episode_id=episode_id,
                )
            return

        self._audit_context_pressure(
            state.pid,
            "llm.context_pressure_detected",
            assessment,
            policy,
            episode_id=episode_id,
        )
        if policy.mode == "disabled":
            observation["action"] = "disabled"
            return
        if policy.mode == "prompt":
            self._apply_context_pressure_prompt(
                state,
                policy=policy,
                base_assessment=assessment,
                provider_lower_bound_tokens=lower_bound,
                episode_id=episode_id,
                observation=observation,
            )
            return
        persistent_context_enabled = self.context_memory.persistent_context_enabled(
            state.pid
        )
        maintenance_authorized = self._capabilities.check(
            state.pid,
            LLM_CONTEXT_MAINTENANCE_RESOURCE,
            CapabilityRight.EXECUTE,
        )
        if not persistent_context_enabled or not maintenance_authorized:
            observation["action"] = "not_authorized"
            self._audit_context_pressure(
                state.pid,
                "llm.context_pressure_maintenance_not_authorized",
                assessment,
                policy,
                episode_id=episode_id,
                extra={
                    "persistent_context_enabled": persistent_context_enabled,
                    "maintenance_authorized": maintenance_authorized,
                },
            )
            return
        if attempted:
            observation["action"] = "deduplicated"
            return
        try:
            # The earlier check is a read-only planning predicate. Consume a
            # finite maintenance lease only when this pressure episode will
            # actually attempt the compaction effect.
            self._capabilities.require(
                state.pid,
                LLM_CONTEXT_MAINTENANCE_RESOURCE,
                CapabilityRight.EXECUTE,
            )
        except CapabilityDenied:
            observation["action"] = "not_authorized"
            self._audit_context_pressure(
                state.pid,
                "llm.context_pressure_maintenance_not_authorized",
                assessment,
                policy,
                episode_id=episode_id,
                extra={
                    "persistent_context_enabled": persistent_context_enabled,
                    "maintenance_authorized": False,
                },
            )
            return
        await self._attempt_auto_context_management(
            state,
            policy=policy,
            assessment=assessment,
            episode_id=episode_id,
            observation=observation,
        )

    def _apply_context_pressure_prompt(
        self,
        state: _LLMCallState,
        *,
        policy: ContextManagementPolicy,
        base_assessment: ContextPressureAssessment,
        provider_lower_bound_tokens: int,
        episode_id: str,
        observation: dict[str, Any],
    ) -> None:
        notice, prompted_messages, assessment = self._pressure_prompt_request(
            state,
            policy=policy,
            base_assessment=base_assessment,
            provider_lower_bound_tokens=provider_lower_bound_tokens,
        )
        notice_tokens = max(
            0,
            assessment.local_input_estimate_tokens
            - base_assessment.local_input_estimate_tokens,
        )
        observation.update(
            {
                **assessment.to_dict(),
                "active": assessment.triggered,
                "action": "prompted",
                "prompt_notice_estimate_tokens": notice_tokens,
            }
        )
        if assessment.projected_tokens > assessment.context_window_tokens:
            observation.update(
                {
                    "action": "failed",
                    "failure_reason": "prompt_notice_exceeds_context_window",
                }
            )
            self._audit_context_pressure(
                state.pid,
                "llm.context_pressure_failed",
                assessment,
                policy,
                episode_id=episode_id,
                extra={
                    "reason": "prompt_notice_exceeds_context_window",
                    "prompt_notice_estimate_tokens": notice_tokens,
                },
            )
            raise ResourceLimitExceeded(
                "context-pressure prompt would exceed the configured LLM "
                "context window: "
                f"projected_tokens={assessment.projected_tokens}, "
                f"context_window_tokens={assessment.context_window_tokens}"
            )
        state.request_messages = prompted_messages
        self._audit_context_pressure(
            state.pid,
            "llm.context_pressure_prompted",
            assessment,
            policy,
            episode_id=episode_id,
            extra={
                "prompt_notice_estimate_tokens": notice_tokens,
                "prompt_notice_sha256": hashlib.sha256(
                    notice.encode("utf-8")
                ).hexdigest(),
            },
        )

    async def _attempt_auto_context_management(
        self,
        state: _LLMCallState,
        *,
        policy: ContextManagementPolicy,
        assessment: ContextPressureAssessment,
        episode_id: str,
        observation: dict[str, Any],
    ) -> None:
        observation.update({"auto_attempted": True, "action": "auto_attempted"})
        action = policy.tool_action()
        pending_context = self._auto_context_pending_metadata(
            policy,
            assessment,
            episode_id=episode_id,
        )
        try:
            self._persist_completed_context_management_marker(
                state.pid,
                action=action,
                metadata={**pending_context, "outcome": "attempted"},
            )
        except Exception as exc:
            public_error, internal_error = self._llm_error_artifacts(
                exc,
                code="llm_context_management_error",
            )
            observation.update(
                {"action": "failed", "failure_reason": type(exc).__name__}
            )
            self._audit_context_pressure(
                state.pid,
                "llm.context_pressure_failed",
                assessment,
                policy,
                episode_id=episode_id,
                extra={
                    "reason": "attempt_marker_persistence_failed",
                    "error_type": type(exc).__name__,
                    "error": public_error["message"],
                    "error_details": public_error,
                    "internal_error": internal_error,
                },
                correlation_id=public_error["correlation_id"],
            )
            return
        self._audit_context_pressure(
            state.pid,
            "llm.context_pressure_auto_attempted",
            assessment,
            policy,
            episode_id=episode_id,
            extra={"tool_name": policy.tool_name},
        )
        try:
            result = await self._dispatch_auto_context_action(
                state.pid,
                action=action,
                pending_context=pending_context,
            )
        except _ContextManagementHandled:
            raise
        except Exception as exc:
            public_error, internal_error = self._llm_error_artifacts(
                exc,
                code="llm_context_management_error",
            )
            generation_after = self._processes.get_llm_context_generation(
                state.pid
            )
            observation.update(
                {"action": "failed", "failure_reason": type(exc).__name__}
            )
            self._audit_context_pressure(
                state.pid,
                "llm.context_pressure_failed",
                assessment,
                policy,
                episode_id=episode_id,
                extra={
                    "reason": type(exc).__name__,
                    "error": public_error["message"],
                    "error_details": public_error,
                    "internal_error": internal_error,
                    "context_generation_after": generation_after,
                },
                correlation_id=public_error["correlation_id"],
            )
            if generation_after != assessment.context_generation:
                raise _ContextManagementHandled(
                    self._context_generation_changed_result(
                        episode_id=episode_id,
                        generation_after=generation_after,
                    )
                ) from exc
            return
        self._finish_auto_context_action(
            state.pid,
            action=action,
            result=result,
            policy=policy,
            assessment=assessment,
            episode_id=episode_id,
            pending_context=pending_context,
            observation=observation,
        )

    @staticmethod
    def _auto_context_pending_metadata(
        policy: ContextManagementPolicy,
        assessment: ContextPressureAssessment,
        *,
        episode_id: str,
    ) -> dict[str, Any]:
        return {
            "kind": "context_management_auto",
            "schema_version": 1,
            "source": "runtime_context_management",
            "episode_id": episode_id,
            "policy_fingerprint": policy.fingerprint,
            "mode": policy.mode,
            "tool_name": policy.tool_name,
            "context_generation": assessment.context_generation,
            "assessment": assessment.to_dict(),
        }

    async def _dispatch_auto_context_action(
        self,
        pid: str,
        *,
        action: dict[str, Any],
        pending_context: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return await self.adispatch(
                pid,
                action,
                context_metadata={
                    "context_management_auto": True,
                    "context_pressure_episode_id": pending_context["episode_id"],
                    "context_pressure_policy_fingerprint": pending_context[
                        "policy_fingerprint"
                    ],
                },
            )
        except HumanApprovalRequired as exc:
            payload = self._wait_for_human_action(
                pid=pid,
                action=action,
                request_id=exc.request_id,
                message=str(exc),
                content_preview="",
                tool_call_count=0,
                pending_metadata=pending_context,
            )
            raise _ContextManagementHandled(payload) from exc
        except ProcessWaitRequired as exc:
            payload = self._wait_for_child_action(
                pid=pid,
                action=exc.resume_action or action,
                child_pid=exc.child_pid,
                message=str(exc),
                content_preview="",
                tool_call_count=0,
                pending_metadata=pending_context,
            )
            raise _ContextManagementHandled(payload) from exc
        except ProcessMessageWaitRequired as exc:
            payload = self._wait_for_message_action(
                pid=pid,
                action=action,
                filters=exc.filters,
                message=str(exc),
                content_preview="",
                tool_call_count=0,
                pending_metadata=pending_context,
            )
            raise _ContextManagementHandled(payload) from exc

    def _finish_auto_context_action(
        self,
        pid: str,
        *,
        action: dict[str, Any],
        result: dict[str, Any],
        policy: ContextManagementPolicy,
        assessment: ContextPressureAssessment,
        episode_id: str,
        pending_context: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        generation_after = self._processes.get_llm_context_generation(pid)
        if self._context_compaction_succeeded(
            result,
            generation_before=assessment.context_generation,
            generation_after=generation_after,
        ):
            self._persist_completed_context_management_marker(
                pid,
                action=action,
                metadata={
                    **pending_context,
                    "context_generation_after": generation_after,
                    "outcome": "compacted",
                },
            )
            self._audit_context_pressure(
                pid,
                "llm.context_pressure_compacted",
                assessment,
                policy,
                episode_id=episode_id,
                extra={
                    "tool_name": policy.tool_name,
                    "context_generation_after": generation_after,
                },
            )
            raise _ContextManagementHandled(
                {
                    "ok": True,
                    "context_compacted": True,
                    "context_pressure_episode_id": episode_id,
                    "context_generation": generation_after,
                }
            )
        reason = self._context_management_failure_reason(
            result,
            generation_before=assessment.context_generation,
            generation_after=generation_after,
        )
        observation.update({"action": "failed", "failure_reason": reason})
        self._audit_context_pressure(
            pid,
            "llm.context_pressure_failed",
            assessment,
            policy,
            episode_id=episode_id,
            extra={
                "reason": reason,
                "result": sanitize_for_observability(result),
                "context_generation_after": generation_after,
            },
        )
        if generation_after != assessment.context_generation:
            raise _ContextManagementHandled(
                self._context_generation_changed_result(
                    episode_id=episode_id,
                    generation_after=generation_after,
                )
            )

    @staticmethod
    def _context_generation_changed_result(
        *,
        episode_id: str,
        generation_after: str,
    ) -> dict[str, Any]:
        """End the quantum so the next request rematerializes fresh context."""

        return {
            "ok": True,
            "context_generation_changed": True,
            "context_management_failed": True,
            "context_pressure_episode_id": episode_id,
            "context_generation": generation_after,
        }

    def _prior_context_pressure_state(
        self,
        pid: str,
        *,
        latest_call: Any | None,
        policy_fingerprint: str,
    ) -> dict[str, Any]:
        durable = self.pending.get(pid) or {}
        metadata = pending_metadata(durable)
        durable_action = dict(durable.get("action") or {})
        durable_pressure: Any = durable_action.get("context_pressure")
        if durable_action.get("kind") == "llm_release_request":
            durable_pressure = dict(
                durable_action.get("request_options") or {}
            ).get("context_pressure")
        durable_is_newer = str(durable.get("updated_at") or "") >= str(
            getattr(latest_call, "completed_at", "") or ""
        )
        if (
            durable_is_newer
            and isinstance(durable_pressure, dict)
            and durable_pressure.get("policy_fingerprint") == policy_fingerprint
        ):
            return dict(durable_pressure)
        if (
            durable_is_newer
            and metadata.get("kind") == "context_management_auto"
            and metadata.get("policy_fingerprint") == policy_fingerprint
        ):
            return {
                "active": True,
                "episode_id": metadata.get("episode_id"),
                "auto_attempted": True,
                "policy_fingerprint": policy_fingerprint,
            }
        if latest_call is not None:
            options = getattr(latest_call, "request_options", {})
            if isinstance(options, dict) and "context_pressure" in options:
                pressure = options.get("context_pressure")
                if (
                    isinstance(pressure, dict)
                    and pressure.get("policy_fingerprint") == policy_fingerprint
                ):
                    return dict(pressure)
                return {}
        if (
            metadata.get("kind") == "context_management_auto"
            and metadata.get("policy_fingerprint") == policy_fingerprint
        ):
            return {
                "active": True,
                "episode_id": metadata.get("episode_id"),
                "auto_attempted": True,
                "policy_fingerprint": policy_fingerprint,
            }
        return {}

    @staticmethod
    def _append_context_pressure_notice(
        messages: list[dict[str, Any]],
        notice: str,
    ) -> None:
        for message in reversed(messages):
            if str(message.get("role") or "") != "user":
                continue
            content = str(message.get("content") or "")
            message["content"] = "\n\n".join(part for part in (content, notice) if part)
            return
        messages.append({"role": "user", "content": notice})

    @classmethod
    def _pressure_prompt_request(
        cls,
        state: _LLMCallState,
        *,
        policy: ContextManagementPolicy,
        base_assessment: ContextPressureAssessment,
        provider_lower_bound_tokens: int,
    ) -> tuple[str, list[dict[str, Any]], ContextPressureAssessment]:
        """Build and assess the exact prompt-mode request.

        The diagnostic notice includes numeric values from the assessment, so
        its own serialized size can change those values at a digit boundary.
        Iterate to a stable notice while always returning an assessment of the
        exact messages that would be sent.
        """

        resolved = state.resolved
        assert resolved is not None
        notice_assessment = base_assessment
        notice = ""
        candidate_messages: list[dict[str, Any]] = []
        candidate_assessment = base_assessment
        for _ in range(8):
            notice = context_pressure_prompt(policy, notice_assessment)
            candidate_messages = [dict(message) for message in state.request_messages]
            cls._append_context_pressure_notice(candidate_messages, notice)
            candidate_assessment = assess_context_pressure(
                messages=candidate_messages,
                tools=state.tools,
                context_window_tokens=resolved.context_window_tokens,
                reserved_output_tokens=resolved.max_tokens,
                threshold_ratio=policy.threshold_ratio,
                profile_id=resolved.profile_id,
                context_generation=base_assessment.context_generation,
                provider_lower_bound_tokens=provider_lower_bound_tokens,
            )
            if context_pressure_prompt(policy, candidate_assessment) == notice:
                break
            notice_assessment = candidate_assessment
        return notice, candidate_messages, candidate_assessment

    @staticmethod
    def _context_compaction_succeeded(
        result: dict[str, Any],
        *,
        generation_before: str,
        generation_after: str,
    ) -> bool:
        payload = result.get("payload")
        return bool(
            result.get("ok") is True
            and isinstance(payload, dict)
            and payload.get("compacted") is True
            and generation_after != generation_before
        )

    @staticmethod
    def _context_management_failure_reason(
        result: dict[str, Any],
        *,
        generation_before: str,
        generation_after: str,
    ) -> str:
        if result.get("ok") is not True:
            return "tool_failed"
        payload = result.get("payload")
        if not isinstance(payload, dict) or payload.get("compacted") is not True:
            return "invalid_compaction_result"
        if generation_after == generation_before:
            return "context_generation_unchanged"
        return "unknown"

    def _persist_completed_context_management_marker(
        self,
        pid: str,
        *,
        action: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        self._persist_pending_action(
            pid,
            wait_type="context_management",
            action=action,
            content_preview="",
            tool_call_count=0,
            pending_metadata=metadata,
            status="completed",
        )

    def _audit_context_pressure(
        self,
        pid: str,
        action: str,
        assessment: ContextPressureAssessment,
        policy: ContextManagementPolicy,
        *,
        episode_id: str,
        extra: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        self._audit.record(
            actor=pid,
            action=action,
            target=f"process:{pid}",
            decision={
                **assessment.to_dict(),
                "episode_id": episode_id,
                "policy_fingerprint": policy.fingerprint,
                "mode": policy.mode,
                **dict(extra or {}),
            },
            correlation_id=correlation_id,
        )

    def _prepare_resumed_llm_request(
        self,
        state: _LLMCallState,
        prepared_request: dict[str, Any],
    ) -> None:
        resolved = self._llms.resolve(state.profile_id)
        state.resolved = resolved
        state.client = resolved.client
        sink_data = dict(prepared_request.get("sink") or {})
        state.sink = DataSink(
            identity=str(sink_data["identity"]),
            identity_sha256=sink_data.get("identity_sha256"),
            trust_identity=sink_data.get("trust_identity"),
            trust_identity_sha256=sink_data.get("trust_identity_sha256"),
        )
        current_sink = DataSink(
            f"llm:{resolved.profile_id}",
            resolved.identity_sha256,
        )
        if current_sink != state.sink:
            raise _LLMProviderChainScopeChanged(
                "LLM profile Sink changed while data release was pending"
            )
        state.data_flow_chain_fingerprint = str(
            prepared_request["data_flow_chain_fingerprint"]
        )
        state.source_refs_fingerprint = str(
            prepared_request["source_refs_fingerprint"]
        )
        prepared_provider = prepared_request.get("provider_chain_fingerprint")
        state.provider_chain_fingerprint = (
            str(prepared_provider) if prepared_provider is not None else None
        )
        prepared_previous_response_id = prepared_request.get("previous_response_id")
        if prepared_previous_response_id is not None:
            raise _LLMProviderChainScopeChanged(
                "provider-side Responses state cannot be resumed by the "
                "full-snapshot executor"
            )
        state.previous_response_id = None
        state.parallel_tool_calls = bool(prepared_request["parallel_tool_calls"])
        state.auto_wait_on_empty_tool_calls = bool(
            prepared_request["auto_wait_on_empty_tool_calls"]
        )
        state.fallback_json_actions = bool(
            prepared_request.get("fallback_json_actions", False)
        )
        state.temperature = float(prepared_request["temperature"])
        state.max_tokens = int(prepared_request["max_tokens"])
        state.egress_payload = dict(prepared_request.get("egress_payload") or {})
        state.canonical_args = dict(
            prepared_request.get("canonical_args") or {}
        )
        self._rebind_resumed_image_only_generation(state)
        self._prepare_llm_budget_envelope(state)
        state.canonical_args.update(
            {
                "call_id": state.call_id,
                "profile_id": resolved.profile_id,
                "attempt": state.attempt,
                "resource_envelope_sha256": state.resource_envelope_sha256,
            }
        )

    def _rebind_resumed_image_only_generation(self, state: _LLMCallState) -> None:
        image = self._images.get(state.process.image_id)
        if image is None or image.prompt_mode != PROMPT_MODE_IMAGE_ONLY:
            return
        anchor = self._frozen_image_only_anchor(state, image=image)
        prepared_generation = state.request_options.get("llm_context_generation")
        if prepared_generation != anchor["llm_context_generation"]:
            raise ValidationError(
                "image_only resumed request generation does not match its frozen anchor"
            )
        current_generation = self._processes.get_llm_context_generation(state.pid)
        if current_generation == prepared_generation:
            return

        rebound_anchor = {
            **anchor,
            "llm_context_generation": current_generation,
        }
        rebound_purpose = self._image_only_request_purpose(rebound_anchor)
        state.request_options["llm_context_generation"] = current_generation
        state.request_options[_IMAGE_ONLY_FROZEN_ANCHOR_KEY] = {
            "schema_version": _IMAGE_ONLY_FROZEN_ANCHOR_SCHEMA_VERSION,
            **rebound_anchor,
            "purpose": rebound_purpose,
        }
        request_marker = state.request_options.get(_IMAGE_ONLY_REQUEST_KEY)
        if request_marker is not None:
            if not isinstance(request_marker, dict):
                raise ValidationError("image_only resumed request anchor marker is invalid")
            if request_marker.get("schema_version") != _IMAGE_ONLY_REQUEST_SCHEMA_VERSION:
                raise ValidationError(
                    "image_only resumed request anchor has an unsupported schema"
                )
            for key, expected in anchor.items():
                if request_marker.get(key) != expected:
                    raise ValidationError(
                        f"image_only resumed request anchor changed {key}"
                    )
            if request_marker.get("purpose") != self._image_only_request_purpose(anchor):
                raise ValidationError("image_only resumed request anchor purpose changed")
            state.request_options[_IMAGE_ONLY_REQUEST_KEY] = {
                **request_marker,
                **rebound_anchor,
                "purpose": rebound_purpose,
            }
        self._audit.record(
            actor=state.pid,
            action="llm.image_only_release_generation_rebound",
            target=f"llm_call:{state.call_id}",
            decision={
                "previous_generation": prepared_generation,
                "current_generation": current_generation,
                "request_payload_changed": False,
            },
        )

    async def _invoke_prepared_llm_request(self, state: _LLMCallState) -> Any:
        assert state.resolved is not None and state.sink is not None
        if state.resource_envelope is None:
            raise RuntimeError("LLM resource envelope was not prepared")
        invocation = ProtectedOperationInvocation(
            pid=state.pid,
            actor=state.pid,
            target=state.sink.identity,
            canonical_args=state.canonical_args,
            observation={
                **state.canonical_args,
                **self._llm_resource_context(state),
                "message_count": len(state.request_messages),
                "tool_count": len(state.tools),
                "source_count": len(state.flow_context.source_refs),
            },
            data_sink=state.sink,
            data_flow_context=state.flow_context,
            data_flow_ingress_context=self._data_flow.unclassified_ingress_context(
                state.flow_context,
                origin="external:llm",
            ),
            data_flow_payload=state.egress_payload,
            data_flow_operation="llm.complete",
            data_flow_allow_recovered_source_snapshots=state.resumed_release,
            reservation_usage=state.resource_envelope,
            resource_source="llm.request",
            resource_context=self._llm_resource_context(state),
            prepare=lambda: self._assert_llm_call_scope(state),
            failure_evidence=lambda error, phase: self._llm_failure_evidence(
                state,
                error,
                phase,
            ),
            post_commit_result_observer=lambda result, observation: (
                self._bind_semantic_provider_observation(
                    state,
                    result,
                    observation,
                )
            ),
        )
        with self._protected_operations.start(
            "primitive.llm.complete",
            invocation,
            provider=state.client,
        ) as protected:

            async def dispatch_bound_request() -> Any:
                self._assert_llm_call_scope(state)
                state.provider_dispatched = True
                return await self._complete_action(
                    state.client,
                    state.request_messages,
                    state.tools,
                    temperature=state.temperature,
                    max_tokens=state.max_tokens,
                    previous_response_id=state.previous_response_id,
                    parallel_tool_calls=state.parallel_tool_calls,
                )

            completion = await protected.acall(
                ProviderPhase(
                    "provider_request",
                    state_mutation=True,
                    information_flow=True,
                ),
                dispatch_bound_request,
            )
            state.provider_trace = self._provider_trace_for_completion(
                state,
                completion,
            )
            self._record_provider_trace_summary(state)
            resource, violation = self._llm_completion_resource_settlement(
                state,
                completion,
            )
            result = protected.complete(
                completion,
                self._llm_success_evidence(state, completion),
                classification_override=ExternalEffectClassification(
                    rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                    rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                    state_mutation=True,
                    information_flow=True,
                    metadata={"outcome": "provider_completed"},
                ),
                resource=resource,
            )
            if violation is not None:
                raise ResourceLimitExceeded(violation)
            return result

    @staticmethod
    def _provider_trace_for_completion(
        state: _LLMCallState,
        completion: Any,
    ) -> dict[str, Any]:
        if type(state.client) is LLMClient:
            trace = getattr(completion, "provider_trace", None)
            if is_provider_trace(trace):
                return trace
        return custom_provider_trace(completion)

    @staticmethod
    def _record_provider_trace_summary(state: _LLMCallState) -> None:
        if state.provider_trace is None:
            state.request_options.pop("provider_trace_summary", None)
            return
        state.request_options["provider_trace_summary"] = provider_trace_summary(
            state.provider_trace
        )

    def _assert_llm_call_scope(self, state: _LLMCallState) -> None:
        assert state.resolved is not None and state.sink is not None
        if state.previous_response_id is not None:
            raise _LLMProviderChainScopeChanged(
                "provider-side Responses state is disabled for full-snapshot "
                "executor requests"
            )
        self._assert_llm_provider_chain_scope(
            pid=state.pid,
            profile_id=state.resolved.profile_id,
            context=state.flow_context,
            expected_sink=state.sink,
            expected_data_flow_fingerprint=state.data_flow_chain_fingerprint,
            expected_provider_fingerprint=state.provider_chain_fingerprint,
            expected_source_refs_fingerprint=state.source_refs_fingerprint,
        )

    def _llm_failure_evidence(
        self,
        state: _LLMCallState,
        error: BaseException,
        phase: str,
    ) -> ProtectedOperationEvidence:
        assert state.resolved is not None and state.sink is not None
        return ProtectedOperationEvidence(
            event_type=EventType.EXTERNAL_WRITE,
            event_source=state.pid,
            event_target=state.sink.identity,
            event_payload={
                "adapter": "llm",
                "profile_id": state.resolved.profile_id,
                "outcome": "unknown",
                "phase": phase,
            },
            audit_action="primitive.llm.complete.failed",
            audit_actor=state.pid,
            audit_target=state.sink.identity,
            audit_decision={
                **state.canonical_args,
                "error_type": type(error).__name__,
                "phase": phase,
                "effect_outcome": "unknown",
            },
            input_refs=tuple(item.oid for item in state.flow_context.source_refs),
        )

    def _llm_success_evidence(
        self,
        state: _LLMCallState,
        completion: Any,
    ) -> ProtectedOperationEvidence:
        assert state.resolved is not None and state.sink is not None
        request_id = getattr(completion, "request_id", None)
        response_id = getattr(completion, "response_id", None)
        return ProtectedOperationEvidence(
            event_type=EventType.EXTERNAL_WRITE,
            event_source=state.pid,
            event_target=state.sink.identity,
            event_payload={
                "adapter": "llm",
                "profile_id": state.resolved.profile_id,
                "status": "ok",
                "request_id": request_id,
                "response_id": response_id,
            },
            audit_action="primitive.llm.complete",
            audit_actor=state.pid,
            audit_target=state.sink.identity,
            audit_decision={
                **state.canonical_args,
                "status": "ok",
                "request_id": request_id,
                "response_id": response_id,
            },
            input_refs=tuple(item.oid for item in state.flow_context.source_refs),
            provider_receipt={
                "request_id": request_id,
                "response_id": response_id,
            },
        )

    def _build_llm_release_request(
        self,
        state: _LLMCallState,
        *,
        previous: dict[str, Any] | None,
        response_scope_fingerprint: str | None,
    ) -> dict[str, Any]:
        assert state.resolved is not None and state.sink is not None
        prepared = dict(previous or {})
        prepared.update(
            {
                "kind": "llm_release_request",
                "schema_version": 1,
                "pid": state.pid,
                "call_id": state.call_id,
                "created_at": state.created_at,
                "profile_id": state.resolved.profile_id,
                "request_messages": list(state.request_messages),
                "tools": list(state.tools),
                "request_options": dict(state.request_options),
                "sink": {
                    "identity": state.sink.identity,
                    "identity_sha256": state.sink.identity_sha256,
                    "trust_identity": state.sink.trust_identity,
                    "trust_identity_sha256": state.sink.trust_identity_sha256,
                },
                "flow_context": state.flow_context.to_dict(),
                "data_flow_chain_fingerprint": state.data_flow_chain_fingerprint,
                "source_refs_fingerprint": state.source_refs_fingerprint,
                "provider_chain_fingerprint": state.provider_chain_fingerprint,
                "previous_response_id": state.previous_response_id,
                "parallel_tool_calls": state.parallel_tool_calls,
                "auto_wait_on_empty_tool_calls": state.auto_wait_on_empty_tool_calls,
                "fallback_json_actions": state.fallback_json_actions,
                "temperature": state.temperature,
                "max_tokens": state.max_tokens,
                "egress_payload": state.egress_payload,
                "canonical_args": state.canonical_args,
                "attempt": state.attempt,
                "max_attempts": state.max_attempts,
                "response_scope_fingerprint": response_scope_fingerprint,
            }
        )
        return prepared

    def _record_llm_call_error(
        self,
        state: _LLMCallState,
        error: Exception,
    ) -> None:
        if state.provider_trace is None:
            if type(state.client) is LLMClient:
                state.provider_trace = provider_trace_from_error(error)
            if state.provider_trace is None:
                state.provider_trace = custom_provider_trace(error=error)
        self._record_provider_trace_summary(state)
        preserve_domain_text = self._preserve_llm_domain_error_text(error)
        public_error: dict[str, str] | None = None
        internal_error: dict[str, Any] | None = None
        selected_error = str(error)
        if not preserve_domain_text:
            public_error, internal_error = self._llm_error_artifacts(error)
            selected_error = public_error["message"]
        observable = observable_llm_call_fields(
            messages=state.request_messages,
            tools=state.tools,
            response_content="",
            tool_calls=[],
            reasoning=state.provider_trace,
            raw_response=None,
            error=selected_error,
            config=self.config,
        )
        observability = dict(observable["observability"])
        if public_error is not None and internal_error is not None:
            observability["failure"] = {
                "public_error": public_error,
                "internal_error": internal_error,
            }
        image = self._images.get(state.process.image_id)
        request_marker = state.request_options.get(_IMAGE_ONLY_REQUEST_KEY)
        request_purpose = (
            str(request_marker.get("purpose") or "")
            if isinstance(request_marker, dict)
            else ""
        )
        purpose = "action_selection"
        if image is not None and image.prompt_mode == PROMPT_MODE_IMAGE_ONLY:
            purpose = (
                request_purpose
                if request_purpose.startswith(
                    f"{_IMAGE_ONLY_REQUEST_PURPOSE_PREFIX}:"
                )
                else _IMAGE_ONLY_ERROR_PURPOSE
            )
        self._processes.insert_llm_call(
            LLMCallRecord(
                call_id=state.call_id,
                pid=state.pid,
                image_id=state.process.image_id,
                purpose=purpose,
                status="error",
                messages=observable["messages"],
                tools=observable["tools"],
                request_options=state.request_options,
                response_content=observable["response_content"],
                tool_calls=observable["tool_calls"],
                reasoning=observable["reasoning"],
                usage=dict(state.completion_usage),
                raw_response=observable["raw_response"],
                observability=observability,
                error=observable["error"],
                created_at=state.created_at,
                completed_at=utc_now(),
            )
        )
        self._operations.link_evidence(
            "llm_call",
            state.call_id,
            "invocation",
            metadata={"attempt": state.attempt, "status": "error"},
        )

    def _record_llm_call_success(
        self,
        state: _LLMCallState,
        completion: Any,
    ) -> tuple[Any, bool, bool, bool, str, str]:
        self._prepare_image_only_transcript_record(state, completion)
        usage = dict(state.completion_usage)
        invalid_usage_fields = set(state.invalid_completion_usage_fields)
        self._record_effective_provider_request_options(state, completion)
        if invalid_usage_fields:
            state.request_options["invalid_usage_fields"] = sorted(
                invalid_usage_fields
            )
        state.request_options["fallback_json_action_used"] = (
            self._fallback_json_action_was_used(state, completion)
        )
        if getattr(completion, "api", None) == "responses":
            manifest = self._response_tool_call_manifest(completion)
            if self.config.llm.persist_full_io:
                state.request_options["openai_response_tool_calls"] = manifest
            else:
                # The manifest carries provider-generated call ids and model-
                # selected tool names. It is needed only for durable, lossless
                # Responses chaining, which is disabled in content-free mode.
                state.request_options.pop("openai_response_tool_calls", None)
                state.request_options["openai_response_tool_call_count"] = len(
                    manifest
                )
        observable = observable_llm_call_fields(
            messages=state.request_messages,
            tools=state.tools,
            response_content=str(getattr(completion, "content", "")),
            tool_calls=list(getattr(completion, "tool_calls", []) or []),
            reasoning=state.provider_trace,
            raw_response=project_provider_raw_response(
                getattr(completion, "raw", None)
            ),
            config=self.config,
        )
        success_record = LLMCallRecord(
            call_id=state.call_id,
            pid=state.pid,
            image_id=state.process.image_id,
            purpose="action_selection",
            status="ok",
            api=getattr(completion, "api", None),
            model=getattr(completion, "model", None),
            request_id=getattr(completion, "request_id", None),
            response_id=getattr(completion, "response_id", None),
            messages=observable["messages"],
            tools=observable["tools"],
            request_options=state.request_options,
            response_content=observable["response_content"],
            tool_calls=observable["tool_calls"],
            reasoning=observable["reasoning"],
            usage=usage,
            raw_response=observable["raw_response"],
            observability=observable["observability"],
            error=observable["error"],
            created_at=state.created_at,
            completed_at=utc_now(),
        )
        self._processes.insert_llm_call(success_record)
        self._operations.link_evidence(
            "llm_call",
            state.call_id,
            "invocation",
            metadata={"attempt": state.attempt, "status": "ok"},
        )
        request_id = getattr(completion, "request_id", None)
        if request_id:
            self._operations.link_evidence(
                "llm_request",
                str(request_id),
                "invocation",
                metadata={"call_id": state.call_id},
            )
        self._observe_host_semantic_result(state, completion, success_record)
        return (
            completion,
            state.parallel_tool_calls,
            state.auto_wait_on_empty_tool_calls,
            state.fallback_json_actions,
            str(state.request_options["llm_profile_id"]),
            state.call_id,
        )

    def _supersede_image_only_request_anchor(
        self,
        *,
        pid: str,
        image_id: str | None,
        transcript_call_id: str,
        request_options: Mapping[str, Any],
    ) -> None:
        frozen = request_options.get(_IMAGE_ONLY_FROZEN_ANCHOR_KEY)
        if not isinstance(frozen, dict):
            return
        purpose = str(frozen.get("purpose") or "")
        if not purpose.startswith(f"{_IMAGE_ONLY_REQUEST_PURPOSE_PREFIX}:"):
            raise ValidationError("image_only frozen request anchor purpose changed")
        request = self._processes.get_latest_llm_call(
            pid=pid,
            purpose=purpose,
        )
        if (
            request is None
            or request.status != "error"
            or not isinstance(
                request.request_options.get(_IMAGE_ONLY_REQUEST_KEY),
                dict,
            )
        ):
            return
        now = utc_now()
        self._processes.insert_llm_call(
            LLMCallRecord(
                call_id=new_id("llmanchor"),
                pid=pid,
                image_id=image_id,
                purpose=purpose,
                status="ok",
                messages=[],
                tools=[],
                request_options={
                    "image_only_request_superseded": {
                        "schema_version": 1,
                        "request_call_id": request.call_id,
                        "transcript_call_id": transcript_call_id,
                    }
                },
                response_content="",
                tool_calls=[],
                observability={"kind": "image_only_request_superseded"},
                created_at=now,
                completed_at=now,
            )
        )

    def _try_supersede_image_only_request_anchor(
        self,
        *,
        pid: str,
        image_id: str | None,
        transcript_call_id: str,
        request_options: Mapping[str, Any],
    ) -> bool:
        try:
            self._supersede_image_only_request_anchor(
                pid=pid,
                image_id=image_id,
                transcript_call_id=transcript_call_id,
                request_options=request_options,
            )
        except Exception as exc:
            # The complete transcript remains canonical and the older request
            # anchor remains protected. Tombstone failure therefore delays
            # retention only; it must not roll back a correct tool output or
            # turn an already-terminal tool action into a second exit.
            try:
                self._audit.record(
                    actor=pid,
                    action="llm.image_only_request_anchor_supersede_failed",
                    target=f"llm_call:{transcript_call_id}",
                    decision={"error_type": type(exc).__name__},
                )
            except Exception:
                pass
            return False
        return True

    @staticmethod
    def _normalize_image_only_completion_tool_calls(
        state: _LLMCallState,
        completion: Any,
    ) -> None:
        tool_calls = list(getattr(completion, "tool_calls", []) or [])
        seen: set[str] = set()
        for ordinal, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            candidate = str(
                tool_call.get("call_id") or tool_call.get("id") or ""
            ).strip()
            if not candidate or candidate in seen:
                candidate = "call_" + hashlib.sha256(
                    f"{state.call_id}:{ordinal}".encode("utf-8")
                ).hexdigest()[:24]
            seen.add(candidate)
            tool_call["call_id"] = candidate
        completion.tool_calls = tool_calls

    def _previous_image_only_input_oids(
        self,
        *,
        state: _LLMCallState,
        anchor: Mapping[str, str],
    ) -> set[str]:
        input_oids: set[str] = set()
        previous = self._processes.get_latest_successful_llm_call(
            pid=state.pid,
            purpose="action_selection",
        )
        if previous is None:
            return input_oids
        previous_marker = self._image_only_marker_for_anchor(previous, anchor)
        if previous_marker is None:
            return input_oids
        previous_oids = previous_marker.get("input_oids")
        if isinstance(previous_oids, list):
            input_oids.update(
                oid for oid in previous_oids if isinstance(oid, str) and oid
            )
        output_key = previous_marker.get("output_key")
        if not isinstance(output_key, str) or not output_key:
            return input_oids
        rows = self._processes.list_llm_tool_outputs(
            pid=state.pid,
            response_id=output_key,
        )
        for row in rows:
            envelope = self._decode_image_only_tool_output(row.get("output_text"))
            result_oid = envelope.get("result_oid")
            if result_oid:
                input_oids.add(result_oid)
        return input_oids

    @staticmethod
    def _image_only_canonical_message_count(state: _LLMCallState) -> int:
        canonical_message_count = len(state.request_messages)
        if state.attempt <= 1:
            return canonical_message_count
        if canonical_message_count < 3:
            raise ValidationError("image_only action-repair request is malformed")
        if state.request_messages[-1].get("role") != "user":
            raise ValidationError("image_only action-repair request is malformed")
        return canonical_message_count - 1

    def _prepare_image_only_request_record(
        self,
        state: _LLMCallState,
        *,
        image: Any,
    ) -> None:
        if image.prompt_mode != PROMPT_MODE_IMAGE_ONLY:
            return
        if not self.config.llm.persist_full_io:
            raise ValidationError(
                "image_only requires llm.persist_full_io=true for durable request retry"
            )
        anchor = self._frozen_image_only_anchor(state, image=image)
        prepared_generation = state.request_options.get("llm_context_generation")
        if prepared_generation != anchor["llm_context_generation"]:
            raise ValidationError(
                "image_only request generation changed during request preparation"
            )
        purpose = self._image_only_request_purpose(anchor)
        canonical_message_count = self._image_only_canonical_message_count(state)
        if canonical_message_count != 2:
            # Once a complete transcript exists, that successful head and its
            # paired output rows—not a failed request snapshot—remain the
            # canonical recovery source.
            return
        previous_request, previous_marker = self._latest_image_only_request(
            pid=state.pid,
            image=image,
            anchor=anchor,
        )
        if previous_request is not None and previous_marker is not None:
            # Keep a single durable goal retry anchor per transcript anchor.
            # Later transient rows use the separate request purpose so they
            # cannot hide a successful transcript head, but need not duplicate
            # the same retained goal payload.
            return
        input_oids = {ref.oid for ref in state.flow_context.source_refs}
        if state.process.goal_oid:
            input_oids.add(str(state.process.goal_oid))
        state.request_options[_IMAGE_ONLY_REQUEST_KEY] = {
            "schema_version": _IMAGE_ONLY_REQUEST_SCHEMA_VERSION,
            **anchor,
            "purpose": purpose,
            "canonical_message_count": canonical_message_count,
            "labels": state.flow_context.labels.to_dict(),
            "input_oids": sorted(input_oids),
        }

    def _prepare_image_only_transcript_record(
        self,
        state: _LLMCallState,
        completion: Any,
    ) -> None:
        image = self._images.get(state.process.image_id)
        if image is None or image.prompt_mode != PROMPT_MODE_IMAGE_ONLY:
            return
        if not self.config.llm.persist_full_io:
            raise ValidationError(
                "image_only requires llm.persist_full_io=true for durable transcript replay"
            )
        self._normalize_image_only_completion_tool_calls(state, completion)
        anchor = self._frozen_image_only_anchor(state, image=image)
        input_oids = {ref.oid for ref in state.flow_context.source_refs}
        input_oids.update(
            self._previous_image_only_input_oids(
                state=state,
                anchor=anchor,
            )
        )
        if state.process.goal_oid:
            input_oids.add(str(state.process.goal_oid))
        manifest = self._response_tool_call_manifest(completion)
        marker: dict[str, Any] = {
            "schema_version": _IMAGE_ONLY_TRANSCRIPT_SCHEMA_VERSION,
            **anchor,
            "canonical_message_count": self._image_only_canonical_message_count(
                state
            ),
            "output_key": state.call_id,
            "tool_calls": manifest,
            "labels": state.flow_context.labels.to_dict(),
            "input_oids": sorted(input_oids),
        }
        state.request_options[_IMAGE_ONLY_TRANSCRIPT_KEY] = marker
        setattr(completion, "_agent_libos_transcript_output_key", state.call_id)

    @staticmethod
    def _record_effective_provider_request_options(
        state: _LLMCallState,
        completion: Any,
    ) -> None:
        """Persist only non-secret facts about the request that succeeded.

        Compatibility retries may remove provider options after the executor
        assembled its request.  Configuration telemetry remains useful, but it
        must not be mistaken for the options the provider actually accepted.
        ``LLMCompletion`` exposes a deliberately small, secret-free outcome
        projection for that distinction.
        """

        effective = getattr(completion, "provider_request_options", None)
        if isinstance(effective, Mapping):
            state.request_options["openai_prompt_cache_key_sent"] = (
                effective.get("prompt_cache_key_sent") is True
            )
            state.request_options["openai_prompt_cache_options_sent"] = (
                effective.get("prompt_cache_options_sent") is True
            )
            retention = effective.get("prompt_cache_retention")
            state.request_options["openai_prompt_cache_retention"] = (
                retention if retention in {"in_memory", "24h"} else None
            )
            state.request_options["openai_safety_identifier_sent"] = (
                effective.get("safety_identifier_sent") is True
            )

        removed = getattr(completion, "compatibility_removed_options", None)
        if isinstance(removed, (list, tuple, set)):
            state.request_options["openai_compatibility_removed_options"] = sorted(
                {
                    item
                    for item in removed
                    if isinstance(item, str) and 0 < len(item) <= 64
                }
            )[:32]

    def _fallback_json_action_was_used(
        self,
        state: _LLMCallState,
        completion: Any,
    ) -> bool:
        if not state.fallback_json_actions:
            return False
        if bool(getattr(completion, "fallback_json_action_used", False)):
            return True
        for tool_call in list(getattr(completion, "tool_calls", []) or []):
            try:
                tool_call_to_action(
                    tool_call,
                    max_argument_bytes=(
                        self.config.tools.tool_call_args_hard_limit_bytes
                    ),
                )
            except Exception:
                continue
            return False
        try:
            parse_json_action(str(getattr(completion, "content", "")))
        except Exception:
            return False
        return True

    def _prepare_llm_budget_envelope(self, state: _LLMCallState) -> None:
        resolved = state.resolved
        if resolved is None:
            raise RuntimeError("LLM profile must be resolved before budget admission")

        state.max_input_tokens_per_call = int(
            resolved.max_input_tokens_per_call
        )
        state.max_total_tokens_per_call = int(
            resolved.max_total_tokens_per_call
        )
        state.estimated_input_tokens = estimate_request_input_tokens(
            state.request_messages,
            state.tools,
        )
        reserved_total_tokens = min(
            state.max_total_tokens_per_call,
            state.max_input_tokens_per_call + state.max_tokens,
        )
        state.resource_envelope = ResourceUsage(
            llm_calls=1,
            llm_prompt_tokens=state.max_input_tokens_per_call,
            llm_completion_tokens=state.max_tokens,
            llm_total_tokens=reserved_total_tokens,
        )
        envelope = {
            "llm_calls": 1,
            "llm_prompt_tokens": state.max_input_tokens_per_call,
            "llm_completion_tokens": state.max_tokens,
            "llm_total_tokens": reserved_total_tokens,
        }
        state.resource_envelope_sha256 = hashlib.sha256(
            dumps(envelope).encode("utf-8")
        ).hexdigest()
        state.request_options["llm_budget"] = {
            "schema_version": 1,
            "estimated_input_tokens": state.estimated_input_tokens,
            "max_input_tokens_per_call": state.max_input_tokens_per_call,
            "max_output_tokens_per_call": state.max_tokens,
            "max_total_tokens_per_call": state.max_total_tokens_per_call,
            "reserved_total_tokens": reserved_total_tokens,
            "resource_envelope_sha256": state.resource_envelope_sha256,
        }

        if state.estimated_input_tokens > state.max_input_tokens_per_call:
            self._deny_llm_budget_admission(
                state,
                reason="estimated_input_exceeds_per_call_limit",
            )
            raise ResourceLimitExceeded(
                "LLM estimated input tokens exceed max_input_tokens_per_call: "
                f"{state.estimated_input_tokens} > "
                f"{state.max_input_tokens_per_call}"
            )
        projected_tokens = state.estimated_input_tokens + state.max_tokens
        if projected_tokens > state.max_total_tokens_per_call:
            self._deny_llm_budget_admission(
                state,
                reason="estimated_total_exceeds_per_call_limit",
            )
            raise ResourceLimitExceeded(
                "LLM estimated input plus max output tokens exceed "
                "max_total_tokens_per_call: "
                f"{projected_tokens} > {state.max_total_tokens_per_call}"
            )

    def _deny_llm_budget_admission(
        self,
        state: _LLMCallState,
        *,
        reason: str,
    ) -> None:
        if state.budget_admission_denial_audited:
            return
        state.budget_admission_denial_audited = True
        self._audit.record(
            actor=state.pid,
            action="llm.budget_admission_denied",
            target=f"llm:{state.profile_id}",
            decision={
                **self._llm_resource_context(state),
                "reason": reason,
            },
        )

    @staticmethod
    def _llm_resource_context(state: _LLMCallState) -> dict[str, Any]:
        return {
            "purpose": "action_selection",
            "call_id": state.call_id,
            "profile_id": state.profile_id,
            "attempt": state.attempt,
            "estimated_input_tokens": state.estimated_input_tokens,
            "max_input_tokens_per_call": state.max_input_tokens_per_call,
            "max_output_tokens_per_call": state.max_tokens,
            "max_total_tokens_per_call": state.max_total_tokens_per_call,
            "resource_envelope_sha256": state.resource_envelope_sha256,
        }

    @staticmethod
    def _canonical_llm_usage(completion: Any) -> tuple[dict[str, int], set[str]]:
        return canonicalize_llm_usage(
            getattr(completion, "usage", None),
            api=getattr(completion, "api", None),
        )

    def _llm_completion_resource_settlement(
        self,
        state: _LLMCallState,
        completion: Any,
    ) -> tuple[ResourceSettlement, str | None]:
        if state.resource_envelope is None:
            raise RuntimeError("LLM resource envelope was not prepared")
        canonical_usage, invalid_fields = self._canonical_llm_usage(completion)
        state.completion_usage = dict(canonical_usage)
        state.invalid_completion_usage_fields = set(invalid_fields)
        token_keys = {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        }
        invalid_token_fields = sorted(invalid_fields & token_keys)
        context: dict[str, Any] = {
            **self._llm_resource_context(state),
            "usage": canonical_usage,
        }
        has_cumulative_token_limit = bool(
            self._resources is not None
            and self._resources.has_limit(state.pid, "max_llm_total_tokens")
        )
        if invalid_token_fields and has_cumulative_token_limit:
            context["invalid_usage_fields"] = invalid_token_fields
            return self._llm_maximum_resource_settlement(
                context,
                "LLM provider returned invalid token usage fields: "
                + ", ".join(invalid_token_fields),
            )
        material_invalid_fields = self._material_invalid_llm_usage_fields(
            canonical_usage,
            invalid_fields,
        )
        if material_invalid_fields:
            context["invalid_usage_fields"] = material_invalid_fields
            return self._llm_maximum_resource_settlement(
                context,
                "LLM provider returned invalid token usage fields without "
                "a valid compatible counter: "
                + ", ".join(material_invalid_fields),
            )

        has_reported_usage = any(key in canonical_usage for key in token_keys)
        if not has_reported_usage:
            if has_cumulative_token_limit:
                context["usage_missing"] = True
                return self._llm_maximum_resource_settlement(
                    context,
                    "LLM token budget is configured, but provider response "
                    "did not include token usage",
                )
            return (
                ResourceSettlement(
                    usage=ResourceUsage(llm_calls=1),
                    source="llm.completion",
                    context={**context, "usage_missing": True},
                ),
                None,
            )

        (
            prompt_tokens,
            completion_tokens,
            total_tokens,
            inconsistent_component_total,
        ) = self._normalized_llm_token_usage(canonical_usage)
        if inconsistent_component_total is not None:
            context["component_total_tokens"] = inconsistent_component_total
            return self._llm_maximum_resource_settlement(
                context,
                "LLM provider total_tokens does not equal prompt/completion usage",
            )

        actual = ResourceUsage(
            llm_calls=1,
            llm_prompt_tokens=prompt_tokens,
            llm_completion_tokens=completion_tokens,
            llm_total_tokens=total_tokens,
        )
        exceeded = [
            (
                "prompt_tokens",
                actual.llm_prompt_tokens,
                state.resource_envelope.llm_prompt_tokens,
            ),
            (
                "completion_tokens",
                actual.llm_completion_tokens,
                state.resource_envelope.llm_completion_tokens,
            ),
            (
                "total_tokens",
                actual.llm_total_tokens,
                state.resource_envelope.llm_total_tokens,
            ),
        ]
        exceeded = [item for item in exceeded if item[1] > item[2]]
        if exceeded:
            context["exceeded_usage"] = {
                name: {"actual": actual_value, "reserved": reserved_value}
                for name, actual_value, reserved_value in exceeded
            }
            return self._llm_maximum_resource_settlement(
                context,
                "LLM provider usage exceeded the reserved per-call token envelope: "
                + ", ".join(
                    f"{name}={actual_value}>{reserved_value}"
                    for name, actual_value, reserved_value in exceeded
                ),
            )
        return (
            ResourceSettlement(
                usage=actual,
                source="llm.completion",
                context=context,
            ),
            None,
        )

    @staticmethod
    def _llm_maximum_resource_settlement(
        context: Mapping[str, Any],
        violation: str,
    ) -> tuple[ResourceSettlement, str]:
        return (
            ResourceSettlement(
                usage=ResourceUsage(),
                source="llm.completion",
                context={**dict(context), "settlement": "fail_closed_maximum"},
                charge_reserved_maximum=True,
            ),
            violation,
        )

    @classmethod
    def _normalized_llm_token_usage(
        cls,
        usage: Mapping[str, int],
    ) -> tuple[int, int, int, int | None]:
        prompt_tokens = int(
            cls._first_usage_counter(
                usage,
                "prompt_tokens",
                "input_tokens",
            )
            or 0
        )
        completion_tokens = int(
            cls._first_usage_counter(
                usage,
                "completion_tokens",
                "output_tokens",
            )
            or 0
        )
        reported_total = cls._first_usage_counter(
            usage,
            "total_tokens",
            default=None,
        )
        component_total = prompt_tokens + completion_tokens
        total_tokens = component_total if reported_total is None else reported_total
        has_both_components = any(
            key in usage for key in ("prompt_tokens", "input_tokens")
        ) and any(
            key in usage for key in ("completion_tokens", "output_tokens")
        )
        inconsistent = bool(
            reported_total is not None
            and has_both_components
            and reported_total != component_total
        )
        return (
            prompt_tokens,
            completion_tokens,
            total_tokens,
            component_total if inconsistent else None,
        )

    @staticmethod
    def _material_invalid_llm_usage_fields(
        usage: Mapping[str, int],
        invalid_fields: set[str],
    ) -> list[str]:
        material: set[str] = set()
        for aliases in (
            {"prompt_tokens", "input_tokens"},
            {"completion_tokens", "output_tokens"},
            {"total_tokens"},
        ):
            invalid_aliases = invalid_fields & aliases
            if invalid_aliases and not usage.keys() & aliases:
                material.update(invalid_aliases)
        return sorted(material)

    @staticmethod
    def _first_usage_counter(
        usage: Mapping[str, int],
        *keys: str,
        default: int | None = 0,
    ) -> int | None:
        for key in keys:
            if key in usage:
                return usage[key]
        return default

    def _data_flow_provider_chain_fingerprint(
        self,
        *,
        pid: str,
        sink: DataSink,
        context: DataFlowContext,
    ) -> str:
        trust = self._data_flow.resolve_sink_trust(sink)
        authority_manifest = self._authority_manifests.get_for_process(pid)
        material = {
            "sink": sink.identity,
            "sink_identity_sha256": sink.identity_sha256,
            "sink_trust_identity": sink.registry_identity,
            "sink_trust_identity_sha256": sink.registry_identity_sha256,
            "registry_generation": self._authority.get_sink_trust_generation(),
            "trust_id": trust.trust_id if trust is not None else None,
            "trust_sha256": trust.spec_hash if trust is not None else None,
            # Provider-side retention is bounded by confidentiality and
            # identity clearance. Trust, integrity, and origin may be lowered
            # by a tool result that is then sent explicitly without changing
            # which data the provider is cleared to retain.
            "clearance_labels_sha256": hashlib.sha256(
                dumps(
                    {
                        "sensitivity": context.labels.sensitivity.value,
                        "tenant": context.labels.tenant,
                        "principal": context.labels.principal,
                    }
                ).encode("utf-8")
            ).hexdigest(),
            "authority_manifest_hash": (
                authority_manifest.manifest_hash
                if authority_manifest is not None
                else None
            ),
        }
        return hashlib.sha256(dumps(material).encode("utf-8")).hexdigest()

    def _combined_provider_chain_fingerprint(
        self,
        client: Any,
        data_flow_fingerprint: str,
    ) -> str | None:
        provider_fingerprint = self._openai_provider_chain_fingerprint(client)
        if provider_fingerprint is None:
            return None
        material = {
            "provider": provider_fingerprint,
            "data_flow": data_flow_fingerprint,
        }
        return hashlib.sha256(dumps(material).encode("utf-8")).hexdigest()

    def _assert_llm_provider_chain_scope(
        self,
        *,
        pid: str,
        profile_id: str,
        context: DataFlowContext,
        expected_sink: DataSink,
        expected_data_flow_fingerprint: str,
        expected_provider_fingerprint: str | None,
        expected_source_refs_fingerprint: str,
    ) -> None:
        current_resolved = self._llms.resolve(profile_id)
        current_sink = DataSink(
            f"llm:{current_resolved.profile_id}",
            current_resolved.identity_sha256,
        )
        current_data_flow_fingerprint = self._data_flow_provider_chain_fingerprint(
            pid=pid,
            sink=current_sink,
            context=context,
        )
        current_provider_fingerprint = self._combined_provider_chain_fingerprint(
            current_resolved.client,
            current_data_flow_fingerprint,
        )
        if (
            current_sink != expected_sink
            or current_data_flow_fingerprint != expected_data_flow_fingerprint
            or current_provider_fingerprint != expected_provider_fingerprint
            or context.source_refs_hash() != expected_source_refs_fingerprint
        ):
            raise _LLMProviderChainScopeChanged(
                "LLM provider-side response state scope changed before dispatch"
            )

    def _previous_response_state_for_state(
        self,
        pid: str,
        profile_id: str,
        client: Any,
        *,
        response_scope_fingerprint: str | None = None,
        provider_chain_fingerprint: str | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        if not isinstance(client, LLMClient):
            return None, []
        if (
            not client.responses_previous_response_id
            or not client.store
            or not client._use_responses_api()
            or not client._use_openai_request_options()
        ):
            return None, []
        call = self._processes.get_latest_llm_call(pid=pid, purpose="action_selection")
        if call is None or call.status != "ok" or call.api != "responses" or not call.response_id:
            return None, []
        if call.request_options.get("llm_profile_id") != profile_id:
            return None, []
        if call.request_options.get("openai_response_scope_fingerprint") != response_scope_fingerprint:
            return None, []
        if (
            provider_chain_fingerprint is None
            or call.request_options.get("openai_provider_chain_fingerprint") != provider_chain_fingerprint
        ):
            return None, []

        raw_manifest = call.request_options.get("openai_response_tool_calls")
        if raw_manifest is None:
            # Rows written before durable tool-output tracking are continuable
            # only when the response made no function calls at all.
            if call.tool_calls != []:
                return None, []
            raw_manifest = []
        if not isinstance(raw_manifest, list):
            return None, []

        manifest: list[dict[str, str]] = []
        seen_call_ids: set[str] = set()
        for item in raw_manifest:
            if not isinstance(item, dict):
                return None, []
            call_id = str(item.get("call_id") or "").strip()
            if not call_id or call_id in seen_call_ids:
                return None, []
            seen_call_ids.add(call_id)
            manifest.append(
                {
                    "call_id": call_id,
                    "name": str(item.get("name") or "").strip(),
                }
            )

        output_rows = self._processes.list_llm_tool_outputs(pid=pid, response_id=str(call.response_id))
        outputs_by_call_id = {str(row.get("call_id") or ""): row for row in output_rows}
        if set(outputs_by_call_id) != seen_call_ids:
            return None, []
        tool_messages: list[dict[str, Any]] = []
        for item in manifest:
            output = outputs_by_call_id[item["call_id"]].get("output_text")
            if not isinstance(output, str):
                return None, []
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "name": item["name"] or None,
                    "content": output,
                }
            )
        return str(call.response_id), tool_messages

    @staticmethod
    def _openai_provider_chain_fingerprint(client: Any) -> str | None:
        """Bind provider-side response state to its actual account boundary.

        The fingerprint is stable across restarts with the same credential but
        does not persist the credential itself.  A model, endpoint, API mode,
        credential/env identity, organization, or project change forces a
        stateless request even when the profile id is reused in place.
        """

        if not isinstance(client, LLMClient):
            return None
        credential = client.api_key
        if not credential and client.inherit_ambient_openai_sdk_config:
            credential = os.getenv(client.api_key_env)
        if not credential:
            return None
        # URL paths are case-sensitive even though scheme/host are not.  Keep
        # the configured spelling so two account gateways that differ only by
        # a case-sensitive path cannot collide and reuse provider-side state.
        # A harmless host-case change may reset the chain, which is safer than
        # treating distinct endpoints as identical.
        base_url = str(client.base_url or "https://api.openai.com/v1").strip().rstrip("/")
        sdk_client = client._async_client or client._client
        organization = getattr(sdk_client, "organization", None) or client.organization
        project = getattr(sdk_client, "project", None) or client.project
        if client.inherit_ambient_openai_sdk_config:
            organization = (
                organization
                or os.getenv("OPENAI_ORGANIZATION")
                or os.getenv("OPENAI_ORG_ID")
            )
            project = (
                project
                or os.getenv("OPENAI_PROJECT")
                or os.getenv("OPENAI_PROJECT_ID")
            )
        material = {
            "client_class": f"{type(client).__module__}.{type(client).__qualname__}",
            "base_url": base_url,
            "model": str(client.model or ""),
            "api_mode": str(client.api_mode or ""),
            "api_key_env": str(client.api_key_env or ""),
            "organization": str(organization or ""),
            "project": str(project or ""),
        }
        return hmac.new(
            str(credential).encode("utf-8"),
            dumps(material).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _messages_with_tool_outputs(
        messages: list[dict[str, Any]],
        tool_outputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not tool_outputs:
            return messages
        instructions = [message for message in messages if str(message.get("role")) in {"system", "developer"}]
        conversation = [message for message in messages if str(message.get("role")) not in {"system", "developer"}]
        return [*instructions, *tool_outputs, *conversation]

    @staticmethod
    def _response_tool_call_manifest(completion: Any) -> list[dict[str, Any]]:
        manifest: list[dict[str, Any]] = []
        for ordinal, tool_call in enumerate(list(getattr(completion, "tool_calls", []) or [])):
            if not isinstance(tool_call, dict):
                manifest.append({"ordinal": ordinal, "call_id": None, "name": None})
                continue
            manifest.append(
                {
                    "ordinal": ordinal,
                    "call_id": tool_call.get("call_id"),
                    "name": tool_call.get("name"),
                }
            )
        return manifest

    def _responses_state_scope_fingerprint(
        self,
        *,
        pid: str,
        process: Any,
        tools: list[dict[str, Any]],
        available_skills: list[dict[str, Any]] | None = None,
        task_run_settlement: bool = False,
    ) -> str:
        context_scope = self._context_scope_for_previous_response(pid)
        material = {
            "pid": pid,
            "image_id": getattr(process, "image_id", None),
            "tool_table": getattr(process, "tool_table", {}),
            "loaded_skills": getattr(process, "loaded_skills", {}),
            "context_scope": context_scope,
            "tools": to_jsonable(tools),
            "available_skills": to_jsonable(available_skills or []),
            "task_run_binding_sha256": (
                self._task_run_settlement_binding_hash(pid)
                if task_run_settlement
                else self._task_run_request_binding_hash(pid)
            ),
        }
        return hashlib.sha256(dumps(material).encode("utf-8")).hexdigest()

    def _current_responses_state_scope_fingerprint(
        self,
        pid: str,
        *,
        settlement: bool = False,
    ) -> str:
        process = self._processes.get_process(pid)
        if process is None:
            raise ValidationError(f"LLM request process does not exist: {pid}")
        prompt_process = replace(
            process,
            tool_table=self._tools.model_tool_table(pid),
            loaded_skills=self._tools.model_loaded_skills(pid),
        )
        return self._responses_state_scope_fingerprint(
            pid=pid,
            process=prompt_process,
            tools=self._tools.openai_tool_schemas(pid),
            available_skills=[],
            task_run_settlement=settlement,
        )

    def _assert_task_run_request_scope_current(
        self,
        pid: str,
        expected_scope_fingerprint: str | None,
        *,
        settlement: bool = False,
    ) -> None:
        process = self._processes.get_process(pid)
        if process is None or getattr(process, "task_run_id", None) is None:
            return
        current = self._current_responses_state_scope_fingerprint(
            pid,
            settlement=settlement,
        )
        if (
            not isinstance(expected_scope_fingerprint, str)
            or not hmac.compare_digest(current, expected_scope_fingerprint)
        ):
            self._mark_task_run_request_scope_drift(pid)
            raise ValidationError(
                "durable TaskRun LLM request binding changed before action commit"
            )

    def _context_scope_for_previous_response(self, pid: str) -> dict[str, Any]:
        return {
            "generation": self._processes.get_llm_context_generation(pid),
        }

    @staticmethod
    def _tool_schema_observation(tools: list[dict[str, Any]]) -> dict[str, int]:
        strict = 0
        non_strict = 0
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            if function.get("strict") is True:
                strict += 1
            else:
                non_strict += 1
        return {"strict": strict, "non_strict": non_strict}

    async def _complete_action(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        previous_response_id: str | None = None,
        parallel_tool_calls: bool,
    ) -> Any:
        return await self.provider.complete_action(
            client,
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            previous_response_id=previous_response_id,
            parallel_tool_calls=parallel_tool_calls,
        )

    def dispatch(
        self,
        pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.actions.dispatch(
            pid,
            action,
            context_metadata=context_metadata,
            expected_tool_id=self._task_run_expected_tool_id(pid, action),
        )

    async def adispatch(
        self,
        pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.actions.adispatch(
            pid,
            action,
            context_metadata=context_metadata,
            expected_tool_id=self._task_run_expected_tool_id(pid, action),
        )

    async def _adispatch_selected_action(
        self,
        pid: str,
        action: dict[str, Any],
        *,
        host_auto_wait: bool,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not host_auto_wait:
            return await self.adispatch(
                pid,
                action,
                context_metadata=context_metadata,
            )
        return await self.actions.adispatch_host_auto_wait(
            pid,
            action,
            context_metadata={
                **(context_metadata or {}),
                "llm_action_source": "host_empty_tool_calls_auto_wait",
            },
            expected_tool_id=self._task_run_expected_tool_id(pid, action),
        )

    def _notify_interrupt_messages(self, pid: str) -> dict[str, Any] | None:
        return self._messages.notice(
            pid,
            kind=ProcessMessageKind.INTERRUPT,
            phase="before_llm_tool_selection",
            source="llm.executor",
            instruction=self._process_message_instruction(pid),
        )

    def _pre_tool_interrupt_notice(
        self,
        pid: str,
        tool_name: str,
        tool_args: dict[str, Any],
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self._is_exact_resuming_ask_human(
            pid,
            tool_name,
            tool_args,
            context_metadata,
        ):
            # The Human provider has already presented this exact question and
            # durably recorded its answer.  Returning that answer settles the
            # admitted wait; it is not a new model-selected dispatch.  Keep the
            # interrupt unread so the next quantum must still handle it.
            return None
        if tool_name in {"read_process_messages", "receive_process_messages"}:
            return None
        if tool_name == "discover_skills":
            # A bounded metadata read is the source-neutral bridge to the
            # hidden message-handling schema. Keep unrelated discovery from
            # deferring a mandatory interrupt indefinitely.
            query = str(tool_args.get("text") or "").casefold()
            if "message" in query or "mailbox" in query:
                return None
        if tool_name == "activate_skill":
            skill_id = tool_args.get("skill_id")
            if isinstance(skill_id, str):
                try:
                    if self._skills.skill_declares_any_tool(
                        skill_id,
                        {"read_process_messages", "receive_process_messages"},
                    ):
                        return None
                except (NotFound, ValidationError):
                    pass
        instruction = self._process_message_instruction(pid)
        notice = self._messages.notice(
            pid,
            kind=ProcessMessageKind.INTERRUPT,
            phase="before_tool_call",
            source="llm.executor",
            instruction=instruction,
        )
        if notice is None:
            return None
        return {
            "ok": False,
            "tool_id": None,
            "result_oid": None,
            "payload": {"message_notice": notice},
            "error": f"unread interrupt process messages are waiting; {instruction}",
            "interrupted_by_message": True,
            "message_notice": notice,
        }

    def _is_exact_resuming_ask_human(
        self,
        pid: str,
        tool_name: str,
        tool_args: Mapping[str, Any],
        context_metadata: Mapping[str, Any] | None,
    ) -> bool:
        """Prove that dispatch only returns one already-recorded Human answer."""

        if tool_name != "ask_human" or not isinstance(context_metadata, Mapping):
            return False
        request_id = context_metadata.get("human_resume_request_id")
        if not isinstance(request_id, str) or not request_id:
            return False
        pending = self.pending.get(pid)
        if not isinstance(pending, Mapping):
            return False
        expected_action = {"action": tool_name, **dict(tool_args)}
        if not (
            pending.get("pid") == pid
            and pending.get("wait_type") == "human"
            and pending.get("status") == "resuming"
            and pending.get("request_id") == request_id
            and pending.get("action") == expected_action
        ):
            return False
        try:
            request = self._human.get(request_id)
        except Exception:
            return False
        return bool(
            request.pid == pid
            and request.status is HumanRequestStatus.APPROVED
        )

    def _notify_normal_messages(self, pid: str) -> dict[str, Any] | None:
        return self._messages.notice(
            pid,
            kind=ProcessMessageKind.NORMAL,
            phase="after_tool_call",
            source="llm.executor",
            instruction=self._process_message_instruction(pid),
        )

    def _process_message_instruction(self, pid: str) -> str:
        process = self._process.get(pid)
        message_tools = {"read_process_messages", "receive_process_messages"}
        if message_tools.isdisjoint(process.model_tool_table):
            return (
                "Resolve message handling before other work. If the preceding discovery "
                "result identifies a Skill declaring read_process_messages or "
                "receive_process_messages, activate that exact returned id with the same "
                "row's package_sha256 as expected_package_sha256; otherwise call "
                "discover_skills with text 'messages' and an unquoted JSON integer limit "
                "such as 5. Then call a "
                "visible message-read tool to inspect and acknowledge unread process messages."
            )
        return (
            "Call read_process_messages or receive_process_messages to inspect and acknowledge "
            "unread process messages."
        )

    def _handles_for_oids(self, pid: str, oids: list[str]) -> list[ObjectHandle]:
        return [self._handle_for_oid(pid, oid) for oid in oids]

    def _handle_for_oid(self, pid: str, oid: str) -> ObjectHandle:
        process = self._process.get(pid)
        if process.memory_view is not None:
            for handle in process.memory_view.roots:
                if handle.oid == oid:
                    return handle
        return self._memory.handle_for_oid(
            pid,
            oid,
            required_rights={ObjectRight.READ.value},
            optional_rights={ObjectRight.MATERIALIZE.value, ObjectRight.LINK.value, ObjectRight.DIFF.value},
            issued_by="llm.executor",
        )

    def _add_to_view(self, pid: str, handle: ObjectHandle) -> None:
        process = self._process.get(pid)
        if process.memory_view is None:
            process.memory_view = self._memory.create_view(pid, [handle], mode=ViewMode.READ_ONLY)
            self._processes.patch_process(
                pid,
                {"memory_view": process.memory_view},
                expected_revision=process.revision,
            )
        elif all(existing.oid != handle.oid for existing in process.memory_view.roots):
            self._processes.append_process_memory_roots(pid, [handle])

    def _persist_pending_action(
        self,
        pid: str,
        *,
        wait_type: str,
        action: dict[str, Any],
        content_preview: str,
        tool_call_count: int,
        request_id: str | None = None,
        child_pid: str | None = None,
        filters: dict[str, Any] | None = None,
        response_id: str | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        resume_token: str | None = None,
        pending_metadata: dict[str, Any] | None = None,
        status: str = "pending",
    ) -> str:
        return self.pending.persist(
            pid,
            wait_type=wait_type,
            action=action,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
            request_id=request_id,
            child_pid=child_pid,
            filters=filters,
            metadata=pending_metadata,
            response_id=response_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            resume_token=resume_token,
            status=status,
        )

    @staticmethod
    def _pending_data_flow_metadata(pending: dict[str, Any]) -> dict[str, Any]:
        return pending_data_flow_metadata(pending)

    def _clear_pending_action(self, pid: str, resume_token: str) -> None:
        self.pending.complete(pid, resume_token=resume_token)

    def _synchronize_pending_action(self, pid: str) -> dict[str, Any] | None:
        return self.pending.synchronize(pid)

    def _clear_in_memory_pending_action(self, pid: str) -> None:
        self.pending.clear_memory(pid)

    def _hydrate_pending_action(self, pending: dict[str, Any]) -> None:
        self.pending.hydrate(pending)

    def _load_pending_actions(self) -> None:
        pending_by_status = self._validated_pending_actions_for_startup()
        for pending in pending_by_status["resuming"]:
            pid = str(pending["pid"])
            process = self._processes.get_process(pid)
            if process is not None and process.status not in {
                ProcessStatus.EXITED,
                ProcessStatus.FAILED,
                ProcessStatus.KILLED,
            }:
                self._process.exit(
                    pid,
                    failed=True,
                    message="interrupted while resuming a durable LLM action; automatic replay is disabled",
                )
            self._audit.record(
                actor="llm.executor",
                action="llm.pending_action_resume_interrupted",
                target=f"process:{pid}",
                decision={
                    "wait_type": pending.get("wait_type"),
                    "status": "resuming",
                    "replayed": False,
                },
            )
        for pending in pending_by_status["pending"]:
            action = dict(pending.get("action") or {})
            if (
                pending.get("wait_type") == "llm_release"
                and action.get("kind") == "llm_release_request_redacted"
            ):
                pid = str(pending["pid"])
                request_id = str(pending.get("request_id") or "")
                try:
                    request = self._human.get(request_id)
                except NotFound:
                    request = None
                if request is not None and request.status not in {
                    HumanRequestStatus.PENDING,
                    HumanRequestStatus.APPROVED,
                }:
                    # Rejection/cancellation needs no provider payload and can
                    # preserve the ordinary durable rejection-resume path.
                    self._hydrate_pending_action(pending)
                    continue
                resume_token = self._pending_resume_token(pending)
                claimed = self.pending.claim(
                    pid,
                    resume_token=resume_token,
                )
                if claimed is None:
                    continue
                error = _LLMReleasePayloadUnavailable(
                    "prepared LLM release payload is unavailable because full-I/O "
                    "retention was disabled and the exact in-memory request was lost"
                )
                self._record_llm_release_payload_unavailable(
                    pid=pid,
                    request_id=request_id,
                    claimed=claimed,
                    error=error,
                )
                self._fail_interrupted_pending_resume(pid, claimed, error)
                continue
            self._hydrate_pending_action(pending)

    def _validated_pending_actions_for_startup(
        self,
    ) -> dict[str, list[dict[str, Any]]]:
        """Decode startup rows individually so corrupt Runs can fail in place.

        TaskRun recovery owns projection of a corrupt Run-local resume bundle to
        ``needs_attention``.  Ordinary process resume state has no such
        isolation boundary, so corruption there remains fatal.  In either
        case this routing pass never exposes the corrupt payload to the audit
        log or hydrates it into executor memory.
        """

        selected: dict[str, list[dict[str, Any]]] = {
            "pending": [],
            "resuming": [],
        }
        cursor: str | None = None
        while True:
            records, next_cursor = self._pending_action_validation_page(cursor)
            last_pid = cursor
            for record in records:
                identity = self._pending_action_validation_identity(
                    record,
                    after_pid=last_pid,
                )
                pid, status, task_run_id, decode_valid, error_code = identity
                last_pid = pid
                routed = self._route_pending_action_validation_identity(
                    pid=pid,
                    status=status,
                    task_run_id=task_run_id,
                    decode_valid=decode_valid,
                    error_code=error_code,
                )
                if routed is not None:
                    selected[status].append(routed)

            if next_cursor is None:
                break
            if not records or next_cursor != last_pid or next_cursor == cursor:
                raise ValidationError(
                    "persisted pending LLM action validation projection is invalid"
                )
            cursor = next_cursor
        return selected

    def _pending_action_validation_page(
        self,
        cursor: str | None,
    ) -> tuple[list[Any] | tuple[Any, ...], str | None]:
        page = self._processes.list_llm_pending_action_validation_rows(
            after_pid=cursor,
            limit=self.config.task_runs.recovery_page_size,
        )
        expected_keys = {"records", "next_cursor", "schema_version"}
        if not isinstance(page, Mapping) or set(page) != expected_keys:
            self._invalid_pending_action_validation_projection()
        if type(page["schema_version"]) is not int or page["schema_version"] != 1:
            self._invalid_pending_action_validation_projection()
        records = page["records"]
        next_cursor = page["next_cursor"]
        if not isinstance(records, (list, tuple)):
            self._invalid_pending_action_validation_projection()
        if next_cursor is not None and (
            not isinstance(next_cursor, str) or not next_cursor
        ):
            self._invalid_pending_action_validation_projection()
        return records, next_cursor

    def _pending_action_validation_identity(
        self,
        record: Any,
        *,
        after_pid: str | None,
    ) -> tuple[str, str, str | None, bool, str | None]:
        expected_keys = {
            "pid",
            "status",
            "task_run_id",
            "decode_valid",
            "error_code",
        }
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            self._invalid_pending_action_validation_projection()
        pid = record["pid"]
        status = record["status"]
        task_run_id = record["task_run_id"]
        decode_valid = record["decode_valid"]
        error_code = record["error_code"]
        valid_identity = (
            isinstance(pid, str)
            and bool(pid)
            and (after_pid is None or pid > after_pid)
            and isinstance(status, str)
            and (
                task_run_id is None
                or (isinstance(task_run_id, str) and bool(task_run_id))
            )
            and type(decode_valid) is bool
            and (
                error_code is None
                or (isinstance(error_code, str) and bool(error_code))
            )
        )
        if not valid_identity:
            self._invalid_pending_action_validation_projection()
        return pid, status, task_run_id, decode_valid, error_code

    def _route_pending_action_validation_identity(
        self,
        *,
        pid: str,
        status: str,
        task_run_id: str | None,
        decode_valid: bool,
        error_code: str | None,
    ) -> dict[str, Any] | None:
        if not decode_valid:
            return self._defer_corrupt_task_run_pending_action(
                pid=pid,
                status=status,
                task_run_id=task_run_id,
                error_code=error_code,
            )
        if status not in {"pending", "resuming", "completed"} or error_code is not None:
            self._invalid_pending_action_validation_projection()
        if status == "completed":
            return None
        pending = self.pending.get(pid)
        if (
            pending is None
            or pending.get("pid") != pid
            or pending.get("status") != status
        ):
            raise ValidationError("invalid persisted pending LLM action")
        return pending

    def _defer_corrupt_task_run_pending_action(
        self,
        *,
        pid: str,
        status: str,
        task_run_id: str | None,
        error_code: str | None,
    ) -> None:
        if status != "invalid" or error_code is None:
            self._invalid_pending_action_validation_projection()
        if task_run_id is None:
            raise ValidationError("invalid persisted pending LLM action")
        self._audit.record(
            actor="llm.executor",
            action="llm.pending_action_corrupt_deferred_to_task_run",
            target=f"process:{pid}",
            decision={
                "task_run_id": task_run_id,
                "error_code": error_code,
                "hydrated": False,
            },
        )
        return None

    @staticmethod
    def _invalid_pending_action_validation_projection() -> None:
        raise ValidationError(
            "persisted pending LLM action validation projection is invalid"
        )

    def _restore_pending_compaction_child_goal(self, pending: dict[str, Any]) -> None:
        try:
            from agent_libos.tools.builtin.context import restore_pending_compaction_child_goal

            restore_pending_compaction_child_goal(
                pending,
                processes=self._processes,
                objects=self._objects,
                memory=self._memory,
            )
        except Exception as exc:
            public_error, internal_error = self._llm_error_artifacts(
                exc,
                code="llm_context_restore_error",
            )
            self._audit.record(
                actor="llm.executor",
                action="llm.pending_compaction_child_restore_failed",
                target=f"process:{pending.get('pid')}",
                decision={
                    "error": public_error["message"],
                    "error_details": public_error,
                    "internal_error": internal_error,
                    "child_pid": pending.get("child_pid"),
                },
                correlation_id=public_error["correlation_id"],
            )
