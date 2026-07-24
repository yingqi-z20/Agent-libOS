from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, TYPE_CHECKING

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import (
    HumanApprovalRequired,
    NotFound,
    ProcessMessageWaitRequired,
    ProcessWaitRequired,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps, to_jsonable
from agent_libos.llm.client import LLMClient
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
    provider_usage_lower_bound,
)
from agent_libos.llm.event_projection import project_prompt_events
from agent_libos.llm.prompt import (
    build_system_prompt,
    build_user_prompt,
    recover_initial_goal_context,
)
from agent_libos.llm.records import observable_llm_call_fields
from agent_libos.llm.usage import canonicalize_llm_usage
from agent_libos.llm.tool_protocol import tool_call_to_action
from agent_libos.llm.pending import (
    LLMPendingActionService,
    pending_data_flow_metadata,
    pending_metadata,
    pending_resume_token,
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
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProviderPhase,
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
_IMAGE_ONLY_TRANSCRIPT_KEY = "image_only_transcript"
_IMAGE_ONLY_TRANSCRIPT_SCHEMA_VERSION = 1
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

    @staticmethod
    def _image_only_transcript_anchor(image: Any, process: Any) -> dict[str, str]:
        system_prompt = str(getattr(image, "system_prompt", "") or "")
        return {
            "image_id": str(image.image_id),
            "goal_oid": str(process.goal_oid or ""),
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
        }

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
        latest = self._processes.get_latest_llm_call(
            pid=pid,
            purpose="action_selection",
        )
        if latest is None:
            return None, None
        marker = self._image_only_marker_for_anchor(latest, anchor)
        if marker is not None or latest.image_id != image.image_id:
            return latest, marker
        if latest.request_options.get(_IMAGE_ONLY_TRANSCRIPT_KEY) is None:
            raise ValidationError(
                "image_only legacy prompt history cannot be resumed transparently"
            )
        return latest, None

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
    ) -> tuple[list[dict[str, Any]], DataFlowContext, list[str]]:
        anchor = self._image_only_transcript_anchor(image, process)
        latest, marker = self._latest_image_only_transcript(
            pid=pid,
            image=image,
            anchor=anchor,
        )
        if latest is None or marker is None:
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
        capabilities: list[Any],
        tools: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        available_skills: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], DataFlowContext, str | None]:
        if image.prompt_mode == PROMPT_MODE_IMAGE_ONLY:
            if not self.config.llm.persist_full_io:
                raise _ImageOnlyFullIORequired(
                    "image_only requires llm.persist_full_io=true for durable transcript replay"
                )
            messages, flow_context, input_refs = self._image_only_messages_and_flow(
                pid=pid,
                image=image,
                process=process,
                context=context,
            )
            self._audit.record(
                actor=pid,
                action="llm.request",
                target=f"image:{image.image_id}",
                input_refs=input_refs,
                decision={
                    "messages": len(messages),
                    "policy": image.context_policy,
                    "prompt_mode": PROMPT_MODE_IMAGE_ONLY,
                    "event_projection": {"model_visible": False},
                },
            )
            return messages, flow_context, None
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
            if goal_is_materialized
            else self._retained_original_goal_context(
                process=process,
                image=image,
            )
        )
        flow_context = self._data_flow.context_from_materialization(pid, context)
        if original_goal_context is not None:
            flow_context = self._include_retained_goal_labels(
                flow_context,
                process.goal_oid,
            )
        event_projection = project_prompt_events(
            events,
            context_object_name=self.context_memory.object_name(pid),
            payload_max_chars=self.config.llm_context.prompt_event_payload_max_chars,
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
            },
        )
        return (
            messages,
            flow_context,
            event_projection.represented_through_event_id,
        )

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
        events = [
            replace(
                event,
                source=self._tools.redact_model_context(pid, event.source),
                payload=self._tools.redact_model_context(pid, event.payload),
                correlation_id=self._tools.redact_model_context(pid, event.correlation_id),
                causality=self._tools.redact_model_context(pid, event.causality),
            )
            for event in self._events.list(
                target=pid,
                limit=self.config.llm_context.recent_event_limit,
                after_event_id=process.event_cursor,
            )
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
        available_skills = self._skills.available_builtin_prompt_context(pid)
        prepared_context = await self._prepare_llm_context(
            pid=pid,
            image=image,
            process=prompt_process,
            source_context=source_context,
            events=events,
            capabilities=capabilities,
            tools=tools,
        )
        if isinstance(prepared_context, dict):
            return prepared_context
        context = prepared_context
        try:
            messages, flow_context, represented_through_event_id = (
                self._assemble_llm_request(
                    pid=pid,
                    image=image,
                    process=prompt_process,
                    context=context,
                    events=events,
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
            response_scope_fingerprint = self._responses_state_scope_fingerprint(
                pid=pid,
                process=prompt_process,
                context=context,
                tools=openai_tools,
                available_skills=available_skills,
            )
            (
                completion,
                actions,
                parallel_tool_calls,
                host_auto_wait,
            ) = await self._complete_valid_action(
                pid,
                messages,
                openai_tools,
                response_scope_fingerprint=response_scope_fingerprint,
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
        durable_error = self._durable_llm_error(error)
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
        if reason is not None:
            decision["reason"] = reason
        self._audit.record(
            actor=pid,
            action="llm.action_failed",
            target=f"process:{pid}",
            decision=decision,
        )
        result = {"ok": False, "error": durable_error}
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
        self._audit.record(
            actor=pid,
            action="llm.context_pressure_detected",
            target=f"process:{pid}",
            decision=common,
        )
        marker_failure = self._record_storage_context_attempt(
            pid,
            action=action,
            pending_context=pending_context,
            common=common,
        )
        if marker_failure is not None:
            return marker_failure
        self._audit.record(
            actor=pid,
            action="llm.context_pressure_auto_attempted",
            target=f"process:{pid}",
            decision=common,
        )
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
            self._persist_completed_context_management_marker(
                pid,
                action=action,
                metadata={**pending_context, "outcome": "attempted"},
            )
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
        if include_error_type:
            decision["error_type"] = type(error).__name__
        if include_error:
            decision["error"] = str(error)
        self._audit.record(
            actor=pid,
            action="llm.context_pressure_failed",
            target=f"process:{pid}",
            decision=decision,
        )
        return self._fail_llm_quantum(pid, error)

    def _durable_llm_error(self, error: BaseException) -> str:
        detail = str(error)
        if self.config.llm.persist_full_io:
            return detail
        encoded = detail.encode("utf-8", errors="replace")
        return (
            f"{type(error).__name__}: provider error detail redacted "
            f"(bytes={len(encoded)}, sha256={hashlib.sha256(encoded).hexdigest()})"
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
    ) -> dict[str, Any]:
        if host_auto_wait and len(actions) != 1:
            raise ValueError("host-generated auto-wait must contain exactly one action")
        if parallel_tool_calls and len(actions) > 1:
            return await self._dispatch_action_batch(
                pid=pid,
                completion=completion,
                actions=actions,
            )
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
                **tool_call_context,
            )
        self._persist_response_tool_output(
            pid=pid,
            result=result,
            **tool_call_context,
        )
        return self._completed_action_result(
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

    async def _dispatch_action_batch(
        self,
        *,
        pid: str,
        completion: Any,
        actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        completed_actions: list[dict[str, Any]] = []
        completed_results: list[dict[str, Any]] = []
        content_preview = self._completion_content_preview(getattr(completion, "content", ""))
        tool_call_count = len(getattr(completion, "tool_calls", []) or [])
        stop_reason = "completed"
        stopped_action: dict[str, Any] | None = None
        stopped_result: dict[str, Any] | None = None

        for action_index, action in enumerate(actions):
            tool_call_context = self._completion_tool_call_context(completion, index=action_index)
            try:
                result = await self.adispatch(pid, action)
            except HumanApprovalRequired as exc:
                stop_reason = "waiting_human"
                self._persist_unexecuted_parallel_tool_outputs(
                    pid=pid,
                    completion=completion,
                    start_index=action_index + 1,
                    reason=stop_reason,
                )
                payload = self._wait_for_human_action(
                    pid=pid,
                    action=action,
                    request_id=exc.request_id,
                    message=str(exc),
                    content_preview=content_preview,
                    tool_call_count=tool_call_count,
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
                    pending_action=action,
                )
                return self._with_parallel_batch_progress(payload, completed_actions, completed_results)
            except ProcessWaitRequired as exc:
                stop_reason = "waiting_child"
                self._persist_unexecuted_parallel_tool_outputs(
                    pid=pid,
                    completion=completion,
                    start_index=action_index + 1,
                    reason=stop_reason,
                )
                pending_action = exc.resume_action or action
                payload = self._wait_for_child_action(
                    pid=pid,
                    action=pending_action,
                    child_pid=exc.child_pid,
                    message=str(exc),
                    content_preview=content_preview,
                    tool_call_count=tool_call_count,
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
                return self._with_parallel_batch_progress(payload, completed_actions, completed_results)
            except ProcessMessageWaitRequired as exc:
                stop_reason = "waiting_message"
                self._persist_unexecuted_parallel_tool_outputs(
                    pid=pid,
                    completion=completion,
                    start_index=action_index + 1,
                    reason=stop_reason,
                )
                payload = self._wait_for_message_action(
                    pid=pid,
                    action=action,
                    filters=exc.filters,
                    message=str(exc),
                    content_preview=content_preview,
                    tool_call_count=tool_call_count,
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
                    pending_action=action,
                )
                return self._with_parallel_batch_progress(payload, completed_actions, completed_results)
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
            self._persist_response_tool_output(pid=pid, result=result, **tool_call_context)
            completed_actions.append(action)
            completed_results.append(result)
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
        return self._complete_action_batch_payload(
            payload=payload,
            completed_actions=completed_actions,
            completed_results=completed_results,
            stopped_action=stopped_action,
            stopped_result=stopped_result,
        )

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
    ) -> dict[str, Any]:
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
            pending_metadata=pending_metadata,
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
            "pending_metadata": dict(pending_metadata or {}),
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
        return {"ok": False, "waiting_human": True, "request_id": request_id}

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
    ) -> dict[str, Any]:
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
            pending_metadata=pending_metadata,
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
            "pending_metadata": dict(pending_metadata or {}),
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
        return {"ok": False, "waiting_event": True, "child_pid": child_pid}

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
    ) -> dict[str, Any]:
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
            pending_metadata=pending_metadata,
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
            "pending_metadata": dict(pending_metadata or {}),
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
        return {"ok": False, "waiting_message": True, "filters": filters}

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
            return self._completed_action_result(
                pid=pid,
                action=action,
                result=result,
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
                resumed_after_human=True,
            )

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
        return self._completed_action_result(
            pid=pid,
            action=action,
            result=result,
            content_preview=str(pending.get("content_preview", "")),
            tool_call_count=int(pending.get("tool_call_count", 0)),
            resumed_after_human=True,
        )

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
        completed_action: tuple[Any, list[dict[str, Any]], bool, bool],
    ) -> dict[str, Any]:
        completion, actions, parallel_tool_calls, host_auto_wait = completed_action
        self._clear_pending_action(pid, self._pending_resume_token(claimed))
        return await self._dispatch_completed_llm_action(
            pid=pid,
            completion=completion,
            actions=actions,
            parallel_tool_calls=parallel_tool_calls,
            host_auto_wait=host_auto_wait,
            resumed_after_human=True,
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
        message = (
            "durable LLM action resume failed after its non-replayable claim; "
            f"automatic replay is disabled: {type(error).__name__}: {error}"
        )
        terminal_error: str | None = None
        process = self._processes.get_process(pid)
        if process is not None and process.status not in {
            ProcessStatus.EXITED,
            ProcessStatus.FAILED,
            ProcessStatus.KILLED,
        }:
            try:
                self._process.exit(pid, failed=True, message=message)
            except Exception as exc:
                terminal_error = f"{type(exc).__name__}: {exc}"
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
                    terminal_error = (
                        f"{terminal_error}; fallback={type(fallback_exc).__name__}: {fallback_exc}"
                    )
        try:
            self._audit.record(
                actor="llm.executor",
                action="llm.pending_action_resume_interrupted",
                target=f"process:{pid}",
                decision={
                    "wait_type": pending.get("wait_type"),
                    "status": pending.get("status"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "terminal_error": terminal_error,
                    "replayed": False,
                },
            )
        except Exception:
            # Preserve the original post-claim failure.  The durable resuming
            # row and FAILED process state remain the primary evidence.
            pass

    @staticmethod
    def _pending_resume_token(pending: dict[str, Any]) -> str:
        return pending_resume_token(pending)

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
        return await self._finish_resumed_context_management(
            pid,
            result={
                "ok": False,
                "tool_id": None,
                "result_oid": None,
                "payload": None,
                "error": str(error),
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

    def _selected_completion_tool_call_context(self, completion: Any) -> dict[str, str | None]:
        tool_calls = list(getattr(completion, "tool_calls", []) or [])
        for index in range(len(tool_calls) - 1, -1, -1):
            tool_call = tool_calls[index]
            if not isinstance(tool_call, dict):
                continue
            try:
                tool_call_to_action(tool_call)
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
        call = self._processes.get_latest_llm_call(pid=pid, purpose="action_selection")
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
        return self._completed_action_result(
            pid=pid,
            action=action,
            result=result,
            content_preview=str(pending.get("content_preview", "")),
            tool_call_count=int(pending.get("tool_call_count", 0)),
            resumed_after_human=False,
        )

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
        _prepared_request: dict[str, Any] | None = None,
    ) -> tuple[Any, list[dict[str, Any]], bool, bool]:
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
                ) = await self._complete_action_recorded(
                    pid=pid,
                    messages=attempt_messages,
                    tools=tools,
                    attempt=attempt_number,
                    max_attempts=selected_max_attempts,
                    response_scope_fingerprint=response_scope_fingerprint,
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
                return completion, actions, parallel_tool_calls, auto_wait_used
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

    def _preflight_parallel_tool_batch(self, pid: str, actions: list[dict[str, Any]]) -> None:
        self.actions.preflight_parallel(pid, actions)

    def _validate_dispatchable_action(self, pid: str, action: dict[str, Any]) -> None:
        self.actions.validate(pid, action)

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
        _force_stateless: bool = False,
        _chain_scope_retry: int = 0,
        _prepared_request: dict[str, Any] | None = None,
    ) -> tuple[Any, bool, bool, bool, str]:
        state = self._initialize_llm_call_state(
            pid=pid,
            messages=messages,
            tools=tools,
            attempt=attempt,
            max_attempts=max_attempts,
            prepared_request=_prepared_request,
        )
        try:
            if _prepared_request is None:
                await self._prepare_fresh_llm_request(
                    state,
                    response_scope_fingerprint=response_scope_fingerprint,
                    force_stateless=_force_stateless,
                )
            else:
                self._prepare_resumed_llm_request(state, _prepared_request)
            completion = await self._invoke_prepared_llm_request(state)
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
                _force_stateless=True,
                _chain_scope_retry=_chain_scope_retry + 1,
            )
        except Exception as exc:
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
    ) -> _LLMCallState:
        process = self._process.get(pid)
        if prepared_request is None:
            profile_id = (
                process.llm_profile_id or self.config.llm.default_profile_id
            )
            return _LLMCallState(
                pid=pid,
                process=process,
                call_id=new_id("llmcall"),
                created_at=utc_now(),
                profile_id=profile_id,
                attempt=attempt,
                max_attempts=max_attempts,
                request_options={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "purpose": "action_selection",
                    "llm_profile_id": profile_id,
                },
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
        state.egress_payload = {
            "messages": state.request_messages,
            "tools": state.tools,
            "profile_id": resolved.profile_id,
            "previous_response_id": state.previous_response_id,
            "parallel_tool_calls": state.parallel_tool_calls,
        }
        # Context pressure is evaluated only after the complete request has
        # been assembled, but before either LLM resource preflight or external
        # data-flow clearance. Prompt-mode notices therefore participate in
        # the exact egress payload that is checked and persisted.
        self._data_flow.precheck_egress_clearance(
            pid=state.pid,
            sink=precheck_sink,
            context=state.flow_context,
            payload=state.egress_payload,
        )
        self._preflight_llm_call(state.pid)
        state.canonical_args = {
            "profile_id": resolved.profile_id,
            "sink_identity_sha256": state.sink.identity_sha256,
            "payload_sha256": hashlib.sha256(
                dumps(to_jsonable(state.egress_payload)).encode("utf-8")
            ).hexdigest(),
            "attempt": state.attempt,
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
                    "error": str(exc),
                },
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
                    "error": str(exc),
                    "context_generation_after": generation_after,
                },
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
        )

    def _prepare_resumed_llm_request(
        self,
        state: _LLMCallState,
        prepared_request: dict[str, Any],
    ) -> None:
        self._preflight_llm_call(state.pid)
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

    async def _invoke_prepared_llm_request(self, state: _LLMCallState) -> Any:
        assert state.resolved is not None and state.sink is not None
        invocation = ProtectedOperationInvocation(
            pid=state.pid,
            actor=state.pid,
            target=state.sink.identity,
            canonical_args=state.canonical_args,
            observation={
                **state.canonical_args,
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
            prepare=lambda: self._assert_llm_call_scope(state),
            failure_evidence=lambda error, phase: self._llm_failure_evidence(
                state,
                error,
                phase,
            ),
        )
        with self._protected_operations.start(
            "primitive.llm.complete",
            invocation,
            provider=state.client,
        ) as protected:

            async def dispatch_bound_request() -> Any:
                self._assert_llm_call_scope(state)
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
            return protected.complete(
                completion,
                self._llm_success_evidence(state, completion),
                classification_override=ExternalEffectClassification(
                    rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                    rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                    state_mutation=True,
                    information_flow=True,
                    metadata={"outcome": "provider_completed"},
                ),
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
        self._charge_llm_attempt(
            state.pid,
            source="llm.error",
            context={"error_type": type(error).__name__},
        )
        observable = observable_llm_call_fields(
            messages=state.request_messages,
            tools=state.tools,
            response_content="",
            tool_calls=[],
            reasoning=None,
            raw_response=None,
            error=str(error),
            config=self.config,
        )
        self._processes.insert_llm_call(
            LLMCallRecord(
                call_id=state.call_id,
                pid=state.pid,
                image_id=state.process.image_id,
                purpose="action_selection",
                status="error",
                messages=observable["messages"],
                tools=observable["tools"],
                request_options=state.request_options,
                response_content=observable["response_content"],
                tool_calls=observable["tool_calls"],
                reasoning=observable["reasoning"],
                raw_response=observable["raw_response"],
                observability=observable["observability"],
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
    ) -> tuple[Any, bool, bool, bool, str]:
        self._prepare_image_only_transcript_record(state, completion)
        usage, invalid_usage_fields = self._canonical_llm_usage(completion)
        self._record_effective_provider_request_options(state, completion)
        if invalid_usage_fields:
            state.request_options["invalid_usage_fields"] = sorted(
                invalid_usage_fields
            )
        state.request_options["fallback_json_action_used"] = (
            self._fallback_json_action_was_used(state, completion)
        )
        self._charge_llm_attempt(
            state.pid,
            source="llm.completion",
            context={"usage": usage},
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
            reasoning=getattr(completion, "reasoning", None),
            raw_response=getattr(completion, "raw", None),
            config=self.config,
        )
        self._processes.insert_llm_call(
            LLMCallRecord(
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
        )
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
        self._charge_llm_completion(
            state.pid,
            usage,
            invalid_usage_fields=invalid_usage_fields,
        )
        return (
            completion,
            state.parallel_tool_calls,
            state.auto_wait_on_empty_tool_calls,
            state.fallback_json_actions,
            str(state.request_options["llm_profile_id"]),
        )

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
        previous = self._processes.get_latest_llm_call(
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
        anchor = self._image_only_transcript_anchor(image, state.process)
        input_oids = {ref.oid for ref in state.flow_context.source_refs}
        input_oids.update(
            self._previous_image_only_input_oids(state=state, anchor=anchor)
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

    @staticmethod
    def _fallback_json_action_was_used(
        state: _LLMCallState,
        completion: Any,
    ) -> bool:
        if not state.fallback_json_actions:
            return False
        if bool(getattr(completion, "fallback_json_action_used", False)):
            return True
        for tool_call in list(getattr(completion, "tool_calls", []) or []):
            try:
                tool_call_to_action(tool_call)
            except Exception:
                continue
            return False
        try:
            parse_json_action(str(getattr(completion, "content", "")))
        except Exception:
            return False
        return True

    def _preflight_llm_call(self, pid: str) -> None:
        resources = self._resources
        if resources is None:
            return
        resources.preflight(
            pid,
            ResourceUsage(llm_calls=1),
            source="llm.request",
            context={"purpose": "action_selection"},
        )

    def _charge_llm_attempt(self, pid: str, *, source: str, context: dict[str, Any] | None = None) -> None:
        resources = self._resources
        if resources is None:
            return
        resources.charge(
            pid,
            ResourceUsage(llm_calls=1),
            source=source,
            context=context or {},
            allow_overage=True,
            kill_on_exceed=True,
        )

    @staticmethod
    def _canonical_llm_usage(completion: Any) -> tuple[dict[str, int], set[str]]:
        return canonicalize_llm_usage(
            getattr(completion, "usage", None),
            api=getattr(completion, "api", None),
        )

    def _charge_llm_completion(
        self,
        pid: str,
        usage: Mapping[str, int],
        *,
        invalid_usage_fields: set[str] | None = None,
    ) -> None:
        resources = self._resources
        if resources is None:
            return
        canonical_usage = dict(usage)
        invalid_fields = invalid_usage_fields or set()
        has_token_limit = resources.has_limit(pid, "max_llm_total_tokens")
        token_keys = {"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"}
        if has_token_limit and not any(
            key in canonical_usage or key in invalid_fields for key in token_keys
        ):
            resources.charge(
                pid,
                ResourceUsage(),
                source="llm.completion",
                context={"usage_missing": True},
                allow_overage=False,
                kill_on_exceed=False,
            )
            raise ResourceLimitExceeded("LLM token budget is configured, but provider response did not include token usage")
        if has_token_limit:
            prompt_value = self._budget_usage_int(
                canonical_usage,
                "prompt_tokens",
                "input_tokens",
                invalid_fields=invalid_fields,
            )
            completion_value = self._budget_usage_int(
                canonical_usage,
                "completion_tokens",
                "output_tokens",
                invalid_fields=invalid_fields,
            )
            total_value = self._budget_usage_int(
                canonical_usage,
                "total_tokens",
                invalid_fields=invalid_fields,
            )
            prompt_tokens = prompt_value or 0
            completion_tokens = completion_value or 0
            component_total = prompt_tokens + completion_tokens
            total_tokens = component_total if total_value is None else total_value
            if total_value is not None and total_value < component_total:
                raise ResourceLimitExceeded(
                    "LLM token budget is configured, but provider total_tokens is smaller than prompt/completion usage"
                )
        else:
            prompt_tokens = self._usage_int(
                canonical_usage, "prompt_tokens", "input_tokens"
            )
            completion_tokens = self._usage_int(
                canonical_usage, "completion_tokens", "output_tokens"
            )
            total_tokens = self._usage_int(canonical_usage, "total_tokens")
            if total_tokens == 0 and (prompt_tokens or completion_tokens):
                total_tokens = prompt_tokens + completion_tokens
        resources.charge(
            pid,
            ResourceUsage(
                llm_prompt_tokens=prompt_tokens,
                llm_completion_tokens=completion_tokens,
                llm_total_tokens=total_tokens,
            ),
            source="llm.completion",
            context={"usage": canonical_usage},
            allow_overage=True,
            kill_on_exceed=True,
        )

    def _usage_int(self, usage: Mapping[str, int], *keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if value is None:
                continue
            return value
        return 0

    def _budget_usage_int(
        self,
        usage: Mapping[str, int],
        *keys: str,
        invalid_fields: set[str],
    ) -> int | None:
        for key in keys:
            if key in invalid_fields:
                raise ResourceLimitExceeded(
                    "LLM token budget is configured, but provider returned "
                    f"invalid {key}"
                )
            if key not in usage:
                continue
            return usage[key]
        return None

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
        credential = client.api_key or os.getenv(client.api_key_env)
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
        context: Any,
        tools: list[dict[str, Any]],
        available_skills: list[dict[str, Any]] | None = None,
    ) -> str:
        context_scope = self._context_scope_for_previous_response(pid)
        material = {
            "pid": pid,
            "image_id": getattr(process, "image_id", None),
            "tool_table": getattr(process, "tool_table", {}),
            "loaded_skills": getattr(process, "loaded_skills", {}),
            "context_scope": context_scope,
            "tools": to_jsonable(tools),
            "available_builtin_skills": to_jsonable(available_skills or []),
        }
        return hashlib.sha256(dumps(material).encode("utf-8")).hexdigest()

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
    ) -> dict[str, Any] | None:
        if tool_name in {"read_process_messages", "receive_process_messages"}:
            return None
        if (
            tool_name == "activate_skill"
            and tool_args.get("skill_id") == "agent-libos-child-processes"
        ):
            # This built-in activation only projects tools already present in
            # the image. It is the minimal no-authority bridge needed to make
            # the interrupt-handling tools visible under Skills projection.
            return None
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
                "Call activate_skill with skill_id agent-libos-child-processes, then call "
                "read_process_messages or receive_process_messages to inspect and acknowledge "
                "unread process messages."
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
        for pending in self.pending.list(status="resuming"):
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
        for pending in self.pending.list(status="pending"):
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
            self._audit.record(
                actor="llm.executor",
                action="llm.pending_compaction_child_restore_failed",
                target=f"process:{pending.get('pid')}",
                decision={"error": str(exc), "child_pid": pending.get("child_pid")},
            )
