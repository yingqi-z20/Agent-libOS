from __future__ import annotations

import copy
import hashlib
import json
import os
from ast import literal_eval
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import (
    ToolsExecutionLoop,
    ToolsExecutor,
    is_string_list,
    tool_result_to_str,
)
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.llm import client as llm_client_module
from agent_libos.llm.client import LLMClient, LLMCompletion
from agent_libos.models import (
    PROMPT_MODE_IMAGE_ONLY,
    PROMPT_MODES,
    AgentImage,
    CapabilityRight,
    ProcessStatus,
)
from agent_libos.models.exceptions import NotFound
from agent_libos.storage import SQLiteStore
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolPolicy, ToolResult
from agent_libos.utils.openai_schema import normalize_openai_strict_schema
from agent_libos.utils.serde import dumps as serde_dumps, to_jsonable
from agent_libos import Runtime

from agent_libos_dojo.contained import (
    CompiledTaskAuthority,
    ContainedAgentDojoFunctionTool,
    ContainedOperationExecutor,
    FunctionPolicyCatalog,
    configured_model_processing_sink,
    register_compiled_sinks,
    register_model_processing_sink,
)


HIDDEN_TERMINAL_TOOL = "agentdojo_runtime_submit_final"
DEFAULT_MAX_TOOL_ITERATIONS = 15
EVALUATION_MAX_COMPLETION_TOKENS = 65_536
EVALUATION_TIMEOUT_S = 240.0
EVALUATION_ENABLE_THINKING = True
EVALUATION_MAX_RETRIES = 2
_IMAGE_ID = "agentdojo-native-semantics:v0"
_CONTAINED_IMAGE_ID = "agentdojo-native-contained:v0"
_AGENTDOJO_NORMALIZATION_SCHEMA_VERSION = 1
_AGENTDOJO_NORMALIZER = "agentdojo-pydantic-defaults-and-string-list-v1"
_NATIVE_TOOL_OUTCOME_SCHEMA_VERSION = 2
_CUSTOM_BASE_URL_POLICY_ENV = "AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(serde_dumps(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class PipelineRunError(RuntimeError):
    """A trajectory did not produce oracle-usable evidence."""


@dataclass(frozen=True)
class ExplicitDotenvSnapshot:
    """Immutable provider configuration captured from one explicit dotenv file."""

    path: Path = field(repr=False)
    _file_sha256: str = field(repr=False)
    model_override: str | None
    _dotenv_items: tuple[tuple[str, str], ...] = field(repr=False)
    _client_init_items: tuple[tuple[str, Any], ...] = field(repr=False)
    _ambient_equality_check_passed: bool = field(repr=False)
    _redaction_check_passed: bool = field(repr=False)
    _custom_base_url_policy_check_passed: bool = field(repr=False)

    def new_client(self) -> LLMClient:
        """Build an independent client without consulting the file or environment."""

        client = LLMClient(
            **{
                name: copy.deepcopy(value)
                for name, value in self._client_init_items
            }
        )
        # Apply the non-secret CLI selection after cloning the captured client
        # snapshot.  This permits a single protected credential file to drive
        # multiple requested model labels without consulting ambient process
        # state or duplicating the credential file.
        if self.model_override is not None:
            client.model = self.model_override
        return client

    def assert_unchanged(self) -> None:
        """Fail closed if the selected dotenv bytes changed after capture."""

        try:
            current_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        except OSError as exc:
            raise PipelineRunError(
                "selected dotenv file became unavailable during the evaluation run"
            ) from exc
        if current_sha256 != self._file_sha256:
            raise PipelineRunError(
                "selected dotenv file changed during the evaluation run"
            )

    def redactions(self) -> dict[str, str]:
        """Return exact run-start values that must never enter artifacts."""

        return _credential_redactions(dict(self._dotenv_items))

    def verification_metadata(
        self,
        *,
        credential_profile_id: str | None = None,
    ) -> dict[str, Any]:
        """Return the minimal public credential provenance projection.

        Exact dotenv bytes and credential-related values stay in this
        in-memory snapshot so the run can fail closed on drift and redact
        artifacts before they are committed.  Public artifacts deliberately
        contain no dotenv, key, endpoint, organization, project, safety, or
        prompt-cache value, hash, or length.  A stable non-secret profile ID
        may be supplied by the frozen master protocol; it is never derived
        from credential material.
        """

        if not self._ambient_equality_check_passed:
            raise PipelineRunError(
                "ambient OpenAI configuration equality was not verified"
            )
        if not self._redaction_check_passed:
            raise PipelineRunError(
                "credential artifact redaction was not configured"
            )
        if not self._custom_base_url_policy_check_passed:
            raise PipelineRunError(
                "custom base URL policy was not verified"
            )
        selected_profile = _normalize_credential_profile_id(
            credential_profile_id
        )
        metadata: dict[str, Any] = {
            "schema_version": 2,
            "source": "explicit_dotenv_whitelist",
            "ambient_configuration_equality_check_passed": True,
            "artifact_redaction_configuration_check_passed": True,
            "custom_base_url_policy_check_passed": True,
            "credential_values_or_fingerprints_persisted": False,
        }
        if selected_profile is not None:
            metadata["credential_profile_id"] = selected_profile
        return metadata


@dataclass
class RunRecorder:
    """In-memory, secret-free projection of one provider/tool trajectory."""

    events: list[ChatMessage] = field(default_factory=list)
    provider_calls: list[dict[str, Any]] = field(default_factory=list)
    provider_requests: list[dict[str, Any]] = field(default_factory=list)
    tool_executions: list[dict[str, Any]] = field(default_factory=list)
    native_tool_outcomes: list[dict[str, Any]] = field(default_factory=list)
    iteration_limit_suppressed_tool_calls: list[dict[str, Any]] = field(
        default_factory=list
    )
    final_answer: str | None = None
    _pending_calls: list[FunctionCall] = field(default_factory=list, repr=False)
    _seen_provider_tool_call_ids: set[str] = field(default_factory=set, repr=False)
    _seen_native_tool_operation_ids: set[str] = field(default_factory=set, repr=False)

    def record_provider_request(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> None:
        rendered_tools = to_jsonable(list(tools))
        canonical_tools = json.dumps(
            rendered_tools,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.provider_requests.append(
            {
                "capture_stage": "llm_client_input_before_provider_normalization",
                "messages": to_jsonable(list(messages)),
                "message_roles": [str(message.get("role") or "") for message in messages],
                "tools": rendered_tools,
                "tool_names": [_openai_tool_name(tool) for tool in tools],
                "tools_sha256": hashlib.sha256(
                    canonical_tools.encode("utf-8")
                ).hexdigest(),
            }
        )

    def record_assistant(self, completion: LLMCompletion) -> None:
        tool_calls = [_function_call(value) for value in completion.tool_calls]
        rendered_tool_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            call_id = str(call.id or "").strip()
            function = str(call.function or "").strip()
            if not call_id or not function:
                raise PipelineRunError(
                    "formal provider tool calls require non-empty function and call IDs"
                )
            if call_id in self._seen_provider_tool_call_ids:
                raise PipelineRunError(
                    "formal provider response reused a tool-call ID"
                )
            self._seen_provider_tool_call_ids.add(call_id)
            rendered = call.model_dump(mode="json")
            rendered["raw_arguments_sha256"] = _canonical_sha256(
                dict(call.args)
            )
            rendered_tool_calls.append(rendered)
        content = (
            [text_content_block_from_string(completion.content)]
            if completion.content
            else None
        )
        self.events.append(
            ChatAssistantMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls or None,
            )
        )
        self._pending_calls.extend(tool_calls)
        call_index = len(self.provider_calls)
        provider_call = {
            "api": completion.api,
            # The request-selected model is recorded in
            # provider_request_options.requested_model.  Keep the
            # provider-authored response model under an explicit name because
            # compatible endpoints may return an alias or routed backend ID.
            "response_model": completion.model,
            # Compatibility alias for pre-existing artifact consumers.  New
            # verification must never use this field as requested-model
            # evidence.
            "model": completion.model,
            "response_id": completion.response_id,
            "request_id": completion.request_id,
            "usage": to_jsonable(completion.usage),
            "content": completion.content,
            "tool_calls": rendered_tool_calls,
            "fallback_json_action_used": completion.fallback_json_action_used,
            "compatibility_removed_options": list(
                completion.compatibility_removed_options
            ),
            "provider_request_options": to_jsonable(
                completion.provider_request_options
            ),
        }
        if call_index < len(self.provider_requests):
            provider_call["request"] = self.provider_requests[call_index]
        self.provider_calls.append(provider_call)

    def record_tool(
        self,
        *,
        function: str,
        args: Mapping[str, Any],
        formatted_result: str,
        raw_result: Any,
        error: str | None,
        runtime_tool_call_id: str | None = None,
        provider_tool_call_id: str | None = None,
        normalization_witness: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        call = self._take_pending(
            function,
            provider_tool_call_id=provider_tool_call_id,
        )
        selected_provider_call_id = provider_tool_call_id
        if not isinstance(selected_provider_call_id, str) or not selected_provider_call_id:
            raise PipelineRunError(
                "formal AgentDojo tool execution requires a provider tool-call ID"
            )
        witness = self._validated_normalization_witness(
            normalization_witness,
            function=function,
            provider_tool_call_id=selected_provider_call_id,
            runtime_tool_call_id=runtime_tool_call_id,
        )
        self.events.append(
            ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(formatted_result)],
                tool_call_id=selected_provider_call_id,
                tool_call=call,
                error=error,
            )
        )
        self.tool_executions.append(
            {
                "function": function,
                "args": to_jsonable(dict(args)),
                "result": to_jsonable(raw_result),
                "formatted_result": formatted_result,
                "error": error,
                "provider_tool_call_id": selected_provider_call_id,
                "runtime_tool_call_id": runtime_tool_call_id,
                "raw_arguments_sha256": witness["raw_arguments_sha256"],
                "schema_sha256": witness["schema_sha256"],
                "normalized_arguments_sha256": witness[
                    "normalized_arguments_sha256"
                ],
                "normalization_witness_sha256": witness["witness_sha256"],
                "normalization_witness": witness,
                "metadata": to_jsonable(dict(metadata or {})),
            }
        )

    def record_wrapper_failure(
        self,
        *,
        function: str,
        args: Mapping[str, Any],
        error_type: str,
        runtime_tool_call_id: str,
        provider_tool_call_id: str,
        normalization_witness: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a wrapper exception before the native terminal ledger exists.

        The durable result/audit/event binding is attached after the enclosing
        tool operation finishes.  Only the exception class crosses this
        boundary; exception text can contain provider or benchmark data.
        """

        call = self._take_pending(
            function,
            provider_tool_call_id=provider_tool_call_id,
        )
        witness = self._validated_normalization_witness(
            normalization_witness,
            function=function,
            provider_tool_call_id=provider_tool_call_id,
            runtime_tool_call_id=runtime_tool_call_id,
        )
        formatted = f"Native AgentDojo wrapper failed ({error_type})."
        self.events.append(
            ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(formatted)],
                tool_call_id=provider_tool_call_id,
                tool_call=call,
                error=formatted,
            )
        )
        self.tool_executions.append(
            {
                "function": function,
                "args": to_jsonable(dict(args)),
                "result": None,
                "formatted_result": formatted,
                "error": formatted,
                "provider_tool_call_id": provider_tool_call_id,
                "runtime_tool_call_id": runtime_tool_call_id,
                "raw_arguments_sha256": witness["raw_arguments_sha256"],
                "schema_sha256": witness["schema_sha256"],
                "normalized_arguments_sha256": witness[
                    "normalized_arguments_sha256"
                ],
                "normalization_witness_sha256": witness["witness_sha256"],
                "normalization_witness": witness,
                "metadata": {
                    "outcome_kind": "wrapper_exception",
                    "failure_phase": "wrapper_or_provider",
                    "native_admission_denial": False,
                    "committed_effect": False,
                    **dict(to_jsonable(dict(metadata or {}))),
                },
            }
        )

    def reconcile_native_tool_outcome(
        self,
        outcome: Mapping[str, Any],
        *,
        schema_sha256: str,
    ) -> None:
        """Attach one durable native terminal outcome to one assistant attempt."""

        selected = dict(to_jsonable(dict(outcome)))
        if not _validate_native_tool_terminal_outcome(selected):
            raise PipelineRunError("native tool terminal outcome evidence is invalid")
        operation_id = str(selected["operation_id"])
        if operation_id in self._seen_native_tool_operation_ids:
            prior = [
                item
                for item in self.native_tool_outcomes
                if item.get("operation_id") == operation_id
            ]
            if len(prior) == 1 and prior[0] == selected:
                return
            raise PipelineRunError(
                "duplicate native tool operation differs from recorded evidence"
            )
        provider_tool_call_id = str(selected["provider_tool_call_id"])
        runtime_tool_call_id = str(selected["runtime_tool_call_id"])
        function = str(selected["function"])
        raw_arguments_sha256 = str(selected["raw_arguments_sha256"])
        matches = [
            execution
            for execution in self.tool_executions
            if execution.get("provider_tool_call_id") == provider_tool_call_id
        ]
        if len(matches) > 1:
            raise PipelineRunError(
                "provider tool-call ID maps to multiple wrapper executions"
            )
        if matches:
            execution = matches[0]
            if (
                execution.get("function") != function
                or execution.get("runtime_tool_call_id") != runtime_tool_call_id
                or execution.get("raw_arguments_sha256") != raw_arguments_sha256
            ):
                raise PipelineRunError(
                    "wrapper execution differs from its native terminal outcome"
                )
            metadata = execution.get("metadata")
            if not isinstance(metadata, dict):
                raise PipelineRunError("wrapper execution metadata is malformed")
            metadata["native_terminal_outcome"] = selected
            metadata["native_terminal_outcome_sha256"] = selected[
                "binding_sha256"
            ]
        else:
            if selected["result"]["ok"] is not False:
                raise PipelineRunError(
                    "successful native tool outcome was not recorded by its wrapper"
                )
            call = self._take_pending(
                function,
                provider_tool_call_id=provider_tool_call_id,
            )
            if _canonical_sha256(dict(call.args)) != raw_arguments_sha256:
                raise PipelineRunError(
                    "native tool failure raw arguments differ from provider attempt"
                )
            if not _is_sha256(schema_sha256):
                raise PipelineRunError(
                    "native tool failure lacks its provider-visible schema hash"
                )
            failure_phase = str(selected.get("failure_phase") or "runtime")
            formatted = (
                "Agent libOS rejected the tool arguments before wrapper execution."
                if failure_phase == "input_validation"
                else "Agent libOS recorded a native tool execution failure."
            )
            self.events.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(formatted)],
                    tool_call_id=provider_tool_call_id,
                    tool_call=call,
                    error=formatted,
                )
            )
            self.tool_executions.append(
                {
                    "function": function,
                    # Normalization did not complete.  Preserve the exact raw
                    # provider object for attempt matching without calling it
                    # a normalized AgentDojo invocation.
                    "args": to_jsonable(dict(call.args)),
                    "result": {
                        "native_tool_failure": {
                            "operation_id": operation_id,
                            "result_oid": selected["result"].get("result_oid"),
                            "result_payload_sha256": selected["result"].get(
                                "payload_sha256"
                            ),
                        }
                    },
                    "formatted_result": formatted,
                    "error": formatted,
                    "provider_tool_call_id": provider_tool_call_id,
                    "runtime_tool_call_id": runtime_tool_call_id,
                    "raw_arguments_sha256": raw_arguments_sha256,
                    "schema_sha256": schema_sha256,
                    "normalized_arguments_sha256": None,
                    "normalization_witness_sha256": None,
                    "normalization_witness": {},
                    "metadata": {
                        "outcome_kind": "native_terminal_failure",
                        "failure_phase": failure_phase,
                        "provider_dispatched": False,
                        "provider_tool_call_id": provider_tool_call_id,
                        "runtime_tool_call_id": runtime_tool_call_id,
                        "llm_response_id": selected["llm_response_id"],
                        "native_admission_denial": False,
                        "committed_effect": False,
                        "native_terminal_outcome": selected,
                        "native_terminal_outcome_sha256": selected[
                            "binding_sha256"
                        ],
                    },
                }
            )
        self.native_tool_outcomes.append(selected)
        self._seen_native_tool_operation_ids.add(operation_id)

    @staticmethod
    def _validated_normalization_witness(
        normalization_witness: Mapping[str, Any],
        *,
        function: str,
        provider_tool_call_id: str,
        runtime_tool_call_id: str | None,
    ) -> dict[str, Any]:
        witness = dict(to_jsonable(dict(normalization_witness)))
        expected_witness_sha256 = witness.pop("witness_sha256", None)
        if (
            not _is_sha256(expected_witness_sha256)
            or _canonical_sha256(witness) != expected_witness_sha256
        ):
            raise PipelineRunError("AgentDojo normalization witness hash mismatch")
        witness["witness_sha256"] = expected_witness_sha256
        if (
            witness.get("function") != function
            or witness.get("provider_tool_call_id") != provider_tool_call_id
            or witness.get("runtime_tool_call_id") != runtime_tool_call_id
        ):
            raise PipelineRunError(
                "AgentDojo tool execution does not match its normalization witness"
            )
        return witness

    def normalization_witness(
        self,
        *,
        function: str,
        normalized_args: Mapping[str, Any],
        schema_sha256: str,
        context_metadata: Mapping[str, Any],
        runtime_tool_call_id: str | None,
    ) -> dict[str, Any]:
        """Bind one Host-captured raw call to its Pydantic-normalized object."""

        provider_tool_call_id = str(
            context_metadata.get("llm_tool_call_id") or ""
        ).strip()
        llm_response_id = str(
            context_metadata.get("llm_transcript_output_key") or ""
        ).strip()
        tool_name = str(context_metadata.get("llm_tool_name") or "").strip()
        raw_arguments_sha256 = str(
            context_metadata.get("llm_tool_raw_arguments_sha256") or ""
        ).strip()
        runtime_id = str(runtime_tool_call_id or "").strip()
        if (
            not provider_tool_call_id
            or not llm_response_id
            or not runtime_id
            or tool_name != function
            or not _is_sha256(raw_arguments_sha256)
            or not _is_sha256(schema_sha256)
        ):
            raise PipelineRunError(
                "formal AgentDojo call lacks complete Host identity/schema evidence"
            )
        call = self._pending_call(provider_tool_call_id)
        if call.function != function:
            raise PipelineRunError(
                "Host tool identity differs from the recorded provider call"
            )
        expected_raw_sha256 = _canonical_sha256(dict(call.args))
        if raw_arguments_sha256 != expected_raw_sha256:
            raise PipelineRunError(
                "Host raw argument hash differs from the recorded provider call"
            )
        matching_provider_calls = [
            provider_call
            for provider_call in self.provider_calls
            if any(
                isinstance(candidate, Mapping)
                and candidate.get("id") == provider_tool_call_id
                for candidate in provider_call.get("tool_calls") or []
            )
        ]
        if len(matching_provider_calls) != 1:
            raise PipelineRunError(
                "provider tool-call ID lacks one captured provider request"
            )
        request = matching_provider_calls[0].get("request")
        schema_matches: list[Mapping[str, Any]] = []
        if isinstance(request, Mapping):
            for tool in request.get("tools") or []:
                if not isinstance(tool, Mapping):
                    continue
                provider_function = tool.get("function")
                if (
                    isinstance(provider_function, Mapping)
                    and provider_function.get("name") == function
                    and isinstance(provider_function.get("parameters"), Mapping)
                ):
                    schema_matches.append(provider_function["parameters"])
        if (
            len(schema_matches) != 1
            or _canonical_sha256(dict(schema_matches[0])) != schema_sha256
        ):
            raise PipelineRunError(
                "AgentDojo schema hash differs from the captured provider schema"
            )
        normalized = dict(to_jsonable(dict(normalized_args)))
        witness = {
            "schema_version": _AGENTDOJO_NORMALIZATION_SCHEMA_VERSION,
            "normalizer": _AGENTDOJO_NORMALIZER,
            "function": function,
            "provider_tool_call_id": provider_tool_call_id,
            "runtime_tool_call_id": runtime_id,
            "llm_response_id": llm_response_id,
            "raw_arguments_sha256": raw_arguments_sha256,
            "schema_sha256": schema_sha256,
            "normalized_arguments_sha256": _canonical_sha256(normalized),
            "raw_call_sha256": _canonical_sha256(
                {"function": function, "args": dict(call.args)}
            ),
            "normalized_call_sha256": _canonical_sha256(
                {"function": function, "arguments": normalized}
            ),
        }
        witness["witness_sha256"] = _canonical_sha256(witness)
        return witness

    def _pending_call(self, provider_tool_call_id: str) -> FunctionCall:
        matches = [
            call
            for call in self._pending_calls
            if call.id == provider_tool_call_id
        ]
        if len(matches) != 1:
            raise PipelineRunError(
                "provider tool-call ID has no unique pending assistant attempt"
            )
        return matches[0]

    def usage(self) -> dict[str, int]:
        totals: Counter[str] = Counter()
        for call in self.provider_calls:
            usage = call.get("usage")
            if not isinstance(usage, Mapping):
                continue
            for key, value in usage.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    continue
                totals[str(key)] += value
        if "total_tokens" not in totals:
            prompt = totals.get("prompt_tokens", totals.get("input_tokens", 0))
            completion = totals.get(
                "completion_tokens", totals.get("output_tokens", 0)
            )
            if prompt or completion:
                totals["total_tokens"] = prompt + completion
        return dict(sorted(totals.items()))

    def _take_pending(
        self,
        function: str,
        *,
        provider_tool_call_id: str | None,
    ) -> FunctionCall:
        if not isinstance(provider_tool_call_id, str) or not provider_tool_call_id:
            raise PipelineRunError(
                "formal AgentDojo execution lacks a provider tool-call ID"
            )
        pending = self._pending_call(provider_tool_call_id)
        if pending.function != function:
            raise PipelineRunError(
                "executed function differs from its pending provider tool call"
            )
        for index, call in enumerate(self._pending_calls):
            if call is pending:
                return self._pending_calls.pop(index)
        raise PipelineRunError("pending provider tool call disappeared")


def _native_tool_terminal_outcome(
    host: Runtime,
    operation: Any,
) -> dict[str, Any]:
    """Project one payload-free, self-bound native tool terminal outcome."""

    identity = operation.metadata.get("provider_tool_identity")
    if not isinstance(identity, Mapping):
        raise PipelineRunError("native tool operation lacks provider identity")
    identity = dict(to_jsonable(dict(identity)))
    identity_sha256 = identity.pop("identity_sha256", None)
    if (
        identity.get("schema_version") != 1
        or not _is_sha256(identity_sha256)
        or _canonical_sha256(identity) != identity_sha256
    ):
        raise PipelineRunError("native tool operation provider identity is invalid")
    identity["identity_sha256"] = identity_sha256
    if (
        operation.state.value != "terminal"
        or operation.name != f"tool.{identity.get('function')}"
        or operation.pid != operation.actor
        or not isinstance(operation.operation_id, str)
        or not operation.operation_id
        or not isinstance(operation.root_operation_id, str)
        or not operation.root_operation_id
        or not isinstance(operation.parent_operation_id, str)
        or not operation.parent_operation_id
        or operation.root_operation_id != operation.parent_operation_id
        or operation.operation_id == operation.parent_operation_id
    ):
        raise PipelineRunError("native tool operation identity/state is invalid")

    raw_links = host.store.list_operation_evidence(
        operation_ids=[operation.operation_id]
    )
    links: list[dict[str, Any]] = []
    for link in raw_links:
        projected = {
            "link_id": link.link_id,
            "operation_id": link.operation_id,
            "evidence_type": link.evidence_type,
            "evidence_id": link.evidence_id,
            "role": link.role,
            "metadata": to_jsonable(link.metadata),
        }
        projected["link_sha256"] = _canonical_sha256(projected)
        links.append(projected)
    links.sort(
        key=lambda item: (
            str(item["evidence_type"]),
            str(item["role"]),
            str(item["evidence_id"]),
            str(item["link_id"]),
        )
    )
    invocation_links = [
        link
        for link in links
        if link["evidence_type"] == "tool_call" and link["role"] == "invocation"
    ]
    result_links = [
        link
        for link in links
        if link["evidence_type"] == "tool_call" and link["role"] == "result"
    ]
    if (
        len(invocation_links) != 1
        or len(result_links) != 1
        or invocation_links[0]["evidence_id"] != result_links[0]["evidence_id"]
    ):
        raise PipelineRunError(
            "native tool operation lacks one invocation/result identity pair"
        )
    runtime_tool_call_id = str(invocation_links[0]["evidence_id"])
    result_metadata = result_links[0].get("metadata")
    if (
        not runtime_tool_call_id
        or not isinstance(result_metadata, Mapping)
        or not isinstance(result_metadata.get("ok"), bool)
    ):
        raise PipelineRunError("native tool result evidence is malformed")
    result_oid = result_metadata.get("result_oid")
    if result_oid is not None and (
        not isinstance(result_oid, str) or not result_oid
    ):
        raise PipelineRunError("native tool result object identity is malformed")
    result_object = host.store.get_object(result_oid) if result_oid is not None else None
    if result_oid is not None and result_object is None:
        raise PipelineRunError("native tool result evidence references a missing Object")
    result_projection = {
        "ok": bool(result_metadata["ok"]),
        "result_oid": result_oid,
        "payload_sha256": (
            _canonical_sha256(to_jsonable(result_object.payload))
            if result_object is not None
            else None
        ),
        "object_type": (
            result_object.type.value if result_object is not None else None
        ),
        "object_version": (
            result_object.version if result_object is not None else None
        ),
        "object_immutable": (
            result_object.immutable if result_object is not None else None
        ),
    }

    audits: list[dict[str, Any]] = []
    for link in links:
        if link["evidence_type"] != "audit" or link["role"] != "audit":
            continue
        record = host.store.get_audit(str(link["evidence_id"]))
        if record is None:
            raise PipelineRunError("native tool operation references a missing audit")
        projected = {
            "record_id": record.record_id,
            "actor": record.actor,
            "action": record.action,
            "target": record.target,
            "input_refs": list(record.input_refs),
            "output_refs": list(record.output_refs),
            "capability_refs": list(record.capability_refs),
            "decision": to_jsonable(record.decision),
            "correlation_id": record.correlation_id,
            "parent_record_id": record.parent_record_id,
        }
        projected["audit_sha256"] = _canonical_sha256(projected)
        audits.append(projected)
    audits.sort(key=lambda item: str(item["record_id"]))

    events: list[dict[str, Any]] = []
    for link in links:
        if link["evidence_type"] != "event" or link["role"] != "event":
            continue
        event = host.store.get_event(str(link["evidence_id"]))
        if event is None:
            raise PipelineRunError("native tool operation references a missing event")
        projected = {
            "event_id": event.event_id,
            "type": event.type.value,
            "source": event.source,
            "target": event.target,
            "payload": to_jsonable(event.payload),
            "priority": event.priority.value,
            "correlation_id": event.correlation_id,
            "causality": to_jsonable(event.causality),
        }
        projected["event_sha256"] = _canonical_sha256(projected)
        events.append(projected)
    events.sort(key=lambda item: str(item["event_id"]))

    failure_phase = None
    if result_projection["ok"] is False:
        durable_payload = result_object.payload if result_object is not None else None
        error_payload = (
            durable_payload.get("error")
            if isinstance(durable_payload, Mapping)
            else None
        )
        failure_phase = (
            "input_validation"
            if (
                isinstance(durable_payload, Mapping)
                and durable_payload.get("policy_decision") == "validation_error"
            )
            or (
                isinstance(error_payload, Mapping)
                and error_payload.get("type") == "InputValidationError"
            )
            or any(
                audit.get("action") == "tool.call"
                and isinstance(audit.get("decision"), Mapping)
                and audit["decision"].get("policy_decision")
                == "validation_error"
                for audit in audits
            )
            else "wrapper_or_provider"
        )
    projection = {
        "schema_version": _NATIVE_TOOL_OUTCOME_SCHEMA_VERSION,
        "operation_id": operation.operation_id,
        "root_operation_id": operation.root_operation_id,
        "parent_operation_id": operation.parent_operation_id,
        "operation_name": operation.name,
        "operation_state": operation.state.value,
        "operation_outcome": operation.outcome.value,
        "pid": operation.pid,
        "function": identity["function"],
        "llm_response_id": identity["llm_response_id"],
        "provider_tool_call_id": identity["provider_tool_call_id"],
        "runtime_tool_call_id": runtime_tool_call_id,
        "raw_arguments_sha256": identity["raw_arguments_sha256"],
        "provider_identity_sha256": identity["identity_sha256"],
        "failure_phase": failure_phase,
        "result": result_projection,
        "audit_records": audits,
        "event_records": events,
        "operation_evidence_links": links,
    }
    projection["binding_sha256"] = _canonical_sha256(projection)
    if not _validate_native_tool_terminal_outcome(projection):
        raise PipelineRunError(
            "native tool terminal outcome failed self-validation: "
            f"result_ok={result_projection['ok']}, audits={len(audits)}, "
            f"events={[event['type'] for event in events]}, "
            f"link_roles={[(link['evidence_type'], link['role']) for link in links]}"
        )
    return projection


def _validate_native_tool_terminal_outcome(value: Mapping[str, Any]) -> bool:
    """Validate the one-to-one operation/result/audit/event terminal binding."""

    outcome = dict(to_jsonable(dict(value)))
    expected_fields = {
        "schema_version",
        "operation_id",
        "root_operation_id",
        "parent_operation_id",
        "operation_name",
        "operation_state",
        "operation_outcome",
        "pid",
        "function",
        "llm_response_id",
        "provider_tool_call_id",
        "runtime_tool_call_id",
        "raw_arguments_sha256",
        "provider_identity_sha256",
        "failure_phase",
        "result",
        "audit_records",
        "event_records",
        "operation_evidence_links",
        "binding_sha256",
    }
    if "query_invocation" in outcome:
        expected_fields.add("query_invocation")
    if set(outcome) != expected_fields:
        return False
    # Aggregation adds this outer shard coordinate after the native operation
    # has been sealed.  It is cross-checked against the paired execution and
    # provider call, not included in the operation-local binding hash.
    query_invocation = outcome.pop("query_invocation", None)
    if query_invocation is not None and (
        isinstance(query_invocation, bool)
        or not isinstance(query_invocation, int)
        or query_invocation <= 0
    ):
        return False
    binding_sha256 = outcome.pop("binding_sha256", None)
    expected_provider_identity = {
        "schema_version": 1,
        "llm_response_id": outcome.get("llm_response_id"),
        "provider_tool_call_id": outcome.get("provider_tool_call_id"),
        "function": outcome.get("function"),
        "raw_arguments_sha256": outcome.get("raw_arguments_sha256"),
    }
    if (
        outcome.get("schema_version") != _NATIVE_TOOL_OUTCOME_SCHEMA_VERSION
        or not _is_sha256(binding_sha256)
        or _canonical_sha256(outcome) != binding_sha256
        or outcome.get("operation_state") != "terminal"
        or outcome.get("operation_name") != f"tool.{outcome.get('function')}"
        or not all(
            isinstance(outcome.get(key), str) and bool(outcome.get(key))
            for key in (
                "operation_id",
                "root_operation_id",
                "parent_operation_id",
                "pid",
                "function",
                "llm_response_id",
                "provider_tool_call_id",
                "runtime_tool_call_id",
            )
        )
        or outcome.get("root_operation_id")
        != outcome.get("parent_operation_id")
        or outcome.get("operation_id") == outcome.get("parent_operation_id")
        or not _is_sha256(outcome.get("raw_arguments_sha256"))
        or not _is_sha256(outcome.get("provider_identity_sha256"))
        or _canonical_sha256(expected_provider_identity)
        != outcome.get("provider_identity_sha256")
    ):
        return False
    result = outcome.get("result")
    if (
        not isinstance(result, Mapping)
        or set(result)
        != {
            "ok",
            "result_oid",
            "payload_sha256",
            "object_type",
            "object_version",
            "object_immutable",
        }
        or not isinstance(result.get("ok"), bool)
        or (
            result.get("result_oid") is None
            and any(
                result.get(key) is not None
                for key in (
                    "payload_sha256",
                    "object_type",
                    "object_version",
                    "object_immutable",
                )
            )
        )
        or (
            result.get("result_oid") is not None
            and (
                not isinstance(result.get("result_oid"), str)
                or not result.get("result_oid")
                or not _is_sha256(result.get("payload_sha256"))
                or result.get("object_type") != "tool_result"
                or not isinstance(result.get("object_version"), int)
                or isinstance(result.get("object_version"), bool)
                or result.get("object_version") <= 0
                or result.get("object_immutable") is not True
            )
        )
    ):
        return False
    if result["ok"] is True and outcome.get("failure_phase") is not None:
        return False
    if result["ok"] is False and outcome.get("failure_phase") not in {
        "input_validation",
        "wrapper_or_provider",
    }:
        return False
    operation_outcome = outcome.get("operation_outcome")
    if (
        (result["ok"] is True and operation_outcome != "succeeded")
        or (
            result["ok"] is False
            and operation_outcome not in {"denied", "failed", "unknown"}
        )
    ):
        return False

    links = outcome.get("operation_evidence_links")
    if not isinstance(links, list) or not links:
        return False
    link_ids: list[str] = []
    for link in links:
        if not isinstance(link, Mapping):
            return False
        if set(link) != {
            "link_id",
            "operation_id",
            "evidence_type",
            "evidence_id",
            "role",
            "metadata",
            "link_sha256",
        }:
            return False
        unsigned = {key: item for key, item in link.items() if key != "link_sha256"}
        if (
            not isinstance(link.get("link_id"), str)
            or not link.get("link_id")
            or link.get("operation_id") != outcome.get("operation_id")
            or link.get("evidence_type") not in {"tool_call", "audit", "event"}
            or not isinstance(link.get("evidence_id"), str)
            or not link.get("evidence_id")
            or not isinstance(link.get("role"), str)
            or not link.get("role")
            or not isinstance(link.get("metadata"), Mapping)
            or not _is_sha256(link.get("link_sha256"))
            or _canonical_sha256(unsigned) != link.get("link_sha256")
        ):
            return False
        link_ids.append(str(link["link_id"]))
    if len(link_ids) != len(set(link_ids)):
        return False
    tool_links = [link for link in links if link.get("evidence_type") == "tool_call"]
    invocation_links = [
        link
        for link in tool_links
        if link.get("role") == "invocation"
    ]
    result_links = [
        link
        for link in tool_links
        if link.get("role") == "result"
    ]
    runtime_id = outcome["runtime_tool_call_id"]
    invocation_metadata = invocation_links[0].get("metadata") if invocation_links else None
    tool_id = (
        invocation_metadata.get("tool_id")
        if isinstance(invocation_metadata, Mapping)
        else None
    )
    tool_resource = f"tool:{tool_id}" if isinstance(tool_id, str) and tool_id else None
    if (
        len(tool_links) != 2
        or len(invocation_links) != 1
        or len(result_links) != 1
        or invocation_links[0].get("evidence_id") != runtime_id
        or not isinstance(invocation_metadata, Mapping)
        or set(invocation_metadata) != {"tool_id", "tool"}
        or invocation_metadata.get("tool") != outcome.get("function")
        or tool_resource is None
        or result_links[0].get("evidence_id") != runtime_id
        or result_links[0].get("metadata")
        != {"ok": result["ok"], "result_oid": result.get("result_oid")}
    ):
        return False

    audits = outcome.get("audit_records")
    if not isinstance(audits, list) or not audits:
        return False
    audit_record_ids: list[str] = []
    for audit in audits:
        if not isinstance(audit, Mapping):
            return False
        if set(audit) != {
            "record_id",
            "actor",
            "action",
            "target",
            "input_refs",
            "output_refs",
            "capability_refs",
            "decision",
            "correlation_id",
            "parent_record_id",
            "audit_sha256",
        }:
            return False
        unsigned_audit = {
            key: item for key, item in audit.items() if key != "audit_sha256"
        }
        if (
            not isinstance(audit.get("record_id"), str)
            or not audit.get("record_id")
            or not _is_sha256(audit.get("audit_sha256"))
            or _canonical_sha256(unsigned_audit) != audit.get("audit_sha256")
        ):
            return False
        audit_record_ids.append(str(audit["record_id"]))
    if len(audit_record_ids) != len(set(audit_record_ids)):
        return False
    audit_links = [
        link for link in links if link.get("evidence_type") == "audit"
    ]
    canonical_audit_links = [
        link for link in audit_links if link.get("role") == "audit"
    ]
    if (
        Counter(
            str(link["evidence_id"]) for link in canonical_audit_links
        )
        != Counter(audit_record_ids)
        or {
            str(link["evidence_id"]) for link in audit_links
        }
        != set(audit_record_ids)
        or any(
            link.get("role") == "audit"
            and link.get("evidence_type") != "audit"
            for link in links
        )
    ):
        return False
    tool_audits = [
        audit for audit in audits if audit.get("action") == "tool.call"
    ]
    if len(tool_audits) != 1:
        return False
    tool_audit = tool_audits[0]
    decision = tool_audit.get("decision")
    if (
        tool_audit.get("actor") != outcome.get("pid")
        or tool_audit.get("target") != tool_resource
        or not isinstance(decision, Mapping)
        or decision.get("ok") is not result["ok"]
        or decision.get("tool") != outcome.get("function")
        or tool_audit.get("output_refs")
        != ([result["result_oid"]] if result.get("result_oid") else [])
    ):
        return False
    resource_audits = [
        audit for audit in audits if audit.get("action") == "resource.charge"
    ]
    if len(resource_audits) != 1:
        return False
    resource_audit = resource_audits[0]
    resource_decision = resource_audit.get("decision")
    resource_context = (
        resource_decision.get("context")
        if isinstance(resource_decision, Mapping)
        else None
    )
    resource_usage = (
        resource_decision.get("usage")
        if isinstance(resource_decision, Mapping)
        else None
    )
    expected_resource_context = {
        "call_id": runtime_id,
        "tool": outcome.get("function"),
        "tool_id": tool_id,
    }
    resource_correlation_id = resource_audit.get("correlation_id")
    if (
        resource_audit.get("actor") != outcome.get("pid")
        or resource_audit.get("target") != f"process:{outcome.get('pid')}"
        or resource_audit.get("input_refs") != []
        or resource_audit.get("output_refs") != []
        or resource_audit.get("capability_refs") != []
        or not isinstance(resource_decision, Mapping)
        or set(resource_decision)
        != {"charged_pids", "context", "source", "usage"}
        or resource_decision.get("charged_pids") != [outcome.get("pid")]
        or resource_decision.get("source") != "tool.call"
        or not isinstance(resource_context, Mapping)
        or dict(resource_context) != expected_resource_context
        or not isinstance(resource_usage, Mapping)
        # Native tool resource charging currently records the runtime call ID
        # in decision.context and deliberately has no independent correlation
        # chain.  Accepting an attacker-supplied correlation string would add
        # an unbound semantic claim to the sealed projection.
        or resource_correlation_id is not None
    ):
        return False
    resource_audit_ids = [str(resource_audit["record_id"])]
    if (
        Counter(
            str(link["evidence_id"])
            for link in audit_links
            if link.get("role") == "result"
        )
        != Counter([str(tool_audit["record_id"])])
        or Counter(
            str(link["evidence_id"])
            for link in audit_links
            if link.get("role") == "resource_charge"
        )
        != Counter(resource_audit_ids)
        or any(
            link.get("role")
            not in {"audit", "decision", "result", "resource_charge"}
            for link in audit_links
        )
    ):
        return False

    events = outcome.get("event_records")
    if not isinstance(events, list):
        return False
    event_record_ids: list[str] = []
    for event in events:
        if not isinstance(event, Mapping):
            return False
        if set(event) != {
            "event_id",
            "type",
            "source",
            "target",
            "payload",
            "priority",
            "correlation_id",
            "causality",
            "event_sha256",
        }:
            return False
        unsigned_event = {
            key: item for key, item in event.items() if key != "event_sha256"
        }
        if (
            not isinstance(event.get("event_id"), str)
            or not event.get("event_id")
            or not _is_sha256(event.get("event_sha256"))
            or _canonical_sha256(unsigned_event) != event.get("event_sha256")
        ):
            return False
        event_record_ids.append(str(event["event_id"]))
    if len(event_record_ids) != len(set(event_record_ids)):
        return False
    event_links = [
        link for link in links if link.get("evidence_type") == "event"
    ]
    canonical_event_links = [
        link for link in event_links if link.get("role") == "event"
    ]
    if (
        Counter(
            str(link["evidence_id"]) for link in canonical_event_links
        )
        != Counter(event_record_ids)
        or {
            str(link["evidence_id"]) for link in event_links
        }
        != set(event_record_ids)
        or any(
            link.get("role") == "event"
            and link.get("evidence_type") != "event"
            for link in links
        )
    ):
        return False
    called = [
        event
        for event in events
        if event.get("type") == "tool_called"
        and isinstance(event.get("payload"), Mapping)
        and event["payload"].get("call_id") == runtime_id
    ]
    terminal_type = "tool_completed" if result["ok"] else "tool_failed"
    terminal = [
        event
        for event in events
        if event.get("type") == terminal_type
        and isinstance(event.get("payload"), Mapping)
        and event["payload"].get("call_id") == runtime_id
    ]
    resource_events = [
        event for event in events if event.get("type") == "resource_charged"
    ]
    if len(resource_events) != 1:
        return False
    resource_event = resource_events[0]
    resource_event_payload = resource_event.get("payload")
    if (
        resource_event.get("source") != "tool.call"
        or resource_event.get("target") != outcome.get("pid")
        or resource_event.get("correlation_id") != resource_correlation_id
        or not isinstance(resource_event_payload, Mapping)
        or set(resource_event_payload) != {"context", "pid", "usage"}
        or resource_event_payload.get("pid") != outcome.get("pid")
        or not isinstance(resource_event_payload.get("context"), Mapping)
        or dict(resource_event_payload["context"])
        != expected_resource_context
        or not isinstance(resource_event_payload.get("usage"), Mapping)
        or _canonical_sha256(resource_event_payload["usage"])
        != _canonical_sha256(resource_usage)
    ):
        return False
    resource_event_ids = [str(resource_event["event_id"])]
    return bool(
        len(called) == 1
        and len(terminal) == 1
        and called[0].get("source") == outcome.get("pid")
        and called[0].get("target") == tool_resource
        and terminal[0].get("source") == tool_resource
        and terminal[0].get("target") == outcome.get("pid")
        and isinstance(terminal[0].get("payload"), Mapping)
        and terminal[0]["payload"].get("result_oid")
        == result.get("result_oid")
        and Counter(
            str(link["evidence_id"])
            for link in event_links
            if link.get("role") == "resource_charge"
        )
        == Counter(resource_event_ids)
        and all(
            link.get("role") in {"event", "resource_charge"}
            for link in event_links
        )
    )


def _capture_native_tool_terminal_outcomes(
    host: Runtime,
    *,
    pid: str,
    recorder: RunRecorder,
    schema_sha256_by_function: Mapping[str, str],
) -> None:
    """Reconcile every new provider-authored native tool operation."""

    operations = sorted(
        host.store.list_operations(pid=pid),
        key=lambda operation: (operation.started_at, operation.operation_id),
    )
    for operation in operations:
        identity = operation.metadata.get("provider_tool_identity")
        if not isinstance(identity, Mapping):
            continue
        function = str(identity.get("function") or "")
        if function == HIDDEN_TERMINAL_TOOL:
            continue
        if operation.operation_id in recorder._seen_native_tool_operation_ids:
            continue
        schema_sha256 = schema_sha256_by_function.get(function)
        if not _is_sha256(schema_sha256):
            raise PipelineRunError(
                f"native tool outcome has no captured provider schema: {function}"
            )
        recorder.reconcile_native_tool_outcome(
            _native_tool_terminal_outcome(host, operation),
            schema_sha256=schema_sha256,
        )


def evaluation_config() -> AgentLibOSConfig:
    """Return the fixed model/runtime controls shared by both arms."""

    llm = replace(
        DEFAULT_CONFIG.llm,
        temperature=0.0,
        max_tokens=EVALUATION_MAX_COMPLETION_TOKENS,
        timeout_s=EVALUATION_TIMEOUT_S,
        max_retries=EVALUATION_MAX_RETRIES,
        api_mode="chat",
        parallel_tool_calls=False,
        auto_wait_on_empty_tool_calls=False,
    )
    return replace(DEFAULT_CONFIG, llm=llm)


def client_from_env(
    env_file: str | Path,
    *,
    config: AgentLibOSConfig,
    model_override: str | None = None,
) -> LLMClient:
    snapshot = capture_explicit_dotenv_environment(
        env_file,
        config=config,
        model_override=model_override,
    )
    return snapshot.new_client()


def capture_explicit_dotenv_environment(
    env_file: str | Path,
    *,
    config: AgentLibOSConfig,
    model_override: str | None = None,
) -> ExplicitDotenvSnapshot:
    """Resolve one stable dotenv/client snapshot before any artifact is created."""

    env_path = Path(env_file).absolute()
    try:
        initial_bytes = env_path.read_bytes()
    except OSError as exc:
        raise PipelineRunError(f"dotenv file does not exist: {env_path}") from exc
    initial_sha256 = hashlib.sha256(initial_bytes).hexdigest()
    dotenv = _read_dotenv_bytes(initial_bytes)
    _validate_explicit_dotenv_values(dotenv)
    selected_model_override = normalize_model_override(model_override)
    resolved_client = _client_from_captured_dotenv(dotenv, config=config)
    try:
        client_init_items = tuple(
            (item.name, copy.deepcopy(getattr(resolved_client, item.name)))
            for item in fields(LLMClient)
            if item.init
        )
    finally:
        if resolved_client is not None:
            resolved_client.close()
    expected_redaction_values = {
        value
        for name in _PRIVATE_PROVIDER_VALUE_NAMES
        if (value := dotenv.get(name))
    }
    return ExplicitDotenvSnapshot(
        path=env_path,
        _file_sha256=initial_sha256,
        model_override=selected_model_override,
        _dotenv_items=tuple(sorted(dotenv.items())),
        _client_init_items=client_init_items,
        _ambient_equality_check_passed=True,
        _redaction_check_passed=(
            set(_credential_redactions(dotenv)) == expected_redaction_values
        ),
        # LLMClient construction above validates that the selected endpoint is
        # official or that this exact captured dotenv explicitly opted into a
        # custom endpoint.  No ambient policy value is inherited.
        _custom_base_url_policy_check_passed=True,
    )


def normalize_model_override(value: str | None) -> str | None:
    """Validate a non-secret provider model label supplied by the host CLI."""

    if value is None:
        return None
    selected = str(value).strip()
    if not selected:
        raise PipelineRunError("model override must be a non-empty string")
    if len(selected) > 256:
        raise PipelineRunError("model override must be at most 256 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in selected):
        raise PipelineRunError("model override must not contain control characters")
    return selected


def _normalize_credential_profile_id(value: str | None) -> str | None:
    """Validate a protocol-owned, non-secret credential profile label."""

    if value is None:
        return None
    raw = str(value)
    selected = raw.strip()
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if (
        not selected
        or selected != raw
        or len(selected) > 128
        or not selected[0].isalnum()
        or any(character not in allowed for character in selected)
    ):
        raise PipelineRunError("credential profile ID is malformed")
    return selected


def _credential_redactions(dotenv: Mapping[str, str]) -> dict[str, str]:
    """Build private in-memory replacements without durable fingerprints."""

    return {
        value: replacement
        for name, replacement in _PRIVATE_PROVIDER_REDACTIONS
        if (value := dotenv.get(name))
        if value
    }


_PRIVATE_PROVIDER_REDACTIONS: tuple[tuple[str, str], ...] = (
    ("OPENAI_API_KEY", "[redacted-api-key]"),
    ("OPENAI_BASE_URL", "[redacted-endpoint]"),
    ("OPENAI_ORGANIZATION", "[redacted-organization]"),
    ("OPENAI_ORG_ID", "[redacted-organization]"),
    ("OPENAI_PROJECT", "[redacted-project]"),
    ("OPENAI_PROJECT_ID", "[redacted-project]"),
    ("OPENAI_SAFETY_IDENTIFIER", "[redacted-safety-identifier]"),
    ("OPENAI_PROMPT_CACHE_KEY", "[redacted-prompt-cache-key]"),
)
_PRIVATE_PROVIDER_VALUE_NAMES = frozenset(
    name for name, _replacement in _PRIVATE_PROVIDER_REDACTIONS
)


def validate_explicit_dotenv_environment(env_file: str | Path) -> dict[str, str]:
    """Require the selected dotenv file to be the effective OpenAI config.

    ``LLMClient.from_env`` intentionally gives the process environment precedence
    over a dotenv file.  Evaluation runs need a stronger provenance contract: an
    ambient setting may be present only when it is byte-for-byte identical to the
    selected file.  Reject conflicts before constructing a provider client, and
    never include secret values in the diagnostic.
    """

    env_path = Path(env_file)
    try:
        dotenv = _read_dotenv_bytes(env_path.read_bytes())
    except OSError as exc:
        raise PipelineRunError(f"dotenv file does not exist: {env_path}") from exc
    _validate_explicit_dotenv_values(dotenv)
    return dotenv


def _read_dotenv_bytes(raw: bytes) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in raw.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _validate_explicit_dotenv_values(dotenv: Mapping[str, str]) -> None:
    conflicts = sorted(
        key
        for key, value in os.environ.items()
        if (
            key.startswith("OPENAI_")
            or key == _CUSTOM_BASE_URL_POLICY_ENV
        )
        and dotenv.get(key) != value
    )
    if conflicts:
        rendered = ", ".join(conflicts)
        raise PipelineRunError(
            "ambient OpenAI configuration conflicts with the selected dotenv "
            f"file for: {rendered}"
        )


def _client_from_captured_dotenv(
    dotenv: Mapping[str, str],
    *,
    config: AgentLibOSConfig,
) -> LLMClient:
    """Mirror ``LLMClient.from_env`` without re-reading mutable process state."""

    env = dict(dotenv)
    defaults = config.llm
    allow_custom_base_url = llm_client_module._bool_env_from(
        env,
        _CUSTOM_BASE_URL_POLICY_ENV,
        default=False,
    )
    return LLMClient(
        base_url=env.get("OPENAI_BASE_URL"),
        model=env.get("OPENAI_LANGUAGE_MODEL") or env.get("OPENAI_MODEL"),
        api_key=env.get("OPENAI_API_KEY"),
        api_key_env="OPENAI_API_KEY",
        # These controls are fixed by the evaluation protocol.  Dotenv
        # values may describe a general client profile, but cannot weaken the
        # realized AgentDojo run configuration.
        timeout=EVALUATION_TIMEOUT_S,
        max_retries=EVALUATION_MAX_RETRIES,
        api_mode="chat",
        store=llm_client_module._bool_env_from(
            env,
            "OPENAI_STORE",
            default=defaults.store,
        ),
        reasoning_effort=llm_client_module._optional_env_from(
            env,
            "OPENAI_REASONING_EFFORT",
        ),
        verbosity=llm_client_module._verbosity_env_from(env, "OPENAI_VERBOSITY"),
        safety_identifier=(
            llm_client_module._optional_env_from(env, "OPENAI_SAFETY_IDENTIFIER")
            or defaults.safety_identifier
        ),
        prompt_cache_key=(
            llm_client_module._optional_env_from(env, "OPENAI_PROMPT_CACHE_KEY")
            or defaults.prompt_cache_key
        ),
        prompt_cache_retention=(
            llm_client_module._prompt_cache_retention_env_from(
                env,
                "OPENAI_PROMPT_CACHE_RETENTION",
            )
            or defaults.prompt_cache_retention
        ),
        responses_previous_response_id=llm_client_module._bool_env_from(
            env,
            "OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID",
            default=defaults.responses_previous_response_id,
        ),
        parallel_tool_calls=llm_client_module._bool_env_from(
            env,
            "OPENAI_PARALLEL_TOOL_CALLS",
            default=defaults.parallel_tool_calls,
        ),
        fallback_json_actions=llm_client_module._bool_env_from(
            env,
            "OPENAI_FALLBACK_JSON_ACTIONS",
            default=defaults.fallback_json_actions,
        ),
        enable_thinking=EVALUATION_ENABLE_THINKING,
        # The formal evaluation protocol requires this exact provider field.
        # Do not let the general-purpose client compatibility layer rename it
        # to ``max_tokens`` after a provider rejection.
        require_max_completion_tokens=True,
        organization=(
            llm_client_module._optional_env_from(env, "OPENAI_ORGANIZATION")
            or llm_client_module._optional_env_from(env, "OPENAI_ORG_ID")
        ),
        project=(
            llm_client_module._optional_env_from(env, "OPENAI_PROJECT")
            or llm_client_module._optional_env_from(env, "OPENAI_PROJECT_ID")
        ),
        inherit_ambient_openai_sdk_config=False,
        allow_custom_base_url=allow_custom_base_url,
        defaults=defaults,
    )


class SharedModelElement(BasePipelineElement):
    """AgentDojo model element backed by Agent libOS's provider client."""

    def __init__(
        self,
        client: LLMClient,
        *,
        recorder: RunRecorder,
        max_output_tokens: int,
        provider_guard: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.recorder = recorder
        self.max_output_tokens = max_output_tokens
        self.provider_guard = provider_guard
        self.name = str(client.model or "agent-libos-llm")

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        provider_messages = _dojo_messages_to_openai(messages)
        provider_tools = _dojo_tools_to_openai(runtime)
        self.recorder.record_provider_request(
            messages=provider_messages,
            tools=provider_tools,
        )
        if self.provider_guard is not None:
            self.provider_guard("before_provider_call")
        try:
            completion = self.client.complete_action(
                messages=provider_messages,
                tools=provider_tools,
                temperature=0.0,
                max_tokens=self.max_output_tokens,
                parallel_tool_calls=False,
            )
        finally:
            if self.provider_guard is not None:
                self.provider_guard("after_provider_call")
        self.recorder.record_assistant(completion)
        return query, runtime, env, [*messages, self.recorder.events[-1]], extra_args


class ControlPipeline(AgentPipeline):
    """AgentDojo-native tool loop using the same provider client as libOS."""

    def __init__(
        self,
        *,
        client: LLMClient,
        system_message: str,
        max_output_tokens: int,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        provider_guard: Callable[[str], None] | None = None,
    ) -> None:
        self.recorder = RunRecorder()
        self.model_element = SharedModelElement(
            client,
            recorder=self.recorder,
            max_output_tokens=max_output_tokens,
            provider_guard=provider_guard,
        )
        loop = ToolsExecutionLoop(
            [ToolsExecutor(tool_result_to_str), self.model_element],
            max_iters=max_tool_iterations,
        )
        super().__init__([SystemMessage(system_message), InitQuery(), self.model_element, loop])
        self.name = f"{client.model or 'unknown-model'}-upstream-control"
        self.last_run: dict[str, Any] = {}
        self._query_invocation_count = 0
        self._query_runs: list[dict[str, Any]] = []

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        self._query_invocation_count += 1
        query_invocation = self._query_invocation_count
        self.recorder = RunRecorder()
        self.model_element.recorder = self.recorder
        returned_messages: list[ChatMessage] = list(messages)
        error: BaseException | None = None
        try:
            outcome = super().query(query, runtime, env, messages, extra_args)
            returned_messages = list(outcome[3])
            return outcome
        except BaseException as exc:
            error = exc
            raise
        finally:
            query_run = {
                "arm": "upstream_control",
                "messages": to_jsonable(returned_messages),
                "logical_model_requests": list(self.recorder.provider_requests),
                "logical_model_invocation_count": len(
                    self.recorder.provider_requests
                ),
                "provider_calls": list(self.recorder.provider_calls),
                "usage": self.recorder.usage(),
                "provider_call_count": len(self.recorder.provider_calls),
                "tool_call_count": sum(
                    len(provider_call.get("tool_calls") or [])
                    for provider_call in self.recorder.provider_calls
                ),
                "executed_tool_call_count": sum(
                    message.get("role") == "tool"
                    for message in returned_messages
                ),
                "error_type": type(error).__name__ if error is not None else None,
                "error": str(error) if error is not None else None,
            }
            query_run["query_invocation"] = query_invocation
            self._query_runs.append(query_run)
            self.last_run = _aggregate_control_query_runs(self._query_runs)

    def close(self) -> None:
        self.model_element.client.close()


@dataclass
class TerminalCaptureLLMClient(LLMClient):
    """Hide the runtime-only terminal tool and capture natural final text."""

    recorder: RunRecorder = field(default_factory=RunRecorder, repr=False)
    terminal_tool_name: str = HIDDEN_TERMINAL_TOOL
    suppress_visible_tool_calls: bool = False
    provider_guard: Callable[[str], None] | None = field(default=None, repr=False)

    @classmethod
    def from_client(
        cls,
        client: LLMClient,
        *,
        recorder: RunRecorder,
    ) -> "TerminalCaptureLLMClient":
        kwargs = {
            item.name: getattr(client, item.name)
            for item in fields(LLMClient)
            if item.init
        }
        return cls(**kwargs, recorder=recorder)

    async def acomplete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        previous_response_id: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMCompletion:
        visible_tools = [
            tool
            for tool in tools
            if _openai_tool_name(tool) != self.terminal_tool_name
        ]
        self.recorder.record_provider_request(
            messages=messages,
            tools=visible_tools,
        )
        if self.provider_guard is not None:
            self.provider_guard("before_provider_call")
        try:
            completion = await super().acomplete_action(
                messages,
                visible_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                previous_response_id=previous_response_id,
                parallel_tool_calls=False,
            )
        finally:
            if self.provider_guard is not None:
                self.provider_guard("after_provider_call")
        return self._prepare_runtime_completion(completion)

    def _prepare_runtime_completion(self, completion: LLMCompletion) -> LLMCompletion:
        """Record the raw response and adapt terminal/iteration-limit actions."""

        self.recorder.record_assistant(copy.deepcopy(completion))
        if completion.tool_calls and self.suppress_visible_tool_calls:
            self.recorder.iteration_limit_suppressed_tool_calls.extend(
                to_jsonable(completion.tool_calls)
            )
            self.recorder.final_answer = completion.content
            completion.tool_calls = [self._terminal_tool_call(completion.content)]
            return completion
        if not completion.tool_calls:
            self.recorder.final_answer = completion.content
            completion.tool_calls = [self._terminal_tool_call(completion.content)]
        return completion

    def _terminal_tool_call(self, content: str) -> dict[str, Any]:
        return {
            "id": f"agentdojo-final-{len(self.recorder.provider_calls)}",
            "name": self.terminal_tool_name,
            "arguments": json.dumps(
                {"content": content},
                ensure_ascii=False,
            ),
        }


class _TerminalArgs(BaseModel):
    content: str = Field(description="The assistant's natural final response.")


class HiddenTerminalTool(SyncAgentTool[_TerminalArgs]):
    name = HIDDEN_TERMINAL_TOOL
    description = "Runtime-internal terminal carrier; never exposed to the model."
    args_schema = _TerminalArgs
    policy = ToolPolicy(side_effects=False, idempotent=True)
    tags = ["evaluation", "internal"]

    def __init__(self, recorder: RunRecorder) -> None:
        self.recorder = recorder

    def run(self, args: _TerminalArgs, ctx: ToolContext) -> ToolResult:
        self.recorder.final_answer = args.content
        return ToolResult.success(
            data={"captured": True},
            model_data="",
        )


class AgentDojoFunctionTool(SyncAgentTool[BaseModel]):
    """Exact-name/schema wrapper around an AgentDojo Function."""

    name = "agentdojo_placeholder"
    description = "AgentDojo function bridge placeholder."
    args_schema = BaseModel
    policy = ToolPolicy(side_effects=False, idempotent=False)
    tags = ["evaluation", "agentdojo", "native-semantics"]

    def __init__(
        self,
        function: Function,
        *,
        dojo_runtime: FunctionsRuntime,
        env: Any,
        recorder: RunRecorder,
    ) -> None:
        self.function = function
        self.name = function.name
        self.description = function.description
        self.args_schema = _agentdojo_compatible_parameters(function)
        provider_schema, _strict = normalize_openai_strict_schema(
            self.args_schema.model_json_schema()
        )
        self.schema_sha256 = _canonical_sha256(provider_schema)
        self.dojo_runtime = dojo_runtime
        self.env = env
        self.recorder = recorder
        self.metadata = {
            "evaluation_semantics": "ambient_native_semantics",
            "agentdojo_function": function.name,
        }

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        arguments = args.model_dump(mode="python")
        context_metadata = (
            dict(ctx.metadata) if isinstance(ctx.metadata, Mapping) else {}
        )
        runtime_tool_call_id = str(ctx.call_id or "").strip() or None
        witness = self.recorder.normalization_witness(
            function=self.function.name,
            normalized_args=arguments,
            schema_sha256=self.schema_sha256,
            context_metadata=context_metadata,
            runtime_tool_call_id=runtime_tool_call_id,
        )
        provider_tool_call_id = str(
            witness["provider_tool_call_id"]
        )
        try:
            raw_result, error = self.dojo_runtime.run_function(
                self.env,
                self.function.name,
                arguments,
            )
        except Exception as exc:
            self.recorder.record_wrapper_failure(
                function=self.function.name,
                args=arguments,
                error_type=type(exc).__name__,
                runtime_tool_call_id=str(runtime_tool_call_id),
                provider_tool_call_id=provider_tool_call_id,
                normalization_witness=witness,
                metadata={
                    "provider_dispatched": True,
                    "provider_tool_call_id": provider_tool_call_id,
                    "runtime_tool_call_id": runtime_tool_call_id,
                    "llm_response_id": witness["llm_response_id"],
                    "normalization_witness_sha256": witness["witness_sha256"],
                },
            )
            raise
        selected_error = str(error) if error is not None else None
        formatted = selected_error or tool_result_to_str(raw_result)
        execution_metadata = {
            "outcome_kind": "wrapper_result",
            "provider_dispatched": True,
            "provider_tool_call_id": provider_tool_call_id,
            "runtime_tool_call_id": runtime_tool_call_id,
            "llm_response_id": witness["llm_response_id"],
            "normalization_witness_sha256": witness["witness_sha256"],
        }
        self.recorder.record_tool(
            function=self.function.name,
            args=arguments,
            formatted_result=formatted,
            raw_result=raw_result,
            error=selected_error,
            runtime_tool_call_id=runtime_tool_call_id,
            provider_tool_call_id=provider_tool_call_id,
            normalization_witness=witness,
            metadata=execution_metadata,
        )
        return ToolResult.success(
            content=formatted,
            data={
                "agentdojo_result": to_jsonable(raw_result),
                "agentdojo_error": error,
            },
            model_data=formatted,
            metadata={
                "agentdojo_error": error,
                "evaluation_semantics": "ambient_native_semantics",
                **execution_metadata,
            },
        )


def _agentdojo_compatible_parameters(function: Function) -> type[BaseModel]:
    """Preserve AgentDojo's pre-validation string-list coercion.

    AgentDojo's native ``ToolsExecutor`` converts arguments such as
    ``'["a@example.com"]'`` to a Python list before ``FunctionsRuntime`` runs
    Pydantic validation.  Agent libOS validates a tool call before invoking the
    wrapper, so the adapter must perform the same conversion in its argument
    model.  Retaining the original schema title keeps the provider-visible JSON
    schema byte-for-byte equivalent to the upstream function schema.
    """

    parameters = function.parameters

    def normalize_string_lists(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            key: (
                literal_eval(item)
                if isinstance(item, str) and is_string_list(item)
                else item
            )
            for key, item in value.items()
        }

    schema_title = str(
        parameters.model_json_schema().get("title") or parameters.__name__
    )
    return create_model(
        f"{parameters.__name__}AgentLibOSAdapter",
        __base__=parameters,
        __config__=ConfigDict(title=schema_title),
        __validators__={
            "normalize_agentdojo_string_lists": model_validator(mode="before")(
                normalize_string_lists
            )
        },
    )


ClientFactory = Callable[[RunRecorder], TerminalCaptureLLMClient]


class AgentLibOSAmbientPipeline(BasePipelineElement):
    """Run AgentDojo functions through the native Agent libOS loop.

    This arm intentionally grants ambient suite-wide tool authority. It tests
    integration and behavior parity, not capability/approval containment.
    """

    def __init__(
        self,
        *,
        client_factory: ClientFactory,
        system_message: str,
        runtime_dir: str | Path,
        config: AgentLibOSConfig,
        max_quanta: int = DEFAULT_MAX_TOOL_ITERATIONS + 1,
        prompt_mode: str = PROMPT_MODE_IMAGE_ONLY,
        provider_guard: Callable[[str], None] | None = None,
    ) -> None:
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"unknown Agent libOS prompt mode: {prompt_mode}")
        self.client_factory = client_factory
        self.system_message = system_message
        self.runtime_dir = Path(runtime_dir)
        self.config = config
        self.max_quanta = max_quanta
        self.prompt_mode = prompt_mode
        self.provider_guard = provider_guard
        self.name = "agent-libos-ambient-native-semantics"
        self.last_run: dict[str, Any] = {}
        self._query_invocation_count = 0
        self._query_runs: list[dict[str, Any]] = []

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        self._query_invocation_count += 1
        query_invocation = self._query_invocation_count
        runtime_subdir = f"query-{query_invocation:03d}"
        recorder = RunRecorder()
        client = self.client_factory(recorder)
        client.provider_guard = self.provider_guard
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        query_runtime_dir = self.runtime_dir / runtime_subdir
        query_runtime_dir.mkdir()
        workspace = query_runtime_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        store = SQLiteStore(query_runtime_dir / "runtime.sqlite")
        host: Runtime | None = None
        pid: str | None = None
        error: BaseException | None = None
        results: list[Any] = []
        replaced_tool_names: list[str] = []
        try:
            host = Runtime(
                store,
                llm_client=client,
                substrate=LocalResourceProviderSubstrate(workspace),
                config=self.config,
            )
            function_tools = [
                AgentDojoFunctionTool(
                    function,
                    dojo_runtime=runtime,
                    env=env,
                    recorder=recorder,
                )
                for function in runtime.functions.values()
            ]
            schema_sha256_by_function = {
                tool.name: tool.schema_sha256 for tool in function_tools
            }
            for tool in function_tools:
                try:
                    existing = host.tools.resolve(tool.name)
                except NotFound:
                    continue
                if not host.tools.unregister_tool(existing):
                    raise PipelineRunError(
                        f"failed to replace colliding runtime tool: {tool.name}"
                    )
                replaced_tool_names.append(tool.name)
            for tool in [*function_tools, HiddenTerminalTool(recorder)]:
                host.tools.register_tool(
                    tool,
                    registered_by="agentdojo-evaluation",
                    ephemeral=True,
                )
            host.register_image(
                AgentImage(
                    image_id=_IMAGE_ID,
                    name="agentdojo-native-semantics",
                    system_prompt=self.system_message,
                    prompt_mode=self.prompt_mode,
                    default_tools=[
                        *(function.name for function in runtime.functions.values()),
                        HIDDEN_TERMINAL_TOOL,
                    ],
                    metadata={
                        "evaluation": "agentdojo",
                        "semantics": "ambient_native_semantics",
                        "prompt_mode": self.prompt_mode,
                        "hidden_terminal_tool": HIDDEN_TERMINAL_TOOL,
                    },
                ),
                actor="agentdojo-evaluation",
            )
            pid = host.process.spawn(image=_IMAGE_ID, goal=query)
            for quantum in range(self.max_quanta):
                # AgentDojo's native loop executes at most ``max_iters`` tool
                # calls and then performs one final model observation whose
                # tool calls remain unexecuted.  Agent libOS normally executes
                # an action in every quantum, so suppress a visible tool only
                # on that final observation quantum.  Natural final text still
                # travels through the runtime-only terminal carrier.
                client.suppress_visible_tool_calls = quantum == self.max_quanta - 1
                try:
                    quantum_results = host.run_process_until_idle(
                        pid,
                        max_quanta=1,
                    )
                finally:
                    _capture_native_tool_terminal_outcomes(
                        host,
                        pid=pid,
                        recorder=recorder,
                        schema_sha256_by_function=schema_sha256_by_function,
                    )
                results.extend(quantum_results)
                if recorder.final_answer is not None:
                    break
                if host.process.get(pid).status in host.process.TERMINAL_STATUSES:
                    break
            process = host.process.get(pid)
            status_before_host_exit = process.status.value
            if (
                process.status in host.process.TERMINAL_STATUSES
                and process.status != ProcessStatus.EXITED
            ):
                raise PipelineRunError(
                    "Agent libOS terminated before the host could commit the captured "
                    f"final response: status={process.status.value}"
                )
            if process.status not in host.process.TERMINAL_STATUSES:
                host.process.exit(
                    pid,
                    message=recorder.final_answer,
                )
            process = host.process.get(pid)
            if process.status != ProcessStatus.EXITED:
                raise PipelineRunError(
                    f"Agent libOS final process status is not exited: {process.status.value}"
                )
            returned_messages: list[ChatMessage] = [
                {
                    "role": "system",
                    "content": [text_content_block_from_string(self.system_message)],
                },
                {
                    "role": "user",
                    "content": [text_content_block_from_string(query)],
                },
                *recorder.events,
            ]
            if returned_messages[-1]["role"] != "assistant":
                raise PipelineRunError("Agent libOS trace did not end with an assistant response")
            query_run = _ambient_run_snapshot(
                host,
                pid=pid,
                results=results,
                recorder=recorder,
                status_before_host_exit=status_before_host_exit,
                replaced_tool_names=replaced_tool_names,
                prompt_mode=self.prompt_mode,
                transcript_messages=returned_messages,
                error=None,
            )
            self._record_query_run(
                query_run,
                query_invocation=query_invocation,
                runtime_subdir=runtime_subdir,
            )
            returned_extra = dict(extra_args)
            returned_extra["agent_libos_dojo"] = {
                "arm": "libos_ambient",
                "pid": pid,
                "usage": recorder.usage(),
            }
            return query, runtime, env, returned_messages, returned_extra
        except BaseException as exc:
            error = exc
            if host is not None and pid is not None:
                query_run = _ambient_run_snapshot(
                    host,
                    pid=pid,
                    results=results,
                    recorder=recorder,
                    status_before_host_exit=None,
                    replaced_tool_names=replaced_tool_names,
                    prompt_mode=self.prompt_mode,
                    transcript_messages=[
                        {
                            "role": "system",
                            "content": [
                                text_content_block_from_string(self.system_message)
                            ],
                        },
                        {
                            "role": "user",
                            "content": [text_content_block_from_string(query)],
                        },
                        *recorder.events,
                    ],
                    error=exc,
                )
            else:
                query_run = {
                    "arm": "libos_ambient",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "provider_calls": list(recorder.provider_calls),
                    "usage": recorder.usage(),
                }
            self._record_query_run(
                query_run,
                query_invocation=query_invocation,
                runtime_subdir=runtime_subdir,
            )
            raise
        finally:
            if host is not None:
                host.close()
            else:
                store.close()
            if error is not None and not self.last_run:
                self._record_query_run(
                    {
                        "arm": "libos_ambient",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    query_invocation=query_invocation,
                    runtime_subdir=runtime_subdir,
                )

    def _record_query_run(
        self,
        query_run: Mapping[str, Any],
        *,
        query_invocation: int,
        runtime_subdir: str,
    ) -> None:
        selected = dict(query_run)
        selected["query_invocation"] = query_invocation
        selected["runtime_subdir"] = runtime_subdir
        self._query_runs.append(selected)
        self.last_run = _aggregate_ambient_query_runs(self._query_runs)


class _DeferredContainedExecutor:
    """Let image validation see exact tools before the task PID exists."""

    def __init__(
        self,
        *,
        suite: str,
        catalog: FunctionPolicyCatalog,
        authority: CompiledTaskAuthority,
    ) -> None:
        self.suite = suite
        self.catalog = catalog
        self.authority = authority
        self._delegate: ContainedOperationExecutor | None = None

    def bind(self, runtime: Runtime, pid: str) -> None:
        if self._delegate is not None:
            raise PipelineRunError("contained executor was bound more than once")
        self._delegate = ContainedOperationExecutor(
            runtime=runtime,
            pid=pid,
            suite=self.suite,
            catalog=self.catalog,
            authority=self.authority,
        )

    @property
    def runtime(self) -> Runtime:
        """Expose the bound runtime without permitting pre-bind access.

        The contained tool records the post-dispatch IFC source context after
        the native executor returns.  Image construction needs this deferred
        proxy before a process exists, so the proxy must forward that read only
        after ``bind`` has installed the real executor.
        """

        if self._delegate is None:
            raise PipelineRunError("contained executor runtime used before PID binding")
        return self._delegate.runtime

    def execute(self, **kwargs: Any) -> Any:
        if self._delegate is None:
            raise PipelineRunError("contained executor used before PID binding")
        return self._delegate.execute(**kwargs)


class AgentLibOSContainedPipeline(AgentLibOSAmbientPipeline):
    """Run one natural AgentDojo trajectory under exact native authority.

    The compiler input is supplied by the Host before any attacked environment
    or injection string exists.  Tool visibility remains identical to the two
    behavioral arms; only native Task Authority, capability admission and IFC
    Sink clearance differ.
    """

    def __init__(
        self,
        *,
        client_factory: ClientFactory,
        system_message: str,
        runtime_dir: str | Path,
        config: AgentLibOSConfig,
        suite: str,
        catalog: FunctionPolicyCatalog,
        authority: CompiledTaskAuthority,
        max_quanta: int = DEFAULT_MAX_TOOL_ITERATIONS + 1,
        prompt_mode: str = PROMPT_MODE_IMAGE_ONLY,
        provider_guard: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(
            client_factory=client_factory,
            system_message=system_message,
            runtime_dir=runtime_dir,
            config=config,
            max_quanta=max_quanta,
            prompt_mode=prompt_mode,
            provider_guard=provider_guard,
        )
        metadata = authority.manifest.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("suite") != suite:
            raise ValueError("contained authority does not match the selected suite")
        if authority.policy_sha256 != catalog.sha256:
            raise ValueError("contained authority and function catalog hashes differ")
        self.suite = suite
        self.catalog = catalog
        self.authority = authority
        self.name = "agent-libos-native-contained"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        del messages
        self._query_invocation_count += 1
        query_invocation = self._query_invocation_count
        runtime_subdir = f"query-{query_invocation:03d}"
        recorder = RunRecorder()
        client = self.client_factory(recorder)
        client.provider_guard = self.provider_guard
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        query_runtime_dir = self.runtime_dir / runtime_subdir
        query_runtime_dir.mkdir()
        workspace = query_runtime_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        store = SQLiteStore(query_runtime_dir / "runtime.sqlite")
        host: Runtime | None = None
        pid: str | None = None
        error: BaseException | None = None
        results: list[Any] = []
        replaced_tool_names: list[str] = []
        model_sink: Any | None = None
        model_sink_rule: Any | None = None
        try:
            host = Runtime(
                store,
                llm_client=client,
                substrate=LocalResourceProviderSubstrate(workspace),
                config=self.config,
            )
            registry_actor = host.process.spawn(
                goal="configure exact AgentDojo containment Sinks"
            )
            host.capability.grant(
                registry_actor,
                host.config.data_flow.registry_resource,
                [CapabilityRight.ADMIN],
                issued_by="agentdojo.contained.host",
            )
            register_compiled_sinks(
                host,
                self.authority,
                registry_actor=registry_actor,
            )
            deferred_executor = _DeferredContainedExecutor(
                suite=self.suite,
                catalog=self.catalog,
                authority=self.authority,
            )
            function_tools = [
                ContainedAgentDojoFunctionTool(
                    function,
                    args_schema=_agentdojo_compatible_parameters(function),
                    dojo_runtime=runtime,
                    environment=env,
                    recorder=recorder,
                    executor=deferred_executor,  # type: ignore[arg-type]
                )
                for function in runtime.functions.values()
            ]
            schema_sha256_by_function = {
                tool.name: tool.schema_sha256 for tool in function_tools
            }
            for tool in function_tools:
                try:
                    existing = host.tools.resolve(tool.name)
                except NotFound:
                    continue
                if not host.tools.unregister_tool(existing):
                    raise PipelineRunError(
                        f"failed to replace colliding runtime tool: {tool.name}"
                    )
                replaced_tool_names.append(tool.name)
            for tool in [*function_tools, HiddenTerminalTool(recorder)]:
                host.tools.register_tool(
                    tool,
                    registered_by="agentdojo-contained-evaluation",
                    ephemeral=True,
                )
            host.register_image(
                AgentImage(
                    image_id=_CONTAINED_IMAGE_ID,
                    name="agentdojo-native-contained",
                    system_prompt=self.system_message,
                    prompt_mode=self.prompt_mode,
                    default_tools=[
                        *(function.name for function in runtime.functions.values()),
                        HIDDEN_TERMINAL_TOOL,
                    ],
                    metadata={
                        "evaluation": "agentdojo",
                        "semantics": "native_capability_ifc_contained",
                        "prompt_mode": self.prompt_mode,
                        "hidden_terminal_tool": HIDDEN_TERMINAL_TOOL,
                        "authority_manifest_template_sha256": (
                            self.authority.manifest_template_sha256
                        ),
                        "function_policy_sha256": self.authority.policy_sha256,
                    },
                ),
                actor="agentdojo-contained-evaluation",
            )
            pid = host.process.spawn(
                image=_CONTAINED_IMAGE_ID,
                goal=query,
                authority_manifest=self.authority.manifest,
            )
            # The LLM processing Sink is derived from the actual process-bound
            # profile and frozen provider configuration.  It is never a dummy
            # provider label or wildcard.
            model_sink = configured_model_processing_sink(host, pid)
            model_sink_rule = register_model_processing_sink(
                host,
                model_sink,
                registry_actor=registry_actor,
            )
            deferred_executor.bind(host, pid)
            for quantum in range(self.max_quanta):
                client.suppress_visible_tool_calls = quantum == self.max_quanta - 1
                try:
                    quantum_results = host.run_process_until_idle(
                        pid,
                        max_quanta=1,
                    )
                finally:
                    _capture_native_tool_terminal_outcomes(
                        host,
                        pid=pid,
                        recorder=recorder,
                        schema_sha256_by_function=schema_sha256_by_function,
                    )
                results.extend(quantum_results)
                if recorder.final_answer is not None:
                    break
                if host.process.get(pid).status in host.process.TERMINAL_STATUSES:
                    break
            process = host.process.get(pid)
            status_before_host_exit = process.status.value
            if (
                process.status in host.process.TERMINAL_STATUSES
                and process.status != ProcessStatus.EXITED
            ):
                raise PipelineRunError(
                    "contained Agent libOS terminated before the Host could commit "
                    f"the captured final response: status={process.status.value}"
                )
            if process.status not in host.process.TERMINAL_STATUSES:
                host.process.exit(pid, message=recorder.final_answer)
            process = host.process.get(pid)
            if process.status != ProcessStatus.EXITED:
                raise PipelineRunError(
                    "contained Agent libOS final process status is not exited: "
                    f"{process.status.value}"
                )
            returned_messages: list[ChatMessage] = [
                {
                    "role": "system",
                    "content": [text_content_block_from_string(self.system_message)],
                },
                {
                    "role": "user",
                    "content": [text_content_block_from_string(query)],
                },
                *recorder.events,
            ]
            if returned_messages[-1]["role"] != "assistant":
                raise PipelineRunError(
                    "contained Agent libOS trace did not end with an assistant response"
                )
            query_run = _contained_run_snapshot(
                host,
                pid=pid,
                results=results,
                recorder=recorder,
                status_before_host_exit=status_before_host_exit,
                replaced_tool_names=replaced_tool_names,
                prompt_mode=self.prompt_mode,
                transcript_messages=returned_messages,
                authority=self.authority,
                model_sink=model_sink,
                model_sink_rule=model_sink_rule,
                error=None,
            )
            self._record_query_run(
                query_run,
                query_invocation=query_invocation,
                runtime_subdir=runtime_subdir,
            )
            returned_extra = dict(extra_args)
            returned_extra["agent_libos_dojo"] = {
                "arm": "libos_contained",
                "pid": pid,
                "usage": recorder.usage(),
                "authority_manifest_template_sha256": (
                    self.authority.manifest_template_sha256
                ),
            }
            return query, runtime, env, returned_messages, returned_extra
        except BaseException as exc:
            error = exc
            if host is not None and pid is not None:
                query_run = _contained_run_snapshot(
                    host,
                    pid=pid,
                    results=results,
                    recorder=recorder,
                    status_before_host_exit=None,
                    replaced_tool_names=replaced_tool_names,
                    prompt_mode=self.prompt_mode,
                    transcript_messages=[
                        {
                            "role": "system",
                            "content": [
                                text_content_block_from_string(self.system_message)
                            ],
                        },
                        {
                            "role": "user",
                            "content": [text_content_block_from_string(query)],
                        },
                        *recorder.events,
                    ],
                    authority=self.authority,
                    model_sink=model_sink,
                    model_sink_rule=model_sink_rule,
                    error=exc,
                )
            else:
                query_run = {
                    "arm": "libos_contained",
                    "semantics": "native_capability_ifc_contained",
                    "authority_manifest_template_sha256": (
                        self.authority.manifest_template_sha256
                    ),
                    "function_policy_sha256": self.authority.policy_sha256,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "provider_calls": list(recorder.provider_calls),
                    "usage": recorder.usage(),
                }
            self._record_query_run(
                query_run,
                query_invocation=query_invocation,
                runtime_subdir=runtime_subdir,
            )
            raise
        finally:
            if host is not None:
                host.close()
            else:
                store.close()
            if error is not None and not self.last_run:
                self._record_query_run(
                    {
                        "arm": "libos_contained",
                        "semantics": "native_capability_ifc_contained",
                        "authority_manifest_template_sha256": (
                            self.authority.manifest_template_sha256
                        ),
                        "function_policy_sha256": self.authority.policy_sha256,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    query_invocation=query_invocation,
                    runtime_subdir=runtime_subdir,
                )


def make_terminal_client_factory(
    snapshot: ExplicitDotenvSnapshot,
) -> ClientFactory:
    def factory(recorder: RunRecorder) -> TerminalCaptureLLMClient:
        base = snapshot.new_client()
        return TerminalCaptureLLMClient.from_client(base, recorder=recorder)

    return factory


def _ambient_run_snapshot(
    host: Runtime,
    *,
    pid: str,
    results: Sequence[Any],
    recorder: RunRecorder,
    status_before_host_exit: str | None,
    replaced_tool_names: Sequence[str],
    prompt_mode: str,
    transcript_messages: Sequence[ChatMessage],
    error: BaseException | None,
) -> dict[str, Any]:
    process = host.process.get(pid)
    calls = sorted(
        host.store.list_llm_calls(
            pid=pid,
            limit=host.config.llm.call_record_hard_limit,
        ),
        key=lambda call: (call.created_at, call.call_id),
    )
    audit = host.audit.trace(actor=pid)
    audit_counts = Counter(record.action for record in audit)
    provider_calls = [dict(item) for item in recorder.provider_calls]
    for index, provider_call in enumerate(provider_calls):
        if index < len(calls):
            provider_call["llm_transcript_output_key"] = calls[index].call_id
    return {
        "arm": "libos_ambient",
        "semantics": "ambient_native_semantics",
        "prompt_mode": prompt_mode,
        "pid": pid,
        "process_status": process.status.value,
        "status_before_host_exit": status_before_host_exit,
        "status_message": process.status_message,
        "scheduler_result_count": len(results),
        "logical_model_invocation_count": len(recorder.provider_requests),
        "provider_call_count": len(recorder.provider_calls),
        "llm_call_record_count": len(calls),
        "tool_call_count": sum(
            len(provider_call.get("tool_calls") or [])
            for provider_call in recorder.provider_calls
        ),
        "executed_tool_call_count": len(recorder.tool_executions),
        "hidden_terminal_tool_calls": 1 if recorder.final_answer is not None else 0,
        "replaced_runtime_tool_names": sorted(replaced_tool_names),
        "usage": recorder.usage(),
        "logical_model_requests": list(recorder.provider_requests),
        "provider_calls": provider_calls,
        "tool_executions": list(recorder.tool_executions),
        "native_tool_outcome_evidence_schema_version": (
            _NATIVE_TOOL_OUTCOME_SCHEMA_VERSION
        ),
        "native_tool_outcomes": list(recorder.native_tool_outcomes),
        "native_tool_outcome_count": len(recorder.native_tool_outcomes),
        "iteration_limit_suppressed_tool_calls": list(
            recorder.iteration_limit_suppressed_tool_calls
        ),
        "messages": to_jsonable(list(transcript_messages)),
        "audit_action_counts": dict(sorted(audit_counts.items())),
        "llm_call_records": [
            {
                "call_id": call.call_id,
                "pid": call.pid,
                "status": call.status,
                "api": call.api,
                "model": call.model,
                "request_id": call.request_id,
                "response_id": call.response_id,
                "tool_calls": to_jsonable(call.tool_calls),
                "created_at": call.created_at,
                "usage": to_jsonable(call.usage),
                "error": call.error,
            }
            for call in calls
        ],
        "external_effect_count": len(host.store.list_external_effects()),
        "external_effect_scope": (
            "Includes protected LLM provider calls; bridged AgentDojo tools are "
            "deliberately not classified as protected effects in this ambient pilot."
        ),
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
    }


def _contained_run_snapshot(
    host: Runtime,
    *,
    pid: str,
    results: Sequence[Any],
    recorder: RunRecorder,
    status_before_host_exit: str | None,
    replaced_tool_names: Sequence[str],
    prompt_mode: str,
    transcript_messages: Sequence[ChatMessage],
    authority: CompiledTaskAuthority,
    model_sink: Any | None,
    model_sink_rule: Any | None,
    error: BaseException | None,
) -> dict[str, Any]:
    base = _ambient_run_snapshot(
        host,
        pid=pid,
        results=results,
        recorder=recorder,
        status_before_host_exit=status_before_host_exit,
        replaced_tool_names=replaced_tool_names,
        prompt_mode=prompt_mode,
        transcript_messages=transcript_messages,
        error=error,
    )
    native_effects = [
        effect
        for effect in host.store.list_external_effects(pid=pid)
        if str(effect.provider) == f"agentdojo.{authority.manifest['metadata']['suite']}"
    ]
    audits = list(host.audit.trace(actor=pid))
    decisions = list(host.store.list_data_flow_decisions(pid=pid))
    denials: list[dict[str, Any]] = []
    for execution in recorder.tool_executions:
        result = execution.get("result")
        if not isinstance(result, Mapping):
            continue
        denial = result.get("contained_denial")
        if isinstance(denial, Mapping):
            denials.append(dict(to_jsonable(denial)))
    model_sink_projection = None
    if model_sink is not None:
        model_sink_projection = {
            "identity": model_sink.identity,
            "identity_sha256": model_sink.identity_sha256,
            "registry_identity": model_sink.registry_identity,
            "registry_identity_sha256": model_sink.registry_identity_sha256,
        }
    model_rule_projection = (
        model_sink_rule.to_dict()
        if model_sink_rule is not None and callable(getattr(model_sink_rule, "to_dict", None))
        else None
    )
    base.update(
        {
            "arm": "libos_contained",
            "semantics": "native_capability_ifc_contained",
            "native_admission_evidence_schema_version": 1,
            "authority_manifest_template_sha256": (
                authority.manifest_template_sha256
            ),
            "authority_ground_truth_calls_sha256": (
                authority.ground_truth_calls_sha256
            ),
            "authority_clean_environment_sha256": (
                authority.clean_environment_sha256
            ),
            "function_policy_sha256": authority.policy_sha256,
            "contained_ablation": authority.ablation.value,
            "authority_source_ref_coverage": authority.source_ref_coverage,
            "authority_metadata": to_jsonable(authority.manifest["metadata"]),
            "model_processing_sink": model_sink_projection,
            "model_processing_sink_rule": model_rule_projection,
            "native_external_effects": [
                _contained_effect_projection(effect) for effect in native_effects
            ],
            "native_external_effect_count": len(native_effects),
            "native_data_flow_decisions": [
                _contained_data_flow_projection(decision) for decision in decisions
            ],
            "native_audit_records": [
                _contained_audit_projection(record) for record in audits
            ],
            "contained_denials": denials,
            "external_effect_scope": (
                "native_external_effects contains only committed or aborted "
                "AgentDojo provider-bound protected operations for this task PID; "
                "LLM processing effects remain separately visible in the Host store"
            ),
        }
    )
    return base


def _contained_effect_projection(effect: Any) -> dict[str, Any]:
    metadata = effect.provider_metadata
    context = metadata.get("context") if isinstance(metadata, Mapping) else None
    receipt = effect.provider_receipt
    return {
        "effect_id": effect.effect_id,
        "record_id": effect.record_id,
        "event_id": effect.event_id,
        "pid": effect.pid,
        "provider": effect.provider,
        "operation": effect.operation,
        "target": effect.target,
        "transaction_state": effect.transaction_state,
        "effect_state": effect.effect_state,
        "state_mutation": effect.state_mutation,
        "information_flow": effect.information_flow,
        "canonical_args_hash": effect.canonical_args_hash,
        "provider_receipt_present": bool(receipt),
        "provider_receipt": (
            to_jsonable(dict(receipt)) if isinstance(receipt, Mapping) else {}
        ),
        "provider_receipt_sha256": (
            hashlib.sha256(
                json.dumps(
                    to_jsonable(receipt),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if receipt
            else None
        ),
        "context": to_jsonable(context) if isinstance(context, Mapping) else {},
    }


def _contained_data_flow_projection(decision: Any) -> dict[str, Any]:
    return {
        "decision_id": decision.decision_id,
        "sink": decision.sink,
        "direction": decision.direction.value,
        "outcome": decision.outcome.value,
        "reason": decision.reason,
        "labels": decision.labels.to_dict(),
        "source_refs": to_jsonable(decision.source_refs),
        "payload_hash": decision.payload_hash,
        "registry_generation": decision.registry_generation,
        "trust_id": decision.trust_id,
        "trust_hash": decision.trust_hash,
        "release_capability_id": decision.release_capability_id,
    }


def _contained_audit_projection(record: Any) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "action": record.action,
        "target": record.target,
        "capability_refs": list(record.capability_refs),
        "decision": to_jsonable(record.decision),
        "correlation_id": record.correlation_id,
        "parent_record_id": record.parent_record_id,
    }


def _aggregate_control_query_runs(
    query_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate every invocation of AgentDojo's native control pipeline."""

    if not query_runs:
        return {}
    aggregate = dict(query_runs[-1])
    logical_model_requests: list[dict[str, Any]] = []
    provider_calls: list[dict[str, Any]] = []
    messages: list[Any] = []
    usage: Counter[str] = Counter()
    query_summaries: list[dict[str, Any]] = []
    query_transcripts: list[dict[str, Any]] = []
    tool_call_count = 0
    executed_tool_call_count = 0
    for query_run in query_runs:
        invocation = int(query_run.get("query_invocation") or 0)
        for request in query_run.get("logical_model_requests") or []:
            if not isinstance(request, Mapping):
                continue
            tagged_request = dict(request)
            tagged_request["query_invocation"] = invocation
            logical_model_requests.append(tagged_request)
        for provider_call in query_run.get("provider_calls") or []:
            if not isinstance(provider_call, Mapping):
                continue
            tagged = dict(provider_call)
            tagged["query_invocation"] = invocation
            provider_calls.append(tagged)
        transcript = list(query_run.get("messages") or [])
        messages.extend(transcript)
        query_transcripts.append(
            {
                "query_invocation": invocation,
                "messages": transcript,
            }
        )
        selected_tool_calls = int(query_run.get("tool_call_count") or 0)
        selected_executed_tool_calls = int(
            query_run.get("executed_tool_call_count") or 0
        )
        tool_call_count += selected_tool_calls
        executed_tool_call_count += selected_executed_tool_calls
        selected_usage = dict(query_run.get("usage") or {})
        for key, value in selected_usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[str(key)] += value
        query_summaries.append(
            {
                "query_invocation": invocation,
                "logical_model_invocation_count": int(
                    query_run.get("logical_model_invocation_count") or 0
                ),
                "provider_call_count": int(
                    query_run.get("provider_call_count") or 0
                ),
                "tool_call_count": selected_tool_calls,
                "executed_tool_call_count": selected_executed_tool_calls,
                "usage": selected_usage,
                "error_type": query_run.get("error_type"),
                "error": query_run.get("error"),
            }
        )
    aggregate.update(
        {
            "query_evidence_schema_version": 1,
            "query_invocation_count": len(query_runs),
            "query_runs": query_summaries,
            "query_transcripts": query_transcripts,
            "logical_model_requests": logical_model_requests,
            "logical_model_invocation_count": len(logical_model_requests),
            "provider_calls": provider_calls,
            "provider_call_count": len(provider_calls),
            "tool_call_count": tool_call_count,
            "executed_tool_call_count": executed_tool_call_count,
            "messages": messages,
            "usage": dict(sorted(usage.items())),
        }
    )
    return aggregate


def _aggregate_ambient_query_runs(
    query_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate every invocation made by AgentDojo's empty-output retry loop."""

    if not query_runs:
        return {}
    aggregate = dict(query_runs[-1])
    logical_model_requests: list[Any] = []
    provider_calls: list[Any] = []
    tool_executions: list[Any] = []
    native_tool_outcomes: list[Any] = []
    suppressed_calls: list[Any] = []
    llm_call_records: list[Any] = []
    native_external_effects: list[Any] = []
    native_data_flow_decisions: list[Any] = []
    native_audit_records: list[Any] = []
    contained_denials: list[Any] = []
    usage: Counter[str] = Counter()
    audit_counts: Counter[str] = Counter()
    replaced_tool_names: set[str] = set()
    query_summaries: list[dict[str, Any]] = []
    query_transcripts: list[dict[str, Any]] = []
    messages_all: list[Any] = []
    scalar_totals: Counter[str] = Counter()
    for query_run in query_runs:
        invocation = int(query_run.get("query_invocation") or 0)
        logical_model_requests.extend(
            {
                **dict(item),
                "query_invocation": invocation,
            }
            for item in query_run.get("logical_model_requests") or []
            if isinstance(item, Mapping)
        )
        provider_calls.extend(
            {
                **dict(item),
                "query_invocation": invocation,
            }
            for item in query_run.get("provider_calls") or []
            if isinstance(item, Mapping)
        )
        tool_executions.extend(
            {
                **dict(item),
                "query_invocation": invocation,
                "pid": query_run.get("pid"),
            }
            for item in query_run.get("tool_executions") or []
            if isinstance(item, Mapping)
        )
        native_tool_outcomes.extend(
            {
                **dict(item),
                "query_invocation": invocation,
            }
            for item in query_run.get("native_tool_outcomes") or []
            if isinstance(item, Mapping)
        )
        suppressed = list(
            query_run.get("iteration_limit_suppressed_tool_calls") or []
        )
        suppressed_calls.extend(
            {
                **dict(item),
                "query_invocation": invocation,
            }
            for item in suppressed
            if isinstance(item, Mapping)
        )
        llm_call_records.extend(
            {
                **dict(item),
                "query_invocation": invocation,
            }
            for item in query_run.get("llm_call_records") or []
            if isinstance(item, Mapping)
        )
        for destination, key in (
            (native_external_effects, "native_external_effects"),
            (native_data_flow_decisions, "native_data_flow_decisions"),
            (native_audit_records, "native_audit_records"),
            (contained_denials, "contained_denials"),
        ):
            destination.extend(
                {
                    **dict(item),
                    "query_invocation": invocation,
                }
                for item in query_run.get(key) or []
                if isinstance(item, Mapping)
            )
        for key, value in dict(query_run.get("usage") or {}).items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[str(key)] += value
        for key, value in dict(query_run.get("audit_action_counts") or {}).items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                audit_counts[str(key)] += value
        replaced_tool_names.update(query_run.get("replaced_runtime_tool_names") or [])
        for key in (
            "scheduler_result_count",
            "hidden_terminal_tool_calls",
            "external_effect_count",
        ):
            value = query_run.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                scalar_totals[key] += value
        messages = list(query_run.get("messages") or [])
        messages_all.extend(messages)
        query_transcripts.append(
            {
                "query_invocation": invocation,
                "messages": messages,
            }
        )
        query_summaries.append(
            {
                "query_invocation": invocation,
                "runtime_subdir": query_run.get("runtime_subdir"),
                "pid": query_run.get("pid"),
                "process_status": query_run.get("process_status"),
                "status_before_host_exit": query_run.get("status_before_host_exit"),
                "logical_model_invocation_count": query_run.get(
                    "logical_model_invocation_count", 0
                ),
                "provider_call_count": query_run.get("provider_call_count", 0),
                "tool_call_count": query_run.get("tool_call_count", 0),
                "executed_tool_call_count": query_run.get(
                    "executed_tool_call_count", 0
                ),
                "native_tool_outcome_count": query_run.get(
                    "native_tool_outcome_count", 0
                ),
                "llm_call_record_count": query_run.get("llm_call_record_count", 0),
                "hidden_terminal_tool_calls": query_run.get(
                    "hidden_terminal_tool_calls", 0
                ),
                "native_external_effect_count": query_run.get(
                    "native_external_effect_count", 0
                ),
                "contained_denial_count": len(
                    query_run.get("contained_denials") or []
                ),
                "model_processing_sink": query_run.get(
                    "model_processing_sink"
                ),
                "iteration_limit_suppressed_tool_call_count": len(suppressed),
                "usage": dict(query_run.get("usage") or {}),
                "error_type": query_run.get("error_type"),
                "error": query_run.get("error"),
            }
        )
    aggregate.update(
        {
            "query_evidence_schema_version": 1,
            "query_invocation_count": len(query_runs),
            "query_runs": query_summaries,
            "query_transcripts": query_transcripts,
            "logical_model_requests": logical_model_requests,
            "logical_model_invocation_count": len(logical_model_requests),
            "provider_calls": provider_calls,
            "provider_call_count": len(provider_calls),
            "tool_executions": tool_executions,
            "native_tool_outcome_evidence_schema_version": (
                _NATIVE_TOOL_OUTCOME_SCHEMA_VERSION
            ),
            "native_tool_outcomes": native_tool_outcomes,
            "native_tool_outcome_count": len(native_tool_outcomes),
            "tool_call_count": sum(
                int(query_run.get("tool_call_count") or 0)
                for query_run in query_runs
            ),
            "executed_tool_call_count": len(tool_executions),
            "messages": messages_all,
            "iteration_limit_suppressed_tool_calls": suppressed_calls,
            "llm_call_records": llm_call_records,
            "llm_call_record_count": len(llm_call_records),
            "native_external_effects": native_external_effects,
            "native_external_effect_count": len(native_external_effects),
            "native_data_flow_decisions": native_data_flow_decisions,
            "native_audit_records": native_audit_records,
            "contained_denials": contained_denials,
            "usage": dict(sorted(usage.items())),
            "audit_action_counts": dict(sorted(audit_counts.items())),
            "replaced_runtime_tool_names": sorted(replaced_tool_names),
            **dict(scalar_totals),
        }
    )
    return aggregate


def _dojo_messages_to_openai(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content_value = message.get("content")
        content = (
            get_text_content_as_str(content_value)
            if isinstance(content_value, list)
            else ""
        )
        if role in {"system", "user"}:
            converted.append({"role": role, "content": content})
            continue
        if role == "assistant":
            assistant: dict[str, Any] = {"role": "assistant", "content": content}
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function,
                            "arguments": json.dumps(call.args, ensure_ascii=False),
                        },
                    }
                    for call in tool_calls
                ]
            converted.append(assistant)
            continue
        if role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id"),
                    "name": message["tool_call"].function,
                    "content": message.get("error") or content,
                }
            )
            continue
        raise ValueError(f"unsupported AgentDojo message role: {role}")
    return converted


def _dojo_tools_to_openai(runtime: FunctionsRuntime) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": function.name,
                "description": function.description,
                "parameters": function.parameters.model_json_schema(),
            },
        }
        for function in sorted(runtime.functions.values(), key=lambda item: item.name)
    ]


def _function_call(value: Mapping[str, Any]) -> FunctionCall:
    raw_args = value.get("arguments", {})
    if isinstance(raw_args, str):
        parsed = json.loads(raw_args or "{}")
    else:
        parsed = raw_args
    if not isinstance(parsed, Mapping):
        raise ValueError("provider tool arguments must decode to an object")
    return FunctionCall(
        function=str(value.get("name") or ""),
        args=dict(parsed),
        id=(str(value["id"]) if value.get("id") is not None else None),
    )


def _openai_tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "")
    return str(tool.get("name") or "")
