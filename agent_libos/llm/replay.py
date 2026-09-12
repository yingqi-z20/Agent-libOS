from __future__ import annotations

"""Host-private, lossless Responses conversation replay.

The payloads in this module are executable provider input, not observations.
They must never be used as GUI, audit, model-tool, or image-package projections.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping, Sequence

from agent_libos.llm.response_items import validate_response_items
from agent_libos.models.data_flow import DataFlowContext, DataSourceRef
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.llm_replay import LLMReplayHead, LLMReplayTurn, canonical_replay_payload
from agent_libos.models.task_runs import canonical_task_run_json
from agent_libos.utils.ids import estimate_tokens, new_id, utc_now


class ReplayStateError(ValidationError):
    """A private continuation cannot be replayed without changing its meaning."""


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    pid: str
    run_id: str | None
    provider_fingerprint: str
    model: str
    context_generation: str
    expected_head: LLMReplayHead | None
    response_items: list[dict[str, Any]] = field(repr=False, metadata={"serialize": False})
    payload: dict[str, Any] = field(repr=False, metadata={"serialize": False})
    input_items: list[dict[str, Any]] = field(repr=False, metadata={"serialize": False})
    flow_context: DataFlowContext = field(repr=False, metadata={"serialize": False})
    estimated_input_tokens: int = 0


def messages_to_response_items(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert logical messages without moving instructions or copying markers."""

    result: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ReplayStateError("Responses replay message role is invalid")
        content = message.get("content", "")
        if role == "tool":
            call_id = message.get("tool_call_id") or message.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ReplayStateError("Responses replay tool output lacks its call identity")
            result.append({"type": "function_call_output", "call_id": call_id, "output": content})
            continue
        if content or role != "assistant":
            result.append({"role": role, "content": deepcopy(content)})
        calls = message.get("tool_calls") or []
        if calls and role != "assistant":
            raise ReplayStateError("Responses replay tool call role is invalid")
        for call in calls:
            function = call.get("function") or call
            call_id = call.get("id") or call.get("call_id")
            arguments = function.get("arguments", "{}")
            result.append({
                "type": "function_call",
                "call_id": call_id,
                "name": function.get("name"),
                "arguments": arguments if isinstance(arguments, str) else canonical_task_run_json(arguments),
            })
    return result


def _counter(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= (1 << 53) - 1 else None


def reasoning_token_bound(items: Sequence[Mapping[str, Any]], usage: Mapping[str, Any], max_output_tokens: int) -> int:
    """Account for opaque state using generation counters, never ciphertext size."""

    if not any(item.get("type") == "reasoning" and item.get("encrypted_content") for item in items):
        return 0
    details = usage.get("output_tokens_details")
    candidate = _counter(details.get("reasoning_tokens")) if isinstance(details, Mapping) else None
    if candidate is None:
        candidate = _counter(usage.get("reasoning_tokens"))
    if candidate is None:
        candidate = _counter(usage.get("output_tokens"))
    if candidate is None:
        candidate = _counter(usage.get("completion_tokens"))
    if candidate is None:
        candidate = _counter(max_output_tokens)
    if candidate is None:
        raise ReplayStateError("Opaque Responses replay has no usable token accounting bound")
    return candidate


def estimate_replay_input_tokens(items: Sequence[Mapping[str, Any]], *, opaque_tokens: int, tools: Sequence[Mapping[str, Any]] = ()) -> int:
    # Ciphertext is storage material; a string tokenizer cannot estimate the
    # reasoning represented by it. Its source-turn generation usage supplies
    # that component instead. Reasoning summaries remain visible input.
    visible = [{key: value for key, value in item.items() if key != "encrypted_content"} for item in items]
    return max(1, sum(estimate_tokens(item) + 8 for item in visible) + sum(estimate_tokens(dict(tool)) + 12 for tool in tools) + opaque_tokens + 16)


class LLMReplayService:
    def __init__(self, store: Any, *, max_bytes: int, max_turns: int = 2_048, publications: Any | None = None) -> None:
        if type(max_bytes) is not int or max_bytes <= 0 or type(max_turns) is not int or max_turns <= 0:
            raise ValueError("Responses replay limits must be positive integers")
        self.store = store
        self.max_bytes = max_bytes
        self.max_turns = max_turns
        self.publications = publications

    def _copy_payload(self, value: Mapping[str, Any]) -> dict[str, Any]:
        encoded = canonical_replay_payload(dict(value))
        if len(encoded.encode("utf-8")) > self.max_bytes:
            raise ReplayStateError("Responses replay exceeds its private payload byte bound")
        return deepcopy(dict(value))

    def validate_turn(self, turn: LLMReplayTurn) -> dict[str, Any]:
        if turn.payload is None or turn.purged_at is not None:
            raise ReplayStateError("Responses replay payload is unavailable or purged")
        payload = self._copy_payload(turn.payload)
        encoded = canonical_replay_payload(payload).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != turn.payload_sha256 or len(encoded) != turn.payload_bytes:
            raise ReplayStateError("Responses replay payload integrity check failed")
        provider = self._payload_provider(payload)
        groups = payload["groups"]
        if not isinstance(groups, list) or len(groups) > 2 * self.max_turns:
            raise ReplayStateError("Responses replay turn bound is invalid")
        if not isinstance(payload["prefix"], list):
            raise ReplayStateError("Responses replay prefix is invalid")
        context = DataFlowContext.from_dict(payload["flow_context"])
        if context.labels.to_dict() != turn.source_labels:
            raise ReplayStateError("Responses replay source labels do not match its payload")
        self._validate_groups(groups, provider=provider)
        if self._retained_turn_count(groups) > self.max_turns:
            raise ReplayStateError("Responses replay turn bound is invalid")
        # Validation permits only the last group to be pending; every prior
        # complete request remains independently representable.
        if not groups or not self._pending_calls(groups[-1]):
            validate_response_items(self._flatten(payload), provider=provider)
        else:
            validate_response_items(self._flatten({**payload, "groups": groups[:-1]}) + groups[-1]["input_items"], provider=provider)
        return payload

    @staticmethod
    def _payload_provider(payload: Mapping[str, Any]) -> str | None:
        legacy_fields = {"schema_version", "prefix", "groups", "flow_context"}
        if set(payload) == legacy_fields and payload["schema_version"] == 1:
            return None
        if set(payload) == legacy_fields | {"provider"} and payload["schema_version"] == 2:
            provider = payload["provider"]
            if isinstance(provider, str) and provider in {"openai", "aliyun"}:
                return provider
        raise ReplayStateError("Responses replay payload schema is invalid")

    def _validate_groups(self, groups: list[Any], *, provider: str | None = None) -> None:
        seen: set[str] = set()
        for index, group in enumerate(groups):
            self._validate_group_shape(group)
            call_id = group["call_id"]
            if not isinstance(call_id, str) or not call_id or call_id in seen:
                raise ReplayStateError("Responses replay has duplicate or invalid local call identities")
            seen.add(call_id)
            validate_response_items(group["output_items"], output=True, provider=provider)
            self._validate_outputs(group)
            if index < len(groups) - 1 and (self._pending_calls(group) or not group["validated"]):
                raise ReplayStateError("Responses replay contains an unresolved historical tool call")

    @staticmethod
    def _validate_group_shape(group: Any) -> None:
        if not isinstance(group, dict) or set(group) != {"call_id", "response_id", "input_items", "output_items", "tool_outputs", "reasoning_tokens", "validated"}:
            raise ReplayStateError("Responses replay turn shape is invalid")
        if _counter(group["reasoning_tokens"]) is None:
            raise ReplayStateError("Responses replay token accounting is invalid")
        if type(group["validated"]) is not bool:
            raise ReplayStateError("Responses replay action validation marker is invalid")
        if not isinstance(group["input_items"], list) or not isinstance(group["tool_outputs"], list):
            raise ReplayStateError("Responses replay turn items are invalid")

    @staticmethod
    def _retained_turn_count(groups: list[dict[str, Any]]) -> int:
        """Count one provider turn and its optional Host wait result together.

        Preserve the existing private group format and wire order. Only the
        exact, adjacent Host observation can share its provider's admission;
        an orphan retained by compaction counts as a turn of its own. Disjoint
        pairs ensure that even a chain of Host-shaped groups stays bounded.
        Callers have already validated the group shapes.
        """
        count = 0
        index = 0
        while index < len(groups):
            owner = groups[index]
            count += 1
            index += 1
            if index == len(groups):
                break
            observation = groups[index]
            if (
                owner["validated"]
                and not any(item.get("type") == "function_call" for item in owner["output_items"])
                and observation["call_id"] == f"replay_host_input:{owner['call_id']}"
                and observation["response_id"] is None
                and observation["validated"]
                and observation["reasoning_tokens"] == 0
                and not observation["output_items"]
                and not observation["tool_outputs"]
                and observation["input_items"]
                and all(
                    isinstance(item, dict) and item.get("role") == "user"
                    and item.get("type", "message") == "message"
                    for item in observation["input_items"]
                )
            ):
                index += 1
        return count

    def load_current(self, pid: str) -> tuple[LLMReplayHead, LLMReplayTurn, dict[str, Any]] | None:
        head = self.store.get_llm_replay_head(pid)
        if head is None:
            return None
        turn = self.store.get_llm_replay_turn(head.turn_id)
        if turn is None or turn.pid != pid:
            raise ReplayStateError("Responses replay head has no matching private payload")
        return head, turn, self.validate_turn(turn)

    def superseded_by_exec(self, *, pid: str, call_id: str, publications: Any | None = None) -> bool:
        """Recognize the last provider call retired by a committed Host exec."""
        publications = self.publications if publications is None else publications
        if publications is None:
            return False
        publication = publications.get_latest_committed_exec_publication(pid)
        if publication is None or publication["kind"] != "process_exec" or publication["state"] != "committed" or publication["pid"] != pid:
            return False
        committed = [item for item in publication["receipt"].get("phases", []) if isinstance(item, dict) and item.get("phase") == "committed"]
        return len(committed) == 1 and committed[0].get("prior_llm_call_id") == call_id

    def advance_context_source(self, *, pid: str, context_generation: str, previous: DataSourceRef, current: DataFlowContext) -> None:
        """Follow one verified Host context append, preserving all other sources.

        The caller holds the Object update transaction and checks READ for both
        snapshots. Arbitrary writes and earlier, unaccounted versions must not
        be laundered into a valid private continuation here.
        """
        retained = self.load_current(pid)
        if retained is None or retained[1].context_generation != context_generation:
            return
        head, turn, payload = retained
        flow = DataFlowContext.from_dict(payload["flow_context"])
        prior_refs = tuple(ref for ref in flow.source_refs if ref.oid == previous.oid)
        if not prior_refs:
            return
        if prior_refs != (previous,):
            raise ReplayStateError("Responses replay context source changed before Host update")
        if len(current.source_refs) != 1 or current.source_refs[0].oid != previous.oid:
            raise ReplayStateError("Responses replay context update source is invalid")
        replacement = current.source_refs[0]
        if replacement == previous:
            return
        if replacement.version != previous.version + 1:
            raise ReplayStateError("Responses replay context update skipped a source version")
        preserved = DataFlowContext(labels=flow.labels, source_refs=tuple(ref for ref in flow.source_refs if ref.oid != previous.oid))
        payload["flow_context"] = DataFlowContext.aggregate((preserved, current)).to_dict()
        self._publish_like(turn, payload, expected=head)

    def freeze_request(self, request: ReplayRequest) -> dict[str, str]:
        """Retain a conditional-release request privately, without moving head."""
        payload = self._copy_payload({
            "schema_version": 1,
            "kind": "frozen_request",
            "request": {
                "payload": request.payload,
                "input_items": request.input_items,
                "response_items": request.response_items,
                "estimated_input_tokens": request.estimated_input_tokens,
                "expected_head": None if request.expected_head is None else {
                    "pid": request.expected_head.pid,
                    "turn_id": request.expected_head.turn_id,
                    "revision": request.expected_head.revision,
                    "updated_at": request.expected_head.updated_at,
                },
                "flow_context": request.flow_context.to_dict(),
            },
        })
        turn = LLMReplayTurn.from_payload(turn_id=new_id("llmreplay_request"), pid=request.pid, run_id=request.run_id, provider_fingerprint=request.provider_fingerprint, model=request.model, context_generation=request.context_generation, payload=payload, source_labels=request.flow_context.labels.to_dict(), created_at=utc_now())
        self.store.insert_llm_replay_turn(turn)
        return {"turn_id": turn.turn_id, "payload_sha256": turn.payload_sha256}

    def load_request(self, reference: Mapping[str, Any], *, pid: str, provider_fingerprint: str, model: str, context_generation: str, run_id: str | None = None, provider: str | None = None) -> ReplayRequest:
        if set(reference) != {"turn_id", "payload_sha256"}:
            raise ReplayStateError("Responses replay request reference shape is invalid")
        turn = self.store.get_llm_replay_turn(reference["turn_id"])
        if turn is None or turn.payload is None or turn.purged_at is not None:
            raise ReplayStateError("Responses replay request payload is missing or purged")
        if (turn.pid, turn.run_id, turn.provider_fingerprint, turn.model, turn.context_generation, turn.payload_sha256) != (pid, run_id, provider_fingerprint, model, context_generation, reference["payload_sha256"]):
            raise ReplayStateError("Responses replay request scope or integrity changed")
        payload = self._copy_payload(turn.payload)
        encoded = canonical_replay_payload(payload).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != turn.payload_sha256 or len(encoded) != turn.payload_bytes:
            raise ReplayStateError("Responses replay request integrity check failed")
        if set(payload) != {"schema_version", "kind", "request"} or payload["schema_version"] != 1 or payload["kind"] != "frozen_request":
            raise ReplayStateError("Responses replay request payload schema is invalid")
        request = payload["request"]
        if not isinstance(request, dict) or set(request) != {"payload", "input_items", "response_items", "estimated_input_tokens", "expected_head", "flow_context"}:
            raise ReplayStateError("Responses replay frozen request shape is invalid")
        expected = None if request["expected_head"] is None else LLMReplayHead(**request["expected_head"])
        if self.store.get_llm_replay_head(pid) != expected:
            raise ReplayStateError("Responses replay request head changed while awaiting release")
        flow = DataFlowContext.from_dict(request["flow_context"])
        if flow.labels.to_dict() != turn.source_labels or _counter(request["estimated_input_tokens"]) is None:
            raise ReplayStateError("Responses replay frozen request evidence is invalid")
        if not isinstance(request["payload"], dict) or self._payload_provider(request["payload"]) != provider:
            raise ReplayStateError("Responses replay frozen request provider changed")
        items = validate_response_items(request["response_items"], provider=provider)
        if items != [*self._flatten(request["payload"]), *request["input_items"]]:
            raise ReplayStateError("Responses replay frozen request wire input changed")
        return ReplayRequest(pid=pid, run_id=run_id, provider_fingerprint=provider_fingerprint, model=model, context_generation=context_generation, expected_head=expected, response_items=items, payload=request["payload"], input_items=request["input_items"], flow_context=flow, estimated_input_tokens=request["estimated_input_tokens"])

    def prepare(self, *, pid: str, provider_fingerprint: str, model: str, context_generation: str, messages: Sequence[Mapping[str, Any]], flow_context: DataFlowContext, run_id: str | None = None, tools: Sequence[Mapping[str, Any]] = (), provider: str | None = None) -> ReplayRequest:
        # Validate the Host-supplied selection even before any history exists.
        validate_response_items([], provider=provider)
        current_items = messages_to_response_items(messages)
        prefix: list[dict[str, Any]] = []
        while current_items and current_items[0].get("role") in {"system", "developer"}:
            prefix.append(current_items.pop(0))
        current = self.load_current(pid)
        head = None
        if current is None:
            payload = {"schema_version": 1, "prefix": prefix, "groups": [], "flow_context": flow_context.to_dict()}
            if provider is not None:
                payload.update(schema_version=2, provider=provider)
        else:
            head, turn, payload = current
            if (turn.provider_fingerprint, turn.model, turn.context_generation, turn.run_id) != (provider_fingerprint, model, context_generation, run_id):
                raise ReplayStateError("Responses replay provider, owner, or context scope changed")
            if self._payload_provider(payload) != provider:
                raise ReplayStateError("Responses replay hosted tool provider changed")
            if payload["prefix"] != prefix:
                raise ReplayStateError("Responses replay instruction prefix changed")
            if payload["groups"] and (self._pending_calls(payload["groups"][-1]) or not payload["groups"][-1]["validated"]):
                raise ReplayStateError("Responses replay is waiting for durable tool outputs")
            flow_context = DataFlowContext.aggregate([DataFlowContext.from_dict(payload["flow_context"]), flow_context])
            payload["flow_context"] = flow_context.to_dict()
        self._require_continuity(pid, context_generation, None if current is None else payload)
        self._admit_request_payload(payload, current_items)
        items = validate_response_items([*self._flatten(payload), *current_items], provider=provider)
        opaque_tokens = sum(group["reasoning_tokens"] for group in payload["groups"])
        self._copy_payload(payload)
        return ReplayRequest(pid=pid, run_id=run_id, provider_fingerprint=provider_fingerprint, model=model, context_generation=context_generation, expected_head=head, response_items=items, payload=payload, input_items=current_items, flow_context=flow_context, estimated_input_tokens=estimate_replay_input_tokens(items, opaque_tokens=opaque_tokens, tools=tools))

    def _admit_request_payload(self, payload: dict[str, Any], input_items: list[dict[str, Any]]) -> None:
        if self._retained_turn_count(payload["groups"]) >= self.max_turns:
            raise ReplayStateError("Responses replay requires semantic compaction before another turn")
        # Include the new input before a paid Provider call. Generated output
        # is still checked separately at stage: ciphertext size cannot be
        # inferred from a generation token limit.
        prospective = {
            **payload,
            "groups": [*payload["groups"], {
                "call_id": "pending-provider-call",
                "response_id": None,
                "input_items": input_items,
                "output_items": [],
                "tool_outputs": [],
                "reasoning_tokens": 0,
                "validated": False,
            }],
        }
        self._copy_payload(prospective)

    def _require_continuity(self, pid: str, context_generation: str, payload: Mapping[str, Any] | None) -> None:
        previous = self.store.get_latest_successful_llm_call(
            pid=pid, purpose="action_selection"
        )
        if previous is None:
            return
        previous_generation = previous.request_options.get("llm_context_generation")
        if previous_generation not in {None, context_generation}:
            return
        if payload is not None:
            if not any(group["call_id"] == previous.call_id for group in payload["groups"]):
                raise ReplayStateError("Responses replay continuity is missing a completed provider turn")
            return
        marker = previous.request_options.get("responses_replay")
        if isinstance(marker, Mapping) and marker.get("enabled") is True:
            if self.superseded_by_exec(pid=pid, call_id=previous.call_id):
                return
            raise ReplayStateError("Responses replay head is missing for an existing conversation")

    def stage(self, request: ReplayRequest, *, call_id: str, response_items: Sequence[Mapping[str, Any]], usage: Mapping[str, Any], max_output_tokens: int, response_id: str | None = None, flow_context: DataFlowContext | None = None) -> LLMReplayHead:
        items = validate_response_items(list(response_items), output=True, provider=self._payload_provider(request.payload))
        payload = deepcopy(request.payload)
        if flow_context is not None:
            payload["flow_context"] = DataFlowContext.aggregate([request.flow_context, flow_context]).to_dict()
        if any(group["call_id"] == call_id for group in payload["groups"]):
            raise ReplayStateError("Responses replay local call identity was reused")
        if self._retained_turn_count(payload["groups"]) >= self.max_turns:
            raise ReplayStateError("Responses replay requires semantic compaction before another turn")
        payload["groups"].append({"call_id": call_id, "response_id": response_id, "input_items": deepcopy(request.input_items), "output_items": items, "tool_outputs": [], "reasoning_tokens": reasoning_token_bound(items, usage, max_output_tokens), "validated": False})
        return self._publish(pid=request.pid, run_id=request.run_id, provider_fingerprint=request.provider_fingerprint, model=request.model, context_generation=request.context_generation, payload=payload, expected=request.expected_head)

    def mark_validated(self, *, pid: str, call_id: str) -> LLMReplayHead:
        current = self.load_current(pid)
        if current is None:
            raise ReplayStateError("Responses replay validation has no staged turn")
        head, turn, payload = current
        if not payload["groups"] or payload["groups"][-1]["call_id"] != call_id:
            raise ReplayStateError("Responses replay validation does not match the current turn")
        if payload["groups"][-1]["validated"]:
            return head
        payload["groups"][-1]["validated"] = True
        return self._publish_like(turn, payload, expected=head)

    def settle(self, *, pid: str, call_id: str, outputs: Sequence[Mapping[str, Any]], flow_context: DataFlowContext | None = None) -> LLMReplayHead:
        current = self.load_current(pid)
        if current is None:
            raise ReplayStateError("Responses replay settlement has no staged provider turn")
        head, turn, payload = current
        groups = payload["groups"]
        if not groups or groups[-1]["call_id"] != call_id:
            raise ReplayStateError("Responses replay settlement does not match the current turn")
        group = groups[-1]
        if not group["validated"]:
            raise ReplayStateError("Responses replay cannot settle an unvalidated action")
        by_call = {item["call_id"]: item for item in group["tool_outputs"]}
        for output in outputs:
            selected = deepcopy(dict(output))
            if set(selected) != {"type", "call_id", "output"} or selected["type"] != "function_call_output" or not isinstance(selected["call_id"], str) or not isinstance(selected["output"], str):
                raise ReplayStateError("Responses replay tool result shape is invalid")
            existing = by_call.get(selected["call_id"])
            if existing is not None and existing != selected:
                raise ReplayStateError("Responses replay tool result changed after publication")
            by_call[selected["call_id"]] = selected
        # Parallel completion order cannot reorder the provider's call order.
        call_order = [item["call_id"] for item in group["output_items"] if item.get("type") == "function_call"]
        if set(by_call) - set(call_order):
            raise ReplayStateError("Responses replay tool result has no provider function call")
        group["tool_outputs"] = [by_call[call] for call in call_order if call in by_call]
        if flow_context is not None:
            payload["flow_context"] = DataFlowContext.aggregate([DataFlowContext.from_dict(payload["flow_context"]), flow_context]).to_dict()
        if payload == turn.payload:
            return head
        return self._publish_like(turn, payload, expected=head)

    def discard_staged(self, *, pid: str, call_id: str) -> LLMReplayHead:
        """Drop a rejected action before any tool result has been committed."""
        current = self.load_current(pid)
        if current is None:
            raise ReplayStateError("Responses replay repair has no staged turn")
        head, turn, payload = current
        if not payload["groups"] or payload["groups"][-1]["call_id"] != call_id or payload["groups"][-1]["tool_outputs"] or payload["groups"][-1]["validated"]:
            raise ReplayStateError("Responses replay repair cannot discard a settled turn")
        # Retain a content-free local tombstone so startup can distinguish an
        # intentional rejected action from a missing/corrupt conversation head.
        payload["groups"][-1].update(input_items=[], output_items=[], tool_outputs=[], reasoning_tokens=0, validated=True)
        return self._publish_like(turn, payload, expected=head)

    def append_host_input(self, *, pid: str, call_id: str, input_items: Sequence[Mapping[str, Any]], flow_context: DataFlowContext | None = None) -> LLMReplayHead:
        """Settle an implicit Host wait as input, without inventing tool calls."""
        items = validate_response_items(list(input_items))
        if not items or any(item.get("role") != "user" or item.get("type", "message") != "message" for item in items):
            raise ReplayStateError("Responses replay Host observation must contain user input")
        current = self.load_current(pid)
        if current is None:
            raise ReplayStateError("Responses replay Host observation has no provider turn")
        head, turn, payload = current
        self._append_host_group(payload, call_id, items)
        if flow_context is not None:
            payload["flow_context"] = DataFlowContext.aggregate([DataFlowContext.from_dict(payload["flow_context"]), flow_context]).to_dict()
        if payload == turn.payload:
            return head
        return self._publish_like(turn, payload, expected=head)

    def _append_host_group(self, payload: dict[str, Any], call_id: str, items: list[dict[str, Any]]) -> None:
        groups = payload["groups"]
        host_id = f"replay_host_input:{call_id}"
        group = {"call_id": host_id, "response_id": None, "input_items": items, "output_items": [], "tool_outputs": [], "reasoning_tokens": 0, "validated": True}
        existing = next((entry for entry in groups if entry["call_id"] == host_id), None)
        if existing is not None:
            if existing != group:
                raise ReplayStateError("Responses replay Host observation changed after publication")
            return
        if not groups or groups[-1]["call_id"] != call_id or not groups[-1]["validated"]:
            raise ReplayStateError("Responses replay Host observation does not match its validated turn")
        if any(item.get("type") == "function_call" for item in groups[-1]["output_items"]):
            raise ReplayStateError("Responses replay Host observation cannot replace a native tool output")
        # This observation settles the admitted provider turn. It must not
        # consume another turn after the Host has already acknowledged input.
        # Publication still validates the logical turn and private byte bounds.
        groups.append(group)

    def compact(self, *, pid: str, context_generation: str, messages: Sequence[Mapping[str, Any]], flow_context: DataFlowContext, retain_groups: int = 0, replaced_context_oid: str | None = None) -> LLMReplayHead:
        """Publish a Host-certified summary and retain only whole latest turns.

        The caller must already hold the semantic-compaction transaction and
        have validated the summary certificate. This method is never a tool.
        """
        current = self.load_current(pid)
        if current is None:
            raise ReplayStateError("Responses replay compaction has no current head")
        head, turn, payload = current
        if type(retain_groups) is not int or retain_groups < 0 or retain_groups > len(payload["groups"]):
            raise ReplayStateError("Responses replay compaction group count is invalid")
        if payload["groups"] and (self._pending_calls(payload["groups"][-1]) or not payload["groups"][-1]["validated"]):
            raise ReplayStateError("Responses replay compaction cannot discard a pending tool group")
        if not context_generation or context_generation == turn.context_generation:
            raise ReplayStateError("Responses replay compaction requires a new context generation")
        prefix: list[dict[str, Any]] = []
        summary = messages_to_response_items(messages)
        while summary and summary[0].get("role") in {"system", "developer"}:
            prefix.append(summary.pop(0))
        retained = payload["groups"][-retain_groups:] if retain_groups else []
        if summary:
            retained = [{"call_id": new_id("replay_compaction"), "response_id": None, "input_items": summary, "output_items": [], "tool_outputs": [], "reasoning_tokens": 0, "validated": True}, *retained]
        context = self._compacted_flow_context(
            DataFlowContext.from_dict(payload["flow_context"]), flow_context,
            replaced_context_oid=replaced_context_oid, retain_groups=retain_groups,
        )
        compacted = {**payload, "prefix": prefix, "groups": retained, "flow_context": context.to_dict()}
        return self._publish(pid=pid, run_id=turn.run_id, provider_fingerprint=turn.provider_fingerprint, model=turn.model, context_generation=context_generation, payload=compacted, expected=head)

    @staticmethod
    def _compacted_flow_context(
        historical: DataFlowContext, current: DataFlowContext, *,
        replaced_context_oid: str | None, retain_groups: int,
    ) -> DataFlowContext:
        if replaced_context_oid is not None:
            # Only the Host's generation-bound compaction certificate can
            # authorize replacement of the old context Object versions. Calls
            # retaining native turns cannot retire their source references.
            matching = [ref for ref in current.source_refs if ref.oid == replaced_context_oid]
            if retain_groups or len(matching) != 1:
                raise ReplayStateError("Responses compaction source replacement is invalid")
            historical = DataFlowContext(
                labels=historical.labels,
                source_refs=tuple(ref for ref in historical.source_refs
                                  if ref.oid != replaced_context_oid),
            )
        return DataFlowContext.aggregate((historical, current))

    def rebind(self, *, turn_id: str, pid: str, context_generation: str, provider_fingerprint: str, model: str, flow_context: DataFlowContext, run_id: str | None = None) -> LLMReplayHead:
        """Rebind a pre-authorized local checkpoint reference without mutation."""
        turn = self.store.get_llm_replay_turn(turn_id)
        if turn is None:
            raise ReplayStateError("Responses replay checkpoint payload is missing")
        payload = self.validate_checkpoint_turn(turn)
        if turn.run_id is not None or run_id is not None:
            raise ReplayStateError("Checkpoint replay cannot resurrect a TaskRun binding")
        if (turn.provider_fingerprint, turn.model) != (provider_fingerprint, model):
            raise ReplayStateError("Responses replay checkpoint provider scope changed")
        # Source refs can be remapped only by the authorized checkpoint layer;
        # labels may never become less restrictive by rebind alone.
        context = DataFlowContext.aggregate([DataFlowContext(labels=DataFlowContext.from_dict(payload["flow_context"]).labels), flow_context])
        payload["flow_context"] = context.to_dict()
        return self._publish(pid=pid, run_id=None, provider_fingerprint=provider_fingerprint, model=model, context_generation=context_generation, payload=payload, expected=self.store.get_llm_replay_head(pid))

    def capture_checkpoint_refs(self, pids: Sequence[str]) -> dict[str, dict[str, Any]]:
        refs: dict[str, dict[str, Any]] = {}
        for pid in pids:
            current = self.load_current(pid)
            if current is None:
                continue
            _head, turn, _payload = current
            if turn.run_id is not None:
                continue
            self.validate_checkpoint_turn(turn)
            refs[pid] = {"turn_id": turn.turn_id, "pid": pid, "payload_sha256": turn.payload_sha256, "provider_fingerprint": turn.provider_fingerprint, "model": turn.model, "context_generation": turn.context_generation, "run_id": None}
        return refs

    def validate_checkpoint_turn(self, turn: LLMReplayTurn) -> dict[str, Any]:
        payload = self.validate_turn(turn)
        if payload["groups"] and (self._pending_calls(payload["groups"][-1]) or not payload["groups"][-1]["validated"]):
            raise ReplayStateError("Checkpoint cannot capture an incomplete Responses replay turn")
        return payload

    @staticmethod
    def _flatten(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        result = deepcopy(payload["prefix"])
        for group in payload["groups"]:
            result.extend(deepcopy(group["input_items"]))
            result.extend(deepcopy(group["output_items"]))
            result.extend(deepcopy(group["tool_outputs"]))
        return result

    @staticmethod
    def _pending_calls(group: Mapping[str, Any]) -> set[str]:
        calls = {item["call_id"] for item in group["output_items"] if item.get("type") == "function_call"}
        return calls - {item["call_id"] for item in group["tool_outputs"]}

    @staticmethod
    def _validate_outputs(group: Mapping[str, Any]) -> None:
        calls = [item["call_id"] for item in group["output_items"] if item.get("type") == "function_call"]
        seen: set[str] = set()
        for output in group["tool_outputs"]:
            if not isinstance(output, dict) or set(output) != {"type", "call_id", "output"} or output["type"] != "function_call_output" or not isinstance(output["call_id"], str) or not isinstance(output["output"], str) or output["call_id"] not in calls or output["call_id"] in seen:
                raise ReplayStateError("Responses replay tool pairing is invalid")
            seen.add(output["call_id"])
        if len(calls) != len(set(calls)):
            raise ReplayStateError("Responses replay provider call identity is duplicated")

    def _publish_like(self, turn: LLMReplayTurn, payload: dict[str, Any], *, expected: LLMReplayHead) -> LLMReplayHead:
        return self._publish(pid=turn.pid, run_id=turn.run_id, provider_fingerprint=turn.provider_fingerprint, model=turn.model, context_generation=turn.context_generation, payload=payload, expected=expected)

    def _publish(self, *, pid: str, run_id: str | None, provider_fingerprint: str, model: str, context_generation: str, payload: dict[str, Any], expected: LLMReplayHead | None) -> LLMReplayHead:
        payload = self._copy_payload(payload)
        encoded = canonical_replay_payload(payload).encode("utf-8")
        now = utc_now()
        turn = LLMReplayTurn(turn_id=new_id("llmreplay"), pid=pid, run_id=run_id, provider_fingerprint=provider_fingerprint, model=model, context_generation=context_generation, payload=payload, source_labels=DataFlowContext.from_dict(payload["flow_context"]).labels.to_dict(), payload_sha256=hashlib.sha256(encoded).hexdigest(), payload_bytes=len(encoded), created_at=now)
        self.validate_turn(turn)
        head = LLMReplayHead(pid=pid, turn_id=turn.turn_id, revision=1 if expected is None else expected.revision + 1, updated_at=now)
        with self.store.transaction():
            self.store.insert_llm_replay_turn(turn)
            if not self.store.compare_and_set_llm_replay_head(head, expected_revision=None if expected is None else expected.revision):
                raise ReplayStateError("Responses replay head changed before publication")
        return head
