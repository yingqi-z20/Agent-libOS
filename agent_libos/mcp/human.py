"""Narrow durable HumanRequest bridge for MCP Elicitation.

The MCP managers do not accept raw UI/CLI responses.  A Runtime adapter first
creates and settles a real durable Human question, whose text answer is the
canonical JSON object of local Elicitation responses.  Managers then consume
that already-approved answer through this SPI and independently validate it
against the broker-held provider request keys and schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from agent_libos.mcp._input import (
    canonical_json_bytes,
    decode_broker_json,
    json_sha256,
)
from agent_libos.mcp.types import JsonValue, McpInputRequest
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.ids import new_id


_MCP_HOST_QUESTION_ACTORS = frozenset({"runtime", "gui", "cli"})


def _require_human_request_id(value: Any) -> None:
    if (
        type(value) is not str
        or not value.startswith("hreq_")
        or len(value) != 21
        or any(character not in "0123456789abcdef" for character in value[5:])
    ):
        raise ValidationError("MCP Human request id is invalid")


@dataclass(frozen=True)
class McpHumanRequestReceipt:
    request_id: str
    revision: int
    preview_sha256: str

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValidationError("MCP Human request id is invalid")
        if type(self.revision) is not int or self.revision < 0:
            raise ValidationError("MCP Human request revision is invalid")
        if (
            type(self.preview_sha256) is not str
            or len(self.preview_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.preview_sha256)
        ):
            raise ValidationError("MCP Human preview digest is invalid")


class McpHumanRequestBridge(Protocol):
    """Runtime-owned adapter backed by durable ``HumanObjectManager`` state."""

    def reserve_question_id(self) -> str: ...

    def create_question(
        self,
        *,
        owner_id: str,
        server_id: str,
        operation: str,
        local_ref: str,
        preview: dict[str, JsonValue],
        preview_sha256: str,
        expires_at: str | None,
        request_id: str | None = None,
    ) -> McpHumanRequestReceipt: ...

    def inspect_question(
        self,
        request_id: str,
        *,
        preview_sha256: str,
    ) -> McpHumanRequestReceipt: ...

    def consume_approved_answer(
        self,
        request_id: str,
        *,
        presented_revision: int,
        preview_sha256: str,
    ) -> dict[str, JsonValue]: ...

    def cancel_question(
        self,
        request_id: str,
        *,
        preview_sha256: str,
        reason: str,
    ) -> None: ...

    def cancel_question_for_recovery(self, request_id: str, *, reason: str) -> None: ...

    def question_preview_sha256_for_recovery(self, request_id: str) -> str: ...


class HumanObjectManagerMcpBridge:
    """Real durable HumanRequest adapter used by Runtime composition.

    ``settle_answer`` is the only method that accepts raw Host form values.  It
    stores their canonical JSON as the answer to an ordinary Human *question*
    (data entry), using HumanObjectManager's atomic revision/preview fence.
    Protected-operation ASK decisions remain a separate authority step.
    """

    def __init__(
        self,
        human_manager: Any,
        *,
        host_question_authorizer: Callable[..., None] | None = None,
    ) -> None:
        if human_manager is None:
            raise TypeError("MCP HumanObjectManager is required")
        if host_question_authorizer is not None and not callable(
            host_question_authorizer
        ):
            raise TypeError("MCP Host question authorizer must be callable")
        self.human_manager = human_manager
        self.host_question_authorizer = host_question_authorizer

    def reserve_question_id(self) -> str:
        return new_id("hreq")

    def create_question(
        self,
        *,
        owner_id: str,
        server_id: str,
        operation: str,
        local_ref: str,
        preview: dict[str, JsonValue],
        preview_sha256: str,
        expires_at: str | None,
        request_id: str | None = None,
    ) -> McpHumanRequestReceipt:
        _require_preview(preview, preview_sha256)
        selected_request_id = request_id or self.reserve_question_id()
        _require_human_request_id(selected_request_id)
        question = (
            f"MCP server {server_id} requires user-provided input for "
            f"{operation}. Review the Host form and submit one response."
        )
        context = {
            "_agent_libos_mcp_preview_sha256": preview_sha256,
            "mcp_server_id": server_id,
            "mcp_operation": operation,
            "mcp_local_ref": local_ref,
            "mcp_preview": preview,
            "mcp_expires_at": expires_at,
            "mcp_answer_format": "canonical-json-local-input-responses",
        }
        request_id = self._create_question(
            owner_id=owner_id,
            server_id=server_id,
            operation=operation,
            local_ref=local_ref,
            preview=preview,
            preview_sha256=preview_sha256,
            question=question,
            context=context,
            request_id=selected_request_id,
        )
        return self.inspect_question(
            request_id,
            preview_sha256=preview_sha256,
        )

    def _create_question(
        self,
        *,
        owner_id: str,
        server_id: str,
        operation: str,
        local_ref: str,
        preview: dict[str, JsonValue],
        preview_sha256: str,
        question: str,
        context: dict[str, JsonValue],
        request_id: str,
    ) -> str:
        if owner_id not in _MCP_HOST_QUESTION_ACTORS:
            return self.human_manager.ask(
                owner_id,
                question,
                context=context,
                blocking=True,
                _request_id=request_id,
            )
        authorizer = self.host_question_authorizer
        if authorizer is None:
            raise ValidationError("MCP Host question authorizer is unavailable")
        authorized = authorizer(
            owner_id=owner_id,
            server_id=server_id,
            operation=operation,
            local_ref=local_ref,
            preview=preview,
            preview_sha256=preview_sha256,
        )
        if authorized is not None:
            raise ValidationError("MCP Host question authorizer returned invalid evidence")
        _require_preview(preview, preview_sha256)
        selected_human = self.human_manager.config.runtime.default_human
        if type(selected_human) is not str or not selected_human:
            raise ValidationError("MCP Host question target is invalid")
        return self.human_manager.query(
            pid=owner_id,
            human=selected_human,
            request={
                "type": "question",
                "question": question,
                "context": context,
            },
            blocking=True,
            _request_id=request_id,
        )

    def inspect_question(
        self,
        request_id: str,
        *,
        preview_sha256: str,
    ) -> McpHumanRequestReceipt:
        request = self.human_manager.get(request_id)
        self._require_question(request, preview_sha256=preview_sha256)
        revision = request.revision
        if str(getattr(request.status, "value", request.status)) == "approved":
            decision = request.decision
            receipt = decision.get("approval_preview_receipt") if isinstance(
                decision, dict
            ) else None
            presented = receipt.get("request_revision") if isinstance(
                receipt, dict
            ) else None
            if type(presented) is not int or presented < 0:
                raise ValidationError("MCP Human answer receipt binding changed")
            revision = presented
        return McpHumanRequestReceipt(
            request_id=request.request_id,
            revision=revision,
            preview_sha256=preview_sha256,
        )

    def settle_answer(
        self,
        request_id: str,
        responses: dict[str, JsonValue],
        *,
        expected_revision: int,
        preview_sha256: str,
        responder: str | None = None,
    ) -> McpHumanRequestReceipt:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValidationError("MCP Human expected revision is invalid")
        request = self.human_manager.get(request_id)
        self._require_question(request, preview_sha256=preview_sha256)
        status = str(getattr(request.status, "value", request.status))
        if status == "approved":
            existing = self.consume_approved_answer(
                request_id,
                presented_revision=expected_revision,
                preview_sha256=preview_sha256,
            )
            encoded_existing = canonical_json_bytes(
                existing,
                label="MCP approved Human Elicitation responses",
            )
            encoded_presented = canonical_json_bytes(
                responses,
                label="MCP Human Elicitation responses",
            )
            if encoded_existing != encoded_presented:
                raise ValidationError("MCP Human question was already answered differently")
            return McpHumanRequestReceipt(
                request_id=request.request_id,
                revision=request.revision,
                preview_sha256=preview_sha256,
            )
        if status != "pending":
            raise ValidationError("MCP Human question is no longer answerable")
        answer = canonical_json_bytes(
            responses,
            label="MCP Human Elicitation responses",
        ).decode("ascii")
        decided = self.human_manager.approve(
            request_id,
            decision={
                "approved": True,
                "answer": answer,
                "mcp_data_entry": True,
            },
            responder=responder,
            expected_revision=expected_revision,
            preview_sha256=preview_sha256,
        )
        return McpHumanRequestReceipt(
            request_id=decided.request_id,
            revision=decided.revision,
            preview_sha256=preview_sha256,
        )

    def consume_approved_answer(
        self,
        request_id: str,
        *,
        presented_revision: int,
        preview_sha256: str,
    ) -> dict[str, JsonValue]:
        if type(presented_revision) is not int or presented_revision < 0:
            raise ValidationError("MCP Human presented revision is invalid")
        request = self.human_manager.get(request_id)
        self._require_question(request, preview_sha256=preview_sha256)
        if str(getattr(request.status, "value", request.status)) != "approved":
            raise ValidationError("MCP Human question has no approved answer")
        decision = request.decision
        if not isinstance(decision, dict) or decision.get("mcp_data_entry") is not True:
            raise ValidationError("MCP Human answer provenance is invalid")
        receipt = decision.get("approval_preview_receipt")
        if (
            not isinstance(receipt, dict)
            or receipt.get("request_revision") != presented_revision
            or receipt.get("preview_sha256") != preview_sha256
        ):
            raise ValidationError("MCP Human answer receipt binding changed")
        answer = decision.get("answer")
        if type(answer) is not str:
            raise ValidationError("MCP Human answer is invalid")
        return decode_broker_json(
            answer.encode("ascii"),
            label="MCP Human Elicitation responses",
        )

    def cancel_question(
        self,
        request_id: str,
        *,
        preview_sha256: str,
        reason: str,
    ) -> None:
        request = self.human_manager.get(request_id)
        self._require_question(request, preview_sha256=preview_sha256)
        if str(getattr(request.status, "value", request.status)) != "pending":
            return
        self.human_manager.reject(
            request_id,
            decision={
                "approved": False,
                "reason": reason,
                "mcp_data_entry": True,
            },
            expected_revision=request.revision,
            preview_sha256=preview_sha256,
        )

    def cancel_question_for_recovery(self, request_id: str, *, reason: str) -> None:
        """Terminalize only a previously persisted MCP question on recovery."""

        try:
            preview_sha256 = self.question_preview_sha256_for_recovery(request_id)
        except ValidationError:
            request = self.human_manager.get(request_id)
            if request is None:
                return
            raise
        self.cancel_question(
            request_id,
            preview_sha256=preview_sha256,
            reason=reason,
        )

    def question_preview_sha256_for_recovery(self, request_id: str) -> str:
        """Return the persisted preview fence without accepting Host input."""

        request = self.human_manager.get(request_id)
        if request is None:
            raise ValidationError("MCP Human question is unavailable")
        context = getattr(request, "payload", {}).get("context")
        preview_sha256 = (
            context.get("_agent_libos_mcp_preview_sha256")
            if isinstance(context, dict)
            else None
        )
        if type(preview_sha256) is not str:
            raise ValidationError("MCP Human recovery binding is invalid")
        self._require_question(request, preview_sha256=preview_sha256)
        return preview_sha256

    @staticmethod
    def _require_question(request: Any, *, preview_sha256: str) -> None:
        if request is None or getattr(request, "payload", {}).get("type") != "question":
            raise ValidationError("MCP Human question is unavailable")
        context = request.payload.get("context")
        if not isinstance(context, dict):
            raise ValidationError("MCP Human question binding is invalid")
        preview = context.get("mcp_preview")
        if type(preview) is not dict:
            raise ValidationError("MCP Human question preview is invalid")
        _require_preview(preview, preview_sha256)
        if context.get("_agent_libos_mcp_preview_sha256") != preview_sha256:
            raise ValidationError("MCP Human question preview binding changed")


def mcp_human_preview(
    *,
    server_id: str,
    operation: str,
    local_ref: str,
    input_requests: tuple[McpInputRequest, ...],
) -> tuple[dict[str, JsonValue], str]:
    """Build the exact non-secret question preview and its canonical digest."""

    for label, value in (
        ("server id", server_id),
        ("operation", operation),
        ("local reference", local_ref),
    ):
        if type(value) is not str or not value:
            raise ValidationError(f"MCP Human preview {label} is invalid")
    if type(input_requests) is not tuple:
        raise ValidationError("MCP Human preview input requests are invalid")
    projected: list[JsonValue] = []
    for request in input_requests:
        if not isinstance(request, McpInputRequest):
            raise ValidationError("MCP Human preview input request is invalid")
        schema = dict(request.schema)
        projected.append(
            {
                "requestId": request.request_id,
                "kind": request.kind.value,
                "mode": request.mode,
                "prompt": request.prompt,
                "inertUrl": request.inert_url,
                "schema": schema,
                "schemaSha256": json_sha256(
                    schema,
                    label="MCP Human input schema",
                ),
            }
        )
    preview: dict[str, JsonValue] = {
        "contract": "agent-libos.mcp.elicitation.v1",
        "serverId": server_id,
        "operation": operation,
        "localRef": local_ref,
        "answerFormat": "canonical-json-local-input-responses",
        "inputRequests": projected,
    }
    digest = json_sha256(preview, label="MCP Human request preview")
    canonical_json_bytes(preview, label="MCP Human request preview")
    return preview, digest


def require_human_receipt(
    value: McpHumanRequestReceipt,
    *,
    preview_sha256: str,
) -> McpHumanRequestReceipt:
    if not isinstance(value, McpHumanRequestReceipt):
        raise ValidationError("MCP Human request factory returned an invalid receipt")
    if value.preview_sha256 != preview_sha256:
        raise ValidationError("MCP Human request preview binding changed")
    return value


def _require_preview(preview: Any, preview_sha256: Any) -> None:
    if type(preview) is not dict:
        raise ValidationError("MCP Human preview is invalid")
    actual = json_sha256(preview, label="MCP Human request preview")
    if type(preview_sha256) is not str or actual != preview_sha256:
        raise ValidationError("MCP Human preview binding changed")


__all__ = [
    "McpHumanRequestBridge",
    "McpHumanRequestReceipt",
    "HumanObjectManagerMcpBridge",
    "mcp_human_preview",
    "require_human_receipt",
]
