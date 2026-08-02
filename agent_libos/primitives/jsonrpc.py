from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import json
import math
import os
import re
import socket  # pylint: disable=unused-import  # shared DNS test seam
import threading
import time
from functools import partial
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.exceptions import SchemaError as JsonSchemaSchemaError
from jsonschema.validators import validator_for as jsonschema_validator_for

from agent_libos.capability.manager import CapabilityManager
from agent_libos.capability.rules import AUTHORITY_RULES_KEY
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.human.manager import HumanObjectManager
from agent_libos.models import (
    CapabilityEffect,
    CapabilityRight,
    DataSink,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    JsonRpcCallResult,
    JsonRpcCallStatus,
    JsonRpcEndpointSpec,
    JsonRpcHeaderSpec,
    JsonRpcMethodSpec,
    JsonRpcTransportResult,
    ResourceUsage,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProviderHostError,
    ValidationError,
)
from agent_libos.models.external_effect import default_external_effect_rollback_status
from agent_libos.ports import AuditPort, EventPort
from agent_libos.storage import UnitOfWork
from agent_libos.substrate import JsonRpcProvider, ProviderEffectNotStarted
from agent_libos.substrate.base import _bounded_provider_getaddrinfo
from agent_libos.sdk import (
    ProviderRegistryBinding,
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProtectedOperationSDK,
    ProviderPhase,
    ResourceSettlement,
)
from agent_libos.tools.observability import sanitize_for_observability
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.serde import dumps, to_jsonable
from agent_libos.utils.yaml_loader import load_yaml_mapping

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@+-]*$")
_ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEADER_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_FORBIDDEN_HEADERS = {"connection", "content-length", "host", "transfer-encoding", "upgrade"}
_LOCAL_HTTP_HOSTS = {"localhost", "127.0.0.1", "::1"}
_FORBIDDEN_JSONRPC_HOSTS = {"metadata.google.internal"}
_CALL_RIGHTS = {CapabilityRight.READ.value, CapabilityRight.WRITE.value, CapabilityRight.EXECUTE.value}
_ALLOWED_HEADER_PREFIXES = {"", "Bearer ", "Token ", "Basic "}
_ALLOWED_HEADER_SUFFIXES = {""}
_PROVIDER_RESULT_RETURNED_ATTR = "_agent_libos_provider_result_returned"
class _JsonRpcTransportNotStarted(TimeoutError, ProviderEffectNotStarted):
    """The shared deadline expired after DNS but before provider dispatch."""


def _mark_provider_result_returned(error: ProviderHostError) -> ProviderHostError:
    """Mark a public provider error as occurring after a result was returned."""

    object.__setattr__(error, _PROVIDER_RESULT_RETURNED_ATTR, True)
    return error


def _bounded_jsonrpc_getaddrinfo(
    host: str,
    port: int,
    *,
    deadline: float,
) -> list[Any]:
    """Resolve one Host name behind bounded daemon capacity and one deadline."""
    return _bounded_provider_getaddrinfo(
        host,
        port,
        deadline=deadline,
        operation="JSON-RPC",
    )


def _provider_result_was_returned(error: BaseException) -> bool:
    try:
        attributes = object.__getattribute__(error, "__dict__")
    except Exception:
        return False
    return attributes.get(_PROVIDER_RESULT_RETURNED_ATTR) is True


class JsonRpcPrimitive:
    """Capability-controlled JSON-RPC 2.0 over HTTP client primitive."""

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        capabilities: CapabilityManager,
        audit: AuditPort,
        events: EventPort,
        *,
        protected_operations: ProtectedOperationSDK,
        human: HumanObjectManager | None,
        provider: JsonRpcProvider,
        config: AgentLibOSConfig | None = None,
        resources: Any | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.unit_of_work = unit_of_work
        self.extensions = unit_of_work.extensions
        self.authority = unit_of_work.authority
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.protected_operations = protected_operations
        self.human = human
        self.provider = provider
        self.resources = resources
        self._registry_phase_lock = threading.RLock()
        self._resolution_deadline = threading.local()

    def endpoint_resource(self, endpoint_id: str) -> str:
        return f"jsonrpc_endpoint:{endpoint_id}"

    def method_resource(self, endpoint_id: str, method_id: str) -> str:
        return f"jsonrpc:{endpoint_id}:{method_id}"

    def register_endpoint(
        self,
        endpoint: JsonRpcEndpointSpec | dict[str, Any],
        *,
        actor: str = "runtime",
        replace: bool = False,
        require_capability: bool = True,
        source: str | None = None,
    ) -> dict[str, Any]:
        spec = self._coerce_endpoint(endpoint)
        authority_decision = None
        if require_capability:
            required_right = CapabilityRight.ADMIN if replace else CapabilityRight.WRITE
            authority_decision = self.capabilities.require(
                actor,
                self.endpoint_resource(spec.endpoint_id),
                required_right,
                consume=False,
            )
        now = utc_now()
        with self._registry_phase_lock, self.capabilities.authority_transaction(
            [authority_decision],
            actor=actor,
            operation="JSON-RPC endpoint registry",
        ):
            existing = self.extensions.get_jsonrpc_endpoint(spec.endpoint_id)
            if existing is not None and not replace:
                raise ValidationError(f"JSON-RPC endpoint already exists: {spec.endpoint_id}")
            self.extensions.upsert_jsonrpc_endpoint(spec, registered_by=actor, created_at=now)
            if existing is not None:
                self._disable_replaced_endpoint_method_capabilities(spec.endpoint_id, actor=actor)
            self.events.emit(
                EventType.EXTERNAL_WRITE,
                source=actor,
                target=self.endpoint_resource(spec.endpoint_id),
                payload={"adapter": "jsonrpc", "operation": "endpoint_register", "endpoint_id": spec.endpoint_id},
            )
            self.audit.record(
                actor=actor,
                action="jsonrpc.endpoint.register" if existing is None else "jsonrpc.endpoint.replace",
                target=self.endpoint_resource(spec.endpoint_id),
                decision={
                    "endpoint_id": spec.endpoint_id,
                    "methods": [method.method_id for method in spec.methods],
                    "replaced": existing is not None,
                    "source": source,
                },
            )
        return self.inspect_endpoint(spec.endpoint_id, actor=actor, require_capability=False)

    def register_endpoint_from_yaml_text(
        self,
        text: str,
        *,
        actor: str,
        replace: bool = False,
        require_capability: bool = True,
        source: str | None = None,
    ) -> dict[str, Any]:
        if len(text.encode("utf-8")) > self.config.jsonrpc.manifest_max_bytes:
            raise ValidationError(f"JSON-RPC manifest exceeds manifest_max_bytes={self.config.jsonrpc.manifest_max_bytes}")
        data = load_yaml_mapping(text)
        if set(data) == {"jsonrpc_endpoint"} and isinstance(data["jsonrpc_endpoint"], dict):
            data = data["jsonrpc_endpoint"]
        if set(data) == {"endpoint"} and isinstance(data["endpoint"], dict):
            data = data["endpoint"]
        return self.register_endpoint(
            data,
            actor=actor,
            replace=replace,
            require_capability=require_capability,
            source=source,
        )

    def list_endpoints(
        self,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        text: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        endpoints, _has_more = self.list_endpoints_window(
            actor=actor,
            require_capability=require_capability,
            text=text,
            limit=limit,
        )
        return endpoints

    def list_endpoints_window(
        self,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        text: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return one bounded page plus an exact signal that another row exists."""

        if require_capability and actor is not None:
            self.capabilities.require(actor, self.config.jsonrpc.registry_resource, CapabilityRight.READ)
        selected_limit = self._bounded_list_limit(limit)
        endpoints: list[dict[str, Any]] = []
        rows = self.extensions.list_jsonrpc_endpoints(text=text, limit=selected_limit + 1)
        for spec, metadata in rows[:selected_limit]:
            self._validate_endpoint(spec)
            endpoints.append(self._endpoint_to_json(spec, metadata, include_sensitive_fields=False))
        return endpoints, len(rows) > selected_limit

    def inspect_endpoint(
        self,
        endpoint_id: str,
        *,
        actor: str | None = None,
        require_capability: bool = True,
        include_sensitive_fields: bool = False,
    ) -> dict[str, Any]:
        if require_capability and actor is not None:
            self.capabilities.require(actor, self.endpoint_resource(endpoint_id), CapabilityRight.READ)
        spec, metadata = self._load_endpoint(endpoint_id)
        return self._endpoint_to_json(spec, metadata, include_sensitive_fields=include_sensitive_fields)

    def unregister_endpoint(
        self,
        endpoint_id: str,
        *,
        actor: str = "runtime",
        require_capability: bool = True,
    ) -> dict[str, Any]:
        authority_decision = None
        if require_capability:
            authority_decision = self.capabilities.require(
                actor,
                self.endpoint_resource(endpoint_id),
                CapabilityRight.ADMIN,
                consume=False,
            )
        with self._registry_phase_lock, self.capabilities.authority_transaction(
            [authority_decision],
            actor=actor,
            operation="JSON-RPC endpoint unregister",
        ):
            self._load_endpoint(endpoint_id)
            self._disable_replaced_endpoint_method_capabilities(endpoint_id, actor=actor)
            self.extensions.delete_jsonrpc_endpoint(endpoint_id)
            self.events.emit(
                EventType.EXTERNAL_WRITE,
                source=actor,
                target=self.endpoint_resource(endpoint_id),
                payload={"adapter": "jsonrpc", "operation": "endpoint_unregister", "endpoint_id": endpoint_id},
            )
            self.audit.record(
                actor=actor,
                action="jsonrpc.endpoint.unregister",
                target=self.endpoint_resource(endpoint_id),
                decision={"endpoint_id": endpoint_id},
            )
        return {"endpoint_id": endpoint_id, "deleted": True}

    def call(
        self,
        pid: str,
        endpoint_id: str,
        method_id: str,
        params: Any = None,
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> JsonRpcCallResult:
        try:
            return self._call(
                pid,
                endpoint_id,
                method_id,
                params,
                source_oids=source_oids,
            )
        except ProviderEffectNotStarted as error:
            raise ProviderHostError(
                code="jsonrpc_provider_not_started",
                error_type=type(error).__name__,
                correlation_id=new_id("corr"),
            ) from None

    def _call(
        self,
        pid: str,
        endpoint_id: str,
        method_id: str,
        params: Any = None,
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> JsonRpcCallResult:
        resource, spec, method, registry_binding = self._resolve_call_target(
            pid,
            endpoint_id,
            method_id,
            params,
            source_oids=source_oids,
        )
        request_id = new_id("jrpc")
        operation_context = self._operation_context(
            pid,
            spec,
            method,
            params,
            request_id=request_id,
            registry_binding=registry_binding,
        )
        self._validate_params_against_schema(method, params)
        flow_context = self._data_flow().context_from_source_oids(pid, source_oids)
        sink = DataSink(
            f"jsonrpc:{endpoint_id}:{method_id}",
            self._endpoint_identity_sha256(spec, method),
        )
        self._data_flow().authorize_egress(
            pid=pid,
            sink=sink,
            context=flow_context,
            payload=params,
            operation="jsonrpc.call",
        )
        decision = self._authorize_call(
            pid,
            resource,
            method.right,
            operation_context,
            source_oids=source_oids,
        )
        profile = self.capabilities.profiles.jsonrpc(
            resource=resource,
            effect=decision.effect or CapabilityEffect.DENY,
            endpoint_id=endpoint_id,
            method_id=method_id,
        )
        operation_context.update(
            {
                "capability_ids": list(decision.matched_capability_ids),
                "selected_capability_id": decision.selected_capability_id,
                "sandbox_profile": self._profile_json(profile),
            }
        )
        request_body = self._bounded_request_body(spec, method, params, request_id)
        resource_context = {
            "endpoint_id": endpoint_id,
            "method_id": method_id,
            "request_bytes": len(request_body),
        }
        resource_progress: dict[str, int | None] = {"response_bytes": None}
        failure_resource = partial(
            self._provider_result_failure_resource,
            request_bytes=len(request_body),
            max_response_bytes=spec.max_response_bytes,
            resource_context=resource_context,
            resource_progress=resource_progress,
        )

        effect_context = self._effect_context(spec, method, operation_context, request_bytes=len(request_body))
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=resource,
            decisions=(decision,),
            canonical_args=operation_context,
            observation=effect_context,
            preflight_usage=ResourceUsage(jsonrpc_request_bytes=len(request_body)),
            resource_source="primitive.jsonrpc.call",
            resource_context=resource_context,
            **self._protected_registry_guard(registry_binding, endpoint_id),
            failure_resource=failure_resource,
            failure_evidence=lambda error, phase: self._protected_failure_evidence(pid, resource, method, operation_context, error, phase),
            data_sink=sink,
            data_flow_context=flow_context,
            data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                flow_context,
                origin="external:jsonrpc",
            ),
            data_flow_payload=params,
            data_flow_operation="jsonrpc.call",
        )
        with self._protected().start("primitive.jsonrpc.call", invocation, provider=self.provider) as protected:
            transport = self._dispatch_transport(
                protected,
                spec,
                method,
                request_body,
            )
            transport = self._validated_transport_result(
                transport,
                max_response_bytes=spec.max_response_bytes,
            )
            resource_progress["response_bytes"] = transport.response_bytes
            classification_override = None
            if transport.error is not None:
                classification_override = ExternalEffectClassification(
                    rollback_class=ExternalEffectRollbackClass.UNKNOWN,
                    rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
                    state_mutation=bool(
                        method.state_mutation or method.right != CapabilityRight.READ.value
                    ),
                    information_flow=True,
                    metadata={
                        "outcome": "unknown_transport_failure",
                        "phase": "transport_not_started_after_dns",
                    },
                )
            result = self._call_result_from_transport(spec, method, request_id, transport)
            result_payload = {
                "status": result.status.value,
                "ok": result.ok,
                "http_status": result.http_status,
                "response_bytes": result.response_bytes,
                "duration_s": result.duration_s,
            }
            completed = protected.complete(
                result,
                self._protected_evidence(pid, resource, result, method, operation_context),
                classification_context=effect_context,
                classification_result=result_payload,
                classification_override=classification_override,
                resource=ResourceSettlement(
                    usage=ResourceUsage(
                        jsonrpc_request_bytes=len(request_body),
                        jsonrpc_response_bytes=result.response_bytes,
                    ),
                    source="primitive.jsonrpc.call",
                    context={
                        **resource_context,
                        "response_bytes": result.response_bytes,
                        "status": result.status.value,
                    },
                ),
            )
            return completed

    def _resolve_call_target(
        self,
        pid: str,
        endpoint_id: str,
        method_id: str,
        params: Any,
        *,
        source_oids: list[str] | tuple[str, ...] | None,
    ) -> tuple[
        str,
        JsonRpcEndpointSpec,
        JsonRpcMethodSpec,
        dict[str, Any],
    ]:
        resource = self.method_resource(endpoint_id, method_id)
        self._validate_json_value(params, "params")
        visibility_context = self._visibility_operation_context(
            pid,
            endpoint_id,
            method_id,
            params,
        )
        self._authorize_call_visibility(
            pid,
            resource,
            visibility_context,
            source_oids=source_oids,
        )
        spec, _metadata = self._load_endpoint(endpoint_id)
        method = spec.method_by_id(method_id)
        if method is None:
            raise NotFound(f"JSON-RPC method not found: {endpoint_id}/{method_id}")
        return (
            resource,
            spec,
            method,
            self._registry_binding_for_endpoint_spec(spec),
        )

    def _bounded_request_body(
        self,
        spec: JsonRpcEndpointSpec,
        method: JsonRpcMethodSpec,
        params: Any,
        request_id: str,
    ) -> bytes:
        request_body = self._request_body(method, params, request_id)
        if len(request_body) > spec.max_request_bytes:
            raise ValidationError(
                "JSON-RPC request exceeds "
                f"max_request_bytes={spec.max_request_bytes}"
            )
        return request_body

    async def acall(
        self,
        pid: str,
        endpoint_id: str,
        method_id: str,
        params: Any = None,
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> JsonRpcCallResult:
        return await self._data_flow().run_sync_in_worker(
            self.call,
            pid,
            endpoint_id,
            method_id,
            params,
            source_oids=source_oids,
        )

    def _protected(self):
        return self.protected_operations

    def _data_flow(self) -> Any:
        manager = getattr(self, "data_flow", None) or getattr(
            self._protected(),
            "data_flow",
            None,
        )
        if manager is None:
            raise ValidationError("JSON-RPC data-flow manager is not attached")
        return manager

    @staticmethod
    def _endpoint_identity_sha256(
        spec: JsonRpcEndpointSpec,
        method: JsonRpcMethodSpec,
    ) -> str:
        return hashlib.sha256(
            dumps(
                to_jsonable(
                    {
                        "schema_version": 1,
                        "endpoint": spec,
                        "method": method,
                    }
                )
            ).encode("utf-8")
        ).hexdigest()

    def _protected_evidence(
        self,
        pid: str,
        resource: str,
        result: JsonRpcCallResult,
        method: JsonRpcMethodSpec,
        operation_context: dict[str, Any],
    ) -> ProtectedOperationEvidence:
        event_type = (
            EventType.EXTERNAL_WRITE
            if method.state_mutation or method.right != CapabilityRight.READ.value
            else EventType.EXTERNAL_READ
        )
        return ProtectedOperationEvidence(
            event_type=event_type,
            event_source=pid,
            event_target=resource,
            event_payload={
                "adapter": "jsonrpc",
                "endpoint_id": result.endpoint_id,
                "method_id": result.method_id,
                "status": result.status.value,
                "ok": result.ok,
                "http_status": result.http_status,
                "response_bytes": result.response_bytes,
                "duration_s": result.duration_s,
            },
            audit_action="primitive.jsonrpc.call",
            audit_actor=pid,
            audit_target=resource,
            audit_decision={
                "endpoint_id": result.endpoint_id,
                "method_id": result.method_id,
                "rpc_method": method.rpc_method,
                "right": method.right,
                "request_id": result.request_id,
                "params_sha256": operation_context["params_sha256"],
                "params_preview": operation_context["params_preview"],
                "params_observation": operation_context["params_observation"],
                "sandbox_profile": operation_context.get("sandbox_profile"),
                "status": result.status.value,
                "ok": result.ok,
                "http_status": result.http_status,
                "response_bytes": result.response_bytes,
                "duration_s": result.duration_s,
            },
            capability_refs=tuple(operation_context.get("capability_ids") or ()),
            effect_metadata={
                "status": result.status.value,
                "ok": result.ok,
                "http_status": result.http_status,
                "response_bytes": result.response_bytes,
                "duration_s": result.duration_s,
            },
        )

    def _protected_failure_evidence(
        self,
        pid: str,
        resource: str,
        method: JsonRpcMethodSpec,
        operation_context: dict[str, Any],
        error: BaseException,
        phase: str,
    ) -> ProtectedOperationEvidence:
        result = JsonRpcCallResult(
            endpoint_id=str(operation_context["endpoint_id"]),
            method_id=method.method_id,
            rpc_method=method.rpc_method,
            request_id=str(operation_context["request_id"]),
            status=JsonRpcCallStatus.TRANSPORT_ERROR,
            http_status=None,
            ok=False,
            error={"message": {"redacted": True}, "phase": phase},
            response_bytes=0,
            duration_s=0.0,
        )
        evidence = self._protected_evidence(pid, resource, result, method, operation_context)
        return ProtectedOperationEvidence(
            **{
                **evidence.__dict__,
                "audit_decision": {
                    **dict(evidence.audit_decision),
                    "effect_outcome": "unknown",
                    "error_type": type(error).__name__,
                    "phase": phase,
                },
            }
        )

    def grant_method(
        self,
        pid: str,
        endpoint_id: str,
        method_id: str,
        *,
        right: str | CapabilityRight,
        issued_by: str = "jsonrpc",
        delegable: bool = True,
    ) -> Any:
        return self.capabilities.grant(
            subject=pid,
            resource=self.method_resource(endpoint_id, method_id),
            rights=[CapabilityRight(str(right))],
            issued_by=issued_by,
            delegable=delegable,
        )

    def _authorize_call(
        self,
        pid: str,
        resource: str,
        right: str,
        context: dict[str, Any],
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> Any:
        decision = self.capabilities.authorize(pid, resource, right, context, audit=True)
        if decision.allowed:
            return decision
        if decision.policy == CapabilityManager.ASK_EACH_TIME:
            if self.human is None:
                raise CapabilityDenied(f"{pid} requires human approval for JSON-RPC call on {resource}")
            profile = self.capabilities.profiles.jsonrpc(
                resource=resource,
                effect=CapabilityEffect.ASK,
                endpoint_id=str(context["endpoint_id"]),
                method_id=str(context["method_id"]),
            )
            approval_context = {**context, "sandbox_profile": self._profile_json(profile)}
            request_id = self.human.query(
                pid=pid,
                human=self.config.runtime.default_human,
                request={
                    "type": "external_operation_approval",
                    "question": f"Allow this process to call remote JSON-RPC method {resource}?",
                    "requested_once_capability": {
                        "subject": pid,
                        "resource": resource,
                        "rights": [right],
                        "constraints": self._approval_constraints(context),
                    },
                    "context": approval_context,
                },
                blocking=True,
                source_oids=source_oids,
            )
            raise HumanApprovalRequired(
                request_id=request_id,
                message=f"{pid} is waiting for per-use human approval to call {resource}",
            )
        raise CapabilityDenied(decision.reason)

    def _authorize_call_visibility(
        self,
        pid: str,
        resource: str,
        context: dict[str, Any],
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        bound_context: dict[str, Any] | None = None
        for right in (CapabilityRight.READ, CapabilityRight.WRITE, CapabilityRight.EXECUTE):
            selected_context = {**context, "right": str(right)}
            decision = self.capabilities.authorize(pid, resource, right, selected_context)
            if decision.allowed:
                return
            # Preserve the no-oracle boundary for callers with no matching
            # authority. Only an ASK policy or an existing constrained grant
            # may resolve the metadata-free registry binding needed by an
            # exact approval.
            if not (
                decision.policy == CapabilityManager.ASK_EACH_TIME
                or decision.matched_capability_ids
            ):
                continue
            if bound_context is None:
                bound_context = {
                    **context,
                    **self._registry_binding_context(str(context["endpoint_id"])),
                }
            rebound = self.capabilities.authorize(
                pid,
                resource,
                right,
                {**bound_context, "right": str(right)},
            )
            if rebound.allowed:
                return
            if rebound.policy == CapabilityManager.ASK_EACH_TIME:
                self._request_visibility_approval(
                    pid,
                    resource,
                    str(right),
                    bound_context,
                    source_oids=source_oids,
                )
        raise CapabilityDenied(f"{pid} lacks JSON-RPC call authority on {resource}")

    def _request_visibility_approval(
        self,
        pid: str,
        resource: str,
        right: str,
        context: dict[str, Any],
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if self.human is None:
            raise CapabilityDenied(f"{pid} requires human approval for JSON-RPC call on {resource}")
        profile = self.capabilities.profiles.jsonrpc(
            resource=resource,
            effect=CapabilityEffect.ASK,
            endpoint_id=str(context["endpoint_id"]),
            method_id=str(context["method_id"]),
        )
        approval_context = {**context, "right": right, "sandbox_profile": self._profile_json(profile)}
        request_id = self.human.query(
            pid=pid,
            human=self.config.runtime.default_human,
            request={
                "type": "external_operation_approval",
                "question": f"Allow this process to call remote JSON-RPC method {resource}?",
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [right],
                    "constraints": self._approval_constraints(context),
                },
                "context": approval_context,
            },
            blocking=True,
            source_oids=source_oids,
        )
        raise HumanApprovalRequired(
            request_id=request_id,
            message=f"{pid} is waiting for per-use human approval to call {resource}",
        )

    def _request_body(self, method: JsonRpcMethodSpec, params: Any, request_id: str) -> bytes:
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method.rpc_method,
        }
        if params is not None:
            payload["params"] = params
        return dumps(payload).encode("utf-8")

    def _invoke_transport_provider(
        self,
        endpoint: JsonRpcEndpointSpec,
        method: JsonRpcMethodSpec,
        request_body: bytes,
        *,
        resolved_addresses: tuple[str, ...],
        resolved_headers: Mapping[str, str],
        deadline: float,
        started: float,
    ) -> JsonRpcTransportResult:
        remaining = min(endpoint.timeout_s, deadline - time.monotonic())
        if remaining <= 0:
            raise _JsonRpcTransportNotStarted(
                "JSON-RPC absolute deadline expired before transport dispatch"
            )
        try:
            raw_result = self.provider.call(
                endpoint,
                method,
                request_body,
                timeout_s=remaining,
                max_response_bytes=endpoint.max_response_bytes,
                resolved_addresses=resolved_addresses,
                resolved_headers=resolved_headers,
            )
        except ProviderEffectNotStarted:
            raise
        except Exception as error:
            return JsonRpcTransportResult(
                status_code=None,
                body=b"",
                elapsed_s=time.monotonic() - started,
                response_bytes=0,
                error="provider transport failed",
                error_type=type(error).__name__,
                correlation_id=new_id("corr"),
            )
        return self._validated_transport_result(
            raw_result,
            max_response_bytes=endpoint.max_response_bytes,
        )

    def _dispatch_transport(
        self,
        protected: Any,
        endpoint: JsonRpcEndpointSpec,
        method: JsonRpcMethodSpec,
        request_body: bytes,
    ) -> JsonRpcTransportResult:
        started = time.monotonic()
        deadline = started + endpoint.timeout_s
        resolved_headers = self._require_header_environment(endpoint)
        previous_deadline = getattr(self._resolution_deadline, "current", None)
        self._resolution_deadline.current = deadline
        try:
            resolved_addresses = protected.call(
                ProviderPhase("dns_resolution", information_flow=True),
                self._validate_runtime_resolution,
                endpoint,
            )
        finally:
            if previous_deadline is None:
                with contextlib.suppress(AttributeError):
                    del self._resolution_deadline.current
            else:
                self._resolution_deadline.current = previous_deadline
        return protected.call(
            ProviderPhase(
                "transport_not_started_after_dns",
                state_mutation=bool(
                    method.state_mutation
                    or method.right != CapabilityRight.READ.value
                ),
                information_flow=True,
            ),
            self._invoke_transport_provider,
            endpoint,
            method,
            request_body,
            resolved_addresses=resolved_addresses,
            resolved_headers=resolved_headers,
            deadline=deadline,
            started=started,
        )

    @staticmethod
    def _provider_result_failure_resource(
        error: BaseException,
        phase: str,
        *,
        request_bytes: int,
        max_response_bytes: int,
        resource_context: dict[str, Any],
        resource_progress: dict[str, int | None],
    ) -> ResourceSettlement | None:
        if not _provider_result_was_returned(error):
            return None
        known_response_bytes = resource_progress["response_bytes"]
        unknown_response_bytes = (
            max_response_bytes if known_response_bytes is None else 0
        )
        return ResourceSettlement(
            usage=ResourceUsage(
                jsonrpc_request_bytes=request_bytes,
                jsonrpc_response_bytes=(
                    known_response_bytes
                    if known_response_bytes is not None
                    else max_response_bytes
                ),
            ),
            source="primitive.jsonrpc.call",
            context={
                **resource_context,
                "failure_phase": phase,
                "response_bytes": known_response_bytes or 0,
                "unknown_response_bytes": unknown_response_bytes,
                "provider_result_returned": True,
            },
        )

    def _call_result_from_transport(
        self,
        endpoint: JsonRpcEndpointSpec,
        method: JsonRpcMethodSpec,
        request_id: str,
        transport: JsonRpcTransportResult,
    ) -> JsonRpcCallResult:
        transport = self._validated_transport_result(
            transport,
            max_response_bytes=endpoint.max_response_bytes,
        )
        if transport.error and transport.status_code is None:
            return JsonRpcCallResult(
                endpoint_id=endpoint.endpoint_id,
                method_id=method.method_id,
                rpc_method=method.rpc_method,
                request_id=request_id,
                status=JsonRpcCallStatus.TRANSPORT_ERROR,
                http_status=None,
                ok=False,
                error={
                    "code": "jsonrpc_transport_error",
                    "error_type": transport.error_type or "TransportError",
                    "correlation_id": transport.correlation_id or new_id("corr"),
                },
                response_bytes=transport.response_bytes,
                duration_s=transport.elapsed_s,
            )
        if transport.too_large:
            return self._failure(
                endpoint,
                method,
                request_id,
                JsonRpcCallStatus.RESPONSE_TOO_LARGE,
                transport,
                f"response exceeded max_response_bytes={endpoint.max_response_bytes}",
            )
        if transport.status_code is None or not 200 <= transport.status_code < 300:
            return self._failure(
                endpoint,
                method,
                request_id,
                JsonRpcCallStatus.HTTP_ERROR,
                transport,
                "HTTP status was not successful",
                extra={"body_observation": self._body_observation(transport.body)},
            )
        try:
            envelope = json.loads(
                transport.body.decode("utf-8"),
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ValueError("non-finite JSON number")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return self._failure(
                endpoint,
                method,
                request_id,
                JsonRpcCallStatus.INVALID_RESPONSE,
                transport,
                "response body is not valid UTF-8 JSON",
            )
        except Exception as error:
            raise _mark_provider_result_returned(
                ProviderHostError(
                    code="jsonrpc_provider_error",
                    error_type=type(error).__name__,
                    correlation_id=new_id("corr"),
                )
            ) from None
        if not isinstance(envelope, dict):
            return self._failure(endpoint, method, request_id, JsonRpcCallStatus.INVALID_RESPONSE, transport, "response is not a JSON object")
        if envelope.get("jsonrpc") != "2.0":
            return self._failure(endpoint, method, request_id, JsonRpcCallStatus.INVALID_RESPONSE, transport, "missing jsonrpc=2.0")
        if envelope.get("id") != request_id:
            return self._failure(endpoint, method, request_id, JsonRpcCallStatus.INVALID_RESPONSE, transport, "response id mismatch")
        has_result = "result" in envelope
        has_error = "error" in envelope
        if has_result == has_error:
            return self._failure(
                endpoint,
                method,
                request_id,
                JsonRpcCallStatus.INVALID_RESPONSE,
                transport,
                "response must contain exactly one of result or error",
            )
        if has_error:
            error = envelope["error"]
            if (
                not isinstance(error, dict)
                or type(error.get("code")) is not int
                or type(error.get("message")) is not str
            ):
                return self._failure(
                    endpoint,
                    method,
                    request_id,
                    JsonRpcCallStatus.INVALID_RESPONSE,
                    transport,
                    "JSON-RPC error object is invalid",
                )
            return JsonRpcCallResult(
                endpoint_id=endpoint.endpoint_id,
                method_id=method.method_id,
                rpc_method=method.rpc_method,
                request_id=request_id,
                status=JsonRpcCallStatus.JSONRPC_ERROR,
                http_status=transport.status_code,
                ok=False,
                error=to_jsonable(error),
                response_bytes=transport.response_bytes,
                duration_s=transport.elapsed_s,
            )
        return JsonRpcCallResult(
            endpoint_id=endpoint.endpoint_id,
            method_id=method.method_id,
            rpc_method=method.rpc_method,
            request_id=request_id,
            status=JsonRpcCallStatus.OK,
            http_status=transport.status_code,
            ok=True,
            result=to_jsonable(envelope.get("result")),
            response_bytes=transport.response_bytes,
            duration_s=transport.elapsed_s,
        )

    @staticmethod
    def _validated_transport_result(
        result: Any,
        *,
        max_response_bytes: int,
    ) -> JsonRpcTransportResult:
        """Decode provider output and derive bounded accounting from its body."""

        try:
            if not isinstance(result, JsonRpcTransportResult):
                raise TypeError("JSON-RPC provider returned an invalid transport result")
            status_code = result.status_code
            body = result.body
            elapsed_s = result.elapsed_s
            error = result.error
            error_type = result.error_type
            correlation_id = result.correlation_id
            if status_code is not None and type(status_code) is not int:
                raise TypeError("JSON-RPC provider status_code is invalid")
            if type(body) is not bytes:
                raise TypeError("JSON-RPC provider body is invalid")
            if (
                type(elapsed_s) not in {int, float}
                or not math.isfinite(elapsed_s)
                or elapsed_s < 0
            ):
                raise TypeError("JSON-RPC provider elapsed_s is invalid")
            for field_name, selected in (
                ("error", error),
                ("error_type", error_type),
                ("correlation_id", correlation_id),
            ):
                if selected is not None and type(selected) is not str:
                    raise TypeError(
                        f"JSON-RPC provider {field_name} is invalid"
                    )
            actual_response_bytes = len(body)
            too_large = actual_response_bytes > max_response_bytes
            return JsonRpcTransportResult(
                status_code=status_code,
                # Preserve one sentinel byte so repeated validation cannot
                # lose an already-observed over-limit condition.
                body=bytes(body[: max_response_bytes + 1]),
                elapsed_s=float(elapsed_s),
                response_bytes=min(actual_response_bytes, max_response_bytes),
                too_large=too_large,
                error=error,
                error_type=error_type,
                correlation_id=correlation_id,
            )
        except ProviderHostError as error:
            _mark_provider_result_returned(error)
            raise
        except Exception as error:
            raise _mark_provider_result_returned(
                ProviderHostError(
                    code="jsonrpc_provider_error",
                    error_type=type(error).__name__,
                    correlation_id=new_id("corr"),
                )
            ) from None

    def _failure(
        self,
        endpoint: JsonRpcEndpointSpec,
        method: JsonRpcMethodSpec,
        request_id: str,
        status: JsonRpcCallStatus,
        transport: JsonRpcTransportResult,
        message: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> JsonRpcCallResult:
        return JsonRpcCallResult(
            endpoint_id=endpoint.endpoint_id,
            method_id=method.method_id,
            rpc_method=method.rpc_method,
            request_id=request_id,
            status=status,
            http_status=transport.status_code,
            ok=False,
            error={"message": message, **dict(extra or {})},
            response_bytes=transport.response_bytes,
            duration_s=transport.elapsed_s,
        )

    def _operation_context(
        self,
        pid: str,
        endpoint: JsonRpcEndpointSpec,
        method: JsonRpcMethodSpec,
        params: Any,
        *,
        request_id: str,
        registry_binding: dict[str, Any],
    ) -> dict[str, Any]:
        params_json = dumps(params)
        params_observation = sanitize_for_observability(
            {"params": params},
            preview_chars=self.config.jsonrpc.audit_preview_chars,
        )
        return {
            "pid": pid,
            "primitive": "runtime.jsonrpc.call",
            "operation": "jsonrpc.call",
            "authority_operation": "jsonrpc.call",
            "endpoint_id": endpoint.endpoint_id,
            "method_id": method.method_id,
            "rpc_method": method.rpc_method,
            "right": method.right,
            "request_id": request_id,
            **registry_binding,
            "params_sha256": hashlib.sha256(params_json.encode("utf-8")).hexdigest(),
            "params_preview": params_observation["preview"],
            "params_observation": params_observation,
        }

    def _visibility_operation_context(self, pid: str, endpoint_id: str, method_id: str, params: Any) -> dict[str, Any]:
        params_json = dumps(params)
        return {
            "pid": pid,
            "primitive": "runtime.jsonrpc.call",
            "operation": "jsonrpc.call",
            "authority_operation": "jsonrpc.call",
            "endpoint_id": endpoint_id,
            "method_id": method_id,
            "params_sha256": hashlib.sha256(params_json.encode("utf-8")).hexdigest(),
        }

    def _approval_constraints(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            AUTHORITY_RULES_KEY: [
                {
                    "rule_id": f"jsonrpc.approval.{context['endpoint_id']}.{context['method_id']}",
                    "operation": "jsonrpc.call",
                    "effect": CapabilityEffect.ALLOW.value,
                    "risk": "high",
                    "conditions": {
                        "endpoint_id": context["endpoint_id"],
                        "method_id": context["method_id"],
                        "registry_spec_sha256": context["registry_spec_sha256"],
                        "registry_generation": context["registry_generation"],
                        "params_sha256": context["params_sha256"],
                    },
                    "description": "one-shot human approval for exact JSON-RPC call payload",
                }
            ]
        }

    @staticmethod
    def _endpoint_spec_sha256(endpoint: JsonRpcEndpointSpec) -> str:
        return hashlib.sha256(dumps(endpoint).encode("utf-8")).hexdigest()

    def _registry_binding_context(self, endpoint_id: str) -> dict[str, Any]:
        binding = self.extensions.get_jsonrpc_registry_binding(endpoint_id)
        if not isinstance(binding, dict):
            raise ValidationError("JSON-RPC registry binding must be an object")
        generation = binding.get("registry_generation")
        digest = binding.get("registry_spec_sha256")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValidationError("JSON-RPC registry generation is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValidationError("JSON-RPC registry spec digest is invalid")
        return {
            "registry_spec_sha256": digest,
            "registry_generation": generation,
        }

    def _registry_binding_for_endpoint_spec(
        self,
        endpoint: JsonRpcEndpointSpec,
    ) -> dict[str, Any]:
        binding = self._registry_binding_context(endpoint.endpoint_id)
        if binding["registry_spec_sha256"] != self._endpoint_spec_sha256(endpoint):
            raise CapabilityDenied(
                "JSON-RPC endpoint registry changed before call authorization"
            )
        return binding

    def _protected_registry_guard(
        self,
        binding: dict[str, Any],
        endpoint_id: str,
    ) -> dict[str, Any]:
        return {
            "provider_registry_binding": ProviderRegistryBinding.from_context(binding),
            "provider_registry_binding_resolver": lambda: ProviderRegistryBinding.from_context(
                self._registry_binding_context(endpoint_id)
            ),
            "provider_registry_phase_guard": lambda: self._registry_phase_lock,
        }

    def _effect_context(
        self,
        endpoint: JsonRpcEndpointSpec,
        method: JsonRpcMethodSpec,
        operation_context: dict[str, Any],
        *,
        request_bytes: int,
    ) -> dict[str, Any]:
        return {
            **operation_context,
            "request_bytes": request_bytes,
            "endpoint_id": endpoint.endpoint_id,
            "method_id": method.method_id,
            "rpc_method": method.rpc_method,
            "method": {
                "right": method.right,
                "rollback_class": method.rollback_class,
                "rollback_status": (
                    method.rollback_status
                    if method.rollback_status is not None
                    else self._default_rollback_status(method.rollback_class)
                ),
                "state_mutation": method.state_mutation,
                "information_flow": method.information_flow,
            },
        }

    def _coerce_endpoint(self, value: JsonRpcEndpointSpec | dict[str, Any]) -> JsonRpcEndpointSpec:
        if isinstance(value, JsonRpcEndpointSpec):
            # Typed callers still run through the same canonical coercion as
            # mapping/YAML callers.  Python does not enforce dataclass
            # annotations at runtime, so a valid ``timeout_s=1`` would
            # otherwise persist as JSON integer ``1`` and decode as float
            # ``1.0``.  Registry digests must not depend on that incidental
            # construction spelling.
            value = to_jsonable(value)
        if not isinstance(value, dict):
            raise ValidationError("JSON-RPC endpoint must be a mapping")
        unknown = sorted(set(value) - {
            "schema_version",
            "endpoint_id",
            "url",
            "headers",
            "methods",
            "timeout_s",
            "max_request_bytes",
            "max_response_bytes",
            "metadata",
        })
        if unknown:
            raise ValidationError(f"unknown JSON-RPC endpoint fields: {unknown}")
        methods = value.get("methods")
        if not isinstance(methods, list) or not methods:
            raise ValidationError("JSON-RPC endpoint requires a non-empty methods list")
        spec = JsonRpcEndpointSpec(
            schema_version=self._coerce_positive_int(value.get("schema_version", 1), "schema_version"),
            endpoint_id=self._require_string(value.get("endpoint_id"), "endpoint_id"),
            url=self._require_string(value.get("url"), "url"),
            headers=self._header_specs(
                self._mapping_field(value, "headers", "JSON-RPC endpoint")
            ),
            methods=[self._method_spec(item) for item in methods],
            timeout_s=self._coerce_positive_float(value.get("timeout_s", self.config.jsonrpc.timeout_s), "timeout_s"),
            max_request_bytes=self._coerce_positive_int(
                value.get("max_request_bytes", self.config.jsonrpc.max_request_bytes),
                "max_request_bytes",
            ),
            max_response_bytes=self._coerce_positive_int(
                value.get("max_response_bytes", self.config.jsonrpc.max_response_bytes),
                "max_response_bytes",
            ),
            metadata=self._mapping_field(value, "metadata", "JSON-RPC endpoint"),
        )
        self._validate_endpoint(spec)
        return spec

    def _validate_endpoint(self, endpoint: JsonRpcEndpointSpec) -> None:
        if endpoint.schema_version != 1:
            raise ValidationError("JSON-RPC endpoint schema_version must be 1")
        self._validate_identifier(endpoint.endpoint_id, "endpoint_id", self.config.jsonrpc.endpoint_id_max_chars)
        self._validate_url(endpoint.url)
        self._validate_positive_finite(endpoint.timeout_s, "timeout_s")
        self._validate_positive_integer(endpoint.max_request_bytes, "max_request_bytes")
        self._validate_positive_integer(endpoint.max_response_bytes, "max_response_bytes")
        if endpoint.timeout_s > self.config.jsonrpc.timeout_hard_limit_s:
            raise ValidationError("JSON-RPC endpoint timeout_s exceeds configured bounds")
        if endpoint.max_request_bytes > self.config.jsonrpc.max_request_hard_limit_bytes:
            raise ValidationError("JSON-RPC endpoint max_request_bytes exceeds configured bounds")
        if endpoint.max_response_bytes > self.config.jsonrpc.max_response_hard_limit_bytes:
            raise ValidationError("JSON-RPC endpoint max_response_bytes exceeds configured bounds")
        self._validate_json_value(endpoint.metadata, "metadata")
        seen: set[str] = set()
        for name, header in endpoint.headers.items():
            self._validate_header_name(name)
            self._validate_header_value_part(header.env, "header env")
            self._validate_header_env_allowed(header.env)
            self._validate_header_value_part(header.prefix, "header prefix")
            self._validate_header_value_part(header.suffix, "header suffix")
        for method in endpoint.methods:
            self._validate_method(method)
            if method.method_id in seen:
                raise ValidationError(f"duplicate JSON-RPC method_id: {method.method_id}")
            seen.add(method.method_id)

    def _method_spec(self, value: Any) -> JsonRpcMethodSpec:
        if not isinstance(value, dict):
            raise ValidationError("JSON-RPC method entries must be mappings")
        unknown = sorted(set(value) - {
            "method_id",
            "rpc_method",
            "right",
            "rollback_class",
            "rollback_status",
            "state_mutation",
            "information_flow",
            "params_schema",
            "metadata",
        })
        if unknown:
            raise ValidationError(f"unknown JSON-RPC method fields: {unknown}")
        return JsonRpcMethodSpec(
            method_id=self._require_string(value.get("method_id"), "method_id"),
            rpc_method=self._require_string(value.get("rpc_method"), "rpc_method"),
            right=self._require_string(value.get("right"), "right"),
            rollback_class=self._require_string(value.get("rollback_class"), "rollback_class"),
            rollback_status=str(value["rollback_status"]) if value.get("rollback_status") is not None else None,
            state_mutation=self._require_bool(value.get("state_mutation"), "state_mutation"),
            information_flow=self._require_bool(value.get("information_flow"), "information_flow"),
            params_schema=self._mapping_field(value, "params_schema", "JSON-RPC method"),
            metadata=self._mapping_field(value, "metadata", "JSON-RPC method"),
        )

    def _validate_method(self, method: JsonRpcMethodSpec) -> None:
        self._validate_identifier(method.method_id, "method_id", self.config.jsonrpc.method_id_max_chars)
        if not method.rpc_method or len(method.rpc_method) > self.config.jsonrpc.rpc_method_max_chars:
            raise ValidationError("JSON-RPC rpc_method is empty or too long")
        if any(ord(char) < 32 for char in method.rpc_method):
            raise ValidationError("JSON-RPC rpc_method contains control characters")
        if method.right not in _CALL_RIGHTS:
            raise ValidationError("JSON-RPC method right must be read, write, or execute")
        try:
            rollback_class = ExternalEffectRollbackClass(method.rollback_class)
            rollback_status = (
                method.rollback_status
                if method.rollback_status is not None
                else self._default_rollback_status(method.rollback_class)
            )
            ExternalEffectRollbackStatus(rollback_status)
        except ValueError as exc:
            raise ValidationError("JSON-RPC method has invalid rollback_class or rollback_status") from exc
        if rollback_class == ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED and bool(method.state_mutation):
            raise ValidationError("no_rollback_required JSON-RPC methods cannot declare state_mutation=true")
        self._validate_params_schema(method.params_schema)
        self._validate_json_value(method.metadata, "method metadata")

    def _validate_params_schema(self, schema: dict[str, Any]) -> None:
        if not schema:
            return
        self._validate_json_value(schema, "method params_schema")
        try:
            jsonschema_validator_for(schema).check_schema(schema)
        except JsonSchemaSchemaError as exc:
            raise ValidationError("JSON-RPC method params_schema is not a valid JSON Schema") from exc

    def _validate_params_against_schema(self, method: JsonRpcMethodSpec, params: Any) -> None:
        schema = method.params_schema
        if not schema:
            return
        try:
            validator_cls = jsonschema_validator_for(schema)
            validator_cls.check_schema(schema)
            validator_cls(schema).validate(params)
        except JsonSchemaSchemaError as exc:
            raise ValidationError("JSON-RPC method params_schema is not a valid JSON Schema") from exc
        except JsonSchemaValidationError as exc:
            path = ".".join(str(item) for item in exc.path)
            location = f" at {path}" if path else ""
            raise ValidationError(f"JSON-RPC params do not match params_schema{location}") from exc

    def _header_specs(self, value: Any) -> dict[str, JsonRpcHeaderSpec]:
        if not isinstance(value, dict):
            raise ValidationError("JSON-RPC headers must be a mapping")
        headers: dict[str, JsonRpcHeaderSpec] = {}
        for raw_name, raw_spec in value.items():
            name = str(raw_name)
            if not isinstance(raw_spec, dict) or "env" not in raw_spec:
                raise ValidationError("JSON-RPC headers must use env-backed mappings")
            unknown = sorted(set(raw_spec) - {"env", "prefix", "suffix"})
            if unknown:
                raise ValidationError(f"unknown JSON-RPC header fields for {name}: {unknown}")
            headers[name] = JsonRpcHeaderSpec(
                env=self._require_string(raw_spec.get("env"), f"headers.{name}.env"),
                prefix=str(raw_spec.get("prefix", "")),
                suffix=str(raw_spec.get("suffix", "")),
            )
        return headers

    def _validate_header_name(self, name: str) -> None:
        if len(name) > self.config.jsonrpc.header_name_max_chars or not _HEADER_PATTERN.match(name):
            raise ValidationError(f"invalid JSON-RPC header name: {name!r}")
        if name.lower() in _FORBIDDEN_HEADERS:
            raise ValidationError(f"JSON-RPC header cannot be configured by endpoint manifest: {name}")

    def _validate_header_value_part(self, value: str, field: str) -> None:
        if len(value) > self.config.jsonrpc.header_value_max_chars:
            raise ValidationError(f"JSON-RPC {field} exceeds configured maximum length")
        if "\r" in value or "\n" in value:
            raise ValidationError(f"JSON-RPC {field} contains a newline")
        if field == "header env" and not _ENV_PATTERN.match(value):
            raise ValidationError(f"JSON-RPC header env is not a valid environment variable name: {value!r}")
        if field == "header prefix" and value not in _ALLOWED_HEADER_PREFIXES:
            raise ValidationError("JSON-RPC header prefix must be empty or an approved auth scheme")
        if field == "header suffix" and value not in _ALLOWED_HEADER_SUFFIXES:
            raise ValidationError("JSON-RPC header suffix must be empty")

    def _validate_header_env_allowed(self, value: str) -> None:
        for pattern in self.config.jsonrpc.header_env_allowlist:
            if pattern.endswith("*") and value.startswith(pattern[:-1]):
                return
            if value == pattern:
                return
        raise ValidationError(f"JSON-RPC header env is not allowed for endpoint manifests: {value!r}")

    def _require_header_environment(
        self,
        endpoint: JsonRpcEndpointSpec,
    ) -> Mapping[str, str]:
        missing: list[str] = []
        invalid: list[str] = []
        resolved_headers: dict[str, str] = {}
        host_environment = dict(os.environ)
        for name, spec in endpoint.headers.items():
            value = host_environment.get(spec.env)
            if value is None:
                missing.append(spec.env)
                continue
            resolved = f"{spec.prefix}{value}{spec.suffix}"
            if len(resolved) > self.config.jsonrpc.header_value_max_chars or "\r" in resolved or "\n" in resolved:
                invalid.append(spec.env)
                continue
            try:
                resolved.encode("iso-8859-1")
            except UnicodeEncodeError:
                invalid.append(spec.env)
                continue
            resolved_headers[name] = resolved
        if missing:
            raise ValidationError(f"missing JSON-RPC header environment variables: {missing}")
        if invalid:
            raise ValidationError(f"invalid JSON-RPC header environment variable values: {invalid}")
        # The provider receives a per-operation immutable snapshot.  It must
        # not re-read ambient environment variables after this preflight,
        # otherwise a concurrent mutation can replace an already-validated
        # credential or introduce invalid header bytes before dispatch.
        return MappingProxyType(resolved_headers)

    def _validate_url(self, url: str) -> None:
        parsed = urlsplit(url)
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValidationError("JSON-RPC endpoint URL has invalid port") from exc
        if parsed.username or parsed.password:
            raise ValidationError("JSON-RPC endpoint URL must not include userinfo")
        if parsed.fragment:
            raise ValidationError("JSON-RPC endpoint URL must not include a fragment")
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValidationError("JSON-RPC endpoint URL must be HTTP(S)")
        host = (parsed.hostname or "").rstrip(".").lower()
        self._validate_remote_host(host)
        if parsed.scheme == "http" and host not in _LOCAL_HTTP_HOSTS:
            raise ValidationError("plain HTTP JSON-RPC endpoints are restricted to localhost")

    def _validate_remote_host(self, host: str) -> None:
        if host in _FORBIDDEN_JSONRPC_HOSTS:
            raise ValidationError("JSON-RPC endpoint host is blocked")
        try:
            address = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            return
        self._validate_remote_address(address, allow_loopback=True)

    def _validate_runtime_resolution(self, endpoint: JsonRpcEndpointSpec) -> tuple[str, ...]:
        parsed = urlsplit(endpoint.url)
        host = (parsed.hostname or "").rstrip(".").lower()
        if not host:
            raise ValidationError("JSON-RPC endpoint URL host is empty")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        deadline = getattr(
            self._resolution_deadline,
            "current",
            time.monotonic() + endpoint.timeout_s,
        )
        try:
            addresses = _bounded_jsonrpc_getaddrinfo(
                host,
                port,
                deadline=deadline,
            )
        except TimeoutError:
            raise
        except OSError as exc:
            raise ValidationError(f"JSON-RPC endpoint host could not be resolved: {host}") from exc
        if not addresses:
            raise ValidationError(f"JSON-RPC endpoint host resolved no addresses: {host}")
        allow_loopback = host in _LOCAL_HTTP_HOSTS
        resolved: list[str] = []
        for item in addresses:
            raw_address = str(item[4][0]).split("%", 1)[0]
            try:
                address = ipaddress.ip_address(raw_address)
            except ValueError as exc:
                raise ValidationError(f"JSON-RPC endpoint resolved invalid address: {raw_address}") from exc
            self._validate_remote_address(address, allow_loopback=allow_loopback)
            text = str(address)
            if text not in resolved:
                resolved.append(text)
        return tuple(resolved)

    def _validate_remote_address(self, address: ipaddress.IPv4Address | ipaddress.IPv6Address, *, allow_loopback: bool) -> None:
        if address.is_loopback:
            if allow_loopback:
                return
            raise ValidationError("JSON-RPC endpoint resolved to loopback address")
        if not address.is_global:
            raise ValidationError("JSON-RPC endpoint resolved to non-public address")
        if (
            address.is_private
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise ValidationError("JSON-RPC endpoint IP address is not allowed")

    def _disable_replaced_endpoint_method_capabilities(self, endpoint_id: str, *, actor: str) -> None:
        prefix = f"jsonrpc:{endpoint_id}:"
        for cap in self.authority.list_capabilities():
            if not cap.active or cap.revoked:
                continue
            if cap.resource == f"jsonrpc:{endpoint_id}:*" or cap.resource.startswith(prefix):
                self.capabilities.disable_subject_capability(
                    cap.cap_id,
                    actor=actor,
                    reason="JSON-RPC endpoint spec replaced; method authority must be reissued",
                )

    def _validate_identifier(self, value: str, field: str, max_chars: int) -> None:
        if len(value) > max_chars or not _ID_PATTERN.match(value):
            raise ValidationError(f"invalid JSON-RPC {field}: {value!r}")

    @staticmethod
    def _mapping_field(
        value: dict[str, Any],
        field: str,
        context: str,
    ) -> dict[str, Any]:
        if field not in value:
            return {}
        selected = value[field]
        if not isinstance(selected, dict):
            raise ValidationError(f"{context} {field} must be a mapping")
        return dict(selected)

    def _require_string(self, value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"JSON-RPC {field} must be a non-empty string")
        return value.strip()

    def _require_bool(self, value: Any, field: str) -> bool:
        if not isinstance(value, bool):
            raise ValidationError(f"JSON-RPC {field} must be a boolean")
        return value

    def _coerce_positive_float(self, value: Any, field: str) -> float:
        if isinstance(value, bool):
            raise ValidationError(f"JSON-RPC {field} must be a number")
        try:
            selected = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"JSON-RPC {field} must be a number") from exc
        self._validate_positive_finite(selected, field)
        return selected

    def _coerce_positive_int(self, value: Any, field: str) -> int:
        if type(value) is not int:
            raise ValidationError(f"JSON-RPC {field} must be an integer")
        selected = value
        self._validate_positive_integer(selected, field)
        return selected

    def _validate_positive_finite(self, value: Any, field: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValidationError(f"JSON-RPC {field} must be finite")
        if float(value) <= 0:
            raise ValidationError(f"JSON-RPC {field} must be > 0")

    def _validate_positive_integer(self, value: Any, field: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(f"JSON-RPC {field} must be an integer")
        if value <= 0:
            raise ValidationError(f"JSON-RPC {field} must be > 0")

    def _validate_json_value(self, value: Any, field: str) -> None:
        if value is not None and type(value) not in {dict, list}:
            raise ValidationError(
                f"JSON-RPC {field} must be an object, array, or null"
            )
        try:
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValidationError(f"JSON-RPC {field} must be JSON-serializable") from exc

    def _bounded_list_limit(self, limit: int | None) -> int:
        selected = self.config.jsonrpc.list_limit if limit is None else limit
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise ValidationError("JSON-RPC endpoint list limit must be an integer")
        value = selected
        if value < 1:
            raise ValidationError("JSON-RPC endpoint list limit must be >= 1")
        if value > self.config.jsonrpc.list_limit:
            raise ValidationError(
                f"JSON-RPC endpoint list limit exceeds configured maximum {self.config.jsonrpc.list_limit}"
            )
        return value

    def _load_endpoint(self, endpoint_id: str) -> tuple[JsonRpcEndpointSpec, dict[str, Any]]:
        self._validate_identifier(endpoint_id, "endpoint_id", self.config.jsonrpc.endpoint_id_max_chars)
        found = self.extensions.get_jsonrpc_endpoint(endpoint_id)
        if found is None:
            raise NotFound(f"JSON-RPC endpoint not found: {endpoint_id}")
        spec, metadata = found
        self._validate_endpoint(spec)
        return spec, metadata

    def _endpoint_to_json(
        self,
        endpoint: JsonRpcEndpointSpec,
        metadata: dict[str, Any],
        *,
        include_sensitive_fields: bool,
    ) -> dict[str, Any]:
        return {
            "schema_version": endpoint.schema_version,
            "endpoint_id": endpoint.endpoint_id,
            "url": endpoint.url if include_sensitive_fields else None,
            "headers": {
                name: {
                    "env": spec.env,
                    "prefix": spec.prefix if include_sensitive_fields else None,
                    "suffix": spec.suffix if include_sensitive_fields else None,
                    "prefix_configured": bool(spec.prefix),
                    "suffix_configured": bool(spec.suffix),
                    "value": "<redacted>",
                }
                for name, spec in endpoint.headers.items()
            },
            "methods": [
                {
                    "method_id": method.method_id,
                    "rpc_method": method.rpc_method,
                    "right": method.right,
                    "resource": self.method_resource(endpoint.endpoint_id, method.method_id),
                    "rollback_class": method.rollback_class,
                    "rollback_status": (
                        method.rollback_status
                        if method.rollback_status is not None
                        else self._default_rollback_status(method.rollback_class)
                    ),
                    "state_mutation": method.state_mutation,
                    "information_flow": method.information_flow,
                    "params_schema": method.params_schema,
                    "metadata": method.metadata,
                }
                for method in endpoint.methods
            ],
            "metadata": endpoint.metadata,
            "registered_by": metadata.get("registered_by"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
        }

    def _default_rollback_status(self, rollback_class: str) -> str:
        return default_external_effect_rollback_status(
            ExternalEffectRollbackClass(rollback_class)
        ).value

    def _profile_json(self, profile: Any) -> dict[str, Any]:
        return {
            "operation": profile.operation,
            "resource": profile.resource,
            "effect": profile.effect.value,
            "risk": profile.risk.value,
            "rule_id": profile.rule_id,
            "restrictions": dict(profile.restrictions),
        }

    def _body_observation(self, value: bytes) -> dict[str, Any]:
        return sanitize_for_observability(
            {"body": value.decode("utf-8", errors="replace")},
            preview_chars=self.config.jsonrpc.audit_preview_chars,
        )
