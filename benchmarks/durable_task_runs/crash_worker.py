from __future__ import annotations

import argparse
import json
import os
import signal
import socket
from dataclasses import replace
from typing import Any, NoReturn

from agent_libos import Runtime, TaskRunSpecV1
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.evidence.external_effects import prepare_external_effect_intent
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    JsonRpcEndpointSpec,
    JsonRpcMethodSpec,
    TaskRunRetention,
)
from agent_libos.substrate import LocalResourceProviderSubstrate

from benchmarks.durable_task_runs.crash_harness import (
    CRASH_EXIT_CODE,
    DurabilityBarrier,
    FsyncIdempotentJsonRpcProvider,
    FsyncProviderLedger,
    ProviderOutcome,
)


_IMAGE_ID = "durable-crash-agent:v1"
_ENDPOINT_ID = "durability-provider"
_METHOD_ID = "commit"


class _ScriptedActionClient:
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self._actions = list(actions)

    def complete_action(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        if not self._actions:
            raise AssertionError("crash worker exhausted its deterministic actions")
        action = self._actions.pop(0)
        name = str(action["action"])
        arguments = {key: value for key, value in action.items() if key != "action"}
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": f"crash_tool_call_{len(self._actions)}",
                    "name": name,
                    "arguments": json.dumps(
                        arguments,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            ],
            request_id=f"crash-request-{len(self._actions)}",
            response_id=f"crash-response-{len(self._actions)}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument(
        "--barrier",
        choices=[item.value for item in DurabilityBarrier],
        required=True,
    )
    parser.add_argument("--crash-before-local-result", action="store_true")
    args = parser.parse_args()
    barrier = DurabilityBarrier(args.barrier)
    if (
        args.crash_before_local_result
        and barrier is not DurabilityBarrier.PROVIDER_RESULT_DURABLE
    ):
        parser.error("--crash-before-local-result requires provider_result_durable")

    idempotency_key = f"durability-key-{barrier.value}"
    actions: list[dict[str, Any]] = []
    if args.crash_before_local_result:
        actions.append({"action": "get_current_time"})
    actions.append(
        {
            "action": "call_jsonrpc_method",
            "endpoint_id": _ENDPOINT_ID,
            "method_id": _METHOD_ID,
            "params": {
                "barrier": barrier.value,
                "idempotency_key": idempotency_key,
            },
        }
    )

    ledger = FsyncProviderLedger(args.ledger)
    provider = FsyncIdempotentJsonRpcProvider(
        ledger,
        barrier=(
            None
            if args.crash_before_local_result
            or barrier is DurabilityBarrier.EFFECT_PREPARED
            else barrier
        ),
        crash=_crash,
    )
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    workspace = os.path.join(
        os.path.dirname(args.database),
        f"{barrier.value}-workspace",
    )
    os.makedirs(workspace, exist_ok=True)
    substrate = LocalResourceProviderSubstrate(workspace)
    substrate.jsonrpc = provider
    runtime = Runtime.open(
        args.database,
        config=config,
        substrate=substrate,
    )
    runtime.llm.client = _ScriptedActionClient(actions)
    runtime.register_image(
        AgentImage(
            image_id=_IMAGE_ID,
            name="durable-crash-agent",
            version="v1",
            default_tools=["get_current_time", "call_jsonrpc_method"],
        ),
        actor="crash-harness",
    )
    summary = runtime.task_runs.create(
        TaskRunSpecV1(
            goal={"barrier": barrier.value},
            display_title=f"Crash barrier: {barrier.value}",
            image_id=_IMAGE_ID,
            retention=TaskRunRetention.PERMANENT,
        ),
        client_request_id=f"create-{barrier.value}",
    )
    if summary.root_pid is None:
        raise AssertionError("crash TaskRun has no root process")
    pid = summary.root_pid
    provider.attach(runtime, pid)
    _register_endpoint(runtime)
    runtime.capability.grant(
        pid,
        f"jsonrpc:{_ENDPOINT_ID}:{_METHOD_ID}",
        [CapabilityRight.WRITE],
        issued_by="crash-harness",
    )
    socket.getaddrinfo = _public_test_resolution  # type: ignore[assignment]

    if barrier is DurabilityBarrier.RUN_COMMITTED:
        _crash(False)

    if args.crash_before_local_result:
        _install_after_primitive_crash(runtime)
        max_quanta = 2
    else:
        _install_commit_barrier(
            runtime,
            barrier,
            ledger=ledger,
            idempotency_key=idempotency_key,
        )
        max_quanta = 1

    runtime.task_runs.run_until_blocked(
        summary.run_id,
        expected_revision=summary.revision,
        command_id=f"run-{barrier.value}",
        max_quanta=max_quanta,
    )
    raise AssertionError(f"barrier did not terminate the worker: {barrier.value}")


def _register_endpoint(runtime: Runtime) -> None:
    runtime.jsonrpc.register_endpoint(
        JsonRpcEndpointSpec(
            schema_version=1,
            endpoint_id=_ENDPOINT_ID,
            url="https://durability.example.test/jsonrpc",
            headers={},
            methods=[
                JsonRpcMethodSpec(
                    method_id=_METHOD_ID,
                    rpc_method="durability.commit",
                    right=CapabilityRight.WRITE.value,
                    rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE.value,
                    rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED.value,
                    state_mutation=True,
                    information_flow=True,
                    params_schema={
                        "type": "object",
                        "required": ["barrier", "idempotency_key"],
                        "additionalProperties": False,
                        "properties": {
                            "barrier": {"type": "string"},
                            "idempotency_key": {"type": "string", "minLength": 1},
                        },
                    },
                )
            ],
            timeout_s=5.0,
            max_request_bytes=16 * 1024,
            max_response_bytes=16 * 1024,
        ),
        actor="crash-harness",
        require_capability=False,
    )


def _install_commit_barrier(
    runtime: Runtime,
    barrier: DurabilityBarrier,
    *,
    ledger: FsyncProviderLedger,
    idempotency_key: str,
) -> None:
    if barrier is DurabilityBarrier.ACTION_COMMITTED:
        original = runtime.task_runs.record_validated_transcript

        def crash_after_validated(**kwargs: Any) -> Any:
            original(**kwargs)
            _crash(False)

        runtime.task_runs.record_validated_transcript = crash_after_validated  # type: ignore[method-assign]
        return
    if barrier is DurabilityBarrier.EFFECT_PREPARED:
        original = runtime.task_runs.expected_tool_id_for_pending_action

        def crash_after_effect_prepare(pid: str, action: Any) -> Any:
            expected_tool_id = original(pid, action)
            point = runtime.store.get_task_run_resume_point(pid, complete_only=True)
            if point is None or point.pending_action_payload_id is None:
                raise AssertionError("prepared barrier lost its claimed action")
            wrapper = runtime.task_runs._decode_pending_resume_payload(point)  # noqa: SLF001
            effect = prepare_external_effect_intent(
                runtime.uow.protected_effects,
                pid=pid,
                provider="jsonrpc",
                operation="call",
                target=f"jsonrpc:{_ENDPOINT_ID}:{_METHOD_ID}",
                state_mutation=True,
                information_flow=False,
                metadata={
                    "task_run_action": {
                        "run_id": point.run_id,
                        "pid": pid,
                        "call_id": wrapper.get("call_id"),
                        "context_generation": wrapper.get("context_generation"),
                        "action_manifest_sha256": wrapper.get("manifest_sha256"),
                        "source_safe_point_seq": point.safe_point_seq,
                    }
                },
                canonical_args={
                    "action": action,
                    "idempotency_key": idempotency_key,
                },
                idempotency_key=idempotency_key,
            )
            ledger.append(
                effect_id=effect.effect_id,
                kind="certification",
                outcome=ProviderOutcome.NOT_STARTED,
                idempotency_key=idempotency_key,
            )
            _crash(False)
            return expected_tool_id

        runtime.task_runs.expected_tool_id_for_pending_action = crash_after_effect_prepare  # type: ignore[method-assign]
        return
    if barrier is DurabilityBarrier.PROVIDER_RESULT_DURABLE:
        original = runtime.task_runs.stage_completed_transcript

        def crash_after_staged(**kwargs: Any) -> Any:
            original(**kwargs)
            _crash(False)

        runtime.task_runs.stage_completed_transcript = crash_after_staged  # type: ignore[method-assign]
        return
    if barrier is DurabilityBarrier.RESUME_POINT_COMMITTED:
        original = runtime.task_runs.record_completed_transcript

        def crash_after_resume_point(**kwargs: Any) -> Any:
            original(**kwargs)
            _crash(False)

        runtime.task_runs.record_completed_transcript = crash_after_resume_point  # type: ignore[method-assign]


def _install_after_primitive_crash(runtime: Runtime) -> None:
    original = runtime.jsonrpc.call

    def crash_after_primitive(*args: Any, **kwargs: Any) -> Any:
        original(*args, **kwargs)
        _crash(False)

    runtime.jsonrpc.call = crash_after_primitive  # type: ignore[method-assign]


def _public_test_resolution(*_args: Any, **_kwargs: Any) -> list[Any]:
    return [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("93.184.216.34", 443),
        )
    ]


def _crash(sigkill: bool) -> NoReturn:
    # No normal interpreter shutdown is allowed to flush Runtime-owned state.
    # The dispatched barrier uses SIGKILL so the matrix covers both an abrupt
    # signal and ``os._exit`` at independently durable commit boundaries.
    # Windows does not expose SIGKILL, so retain the same no-cleanup crash
    # property there with the portable os._exit fallback.
    kill_signal = getattr(signal, "SIGKILL", None)
    if sigkill and kill_signal is not None:
        os.kill(os.getpid(), kill_signal)
        signal.pause()  # pragma: no cover - SIGKILL cannot return control
    os._exit(CRASH_EXIT_CODE)


if __name__ == "__main__":
    main()
