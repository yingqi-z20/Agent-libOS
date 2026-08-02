from __future__ import annotations

import argparse
import hmac
import json
import os
import signal
import threading
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MethodType
from typing import Any
from urllib.parse import urlsplit

from agent_libos import Runtime
from agent_libos.api.gui.server import create_gui_http_server
from agent_libos.config import DEFAULT_CONFIG, LLMProfile
from agent_libos.evidence import (
    PayloadRetentionTier,
    llm_call_payload_sha256,
    llm_call_payload_retention_tier,
    retain_llm_call_payload,
)
from agent_libos.llm.provider_trace import provider_trace_summary
from agent_libos.llm.client import LLMClient
from agent_libos.models.llm import LLMCallRecord


PROVIDER_KEY_ENV = "AGENT_LIBOS_E2E_PROVIDER_KEY"
CONTROL_TOKEN_ENV = "AGENT_LIBOS_E2E_CONTROL_TOKEN"
LIVE_REASONING_START = "E2E_REASONING_START"
LIVE_REASONING_END = "E2E_REASONING_END"
LIVE_OUTPUT = "E2E_OUTPUT_COMPLETE"
MALICIOUS_TEXT = (
    "<script>window.__providerTracePwned=true</script> "
    "[危险链接](javascript:alert(1)) \u202egnp.exe"
)
SUMMARY_CALL_ID = "llmcall_e2e_summary"
HASH_CALL_ID = "llmcall_e2e_hash"
CONFLICT_CALL_ID = "llmcall_e2e_chunk_conflict"
LIMITED_CALL_ID = "llmcall_e2e_limited"
OLDEST_CALL_ID = "llmcall_e2e_filler_00"
FILLER_COUNT = 55
MAX_REQUEST_BYTES = 2 * 1024 * 1024


@dataclass
class FixtureState:
    provider_key: str
    control_token: str
    provider_requests: int = 0
    runtime: Runtime | None = None
    pid: str | None = None
    mutation_lock: Any = None

    def __post_init__(self) -> None:
        self.lock = threading.RLock()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    gui_token = _required_secret(parser, "AGENT_LIBOS_E2E_GUI_TOKEN")
    provider_key = _required_secret(parser, PROVIDER_KEY_ENV)
    control_token = _required_secret(parser, CONTROL_TOKEN_ENV)

    args.db.parent.mkdir(parents=True, exist_ok=True)
    state = FixtureState(provider_key=provider_key, control_token=control_token)
    provider_server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler_for(state),
    )
    provider_thread = threading.Thread(
        target=provider_server.serve_forever,
        name="e2e-loopback-provider",
        daemon=True,
    )
    provider_thread.start()

    runtime: Runtime | None = None
    gui_server: Any = None
    try:
        provider_host, provider_port = provider_server.server_address
        config = _fixture_config(
            f"http://{provider_host}:{provider_port}/v1"
        )
        runtime = Runtime.open(str(args.db), config=config)
        # Production auto mode intentionally chooses Chat immediately for custom
        # endpoints. This fixture forces the one initial Responses probe needed
        # to exercise the built-in Responses -> Chat compatibility branch while
        # retaining the truthful loopback endpoint and exact trusted client type.
        client = LLMClient(
            base_url=f"http://{provider_host}:{provider_port}/v1",
            model="e2e-provider-model",
            api_key=provider_key,
            api_key_env=PROVIDER_KEY_ENV,
            timeout=5.0,
            max_retries=1,
            api_mode="auto",
            fallback_json_actions=True,
            inherit_ambient_openai_sdk_config=False,
            allow_custom_base_url=True,
            defaults=config.llm,
        )
        client._use_responses_api = MethodType(lambda _client: True, client)  # type: ignore[method-assign]
        runtime.llm.client = client
        state.runtime = runtime
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal=(
                "Exercise the deterministic loopback Provider and finish by calling "
                "process_exit with a compact structured payload."
            ),
        )
        state.pid = pid
        result = runtime.run_process_once(pid)
        live_call = _validate_live_runtime_trace(runtime, pid, result, state)
        _seed_retention_and_pagination_records(runtime, pid)

        gui_server = create_gui_http_server(
            db=str(args.db),
            runtime=runtime,
            token=gui_token,
            auto_run=False,
            llm_profiles_file=args.db.parent / "profiles.json",
        )
        state.mutation_lock = gui_server.service.runtime_lock

        def request_shutdown(_signum: int, _frame: Any) -> None:
            def shutdown() -> None:
                if gui_server is not None:
                    gui_server.shutdown()
                provider_server.shutdown()

            threading.Thread(target=shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, request_shutdown)
        signal.signal(signal.SIGINT, request_shutdown)
        host, port = gui_server.server_address
        print(
            json.dumps(
                {
                    "url": f"http://{host}:{port}",
                    "db": str(args.db),
                    "pid": pid,
                    "control_url": (
                        f"http://{provider_host}:{provider_port}"
                        "/__e2e__/retention"
                    ),
                    "live_call_id": live_call.call_id,
                    "summary_call_id": SUMMARY_CALL_ID,
                    "hash_call_id": HASH_CALL_ID,
                    "conflict_call_id": CONFLICT_CALL_ID,
                    "limited_call_id": LIMITED_CALL_ID,
                    "oldest_call_id": OLDEST_CALL_ID,
                    "expected_call_count": 1 + FILLER_COUNT + 4,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        gui_server.serve_forever()
    finally:
        if gui_server is not None:
            try:
                gui_server.service.shutdown()
            finally:
                gui_server.server_close()
        provider_server.shutdown()
        provider_server.server_close()
        provider_thread.join(timeout=2)
        if runtime is not None:
            runtime.close()


def _required_secret(parser: argparse.ArgumentParser, name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        parser.error(f"{name} is required")
    return value


def _fixture_config(provider_url: str) -> Any:
    llm = replace(
        DEFAULT_CONFIG.llm,
        default_profile_id="e2e",
        profiles={
            "e2e": LLMProfile(
                base_url=provider_url,
                model="e2e-provider-model",
                api_key_env=PROVIDER_KEY_ENV,
                api_mode="auto",
                timeout_s=5.0,
                max_retries=1,
                fallback_json_actions=True,
                temperature=0.0,
                max_tokens=512,
                allow_custom_base_url=True,
            )
        },
        fallback_json_actions=True,
    )
    gui = replace(DEFAULT_CONFIG.gui, snapshot_llm_call_limit=1)
    return replace(DEFAULT_CONFIG, gui=gui, llm=llm)


def _handler_for(state: FixtureState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlsplit(self.path).path
            if path == "/v1/responses":
                self._responses_compatibility_rejection()
                return
            if path == "/v1/chat/completions":
                self._provider_completion()
                return
            if path == "/__e2e__/retention":
                self._retention_control()
                return
            self._send_json(404, {"error": "not_found"})

        def _responses_compatibility_rejection(self) -> None:
            authorization = self.headers.get("Authorization", "")
            if not hmac.compare_digest(authorization, f"Bearer {state.provider_key}"):
                self._discard_body()
                self._send_json(401, {"error": {"type": "unauthorized"}})
                return
            payload = self._read_json()
            if not isinstance(payload, dict) or payload.get("model") != "e2e-provider-model":
                self._send_json(400, {"error": {"type": "invalid_request"}})
                return
            with state.lock:
                state.provider_requests += 1
                request_number = state.provider_requests
            if request_number != 1:
                self._send_json(409, {"error": {"type": "unexpected_request_order"}})
                return
            self._send_json(
                404,
                {
                    "error": {
                        "message": "Responses endpoint not found",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "endpoint_not_found",
                    }
                },
                extra_headers={"x-request-id": "req-e2e-responses"},
            )

        def _provider_completion(self) -> None:
            authorization = self.headers.get("Authorization", "")
            if not hmac.compare_digest(
                authorization,
                f"Bearer {state.provider_key}",
            ):
                self._discard_body()
                self._send_json(401, {"error": {"type": "unauthorized"}})
                return
            payload = self._read_json()
            if not isinstance(payload, dict):
                self._send_json(400, {"error": {"type": "invalid_request"}})
                return
            if (
                payload.get("model") != "e2e-provider-model"
                or not isinstance(payload.get("messages"), list)
                or not isinstance(payload.get("tools"), list)
            ):
                self._send_json(400, {"error": {"type": "invalid_request"}})
                return
            with state.lock:
                state.provider_requests += 1
                request_number = state.provider_requests
            if request_number == 2:
                self._send_json(
                    429,
                    {
                        "error": {
                            "message": "deterministic fixture rate limit",
                            "type": "rate_limit_error",
                            "param": None,
                            "code": "rate_limit_exceeded",
                        }
                    },
                    extra_headers={"Retry-After": "0", "x-request-id": "req-e2e-1"},
                )
                return
            reasoning = (
                f"{LIVE_REASONING_START}\n"
                + ("推理块🧪安全边界\n" * 8_000)
                + f"{MALICIOUS_TEXT}\n{LIVE_REASONING_END}"
            )
            self._send_json(
                200,
                {
                    "id": "chatcmpl-e2e-runtime",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "e2e-provider-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": LIVE_OUTPUT,
                                "reasoning_content": reasoning,
                                "tool_calls": [
                                    {
                                        "id": "call-e2e-process-exit",
                                        "type": "function",
                                        "function": {
                                            "name": "process_exit",
                                            "arguments": json.dumps(
                                                {
                                                    "payload": {
                                                        "summary": "fixture complete",
                                                        "verified": True,
                                                    }
                                                },
                                                separators=(",", ":"),
                                            ),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 24,
                        "completion_tokens": 12,
                        "total_tokens": 36,
                    },
                },
                extra_headers={"x-request-id": "req-e2e-2"},
            )

        def _retention_control(self) -> None:
            supplied = self.headers.get("X-E2E-Control-Token", "")
            if not hmac.compare_digest(supplied, state.control_token):
                self._discard_body()
                self._send_json(403, {"error": "forbidden"})
                return
            payload = self._read_json()
            if payload != {"call_id": CONFLICT_CALL_ID, "target": "summary"}:
                self._send_json(400, {"error": "invalid_control_request"})
                return
            runtime = state.runtime
            if runtime is None or state.pid is None:
                self._send_json(503, {"error": "fixture_not_ready"})
                return
            selected_lock = state.mutation_lock or state.lock
            with selected_lock:
                current = runtime.store.get_llm_call(CONFLICT_CALL_ID)
                if current is None or current.pid != state.pid:
                    self._send_json(404, {"error": "call_not_found"})
                    return
                current_tier = llm_call_payload_retention_tier(current)
                if current_tier is PayloadRetentionTier.SUMMARY:
                    self._send_json(200, {"ok": True, "changed": False})
                    return
                if current_tier is not PayloadRetentionTier.FULL:
                    self._send_json(409, {"error": "unexpected_retention_tier"})
                    return
                retained = retain_llm_call_payload(
                    current,
                    PayloadRetentionTier.SUMMARY,
                    provider_chain_head=False,
                )
                changed = runtime.store.update_llm_call_payload_retention(
                    retained,
                    expected_payload_sha256=llm_call_payload_sha256(current),
                    expected_tier=PayloadRetentionTier.FULL,
                )
            self._send_json(200 if changed else 409, {"ok": changed, "changed": changed})

        def _read_json(self) -> Any:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return None
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._discard_body(max_bytes=MAX_REQUEST_BYTES)
                return None
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None

        def _discard_body(self, *, max_bytes: int = MAX_REQUEST_BYTES) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return
            if 0 < length <= max_bytes:
                self.rfile.read(length)

        def _send_json(
            self,
            status: int,
            value: Any,
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def _validate_live_runtime_trace(
    runtime: Runtime,
    pid: str,
    result: Any,
    state: FixtureState,
) -> LLMCallRecord:
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError("deterministic Runtime LLM action did not complete")
    if result.get("action", {}).get("action") != "process_exit":
        raise RuntimeError("deterministic Runtime did not execute process_exit")
    calls = runtime.store.list_llm_calls(pid)
    if len(calls) != 1 or state.provider_requests != 3:
        raise RuntimeError("deterministic Provider retry count was not preserved")
    call = calls[0]
    trace = call.reasoning if isinstance(call.reasoning, dict) else {}
    attempts = trace.get("attempts") if isinstance(trace.get("attempts"), list) else []
    if (
        call.status != "ok"
        or call.api != "chat"
        or call.model != "e2e-provider-model"
        or trace.get("coverage") != "complete"
        or trace.get("selected_attempt") != 3
        or len(attempts) != 3
        or attempts[0].get("kind") != "initial"
        or attempts[0].get("api") != "responses"
        or attempts[0].get("status") != "error"
        or attempts[1].get("kind") != "responses_to_chat"
        or attempts[1].get("status") != "error"
        or attempts[2].get("kind") != "transport_retry"
        or attempts[2].get("status") != "ok"
        or LIVE_REASONING_START not in json.dumps(attempts[2], ensure_ascii=False)
        or LIVE_REASONING_END not in json.dumps(attempts[2], ensure_ascii=False)
        or attempts[2].get("output") != LIVE_OUTPUT
        or [item.get("name") for item in attempts[2].get("tool_calls", [])]
        != ["process_exit"]
        or call.request_options.get("fallback_json_actions_enabled") is not True
        or call.request_options.get("fallback_json_action_used") is not False
    ):
        raise RuntimeError("deterministic Provider trace invariant failed")
    completed_actions = [
        item
        for item in runtime.audit.trace(actor=pid)
        if item.action == "llm.action" and item.target == "process_exit"
    ]
    if len(completed_actions) != 1:
        raise RuntimeError("Runtime action execution evidence is missing")
    return call


def _seed_retention_and_pagination_records(runtime: Runtime, pid: str) -> None:
    for index in range(FILLER_COUNT):
        runtime.store.insert_llm_call(
            _synthetic_trace_record(
                pid,
                f"llmcall_e2e_filler_{index:02d}",
                created_at=f"2019-01-01T00:00:{index:02d}+00:00",
                response_content=f"filler-{index:02d}",
            )
        )

    summary = _synthetic_trace_record(
        pid,
        SUMMARY_CALL_ID,
        created_at="2020-01-01T00:00:01+00:00",
        response_content="summary-retention-source",
    )
    runtime.store.insert_llm_call(summary)
    _reduce_record(runtime, summary, PayloadRetentionTier.SUMMARY)

    hash_only = _synthetic_trace_record(
        pid,
        HASH_CALL_ID,
        created_at="2020-01-01T00:00:02+00:00",
        response_content="hash-retention-source",
    )
    runtime.store.insert_llm_call(hash_only)
    retained_summary = _reduce_record(
        runtime,
        hash_only,
        PayloadRetentionTier.SUMMARY,
    )
    _reduce_record(runtime, retained_summary, PayloadRetentionTier.HASH_ONLY)

    conflict_content = (
        "E2E_CONFLICT_START\n"
        + ("冲突分块🧩\n" * 6_000)
        + "E2E_CONFLICT_END"
    )
    runtime.store.insert_llm_call(
        _synthetic_trace_record(
            pid,
            CONFLICT_CALL_ID,
            created_at="2020-01-01T00:00:03+00:00",
            response_content=conflict_content,
        )
    )
    runtime.store.insert_llm_call(_limited_trace_record(pid))


def _limited_trace_record(pid: str) -> LLMCallRecord:
    created_at = "2020-01-01T00:00:04+00:00"
    record = _synthetic_trace_record(
        pid,
        LIMITED_CALL_ID,
        created_at=created_at,
        response_content="limited-output",
    )
    trace = dict(record.reasoning)
    trace["limited"] = True
    trace["omitted_attempts"] = 2
    attempts = [dict(trace["attempts"][0])]
    attempts[0]["reasoning"] = {
        "availability": "limited",
        "blocks": [
            {
                "type": "reasoning_text",
                "text": "E2E_LIMITED_VISIBLE",
                "source": "reasoning_content",
            },
            {
                "type": "omitted",
                "reason": "bounds",
                "chars": 999_999,
                "bytes": 1_999_998,
                "sha256": "a" * 64,
            },
        ],
    }
    trace["attempts"] = attempts
    record.reasoning = trace
    record.request_options["provider_trace_summary"] = provider_trace_summary(trace)
    return record


def _reduce_record(
    runtime: Runtime,
    record: LLMCallRecord,
    target: PayloadRetentionTier,
) -> LLMCallRecord:
    source_tier = llm_call_payload_retention_tier(record)
    retained = retain_llm_call_payload(
        record,
        target,
        provider_chain_head=False,
    )
    changed = runtime.store.update_llm_call_payload_retention(
        retained,
        expected_payload_sha256=llm_call_payload_sha256(record),
        expected_tier=source_tier,
    )
    if not changed:
        raise RuntimeError("fixture retention transition failed")
    return retained


def _synthetic_trace_record(
    pid: str,
    call_id: str,
    *,
    created_at: str,
    response_content: str,
) -> LLMCallRecord:
    trace = {
        "kind": "provider_trace",
        "schema_version": 1,
        "coverage": "complete",
        "selected_attempt": 1,
        "limited": False,
        "omitted_attempts": 0,
        "attempts": [
            {
                "sequence": 1,
                "kind": "initial",
                "api": "chat",
                "status": "ok",
                "reasoning": {
                    "availability": "returned",
                    "blocks": [
                        {
                            "type": "reasoning_text",
                            "text": f"synthetic reasoning for {call_id}",
                            "source": "reasoning_content",
                        }
                    ],
                },
                "output": response_content,
                "tool_calls": [
                    {
                        "name": "process_exit",
                        "arguments": {"payload": {"fixture": call_id}},
                    }
                ],
                "usage": {"total_tokens": 1},
                "model": "synthetic-fixture-model",
                "request_id": None,
                "response_id": f"response-{call_id}",
                "started_at": created_at,
                "completed_at": created_at,
                "duration_ms": 0,
                "error": None,
            }
        ],
    }
    return LLMCallRecord(
        call_id=call_id,
        pid=pid,
        image_id="base-agent:v0",
        purpose="fixture_pagination",
        status="ok",
        api="chat",
        model="synthetic-fixture-model",
        request_options={
            "provider_trace_summary": provider_trace_summary(trace),
            "fallback_json_actions_enabled": True,
            "fallback_json_action_used": False,
        },
        response_content=response_content,
        # Synthetic rows never carry executable recovery semantics. The real
        # Runtime-generated row above is the fixture's tool-action evidence.
        tool_calls=[],
        reasoning=trace,
        usage={"total_tokens": 1},
        raw_response={"id": f"response-{call_id}"},
        created_at=created_at,
        completed_at=created_at,
    )


if __name__ == "__main__":
    main()
