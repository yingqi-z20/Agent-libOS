from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Protocol, TYPE_CHECKING
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, Field

from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ObjectMetadata,
    ObjectRight,
    ObjectType,
    Provenance,
    ResourceUsage,
    ViewMode,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.runtime.external_effects import classify_external_effect, record_external_effect
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy
from agent_libos.utils.ids import new_id, utc_now

if TYPE_CHECKING:
    from agent_libos.runtime.runtime import Runtime


_VIVADO_ADAPTER_ATTR = "_agent_libos_vivado_adapter"
_VIVADO_ENV_PREFIX = "AGENT_LIBOS_VIVADO_"
_PROJECT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_SYNC_RESERVED_DIR = ".vivado-server-sync"
_LOCAL_HTTP_HOSTS = {"localhost", "127.0.0.1", "::1"}
_TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "n", "off"}


class VivadoProvider(Protocol):
    def health(self, *, timeout_s: float) -> dict[str, Any]: ...

    def create_push_plan(
        self,
        project: str,
        entries: list[dict[str, Any]],
        *,
        delete_extra: bool,
        include_globs: list[str],
        exclude_globs: list[str],
        timeout_s: float,
    ) -> dict[str, Any]: ...

    def upload_push_file(
        self,
        project: str,
        sync_id: str,
        sync_path: str,
        source_path: Path,
        *,
        size_bytes: int,
        sha256: str,
        timeout_s: float,
    ) -> dict[str, Any]: ...

    def commit_push(self, project: str, sync_id: str, *, force: bool, timeout_s: float) -> dict[str, Any]: ...

    def abort_push(self, project: str, sync_id: str, *, timeout_s: float) -> dict[str, Any]: ...

    def create_pull_plan(
        self,
        project: str,
        entries: list[dict[str, Any]],
        *,
        delete_extra: bool,
        include_globs: list[str],
        exclude_globs: list[str],
        timeout_s: float,
    ) -> dict[str, Any]: ...

    def download_file(self, project: str, sync_path: str, target_path: Path, *, timeout_s: float) -> dict[str, Any]: ...

    def create_session(self, project: str, args: list[str], *, timeout_s: float) -> dict[str, Any]: ...

    def send_stdin(self, session_id: str, text: str, *, timeout_s: float) -> dict[str, Any]: ...

    def read_output(self, session_id: str, *, cursor: int, timeout_ms: int, timeout_s: float) -> dict[str, Any]: ...

    def heartbeat(self, session_id: str, *, timeout_s: float) -> dict[str, Any]: ...

    def get_session(self, session_id: str, *, timeout_s: float) -> dict[str, Any]: ...

    def delete_session(self, session_id: str, *, timeout_s: float) -> dict[str, Any]: ...

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification: ...


@dataclass(frozen=True)
class VivadoModuleSettings:
    base_url: str = "http://127.0.0.1:8080"
    token_env: str = "VIVADO_SERVER_TOKEN"
    request_timeout_s: float = 30.0
    output_timeout_ms: int = 30_000
    max_sessions_global: int = 16
    max_sessions_per_process: int = 4
    max_manifest_entries: int = 200_000
    max_file_bytes: int = 1_073_741_824
    file_chunk_bytes: int = 1_048_576
    max_output_chars: int = 32_000
    output_hard_limit_chars: int = 200_000
    input_max_chars: int = 32_768
    input_hard_limit_chars: int = 131_072
    max_session_args: int = 32
    max_arg_chars: int = 4096
    session_name_prefix: str = "vivado_session"
    allow_plain_http_non_loopback: bool = False
    default_exclude_globs: tuple[str, ...] = (
        ".git/**",
        ".vivado-server-sync/**",
        "*.jou",
        "*.log",
        "*.str",
        ".Xil/**",
        "*.cache/**",
        "*.runs/**",
        "*.sim/**",
    )


@dataclass(frozen=True)
class VivadoModuleConfig:
    vivado: VivadoModuleSettings = field(default_factory=VivadoModuleSettings)


@dataclass(frozen=True)
class LocalManifest:
    entries: list[dict[str, Any]]
    root_relative: str
    root_path: Path
    hashed_file_bytes: int


@dataclass(frozen=True)
class VivadoHealthResult:
    ok: bool
    response: dict[str, Any]


@dataclass(frozen=True)
class VivadoPushResult:
    project: str
    sync_id: str
    status: str
    uploaded_files: list[str]
    created_dirs: list[str]
    deleted_files: list[str]
    deleted_dirs: list[str]
    upload_count: int


@dataclass(frozen=True)
class VivadoPullResult:
    project: str
    downloaded_files: list[str]
    created_dirs: list[str]
    deleted_files: list[str]
    deleted_dirs: list[str]
    download_count: int


@dataclass(frozen=True)
class VivadoSessionCreateResult:
    session_oid: str
    namespace: str
    name: str
    type: str
    session_id: str
    project: str
    status: str
    cursor: int


@dataclass(frozen=True)
class VivadoSessionWriteResult:
    session_oid: str
    session_id: str
    chars_written: int
    status: str


@dataclass(frozen=True)
class VivadoSessionOutputResult:
    session_oid: str
    session_id: str
    cursor: int
    status: str
    output: str
    output_truncated: bool
    chunks: list[dict[str, Any]]
    overrun: bool
    heartbeat_sent: bool


@dataclass(frozen=True)
class VivadoSessionStatusResult:
    session_oid: str
    session_id: str
    project: str
    status: str
    started_at: str | None
    last_heartbeat_at: str | None
    exit_code: int | None


@dataclass(frozen=True)
class VivadoSessionHeartbeatResult:
    session_oid: str
    session_id: str
    status: str


@dataclass(frozen=True)
class VivadoSessionCloseResult:
    session_oid: str
    session_id: str
    closed: bool
    status: str


@dataclass
class _VivadoRuntimeSession:
    session_oid: str
    session_id: str
    owner_pid: str
    project: str
    args: list[str]
    status: str
    cursor: int
    started_at: str | None
    last_heartbeat_at: str | None
    exit_code: int | None
    created_at: str
    lock: threading.RLock = field(default_factory=threading.RLock)


def _coerce_vivado_settings(value: Any) -> VivadoModuleSettings:
    settings_data: dict[str, Any]
    if value is None:
        settings_data = {}
    elif isinstance(value, VivadoModuleSettings):
        settings_data = asdict(value)
    elif isinstance(value, dict):
        settings_data = dict(value)
    else:
        settings_data = {
            item.name: getattr(value, item.name)
            for item in fields(VivadoModuleSettings)
            if hasattr(value, item.name)
        }
    settings_data.update(_vivado_settings_env_overrides(os.environ))
    settings = VivadoModuleSettings(**settings_data)
    _validate_vivado_settings(settings)
    return settings


def _vivado_settings_env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    defaults = VivadoModuleSettings()
    overrides: dict[str, Any] = {}
    for item in fields(VivadoModuleSettings):
        env_name = f"{_VIVADO_ENV_PREFIX}{item.name.upper()}"
        if env_name not in env:
            continue
        overrides[item.name] = _parse_vivado_env_value(env_name, env[env_name], getattr(defaults, item.name))
    return overrides


def _parse_vivado_env_value(env_name: str, raw_value: str, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return _parse_vivado_bool_env(env_name, raw_value, default=default_value)
    if isinstance(default_value, int):
        return _parse_vivado_int_env(env_name, raw_value, default=default_value)
    if isinstance(default_value, float):
        return _parse_vivado_float_env(env_name, raw_value, default=default_value)
    if isinstance(default_value, str):
        return raw_value.strip() or default_value
    if isinstance(default_value, tuple):
        return _parse_vivado_tuple_env(env_name, raw_value, default=default_value)
    raise ValidationError(f"unsupported Vivado environment setting: {env_name}")


def _parse_vivado_bool_env(env_name: str, raw_value: str, *, default: bool) -> bool:
    normalized = raw_value.strip().lower()
    if not normalized:
        return default
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    raise ValidationError(f"{env_name} must be a boolean value")


def _parse_vivado_int_env(env_name: str, raw_value: str, *, default: int) -> int:
    selected = raw_value.strip()
    if not selected:
        return default
    try:
        return int(selected)
    except ValueError as exc:
        raise ValidationError(f"{env_name} must be an integer") from exc


def _parse_vivado_float_env(env_name: str, raw_value: str, *, default: float) -> float:
    selected = raw_value.strip()
    if not selected:
        return default
    try:
        return float(selected)
    except ValueError as exc:
        raise ValidationError(f"{env_name} must be numeric") from exc


def _parse_vivado_tuple_env(env_name: str, raw_value: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    selected = raw_value.strip()
    if not selected:
        return default
    if selected.startswith("["):
        try:
            parsed = json.loads(selected)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{env_name} must be a JSON string list or {os.pathsep!r}-separated string") from exc
        if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
            raise ValidationError(f"{env_name} must be a JSON string list")
        return tuple(item.strip() for item in parsed if item.strip())
    parts = [item.strip() for item in selected.split(os.pathsep)]
    if len(parts) == 1 and "," in selected:
        parts = [item.strip() for item in selected.split(",")]
    return tuple(item for item in parts if item)


def _validate_vivado_settings(settings: VivadoModuleSettings) -> None:
    positive_ints = (
        "output_timeout_ms",
        "max_sessions_global",
        "max_sessions_per_process",
        "max_manifest_entries",
        "max_file_bytes",
        "file_chunk_bytes",
        "max_output_chars",
        "output_hard_limit_chars",
        "input_max_chars",
        "input_hard_limit_chars",
        "max_session_args",
        "max_arg_chars",
    )
    for name in positive_ints:
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValidationError(f"vivado module setting {name} must be a positive integer")
    if settings.max_sessions_global < settings.max_sessions_per_process:
        raise ValidationError("vivado module max_sessions_global must be >= max_sessions_per_process")
    if settings.output_hard_limit_chars < settings.max_output_chars:
        raise ValidationError("vivado module output_hard_limit_chars must cover max_output_chars")
    if settings.input_hard_limit_chars < settings.input_max_chars:
        raise ValidationError("vivado module input_hard_limit_chars must cover input_max_chars")
    if isinstance(settings.request_timeout_s, bool) or not isinstance(settings.request_timeout_s, (int, float)):
        raise ValidationError("vivado module request_timeout_s must be numeric")
    if not math.isfinite(float(settings.request_timeout_s)) or settings.request_timeout_s <= 0:
        raise ValidationError("vivado module request_timeout_s must be > 0")
    if not settings.token_env.strip():
        raise ValidationError("vivado module token_env must be non-empty")
    if not settings.session_name_prefix.strip():
        raise ValidationError("vivado module session_name_prefix must be non-empty")
    _validate_base_url(settings.base_url, allow_plain_http_non_loopback=settings.allow_plain_http_non_loopback)


def initialize_vivado(runtime: "Runtime") -> None:
    if getattr(runtime, _VIVADO_ADAPTER_ATTR, None) is not None:
        return
    settings = _coerce_vivado_settings(getattr(runtime.substrate, "vivado_settings", None))
    provider = getattr(runtime.substrate, "vivado", None) or HttpVivadoProvider(settings)
    adapter = VivadoAdapter(runtime, provider=provider, config=VivadoModuleConfig(settings))
    adapter.release_stale_session_objects()
    setattr(runtime, _VIVADO_ADAPTER_ATTR, adapter)
    runtime.memory.bind_object_release_finalizer(_object_release_finalizer(adapter))
    bind_shutdown = getattr(runtime, "bind_shutdown_finalizer", None)
    if callable(bind_shutdown):
        bind_shutdown(adapter.shutdown)


def _object_release_finalizer(adapter: "VivadoAdapter"):
    def finalize(obj: Any, actor: str, reason: str) -> None:
        if getattr(obj, "type", None) == ObjectType.EXTERNAL_REF and isinstance(getattr(obj, "payload", None), dict):
            if obj.payload.get("kind") == "vivado_session":
                adapter.close_for_object_release(obj.oid, actor=actor, reason=reason)

    return finalize


class HttpVivadoProvider:
    """HTTP client for VivadoServer v1."""

    def __init__(self, settings: VivadoModuleSettings | None = None):
        self.settings = settings or VivadoModuleSettings()
        _validate_base_url(
            self.settings.base_url,
            allow_plain_http_non_loopback=self.settings.allow_plain_http_non_loopback,
        )

    def health(self, *, timeout_s: float) -> dict[str, Any]:
        return self._request_json("GET", "/healthz", auth=False, timeout_s=timeout_s)

    def create_push_plan(
        self,
        project: str,
        entries: list[dict[str, Any]],
        *,
        delete_extra: bool,
        include_globs: list[str],
        exclude_globs: list[str],
        timeout_s: float,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"/v1/projects/{_quote_segment(project)}/sync/push/plan",
            {
                "entries": entries,
                "delete_extra": delete_extra,
                "include_globs": include_globs,
                "exclude_globs": exclude_globs,
            },
            timeout_s=timeout_s,
        )

    def upload_push_file(
        self,
        project: str,
        sync_id: str,
        sync_path: str,
        source_path: Path,
        *,
        size_bytes: int,
        sha256: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        path = f"/v1/projects/{_quote_segment(project)}/sync/{_quote_segment(sync_id)}/files/{_quote_sync_path(sync_path)}"
        url = self._url(path)
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValidationError("VivadoServer URL must be HTTP(S)")
        conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        host = parsed.hostname or ""
        port = parsed.port
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        headers = {
            "Authorization": f"Bearer {self._token()}",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size_bytes),
        }
        conn = conn_cls(host, port=port, timeout=timeout_s)
        try:
            with source_path.open("rb") as handle:
                conn.request("PUT", target, body=handle, headers=headers)
                response = conn.getresponse()
                body = response.read()
        except OSError as exc:
            raise ValidationError(f"VivadoServer upload failed: {type(exc).__name__}: {exc}") from exc
        finally:
            conn.close()
        self._raise_for_status(response.status, body)
        data = self._decode_json(body)
        if data.get("sha256", "").lower() not in {"", sha256.lower()}:
            raise ValidationError("VivadoServer upload response sha256 mismatch")
        return data

    def commit_push(self, project: str, sync_id: str, *, force: bool, timeout_s: float) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"/v1/projects/{_quote_segment(project)}/sync/{_quote_segment(sync_id)}/commit",
            {"force": force},
            timeout_s=timeout_s,
        )

    def abort_push(self, project: str, sync_id: str, *, timeout_s: float) -> dict[str, Any]:
        return self._request_json(
            "DELETE",
            f"/v1/projects/{_quote_segment(project)}/sync/{_quote_segment(sync_id)}",
            timeout_s=timeout_s,
        )

    def create_pull_plan(
        self,
        project: str,
        entries: list[dict[str, Any]],
        *,
        delete_extra: bool,
        include_globs: list[str],
        exclude_globs: list[str],
        timeout_s: float,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"/v1/projects/{_quote_segment(project)}/sync/pull/plan",
            {
                "entries": entries,
                "delete_extra": delete_extra,
                "include_globs": include_globs,
                "exclude_globs": exclude_globs,
            },
            timeout_s=timeout_s,
        )

    def download_file(self, project: str, sync_path: str, target_path: Path, *, timeout_s: float) -> dict[str, Any]:
        request = urlrequest.Request(
            self._url(f"/v1/projects/{_quote_segment(project)}/sync/files/{_quote_sync_path(sync_path)}"),
            headers={"Authorization": f"Bearer {self._token()}"},
            method="GET",
        )
        sha = hashlib.sha256()
        bytes_written = 0
        try:
            with urlrequest.urlopen(request, timeout=timeout_s) as response:
                expected_size = _optional_int(response.headers.get("x-sync-size-bytes"))
                expected_mtime = _optional_int(response.headers.get("x-sync-mtime-unix-ms"))
                expected_sha = str(response.headers.get("x-sync-sha256") or "").lower()
                if expected_size is not None and expected_size > self.settings.max_file_bytes:
                    raise ValidationError(
                        f"Vivado downloaded file exceeds max_file_bytes={self.settings.max_file_bytes}: {sync_path}"
                    )
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with target_path.open("wb") as handle:
                    while True:
                        chunk = response.read(self.settings.file_chunk_bytes)
                        if not chunk:
                            break
                        if bytes_written + len(chunk) > self.settings.max_file_bytes:
                            raise ValidationError(
                                f"Vivado downloaded file exceeds max_file_bytes={self.settings.max_file_bytes}: {sync_path}"
                            )
                        sha.update(chunk)
                        bytes_written += len(chunk)
                        handle.write(chunk)
        except urlerror.HTTPError as exc:
            body = exc.read()
            self._raise_for_status(exc.code, body)
            raise
        except (OSError, urlerror.URLError) as exc:
            raise ValidationError(f"VivadoServer download failed: {type(exc).__name__}: {exc}") from exc
        actual_sha = sha.hexdigest()
        if not expected_sha:
            raise ValidationError("VivadoServer download response missing x-sync-sha256")
        return {
            "path": sync_path,
            "size_bytes": expected_size if expected_size is not None else bytes_written,
            "mtime_unix_ms": expected_mtime,
            "sha256": expected_sha,
            "bytes_written": bytes_written,
            "actual_sha256": actual_sha,
        }

    def create_session(self, project: str, args: list[str], *, timeout_s: float) -> dict[str, Any]:
        return self._request_json("POST", "/v1/sessions", {"project": project, "args": args}, timeout_s=timeout_s)

    def send_stdin(self, session_id: str, text: str, *, timeout_s: float) -> dict[str, Any]:
        result = self._request_json(
            "POST",
            f"/v1/sessions/{_quote_segment(session_id)}/stdin",
            {"text": text},
            timeout_s=timeout_s,
            empty_ok=True,
        )
        return result or {"session_id": session_id, "status": "sent"}

    def read_output(self, session_id: str, *, cursor: int, timeout_ms: int, timeout_s: float) -> dict[str, Any]:
        return self._request_json(
            "GET",
            f"/v1/sessions/{_quote_segment(session_id)}/output?cursor={int(cursor)}&timeout_ms={int(timeout_ms)}",
            timeout_s=timeout_s,
        )

    def heartbeat(self, session_id: str, *, timeout_s: float) -> dict[str, Any]:
        result = self._request_json(
            "POST",
            f"/v1/sessions/{_quote_segment(session_id)}/heartbeat",
            timeout_s=timeout_s,
            empty_ok=True,
        )
        return result or {"session_id": session_id, "status": "ok"}

    def get_session(self, session_id: str, *, timeout_s: float) -> dict[str, Any]:
        return self._request_json("GET", f"/v1/sessions/{_quote_segment(session_id)}", timeout_s=timeout_s)

    def delete_session(self, session_id: str, *, timeout_s: float) -> dict[str, Any]:
        result = self._request_json(
            "DELETE",
            f"/v1/sessions/{_quote_segment(session_id)}",
            timeout_s=timeout_s,
            empty_ok=True,
        )
        return result or {"session_id": session_id, "status": "terminated"}

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        read_ops = {"health", "pull_plan", "download_file", "read_output", "session_status"}
        if operation in read_ops:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={"project": context.get("project"), "session_id": context.get("session_id")},
            )
        if operation in {
            "push_plan",
            "upload_file",
            "commit_push",
            "abort_push",
            "session_create",
            "send_stdin",
            "heartbeat",
            "delete_session",
        }:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                state_mutation=True,
                information_flow=True,
                metadata={"project": context.get("project"), "session_id": context.get("session_id")},
            )
        raise ValueError(f"unsupported vivado external effect operation: {operation}")

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        auth: bool = True,
        timeout_s: float,
        empty_ok: bool = False,
    ) -> dict[str, Any]:
        data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if auth:
            headers["Authorization"] = f"Bearer {self._token()}"
        request = urlrequest.Request(self._url(path), data=data, headers=headers, method=method)
        try:
            with urlrequest.urlopen(request, timeout=timeout_s) as response:
                raw = response.read()
        except urlerror.HTTPError as exc:
            raw = exc.read()
            self._raise_for_status(exc.code, raw)
            raise
        except (OSError, urlerror.URLError) as exc:
            raise ValidationError(f"VivadoServer request failed: {type(exc).__name__}: {exc}") from exc
        if empty_ok and not raw:
            return {}
        return self._decode_json(raw)

    def _url(self, path: str) -> str:
        base = self.settings.base_url.rstrip("/")
        return f"{base}{path}"

    def _token(self) -> str:
        token = os.environ.get(self.settings.token_env)
        if token is None or not token:
            raise ValidationError(f"missing VivadoServer token environment variable: {self.settings.token_env}")
        if "\r" in token or "\n" in token:
            raise ValidationError(f"invalid VivadoServer token environment variable: {self.settings.token_env}")
        return token

    def _decode_json(self, raw: bytes) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValidationError("VivadoServer returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValidationError("VivadoServer JSON response must be an object")
        return data

    def _raise_for_status(self, status: int, raw: bytes) -> None:
        if 200 <= status < 300:
            return
        message = raw.decode("utf-8", errors="replace")
        with contextlib.suppress(Exception):
            decoded = json.loads(message)
            if isinstance(decoded, dict) and decoded.get("error"):
                message = str(decoded["error"])
        raise ValidationError(f"VivadoServer returned HTTP {status}: {message[:500]}")


class VivadoAdapter:
    """Object-bound VivadoServer client primitive."""

    def __init__(
        self,
        runtime: "Runtime",
        *,
        provider: VivadoProvider,
        config: VivadoModuleConfig | None = None,
    ) -> None:
        self.runtime = runtime
        self.provider = provider
        self.config = config or VivadoModuleConfig()
        self._sessions: dict[str, _VivadoRuntimeSession] = {}
        self._pending_session_creates = 0
        self._pending_session_creates_by_process: dict[str, int] = {}
        self._lock = threading.RLock()

    def server_resource(self) -> str:
        return "vivado:server"

    def project_resource(self, project: str) -> str:
        return f"vivado:project:{self._validate_project(project)}"

    def health(self, pid: str) -> VivadoHealthResult:
        resource = self.server_resource()
        decision = self.runtime.capability.require(pid, resource, CapabilityRight.READ)
        started = time.monotonic()
        self._consume_decision(decision, used_by="vivado")
        response = self.provider.health(timeout_s=self.config.vivado.request_timeout_s)
        self._record_provider_effect(
            pid,
            operation="health",
            target=resource,
            context={"resource": resource},
            result={"ok": True, "elapsed_s": time.monotonic() - started},
            event_type=EventType.EXTERNAL_READ,
            decision={"ok": True},
        )
        return VivadoHealthResult(ok=True, response=response)

    def sync_push(
        self,
        pid: str,
        project: str,
        *,
        local_root: str,
        include_globs: list[str],
        exclude_globs: list[str] | None,
        delete_extra: bool,
        force: bool,
    ) -> VivadoPushResult:
        selected_project = self._validate_project(project)
        selected_include = self._normalize_globs(include_globs, "include_globs")
        selected_exclude = self._default_push_excludes(exclude_globs)
        project_decision = self.runtime.capability.require(
            pid,
            self.project_resource(selected_project),
            CapabilityRight.WRITE,
            {"operation": "vivado.sync_push", "project": selected_project},
        )
        manifest = self._scan_local_manifest(
            pid,
            local_root=local_root,
            include_globs=selected_include,
            exclude_globs=selected_exclude,
            required_filesystem_right=CapabilityRight.READ,
            operation="vivado.sync_push.scan",
        )
        self._charge(pid, ResourceUsage(external_read_bytes=manifest.hashed_file_bytes), source="primitive.vivado.scan_push")
        plan: dict[str, Any] | None = None
        sync_id: str | None = None
        try:
            self._consume_decision(project_decision, used_by="vivado")
            plan = self.provider.create_push_plan(
                selected_project,
                manifest.entries,
                delete_extra=delete_extra,
                include_globs=selected_include,
                exclude_globs=selected_exclude,
                timeout_s=self.config.vivado.request_timeout_s,
            )
            sync_id = self._require_response_string(plan, "sync_id")
            self._record_provider_effect(
                pid,
                operation="push_plan",
                target=f"vivado:sync:{sync_id}",
                context={"project": selected_project, "delete_extra": delete_extra},
                result={"sync_id": sync_id, "uploads": len(plan.get("upload_files") or [])},
                event_type=EventType.EXTERNAL_WRITE,
                decision={"project": selected_project, "sync_id": sync_id, "upload_count": len(plan.get("upload_files") or [])},
            )
            uploaded: list[str] = []
            file_map = {entry["path"]: entry for entry in manifest.entries if entry.get("kind") == "file"}
            for file_entry in self._list_of_mappings(plan.get("upload_files"), "upload_files"):
                sync_path = _validate_sync_path(str(file_entry.get("path", "")))
                manifest_entry = file_map.get(sync_path)
                if manifest_entry is None:
                    raise ValidationError(f"VivadoServer requested upload absent from local manifest: {sync_path}")
                source_path = self._path_for_sync_path(manifest.root_path, sync_path)
                self._verify_file_matches_manifest(source_path, manifest_entry)
                uploaded_result = self.provider.upload_push_file(
                    selected_project,
                    sync_id,
                    sync_path,
                    source_path,
                    size_bytes=int(manifest_entry["size_bytes"]),
                    sha256=str(manifest_entry["sha256"]),
                    timeout_s=self.config.vivado.request_timeout_s,
                )
                uploaded.append(sync_path)
                self._charge(
                    pid,
                    ResourceUsage(external_write_bytes=int(manifest_entry["size_bytes"])),
                    source="primitive.vivado.upload_file",
                    context={"project": selected_project, "path": sync_path},
                )
                self._record_provider_effect(
                    pid,
                    operation="upload_file",
                    target=f"vivado:sync:{sync_id}:{sync_path}",
                    context={"project": selected_project, "sync_id": sync_id, "path": sync_path},
                    result={"path": uploaded_result.get("path", sync_path), "size_bytes": manifest_entry["size_bytes"]},
                    event_type=EventType.EXTERNAL_WRITE,
                    decision={"project": selected_project, "sync_id": sync_id, "path": sync_path},
                )
            commit = self.provider.commit_push(
                selected_project,
                sync_id,
                force=force,
                timeout_s=self.config.vivado.request_timeout_s,
            )
            self._record_provider_effect(
                pid,
                operation="commit_push",
                target=f"vivado:sync:{sync_id}",
                context={"project": selected_project, "sync_id": sync_id, "force": force},
                result={"status": commit.get("status"), "uploaded_files": commit.get("uploaded_files")},
                event_type=EventType.EXTERNAL_WRITE,
                decision={"project": selected_project, "sync_id": sync_id, "force": force, "status": commit.get("status")},
            )
            return VivadoPushResult(
                project=selected_project,
                sync_id=sync_id,
                status=str(commit.get("status") or "committed"),
                uploaded_files=[str(item) for item in commit.get("uploaded_files", uploaded)],
                created_dirs=[str(item) for item in commit.get("created_dirs", plan.get("create_dirs", []))],
                deleted_files=[str(item) for item in commit.get("deleted_files", plan.get("delete_files", []))],
                deleted_dirs=[str(item) for item in commit.get("deleted_dirs", plan.get("delete_dirs", []))],
                upload_count=len(uploaded),
            )
        except Exception:
            if sync_id is not None:
                with contextlib.suppress(Exception):
                    aborted = self.provider.abort_push(
                        selected_project,
                        sync_id,
                        timeout_s=self.config.vivado.request_timeout_s,
                    )
                    self._record_provider_effect(
                        pid,
                        operation="abort_push",
                        target=f"vivado:sync:{sync_id}",
                        context={"project": selected_project, "sync_id": sync_id},
                        result=aborted,
                        event_type=EventType.EXTERNAL_WRITE,
                        decision={"project": selected_project, "sync_id": sync_id, "status": aborted.get("status")},
                    )
            raise

    def sync_pull(
        self,
        pid: str,
        project: str,
        *,
        local_root: str,
        include_globs: list[str],
        exclude_globs: list[str],
        delete_extra: bool,
        apply_mtime: bool,
    ) -> VivadoPullResult:
        selected_project = self._validate_project(project)
        selected_include = self._normalize_globs(include_globs, "include_globs")
        selected_exclude = self._normalize_globs(exclude_globs, "exclude_globs")
        project_decision = self.runtime.capability.require(
            pid,
            self.project_resource(selected_project),
            CapabilityRight.READ,
            {"operation": "vivado.sync_pull", "project": selected_project},
        )
        manifest = self._scan_local_manifest(
            pid,
            local_root=local_root,
            include_globs=selected_include,
            exclude_globs=selected_exclude,
            required_filesystem_right=CapabilityRight.READ,
            operation="vivado.sync_pull.scan",
        )
        write_decision = self._require_root_filesystem_right(
            pid,
            manifest.root_relative,
            CapabilityRight.WRITE,
            operation="vivado.sync_pull.write",
        )
        delete_decision = None
        if delete_extra:
            delete_decision = self._require_root_filesystem_right(
                pid,
                manifest.root_relative,
                CapabilityRight.DELETE,
                operation="vivado.sync_pull.delete_extra",
            )
        self._charge(pid, ResourceUsage(external_read_bytes=manifest.hashed_file_bytes), source="primitive.vivado.scan_pull")
        self._consume_decision(project_decision, used_by="vivado")
        self._consume_decision(write_decision, used_by="vivado")
        if delete_decision is not None:
            self._consume_decision(delete_decision, used_by="vivado")
        plan = self.provider.create_pull_plan(
            selected_project,
            manifest.entries,
            delete_extra=delete_extra,
            include_globs=selected_include,
            exclude_globs=selected_exclude,
            timeout_s=self.config.vivado.request_timeout_s,
        )
        self._record_provider_effect(
            pid,
            operation="pull_plan",
            target=self.project_resource(selected_project),
            context={"project": selected_project, "delete_extra": delete_extra},
            result={"downloads": len(plan.get("download_files") or [])},
            event_type=EventType.EXTERNAL_READ,
            decision={"project": selected_project, "download_count": len(plan.get("download_files") or [])},
        )
        for sync_path in [str(item) for item in plan.get("create_dirs", [])]:
            self._make_local_dir(manifest.root_path, _validate_sync_path(sync_path))
        downloaded: list[str] = []
        for file_entry in self._list_of_mappings(plan.get("download_files"), "download_files"):
            sync_path = _validate_sync_path(str(file_entry.get("path", "")))
            target_path = self._path_for_sync_path(manifest.root_path, sync_path)
            self._require_local_target_safe(manifest.root_path, target_path, allow_missing=True)
            expected_size = self._require_plan_file_size(file_entry, sync_path)
            if expected_size > self.config.vivado.max_file_bytes:
                raise ValidationError(
                    f"Vivado pull file exceeds max_file_bytes={self.config.vivado.max_file_bytes}: {sync_path}"
                )
            tmp_path = self._tmp_download_path(manifest.root_path, sync_path)
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            result = self.provider.download_file(
                selected_project,
                sync_path,
                tmp_path,
                timeout_s=self.config.vivado.request_timeout_s,
            )
            try:
                expected_sha = str(file_entry.get("sha256") or result.get("sha256") or "").lower()
                reported_sha = str(result.get("sha256") or "").lower()
                reported_size = int(result.get("size_bytes", -1))
                expected_mtime = file_entry.get("mtime_unix_ms", result.get("mtime_unix_ms"))
                if reported_size > self.config.vivado.max_file_bytes:
                    raise ValidationError(
                        f"Vivado downloaded file exceeds max_file_bytes={self.config.vivado.max_file_bytes}: {sync_path}"
                    )
                self._verify_download(tmp_path, expected_size=reported_size, expected_sha=reported_sha)
                if reported_size != expected_size:
                    raise ValidationError("Vivado downloaded file size mismatch with pull plan")
                if reported_sha != expected_sha:
                    raise ValidationError("Vivado downloaded file sha256 mismatch with pull plan")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                self._require_local_target_safe(manifest.root_path, target_path, allow_missing=True)
                os.replace(tmp_path, target_path)
                if apply_mtime and expected_mtime is not None:
                    mtime_s = int(expected_mtime) / 1000.0
                    os.utime(target_path, (mtime_s, mtime_s))
            except Exception:
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()
                raise
            downloaded.append(sync_path)
            self._charge(
                pid,
                ResourceUsage(external_read_bytes=expected_size, external_write_bytes=expected_size),
                source="primitive.vivado.download_file",
                context={"project": selected_project, "path": sync_path},
            )
            self._record_provider_effect(
                pid,
                operation="download_file",
                target=f"vivado:project:{selected_project}:{sync_path}",
                context={"project": selected_project, "path": sync_path},
                result={"path": sync_path, "size_bytes": expected_size},
                event_type=EventType.EXTERNAL_READ,
                decision={"project": selected_project, "path": sync_path, "size_bytes": expected_size},
            )
        deleted_files: list[str] = []
        deleted_dirs: list[str] = []
        if delete_extra:
            for sync_path in [str(item) for item in plan.get("delete_files", [])]:
                path = self._path_for_sync_path(manifest.root_path, _validate_sync_path(sync_path))
                self._delete_local_file(manifest.root_path, path)
                deleted_files.append(sync_path)
            for sync_path in sorted([str(item) for item in plan.get("delete_dirs", [])], key=lambda value: value.count("/"), reverse=True):
                path = self._path_for_sync_path(manifest.root_path, _validate_sync_path(sync_path))
                self._delete_local_dir(manifest.root_path, path)
                deleted_dirs.append(sync_path)
        return VivadoPullResult(
            project=selected_project,
            downloaded_files=downloaded,
            created_dirs=[str(item) for item in plan.get("create_dirs", [])],
            deleted_files=deleted_files,
            deleted_dirs=deleted_dirs,
            download_count=len(downloaded),
        )

    def create_session(self, pid: str, project: str, args: list[str], *, name: str | None = None) -> VivadoSessionCreateResult:
        selected_project = self._validate_project(project)
        selected_args = self._validate_session_args(args)
        resource = self.project_resource(selected_project)
        decision = self.runtime.capability.require(
            pid,
            resource,
            CapabilityRight.EXECUTE,
            {"operation": "vivado.session_create", "project": selected_project, "args": selected_args},
        )
        self._reserve_session_capacity(pid)
        reserved_capacity = True
        try:
            self._consume_decision(decision, used_by="vivado")
            response = self.provider.create_session(
                selected_project,
                selected_args,
                timeout_s=self.config.vivado.request_timeout_s,
            )
            session_id = self._require_response_string(response, "session_id")
            session_oid, object_name, namespace = self._create_session_object(
                pid,
                session_id=session_id,
                project=selected_project,
                args=selected_args,
                status=str(response.get("status") or "running"),
                name=name,
            )
            session = _VivadoRuntimeSession(
                session_oid=session_oid,
                session_id=session_id,
                owner_pid=pid,
                project=selected_project,
                args=selected_args,
                status=str(response.get("status") or "running"),
                cursor=0,
                started_at=response.get("started_at"),
                last_heartbeat_at=response.get("last_heartbeat_at"),
                exit_code=response.get("exit_code"),
                created_at=utc_now(),
            )
            with self._lock:
                self._sessions[session_oid] = session
                self._release_session_capacity_locked(pid)
                reserved_capacity = False
            self._consume_decision(decision, used_by="vivado")
            self._record_provider_effect(
                pid,
                operation="session_create",
                target=f"vivado:session:{session_id}",
                context={"project": selected_project, "session_id": session_id},
                result={"session_id": session_id, "status": session.status},
                event_type=EventType.EXTERNAL_WRITE,
                output_refs=[session_oid],
                decision={"project": selected_project, "session_id": session_id, "status": session.status},
            )
            return VivadoSessionCreateResult(
                session_oid=session_oid,
                namespace=namespace,
                name=object_name,
                type=ObjectType.EXTERNAL_REF.value,
                session_id=session_id,
                project=selected_project,
                status=session.status,
                cursor=session.cursor,
            )
        except Exception:
            if reserved_capacity:
                self._release_session_capacity(pid)
            raise

    def send_stdin(self, pid: str, session_oid: str, text: str) -> VivadoSessionWriteResult:
        if "\x00" in text:
            raise ValidationError("vivado stdin cannot contain NUL bytes")
        if len(text) > self.config.vivado.input_max_chars:
            raise ValidationError(f"vivado stdin exceeds configured limit {self.config.vivado.input_max_chars} chars")
        if len(text) > self.config.vivado.input_hard_limit_chars:
            raise ValidationError(f"vivado stdin exceeds hard limit {self.config.vivado.input_hard_limit_chars} chars")
        session = self._require_session(session_oid)
        self._require_owner(pid, session, "send stdin")
        self._require_object_right(pid, session_oid, ObjectRight.WRITE.value)
        response = self.provider.send_stdin(session.session_id, text, timeout_s=self.config.vivado.request_timeout_s)
        with session.lock:
            status = str(response.get("status") or session.status)
            session.status = status
        self._record_provider_effect(
            pid,
            operation="send_stdin",
            target=f"vivado:session:{session.session_id}",
            context={"project": session.project, "session_id": session.session_id, "chars": len(text)},
            result={"status": status, "chars": len(text)},
            event_type=EventType.EXTERNAL_WRITE,
            input_refs=[session_oid],
            decision={"session_id": session.session_id, "chars": len(text), "status": status},
        )
        return VivadoSessionWriteResult(
            session_oid=session_oid,
            session_id=session.session_id,
            chars_written=len(text),
            status=status,
        )

    def read_output(
        self,
        pid: str,
        session_oid: str,
        *,
        timeout_ms: int | None = None,
        max_chars: int | None = None,
        heartbeat: bool = True,
    ) -> VivadoSessionOutputResult:
        selected_timeout_ms = self._validate_timeout_ms(timeout_ms)
        selected_max_chars = self._validate_output_chars(max_chars)
        self._require_object_right(pid, session_oid, ObjectRight.READ.value)
        session = self._require_session(session_oid)
        heartbeat_sent = False
        if heartbeat and session.owner_pid == pid and self.runtime.capability.check(pid, f"object:{session_oid}", ObjectRight.WRITE):
            heartbeat_decision = self.runtime.capability.require(pid, f"object:{session_oid}", ObjectRight.WRITE)
            self._consume_decision(heartbeat_decision, used_by="vivado")
            heartbeat_response = self.provider.heartbeat(session.session_id, timeout_s=self.config.vivado.request_timeout_s)
            with session.lock:
                session.status = str(heartbeat_response.get("status") or session.status)
                session.last_heartbeat_at = heartbeat_response.get("last_heartbeat_at", session.last_heartbeat_at)
            self._record_provider_effect(
                pid,
                operation="heartbeat",
                target=f"vivado:session:{session.session_id}",
                context={"project": session.project, "session_id": session.session_id, "source": "read_output"},
                result={"status": session.status},
                event_type=EventType.EXTERNAL_WRITE,
                input_refs=[session_oid],
                decision={"session_id": session.session_id, "status": session.status, "source": "read_output"},
            )
            heartbeat_sent = True
        with session.lock:
            cursor = session.cursor
        response = self.provider.read_output(
            session.session_id,
            cursor=cursor,
            timeout_ms=selected_timeout_ms,
            timeout_s=max(self.config.vivado.request_timeout_s, selected_timeout_ms / 1000.0 + 1.0),
        )
        chunks = self._list_of_mappings(response.get("chunks"), "chunks")
        bounded_chunks, output, truncated = self._bounded_chunks(chunks, selected_max_chars)
        with session.lock:
            session.cursor = int(response.get("cursor", session.cursor))
            session.status = str(response.get("status") or session.status)
        chars = len(output)
        self._charge(
            pid,
            ResourceUsage(external_read_bytes=len(output.encode("utf-8", errors="replace"))),
            source="primitive.vivado.read_output",
            context={"session_id": session.session_id, "chars": chars},
        )
        self._record_provider_effect(
            pid,
            operation="read_output",
            target=f"vivado:session:{session.session_id}",
            context={"project": session.project, "session_id": session.session_id},
            result={"cursor": session.cursor, "chunks": len(chunks), "chars": chars, "overrun": bool(response.get("overrun"))},
            event_type=EventType.EXTERNAL_READ,
            input_refs=[session_oid],
            decision={"session_id": session.session_id, "cursor": session.cursor, "chunks": len(chunks), "chars": chars},
        )
        return VivadoSessionOutputResult(
            session_oid=session_oid,
            session_id=session.session_id,
            cursor=session.cursor,
            status=session.status,
            output=output,
            output_truncated=truncated,
            chunks=bounded_chunks,
            overrun=bool(response.get("overrun")),
            heartbeat_sent=heartbeat_sent,
        )

    def session_status(self, pid: str, session_oid: str) -> VivadoSessionStatusResult:
        self._require_object_right(pid, session_oid, ObjectRight.READ.value)
        session = self._require_session(session_oid)
        response = self.provider.get_session(session.session_id, timeout_s=self.config.vivado.request_timeout_s)
        with session.lock:
            session.status = str(response.get("status") or session.status)
            session.last_heartbeat_at = response.get("last_heartbeat_at", session.last_heartbeat_at)
            session.exit_code = response.get("exit_code", session.exit_code)
        self._record_provider_effect(
            pid,
            operation="session_status",
            target=f"vivado:session:{session.session_id}",
            context={"project": session.project, "session_id": session.session_id},
            result={"status": session.status},
            event_type=EventType.EXTERNAL_READ,
            input_refs=[session_oid],
            decision={"session_id": session.session_id, "status": session.status},
        )
        return VivadoSessionStatusResult(
            session_oid=session_oid,
            session_id=session.session_id,
            project=session.project,
            status=session.status,
            started_at=response.get("started_at", session.started_at),
            last_heartbeat_at=session.last_heartbeat_at,
            exit_code=session.exit_code,
        )

    def heartbeat(self, pid: str, session_oid: str) -> VivadoSessionHeartbeatResult:
        session = self._require_session(session_oid)
        self._require_owner(pid, session, "heartbeat")
        self._require_object_right(pid, session_oid, ObjectRight.WRITE.value)
        response = self.provider.heartbeat(session.session_id, timeout_s=self.config.vivado.request_timeout_s)
        with session.lock:
            session.status = str(response.get("status") or session.status)
            session.last_heartbeat_at = response.get("last_heartbeat_at", session.last_heartbeat_at)
        self._record_provider_effect(
            pid,
            operation="heartbeat",
            target=f"vivado:session:{session.session_id}",
            context={"project": session.project, "session_id": session.session_id},
            result={"status": session.status},
            event_type=EventType.EXTERNAL_WRITE,
            input_refs=[session_oid],
            decision={"session_id": session.session_id, "status": session.status},
        )
        return VivadoSessionHeartbeatResult(session_oid=session_oid, session_id=session.session_id, status=session.status)

    def close(self, pid: str, session_oid: str) -> VivadoSessionCloseResult:
        self._require_object_right(pid, session_oid, ObjectRight.DELETE.value)
        session_id, status = self._close_session(session_oid, actor=pid, reason="vivado_session_close")
        self.runtime.memory.delete_object_trusted(pid, session_oid, reason="vivado_session_close")
        return VivadoSessionCloseResult(session_oid=session_oid, session_id=session_id, closed=True, status=status)

    def close_for_object_release(self, oid: str, *, actor: str, reason: str) -> None:
        if oid not in self._sessions:
            return
        self._close_session(oid, actor=actor, reason=f"object_release:{reason}")

    def release_stale_session_objects(self) -> list[str]:
        released: list[str] = []
        for obj in list(self.runtime.store.list_objects()):
            if obj.type != ObjectType.EXTERNAL_REF:
                continue
            if not isinstance(obj.payload, dict) or obj.payload.get("kind") != "vivado_session":
                continue
            if obj.oid in self._sessions:
                continue
            if self.runtime.memory.delete_object_trusted("runtime.vivado", obj.oid, reason="stale_vivado_session"):
                released.append(obj.oid)
        if released:
            self.runtime.audit.record(
                actor="runtime.vivado",
                action="primitive.vivado.release_stale_objects",
                target="vivado:session:*",
                input_refs=released,
                decision={"released": released},
            )
        return released

    def shutdown(self) -> bool:
        with self._lock:
            session_oids = list(self._sessions)
        for oid in session_oids:
            with contextlib.suppress(Exception):
                self._close_session(oid, actor="runtime", reason="runtime.shutdown")
        return True

    def _scan_local_manifest(
        self,
        pid: str,
        *,
        local_root: str,
        include_globs: list[str],
        exclude_globs: list[str],
        required_filesystem_right: CapabilityRight,
        operation: str,
    ) -> LocalManifest:
        root_relative = self.runtime.resolve_process_working_directory(pid, local_root)
        decision = self._require_root_filesystem_right(pid, root_relative, required_filesystem_right, operation=operation)
        self._consume_decision(decision, used_by="vivado")
        target, _relative = self.runtime.filesystem.resolve_path(root_relative)
        root_path = Path(target.display)
        entries, hashed_bytes = self._build_manifest(root_path, include_globs=include_globs, exclude_globs=exclude_globs)
        return LocalManifest(entries=entries, root_relative=root_relative, root_path=root_path, hashed_file_bytes=hashed_bytes)

    def _require_root_filesystem_right(
        self,
        pid: str,
        root_relative: str,
        right: CapabilityRight,
        *,
        operation: str,
    ) -> Any:
        resource = self.runtime.filesystem.directory_resource_for(root_relative)
        return self.runtime.capability.require(
            pid,
            resource,
            right,
            {"operation": operation, "path": root_relative, "resource": resource},
        )

    def _build_manifest(self, root: Path, *, include_globs: list[str], exclude_globs: list[str]) -> tuple[list[dict[str, Any]], int]:
        if not root.exists() or not root.is_dir():
            raise NotFound(f"Vivado local root is not a directory: {root}")
        entries: list[dict[str, Any]] = []
        hashed_bytes = 0

        def scan(path: Path) -> None:
            nonlocal hashed_bytes
            rel = path.relative_to(root).as_posix()
            if rel == ".":
                return
            if path.is_symlink():
                if _is_reserved_sync_path(rel):
                    raise ValidationError(f"Vivado reserved sync directory must not be a symlink: {rel}")
                raise ValidationError(f"Vivado sync does not support symlinks: {rel}")
            if _is_reserved_sync_path(rel) or _glob_matches(rel, exclude_globs):
                return
            stat = path.stat()
            include = _glob_matches(rel, include_globs) if include_globs else True
            if path.is_dir():
                if include or _directory_may_contain_include(rel, include_globs):
                    if include:
                        entries.append({"path": rel, "kind": "dir", "mtime_unix_ms": stat.st_mtime_ns // 1_000_000})
                    for child in sorted(path.iterdir(), key=lambda item: item.name):
                        scan(child)
                return
            if not path.is_file():
                raise ValidationError(f"Vivado sync supports only regular files and directories: {rel}")
            if not include:
                return
            if stat.st_size > self.config.vivado.max_file_bytes:
                raise ValidationError(f"Vivado sync file exceeds max_file_bytes={self.config.vivado.max_file_bytes}: {rel}")
            digest = self._sha256_file(path)
            hashed_bytes += stat.st_size
            entries.append(
                {
                    "path": rel,
                    "kind": "file",
                    "size_bytes": stat.st_size,
                    "mtime_unix_ms": stat.st_mtime_ns // 1_000_000,
                    "sha256": digest,
                }
            )
            if len(entries) > self.config.vivado.max_manifest_entries:
                raise ValidationError(
                    f"Vivado sync manifest exceeds max_manifest_entries={self.config.vivado.max_manifest_entries}"
                )

        for child in sorted(root.iterdir(), key=lambda item: item.name):
            scan(child)
        entries.sort(key=lambda item: item["path"])
        return entries, hashed_bytes

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(self.config.vivado.file_chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_file_matches_manifest(self, path: Path, entry: dict[str, Any]) -> None:
        stat = path.stat()
        if stat.st_size != int(entry["size_bytes"]):
            raise ValidationError(f"Vivado upload file changed after planning: {entry['path']}")
        digest = self._sha256_file(path)
        if digest != str(entry["sha256"]).lower():
            raise ValidationError(f"Vivado upload file changed after planning: {entry['path']}")

    def _verify_download(self, path: Path, *, expected_size: int, expected_sha: str) -> None:
        if expected_size < 0:
            raise ValidationError("Vivado pull file is missing size_bytes")
        if not re.match(r"^[a-fA-F0-9]{64}$", expected_sha or ""):
            raise ValidationError("Vivado pull file is missing sha256")
        stat = path.stat()
        if stat.st_size != expected_size:
            raise ValidationError("Vivado downloaded file size mismatch")
        digest = self._sha256_file(path)
        if digest != expected_sha.lower():
            raise ValidationError("Vivado downloaded file sha256 mismatch")

    def _require_plan_file_size(self, file_entry: dict[str, Any], sync_path: str) -> int:
        if "size_bytes" not in file_entry:
            raise ValidationError(f"Vivado pull plan file is missing size_bytes: {sync_path}")
        try:
            selected = int(file_entry["size_bytes"])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Vivado pull plan file has invalid size_bytes: {sync_path}") from exc
        if selected < 0:
            raise ValidationError(f"Vivado pull plan file has invalid size_bytes: {sync_path}")
        return selected

    def _path_for_sync_path(self, root: Path, sync_path: str) -> Path:
        selected = root.joinpath(*_validate_sync_path(sync_path).split("/"))
        self._require_local_target_safe(root, selected, allow_missing=True)
        return selected

    def _require_local_target_safe(self, root: Path, target: Path, *, allow_missing: bool) -> None:
        resolved_parent = target.parent.resolve()
        root_resolved = root.resolve()
        if resolved_parent != root_resolved and root_resolved not in resolved_parent.parents:
            raise CapabilityDenied(f"Vivado sync target escapes local root: {target}")
        if target.exists():
            if target.is_symlink():
                raise ValidationError(f"Vivado sync target must not be a symlink: {target}")
            if not allow_missing and not target.exists():
                raise NotFound(f"Vivado sync target not found: {target}")

    def _make_local_dir(self, root: Path, sync_path: str) -> None:
        target = self._path_for_sync_path(root, sync_path)
        if target.exists() and target.is_symlink():
            raise ValidationError(f"Vivado sync directory target must not be a symlink: {sync_path}")
        target.mkdir(parents=True, exist_ok=True)

    def _delete_local_file(self, root: Path, target: Path) -> None:
        self._require_local_target_safe(root, target, allow_missing=False)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise ValidationError(f"Vivado pull delete target is not a regular file: {target}")
            target.unlink()

    def _delete_local_dir(self, root: Path, target: Path) -> None:
        self._require_local_target_safe(root, target, allow_missing=False)
        if target.exists():
            if target.is_symlink() or not target.is_dir():
                raise ValidationError(f"Vivado pull delete target is not a directory: {target}")
            target.rmdir()

    def _tmp_download_path(self, root: Path, sync_path: str) -> Path:
        digest = hashlib.sha256(sync_path.encode("utf-8")).hexdigest()[:16]
        reserved_root = root / _SYNC_RESERVED_DIR
        downloads_root = reserved_root / "downloads"
        if reserved_root.exists() and reserved_root.is_symlink():
            raise ValidationError(f"Vivado reserved sync directory must not be a symlink: {_SYNC_RESERVED_DIR}")
        if downloads_root.exists() and downloads_root.is_symlink():
            raise ValidationError(f"Vivado download directory must not be a symlink: {downloads_root}")
        tmp_path = downloads_root / f"{digest}.tmp"
        self._require_local_target_safe(root, tmp_path, allow_missing=True)
        return tmp_path

    def _create_session_object(
        self,
        pid: str,
        *,
        session_id: str,
        project: str,
        args: list[str],
        status: str,
        name: str | None,
    ) -> tuple[str, str, str]:
        object_name = name or f"{self.config.vivado.session_name_prefix}:{session_id.rsplit('-', 1)[-1]}"
        payload = {
            "kind": "vivado_session",
            "session_id": session_id,
            "project": project,
            "args": list(args),
            "status": status,
            "created_at": utc_now(),
        }
        handle = self.runtime.memory.create_object(
            pid=pid,
            object_type=ObjectType.EXTERNAL_REF,
            payload=payload,
            metadata=ObjectMetadata(title="Vivado session", tags=["vivado", "external_ref"]),
            immutable=False,
            provenance=Provenance(created_from_action="vivado.session_create"),
            name=object_name,
        )
        obj = self.runtime.memory.get_object(pid, handle)
        with self.runtime.store._lock:
            process = self.runtime.process.get(pid)
            if process.memory_view is None:
                process.memory_view = self.runtime.memory.create_view(pid, [handle], mode=ViewMode.READ_ONLY)
            elif all(existing.oid != handle.oid for existing in process.memory_view.roots):
                process.memory_view.roots.append(handle)
            self.runtime.store.update_process(process)
        return handle.oid, obj.name, obj.namespace

    def _close_session(self, session_oid: str, *, actor: str, reason: str) -> tuple[str, str]:
        session = self._require_session(session_oid)
        response = self.provider.delete_session(session.session_id, timeout_s=self.config.vivado.request_timeout_s)
        status = str(response.get("status") or "terminated")
        with self._lock:
            self._sessions.pop(session_oid, None)
        self._record_provider_effect(
            actor,
            operation="delete_session",
            target=f"vivado:session:{session.session_id}",
            context={"project": session.project, "session_id": session.session_id, "reason": reason},
            result={"status": status},
            event_type=EventType.EXTERNAL_WRITE,
            input_refs=[session_oid],
            decision={"session_id": session.session_id, "status": status, "reason": reason},
        )
        return session.session_id, status

    def _require_session(self, session_oid: str) -> _VivadoRuntimeSession:
        with self._lock:
            session = self._sessions.get(session_oid)
        if session is None:
            raise NotFound(f"Vivado session is not active: {session_oid}")
        return session

    def _require_owner(self, pid: str, session: _VivadoRuntimeSession, operation: str) -> None:
        if session.owner_pid != pid:
            raise CapabilityDenied(f"{pid} cannot {operation} for Vivado session owned by {session.owner_pid}")

    def _require_object_right(self, pid: str, oid: str, right: str) -> None:
        if self.runtime.store.get_object(oid) is None:
            raise NotFound(f"object not found: {oid}")
        decision = self.runtime.capability.require(pid, f"object:{oid}", right)
        self._consume_decision(decision, used_by="vivado")

    def _reserve_session_capacity(self, pid: str) -> None:
        with self._lock:
            active_sessions = list(self._sessions.values())
            global_count = len(active_sessions) + self._pending_session_creates
            process_count = (
                sum(1 for session in active_sessions if session.owner_pid == pid)
                + self._pending_session_creates_by_process.get(pid, 0)
            )
            if global_count >= self.config.vivado.max_sessions_global:
                raise ValidationError("Vivado session global limit reached")
            if process_count >= self.config.vivado.max_sessions_per_process:
                raise ValidationError("Vivado session per-process limit reached")
            self._pending_session_creates += 1
            self._pending_session_creates_by_process[pid] = self._pending_session_creates_by_process.get(pid, 0) + 1

    def _release_session_capacity(self, pid: str) -> None:
        with self._lock:
            self._release_session_capacity_locked(pid)

    def _release_session_capacity_locked(self, pid: str) -> None:
        if self._pending_session_creates <= 0:
            return
        self._pending_session_creates -= 1
        process_pending = self._pending_session_creates_by_process.get(pid, 0)
        if process_pending <= 1:
            self._pending_session_creates_by_process.pop(pid, None)
        else:
            self._pending_session_creates_by_process[pid] = process_pending - 1

    def _record_provider_effect(
        self,
        pid: str,
        *,
        operation: str,
        target: str,
        context: dict[str, Any],
        result: Any,
        event_type: EventType,
        decision: dict[str, Any],
        input_refs: list[str] | None = None,
        output_refs: list[str] | None = None,
    ) -> None:
        event = self.runtime.events.emit(
            event_type,
            source=pid,
            target=target,
            payload={"adapter": "vivado", "operation": operation, **decision},
        )
        audit_record = self.runtime.audit.record(
            actor=pid,
            action=f"primitive.vivado.{operation}",
            target=target,
            input_refs=input_refs or [],
            output_refs=output_refs or [],
            decision=decision,
        )
        classification = classify_external_effect(self.provider, operation, context, result)
        record_external_effect(
            self.runtime.store,
            pid=pid,
            provider="vivado",
            operation=operation,
            target=target,
            classification=classification,
            audit_record=audit_record,
            event=event,
            metadata={"operation": operation},
        )

    def _charge(
        self,
        pid: str,
        usage: ResourceUsage,
        *,
        source: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        resources = getattr(self.runtime, "resources", None)
        if resources is not None:
            resources.charge(pid, usage, source=source, context=context or {})

    def _consume_decision(self, decision: Any, *, used_by: str) -> None:
        cap_id = getattr(decision, "consume_capability_id", None)
        if cap_id is not None:
            self.runtime.capability.consume_use(cap_id, used_by=used_by, reason="one-time Vivado permission consumed")

    def _default_push_excludes(self, value: list[str] | None) -> list[str]:
        if value is None:
            globs = list(self.config.vivado.default_exclude_globs)
        else:
            globs = list(value)
        return self._normalize_globs([*globs, _SYNC_RESERVED_DIR, f"{_SYNC_RESERVED_DIR}/**"], "exclude_globs")

    def _normalize_globs(self, value: list[str] | tuple[str, ...] | None, field: str) -> list[str]:
        if value is None:
            return []
        result: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValidationError(f"Vivado {field} entries must be non-empty strings")
            text = item.strip().replace("\\", "/")
            if text.startswith("/") or ":" in text:
                raise ValidationError(f"Vivado {field} entries must be relative sync globs: {item}")
            result.append(text)
        return result

    def _validate_project(self, project: str) -> str:
        if not isinstance(project, str) or not project.strip():
            raise ValidationError("Vivado project must be a non-empty string")
        selected = project.strip()
        if selected in {".", ".."} or not _PROJECT_PATTERN.match(selected):
            raise ValidationError(f"invalid Vivado project name: {project!r}")
        return selected

    def _validate_session_args(self, args: list[str]) -> list[str]:
        if not isinstance(args, list) or not args:
            raise ValidationError("Vivado session args must be a non-empty list")
        if len(args) > self.config.vivado.max_session_args:
            raise ValidationError(f"Vivado session args exceed max_session_args={self.config.vivado.max_session_args}")
        selected: list[str] = []
        for arg in args:
            if not isinstance(arg, str):
                raise ValidationError("Vivado session args must be strings")
            if "\x00" in arg or any(ord(char) < 32 for char in arg):
                raise ValidationError("Vivado session args must not contain control characters")
            if len(arg) > self.config.vivado.max_arg_chars:
                raise ValidationError(f"Vivado session arg exceeds max_arg_chars={self.config.vivado.max_arg_chars}")
            selected.append(arg)
        return selected

    def _validate_timeout_ms(self, value: int | None) -> int:
        selected = self.config.vivado.output_timeout_ms if value is None else int(value)
        if selected < 0:
            raise ValidationError("Vivado output timeout_ms must be >= 0")
        return selected

    def _validate_output_chars(self, value: int | None) -> int:
        selected = self.config.vivado.max_output_chars if value is None else int(value)
        if selected < 1:
            raise ValidationError("Vivado max_chars must be >= 1")
        if selected > self.config.vivado.output_hard_limit_chars:
            raise ValidationError(f"Vivado max_chars exceeds hard limit {self.config.vivado.output_hard_limit_chars}")
        return selected

    def _bounded_chunks(self, chunks: list[dict[str, Any]], max_chars: int) -> tuple[list[dict[str, Any]], str, bool]:
        output = "".join(str(chunk.get("text", "")) for chunk in chunks)
        if len(output) <= max_chars:
            return chunks, output, False
        remaining = max_chars
        selected: list[dict[str, Any]] = []
        for chunk in reversed(chunks):
            text = str(chunk.get("text", ""))
            if remaining <= 0:
                break
            if len(text) > remaining:
                copy = dict(chunk)
                copy["text"] = text[-remaining:]
                selected.append(copy)
                remaining = 0
            else:
                selected.append(dict(chunk))
                remaining -= len(text)
        selected.reverse()
        return selected, output[-max_chars:], True

    def _require_response_string(self, value: dict[str, Any], field: str) -> str:
        selected = value.get(field)
        if not isinstance(selected, str) or not selected.strip():
            raise ValidationError(f"VivadoServer response missing {field}")
        return selected.strip()

    def _list_of_mappings(self, value: Any, field: str) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValidationError(f"VivadoServer response {field} must be a list")
        result: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                raise ValidationError(f"VivadoServer response {field} entries must be objects")
            result.append(dict(item))
        return result


class VivadoHealthArgs(BaseModel):
    pass


class VivadoHealthOutput(BaseModel):
    ok: bool
    response: dict[str, Any]


class VivadoSyncPushArgs(BaseModel):
    project: str = Field(description="VivadoServer project name.")
    local_root: str = Field(default=".", description="Workspace-relative local project root.")
    include_globs: list[str] = Field(default_factory=list, description="Relative sync globs to include.")
    exclude_globs: list[str] | None = Field(default=None, description="Relative sync globs to exclude; omitted uses Vivado defaults.")
    delete_extra: bool = Field(default=False, description="Delete extra remote project files during commit.")
    force: bool = Field(default=False, description="Force commit over server-side conflicts.")


class VivadoSyncPushOutput(BaseModel):
    project: str
    sync_id: str
    status: str
    uploaded_files: list[str]
    created_dirs: list[str]
    deleted_files: list[str]
    deleted_dirs: list[str]
    upload_count: int


class VivadoSyncPullArgs(BaseModel):
    project: str = Field(description="VivadoServer project name.")
    local_root: str = Field(default=".", description="Workspace-relative local project root.")
    include_globs: list[str] = Field(default_factory=list, description="Relative sync globs to include.")
    exclude_globs: list[str] = Field(default_factory=list, description="Relative sync globs to exclude.")
    delete_extra: bool = Field(default=False, description="Delete extra local files returned by the pull plan.")
    apply_mtime: bool = Field(default=True, description="Apply server mtime metadata to downloaded files.")


class VivadoSyncPullOutput(BaseModel):
    project: str
    downloaded_files: list[str]
    created_dirs: list[str]
    deleted_files: list[str]
    deleted_dirs: list[str]
    download_count: int


class VivadoSessionCreateArgs(BaseModel):
    project: str = Field(description="VivadoServer project name.")
    args: list[str] = Field(default_factory=lambda: ["-mode", "tcl"], description="Vivado CLI argv after the executable.")
    name: str | None = Field(default=None, description="Optional Object Memory name for the session object.")


class VivadoSessionCreateOutput(BaseModel):
    session_oid: str
    namespace: str
    name: str
    type: str
    session_id: str
    project: str
    status: str
    cursor: int


class VivadoSessionStdinArgs(BaseModel):
    session_oid: str = Field(description="Object oid returned by vivado_session_create.")
    text: str = Field(description="Text to write to Vivado stdin.")


class VivadoSessionStdinOutput(BaseModel):
    session_oid: str
    session_id: str
    chars_written: int
    status: str


class VivadoSessionReadOutputArgs(BaseModel):
    session_oid: str = Field(description="Object oid returned by vivado_session_create.")
    timeout_ms: int | None = Field(default=None, ge=0, description="Server long-poll timeout in milliseconds.")
    max_chars: int | None = Field(default=None, ge=1, description="Maximum output chars to return.")
    heartbeat: bool = Field(default=True, description="Send heartbeat before reading when called by the owning process.")


class VivadoSessionReadOutputOutput(BaseModel):
    session_oid: str
    session_id: str
    cursor: int
    status: str
    output: str
    output_truncated: bool
    chunks: list[dict[str, Any]]
    overrun: bool
    heartbeat_sent: bool


class VivadoSessionStatusArgs(BaseModel):
    session_oid: str = Field(description="Object oid returned by vivado_session_create.")


class VivadoSessionStatusOutput(BaseModel):
    session_oid: str
    session_id: str
    project: str
    status: str
    started_at: str | None
    last_heartbeat_at: str | None
    exit_code: int | None


class VivadoSessionHeartbeatArgs(BaseModel):
    session_oid: str = Field(description="Object oid returned by vivado_session_create.")


class VivadoSessionHeartbeatOutput(BaseModel):
    session_oid: str
    session_id: str
    status: str


class VivadoSessionCloseArgs(BaseModel):
    session_oid: str = Field(description="Object oid returned by vivado_session_create.")


class VivadoSessionCloseOutput(BaseModel):
    session_oid: str
    session_id: str
    closed: bool
    status: str


class VivadoHealthTool(SyncAgentTool[VivadoHealthArgs]):
    name = "vivado_health"
    description = "Check VivadoServer health."
    args_schema = VivadoHealthArgs
    output_schema = VivadoHealthOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, declared_permissions={"vivado.read"}, timeout_s=None)
    tags = ["vivado", "health"]

    def run(self, args: VivadoHealthArgs, ctx: ToolContext) -> VivadoHealthOutput:
        return VivadoHealthOutput(**asdict(_vivado_adapter(_runtime(ctx)).health(ctx.pid)))


class VivadoSyncPushTool(SyncAgentTool[VivadoSyncPushArgs]):
    name = "vivado_sync_push"
    description = "Push a local workspace project tree to VivadoServer using a safe sync plan."
    args_schema = VivadoSyncPushArgs
    output_schema = VivadoSyncPushOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"vivado.write", "filesystem.read"},
        timeout_s=None,
    )
    tags = ["vivado", "sync", "push", "external", "side_effect"]

    def run(self, args: VivadoSyncPushArgs, ctx: ToolContext) -> VivadoSyncPushOutput:
        result = _vivado_adapter(_runtime(ctx)).sync_push(
            ctx.pid,
            args.project,
            local_root=args.local_root,
            include_globs=args.include_globs,
            exclude_globs=args.exclude_globs,
            delete_extra=args.delete_extra,
            force=args.force,
        )
        return VivadoSyncPushOutput(**asdict(result))


class VivadoSyncPullTool(SyncAgentTool[VivadoSyncPullArgs]):
    name = "vivado_sync_pull"
    description = "Pull selected VivadoServer project files into a local workspace project tree."
    args_schema = VivadoSyncPullArgs
    output_schema = VivadoSyncPullOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"vivado.read", "filesystem.write"},
        timeout_s=None,
    )
    tags = ["vivado", "sync", "pull", "external", "side_effect"]

    def run(self, args: VivadoSyncPullArgs, ctx: ToolContext) -> VivadoSyncPullOutput:
        result = _vivado_adapter(_runtime(ctx)).sync_pull(
            ctx.pid,
            args.project,
            local_root=args.local_root,
            include_globs=args.include_globs,
            exclude_globs=args.exclude_globs,
            delete_extra=args.delete_extra,
            apply_mtime=args.apply_mtime,
        )
        return VivadoSyncPullOutput(**asdict(result))


class VivadoSessionCreateTool(SyncAgentTool[VivadoSessionCreateArgs]):
    name = "vivado_session_create"
    description = "Create a remote Vivado session and return an Object Memory EXTERNAL_REF handle for it."
    args_schema = VivadoSessionCreateArgs
    output_schema = VivadoSessionCreateOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"vivado.execute", "object.write"},
        timeout_s=None,
    )
    tags = ["vivado", "session", "external", "side_effect"]

    def run(self, args: VivadoSessionCreateArgs, ctx: ToolContext) -> VivadoSessionCreateOutput:
        result = _vivado_adapter(_runtime(ctx)).create_session(ctx.pid, args.project, args.args, name=args.name)
        return VivadoSessionCreateOutput(**asdict(result))


class VivadoSessionStdinTool(SyncAgentTool[VivadoSessionStdinArgs]):
    name = "vivado_session_send_stdin"
    description = "Write text to an active Object-bound Vivado session."
    args_schema = VivadoSessionStdinArgs
    output_schema = VivadoSessionStdinOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"vivado.execute", "object.write"},
        timeout_s=None,
    )
    tags = ["vivado", "stdin", "external", "side_effect"]

    def run(self, args: VivadoSessionStdinArgs, ctx: ToolContext) -> VivadoSessionStdinOutput:
        result = _vivado_adapter(_runtime(ctx)).send_stdin(ctx.pid, args.session_oid, args.text)
        return VivadoSessionStdinOutput(**asdict(result))


class VivadoSessionReadOutputTool(SyncAgentTool[VivadoSessionReadOutputArgs]):
    name = "vivado_session_read_output"
    description = "Read buffered output from an active Object-bound Vivado session."
    args_schema = VivadoSessionReadOutputArgs
    output_schema = VivadoSessionReadOutputOutput
    policy = ToolPolicy(side_effects=False, idempotent=False, declared_permissions={"object.read"}, timeout_s=None)
    tags = ["vivado", "output", "read"]

    def run(self, args: VivadoSessionReadOutputArgs, ctx: ToolContext) -> VivadoSessionReadOutputOutput:
        result = _vivado_adapter(_runtime(ctx)).read_output(
            ctx.pid,
            args.session_oid,
            timeout_ms=args.timeout_ms,
            max_chars=args.max_chars,
            heartbeat=args.heartbeat,
        )
        return VivadoSessionReadOutputOutput(**asdict(result))


class VivadoSessionStatusTool(SyncAgentTool[VivadoSessionStatusArgs]):
    name = "vivado_session_status"
    description = "Get status for an active Object-bound Vivado session."
    args_schema = VivadoSessionStatusArgs
    output_schema = VivadoSessionStatusOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, declared_permissions={"object.read"}, timeout_s=None)
    tags = ["vivado", "session", "status"]

    def run(self, args: VivadoSessionStatusArgs, ctx: ToolContext) -> VivadoSessionStatusOutput:
        result = _vivado_adapter(_runtime(ctx)).session_status(ctx.pid, args.session_oid)
        return VivadoSessionStatusOutput(**asdict(result))


class VivadoSessionHeartbeatTool(SyncAgentTool[VivadoSessionHeartbeatArgs]):
    name = "vivado_session_heartbeat"
    description = "Send a heartbeat for an active Object-bound Vivado session."
    args_schema = VivadoSessionHeartbeatArgs
    output_schema = VivadoSessionHeartbeatOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"vivado.execute", "object.write"},
        timeout_s=None,
    )
    tags = ["vivado", "heartbeat", "external", "side_effect"]

    def run(self, args: VivadoSessionHeartbeatArgs, ctx: ToolContext) -> VivadoSessionHeartbeatOutput:
        result = _vivado_adapter(_runtime(ctx)).heartbeat(ctx.pid, args.session_oid)
        return VivadoSessionHeartbeatOutput(**asdict(result))


class VivadoSessionCloseTool(SyncAgentTool[VivadoSessionCloseArgs]):
    name = "vivado_session_close"
    description = "Close an active Object-bound Vivado session and release its Object Memory handle."
    args_schema = VivadoSessionCloseArgs
    output_schema = VivadoSessionCloseOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"vivado.execute", "object.delete"},
        timeout_s=None,
    )
    tags = ["vivado", "close", "external", "side_effect"]

    def run(self, args: VivadoSessionCloseArgs, ctx: ToolContext) -> VivadoSessionCloseOutput:
        result = _vivado_adapter(_runtime(ctx)).close(ctx.pid, args.session_oid)
        return VivadoSessionCloseOutput(**asdict(result))


def register_module(ctx: Any) -> None:
    for tool in [
        VivadoHealthTool(),
        VivadoSyncPushTool(),
        VivadoSyncPullTool(),
        VivadoSessionCreateTool(),
        VivadoSessionStdinTool(),
        VivadoSessionReadOutputTool(),
        VivadoSessionStatusTool(),
        VivadoSessionHeartbeatTool(),
        VivadoSessionCloseTool(),
    ]:
        ctx.register_tool(tool)

    ctx.register_image(
        AgentImage(
            image_id="vivado-agent:v0",
            name="vivado-agent",
            default_tools=[
                "process_exit",
                "vivado_health",
                "vivado_session_close",
                "vivado_session_create",
                "vivado_session_heartbeat",
                "vivado_session_read_output",
                "vivado_session_send_stdin",
                "vivado_session_status",
                "vivado_sync_pull",
                "vivado_sync_push",
            ],
            required_capabilities=[
                {"resource": "vivado:server", "rights": ["read"]},
                {"resource": "vivado:project:*", "rights": ["read", "write", "execute"]},
            ],
            metadata={"module": "agent-libos-vivado:v0"},
        )
    )
    ctx.add_startup_hook(initialize_vivado)


def _runtime(ctx: ToolContext) -> Any:
    if ctx.runtime is None:
        raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
    return ctx.runtime


def _vivado_adapter(runtime: Any) -> VivadoAdapter:
    adapter = getattr(runtime, _VIVADO_ADAPTER_ATTR, None)
    if adapter is None:
        raise ToolExecutionError("Vivado module has not initialized.", code=ToolErrorCode.EXECUTION_ERROR)
    return adapter


def _validate_base_url(base_url: str, *, allow_plain_http_non_loopback: bool) -> None:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValidationError("VivadoServer base_url must be HTTP(S)")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValidationError("VivadoServer base_url must not include userinfo or fragment")
    host = (parsed.hostname or "").rstrip(".").lower()
    if parsed.scheme == "http" and not allow_plain_http_non_loopback and not _is_loopback_host(host):
        raise ValidationError("plain HTTP VivadoServer base_url is restricted to loopback")


def _is_loopback_host(host: str) -> bool:
    if host in _LOCAL_HTTP_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _validate_sync_path(path: str) -> str:
    if not isinstance(path, str):
        raise ValidationError("Vivado sync path must be a string")
    selected = path.strip().replace("\\", "/")
    if (
        not selected
        or selected in {".", ".."}
        or selected.startswith("/")
        or "\\" in path
        or ":" in selected
        or "//" in selected
    ):
        raise ValidationError(f"invalid Vivado sync path: {path!r}")
    parts = selected.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise ValidationError(f"invalid Vivado sync path: {path!r}")
    if _is_reserved_sync_path(selected):
        raise ValidationError(f"Vivado sync path uses reserved directory: {path!r}")
    return selected


def _is_reserved_sync_path(path: str) -> bool:
    return path == _SYNC_RESERVED_DIR or path.startswith(f"{_SYNC_RESERVED_DIR}/")


def _glob_matches(path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/**"):
            base = pattern[:-3].rstrip("/")
            if path == base or path.startswith(f"{base}/"):
                return True
        if fnmatch.fnmatchcase(path, pattern):
            return True
    return False


def _directory_may_contain_include(path: str, include_globs: list[str]) -> bool:
    if not include_globs:
        return True
    prefix = f"{path.rstrip('/')}/"
    for pattern in include_globs:
        literal = pattern.split("*", 1)[0]
        if literal.startswith(prefix) or prefix.startswith(literal):
            return True
    return False


def _quote_segment(value: str) -> str:
    return quote(value, safe="")


def _quote_sync_path(path: str) -> str:
    return "/".join(_quote_segment(segment) for segment in _validate_sync_path(path).split("/"))


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
