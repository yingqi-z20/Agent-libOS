"""Host-selected provider tools and bounded, untrusted result projections.

This module never dispatches tool calls. Managed activities are observations of
work done inside the LLM provider; only function calls enter the Runtime loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urlsplit

from agent_libos.config import ProviderToolsConfig
from agent_libos.llm.sdk_fields import omit_unset_sdk_defaults


PROVIDER_TOOL_MAX_ITEMS = 256
PROVIDER_TOOL_TEXT_MAX_CHARS = 262_144
PROVIDER_TOOL_TOTAL_MAX_BYTES = 1_048_576
_TOOL_NAMES = ("web_search", "web_extractor", "code_interpreter")
_MANAGED_TYPES = {f"{name}_call": name for name in _TOOL_NAMES}


class ProviderToolsResponseError(ValueError):
    """A provider tool response violated its inbound protocol or size bounds."""


def configured_tool_names(config: ProviderToolsConfig | None) -> list[str]:
    return [name for name in _TOOL_NAMES if config is not None and getattr(config, name)]


def apply_provider_tools(
    payload: dict[str, Any], config: ProviderToolsConfig | None, *, api: str, enabled: bool,
) -> None:
    """Add the exact Host-approved tool surface, with no free-form options."""
    if config is None or not enabled:
        return
    names = configured_tool_names(config)
    if api == "chat":
        if config.provider != "aliyun" or names != ["web_search"]:
            raise ValueError("Chat supports only Aliyun web_search")
        payload["extra_body"] = {**payload.get("extra_body", {}), "enable_search": True}
        return
    tools = list(payload.get("tools", []))
    for name in names:
        tool: dict[str, Any] = {"type": name}
        if name == "code_interpreter" and config.provider == "openai":
            container: dict[str, Any] = {"type": "auto"}
            if config.file_ids:
                container["file_ids"] = list(config.file_ids)
            tool["container"] = container
        tools.append(tool)
    payload["tools"] = tools
    if config.provider == "openai":
        include = list(payload.get("include", []))
        if config.web_search:
            include.append("web_search_call.action.sources")
        if config.code_interpreter:
            include.append("code_interpreter_call.outputs")
        payload["include"] = list(dict.fromkeys(include))
    if config.provider == "aliyun" and (config.code_interpreter or config.web_extractor):
        payload["extra_body"] = {**payload.get("extra_body", {}), "enable_thinking": True}


def provider_tool_request_observation(
    config: ProviderToolsConfig | None, request: dict[str, Any], *, replay: bool,
) -> dict[str, Any] | None:
    if config is None:
        return None
    effective = [tool.get("type") for tool in request.get("tools", [])
                 if isinstance(tool, dict) and tool.get("type") in _TOOL_NAMES]
    if request.get("extra_body", {}).get("enable_search") is True:
        effective.append("web_search")
    return {
        "provider": config.provider,
        "configured": configured_tool_names(config),
        "effective": effective,
        "observed": "unknown" if effective else "not_returned",
        "replay": "native" if replay and not config.code_interpreter else "stateless",
        "file_count": len(config.file_ids),
        "usage": None,
    }


@dataclass
class ProviderToolResults:
    activities: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None


@dataclass
class _Bounds:
    bytes: int = 0
    items: int = 0

    def text(self, value: Any, *, required: bool = True) -> str | None:
        if value is None and not required:
            return None
        if not isinstance(value, str) or (required and not value):
            raise ProviderToolsResponseError("Invalid provider tool text")
        if len(value) > PROVIDER_TOOL_TEXT_MAX_CHARS:
            raise ProviderToolsResponseError("Provider tool text exceeds bounds")
        self.bytes += len(value.encode("utf-8"))
        if self.bytes > PROVIDER_TOOL_TOTAL_MAX_BYTES:
            raise ProviderToolsResponseError("Provider tool output exceeds byte bounds")
        return value

    def array(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)) or len(value) > PROVIDER_TOOL_MAX_ITEMS:
            raise ProviderToolsResponseError("Provider tool array exceeds bounds")
        self.items += len(value)
        if self.items > 4096:
            raise ProviderToolsResponseError("Provider tool output exceeds item bounds")
        return list(value)

    def url(self, value: Any) -> str:
        result = self.text(value)
        assert result is not None
        try:
            parsed = urlsplit(result)
            valid = parsed.scheme in {"https", "http"} and bool(parsed.hostname)
        except ValueError:
            valid = False
        if not valid or any(ord(char) < 32 for char in result):
            raise ProviderToolsResponseError("Invalid provider tool URL")
        return result


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _check_fields(value: Any, allowed: set[str]) -> None:
    fields = value if isinstance(value, dict) else getattr(value, "__dict__", None)
    if isinstance(fields, dict) and not isinstance(value, dict):
        fields = omit_unset_sdk_defaults(value, fields, allowed)
    extras = getattr(value, "__pydantic_extra__", None)
    for mapping in (fields, extras):
        if mapping is not None and (
            not isinstance(mapping, dict) or len(mapping) > len(allowed)
            or any(key not in allowed for key in mapping)
        ):
            raise ProviderToolsResponseError("Unsupported provider tool fields")


def _annotation(value: Any, result: ProviderToolResults, bounds: _Bounds, config: ProviderToolsConfig) -> None:
    kind = _get(value, "type")
    if kind == "url_citation":
        _check_fields(value, {"type", "url", "title", "start_index", "end_index"})
        annotation: dict[str, Any] = {"type": kind, "url": bounds.url(_get(value, "url"))}
        title = bounds.text(_get(value, "title"), required=False)
        if title is not None:
            annotation["title"] = title
        target = result.citations
    elif kind in {"container_file_citation", "file_citation", "file_path"} and config.code_interpreter:
        _check_fields(value, {"type", "file_id", "container_id", "filename", "start_index", "end_index", "index"})
        annotation = {"type": kind, "file_id": bounds.text(_get(value, "file_id"))}
        for name in ("container_id", "filename"):
            text = bounds.text(_get(value, name), required=False)
            if text is not None:
                annotation[name] = text
        target = result.artifacts
    else:
        raise ProviderToolsResponseError("Unsupported provider tool annotation")
    for name in ("start_index", "end_index", "index"):
        index = _get(value, name)
        if index is not None:
            if type(index) is not int or not 0 <= index <= PROVIDER_TOOL_TEXT_MAX_CHARS:
                raise ProviderToolsResponseError("Invalid provider tool annotation index")
            annotation[name] = index
    if ("start_index" in annotation and "end_index" in annotation
            and annotation["start_index"] > annotation["end_index"]):
        raise ProviderToolsResponseError("Invalid provider tool annotation range")
    if len(target) >= PROVIDER_TOOL_MAX_ITEMS:
        raise ProviderToolsResponseError("Provider tool annotations exceed bounds")
    target.append(annotation)


def _activity(value: Any, name: str, config: ProviderToolsConfig, bounds: _Bounds) -> dict[str, Any]:
    fields = {"type", "id", "status"} | {
        "web_search": {"action"}, "web_extractor": {"goal", "urls", "output"},
        "code_interpreter": {"code", "outputs", "container_id"},
    }[name]
    _check_fields(value, fields)
    status = _get(value, "status")
    terminal = {"completed", "failed"} if name != "web_extractor" else {"completed"}
    if status not in terminal:
        raise ProviderToolsResponseError("Provider tool returned a nonterminal status")
    activity: dict[str, Any] = {
        "type": _get(value, "type"), "id": bounds.text(_get(value, "id")), "status": status,
    }
    if name == "web_search":
        activity["action"] = _search_activity_action(value, config, bounds)
    elif name == "web_extractor":
        activity.update(_extractor_activity_result(value, bounds))
    else:
        activity.update(_code_activity_result(value, config, bounds))
    return activity


def _search_activity_action(value: Any, config: ProviderToolsConfig, bounds: _Bounds) -> dict[str, Any]:
    source = _get(value, "action")
    action_type = _get(source, "type")
    allowed = {"search"} if config.provider == "aliyun" else {"search", "open_page", "find_in_page"}
    if action_type not in allowed:
        raise ProviderToolsResponseError("Unsupported provider search action")
    _check_fields(source, {"type"} | {
        "search": {"query", "queries", "sources"},
        "open_page": {"url"}, "find_in_page": {"url", "pattern"},
    }[action_type])
    action: dict[str, Any] = {"type": action_type}
    for field_name in ("query", "pattern"):
        text = bounds.text(_get(source, field_name), required=False)
        if text is not None:
            action[field_name] = text
    if _get(source, "url") is not None:
        action["url"] = bounds.url(_get(source, "url"))
    if action_type == "find_in_page" and not all(key in action for key in ("pattern", "url")):
        raise ProviderToolsResponseError("Incomplete provider find action")
    if _get(source, "queries") is not None:
        action["queries"] = [bounds.text(query) for query in bounds.array(_get(source, "queries"))]
    if _get(source, "sources") is not None:
        sources = []
        for entry in bounds.array(_get(source, "sources")):
            _check_fields(entry, {"type", "url"})
            if _get(entry, "type") != "url":
                raise ProviderToolsResponseError("Unsupported provider search source")
            sources.append({"type": "url", "url": bounds.url(_get(entry, "url"))})
        action["sources"] = sources
    return action


def _extractor_activity_result(value: Any, bounds: _Bounds) -> dict[str, Any]:
    activity: dict[str, Any] = {}
    activity["goal"] = bounds.text(_get(value, "goal"))
    if _get(value, "urls") is None or not isinstance(_get(value, "output"), str):
        raise ProviderToolsResponseError("Incomplete provider extractor result")
    activity["urls"] = [bounds.url(url) for url in bounds.array(_get(value, "urls"))]
    activity["output"] = bounds.text(_get(value, "output"), required=False) or ""
    return activity


def _code_activity_result(value: Any, config: ProviderToolsConfig, bounds: _Bounds) -> dict[str, Any]:
    activity: dict[str, Any] = {}
    activity["code"] = bounds.text(_get(value, "code"), required=False)
    activity["container_id"] = bounds.text(_get(value, "container_id"), required=False)
    outputs = []
    for entry in bounds.array(_get(value, "outputs")):
        kind = _get(entry, "type")
        if kind == "logs":
            _check_fields(entry, {"type", "logs"})
            if not isinstance(_get(entry, "logs"), str):
                raise ProviderToolsResponseError("Invalid provider code logs")
            outputs.append({"type": "logs", "logs": bounds.text(_get(entry, "logs"), required=False) or ""})
        elif kind == "image" and config.provider == "openai":
            _check_fields(entry, {"type", "url"})
            outputs.append({"type": "image", "url": bounds.url(_get(entry, "url"))})
        else:
            raise ProviderToolsResponseError("Unsupported provider code output")
    activity["outputs"] = outputs
    return activity


def project_provider_tool_results(
    output: list[Any], config: ProviderToolsConfig | None, request: dict[str, Any], *, response: Any = None,
) -> ProviderToolResults:
    """Validate provider-specific activities independently of native replay."""
    result = ProviderToolResults()
    if config is None:
        return result
    bounds = _Bounds()
    observation = provider_tool_request_observation(config, request, replay=False)
    assert observation is not None
    enabled = observation["effective"]
    for item in output:
        kind = _get(item, "type")
        if kind in _MANAGED_TYPES:
            name = _MANAGED_TYPES[kind]
            if name not in enabled or (name == "web_extractor" and config.provider != "aliyun"):
                raise ProviderToolsResponseError("Provider returned a disabled managed tool")
            if len(result.activities) >= PROVIDER_TOOL_MAX_ITEMS:
                raise ProviderToolsResponseError("Provider tool activities exceed bounds")
            result.activities.append(_activity(item, name, config, bounds))
        elif kind == "message":
            for content in bounds.array(_get(item, "content")):
                for annotation in bounds.array(_get(content, "annotations")):
                    _annotation(annotation, result, bounds, config)
        elif kind not in {"reasoning", "function_call"}:
            raise ProviderToolsResponseError("Unsupported provider output item")
    # Aliyun exposes tool counts independently of token usage. Never infer
    # counts from activities or charge them to the Runtime function-tool meter.
    returned_usage = _get(response, "x_tools")
    if returned_usage is None:
        returned_usage = _get(_get(response, "usage"), "plugins")
    if returned_usage is not None:
        usage = {}
        for name in _TOOL_NAMES:
            entry = _get(returned_usage, name)
            if entry is None:
                continue
            count = _get(entry, "count")
            if type(count) is not int or not 0 <= count <= 1_000_000:
                raise ProviderToolsResponseError("Invalid provider tool usage")
            usage[name] = {"count": count}
        result.usage = usage or None
    return result


def provider_tool_result_text(
    content: str,
    activities: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> str:
    """Project results into plain, bounded history with no sandbox protocol state.

    Remote IDs are inert text references. No native call, code, container state,
    annotations object, or provider continuation token is carried into history.
    """
    parts: list[str] = []
    remaining = PROVIDER_TOOL_TEXT_MAX_CHARS

    def append(value: Any) -> None:
        nonlocal remaining
        if not isinstance(value, str) or not value or remaining <= 0:
            return
        prefix = "\n\n" if parts else ""
        selected = value[:max(0, remaining - len(prefix))]
        if selected:
            parts.append(prefix + selected)
            remaining -= len(prefix) + len(selected)

    append(content)
    for activity in activities[:PROVIDER_TOOL_MAX_ITEMS]:
        if not isinstance(activity, dict):
            continue
        kind = activity.get("type")
        for part in _activity_result_text_parts(activity):
            append(part)
        if not parts and kind in _MANAGED_TYPES:
            append(f"Provider {_MANAGED_TYPES[kind]} activity: {activity.get('status', 'unknown')}")
    for citation in citations[:PROVIDER_TOOL_MAX_ITEMS]:
        append(_citation_result_text(citation))
    for artifact in artifacts[:PROVIDER_TOOL_MAX_ITEMS]:
        append(_artifact_result_text(artifact))
    return "".join(parts)


def _activity_result_text_parts(activity: dict[str, Any]) -> Iterator[Any]:
    kind = activity.get("type")
    if kind == "web_extractor_call":
        yield activity.get("output")
    elif kind == "code_interpreter_call":
        outputs = activity.get("outputs")
        for output in outputs[:PROVIDER_TOOL_MAX_ITEMS] if isinstance(outputs, list) else []:
            if isinstance(output, dict) and output.get("type") == "logs":
                yield output.get("logs")
    elif kind == "web_search_call":
        action = activity.get("action")
        sources = action.get("sources") if isinstance(action, dict) else None
        for source in sources[:PROVIDER_TOOL_MAX_ITEMS] if isinstance(sources, list) else []:
            if isinstance(source, dict) and isinstance(source.get("url"), str):
                yield f"Source: {source['url']}"


def _citation_result_text(citation: Any) -> str | None:
    if not isinstance(citation, dict) or not isinstance(citation.get("url"), str):
        return None
    title = citation.get("title")
    return f"Source: {title if isinstance(title, str) else ''} {citation['url']}"


def _artifact_result_text(artifact: Any) -> str | None:
    if not isinstance(artifact, dict) or not isinstance(artifact.get("file_id"), str):
        return None
    filename = artifact.get("filename")
    return f"Artifact: {filename if isinstance(filename, str) else ''} (file ID: {artifact['file_id']})"
