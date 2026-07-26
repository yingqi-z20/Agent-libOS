from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from agent_libos.config.defaults import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.yaml_loader import YAML_MAX_UTF8_BYTES, load_yaml_mapping

_CONFIG_ADAPTER = TypeAdapter(AgentLibOSConfig)


def get_project_root(start: str | Path | None = None) -> Path:
    """Return the repository root that owns the installed ``agent_libos`` tree."""

    selected = Path(start).expanduser().resolve() if start is not None else Path(__file__).resolve()
    base = selected if selected.is_dir() else selected.parent
    for candidate in (base, *base.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "agent_libos").is_dir():
            return candidate
    return Path(__file__).resolve().parents[2]


def load_config_file(path: str | Path, *, base: AgentLibOSConfig = DEFAULT_CONFIG) -> AgentLibOSConfig:
    """Load a YAML config file as a strict overlay on top of ``base``."""

    selected = Path(path)
    if not selected.exists():
        raise FileNotFoundError(f"config file not found: {selected}")
    try:
        data = load_yaml_mapping(_read_config_text(selected))
    except ValidationError as exc:
        if str(exc) == "YAML document must be a mapping":
            raise ValueError(
                f"config YAML root must be a mapping: {selected}"
            ) from exc
        raise ValueError(f"invalid YAML config {selected}: {exc}") from exc

    merged = _deep_merge(asdict(base), data)
    return _CONFIG_ADAPTER.validate_python(merged)


def load_config_from_project_root(
    filename: str | Path = "config.yaml",
    *,
    base: AgentLibOSConfig = DEFAULT_CONFIG,
    root: str | Path | None = None,
) -> AgentLibOSConfig:
    """Load ``filename`` from the project root when it exists."""

    selected_root = Path(root).expanduser().resolve() if root is not None else get_project_root()
    requested = Path(filename).expanduser()
    selected = requested if requested.is_absolute() else selected_root / requested
    if not selected.exists():
        return base
    return load_config_file(selected, base=base)


def load_config_from_cwd(
    filename: str | Path = "config.yaml", *, base: AgentLibOSConfig = DEFAULT_CONFIG
) -> AgentLibOSConfig:
    """Load ``filename`` from the current working directory when it exists."""

    selected = Path.cwd() / Path(filename)
    if not selected.exists():
        return base
    return load_config_file(selected, base=base)


def _deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(overlay, Mapping):
        merged = dict(base)
        for key, value in overlay.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return overlay


def _read_config_text(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(YAML_MAX_UTF8_BYTES + 1)
    if len(raw) > YAML_MAX_UTF8_BYTES:
        raise ValidationError(
            "YAML document exceeds "
            f"YAML_MAX_UTF8_BYTES={YAML_MAX_UTF8_BYTES}"
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("YAML document must be valid UTF-8 text") from exc
