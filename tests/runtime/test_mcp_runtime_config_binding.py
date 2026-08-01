from __future__ import annotations

from dataclasses import replace

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.substrate import LocalResourceProviderSubstrate, SdkMcpProvider


def _selected_config():
    return replace(
        DEFAULT_CONFIG,
        mcp=replace(
            DEFAULT_CONFIG.mcp,
            protocol_probe_timeout_s=0.125,
            list_max_pages=3,
            list_limit=7,
        ),
    )


def test_runtime_reconstructs_builtin_mcp_provider_with_runtime_config(
    tmp_path,
) -> None:
    config = _selected_config()
    substrate = LocalResourceProviderSubstrate(tmp_path)
    substrate_provider = substrate.mcp

    runtime = Runtime.open("local", substrate=substrate, config=config)
    try:
        provider = runtime.mcp.provider
        assert type(provider) is SdkMcpProvider
        assert provider is not substrate_provider
        assert provider.mcp_config is config.mcp
        assert provider.mcp_config.protocol_probe_timeout_s == 0.125
        assert provider.mcp_config.list_max_pages == 3
        assert provider.mcp_config.list_limit == 7

        # Runtime assembly must not rewrite a caller-owned substrate. This also
        # keeps a failed or competing assembly from changing another Runtime's
        # effective provider through the shared substrate object.
        assert substrate.mcp is substrate_provider
    finally:
        runtime.close()


def test_runtime_preserves_custom_mcp_provider_identity(tmp_path) -> None:
    class CustomSdkMcpProvider(SdkMcpProvider):
        pass

    config = _selected_config()
    substrate = LocalResourceProviderSubstrate(tmp_path)
    custom_provider = CustomSdkMcpProvider(
        tmp_path,
        mcp_config=DEFAULT_CONFIG.mcp,
    )
    substrate.mcp = custom_provider

    runtime = Runtime.open("local", substrate=substrate, config=config)
    try:
        assert runtime.mcp.provider is custom_provider
        assert substrate.mcp is custom_provider
        assert custom_provider.mcp_config is DEFAULT_CONFIG.mcp
    finally:
        runtime.close()


def test_runtime_without_substrate_binds_default_mcp_provider_to_config(
    tmp_path,
    monkeypatch,
) -> None:
    config = _selected_config()
    monkeypatch.chdir(tmp_path)

    runtime = Runtime.open("local", config=config)
    try:
        provider = runtime.mcp.provider
        assert type(provider) is SdkMcpProvider
        assert provider.mcp_config is config.mcp
    finally:
        runtime.close()
