from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG, DataFlowDefaults, load_config_file
from agent_libos.models import (
    DataFlowContext,
    DataIntegrity,
    DataLabels,
    DataReleaseBinding,
    DataSensitivity,
    DataSourceRef,
    DataTrustLevel,
    SinkTrustLevel,
    SinkTrustRule,
    SinkTrustSpec,
    sink_pattern_matches,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage import SQLiteStore
from agent_libos.utils.ids import utc_now


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def test_data_labels_are_strict_and_aggregate_conservatively() -> None:
    confidential = DataLabels(
        sensitivity="confidential",
        trust_level="verified",
        integrity="verified",
        tenant="tenant-a",
        principal="principal-a",
    )
    secret = DataLabels(
        sensitivity=DataSensitivity.SECRET,
        trust_level=DataTrustLevel.UNTRUSTED,
        integrity=DataIntegrity.CHECKED,
        tenant="tenant-b",
        principal="principal-a",
    )

    merged = DataLabels.aggregate([confidential, secret])

    assert merged.sensitivity is DataSensitivity.SECRET
    assert merged.trust_level is DataTrustLevel.UNTRUSTED
    assert merged.integrity is DataIntegrity.CHECKED
    assert merged.tenant == "mixed"
    assert merged.principal == "principal-a"
    assert merged.is_mixed_identity
    assert len(merged.labels_hash()) == 64

    with pytest.raises(ValueError, match="sensitivity must be one of"):
        DataLabels(sensitivity="top-secret")
    with pytest.raises(ValueError, match="unknown fields"):
        DataLabels.from_dict({"sensitivity": "secret", "payload": "must-not-be-accepted"})


def test_data_flow_context_deduplicates_and_hashes_versioned_sources() -> None:
    first = DataSourceRef("oid-a", 1, _SHA_A)
    second = DataSourceRef("oid-a", 2, _SHA_B)
    context = DataFlowContext(
        labels=DataLabels(sensitivity="restricted"),
        source_refs=(second, first, first),
        materialization_id="materialization-a",
    )

    assert context.source_refs == (first, second)
    assert len(context.source_refs_hash()) == 64

    with pytest.raises(ValueError, match="positive integer"):
        DataSourceRef("oid-a", 0, _SHA_A)
    with pytest.raises(ValueError, match="SHA-256"):
        DataSourceRef("oid-a", 1, "not-a-hash")


def test_sink_rules_enforce_trust_clearance_and_trailing_wildcards() -> None:
    trusted = SinkTrustRule(
        pattern="llm:corp-secure",
        trust_level="trusted",
        max_sensitivity="restricted",
        tenants=("tenant-a",),
        principals=("principal-a",),
        identity_sha256=_SHA_A,
    )

    assert trusted.trust_level is SinkTrustLevel.TRUSTED
    assert trusted.max_sensitivity is DataSensitivity.RESTRICTED
    assert sink_pattern_matches("jsonrpc:crm:*", "jsonrpc:crm:read")
    assert not sink_pattern_matches("jsonrpc:crm:*", "jsonrpc:other:read")
    assert len(trusted.spec_hash()) == 64

    with pytest.raises(ValueError, match="trailing wildcard"):
        SinkTrustRule(pattern="jsonrpc:*:read")
    with pytest.raises(ValueError, match="must not exceed normal"):
        SinkTrustRule(
            pattern="filesystem:workspace:secret.txt",
            trust_level="untrusted",
            max_sensitivity="secret",
        )
    with pytest.raises(ValueError, match="requires identity_sha256"):
        SinkTrustRule(pattern="mcp:corp:search", trust_level="trusted", max_sensitivity="confidential")
    with pytest.raises(ValueError, match="requires identity_sha256"):
        SinkTrustRule(pattern="shell:*", trust_level="trusted", max_sensitivity="confidential")
    with pytest.raises(ValueError, match="requires identity_sha256"):
        SinkTrustRule(pattern="pty:spawn:*", trust_level="trusted", max_sensitivity="confidential")
    with pytest.raises(ValueError, match="explicitly enumerate"):
        SinkTrustRule(pattern="human:owner:terminal", tenants=("*",))


def test_release_binding_normalizes_to_exact_payload_free_json_shape() -> None:
    binding = DataReleaseBinding(
        sink="jsonrpc:crm:update",
        sink_identity_sha256=_SHA_A,
        trust_id="trust-a",
        trust_hash=_SHA_B,
        registry_generation=7,
        manifest_hash=_SHA_C,
        labels_hash=_SHA_A,
        source_refs_hash=_SHA_B,
        payload_hash=_SHA_C,
        operation="jsonrpc.call",
        target_state_version=4,
    )

    normalized = DataReleaseBinding.normalize(binding.to_dict())

    assert normalized == binding.to_dict()
    assert set(normalized) == {
        "schema_version",
        "sink",
        "sink_identity_sha256",
        "trust_id",
        "trust_hash",
        "registry_generation",
        "manifest_hash",
        "labels_hash",
        "source_refs_hash",
        "payload_hash",
        "operation",
        "target_state_version",
    }
    with pytest.raises(ValueError, match="unknown fields"):
        DataReleaseBinding.from_dict({**binding.to_dict(), "payload": "secret"})


def test_data_flow_config_defaults_to_untrusted_normal_and_loads_rules(tmp_path: Path) -> None:
    assert DEFAULT_CONFIG.data_flow.default_trust_level is SinkTrustLevel.UNTRUSTED
    assert DEFAULT_CONFIG.data_flow.default_max_sensitivity is DataSensitivity.NORMAL
    assert DEFAULT_CONFIG.data_flow.sink_rules == ()

    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "data_flow:",
                "  sink_rules:",
                "    - pattern: 'llm:corp-secure'",
                "      trust_level: trusted",
                "      max_sensitivity: restricted",
                f"      identity_sha256: '{_SHA_A}'",
                "      tenants: [tenant-a]",
                "      principals: [principal-a]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config_file(path)

    assert config.data_flow.sink_rules[0].pattern == "llm:corp-secure"
    assert config.data_flow.sink_rules[0].tenants == ("tenant-a",)

    integrity_path = tmp_path / "integrity.yaml"
    integrity_path.write_text(
        "data_flow:\n"
        "  operation_minimum_integrity:\n"
        "    primitive.filesystem.write_text: checked\n",
        encoding="utf-8",
    )
    configured_integrity = load_config_file(integrity_path).data_flow
    assert (
        configured_integrity.operation_minimum_integrity[
            "primitive.filesystem.write_text"
        ]
        is DataIntegrity.CHECKED
    )
    with pytest.raises(TypeError, match="configuration mappings are immutable"):
        configured_integrity.operation_minimum_integrity[
            "primitive.filesystem.write_text"
        ] = DataIntegrity.UNTRUSTED

    with pytest.raises(ValueError, match="must remain untrusted"):
        AgentLibOSConfig(
            data_flow=replace(DEFAULT_CONFIG.data_flow, default_trust_level=SinkTrustLevel.TRUSTED)
        )
    with pytest.raises(ValueError, match="duplicate pattern"):
        AgentLibOSConfig(
            data_flow=DataFlowDefaults(
                sink_rules=(
                    SinkTrustRule("filesystem:workspace:*"),
                    SinkTrustRule("filesystem:workspace:*"),
                )
            )
        )
    with pytest.raises(ValueError, match="equal-priority overlapping"):
        AgentLibOSConfig(
            data_flow=DataFlowDefaults(
                sink_rules=(
                    SinkTrustRule("filesystem:workspace:report"),
                    SinkTrustRule("filesystem:workspace:report*"),
                )
            )
        )

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "data_flow:\n  sink_rules:\n    - pattern: 'jsonrpc:crm:*'\n"
        "      trust_level: trusted\n      max_sensitivity: secret\n",
        encoding="utf-8",
    )
    with pytest.raises(PydanticValidationError, match="identity_sha256"):
        load_config_file(invalid)


def test_store_revalidates_persisted_sink_trust_and_label_evidence() -> None:
    store = SQLiteStore(":memory:")
    try:
        trust = SinkTrustSpec(
            trust_id="persisted-trust",
            pattern="human:owner:terminal",
            trust_level="trusted",
            max_sensitivity="secret",
            generation=1,
            created_by="test",
            created_at=utc_now(),
        )
        store.register_sink_trust(trust)
        store.conn.execute(
            "UPDATE sink_trust_records SET spec_hash = ? WHERE trust_id = ?",
            ("0" * 64, trust.trust_id),
        )
        store.conn.commit()

        with pytest.raises(ValidationError, match="invalid persisted sink trust record"):
            store.get_sink_trust(trust.trust_id)
    finally:
        store.close()
