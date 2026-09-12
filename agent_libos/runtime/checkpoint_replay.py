from __future__ import annotations

"""Local checkpoint authority boundary for private Responses continuations."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_libos.config import AgentLibOSConfig
from agent_libos.llm.replay import LLMReplayService
from agent_libos.models import Capability, CapabilityEffect, CapabilityRight, CapabilityStatus
from agent_libos.models.data_flow import DataFlowContext, DataSourceRef
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.models.snapshot import LocalReplayReference
from agent_libos.runtime.checkpoint_continuation import CheckpointContinuationAdapter
from agent_libos.utils.object_payload import object_payload_sha256
from agent_libos.utils.serde import loads


class CheckpointReplayAdapter:
    """Validate immutable local refs before checkpoint publication effects.

    The adapter is bound by Host composition only. Nothing in a checkpoint
    chooses the provider resolver or supplies transport configuration.
    """

    def __init__(
        self,
        unit_of_work: Any,
        *,
        config: AgentLibOSConfig,
        capabilities: Any,
        data_flow: Any,
        filesystem: Any,
        profile_snapshot: Callable[[str], Any],
    ) -> None:
        self.processes = unit_of_work.processes
        self.objects = unit_of_work.objects
        self.config = config
        self.capabilities = capabilities
        self.data_flow = data_flow
        self.filesystem = filesystem
        self.profile_snapshot = profile_snapshot
        self.service = LLMReplayService(
            self.processes,
            max_bytes=config.llm.responses_replay_max_bytes,
            max_turns=config.llm.responses_replay_max_turns,
        )
        self.continuations = CheckpointContinuationAdapter(
            processes=self.processes, config=config, profile_scope=self._scope,
            validate_sources=self._validate_sources, fork_context=self._fork_context,
            validate_fork_sources=self._validate_fork_sources,
        )

    def capture_provider_continuations(self, pids: Sequence[str]) -> dict[str, dict[str, Any]]:
        return self.continuations.capture(pids)

    def capture(self, pids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not self.config.llm.persist_full_io:
            return {}
        return self.service.capture_checkpoint_refs(pids)

    def _entries(self, snapshot: Mapping[str, Any]) -> dict[str, LocalReplayReference]:
        return {
            pid: LocalReplayReference.from_mapping(value)
            for pid, value in snapshot.get("responses_replay_refs", {}).items()
        }

    def _payload(self, reference: LocalReplayReference) -> dict[str, Any]:
        if not self.config.llm.persist_full_io:
            raise ValidationError("checkpoint private Responses replay is disabled by payload retention")
        turn = self.processes.get_llm_replay_turn(reference.turn_id)
        if turn is None:
            raise ValidationError("checkpoint private Responses replay payload is missing")
        if any(getattr(turn, name) != value for name, value in reference.to_mapping().items()):
            raise ValidationError("checkpoint private Responses replay scope or integrity changed")
        return self.service.validate_checkpoint_turn(turn)

    def _scope(self, row: Mapping[str, Any]) -> tuple[str, str]:
        profile_id = row.get("llm_profile_id")
        if not isinstance(profile_id, str) or not profile_id:
            raise ValidationError("checkpoint private Responses replay has no LLM profile binding")
        snapshot = self.profile_snapshot(profile_id)
        return snapshot.identity_sha256, snapshot.policy.model

    def validate(self, snapshot: Mapping[str, Any], *, remapped: Mapping[str, Any] | None = None) -> None:
        self.continuations.validate(snapshot, remapped=remapped)
        rows = {str(row["pid"]): row for row in snapshot["rows"]["processes"]}
        for pid, reference in self._entries(snapshot).items():
            if pid not in rows or reference.pid != pid:
                raise ValidationError("checkpoint private Responses replay is outside its process scope")
            payload = self._payload(reference)
            if self._scope(rows[pid]) != (reference.provider_fingerprint, reference.model):
                raise ValidationError("checkpoint private Responses replay provider scope changed")
            context = DataFlowContext.from_dict(payload["flow_context"])
            self._validate_sources(pid, context, snapshot=snapshot)
            if remapped is not None:
                target_pid = remapped["pid_map"][pid]
                self._validate_fork_sources(
                    target_pid,
                    context,
                    remapped=remapped,
                )

    def _source_resource(self, reference: DataSourceRef) -> str:
        if not reference.oid.startswith(self.data_flow.FILE_BINDING_SOURCE_REF_PREFIX):
            return f"object:{reference.oid}"
        binding_id = reference.oid.removeprefix(self.data_flow.FILE_BINDING_SOURCE_REF_PREFIX)
        binding = self.data_flow.store.get_file_label_binding_by_id(binding_id)
        if binding is None:
            raise ValidationError("checkpoint replay source file binding is missing")
        return self.filesystem.resource_for(binding.normalized_path)

    def _validate_sources(self, pid: str, context: DataFlowContext, *, snapshot: Mapping[str, Any]) -> None:
        object_rows = {str(row["oid"]): row for row in snapshot["rows"]["objects"]}
        payloads = snapshot.get("object_payloads", {})
        self.data_flow.validate_replay_sources(
            pid, context,
            file_resource_resolver=self.filesystem.resource_for,
            captured_objects={oid: (object_rows[oid]["version"], payload) for oid, payload in payloads.items() if oid in object_rows},
        )

    @staticmethod
    def _capability(row: Mapping[str, Any]) -> Capability:
        return Capability(
            cap_id=row["cap_id"], subject=row["subject"], resource=row["resource"],
            rights=set(loads(row["rights_json"])), constraints=loads(row["constraints_json"]),
            issued_by=row["issued_by"], issued_at=row["issued_at"], expires_at=row["expires_at"],
            delegable=row["delegable"], revocable=row["revocable"],
            effect=CapabilityEffect(row["effect"]), issuer_cap_id=row["issuer_cap_id"],
            parent_cap_id=row["parent_cap_id"], delegation_depth=row["delegation_depth"],
            max_delegation_depth=row["max_delegation_depth"], uses_remaining=row["uses_remaining"],
            status=CapabilityStatus(row["status"]), metadata=loads(row["metadata_json"]),
        )

    def _fork_context(self, context: DataFlowContext, remapped: Mapping[str, Any]) -> DataFlowContext:
        references: list[DataSourceRef] = []
        objects = remapped.get("object_map", {})
        rows = {str(row["oid"]): row for row in remapped["rows"]["objects"]}
        for reference in context.source_refs:
            target_oid = objects.get(reference.oid, reference.oid)
            if target_oid != reference.oid:
                payload = remapped["object_payloads"].get(target_oid)
                if target_oid not in rows or target_oid not in remapped["object_payloads"]:
                    raise ValidationError("checkpoint replay source cannot be cloned into the fork")
                # Native encrypted items are immutable. A remapped Object
                # whose content changes cannot stand in for its original bytes.
                digest = object_payload_sha256(payload)
                if digest != reference.content_sha256:
                    raise ValidationError("checkpoint replay fork changed source Object content")
                references.append(DataSourceRef(target_oid, rows[target_oid]["version"], digest))
            else:
                references.append(reference)
        return DataFlowContext(labels=context.labels, source_refs=tuple(references))

    def _validate_fork_sources(self, pid: str, context: DataFlowContext, *, remapped: Mapping[str, Any]) -> None:
        projected = self._fork_context(context, remapped)
        capabilities = [self._capability(row) for row in remapped["rows"]["capabilities"]]
        for reference in projected.source_refs:
            decision = self.capabilities.authorize_matching_capabilities(
                pid, self._source_resource(reference), CapabilityRight.READ, capabilities,
            )
            if not decision.allowed:
                raise CapabilityDenied("checkpoint fork lacks authority to read private replay sources")

    def publish_restore(self, snapshot: Mapping[str, Any], *, current_pids: Sequence[str]) -> None:
        entries = self._entries(snapshot)
        for pid in set(current_pids) | set(snapshot["subtree_pids"]):
            if pid not in entries:
                self.processes.clear_llm_replay_head(pid)
        for pid, reference in entries.items():
            payload = self._payload(reference)
            self.service.rebind(
                turn_id=reference.turn_id, pid=pid, run_id=None,
                context_generation=self.processes.get_llm_context_generation(pid),
                provider_fingerprint=reference.provider_fingerprint, model=reference.model,
                flow_context=DataFlowContext.from_dict(payload["flow_context"]),
            )
        self.continuations.publish_restore(snapshot)

    def publish_fork(self, snapshot: Mapping[str, Any], *, remapped: Mapping[str, Any]) -> None:
        for pid, reference in self._entries(snapshot).items():
            payload = self._payload(reference)
            target_pid = remapped["pid_map"][pid]
            self.service.rebind(
                turn_id=reference.turn_id, pid=target_pid, run_id=None,
                context_generation=self.processes.get_llm_context_generation(target_pid),
                provider_fingerprint=reference.provider_fingerprint, model=reference.model,
                flow_context=self._fork_context(DataFlowContext.from_dict(payload["flow_context"]), remapped),
            )
        self.continuations.publish_fork(snapshot, remapped=remapped)
