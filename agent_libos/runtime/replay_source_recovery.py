from __future__ import annotations

"""Bounded startup retention of existing READ authority for private history."""

from collections.abc import Callable, Iterator
from typing import Any

from agent_libos.config import AgentLibOSConfig
from agent_libos.llm.replay import LLMReplayService
from agent_libos.models import CapabilityRight, DataFlowContext, ObjectLifecycleState, ProcessStatus
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.object_payload import object_payload_sha256


class LLMReplaySourceRecovery:
    """Keep only already-authorized reads when volatile source bytes are lost.

    This is a Host startup collaborator, never a process capability issuer.
    Private payloads stay in their sidecar and no grant is recreated. The
    existing Object sweep independently CAS-rechecks and narrows selected rows.
    """

    def __init__(
        self,
        unit_of_work: Any,
        *,
        config: AgentLibOSConfig,
        capabilities: Any,
        profile_snapshot: Callable[[str], Any],
        excluded_run_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.processes = unit_of_work.processes
        self.objects = unit_of_work.objects
        self.authority = unit_of_work.authority
        self.config = config
        self.capabilities = capabilities
        self.profile_snapshot = profile_snapshot
        self.excluded_run_ids = excluded_run_ids
        self.service = LLMReplayService(self.processes, max_bytes=config.llm.responses_replay_max_bytes, max_turns=config.llm.responses_replay_max_turns)
        self._retained: dict[str, dict[str, frozenset[str]]] | None = None

    def _contexts(self) -> Iterator[tuple[str, DataFlowContext]]:
        if not self.config.llm.persist_full_io:
            return
        after = None
        while True:
            pids = self.processes.list_llm_replay_recovery_pids(after_pid=after, limit=self.config.llm.call_record_list_limit)
            if not pids:
                return
            for pid in pids:
                contexts = [context for _owner, context in self._owner_contexts(pid)]
                if contexts:
                    yield pid, DataFlowContext.aggregate(contexts)
            after = pids[-1]

    def _owner_contexts(self, pid: str) -> Iterator[tuple[str, DataFlowContext]]:
        process = self.processes.get_process(pid)
        if (
            process is None
            or process.status in {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
            or process.task_run_id in self.excluded_run_ids
        ):
            # Terminal owners need no volatile read grants. TaskRun preflight
            # already classified invalid Runs; recovery must isolate them.
            return
        head = self.processes.get_llm_replay_head(pid)
        pending = self.processes.get_llm_pending_action(pid)
        prepared = {}
        if pending is not None and pending["status"] == "pending":
            prepared = pending.get("action") or {}
        if not isinstance(prepared, dict):
            # The pending-action validator owns malformed action isolation.
            # It cannot establish authority for any retained replay sources.
            return
        reference = prepared.get("responses_replay_request")
        if head is None and reference is None:
            return
        profile = self.profile_snapshot(process.llm_profile_id or self.config.llm.default_profile_id)
        if not profile.policy.responses_replay:
            return
        expected = (pid, process.task_run_id, profile.identity_sha256, profile.policy.model, self.processes.get_llm_context_generation(pid))
        current = self.service.load_current(pid)
        if current is not None:
            _head, turn, payload = current
            actual = (turn.pid, turn.run_id, turn.provider_fingerprint, turn.model, turn.context_generation)
            if actual == expected:
                yield pid, DataFlowContext.from_dict(payload["flow_context"])
        if reference is None:
            return
        if not isinstance(reference, dict):
            raise ValidationError("private replay recovery pending request reference is invalid")
        if prepared.get("kind") != "llm_release_request" or prepared.get("pid") != pid:
            raise ValidationError("private replay recovery pending request owner is invalid")
        request = self.service.load_request(
            reference, pid=pid, run_id=process.task_run_id,
            provider_fingerprint=profile.identity_sha256, model=profile.policy.model,
            context_generation=expected[-1],
        )
        yield pid, request.flow_context

    def preflight(self) -> None:
        """Validate every candidate private payload before any recovery writes."""

        retained: dict[tuple[str, str], set[str]] = {}
        count = 0
        maximum = self.config.llm.call_record_hard_limit * self.config.capability.list_limit
        for pid, context in self._contexts():
            oids = self._eligible_sources(pid, context)
            for oid, cap_ids in self._read_policies(pid, oids).items():
                retained[(pid, oid)] = cap_ids
                count += len(cap_ids)
            if count > maximum:
                raise ValidationError("private replay recovery READ authority exceeds its configured bound")
        self._remove_unretained_parent_chains(retained, {oid for _pid, oid in retained})
        indexed: dict[str, dict[str, frozenset[str]]] = {}
        for (pid, oid), cap_ids in retained.items():
            if cap_ids:
                indexed.setdefault(oid, {})[pid] = frozenset(cap_ids)
        self._retained = indexed

    def _eligible_sources(self, pid: str, context: DataFlowContext) -> set[str]:
        eligible: set[str] = set()
        for reference in context.source_refs:
            state = self.objects.get_persisted_object_state(reference.oid)
            if state is None or state.lifecycle_state is not ObjectLifecycleState.LIVE or state.version != reference.version:
                continue
            obj = self.objects.get_object(reference.oid)
            if obj is not None and object_payload_sha256(obj.payload) != reference.content_sha256:
                continue
            decision = self.capabilities.authorize(pid, f"object:{reference.oid}", CapabilityRight.READ, audit=False)
            if decision.allowed:
                eligible.add(reference.oid)
        return eligible

    def _read_policies(self, pid: str, oids: set[str]) -> dict[str, set[str]]:
        selected: dict[str, set[str]] = {}
        after = None
        while oids:
            caps = self.authority.query_capabilities(pid, active_only=True, after_cap_id=after, limit=self.config.capability.list_limit)
            if not caps:
                break
            for cap in caps:
                oid = cap.resource.removeprefix("object:") if cap.resource.startswith("object:") else None
                if oid in oids and CapabilityRight.READ in cap.rights and not self.capabilities.is_expired(cap):
                    selected.setdefault(oid, set()).add(cap.cap_id)
            after = caps[-1].cap_id
        return selected

    def retained_read_capabilities(self, oids: tuple[str, ...]) -> dict[tuple[str, str], frozenset[str]]:
        """Resolve only one Object sweep page, without mutating authority."""

        if self._retained is None:
            raise ValidationError("private replay source recovery requires completed preflight")
        retained: dict[tuple[str, str], frozenset[str]] = {}
        for oid in oids:
            for pid, cap_ids in self._retained.get(oid, {}).items():
                resource = f"object:{oid}"
                decision = self.capabilities.authorize(pid, resource, CapabilityRight.READ, audit=False)
                if not decision.allowed:
                    continue
                retained[(pid, oid)] = cap_ids
        return retained

    def _remove_unretained_parent_chains(self, retained: dict[tuple[str, str], set[str]], oids: set[str]) -> None:
        # An exact-Object ancestor swept in this page must independently have
        # qualified for retention. Never invent an ancestor or extend retention
        # to another subject merely because a child has private LLM history.
        changed = True
        while changed:
            changed = False
            all_ids = set().union(*retained.values()) if retained else set()
            for cap_ids in retained.values():
                for cap_id in tuple(cap_ids):
                    cap = self.authority.get_capability(cap_id)
                    parent = self.authority.get_capability(cap.parent_cap_id) if cap is not None and cap.parent_cap_id else None
                    if parent is not None and parent.resource.startswith("object:") and parent.resource.removeprefix("object:") in oids and parent.cap_id not in all_ids:
                        cap_ids.remove(cap_id)
                        changed = True
