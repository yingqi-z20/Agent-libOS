from __future__ import annotations

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    CapabilityRight,
    ChildProcessWait,
    HostResumeProcessWait,
    HumanRequestStatus,
    HumanProcessWait,
    KilledProcessOutcome,
    MessageProcessWait,
    PausedProcessWait,
    ProcessStatus,
    ToolProcessWait,
)
from agent_libos.models.exceptions import (
    ProcessError,
    ProcessMessageWaitRequired,
    ValidationError,
)


class TestTypedMessageAndResourceProcessState:
    def test_composed_runtime_injects_one_process_transition_service(self) -> None:
        runtime = Runtime.open("local")
        try:
            transitions = runtime.process_transitions
            assert runtime.process.transitions is transitions
            assert runtime.resources._transitions is transitions
            assert runtime.messages._transitions is transitions
            assert runtime.human._transitions is transitions
            assert runtime.scheduler._transitions is transitions
            assert runtime.checkpoint._process_transitions is transitions
            assert runtime.llm._process_transitions is transitions
        finally:
            runtime.close()

    def test_execution_completion_never_parses_status_message_as_outcome(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="typed execution completion",
            )
            token = runtime.store.claim_execution(pid, owner_id="typed-test")
            assert token is not None

            with pytest.raises(ValidationError, match="killed requires"):
                runtime.store.complete_execution(
                    token,
                    status=ProcessStatus.KILLED,
                    status_message="result_oid:obj_forged",
                )

            assert runtime.process.get(pid).status == ProcessStatus.RUNNING
            assert runtime.store.complete_execution(
                token,
                status=ProcessStatus.KILLED,
                outcome=KilledProcessOutcome(code="typed_completion"),
            )
            terminal = runtime.process.get(pid)
            assert terminal.outcome == KilledProcessOutcome(
                code="typed_completion"
            )
            assert terminal.status_message is None
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ("status", "wait_state"),
        (
            (
                ProcessStatus.WAITING_EVENT,
                ChildProcessWait(child_pid="pid_pending_child"),
            ),
            (
                ProcessStatus.WAITING_EVENT,
                MessageProcessWait(filters={"channel": "control"}),
            ),
            (
                ProcessStatus.WAITING_HUMAN,
                HumanProcessWait(request_ids=("hreq_pending",)),
            ),
            (
                ProcessStatus.WAITING_TOOL,
                ToolProcessWait(operation_id="op_pending"),
            ),
            (ProcessStatus.PAUSED, PausedProcessWait()),
            (
                ProcessStatus.PAUSED,
                HostResumeProcessWait(reason_oid="obj_host_resume"),
            ),
        ),
        ids=("child", "message", "human", "tool", "paused", "host-resume"),
    )
    def test_exec_rejects_active_typed_wait_before_publication(
        self,
        status: ProcessStatus,
        wait_state: object,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="typed wait must survive rejected exec",
            )
            runnable = runtime.process.get(pid)
            waiting = runtime.process_transitions.transition(
                pid,
                status,
                expected_revision=runnable.revision,
                expected_status=ProcessStatus.RUNNABLE,
                expected_state_generation=runnable.state_generation,
                wait_state=wait_state,  # type: ignore[arg-type]
            )
            publication_ids = {
                item["publication_id"]
                for item in runtime.store.list_runtime_publications()
            }

            with pytest.raises(ValidationError, match="active typed wait"):
                runtime.exec_process(pid, "base-agent:v0")

            unchanged = runtime.process.get(pid)
            assert unchanged.status == waiting.status
            assert unchanged.wait_state == waiting.wait_state
            assert unchanged.outcome == waiting.outcome
            assert unchanged.state_generation == waiting.state_generation
            assert unchanged.revision == waiting.revision
            assert {
                item["publication_id"]
                for item in runtime.store.list_runtime_publications()
            } == publication_ids
        finally:
            runtime.close()

    def test_message_wait_and_wakeup_use_typed_state_generation(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="typed message wait")

            with pytest.raises(ProcessMessageWaitRequired):
                runtime.messages.receive(
                    pid,
                    block=True,
                    channel="control",
                    correlation_id="job-1",
                )

            waiting = runtime.process.get(pid)
            assert waiting.status == ProcessStatus.WAITING_EVENT
            assert isinstance(waiting.wait_state, MessageProcessWait)
            assert waiting.wait_state.filters["channel"] == "control"
            assert waiting.wait_state.filters["correlation_id"] == "job-1"
            assert waiting.status_message is not None
            assert waiting.status_message.startswith("waiting_message:")
            wait_generation = waiting.state_generation

            runtime.messages.post(
                sender="test",
                recipient_pid=pid,
                channel="noise",
                correlation_id="job-1",
                subject="not a match",
            )
            still_waiting = runtime.process.get(pid)
            assert still_waiting.status == ProcessStatus.WAITING_EVENT
            assert still_waiting.wait_state == waiting.wait_state
            assert still_waiting.state_generation == wait_generation

            runtime.messages.post(
                sender="test",
                recipient_pid=pid,
                channel="control",
                correlation_id="job-1",
                subject="wake",
            )
            resumed = runtime.process.get(pid)
            assert resumed.status == ProcessStatus.RUNNABLE
            assert resumed.wait_state is None
            assert resumed.outcome is None
            assert resumed.status_message is None
            assert resumed.state_generation == wait_generation + 1
        finally:
            runtime.close()

    def test_blocking_human_query_cannot_replace_message_wait(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="message owner must keep its wait",
            )
            with pytest.raises(ProcessMessageWaitRequired):
                runtime.messages.receive(pid, block=True, channel="control")
            waiting = runtime.process.get(pid)

            with pytest.raises(ProcessError, match="owned by another condition"):
                runtime.human.query(
                    pid,
                    "owner",
                    {"type": "question", "question": "overwrite message wait?"},
                    blocking=True,
                )

            assert runtime.process.get(pid) == waiting
            assert runtime.human.list(pid) == []
        finally:
            runtime.close()

    @pytest.mark.parametrize("paged", [False, True], ids=("receive", "receive-page"))
    def test_blocking_message_receive_cannot_replace_human_wait(
        self,
        paged: bool,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="Human owner must keep its wait",
            )
            request_id = runtime.human.query(
                pid,
                "owner",
                {"type": "question", "question": "hold this wait?"},
                blocking=True,
            )
            waiting = runtime.process.get(pid)

            with pytest.raises(ProcessError, match="owned by another condition"):
                if paged:
                    runtime.messages.receive_page(pid, block=True, channel="control")
                else:
                    runtime.messages.receive(pid, block=True, channel="control")

            assert runtime.process.get(pid) == waiting
            assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        finally:
            runtime.close()

    def test_child_wait_cannot_replace_human_wait(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="Human owner must keep its wait",
            )
            runtime.capability.grant(
                parent,
                "process:spawn",
                [CapabilityRight.WRITE],
                issued_by="test",
            )
            child = runtime.process.spawn_child(parent, goal="remain nonterminal")
            request_id = runtime.human.query(
                parent,
                "owner",
                {"type": "question", "question": "hold this wait?"},
                blocking=True,
            )
            waiting = runtime.process.get(parent)

            with pytest.raises(ProcessError, match="owned by another condition"):
                runtime.process.wait(parent, child)

            assert runtime.process.get(parent) == waiting
            assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        finally:
            runtime.close()

    @pytest.mark.parametrize("paged", [False, True], ids=("receive", "receive-page"))
    def test_message_receive_cannot_bypass_host_resume_gate(
        self,
        paged: bool,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="Host resume gate must remain owned",
            )
            runnable = runtime.process.get(pid)
            gated = runtime.process_transitions.transition(
                pid,
                ProcessStatus.PAUSED,
                expected_revision=runnable.revision,
                expected_status=runnable.status,
                expected_state_generation=runnable.state_generation,
                wait_state=HostResumeProcessWait(reason_oid="obj_manual_recovery"),
            )

            with pytest.raises(ProcessError, match="owned by another condition"):
                if paged:
                    runtime.messages.receive_page(pid, block=True, channel="control")
                else:
                    runtime.messages.receive(pid, block=True, channel="control")

            assert runtime.process.get(pid) == gated
        finally:
            runtime.close()

    def test_message_wakeup_ignores_forged_compatibility_projection(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="typed child wait")
            child = runtime.process.spawn_child(parent, goal="unrelated child")
            process = runtime.process.get(parent)
            waiting = runtime.messages._transitions.transition(
                parent,
                ProcessStatus.WAITING_EVENT,
                expected_revision=process.revision,
                expected_status=process.status,
                wait_state=ChildProcessWait(child_pid=child),
            )
            runtime.store.patch_process(
                parent,
                {
                    "status_message": (
                        'waiting_message:{"channel":"control","correlation_id":null,'
                        '"kind":null,"message_ids":null,"reply_to":null,"sender":null}'
                    )
                },
                expected_revision=waiting.revision,
            )

            runtime.messages.post(
                sender="test",
                recipient_pid=parent,
                channel="control",
                subject="must not wake a child waiter",
            )

            persisted = runtime.process.get(parent)
            assert persisted.status == ProcessStatus.WAITING_EVENT
            assert persisted.wait_state == ChildProcessWait(child_pid=child)
            assert persisted.state_generation == waiting.state_generation
        finally:
            runtime.close()

    def test_resource_kill_persists_typed_outcome_and_wakes_only_typed_child_wait(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="typed parent wait")
            matching_child = runtime.process.spawn_child(parent, goal="matching child")
            other_child = runtime.process.spawn_child(parent, goal="other child")
            process = runtime.process.get(parent)
            waiting = runtime.resources._transitions.transition(
                parent,
                ProcessStatus.WAITING_EVENT,
                expected_revision=process.revision,
                expected_status=process.status,
                wait_state=ChildProcessWait(child_pid=matching_child),
            )
            forged = runtime.store.patch_process(
                parent,
                {"status_message": f"waiting for {other_child}"},
                expected_revision=waiting.revision,
            )

            runtime.resources.kill_if_exceeded(other_child, reason="other exhausted")
            still_waiting = runtime.process.get(parent)
            assert still_waiting.status == ProcessStatus.WAITING_EVENT
            assert still_waiting.wait_state == ChildProcessWait(child_pid=matching_child)
            assert still_waiting.state_generation == forged.state_generation

            runtime.resources.kill_if_exceeded(matching_child, reason="matching exhausted")
            resumed = runtime.process.get(parent)
            killed = runtime.process.get(matching_child)
            assert resumed.status == ProcessStatus.RUNNABLE
            assert resumed.wait_state is None
            assert resumed.status_message is None
            assert resumed.state_generation == forged.state_generation + 1
            assert killed.status == ProcessStatus.KILLED
            assert killed.wait_state is None
            assert killed.status_message == "matching exhausted"
            assert killed.outcome == KilledProcessOutcome(code="resource_limit_exceeded")
        finally:
            runtime.close()
