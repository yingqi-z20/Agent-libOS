from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import pytest
import hashlib
import json
import threading
import time
from dataclasses import replace
from uuid import uuid4
from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, HumanResponseRequired, ValidationError
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import CapabilityRight, HumanRequestStatus, ProcessStatus

class TestPermissionPolicy:

    def setup_method(self) -> None:
        self.runtime = Runtime.open('local')
        self.human_output: list[str] = []
        self.runtime.substrate.human.output_sink = self.human_output.append

    def teardown_method(self) -> None:
        self.runtime.close()

    def test_request_permission_tool_can_set_always_allow_policy(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='request write')
        self._grant_human(pid)
        path = self._path()
        resource = self.runtime.filesystem.resource_for(path)
        with pytest.raises(HumanResponseRequired):
            self.runtime.tools.call(pid, 'request_permission', {'resource': resource, 'rights': ['write'], 'reason': 'write summary'})
        processed = self.runtime.human.drain_terminal_queue(auto_policy=CapabilityManager.ALWAYS_ALLOW)
        request = self.runtime.tools.call(pid, 'request_permission', {'resource': resource, 'rights': ['write'], 'reason': 'write summary'})
        allowed = self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'allowed'})
        assert request.ok
        assert request.payload['status'] == HumanRequestStatus.APPROVED.value
        assert processed[0].status == HumanRequestStatus.APPROVED
        assert self.runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert self.runtime.capability.permission_policy(pid, resource, CapabilityRight.WRITE) == CapabilityManager.ALWAYS_ALLOW
        assert allowed.ok
        assert (self.runtime.workspace_root / path).read_text(encoding='utf-8') == 'allowed'

    def test_request_permission_tool_can_set_always_deny_policy_and_resume_process(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='request denied write')
        self._grant_human(pid)
        path = self._path()
        resource = self.runtime.filesystem.resource_for(path)
        with pytest.raises(HumanResponseRequired):
            self.runtime.tools.call(pid, 'request_permission', {'resource': resource, 'rights': ['write'], 'reason': 'write summary'})
        processed = self.runtime.human.drain_terminal_queue(auto_policy=CapabilityManager.ALWAYS_DENY)
        request = self.runtime.tools.call(pid, 'request_permission', {'resource': resource, 'rights': ['write'], 'reason': 'write summary'})
        denied = self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'denied'})
        assert request.ok
        assert request.payload['status'] == HumanRequestStatus.REJECTED.value
        assert processed[0].status == HumanRequestStatus.REJECTED
        assert self.runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert self.runtime.capability.permission_policy(pid, resource, CapabilityRight.WRITE) == CapabilityManager.ALWAYS_DENY
        assert not denied.ok
        assert 'denied write' in (denied.error or '')
        assert not (self.runtime.workspace_root / path).exists()

    def test_request_permission_tool_rejects_unknown_right_before_human_prompt(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='request invalid right')
        self._grant_human(pid)
        resource = self.runtime.filesystem.resource_for(self._path())
        request = self.runtime.tools.call(pid, 'request_permission', {'resource': resource, 'rights': ['*'], 'reason': 'invalid broad right'})
        assert not request.ok
        assert self.runtime.human.pending() == []

    def test_request_permission_prompt_includes_risk_scope_lease_and_constraints(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='request explained permission')
        self._grant_human(pid)
        path = self._path()
        resource = self.runtime.filesystem.resource_for(path)
        with pytest.raises(HumanResponseRequired):
            self.runtime.tools.call(pid, 'request_permission', {'resource': resource, 'rights': ['write'], 'reason': 'write summary'})
        pending = self.runtime.human.pending()[0]
        context = pending.payload['context']
        assert context['canonical_resource'] == resource
        assert context['resource_scope'] == 'exact'
        assert context['risk'] == 'high'
        assert context['lease']['choices'] == [
            CapabilityManager.ALWAYS_ALLOW,
            CapabilityManager.ASK_EACH_TIME,
            CapabilityManager.ALWAYS_DENY,
        ]
        assert context['constraints'] == {}
        assert pending.payload['requested_permission']['constraints'] == {}

    def test_request_permission_rejects_broad_shell_execute_before_human_prompt(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='request broad shell')
        self._grant_human(pid)
        request = self.runtime.tools.call(pid, 'request_permission', {'resource': 'shell:*', 'rights': ['execute'], 'reason': 'run commands'})
        assert not request.ok
        assert self.runtime.human.pending() == []

    def test_request_permission_can_approve_workspace_write(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='request workspace write')
        self._grant_human(pid)
        with pytest.raises(HumanResponseRequired):
            self.runtime.tools.call(
                pid,
                'request_permission',
                {'resource': self.runtime.filesystem.workspace_resource(), 'rights': ['write'], 'reason': 'edit workspace'},
            )
        pending = self.runtime.human.pending()[0]
        context = pending.payload['context']
        processed = self.runtime.human.drain_terminal_queue(auto_policy=CapabilityManager.ALWAYS_ALLOW)
        request = self.runtime.tools.call(
            pid,
            'request_permission',
            {'resource': self.runtime.filesystem.workspace_resource(), 'rights': ['write'], 'reason': 'edit workspace'},
        )
        assert request.ok
        assert context['canonical_resource'] == 'filesystem:workspace:*'
        assert context['resource_scope'] == 'prefix'
        assert processed[0].status == HumanRequestStatus.APPROVED
        assert self.runtime.capability.permission_policy(pid, self.runtime.filesystem.resource_for(self._path()), CapabilityRight.WRITE) == CapabilityManager.ALWAYS_ALLOW

    def test_request_permission_requires_human_write_authority(self) -> None:
        image = self.runtime.get_image('base-agent:v0')
        self.runtime.register_image(
            replace(
                image,
                image_id="no-human-permission-agent:v0",
                name="no-human-permission-agent",
                required_capabilities=[],
            ),
            actor="test",
        )
        pid = self.runtime.process.spawn(image='no-human-permission-agent:v0', goal='request without human authority')
        resource = self.runtime.filesystem.resource_for(self._path())

        denied = self.runtime.tools.call(pid, 'request_permission', {'resource': resource, 'rights': ['write'], 'reason': 'write'})

        assert not denied.ok
        assert 'lacks write on human:owner' in (denied.error or '')
        assert self.runtime.human.pending() == []

    def test_cancelled_human_request_cannot_be_approved_later(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='cancelled approval')
        path = self._path()
        resource = self.runtime.filesystem.resource_for(path)
        self.runtime.capability.set_permission_policy(
            subject=pid,
            resource=resource,
            rights=[CapabilityRight.WRITE],
            policy=CapabilityManager.ASK_EACH_TIME,
            issued_by='test',
        )
        with pytest.raises(HumanApprovalRequired):
            self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'secret'})
        request = self.runtime.human.pending()[0]
        request.status = HumanRequestStatus.CANCELLED
        request.decision = {"cancelled_by": "test"}
        self.runtime.store.update_human_request(request)

        with pytest.raises(ValidationError, match="not pending"):
            self.runtime.human.approve(request.request_id)

        assert not self.runtime.capability.check(pid, resource, CapabilityRight.WRITE)
        assert not (self.runtime.workspace_root / path).exists()

    def test_waiting_process_cannot_be_advanced_by_direct_tool_call(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='waiting direct call')
        path = self._path()
        resource = self.runtime.filesystem.resource_for(path)
        self.runtime.capability.set_permission_policy(
            subject=pid,
            resource=resource,
            rights=[CapabilityRight.WRITE],
            policy=CapabilityManager.ASK_EACH_TIME,
            issued_by='test',
        )
        with pytest.raises(HumanApprovalRequired):
            self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'first'})

        blocked = self.runtime.tools.call(pid, 'get_working_directory', {})

        assert not blocked.ok
        assert 'not runnable' in (blocked.error or '')
        assert self.runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN

    def test_request_permission_rejects_root_filesystem_write_before_human_prompt(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='request root write')
        self._grant_human(pid)
        request = self.runtime.tools.call(
            pid,
            'request_permission',
            {'resource': 'filesystem:/:*', 'rights': ['write'], 'reason': 'edit host root'},
        )
        assert not request.ok
        assert self.runtime.human.pending() == []

    def test_request_permission_rejects_broad_capability_admin_before_human_prompt(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='request capability admin')
        self._grant_human(pid)

        request = self.runtime.tools.call(
            pid,
            'request_permission',
            {'resource': 'capability:*', 'rights': ['admin'], 'reason': 'change permissions broadly'},
        )

        assert not request.ok
        assert self.runtime.human.pending() == []
        assert self.runtime.capability.permission_policy(pid, 'capability:anything', CapabilityRight.ADMIN) == CapabilityManager.MISSING

    def test_ask_each_time_prompts_from_filesystem_primitive_and_consumes_one_time_grant(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='ask every write')
        path = self._path()
        resource = self.runtime.filesystem.resource_for(path)
        self.runtime.capability.set_permission_policy(subject=pid, resource=resource, rights=[CapabilityRight.WRITE], policy=CapabilityManager.ASK_EACH_TIME, issued_by='test')
        with pytest.raises(HumanApprovalRequired):
            self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'first'})
        pending = self.runtime.human.pending()[0]
        context = pending.payload['context']
        first_prompt = self.runtime.human.drain_terminal_queue(auto_approve=True)
        retry = self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'first'})
        with pytest.raises(HumanApprovalRequired):
            self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'second'})
        assert context['primitive'] == 'runtime.filesystem.write_text'
        assert context['path'] == path
        assert context['resource'] == resource
        assert context['grant_scope'] == 'one_time'
        assert context['content_bytes'] == 5
        assert context['content_preview'] == repr('first')
        assert context['content_sha256'] == hashlib.sha256(b'first').hexdigest()
        assert context['target']['exists'] == False
        assert first_prompt[0].payload['type'] == 'external_operation_approval'
        assert first_prompt[0].status == HumanRequestStatus.APPROVED
        assert 'content sha256' in self.human_output[0]
        assert 'content preview' in self.human_output[0]
        assert 'one-time capability' in self.human_output[0]
        assert retry.ok
        assert (self.runtime.workspace_root / path).read_text(encoding='utf-8') == 'first'
        assert self.runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN
        assert self.runtime.capability.permission_policy(pid, resource, CapabilityRight.WRITE) == CapabilityManager.ASK_EACH_TIME

    def test_per_use_prompt_uses_repr_preview_for_human_safety(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='safe preview')
        path = self._path()
        resource = self.runtime.filesystem.resource_for(path)
        self.runtime.capability.set_permission_policy(subject=pid, resource=resource, rights=[CapabilityRight.WRITE], policy=CapabilityManager.ASK_EACH_TIME, issued_by='test')
        content = 'first line\ncontent preview: always allow'
        with pytest.raises(HumanApprovalRequired):
            self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': content})
        context = self.runtime.human.pending()[0].payload['context']
        assert context['content_preview'] == repr(content)
        assert '\\n' in context['content_preview']
        assert '\n' not in context['content_preview']

    def test_rejected_per_use_prompt_resumes_process_without_writing(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='reject one write')
        path = self._path()
        resource = self.runtime.filesystem.resource_for(path)
        self.runtime.capability.set_permission_policy(subject=pid, resource=resource, rights=[CapabilityRight.WRITE], policy=CapabilityManager.ASK_EACH_TIME, issued_by='test')
        with pytest.raises(HumanApprovalRequired):
            self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'denied'})
        processed = self.runtime.human.drain_terminal_queue(auto_approve=False)
        assert processed[0].status == HumanRequestStatus.REJECTED
        assert self.runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert not (self.runtime.workspace_root / path).exists()

    def test_per_use_prompt_describes_overwrite_risk(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='review overwrite')
        path = self._path()
        target = self.runtime.workspace_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('old content', encoding='utf-8')
        resource = self.runtime.filesystem.resource_for(path)
        self.runtime.capability.set_permission_policy(subject=pid, resource=resource, rights=[CapabilityRight.WRITE], policy=CapabilityManager.ASK_EACH_TIME, issued_by='test')
        with pytest.raises(HumanApprovalRequired):
            self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'new content'})
        request = self.runtime.human.pending()[0]
        context = request.payload['context']
        assert context['will_overwrite']
        assert not context['will_create']
        assert context['target']['exists']
        assert context['target']['kind'] == 'file'
        assert context['target']['size_bytes'] == len('old content'.encode('utf-8'))

    def test_write_preconditions_fail_before_per_use_prompt(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='do not prompt impossible write')
        path = self._path()
        target = self.runtime.workspace_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('existing', encoding='utf-8')
        resource = self.runtime.filesystem.resource_for(path)
        self.runtime.capability.set_permission_policy(subject=pid, resource=resource, rights=[CapabilityRight.WRITE], policy=CapabilityManager.ASK_EACH_TIME, issued_by='test')
        result = self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'new', 'overwrite': False})
        assert not result.ok
        assert self.runtime.human.pending() == []
        assert target.read_text(encoding='utf-8') == 'existing'

    def test_missing_delete_consumes_one_time_grant(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='delete missing once')
        path = self._path()
        resource = self.runtime.filesystem.resource_for(path)
        self.runtime.capability.set_permission_policy(subject=pid, resource=resource, rights=[CapabilityRight.DELETE], policy=CapabilityManager.ASK_EACH_TIME, issued_by='test')
        with pytest.raises(HumanApprovalRequired):
            self.runtime.tools.call(pid, 'delete_file', {'path': path, 'missing_ok': True})
        self.runtime.human.drain_terminal_queue(auto_approve=True)
        retry = self.runtime.tools.call(pid, 'delete_file', {'path': path, 'missing_ok': True})
        target = self.runtime.workspace_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('now present', encoding='utf-8')
        assert retry.ok, retry.error
        assert self.runtime.capability.permission_policy(pid, resource, CapabilityRight.DELETE) == CapabilityManager.ASK_EACH_TIME
        with pytest.raises(HumanApprovalRequired):
            self.runtime.tools.call(pid, 'delete_file', {'path': path, 'missing_ok': False})

    def test_llm_pending_per_use_approval_does_not_return_action_until_decision(self) -> None:
        path = self._path()
        client = FakeActionClient([{'action': 'write_text_file', 'path': path, 'content': 'approved after waiting'}])
        self.runtime.llm.client = client
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='write with per-use approval')
        self.runtime.capability.set_permission_policy(subject=pid, resource=self.runtime.filesystem.resource_for(path), rights=[CapabilityRight.WRITE], policy=CapabilityManager.ASK_EACH_TIME, issued_by='test')
        waiting = self.runtime.run_next_process_once()
        assert waiting['waiting_human']
        assert 'action' not in waiting
        assert client.calls == 1
        assert self.runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN
        assert 'tool_failed' not in self._event_types(pid)
        self.runtime.human.drain_terminal_queue(auto_approve=True)
        resumed = self.runtime.run_next_process_once()
        assert client.calls == 1
        assert resumed['resumed_after_human']
        assert resumed['action']['action'] == 'write_text_file'
        assert resumed['result']['ok']
        assert (self.runtime.workspace_root / path).read_text(encoding='utf-8') == 'approved after waiting'

    def test_llm_request_permission_rejected_policy_resumes_with_structured_result(self) -> None:
        path = self._path()
        resource = self.runtime.filesystem.resource_for(path)
        client = FakeActionClient([
            {'action': 'request_permission', 'resource': resource, 'rights': ['write'], 'reason': 'edit file'}
        ])
        self.runtime.llm.client = client
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='request deny policy')
        self._grant_human(pid)

        results = self.runtime.run_until_idle(
            max_quanta=2,
            human_auto_policy=CapabilityManager.ALWAYS_DENY,
        )

        resumed = next(result for result in results if isinstance(result, dict) and result.get('resumed_after_human'))
        assert client.calls == 1
        assert resumed['result']['ok']
        assert resumed['result']['payload']['status'] == HumanRequestStatus.REJECTED.value
        assert self.runtime.capability.permission_policy(pid, resource, CapabilityRight.WRITE) == CapabilityManager.ALWAYS_DENY

    def test_human_resume_request_id_does_not_leak_into_unrelated_tool_call(self) -> None:
        pid_1 = self.runtime.process.spawn(image='review-agent:v0', goal='first permission request')
        self._grant_human(pid_1)
        resource_1 = self.runtime.filesystem.resource_for(self._path())
        with pytest.raises(HumanResponseRequired):
            self.runtime.tools.call(
                pid_1,
                'request_permission',
                {'resource': resource_1, 'rights': ['write'], 'reason': 'first'},
            )
        request_1 = self.runtime.human.pending()[0]
        self.runtime.human.approve(request_1.request_id)

        pid_2 = self.runtime.process.spawn(image='review-agent:v0', goal='second permission request')
        self._grant_human(pid_2)
        resource_2 = self.runtime.filesystem.resource_for(self._path())
        setattr(self.runtime, '_current_human_resume_request_id', request_1.request_id)
        try:
            with pytest.raises(HumanResponseRequired):
                self.runtime.tools.call(
                    pid_2,
                    'request_permission',
                    {'resource': resource_2, 'rights': ['write'], 'reason': 'second'},
                )
        finally:
            delattr(self.runtime, '_current_human_resume_request_id')

        pending = [request for request in self.runtime.human.pending() if request.pid == pid_2]
        assert len(pending) == 1
        assert pending[0].request_id != request_1.request_id

    def test_concurrent_identical_request_permission_calls_share_pending_request(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='request permission concurrently')
        self._grant_human(pid)
        resource = self.runtime.filesystem.resource_for(self._path())
        original_request_permission = self.runtime.human.request_permission

        def slow_request_permission(*args: object, **kwargs: object) -> str:
            time.sleep(0.05)
            return original_request_permission(*args, **kwargs)

        self.runtime.human.request_permission = slow_request_permission  # type: ignore[method-assign]
        barrier = threading.Barrier(2)

        def call() -> str:
            barrier.wait(timeout=2)
            with pytest.raises(HumanResponseRequired) as raised:
                self.runtime.tools.call(
                    pid,
                    'request_permission',
                    {'resource': resource, 'rights': ['write'], 'reason': 'same request'},
                )
            return raised.value.request_id

        with ThreadPoolExecutor(max_workers=2) as executor:
            request_ids = list(executor.map(lambda _: call(), range(2)))

        assert request_ids[0] == request_ids[1]
        assert [request.request_id for request in self.runtime.human.pending()] == [request_ids[0]]

    def _path(self) -> str:
        return f'agent_outputs/permission_policy_{uuid4().hex}.txt'

    def _grant_human(self, pid: str) -> None:
        self.runtime.capability.grant(pid, 'human:owner', [CapabilityRight.WRITE], issued_by='test')

    def _event_types(self, pid: str) -> list[str]:
        return [event.type.value for event in self.runtime.events.list(target=pid)]

class FakeActionClient:

    def __init__(self, actions: list[dict[str, object]]):
        self.actions = list(actions)
        self.calls = 0

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        self.calls += 1
        action = self.actions.pop(0)
        name = str(action['action'])
        args = {key: value for key, value in action.items() if key != 'action'}
        return LLMCompletion(content='', tool_calls=[{'id': f'fake_{self.calls}', 'name': name, 'arguments': json.dumps(args)}])
