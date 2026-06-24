from __future__ import annotations
import agent_libos.substrate.local as local_substrate
import os
import pytest
import tempfile
from pathlib import Path
from typing import Any
from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import AgentLibOSConfig, ShellCommandRule, ShellDefaults
from agent_libos.models import AuthorityRisk, AuthorityRule, CapabilityEffect, CapabilityRight, ExternalEffectClassification, ExternalEffectRollbackClass, ExternalEffectRollbackStatus, HumanRequestStatus
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, HumanResponseRequired, ValidationError
from agent_libos.substrate import CommandResult, LocalClockProvider, LocalFilesystemProvider, LocalHumanProvider, LocalResourceProviderSubstrate

class TestShellPrimitive:
    def setup_method(self) -> None:
        self._temp_dirs: list[tempfile.TemporaryDirectory[str]] = []

    def teardown_method(self) -> None:
        while self._temp_dirs:
            self._temp_dirs.pop().cleanup()

    def test_whitelist_policy_auto_allows_exact_safe_command(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='run safe shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            result = runtime.shell.run(pid, ['git', 'status', '--short'], timeout=2.0)
            assert result.stdout == 'ok\n'
            assert provider.calls == [(['git', 'status', '--short'], 2.0)]
            assert 'primitive.shell.run' in self._audit_actions(runtime)
        finally:
            runtime.close()

    def test_shell_run_records_intent_and_links_result_audit(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='audit shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')

            runtime.shell.run(pid, ['git', 'status', '--short'], timeout=2.0)

            records = runtime.audit.trace()
            intent = next(record for record in records if record.action == 'primitive.shell.intent')
            result = next(record for record in records if record.action == 'primitive.shell.run')
            event = next(event for event in runtime.events.list(target='shell:git') if event.payload.get('operation') == 'run')
            assert result.parent_record_id == intent.record_id
            assert result.correlation_id == intent.record_id
            assert event.correlation_id == intent.record_id
            assert intent.decision['sandbox_profile']['operation'] == 'shell.run'
        finally:
            runtime.close()

    def test_denied_shell_command_does_not_record_execution_intent(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='denied shell audit')
            runtime.shell.grant_policy(pid, runtime.config.shell.always_allow_level, issued_by='test')

            with pytest.raises(CapabilityDenied):
                runtime.shell.run(pid, ['rm', '-rf', 'agent_outputs'])

            assert provider.calls == []
            assert 'primitive.shell.intent' not in self._audit_actions(runtime)
        finally:
            runtime.close()

    def test_unlisted_command_requires_approval_and_consumes_once(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='run unlisted shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['git', 'show', '--stat'])
            pending = runtime.human.pending()
            assert len(pending) == 1
            assert pending[0].payload['context']['argv'] == ['git', 'show', '--stat']
            assert runtime.human.drain_terminal_queue(auto_approve=True)[0].status == HumanRequestStatus.APPROVED
            allowed = runtime.shell.run(pid, ['git', 'show', '--stat'])
            assert allowed.stdout == 'ok\n'
            assert provider.calls == [(['git', 'show', '--stat'], runtime.config.tools.shell_timeout_s)]
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['git', 'show', '--stat'])
        finally:
            runtime.close()

    def test_human_shell_approval_is_bound_to_exact_argv(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='argv-bound shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['git', 'show', '--stat'])
            runtime.human.drain_terminal_queue(auto_approve=True)
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['git', 'push'])
            assert provider.calls == []
            allowed = runtime.shell.run(pid, ['git', 'show', '--stat'])
            assert allowed.stdout == 'ok\n'
            assert provider.calls == [(['git', 'show', '--stat'], runtime.config.tools.shell_timeout_s)]
        finally:
            runtime.close()

    def test_git_diff_with_output_requires_approval(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='diff output')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['git', 'diff', '--output=patch.diff'])
            context = runtime.human.pending()[0].payload['context']
            assert context['rule_id'] != 'shell.low.git'
        finally:
            runtime.close()

    def test_high_risk_network_command_requires_approval_with_rule_profile(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='network shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['curl', 'https://example.test'])
            context = runtime.human.pending()[0].payload['context']
            assert context['risk'] == 'high'
            assert context['rule_id'] == 'shell.network.default'
            assert context['sandbox_profile']['operation'] == 'shell.run'
            assert context['sandbox_profile']['restrictions']['network']
        finally:
            runtime.close()

    def test_destructive_command_is_denied_even_with_always_allow_policy(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='destructive shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.always_allow_level, issued_by='test')
            with pytest.raises(CapabilityDenied):
                runtime.shell.run(pid, ['rm', '-rf', 'agent_outputs'])
            assert provider.calls == []
        finally:
            runtime.close()

    def test_medium_project_code_execution_requires_approval(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='pytest shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['pytest'])
            context = runtime.human.pending()[0].payload['context']
            assert context['risk'] == 'medium'
            assert context['rule_id'] == 'shell.medium.pytest'
        finally:
            runtime.close()

    def test_compileall_requires_approval_because_it_writes_artifacts(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='compileall shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['python', '-m', 'compileall', 'agent_libos'])
            context = runtime.human.pending()[0].payload['context']
            assert context['rule_id'] == 'shell.high.compileall'
            assert context['risk'] == 'high'
            assert provider.calls == []
        finally:
            runtime.close()

    def test_pytest_collect_only_requires_approval_because_collection_executes_project_code(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='pytest collect shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['pytest', '--collect-only'])
            context = runtime.human.pending()[0].payload['context']
            assert context['rule_id'] == 'shell.medium.pytest'
            assert context['risk'] == 'medium'
            assert provider.calls == []
        finally:
            runtime.close()

    def test_shell_denies_workspace_external_path_arguments_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            outside_path = Path(outside)
            (outside_path / 'outside_mod.py').write_text('x = 1\n', encoding='utf-8')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(workspace))
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='outside path shell')
                runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
                with pytest.raises(CapabilityDenied):
                    runtime.shell.run(pid, ['python', '-m', 'compileall', str(outside_path)], timeout=5.0)
                assert runtime.human.pending() == []
                assert not any(outside_path.rglob('__pycache__'))
            finally:
                runtime.close()

    def test_always_allow_policy_passes_shell_syntax_payload_as_literal_argv(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            payload = 'safe$(printf${IFS}__INJECTED__)'
            pid = runtime.process.spawn(image='review-agent:v0', goal='explicit broad shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.always_allow_level, issued_by='test')
            result = runtime.shell.run(pid, ['python', '-m', 'compileall', payload])
            assert result.stdout == 'ok\n'
            assert provider.calls == [(['python', '-m', 'compileall', payload], runtime.config.tools.shell_timeout_s)]
        finally:
            runtime.close()

    def test_always_deny_shell_policy_overrides_exact_command_grant(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='deny shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.always_deny_level, issued_by='test')
            runtime.capability.grant(pid, 'shell:git', [CapabilityRight.EXECUTE], issued_by='test')
            with pytest.raises(CapabilityDenied):
                runtime.shell.run(pid, ['git', 'status', '--short'])
        finally:
            runtime.close()

    def test_non_shell_typed_wildcard_capability_does_not_enable_shell(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='typed wildcard shell')
            runtime.capability.grant(pid, 'filesystem:*', [CapabilityRight.EXECUTE], issued_by='test')
            with pytest.raises(CapabilityDenied):
                runtime.shell.run(pid, ['git', 'status', '--short'])
        finally:
            runtime.close()

    def test_shell_authorization_uses_capability_resource_matching(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='typed shell matching')
            runtime.capability.grant(pid, 'shell:gi:*', [CapabilityRight.EXECUTE], issued_by='test')
            with pytest.raises(CapabilityDenied):
                runtime.shell.run(pid, ['git', 'status'])
            runtime.capability.grant(pid, 'shell:git:*', [CapabilityRight.EXECUTE], issued_by='test')
            result = runtime.shell.run(pid, ['git', 'status'])
            with pytest.raises(CapabilityDenied):
                runtime.shell.run(pid, ['git', 'push'])
            assert result.stdout == 'ok\n'
            assert provider.calls == [(['git', 'status'], runtime.config.tools.shell_timeout_s)]
        finally:
            runtime.close()

    def test_request_permission_shell_command_class_is_rule_constrained(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='request git shell')
            runtime.capability.grant(pid, 'human:owner', [CapabilityRight.WRITE], issued_by='test')
            with pytest.raises(HumanResponseRequired):
                runtime.tools.call(pid, 'request_permission', {'resource': 'shell:git', 'rights': ['execute'], 'reason': 'inspect git state'})
            pending = runtime.human.pending()[0]
            rules = pending.payload['requested_permission']['constraints']['authority_rules']
            runtime.human.drain_terminal_queue(auto_policy=CapabilityManager.ALWAYS_ALLOW)
            request = runtime.tools.call(pid, 'request_permission', {'resource': 'shell:git', 'rights': ['execute'], 'reason': 'inspect git state'})
            allowed = runtime.shell.run(pid, ['git', 'status'])
            with pytest.raises(CapabilityDenied):
                runtime.shell.run(pid, ['git', 'push', 'origin'])
            assert request.ok
            assert request.payload['status'] == 'approved'
            assert any(rule['rule_id'] == 'shell.git.deny.push' for rule in rules)
            assert allowed.stdout == 'ok\n'
            assert provider.calls == [(['git', 'status'], runtime.config.tools.shell_timeout_s)]
        finally:
            runtime.close()

    def test_bare_shell_wildcard_allow_does_not_bypass_shell_policy(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='bare wildcard')
            runtime.capability.grant(pid, 'shell:*', [CapabilityRight.EXECUTE], issued_by='test')
            with pytest.raises(CapabilityDenied):
                runtime.shell.run(pid, ['git', 'status'])
        finally:
            runtime.close()

    def test_blacklist_policy_asks_for_nested_shell_interpreter(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='blacklist shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.blocklist_ask_else_auto_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['env', 'bash', '-c', 'echo unsafe'])
            request = runtime.human.pending()[0]
            assert request.payload['context']['matched_rule'] == ['bash']
        finally:
            runtime.close()

    def test_path_qualified_whitelist_command_does_not_match_bare_rule(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='path bypass')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['./git', 'status', '--short'])
        finally:
            runtime.close()

    def test_shell_tool_uses_runtime_shell_primitive(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='run shell tool')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            result = runtime.tools.call(pid, 'run_shell_command', {'argv': ['git', 'status', '--short']})
            assert result.ok, result.error
            assert result.payload['stdout'] == 'ok\n'
            assert provider.calls == [(['git', 'status', '--short'], runtime.config.tools.shell_timeout_s)]
        finally:
            runtime.close()

    def test_shell_primitive_truncates_output_before_tool_layer(self) -> None:
        config = AgentLibOSConfig(shell=ShellDefaults(max_stdout_chars=3, max_stderr_chars=2, whitelist=(ShellCommandRule(('tool',)),), blacklist=()))
        runtime, provider = self._runtime_with_config(config)
        provider.stdout = 'abcdef'
        provider.stderr = 'wxyz'
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='bounded shell')
            runtime.shell.grant_policy(pid, config.shell.allowlist_auto_else_ask_level, issued_by='test')
            result = runtime.shell.run(pid, ['tool'])
            tool_result = runtime.tools.call(pid, 'run_shell_command', {'argv': ['tool'], 'max_stdout_chars': 10, 'max_stderr_chars': 10})
            assert result.stdout == 'abc'
            assert result.stderr == 'wx'
            assert result.stdout_truncated
            assert result.stderr_truncated
            assert tool_result.payload['stdout'] == 'abc'
            assert tool_result.payload['stdout_truncated']
            assert tool_result.payload['stderr'] == 'wx'
            assert tool_result.payload['stderr_truncated']
        finally:
            runtime.close()

    def test_shell_primitive_enforces_timeout_limit_without_tool_schema(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='bounded timeout')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(ValidationError):
                runtime.shell.run(pid, ['git', 'status', '--short'], timeout=0)
            with pytest.raises(ValidationError):
                runtime.shell.run(pid, ['git', 'status', '--short'], timeout=float('nan'))
            with pytest.raises(ValidationError):
                runtime.shell.run(pid, ['git', 'status', '--short'], timeout=runtime.config.shell.timeout_hard_limit_s + 1)
        finally:
            runtime.close()

    def _runtime_with_fake_shell(self) -> tuple[Runtime, 'FakeShellProvider']:
        temp_dir = tempfile.TemporaryDirectory()
        self._temp_dirs.append(temp_dir)
        provider = FakeShellProvider()
        substrate = RecordingShellSubstrate(temp_dir.name, provider)
        runtime = Runtime.open('local', substrate=substrate)
        runtime.substrate.human.output_sink = lambda _message: None
        return (runtime, provider)

    def _runtime_with_config(self, config: AgentLibOSConfig) -> tuple[Runtime, 'FakeShellProvider']:
        temp_dir = tempfile.TemporaryDirectory()
        self._temp_dirs.append(temp_dir)
        provider = FakeShellProvider()
        substrate = RecordingShellSubstrate(temp_dir.name, provider)
        runtime = Runtime.open('local', substrate=substrate, config=config)
        runtime.substrate.human.output_sink = lambda _message: None
        return (runtime, provider)

    def _audit_actions(self, runtime: Runtime) -> list[str]:
        return [record.action for record in runtime.audit.trace()]

class TestShellMatcher:
    def setup_method(self) -> None:
        self._temp_dirs: list[tempfile.TemporaryDirectory[str]] = []

    def teardown_method(self) -> None:
        while self._temp_dirs:
            self._temp_dirs.pop().cleanup()

    def test_custom_exact_whitelist_rule_does_not_prefix_match_extra_args(self) -> None:
        config = AgentLibOSConfig(shell=ShellDefaults(whitelist=(ShellCommandRule(('tool', 'safe')),), blacklist=()))
        runtime, _provider = self._runtime_with_config(config)
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='exact shell')
            runtime.shell.grant_policy(pid, config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['tool', 'safe', '--extra'])
        finally:
            runtime.close()

    def test_custom_allow_rule_with_shell_syntax_payload_requires_approval(self) -> None:
        config = AgentLibOSConfig(
            shell=ShellDefaults(
                rules=(
                    AuthorityRule(
                        rule_id='custom.tool.inspect',
                        operation='shell.run',
                        effect=CapabilityEffect.ALLOW,
                        risk=AuthorityRisk.LOW,
                        conditions={'argv': ['tool', 'inspect'], 'match': 'prefix'},
                    ),
                ),
                whitelist=(),
                blacklist=(),
            )
        )
        runtime, provider = self._runtime_with_config(config)
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='custom allow syntax')
            runtime.shell.grant_policy(pid, config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['tool', 'inspect', 'safe$(printf${IFS}__INJECTED__)'])
            context = runtime.human.pending()[0].payload['context']
            assert context['rule_id'] == 'shell.syntax.default'
            assert provider.calls == []
        finally:
            runtime.close()

    def test_custom_allow_rule_cannot_override_script_interpreter_risk(self) -> None:
        config = AgentLibOSConfig(
            shell=ShellDefaults(
                rules=(
                    AuthorityRule(
                        rule_id='custom.bash.allow',
                        operation='shell.run',
                        effect=CapabilityEffect.ALLOW,
                        risk=AuthorityRisk.LOW,
                        conditions={'argv': ['bash'], 'match': 'prefix'},
                    ),
                ),
                whitelist=(),
                blacklist=(),
            )
        )
        runtime, provider = self._runtime_with_config(config)
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='custom bash allow')
            runtime.shell.grant_policy(pid, config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['bash', '-c', 'printf __INJECTED__'])
            context = runtime.human.pending()[0].payload['context']
            assert context['rule_id'] == 'shell.interpreter.default'
            assert provider.calls == []
        finally:
            runtime.close()

    def test_local_shell_provider_uses_argv_without_shell_interpreter(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        class FakePopen:
            pid = 12345
            returncode = 0

            def __init__(self, argv: list[str], **kwargs: Any) -> None:
                captured['argv'] = argv
                captured['kwargs'] = kwargs
                kwargs['stdout'].write(b'ok\n')
                kwargs['stdout'].flush()

            def poll(self) -> int:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                return 'ok\n', ''

        class FakePsutilProcess:
            def __init__(self, pid: int) -> None:
                self.pid = pid

            def children(self, recursive: bool = False) -> list[Any]:
                return []

            def cpu_times(self) -> Any:
                return type('Times', (), {'user': 0.0, 'system': 0.0})()

            def memory_info(self) -> Any:
                return type('Mem', (), {'rss': 0})()

        def fake_popen(argv: list[str], **kwargs: Any) -> FakePopen:
            captured['argv'] = argv
            captured['kwargs'] = kwargs
            return FakePopen(argv, **kwargs)

        monkeypatch.setattr(local_substrate.subprocess, 'Popen', fake_popen)
        monkeypatch.setattr(local_substrate.psutil, 'Process', FakePsutilProcess)
        provider = local_substrate.LocalShellProvider(tmp_path)
        result = provider.run(['echo', 'safe$(printf${IFS}__INJECTED__)'])

        assert result.stdout == 'ok\n'
        assert Path(captured['argv'][0]).name == 'echo'
        assert captured['argv'][1:] == ['safe$(printf${IFS}__INJECTED__)']
        assert captured['kwargs']['stdout'] is not local_substrate.subprocess.PIPE
        assert captured['kwargs']['stderr'] is not local_substrate.subprocess.PIPE
        assert captured['kwargs']['shell'] is False
        assert 'OPENAI_API_KEY' not in captured['kwargs']['env']

    def test_local_shell_provider_does_not_execute_workspace_path_hijack(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}
        fake_git = tmp_path / 'git'
        fake_git.write_text('#!/bin/sh\nprintf hijacked\n', encoding='utf-8')
        fake_git.chmod(0o755)

        class FakePopen:
            pid = 12345
            returncode = 0

            def __init__(self, argv: list[str], **kwargs: Any) -> None:
                captured['argv'] = argv
                captured['kwargs'] = kwargs
                kwargs['stdout'].write(b'ok\n')
                kwargs['stdout'].flush()

            def poll(self) -> int:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        class FakePsutilProcess:
            def __init__(self, pid: int) -> None:
                self.pid = pid

            def children(self, recursive: bool = False) -> list[Any]:
                return []

            def cpu_times(self) -> Any:
                return type('Times', (), {'user': 0.0, 'system': 0.0})()

            def memory_info(self) -> Any:
                return type('Mem', (), {'rss': 0})()

        def fake_popen(argv: list[str], **kwargs: Any) -> FakePopen:
            return FakePopen(argv, **kwargs)

        searched_paths: list[str] = []

        def fake_which(command: str, *, path: str | None = None) -> str | None:
            searched_paths.append(path or '')
            assert command == 'git'
            return '/usr/bin/git'

        monkeypatch.setenv('PATH', os.pathsep.join([str(tmp_path), '/usr/bin']))
        monkeypatch.setattr(local_substrate.shutil, 'which', fake_which)
        monkeypatch.setattr(local_substrate.subprocess, 'Popen', fake_popen)
        monkeypatch.setattr(local_substrate.psutil, 'Process', FakePsutilProcess)
        provider = local_substrate.LocalShellProvider(tmp_path)

        result = provider.run(['git', 'status'])

        assert result.stdout == 'ok\n'
        assert captured['argv'][0] == '/usr/bin/git'
        assert str(tmp_path) not in searched_paths[0].split(os.pathsep)
        assert str(tmp_path) not in captured['kwargs']['env']['PATH'].split(os.pathsep)

    def _runtime_with_config(self, config: AgentLibOSConfig) -> tuple[Runtime, 'FakeShellProvider']:
        temp_dir = tempfile.TemporaryDirectory()
        self._temp_dirs.append(temp_dir)
        provider = FakeShellProvider()
        substrate = RecordingShellSubstrate(temp_dir.name, provider)
        runtime = Runtime.open('local', substrate=substrate, config=config)
        runtime.substrate.human.output_sink = lambda _message: None
        return (runtime, provider)

class RecordingShellSubstrate(LocalResourceProviderSubstrate):

    def __init__(self, root: str, shell: 'FakeShellProvider'):
        super().__init__(Path(root).resolve())
        self.workspace_root = Path(root).resolve()
        self.workspace_display = str(self.workspace_root)
        self.filesystem = LocalFilesystemProvider(root)
        self.clock = LocalClockProvider()
        self.shell = shell
        self.human = LocalHumanProvider()

class FakeShellProvider:

    def __init__(self):
        self.calls: list[tuple[list[str], float]] = []
        self.stdout = 'ok\n'
        self.stderr = ''

    def run(
        self,
        argv: list[str],
        *,
        timeout: float = 30.0,
        cwd: str | None = None,
        limits: Any | None = None,
        stdout_limit_chars: int | None = None,
        stderr_limit_chars: int | None = None,
    ) -> CommandResult:
        self.calls.append((list(argv), timeout))
        return CommandResult(argv=list(argv), returncode=0, stdout=self.stdout, stderr=self.stderr)

    def classify_external_effect(self, operation: str, context: dict[str, Any], result: Any) -> ExternalEffectClassification:
        return ExternalEffectClassification(rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE, rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED, state_mutation=True, information_flow=True, metadata={'operation': operation})
