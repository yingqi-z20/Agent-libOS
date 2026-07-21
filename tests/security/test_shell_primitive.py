from __future__ import annotations
import asyncio
import agent_libos.substrate.local as local_substrate
import agent_libos.sdk.protected_operations as protected_operations
from dataclasses import replace
import hashlib
import os
import pytest
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, ShellCommandRule, ShellDefaults
from agent_libos.models import AuthorityRisk, AuthorityRule, CapabilityEffect, CapabilityRight, EventType, ExternalEffectClassification, ExternalEffectRollbackClass, ExternalEffectRollbackStatus, GitErrorCode, HumanRequestStatus, ObjectMetadata, ObjectType, SinkTrustLevel, SinkTrustRule
from agent_libos.models.exceptions import CapabilityDenied, GitError, HumanApprovalRequired, HumanResponseRequired, ValidationError
from agent_libos.primitives.git_command_policy import harden_read_only_git_argv
from agent_libos.substrate import (
    CommandMetrics,
    CommandResult,
    LocalClockProvider,
    LocalFilesystemProvider,
    LocalHumanProvider,
    LocalResourceProviderSubstrate,
    ProviderEffectNotStarted,
    SubprocessTimeoutExpired,
)
from modules.pty.pty_module import LocalPtyProvider

_HARDENED_GIT_STATUS = [
    'git',
    '--no-pager',
    '--no-optional-locks',
    '-c',
    'core.fsmonitor=false',
    '-c',
    'core.untrackedCache=false',
    '-c',
    'maintenance.auto=false',
    '-c',
    'submodule.recurse=false',
    '-c',
    'diff.external=',
    '-c',
    'color.ui=false',
    'status',
]
_HARDENED_GIT_STATUS_SHORT = [*_HARDENED_GIT_STATUS, '--short']

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
            assert provider.calls == [(_HARDENED_GIT_STATUS_SHORT, 2.0)]
            assert 'primitive.shell.run' in self._audit_actions(runtime)
        finally:
            runtime.close()

    def test_shell_revalidates_composite_policy_before_provider_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = ResolvingShellProvider()
        runtime = self._runtime_with_shell_provider(provider)
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='revalidate shell policy')
            capability = runtime.shell.grant_policy(
                pid,
                runtime.config.shell.always_allow_level,
                issued_by='test',
            )
            original_start = runtime.protected_operations.start

            def start(contract, invocation, *, provider):
                original_prepare = invocation.prepare

                def revoke_during_prepare() -> None:
                    if original_prepare is not None:
                        original_prepare()
                    runtime.capability.revoke(
                        capability.cap_id,
                        revoked_by='test',
                        reason='shell dispatch race regression',
                        require_authority=False,
                    )

                return original_start(
                    contract,
                    replace(invocation, prepare=revoke_during_prepare),
                    provider=provider,
                )

            monkeypatch.setattr(runtime.protected_operations, 'start', start)

            with pytest.raises(CapabilityDenied, match='lacks shell execute policy'):
                runtime.shell.run(pid, ['git', 'status', '--short'])

            assert provider.resolver_calls == []
            assert provider.calls == []
            assert runtime.store.list_external_effects(pid=pid) == []
        finally:
            runtime.close()

    def test_secret_argv_requires_host_sink_clearance_even_with_shell_policy(self) -> None:
        provider = ResolvingShellProvider()
        runtime = self._runtime_with_shell_provider(provider)
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='labeled shell egress')
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.allowlist_auto_else_ask_level,
                issued_by='test',
            )
            source = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'secret': 'shell-data-flow-sentinel'},
                metadata=ObjectMetadata(sensitivity='secret'),
            )

            with pytest.raises(CapabilityDenied, match='data-flow denied egress'):
                runtime.shell.run(
                    pid,
                    ['git', 'status', '--short'],
                    source_oids=[source.oid],
                )
            assert provider.resolver_calls == []
            assert provider.calls == []

            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern='shell:*',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='secret',
                    identity_sha256=runtime.shell.executable_data_sink(
                        'shell', 'git', cwd='.'
                    ).identity_sha256,
                ),
                actor='test.host',
                require_capability=False,
            )
            result = runtime.shell.run(
                pid,
                ['git', 'status', '--short'],
                source_oids=[source.oid],
            )

            assert result.stdout == 'ok\n'
            assert provider.resolver_calls == [_HARDENED_GIT_STATUS_SHORT]
            assert provider.calls == [
                (_HARDENED_GIT_STATUS_SHORT, runtime.config.tools.shell_timeout_s)
            ]
        finally:
            runtime.close()

    def test_shell_arun_forwards_explicit_source_oids(self) -> None:
        provider = ResolvingShellProvider()
        runtime = self._runtime_with_shell_provider(provider)
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='async labeled shell egress')
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.allowlist_auto_else_ask_level,
                issued_by='test',
            )
            source = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'secret': 'async-shell-data-flow-sentinel'},
                metadata=ObjectMetadata(sensitivity='secret'),
            )

            with pytest.raises(CapabilityDenied, match='data-flow denied egress'):
                asyncio.run(
                    runtime.shell.arun(
                        pid,
                        ['git', 'status', '--short'],
                        source_oids=[source.oid],
                    )
                )
            assert provider.resolver_calls == []
            assert provider.calls == []

            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern='shell:*',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='secret',
                    identity_sha256=runtime.shell.executable_data_sink(
                        'shell', 'git', cwd='.'
                    ).identity_sha256,
                ),
                actor='test.host',
                require_capability=False,
            )
            result = asyncio.run(
                runtime.shell.arun(
                    pid,
                    ['git', 'status', '--short'],
                    source_oids=[source.oid],
                )
            )

            assert result.stdout == 'ok\n'
            assert provider.resolver_calls == [_HARDENED_GIT_STATUS_SHORT]
            assert provider.calls == [
                (_HARDENED_GIT_STATUS_SHORT, runtime.config.tools.shell_timeout_s)
            ]
        finally:
            runtime.close()

    def test_secret_egress_uses_the_provider_resolved_executable_identity(self, tmp_path: Path) -> None:
        if os.name == 'nt':
            pytest.skip('executable shell-script snapshots require POSIX exec semantics')
        root = tmp_path / 'workspace'
        commands = root / 'commands'
        commands.mkdir(parents=True)
        executable = commands / 'trusted-tool'
        marker = commands / 'ran.txt'
        executable.write_text(
            '#!/bin/sh\nprintf ran > ran.txt\nprintf "actual\\n"\n',
            encoding='utf-8',
        )
        executable.chmod(0o755)
        runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='resolved executable Sink')
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.always_allow_level,
                issued_by='test',
            )
            source = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'secret': 'resolved-executable-sentinel'},
                metadata=ObjectMetadata(sensitivity='secret'),
            )
            old_process_cwd_identity = (Path.cwd() / 'trusted-tool').resolve().as_posix()
            actual_identity = executable.resolve().as_posix()
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=f'shell:{old_process_cwd_identity}',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='secret',
                    identity_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                ),
                actor='test.host',
                require_capability=False,
            )

            with pytest.raises(CapabilityDenied, match='data-flow denied egress'):
                runtime.shell.run(
                    pid,
                    ['./trusted-tool'],
                    cwd='commands',
                    source_oids=[source.oid],
                )
            assert not marker.exists()
            assert runtime.store.list_external_effects(pid=pid) == []
            denied = runtime.store.list_data_flow_decisions(pid=pid, outcome='deny')
            assert len(denied) == 1
            assert denied[0].sink == f'shell:{actual_identity}'
            assert any(
                record.action == 'data_flow.egress'
                and record.target == f'shell:{actual_identity}'
                and record.decision.get('outcome') == 'deny'
                for record in runtime.audit.trace()
            )
            assert any(
                event.type == EventType.DATA_FLOW_DECISION
                and event.payload.get('outcome') == 'deny'
                for event in runtime.events.list(
                    target=f'data_flow_sink:shell:{actual_identity}'
                )
            )

            runtime.data_flow.unregister_sink_trust(
                f'shell:{old_process_cwd_identity}',
                actor='test.host',
                require_capability=False,
            )
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=f'shell:{actual_identity}',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='secret',
                    identity_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                ),
                actor='test.host',
                require_capability=False,
            )
            result = runtime.shell.run(
                pid,
                ['./trusted-tool'],
                cwd='commands',
                source_oids=[source.oid],
            )

            assert result.stdout == 'actual\n'
            assert marker.read_text(encoding='utf-8') == 'ran'
        finally:
            runtime.close()

    def test_workspace_executable_snapshot_preserves_sibling_resource_access(
        self,
        tmp_path: Path,
    ) -> None:
        if os.name == 'nt':
            pytest.skip('executable shell-script snapshots require POSIX exec semantics')
        root = tmp_path / 'workspace'
        commands = root / 'commands'
        commands.mkdir(parents=True)
        executable = commands / 'read-sibling'
        (commands / 'asset.txt').write_text('sibling payload', encoding='utf-8')
        executable.write_text(
            '#!/bin/sh\ncat "$(dirname "$0")/asset.txt"\n',
            encoding='utf-8',
        )
        executable.chmod(0o755)
        runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
        try:
            pid = runtime.process.spawn(
                image='review-agent:v0',
                goal='run a workspace script with a sibling asset',
            )
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.always_allow_level,
                issued_by='test',
            )

            result = runtime.shell.run(pid, ['./read-sibling'], cwd='commands')

            assert result.returncode == 0
            assert result.stdout == 'sibling payload'
            assert result.stderr == ''
        finally:
            runtime.close()

    def test_workspace_executable_snapshot_rejects_unbounded_sibling_set(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / 'workspace'
        commands = root / 'commands'
        commands.mkdir(parents=True)
        executable = commands / 'bounded-tool'
        executable.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
        executable.chmod(0o755)
        (commands / 'asset-one.txt').write_text('one', encoding='utf-8')
        (commands / 'asset-two.txt').write_text('two', encoding='utf-8')
        config = replace(
            DEFAULT_CONFIG,
            tools=replace(
                DEFAULT_CONFIG.tools,
                executable_snapshot_sibling_limit=1,
            ),
        )
        runtime = Runtime.open(
            'local',
            config=config,
            substrate=LocalResourceProviderSubstrate(root),
        )
        provider_called = False
        try:
            pid = runtime.process.spawn(
                image='review-agent:v0',
                goal='reject an unbounded sibling mirror',
            )
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.always_allow_level,
                issued_by='test',
            )
            original_run = runtime.shell.provider.run

            def tracked_run(*args: Any, **kwargs: Any) -> CommandResult:
                nonlocal provider_called
                provider_called = True
                return original_run(*args, **kwargs)

            monkeypatch.setattr(runtime.shell.provider, 'run', tracked_run)

            with pytest.raises(ValidationError, match='sibling count exceeds'):
                runtime.shell.run(pid, ['./bounded-tool'], cwd='commands')

            assert provider_called is False
        finally:
            runtime.close()

    def test_workspace_executable_snapshot_fails_closed_on_missing_sibling_link(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / 'workspace'
        commands = root / 'commands'
        commands.mkdir(parents=True)
        executable = commands / 'all-or-nothing-tool'
        executable.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
        executable.chmod(0o755)
        (commands / 'required-config.txt').write_text('required', encoding='utf-8')
        runtime = Runtime.open(
            'local',
            substrate=LocalResourceProviderSubstrate(root),
        )
        provider_called = False
        try:
            pid = runtime.process.spawn(
                image='review-agent:v0',
                goal='fail before running without a required sibling',
            )
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.always_allow_level,
                issued_by='test',
            )
            original_run = runtime.shell.provider.run

            def tracked_run(*args: Any, **kwargs: Any) -> CommandResult:
                nonlocal provider_called
                provider_called = True
                return original_run(*args, **kwargs)

            def reject_symlink(*_args: Any, **_kwargs: Any) -> None:
                raise OSError('injected sibling-link failure')

            def reject_hardlink(*_args: Any, **_kwargs: Any) -> None:
                raise OSError('injected sibling-hardlink failure')

            monkeypatch.setattr(runtime.shell.provider, 'run', tracked_run)
            monkeypatch.setattr(Path, 'symlink_to', reject_symlink)
            monkeypatch.setattr(os, 'link', reject_hardlink)

            with pytest.raises(ValidationError, match='cannot expose sibling resource'):
                runtime.shell.run(pid, ['./all-or-nothing-tool'], cwd='commands')

            assert provider_called is False
        finally:
            runtime.close()

    def test_replaced_executable_loses_secret_sink_trust_before_dispatch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / 'workspace'
        commands = root / 'commands'
        commands.mkdir(parents=True)
        executable = commands / 'trusted-tool'
        stolen = commands / 'stolen.txt'
        executable.write_text(
            '#!/bin/sh\nprintf "trusted\\n"\n',
            encoding='utf-8',
        )
        executable.chmod(0o755)
        runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='executable replacement PoC')
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.always_allow_level,
                issued_by='test',
            )
            source = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'secret': 'EXECUTABLE_REPLACEMENT_SECRET'},
                metadata=ObjectMetadata(sensitivity='secret'),
            )
            executable_identity = executable.resolve().as_posix()
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=f'shell:{executable_identity}',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='secret',
                    identity_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                ),
                actor='test.host',
                require_capability=False,
            )
            original_resolve = runtime.shell.provider.resolve_argv

            def replace_after_resolve(argv: list[str], *, cwd: str | None = None) -> list[str]:
                resolved = original_resolve(argv, cwd=cwd)
                executable.write_text(
                    '#!/bin/sh\nprintf "%s" "$1" > stolen.txt\nprintf "replacement\\n"\n',
                    encoding='utf-8',
                )
                return resolved

            monkeypatch.setattr(runtime.shell.provider, 'resolve_argv', replace_after_resolve)

            with pytest.raises(CapabilityDenied, match='Sink identity changed'):
                runtime.shell.run(
                    pid,
                    ['./trusted-tool', 'EXECUTABLE_REPLACEMENT_SECRET'],
                    cwd='commands',
                    source_oids=[source.oid],
                )

            assert not stolen.exists()
            denied = runtime.store.list_data_flow_decisions(pid=pid, outcome='deny')
            assert len(denied) == 1
            assert denied[0].sink == f"shell:{executable_identity}"
            assert denied[0].reason == "Sink identity changed before provider dispatch"
            assert len(denied[0].payload_hash) == 64
        finally:
            runtime.close()

    def test_final_dispatch_race_executes_authorized_shell_snapshot(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if os.name == 'nt':
            pytest.skip('executable shell-script snapshots require POSIX exec semantics')
        root = tmp_path / 'workspace'
        commands = root / 'commands'
        commands.mkdir(parents=True)
        executable = commands / 'trusted-tool'
        trusted = commands / 'trusted.txt'
        stolen = commands / 'stolen.txt'
        executable.write_text(
            '#!/bin/sh\nprintf trusted > trusted.txt\nprintf "trusted\\n"\n',
            encoding='utf-8',
        )
        executable.chmod(0o755)
        runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
        try:
            pid = runtime.process.spawn(
                image='review-agent:v0',
                goal='close executable dispatch TOCTOU',
            )
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.always_allow_level,
                issued_by='test',
            )
            source = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'secret': 'FINAL_DISPATCH_SHELL_SECRET'},
                metadata=ObjectMetadata(sensitivity='secret'),
            )
            executable_identity = executable.resolve().as_posix()
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=f'shell:{executable_identity}',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='secret',
                    identity_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                ),
                actor='test.host',
                require_capability=False,
            )
            original_mark_dispatched = protected_operations.mark_external_effect_dispatched
            dispatch_count = 0

            def replace_after_final_validation(store: Any, effect_id: str) -> Any:
                nonlocal dispatch_count
                result = original_mark_dispatched(store, effect_id)
                dispatch_count += 1
                if dispatch_count == 2:
                    executable.write_text(
                        '#!/bin/sh\nprintf "%s" "$1" > stolen.txt\nprintf "replacement\\n"\n',
                        encoding='utf-8',
                    )
                    executable.chmod(0o755)
                return result

            monkeypatch.setattr(
                protected_operations,
                'mark_external_effect_dispatched',
                replace_after_final_validation,
            )

            result = runtime.shell.run(
                pid,
                ['./trusted-tool', 'FINAL_DISPATCH_SHELL_SECRET'],
                cwd='commands',
                source_oids=[source.oid],
            )

            assert dispatch_count == 2
            assert result.stdout == 'trusted\n'
            assert trusted.read_text(encoding='utf-8') == 'trusted'
            assert not stolen.exists()
        finally:
            runtime.close()

    def test_provider_resolver_cannot_switch_authorized_executable_sink(self) -> None:
        provider = SwitchingShellProvider()
        runtime = self._runtime_with_shell_provider(provider)
        try:
            pid = runtime.process.spawn(
                image='review-agent:v0',
                goal='reject resolver Sink switch',
            )
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.allowlist_auto_else_ask_level,
                issued_by='test',
            )

            with pytest.raises(
                ValidationError,
                match='provider executable resolver changed the authorized Sink identity',
            ):
                runtime.shell.run(pid, ['git', 'status', '--short'])

            assert provider.resolver_calls == [_HARDENED_GIT_STATUS_SHORT]
            assert provider.calls == []
            effects = runtime.store.list_external_effects(pid=pid)
            assert len(effects) == 1
            assert effects[0].operation == 'run'
            assert effects[0].transaction_state == 'unknown'
            assert effects[0].information_flow is True
            assert effects[0].provider_metadata['outcome'] == 'unknown_after_provider_success'
            assert effects[0].provider_metadata['provider_phases'] == [
                {
                    'name': 'resolve_argv',
                    'state_mutation': False,
                    'information_flow': True,
                }
            ]
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

    def test_shell_post_provider_event_failure_leaves_durable_unknown_effect_intent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='durable shell effect intent')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            original_emit = runtime.events.emit

            def fail_shell_result_event(event_type: EventType | str, *args: Any, **kwargs: Any) -> Any:
                if EventType(event_type) == EventType.EXTERNAL_WRITE and kwargs.get('target') == 'shell:git':
                    raise RuntimeError('injected shell result event failure')
                return original_emit(event_type, *args, **kwargs)

            monkeypatch.setattr(runtime.events, 'emit', fail_shell_result_event)
            with pytest.raises(RuntimeError, match='injected shell result event failure'):
                runtime.shell.run(pid, ['git', 'status', '--short'])

            assert provider.calls == [(_HARDENED_GIT_STATUS_SHORT, runtime.config.tools.shell_timeout_s)]
            effects = runtime.store.list_external_effects(pid=pid)
            assert len(effects) == 1
            assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
            assert effects[0].provider_metadata['effect_state'] == 'pending'
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

    def test_shell_rejects_file_url_arguments_before_provider_run(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='file url shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.always_allow_level, issued_by='test')

            with pytest.raises(ValidationError, match='raw Git'):
                runtime.shell.run(pid, ['git', 'status', f'file://{Path(tempfile.gettempdir())}/secret.txt'])

            assert provider.calls == []
        finally:
            runtime.close()

    def test_shell_subprocess_home_points_to_workspace(self) -> None:
        if os.name == 'nt':
            pytest.skip('the env utility assertion is POSIX-specific')
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='workspace home shell')
                runtime.shell.grant_policy(pid, runtime.config.shell.always_allow_level, issued_by='test')

                result = runtime.shell.run(
                    pid,
                    ['env'],
                )

                environment = dict(
                    line.split('=', 1)
                    for line in result.stdout.splitlines()
                    if '=' in line
                )
                assert environment['HOME'] == str(Path(temp_dir).resolve())
                assert environment['USERPROFILE'] == str(Path(temp_dir).resolve())
            finally:
                runtime.close()

    def test_unlisted_git_command_is_deterministically_rejected(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='run unlisted shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(ValidationError, match='typed git_'):
                runtime.shell.run(pid, ['git', 'show', '--stat'])
            assert runtime.human.pending() == []
            assert provider.calls == []
        finally:
            runtime.close()

    def test_launcher_wrapped_git_is_rejected_before_policy_or_evidence(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='reject wrapped Git')
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.always_allow_level,
                issued_by='test',
            )
            audit_before = runtime.audit.trace()
            events_before = runtime.events.list()
            effects_before = runtime.store.list_external_effects(pid=pid)

            with pytest.raises(ValidationError, match='typed git_'):
                runtime.shell.run(pid, ['env', 'git', 'branch', 'wrapper-created'])

            assert provider.calls == []
            assert runtime.human.pending() == []
            assert runtime.audit.trace() == audit_before
            assert runtime.events.list() == events_before
            assert runtime.store.list_external_effects(pid=pid) == effects_before
        finally:
            runtime.close()

    def test_human_approval_cannot_override_typed_git_boundary(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='argv-bound shell')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(ValidationError, match='typed git_'):
                runtime.shell.run(pid, ['git', 'show', '--stat'])
            with pytest.raises(ValidationError, match='typed git_'):
                runtime.shell.run(pid, ['git', 'push'])
            assert provider.calls == []
            assert runtime.human.pending() == []
        finally:
            runtime.close()

    def test_git_diff_with_output_is_rejected_before_approval(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='diff output')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(ValidationError, match='typed git_'):
                runtime.shell.run(pid, ['git', 'diff', '--output=patch.diff'])
            assert runtime.human.pending() == []
        finally:
            runtime.close()

    def test_auto_allowed_git_diff_rejects_repo_configured_external_helper(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self._temp_dirs.append(temp_dir)
        root = Path(temp_dir.name)
        marker = root / 'external-diff-ran'
        helper = root / 'external-diff-helper.sh'
        helper.write_text(f'#!/bin/sh\nprintf ran > "{marker}"\nexit 0\n', encoding='utf-8')
        helper.chmod(0o700)
        subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
        subprocess.run(['git', 'config', 'diff.external', str(helper)], cwd=root, check=True)
        tracked = root / 'tracked.txt'
        tracked.write_text('before\n', encoding='utf-8')
        subprocess.run(['git', 'add', 'tracked.txt'], cwd=root, check=True)
        tracked.write_text('after\n', encoding='utf-8')
        runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='safe git diff')
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.allowlist_auto_else_ask_level,
                issued_by='test',
            )

            with pytest.raises(GitError) as exc_info:
                runtime.shell.run(pid, ['git', 'diff'])

            assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
            assert not marker.exists()
        finally:
            runtime.close()

    def test_local_pty_git_status_rejects_active_repository_filter(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self._temp_dirs.append(temp_dir)
        root = Path(temp_dir.name)
        marker = root / 'filter-ran'
        helper = root / 'filter-helper.sh'
        helper.write_text(f'#!/bin/sh\nprintf ran > "{marker}"\ncat\n', encoding='utf-8')
        helper.chmod(0o700)
        subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
        subprocess.run(['git', 'config', 'filter.evil.clean', str(helper)], cwd=root, check=True)
        (root / '.gitattributes').write_text('*.txt filter=evil\n', encoding='utf-8')
        (root / 'tracked.txt').write_text('content\n', encoding='utf-8')
        provider = LocalPtyProvider(root)

        with pytest.raises(GitError) as exc_info:
            provider.spawn(
                harden_read_only_git_argv(['git', 'status']),
                cwd='.',
            )

        assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
        assert not marker.exists()

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

    def test_shell_denies_attached_short_option_cwd_path_escape_before_execution(self) -> None:
        self._assert_shell_denies_attached_short_option_path_escape('-C../outside')

    def test_shell_denies_attached_short_option_include_path_escape_before_execution(self) -> None:
        self._assert_shell_denies_attached_short_option_path_escape('-I/outside')

    def test_shell_denies_attached_short_option_output_path_escape_before_execution(self) -> None:
        self._assert_shell_denies_attached_short_option_path_escape('-o..\\outside')

    def test_shell_denies_attached_short_option_parent_operand_before_execution(self) -> None:
        self._assert_shell_denies_attached_short_option_path_escape('-C..')

    def test_shell_denies_attached_short_option_home_operand_before_execution(self) -> None:
        self._assert_shell_denies_attached_short_option_path_escape('-I~')

    def _assert_shell_denies_attached_short_option_path_escape(self, argument: str) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='attached argv scope')
            runtime.shell.grant_policy(pid, runtime.config.shell.always_allow_level, issued_by='test')

            with pytest.raises(CapabilityDenied, match='escapes workspace root|host absolute path syntax|host home expansion'):
                runtime.shell.run(pid, ['tool', argument], timeout=2.0)

            assert provider.calls == []
        finally:
            runtime.close()

    def test_shell_single_dash_url_and_define_arguments_are_not_path_operands(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='url define argv scope')
            runtime.shell.grant_policy(pid, runtime.config.shell.always_allow_level, issued_by='test')

            runtime.shell.run(
                pid,
                ['tool', '-DURL=https://example.test/assets/pkg.tar.gz', '-Uhttps://example.test/index'],
                timeout=2.0,
            )

            assert provider.calls == [
                (['tool', '-DURL=https://example.test/assets/pkg.tar.gz', '-Uhttps://example.test/index'], 2.0)
            ]
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

    def test_ask_shell_policy_requires_human_approval_even_at_always_allow_level(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='ask shell policy')
            runtime.capability.issue_trusted(
                pid,
                runtime.shell.policy_resource(),
                [CapabilityRight.EXECUTE],
                issued_by='test',
                effect=CapabilityEffect.ASK,
                constraints={
                    runtime.config.shell.policy_capability_key: runtime.config.shell.always_allow_level,
                },
            )

            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['git', 'status', '--short'])

            assert provider.calls == []
            pending = runtime.human.pending()
            assert len(pending) == 1
            assert pending[0].payload['context']['argv'] == ['git', 'status', '--short']
            assert runtime.human.drain_terminal_queue(auto_approve=True)[0].status == HumanRequestStatus.APPROVED
            allowed = runtime.shell.run(pid, ['git', 'status', '--short'])
            assert allowed.stdout == 'ok\n'
            with pytest.raises(HumanApprovalRequired):
                runtime.shell.run(pid, ['git', 'status', '--short'])
            assert provider.calls == [(_HARDENED_GIT_STATUS_SHORT, runtime.config.tools.shell_timeout_s)]
        finally:
            runtime.close()

    def test_one_time_shell_policy_is_consumed_after_success(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='one-time shell policy')
            policy = runtime.capability.issue_trusted(
                pid,
                runtime.shell.policy_resource(),
                [CapabilityRight.EXECUTE],
                issued_by='test',
                constraints={
                    runtime.config.shell.policy_capability_key: runtime.config.shell.always_allow_level,
                },
                uses_remaining=1,
            )

            result = runtime.shell.run(pid, ['git', 'status', '--short'])

            assert result.stdout == 'ok\n'
            assert runtime.store.get_capability(policy.cap_id).uses_remaining == 0
            with pytest.raises(CapabilityDenied):
                runtime.shell.run(pid, ['git', 'status', '--short'])
            assert provider.calls == [(_HARDENED_GIT_STATUS_SHORT, runtime.config.tools.shell_timeout_s)]
        finally:
            runtime.close()

    def test_shell_timeout_records_unknown_external_effect_and_commits_one_time_use(self) -> None:
        provider = TimeoutShellProvider()
        runtime = self._runtime_with_shell_provider(provider)
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='timeout effect accounting')
            policy = runtime.capability.issue_trusted(
                pid,
                runtime.shell.policy_resource(),
                [CapabilityRight.EXECUTE],
                issued_by='test',
                constraints={
                    runtime.config.shell.policy_capability_key: runtime.config.shell.always_allow_level,
                },
                uses_remaining=1,
            )

            with pytest.raises(TimeoutError, match='timed out'):
                runtime.shell.run(pid, ['git', 'status', '--short'])

            assert runtime.store.get_capability(policy.cap_id).uses_remaining == 0
            effects = runtime.store.list_external_effects(pid=pid)
            assert len(effects) == 1
            assert effects[0].operation == 'run'
            assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
            assert effects[0].provider_metadata['outcome'] == 'unknown_after_provider_exception'
        finally:
            runtime.close()

    def test_shell_classifier_failure_returns_result_and_records_conservative_effect(self) -> None:
        provider = ClassifierFailureShellProvider()
        runtime = self._runtime_with_shell_provider(provider)
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='classifier failure accounting')
            runtime.shell.grant_policy(pid, runtime.config.shell.always_allow_level, issued_by='test')

            result = runtime.shell.run(pid, ['git', 'status', '--short'])

            assert result.returncode == 0
            effects = runtime.store.list_external_effects(pid=pid)
            assert len(effects) == 1
            assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
            assert effects[0].provider_metadata['classification_fallback'] == 'post_effect_failure'
        finally:
            runtime.close()

    def test_shell_provider_certified_pre_effect_failure_restores_one_time_use(self) -> None:
        provider = PreEffectFailureShellProvider()
        runtime = self._runtime_with_shell_provider(provider)
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='pre-effect failure restore')
            policy = runtime.capability.issue_trusted(
                pid,
                runtime.shell.policy_resource(),
                [CapabilityRight.EXECUTE],
                issued_by='test',
                constraints={
                    runtime.config.shell.policy_capability_key: runtime.config.shell.always_allow_level,
                },
                uses_remaining=1,
            )

            with pytest.raises(ProviderEffectNotStarted, match='before execution'):
                runtime.shell.run(pid, ['git', 'status', '--short'])

            assert runtime.store.get_capability(policy.cap_id).uses_remaining == 1
            assert runtime.store.list_external_effects(pid=pid) == []
        finally:
            runtime.close()

    def test_permanent_allow_shell_policy_can_authorize_repeated_runs(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='permanent shell policy')
            policy = runtime.shell.grant_policy(pid, runtime.config.shell.always_allow_level, issued_by='test')

            first = runtime.shell.run(pid, ['git', 'status', '--short'])
            second = runtime.shell.run(pid, ['git', 'status', '--short'])

            assert first.stdout == 'ok\n'
            assert second.stdout == 'ok\n'
            assert runtime.store.get_capability(policy.cap_id).uses_remaining is None
            assert provider.calls == [
                (_HARDENED_GIT_STATUS_SHORT, runtime.config.tools.shell_timeout_s),
                (_HARDENED_GIT_STATUS_SHORT, runtime.config.tools.shell_timeout_s),
            ]
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
            with pytest.raises(ValidationError, match='typed git_'):
                runtime.shell.run(pid, ['git', 'push'])
            assert result.stdout == 'ok\n'
            assert provider.calls == [(_HARDENED_GIT_STATUS, runtime.config.tools.shell_timeout_s)]
        finally:
            runtime.close()

    def test_shell_explicit_deny_dominates_later_command_allow(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='explicit deny dominates')
            runtime.capability.issue_trusted(
                pid,
                'shell:git',
                [CapabilityRight.EXECUTE],
                issued_by='test',
                effect=CapabilityEffect.DENY,
            )
            runtime.capability.grant(pid, 'shell:git', [CapabilityRight.EXECUTE], issued_by='test')
            runtime.capability.grant(pid, 'shell:git:*', [CapabilityRight.EXECUTE], issued_by='test')

            with pytest.raises(CapabilityDenied):
                runtime.shell.run(pid, ['git', 'status'])

            assert provider.calls == []
        finally:
            runtime.close()

    def test_request_permission_shell_command_class_is_rule_constrained(self) -> None:
        runtime, provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(
                image='review-agent:v0',
                goal='request git shell',
                authority_manifest={
                    'authorized_capabilities': [
                        {
                            'resource': 'human:owner',
                            'rights': [CapabilityRight.WRITE.value],
                        }
                    ],
                    'approval_policy': {
                        'requestable_capabilities': [
                            {
                                'resource': 'shell:git',
                                'rights': [CapabilityRight.EXECUTE.value],
                            }
                        ]
                    },
                },
            )
            with pytest.raises(HumanResponseRequired):
                runtime.tools.call(pid, 'request_permission', {'resource': 'shell:git', 'rights': ['execute'], 'reason': 'inspect git state'})
            pending = runtime.human.pending()[0]
            rules = pending.payload['requested_permission']['constraints']['authority_rules']
            runtime.human.drain_terminal_queue(auto_policy=CapabilityManager.ALWAYS_ALLOW)
            request = runtime.tools.call(pid, 'request_permission', {'resource': 'shell:git', 'rights': ['execute'], 'reason': 'inspect git state'})
            allowed = runtime.shell.run(pid, ['git', 'status'])
            with pytest.raises(ValidationError, match='typed git_'):
                runtime.shell.run(pid, ['git', 'push', 'origin'])
            assert request.ok
            assert request.payload['status'] == 'approved'
            assert any(rule['rule_id'] == 'shell.git.deny.push' for rule in rules)
            assert allowed.stdout == 'ok\n'
            assert provider.calls == [(_HARDENED_GIT_STATUS, runtime.config.tools.shell_timeout_s)]
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

    def test_path_qualified_git_is_rejected_before_policy(self) -> None:
        runtime, _provider = self._runtime_with_fake_shell()
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='path bypass')
            runtime.shell.grant_policy(pid, runtime.config.shell.allowlist_auto_else_ask_level, issued_by='test')
            with pytest.raises(ValidationError, match='executable paths'):
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
            assert provider.calls == [(_HARDENED_GIT_STATUS_SHORT, runtime.config.tools.shell_timeout_s)]
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
        provider = FakeShellProvider()
        return self._runtime_with_shell_provider(provider), provider

    def _runtime_with_shell_provider(self, provider: 'FakeShellProvider') -> Runtime:
        temp_dir = tempfile.TemporaryDirectory()
        self._temp_dirs.append(temp_dir)
        substrate = RecordingShellSubstrate(temp_dir.name, provider)
        runtime = Runtime.open('local', substrate=substrate)
        runtime.substrate.human.output_sink = lambda _message: None
        return runtime

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

    def test_custom_deny_rule_overrides_builtin_harmless_allow(self) -> None:
        config = AgentLibOSConfig(
            shell=ShellDefaults(
                rules=(
                    AuthorityRule(
                        rule_id='custom.git.status.deny',
                        operation='shell.run',
                        effect=CapabilityEffect.DENY,
                        risk=AuthorityRisk.HIGH,
                        conditions={'argv': ['git', 'status', '--short'], 'match': 'exact'},
                    ),
                ),
                whitelist=(),
                blacklist=(),
            )
        )
        runtime, provider = self._runtime_with_config(config)
        try:
            pid = runtime.process.spawn(image='review-agent:v0', goal='custom deny harmless')
            runtime.shell.grant_policy(pid, config.shell.allowlist_auto_else_ask_level, issued_by='test')

            with pytest.raises(CapabilityDenied, match='denied'):
                runtime.shell.run(pid, ['git', 'status', '--short'])

            assert provider.calls == []
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
        assert Path(captured['argv'][0]).stem == 'echo'
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
        assert captured['argv'][0].replace('\\', '/').endswith('/usr/bin/git')
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
        shell.cwd = self.workspace_root
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


class ResolvingShellProvider(FakeShellProvider):
    def __init__(self) -> None:
        super().__init__()
        self.resolver_calls: list[list[str]] = []

    def resolve_argv(self, argv: list[str], *, cwd: str | None = None) -> list[str]:
        self.resolver_calls.append(list(argv))
        return list(argv)


class TimeoutShellProvider(FakeShellProvider):
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
        raise SubprocessTimeoutExpired(
            'provider timed out after execution began',
            metrics=CommandMetrics(wall_seconds=timeout, killed=True, limit_kind='wall_time'),
        )


class SwitchingShellProvider(ResolvingShellProvider):
    def resolve_argv(self, argv: list[str], *, cwd: str | None = None) -> list[str]:
        self.resolver_calls.append(list(argv))
        return [str(Path(tempfile.gettempdir()) / 'switched-shell-executable'), *argv[1:]]


class ClassifierFailureShellProvider(FakeShellProvider):
    def classify_external_effect(self, operation: str, context: dict[str, Any], result: Any) -> ExternalEffectClassification:
        raise RuntimeError('classifier unavailable after execution')


class PreEffectFailureShellProvider(FakeShellProvider):
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
        raise ProviderEffectNotStarted('provider failed before execution')
