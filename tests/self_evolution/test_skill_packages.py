from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, SkillDefaults, ToolDefaults
from agent_libos.models import AgentImage, CapabilityRight, EventType
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, NotFound, ValidationError
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.skills.schema import JitToolSpec, SkillPackage, SkillResource
from agent_libos.substrate import LocalResourceProviderSubstrate
from tests.support.public_errors import assert_public_error_message
from tests.support.skills import write_raw_skill, write_skill_package


class TestSkillPackageLoading:
    def test_skill_discovery_window_reports_registered_packages_beyond_requested_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = Runtime.open('local')
            try:
                for index in range(3):
                    skill_dir = write_skill_package(root, f'window-skill-{index}', allowed_tools=['echo'])
                    package, _source = runtime.skills._load_package_from_host_path(skill_dir)
                    runtime.skills.register_skill_package(package, actor='cli', require_capability=False)

                bounded, has_more = runtime.skills.discover_skills_window(
                    text='window-skill',
                    actor='test',
                    require_capability=False,
                    limit=2,
                )
                complete, complete_has_more = runtime.skills.discover_skills_window(
                    text='window-skill',
                    actor='test',
                    require_capability=False,
                    limit=3,
                )

                assert {
                    item['skill_id']
                    for item in bounded
                } == {'window-skill-0', 'window-skill-1'}
                assert has_more is True
                assert {
                    item['skill_id']
                    for item in complete
                } == {'window-skill-0', 'window-skill-1', 'window-skill-2'}
                assert complete_has_more is False
            finally:
                runtime.close()

    def test_skill_discovery_ranks_registered_metadata_before_applying_limit(self) -> None:
        runtime = Runtime.open('local')
        try:
            for index in range(3):
                runtime.skills.register_skill_package(
                    SkillPackage(
                        skill_id=f'aaa-weak-route-{index}',
                        name=f'aaa-weak-route-{index}',
                        description=(
                            'Quasar workflows use a ledger and reconcile one matching intent.'
                        ),
                        instructions='Use echo.',
                        allowed_tools=['echo'],
                    ),
                    actor='test.host',
                    require_capability=False,
                )
            runtime.skills.register_skill_package(
                SkillPackage(
                    skill_id='zzz-quasar-ledger-reconcile',
                    name='zzz-quasar-ledger-reconcile',
                    description='Handle the matching intent.',
                    instructions='Use echo.',
                    allowed_tools=['echo'],
                ),
                actor='test.host',
                require_capability=False,
            )

            page, has_more = runtime.skills.discover_skills_window(
                text='quasar ledger reconcile intent',
                actor='test',
                require_capability=False,
                limit=1,
            )

            assert [item['skill_id'] for item in page] == [
                'zzz-quasar-ledger-reconcile'
            ]
            assert has_more is True
        finally:
            runtime.close()

    def test_skill_discovery_returns_split_owner_partial_matches(self) -> None:
        runtime = Runtime.open('local')
        try:
            for skill_id, description in (
                (
                    'constellation-reader',
                    'Read constellation workspace text safely.',
                ),
                (
                    'constellation-writer',
                    'Write constellation workspace text safely.',
                ),
                (
                    'constellation-unrelated',
                    'Observe constellation telemetry.',
                ),
            ):
                runtime.skills.register_skill_package(
                    SkillPackage(
                        skill_id=skill_id,
                        name=skill_id,
                        description=description,
                        instructions='Use echo.',
                        allowed_tools=['echo'],
                    ),
                    actor='test.host',
                    require_capability=False,
                )

            discovered = runtime.skills.discover_skills(
                text='constellation read write',
                actor='test',
                require_capability=False,
                limit=5,
            )

            discovered_ids = [item['skill_id'] for item in discovered]
            assert set(discovered_ids[:2]) == {
                'constellation-reader',
                'constellation-writer',
            }
            assert 'constellation-unrelated' not in discovered_ids
        finally:
            runtime.close()

    def test_skill_discovery_rejects_unbounded_limits(self) -> None:
        runtime = Runtime.open('local')
        try:
            for limit in (0, -1, True, runtime.config.skills.discover_limit + 1):
                with pytest.raises(ValidationError, match='limit'):
                    runtime.skills.discover_skills(require_capability=False, limit=limit)  # type: ignore[arg-type]
        finally:
            runtime.close()

    def test_skill_discovery_searches_metadata_not_instruction_bodies(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.skills.register_skill_package(
                SkillPackage(
                    skill_id='metadata-search-skill',
                    name='metadata-search-skill',
                    description='Find this package using the visible-description-token.',
                    instructions='The private-instruction-token is body content only.',
                ),
                actor='test.host',
                require_capability=False,
            )

            visible = runtime.skills.discover_skills(
                text='visible-description-token',
                actor='test',
                require_capability=False,
            )
            hidden = runtime.skills.discover_skills(
                text='private-instruction-token',
                actor='test',
                require_capability=False,
            )

            assert [item['skill_id'] for item in visible] == ['metadata-search-skill']
            assert hidden == []

            natural_phrase = runtime.skills.discover_skills(
                text='find visible description token',
                actor='test',
                require_capability=False,
            )
            assert [item['skill_id'] for item in natural_phrase] == [
                'metadata-search-skill'
            ]
        finally:
            runtime.close()

    def test_skill_discovery_uses_only_configured_workspace_roots_and_deduplicates_aliases(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_raw_skill(
                root / 'custom-catalog',
                'configured-skill',
                'name: configured-skill\ndescription: Configured catalog Skill.\n',
            )
            write_raw_skill(
                root / 'skills',
                'implicit-skill',
                'name: implicit-skill\ndescription: Must not be discovered implicitly.\n',
            )
            monkeypatch.chdir(root)
            config = AgentLibOSConfig(
                skills=replace(
                    SkillDefaults(),
                    workspace_dirs=('custom-catalog', './custom-catalog'),
                    global_dirs=(str(root / 'global-catalog'),),
                )
            )
            runtime = Runtime.open('local', config=config)
            try:
                loaded_paths: list[Path] = []
                original_load = runtime.skills._load_package_from_host_path

                def record_load(path: str | Path):
                    loaded_paths.append(Path(path).resolve())
                    return original_load(path)

                monkeypatch.setattr(runtime.skills, '_load_package_from_host_path', record_load)
                discovered = runtime.skills.discover_skills(
                    text='configured-skill',
                    require_capability=False,
                )

                assert [item['skill_id'] for item in discovered] == ['configured-skill']
                assert loaded_paths == [(root / 'custom-catalog' / 'configured-skill').resolve()]
            finally:
                runtime.close()


    def test_standard_package_validation_and_global_trust(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            global_dir = root / 'global-skills'
            skill_dir = write_skill_package(global_dir, 'global-skill', allowed_tools=['echo'])
            config = AgentLibOSConfig(skills=replace(SkillDefaults(), global_dirs=(str(global_dir),)))
            runtime = Runtime.open('local', config=config)
            try:
                parsed = runtime.skills.validate_package_path(skill_dir)
                assert parsed['allowed_tools'] == ['echo']
                assert 'allowed-tools: echo' in (skill_dir / 'SKILL.md').read_text(encoding='utf-8')

                legacy_list_dir = write_raw_skill(
                    root,
                    'legacy-list-skill',
                    'name: legacy-list-skill\n'
                    'description: Retains compatibility with YAML-list allowed tools.\n'
                    'allowed-tools:\n'
                    '  - echo\n',
                )
                assert runtime.skills.validate_package_path(legacy_list_dir)['allowed_tools'] == ['echo']

                with pytest.raises(CapabilityDenied):
                    runtime.skills.register_global_skill_from_path(skill_dir, actor='cli', require_capability=False)
                trust = runtime.skills.global_package_info(skill_dir)
                runtime.skills.trust_skill_source(actor='cli', source_type='global', source=trust['source'], package_sha256=trust['package_sha256'], require_capability=False)
                registered = runtime.skills.register_global_skill_from_path(skill_dir, actor='cli', require_capability=False)
                assert registered['skill_id'] == 'global-skill'
                assert registered['source_type'] == 'global'
                assert 'package_sha256' in registered
                with pytest.raises(ValidationError):
                    runtime.skills.validate_package_path(write_raw_skill(root, 'bad', 'name: bad\ndescription: Bad\nunknown: nope\n'))
                with pytest.raises(ValidationError):
                    runtime.skills.validate_package_path(write_raw_skill(root, 'BadName', 'name: BadName\ndescription: Bad\n'))
                with pytest.raises(ValidationError):
                    runtime.skills.validate_package_path(write_raw_skill(root, 'bad-', 'name: bad-\ndescription: Bad\n'))
                with pytest.raises(ValidationError):
                    runtime.skills.validate_package_path(write_raw_skill(root, 'bad--name', 'name: bad--name\ndescription: Bad\n'))
                overlong_name = 'a' * 65
                with pytest.raises(ValidationError):
                    runtime.skills.validate_package_path(
                        write_raw_skill(
                            root,
                            overlong_name,
                            f'name: {overlong_name}\ndescription: Bad\n',
                        )
                    )
                with pytest.raises(ValidationError):
                    runtime.skills.validate_package_path(write_raw_skill(root, 'bad-metadata', 'name: bad-metadata\ndescription: Bad\nmetadata: {agent-libos.version: 1}\n'))
                old_yaml = root / 'legacy.yaml'
                old_yaml.write_text('schema_version: 1\nskill_id: legacy:v0\nname: Legacy\n', encoding='utf-8')
                with pytest.raises(ValidationError):
                    runtime.skills.validate_package_path(old_yaml)
                with pytest.raises(ValidationError):
                    runtime.skills.register_skill_package({'schema_version': 1, 'skill_id': 'legacy', 'name': 'legacy', 'description': 'Legacy shape.', 'tools': ['echo']}, actor='cli', require_capability=False)
                with pytest.raises(ValidationError):
                    write_skill_package(root, 'bad-jit', jit_tools=[{'name': 'bad', 'description': 'bad', 'source_path': '../escaped.ts'}])
                with pytest.raises(ValidationError):
                    write_skill_package(root, 'bad-right', required_capabilities=[{'resource': 'filesystem:workspace:*', 'rights': ['*']}])
                with pytest.raises(
                    ValidationError,
                    match='capability spec contains unknown fields',
                ):
                    write_skill_package(
                        root,
                        'unknown-capability-field',
                        required_capabilities=[
                            {
                                'resource': 'filesystem:workspace:*',
                                'rights': ['read'],
                                'constrants': {'path': 'README.md'},
                            }
                        ],
                    )
            finally:
                runtime.close()

    def test_programmatic_capability_spec_rejects_mixed_unknown_key_types(
        self,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            capability_spec: dict[Any, Any] = {
                'resource': 'filesystem:workspace:*',
                'rights': ['read'],
                'constrants': {'path': 'README.md'},
                1: 'unexpected',
            }
            package = SkillPackage(
                skill_id='mixed-capability-keys',
                name='mixed-capability-keys',
                description='Reject mixed unknown capability keys.',
                instructions='Do not load this invalid package.',
                required_capabilities=[capability_spec],  # type: ignore[list-item]
            )

            with pytest.raises(
                ValidationError,
                match='capability spec contains unknown fields',
            ):
                runtime.skills.register_skill_package(
                    package,
                    actor='test.host',
                    require_capability=False,
                )
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('kind', 'content', 'content_base64', 'expected_error'),
        [
            ('text', 'x', 'eA==', 'must not contain content_base64'),
            ('base64', 'x', 'eA==', 'must not contain text content'),
        ],
    )
    def test_programmatic_skill_resource_rejects_dual_payload_forms(
        self,
        kind: str,
        content: str | None,
        content_base64: str | None,
        expected_error: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            package = SkillPackage(
                skill_id=f'dual-resource-{kind}',
                name=f'dual-resource-{kind}',
                description='Reject unhashed alternate resource payloads.',
                instructions='Do not load this invalid package.',
                resources=[
                    SkillResource(
                        path='references/value.bin',
                        size_bytes=1,
                        sha256=hashlib.sha256(b'x').hexdigest(),
                        kind=kind,
                        content=content,
                        content_base64=content_base64,
                    )
                ],
            )

            with pytest.raises(ValidationError, match=expected_error):
                runtime.skills.register_skill_package(
                    package,
                    actor='test.host',
                    require_capability=False,
                )
        finally:
            runtime.close()

    @pytest.mark.parametrize('claim_source', ('embedded', 'argument'))
    def test_skill_registration_rejects_claimed_hash_mismatch_before_authority_use(
        self,
        claim_source: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(
                image='base-agent:v0',
                goal='reject a mismatched Skill content hash',
            )
            skill_id = f'hash-mismatch-{claim_source}'
            authority = runtime.capability.grant_once(
                actor,
                f'skill:{skill_id}',
                [CapabilityRight.WRITE],
                issued_by='test',
            )
            package = SkillPackage(
                skill_id=skill_id,
                name=skill_id,
                description='Reject a caller-supplied mismatched hash.',
                instructions='This content has one canonical hash.',
                package_sha256=('a' * 64 if claim_source == 'embedded' else ''),
            )
            kwargs = (
                {'package_sha256': 'a' * 64}
                if claim_source == 'argument'
                else {}
            )

            with pytest.raises(ValidationError, match='does not match'):
                runtime.skills.register_skill_package(
                    package,
                    actor=actor,
                    **kwargs,
                )

            assert runtime.store.get_skill(skill_id) is None
            assert runtime.store.get_capability(authority.cap_id).uses_remaining == 1
        finally:
            runtime.close()

    def test_global_skill_registration_cannot_rebind_trusted_hash_to_other_content(
        self,
    ) -> None:
        runtime = Runtime.open('local')
        fake_hash = 'b' * 64
        source = 'global/hash-pin'
        try:
            runtime.skills.trust_skill_source(
                actor='test.host',
                source_type='global',
                source=source,
                package_sha256=fake_hash,
                require_capability=False,
            )
            package = SkillPackage(
                skill_id='global-hash-pin',
                name='global-hash-pin',
                description='Content must match the trusted hash pin.',
                instructions='Different content cannot reuse a trusted pin.',
            )

            with pytest.raises(ValidationError, match='does not match'):
                runtime.skills.register_skill_package(
                    package,
                    actor='test.host',
                    source_type='global',
                    source=source,
                    package_sha256=fake_hash,
                    require_capability=False,
                )

            assert runtime.store.get_skill(package.skill_id) is None
        finally:
            runtime.close()

    def test_programmatic_skill_registration_rejects_invalid_jit_timeouts(self) -> None:
        hard_limit = 7.0
        config = AgentLibOSConfig(
            tools=replace(
                ToolDefaults(),
                deno_timeout_s=5.0,
                deno_timeout_hard_limit_s=hard_limit,
            )
        )
        invalid_cases = (
            (True, 'must be a number'),
            ('5', 'must be a number'),
            (float('nan'), 'must be finite and > 0'),
            (float('inf'), 'must be finite and > 0'),
            (0, 'must be finite and > 0'),
            (-1.0, 'must be finite and > 0'),
            (hard_limit + 0.01, 'deno_timeout_hard_limit_s=7.0'),
            (10**400, 'deno_timeout_hard_limit_s=7.0'),
        )
        runtime = Runtime.open('local', config=config)
        try:
            for index, (timeout_s, message) in enumerate(invalid_cases):
                skill_id = f'invalid-jit-timeout-{index}'
                package = _programmatic_jit_skill(skill_id, timeout_s)

                with pytest.raises(ValidationError, match=message):
                    runtime.skills.register_skill_package(
                        package,
                        actor='cli',
                        require_capability=False,
                    )

                assert runtime.store.get_skill(skill_id) is None
        finally:
            runtime.close()

    def test_programmatic_skill_registration_accepts_valid_jit_timeouts(self) -> None:
        hard_limit = 7.0
        config = AgentLibOSConfig(
            tools=replace(
                ToolDefaults(),
                deno_timeout_s=5.0,
                deno_timeout_hard_limit_s=hard_limit,
            )
        )
        runtime = Runtime.open('local', config=config)
        try:
            for index, timeout_s in enumerate((None, 1, 1.25, hard_limit)):
                skill_id = f'valid-jit-timeout-{index}'
                registered = runtime.skills.register_skill_package(
                    _programmatic_jit_skill(skill_id, timeout_s),
                    actor='cli',
                    require_capability=False,
                )
                persisted, _metadata = runtime.store.get_skill(skill_id)

                assert registered['skill_id'] == skill_id
                assert persisted.jit_tools[0].timeout_s == timeout_s
        finally:
            runtime.close()

    def test_failed_skill_replace_rolls_back_registry_and_restores_one_time_write(self, monkeypatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_dir = write_skill_package(root, 'atomic-skill', allowed_tools=['echo'], body='original instructions\n')
            runtime = Runtime.open('local')
            try:
                original, _source = runtime.skills._load_package_from_host_path(skill_dir)
                runtime.skills.register_skill_package(original, actor='cli', require_capability=False)
                write_skill_package(root, 'atomic-skill', allowed_tools=['human_output'], body='replacement instructions\n')
                replacement, _source = runtime.skills._load_package_from_host_path(skill_dir)
                actor = runtime.process.spawn(image='base-agent:v0', goal='replace skill')
                cap = runtime.capability.grant_once(
                    actor,
                    'skill:atomic-skill',
                    [CapabilityRight.WRITE],
                    issued_by='test',
                )
                real_record = runtime.audit.record

                def fail_registration_audit(*args, **kwargs):
                    if kwargs.get('action') == 'skill.register':
                        raise RuntimeError('registration audit failed')
                    return real_record(*args, **kwargs)

                monkeypatch.setattr(runtime.audit, 'record', fail_registration_audit)
                with pytest.raises(RuntimeError, match='registration audit failed'):
                    runtime.skills.register_skill_package(replacement, actor=actor, replace=True)

                persisted, _metadata = runtime.store.get_skill('atomic-skill')
                assert persisted.package_sha256 == original.package_sha256
                assert persisted.allowed_tools == ['echo']
                assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
                assert not any(
                    event.type.value == 'skill_registered' and event.source == actor
                    for event in runtime.events.list()
                )
            finally:
                runtime.close()

    def test_failed_skill_trust_rolls_back_record_and_restores_one_time_admin(self, monkeypatch) -> None:
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(image='base-agent:v0', goal='trust skill')
            cap = runtime.capability.grant_once(
                actor,
                runtime.config.skills.trust_resource,
                [CapabilityRight.ADMIN],
                issued_by='test',
            )
            real_emit = runtime.events.emit

            def fail_trust_event(event_type, *args, **kwargs):
                if str(getattr(event_type, 'value', event_type)) == 'skill_trusted':
                    raise RuntimeError('trust event failed')
                return real_emit(event_type, *args, **kwargs)

            monkeypatch.setattr(runtime.events, 'emit', fail_trust_event)
            with pytest.raises(RuntimeError, match='trust event failed'):
                runtime.skills.trust_skill_source(
                    actor=actor,
                    source_type='global',
                    source='global/example',
                    package_sha256='a' * 64,
                )

            assert not runtime.store.is_skill_trusted(
                source_type='global',
                source='global/example',
                package_sha256='a' * 64,
            )
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
        finally:
            runtime.close()

    @pytest.mark.parametrize('operation', ['trust', 'untrust'])
    def test_skill_trust_reauthorizes_unlimited_admin_before_mutation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        operation: str,
    ) -> None:
        runtime = Runtime.open('local')
        source_type = 'global'
        source = 'global/reauthorization'
        package_sha256 = 'b' * 64
        try:
            actor = runtime.process.spawn(
                image='base-agent:v0',
                goal=f'skill trust authority race {operation}',
            )
            if operation == 'untrust':
                runtime.skills.trust_skill_source(
                    actor='test.host',
                    source_type=source_type,
                    source=source,
                    package_sha256=package_sha256,
                    require_capability=False,
                )
            authority = runtime.capability.grant(
                actor,
                runtime.config.skills.trust_resource,
                [CapabilityRight.ADMIN],
                issued_by='test.host',
            )
            original_require = runtime.capability.require

            def revoke_after_outer_authorization(*args: Any, **kwargs: Any):
                decision = original_require(*args, **kwargs)
                runtime.capability.revoke(
                    authority.cap_id,
                    revoked_by='test.host',
                    reason='skill trust revocation race regression',
                    require_authority=False,
                )
                return decision

            monkeypatch.setattr(runtime.capability, 'require', revoke_after_outer_authorization)

            with pytest.raises(CapabilityDenied, match='authority changed'):
                if operation == 'trust':
                    runtime.skills.trust_skill_source(
                        actor=actor,
                        source_type=source_type,
                        source=source,
                        package_sha256=package_sha256,
                    )
                else:
                    runtime.skills.untrust_skill_source(
                        actor=actor,
                        source_type=source_type,
                        source=source,
                        package_sha256=package_sha256,
                    )

            persisted = runtime.store.is_skill_trusted(
                source_type=source_type,
                source=source,
                package_sha256=package_sha256,
            )
            assert persisted is (operation == 'untrust')
        finally:
            runtime.close()

    def test_skill_registration_reauthorizes_inside_publication_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(
                Path(temp_dir),
                'registration-race-skill',
                allowed_tools=['echo'],
            )
            runtime = Runtime.open('local')
            try:
                package, _source = runtime.skills._load_package_from_host_path(skill_dir)
                actor = runtime.process.spawn(goal='skill registration authority race')
                authority = runtime.capability.grant(
                    actor,
                    'skill:registration-race-skill',
                    [CapabilityRight.WRITE],
                    issued_by='test.host',
                )
                original_transaction = runtime.capability.authority_transaction

                def revoke_before_publication(decisions, *, actor: str, operation: str):
                    if operation == 'skill registration':
                        runtime.capability.revoke(
                            authority.cap_id,
                            revoked_by='test.host',
                            reason='registration race regression',
                            require_authority=False,
                        )
                    return original_transaction(
                        decisions,
                        actor=actor,
                        operation=operation,
                    )

                monkeypatch.setattr(
                    runtime.capability,
                    'authority_transaction',
                    revoke_before_publication,
                )

                with pytest.raises(CapabilityDenied, match='authority changed'):
                    runtime.skills.register_skill_package(package, actor=actor)

                assert runtime.store.get_skill('registration-race-skill') is None
            finally:
                runtime.close()

    def test_global_skill_registration_rechecks_exact_trust_inside_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            global_dir = root / 'global-skills'
            skill_dir = write_skill_package(
                global_dir,
                'global-registration-race',
                allowed_tools=['echo'],
            )
            config = AgentLibOSConfig(
                skills=replace(SkillDefaults(), global_dirs=(str(global_dir),))
            )
            runtime = Runtime.open('local', config=config)
            try:
                trust = runtime.skills.global_package_info(skill_dir)
                runtime.skills.trust_skill_source(
                    actor='cli',
                    source_type='global',
                    source=trust['source'],
                    package_sha256=trust['package_sha256'],
                    require_capability=False,
                )
                original_transaction = runtime.capability.authority_transaction

                def untrust_before_publication(decisions, *, actor: str, operation: str):
                    if operation == 'skill registration':
                        runtime.skills.store.delete_skill_trust(
                            source_type='global',
                            source=trust['source'],
                            package_sha256=trust['package_sha256'],
                        )
                    return original_transaction(
                        decisions,
                        actor=actor,
                        operation=operation,
                    )

                monkeypatch.setattr(
                    runtime.capability,
                    'authority_transaction',
                    untrust_before_publication,
                )

                with pytest.raises(CapabilityDenied, match='not trusted'):
                    runtime.skills.register_global_skill_from_path(
                        skill_dir,
                        actor='cli',
                        require_capability=False,
                    )

                assert runtime.store.get_skill('global-registration-race') is None
            finally:
                runtime.close()

    def test_skill_activation_reauthorizes_inside_publication_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(
                Path(temp_dir),
                'activation-race-skill',
                allowed_tools=['echo'],
            )
            runtime = Runtime.open('local')
            try:
                runtime.skills.register_skill_from_path(
                    skill_dir,
                    actor='cli',
                    require_capability=False,
                )
                actor = runtime.process.spawn(goal='skill activation authority race')
                authority = runtime.capability.grant(
                    actor,
                    'skill:activation-race-skill',
                    [CapabilityRight.EXECUTE],
                    issued_by='test.host',
                )
                original_transaction = runtime.capability.authority_transaction

                def revoke_before_publication(decisions, *, actor: str, operation: str):
                    if operation == 'skill activation':
                        runtime.capability.revoke(
                            authority.cap_id,
                            revoked_by='test.host',
                            reason='activation race regression',
                            require_authority=False,
                        )
                    return original_transaction(
                        decisions,
                        actor=actor,
                        operation=operation,
                    )

                monkeypatch.setattr(
                    runtime.capability,
                    'authority_transaction',
                    revoke_before_publication,
                )

                with pytest.raises(CapabilityDenied, match='authority changed'):
                    runtime.skills.activate_skill(
                        actor,
                        'activation-race-skill',
                        actor=actor,
                    )

                assert 'activation-race-skill' not in runtime.process.get(actor).loaded_skills
                assert 'echo' not in runtime.process.get(actor).tool_table
            finally:
                runtime.close()

    def test_skill_activation_rejects_failed_reservation_settlement_atomically(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(
                Path(temp_dir),
                'activation-settlement-skill',
                allowed_tools=['echo'],
            )
            runtime = Runtime.open('local')
            try:
                runtime.skills.register_skill_from_path(
                    skill_dir,
                    actor='cli',
                    require_capability=False,
                )
                actor = runtime.process.spawn(goal='skill activation settlement')
                authority = runtime.capability.grant_once(
                    actor,
                    'skill:activation-settlement-skill',
                    [CapabilityRight.EXECUTE],
                    issued_by='test.host',
                )
                monkeypatch.setattr(
                    runtime.capability,
                    'commit_reserved_use',
                    lambda *args, **kwargs: False,
                )

                with pytest.raises(CapabilityDenied, match='reservation is no longer active'):
                    runtime.skills.activate_skill(
                        actor,
                        'activation-settlement-skill',
                        actor=actor,
                    )

                persisted = runtime.store.get_capability(authority.cap_id)
                assert persisted is not None
                assert persisted.active
                assert persisted.uses_remaining == 1
                assert 'activation-settlement-skill' not in runtime.process.get(actor).loaded_skills
                assert 'echo' not in runtime.process.get(actor).tool_table
            finally:
                runtime.close()

    def test_workspace_register_and_activate_reads_via_filesystem_and_uses_human_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_skill_package(Path(temp_dir), 'workspace-skill', allowed_tools=['echo'], extra_resources={'references/guide.md': 'Workspace resource guide.'})
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='load workspace skill')
                runtime.filesystem.grant_path(pid, 'workspace-skill/SKILL.md', [CapabilityRight.READ], issued_by='test')
                runtime.filesystem.grant_directory(pid, 'workspace-skill/references', [CapabilityRight.READ], issued_by='test')
                with pytest.raises(HumanApprovalRequired) as raised:
                    runtime.skills.activate_skill_from_workspace_path(pid, 'workspace-skill')
                runtime.human.approve(raised.value.request_id)
                loaded = runtime.skills.activate_skill_from_workspace_path(pid, 'workspace-skill')
                assert loaded['skill_id'] == 'workspace-skill'
                assert 'echo' in runtime.process.get(pid).tool_table
                assert not runtime.capability.check(pid, 'skill:workspace-skill', CapabilityRight.EXECUTE)
                resource = runtime.skills.read_skill_resource(pid, 'workspace-skill', 'references/guide.md')
                assert resource['content'] == 'Workspace resource guide.'
            finally:
                runtime.close()

    def test_host_skill_json_metadata_rejects_pathological_parser_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = Runtime.open('local')
            try:
                for index, payload in enumerate(_pathological_json_payloads()):
                    skill_dir = write_skill_package(
                        root,
                        f'host-json-limit-{index}',
                        actions=[{'name': 'review', 'use_cases': ['Review a change.']}],
                    )
                    (skill_dir / 'references' / 'agent-libos' / 'actions.json').write_text(
                        payload,
                        encoding='utf-8',
                    )

                    with pytest.raises(ValidationError, match='invalid JSON skill metadata resource'):
                        runtime.skills.validate_package_path(skill_dir)
            finally:
                runtime.close()

    def test_workspace_skill_json_metadata_rejects_pathological_parser_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_names: list[str] = []
            for index, payload in enumerate(_pathological_json_payloads()):
                name = f'workspace-json-limit-{index}'
                package_names.append(name)
                skill_dir = write_skill_package(
                    root,
                    name,
                    jit_tools=[
                        {
                            'name': f'check_{index}',
                            'description': 'Return a deterministic result.',
                            'source_path': 'scripts/check.ts',
                        }
                    ],
                    scripts={
                        'scripts/check.ts': (
                            'export async function run() { return {ok: true}; }\n'
                        )
                    },
                )
                (skill_dir / 'references' / 'agent-libos' / 'jit-tools.json').write_text(
                    payload,
                    encoding='utf-8',
                )

            runtime = Runtime.open(
                'local',
                substrate=LocalResourceProviderSubstrate(temp_dir),
            )
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='reject pathological Skill JSON')
                for name in package_names:
                    runtime.filesystem.grant_path(
                        pid,
                        f'{name}/SKILL.md',
                        [CapabilityRight.READ],
                        issued_by='test',
                    )
                    runtime.filesystem.grant_path(
                        pid,
                        f'{name}/references/agent-libos/jit-tools.json',
                        [CapabilityRight.READ],
                        issued_by='test',
                    )

                    with pytest.raises(ValidationError, match='invalid JSON skill metadata resource'):
                        runtime.skills.register_skill_from_workspace_path(
                            pid,
                            name,
                            require_capability=False,
                        )
                    assert runtime.store.get_skill(name) is None
            finally:
                runtime.close()

    def test_workspace_skill_rejects_truncated_skill_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_dir = root / 'truncated-skill-md'
            skill_dir.mkdir()
            complete_prefix = (
                '---\n'
                'name: truncated-skill-md\n'
                'description: Reject a partial Skill manifest.\n'
                '---\n\n'
                '# Complete visible prefix\n'
            ).encode('utf-8')
            (skill_dir / 'SKILL.md').write_bytes(complete_prefix + b'Hidden suffix must not be omitted.\n')
            config = AgentLibOSConfig(
                skills=replace(SkillDefaults(), skill_md_max_bytes=len(complete_prefix))
            )
            runtime = Runtime.open(
                'local',
                substrate=LocalResourceProviderSubstrate(temp_dir),
                config=config,
            )
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='reject truncated Skill')
                runtime.filesystem.grant_path(
                    pid,
                    'truncated-skill-md/SKILL.md',
                    [CapabilityRight.READ],
                    issued_by='test',
                )

                with pytest.raises(ValidationError, match='skill_md_max_bytes'):
                    runtime.skills.register_skill_from_workspace_path(
                        pid,
                        'truncated-skill-md',
                        require_capability=False,
                    )
                assert runtime.store.get_skill('truncated-skill-md') is None
            finally:
                runtime.close()

    def test_workspace_skill_rejects_truncated_explicit_metadata_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_dir = write_skill_package(
                root,
                'truncated-reference',
                actions=[{'name': 'review', 'use_cases': ['Review a change.']}],
            )
            reference = skill_dir / 'references' / 'agent-libos' / 'actions.json'
            complete_prefix = reference.read_bytes()
            reference.write_bytes(complete_prefix + b' ')
            config = AgentLibOSConfig(
                skills=replace(SkillDefaults(), resource_read_max_bytes=len(complete_prefix))
            )
            runtime = Runtime.open(
                'local',
                substrate=LocalResourceProviderSubstrate(temp_dir),
                config=config,
            )
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='reject truncated Skill reference')
                runtime.filesystem.grant_path(
                    pid,
                    'truncated-reference/SKILL.md',
                    [CapabilityRight.READ],
                    issued_by='test',
                )
                runtime.filesystem.grant_path(
                    pid,
                    'truncated-reference/references/agent-libos/actions.json',
                    [CapabilityRight.READ],
                    issued_by='test',
                )

                with pytest.raises(ValidationError, match='metadata resource exceeds'):
                    runtime.skills.register_skill_from_workspace_path(
                        pid,
                        'truncated-reference',
                        require_capability=False,
                    )
                assert runtime.store.get_skill('truncated-reference') is None
            finally:
                runtime.close()

    def test_workspace_skill_rejects_truncated_jit_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = 'scripts/check.ts'
            source_prefix = 'export default async function main() { return {ok: true}; }\n'
            skill_dir = write_skill_package(
                root,
                'truncated-jit-source',
                jit_tools=[
                    {
                        'name': 'check',
                        'description': 'Return a deterministic result.',
                        'source_path': source_path,
                    }
                ],
                scripts={source_path: source_prefix},
            )
            jit_manifest = skill_dir / 'references' / 'agent-libos' / 'jit-tools.json'
            read_limit = max(len(jit_manifest.read_bytes()), len(source_prefix.encode('utf-8')))
            padded_source = source_prefix.encode('utf-8').ljust(read_limit, b' ')
            (skill_dir / source_path).write_bytes(padded_source + b'// omitted suffix\n')
            config = AgentLibOSConfig(
                skills=replace(
                    SkillDefaults(),
                    resource_read_max_bytes=read_limit,
                    max_jit_source_chars=read_limit,
                )
            )
            runtime = Runtime.open(
                'local',
                substrate=LocalResourceProviderSubstrate(temp_dir),
                config=config,
            )
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='reject truncated Skill JIT')
                for path in (
                    'truncated-jit-source/SKILL.md',
                    'truncated-jit-source/references/agent-libos/jit-tools.json',
                    'truncated-jit-source/scripts/check.ts',
                ):
                    runtime.filesystem.grant_path(pid, path, [CapabilityRight.READ], issued_by='test')

                with pytest.raises(ValidationError, match='JIT source exceeds'):
                    runtime.skills.register_skill_from_workspace_path(
                        pid,
                        'truncated-jit-source',
                        require_capability=False,
                    )
                assert runtime.store.get_skill('truncated-jit-source') is None
            finally:
                runtime.close()

    def test_workspace_jit_source_character_limit_preserves_complete_multibyte_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = 'scripts/unicode.ts'
            source = (
                f"// {'界' * 20}\n"
                'export default async function main() { return {ok: true}; }\n'
            )
            skill_dir = write_skill_package(
                root,
                'unicode-jit-source',
                jit_tools=[
                    {
                        'name': 'unicode_check',
                        'description': 'Keep the complete Unicode source.',
                        'source_path': source_path,
                    }
                ],
                scripts={source_path: source},
            )
            jit_manifest = skill_dir / 'references' / 'agent-libos' / 'jit-tools.json'
            resource_limit = max(len(source.encode('utf-8')), len(jit_manifest.read_bytes()))
            assert len(source.encode('utf-8')) > len(source)
            config = AgentLibOSConfig(
                skills=replace(
                    SkillDefaults(),
                    resource_read_max_bytes=resource_limit,
                    max_jit_source_chars=len(source),
                ),
                # Skill packages have a fixed UTF-8 format and their hashes
                # must not depend on the general-purpose text-tool encoding.
                tools=replace(ToolDefaults(), default_text_encoding='utf-16'),
            )
            runtime = Runtime.open(
                'local',
                substrate=LocalResourceProviderSubstrate(temp_dir),
                config=config,
            )
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='snapshot Unicode Skill JIT')
                for path in (
                    'unicode-jit-source/SKILL.md',
                    'unicode-jit-source/references/agent-libos/jit-tools.json',
                    'unicode-jit-source/scripts/unicode.ts',
                ):
                    runtime.filesystem.grant_path(pid, path, [CapabilityRight.READ], issued_by='test')

                registered = runtime.skills.register_skill_from_workspace_path(
                    pid,
                    'unicode-jit-source',
                    require_capability=False,
                )
                package, _metadata = runtime.store.get_skill('unicode-jit-source')

                assert registered['skill_id'] == 'unicode-jit-source'
                assert package.jit_tools[0].source == source
                source_resource = next(
                    resource for resource in package.resources if resource.path == source_path
                )
                assert source_resource.size_bytes == len(source.encode('utf-8'))
                assert source_resource.content == source
            finally:
                runtime.close()

    def test_workspace_activate_failure_keeps_committed_write_one_shot_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_skill_package(Path(temp_dir), 'broken-skill', allowed_tools=['missing_workspace_tool'])
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='load broken workspace skill')
                runtime.filesystem.grant_path(pid, 'broken-skill/SKILL.md', [CapabilityRight.READ], issued_by='test')
                write_cap = runtime.capability.grant_once(
                    pid,
                    'skill:broken-skill',
                    [CapabilityRight.WRITE],
                    issued_by='test',
                )
                execute_cap = runtime.capability.grant_once(
                    pid,
                    'skill:broken-skill',
                    [CapabilityRight.EXECUTE],
                    issued_by='test',
                )

                with pytest.raises(NotFound, match='tool not found'):
                    runtime.skills.activate_skill_from_workspace_path(pid, 'broken-skill')

                discovered = runtime.skills.discover_skills(
                    text='broken-skill',
                    actor=pid,
                    require_capability=False,
                )

                assert runtime.store.get_skill('broken-skill') is not None
                assert [item['skill_id'] for item in discovered] == ['broken-skill']
                assert runtime.store.get_capability(write_cap.cap_id).uses_remaining == 0
                assert runtime.store.get_capability(execute_cap.cap_id).uses_remaining == 1
                assert 'broken-skill' not in runtime.process.get(pid).loaded_skills
            finally:
                runtime.close()

    def test_host_skill_package_rejects_hardlinked_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as outside:
            root = Path(temp_dir)
            skill_dir = write_skill_package(root, 'hardlink-skill', allowed_tools=['echo'], extra_resources={'references/guide.md': 'resource'})
            outside_file = Path(outside) / 'external-secret.txt'
            outside_file.write_text('external secret\n', encoding='utf-8')
            resource = skill_dir / 'references' / 'guide.md'
            resource.unlink()
            try:
                os.link(outside_file, resource)
            except OSError:
                pytest.skip('hardlink creation is not available in this environment')
            runtime = Runtime.open('local')
            try:
                with pytest.raises(ValidationError, match='hard links'):
                    runtime.skills.validate_package_path(skill_dir)
            finally:
                runtime.close()

    def test_package_hash_binds_instructions_and_resource_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_dir = write_skill_package(
                root,
                'integrity-skill',
                allowed_tools=['echo'],
                extra_resources={'references/guide.md': 'resource-v1\n'},
                body='# integrity-skill\n\ninstruction-v1\n',
            )
            runtime = Runtime.open('local')
            try:
                first = runtime.skills.validate_package_path(skill_dir)['package_sha256']

                write_skill_package(
                    root,
                    'integrity-skill',
                    allowed_tools=['echo'],
                    extra_resources={'references/guide.md': 'resource-v1\n'},
                    body='# integrity-skill\n\ninstruction-v2\n',
                )
                instruction_changed = runtime.skills.validate_package_path(skill_dir)['package_sha256']

                write_skill_package(
                    root,
                    'integrity-skill',
                    allowed_tools=['echo'],
                    extra_resources={'references/guide.md': 'resource-v2\n'},
                    body='# integrity-skill\n\ninstruction-v1\n',
                )
                resource_changed = runtime.skills.validate_package_path(skill_dir)['package_sha256']

                assert first != instruction_changed
                assert first != resource_changed
                assert instruction_changed != resource_changed
            finally:
                runtime.close()

    def test_loaded_skill_snapshot_hash_rejects_tampered_resource_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(
                Path(temp_dir),
                'tamper-skill',
                allowed_tools=['echo'],
                extra_resources={'references/guide.md': 'resource-v1\n'},
            )
            runtime = Runtime.open('local')
            try:
                package, _source = runtime.skills._load_package_from_host_path(skill_dir)
                snapshot = runtime.skills._skill_snapshot(package)
                for resource in snapshot['resources']:
                    if resource['path'] == 'references/guide.md':
                        resource['content'] = 'resource-v2\n'
                        break

                with pytest.raises(ValidationError, match='snapshot hash'):
                    runtime.skills._package_from_snapshot(snapshot, context='tampered skill')

                alternate_payload = runtime.skills._skill_snapshot(package)
                for resource in alternate_payload['resources']:
                    if resource['path'] == 'references/guide.md':
                        resource['content_base64'] = 'dW5oYXNoZWQtc2VjcmV0'
                        break

                with pytest.raises(
                    ValidationError,
                    match='must not contain content_base64',
                ):
                    runtime.skills._package_from_snapshot(
                        alternate_payload,
                        context='alternate-payload skill',
                    )
            finally:
                runtime.close()

    def test_skill_syscalls_use_primitive_capabilities_not_tool_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_skill_package(Path(temp_dir), 'syscall-skill', allowed_tools=['echo'])
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='syscall skill')
                process = runtime.process.get(pid)
                process.tool_table.pop('activate_skill', None)
                runtime.store.update_process(process)
                runtime.filesystem.grant_path(pid, 'syscall-skill/SKILL.md', [CapabilityRight.READ], issued_by='test')
                runtime.capability.grant(pid, 'skill:syscall-skill', [CapabilityRight.WRITE, CapabilityRight.EXECUTE], issued_by='test')
                registered = self._run(LibOSSyscallSession(runtime, pid).handle('skill.register_path', {'path': 'syscall-skill'}))
                loaded = self._run(
                    LibOSSyscallSession(runtime, pid).handle(
                        'skill.activate',
                        {
                            'skill_id': 'syscall-skill',
                            'expected_package_sha256': registered['package_sha256'],
                        },
                    )
                )
                assert registered['skill_id'] == 'syscall-skill'
                assert loaded['skill_id'] == 'syscall-skill'
                assert 'echo' in runtime.process.get(pid).tool_table
                with pytest.raises(NotFound):
                    self._run(LibOSSyscallSession(runtime, pid).handle('skill.register', {'skill': {'schema_version': 1, 'skill_id': 'inline-skill', 'name': 'inline-skill', 'description': 'Inline package should not be syscall-visible.', 'instructions': 'inline'}}))
            finally:
                runtime.close()

    def test_loaded_existing_tool_visibility_does_not_grant_resource_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'read-skill', allowed_tools=['read_text_file'])
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='load read tool')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'skill:read-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(pid, 'read-skill', actor=pid)
                result = runtime.tools.call(pid, 'read_text_file', {'path': 'secret.txt'})
                assert 'read_text_file' in runtime.process.get(pid).tool_table
                assert not runtime.capability.check(pid, 'filesystem:workspace:secret.txt', CapabilityRight.READ)
                assert not result.ok
                assert_public_error_message(
                    result.error,
                    code='permission_denied',
                    error_type='CapabilityDenied',
                    forbidden=('lacks read', 'secret.txt'),
                )
            finally:
                runtime.close()

    def test_read_skill_resource_requires_loaded_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'resource-skill', allowed_tools=['echo'], extra_resources={'references/guide.md': 'Remember resource-token.\n'})
            binary_path = skill_dir / 'assets' / 'sample.bin'
            binary_path.parent.mkdir(parents=True, exist_ok=True)
            binary_path.write_bytes(b'\xff\x00')
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='read resource')
                registered = runtime.skills.register_skill_from_path(
                    skill_dir,
                    actor='cli',
                    require_capability=False,
                )
                with pytest.raises(CapabilityDenied):
                    runtime.skills.read_skill_resource(pid, 'resource-skill', 'references/guide.md')
                runtime.capability.grant(pid, 'skill:resource-skill', [CapabilityRight.EXECUTE], issued_by='test')
                activated = runtime.tools.call(
                    pid,
                    'activate_skill',
                    {
                        'skill_id': 'resource-skill',
                        'expected_package_sha256': registered['package_sha256'],
                    },
                )
                assert activated.ok, activated.error
                assert set(activated.payload) == {'result'}
                assert set(activated.payload['result']) == {
                    'pid',
                    'skill_id',
                    'name',
                    'version',
                    'tool_names',
                    'tool_ids',
                    'jit_tool_ids',
                    'instructions_hash',
                    'package_sha256',
                }
                resource = runtime.skills.read_skill_resource(pid, 'resource-skill', 'references/guide.md')
                assert 'resource-token' in resource['content']
                assert resource['content_base64'] is None

                with pytest.raises(ValidationError, match='max_bytes must be >= 1'):
                    runtime.skills.read_skill_resource(
                        pid,
                        'resource-skill',
                        'references/guide.md',
                        max_bytes=0,
                    )

                tool_result = runtime.tools.call(
                    pid,
                    'read_skill_resource',
                    {
                        'skill_id': 'resource-skill',
                        'path': 'references/guide.md',
                    },
                )
                zero_limit = runtime.tools.call(
                    pid,
                    'read_skill_resource',
                    {
                        'skill_id': 'resource-skill',
                        'path': 'references/guide.md',
                        'max_bytes': 0,
                    },
                )
                binary_result = runtime.tools.call(
                    pid,
                    'read_skill_resource',
                    {
                        'skill_id': 'resource-skill',
                        'path': 'assets/sample.bin',
                    },
                )

                assert tool_result.ok, tool_result.error
                assert set(tool_result.payload) == {'resource'}
                assert tool_result.payload['resource']['content_base64'] is None
                assert 'resource-token' in tool_result.payload['resource']['content']
                assert not zero_limit.ok
                assert binary_result.ok, binary_result.error
                assert binary_result.payload['resource']['kind'] == 'base64'
                assert binary_result.payload['resource']['content'] is None
                assert binary_result.payload['resource']['content_base64'] == '/wA='

                unloaded = runtime.tools.call(
                    pid,
                    'unload_skill',
                    {'skill_id': 'resource-skill'},
                )
                assert unloaded.ok, unloaded.error
                assert set(unloaded.payload) == {'result'}
                assert set(unloaded.payload['result']) == {
                    'pid',
                    'skill_id',
                    'removed_tools',
                }
            finally:
                runtime.close()

    @pytest.mark.real_deno
    def test_multiplexed_prompt_hides_jit_resource_discovery_but_known_path_is_readable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(
                Path(temp_dir),
                'multiplexed-resource-skill',
                jit_tools=[
                    {
                        'name': 'multiplexed_contract_tool',
                        'description': 'Return a bounded deterministic result.',
                        'source_path': 'scripts/contract.ts',
                        'input_schema': {'type': 'object'},
                        'output_schema': {'type': 'object'},
                        'tests': [{'args': {}, 'expected': {'ok': True}}],
                    }
                ],
                scripts={
                    'scripts/contract.ts': (
                        'export function run(args, libos) { return {ok: true}; }\n'
                    )
                },
            )
            runtime = Runtime.open('local')
            try:
                image_id = 'multiplexed-skill-resource:v0'
                runtime.register_image(
                    AgentImage(
                        image_id=image_id,
                        name='multiplexed-skill-resource',
                        default_tools=['process_exit'],
                        jit_tool_exposure='multiplexed',
                    ),
                    actor='test',
                )
                runtime.skills.register_skill_from_path(
                    skill_dir,
                    actor='cli',
                    require_capability=False,
                )
                pid = runtime.process.spawn(
                    image=image_id,
                    goal='read one already-known loaded resource path',
                )
                runtime.capability.grant(
                    pid,
                    'skill:multiplexed-resource-skill',
                    [CapabilityRight.EXECUTE],
                    issued_by='test',
                )
                runtime.skills.activate_skill(
                    pid,
                    'multiplexed-resource-skill',
                    actor=pid,
                )

                context = next(
                    item
                    for item in runtime.skills.prompt_context(pid)
                    if item['skill_id'] == 'multiplexed-resource-skill'
                )
                visible_paths = {item['path'] for item in context['resources']}
                known = runtime.skills.read_skill_resource(
                    pid,
                    'multiplexed-resource-skill',
                    'references/agent-libos/jit-tools.json',
                )

                assert context['jit_tools'] == []
                assert 'references/agent-libos/jit-tools.json' not in visible_paths
                assert 'scripts/contract.ts' not in visible_paths
                assert 'multiplexed_contract_tool' in known['content']
            finally:
                runtime.close()

    def test_loaded_skill_uses_activation_snapshot_after_registry_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_dir = write_skill_package(
                root,
                'snapshot-skill',
                allowed_tools=['echo'],
                extra_resources={'references/guide.md': 'original-resource-token\n'},
                body='# snapshot-skill\n\nUse original-instruction-token.\n',
            )
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='snapshot skill')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'skill:snapshot-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(pid, 'snapshot-skill', actor=pid)

                write_skill_package(
                    root,
                    'snapshot-skill',
                    allowed_tools=['human_output'],
                    extra_resources={'references/guide.md': 'replaced-resource-token\n'},
                    body='# snapshot-skill\n\nUse replaced-instruction-token.\n',
                )
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', replace=True, require_capability=False)

                context = next(
                    item
                    for item in runtime.skills.prompt_context(pid)
                    if item['skill_id'] == 'snapshot-skill'
                )
                resource = runtime.skills.read_skill_resource(pid, 'snapshot-skill', 'references/guide.md')

                assert 'original-instruction-token' in context['instructions']
                assert 'replaced-instruction-token' not in context['instructions']
                assert context['allowed_tools'] == ['echo']
                assert resource['content'].replace('\r\n', '\n') == 'original-resource-token\n'
            finally:
                runtime.close()

    def test_reopen_marks_replaced_package_inactive_and_activation_uses_hash_cas(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = root / 'skill-replace-reopen.sqlite'
            skill_id = 'reopen-cas-skill'
            skill_dir = write_skill_package(
                root,
                skill_id,
                allowed_tools=['echo'],
                body='# reopen-cas-skill\n\nUse package A.\n',
            )
            runtime = Runtime.open(database)
            try:
                pid = runtime.process.spawn(
                    image='coding-agent:v0',
                    goal='activate only the discovered Skill content',
                )
                package_a = runtime.skills.register_skill_from_path(
                    skill_dir,
                    actor='test.host',
                    require_capability=False,
                )
                runtime.skills.activate_skill(
                    pid,
                    skill_id,
                    actor=pid,
                    require_capability=False,
                    expected_package_sha256=package_a['package_sha256'],
                )
                write_skill_package(
                    root,
                    skill_id,
                    allowed_tools=['echo'],
                    body='# reopen-cas-skill\n\nUse package B.\n',
                )
                package_b = runtime.skills.register_skill_from_path(
                    skill_dir,
                    actor='test.host',
                    replace=True,
                    require_capability=False,
                )
                assert package_b['package_sha256'] != package_a['package_sha256']
                runtime.capability.grant(
                    pid,
                    runtime.config.skills.registry_resource,
                    [CapabilityRight.READ],
                    issued_by='test.host',
                )
                execute = runtime.capability.grant_once(
                    pid,
                    runtime.skills.resource_for(skill_id),
                    [CapabilityRight.EXECUTE],
                    issued_by='test.host',
                )
            finally:
                runtime.close()

            reopened = Runtime.open(database)
            try:
                discovered = reopened.tools.call(
                    pid,
                    'discover_skills',
                    {'text': skill_id, 'limit': 1},
                )
                assert discovered.ok
                summary = discovered.payload['skills'][0]
                assert summary['package_sha256'] == package_b['package_sha256']
                assert summary['active'] is False
                assert discovered.payload['next_step'] == 'activate_skill'

                before = reopened.process.get(pid)
                before_loaded = deepcopy(before.loaded_skills)
                before_tool_table = dict(before.tool_table)
                before_model_tool_table = dict(before.model_tool_table)
                before_candidates = reopened.store.select_table_rows('tool_candidates')
                before_event_ids = {event.event_id for event in reopened.events.list()}
                before_audit_ids = {record.record_id for record in reopened.audit.trace()}
                prepare_calls = 0
                original_prepare = reopened.skills._prepare_jit_tools

                def observe_prepare(*args: Any, **kwargs: Any) -> Any:
                    nonlocal prepare_calls
                    prepare_calls += 1
                    return original_prepare(*args, **kwargs)

                monkeypatch.setattr(reopened.skills, '_prepare_jit_tools', observe_prepare)
                stale = reopened.tools.call(
                    pid,
                    'activate_skill',
                    {
                        'skill_id': skill_id,
                        'expected_package_sha256': package_a['package_sha256'],
                    },
                )

                assert not stale.ok
                assert (
                    stale.payload['error']['details']['error_type']
                    == 'SkillPackageChanged'
                )
                after_stale = reopened.process.get(pid)
                assert after_stale.loaded_skills == before_loaded
                assert after_stale.tool_table == before_tool_table
                assert after_stale.model_tool_table == before_model_tool_table
                assert reopened.store.select_table_rows('tool_candidates') == before_candidates
                assert reopened.store.get_capability(execute.cap_id).uses_remaining == 1
                assert prepare_calls == 0
                assert not [
                    event
                    for event in reopened.events.list()
                    if event.event_id not in before_event_ids
                    and event.type == EventType.SKILL_LOADED
                ]
                assert not [
                    record
                    for record in reopened.audit.trace()
                    if record.record_id not in before_audit_ids
                    and record.action == 'skill.activate'
                ]

                activated = reopened.tools.call(
                    pid,
                    'activate_skill',
                    {
                        'skill_id': skill_id,
                        'expected_package_sha256': package_b['package_sha256'],
                    },
                )
                assert activated.ok
                assert activated.payload['result']['package_sha256'] == package_b['package_sha256']
                assert reopened.store.get_capability(execute.cap_id).uses_remaining == 0
                assert prepare_calls == 1

                current = reopened.tools.call(
                    pid,
                    'discover_skills',
                    {'text': skill_id, 'limit': 1},
                )
                assert current.ok
                assert current.payload['skills'][0]['active'] is True
                assert current.payload['next_step'] == 'use_loaded_skill'
            finally:
                reopened.close()

    def test_checkpoint_restore_and_fork_do_not_resurrect_global_skill_trust(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            global_dir = root / 'global'
            skill_dir = write_skill_package(global_dir, 'trust-checkpoint-skill', allowed_tools=['echo'])
            config = AgentLibOSConfig(skills=replace(SkillDefaults(), global_dirs=(str(global_dir),)))
            runtime = Runtime.open('local', config=config)
            try:
                trust = runtime.skills.global_package_info(skill_dir)
                runtime.skills.trust_skill_source(
                    actor='cli',
                    source_type='global',
                    source=trust['source'],
                    package_sha256=trust['package_sha256'],
                    require_capability=False,
                )
                runtime.skills.register_global_skill_from_path(skill_dir, actor='cli', require_capability=False)
                pid = runtime.process.spawn(image='base-agent:v0', goal='checkpoint trust')
                runtime.capability.grant(pid, 'skill:trust-checkpoint-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(pid, 'trust-checkpoint-skill', actor=pid)
                checkpoint_id = runtime.checkpoint.create(pid, 'trusted skill loaded', actor=pid)

                runtime.skills.untrust_skill_source(
                    actor='cli',
                    source_type='global',
                    source=trust['source'],
                    package_sha256=trust['package_sha256'],
                    require_capability=False,
                )
                assert not runtime.store.is_skill_trusted(
                    source_type='global',
                    source=trust['source'],
                    package_sha256=trust['package_sha256'],
                )

                runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
                assert not runtime.store.is_skill_trusted(
                    source_type='global',
                    source=trust['source'],
                    package_sha256=trust['package_sha256'],
                )
                runtime.checkpoint.fork_from_checkpoint('cli', checkpoint_id, require_capability=False)
                assert not runtime.store.is_skill_trusted(
                    source_type='global',
                    source=trust['source'],
                    package_sha256=trust['package_sha256'],
                )
            finally:
                runtime.close()

    def test_cross_process_skill_activate_requires_target_process_admin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'cross-load-skill', allowed_tools=['echo'])
            runtime = Runtime.open('local')
            try:
                actor = runtime.process.spawn(image='base-agent:v0', goal='actor')
                target = runtime.process.spawn(image='base-agent:v0', goal='target')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(actor, 'skill:cross-load-skill', [CapabilityRight.EXECUTE], issued_by='test')
                with pytest.raises(CapabilityDenied):
                    runtime.skills.activate_skill(target, 'cross-load-skill', actor=actor)
                runtime.capability.grant(actor, f'process:{target}', [CapabilityRight.ADMIN], issued_by='test')
                loaded = runtime.skills.activate_skill(target, 'cross-load-skill', actor=actor)
                assert loaded['pid'] == target
                assert 'echo' in runtime.process.get(target).tool_table
            finally:
                runtime.close()

    def test_cross_process_failed_activation_restores_execute_and_admin_one_shots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'cross-fail-skill', allowed_tools=['missing_cross_tool'])
            runtime = Runtime.open('local')
            try:
                actor = runtime.process.spawn(image='base-agent:v0', goal='actor')
                target = runtime.process.spawn(image='base-agent:v0', goal='target')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                execute_cap = runtime.capability.grant_once(
                    actor,
                    'skill:cross-fail-skill',
                    [CapabilityRight.EXECUTE],
                    issued_by='test',
                )
                admin_cap = runtime.capability.grant_once(
                    actor,
                    f'process:{target}',
                    [CapabilityRight.ADMIN],
                    issued_by='test',
                )

                with pytest.raises(NotFound, match='tool not found'):
                    runtime.skills.activate_skill(target, 'cross-fail-skill', actor=actor)

                assert runtime.store.get_capability(execute_cap.cap_id).uses_remaining == 1
                assert runtime.store.get_capability(admin_cap.cap_id).uses_remaining == 1
                assert 'cross-fail-skill' not in runtime.process.get(target).loaded_skills
            finally:
                runtime.close()

    def test_unload_skill_consumes_one_time_execute_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'unload-skill', allowed_tools=['echo'])
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='unload skill')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.activate_skill(pid, 'unload-skill')
                runtime.capability.grant_once(pid, 'skill:unload-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.unload_skill(pid, 'unload-skill', actor=pid)
                assert not runtime.capability.check(pid, 'skill:unload-skill', CapabilityRight.EXECUTE)
                assert 'echo' not in runtime.process.get(pid).tool_table
            finally:
                runtime.close()

    def test_unload_skill_restores_same_tool_from_process_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'image-shared-skill', allowed_tools=['echo'])
            runtime = Runtime.open('local')
            try:
                image_id = 'skill-shared-image:v0'
                runtime.register_image(
                    AgentImage(
                        image_id=image_id,
                        name='skill-shared-image',
                        default_tools=['echo'],
                    ),
                    actor='test',
                )
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                pid = runtime.process.spawn(image=image_id, goal='preserve image tool after skill unload')
                before_tool_table = dict(runtime.process.get(pid).tool_table)
                before_model_tool_table = dict(runtime.process.get(pid).model_tool_table)

                runtime.activate_skill(pid, 'image-shared-skill')
                runtime.unload_skill(pid, 'image-shared-skill')

                restored = runtime.process.get(pid)
                assert restored.tool_table == before_tool_table
                assert restored.model_tool_table == before_model_tool_table
            finally:
                runtime.close()

    def test_unload_one_of_two_skills_keeps_shared_tool_until_last_source_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_dir = write_skill_package(root, 'shared-tool-first', allowed_tools=['echo'])
            second_dir = write_skill_package(root, 'shared-tool-second', allowed_tools=['echo'])
            runtime = Runtime.open('local')
            try:
                runtime.skills.register_skill_from_path(first_dir, actor='cli', require_capability=False)
                runtime.skills.register_skill_from_path(second_dir, actor='cli', require_capability=False)
                pid = runtime.process.spawn(image='base-agent:v0', goal='shared skill tool provenance')
                runtime.activate_skill(pid, 'shared-tool-first')
                runtime.activate_skill(pid, 'shared-tool-second')

                runtime.unload_skill(pid, 'shared-tool-first')

                after_first_unload = runtime.process.get(pid)
                assert 'shared-tool-second' in after_first_unload.loaded_skills
                assert 'echo' in after_first_unload.tool_table
                assert 'echo' in after_first_unload.model_tool_table

                runtime.unload_skill(pid, 'shared-tool-second')
                after_last_unload = runtime.process.get(pid)
                assert 'echo' not in after_last_unload.tool_table
                assert 'echo' not in after_last_unload.model_tool_table
            finally:
                runtime.close()

    @pytest.mark.parametrize("base_source", ["image", "manual"])
    def test_unload_rejects_noncanonical_persisted_skill_provenance(
        self,
        tmp_path: Path,
        base_source: str,
    ) -> None:
        skill_dir = write_skill_package(tmp_path, f'legacy-{base_source}-skill', allowed_tools=['echo'])
        database = tmp_path / f'legacy-{base_source}-skill.sqlite'
        runtime = Runtime.open(database)
        try:
            image_id = 'base-agent:v0'
            if base_source == 'image':
                image_id = 'legacy-skill-base-image:v0'
                runtime.register_image(
                    AgentImage(
                        image_id=image_id,
                        name='legacy-skill-base-image',
                        default_tools=['echo'],
                    ),
                    actor='test',
                )
            runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
            pid = runtime.process.spawn(image=image_id, goal='strict Skill provenance')
            if base_source == 'manual':
                runtime.tools.configure_process_tools(pid, ['echo'], assigned_by='test:manual-base')
            runtime.activate_skill(pid, f'legacy-{base_source}-skill')
            process = runtime.process.get(pid)
            loaded = dict(process.loaded_skills[f'legacy-{base_source}-skill'])
            loaded.pop('base_tool_ids', None)
            loaded.pop('base_model_tool_ids', None)
            process.loaded_skills[f'legacy-{base_source}-skill'] = loaded
            runtime.store.update_process(process)
        finally:
            runtime.close()

        reopened = Runtime.open(database)
        try:
            with pytest.raises(ValidationError, match='canonical tool provenance'):
                reopened.unload_skill(pid, f'legacy-{base_source}-skill')

            process = reopened.process.get(pid)
            assert f'legacy-{base_source}-skill' in process.loaded_skills
            assert 'echo' in process.tool_table
            assert 'echo' in process.model_tool_table
        finally:
            reopened.close()

    def _run(self, awaitable: Any) -> Any:
        return asyncio.run(awaitable)


def _programmatic_jit_skill(skill_id: str, timeout_s: Any) -> SkillPackage:
    return SkillPackage(
        skill_id=skill_id,
        name=skill_id,
        description='Programmatic Skill JIT timeout validation.',
        instructions='Use the programmatic JIT tool.',
        jit_tools=[
            JitToolSpec(
                name=f'{skill_id}_tool'.replace('-', '_'),
                description='Return a deterministic result.',
                source_path='scripts/check.ts',
                source=(
                    'export async function run(args: unknown, libos: unknown) '
                    '{ return {ok: true}; }\n'
                ),
                timeout_s=timeout_s,
            )
        ],
    )


def _pathological_json_payloads() -> tuple[str, str]:
    oversized_integer = '9' * 5_000
    excessively_nested = ('[' * 2_000) + '0' + (']' * 2_000)
    return oversized_integer, excessively_nested
