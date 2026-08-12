from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event as ThreadEvent
import pytest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from agent_libos import Runtime
from agent_libos.api.cli import main as cli_main
from agent_libos.capability.effect_binding import canonical_effect_hash
from agent_libos.models import Capability, CapabilityEffect, CapabilityRight, CapabilitySpec, CapabilityStatus, DelegationPolicy, EventType, TaskAuthorityManifest
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.tools.base import ToolContext, ToolErrorCode
from agent_libos.tools.builtin.capabilities import DelegateCapabilityTool


def _grant_process_spawn(runtime: Runtime, pid: str) -> None:
    runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')


def _capability_admission_state(runtime: Runtime, pid: str) -> tuple[object, ...]:
    return (
        runtime.store.list_capabilities(subject=pid),
        [record.record_id for record in runtime.store.list_audit()],
        [event.event_id for event in runtime.events.list()],
        runtime.store.select_table_rows(
            'capability_use_reservations',
            order_by='reservation_id',
        ),
    )


def _inject_legacy_capability_constraints(
    runtime: Runtime,
    capability: Capability,
    constraints: dict[str, object],
) -> Capability:
    """Simulate a pre-admission capability row without using a public writer."""

    injected = replace(capability, constraints=dict(constraints))
    runtime.store.update_capability(injected)
    return injected


def _assert_rule_rejected_at_admission(
    runtime: Runtime,
    pid: str,
    rule: dict[str, object],
    *,
    error_match: str,
) -> None:
    runtime.capability.grant(
        pid,
        'shell:git',
        [CapabilityRight.EXECUTE],
        issued_by='test',
    )
    finite_grant = runtime.capability.grant_once(
        pid,
        'shell:git',
        [CapabilityRight.GRANT],
        issued_by='test',
    )
    before = _capability_admission_state(runtime, pid)

    with pytest.raises(ValidationError, match=error_match):
        runtime.capability.issue(
            pid,
            pid,
            CapabilitySpec(
                resource='shell:git',
                rights={CapabilityRight.EXECUTE.value},
                constraints={'authority_rules': [rule]},
            ),
        )

    assert _capability_admission_state(runtime, pid) == before
    persisted_grant = runtime.store.get_capability(finite_grant.cap_id)
    assert persisted_grant is not None
    assert persisted_grant.uses_remaining == 1


class TestCapabilityManager:

    def test_typed_resource_matching_rejects_prefix_collision(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='resource matching')
            runtime.capability.grant(pid, 'filesystem:workspace:src/*', [CapabilityRight.READ], issued_by='test')
            assert runtime.capability.check(pid, 'filesystem:workspace:src/main.py', CapabilityRight.READ)
            assert runtime.capability.check(pid, 'filesystem:workspace:src', CapabilityRight.READ)
            assert not runtime.capability.check(pid, 'filesystem:workspace:src2/main.py', CapabilityRight.READ)
            with pytest.raises(CapabilityDenied):
                runtime.capability.parse_resource_pattern('filesystem:workspace:src*')
            with pytest.raises(CapabilityDenied):
                runtime.capability.grant(pid, '*', [CapabilityRight.READ], issued_by='test')
            with pytest.raises(ValidationError):
                runtime.capability.grant(pid, 'filesystem:workspace:src/main.py', ['*'], issued_by='test')
        finally:
            runtime.close()

    def test_deny_dominates_matching_allow(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='deny dominates')
            runtime.capability.grant(pid, 'filesystem:workspace:*', [CapabilityRight.READ], issued_by='test')
            runtime.capability.issue_trusted(pid, 'filesystem:workspace:secret.txt', [CapabilityRight.READ], issued_by='test', effect=CapabilityEffect.DENY)
            decision = runtime.capability.authorize(pid, 'filesystem:workspace:secret.txt', CapabilityRight.READ)
            assert not decision.allowed
            assert decision.effect == CapabilityEffect.DENY
            assert decision.matched_capability_ids
        finally:
            runtime.close()

    def test_deny_dominates_unordered_matching_capability_candidates(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='deny dominates unordered')
            deny = runtime.capability.issue_trusted(
                pid,
                'shell:git',
                [CapabilityRight.EXECUTE],
                issued_by='test',
                effect=CapabilityEffect.DENY,
            )
            allow = runtime.capability.grant(pid, 'shell:git', [CapabilityRight.EXECUTE], issued_by='test')

            decision = runtime.capability.authorize_matching_capabilities(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                [allow, deny],
            )

            assert not decision.allowed
            assert decision.effect == CapabilityEffect.DENY
            assert decision.selected_capability_id == deny.cap_id
        finally:
            runtime.close()

    @pytest.mark.parametrize("restrictive_rule_first", [True, False])
    def test_deny_authority_rule_inside_allow_capability_dominates_across_issuance_order(
        self,
        restrictive_rule_first: bool,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='rule deny precedence')

            def issue_restrictive_rule() -> Capability:
                return runtime.capability.issue_trusted(
                    pid,
                    'shell:git',
                    [CapabilityRight.EXECUTE],
                    issued_by='test',
                    constraints={
                        'authority_rules': [
                            {
                                'rule_id': 'test.git.push.cross-capability-deny',
                                'operation': 'shell.run',
                                'effect': 'deny',
                                'risk': 'high',
                                'conditions': {'argv': ['git', 'push'], 'match': 'prefix'},
                            }
                        ]
                    },
                )

            if restrictive_rule_first:
                restrictive = issue_restrictive_rule()
                runtime.capability.grant(pid, 'shell:git', [CapabilityRight.EXECUTE], issued_by='test')
            else:
                runtime.capability.grant(pid, 'shell:git', [CapabilityRight.EXECUTE], issued_by='test')
                restrictive = issue_restrictive_rule()

            decision = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {
                    'operation': 'shell.run',
                    'authority_operation': 'shell.run',
                    'argv': ['git', 'push', 'origin'],
                },
            )

            assert not decision.allowed
            assert decision.effect == CapabilityEffect.DENY
            assert decision.selected_capability_id == restrictive.cap_id
        finally:
            runtime.close()

    @pytest.mark.parametrize("ask_first", [True, False])
    def test_ask_dominates_allow_across_issuance_order(self, ask_first: bool) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='ask precedence')

            def issue_ask() -> Capability:
                return runtime.capability.issue_trusted(
                    pid,
                    'shell:tool',
                    [CapabilityRight.EXECUTE],
                    issued_by='test',
                    effect=CapabilityEffect.ASK,
                )

            if ask_first:
                ask = issue_ask()
                runtime.capability.grant(pid, 'shell:tool', [CapabilityRight.EXECUTE], issued_by='test')
            else:
                runtime.capability.grant(pid, 'shell:tool', [CapabilityRight.EXECUTE], issued_by='test')
                ask = issue_ask()

            decision = runtime.capability.authorize(pid, 'shell:tool', CapabilityRight.EXECUTE)

            assert not decision.allowed
            assert decision.effect == CapabilityEffect.ASK
            assert decision.selected_capability_id == ask.cap_id
        finally:
            runtime.close()

    def test_precedence_keeps_finite_allow_selection_and_consumption(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='finite allow selection')
            runtime.capability.grant(pid, 'object:finite-selection', [CapabilityRight.READ], issued_by='test')
            finite = runtime.capability.issue_trusted(
                pid,
                'object:finite-selection',
                [CapabilityRight.READ],
                issued_by='test',
                uses_remaining=1,
            )

            decision = runtime.capability.authorize(pid, finite.resource, CapabilityRight.READ)

            assert decision.allowed
            assert decision.selected_capability_id == finite.cap_id
            assert decision.consume_capability_id == finite.cap_id
        finally:
            runtime.close()

    def test_exact_one_shot_approval_binding_resolves_ask_and_is_selected_for_consumption(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='bound approval precedence')
            resource = 'mcp:approved:echo'
            context = {
                'operation': 'mcp.call',
                'authority_operation': 'mcp.call',
                'resource': resource,
                'right': CapabilityRight.READ.value,
                'registry_generation': 2,
            }
            runtime.capability.set_permission_policy(
                pid,
                resource,
                [CapabilityRight.READ],
                runtime.capability.ASK_EACH_TIME,
                issued_by='test',
            )
            approval = runtime.capability.issue_trusted(
                pid,
                resource,
                [CapabilityRight.READ],
                issued_by='human',
                uses_remaining=1,
                constraints={
                    runtime.capability.APPROVAL_BINDING_KEY: {
                        'effect_id': 'eff_exact_approval',
                        'canonical_args_hash': canonical_effect_hash(context),
                        'target_state_version': None,
                    }
                },
            )

            decision = runtime.capability.authorize(pid, resource, CapabilityRight.READ, context)

            assert decision.allowed
            assert decision.selected_capability_id == approval.cap_id
            assert decision.consume_capability_id == approval.cap_id
            assert decision.constraint_results[runtime.capability.APPROVAL_BINDING_KEY]['ok']
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('approved_context', 'actual_context'),
        [
            (
                {
                    'operation': 'mcp.call',
                    'authority_operation': 'mcp.call',
                    'resource': 'mcp:approved:echo',
                    'right': 'read',
                },
                {
                    'operation': 'mcp.list_tools',
                    'authority_operation': 'mcp.list_tools',
                    'resource': 'mcp:approved:echo',
                    'right': 'read',
                },
            ),
            (
                {
                    'operation': 'mcp.call',
                    'authority_operation': 'mcp.call',
                    'resource': 'mcp:approved:echo',
                    'right': 'read',
                    'registry_generation': 1,
                },
                {
                    'operation': 'mcp.call',
                    'authority_operation': 'mcp.call',
                    'resource': 'mcp:approved:echo',
                    'right': 'read',
                    'registry_generation': 2,
                },
            ),
        ],
    )
    def test_stale_or_other_operation_approval_binding_cannot_resolve_ask(
        self,
        approved_context: dict[str, object],
        actual_context: dict[str, object],
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='stale approval rejection')
            resource = 'mcp:approved:echo'
            ask = runtime.capability.set_permission_policy(
                pid,
                resource,
                [CapabilityRight.READ],
                runtime.capability.ASK_EACH_TIME,
                issued_by='test',
            )
            runtime.capability.issue_trusted(
                pid,
                resource,
                [CapabilityRight.READ],
                issued_by='human',
                uses_remaining=1,
                constraints={
                    runtime.capability.APPROVAL_BINDING_KEY: {
                        'effect_id': 'eff_stale_approval',
                        'canonical_args_hash': canonical_effect_hash(approved_context),
                        'target_state_version': None,
                    }
                },
            )

            decision = runtime.capability.authorize(
                pid,
                resource,
                CapabilityRight.READ,
                actual_context,
            )

            assert not decision.allowed
            assert decision.effect == CapabilityEffect.ASK
            assert decision.selected_capability_id == ask.cap_id
            assert decision.consume_capability_id is None
        finally:
            runtime.close()

    def test_matching_capability_candidates_cannot_cross_subject_boundary(self) -> None:
        runtime = Runtime.open('local')
        try:
            requester = runtime.process.spawn(image='base-agent:v0', goal='requester')
            other = runtime.process.spawn(image='base-agent:v0', goal='other subject')
            borrowed = runtime.capability.grant(
                other,
                'shell:git',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )

            decision = runtime.capability.authorize_matching_capabilities(
                requester,
                'shell:git',
                CapabilityRight.EXECUTE,
                [borrowed],
            )

            assert not decision.allowed
            assert decision.selected_capability_id is None
        finally:
            runtime.close()

    def test_human_capability_decision_rejects_cross_subject_override(self) -> None:
        runtime = Runtime.open('local')
        try:
            requester = runtime.process.spawn(image='base-agent:v0', goal='request grant')
            victim = runtime.process.spawn(image='base-agent:v0', goal='unrelated victim')
            with pytest.raises(
                ValidationError,
                match='generic human query cannot contain authority-shaping fields',
            ):
                runtime.human.query(
                    requester,
                    'owner',
                    {
                        'type': 'approval',
                        'question': 'grant requested authority',
                        'requested_capability': {
                            'subject': victim,
                            'resource': 'object:cross-subject',
                            'rights': ['read'],
                        },
                    },
                    blocking=False,
                )

            assert not runtime.capability.check(
                victim,
                'object:cross-subject',
                CapabilityRight.READ,
            )
            assert runtime.human.list(requester) == []
        finally:
            runtime.close()

    def test_trusted_actor_names_do_not_bypass_issue_or_revoke_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            subject = runtime.process.spawn(image='base-agent:v0', goal='trusted actor spoof subject')
            resource = 'object:trusted-actor-spoof'

            with pytest.raises(CapabilityDenied, match='lacks grant/admin authority'):
                runtime.capability.issue(
                    'human:owner',
                    subject,
                    {'resource': resource, 'rights': ['read']},
                    require_authority=True,
                )

            cap = runtime.capability.grant(subject, resource, [CapabilityRight.READ], issued_by='test')
            with pytest.raises(CapabilityDenied, match='lacks revoke/admin authority'):
                runtime.capability.revoke(cap.cap_id, revoked_by='human:owner')

            assert runtime.store.get_capability(cap.cap_id).active
        finally:
            runtime.close()

    def test_restrictive_capability_with_bad_constraint_fails_closed_over_allow(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='malformed restrictive policy')
            runtime.capability.grant(pid, 'filesystem:workspace:*', [CapabilityRight.READ], issued_by='test')
            restrictive = runtime.capability.issue_trusted(
                pid,
                'filesystem:workspace:secret.txt',
                [CapabilityRight.READ],
                issued_by='test',
                effect=CapabilityEffect.DENY,
            )
            _inject_legacy_capability_constraints(
                runtime,
                restrictive,
                {'unknown_constraint': True},
            )
            decision = runtime.capability.authorize(pid, 'filesystem:workspace:secret.txt', CapabilityRight.READ)
            assert not decision.allowed
            assert decision.effect == CapabilityEffect.DENY
            assert 'invalid_persisted_constraint' in decision.constraint_results
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        'constraints',
        [
            {'unknown_constraint': True},
            {'git_remote': None},
            {'git_remote': 7},
            {'git_allowed_refs': 'refs/heads/main'},
            {'git_allowed_refs': ['refs/heads/main', None]},
            {'authority_rules': None},
            {'approval_binding': None},
            {'data_release_binding': None},
            {'shell_policy_level': None},
            {'inherited_from': None},
        ],
        ids=[
            'unknown',
            'git-remote-null',
            'git-remote-type',
            'git-refs-container',
            'git-refs-item',
            'authority-rules-null',
            'approval-binding-null',
            'data-release-binding-null',
            'shell-policy-null',
            'inherited-from-null',
        ],
    )
    def test_new_capability_writes_reject_undefined_constraints(
        self,
        constraints: dict[str, object],
    ) -> None:
        runtime = Runtime.open('local')
        try:
            before = _capability_admission_state(runtime, 'worker')

            with pytest.raises(ValidationError, match='constraint|binding|rules'):
                runtime.capability.issue_trusted(
                    'worker',
                    'object:strict-constraints',
                    [CapabilityRight.READ],
                    issued_by='test',
                    constraints=constraints,
                )

            assert _capability_admission_state(runtime, 'worker') == before
        finally:
            runtime.close()

    def test_one_shot_capability_is_consumed_after_successful_use(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='one shot')
            runtime.capability.grant_once(pid, 'filesystem:workspace:once.txt', [CapabilityRight.WRITE], issued_by='test')
            assert runtime.capability.permission_policy(pid, 'filesystem:workspace:once.txt', CapabilityRight.WRITE) == runtime.capability.ALLOW_ONCE
            runtime.capability.consume_allow_once(pid, 'filesystem:workspace:once.txt', CapabilityRight.WRITE, used_by=pid)
            assert not runtime.capability.check(pid, 'filesystem:workspace:once.txt', CapabilityRight.WRITE)
        finally:
            runtime.close()

    def test_structured_rules_lease_and_delegation_are_canonicalized(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='structured capability spec')
            cap = runtime.capability.issue(
                'test',
                pid,
                {
                    'resource': 'shell:git',
                    'rights': ['execute'],
                    'rules': [
                        {
                            'rule_id': 'test.git.status',
                            'operation': 'shell.run',
                            'effect': 'allow',
                            'risk': 'harmless',
                            'conditions': {'argv': ['git', 'status'], 'match': 'exact'},
                        }
                    ],
                    'lease': {'uses_remaining': 1},
                    'delegation': {'delegable': True, 'revocable': False},
                    'metadata': {'purpose': 'structured spec'},
                },
                require_authority=False,
            )
            inspected = runtime.capability.inspect(cap.cap_id)
            assert inspected['rules'][0]['rule_id'] == 'test.git.status'
            assert inspected['rules'][0]['risk'] == 'harmless'
            assert inspected['constraints']['authority_rules'][0]['effect'] == 'allow'
            assert inspected['lease']['uses_remaining'] == 1
            assert inspected['delegation']['delegable']
            assert not inspected['delegation']['revocable']
            assert runtime.capability.permission_policy(pid, 'shell:git', CapabilityRight.EXECUTE) == runtime.capability.MISSING
            assert runtime.capability.permission_policy(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'argv': ['git', 'status']},
            ) == runtime.capability.ALLOW_ONCE
        finally:
            runtime.close()

    def test_capability_use_lease_requires_a_positive_json_integer(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='reject malformed finite-use leases',
            )
            invalid_values = [True, 0, -1, 1.5, float('nan'), float('inf')]
            for index, uses_remaining in enumerate(invalid_values):
                before = runtime.store.list_capabilities(subject=pid)

                with pytest.raises(ValidationError, match='uses_remaining'):
                    runtime.capability.issue_trusted(
                        pid,
                        f'object:invalid-direct-lease:{index}',
                        [CapabilityRight.READ],
                        issued_by='test',
                        uses_remaining=uses_remaining,
                    )
                with pytest.raises(ValidationError, match='uses_remaining'):
                    runtime.capability.issue(
                        'test',
                        pid,
                        {
                            'resource': f'object:invalid-structured-lease:{index}',
                            'rights': ['read'],
                            'lease': {'uses_remaining': uses_remaining},
                        },
                        require_authority=False,
                    )

                assert runtime.store.list_capabilities(subject=pid) == before
        finally:
            runtime.close()

    def test_delegation_policy_fields_require_exact_types_before_issue_side_effects(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='strict delegation fields')
            invalid_specs: list[CapabilitySpec | dict[str, object]] = [
                {'resource': 'object:invalid-top-delegable', 'rights': ['read'], 'delegable': 'false'},
                {'resource': 'object:invalid-top-revocable-zero', 'rights': ['read'], 'revocable': 0},
                {'resource': 'object:invalid-top-revocable-one', 'rights': ['read'], 'revocable': 1},
                {'resource': 'object:invalid-top-depth-bool', 'rights': ['read'], 'max_delegation_depth': True},
                {'resource': 'object:invalid-top-depth-float', 'rights': ['read'], 'max_delegation_depth': 1.0},
                {'resource': 'object:invalid-top-depth-string', 'rights': ['read'], 'max_delegation_depth': '1'},
                {'resource': 'object:invalid-top-depth-negative', 'rights': ['read'], 'max_delegation_depth': -1},
                {
                    'resource': 'object:invalid-nested-delegable',
                    'rights': ['read'],
                    'delegation': {'delegable': 'false'},
                },
                {
                    'resource': 'object:invalid-nested-revocable',
                    'rights': ['read'],
                    'delegation': {'revocable': 1},
                },
                {
                    'resource': 'object:invalid-nested-depth',
                    'rights': ['read'],
                    'delegation': {'max_delegation_depth': True},
                },
                CapabilitySpec(
                    resource='object:invalid-spec-instance',
                    rights={CapabilityRight.READ.value},
                    delegable='false',  # type: ignore[arg-type]
                ),
                CapabilitySpec(
                    resource='object:invalid-policy-instance',
                    rights={CapabilityRight.READ.value},
                    delegation=DelegationPolicy(
                        delegable='false',  # type: ignore[arg-type]
                        max_delegation_depth=1.0,  # type: ignore[arg-type]
                    ),
                ),
            ]

            for spec in invalid_specs:
                before_caps = runtime.store.list_capabilities(subject=pid)
                before_audit = runtime.store.list_audit()

                with pytest.raises(
                    ValidationError,
                    match='delegable|revocable|max_delegation_depth',
                ):
                    runtime.capability.issue(
                        'test',
                        pid,
                        spec,
                        require_authority=False,
                    )

                assert runtime.store.list_capabilities(subject=pid) == before_caps
                assert runtime.store.list_audit() == before_audit

            zero_depth = runtime.capability.issue(
                'test',
                pid,
                {
                    'resource': 'object:valid-depth-zero',
                    'rights': ['read'],
                    'delegable': False,
                    'revocable': True,
                    'max_delegation_depth': 0,
                },
                require_authority=False,
            )
            one_depth = runtime.capability.issue(
                'test',
                pid,
                {
                    'resource': 'object:valid-depth-one',
                    'rights': ['read'],
                    'delegation': {
                        'delegable': True,
                        'revocable': False,
                        'max_delegation_depth': 1,
                    },
                },
                require_authority=False,
            )
            assert (zero_depth.delegable, zero_depth.revocable, zero_depth.max_delegation_depth) == (False, True, 0)
            assert (one_depth.delegable, one_depth.revocable, one_depth.max_delegation_depth) == (True, False, 1)
        finally:
            runtime.close()

    def test_capability_spec_containers_reject_coercion_before_issue_side_effects(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='strict capability containers')
            invalid_specs: list[CapabilitySpec | dict[str, object]] = [
                {'resource': 7, 'rights': ['read']},
                {'resource': 'object:rights-string', 'rights': 'read'},
                {'resource': 'object:rights-mapping', 'rights': {'read': False}},
                {'resource': 'object:rights-tuple', 'rights': ('read',)},
                {'resource': 'object:constraints-false', 'rights': ['read'], 'constraints': False},
                {'resource': 'object:constraints-list', 'rights': ['read'], 'constraints': []},
                {
                    'resource': 'object:constraints-pairs',
                    'rights': ['read'],
                    'constraints': [('purpose', 'laundered')],
                },
                {'resource': 'object:constraints-key', 'rights': ['read'], 'constraints': {1: 'bad'}},
                {'resource': 'object:metadata-false', 'rights': ['read'], 'metadata': False},
                {'resource': 'object:metadata-list', 'rights': ['read'], 'metadata': []},
                {'resource': 'object:metadata-key', 'rights': ['read'], 'metadata': {1: 'bad'}},
                {'resource': 'object:rules-false', 'rights': ['read'], 'rules': False},
                {'resource': 'object:rules-mapping', 'rights': ['read'], 'rules': {}},
                CapabilitySpec(
                    resource='object:spec-rights-mapping',
                    rights={'read': False},  # type: ignore[arg-type]
                ),
                CapabilitySpec(
                    resource='object:spec-constraints-false',
                    rights={CapabilityRight.READ.value},
                    constraints=False,  # type: ignore[arg-type]
                ),
                CapabilitySpec(
                    resource='object:spec-metadata-list',
                    rights={CapabilityRight.READ.value},
                    metadata=[],  # type: ignore[arg-type]
                ),
            ]

            for spec in invalid_specs:
                before_caps = runtime.store.list_capabilities(subject=pid)
                before_audit = runtime.store.list_audit()
                before_events = runtime.events.list()

                with pytest.raises(ValidationError, match='resource|rights|constraints|metadata|rules'):
                    runtime.capability.issue(
                        'test',
                        pid,
                        spec,
                        require_authority=False,
                    )

                assert runtime.store.list_capabilities(subject=pid) == before_caps
                assert runtime.store.list_audit() == before_audit
                assert runtime.events.list() == before_events

            valid_mapping = runtime.capability.issue(
                'test',
                pid,
                {
                    'resource': 'object:strict-valid-mapping',
                    'rights': ['read'],
                    'constraints': {},
                    'metadata': {'source': 'test'},
                    'rules': [],
                },
                require_authority=False,
            )
            valid_spec = runtime.capability.issue(
                'test',
                pid,
                CapabilitySpec(
                    resource='object:strict-valid-spec',
                    rights=frozenset({CapabilityRight.READ.value}),  # type: ignore[arg-type]
                    constraints={},
                    metadata={},
                    rules=(),  # type: ignore[arg-type]
                ),
                require_authority=False,
            )
            assert valid_mapping.rights == {CapabilityRight.READ.value}
            assert valid_spec.rights == {CapabilityRight.READ.value}
        finally:
            runtime.close()

    def test_trusted_capability_helpers_reject_falsey_container_laundering(self) -> None:
        runtime = Runtime.open('local')
        try:
            invalid_calls = [
                lambda: runtime.capability.issue_trusted(
                    'worker',
                    'object:trusted-constraints-list',
                    [CapabilityRight.READ],
                    issued_by='test',
                    constraints=[],  # type: ignore[arg-type]
                ),
                lambda: runtime.capability.issue_trusted(
                    'worker',
                    'object:trusted-metadata-list',
                    [CapabilityRight.READ],
                    issued_by='test',
                    metadata=[],  # type: ignore[arg-type]
                ),
                lambda: runtime.capability.grant(
                    'worker',
                    'object:trusted-grant-constraints-list',
                    [CapabilityRight.READ],
                    issued_by='test',
                    constraints=[],  # type: ignore[arg-type]
                ),
                lambda: runtime.capability.grant_once(
                    'worker',
                    'object:trusted-once-constraints-list',
                    [CapabilityRight.READ],
                    issued_by='test',
                    constraints=[],  # type: ignore[arg-type]
                ),
            ]

            for call in invalid_calls:
                before_caps = runtime.store.list_capabilities(subject='worker')
                before_audit = runtime.store.list_audit()
                before_events = runtime.events.list()

                with pytest.raises(ValidationError, match='constraints|metadata'):
                    call()

                assert runtime.store.list_capabilities(subject='worker') == before_caps
                assert runtime.store.list_audit() == before_audit
                assert runtime.events.list() == before_events
        finally:
            runtime.close()

    def test_capability_spec_schema_rejects_unknown_fields_before_issue(self) -> None:
        runtime = Runtime.open('local')
        try:
            invalid_specs = [
                {
                    'resource': 'object:unknown-effect-field',
                    'rights': ['read'],
                    'effectt': 'deny',
                },
                {
                    'resource': 'object:unknown-lease-field',
                    'rights': ['read'],
                    'lease': {'uses_remaning': 1},
                },
                {
                    'resource': 'object:unknown-delegation-field',
                    'rights': ['read'],
                    'delegation': {
                        'delegable': True,
                        'max_delegation_dept': 0,
                    },
                },
                {
                    'resource': 'object:conflicting-lease-fields',
                    'rights': ['read'],
                    'uses_remaining': 1,
                    'lease': {},
                },
                {
                    'resource': 'object:conflicting-delegation-fields',
                    'rights': ['read'],
                    'delegable': False,
                    'delegation': {
                        'delegable': True,
                        'max_delegation_depth': 2,
                    },
                },
                {
                    'resource': 'object:conflicting-policy-fields',
                    'rights': ['read'],
                    'policy': 'always_deny',
                    'permission_policy': 'always_allow',
                },
            ]

            for spec in invalid_specs:
                before_caps = runtime.store.list_capabilities(subject='worker')
                before_audit = runtime.store.list_audit()
                before_events = runtime.events.list()

                with pytest.raises(ValidationError, match='unknown|conflicting'):
                    runtime.capability.issue(
                        'test',
                        'worker',
                        spec,
                        require_authority=False,
                    )

                assert runtime.store.list_capabilities(subject='worker') == before_caps
                assert runtime.store.list_audit() == before_audit
                assert runtime.events.list() == before_events
        finally:
            runtime.close()

    def test_invalid_delegation_dataclass_is_rejected_before_delegate_side_effects(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='strict delegate parent')
            child = runtime.process.spawn(image='base-agent:v0', goal='strict delegate child')
            runtime.capability.grant(
                parent,
                'object:strict-delegate',
                [CapabilityRight.READ],
                issued_by='test',
                delegable=True,
            )
            malformed = CapabilitySpec(
                resource='object:strict-delegate',
                rights={CapabilityRight.READ.value},
                delegation=DelegationPolicy(
                    delegable=True,
                    revocable='false',  # type: ignore[arg-type]
                    max_delegation_depth=1,
                ),
            )
            before_caps = runtime.store.list_capabilities(subject=child)
            before_audit = runtime.store.list_audit()

            with pytest.raises(ValidationError, match='revocable'):
                runtime.capability.delegate(parent, child, malformed)

            assert runtime.store.list_capabilities(subject=child) == before_caps
            assert runtime.store.list_audit() == before_audit
        finally:
            runtime.close()

    def test_manifest_compile_prevalidates_all_delegation_fields_before_issuing_any_capability(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='strict manifest compile')
            manifest = TaskAuthorityManifest(
                manifest_id='authm_strict_delegation_test',
                pid=pid,
                image_id='base-agent:v0',
                goal_ref=None,
                authorized_capabilities=[
                    {'resource': 'object:manifest-valid-first', 'rights': ['read']},
                    {
                        'resource': 'object:manifest-invalid-second',
                        'rights': ['read'],
                        'delegable': 'false',
                    },
                ],
                issued_by='test',
            )
            before_caps = runtime.store.list_capabilities(subject=pid)
            before_audit = runtime.store.list_audit()

            with pytest.raises(ValidationError, match='delegable'):
                runtime.authority_manifests.compile_root_capabilities(manifest)

            assert runtime.store.list_capabilities(subject=pid) == before_caps
            assert runtime.store.list_audit() == before_audit
        finally:
            runtime.close()

    def test_permission_policy_aliases_are_converted_to_effect_and_lease(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='policy conversion')
            converted = runtime.capability.grant(pid, 'object:converted-policy', [CapabilityRight.READ], issued_by='test', constraints={runtime.capability.POLICY_KEY: runtime.capability.ALWAYS_DENY})
            one_shot = runtime.capability.issue_trusted(pid, 'object:one-shot-policy', [CapabilityRight.READ], issued_by='test', constraints={runtime.capability.POLICY_KEY: runtime.capability.ALLOW_ONCE})
            converted_decision = runtime.capability.authorize(pid, 'object:converted-policy', CapabilityRight.READ)
            one_shot_decision = runtime.capability.authorize(pid, 'object:one-shot-policy', CapabilityRight.READ)
            assert converted.effect == CapabilityEffect.DENY
            assert not converted_decision.allowed
            assert runtime.capability.POLICY_KEY not in runtime.capability.inspect(one_shot.cap_id)['constraints']
            assert one_shot_decision.allowed
            assert one_shot_decision.consume_capability_id == one_shot.cap_id
        finally:
            runtime.close()

    def test_authority_rules_are_enforced_against_operation_context(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='rule constrained shell')
            runtime.capability.issue_trusted(
                pid,
                'shell:git',
                [CapabilityRight.EXECUTE],
                issued_by='test',
                constraints={
                    'authority_rules': [
                        {
                            'rule_id': 'test.git.status.only',
                            'operation': 'shell.run',
                            'effect': 'allow',
                            'risk': 'harmless',
                            'conditions': {'argv': ['git', 'status'], 'match': 'exact'},
                        }
                    ]
                },
            )
            allowed = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'argv': ['git', 'status']},
            )
            denied = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'argv': ['git', 'push']},
            )
            assert allowed.allowed
            assert not denied.allowed
            assert 'constraints rejected' in denied.reason
        finally:
            runtime.close()

    def test_authority_rule_unknown_top_level_field_is_rejected(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='reject malformed authority rule')

            with pytest.raises(ValidationError, match='unknown fields: condition'):
                runtime.capability.issue_trusted(
                    pid,
                    'shell:git',
                    [CapabilityRight.EXECUTE],
                    issued_by='test',
                    constraints={
                        'authority_rules': [
                            {
                                'rule_id': 'test.git.status.typo',
                                'operation': 'shell.run',
                                'effect': 'allow',
                                'risk': 'harmless',
                                'condition': {'argv': ['git', 'status'], 'match': 'exact'},
                            }
                        ]
                    },
                )

            decision = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'argv': ['git', 'push']},
            )
            assert not decision.allowed
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        'conditions',
        [[], '', 0, False],
        ids=['list', 'empty-string', 'zero', 'false'],
    )
    def test_authority_rule_falsey_non_mapping_conditions_are_rejected(
        self,
        conditions: object,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='strict rule conditions')

            with pytest.raises(ValidationError, match='conditions must be a mapping'):
                runtime.capability.issue_trusted(
                    pid,
                    'shell:git',
                    [CapabilityRight.EXECUTE],
                    issued_by='test',
                    constraints={
                        'authority_rules': [
                            {
                                'rule_id': 'test.strict.conditions',
                                'operation': 'shell.run',
                                'effect': 'allow',
                                'risk': 'low',
                                'conditions': conditions,
                            }
                        ]
                    },
                )

            assert not runtime.capability.check(pid, 'shell:git', CapabilityRight.EXECUTE)
        finally:
            runtime.close()

    def test_authority_rule_without_conditions_is_valid_unconditional_rule(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='unconditional authority rule')
            capability = runtime.capability.issue_trusted(
                pid,
                'shell:git',
                [CapabilityRight.EXECUTE],
                issued_by='test',
                constraints={
                    'authority_rules': [
                        {
                            'rule_id': 'test.git.unconditional',
                            'operation': 'shell.run',
                            'effect': 'allow',
                            'risk': 'low',
                        }
                    ]
                },
            )

            decision = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'argv': ['git', 'push']},
            )
            assert decision.allowed
            assert runtime.capability.inspect(capability.cap_id)['rules'][0]['conditions'] == {}
        finally:
            runtime.close()

    def test_scoped_deny_rule_only_denies_matching_operation_context(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='scoped deny shell')
            runtime.capability.issue_trusted(
                pid,
                'shell:git',
                [CapabilityRight.EXECUTE],
                issued_by='test',
                constraints={
                    'authority_rules': [
                        {
                            'rule_id': 'test.git.any',
                            'operation': 'shell.run',
                            'effect': 'allow',
                            'risk': 'low',
                            'conditions': {'argv': ['git'], 'match': 'prefix'},
                        }
                    ]
                },
            )
            runtime.capability.issue_trusted(
                pid,
                'shell:git',
                [CapabilityRight.EXECUTE],
                issued_by='test',
                effect=CapabilityEffect.DENY,
                constraints={
                    'authority_rules': [
                        {
                            'rule_id': 'test.git.push.deny',
                            'operation': 'shell.run',
                            'effect': 'deny',
                            'risk': 'high',
                            'conditions': {'argv': ['git', 'push'], 'match': 'prefix'},
                        }
                    ]
                },
            )
            allowed = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'argv': ['git', 'status']},
            )
            denied = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'argv': ['git', 'push', 'origin']},
            )
            assert allowed.allowed
            assert not denied.allowed
            assert denied.effect == CapabilityEffect.DENY
            assert denied.constraint_results['authority_rules']['rule_id'] == 'test.git.push.deny'
        finally:
            runtime.close()

    def test_malformed_authority_rule_condition_fails_closed_over_allow(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='malformed authority rule')
            _assert_rule_rejected_at_admission(
                runtime,
                pid,
                {
                    'rule_id': 'test.git.push.deny.typo',
                    'operation': 'shell.run',
                    'effect': 'deny',
                    'risk': 'high',
                    'conditions': {'argv_typo': ['git', 'push']},
                },
                error_match='unknown conditions: argv_typo',
            )

            decision = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'argv': ['git', 'push']},
            )

            assert decision.allowed
        finally:
            runtime.close()

    def test_malformed_known_authority_rule_condition_fails_closed_over_allow(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='malformed known authority rule')
            _assert_rule_rejected_at_admission(
                runtime,
                pid,
                {
                    'rule_id': 'test.git.regex.malformed',
                    'operation': 'shell.run',
                    'effect': 'allow',
                    'risk': 'low',
                    'conditions': {'regex_token': '['},
                },
                error_match='malformed conditions: regex_token',
            )

            decision = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'argv': ['git', 'status']},
            )

            assert decision.allowed
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('condition_name', 'condition_value'),
        [
            ('timeout_s', True),
            ('timeout_s', float('nan')),
            ('timeout_s', float('inf')),
            ('timeout_s', -0.1),
            ('timeout_max_s', False),
            ('timeout_max_s', float('nan')),
            ('timeout_max_s', float('-inf')),
            ('timeout_max_s', -1),
        ],
    )
    def test_authority_rule_rejects_invalid_timeout_condition(
        self,
        condition_name: str,
        condition_value: object,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='invalid authority timeout')
            _assert_rule_rejected_at_admission(
                runtime,
                pid,
                {
                    'rule_id': f'test.timeout.invalid.{condition_name}',
                    'operation': 'shell.run',
                    'effect': 'allow',
                    'risk': 'low',
                    'conditions': {condition_name: condition_value},
                },
                error_match=f'malformed conditions: {condition_name}',
            )

            decision = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'timeout_s': 1.0},
            )

            assert decision.allowed
        finally:
            runtime.close()

    @pytest.mark.parametrize('actual_timeout', [True, float('nan'), float('inf'), -0.1])
    def test_authority_rule_timeout_ceiling_rejects_invalid_operation_timeout(self, actual_timeout: object) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='invalid operation timeout')
            runtime.capability.issue_trusted(
                pid,
                'shell:git',
                [CapabilityRight.EXECUTE],
                issued_by='test',
                constraints={
                    'authority_rules': [
                        {
                            'rule_id': 'test.timeout.ceiling',
                            'operation': 'shell.run',
                            'effect': 'allow',
                            'risk': 'low',
                            'conditions': {'timeout_max_s': 5.0},
                        }
                    ]
                },
            )

            decision = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'timeout_s': actual_timeout},
            )

            assert not decision.allowed
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('conditions', 'actual_timeout'),
        [
            ({'timeout_s': 0.0}, 0.0),
            ({'timeout_s': 0.25}, 0.25),
            ({'timeout_max_s': 0.0}, 0.0),
            ({'timeout_max_s': 1.5}, 1.25),
        ],
    )
    def test_authority_rule_accepts_finite_nonnegative_timeout(
        self,
        conditions: dict[str, float],
        actual_timeout: float,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='valid authority timeout')
            runtime.capability.issue_trusted(
                pid,
                'shell:git',
                [CapabilityRight.EXECUTE],
                issued_by='test',
                constraints={
                    'authority_rules': [
                        {
                            'rule_id': 'test.timeout.valid',
                            'operation': 'shell.run',
                            'effect': 'allow',
                            'risk': 'low',
                            'conditions': conditions,
                        }
                    ]
                },
            )

            decision = runtime.capability.authorize(
                pid,
                'shell:git',
                CapabilityRight.EXECUTE,
                {'authority_operation': 'shell.run', 'operation': 'shell.run', 'timeout_s': actual_timeout},
            )

            assert decision.allowed
        finally:
            runtime.close()

    def test_one_shot_grant_authority_is_consumed_after_successful_issue(self) -> None:
        runtime = Runtime.open('local')
        try:
            issuer = runtime.process.spawn(image='base-agent:v0', goal='issuer')
            subject = runtime.process.spawn(image='base-agent:v0', goal='subject')
            runtime.capability.issue_trusted(issuer, 'object:alpha', [CapabilityRight.READ], issued_by='test')
            grant_cap = runtime.capability.grant_once(issuer, 'object:alpha', [CapabilityRight.GRANT], issued_by='test')
            issued = runtime.capability.issue(issuer, subject, CapabilitySpec(resource='object:alpha', rights={CapabilityRight.READ.value}))
            assert runtime.capability.check(subject, 'object:alpha', CapabilityRight.READ)
            assert issued.issuer_cap_id == grant_cap.cap_id
            assert issued.parent_cap_id != grant_cap.cap_id
            assert runtime.capability.inspect(grant_cap.cap_id)['status'] == 'revoked'
            with pytest.raises(CapabilityDenied):
                runtime.capability.issue(issuer, subject, CapabilitySpec(resource='object:alpha', rights={CapabilityRight.WRITE.value}))
        finally:
            runtime.close()

    def test_issue_rolls_back_capability_and_one_shot_authority_when_event_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            issuer = runtime.process.spawn(image='base-agent:v0', goal='issuer')
            subject = runtime.process.spawn(image='base-agent:v0', goal='subject')
            runtime.capability.issue_trusted(issuer, 'object:alpha', [CapabilityRight.READ], issued_by='test')
            grant_cap = runtime.capability.grant_once(
                issuer,
                'object:alpha',
                [CapabilityRight.GRANT],
                issued_by='test',
            )
            before_ids = {cap.cap_id for cap in runtime.capability.capabilities_for(subject)}
            original_emit = runtime.events.emit

            def fail_grant_event(event_type, *args, **kwargs):
                if event_type == EventType.CAPABILITY_GRANTED:
                    raise RuntimeError('injected capability grant event failure')
                return original_emit(event_type, *args, **kwargs)

            monkeypatch.setattr(runtime.events, 'emit', fail_grant_event)
            with pytest.raises(RuntimeError, match='injected capability grant event failure'):
                runtime.capability.issue(
                    issuer,
                    subject,
                    CapabilitySpec(resource='object:alpha', rights={CapabilityRight.READ.value}),
                )

            assert {cap.cap_id for cap in runtime.capability.capabilities_for(subject)} == before_ids
            grant_after_failure = runtime.capability.inspect(grant_cap.cap_id)
            assert grant_after_failure['status'] == 'active'
            assert grant_after_failure['uses_remaining'] == 1

            monkeypatch.setattr(runtime.events, 'emit', original_emit)
            issued = runtime.capability.issue(
                issuer,
                subject,
                CapabilitySpec(resource='object:alpha', rights={CapabilityRight.READ.value}),
            )
            assert {cap.cap_id for cap in runtime.capability.capabilities_for(subject)} == before_ids | {issued.cap_id}
            assert runtime.capability.inspect(grant_cap.cap_id)['status'] == 'revoked'
        finally:
            runtime.close()

    def test_grant_authority_can_only_transfer_existing_allow_rights(self) -> None:
        runtime = Runtime.open('local')
        try:
            issuer = runtime.process.spawn(image='base-agent:v0', goal='issuer')
            subject = runtime.process.spawn(image='base-agent:v0', goal='subject')
            spec = CapabilitySpec(resource='object:alpha', rights={CapabilityRight.READ.value})
            runtime.capability.issue_trusted(issuer, 'object:alpha', [CapabilityRight.GRANT], issued_by='test')
            with pytest.raises(CapabilityDenied):
                runtime.capability.issue(issuer, subject, spec)
            source = runtime.capability.issue_trusted(issuer, 'object:alpha', [CapabilityRight.READ], issued_by='test')
            issued = runtime.capability.issue(issuer, subject, spec)
            assert issued.parent_cap_id == source.cap_id
            assert runtime.capability.check(subject, 'object:alpha', CapabilityRight.READ)
            with pytest.raises(CapabilityDenied):
                runtime.capability.issue(issuer, subject, CapabilitySpec(resource='object:alpha', rights={CapabilityRight.ADMIN.value}))
            with pytest.raises(CapabilityDenied):
                runtime.capability.issue(
                    issuer,
                    subject,
                    CapabilitySpec(
                        resource='object:alpha',
                        rights={CapabilityRight.READ.value},
                        effect=CapabilityEffect.DENY,
                    ),
                )
        finally:
            runtime.close()

    def test_grant_transfer_inherits_parent_expiration(self) -> None:
        runtime = Runtime.open('local')
        try:
            issuer = runtime.process.spawn(image='base-agent:v0', goal='issuer')
            subject = runtime.process.spawn(image='base-agent:v0', goal='subject')
            source = runtime.capability.issue_trusted(
                issuer,
                'object:leased',
                [CapabilityRight.READ],
                issued_by='test',
                expires_at='2999-01-01T00:00:00Z',
            )
            runtime.capability.issue_trusted(issuer, 'object:leased', [CapabilityRight.GRANT], issued_by='test')

            issued = runtime.capability.issue(
                issuer,
                subject,
                CapabilitySpec(resource='object:leased', rights={CapabilityRight.READ.value}),
            )

            assert issued.parent_cap_id == source.cap_id
            assert issued.expires_at == source.expires_at
            assert runtime.capability.inspect(issued.cap_id)['expires_at'] == source.expires_at
            with pytest.raises(CapabilityDenied):
                runtime.capability.issue(
                    issuer,
                    subject,
                    CapabilitySpec(
                        resource='object:leased',
                        rights={CapabilityRight.READ.value},
                        expires_at='3000-01-01T00:00:00Z',
                    ),
                )
        finally:
            runtime.close()

    def test_one_time_capability_claim_is_conditional(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='claim once')
            cap = runtime.capability.grant_once(pid, 'object:once', [CapabilityRight.READ], issued_by='test')
            first = runtime.capability.authorize(pid, 'object:once', CapabilityRight.READ)
            second = runtime.capability.authorize(pid, 'object:once', CapabilityRight.READ)
            runtime.capability.claim_decision_use(first, used_by=pid, reason='test claim')
            with pytest.raises(CapabilityDenied):
                runtime.capability.claim_decision_use(second, used_by=pid, reason='test claim')
            assert runtime.capability.inspect(cap.cap_id)['status'] == 'revoked'
        finally:
            runtime.close()

    def test_claim_decision_use_reauthorizes_before_consuming_stale_allow(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='stale claim deny')
            cap = runtime.capability.grant_once(
                pid,
                'object:stale-claim',
                [CapabilityRight.READ],
                issued_by='test',
            )
            decision = runtime.capability.authorize(
                pid,
                cap.resource,
                CapabilityRight.READ,
            )
            runtime.capability.issue_trusted(
                pid,
                cap.resource,
                [CapabilityRight.READ],
                issued_by='test',
                effect=CapabilityEffect.DENY,
            )
            before_reservations = runtime.store.select_table_rows(
                'capability_use_reservations'
            )

            with pytest.raises(CapabilityDenied, match='authority changed'):
                runtime.capability.claim_decision_use(
                    decision,
                    used_by=pid,
                    reason='stale one-shot claim',
                )

            persisted = runtime.store.get_capability(cap.cap_id)
            assert persisted is not None and persisted.active
            assert persisted.uses_remaining == 1
            assert runtime.store.select_table_rows(
                'capability_use_reservations'
            ) == before_reservations
        finally:
            runtime.close()

    def test_claim_decision_use_reauthorizes_unlimited_cached_allow(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='stale unlimited claim')
            allow = runtime.capability.issue_trusted(
                pid,
                'object:stale-unlimited-claim',
                [CapabilityRight.READ],
                issued_by='test',
            )
            decision = runtime.capability.authorize(pid, allow.resource, CapabilityRight.READ)
            assert decision.consume_capability_id is None
            runtime.capability.issue_trusted(
                pid,
                allow.resource,
                [CapabilityRight.READ],
                issued_by='test',
                effect=CapabilityEffect.DENY,
            )

            with pytest.raises(CapabilityDenied, match='authority changed'):
                runtime.capability.claim_decision_use(
                    decision,
                    used_by=pid,
                    reason='stale unlimited claim',
                )

            assert runtime.store.get_capability(allow.cap_id) == allow
        finally:
            runtime.close()

    def test_reserve_decision_use_reauthorizes_before_reserving(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='stale reserve')
            cap = runtime.capability.grant_once(
                pid,
                'object:stale-reserve',
                [CapabilityRight.READ],
                issued_by='test',
            )
            decision = runtime.capability.authorize(pid, cap.resource, CapabilityRight.READ)
            runtime.capability.issue_trusted(
                pid,
                cap.resource,
                [CapabilityRight.READ],
                issued_by='test',
                effect=CapabilityEffect.DENY,
            )
            before_rows = runtime.store.select_table_rows('capability_use_reservations')

            with pytest.raises(CapabilityDenied, match='authority changed'):
                runtime.capability.reserve_decision_use(
                    decision,
                    used_by=pid,
                    reason='stale reservation',
                )

            persisted = runtime.store.get_capability(cap.cap_id)
            assert persisted is not None and persisted.uses_remaining == 1
            assert runtime.store.select_table_rows('capability_use_reservations') == before_rows
        finally:
            runtime.close()

    def test_require_reauthorizes_inside_claim_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='require race')
            cap = runtime.capability.grant_once(
                pid,
                'object:require-race',
                [CapabilityRight.READ],
                issued_by='test',
            )
            original_transaction = runtime.capability.authority_transaction
            inserted = False

            def install_deny_before_transaction(decisions, *, actor, operation):
                nonlocal inserted
                if not inserted and operation == 'test require transaction':
                    inserted = True
                    runtime.capability.issue_trusted(
                        pid,
                        cap.resource,
                        [CapabilityRight.READ],
                        issued_by='test',
                        effect=CapabilityEffect.DENY,
                    )
                return original_transaction(
                    decisions,
                    actor=actor,
                    operation=operation,
                )

            monkeypatch.setattr(
                runtime.capability,
                'authority_transaction',
                install_deny_before_transaction,
            )

            with pytest.raises(CapabilityDenied, match='authority changed'):
                runtime.capability.require(
                    pid,
                    cap.resource,
                    CapabilityRight.READ,
                    reason='test require transaction',
                )

            persisted = runtime.store.get_capability(cap.cap_id)
            assert persisted is not None and persisted.uses_remaining == 1
        finally:
            runtime.close()

    def test_assert_handle_reauthorizes_exact_named_capability_without_broad_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='exact handle claim')
            handle = runtime.capability.handle_for_object(
                pid,
                'exact-handle-claim',
                [CapabilityRight.READ],
                issued_by='test',
                uses_remaining=1,
            )
            runtime.capability.issue_trusted(
                pid,
                'object:*',
                [CapabilityRight.READ],
                issued_by='test',
            )
            original_authorize_handle = runtime.capability.authorize_handle
            first = True

            def revoke_after_cached_handle_allow(subject, selected_handle, right):
                nonlocal first
                decision = original_authorize_handle(subject, selected_handle, right)
                if first:
                    first = False
                    runtime.capability.revoke(
                        handle.capability_id,
                        revoked_by='test',
                        reason='invalidate cached handle decision',
                        require_authority=False,
                    )
                return decision

            monkeypatch.setattr(
                runtime.capability,
                'authorize_handle',
                revoke_after_cached_handle_allow,
            )

            with pytest.raises(CapabilityDenied, match='handle authority changed'):
                runtime.capability.assert_handle(pid, handle, CapabilityRight.READ)

            persisted = runtime.store.get_capability(handle.capability_id)
            assert persisted is not None
            assert persisted.status == CapabilityStatus.REVOKED
            assert persisted.uses_remaining == 1
            assert runtime.capability.check(pid, f'object:{handle.oid}', CapabilityRight.READ)
        finally:
            runtime.close()

    @pytest.mark.parametrize('state_change', ['revoke', 'expire'])
    def test_claim_decision_use_rejects_revoked_or_expired_binding(
        self,
        state_change: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal=f'stale claim {state_change}')
            cap = runtime.capability.grant_once(
                pid,
                f'object:stale-claim-{state_change}',
                [CapabilityRight.READ],
                issued_by='test',
                expires_at='2999-01-01T00:00:00Z',
            )
            decision = runtime.capability.authorize(
                pid,
                cap.resource,
                CapabilityRight.READ,
            )
            if state_change == 'revoke':
                runtime.capability.revoke(
                    cap.cap_id,
                    revoked_by='test',
                    reason='invalidate cached decision',
                )
            else:
                runtime.store.update_capability(
                    replace(cap, expires_at='2000-01-01T00:00:00+00:00')
                )
            before_reservations = runtime.store.select_table_rows(
                'capability_use_reservations'
            )

            with pytest.raises(CapabilityDenied, match='authority changed'):
                runtime.capability.claim_decision_use(
                    decision,
                    used_by=pid,
                    reason=f'stale {state_change} claim',
                )

            persisted = runtime.store.get_capability(cap.cap_id)
            assert persisted is not None
            assert persisted.uses_remaining == 1
            assert runtime.store.select_table_rows(
                'capability_use_reservations'
            ) == before_reservations
        finally:
            runtime.close()

    def test_concurrent_deny_commit_linearizes_before_cached_claim(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='concurrent deny claim')
            cap = runtime.capability.grant_once(
                pid,
                'object:concurrent-deny-claim',
                [CapabilityRight.READ],
                issued_by='test',
            )
            decision = runtime.capability.authorize(
                pid,
                cap.resource,
                CapabilityRight.READ,
            )
            deny_inserted = ThreadEvent()
            claim_started = ThreadEvent()

            def install_deny() -> str:
                with runtime.store.transaction():
                    runtime.capability.issue_trusted(
                        pid,
                        cap.resource,
                        [CapabilityRight.READ],
                        issued_by='test',
                        effect=CapabilityEffect.DENY,
                    )
                    deny_inserted.set()
                    assert claim_started.wait(timeout=5)
                return 'deny_committed'

            def claim() -> str:
                assert deny_inserted.wait(timeout=5)
                claim_started.set()
                try:
                    runtime.capability.claim_decision_use(
                        decision,
                        used_by=pid,
                        reason='concurrent stale claim',
                    )
                except CapabilityDenied:
                    return 'denied'
                return 'claimed'

            with ThreadPoolExecutor(max_workers=2) as pool:
                deny_future = pool.submit(install_deny)
                claim_future = pool.submit(claim)
                assert deny_future.result(timeout=10) == 'deny_committed'
                assert claim_future.result(timeout=10) == 'denied'

            persisted = runtime.store.get_capability(cap.cap_id)
            assert persisted is not None and persisted.active
            assert persisted.uses_remaining == 1
        finally:
            runtime.close()

    def test_concurrent_cached_one_shot_claim_has_one_winner(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='concurrent one-shot claim')
            cap = runtime.capability.grant_once(
                pid,
                'object:concurrent-one-shot-claim',
                [CapabilityRight.READ],
                issued_by='test',
            )
            decisions = [
                runtime.capability.authorize(pid, cap.resource, CapabilityRight.READ)
                for _ in range(2)
            ]
            barrier = Barrier(2)

            def claim(decision: object) -> str:
                barrier.wait(timeout=5)
                try:
                    runtime.capability.claim_decision_use(
                        decision,  # type: ignore[arg-type]
                        used_by=pid,
                        reason='concurrent one-shot claim',
                    )
                except CapabilityDenied:
                    return 'denied'
                return 'claimed'

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(claim, decisions))

            assert sorted(results) == ['claimed', 'denied']
            persisted = runtime.store.get_capability(cap.cap_id)
            assert persisted is not None
            assert persisted.uses_remaining == 0
            assert persisted.status == CapabilityStatus.REVOKED
        finally:
            runtime.close()

    @pytest.mark.parametrize('failed_sink', ['reserve_audit', 'revoke_event', 'commit_audit'])
    def test_claim_decision_use_sink_failure_rolls_back_lease_and_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failed_sink: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal=f'claim rollback {failed_sink}')
            cap = runtime.capability.grant_once(
                pid,
                f'object:claim-rollback-{failed_sink}',
                [CapabilityRight.READ],
                issued_by='test',
            )
            decision = runtime.capability.authorize(
                pid,
                cap.resource,
                CapabilityRight.READ,
            )
            before_reservations = runtime.store.select_table_rows(
                'capability_use_reservations'
            )
            before_audit_ids = {
                record.record_id for record in runtime.store.list_audit()
            }
            before_event_ids = {
                event.event_id for event in runtime.events.list()
            }
            if failed_sink in {'reserve_audit', 'commit_audit'}:
                original_record = runtime.audit.record
                failed_action = (
                    'capability.reserve_use'
                    if failed_sink == 'reserve_audit'
                    else 'capability.commit_reserved_use'
                )

                def fail_audit_after_write(*args, **kwargs):
                    result = original_record(*args, **kwargs)
                    if kwargs.get('action') == failed_action:
                        raise RuntimeError(f'injected {failed_sink} failure')
                    return result

                monkeypatch.setattr(runtime.audit, 'record', fail_audit_after_write)
            else:
                original_emit = runtime.events.emit

                def fail_event_after_write(event_type, *args, **kwargs):
                    result = original_emit(event_type, *args, **kwargs)
                    if event_type == EventType.CAPABILITY_REVOKED:
                        raise RuntimeError('injected revoke_event failure')
                    return result

                monkeypatch.setattr(runtime.events, 'emit', fail_event_after_write)

            with pytest.raises(RuntimeError, match=f'injected {failed_sink} failure'):
                runtime.capability.claim_decision_use(
                    decision,
                    used_by=pid,
                    reason='atomic cached claim',
                )

            persisted = runtime.store.get_capability(cap.cap_id)
            assert persisted is not None and persisted.active
            assert persisted.uses_remaining == 1
            assert runtime.store.select_table_rows(
                'capability_use_reservations'
            ) == before_reservations
            assert {
                event.event_id for event in runtime.events.list()
            } == before_event_ids
            new_audits = [
                record
                for record in runtime.store.list_audit()
                if record.record_id not in before_audit_ids
            ]
            assert all(
                record.action not in {
                    'capability.reserve_use',
                    'capability.commit_reserved_use',
                    'capability.consume',
                }
                for record in new_audits
            )
        finally:
            runtime.close()

    def test_capability_lease_counts_require_positive_exact_integers_without_side_effects(self) -> None:
        runtime = Runtime.open('local')
        try:
            cap = runtime.capability.issue_trusted(
                'worker',
                'object:strict-lease-counts',
                [CapabilityRight.READ],
                issued_by='test',
                uses_remaining=3,
            )
            invalid_counts = [True, 1.0, '1', 0, -1]
            for index, count in enumerate(invalid_counts):
                for operation in ('consume', 'reserve'):
                    before_cap = runtime.store.get_capability(cap.cap_id)
                    before_reservations = runtime.store.select_table_rows(
                        'capability_use_reservations'
                    )
                    before_audit = runtime.store.list_audit()
                    before_events = runtime.events.list()

                    with pytest.raises(ValidationError, match='count'):
                        if operation == 'consume':
                            runtime.capability.consume_use(
                                cap.cap_id,
                                used_by='test',
                                count=count,  # type: ignore[arg-type]
                            )
                        else:
                            runtime.capability.reserve_use(
                                cap.cap_id,
                                reserved_by='test',
                                count=count,  # type: ignore[arg-type]
                            )

                    assert runtime.store.get_capability(cap.cap_id) == before_cap
                    assert runtime.store.select_table_rows(
                        'capability_use_reservations'
                    ) == before_reservations
                    assert runtime.store.list_audit() == before_audit
                    assert runtime.events.list() == before_events

                with pytest.raises(ValueError, match='count'):
                    runtime.store.consume_capability_uses(
                        cap.cap_id,
                        count,  # type: ignore[arg-type]
                    )
                with pytest.raises(ValueError, match='count'):
                    runtime.store.reserve_capability_uses(
                        cap.cap_id,
                        f'invalid-reservation-{index}',
                        count=count,  # type: ignore[arg-type]
                        reserved_by='test',
                        reason='invalid direct store count',
                        created_at='2026-01-01T00:00:00Z',
                    )
        finally:
            runtime.close()

    def test_restore_rejects_fractional_persisted_reservation_before_side_effects(self) -> None:
        runtime = Runtime.open('local')
        try:
            cap = runtime.capability.issue_trusted(
                'worker',
                'object:strict-reservation-restore',
                [CapabilityRight.READ],
                issued_by='test',
                uses_remaining=2,
            )
            reservation_id = runtime.capability.reserve_use(
                cap.cap_id,
                reserved_by='test',
                count=1,
            )
            runtime.store._execute(  # type: ignore[attr-defined]
                'UPDATE capability_use_reservations SET count = ? WHERE reservation_id = ?',
                (1.5, reservation_id),
            )
            before_cap = runtime.store.get_capability(cap.cap_id)
            before_audit = runtime.store.list_audit()
            before_events = runtime.events.list()
            before_rows = runtime.store.select_table_rows(
                'capability_use_reservations'
            )

            with pytest.raises(ValidationError, match='reservation count'):
                runtime.capability.restore_reserved_use(
                    reservation_id,
                    restored_by='test',
                )

            assert runtime.store.get_capability(cap.cap_id) == before_cap
            assert runtime.store.list_audit() == before_audit
            assert runtime.events.list() == before_events
            assert runtime.store.select_table_rows(
                'capability_use_reservations'
            ) == before_rows
        finally:
            runtime.close()

    def test_fractional_persisted_capability_lease_fails_closed_on_decode(self) -> None:
        runtime = Runtime.open('local')
        try:
            cap = runtime.capability.issue_trusted(
                'worker',
                'object:strict-persisted-lease',
                [CapabilityRight.READ],
                issued_by='test',
                uses_remaining=2,
            )
            runtime.store._execute(  # type: ignore[attr-defined]
                'UPDATE capabilities SET uses_remaining = ? WHERE cap_id = ?',
                (1.5, cap.cap_id),
            )

            with pytest.raises(ValidationError, match='uses_remaining'):
                runtime.store.get_capability(cap.cap_id)
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('column', 'value', 'message'),
        [
            ('delegable', 'false', 'delegable'),
            ('revocable', 2, 'revocable'),
            ('delegation_depth', 1.5, 'delegation_depth'),
            ('max_delegation_depth', 'two', 'max_delegation_depth'),
        ],
    )
    def test_persisted_capability_authority_scalars_fail_closed_on_decode(
        self,
        column: str,
        value: object,
        message: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            cap = runtime.capability.issue_trusted(
                'worker',
                'object:strict-persisted-authority',
                [CapabilityRight.READ],
                issued_by='test',
            )
            runtime.store._execute(  # type: ignore[attr-defined]
                f'UPDATE capabilities SET {column} = ? WHERE cap_id = ?',
                (value, cap.cap_id),
            )

            with pytest.raises(ValidationError, match=message):
                runtime.store.get_capability(cap.cap_id)
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('delegable', 1),
            ('revocable', 'false'),
            ('delegation_depth', True),
            ('max_delegation_depth', 1.5),
        ],
    )
    def test_capability_store_rejects_non_exact_authority_scalars_on_write(
        self,
        field: str,
        value: object,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            cap = runtime.capability.issue_trusted(
                'worker',
                'object:strict-store-authority',
                [CapabilityRight.READ],
                issued_by='test',
            )
            malformed = replace(cap, **{field: value})

            with pytest.raises(ValueError, match=field):
                runtime.store.update_capability(malformed)

            assert runtime.store.get_capability(cap.cap_id) == cap
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('column', 'payload', 'message'),
        [
            ('rights_json', '{"read": false}', 'rights'),
            ('constraints_json', '[]', 'constraints'),
            ('metadata_json', '[]', 'metadata'),
        ],
    )
    def test_persisted_capability_containers_fail_closed_on_decode(
        self,
        column: str,
        payload: str,
        message: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            cap = runtime.capability.issue_trusted(
                'worker',
                'object:strict-persisted-containers',
                [CapabilityRight.READ],
                issued_by='test',
            )
            runtime.store._execute(  # type: ignore[attr-defined]
                f'UPDATE capabilities SET {column} = ? WHERE cap_id = ?',
                (payload, cap.cap_id),
            )

            with pytest.raises(ValidationError, match=message):
                runtime.store.get_capability(cap.cap_id)
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('rights', {'read': False}),
            ('constraints', []),
            ('metadata', False),
        ],
    )
    def test_capability_store_rejects_non_exact_containers_on_write(
        self,
        field: str,
        value: object,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            cap = runtime.capability.issue_trusted(
                'worker',
                'object:strict-store-containers',
                [CapabilityRight.READ],
                issued_by='test',
            )
            malformed = replace(cap, **{field: value})

            with pytest.raises(ValueError, match=field):
                runtime.store.update_capability(malformed)

            assert runtime.store.get_capability(cap.cap_id) == cap
        finally:
            runtime.close()

    def test_issue_requires_trusted_actor_or_grant_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            issuer = runtime.process.spawn(image='base-agent:v0', goal='issuer')
            subject = runtime.process.spawn(image='base-agent:v0', goal='subject')
            spec = CapabilitySpec(resource='object:alpha', rights={CapabilityRight.READ.value})
            with pytest.raises(CapabilityDenied):
                runtime.capability.issue(issuer, subject, spec)
            runtime.capability.issue_trusted(issuer, 'object:alpha', [CapabilityRight.READ], issued_by='test')
            runtime.capability.issue_trusted(issuer, 'object:alpha', [CapabilityRight.GRANT], issued_by='test')
            cap = runtime.capability.issue(issuer, subject, spec)
            assert runtime.capability.check(subject, 'object:alpha', CapabilityRight.READ)
            assert cap.issuer_cap_id == runtime.capability.capabilities_for(issuer)[-1].cap_id
        finally:
            runtime.close()

    def test_actor_names_cannot_gain_trust_by_prefix(self) -> None:
        runtime = Runtime.open('local')
        try:
            subject = runtime.process.spawn(image='base-agent:v0', goal='subject')
            for actor in ['memoryevil', 'image:evil', 'jit.evil', 'process.fork:evil']:
                with pytest.raises(CapabilityDenied):
                    runtime.capability.issue(actor, subject, CapabilitySpec(resource='object:prefix-collision', rights={CapabilityRight.READ.value}))
        finally:
            runtime.close()

    def test_delegate_can_only_attenuate_delegable_parent_capability(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            runtime.capability.grant(parent, 'filesystem:workspace:src/*', [CapabilityRight.READ, CapabilityRight.WRITE], issued_by='test', delegable=True)
            delegated = runtime.capability.delegate(parent, child, CapabilitySpec(resource='filesystem:workspace:src/main.py', rights={CapabilityRight.READ.value}))
            assert delegated.parent_cap_id == runtime.capability.capabilities_for(parent)[-1].cap_id
            assert runtime.capability.check(child, 'filesystem:workspace:src/main.py', CapabilityRight.READ)
            assert not runtime.capability.check(child, 'filesystem:workspace:src/main.py', CapabilityRight.WRITE)
            with pytest.raises(CapabilityDenied):
                runtime.capability.delegate(parent, child, CapabilitySpec(resource='filesystem:workspace:other.py', rights={CapabilityRight.READ.value}))
        finally:
            runtime.close()

    def test_delegate_audit_failure_rolls_back_capability_attachment_and_event(self, monkeypatch) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='atomic delegator')
            child = runtime.process.spawn(image='base-agent:v0', goal='atomic delegatee')
            runtime.capability.grant(
                parent,
                'object:delegated-atomic',
                [CapabilityRight.READ],
                issued_by='test',
                delegable=True,
            )
            before_capabilities = list(runtime.process.get(child).capabilities)
            before_events = [
                event.event_id
                for event in runtime.events.list(target=child)
                if event.type == EventType.CAPABILITY_GRANTED
            ]
            original_record = runtime.audit.record

            def fail_delegate_audit(*args, **kwargs):
                if kwargs.get('action') == 'capability.delegate':
                    raise RuntimeError('injected delegate audit failure')
                return original_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_delegate_audit)
            with pytest.raises(RuntimeError, match='injected delegate audit failure'):
                runtime.capability.delegate(
                    parent,
                    child,
                    CapabilitySpec(
                        resource='object:delegated-atomic',
                        rights={CapabilityRight.READ.value},
                    ),
                )

            assert runtime.process.get(child).capabilities == before_capabilities
            assert not runtime.capability.check(child, 'object:delegated-atomic', CapabilityRight.READ)
            assert [
                event.event_id
                for event in runtime.events.list(target=child)
                if event.type == EventType.CAPABILITY_GRANTED
            ] == before_events
        finally:
            runtime.close()

    def test_derive_authority_late_validation_failure_is_all_or_nothing(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='atomic authority source')
            child = runtime.process.spawn(image='base-agent:v0', goal='atomic authority target')
            runtime.capability.grant(
                parent,
                'object:allowed',
                [CapabilityRight.READ],
                issued_by='test',
                delegable=True,
            )
            before_capabilities = list(runtime.process.get(child).capabilities)

            with pytest.raises(CapabilityDenied):
                runtime.capability.derive_authority(
                    source_subject=parent,
                    target_subject=child,
                    requested_specs=[
                        CapabilitySpec(
                            resource='object:allowed',
                            rights={CapabilityRight.READ.value},
                        ),
                        CapabilitySpec(
                            resource='object:not-allowed',
                            rights={CapabilityRight.READ.value},
                        ),
                    ],
                    transition_kind='test_atomic_transition',
                )

            assert runtime.process.get(child).capabilities == before_capabilities
            assert not runtime.capability.check(child, 'object:allowed', CapabilityRight.READ)
        finally:
            runtime.close()

    def test_derive_authority_final_audit_failure_rolls_back_every_delegation(self, monkeypatch) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='atomic authority source')
            child = runtime.process.spawn(image='base-agent:v0', goal='atomic authority target')
            for resource in ['object:first', 'object:second']:
                runtime.capability.grant(
                    parent,
                    resource,
                    [CapabilityRight.READ],
                    issued_by='test',
                    delegable=True,
                )
            before_capabilities = list(runtime.process.get(child).capabilities)
            before_event_ids = [event.event_id for event in runtime.events.list(target=child)]
            before_audit_ids = [record.record_id for record in runtime.audit.trace()]
            original_record = runtime.audit.record

            def fail_final_derive_audit(*args, **kwargs):
                if kwargs.get('action') == 'capability.derive_authority':
                    raise RuntimeError('injected derive audit failure')
                return original_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_final_derive_audit)
            with pytest.raises(RuntimeError, match='injected derive audit failure'):
                runtime.capability.derive_authority(
                    source_subject=parent,
                    target_subject=child,
                    requested_specs=[
                        CapabilitySpec(resource='object:first', rights={CapabilityRight.READ.value}),
                        CapabilitySpec(resource='object:second', rights={CapabilityRight.READ.value}),
                    ],
                    transition_kind='test_atomic_transition',
                )

            assert runtime.process.get(child).capabilities == before_capabilities
            assert not runtime.capability.check(child, 'object:first', CapabilityRight.READ)
            assert not runtime.capability.check(child, 'object:second', CapabilityRight.READ)
            assert [event.event_id for event in runtime.events.list(target=child)] == before_event_ids
            assert [record.record_id for record in runtime.audit.trace()] == before_audit_ids
        finally:
            runtime.close()

    def test_delegated_capability_stops_authorizing_when_parent_is_revoked(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            parent_cap = runtime.capability.grant(parent, 'object:shared', [CapabilityRight.READ], issued_by='test', delegable=True)
            delegated = runtime.capability.delegate(parent, child, CapabilitySpec(resource='object:shared', rights={CapabilityRight.READ.value}))
            assert runtime.capability.check(child, 'object:shared', CapabilityRight.READ)
            assert delegated.parent_cap_id == parent_cap.cap_id
            runtime.capability.revoke(parent_cap.cap_id, revoked_by='test')
            assert not runtime.capability.check(child, 'object:shared', CapabilityRight.READ)
        finally:
            runtime.close()

    def test_finite_use_capability_cannot_be_delegated_or_granted_onward(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            subject = runtime.process.spawn(image='base-agent:v0', goal='subject')
            runtime.capability.issue_trusted(
                parent,
                'object:finite',
                [CapabilityRight.READ],
                issued_by='test',
                uses_remaining=1,
                delegable=True,
            )
            with pytest.raises(CapabilityDenied):
                runtime.capability.delegate(parent, child, CapabilitySpec(resource='object:finite', rights={CapabilityRight.READ.value}))
            runtime.capability.issue_trusted(parent, 'object:finite', [CapabilityRight.GRANT], issued_by='test')
            with pytest.raises(CapabilityDenied):
                runtime.capability.issue(parent, subject, CapabilitySpec(resource='object:finite', rights={CapabilityRight.READ.value}))
        finally:
            runtime.close()

    def test_delegate_cannot_launder_restrictive_parent_boundary(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='restricted delegator')
            child = runtime.process.spawn(image='base-agent:v0', goal='restricted child')
            runtime.capability.grant(
                parent,
                'filesystem:workspace:*',
                [CapabilityRight.READ],
                issued_by='test',
                delegable=True,
            )
            runtime.capability.issue_trusted(
                parent,
                'filesystem:workspace:secret.txt',
                [CapabilityRight.READ],
                issued_by='test',
                effect=CapabilityEffect.DENY,
            )

            with pytest.raises(CapabilityDenied, match='restrictive capability'):
                runtime.capability.delegate(
                    parent,
                    child,
                    CapabilitySpec(resource='filesystem:workspace:*', rights={CapabilityRight.READ.value}),
                )

            assert not runtime.capability.check(child, 'filesystem:workspace:public.txt', CapabilityRight.READ)
        finally:
            runtime.close()

    def test_delegate_cannot_use_malformed_allow_parent_authority_rules(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='malformed delegator')
            child = runtime.process.spawn(image='base-agent:v0', goal='malformed child')
            rules = [
                {
                    'rule_id': 'bad.regex.allow.parent',
                    'operation': 'filesystem.read',
                    'effect': 'allow',
                    'risk': 'harmless',
                    'conditions': {'regex_token': '['},
                }
            ]
            before = _capability_admission_state(runtime, parent)

            with pytest.raises(
                ValidationError,
                match='malformed conditions: regex_token',
            ):
                runtime.capability.grant(
                    parent,
                    'filesystem:workspace:*',
                    [CapabilityRight.READ],
                    issued_by='test',
                    constraints={'authority_rules': rules},
                    delegable=True,
                )

            assert _capability_admission_state(runtime, parent) == before
            assert not runtime.capability.check(child, 'filesystem:workspace:public.txt', CapabilityRight.READ)
        finally:
            runtime.close()

    def test_grant_transfer_cannot_launder_ask_parent_boundary(self) -> None:
        runtime = Runtime.open('local')
        try:
            issuer = runtime.process.spawn(image='base-agent:v0', goal='restricted issuer')
            subject = runtime.process.spawn(image='base-agent:v0', goal='restricted subject')
            runtime.capability.grant(issuer, 'object:*', [CapabilityRight.GRANT], issued_by='test')
            runtime.capability.grant(issuer, 'object:*', [CapabilityRight.READ], issued_by='test')
            runtime.capability.issue_trusted(
                issuer,
                'object:needs-human',
                [CapabilityRight.READ],
                issued_by='test',
                effect=CapabilityEffect.ASK,
            )

            with pytest.raises(CapabilityDenied, match='restrictive capability'):
                runtime.capability.issue(
                    issuer,
                    subject,
                    CapabilitySpec(resource='object:*', rights={CapabilityRight.READ.value}),
                )

            assert not runtime.capability.check(subject, 'object:public', CapabilityRight.READ)
        finally:
            runtime.close()

    def test_delegate_cannot_drop_parent_constraints(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            policy_cap = runtime.capability.grant(parent, 'shell:*', [CapabilityRight.EXECUTE], issued_by='test', constraints={runtime.config.shell.policy_capability_key: runtime.config.shell.always_allow_level}, delegable=True)
            with pytest.raises(CapabilityDenied):
                runtime.capability.delegate(parent, child, CapabilitySpec(resource='shell:git', rights={CapabilityRight.EXECUTE.value}))
            delegated = runtime.capability.delegate(parent, child, CapabilitySpec(resource='shell:*', rights={CapabilityRight.EXECUTE.value}, constraints=dict(policy_cap.constraints)))
            assert delegated.constraints == policy_cap.constraints
        finally:
            runtime.close()

    def test_legacy_null_parent_constraint_is_not_covered_by_an_omitted_child_constraint(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='legacy parent')
            parent_capability = runtime.capability.grant(
                parent,
                'git_remote:workspace:origin',
                [CapabilityRight.READ],
                issued_by='test',
                delegable=True,
            )
            legacy = _inject_legacy_capability_constraints(
                runtime,
                parent_capability,
                {'git_remote': None},
            )

            assert not runtime.capability.spec_covers(
                legacy,
                CapabilitySpec(
                    resource='git_remote:workspace:origin',
                    rights={CapabilityRight.READ.value},
                ),
            )
        finally:
            runtime.close()

    def test_legacy_invalid_constraints_remain_readable_but_fail_closed(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='legacy reader')
            capability = runtime.capability.grant(
                pid,
                'object:legacy-invalid',
                [CapabilityRight.READ],
                issued_by='test',
            )
            legacy_constraints = {'authority_rules': {'invalid': 'container'}}
            _inject_legacy_capability_constraints(
                runtime,
                capability,
                legacy_constraints,
            )

            inspected = runtime.capability.inspect(capability.cap_id)
            decision = runtime.capability.authorize(
                pid,
                'object:legacy-invalid',
                CapabilityRight.READ,
            )

            assert inspected['constraints'] == legacy_constraints
            assert inspected['rules'] == []
            assert not decision.allowed
            assert 'invalid_persisted_constraint' in decision.constraint_results
        finally:
            runtime.close()

    def test_delegate_cannot_drop_legacy_null_parent_constraint(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='legacy parent')
            child = runtime.process.spawn(image='base-agent:v0', goal='legacy child')
            parent_capability = runtime.capability.grant(
                parent,
                'git_remote:workspace:origin',
                [CapabilityRight.READ],
                issued_by='test',
                delegable=True,
            )
            _inject_legacy_capability_constraints(
                runtime,
                parent_capability,
                {'git_remote': None},
            )

            with pytest.raises(CapabilityDenied, match='invalid parent constraint'):
                runtime.capability.delegate(
                    parent,
                    child,
                    CapabilitySpec(
                        resource='git_remote:workspace:origin',
                        rights={CapabilityRight.READ.value},
                    ),
                )
        finally:
            runtime.close()

    def test_grant_transfer_cannot_drop_legacy_null_parent_constraint(self) -> None:
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(image='base-agent:v0', goal='legacy grant actor')
            recipient = runtime.process.spawn(image='base-agent:v0', goal='legacy recipient')
            parent_capability = runtime.capability.grant(
                actor,
                'git_remote:workspace:origin',
                [CapabilityRight.READ],
                issued_by='test',
            )
            _inject_legacy_capability_constraints(
                runtime,
                parent_capability,
                {'git_remote': None},
            )
            runtime.capability.grant(
                actor,
                'git_remote:workspace:origin',
                [CapabilityRight.GRANT],
                issued_by='test',
            )

            with pytest.raises(CapabilityDenied, match='invalid parent constraint'):
                runtime.capability.issue(
                    actor,
                    recipient,
                    CapabilitySpec(
                        resource='git_remote:workspace:origin',
                        rights={CapabilityRight.READ.value},
                    ),
                )
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        'legacy_constraints',
        [
            {'unknown_constraint': True},
            {'git_remote': None},
            {'git_allowed_refs': 'refs/heads/main'},
        ],
        ids=['unknown', 'null', 'wrong-type'],
    )
    def test_historical_invalid_ancestor_blocks_continued_derivation_after_reopen(
        self,
        legacy_constraints: dict[str, object],
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / 'runtime.sqlite'
            runtime = Runtime.open(db_path)
            try:
                root_subject = runtime.process.spawn(goal='legacy constraint root')
                child_subject = runtime.process.spawn(goal='legacy constraint child')
                root_read = runtime.capability.grant(
                    root_subject,
                    'object:legacy-chain',
                    [CapabilityRight.READ],
                    issued_by='test',
                    delegable=True,
                )
                child_read = runtime.capability.delegate(
                    root_subject,
                    child_subject,
                    CapabilitySpec(
                        resource='object:legacy-chain',
                        rights={CapabilityRight.READ.value},
                        delegable=True,
                    ),
                )
                runtime.capability.grant(
                    root_subject,
                    'object:legacy-chain',
                    [CapabilityRight.GRANT],
                    issued_by='test',
                    delegable=True,
                )
                runtime.capability.delegate(
                    root_subject,
                    child_subject,
                    CapabilitySpec(
                        resource='object:legacy-chain',
                        rights={CapabilityRight.GRANT.value},
                    ),
                )
                _inject_legacy_capability_constraints(
                    runtime,
                    root_read,
                    legacy_constraints,
                )
            finally:
                runtime.close()

            reopened = Runtime.open(db_path)
            try:
                persisted_root = reopened.store.get_capability(root_read.cap_id)
                persisted_child = reopened.store.get_capability(child_read.cap_id)
                assert persisted_root is not None
                assert persisted_root.constraints == legacy_constraints
                assert persisted_child is not None
                assert persisted_child.parent_cap_id == persisted_root.cap_id
                assert not reopened.capability.check(
                    child_subject,
                    'object:legacy-chain',
                    CapabilityRight.READ,
                )

                grandchild = reopened.process.spawn(goal='blocked delegate target')
                recipient = reopened.process.spawn(goal='blocked grant target')

                def mutation_state(pid: str) -> tuple[object, ...]:
                    process = reopened.process.get(pid)
                    return (
                        tuple(process.capabilities),
                        _capability_admission_state(reopened, pid),
                    )

                before_delegate = mutation_state(grandchild)
                with pytest.raises(
                    CapabilityDenied,
                    match='invalid parent constraint chain',
                ):
                    reopened.capability.delegate(
                        child_subject,
                        grandchild,
                        CapabilitySpec(
                            resource='object:legacy-chain',
                            rights={CapabilityRight.READ.value},
                        ),
                    )
                assert mutation_state(grandchild) == before_delegate

                before_transfer = mutation_state(recipient)
                with pytest.raises(
                    CapabilityDenied,
                    match='invalid parent constraint chain',
                ):
                    reopened.capability.issue(
                        child_subject,
                        recipient,
                        CapabilitySpec(
                            resource='object:legacy-chain',
                            rights={CapabilityRight.READ.value},
                        ),
                    )
                assert mutation_state(recipient) == before_transfer
            finally:
                reopened.close()

    def test_delegate_can_add_git_ref_allowlist_as_restrictive_constraint(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            child = runtime.process.spawn(image='base-agent:v0', goal='child')
            parent_capability = runtime.capability.grant(
                parent,
                'git_remote:workspace:origin',
                [CapabilityRight.READ],
                issued_by='test',
                delegable=True,
            )

            assert runtime.capability.spec_covers(
                parent_capability,
                CapabilitySpec(
                    resource='git_remote:workspace:origin',
                    rights={CapabilityRight.READ.value},
                    constraints={'git_allowed_refs': ['refs/heads/main']},
                ),
            )

            delegated = runtime.capability.delegate(
                parent,
                child,
                CapabilitySpec(
                    resource='git_remote:workspace:origin',
                    rights={CapabilityRight.READ.value},
                    constraints={'git_allowed_refs': ['refs/heads/main']},
                ),
            )

            assert delegated.constraints == {'git_allowed_refs': ['refs/heads/main']}
            assert runtime.capability.authorize(
                child,
                'git_remote:workspace:origin',
                CapabilityRight.READ,
                {'git_remote_ref': 'refs/heads/main'},
            ).allowed
            assert not runtime.capability.authorize(
                child,
                'git_remote:workspace:origin',
                CapabilityRight.READ,
                {'git_remote_ref': 'refs/heads/private'},
            ).allowed
        finally:
            runtime.close()

    def test_delegate_can_narrow_but_not_widen_parent_git_ref_allowlist(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            child = runtime.process.spawn(image='base-agent:v0', goal='child')
            parent_capability = runtime.capability.grant(
                parent,
                'git_remote:workspace:origin',
                [CapabilityRight.READ],
                issued_by='test',
                constraints={
                    'git_allowed_refs': [
                        'refs/heads/main',
                        'refs/heads/release',
                    ]
                },
                delegable=True,
            )
            narrowed = CapabilitySpec(
                resource='git_remote:workspace:origin',
                rights={CapabilityRight.READ.value},
                constraints={'git_allowed_refs': ['refs/heads/main']},
            )
            widened = CapabilitySpec(
                resource='git_remote:workspace:origin',
                rights={CapabilityRight.READ.value},
                constraints={
                    'git_allowed_refs': [
                        'refs/heads/main',
                        'refs/heads/private',
                    ]
                },
            )

            assert runtime.capability.spec_covers(parent_capability, narrowed)
            assert not runtime.capability.spec_covers(parent_capability, widened)
            delegated = runtime.capability.delegate(parent, child, narrowed)
            assert delegated.constraints == {
                'git_allowed_refs': ['refs/heads/main']
            }
            with pytest.raises(CapabilityDenied, match='cannot widen'):
                runtime.capability.delegate(parent, child, widened)
        finally:
            runtime.close()

    def test_delegate_selects_a_valid_parent_when_a_more_specific_parent_is_constrained(
        self,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            child = runtime.process.spawn(image='base-agent:v0', goal='child')
            unconstrained = runtime.capability.grant(
                parent,
                'filesystem:workspace:*',
                [CapabilityRight.READ],
                issued_by='test',
                delegable=True,
            )
            runtime.capability.grant(
                parent,
                'filesystem:workspace:README.md',
                [CapabilityRight.READ],
                issued_by='test',
                constraints={
                    'authority_rules': [
                        {
                            'rule_id': 'parent.readme.allow',
                            'operation': 'filesystem.read',
                            'effect': 'allow',
                            'risk': 'harmless',
                            'conditions': {'path': 'README.md'},
                        }
                    ]
                },
                delegable=True,
            )
            spec = CapabilitySpec(
                resource='filesystem:workspace:README.md',
                rights={CapabilityRight.READ.value},
            )

            selected_parent = runtime.capability.validate_delegation(parent, spec)
            delegated = runtime.capability.delegate(parent, child, spec)

            assert selected_parent.cap_id == unconstrained.cap_id
            assert delegated.parent_cap_id == unconstrained.cap_id
            assert delegated.constraints == {}
        finally:
            runtime.close()

    def test_spec_coverage_rejects_new_authority_shaping_constraints(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = CapabilitySpec(
                resource='shell:python',
                rights={CapabilityRight.EXECUTE.value},
                delegable=True,
            )
            requested = CapabilitySpec(
                resource='shell:python',
                rights={CapabilityRight.EXECUTE.value},
                constraints={
                    'authority_rules': [
                        {
                            'rule_id': 'delegated.python.allow',
                            'operation': 'shell.run',
                            'effect': 'allow',
                            'risk': 'harmless',
                            'conditions': {
                                'argv': ['python', '--version'],
                                'match': 'exact',
                            },
                        }
                    ]
                },
            )

            assert not runtime.capability.spec_covers(parent, requested)
        finally:
            runtime.close()

    def test_delegation_cannot_increase_parent_max_depth(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            runtime.capability.issue(
                'test',
                parent,
                {
                    'resource': 'object:limited',
                    'rights': ['read'],
                    'delegation': {'delegable': True, 'max_delegation_depth': 1},
                },
                require_authority=False,
            )
            with pytest.raises(CapabilityDenied):
                runtime.capability.delegate(
                    parent,
                    child,
                    CapabilitySpec(resource='object:limited', rights={CapabilityRight.READ.value}, max_delegation_depth=10),
                )
            delegated = runtime.capability.delegate(parent, child, CapabilitySpec(resource='object:limited', rights={CapabilityRight.READ.value}))
            assert delegated.max_delegation_depth == 1
        finally:
            runtime.close()

    def test_revoke_requires_holder_issuer_or_revoke_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='owner')
            stranger = runtime.process.spawn(image='base-agent:v0', goal='stranger')
            cap = runtime.capability.grant(owner, 'object:revocable', [CapabilityRight.READ], issued_by='test')
            with pytest.raises(CapabilityDenied):
                runtime.capability.revoke(cap.cap_id, revoked_by=stranger)
            runtime.capability.revoke(cap.cap_id, revoked_by=owner, reason='holder abandoned')
            assert not runtime.capability.check(owner, 'object:revocable', CapabilityRight.READ)
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        'defensive_status',
        [CapabilityStatus.REVOKED, CapabilityStatus.DISABLED],
    )
    def test_exec_revocation_staging_preserves_newer_defensive_state(
        self,
        defensive_status: CapabilityStatus,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            subject = runtime.process.spawn(goal='exec revocation defensive state')
            cap = runtime.capability.grant(
                subject,
                'shell:defensive-state',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            if defensive_status == CapabilityStatus.REVOKED:
                runtime.capability.revoke(
                    cap.cap_id,
                    revoked_by='defender',
                    require_authority=False,
                )
            else:
                runtime.capability.disable_subject_capability(
                    cap.cap_id,
                    actor='defender',
                    reason='defensive disable',
                )
            defended = runtime.store.get_capability(cap.cap_id)
            assert defended is not None and defended.status == defensive_status
            before_event_ids = [event.event_id for event in runtime.events.list()]
            before_audit_ids = [record.record_id for record in runtime.audit.trace()]

            staged = runtime.capability.stage_exec_revocation(
                cap.cap_id,
                rollback_token='exec-publication-token',
            )

            assert staged == defended
            assert runtime.store.get_capability(cap.cap_id) == defended
            assert [event.event_id for event in runtime.events.list()] == before_event_ids
            assert [record.record_id for record in runtime.audit.trace()] == before_audit_ids
        finally:
            runtime.close()

    def test_exec_revocation_staging_transitions_active_capability(self) -> None:
        runtime = Runtime.open('local')
        try:
            subject = runtime.process.spawn(goal='exec revocation active control')
            cap = runtime.capability.grant(
                subject,
                'shell:active-control',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            before_event_ids = {event.event_id for event in runtime.events.list()}
            before_audit_ids = {record.record_id for record in runtime.audit.trace()}

            staged = runtime.capability.stage_exec_revocation(
                cap.cap_id,
                rollback_token='exec-publication-token',
            )

            assert staged.status == CapabilityStatus.EXEC_REVOKED
            assert staged.metadata['_agent_libos_exec_rollback_token'] == 'exec-publication-token'
            assert runtime.store.get_capability(cap.cap_id) == staged
            new_events = [
                event
                for event in runtime.events.list()
                if event.event_id not in before_event_ids
            ]
            new_audits = [
                record
                for record in runtime.audit.trace()
                if record.record_id not in before_audit_ids
            ]
            assert len(new_events) == 1
            assert new_events[0].type == EventType.CAPABILITY_REVOKED
            assert new_events[0].source == 'process.exec'
            assert new_events[0].target == subject
            assert len(new_audits) == 1
            assert new_audits[0].actor == 'process.exec'
            assert new_audits[0].action == 'capability.revoke'
            assert new_audits[0].capability_refs == [cap.cap_id]
        finally:
            runtime.close()

    def test_exec_revocation_staging_loses_exact_state_cas_to_defensive_mutation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            subject = runtime.process.spawn(goal='exec revocation CAS race')
            cap = runtime.capability.grant(
                subject,
                'shell:cas-race',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            original_require = runtime.capability.mutations._require_revocable

            def disable_after_read(cap_id: str):
                current = original_require(cap_id)
                runtime.store.update_capability(
                    replace(
                        current,
                        status=CapabilityStatus.DISABLED,
                        metadata={**current.metadata, 'disabled_by': 'defender'},
                    )
                )
                return current

            monkeypatch.setattr(
                runtime.capability.mutations,
                '_require_revocable',
                disable_after_read,
            )
            before_event_ids = [event.event_id for event in runtime.events.list()]
            before_audit_ids = [record.record_id for record in runtime.audit.trace()]

            staged = runtime.capability.stage_exec_revocation(
                cap.cap_id,
                rollback_token='exec-publication-token',
            )

            assert staged.status == CapabilityStatus.DISABLED
            assert staged.metadata['disabled_by'] == 'defender'
            assert runtime.store.get_capability(cap.cap_id) == staged
            assert [event.event_id for event in runtime.events.list()] == before_event_ids
            assert [record.record_id for record in runtime.audit.trace()] == before_audit_ids
        finally:
            runtime.close()

    def test_issue_reauthorizes_unlimited_grant_inside_mutation_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(goal='issue authority race actor')
            subject = runtime.process.spawn(goal='issue authority race subject')
            authority = runtime.capability.issue_trusted(
                actor,
                'object:authority-race',
                [CapabilityRight.ADMIN],
                issued_by='test.host',
            )
            original = runtime.capability._require_issue_authority

            def revoke_after_preflight(who: str, spec: CapabilitySpec):
                decision = original(who, spec)
                runtime.capability.revoke(
                    authority.cap_id,
                    revoked_by='test.host',
                    require_authority=False,
                )
                return decision

            monkeypatch.setattr(
                runtime.capability,
                '_require_issue_authority',
                revoke_after_preflight,
            )

            with pytest.raises(CapabilityDenied, match='authority changed'):
                runtime.capability.issue(
                    actor,
                    subject,
                    CapabilitySpec(
                        resource='object:authority-race',
                        rights={CapabilityRight.READ.value},
                    ),
                )

            assert not runtime.capability.check(
                subject,
                'object:authority-race',
                CapabilityRight.READ,
            )
        finally:
            runtime.close()

    def test_authority_transaction_rejects_new_deny_after_preflight(self) -> None:
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(goal='new deny authority race actor')
            resource = 'object:new-deny-race'
            runtime.capability.issue_trusted(
                actor,
                resource,
                [CapabilityRight.ADMIN],
                issued_by='test.host',
            )
            preflight = runtime.capability.require(
                actor,
                resource,
                CapabilityRight.ADMIN,
                consume=False,
            )
            runtime.capability.issue_trusted(
                actor,
                resource,
                [CapabilityRight.ADMIN],
                issued_by='test.host',
                effect=CapabilityEffect.DENY,
            )

            with pytest.raises(CapabilityDenied, match='authority changed'):
                with runtime.capability.authority_transaction(
                    [preflight],
                    actor=actor,
                    operation='new deny race regression',
                ):
                    pytest.fail('mutation body must not run after a new deny')
        finally:
            runtime.close()

    def test_one_time_revoke_authority_is_reserved_before_target_mutation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='owner')
            actor = runtime.process.spawn(image='base-agent:v0', goal='actor')
            first = runtime.capability.issue_trusted(
                owner,
                'object:protected-a',
                [CapabilityRight.READ],
                issued_by='issuer',
                effect=CapabilityEffect.DENY,
            )
            second = runtime.capability.issue_trusted(
                owner,
                'object:protected-b',
                [CapabilityRight.READ],
                issued_by='issuer',
                effect=CapabilityEffect.DENY,
            )
            runtime.capability.issue_trusted(
                actor,
                'object:*',
                [CapabilityRight.REVOKE],
                issued_by='issuer',
                uses_remaining=1,
            )
            barrier = Barrier(2)
            original = runtime.capability._require_revoke_authority

            def gated_require(who: str, cap: object):
                decision = original(who, cap)
                barrier.wait(timeout=5)
                return decision

            monkeypatch.setattr(runtime.capability, '_require_revoke_authority', gated_require)

            def revoke(cap_id: str) -> str:
                try:
                    runtime.capability.revoke(cap_id, revoked_by=actor)
                    return 'ok'
                except CapabilityDenied:
                    return 'denied'

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(revoke, [first.cap_id, second.cap_id]))

            statuses = {
                runtime.store.get_capability(first.cap_id).status.value,
                runtime.store.get_capability(second.cap_id).status.value,
            }
            assert sorted(results) == ['denied', 'ok']
            assert statuses == {'active', 'revoked'}
        finally:
            runtime.close()

    @pytest.mark.parametrize('failed_sink', ['event', 'audit'])
    def test_revoke_sink_failure_rolls_back_target_and_one_time_authority(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failed_sink: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='revoke rollback owner')
            actor = runtime.process.spawn(image='base-agent:v0', goal='revoke rollback actor')
            target = runtime.capability.issue_trusted(
                owner,
                'object:revoke-atomic',
                [CapabilityRight.READ],
                issued_by='issuer',
            )
            authority = runtime.capability.issue_trusted(
                actor,
                target.resource,
                [CapabilityRight.REVOKE],
                issued_by='issuer',
                uses_remaining=1,
            )
            if failed_sink == 'event':
                original_emit = runtime.events.emit

                def fail_revoke_event(event_type, *args, **kwargs):
                    if event_type == EventType.CAPABILITY_REVOKED:
                        raise RuntimeError('injected revoke event failure')
                    return original_emit(event_type, *args, **kwargs)

                monkeypatch.setattr(runtime.events, 'emit', fail_revoke_event)
            else:
                original_record = runtime.audit.record

                def fail_revoke_audit(*args, **kwargs):
                    if kwargs.get('action') == 'capability.revoke':
                        raise RuntimeError('injected revoke audit failure')
                    return original_record(*args, **kwargs)

                monkeypatch.setattr(runtime.audit, 'record', fail_revoke_audit)

            with pytest.raises(RuntimeError, match=f'revoke {failed_sink} failure'):
                runtime.capability.revoke(target.cap_id, revoked_by=actor)

            persisted_target = runtime.store.get_capability(target.cap_id)
            persisted_authority = runtime.store.get_capability(authority.cap_id)
            assert persisted_target is not None and persisted_target.active
            assert persisted_authority is not None and persisted_authority.active
            assert persisted_authority.uses_remaining == 1
            assert runtime.capability.check(owner, target.resource, CapabilityRight.READ)
        finally:
            runtime.close()

    def test_reserved_use_restore_does_not_reactivate_explicit_revoke(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='reservation revoke race')
            cap = runtime.capability.issue_trusted(
                pid,
                'object:reservation-race',
                [CapabilityRight.READ],
                issued_by='issuer',
                uses_remaining=1,
            )
            decision = runtime.capability.authorize(pid, cap.resource, CapabilityRight.READ)
            reservation_id = runtime.capability.reserve_decision_use(
                decision,
                used_by='test',
                reason='provider preflight reservation',
            )
            assert reservation_id is not None

            runtime.capability.revoke(cap.cap_id, revoked_by='issuer', reason='explicit revoke wins')
            restored = runtime.capability.restore_reserved_use(
                reservation_id,
                restored_by='test',
                reason='provider failed before commit',
            )

            assert restored is None
            after = runtime.store.get_capability(cap.cap_id)
            assert after.status.value == 'revoked'
            assert after.uses_remaining == 0
            assert not runtime.capability.check(pid, cap.resource, CapabilityRight.READ)
        finally:
            runtime.close()

    def test_require_consumes_finite_use_by_default(self) -> None:
        runtime = Runtime.open('local')
        try:
            cap = runtime.capability.issue_trusted(
                'worker',
                'object:required-once',
                [CapabilityRight.READ],
                issued_by='issuer',
                uses_remaining=1,
            )

            decision = runtime.capability.require('worker', cap.resource, CapabilityRight.READ)

            assert decision.allowed
            assert decision.consume_capability_id is None
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
            with pytest.raises(CapabilityDenied):
                runtime.capability.require('worker', cap.resource, CapabilityRight.READ)
        finally:
            runtime.close()

    def test_require_consume_false_supports_explicit_effect_reservation(self) -> None:
        runtime = Runtime.open('local')
        try:
            cap = runtime.capability.issue_trusted(
                'worker',
                'object:reserved-once',
                [CapabilityRight.READ],
                issued_by='issuer',
                uses_remaining=1,
            )

            decision = runtime.capability.require(
                'worker',
                cap.resource,
                CapabilityRight.READ,
                consume=False,
            )

            assert decision.consume_capability_id == cap.cap_id
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
            reservation_id = runtime.capability.reserve_decision_use(
                decision,
                used_by='test',
                reason='explicit boundary reservation',
            )
            assert reservation_id is not None
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
        finally:
            runtime.close()

    def test_inflight_reservation_is_abandoned_fail_closed_after_runtime_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / 'runtime.sqlite')
            runtime = Runtime.open(db_path)
            try:
                cap = runtime.capability.issue_trusted(
                    'crashed-worker',
                    'object:crash-boundary',
                    [CapabilityRight.READ],
                    issued_by='issuer',
                    uses_remaining=1,
                )
                decision = runtime.capability.authorize('crashed-worker', cap.resource, CapabilityRight.READ)
                reservation_id = runtime.capability.reserve_decision_use(
                    decision,
                    used_by='test',
                    reason='simulate provider call interrupted by runtime exit',
                )
                assert reservation_id is not None
            finally:
                runtime.close()

            reopened = Runtime.open(db_path)
            try:
                assert reopened.capability.restore_reserved_use(
                    reservation_id,
                    restored_by='test',
                    reason='late cleanup from previous runtime',
                ) is None
                persisted = reopened.store.get_capability(cap.cap_id)
                assert persisted is not None
                assert persisted.uses_remaining == 0
                assert persisted.status.value == 'revoked'
                rows = reopened.store.select_table_rows(
                    'capability_use_reservations',
                    'reservation_id = ?',
                    [reservation_id],
                )
                assert rows[0]['status'] == 'abandoned'
            finally:
                reopened.close()

    def test_holder_cannot_self_revoke_restrictive_capability(self) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='owner')
            deny = runtime.capability.issue_trusted(owner, 'filesystem:workspace:secret.txt', [CapabilityRight.READ], issued_by='test', effect=CapabilityEffect.DENY)
            with pytest.raises(CapabilityDenied):
                runtime.capability.revoke(deny.cap_id, revoked_by=owner)
            assert not runtime.capability.check(owner, 'filesystem:workspace:secret.txt', CapabilityRight.READ)
            runtime.capability.revoke(deny.cap_id, revoked_by='test')
            assert runtime.capability.inspect(deny.cap_id)['status'] == 'revoked'
        finally:
            runtime.close()

    def test_capability_expiry_must_be_valid_iso_timestamp(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='expiry')
            with pytest.raises(ValidationError):
                runtime.capability.grant(pid, 'object:bad-expiry', [CapabilityRight.READ], issued_by='test', expires_at='zzzz')
            cap = runtime.capability.grant(
                pid,
                'object:good-expiry',
                [CapabilityRight.READ],
                issued_by='test',
                expires_at='2999-01-01T00:00:00Z',
            )
            assert runtime.capability.inspect(cap.cap_id)['expires_at'] == '2999-01-01T00:00:00+00:00'
        finally:
            runtime.close()

class TestCapabilityRuntimeInterface:

    def test_default_images_expose_only_low_risk_capability_tools(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='tool table')
            process = runtime.process.get(pid)
            assert 'list_capabilities' in process.tool_table
            assert 'inspect_capability' in process.tool_table
            assert 'delegate_capability' not in process.tool_table
            assert 'revoke_capability' not in process.tool_table
            listed = runtime.tools.call(pid, 'list_capabilities', {})
            assert listed.ok, listed.error
            assert listed.payload['capabilities']
        finally:
            runtime.close()

    def test_capability_syscalls_do_not_bypass_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            other = runtime.process.spawn(image='base-agent:v0', goal='other')
            runtime.capability.grant(parent, 'object:shared', [CapabilityRight.READ], issued_by='test', delegable=True)
            parent_session = LibOSSyscallSession(runtime, parent)
            other_session = LibOSSyscallSession(runtime, other)
            listed = self._run(parent_session.handle('capability.list', {}))
            delegated = self._run(parent_session.handle('capability.delegate', {'child_pid': child, 'resource': 'object:shared', 'rights': [CapabilityRight.READ.value]}))
            assert listed['capabilities']
            assert runtime.capability.check(child, 'object:shared', CapabilityRight.READ)
            assert delegated['capability']['subject'] == child
            with pytest.raises(CapabilityDenied):
                self._run(other_session.handle('capability.inspect', {'capability_id': delegated['capability']['cap_id']}))
            with pytest.raises(CapabilityDenied):
                self._run(parent_session.handle('capability.delegate', {'child_pid': other, 'resource': 'object:shared', 'rights': [CapabilityRight.READ.value]}))
            deny = runtime.capability.issue_trusted(parent, 'object:blocked', [CapabilityRight.READ], issued_by='test', effect=CapabilityEffect.DENY)
            with pytest.raises(CapabilityDenied):
                self._run(parent_session.handle('capability.revoke', {'capability_id': deny.cap_id}))
        finally:
            runtime.close()

    def test_capability_delegate_syscall_rejects_type_laundering_before_delegation(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='strict syscall parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'strict syscall child')
            runtime.capability.grant(
                parent,
                'object:strict-syscall-delegate',
                [CapabilityRight.READ],
                issued_by='test',
                delegable=True,
            )
            session = LibOSSyscallSession(runtime, parent)
            invalid_fields = [
                {'delegable': 'false'},
                {'revocable': 0},
                {'uses_remaining': True},
                {'uses_remaining': 1.0},
                {'uses_remaining': '1'},
                {'uses_remaining': 0},
                {'uses_remaining': -1},
                {'resource': 7},
                {'rights': 'read'},
                {'rights': {'read': False}},
                {'rights': ('read',)},
                {'constraints': False},
                {'constraints': []},
                {'constraints': [('purpose', 'laundered')]},
                {'metadata': False},
                {'metadata': []},
                {'child_pid': 7},
            ]

            for invalid in invalid_fields:
                before_caps = runtime.store.list_capabilities(subject=child)
                before_audit_ids = {
                    record.record_id for record in runtime.store.list_audit()
                }
                with pytest.raises(
                    ValidationError,
                    match='delegable|revocable|uses_remaining|resource|rights|constraints|metadata|child_pid',
                ):
                    self._run(
                        session.handle(
                            'capability.delegate',
                            {
                                'child_pid': child,
                                'resource': 'object:strict-syscall-delegate',
                                'rights': [CapabilityRight.READ.value],
                                **invalid,
                            },
                        )
                    )

                assert runtime.store.list_capabilities(subject=child) == before_caps
                new_audit = [
                    record
                    for record in runtime.store.list_audit()
                    if record.record_id not in before_audit_ids
                ]
                assert all(
                    record.action not in {'capability.issue', 'capability.delegate'}
                    for record in new_audit
                )

            delegated = self._run(
                session.handle(
                    'capability.delegate',
                    {
                        'child_pid': child,
                        'resource': 'object:strict-syscall-delegate',
                        'rights': [CapabilityRight.READ.value],
                        'delegable': False,
                        'revocable': True,
                        'uses_remaining': 1,
                    },
                )
            )
            assert delegated['capability']['uses_remaining'] == 1
            assert not delegated['capability']['delegable']
            assert delegated['capability']['revocable']
        finally:
            runtime.close()

    def test_delegate_capability_tool_rejects_authority_scalar_coercion_without_side_effects(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='strict tool parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'strict tool child')
            runtime.capability.grant(
                parent,
                'object:strict-tool-delegate',
                [CapabilityRight.READ],
                issued_by='test',
                delegable=True,
            )
            tool = DelegateCapabilityTool()
            context = ToolContext(
                trace_id='strict-delegate-trace',
                call_id='strict-delegate-call',
                pid=parent,
                runtime=runtime,
            )
            invalid_fields = [
                {'uses_remaining': True},
                {'uses_remaining': '1'},
                {'uses_remaining': 1.0},
                {'uses_remaining': 0},
                {'uses_remaining': -1},
                {'delegable': 'false'},
                {'delegable': 0},
                {'delegable': 1},
            ]

            for invalid in invalid_fields:
                before_caps = runtime.store.list_capabilities(subject=child)
                before_audit = runtime.store.list_audit()
                before_events = runtime.events.list()
                result = tool.invoke(
                    {
                        'child_pid': child,
                        'resource': 'object:strict-tool-delegate',
                        'rights': [CapabilityRight.READ.value],
                        **invalid,
                    },
                    context,
                )

                assert not result.ok
                assert result.error is not None
                assert result.error.code == ToolErrorCode.VALIDATION_ERROR
                assert runtime.store.list_capabilities(subject=child) == before_caps
                assert runtime.store.list_audit() == before_audit
                assert runtime.events.list() == before_events

            valid = tool.invoke(
                {
                    'child_pid': child,
                    'resource': 'object:strict-tool-delegate',
                    'rights': [CapabilityRight.READ.value],
                    'uses_remaining': 1,
                    'delegable': False,
                },
                context,
            )
            assert valid.ok, valid.error
            assert valid.data['capability']['uses_remaining'] == 1
            assert not valid.data['capability']['delegable']
        finally:
            runtime.close()

    def test_spawn_child_invalid_capability_inheritance_is_preflighted(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            runtime.capability.grant(parent, 'shell:*', [CapabilityRight.EXECUTE], issued_by='test', constraints={runtime.config.shell.policy_capability_key: runtime.config.shell.always_allow_level}, delegable=True)
            before = len(runtime.process.list())
            with pytest.raises(CapabilityDenied):
                runtime.spawn_child_process(parent, 'should fail', inherit_capabilities=[{'resource': 'shell:git', 'rights': [CapabilityRight.EXECUTE.value]}])
            assert len(runtime.process.list()) == before
        finally:
            runtime.close()

    def test_capabilities_cli_outputs_stable_json_and_enforces_actor_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / 'runtime.sqlite')
            parent = _run_cli_json(['--db', db_path, 'spawn', '--goal', 'parent'])
            granted = _run_cli_json(['--db', db_path, 'capabilities', 'grant', parent['pid'], 'object:cli', '--rights', 'read', '--delegable'])
            listed = _run_cli_json(['--db', db_path, 'capabilities', 'list', '--subject', parent['pid']])
            explained = _run_cli_json(['--db', db_path, 'capabilities', 'explain', parent['pid'], 'object:cli', 'read'])
            assert granted['subject'] == parent['pid']
            assert granted['cap_id'] in {capability['cap_id'] for capability in listed}
            assert explained['allowed']
            with pytest.raises(CapabilityDenied):
                _run_cli_json(['--db', db_path, 'capabilities', '--actor-pid', parent['pid'], 'grant', 'other', 'object:denied', '--rights', 'read'])

    def _run(self, awaitable):
        import asyncio
        return asyncio.run(awaitable)

def _run_cli_json(argv: list[str]):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        cli_main(argv)
    return json.loads(stdout.getvalue())
