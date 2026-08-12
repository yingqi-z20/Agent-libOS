from __future__ import annotations

from agent_libos.models import (
    DataFlowDirection,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
)
from agent_libos.sdk import AuthorityMode, ResourcePolicy
from agent_libos.sdk.descriptors import protected_operation_descriptor as operation


PROTECTED_OPERATION_DESCRIPTORS = (
    operation(
        "primitive.filesystem.validate_directory",
        "filesystem",
        "state",
        resource_policy=ResourcePolicy.OPTIONAL,
        information_flow=True,
    ),
    operation(
        "primitive.filesystem.read_text",
        "filesystem",
        "read_bytes",
        resource_policy=ResourcePolicy.REQUIRED,
        information_flow=True,
        data_flow_direction=DataFlowDirection.INGRESS,
    ),
    operation(
        "primitive.filesystem.read_bytes",
        "filesystem",
        "read_bytes",
        resource_policy=ResourcePolicy.REQUIRED,
        information_flow=True,
        data_flow_direction=DataFlowDirection.INGRESS,
    ),
    operation(
        "primitive.filesystem.write_text",
        "filesystem",
        "write_text",
        resource_policy=ResourcePolicy.REQUIRED,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.EGRESS,
    ),
    operation(
        "primitive.filesystem.read_directory",
        "filesystem",
        "list_directory",
        resource_policy=ResourcePolicy.REQUIRED,
        information_flow=True,
        data_flow_direction=DataFlowDirection.INGRESS,
    ),
    operation(
        "primitive.filesystem.write_directory",
        "filesystem",
        "make_directory",
        resource_policy=ResourcePolicy.NONE,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.EGRESS,
    ),
    operation(
        "primitive.filesystem.delete_file",
        "filesystem",
        "delete_file",
        resource_policy=ResourcePolicy.NONE,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.EGRESS,
    ),
    operation(
        "primitive.filesystem.delete_directory",
        "filesystem",
        "delete_directory",
        resource_policy=ResourcePolicy.NONE,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.EGRESS,
    ),
    operation(
        "primitive.clock.now",
        "clock",
        "now",
        resource_policy=ResourcePolicy.NONE,
        information_flow=True,
    ),
    operation(
        "primitive.clock.sleep",
        "clock",
        "sleep",
        resource_policy=ResourcePolicy.NONE,
        information_flow=True,
    ),
    operation(
        "primitive.shell.run",
        "shell",
        "run",
        resource_policy=ResourcePolicy.OPTIONAL,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
    ),
    operation(
        "primitive.git.read",
        "git",
        "read",
        resource_policy=ResourcePolicy.OPTIONAL,
        information_flow=True,
        data_flow_direction=DataFlowDirection.INGRESS,
    ),
    operation(
        "primitive.git.mutate",
        "git",
        "mutate",
        resource_policy=ResourcePolicy.OPTIONAL,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
        classifier_failure_label="post_git_mutation_failure",
    ),
    operation(
        "primitive.git.fetch",
        "git",
        "fetch",
        resource_policy=ResourcePolicy.OPTIONAL,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
        classifier_failure_label="post_git_fetch_failure",
    ),
    operation(
        "primitive.git.push",
        "git",
        "push",
        resource_policy=ResourcePolicy.OPTIONAL,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.EGRESS,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
        classifier_failure_label="post_git_push_failure",
    ),
    operation(
        "primitive.git.pull_request",
        "git",
        "pull_request",
        resource_policy=ResourcePolicy.OPTIONAL,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
        classifier_failure_label="post_git_pull_request_failure",
    ),
    operation(
        "primitive.jsonrpc.call",
        "jsonrpc",
        "call",
        resource_policy=ResourcePolicy.REQUIRED,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
        classifier_failure_label="post_call_failure",
    ),
    operation(
        "primitive.mcp.list_tools",
        "mcp",
        "list_tools",
        resource_policy=ResourcePolicy.OPTIONAL,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        preflight_classifier=True,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
        classifier_failure_label="post_list_tools_failure",
    ),
    operation(
        "primitive.mcp.discover",
        "mcp",
        "discover",
        resource_policy=ResourcePolicy.OPTIONAL,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        preflight_classifier=True,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
        classifier_failure_label="post_discover_failure",
    ),
    operation(
        "primitive.mcp.discover.internal",
        "mcp",
        "discover",
        resource_policy=ResourcePolicy.OPTIONAL,
        authority_mode=AuthorityMode.RUNTIME_INTERNAL,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        internal_reason="host protocol discovery is an operator-controlled provider operation",
        preflight_classifier=True,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
        classifier_failure_label="post_discover_failure",
    ),
    operation(
        "primitive.mcp.list_tools.internal",
        "mcp",
        "list_tools",
        resource_policy=ResourcePolicy.OPTIONAL,
        authority_mode=AuthorityMode.RUNTIME_INTERNAL,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        internal_reason="host registry refresh is an operator-controlled provider operation",
        preflight_classifier=True,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
        classifier_failure_label="post_list_tools_failure",
    ),
    operation(
        "primitive.mcp.probe_candidate.internal",
        "mcp",
        "probe_candidate",
        resource_policy=ResourcePolicy.OPTIONAL,
        authority_mode=AuthorityMode.RUNTIME_INTERNAL,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        internal_reason=(
            "explicit reviewed Host onboarding probe of an unregistered MCP "
            "transport; it creates evidence but never registry authority"
        ),
        require_classifier=False,
    ),
    operation(
        "primitive.mcp.call",
        "mcp",
        "call_tool",
        resource_policy=ResourcePolicy.REQUIRED,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        preflight_classifier=True,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
        classifier_failure_label="post_call_failure",
    ),
    *tuple(
        operation(
            f"primitive.mcp.{name}",
            "mcp",
            name,
            resource_policy=ResourcePolicy.REQUIRED,
            state_mutation=name
            in {
                "continuation.respond",
                "continuation.cancel",
                "tasks.update",
                "tasks.cancel",
            },
            information_flow=True,
            data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
            classifier_failure_rollback_class=(
                ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
                if name == "tasks.get"
                else ExternalEffectRollbackClass.IRREVERSIBLE
            ),
            classifier_failure_rollback_status=(
                ExternalEffectRollbackStatus.NOT_REQUIRED
                if name == "tasks.get"
                else ExternalEffectRollbackStatus.NOT_SUPPORTED
            ),
            classifier_failure_label=f"post_{name.replace('.', '_')}_failure",
            require_classifier=False,
        )
        for name in (
            "continuation.respond",
            "continuation.cancel",
            "tasks.get",
            "tasks.update",
            "tasks.cancel",
        )
    ),
    *tuple(
        operation(
            f"primitive.mcp.{name}.internal",
            "mcp",
            name,
            resource_policy=ResourcePolicy.OPTIONAL,
            authority_mode=AuthorityMode.RUNTIME_INTERNAL,
            state_mutation=name
            in {
                "continuation.respond",
                "continuation.cancel",
                "tasks.update",
                "tasks.cancel",
            },
            information_flow=True,
            data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
            internal_reason=(
                "Host continuation and Task actions revalidate the durable "
                "owner, registry, authorization, Human, and provider fences"
            ),
            classifier_failure_rollback_class=(
                ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
                if name == "tasks.get"
                else ExternalEffectRollbackClass.IRREVERSIBLE
            ),
            classifier_failure_rollback_status=(
                ExternalEffectRollbackStatus.NOT_REQUIRED
                if name == "tasks.get"
                else ExternalEffectRollbackStatus.NOT_SUPPORTED
            ),
            classifier_failure_label=f"post_{name.replace('.', '_')}_failure",
            require_classifier=False,
        )
        for name in (
            "continuation.respond",
            "continuation.cancel",
            "tasks.get",
            "tasks.update",
            "tasks.cancel",
        )
    ),
    *tuple(
        operation(
            f"primitive.mcp.{name}",
            "mcp",
            name,
            resource_policy=ResourcePolicy.OPTIONAL,
            information_flow=True,
            data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
            require_classifier=False,
        )
        for name in (
            "resources.list",
            "resource_templates.list",
            "resources.read",
            "prompts.list",
            "prompts.get",
            "completion.complete",
        )
    ),
    *tuple(
        operation(
            f"primitive.mcp.{name}.internal",
            "mcp",
            name,
            resource_policy=ResourcePolicy.OPTIONAL,
            authority_mode=AuthorityMode.RUNTIME_INTERNAL,
            information_flow=True,
            data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
            internal_reason=(
                "explicit local Host MCP v3 client operation through the same "
                "transport, data-flow, registry, effect, and evidence boundary"
            ),
            require_classifier=False,
        )
        for name in (
            "resources.list",
            "resource_templates.list",
            "resources.read",
            "prompts.list",
            "prompts.get",
            "completion.complete",
        )
    ),
    *tuple(
        operation(
            f"primitive.mcp.{name}",
            "mcp",
            name,
            resource_policy=ResourcePolicy.OPTIONAL,
            information_flow=True,
            data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
            require_classifier=False,
        )
        for name in (
            "subscriptions.status",
            "subscriptions.events",
        )
    ),
    operation(
        "primitive.mcp.subscriptions.start",
        "mcp",
        "subscriptions.start",
        resource_policy=ResourcePolicy.OPTIONAL,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.ROLLBACKABLE,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_APPLIED,
        classifier_failure_label="post_subscription_start_failure",
        require_classifier=False,
    ),
    operation(
        "primitive.mcp.subscriptions.stop",
        "mcp",
        "subscriptions.stop",
        resource_policy=ResourcePolicy.OPTIONAL,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
        classifier_failure_label="post_subscription_stop_failure",
        require_classifier=False,
    ),
    *tuple(
        operation(
            f"primitive.mcp.{name}.internal",
            "mcp",
            name,
            resource_policy=ResourcePolicy.OPTIONAL,
            authority_mode=AuthorityMode.RUNTIME_INTERNAL,
            state_mutation=name in {"subscriptions.start", "subscriptions.stop"},
            information_flow=True,
            data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
            classifier_failure_rollback_class=(
                ExternalEffectRollbackClass.ROLLBACKABLE
                if name == "subscriptions.start"
                else (
                    ExternalEffectRollbackClass.IRREVERSIBLE
                    if name == "subscriptions.stop"
                    else ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
                )
            ),
            classifier_failure_rollback_status=(
                ExternalEffectRollbackStatus.NOT_APPLIED
                if name == "subscriptions.start"
                else (
                    ExternalEffectRollbackStatus.NOT_SUPPORTED
                    if name == "subscriptions.stop"
                    else ExternalEffectRollbackStatus.NOT_REQUIRED
                )
            ),
            classifier_failure_label=f"post_{name.replace('.', '_')}_failure",
            internal_reason=(
                "explicit local Host MCP subscription lifecycle operation through "
                "the same registry, transport, data-flow, effect, and evidence boundary"
            ),
            require_classifier=False,
        )
        for name in (
            "subscriptions.start",
            "subscriptions.status",
            "subscriptions.events",
            "subscriptions.stop",
        )
    ),
    operation(
        "primitive.mcp.auth.begin.internal",
        "mcp",
        "auth.begin",
        resource_policy=ResourcePolicy.OPTIONAL,
        authority_mode=AuthorityMode.RUNTIME_INTERNAL,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        internal_reason=(
            "OAuth browser authorization is an explicit Host-only MCP operation"
        ),
        require_classifier=False,
    ),
    operation(
        "primitive.mcp.auth.challenge.internal",
        "mcp",
        "auth.challenge",
        resource_policy=ResourcePolicy.OPTIONAL,
        authority_mode=AuthorityMode.RUNTIME_INTERNAL,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        internal_reason=(
            "OAuth scope step-up is an explicit Host-only MCP operation"
        ),
        require_classifier=False,
    ),
    operation(
        "primitive.mcp.auth.complete.internal",
        "mcp",
        "auth.complete",
        resource_policy=ResourcePolicy.OPTIONAL,
        authority_mode=AuthorityMode.RUNTIME_INTERNAL,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
        classifier_failure_label="post_oauth_token_exchange_failure",
        internal_reason=(
            "OAuth code exchange is explicit Host-only and never automatically replayed"
        ),
        require_classifier=False,
    ),
    operation(
        "primitive.mcp.auth.revoke.internal",
        "mcp",
        "auth.revoke",
        resource_policy=ResourcePolicy.OPTIONAL,
        authority_mode=AuthorityMode.RUNTIME_INTERNAL,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        classifier_failure_rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
        classifier_failure_rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
        classifier_failure_label="post_oauth_revocation_failure",
        internal_reason=(
            "OAuth revocation is explicit Host-only and never automatically replayed"
        ),
        require_classifier=False,
    ),
)


__all__ = ["PROTECTED_OPERATION_DESCRIPTORS"]
