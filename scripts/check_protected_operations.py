from __future__ import annotations

import ast
from pathlib import Path
import sys


FORBIDDEN_EFFECT_LIFECYCLE = frozenset(
    {
        "abandon_external_effect_intent",
        "begin_external_effect_intent",
        "mark_external_effect_dispatched",
        "prepare_external_effect_intent",
        "record_external_effect",
    }
)
ALLOWED_LIFECYCLE_FILES = frozenset(
    {
        Path("agent_libos/evidence/external_effects.py"),
        Path("agent_libos/evidence/__init__.py"),
        Path("agent_libos/sdk/protected_operations.py"),
        Path("agent_libos/storage/repositories.py"),
    }
)
SAFE_PROVIDER_CALLS = frozenset(
    {
        (Path("agent_libos/primitives/filesystem.py"), "resolve"),
        # This Git call is a non-dispatching, local security preflight.  The
        # primitive performs the Task Authority and remote capability gates
        # before invoking it, then re-runs the ordinary fingerprint method in
        # the protected provider phase immediately before network dispatch.
        (Path("agent_libos/primitives/git.py"), "preflight_remote_fingerprint"),
        # Path type selects exact-file versus subtree filesystem authority.
        # This narrow, non-dispatching observation follows repository authority
        # and is repeated in the protected phase before a Git mutation runs.
        (Path("agent_libos/primitives/git.py"), "preflight_path_kind"),
        # Repository locking is a payload-free local synchronization fence.
        # Git acquires it only after protected-operation preparation and holds
        # it across final dispatch revalidation, provider I/O, and completion.
        (Path("agent_libos/primitives/git.py"), "repository_lock"),
        # This payload-free Git safety preflight validates repository identity,
        # configuration, and active drivers before a read intent is created;
        # the protected phase repeats the full state observation before use.
        (Path("agent_libos/primitives/git.py"), "validate_read_only_operation"),
        (Path("agent_libos/human/delivery.py"), "read"),
        (Path("agent_libos/human/delivery.py"), "write"),
    }
)
PROVIDER_HANDLE_METHODS = frozenset(
    {"close", "exit_code", "is_alive", "read", "resize", "write"}
)
# Recovery diagnostics handoff has one deliberately evidence-free provider
# action: closing an already-published transient handle while durable state is
# commit-fenced for same-process reopen.  The runtime guard is opaque and
# ContextVar-scoped to an explicitly registered recovery-safe callback.  The
# checker recognizes only this exact leading guard and only handle.close(); it
# is not a general provider-operation allowlist.
RECOVERY_CLEANUP_GUARD_PATH = (
    "self",
    "host",
    "require_recovery_cleanup_lease",
)
RECOVERY_CLEANUP_PROVIDER_CALLS = frozenset(
    {("provider handle method", "close")}
)
EGRESS_CONTRACTS = frozenset(
    {
        "primitive.filesystem.write_text",
        "primitive.filesystem.write_directory",
        "primitive.filesystem.delete_file",
        "primitive.filesystem.delete_directory",
        "primitive.shell.run",
        "primitive.git.mutate",
        "primitive.git.fetch",
        "primitive.git.push",
        "primitive.git.pull_request",
        "primitive.jsonrpc.call",
        "primitive.mcp.list_tools",
        "primitive.mcp.list_tools.internal",
        "primitive.mcp.call",
        "primitive.llm.complete",
        "primitive.human.read",
        "primitive.human.write",
        "primitive.pty.spawn",
        "primitive.pty.write",
        "primitive.pty.resize",
        "primitive.pty.close",
    }
)
INGRESS_CONTRACTS = frozenset(
    {
        "primitive.filesystem.read_text",
        "primitive.filesystem.read_bytes",
        "primitive.filesystem.read_directory",
        "primitive.shell.run",
        "primitive.git.read",
        "primitive.git.mutate",
        "primitive.git.fetch",
        "primitive.git.pull_request",
        "primitive.jsonrpc.call",
        "primitive.mcp.list_tools",
        "primitive.mcp.list_tools.internal",
        "primitive.mcp.call",
        "primitive.llm.complete",
        "primitive.human.read",
        "primitive.pty.spawn",
        "primitive.pty.read",
        "primitive.pty.ingest",
    }
)
DATA_FLOW_DESCRIPTOR_FIELDS = frozenset(
    {"data_sink", "data_flow_context", "data_flow_payload", "data_flow_operation"}
)
INGRESS_DESCRIPTOR_FIELD = "data_flow_ingress_context"

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _is_protected_phase_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in {"call", "acall"} or len(node.args) < 2:
        return False
    phase = node.args[0]
    return (
        isinstance(phase, ast.Call)
        and _attribute_path(phase.func)[-1:] == ("ProviderPhase",)
    )


def _nearest_owner(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> FunctionNode | None:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return current
        current = parents.get(current)
    return None


def _nearest_class(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.ClassDef | None:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, ast.ClassDef):
            return current
        current = parents.get(current)
    return None


def _function_name(function: FunctionNode) -> str:
    if isinstance(function, ast.Lambda):
        return f"lambda@{function.lineno}"
    return function.name


def _has_leading_recovery_cleanup_guard(function: FunctionNode) -> bool:
    if isinstance(function, ast.Lambda):
        return False
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body or not isinstance(body[0], ast.Expr):
        return False
    call = body[0].value
    return (
        isinstance(call, ast.Call)
        and not call.args
        and not call.keywords
        and _attribute_path(call.func) == RECOVERY_CLEANUP_GUARD_PATH
    )


class _CallGraph:
    def __init__(self, tree: ast.AST, parents: dict[ast.AST, ast.AST]) -> None:
        self.parents = parents
        self.functions = tuple(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        )
        self.methods: dict[tuple[ast.ClassDef, str], FunctionNode] = {}
        self.locals: dict[tuple[FunctionNode | None, str], FunctionNode] = {}
        for function in self.functions:
            if isinstance(function, ast.Lambda):
                continue
            containing_function = _nearest_owner(parents.get(function, tree), parents)
            containing_class = _nearest_class(function, parents)
            if containing_class is not None and containing_function is None:
                self.methods[(containing_class, function.name)] = function
            else:
                self.locals[(containing_function, function.name)] = function
        self.calls: dict[FunctionNode, set[FunctionNode]] = {
            function: set() for function in self.functions
        }
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            owner = _nearest_owner(call, parents)
            callee = self.resolve(call.func, owner)
            if owner is not None and callee is not None:
                self.calls[owner].add(callee)

    def resolve(
        self,
        callable_node: ast.AST,
        owner: FunctionNode | None,
    ) -> FunctionNode | None:
        path = _attribute_path(callable_node)
        if len(path) == 2 and path[0] in {"self", "cls"} and owner is not None:
            containing_class = _nearest_class(owner, self.parents)
            if containing_class is not None:
                return self.methods.get((containing_class, path[1]))
        if isinstance(callable_node, ast.Name):
            container = owner
            while True:
                selected = self.locals.get((container, callable_node.id))
                if selected is not None:
                    return selected
                if container is None:
                    break
                container = _nearest_owner(
                    self.parents.get(container, ast.Module(body=[], type_ignores=[])),
                    self.parents,
                )
        return None

    def protected_functions(self, tree: ast.AST) -> set[FunctionNode]:
        protected: set[FunctionNode] = set()
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not _is_protected_phase_call(call):
                continue
            callable_node = call.args[1]
            if isinstance(callable_node, ast.Lambda):
                protected.add(callable_node)
                continue
            selected = self.resolve(callable_node, _nearest_owner(call, self.parents))
            if selected is not None:
                protected.add(selected)
        pending = list(protected)
        while pending:
            function = pending.pop()
            for callee in self.calls.get(function, ()):
                if callee not in protected:
                    protected.add(callee)
                    pending.append(callee)
        # A primitive may centralize the SDK lifecycle in a gateway such as
        # ``_read(callback)`` or ``_mutate(callback)``.  Treat a call-site
        # callback as protected only when the gateway invokes that exact
        # formal parameter exclusively from a function already passed to
        # ProtectedOperation.call/acall.  This keeps the scan useful without
        # requiring every typed operation to duplicate the lifecycle body.
        changed = True
        calls = tuple(node for node in ast.walk(tree) if isinstance(node, ast.Call))
        while changed:
            changed = False
            gateways = self._protected_callback_parameters(protected)
            for call in calls:
                owner = _nearest_owner(call, self.parents)
                callee = self.resolve(call.func, owner)
                if callee is None:
                    continue
                for parameter in gateways.get(callee, ()):
                    callback = self._bound_callback(call, callee, parameter)
                    if callback is None:
                        continue
                    selected = (
                        callback
                        if isinstance(callback, ast.Lambda)
                        else self.resolve(callback, owner)
                    )
                    if selected is not None and selected not in protected:
                        protected.add(selected)
                        pending.append(selected)
                        changed = True
            while pending:
                function = pending.pop()
                for callee in self.calls.get(function, ()):
                    if callee not in protected:
                        protected.add(callee)
                        pending.append(callee)
                        changed = True
        return protected

    @staticmethod
    def _parameters(function: FunctionNode) -> tuple[str, ...]:
        if isinstance(function, ast.Lambda):
            arguments = function.args
        else:
            arguments = function.args
        return tuple(
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        )

    def _protected_callback_parameters(
        self,
        protected: set[FunctionNode],
    ) -> dict[FunctionNode, set[str]]:
        candidates: dict[FunctionNode, set[str]] = {}
        for call in (
            node
            for function in protected
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and _nearest_owner(node, self.parents) is function
            and isinstance(node.func, ast.Name)
        ):
            callback_name = call.func.id
            container = _nearest_owner(call, self.parents)
            while container is not None:
                if callback_name in self._parameters(container):
                    candidates.setdefault(container, set()).add(callback_name)
                    break
                container = _nearest_owner(
                    self.parents.get(container, container),
                    self.parents,
                )

        safe: dict[FunctionNode, set[str]] = {}
        for gateway, parameters in candidates.items():
            for parameter in parameters:
                invocations = [
                    node
                    for node in ast.walk(gateway)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == parameter
                ]
                reassigned = any(
                    isinstance(node, ast.Name)
                    and node.id == parameter
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                    for node in ast.walk(gateway)
                )
                if (
                    invocations
                    and not reassigned
                    and all(
                        _nearest_owner(invocation, self.parents) in protected
                        for invocation in invocations
                    )
                ):
                    safe.setdefault(gateway, set()).add(parameter)
        return safe

    def _bound_callback(
        self,
        call: ast.Call,
        callee: FunctionNode,
        parameter: str,
    ) -> ast.AST | None:
        for keyword in call.keywords:
            if keyword.arg == parameter:
                return keyword.value
        parameters = self._parameters(callee)
        try:
            index = parameters.index(parameter)
        except ValueError:
            return None
        path = _attribute_path(call.func)
        if parameters and parameters[0] in {"self", "cls"} and path[:1] in {
            ("self",),
            ("cls",),
        }:
            index -= 1
        if index < 0 or index >= len(call.args):
            return None
        return call.args[index]

    def provider_reaching_functions(
        self,
        direct: set[FunctionNode],
    ) -> set[FunctionNode]:
        reaching = set(direct)
        changed = True
        while changed:
            changed = False
            for function, callees in self.calls.items():
                if function not in reaching and any(callee in reaching for callee in callees):
                    reaching.add(function)
                    changed = True
        return reaching


def _provider_call_kind(
    node: ast.Call,
    *,
    owner: FunctionNode | None,
    provider_handle_names: dict[FunctionNode, set[str]],
) -> tuple[str, str] | None:
    path = _attribute_path(node.func)
    if len(path) == 3 and path[:2] == ("self", "provider"):
        return "provider method", path[2]
    if (
        len(path) >= 3
        and path[-2] == "handle"
        and path[-1] in PROVIDER_HANDLE_METHODS
    ):
        return "provider handle method", path[-1]
    if (
        owner is not None
        and len(path) == 2
        and path[0] in provider_handle_names.get(owner, set())
        and path[1] in PROVIDER_HANDLE_METHODS
    ):
        return "provider handle method", path[1]
    return None


def _provider_handle_names(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> dict[FunctionNode, set[str]]:
    names: dict[FunctionNode, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not _is_protected_phase_call(value):
            continue
        callable_path = _attribute_path(value.args[1])
        if len(callable_path) != 3 or callable_path[:2] != ("self", "provider"):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        owner = _nearest_owner(node, parents)
        if owner is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                names.setdefault(owner, set()).add(target.id)
    return names


def _invocation_descriptor_error(call: ast.Call) -> str | None:
    fields = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
    missing = sorted(DATA_FLOW_DESCRIPTOR_FIELDS - fields)
    if missing:
        return "missing data-flow descriptor fields: " + ", ".join(missing)
    return None


def _invocation_ingress_descriptor_error(call: ast.Call) -> str | None:
    fields = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
    if INGRESS_DESCRIPTOR_FIELD not in fields:
        return f"missing ingress data-flow descriptor field: {INGRESS_DESCRIPTOR_FIELD}"
    return None


def _literal_contract_names(node: ast.AST) -> frozenset[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return frozenset({node.value})
    if isinstance(node, ast.IfExp):
        return _literal_contract_names(node.body) | _literal_contract_names(node.orelse)
    return frozenset()


def _assigned_contract_names(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> dict[tuple[FunctionNode | None, str], frozenset[str]]:
    assigned: dict[tuple[FunctionNode | None, str], frozenset[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        names = _literal_contract_names(node.value)
        if not names:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        owner = _nearest_owner(node, parents)
        for target in targets:
            if isinstance(target, ast.Name):
                assigned[(owner, target.id)] = names
    return assigned


def _assigned_invocations(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> dict[tuple[FunctionNode | None, str], ast.Call]:
    assigned: dict[tuple[FunctionNode | None, str], ast.Call] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or _attribute_path(value.func)[-1:] != (
            "ProtectedOperationInvocation",
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        owner = _nearest_owner(node, parents)
        for target in targets:
            if isinstance(target, ast.Name):
                assigned[(owner, target.id)] = value
    return assigned


def _returned_invocation(
    function: FunctionNode,
    parents: dict[ast.AST, ast.AST],
) -> ast.Call | None:
    candidates = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and _nearest_owner(node, parents) is function
        and isinstance(node.value, ast.Call)
        and _attribute_path(node.value.func)[-1:] == (
            "ProtectedOperationInvocation",
        )
    ]
    return candidates[0] if len(candidates) == 1 else None


def _factory_invocation(
    call: ast.Call,
    owner: FunctionNode | None,
    call_graph: _CallGraph,
) -> ast.Call | None:
    factory = call_graph.resolve(call.func, owner)
    if factory is None:
        return None
    return _returned_invocation(factory, call_graph.parents)


def _assigned_factory_invocations(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
    call_graph: _CallGraph,
) -> dict[tuple[FunctionNode | None, str], ast.Call]:
    assigned: dict[tuple[FunctionNode | None, str], ast.Call] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        owner = _nearest_owner(node, parents)
        invocation = _factory_invocation(value, owner, call_graph)
        if invocation is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assigned[(owner, target.id)] = invocation
    return assigned


def scan_source(path: Path, *, relative: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
    except (OSError, SyntaxError) as error:
        return [f"{relative}: cannot inspect protected-operation coverage: {error}"]
    errors: list[str] = []
    lifecycle_allowed = relative in ALLOWED_LIFECYCLE_FILES
    parents: dict[ast.AST, ast.AST] = {
        child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
    }
    call_graph = _CallGraph(tree, parents)
    assigned_invocations = _assigned_invocations(tree, parents)
    assigned_factory_invocations = _assigned_factory_invocations(
        tree,
        parents,
        call_graph,
    )
    assigned_contract_names = _assigned_contract_names(tree, parents)
    protected_functions = call_graph.protected_functions(tree)
    recovery_cleanup_functions = {
        function
        for function in call_graph.functions
        if _has_leading_recovery_cleanup_guard(function)
    }
    provider_handle_names = _provider_handle_names(tree, parents)
    direct_provider_functions: set[FunctionNode] = set()
    provider_calls: list[tuple[ast.Call, FunctionNode | None, str, str]] = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        owner = _nearest_owner(call, parents)
        provider_call = _provider_call_kind(
            call,
            owner=owner,
            provider_handle_names=provider_handle_names,
        )
        if provider_call is None:
            continue
        kind, method = provider_call
        if kind == "provider method" and (relative, method) in SAFE_PROVIDER_CALLS:
            continue
        provider_calls.append((call, owner, kind, method))
        if owner is not None and owner not in recovery_cleanup_functions:
            direct_provider_functions.add(owner)
    provider_reaching = call_graph.provider_reaching_functions(direct_provider_functions)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "agent_libos.evidence.external_effects",
            "agent_libos.runtime.external_effects",
        }:
            for alias in node.names:
                if alias.name in FORBIDDEN_EFFECT_LIFECYCLE and not lifecycle_allowed:
                    errors.append(
                        f"{relative}:{node.lineno}: direct import of {alias.name} bypasses agent_libos.sdk"
                    )
        if isinstance(node, ast.Call):
            name: str | None = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name == "_restore_reserved_use":
                    errors.append(
                        f"{relative}:{node.lineno}: use public restore_reserved_use or ProtectedOperationSDK"
                    )
            if name in FORBIDDEN_EFFECT_LIFECYCLE and not lifecycle_allowed:
                errors.append(
                    f"{relative}:{node.lineno}: direct {name} call bypasses agent_libos.sdk"
                )
            if _attribute_path(node.func)[-1:] == ("ProtectedOperationInvocation",):
                fields = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
                if fields & DATA_FLOW_DESCRIPTOR_FIELDS:
                    descriptor_error = _invocation_descriptor_error(node)
                    if descriptor_error is not None:
                        errors.append(
                            f"{relative}:{node.lineno}: ProtectedOperationInvocation {descriptor_error}"
                        )
            path = _attribute_path(node.func)
            if path[-1:] == ("start",) and node.args:
                contract = node.args[0]
                contract_names = _literal_contract_names(contract)
                if isinstance(contract, ast.Name):
                    contract_names = assigned_contract_names.get(
                        (_nearest_owner(node, parents), contract.id),
                        frozenset(),
                    )
                needs_egress = bool(contract_names & EGRESS_CONTRACTS)
                needs_ingress = bool(contract_names & INGRESS_CONTRACTS)
                if needs_egress or needs_ingress:
                    invocation_node: ast.Call | None = None
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Call):
                        candidate = node.args[1]
                        if _attribute_path(candidate.func)[-1:] == (
                            "ProtectedOperationInvocation",
                        ):
                            invocation_node = candidate
                        else:
                            invocation_node = _factory_invocation(
                                candidate,
                                _nearest_owner(node, parents),
                                call_graph,
                            )
                    elif len(node.args) >= 2 and isinstance(node.args[1], ast.Name):
                        key = (_nearest_owner(node, parents), node.args[1].id)
                        invocation_node = assigned_invocations.get(key)
                        if invocation_node is None:
                            invocation_node = assigned_factory_invocations.get(key)
                    if invocation_node is None:
                        errors.append(
                            f"{relative}:{node.lineno}: data-flow contract "
                            f"{sorted(contract_names)} "
                            "does not resolve to a local ProtectedOperationInvocation"
                        )
                    else:
                        if needs_egress:
                            descriptor_error = _invocation_descriptor_error(invocation_node)
                            if descriptor_error is not None:
                                errors.append(
                                    f"{relative}:{node.lineno}: egress contract "
                                    f"{sorted(contract_names & EGRESS_CONTRACTS)} "
                                    f"{descriptor_error}"
                                )
                        if needs_ingress:
                            ingress_error = _invocation_ingress_descriptor_error(
                                invocation_node
                            )
                            if ingress_error is not None:
                                errors.append(
                                    f"{relative}:{node.lineno}: ingress contract "
                                    f"{sorted(contract_names & INGRESS_CONTRACTS)} "
                                    f"{ingress_error}"
                                )
            owner = _nearest_owner(node, parents)
            callee = call_graph.resolve(node.func, owner)
            if (
                callee is not None
                and callee in provider_reaching
                and owner not in protected_functions
            ):
                errors.append(
                    f"{relative}:{node.lineno}: provider helper {_function_name(callee)} is called "
                    "outside an active ProtectedOperation phase"
                )
    for node, owner, kind, method in provider_calls:
        if owner in recovery_cleanup_functions:
            if (kind, method) not in RECOVERY_CLEANUP_PROVIDER_CALLS:
                errors.append(
                    f"{relative}:{node.lineno}: recovery cleanup lease permits "
                    "only provider handle close"
                )
            continue
        if owner not in protected_functions:
            errors.append(
                f"{relative}:{node.lineno}: {kind} {method} is called "
                "outside an active ProtectedOperation phase"
            )
    return errors


def check_tree(root: Path) -> list[str]:
    errors: list[str] = []
    for source_root in (root / "agent_libos", root / "modules"):
        for path in sorted(source_root.rglob("*.py")):
            errors.extend(scan_source(path, relative=path.relative_to(root)))
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = check_tree(root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Protected provider operation coverage is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
