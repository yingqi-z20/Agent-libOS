"""AST architecture ratchet.

The checked-in JSON stores ceilings for known debt, not exemptions from the
rules.  Component-coupling ceilings are grouped by class and dependency so
method extraction does not churn the file; long-function and complexity
ceilings stay attached to their qualified function names.  Removing an entry
or lowering a number makes an improvement permanent.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
MAX_FUNCTION_LINES = 200
COMPLEXITY_HOTSPOT_THRESHOLD = 20
SOURCE_PACKAGE = "agent_libos"
RUNTIME_MODULE_ROOT = "modules"
SOURCE_ROOTS = (SOURCE_PACKAGE, RUNTIME_MODULE_ROOT)
RATCHET_RULES = (
    "composition-late-binding",
    "concrete-api-import",
    "concrete-runtime-import",
    "cross-component-private-access",
    "runtime-service-locator",
    "runtime-storage-sql-bypass",
    "typed-storage-reflection-bypass",
    "untracked-blocking-dispatch",
    "undeclared-runtime-component",
    "whole-process-write",
)

_TYPED_STORAGE_RUNTIME_FILES = frozenset(
    {
        "agent_libos/runtime/process_manager.py",
        "agent_libos/runtime/resource_manager.py",
        "agent_libos/runtime/checkpoint_manager.py",
        "agent_libos/runtime/checkpoint_reconciliation.py",
        "agent_libos/runtime/checkpoint_image.py",
        "agent_libos/runtime/image_boot.py",
        "agent_libos/runtime/data_flow_manager.py",
        "agent_libos/runtime/snapshots/exec_state.py",
        "agent_libos/modules/registry.py",
        "agent_libos/process_transition.py",
    }
)
_RAW_STORAGE_CALLS = frozenset(
    {
        "execute",
        "select_table_rows",
        "insert_table_row",
        "delete_table_rows",
        "snapshot_tables",
        "restore_tables",
    }
)
_RAW_SQL_CALLS = frozenset(
    {"execute", "executemany", "executescript", "_execute", "_execute_script"}
)
_CHECKPOINT_MANAGER_STORAGE_METHOD_OWNERS = {
    "transaction": frozenset({"_unit_of_work"}),
    "record_runtime_publication_artifact": frozenset({"_publications"}),
    "current_effect_ledger_seq": frozenset({"_evidence"}),
    "list_external_effects_changed_after": frozenset({"_evidence"}),
    "list_events": frozenset({"_evidence"}),
    "get_process": frozenset({"_processes"}),
    "patch_process": frozenset({"_processes"}),
    "list_processes": frozenset({"_processes"}),
    "list_object_tasks": frozenset({"_processes"}),
    "get_object_task": frozenset({"_processes"}),
    "get_human_request": frozenset({"_processes"}),
    "get_object": frozenset({"_objects"}),
    "list_objects_owned_by": frozenset({"_objects"}),
    "has_object_payload": frozenset({"_objects"}),
    "object_payload": frozenset({"_objects"}),
    "list_namespaces_created_by": frozenset({"_objects"}),
    "get_namespace": frozenset({"_objects"}),
    "namespace_exists": frozenset({"_objects"}),
    "get_capability": frozenset({"_authority"}),
    "get_image_artifact": frozenset({"_extensions"}),
    "list_tools": frozenset({"_extensions"}),
    "delete_tool": frozenset({"_extensions"}),
    "upsert_image": frozenset({"_extensions"}),
    "insert_image_artifact": frozenset({"_extensions"}),
}
_PROCESS_MANAGER_STORAGE_METHOD_OWNERS = {
    "get_operation": ("self", "_evidence"),
    "list_capabilities": ("self", "_authority"),
}
_TOOL_BROKER_STORAGE_METHOD_OWNERS = {
    "transaction": ("self", "unit_of_work"),
    "get_process": ("self", "processes"),
    "list_processes": ("self", "processes"),
    "patch_process": ("self", "processes"),
    "tool_id_referenced_outside_process": ("self", "processes"),
    "get_object": ("self", "objects"),
    "list_objects_owned_by": ("self", "objects"),
    "delete_tool_candidate": ("self", "extensions"),
    "get_existing_tool_ids": ("self", "extensions"),
    "get_tool_candidate": ("self", "extensions"),
    "get_tool_spec": ("self", "extensions"),
    "list_registered_tool_candidate_rows": ("self", "extensions"),
    "list_tools": ("self", "extensions"),
}
_JIT_PROJECTION_SQL_TRUSTED_PATHS = frozenset(
    {
        "agent_libos/storage/sql.py",
        "agent_libos/storage/migrations.py",
        # Explicit offline content migration. It rebuilds the derived process
        # binding projection in the same transaction as the authoritative
        # process/tool rows and is never imported by runtime startup.
        "agent_libos/storage/tool_skill_migration.py",
    }
)
_JIT_PROJECTION_SQL_TRUSTED_PREFIXES = (
    "agent_libos/storage/migrations/",
)
_JIT_PROJECTION_MUTATION = re.compile(
    r"\b(?:"
    r"(?:insert(?:\s+or\s+[a-z_]+)?|replace|merge)\s+into|"
    r"update(?:\s+or\s+[a-z_]+)?|delete\s+from|"
    r"truncate(?:\s+table)?"
    r")\s+[\"`\[]?(tools|process_tool_bindings)\b",
    re.IGNORECASE,
)
_TYPED_STORAGE_REPOSITORY_CLASSES = frozenset(
    {
        "ProcessRepository",
        "ResourceRepository",
        "RuntimePublicationRepository",
        "CheckpointRestorePublicationWriter",
        "SnapshotCheckpointRepository",
        "EvidenceRepository",
        "RuntimeModuleRepository",
        "PayloadRetentionRepository",
    }
)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    path: str
    owner: str
    detail: str
    line: int

    @property
    def budget_key(self) -> str:
        if self.rule in {
            "composition-late-binding",
            "cross-component-private-access",
            "runtime-service-locator",
        }:
            # Classes are the useful migration unit for component coupling.
            # Keeping the accessed member preserves dependency visibility while
            # allowing a method to be split or renamed without resetting the
            # ratchet.
            owner_namespace = self.owner.split(".", 1)[0]
            return "::".join((self.path, owner_namespace, self.detail))
        return "::".join((self.path, self.owner, self.detail))

    def describe(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.detail} (owner: {self.owner})"


@dataclass(frozen=True, slots=True)
class FunctionMetric:
    path: str
    owner: str
    line: int
    lines: int
    complexity: int

    @property
    def budget_key(self) -> str:
        return "::".join((self.path, self.owner))


@dataclass(frozen=True, slots=True)
class ArchitectureReport:
    violations: tuple[Violation, ...]
    functions: tuple[FunctionMetric, ...]


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _static_getattr_path(node: ast.AST) -> tuple[str, ...]:
    """Resolve ``getattr(base, "literal")`` as a normal attribute path."""

    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "getattr"
        or len(node.args) < 2
        or not isinstance(node.args[1], ast.Constant)
        or not isinstance(node.args[1].value, str)
    ):
        return ()
    base = _attribute_path(node.args[0]) or _static_getattr_path(node.args[0])
    return (*base, node.args[1].value) if base else ()


def _is_default_executor_dispatch(node: ast.Call) -> bool:
    """Recognize raw ``run_in_executor(None, ...)`` dispatch."""

    if not isinstance(node.func, ast.Attribute) or node.func.attr != "run_in_executor":
        return False
    if node.args:
        executor = node.args[0]
    else:
        executor = next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "executor"
            ),
            None,
        )
    return isinstance(executor, ast.Constant) and executor.value is None


def _nearest_function_scope(
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
) -> FunctionNode | None:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _function_scope_chain(
    scope: FunctionNode | None,
    parents: Mapping[ast.AST, ast.AST],
) -> tuple[FunctionNode | None, ...]:
    chain: list[FunctionNode | None] = []
    current = scope
    while current is not None:
        chain.append(current)
        parent = parents.get(current)
        current = (
            _nearest_function_scope(parent, parents)
            if parent is not None
            else None
        )
    chain.append(None)
    return tuple(chain)


def _static_name_environment(
    tree: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
) -> tuple[
    dict[tuple[FunctionNode | None, str], ast.AST | None],
    set[tuple[FunctionNode | None, str]],
]:
    values: dict[tuple[FunctionNode | None, str], list[ast.AST]] = {}
    definitions: set[tuple[FunctionNode | None, str]] = set()
    for node in ast.walk(tree):
        scope = _nearest_function_scope(node, parents)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ):
                definitions.add((node, argument.arg))
            if node.args.vararg is not None:
                definitions.add((node, node.args.vararg.arg))
            if node.args.kwarg is not None:
                definitions.add((node, node.args.kwarg.arg))
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    key = (scope, target.id)
                    definitions.add(key)
                    values.setdefault(key, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            key = (scope, node.target.id)
            definitions.add(key)
            if node.value is not None:
                values.setdefault(key, []).append(node.value)
        elif isinstance(node, ast.arg):
            definitions.add((scope, node.arg))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    definitions.add(
                        (scope, alias.asname or alias.name.split(".", 1)[0])
                    )
    bindings = {
        key: candidates[0] if len(candidates) == 1 else None
        for key, candidates in values.items()
    }
    return bindings, definitions


def _static_sql_text(
    node: ast.AST,
    *,
    scope: FunctionNode | None = None,
    bindings: Mapping[tuple[FunctionNode | None, str], ast.AST | None] | None = None,
    definitions: set[tuple[FunctionNode | None, str]] | None = None,
    parents: Mapping[ast.AST, ast.AST] | None = None,
    seen: frozenset[tuple[FunctionNode | None, str]] = frozenset(),
) -> str | None:
    """Return statically visible SQL text without evaluating source code."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
            else " ? "
            for value in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_sql_text(
            node.left,
            scope=scope,
            bindings=bindings,
            definitions=definitions,
            parents=parents,
            seen=seen,
        )
        right = _static_sql_text(
            node.right,
            scope=scope,
            bindings=bindings,
            definitions=definitions,
            parents=parents,
            seen=seen,
        )
        if left is not None and right is not None:
            return left + right
    if (
        isinstance(node, ast.Name)
        and bindings is not None
        and definitions is not None
        and parents is not None
    ):
        for candidate_scope in _function_scope_chain(scope, parents):
            key = (candidate_scope, node.id)
            if key in seen:
                return None
            if key in bindings:
                value = bindings[key]
                if value is None:
                    return None
                return _static_sql_text(
                    value,
                    scope=candidate_scope,
                    bindings=bindings,
                    definitions=definitions,
                    parents=parents,
                    seen=seen | {key},
                )
            if key in definitions:
                return None
    return None


def _raw_jit_projection_mutation(
    node: ast.Call,
    *,
    relative: str,
    scope: FunctionNode | None,
    bindings: Mapping[tuple[FunctionNode | None, str], ast.AST | None],
    definitions: set[tuple[FunctionNode | None, str]],
    parents: Mapping[ast.AST, ast.AST],
) -> str | None:
    """Identify raw writes to the two transactionally derived JIT tables.

    Tests are outside ``SOURCE_ROOTS``. Schema migrations and the owning SQL
    backend are the only production-source trust boundary for these writes;
    all other runtime code must use typed repository mutations so the durable
    eligibility bit cannot diverge from Tool metadata or process bindings.
    """

    if relative in _JIT_PROJECTION_SQL_TRUSTED_PATHS or relative.startswith(
        _JIT_PROJECTION_SQL_TRUSTED_PREFIXES
    ):
        return None
    call_path = _attribute_path(node.func) or _static_getattr_path(node.func)
    if not call_path or call_path[-1] not in _RAW_SQL_CALLS:
        return None
    sql_node = node.args[0] if node.args else next(
        (
            keyword.value
            for keyword in node.keywords
            if keyword.arg in {"sql", "query", "statement"}
        ),
        None,
    )
    if sql_node is None:
        return None
    sql = _static_sql_text(
        sql_node,
        scope=scope,
        bindings=bindings,
        definitions=definitions,
        parents=parents,
    )
    if sql is None:
        return None
    match = _JIT_PROJECTION_MUTATION.search(sql)
    return match.group(1).lower() if match is not None else None


def _source_paths(root: Path) -> Iterable[Path]:
    return tuple(
        sorted(
            path
            for source_root in SOURCE_ROOTS
            for path in (root / source_root).rglob("*.py")
            if (root / source_root).is_dir() and "__pycache__" not in path.parts
        )
    )


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


class _QualifiedOwnerVisitor(ast.NodeVisitor):
    def __init__(self, tree: ast.AST) -> None:
        self.owners: dict[ast.AST, str] = {tree: "<module>"}
        self._prefix: list[str] = []

    def _visit_owner(self, node: ast.ClassDef | FunctionNode) -> None:
        self._prefix.append(node.name)
        self.owners[node] = ".".join(self._prefix)
        self.generic_visit(node)
        self._prefix.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._visit_owner(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_owner(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_owner(node)


def _qualified_owners(tree: ast.AST) -> dict[ast.AST, str]:
    visitor = _QualifiedOwnerVisitor(tree)
    visitor.visit(tree)
    return visitor.owners


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _owner_for(
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
    owners: Mapping[ast.AST, str],
) -> str:
    current: ast.AST | None = node
    while current is not None:
        selected = owners.get(current)
        if selected is not None:
            return selected
        current = parents.get(current)
    return "<module>"


def _is_outermost_attribute(
    node: ast.Attribute,
    parents: Mapping[ast.AST, ast.AST],
) -> bool:
    parent = parents.get(node)
    return not (isinstance(parent, ast.Attribute) and parent.value is node)


def _is_direct_call_target(
    node: ast.Attribute,
    parents: Mapping[ast.AST, ast.AST],
) -> bool:
    parent = parents.get(node)
    return isinstance(parent, ast.Call) and parent.func is node


def _assignment_names(node: ast.AST) -> tuple[str, ...]:
    targets: tuple[ast.AST, ...]
    if isinstance(node, ast.Assign):
        targets = tuple(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
        targets = (node.target,)
    else:
        return ()
    return tuple(
        target.id
        for target in targets
        if isinstance(target, ast.Name)
    )


def _runtime_aliases(
    tree: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
    owners: Mapping[ast.AST, str],
) -> dict[str, frozenset[str]]:
    assignments: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    aliases: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            continue
        value = node.value
        value_path = _attribute_path(value) or _static_getattr_path(value)
        owner = _owner_for(node, parents, owners)
        assignments.append((owner, _assignment_names(node), value_path))
    changed = True
    while changed:
        changed = False
        for owner, names, value_path in assignments:
            owner_aliases = aliases.setdefault(owner, set())
            is_runtime = value_path in {
                ("self", "runtime"),
                ("self", "_runtime"),
            } or (len(value_path) == 1 and value_path[0] in owner_aliases)
            if is_runtime:
                before = len(owner_aliases)
                owner_aliases.update(names)
                changed = changed or len(owner_aliases) != before
    return {owner: frozenset(names) for owner, names in aliases.items()}


def _component_aliases(
    tree: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
    owners: Mapping[ast.AST, str],
) -> dict[str, dict[str, str]]:
    assignments: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    aliases: dict[str, dict[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            continue
        value_path = _attribute_path(node.value) or _static_getattr_path(node.value)
        owner = _owner_for(node, parents, owners)
        assignments.append((owner, _assignment_names(node), value_path))
    changed = True
    while changed:
        changed = False
        for owner, names, value_path in assignments:
            owner_aliases = aliases.setdefault(owner, {})
            dependency = (
                value_path[1]
                if len(value_path) == 2 and value_path[0] == "self"
                else owner_aliases.get(value_path[0])
                if len(value_path) == 1
                else None
            )
            if dependency is None:
                continue
            for name in names:
                if name not in owner_aliases:
                    owner_aliases[name] = dependency
                    changed = True
    return aliases


def _cross_component_private_detail(
    access_path: tuple[str, ...],
    aliases: Mapping[str, str],
) -> str | None:
    if len(access_path) >= 3 and access_path[0] == "self":
        dependency = access_path[1]
        tail = access_path[2:]
    elif len(access_path) >= 2 and access_path[0] in aliases:
        dependency = aliases[access_path[0]]
        tail = access_path[1:]
    else:
        return None
    for index, part in enumerate(tail):
        if part.startswith("_") and not part.startswith("__"):
            return ".".join((dependency, *tail[: index + 1]))
    return None


def _runtime_service_detail(
    access_path: tuple[str, ...],
    runtime_aliases: frozenset[str],
    *,
    direct_call: bool,
) -> str | None:
    direct_runtime = access_path[:2] in {
        ("self", "runtime"),
        ("self", "_runtime"),
    }
    aliased_runtime = bool(access_path) and access_path[0] in runtime_aliases
    if not direct_runtime and not aliased_runtime:
        return None
    service_index = 2 if direct_runtime else 1
    if len(access_path) > service_index + 1:
        return access_path[service_index]
    if len(access_path) == service_index + 1 and not direct_call:
        return access_path[service_index]
    return None


def _resolved_import_from_module(node: ast.ImportFrom, relative: str) -> str:
    module = node.module or ""
    if node.level == 0:
        return module
    source_parts = list(Path(relative).with_suffix("").parts)
    source_parts.pop()
    parent_count = node.level - 1
    if parent_count > len(source_parts):
        return module
    base = source_parts[: len(source_parts) - parent_count]
    return ".".join((*base, *module.split("."))) if module else ".".join(base)


def _concrete_import_details(
    node: ast.AST,
    relative: str,
    package: str,
) -> tuple[str, ...]:
    target = f"agent_libos.{package}"
    if isinstance(node, ast.ImportFrom):
        module = _resolved_import_from_module(node, relative)
        if module == target or module.startswith(f"{target}."):
            return tuple(f"{module}:{alias.name}" for alias in node.names)
        if module == "agent_libos":
            return tuple(
                target
                for alias in node.names
                if alias.name == package
            )
        return ()
    if isinstance(node, ast.Import):
        return tuple(
            alias.name
            for alias in node.names
            if alias.name == target or alias.name.startswith(f"{target}.")
        )
    return ()


class _ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.score = 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return None

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        self.score += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:  # noqa: N802
        self.score += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self.score += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
        self.score += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self.score += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:  # noqa: N802
        self.score += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        self.score += len(node.handlers) + bool(node.orelse)
        self.generic_visit(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:  # noqa: N802
        self.score += len(node.handlers) + bool(node.orelse)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:  # noqa: N802
        self.score += len(node.cases)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.score += 1 + len(node.ifs)
        self.generic_visit(node)


def _function_complexity(node: FunctionNode) -> int:
    visitor = _ComplexityVisitor()
    for statement in node.body:
        visitor.visit(statement)
    return visitor.score


def _function_start(node: FunctionNode) -> int:
    decorator_lines = [decorator.lineno for decorator in node.decorator_list]
    return min([node.lineno, *decorator_lines])


def _scan_file(path: Path, root: Path) -> ArchitectureReport:
    relative = _relative_path(path, root)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents = _parents(tree)
    owners = _qualified_owners(tree)
    runtime_aliases = _runtime_aliases(tree, parents, owners)
    component_aliases = _component_aliases(tree, parents, owners)
    static_name_bindings, static_name_definitions = _static_name_environment(
        tree,
        parents,
    )
    violations: list[Violation] = []
    functions: list[FunctionMetric] = []

    package_parts = Path(relative).parts
    package_source = (
        len(package_parts) >= 2
        and package_parts[0] == SOURCE_PACKAGE
        and package_parts[1] not in {"__init__.py", "__main__.py"}
    )
    runtime_module_source = bool(
        package_parts and package_parts[0] == RUNTIME_MODULE_ROOT
    )
    package_name = Path(package_parts[1]).stem if package_source else None
    api_import_forbidden = runtime_module_source or (
        package_source and package_name != "api"
    )
    runtime_import_forbidden = runtime_module_source or (
        package_source and package_name not in {"api", "runtime"}
    )

    if relative == "agent_libos/runtime/runtime.py":
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _owner_for(function, parents, owners) != "Runtime._add_handle_to_process_view":
                continue
            call_paths = [
                _attribute_path(candidate.func)
                for candidate in ast.walk(function)
                if isinstance(candidate, ast.Call)
            ]
            if call_paths != [("self", "process", "add_handle_to_process_view")]:
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner="Runtime._add_handle_to_process_view",
                        detail="handle-publication:must-delegate-to-process-manager",
                        line=function.lineno,
                    )
                )

    for node in ast.walk(tree):
        owner = _owner_for(node, parents, owners)
        if (
            relative == "agent_libos/runtime/checkpoint_manager.py"
            and isinstance(node, ast.arg)
            and (
                node.arg in {"store", "snapshot_repository"}
                or _attribute_path(node.annotation) == ("RuntimeStore",)
            )
        ):
            violations.append(
                Violation(
                    rule="typed-storage-reflection-bypass",
                    path=relative,
                    owner=owner,
                    detail=f"{node.arg}:raw-or-overridable-storage-port",
                    line=node.lineno,
                )
            )
        if (
            relative == "agent_libos/tools/broker.py"
            and isinstance(node, ast.arg)
            and node.arg == "store"
        ):
            violations.append(
                Violation(
                    rule="typed-storage-reflection-bypass",
                    path=relative,
                    owner=owner,
                    detail="store:raw-tool-broker-storage-port",
                    line=node.lineno,
                )
            )
        if (
            relative == "agent_libos/tools/broker.py"
            and isinstance(node, ast.arg)
            and _attribute_path(node.annotation) == ("RuntimeStore",)
        ):
            violations.append(
                Violation(
                    rule="typed-storage-reflection-bypass",
                    path=relative,
                    owner=owner,
                    detail=f"{node.arg}:raw-tool-broker-storage-type",
                    line=node.lineno,
                )
            )
        if (
            relative == "agent_libos/tools/broker.py"
            and isinstance(node, ast.arg)
            and node.arg == "unit_of_work"
            and _attribute_path(node.annotation) != ("UnitOfWork",)
        ):
            violations.append(
                Violation(
                    rule="typed-storage-reflection-bypass",
                    path=relative,
                    owner=owner,
                    detail="unit_of_work:untyped-tool-broker-storage-port",
                    line=node.lineno,
                )
            )
        if (
            relative == "agent_libos/tools/broker.py"
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and owner == "ToolBroker.__init__"
            and "unit_of_work"
            not in {
                argument.arg
                for argument in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
            }
        ):
            violations.append(
                Violation(
                    rule="typed-storage-reflection-bypass",
                    path=relative,
                    owner=owner,
                    detail="unit_of_work:missing-tool-broker-storage-port",
                    line=node.lineno,
                )
            )
        if (
            relative == "agent_libos/runtime/checkpoint_manager.py"
            and isinstance(node, ast.arg)
            and node.arg == "unit_of_work"
            and _attribute_path(node.annotation) != ("UnitOfWork",)
        ):
            violations.append(
                Violation(
                    rule="typed-storage-reflection-bypass",
                    path=relative,
                    owner=owner,
                    detail="unit_of_work:untyped-storage-port",
                    line=node.lineno,
                )
            )
        if (
            relative == "agent_libos/runtime/checkpoint_manager.py"
            and isinstance(node, ast.arg)
            and node.arg == "checkpoint_publication_writer"
            and _attribute_path(node.annotation)
            != ("CheckpointRestorePublicationWriterPort",)
        ):
            violations.append(
                Violation(
                    rule="typed-storage-reflection-bypass",
                    path=relative,
                    owner=owner,
                    detail="checkpoint_publication_writer:untyped-publication-port",
                    line=node.lineno,
                )
            )
        if (
            relative == "agent_libos/runtime/checkpoint_manager.py"
            and isinstance(node, ast.arg)
            and node.arg == "operations"
            and _attribute_path(node.annotation)
            != ("RuntimePublicationOperationPort",)
        ):
            violations.append(
                Violation(
                    rule="typed-storage-reflection-bypass",
                    path=relative,
                    owner=owner,
                    detail="operations:untyped-publication-operation-port",
                    line=node.lineno,
                )
            )
        if (
            relative == "agent_libos/runtime/checkpoint_reconciliation.py"
            and isinstance(node, ast.arg)
            and node.arg in {"store", "writer", "operations"}
        ):
            expected = {
                "store": "CheckpointRestorePublicationReader",
                "writer": "CheckpointRestorePublicationWriterPort",
                "operations": "RuntimePublicationOperationPort",
            }[node.arg]
            if _attribute_path(node.annotation) != (expected,):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail=f"{node.arg}:untyped-publication-port",
                        line=node.lineno,
                    )
                )
        if (
            relative == "agent_libos/runtime/builder.py"
            and isinstance(node, ast.Call)
            and _attribute_path(node.func) == ("CheckpointManager",)
        ):
            unit_of_work_value = (
                node.args[0]
                if node.args
                else next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "unit_of_work"
                    ),
                    None,
                )
            )
            if _attribute_path(unit_of_work_value) != ("host", "uow"):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail="unit_of_work:miswired-storage-port",
                        line=node.lineno,
                    )
                )
            writer_keyword = next(
                (
                    keyword
                    for keyword in node.keywords
                    if keyword.arg == "checkpoint_publication_writer"
                ),
                None,
            )
            if writer_keyword is None or _attribute_path(writer_keyword.value) != (
                "host",
                "uow",
                "checkpoint_restore_publications",
            ):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail="checkpoint_publication_writer:miswired-publication-port",
                        line=node.lineno,
                    )
                )
        if (
            relative == "agent_libos/runtime/builder.py"
            and isinstance(node, ast.Call)
            and _attribute_path(node.func) == ("ToolBroker",)
        ):
            tool_broker_uow = (
                node.args[0]
                if node.args
                else next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "unit_of_work"
                    ),
                    None,
                )
            )
            if _attribute_path(tool_broker_uow) != ("host", "uow"):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail="tool-broker:miswired-unit-of-work",
                        line=node.lineno,
                    )
                )
        if (
            relative == "agent_libos/runtime/builder.py"
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ):
            call_path = _attribute_path(node.func)
            if (
                len(call_path) >= 3
                and call_path[0] == "host"
                and (
                    call_path[-1].startswith("bind_")
                    or (
                        call_path[-1].startswith("add_")
                        and call_path[-1].endswith("_hook")
                    )
                    or (
                        call_path[-1].startswith("register_")
                        and call_path[-1].endswith("_recovery")
                    )
                )
            ):
                violations.append(
                    Violation(
                        rule="composition-late-binding",
                        path=relative,
                        owner=owner,
                        detail=".".join(call_path[1:]),
                        line=node.lineno,
                    )
                )
        for rule, package, forbidden in (
            ("concrete-api-import", "api", api_import_forbidden),
            ("concrete-runtime-import", "runtime", runtime_import_forbidden),
        ):
            if forbidden:
                for detail in _concrete_import_details(node, relative, package):
                    violations.append(
                        Violation(
                            rule=rule,
                            path=relative,
                            owner=owner,
                            detail=detail,
                            line=node.lineno,
                        )
                    )

        if isinstance(node, ast.Attribute) and _is_outermost_attribute(node, parents):
            access_path = _attribute_path(node)
            if (
                relative == "agent_libos/runtime/runtime.py"
                and access_path[:2] == ("self", "store")
            ):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail=".".join(access_path),
                        line=node.lineno,
                    )
                )
            if (
                relative == "agent_libos/tools/broker.py"
                and access_path[:2] == ("self", "store")
            ):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail=".".join(access_path),
                        line=node.lineno,
                    )
                )
            if (
                relative == "agent_libos/runtime/syscalls.py"
                and access_path[:3] == ("self", "runtime", "store")
            ):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail=".".join(access_path),
                        line=node.lineno,
                    )
                )
            if (
                relative == "agent_libos/runtime/checkpoint_manager.py"
                and access_path[:2] in {("self", "store"), ("self", "_store")}
            ):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail=".".join(access_path),
                        line=node.lineno,
                    )
                )
            if (
                relative == "agent_libos/runtime/checkpoint_reconciliation.py"
                and access_path[:3] == ("self", "_operations", "store")
            ):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail=".".join(access_path),
                        line=node.lineno,
                    )
                )
            if (
                relative != "agent_libos/storage/sql.py"
                and access_path
                and access_path[-1] == "update_process"
            ):
                violations.append(
                    Violation(
                        rule="whole-process-write",
                        path=relative,
                        owner=owner,
                        detail=".".join(access_path),
                        line=node.lineno,
                    )
                )
            private_detail = _cross_component_private_detail(
                access_path,
                component_aliases.get(owner, {}),
            )
            if private_detail is not None:
                violations.append(
                    Violation(
                        rule="cross-component-private-access",
                        path=relative,
                        owner=owner,
                        detail=private_detail,
                        line=node.lineno,
                    )
                )

            service_detail = _runtime_service_detail(
                access_path,
                runtime_aliases.get(owner, frozenset()),
                direct_call=_is_direct_call_target(node, parents),
            )
            if service_detail is not None:
                violations.append(
                    Violation(
                        rule="runtime-service-locator",
                        path=relative,
                        owner=owner,
                        detail=service_detail,
                        line=node.lineno,
                    )
                )

        if isinstance(node, ast.Call):
            call_path = _attribute_path(node.func)
            owner_class = owner.split(".", 1)[0]
            if (
                relative == "agent_libos/runtime/process_manager.py"
                and call_path
                and call_path[-1] in _PROCESS_MANAGER_STORAGE_METHOD_OWNERS
                and call_path[:-1]
                != _PROCESS_MANAGER_STORAGE_METHOD_OWNERS[call_path[-1]]
            ):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail=".".join(call_path),
                        line=node.lineno,
                    )
                )
            if (
                relative == "agent_libos/tools/broker.py"
                and call_path
                and call_path[-1] in _TOOL_BROKER_STORAGE_METHOD_OWNERS
                and call_path[:-1]
                != _TOOL_BROKER_STORAGE_METHOD_OWNERS[call_path[-1]]
            ):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail=".".join(call_path),
                        line=node.lineno,
                    )
                )
            if (
                relative == "agent_libos/runtime/checkpoint_manager.py"
                and len(call_path) == 3
                and call_path[0] == "self"
                and call_path[-1] in _CHECKPOINT_MANAGER_STORAGE_METHOD_OWNERS
                and call_path[1]
                not in _CHECKPOINT_MANAGER_STORAGE_METHOD_OWNERS[call_path[-1]]
            ):
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail=".".join(call_path),
                        line=node.lineno,
                    )
                )
            projection_table = _raw_jit_projection_mutation(
                node,
                relative=relative,
                scope=_nearest_function_scope(node, parents),
                bindings=static_name_bindings,
                definitions=static_name_definitions,
                parents=parents,
            )
            if projection_table is not None:
                violations.append(
                    Violation(
                        rule="runtime-storage-sql-bypass",
                        path=relative,
                        owner=owner,
                        detail=f"raw-jit-projection-sql:{projection_table}",
                        line=node.lineno,
                    )
                )
            typed_storage_reflection = (
                relative == "agent_libos/storage/repositories.py"
                and owner_class in _TYPED_STORAGE_REPOSITORY_CLASSES
                and (
                    (call_path and call_path[-1] == "_delegate")
                    or (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                    )
                )
            )
            if typed_storage_reflection:
                detail = ".".join(call_path) if call_path else "getattr"
                violations.append(
                    Violation(
                        rule="typed-storage-reflection-bypass",
                        path=relative,
                        owner=owner,
                        detail=detail,
                        line=node.lineno,
                    )
                )
            if (
                relative in _TYPED_STORAGE_RUNTIME_FILES
                and call_path
                and call_path[-1] in _RAW_STORAGE_CALLS
            ):
                violations.append(
                    Violation(
                        rule="runtime-storage-sql-bypass",
                        path=relative,
                        owner=owner,
                        detail=".".join(call_path),
                        line=node.lineno,
                    )
                )
            if call_path in {("asyncio", "to_thread"), ("to_thread",)}:
                violations.append(
                    Violation(
                        rule="untracked-blocking-dispatch",
                        path=relative,
                        owner=owner,
                        detail=".".join(call_path),
                        line=node.lineno,
                    )
                )
            if _is_default_executor_dispatch(node):
                violations.append(
                    Violation(
                        rule="untracked-blocking-dispatch",
                        path=relative,
                        owner=owner,
                        detail="run_in_executor(None)",
                        line=node.lineno,
                    )
                )
            access_path = _static_getattr_path(node)
            if access_path:
                if (
                    relative in _TYPED_STORAGE_RUNTIME_FILES
                    and access_path[-1] in _RAW_STORAGE_CALLS
                ):
                    violations.append(
                        Violation(
                            rule="runtime-storage-sql-bypass",
                            path=relative,
                            owner=owner,
                            detail=".".join(access_path),
                            line=node.lineno,
                        )
                    )
                if (
                    relative != "agent_libos/storage/sql.py"
                    and access_path[-1] == "update_process"
                ):
                    violations.append(
                        Violation(
                            rule="whole-process-write",
                            path=relative,
                            owner=owner,
                            detail=".".join(access_path),
                            line=node.lineno,
                        )
                    )
                private_detail = _cross_component_private_detail(
                    access_path,
                    component_aliases.get(owner, {}),
                )
                if private_detail is not None:
                    violations.append(
                        Violation(
                            rule="cross-component-private-access",
                            path=relative,
                            owner=owner,
                            detail=private_detail,
                            line=node.lineno,
                        )
                    )
                service_detail = _runtime_service_detail(
                    access_path,
                    runtime_aliases.get(owner, frozenset()),
                    direct_call=False,
                )
                if service_detail is not None:
                    violations.append(
                        Violation(
                            rule="runtime-service-locator",
                            path=relative,
                            owner=owner,
                            detail=service_detail,
                            line=node.lineno,
                        )
                    )

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = _function_start(node)
            functions.append(
                FunctionMetric(
                    path=relative,
                    owner=owners[node],
                    line=start,
                    lines=(node.end_lineno or node.lineno) - start + 1,
                    complexity=_function_complexity(node),
                )
            )

    return ArchitectureReport(tuple(violations), tuple(functions))


def scan_architecture(root: Path) -> ArchitectureReport:
    violations: list[Violation] = []
    functions: list[FunctionMetric] = []
    for path in _source_paths(root):
        report = _scan_file(path, root)
        violations.extend(report.violations)
        functions.extend(report.functions)
    violations.extend(_runtime_component_declaration_violations(root))
    return ArchitectureReport(
        tuple(
            sorted(
                violations,
                key=lambda item: (item.path, item.line, item.rule, item.detail),
            )
        ),
        tuple(sorted(functions, key=lambda item: (item.path, item.line, item.owner))),
    )


def _runtime_component_declaration_violations(root: Path) -> tuple[Violation, ...]:
    """Keep the mutable composition host explicit to static tooling."""

    runtime_path = root / "agent_libos" / "runtime" / "runtime.py"
    builder_path = root / "agent_libos" / "runtime" / "builder.py"
    if not runtime_path.is_file() or not builder_path.is_file():
        return ()
    runtime_tree = ast.parse(
        runtime_path.read_text(encoding="utf-8"),
        filename=str(runtime_path),
    )
    declared: set[str] = set()
    for node in runtime_tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Runtime":
            continue
        declared.update(
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        )
        break

    builder_tree = ast.parse(
        builder_path.read_text(encoding="utf-8"),
        filename=str(builder_path),
    )
    parents = _parents(builder_tree)
    owners = _qualified_owners(builder_tree)
    violations: list[Violation] = []
    for node in ast.walk(builder_tree):
        targets: tuple[ast.AST, ...]
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        else:
            continue
        owner = _owner_for(node, parents, owners)
        if not owner.startswith("RuntimeBuilder."):
            continue
        for target in targets:
            access_path = _attribute_path(target)
            if (
                len(access_path) == 2
                and access_path[0] == "host"
                and access_path[1] not in declared
            ):
                violations.append(
                    Violation(
                        rule="undeclared-runtime-component",
                        path="agent_libos/runtime/builder.py",
                        owner=owner,
                        detail=access_path[1],
                        line=node.lineno,
                    )
                )
    return tuple(violations)


def _integer_mapping(value: object, *, field: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    result: dict[str, int] = {}
    for key, limit in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 0
        ):
            raise ValueError(f"{field} entries must map strings to non-negative integers")
        result[key] = limit
    return result


def load_allowlist(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"architecture allowlist not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid architecture allowlist JSON: {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"architecture allowlist schema_version must be {SCHEMA_VERSION}")
    rules = data.get("violation_budgets")
    if not isinstance(rules, dict) or set(rules) != set(RATCHET_RULES):
        raise ValueError(f"violation_budgets must contain exactly: {', '.join(RATCHET_RULES)}")
    normalized_rules = {
        rule: _integer_mapping(rules[rule], field=f"violation_budgets.{rule}")
        for rule in RATCHET_RULES
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "violation_budgets": normalized_rules,
        "function_line_budgets": _integer_mapping(
            data.get("function_line_budgets"), field="function_line_budgets"
        ),
        "complexity_budgets": _integer_mapping(
            data.get("complexity_budgets"), field="complexity_budgets"
        ),
    }


def allowlist_for(report: ArchitectureReport) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {rule: Counter() for rule in RATCHET_RULES}
    for violation in report.violations:
        counts[violation.rule][violation.budget_key] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "violation_budgets": {
            rule: dict(sorted(counts[rule].items())) for rule in RATCHET_RULES
        },
        "function_line_budgets": {
            metric.budget_key: metric.lines
            for metric in report.functions
            if metric.lines > MAX_FUNCTION_LINES
        },
        "complexity_budgets": {
            metric.budget_key: metric.complexity
            for metric in report.functions
            if metric.complexity > COMPLEXITY_HOTSPOT_THRESHOLD
        },
    }


def check_report(report: ArchitectureReport, allowlist: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    observed: dict[str, Counter[str]] = {rule: Counter() for rule in RATCHET_RULES}
    examples: dict[tuple[str, str], Violation] = {}
    for violation in report.violations:
        observed[violation.rule][violation.budget_key] += 1
        examples.setdefault((violation.rule, violation.budget_key), violation)

    violation_budgets = allowlist["violation_budgets"]
    for rule in RATCHET_RULES:
        budgets = violation_budgets[rule]
        for key, count in sorted(observed[rule].items()):
            maximum = budgets.get(key, 0)
            if count > maximum:
                example = examples[(rule, key)]
                errors.append(
                    f"{example.describe()}; observed {count}, ratchet maximum {maximum}"
                )
        for key, maximum in sorted(budgets.items()):
            count = observed[rule].get(key, 0)
            if count < maximum or maximum == 0:
                errors.append(
                    f"stale {rule} budget: {key}; observed {count}, "
                    f"ratchet maximum {maximum}; lower or remove the budget"
                )

    line_budgets = allowlist["function_line_budgets"]
    complexity_budgets = allowlist["complexity_budgets"]
    functions_by_key = {metric.budget_key: metric for metric in report.functions}
    for metric in report.functions:
        if metric.lines > MAX_FUNCTION_LINES:
            maximum = line_budgets.get(metric.budget_key, MAX_FUNCTION_LINES)
            if metric.lines > maximum:
                errors.append(
                    f"{metric.path}:{metric.line}: long-function: {metric.owner} has "
                    f"{metric.lines} lines; ratchet maximum {maximum}"
                )
        if metric.complexity > COMPLEXITY_HOTSPOT_THRESHOLD:
            maximum = complexity_budgets.get(
                metric.budget_key, COMPLEXITY_HOTSPOT_THRESHOLD
            )
            if metric.complexity > maximum:
                errors.append(
                    f"{metric.path}:{metric.line}: complexity-hotspot: {metric.owner} has "
                    f"complexity {metric.complexity}; ratchet maximum {maximum}"
                )
    for key, maximum in sorted(line_budgets.items()):
        metric = functions_by_key.get(key)
        observed_lines = (
            metric.lines
            if metric is not None and metric.lines > MAX_FUNCTION_LINES
            else 0
        )
        if observed_lines < maximum or maximum <= MAX_FUNCTION_LINES:
            errors.append(
                f"stale long-function budget: {key}; observed {observed_lines}, "
                f"ratchet maximum {maximum}; lower or remove the budget"
            )
    for key, maximum in sorted(complexity_budgets.items()):
        metric = functions_by_key.get(key)
        observed_complexity = (
            metric.complexity
            if metric is not None
            and metric.complexity > COMPLEXITY_HOTSPOT_THRESHOLD
            else 0
        )
        if (
            observed_complexity < maximum
            or maximum <= COMPLEXITY_HOTSPOT_THRESHOLD
        ):
            errors.append(
                f"stale complexity-hotspot budget: {key}; "
                f"observed {observed_complexity}, ratchet maximum {maximum}; "
                "lower or remove the budget"
            )
    return errors


def check_architecture(root: Path, allowlist_path: Path) -> list[str]:
    return check_report(scan_architecture(root), load_allowlist(allowlist_path))


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reject architecture debt beyond the checked-in ratchet allowlist."
    )
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=repository_root)
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path(__file__).with_name("architecture_allowlist.json"),
    )
    parser.add_argument(
        "--print-baseline",
        action="store_true",
        help="print the currently observed budgets for manual review; does not write files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    try:
        report = scan_architecture(root)
        if args.print_baseline:
            sys.stdout.write(_json_text(allowlist_for(report)))
            return 0
        errors = check_report(report, load_allowlist(args.allowlist.resolve()))
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"architecture check failed: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("Architecture ratchet failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print(
            "Reduce the debt or deliberately lower/review scripts/architecture_allowlist.json; "
            "use --print-baseline only to inspect candidate budgets.",
            file=sys.stderr,
        )
        return 1
    print("Architecture ratchet passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
