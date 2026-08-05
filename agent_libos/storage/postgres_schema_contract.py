from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent_libos.models.exceptions import UnsupportedStoreVersion


_MANIFEST_FORMAT_VERSION = 1
_MANIFEST_PATH = Path(__file__).with_name("postgres_schema_manifest.json")
_SEMANTIC_TABLES = frozenset(
    {"semantic_assessment_jobs", "semantic_assessments"}
)
_POSTGRES_USER_RELKINDS = ("r", "p", "v", "m", "S", "f", "c")
# Captured from an actual PostgreSQL 17.10 schema initialized by clean baseline
# commit 4b43cb7.  This makes the v4 migration admission contract independent of
# merely assuming that "v5 minus the new objects" still matches production v4.
POSTGRES_V4_BASELINE_4B43CB7_CATALOG_SHA256 = (
    "268c89fe9291aae79650ce4951bc802f298b3e62b75fc5098b3825a0e16cab6d"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_expression(value: object) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", " ", str(value).strip())


def _rows(conn: Any, sql: str, params: tuple[object, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, params)]


def capture_postgres_catalog(conn: Any) -> dict[str, Any]:
    """Capture the complete current-schema storage contract from pg_catalog.

    This function is deliberately read-only.  The generated artifact is
    committed to the package and runtime startup compares a new capture with
    that artifact; startup never derives or repairs its expected schema from a
    live database.
    """

    server_row = conn.execute(
        "SELECT current_setting('server_version_num') AS version_num"
    ).fetchone()
    version_num = int(server_row["version_num"])
    relations = _capture_relations(conn)
    columns = _capture_columns(conn)
    constraints = _capture_constraints(conn)
    indexes = _capture_indexes(conn)
    table_names = tuple(
        item["name"] for item in relations if item["kind"] in {"r", "p"}
    )
    hooks = _capture_mutation_hooks(conn, table_names)
    inheritance = _capture_inheritance(conn)
    return {
        "postgres_major": version_num // 10_000,
        "relations": relations,
        "columns": columns,
        "constraints": constraints,
        "indexes": indexes,
        "triggers": hooks["triggers"],
        "policies": hooks["policies"],
        "rewrite_rules": hooks["rewrite_rules"],
        "inheritance": inheritance,
    }


def _capture_relations(conn: Any) -> list[dict[str, Any]]:
    relkinds = ", ".join("?" for _ in _POSTGRES_USER_RELKINDS)
    rows = _rows(
        conn,
        f"""
        SELECT relation.relname AS relation_name,
               relation.relkind AS relation_kind,
               relation.relpersistence AS persistence,
               access_method.amname AS access_method,
               relation.relreplident AS replica_identity,
               relation.reloptions AS relation_options,
               tablespace.spcname AS tablespace,
               relation.relispartition AS is_partition,
               relation.relrowsecurity AS row_security,
               relation.relforcerowsecurity AS force_row_security
          FROM pg_catalog.pg_class AS relation
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
     LEFT JOIN pg_catalog.pg_am AS access_method
            ON access_method.oid = relation.relam
     LEFT JOIN pg_catalog.pg_tablespace AS tablespace
            ON tablespace.oid = relation.reltablespace
         WHERE namespace.nspname = current_schema()
           AND relation.relkind IN ({relkinds})
         ORDER BY relation.relname
        """,
        _POSTGRES_USER_RELKINDS,
    )
    return [
        {
            "name": str(row["relation_name"]),
            "kind": str(row["relation_kind"]),
            "persistence": str(row["persistence"]),
            "access_method": (
                str(row["access_method"])
                if row["access_method"] is not None
                else None
            ),
            "replica_identity": str(row["replica_identity"]),
            "options": sorted(str(value) for value in (row["relation_options"] or ())),
            "tablespace": (
                str(row["tablespace"])
                if row["tablespace"] is not None
                else None
            ),
            "is_partition": bool(row["is_partition"]),
            "row_security": bool(row["row_security"]),
            "force_row_security": bool(row["force_row_security"]),
        }
        for row in rows
    ]


def _capture_columns(conn: Any) -> dict[str, list[dict[str, Any]]]:
    rows = _rows(
        conn,
        """
        SELECT relation.relname AS table_name,
               attribute.attname AS column_name,
               attribute.attnum AS physical_ordinal,
               pg_catalog.format_type(
                 attribute.atttypid, attribute.atttypmod
               ) AS formatted_type,
               attribute.attnotnull AS not_null,
               pg_catalog.pg_get_expr(
                 default_row.adbin, default_row.adrelid, true
               ) AS default_expression,
               CASE
                 WHEN attribute.attcollation = type_row.typcollation THEN NULL
                 ELSE collation_namespace.nspname || '.' || collation_row.collname
               END AS explicit_collation,
               attribute.attidentity AS identity_kind,
               attribute.attgenerated AS generated_kind
          FROM pg_catalog.pg_attribute AS attribute
          JOIN pg_catalog.pg_class AS relation
            ON relation.oid = attribute.attrelid
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
          JOIN pg_catalog.pg_type AS type_row
            ON type_row.oid = attribute.atttypid
     LEFT JOIN pg_catalog.pg_attrdef AS default_row
            ON default_row.adrelid = relation.oid
           AND default_row.adnum = attribute.attnum
     LEFT JOIN pg_catalog.pg_collation AS collation_row
            ON collation_row.oid = attribute.attcollation
     LEFT JOIN pg_catalog.pg_namespace AS collation_namespace
            ON collation_namespace.oid = collation_row.collnamespace
         WHERE namespace.nspname = current_schema()
           AND relation.relkind IN ('r', 'p', 'v', 'm', 'f', 'c')
           AND attribute.attnum > 0
           AND NOT attribute.attisdropped
         ORDER BY relation.relname, attribute.attnum
        """,
    )
    selected: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        table = str(row["table_name"])
        table_columns = selected.setdefault(table, [])
        # Visible ordinal order is canonical.  Physical attnum is intentionally
        # not persisted: a historical v4 store produced by dropping the final
        # v5 human revision column retains a harmless hidden pg_attribute row.
        table_columns.append(
            {
                "name": str(row["column_name"]),
                "ordinal": len(table_columns) + 1,
                "type": str(row["formatted_type"]),
                "nullable": not bool(row["not_null"]),
                "default": _canonical_expression(row["default_expression"]),
                "collation": (
                    str(row["explicit_collation"])
                    if row["explicit_collation"] is not None
                    else None
                ),
                "identity": str(row["identity_kind"] or ""),
                "generated": str(row["generated_kind"] or ""),
            }
        )
    return selected


def _capture_constraints(conn: Any) -> list[dict[str, Any]]:
    rows = _rows(
        conn,
        """
        SELECT relation.relname AS table_name,
               constraint_row.conname AS constraint_name,
               constraint_row.contype AS constraint_type,
               pg_catalog.pg_get_constraintdef(
                 constraint_row.oid, true
               ) AS definition,
               constraint_row.condeferrable AS is_deferrable,
               constraint_row.condeferred AS is_deferred,
               constraint_row.convalidated AS is_validated,
               constraint_row.connoinherit AS no_inherit,
               constraint_row.conislocal AS is_local,
               constraint_row.coninhcount AS inheritance_count,
               constraint_row.confupdtype AS update_action,
               constraint_row.confdeltype AS delete_action,
               constraint_row.confmatchtype AS match_type,
               CASE
                 WHEN referenced_namespace.nspname = current_schema()
                   THEN '<current>'
                 ELSE referenced_namespace.nspname
               END AS referenced_schema,
               referenced_relation.relname AS referenced_table,
               ARRAY(
                 SELECT attribute.attname
                   FROM unnest(constraint_row.conkey)
                        WITH ORDINALITY AS key_row(attnum, ordinality)
                   JOIN pg_catalog.pg_attribute AS attribute
                     ON attribute.attrelid = relation.oid
                    AND attribute.attnum = key_row.attnum
                  ORDER BY key_row.ordinality
               ) AS columns,
               ARRAY(
                 SELECT attribute.attname
                   FROM unnest(constraint_row.confkey)
                        WITH ORDINALITY AS key_row(attnum, ordinality)
                   JOIN pg_catalog.pg_attribute AS attribute
                     ON attribute.attrelid = referenced_relation.oid
                    AND attribute.attnum = key_row.attnum
                  ORDER BY key_row.ordinality
               ) AS referenced_columns
          FROM pg_catalog.pg_constraint AS constraint_row
          JOIN pg_catalog.pg_class AS relation
            ON relation.oid = constraint_row.conrelid
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
     LEFT JOIN pg_catalog.pg_class AS referenced_relation
            ON referenced_relation.oid = constraint_row.confrelid
     LEFT JOIN pg_catalog.pg_namespace AS referenced_namespace
            ON referenced_namespace.oid = referenced_relation.relnamespace
         WHERE namespace.nspname = current_schema()
         ORDER BY relation.relname, constraint_row.conname
        """,
    )
    return [
        {
            "table": str(row["table_name"]),
            "name": str(row["constraint_name"]),
            "kind": str(row["constraint_type"]),
            "columns": [str(value) for value in row["columns"]],
            "definition": _canonical_expression(row["definition"]),
            "deferrable": bool(row["is_deferrable"]),
            "deferred": bool(row["is_deferred"]),
            "validated": bool(row["is_validated"]),
            "no_inherit": bool(row["no_inherit"]),
            "is_local": bool(row["is_local"]),
            "inheritance_count": int(row["inheritance_count"]),
            "referenced_schema": (
                str(row["referenced_schema"])
                if row["referenced_schema"] is not None
                else None
            ),
            "referenced_table": (
                str(row["referenced_table"])
                if row["referenced_table"] is not None
                else None
            ),
            "referenced_columns": [
                str(value) for value in row["referenced_columns"]
            ],
            "update_action": str(row["update_action"]),
            "delete_action": str(row["delete_action"]),
            "match_type": str(row["match_type"]),
        }
        for row in rows
    ]


def _capture_indexes(conn: Any) -> list[dict[str, Any]]:
    rows = _rows(
        conn,
        """
        SELECT table_relation.relname AS table_name,
               index_relation.relname AS index_name,
               access_method.amname AS access_method,
               index_row.indisunique AS is_unique,
               index_row.indisprimary AS is_primary,
               index_row.indisexclusion AS is_exclusion,
               index_row.indimmediate AS is_immediate,
               index_row.indisclustered AS is_clustered,
               index_row.indisvalid AS is_valid,
               index_row.indisready AS is_ready,
               index_row.indislive AS is_live,
               index_row.indisreplident AS is_replica_identity,
               index_row.indnullsnotdistinct AS nulls_not_distinct,
               index_row.indnkeyatts AS key_count,
               index_row.indnatts AS attribute_count,
               pg_catalog.pg_get_expr(
                 index_row.indpred, index_row.indrelid, true
               ) AS predicate,
               pg_catalog.pg_get_indexdef(
                 index_row.indexrelid, 0, true
               ) AS definition,
               ARRAY(
                 SELECT pg_catalog.pg_get_indexdef(
                   index_row.indexrelid, ordinal.position, true
                 )
                   FROM generate_series(1, index_row.indnatts)
                        AS ordinal(position)
                  ORDER BY ordinal.position
               ) AS attributes,
               ARRAY(
                 SELECT CASE
                          WHEN collation_oid = 0 THEN NULL
                          ELSE collation_namespace.nspname || '.' || collation_row.collname
                        END
                   FROM unnest(index_row.indcollation)
                        WITH ORDINALITY AS item(collation_oid, position)
              LEFT JOIN pg_catalog.pg_collation AS collation_row
                     ON collation_row.oid = item.collation_oid
              LEFT JOIN pg_catalog.pg_namespace AS collation_namespace
                     ON collation_namespace.oid = collation_row.collnamespace
                  ORDER BY item.position
               ) AS collations,
               ARRAY(
                 SELECT operator_namespace.nspname || '.' || operator_class.opcname
                   FROM unnest(index_row.indclass)
                        WITH ORDINALITY AS item(operator_class_oid, position)
                   JOIN pg_catalog.pg_opclass AS operator_class
                     ON operator_class.oid = item.operator_class_oid
                   JOIN pg_catalog.pg_namespace AS operator_namespace
                     ON operator_namespace.oid = operator_class.opcnamespace
                  ORDER BY item.position
               ) AS operator_classes,
               index_row.indoption::smallint[] AS options
          FROM pg_catalog.pg_index AS index_row
          JOIN pg_catalog.pg_class AS index_relation
            ON index_relation.oid = index_row.indexrelid
          JOIN pg_catalog.pg_class AS table_relation
            ON table_relation.oid = index_row.indrelid
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = table_relation.relnamespace
          JOIN pg_catalog.pg_am AS access_method
            ON access_method.oid = index_relation.relam
         WHERE namespace.nspname = current_schema()
         ORDER BY table_relation.relname, index_relation.relname
        """,
    )
    return [
        {
            "table": str(row["table_name"]),
            "name": str(row["index_name"]),
            "access_method": str(row["access_method"]),
            "unique": bool(row["is_unique"]),
            "primary": bool(row["is_primary"]),
            "exclusion": bool(row["is_exclusion"]),
            "immediate": bool(row["is_immediate"]),
            "clustered": bool(row["is_clustered"]),
            "valid": bool(row["is_valid"]),
            "ready": bool(row["is_ready"]),
            "live": bool(row["is_live"]),
            "replica_identity": bool(row["is_replica_identity"]),
            "nulls_not_distinct": bool(row["nulls_not_distinct"]),
            "key_count": int(row["key_count"]),
            "attribute_count": int(row["attribute_count"]),
            "predicate": _canonical_expression(row["predicate"]),
            "definition": _canonical_expression(row["definition"]),
            "attributes": [
                _canonical_expression(value) for value in row["attributes"]
            ],
            "collations": [
                str(value) if value is not None else None
                for value in row["collations"]
            ],
            "operator_classes": [
                str(value) for value in row["operator_classes"]
            ],
            "options": [int(value) for value in row["options"]],
        }
        for row in rows
    ]


def _capture_mutation_hooks(
    conn: Any,
    table_names: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    if not table_names:
        return {"triggers": [], "policies": [], "rewrite_rules": []}
    placeholders = ", ".join("?" for _ in table_names)
    triggers = _rows(
        conn,
        f"""
        SELECT relation.relname AS table_name,
               trigger_row.tgname AS trigger_name,
               pg_catalog.pg_get_triggerdef(trigger_row.oid, true) AS definition
          FROM pg_catalog.pg_trigger AS trigger_row
          JOIN pg_catalog.pg_class AS relation
            ON relation.oid = trigger_row.tgrelid
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname = current_schema()
           AND relation.relname IN ({placeholders})
           AND NOT trigger_row.tgisinternal
         ORDER BY relation.relname, trigger_row.tgname
        """,
        table_names,
    )
    policies = _rows(
        conn,
        f"""
        SELECT relation.relname AS table_name,
               policy_row.polname AS policy_name,
               policy_row.polcmd AS command,
               policy_row.polpermissive AS permissive,
               pg_catalog.pg_get_expr(
                 policy_row.polqual, policy_row.polrelid, true
               ) AS using_expression,
               pg_catalog.pg_get_expr(
                 policy_row.polwithcheck, policy_row.polrelid, true
               ) AS check_expression
          FROM pg_catalog.pg_policy AS policy_row
          JOIN pg_catalog.pg_class AS relation
            ON relation.oid = policy_row.polrelid
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname = current_schema()
           AND relation.relname IN ({placeholders})
         ORDER BY relation.relname, policy_row.polname
        """,
        table_names,
    )
    rewrite_rules = _rows(
        conn,
        f"""
        SELECT relation.relname AS table_name,
               rewrite_row.rulename AS rule_name,
               pg_catalog.pg_get_ruledef(rewrite_row.oid, true) AS definition
          FROM pg_catalog.pg_rewrite AS rewrite_row
          JOIN pg_catalog.pg_class AS relation
            ON relation.oid = rewrite_row.ev_class
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname = current_schema()
           AND relation.relname IN ({placeholders})
         ORDER BY relation.relname, rewrite_row.rulename
        """,
        table_names,
    )
    return {
        "triggers": [
            {
                "table": str(row["table_name"]),
                "name": str(row["trigger_name"]),
                "definition": _canonical_expression(row["definition"]),
            }
            for row in triggers
        ],
        "policies": [
            {
                "table": str(row["table_name"]),
                "name": str(row["policy_name"]),
                "command": str(row["command"]),
                "permissive": bool(row["permissive"]),
                "using": _canonical_expression(row["using_expression"]),
                "check": _canonical_expression(row["check_expression"]),
            }
            for row in policies
        ],
        "rewrite_rules": [
            {
                "table": str(row["table_name"]),
                "name": str(row["rule_name"]),
                "definition": _canonical_expression(row["definition"]),
            }
            for row in rewrite_rules
        ],
    }


def _capture_inheritance(conn: Any) -> list[dict[str, Any]]:
    rows = _rows(
        conn,
        """
        SELECT child.relname AS child_table,
               CASE
                 WHEN parent_namespace.nspname = current_schema()
                   THEN '<current>'
                 ELSE parent_namespace.nspname
               END AS parent_schema,
               parent.relname AS parent_table,
               inheritance.inhseqno AS sequence
          FROM pg_catalog.pg_inherits AS inheritance
          JOIN pg_catalog.pg_class AS child
            ON child.oid = inheritance.inhrelid
          JOIN pg_catalog.pg_namespace AS child_namespace
            ON child_namespace.oid = child.relnamespace
          JOIN pg_catalog.pg_class AS parent
            ON parent.oid = inheritance.inhparent
          JOIN pg_catalog.pg_namespace AS parent_namespace
            ON parent_namespace.oid = parent.relnamespace
         WHERE child_namespace.nspname = current_schema()
         ORDER BY child.relname, inheritance.inhseqno
        """,
    )
    return [
        {
            "child": str(row["child_table"]),
            "parent_schema": str(row["parent_schema"]),
            "parent": str(row["parent_table"]),
            "sequence": int(row["sequence"]),
        }
        for row in rows
    ]


def build_postgres_manifest(
    catalog: Mapping[str, Any],
    *,
    generated_postgres_version_num: int | None = None,
) -> dict[str, Any]:
    selected = copy.deepcopy(dict(catalog))
    return {
        "format_version": _MANIFEST_FORMAT_VERSION,
        "generated_postgres_version_num": generated_postgres_version_num,
        "catalog_sha256": _sha256(selected),
        "catalog": selected,
    }


@lru_cache(maxsize=1)
def load_postgres_v5_manifest() -> dict[str, Any]:
    try:
        payload = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedStoreVersion(
            "Agent libOS PostgreSQL canonical schema manifest is unavailable"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != _MANIFEST_FORMAT_VERSION
        or type(payload.get("generated_postgres_version_num")) is not int
        or not isinstance(payload.get("catalog"), dict)
        or payload.get("catalog_sha256") != _sha256(payload.get("catalog"))
        or payload["generated_postgres_version_num"] // 10_000
        != payload["catalog"].get("postgres_major")
    ):
        raise UnsupportedStoreVersion(
            "Agent libOS PostgreSQL canonical schema manifest is invalid"
        )
    return copy.deepcopy(payload)


def expected_postgres_catalog(store_version: int) -> dict[str, Any]:
    manifest = load_postgres_v5_manifest()
    catalog = copy.deepcopy(manifest["catalog"])
    if store_version == 5:
        return catalog
    if store_version != 4:
        raise ValueError(f"unsupported PostgreSQL catalog version: {store_version}")
    catalog["relations"] = [
        item for item in catalog["relations"] if item["name"] not in _SEMANTIC_TABLES
    ]
    for table in _SEMANTIC_TABLES:
        catalog["columns"].pop(table, None)
    catalog["columns"]["human_requests"] = [
        item
        for item in catalog["columns"]["human_requests"]
        if item["name"] != "revision"
    ]
    catalog["constraints"] = [
        item
        for item in catalog["constraints"]
        if item["table"] not in _SEMANTIC_TABLES
        and not (
            item["table"] == "human_requests"
            and item["name"] == "human_requests_revision_check"
        )
    ]
    catalog["indexes"] = [
        item for item in catalog["indexes"] if item["table"] not in _SEMANTIC_TABLES
    ]
    if _sha256(catalog) != POSTGRES_V4_BASELINE_4B43CB7_CATALOG_SHA256:
        raise UnsupportedStoreVersion(
            "Agent libOS PostgreSQL canonical schema v4 derivation no longer "
            "matches baseline 4b43cb7"
        )
    return catalog


def require_postgres_catalog_contract(conn: Any, *, store_version: int) -> None:
    expected = expected_postgres_catalog(store_version)
    actual = capture_postgres_catalog(conn)
    if actual == expected:
        return
    problems = _catalog_problems(expected, actual)
    index_problems = problems.get("indexes")
    binding_index_changed = (
        isinstance(index_problems, Mapping)
        and "operations.idx_operations_runtime_publication"
        in {
            *index_problems.get("missing", ()),
            *index_problems.get("extra", ()),
            *index_problems.get("changed", ()),
        }
    )
    binding_detail = (
        "exact durable runtime-publication binding index differs; "
        if binding_index_changed
        else ""
    )
    raise UnsupportedStoreVersion(
        "unsupported Agent libOS PostgreSQL canonical schema "
        f"v{store_version} catalog manifest: {binding_detail}{problems}; "
        f"expected_sha256={_sha256(expected)}, actual_sha256={_sha256(actual)}"
    )


def _catalog_problems(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, Any]:
    problems: dict[str, Any] = {}
    for section in (
        "relations",
        "columns",
        "constraints",
        "indexes",
        "triggers",
        "policies",
        "rewrite_rules",
        "inheritance",
    ):
        expected_section = expected.get(section)
        actual_section = actual.get(section)
        if expected_section == actual_section:
            continue
        if section == "columns":
            problems[section] = _mapping_diff(expected_section, actual_section)
        elif section in {"relations"}:
            problems[section] = _list_diff(expected_section, actual_section, ("name",))
        elif section in {"constraints", "indexes"}:
            problems[section] = _list_diff(
                expected_section, actual_section, ("table", "name")
            )
        elif section in {"triggers", "policies", "rewrite_rules"}:
            problems[section] = _list_diff(
                expected_section, actual_section, ("table", "name")
            )
        else:
            problems[section] = {
                "expected_count": len(expected_section or ()),
                "actual_count": len(actual_section or ()),
            }
    return problems


def _mapping_diff(expected: object, actual: object) -> dict[str, Any]:
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return {"expected_type": type(expected).__name__, "actual_type": type(actual).__name__}
    expected_keys = set(expected)
    actual_keys = set(actual)
    return {
        "missing": sorted(expected_keys - actual_keys)[:20],
        "extra": sorted(actual_keys - expected_keys)[:20],
        "changed": sorted(
            key
            for key in expected_keys & actual_keys
            if expected[key] != actual[key]
        )[:20],
    }


def _list_diff(
    expected: object,
    actual: object,
    key_fields: tuple[str, ...],
) -> dict[str, Any]:
    def keyed(value: object) -> dict[str, object]:
        if not isinstance(value, list):
            return {}
        return {
            ".".join(str(item.get(field, "")) for field in key_fields): item
            for item in value
            if isinstance(item, dict)
        }

    expected_by_key = keyed(expected)
    actual_by_key = keyed(actual)
    expected_keys = set(expected_by_key)
    actual_keys = set(actual_by_key)
    return {
        "missing": sorted(expected_keys - actual_keys)[:20],
        "extra": sorted(actual_keys - expected_keys)[:20],
        "changed": sorted(
            key
            for key in expected_keys & actual_keys
            if expected_by_key[key] != actual_by_key[key]
        )[:20],
    }
