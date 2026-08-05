from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from agent_libos.storage.postgres import PostgresStore
from agent_libos.storage.postgres_schema_contract import (
    build_postgres_manifest,
    capture_postgres_catalog,
)


class _ManifestBootstrapPostgresStore(PostgresStore):
    @classmethod
    def _require_canonical_catalog_contract(
        cls,
        conn: object,
        *,
        store_version: int,
    ) -> None:
        del conn, store_version


def _schema_dsn(base_dsn: str, schema: str) -> str:
    parsed = urlsplit(base_dsn)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "options"
    ]
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dsn",
        default=os.environ.get("AGENT_LIBOS_POSTGRES_DSN"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("agent_libos/storage/postgres_schema_manifest.json"),
    )
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or AGENT_LIBOS_POSTGRES_DSN is required")

    import psycopg
    from psycopg import sql

    schema = f"agent_libos_manifest_{uuid4().hex}"
    selected_dsn = _schema_dsn(args.dsn, schema)
    with psycopg.connect(args.dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
        )
    try:
        store = _ManifestBootstrapPostgresStore(selected_dsn)
        try:
            server_row = store.conn.execute(
                "SELECT current_setting('server_version_num') AS version_num"
            ).fetchone()
            manifest = build_postgres_manifest(
                capture_postgres_catalog(store.conn),
                generated_postgres_version_num=int(server_row["version_num"]),
            )
        finally:
            store.close()
        args.output.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        with psycopg.connect(args.dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
