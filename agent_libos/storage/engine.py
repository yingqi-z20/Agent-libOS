from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Protocol, runtime_checkable


def split_sql_script(script: str) -> list[str]:
    """Split the repository's static DDL without executing implicit commits."""

    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(script):
        char = script[index]
        if quote is not None:
            current.append(char)
            if char == quote:
                if index + 1 < len(script) and script[index + 1] == quote:
                    current.append(script[index + 1])
                    index += 1
                else:
                    quote = None
        elif char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(char)
        index += 1
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


@runtime_checkable
class SqlSession(Protocol):
    """Small backend-neutral cursor contract used by SQL repositories."""

    def close(self) -> None:
        ...

    @property
    def rowcount(self) -> int:
        ...

    def execute(self, sql: str, params: Iterable[Any] = ()) -> "SqlSession":
        ...

    def executemany(self, sql: str, seq_of_params: Iterable[Iterable[Any]]) -> Any:
        ...

    def fetchone(self) -> Any | None:
        ...

    def __iter__(self) -> Iterator[Any]:
        ...


@runtime_checkable
class SqlEngine(Protocol):
    """Connection-level SQL surface shared by SQLite and PostgreSQL.

    This intentionally describes only the operations already consumed by the
    shared repositories. Backend creation, dialect conversion, fresh-schema
    creation, and version validation remain backend concerns.
    """

    def close(self) -> None:
        ...

    def commit(self) -> None:
        ...

    def rollback(self) -> None:
        ...

    def cursor(self) -> SqlSession:
        ...

    def execute(self, sql: str, params: Iterable[Any] = ()) -> SqlSession:
        ...

    def executemany(self, sql: str, seq_of_params: Iterable[Iterable[Any]]) -> Any:
        ...

    def executescript(self, script: str) -> Any:
        ...
