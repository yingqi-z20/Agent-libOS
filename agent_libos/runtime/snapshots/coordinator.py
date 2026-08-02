from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol, TypeVar

from agent_libos.models.snapshot import ProcessSnapshot
from agent_libos.runtime.snapshots.codec import SnapshotCodec
PreparedT = TypeVar("PreparedT")
PublishedT = TypeVar("PublishedT")


class SnapshotTransactionPort(Protocol):
    """Shared transaction/lock boundary required by snapshot orchestration."""

    def locked(self) -> AbstractContextManager[None]: ...

    def transaction(
        self,
        *,
        include_object_payloads: bool = False,
    ) -> AbstractContextManager[Any]: ...


class SnapshotCoordinator:
    """Coordinates typed snapshot capture and atomic state publication.

    The manager supplies domain-specific discovery and row mutation callbacks;
    this service owns the invariant-sensitive ordering around validation,
    the store transaction and typed publication boundary. Authority-gated
    callers wrap this method in their domain AuthorityTransaction so
    reauthorization, publication, evidence, and settlement share one UoW.
    """

    def __init__(
        self,
        transaction_port: SnapshotTransactionPort,
        *,
        codec: type[SnapshotCodec] = SnapshotCodec,
    ):
        self._transaction_port = transaction_port
        self._codec = codec

    def decode(self, snapshot: Mapping[str, Any]) -> ProcessSnapshot:
        return self._codec.decode_mapping(snapshot)

    def encode(self, snapshot: ProcessSnapshot) -> dict[str, Any]:
        return self._codec.encode_mapping(snapshot)

    def normalize(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        _typed, normalized = self.canonicalize(snapshot)
        return normalized

    def canonicalize(
        self,
        snapshot: Mapping[str, Any],
    ) -> tuple[ProcessSnapshot, dict[str, Any]]:
        return self._codec.canonicalize_mapping(snapshot)

    @contextmanager
    def capture_scope(self) -> Iterator[None]:
        """Hold the one transaction used by discovery and checkpoint publish."""
        with self._transaction_port.transaction(include_object_payloads=True):
            yield

    @contextmanager
    def restore_runtime_scope(
        self,
        runtime_quiescence: Callable[[], Any],
    ) -> Iterator[None]:
        """Keep scheduler quanta stopped through post-commit finalizers."""
        with runtime_quiescence():
            yield

    @contextmanager
    def restore_registry_scope(
        self,
        registry_quiescence: Callable[[], Any],
    ) -> Iterator[None]:
        """Hold the registry lock around atomic publish and reconciliation."""
        with registry_quiescence():
            yield

    @contextmanager
    def restore_atomic_scope(
        self,
        ownership_quiescence: Callable[[], Any],
    ) -> Iterator[None]:
        """Acquire ownership before the store lock for restore preflight."""
        with ownership_quiescence():
            with self._transaction_port.locked():
                yield

    def atomic_publish(
        self,
        snapshot: Mapping[str, Any] | ProcessSnapshot,
        *,
        prepare: Callable[[ProcessSnapshot], PreparedT],
        publish: Callable[[ProcessSnapshot, PreparedT], PublishedT],
    ) -> tuple[PreparedT, PublishedT]:
        """Prepare and publish once in one payload-aware store transaction."""
        typed = (
            snapshot
            if isinstance(snapshot, ProcessSnapshot)
            else self.decode(snapshot)
        )
        with self._transaction_port.transaction(include_object_payloads=True):
            prepared = prepare(typed)
            published = publish(typed, prepared)
        return prepared, published


__all__ = ["SnapshotCoordinator"]
