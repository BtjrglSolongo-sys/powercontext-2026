# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persist scoped evidence dependencies and apply Source availability events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import insert, select, update

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_TABLE
from powercontext.builtin.records import BaseValueNotFoundError
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

    from powercontext.builtin.persistence.database import AsyncDatabase
    from powercontext.builtin.runtime.application import BuiltinRuntime

_Policy = Literal["all", "any"]
_Reference = SourceRef | MemoryCitation | ArtifactRef


class SourceLifecycleError(ValueError):
    """Invalid lifecycle registration, event, or resolution."""


class SourceLifecycleEvent(BaseModel):
    """One ordered availability change for a registered Source."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    node_id: str
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace text")  # noqa: TRY003
        return value


class _Node(BaseModel):
    node_id: str
    reference: _Reference
    revision: int | None = None
    parents: tuple[str, ...] = ()
    policy: _Policy = "all"


class _Graph(BaseModel):
    # Registration requires parents to exist, so persisted order is topological.
    nodes: list[_Node] = Field(default_factory=list)
    events: dict[str, SourceLifecycleEvent] = Field(default_factory=dict)

    def availability(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for node in self.nodes:
            if isinstance(node.reference, SourceRef):
                event = self.events.get(node.node_id)
                result[node.node_id] = event is None or event.available
            else:
                supported = (result[parent] for parent in node.parents)
                result[node.node_id] = all(supported) if node.policy == "all" else any(supported)
        return result


async def _load(connection: AsyncConnection, scope_id: str) -> _Graph | None:
    table = SOURCE_LIFECYCLE_TABLE
    payload = await connection.scalar(select(table.c.payload).where(table.c.scope_id == scope_id))
    return None if payload is None else _Graph.model_validate_json(payload)


async def unavailable_topic_refs(connection: AsyncConnection, scope_id: str) -> frozenset[tuple[str, int]]:
    """Return exact Topic revisions excluded from live retrieval."""
    graph = await _load(connection, scope_id)
    if graph is None:
        return frozenset()
    states = graph.availability()
    active, inactive = set(), set()
    for node in graph.nodes:
        if isinstance(node.reference, ArtifactRef):
            identity = (node.reference.artifact_id, node.reference.revision)
            (active if states[node.node_id] else inactive).add(identity)
    return frozenset(inactive - active)


class SourceLifecycleApplication:
    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        if self._database is None:
            raise SourceLifecycleError("lifecycle persistence is unavailable")  # noqa: TRY003
        return ScopedSourceLifecycle(self._runtime, self._database, validate_scope_id(scope_id))


class ScopedSourceLifecycle:
    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self._scope_id = scope_id

    async def register_source(self, node_id: str, reference: SourceRef, *, revision: int) -> None:
        if not isinstance(reference, SourceRef) or type(revision) is not int or revision < 1:
            raise SourceLifecycleError("invalid Source reference or revision")  # noqa: TRY003
        await self._register(_Node(node_id=node_id, reference=reference, revision=revision))

    async def register_memory(
        self, node_id: str, reference: MemoryCitation, *, parents: tuple[str, ...], policy: _Policy
    ) -> None:
        if not isinstance(reference, MemoryCitation):
            raise SourceLifecycleError("expected a Memory citation")  # noqa: TRY003
        await self._register_derived(node_id, reference, parents, policy)

    async def register_topic(
        self, node_id: str, reference: ArtifactRef, *, parents: tuple[str, ...], policy: _Policy
    ) -> None:
        if not isinstance(reference, ArtifactRef) or reference.family != "topic-memory":
            raise SourceLifecycleError("expected a Topic Memory reference")  # noqa: TRY003
        await self._register_derived(node_id, reference, parents, policy)

    async def _register_derived(
        self, node_id: str, reference: MemoryCitation | ArtifactRef, parents: tuple[str, ...], policy: _Policy
    ) -> None:
        if policy not in ("all", "any") or not parents or any(not isinstance(parent, str) for parent in parents):
            raise SourceLifecycleError("invalid lifecycle parents or policy")  # noqa: TRY003
        # Parent order has no meaning under either policy.
        await self._register(_Node(node_id=node_id, reference=reference, parents=tuple(sorted(parents)), policy=policy))

    async def _register(self, node: _Node) -> None:
        if not node.node_id.strip():
            raise SourceLifecycleError("node_id must contain non-whitespace text")  # noqa: TRY003
        async with self._runtime._scope_operation(self._scope_id), self._runtime._locked(self._scope_id):
            graph = await self._read()
            nodes = {} if graph is None else {item.node_id: item for item in graph.nodes}
            existing = nodes.get(node.node_id)
            if existing is not None:
                if existing != node:
                    raise SourceLifecycleError("node already has a different binding")  # noqa: TRY003
                return
            if len(set(node.parents)) != len(node.parents) or any(parent not in nodes for parent in node.parents):
                raise SourceLifecycleError("parents must be distinct previously registered nodes")  # noqa: TRY003
            await self._validate_reference(node.reference)
            candidate = graph or _Graph()
            candidate.nodes.append(node)
            if isinstance(node.reference, MemoryCitation):
                await self._sync_memory(candidate, candidate.availability())
            await self._write(candidate, existing=graph is not None)

    async def _validate_reference(self, reference: _Reference) -> None:
        records = self._runtime.records.for_scope(self._scope_id)
        try:
            if isinstance(reference, SourceRef):
                await records.get_source(reference.source_type, reference.source_id)
            elif isinstance(reference, MemoryCitation):
                await self._runtime.memory.for_scope(self._scope_id).get(GetMemoryEntryRequest(citation=reference))
            else:
                await records.get_artifact_revision(reference.family, reference.artifact_id, reference.revision)
        except (BaseValueNotFoundError, LookupError, ValueError) as error:
            raise SourceLifecycleError("registered reference is not readable") from error  # noqa: TRY003

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        async with self._runtime._scope_operation(self._scope_id), self._runtime._locked(self._scope_id):
            graph = await self._read() or _Graph()
            node = next((node for node in graph.nodes if node.node_id == event.node_id), None)
            if node is None or not isinstance(node.reference, SourceRef) or node.revision != event.revision:
                raise SourceLifecycleError("unknown Source node or mismatched revision")  # noqa: TRY003
            previous = graph.events.get(event.node_id)
            if previous == event:
                return sorted(node_id for node_id, active in graph.availability().items() if active)
            if previous is not None and event.sequence <= previous.sequence:
                raise SourceLifecycleError("stale or conflicting lifecycle event")  # noqa: TRY003
            graph.events[event.node_id] = event
            states = graph.availability()
            # Publish only after native writes succeed. A retry compares the
            # desired states with the current manifest, including partial writes.
            await self._sync_memory(graph, states)
            await self._write(graph, existing=True)
            return sorted(node_id for node_id, active in states.items() if active)

    async def resolve(self, node_id: str, *, purpose: Literal["generation", "audit"]) -> _Reference:
        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError("unknown lifecycle resolution purpose")  # noqa: TRY003
        graph = await self._read() or _Graph()
        for node in graph.nodes:
            if node.node_id == node_id:
                if purpose == "generation" and not graph.availability()[node_id]:
                    raise SourceLifecycleError("node is unavailable for generation")  # noqa: TRY003
                return node.reference
        raise SourceLifecycleError("unknown lifecycle node")  # noqa: TRY003

    async def _sync_memory(self, graph: _Graph, states: dict[str, bool]) -> None:
        desired: dict[str, tuple[str, bool]] = {}
        for node in graph.nodes:
            if isinstance(node.reference, MemoryCitation):
                entry_id = node.reference.entry_id
                version = node.reference.entry_version_id
                previous = desired.get(entry_id)
                if previous is not None and previous[0] != version:
                    raise SourceLifecycleError("registered Memory versions conflict")  # noqa: TRY003
                desired[entry_id] = (version, states[node.node_id] or (previous is not None and previous[1]))
        if desired:
            await self._runtime.memory.for_scope(self._scope_id)._sync_source_lifecycle(desired)

    async def _read(self) -> _Graph | None:
        async with self._database.transaction() as connection:
            return await _load(connection, self._scope_id)

    async def _write(self, graph: _Graph, *, existing: bool) -> None:
        table = SOURCE_LIFECYCLE_TABLE
        payload = graph.model_dump_json().encode("utf-8")
        async with self._database.transaction() as connection:
            if existing:
                await connection.execute(
                    update(table).where(table.c.scope_id == self._scope_id).values(payload=payload)
                )
            else:
                await connection.execute(insert(table).values(scope_id=self._scope_id, payload=payload))
