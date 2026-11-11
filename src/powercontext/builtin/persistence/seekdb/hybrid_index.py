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

"""Single-statement candidate retrieval for the embedded seekdb backend."""

from __future__ import annotations

from sqlalchemy import column, literal_column, select, union_all
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.artifacts.memory import EmbeddingProfile, MemorySearchChannels, MemorySearchRequest
from powercontext.builtin.persistence.memory_index import CompositeMemoryIndex, memory_channel_hits
from powercontext.builtin.persistence.oceanbase.memory_index import OceanBaseMemoryFTSIndex, OceanBaseMemoryVectorIndex

_FIELDS = ("memory_artifact_id", "head_revision", "entry_id", "entry_version_id", "text")


class SeekDBHybridMemoryIndex(CompositeMemoryIndex):
    """Use the same native channel queries and fuse their candidates as before."""

    def __init__(self, profile: EmbeddingProfile) -> None:
        self._fts = OceanBaseMemoryFTSIndex()
        self._vector = OceanBaseMemoryVectorIndex(profile)
        super().__init__(self._fts, self._vector)

    async def search(
        self, connection: AsyncConnection, scope_id: str, request: MemorySearchRequest, /
    ) -> MemorySearchChannels:
        if request.mode != "hybrid":
            return await super().search(connection, scope_id, request)

        parameters = await self._vector.candidate_parameters(connection, scope_id, request)
        fts = self._fts.candidates(scope_id, request, include_score=True).subquery("fts_candidates")
        vector = (
            self._vector
            .approximate_candidates(tagged=request.tag_filter is not None)
            .columns(*(column(field) for field in (*_FIELDS, "distance")))
            .subquery("vector_candidates")
        )
        channels = union_all(
            select(
                *(fts.c[field] for field in _FIELDS),
                literal_column("0").label("channel"),
                (-fts.c.score).label("sort_score"),
                literal_column("NULL").label("distance"),
            ),
            select(
                *(vector.c[field] for field in _FIELDS),
                literal_column("1").label("channel"),
                vector.c.distance.label("sort_score"),
                vector.c.distance,
            ),
        ).subquery("channels")
        statement = select(channels).order_by(
            channels.c.channel,
            channels.c.sort_score,
            channels.c.memory_artifact_id,
            channels.c.entry_id,
            channels.c.entry_version_id,
        )
        rows = (await connection.execute(statement, parameters)).mappings().all()
        return MemorySearchChannels(
            fts=memory_channel_hits((row for row in rows if row["channel"] == 0), request.memories),
            vector=memory_channel_hits((row for row in rows if row["channel"] == 1), request.memories),
        )
