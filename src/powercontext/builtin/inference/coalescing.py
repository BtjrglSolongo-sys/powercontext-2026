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

"""Share concurrent embedding work and cache successful vectors."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel, QueryEmbeddingModel


@dataclass(eq=False, frozen=True, slots=True)
class _BatchUsage:
    usage: InferenceUsage


_ResolvedVector: TypeAlias = tuple[EmbeddingVector, _BatchUsage]
_VectorFuture: TypeAlias = asyncio.Future[_ResolvedVector]
_VectorCache: TypeAlias = OrderedDict[tuple[bool, str], EmbeddingVector]
_Embed: TypeAlias = Callable[[tuple[str, ...]], Awaitable[EmbeddingResult]]


def _sum_tokens(values: list[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _observe_result(future: _VectorFuture) -> None:
    # All callers may cancel before a shared batch fails.
    if not future.cancelled():
        future.exception()


class _EmbeddingQueue:
    def __init__(
        self, embed: _Embed, batch_size: int, delay: float, cache_size: int, cache: _VectorCache, *, query: bool
    ) -> None:
        self._embed = embed
        self._batch_size = batch_size
        self._delay = delay
        self._cache_size = cache_size
        self._cache = cache
        self._query = query
        self._inflight: dict[str, _VectorFuture] = {}
        self._pending: dict[str, _VectorFuture] = {}
        self._workers: set[asyncio.Task[None]] = set()
        self._timer: asyncio.TimerHandle | None = None

    def _lookup(self, text: str, loop: asyncio.AbstractEventLoop) -> EmbeddingVector | _VectorFuture:
        key = (self._query, text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        future = self._inflight.get(text)
        if future is None:
            future = loop.create_future()
            future.add_done_callback(_observe_result)
            self._pending[text] = self._inflight[text] = future
        return future

    async def embed(self, texts: tuple[str, ...]) -> EmbeddingResult:
        loop = asyncio.get_running_loop()
        slots = [self._lookup(text, loop) for text in texts]
        if self._pending and self._timer is None:
            self._timer = loop.call_later(self._delay, self._flush, loop)

        waiting = tuple(dict.fromkeys(slot for slot in slots if isinstance(slot, asyncio.Future)))
        resolved = await asyncio.gather(*(asyncio.shield(future) for future in waiting))
        by_future = dict(zip(waiting, resolved, strict=True))
        batches = dict.fromkeys(batch for _, batch in resolved)
        usage = [batch.usage for batch in batches]
        return EmbeddingResult(
            vectors=tuple(by_future[slot][0] if isinstance(slot, asyncio.Future) else slot for slot in slots),
            usage=InferenceUsage(
                requests=sum(item.requests for item in usage),
                input_tokens=_sum_tokens([item.input_tokens for item in usage]),
                output_tokens=_sum_tokens([item.output_tokens for item in usage]),
            ),
        )

    def _flush(self, loop: asyncio.AbstractEventLoop) -> None:
        self._timer = None
        pending = tuple(self._pending.items())
        self._pending.clear()
        for start in range(0, len(pending), self._batch_size):
            worker = loop.create_task(self._run(pending[start : start + self._batch_size]))
            self._workers.add(worker)
            worker.add_done_callback(self._workers.discard)

    async def _run(self, items: tuple[tuple[str, _VectorFuture], ...]) -> None:
        try:
            result = await self._embed(tuple(text for text, _ in items))
            self._publish(items, result)
        except asyncio.CancelledError:
            for _, future in items:
                future.cancel()
            raise
        except Exception as error:
            for _, future in items:
                future.set_exception(error)
        finally:
            for text, _ in items:
                del self._inflight[text]

    def _publish(self, items: tuple[tuple[str, _VectorFuture], ...], result: EmbeddingResult) -> None:
        if len(result.vectors) != len(items):
            raise ValueError("provider returned the wrong number of embedding vectors")  # noqa: TRY003
        batch = _BatchUsage(result.usage)
        for (text, future), vector in zip(items, result.vectors, strict=True):
            if self._cache_size:
                self._cache[self._query, text] = vector
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            future.set_result((vector, batch))


class CoalescingEmbeddingModel:
    """Batch and deduplicate calls to a shared model within one event loop."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if not math.isfinite(coalesce_ms) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be finite and nonnegative")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")  # noqa: TRY003
        self.profile = model.profile
        delay = coalesce_ms / 1000
        cache: _VectorCache = OrderedDict()
        self._documents = _EmbeddingQueue(model.embed, max_batch_size, delay, cache_size, cache, query=False)
        # A provider may embed the same string differently as a query.
        self._queries = (
            _EmbeddingQueue(model.embed_query, max_batch_size, delay, cache_size, cache, query=True)
            if isinstance(model, QueryEmbeddingModel)
            else self._documents
        )

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        return await self._documents.embed(texts)

    async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        return await self._queries.embed(texts)
