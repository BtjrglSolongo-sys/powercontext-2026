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

"""Rerank Memory hits using an HTTP scoring service."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score batches independently and rank the original candidates globally."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(self, query: str, candidates: tuple[MemoryHit, ...], limit: int) -> MemoryRerankDecision:
        if type(limit) is not int or not 1 <= limit <= len(candidates):
            raise ValueError("limit must be between 1 and the candidate count")  # noqa: TRY003

        scores: dict[int, int | float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = range(start + 1, start + 1 + len(batch))
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [{"rank": rank, "text": hit.text} for rank, hit in zip(ranks, batch, strict=True)],
                },
            )
            requests += 1
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or set(payload) != {"scores"} or not isinstance(payload["scores"], list):
                raise ValueError("invalid reranker response")  # noqa: TRY003
            for item in payload["scores"]:
                if not isinstance(item, dict) or set(item) != {"rank", "score"}:
                    raise ValueError("invalid reranker score record")  # noqa: TRY003
                rank, score = item["rank"], item["score"]
                if type(rank) is not int or rank not in ranks or rank in scores:
                    raise ValueError("invalid or repeated reranker rank")  # noqa: TRY003
                if type(score) not in (int, float) or (isinstance(score, float) and not math.isfinite(score)):
                    raise ValueError("reranker score must be finite")  # noqa: TRY003
                scores[rank] = score
            if any(rank not in scores for rank in ranks):
                raise ValueError("reranker response omitted a rank")  # noqa: TRY003

        selected = sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit]
        return MemoryRerankDecision(tuple(selected), InferenceUsage(requests=requests))
