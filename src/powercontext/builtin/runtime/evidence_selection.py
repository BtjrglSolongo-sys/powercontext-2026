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

"""Rank literal sentence spans for the prepared-context byte budget."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from powercontext.builtin.artifacts.search import analyze_fts_query, analyze_text

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])\s*|\n+")
_WORD = re.compile(r"\S+")
_STOP = frozenset([
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "of",
    "on",
    "or",
    "our",
    "should",
    "than",
    "that",
    "the",
    "their",
    "them",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "which",
    "while",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
])


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    document: int
    start: int
    end: int


def _stem(term: str) -> str:
    # Match ordinary inflections without imposing an English tokenizer on other scripts.
    if not term.isascii() or not term.isalpha() or len(term) < 5:
        return term
    if term.endswith("ies"):
        return term[:-3] + "y"
    if term.endswith(("sses", "shes", "ches", "xes", "zes")):
        return term[:-2]
    if term.endswith("s") and not term.endswith(("ss", "us", "is")):
        term = term[:-1]
    for suffix in ("ing", "ed", "ly"):
        if len(term) - len(suffix) >= 4 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def _sentences(text: str, document: int) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    start = 0
    for boundary in _BOUNDARY.finditer(text):
        if text[start : boundary.start()].strip():
            spans.append(EvidenceSpan(document, start, boundary.start()))
        start = boundary.end()
    if text[start:].strip():
        spans.append(EvidenceSpan(document, start, len(text)))
    return spans


def _rank(query: str, documents: Sequence[str]) -> list[tuple[float, EvidenceSpan]]:
    raw_query = set(analyze_fts_query(query).split())
    terms = {_stem(term) for term in (raw_query - _STOP or raw_query)}
    tokenized = [
        [
            (span, [_stem(term) for term in analyze_text(text[span.start : span.end]).split()])
            for span in _sentences(text, index)
        ]
        for index, text in enumerate(documents)
    ]
    vocabularies = [{term for _, sentence in document for term in sentence} for document in tokenized]
    frequency = Counter(term for words in vocabularies for term in words & terms)
    weights = {term: 1 + math.log((len(documents) + 1) / (frequency[term] + 1)) for term in terms}

    ranked = []
    for index, sentences in enumerate(tokenized):
        document_score = math.fsum(weights[term] for term in terms & vocabularies[index])
        scores = [
            math.fsum(weights[term] for term in terms.intersection(words)) / (1 + 0.012 * max(0, len(words) - 24))
            for _, words in sentences
        ]
        peak = max(scores, default=0.0)
        for position, ((span, _), score) in enumerate(zip(sentences, scores, strict=True)):
            neighbors = scores[max(0, position - 1) : position + 2]
            priority = score + 0.20 * max(neighbors) + 0.15 * peak + 0.12 * document_score
            ranked.append((priority, span))
    return sorted(ranked, key=lambda item: (-item[0], item[1].document, item[1].start))


def _merge(
    documents: Sequence[str], selected: tuple[EvidenceSpan, ...], added: EvidenceSpan
) -> tuple[EvidenceSpan, ...]:
    result: list[EvidenceSpan] = []
    for span in sorted((*selected, added), key=lambda item: (item.document, item.start)):
        if result and span.document == result[-1].document:
            previous = result[-1]
            gap = documents[span.document][previous.end : span.start]
            if not gap.strip():
                result[-1] = EvidenceSpan(span.document, previous.start, max(previous.end, span.end))
                continue
        result.append(span)
    return tuple(result)


def _shorten(query: str, text: str, span: EvidenceSpan, fits: Callable[[EvidenceSpan], bool]) -> EvidenceSpan | None:
    words = tuple(_WORD.finditer(text, span.start, span.end))
    if not words:
        return None
    query_terms = {_stem(term) for term in analyze_fts_query(query).split()} - _STOP
    anchor = next(
        (
            index
            for index, word in enumerate(words)
            if query_terms.intersection(_stem(term) for term in analyze_text(word[0]).split())
        ),
        0,
    )
    # Grow a literal window around a matched word, keeping both sides when possible.
    low, high = 1, len(words)
    result = None
    while low <= high:
        size = (low + high) // 2
        first = min(max(0, anchor - size // 3), len(words) - size)
        trial = EvidenceSpan(span.document, words[first].start(), words[first + size - 1].end())
        if fits(trial):
            result = trial
            low = size + 1
        else:
            high = size - 1
    if result is not None:
        return result

    # A long token or text without spaces can still supply a valid character span.
    start = words[anchor].start()
    low, high = start + 1, span.end
    while low <= high:
        end = (low + high) // 2
        trial = EvidenceSpan(span.document, start, end)
        if fits(trial):
            result = trial
            low = end + 1
        else:
            high = end - 1
    return result


def select_evidence(
    query: str,
    documents: Sequence[str],
    fits: Callable[[tuple[EvidenceSpan, ...]], bool],
    *,
    limit: int,
) -> tuple[EvidenceSpan, ...]:
    """Fit high-scoring sentences first and combine adjacent excerpts to share citation overhead."""
    selected: tuple[EvidenceSpan, ...] = ()
    for _, span in _rank(query, documents):
        candidate = _merge(documents, selected, span)
        if len(candidate) <= limit and fits(candidate):
            selected = candidate
        elif not selected:
            # A single long sentence should not force an otherwise useful context to be empty.
            excerpt = _shorten(query, documents[span.document], span, lambda item: fits((item,)))
            if excerpt is not None:
                selected = (excerpt,)
    return selected
