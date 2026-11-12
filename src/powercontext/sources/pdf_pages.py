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

"""Capture complete physical PDF pages and retrieve their exact Memory citations."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

if TYPE_CHECKING:
    from powercontext.artifacts import MemoryCitation
    from powercontext.builtin.runtime.application import ScopedMemoryApplication
    from powercontext.builtin.runtime.models import MemoryEntryRecord

_SOURCE_NAME = "pdf-page"


class _PDFDocumentIdentity(BaseModel):
    document_id: str

    @field_validator("document_id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be blank")  # noqa: TRY003
        return value


class PDFDocumentInput(_PDFDocumentIdentity):
    content: bytes


class PDFPageInput(_PDFDocumentIdentity):
    document_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_number: int = Field(strict=True, gt=0)
    text: str


class PDFPageSource(Source, PDFPageInput):
    """A captured page of one exact PDF revision."""

    @property
    def source_id(self) -> str:
        return self.name


_LIGATURES = str.maketrans({chr(code): unicodedata.normalize("NFKC", chr(code)) for code in range(0xFB00, 0xFB07)})


def _normalize_pdf_text(text: str) -> str:
    # Keep table numbers and ordinary hyphens; these replacements only undo
    # typographic ligatures and discretionary hyphenation in the text layer.
    text = re.sub(r"(?<=\w)\u00ad[ \t]*\r?\n[ \t]*(?=\w)", "", text)
    return unicodedata.normalize("NFC", text.translate(_LIGATURES).replace("\u00ad", "")).strip()


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract every physical page before publishing any results."""
    from pypdf import PdfReader

    if not value.content.startswith(b"%PDF-"):
        raise ValueError("expected a PDF document")  # noqa: TRY003
    digest = hashlib.sha256(value.content).hexdigest()
    # pypdf tries the empty password automatically, including owner-locked files.
    # Decoder errors can originate in individual page streams or fonts.
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=True)
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=digest,
                page_number=number,
                text=_normalize_pdf_text(page.extract_text() or ""),
            )
            for number, page in enumerate(reader.pages, start=1)
        )
    except Exception as error:
        raise ValueError("cannot completely parse PDF") from error  # noqa: TRY003
    if not pages:
        raise ValueError("PDF contains no physical pages")  # noqa: TRY003
    return pages


class _PageAdapter:
    name = _SOURCE_NAME
    input_class = PDFPageInput
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        identity = json.dumps(
            [value.document_id, value.document_sha256, value.page_number],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return PDFPageSource(
            name="pdf:" + hashlib.sha256(identity.encode("ascii")).hexdigest(),
            materialization=SourceMaterialization.CAPTURED,
            **value.model_dump(),
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput.model_validate(source.model_dump())


class _PageProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=_SOURCE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(_PageAdapter(), projections=(_PageProjection(),))

_TERMS = re.compile(r"[a-z]+|\d+")
_STOP = frozenset(
    "a about according all also an and any are as at be been but by can calculate could define did do does each "  # noqa: SIM905
    "during end fiscal for form from give had has have how if in into is it its many may million millions most "
    "of on only or our please provide reported response should show shown state than that the their them these "
    "they this those to up us usd use used using was were what when where whether which who will with would year".split()
)
# Financial questions often use terminology that differs from the report's row or section heading.
_INCOME_STATEMENT = (
    "income statement",
    "statement of income",
    "statements of income",
    "statement of earnings",
    "statements of earnings",
    "statement of operations",
    "statement of profit or loss",
    "statements of operations",
    "profit and loss",
    "p l statement",
)
_CASH_FLOW_STATEMENT = (
    "cash flow statement",
    "statement of cash flow",
    "statements of cash flow",
)
_BALANCE_SHEET = ("balance sheet", "statement of financial position", "statements of financial position")
_CONCEPTS = (
    _INCOME_STATEMENT,
    _CASH_FLOW_STATEMENT,
    _BALANCE_SHEET,
    ("capital expenditure", "capex", "capital spending", "purchase of property", "addition to property"),
    ("depreciation and amortization", "depreciation amortization", "d a"),
    ("operating income", "operating profit", "income from operation", "earnings from operation"),
    ("net income", "net earning", "net profit"),
    ("revenue", "net sales", "net revenue", "turnover"),
    (
        "geography",
        "geographic",
        "geographic region",
        "country",
        "international operation",
        "principal market",
        "geographic information",
        "description of business",
    ),
    ("business segment", "operating segment", "reportable segment", "segment information"),
    ("legal battle", "legal proceeding", "legal matter", "litigation", "lawsuit", "legal contingency"),
    ("restructuring cost", "restructuring charge", "restructuring expense", "restructuring and impairment"),
    ("share repurchase", "stock repurchase", "share buyback", "treasury stock"),
    ("working capital", "current asset", "current liability"),
    ("research and development", "r d", "research development"),
    ("inventory", "inventories", "merchandise inventory"),
    ("account receivable", "trade receivable", "receivable"),
    ("account payable", "trade payable", "payable"),
    ("interest expense", "interest cost", "finance cost"),
    ("income tax", "tax expense", "tax provision", "effective tax rate"),
    ("dividend", "distribution to shareholder"),
    ("goodwill impairment", "impairment of goodwill", "goodwill"),
    ("debt maturity", "maturity of debt", "scheduled maturity"),
    ("gross margin", "gross profit", "cost of sale", "cost of revenue", "cost of good sold"),
    ("free cash flow", "cash from operation", "net cash provided by operating activities"),
    ("selling general and administrative", "selling and administrative", "sg a", "s g a"),
)


def _terms(text: str) -> list[str]:
    terms = _TERMS.findall(text.casefold())
    return [
        term[:-3] + "y"
        if len(term) > 4 and term.endswith("ies")
        else term[:-1]
        if len(term) > 4 and term.endswith("s") and not term.endswith(("ss", "us", "is"))
        else term
        for term in terms
    ]


# These metrics often require inputs from multiple statements even when
# the question names only the metric.
_METRIC_STATEMENTS = (
    (("ebitda", "capital intensity"), (_INCOME_STATEMENT, _CASH_FLOW_STATEMENT)),
    (("free cash flow",), (_CASH_FLOW_STATEMENT,)),
    (("current ratio", "quick ratio", "debt to equity", "net debt", "working capital"), (_BALANCE_SHEET,)),
    (
        (
            "asset turnover",
            "return on asset",
            "return on equity",
            "days sales outstanding",
            "days payable outstanding",
            "inventory turnover",
            "cash conversion cycle",
        ),
        (_INCOME_STATEMENT, _BALANCE_SHEET),
    ),
)


def _phrase(phrase: str, text: str) -> bool:
    return f" {phrase} " in text


@dataclass(slots=True)
class _SearchPage:
    record: MemoryEntryRecord
    source_ids: tuple[str, ...]
    document: str
    terms: list[str]
    headings: tuple[str, ...]


def _page_candidates(records: tuple[MemoryEntryRecord, ...]) -> list[_SearchPage]:
    pages = []
    for record in records:
        ids = tuple(ref.source_id for ref in record.entry.sources if ref.source_type == _SOURCE_NAME)
        if not ids:
            continue
        document, _, text = record.entry.text.partition("\n")
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and line.strip().casefold() not in {"table of contents", "contents"}
        ]
        headings = tuple(
            " " + " ".join(_terms(line)) + " " if len(line) <= 100 and not line.endswith(".") else ""
            for line in lines[:6]
        )
        pages.append(_SearchPage(record, ids, document, _terms(text), headings))
    return pages


def _rank(pages: list[_SearchPage], query: list[str], native: dict[str, int]) -> list[_SearchPage]:
    words = set(query) - _STOP
    question = " " + " ".join(query) + " "
    relevant = set()
    for metrics, statements in _METRIC_STATEMENTS:
        if any(_phrase(" ".join(_terms(metric)), question) for metric in metrics):
            relevant.update(statements)
    concepts = []
    for group in _CONCEPTS:
        aliases = tuple(" ".join(_terms(alias)) for alias in group)
        if group in relevant or any(_phrase(alias, question) for alias in aliases):
            concepts.append(aliases)
    document_words = {page.document: set(_terms(page.document)) for page in pages}
    years = {word for word in words if word.isdecimal() and 1900 <= int(word) <= 2099}
    boilerplate = {"annual", "report", "form", "fy", "k", "q", "quarter"}
    companies = {
        document: len((terms - _STOP - boilerplate) & {word for word in words if word.isalpha()})
        for document, terms in document_words.items()
    }
    best_company = max(companies.values(), default=0)
    best_year = any(companies[doc] == best_company and terms & years for doc, terms in document_words.items())
    document_preference = {
        doc: (2.0 if companies[doc] == best_company else 0.55) * (1.5 if not best_year or terms & years else 0.72)
        for doc, terms in document_words.items()
    }
    title_terms = set().union(*document_words.values())
    counters = [Counter(page.terms) for page in pages]
    frequency = Counter(word for counts in counters for word in words & counts.keys())
    average = max(1.0, sum(len(page.terms) for page in pages) / len(pages))
    weights = {
        word: math.log1p((len(pages) - frequency[word] + 0.5) / (frequency[word] + 0.5))
        * (0.15 if word in years or word in title_terms else 1.0)
        for word in words
    }
    pairs = {f"{a} {b}" for a, b in pairwise(query) if a in words - years and b in words - years}
    base = []
    signals = []
    for page, counts in zip(pages, counters, strict=True):
        body = " " + " ".join(page.terms) + " "
        norm = 0.65 + 0.55 * len(page.terms) / average
        lexical = math.fsum(weights[word] * counts[word] / (counts[word] + norm) for word in words)
        lexical += 1.2 * sum(_phrase(pair, body) for pair in pairs)
        lexical += 0.4 / (1 + native.get(page.record.entry.entry_id, 999))
        matches = []
        for aliases in concepts:
            matched = any(_phrase(alias, body) for alias in aliases)
            # Short titles identify the statement itself. A passing reference
            # in a table of contents or audit opinion is weaker evidence.
            heading = any(
                len(line.split()) <= len(alias.split()) + 5 and not line.split()[-1].isdigit() and _phrase(alias, line)
                for alias in aliases
                for line in page.headings
                if line.strip()
            )
            matches.append(3.0 if heading else 1.0 if matched else 0.0)
        preference = document_preference[page.document]
        base.append((lexical + 1.6 * sum(matches)) * preference if page.terms else -1.0)
        signals.append((matches, lexical, preference))

    # Reserve room for distinct parts of the question. Ranking only by the
    # whole question lets pages that mention many metrics crowd out the
    # original statement that provides just one of the required inputs.
    overall = sorted(range(len(pages)), key=lambda index: -base[index])
    lanes = [
        sorted(
            (i for i, (matches, _, _) in enumerate(signals) if matches[concept]),
            key=lambda i: (
                -signals[i][0][concept] * signals[i][2],
                -signals[i][1] * signals[i][2],
            ),
        )
        for concept in range(len(concepts))
    ]
    order = dict.fromkeys(overall[:3])
    for depth in range(2):
        order.update(dict.fromkeys(lane[depth] for lane in lanes if len(lane) > depth))
    order.update(dict.fromkeys(overall))
    return [pages[index] for index in order]


def _requested_documents(pages: list[_SearchPage], question: str) -> list[_SearchPage]:
    """Resolve explicit report identifiers before ranking pages across reports."""
    query = set(_terms(question))
    ignored = _STOP | {"annual", "report", "form", "fy", "k", "q", "inc", "corp", "corporation", "company"}
    documents = {page.document: set(_terms(page.document)) for page in pages}
    matches = {
        document: len({term for term in terms & query - ignored if term.isalpha() and len(term) > 1})
        for document, terms in documents.items()
    }
    best = max(matches.values(), default=0)
    if not best:
        return pages
    selected = {document for document, count in matches.items() if count == best}
    years = {term for term in query if term.isdecimal() and len(term) == 4}
    dated = {document for document in selected if documents[document] & years}
    if dated:
        selected = dated
    return [page for page in pages if page.document in selected]


async def search_pdf_pages(
    memory: ScopedMemoryApplication, question: str, limit: int = 10
) -> tuple[MemoryCitation, ...]:
    """Rank pages in the current scoped Memory and return their stored citations."""
    from powercontext.builtin.runtime.models import SearchMemoryRequest

    query = _terms(question)
    if limit <= 0 or not query:
        return ()
    keywords = " ".join(dict.fromkeys(word for word in query if word not in _STOP)) or " ".join(query)
    native_page = await memory.search(SearchMemoryRequest(query=keywords, mode="fts", limit=limit))
    native = {hit.entry_id: rank for rank, hit in enumerate(native_page.hits)}
    # Strict FTS can miss paraphrased questions and pages containing just one
    # component of a calculation. Rescore the live page entries within this scope.
    pages = _page_candidates((await memory.list()).entries)
    if not pages:
        return ()
    seen: set[str] = set()
    citations = []
    for page in _rank(_requested_documents(pages, question), query, native):
        if seen.intersection(page.source_ids):
            continue
        seen.update(page.source_ids)
        citations.append(page.record.citation)
        if len(citations) == limit:
            break
    return tuple(citations)
