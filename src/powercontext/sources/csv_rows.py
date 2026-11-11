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

"""Capture offline CSV records as independently addressable Sources."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization


class CSVDocumentInput(BaseModel):
    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    @property
    def source_id(self) -> str:
        return self.name

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be nonempty without surrounding whitespace")  # noqa: TRY003
    if not primary_key_value.strip():
        raise ValueError("primary key value must not be blank")  # noqa: TRY003
    identity = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > 256:
        raise ValueError("CSV row identity exceeds 256 characters")  # noqa: TRY003
    return identity


def parse_csv_document(document: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    if not document.document_id or document.document_id != document.document_id.strip():
        raise ValueError("document_id must be nonempty without surrounding whitespace")  # noqa: TRY003
    rows: list[CSVRowInput] = []
    seen: set[str] = set()
    try:
        records = csv.reader(io.StringIO(document.content.decode("utf-8-sig"), newline=""), strict=True)
        columns = tuple(next(records, ()))
        if not columns or any(not column.strip() for column in columns) or len(set(columns)) != len(columns):
            raise ValueError("CSV requires unique nonblank headers")  # noqa: TRY003
        key_index = columns.index(document.primary_key)
        for record in records:
            if not record:
                continue
            if len(record) != len(columns):
                raise ValueError("CSV row width differs from the header")  # noqa: TRY003
            key = record[key_index]
            csv_row_source_id(document.document_id, key)
            if key in seen:
                raise ValueError("CSV primary keys must be unique")  # noqa: TRY003
            seen.add(key)
            rows.append(
                CSVRowInput(
                    document_id=document.document_id,
                    primary_key=document.primary_key,
                    primary_key_value=key,
                    columns=columns,
                    values=tuple(record),
                )
            )
    except csv.Error as error:
        raise ValueError("invalid CSV quoting") from error  # noqa: TRY003
    return tuple(rows)


class _CSVRowAdapter:
    name = "csv-row"
    input_class = CSVRowInput
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(name=identity, materialization=SourceMaterialization.CAPTURED, **value.model_dump())

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput.model_validate(source.model_dump())


class _CSVRowProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type="csv-row",
            source_id=source.name,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(_CSVRowAdapter(), projections=(_CSVRowProjection(),))
