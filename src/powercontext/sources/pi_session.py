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

"""Capture the visible, uncompressed history of a Pi v3 session branch."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field, JsonValue

from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.models import Source, SourceMaterialization

_SOURCE_NAME = "pi-event"


class PiSessionInput(BaseModel):
    content: bytes
    leaf_id: str | None = None


class _EventFields(BaseModel):
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventInput(_EventFields):
    """A complete visible event and its original Pi record."""


class PiEventSource(Source, _EventFields):
    """An immutable captured event addressed by session and event identity."""

    @property
    def source_id(self) -> str:
        return self.name


class _Checkpoint(BaseModel):
    sessions: dict[str, dict[str, str]] = Field(default_factory=dict)


def _json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(session_id: str, event_id: str) -> str:
    return "pi:" + hashlib.sha256(_json([session_id, event_id]).encode("utf-8")).hexdigest()


def _text(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _body(content: JsonValue, *, tool_calls: bool = False) -> str:
    if isinstance(content, str):
        return content
    blocks: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                blocks.append(_text(block["text"]))
            elif tool_calls and block.get("type") == "toolCall":
                name, call_id = _text(block.get("name")), _text(block.get("id"))
                blocks.append(f"[tool_call {name} {call_id}] {_json(block.get('arguments', {}))}")
    return "\n".join(blocks)


def _visible(record: dict[str, JsonValue], event_type: str) -> tuple[str, str]:
    if event_type in {"compaction", "branch_summary"}:
        return event_type, _text(record.get("summary"))
    if event_type == "custom_message":
        return "custom", _body(record.get("content"))
    message = record.get("message")
    if event_type != "message" or not isinstance(message, dict):
        return "", ""
    role = _text(message.get("role"))
    if role == "bashExecution":
        command, output = _text(message.get("command")), _text(message.get("output"))
        return role, f"$ {command}\n{output}" if command or output else ""
    body = _body(message.get("content"), tool_calls=role != "toolResult")
    if role == "toolResult" and body:
        name, call_id = _text(message.get("toolName")), _text(message.get("toolCallId"))
        body = f"[tool_result {name} {call_id}] {body}"
    return role, body


def _load_record(line: str) -> dict[str, JsonValue]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")  # noqa: TRY003

    def unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate Pi JSON object key")  # noqa: TRY003
        return result

    record = json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(record, dict):
        raise ValueError("Pi JSONL records must be objects")  # noqa: TRY003, TRY004
    # Reject data that cannot be retained losslessly in a Source projection.
    _json(record).encode("utf-8")
    return record


def _parse(document: PiSessionInput) -> tuple[str, dict[str, dict[str, JsonValue]], tuple[PiEventInput, ...]]:
    lines = (line for line in document.content.decode("utf-8-sig").split("\n") if line.strip())
    header = _load_record(next(lines, "{}"))
    session_id = _text(header.get("id"))
    if (
        header.get("type") != "session"
        or type(header.get("version")) is not int
        or header["version"] != 3
        or not session_id.strip()
    ):
        raise ValueError("expected a Pi v3 session header")  # noqa: TRY003

    records: dict[str, dict[str, JsonValue]] = {}
    parents: dict[str, str | None] = {}
    for line in lines:
        record = _load_record(line)
        event_id = _text(record.get("id"))
        event_type = _text(record.get("type"))
        if not event_id.strip() or event_id in records or not event_type.strip() or "parentId" not in record:
            raise ValueError("invalid or duplicate Pi event")  # noqa: TRY003
        parent = record["parentId"]
        if parent is not None and (not isinstance(parent, str) or parent not in records):
            raise ValueError("Pi event parent must precede the event")  # noqa: TRY003
        records[event_id] = record
        parents[event_id] = parent

    leaf = document.leaf_id
    if leaf is not None and leaf not in records:
        raise ValueError("Pi session leaf does not exist")  # noqa: TRY003
    if leaf is None:
        leaf = next(reversed(records), None)
    events: list[PiEventInput] = []
    while leaf is not None:
        record, parent = records[leaf], parents[leaf]
        event_type = _text(record["type"])
        role, body = _visible(record, event_type)
        if role and body.strip():
            events.append(
                PiEventInput(
                    session_id=session_id,
                    event_id=leaf,
                    parent_id=parent,
                    event_type=event_type,
                    role=role,
                    content=f"{role}: {body}",
                    record=record,
                )
            )
        leaf = parent
    return session_id, records, tuple(reversed(events))


def parse_pi_session(document: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Return the original visible events on the selected path, in history order."""
    return _parse(document)[2]


class _PiEventAdapter:
    name = _SOURCE_NAME
    input_class = PiEventInput
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=_identity(value.session_id, value.event_id),
            materialization=SourceMaterialization.CAPTURED,
            **value.model_dump(),
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput.model_validate(source.model_dump())


class _PiEventProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(_PiEventAdapter(), projections=(_PiEventProjection(),))


def _digest(record: dict[str, JsonValue]) -> str:
    return hashlib.sha256(_json(record).encode("utf-8")).hexdigest()


class PiSessionConnector:
    """Import unseen visible events with checkpoint commits owned by ConnectorLifecycle."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        content = await asyncio.to_thread(self._path.read_bytes)
        session_id, records, events = _parse(PiSessionInput(content=content, leaf_id=self._leaf_id))
        checkpoint = _Checkpoint() if session.checkpoint is None else _Checkpoint.model_validate(session.checkpoint)
        imported = checkpoint.sessions.setdefault(session_id, {})
        # Check previously imported sibling branches before writing any new events.
        for event_id, expected in imported.items():
            if event_id not in records or _digest(records[event_id]) != expected:
                raise ValueError(f"imported Pi event changed: {event_id}")  # noqa: TRY003
        for event in events:
            if event.event_id not in imported:
                await session.submit(_identity(session_id, event.event_id), _SOURCE_NAME, event)
                imported[event.event_id] = _digest(event.record)
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint.model_dump())
