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

"""Wire format for incremental context transfers."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

_PACK = "powercontext.context-pack.v1"
_DELTA = "powercontext.context-pack-delta.v1"


class ContextPackError(ValueError):
    """The supplied context pack, delta, or chunk size is invalid."""


def _wire(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode(items: object) -> tuple[bytes, ...]:
    if not isinstance(items, list):
        raise ContextPackError("expected a Base64 array")  # noqa: TRY003
    try:
        if any(not isinstance(item, str) for item in items):
            raise ValueError("Base64 items must be strings")  # noqa: TRY003, TRY301
        decoded = tuple(base64.b64decode(item, validate=True) for item in items)
        if _encode(decoded) != items:
            raise ValueError("noncanonical Base64 item")  # noqa: TRY003, TRY301
    except ValueError as error:
        raise ContextPackError("invalid Base64 item") from error  # noqa: TRY003
    return decoded


def _object(data: bytes, schema: str, fields: set[str]) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ContextPackError("duplicate wire field")  # noqa: TRY003
        return result

    try:
        payload = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    except (ValueError, RecursionError) as error:
        raise ContextPackError("invalid context JSON") from error  # noqa: TRY003
    if not isinstance(payload, dict) or set(payload) != fields | {"schema"} or payload["schema"] != schema:
        raise ContextPackError("invalid context schema")  # noqa: TRY003
    return payload


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


@dataclass(frozen=True, slots=True)
class ContextPack:
    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        if not isinstance(items, tuple) or any(not isinstance(item, bytes) for item in items):
            raise ContextPackError("pack items must be a tuple of bytes")  # noqa: TRY003
        payload = {"schema": _PACK, "items": _encode(items)}
        digest = "sha256:" + hashlib.sha256(_wire(payload)).hexdigest()
        # The byte count is part of the wire format. Iterate until its digit width is stable.
        payload.update(digest=digest, byte_count=0)
        while (size := len(_wire(payload))) != payload["byte_count"]:
            payload["byte_count"] = size
        return cls(items, digest, size)

    @classmethod
    def from_content(cls, content: str | None, *, chunk_bytes: int = 512) -> ContextPack:
        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        try:
            data = (content or "").encode("utf-8")
        except UnicodeError as error:
            raise ContextPackError("content is not valid UTF-8") from error  # noqa: TRY003
        items = []
        start = 0
        while start < len(data):
            end = min(start + chunk_bytes, len(data))
            while end < len(data) and end > start and data[end] & 0xC0 == 0x80:
                end -= 1
            if end == start:
                raise ContextPackError("chunk size cannot hold the next UTF-8 character")  # noqa: TRY003
            items.append(data[start:end])
            start = end
        return cls.build(tuple(items))

    def to_bytes(self) -> bytes:
        return _wire({
            "schema": _PACK,
            "items": _encode(self.items),
            "digest": self.digest,
            "byte_count": self.byte_count,
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        payload = _object(data, _PACK, {"items", "digest", "byte_count"})
        pack = cls.build(_decode(payload["items"]))
        if (
            payload["digest"] != pack.digest
            or type(payload["byte_count"]) is not int
            or payload["byte_count"] != pack.byte_count
        ):
            raise ContextPackError("pack digest or byte count does not match")  # noqa: TRY003
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        for previous, current in zip(base.items, target.items, strict=False):
            if previous != current:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _wire({
            "schema": _DELTA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encode(self.appended),
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        payload = _object(data, _DELTA, {"base_digest", "target_digest", "prefix_items", "appended"})
        base, target, prefix = payload["base_digest"], payload["target_digest"], payload["prefix_items"]
        if not _valid_digest(base) or not _valid_digest(target) or type(prefix) is not int or prefix < 0:
            raise ContextPackError("invalid delta digest or prefix")  # noqa: TRY003
        return cls(base, target, prefix, _decode(payload["appended"]))


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    if base != ContextPack.build(base.items) or delta.base_digest != base.digest:
        raise ContextPackError("base digest does not match")  # noqa: TRY003
    prefix = delta.prefix_items
    if type(prefix) is not int or not 0 <= prefix <= len(base.items):
        raise ContextPackError("delta prefix is out of bounds")  # noqa: TRY003
    if not isinstance(delta.appended, tuple):
        raise ContextPackError("appended items must be a tuple")  # noqa: TRY003
    target = ContextPack.build(base.items[:prefix] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("target digest does not match")  # noqa: TRY003
    return target
