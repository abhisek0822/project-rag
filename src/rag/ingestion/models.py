"""Provider-neutral data contracts for parsed blocks and searchable chunks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

BLOCK_KINDS = frozenset({"heading", "paragraph", "list", "table", "code"})


@dataclass(frozen=True)
class Block:
    """An ordered, structural fragment emitted by a document parser.

    ``page`` is one-based when a source has pages. ``source_locator`` is kept
    parser-specific on purpose (for example, a PDF bounding box or DOCX
    paragraph index), while the other fields remain portable.
    """

    kind: str
    text: str
    order: int
    page: int | None = None
    section_path: tuple[str, ...] = ()
    source_locator: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in BLOCK_KINDS:
            raise ValueError(
                f"Unsupported block kind {self.kind!r}; expected one of {sorted(BLOCK_KINDS)}"
            )
        if self.order < 0:
            raise ValueError("Block order must be non-negative")
        if self.page is not None and self.page < 1:
            raise ValueError("Block page numbers are one-based")


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit with source text, embedding text, and provenance."""

    id: str
    version_id: str
    ordinal: int
    text: str
    embedding_text: str
    token_count: int
    content_hash: str
    block_type: str
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...]
    source_locator: Mapping[str, Any]

    @property
    def chunk_id(self) -> str:
        """Compatibility alias useful to vector-index adapters."""

        return self.id
