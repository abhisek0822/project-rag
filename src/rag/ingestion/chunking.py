"""Structure-aware, token-aware chunk construction."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .errors import ChunkingError
from .identifiers import content_hash, deterministic_chunk_id
from .models import Block, Chunk

TokenCounter = Callable[[str], int]


def approximate_token_count(text: str) -> int:
    """A deterministic fallback counter for deployments without a tokenizer.

    Provider adapters can inject their model's tokenizer through ``Chunker``.
    The fallback counts word-like spans and punctuation separately, making it a
    safer approximation than a simple character ratio.
    """

    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


@dataclass(frozen=True)
class _Piece:
    text: str
    kind: str
    page: int | None
    section_path: tuple[str, ...]
    block_order: int
    locator: Mapping[str, Any]
    is_overlap: bool = False


class Chunker:
    """Split ordered blocks while respecting structure and a hard token cap."""

    def __init__(
        self,
        *,
        target_tokens: int = 500,
        max_tokens: int = 750,
        min_tokens: int = 100,
        overlap_tokens: int = 75,
        token_counter: TokenCounter = approximate_token_count,
    ) -> None:
        if not 0 < min_tokens <= target_tokens <= max_tokens:
            raise ValueError("Expected 0 < min_tokens <= target_tokens <= max_tokens")
        if not 0 <= overlap_tokens < target_tokens:
            raise ValueError("overlap_tokens must be non-negative and below target_tokens")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.count_tokens = token_counter

    def chunk(
        self,
        blocks: Sequence[Block],
        version_id: str,
        document_title: str = "",
    ) -> list[Chunk]:
        if not version_id:
            raise ChunkingError("version_id is required for deterministic chunk IDs")
        ordered = [
            block for block in sorted(blocks, key=lambda item: item.order) if block.text.strip()
        ]
        if not ordered:
            return []

        groups: list[list[Block]] = []
        for block in ordered:
            if not groups or groups[-1][-1].section_path != block.section_path:
                groups.append([block])
            else:
                groups[-1].append(block)

        drafts: list[list[_Piece]] = []
        for group in groups:
            pieces: list[_Piece] = []
            for block in group:
                pieces.extend(self._split_block(block))
            drafts.extend(self._pack_group(pieces))

        chunks: list[Chunk] = []
        for ordinal, pieces in enumerate(drafts):
            text = self._join_pieces(pieces)
            token_count = self.count_tokens(text)
            if token_count <= 0:
                continue
            if token_count > self.max_tokens:
                raise ChunkingError(
                    f"Chunk {ordinal} contains {token_count} tokens, above hard maximum "
                    f"{self.max_tokens}"
                )

            section_path = pieces[0].section_path
            pages = [piece.page for piece in pieces if piece.page is not None]
            kinds = {piece.kind for piece in pieces}
            block_type = next(iter(kinds)) if len(kinds) == 1 else "mixed"
            chunk_hash = content_hash(text)
            embedding_text = self._embedding_text(document_title, section_path, text)
            chunks.append(
                Chunk(
                    id=deterministic_chunk_id(version_id, ordinal, chunk_hash),
                    version_id=version_id,
                    ordinal=ordinal,
                    text=text,
                    embedding_text=embedding_text,
                    token_count=token_count,
                    content_hash=chunk_hash,
                    block_type=block_type,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    section_path=section_path,
                    source_locator={"blocks": self._provenance(pieces)},
                )
            )
        return chunks

    def chunk_blocks(
        self,
        blocks: Sequence[Block],
        version_id: str,
        document_title: str = "",
    ) -> list[Chunk]:
        """Readable alias for callers that prefer an explicit method name."""

        return self.chunk(blocks, version_id, document_title)

    def _split_block(self, block: Block) -> list[_Piece]:
        if self.count_tokens(block.text) <= self.target_tokens:
            return [
                _Piece(
                    text=block.text.strip(),
                    kind=block.kind,
                    page=block.page,
                    section_path=block.section_path,
                    block_order=block.order,
                    locator=block.source_locator,
                )
            ]

        if block.kind in {"code", "table", "list"}:
            raw_units = [line for line in block.text.splitlines() if line.strip()]
            joiner = "\n"
        else:
            raw_units = [
                unit for unit in re.split(r"(?<=[.!?])\s+(?=\S)", block.text) if unit.strip()
            ]
            joiner = " "

        texts = self._pack_text_units(raw_units, joiner, self.target_tokens)
        pieces: list[_Piece] = []
        for split_part, text in enumerate(texts):
            locator = dict(block.source_locator)
            locator["split_part"] = split_part
            pieces.append(
                _Piece(
                    text=text,
                    kind=block.kind,
                    page=block.page,
                    section_path=block.section_path,
                    block_order=block.order,
                    locator=locator,
                )
            )
        return pieces

    def _pack_text_units(self, units: Sequence[str], joiner: str, limit: int) -> list[str]:
        atomic: list[str] = []
        for unit in units:
            cleaned = unit.strip("\n") if joiner == "\n" else unit.strip()
            if not cleaned:
                continue
            if self.count_tokens(cleaned) <= limit:
                atomic.append(cleaned)
            else:
                atomic.extend(self._hard_split(cleaned, limit))

        packed: list[str] = []
        current = ""
        for unit in atomic:
            proposed = unit if not current else f"{current}{joiner}{unit}"
            if current and self.count_tokens(proposed) > limit:
                packed.append(current)
                current = unit
            else:
                current = proposed
        if current:
            packed.append(current)
        return packed

    def _hard_split(self, text: str, limit: int) -> list[str]:
        wordish = re.findall(r"\S+(?:\s+|$)", text)
        parts: list[str] = []
        current = ""
        for unit in wordish:
            if self.count_tokens(unit) > limit:
                if current.strip():
                    parts.append(current.strip())
                    current = ""
                parts.extend(self._split_long_span(unit.strip(), limit))
                continue
            proposed = f"{current}{unit}"
            if current and self.count_tokens(proposed) > limit:
                parts.append(current.strip())
                current = unit
            else:
                current = proposed
        if current.strip():
            parts.append(current.strip())
        return parts

    def _split_long_span(self, text: str, limit: int) -> list[str]:
        """Binary-search maximal character spans for unusually long tokens."""

        output: list[str] = []
        remaining = text
        while remaining:
            low, high = 1, len(remaining)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if self.count_tokens(remaining[:middle]) <= limit:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise ChunkingError("Token counter cannot split an oversized text span")
            output.append(remaining[:best])
            remaining = remaining[best:]
        return output

    def _pack_group(self, pieces: Sequence[_Piece]) -> list[list[_Piece]]:
        drafts: list[list[_Piece]] = []
        current: list[_Piece] = []
        index = 0
        while index < len(pieces):
            piece = pieces[index]
            if not current:
                current = [piece]
                index += 1
                continue

            proposed = self._join_pieces([*current, piece])
            proposed_tokens = self.count_tokens(proposed)
            current_tokens = self.count_tokens(self._join_pieces(current))
            if proposed_tokens <= self.target_tokens or (
                current_tokens < self.min_tokens and proposed_tokens <= self.max_tokens
            ):
                current.append(piece)
                index += 1
                continue

            drafts.append(current)
            overlap = self._overlap_tail(current)
            # In the unlikely case that an injected tokenizer makes the next
            # piece plus overlap exceed the hard maximum, provenance is more
            # important than forcing overlap.
            overlap_with_piece = self.count_tokens(self._join_pieces([*overlap, piece]))
            if overlap and overlap_with_piece <= self.max_tokens:
                current = overlap
            else:
                current = []

        if current:
            drafts.append(current)
        return drafts

    def _overlap_tail(self, pieces: Sequence[_Piece]) -> list[_Piece]:
        if self.overlap_tokens == 0:
            return []
        remaining = self.overlap_tokens
        tail: list[_Piece] = []
        for piece in reversed(pieces):
            piece_tokens = self.count_tokens(piece.text)
            if piece_tokens <= remaining:
                tail.append(replace(piece, is_overlap=True))
                remaining -= piece_tokens
            else:
                suffix = self._suffix(piece.text, remaining)
                if suffix:
                    tail.append(replace(piece, text=suffix, is_overlap=True))
                remaining = 0
            if remaining <= 0:
                break
        tail.reverse()
        return tail

    def _suffix(self, text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        units = re.findall(r"\S+(?:\s+|$)", text)
        selected: list[str] = []
        for unit in reversed(units):
            proposed = "".join([unit, *selected]).strip()
            if self.count_tokens(proposed) > limit:
                break
            selected.insert(0, unit)
        if selected:
            return "".join(selected).strip()

        # A single lexical unit may itself comprise many provider tokens.
        low, high = 1, len(text)
        best = ""
        while low <= high:
            size = (low + high) // 2
            candidate = text[-size:]
            if self.count_tokens(candidate) <= limit:
                best = candidate
                low = size + 1
            else:
                high = size - 1
        return best.strip()

    @staticmethod
    def _join_pieces(pieces: Sequence[_Piece]) -> str:
        return "\n\n".join(piece.text.strip() for piece in pieces if piece.text.strip()).strip()

    @staticmethod
    def _embedding_text(document_title: str, section_path: tuple[str, ...], text: str) -> str:
        context: list[str] = []
        if document_title.strip():
            context.append(f"Document: {document_title.strip()}")
        if section_path:
            context.append(f"Section: {' > '.join(section_path)}")
        if context:
            return "\n".join(context) + "\n\n" + text
        return text

    @staticmethod
    def _provenance(pieces: Sequence[_Piece]) -> list[Mapping[str, Any]]:
        seen = set()
        provenance: list[Mapping[str, Any]] = []
        for piece in pieces:
            key = (piece.block_order, repr(sorted(piece.locator.items())))
            if key in seen:
                continue
            seen.add(key)
            provenance.append(
                {
                    "order": piece.block_order,
                    "page": piece.page,
                    "locator": dict(piece.locator),
                }
            )
        return provenance


def deduplicate_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Keep the first exact-content occurrence without changing its provenance.

    The main pipeline intentionally retains repeated source passages because
    their page citations can differ. Retrieval/index adapters may opt into this
    helper when a corpus's policy calls for exact-content deduplication.
    """

    seen = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        if chunk.content_hash in seen:
            continue
        seen.add(chunk.content_hash)
        unique.append(chunk)
    return unique
