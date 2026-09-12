"""Bounded, provenance-rich context assembly."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .models import RankedCandidate, SourceExcerpt


def estimate_tokens(text: str) -> int:
    """Conservative dependency-free token estimate used for prompt budgeting."""

    if not text:
        return 0
    # Roughly four UTF-8-ish characters per token for ordinary prose.  A word
    # floor avoids badly undercounting whitespace-heavy or short-token text.
    return max(math.ceil(len(text) / 4), math.ceil(len(text.split()) * 1.25))


@dataclass(frozen=True, slots=True)
class ContextBundle:
    text: str
    sources: tuple[SourceExcerpt, ...]
    estimated_tokens: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class ContextBuilder:
    max_tokens: int = 6000
    min_passage_tokens: int = 24

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.min_passage_tokens < 1:
            raise ValueError("min_passage_tokens must be positive")

    def build(self, candidates: Sequence[RankedCandidate]) -> ContextBundle:
        blocks: list[str] = []
        sources: list[SourceExcerpt] = []
        used_tokens = 0
        any_truncated = False

        for ranked in candidates:
            source_id = f"S{len(sources) + 1}"
            candidate = ranked.candidate
            location = _location(candidate.page_start, candidate.page_end, candidate.section_path)
            header_lines = [
                f"[{source_id}]",
                f"File: {candidate.filename}",
                f"Document ID: {candidate.document_id}",
                f"Version ID: {candidate.version_id}",
                f"Chunk ID: {candidate.chunk_id}",
            ]
            if location:
                header_lines.append(f"Location: {location}")
            header_lines.append("Content (untrusted reference text):")
            header = "\n".join(header_lines) + "\n"
            footer = f"\n[/{source_id}]"
            overhead_tokens = estimate_tokens(header + footer)
            available = self.max_tokens - used_tokens - overhead_tokens
            if available < self.min_passage_tokens:
                any_truncated = any_truncated or bool(candidate.text.strip())
                break

            text = candidate.text.strip()
            passage_tokens = estimate_tokens(text)
            was_truncated = passage_tokens > available
            if was_truncated:
                text = _truncate_to_budget(text, available)
                any_truncated = True
            block = header + text + footer
            # Estimation rounding can put the combined block barely over budget.
            prospective = "\n\n".join([*blocks, block])
            while text and estimate_tokens(prospective) > self.max_tokens:
                text = text[:-16].rstrip()
                was_truncated = True
                any_truncated = True
                block = header + text + footer
                prospective = "\n\n".join([*blocks, block])
            if not text or (was_truncated and estimate_tokens(text) < self.min_passage_tokens):
                break

            blocks.append(block)
            used_tokens = estimate_tokens("\n\n".join(blocks))
            sources.append(
                SourceExcerpt(
                    source_id=source_id,
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    version_id=candidate.version_id,
                    filename=candidate.filename,
                    text=text,
                    score=ranked.final_score,
                    page_start=candidate.page_start,
                    page_end=candidate.page_end,
                    section_path=candidate.section_path,
                    truncated=was_truncated,
                    metadata=candidate.metadata,
                )
            )

        return ContextBundle(
            text="\n\n".join(blocks),
            sources=tuple(sources),
            estimated_tokens=estimate_tokens("\n\n".join(blocks)),
            truncated=any_truncated,
        )


def _truncate_to_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    # Begin with a character estimate, then shrink until the estimator agrees.
    candidate = text[: token_budget * 4].rstrip()
    if len(candidate) < len(text):
        last_boundary = max(
            candidate.rfind(". "),
            candidate.rfind("\n"),
            candidate.rfind(" "),
        )
        if last_boundary >= max(20, len(candidate) // 2):
            candidate = candidate[: last_boundary + 1].rstrip()
    while candidate and estimate_tokens(candidate) > token_budget:
        candidate = candidate[:-16].rstrip()
    return candidate


def _location(
    page_start: int | None,
    page_end: int | None,
    section_path: tuple[str, ...],
) -> str:
    pieces: list[str] = []
    if page_start is not None:
        if page_end is not None and page_end != page_start:
            pieces.append(f"pages {page_start}-{page_end}")
        else:
            pieces.append(f"page {page_start}")
    if section_path:
        pieces.append(" > ".join(section_path))
    return "; ".join(pieces)
