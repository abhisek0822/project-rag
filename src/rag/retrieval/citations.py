"""Validation for immutable source labels emitted by a generator."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .models import GenerationResult

_INLINE_CITATION_RE = re.compile(r"\[S(\d+)\]")
_SOURCE_ID_RE = re.compile(r"^\[?S(\d+)\]?$", flags=re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CitationValidation:
    cited_source_ids: tuple[str, ...]
    invalid_source_ids: tuple[str, ...]
    has_required_citation: bool

    @property
    def valid(self) -> bool:
        return not self.invalid_source_ids and self.has_required_citation


def validate_citations(
    result: GenerationResult,
    allowed_source_ids: Iterable[str],
    *,
    require_citation: bool = True,
) -> CitationValidation:
    allowed = tuple(allowed_source_ids)
    allowed_set = set(allowed)
    inline = tuple(f"S{match}" for match in _INLINE_CITATION_RE.findall(result.answer))
    declared = tuple(_normalize_source_id(value) for value in result.cited_source_ids)
    all_references = _ordered_unique((*inline, *declared))
    invalid = tuple(value for value in all_references if value not in allowed_set)
    valid_inline = tuple(value for value in _ordered_unique(inline) if value in allowed_set)

    # An explicit structured list alone is not a visible citation.  Grounded
    # non-abstaining answers need at least one label in their answer text.
    needs_one = require_citation and not result.abstained
    has_required = bool(valid_inline) or not needs_one
    return CitationValidation(
        cited_source_ids=valid_inline,
        invalid_source_ids=invalid,
        has_required_citation=has_required,
    )


def _normalize_source_id(value: str) -> str:
    match = _SOURCE_ID_RE.match(str(value).strip())
    if not match:
        return str(value).strip()
    return f"S{int(match.group(1))}"


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
