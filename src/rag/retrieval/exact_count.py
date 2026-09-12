"""Deterministic whole-source counting for explicit word or phrase count questions.

Semantic retrieval is intentionally optimized for finding relevant passages, not
for corpus-wide aggregation. Chunks can overlap, so counting matches in retrieved
chunks can both miss occurrences and count the same source text twice. This module
operates on the parser's original, non-overlapping blocks instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.ingestion.models import Block

_COUNT_INTENT_RE = re.compile(
    r"\b(?:how\s+many\s+times|how\s+often|count|number\s+of\s+occurrences?)\b",
    flags=re.IGNORECASE,
)
_QUOTED_TARGET_PATTERNS = (
    re.compile(r'"([^"\n]{1,200})"'),
    re.compile(r"“([^”\n]{1,200})”"),
    re.compile(r"'([^'\n]{1,200})'"),
    re.compile(r"‘([^’\n]{1,200})’"),
    re.compile(r"`([^`\n]{1,200})`"),
)
_UNQUOTED_TARGET_PATTERNS = (
    re.compile(
        r"\b(?:word|term|phrase)\s+(?:called\s+|named\s+)?"
        r"([A-Za-z0-9][\w.+#/-]{0,99})\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\boccurrences?\s+of\s+([A-Za-z0-9][\w.+#/-]{0,99})\b",
        flags=re.IGNORECASE,
    ),
)
_CASE_SENSITIVE_RE = re.compile(
    r"\b(?:case[- ]sensitive|matching\s+case|exact\s+case)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ExactCountRequest:
    term: str
    case_sensitive: bool = False

    @property
    def label(self) -> str:
        return "phrase" if any(character.isspace() for character in self.term) else "word"


@dataclass(frozen=True, slots=True)
class CountableDocument:
    document_id: str
    version_id: str
    filename: str
    blocks: tuple[Block, ...]


@dataclass(frozen=True, slots=True)
class ExactCountEvidence:
    document_id: str
    version_id: str
    filename: str
    block_order: int
    text: str
    occurrence_count: int
    page: int | None
    section_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExactDocumentCount:
    document_id: str
    version_id: str
    filename: str
    count: int
    evidence: tuple[ExactCountEvidence, ...]


@dataclass(frozen=True, slots=True)
class ExactCountResult:
    request: ExactCountRequest
    documents: tuple[ExactDocumentCount, ...]

    @property
    def total(self) -> int:
        return sum(document.count for document in self.documents)

    @property
    def evidence(self) -> tuple[ExactCountEvidence, ...]:
        # Keep responses inspectable without returning hundreds of source cards.
        # Up to eight blocks per document gives small documents a complete audit trail.
        candidates = [document.evidence[:8] for document in self.documents if document.evidence]
        selected: list[ExactCountEvidence] = []
        for offset in range(8):
            for group in candidates:
                if offset < len(group):
                    selected.append(group[offset])
                if len(selected) >= 20:
                    return tuple(selected)
        return tuple(selected)


def parse_exact_count_request(question: str) -> ExactCountRequest | None:
    """Recognize an unambiguous request to count a quoted word or phrase."""

    normalized = " ".join(question.split())
    if not normalized or _COUNT_INTENT_RE.search(normalized) is None:
        return None

    target: str | None = None
    for pattern in _QUOTED_TARGET_PATTERNS:
        match = pattern.search(normalized)
        if match:
            target = match.group(1)
            break
    if target is None:
        for pattern in _UNQUOTED_TARGET_PATTERNS:
            match = pattern.search(normalized)
            if match:
                target = match.group(1)
                break

    if target is None:
        return None
    target = " ".join(target.split()).strip()
    if not target:
        return None
    return ExactCountRequest(
        term=target,
        case_sensitive=_CASE_SENSITIVE_RE.search(normalized) is not None,
    )


def count_exact_occurrences(
    request: ExactCountRequest,
    documents: tuple[CountableDocument, ...],
) -> ExactCountResult:
    """Count against every normalized parser block exactly once."""

    pattern = _literal_pattern(request)
    document_counts: list[ExactDocumentCount] = []
    for document in documents:
        total = 0
        evidence: list[ExactCountEvidence] = []
        for block in sorted(document.blocks, key=lambda item: item.order):
            matches = tuple(pattern.finditer(block.text))
            if not matches:
                continue
            total += len(matches)
            evidence.append(
                ExactCountEvidence(
                    document_id=document.document_id,
                    version_id=document.version_id,
                    filename=document.filename,
                    block_order=block.order,
                    text=_evidence_excerpt(block.text, matches),
                    occurrence_count=len(matches),
                    page=block.page,
                    section_path=block.section_path,
                )
            )
        document_counts.append(
            ExactDocumentCount(
                document_id=document.document_id,
                version_id=document.version_id,
                filename=document.filename,
                count=total,
                evidence=tuple(evidence),
            )
        )
    return ExactCountResult(request=request, documents=tuple(document_counts))


def format_exact_count_answer(result: ExactCountResult) -> str:
    """Render a concise result with labels matching ``result.evidence`` order."""

    casing = "case-sensitive" if result.request.case_sensitive else "case-insensitive"
    match_kind = "whole-word" if result.request.label == "word" else "literal-phrase"
    match_mode = f"{casing} exact {match_kind} match"
    source_numbers: dict[str, list[int]] = {}
    for number, evidence in enumerate(result.evidence, start=1):
        source_numbers.setdefault(evidence.document_id, []).append(number)

    if len(result.documents) == 1:
        document = result.documents[0]
        citations = _citation_text(source_numbers.get(document.document_id, []))
        times = "time" if document.count == 1 else "times"
        return (
            f'The {result.request.label} “{result.request.term}” appears {document.count} '
            f"{times} in {document.filename} ({match_mode}).{citations}"
        )

    times = "time" if result.total == 1 else "times"
    details: list[str] = []
    for document in result.documents:
        citations = _citation_text(source_numbers.get(document.document_id, [])).strip()
        detail = f"{document.filename}: {document.count}"
        if citations:
            detail += f" {citations}"
        details.append(detail)
    breakdown = "; ".join(details)
    return (
        f'The {result.request.label} “{result.request.term}” appears {result.total} {times} '
        f"across {len(result.documents)} selected documents ({match_mode}). "
        f"Breakdown: {breakdown}."
    )


def _literal_pattern(request: ExactCountRequest) -> re.Pattern[str]:
    parts = re.split(r"\s+", request.term)
    literal = r"\s+".join(re.escape(part) for part in parts)
    if request.term[0].isalnum() or request.term[0] == "_":
        literal = r"(?<!\w)" + literal
    if request.term[-1].isalnum() or request.term[-1] == "_":
        literal += r"(?!\w)"
    flags = 0 if request.case_sensitive else re.IGNORECASE
    return re.compile(literal, flags=flags)


def _evidence_excerpt(
    text: str,
    matches: tuple[re.Match[str], ...],
    *,
    maximum_characters: int = 1600,
) -> str:
    cleaned = text.strip()
    if len(cleaned) <= maximum_characters:
        return cleaned
    first = matches[0]
    half_window = maximum_characters // 2
    start = max(0, first.start() - half_window)
    end = min(len(text), start + maximum_characters)
    if end - start < maximum_characters:
        start = max(0, end - maximum_characters)
    excerpt = text[start:end].strip()
    if start:
        excerpt = "…" + excerpt
    if end < len(text):
        excerpt += "…"
    return excerpt


def _citation_text(numbers: list[int]) -> str:
    if not numbers:
        return ""
    return " " + " ".join(f"[S{number}]" for number in numbers)
