"""Small, provider-neutral data objects used by the retrieval pipeline.

The rest of the application may use SQLAlchemy or Pydantic objects at its
boundaries, but the retrieval core intentionally uses plain dataclasses.  This
keeps ranking and generation easy to test without a database or web server.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Vector = Sequence[float]


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """Security and metadata scope applied *inside* every repository search.

    ``tenant_id`` is deliberately required.  A repository implementation must
    apply it before vector or lexical ranking; filtering results afterwards can
    leak cross-tenant data and also produces incorrect top-k results.
    """

    tenant_id: str
    collection_id: str | None = None
    document_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    created_after: str | None = None
    created_before: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id is required for retrieval")


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """One searchable chunk returned by either repository search method."""

    chunk_id: str
    document_id: str
    version_id: str
    filename: str
    text: str
    score: float
    page_start: int | None = None
    page_end: int | None = None
    section_path: tuple[str, ...] = ()
    content_hash: str | None = None
    embedding: tuple[float, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id is required")
        if not self.document_id:
            raise ValueError("document_id is required")
        if not self.version_id:
            raise ValueError("version_id is required")
        if not self.text.strip():
            raise ValueError("candidate text cannot be empty")

    @property
    def section_key(self) -> tuple[str, tuple[str, ...]]:
        return self.document_id, self.section_path


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """A candidate plus explainable scores produced by our ranking stages."""

    candidate: SearchCandidate
    fusion_score: float
    final_score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.candidate.chunk_id


@dataclass(frozen=True, slots=True)
class SourceExcerpt:
    """A labelled piece of evidence that was actually placed in the prompt."""

    source_id: str
    chunk_id: str
    document_id: str
    version_id: str
    filename: str
    text: str
    score: float
    page_start: int | None = None
    page_end: int | None = None
    section_path: tuple[str, ...] = ()
    truncated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"[{self.source_id}]"

    @property
    def location(self) -> str | None:
        parts: list[str] = []
        if self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                parts.append(f"pages {self.page_start}-{self.page_end}")
            else:
                parts.append(f"page {self.page_start}")
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        return "; ".join(parts) or None


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    question: str
    context: str
    sources: tuple[SourceExcerpt, ...]
    conversation: tuple[ChatMessage, ...] = ()
    validation_feedback: str | None = None

    @property
    def allowed_source_ids(self) -> tuple[str, ...]:
        return tuple(source.source_id for source in self.sources)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    answer: str
    cited_source_ids: tuple[str, ...] = ()
    confidence: float | None = None
    missing_information: tuple[str, ...] = ()
    abstained: bool = False
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    normalized_query: str
    embedding_provider: str
    dense_candidates: int = 0
    lexical_candidates: int = 0
    fused_candidates: int = 0
    deduplicated_candidates: int = 0
    selected_candidates: int = 0
    context_estimated_tokens: int = 0
    context_truncated: bool = False
    reranker_used: bool = False
    diversity_used: bool = False
    invalid_citations: tuple[str, ...] = ()
    generation_attempts: int = 0
    warnings: tuple[str, ...] = ()
    timings_ms: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    question: str
    candidates: tuple[RankedCandidate, ...]
    diagnostics: RetrievalDiagnostics


@dataclass(frozen=True, slots=True)
class AnswerResult:
    question: str
    answer: str
    sources: tuple[SourceExcerpt, ...]
    cited_source_ids: tuple[str, ...]
    confidence: float | None
    missing_information: tuple[str, ...]
    abstained: bool
    abstention_reason: str | None
    diagnostics: RetrievalDiagnostics
