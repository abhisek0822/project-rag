"""Ports consumed by retrieval and generation business logic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .models import (
    GenerationRequest,
    GenerationResult,
    RetrievalFilters,
    SearchCandidate,
    Vector,
)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Produces compatible vectors for both indexed text and user queries."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int | None: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Ingestion-friendly alias for :meth:`embed_documents`."""
        ...

    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class RetrievalRepository(Protocol):
    """Database-facing search port.

    Implementations must scope both methods by ``filters`` in the SQL/query
    itself and only expose READY chunks from current document versions.
    """

    async def dense_search(
        self,
        *,
        query_embedding: Vector,
        limit: int,
        filters: RetrievalFilters,
    ) -> Sequence[SearchCandidate]: ...

    async def lexical_search(
        self,
        *,
        query: str,
        limit: int,
        filters: RetrievalFilters,
    ) -> Sequence[SearchCandidate]: ...


@runtime_checkable
class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    async def rerank(self, query: str, candidates: Sequence[SearchCandidate]) -> Sequence[float]:
        """Return one relevance score per candidate, in the same order."""
        ...


@runtime_checkable
class Generator(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...
