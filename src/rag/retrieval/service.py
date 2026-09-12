"""End-to-end retrieval, ranking, context, and grounded answer orchestration."""

from __future__ import annotations

import asyncio
import math
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .citations import CitationValidation, validate_citations
from .context import ContextBuilder
from .diversity import select_diverse_candidates
from .fusion import (
    attach_rerank_scores,
    deduplicate_ranked_candidates,
    reciprocal_rank_fusion,
)
from .interfaces import EmbeddingProvider, Generator, Reranker, RetrievalRepository
from .models import (
    AnswerResult,
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    RetrievalDiagnostics,
    RetrievalFilters,
    RetrievalResult,
    SearchCandidate,
    SourceExcerpt,
)


class RetrievalError(RuntimeError):
    """Raised when no configured retrieval channel can be executed."""


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    dense_limit: int = 40
    lexical_limit: int = 40
    fusion_limit: int = 30
    source_limit: int = 8
    dense_weight: float = 0.65
    lexical_weight: float = 0.35
    rrf_rank_constant: float = 60.0
    rerank_weight: float = 0.70
    mmr_lambda: float = 0.75
    max_per_document: int = 3
    max_per_section: int = 2
    near_duplicate_threshold: float = 0.92
    max_context_tokens: int = 6000
    min_context_passage_tokens: int = 24
    minimum_top_score: float | None = None
    require_citations: bool = True
    retry_invalid_citations: bool = True

    def __post_init__(self) -> None:
        integer_limits = (
            self.dense_limit,
            self.lexical_limit,
            self.fusion_limit,
            self.source_limit,
            self.max_per_document,
            self.max_per_section,
            self.max_context_tokens,
            self.min_context_passage_tokens,
        )
        if any(value < 1 for value in integer_limits):
            raise ValueError("retrieval limits and budgets must be positive")
        for name, value in (
            ("rerank_weight", self.rerank_weight),
            ("mmr_lambda", self.mmr_lambda),
            ("near_duplicate_threshold", self.near_duplicate_threshold),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.dense_weight < 0 or self.lexical_weight < 0:
            raise ValueError("retrieval weights must be non-negative")
        if self.dense_weight + self.lexical_weight <= 0:
            raise ValueError("at least one retrieval weight must be positive")
        if self.rrf_rank_constant <= 0:
            raise ValueError("rrf_rank_constant must be positive")


class RetrievalService:
    """Owns the complete query-side RAG pipeline.

    The service never asks a model provider to search files.  It embeds the
    question, calls our repository, fuses and diversifies our results, builds a
    labelled prompt, and only then asks a generator to synthesize an answer.
    """

    def __init__(
        self,
        *,
        repository: RetrievalRepository,
        embedder: EmbeddingProvider,
        generator: Generator,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._generator = generator
        self._reranker = reranker
        self._config = config or RetrievalConfig()
        self._context_builder = context_builder or ContextBuilder(
            max_tokens=self._config.max_context_tokens,
            min_passage_tokens=self._config.min_context_passage_tokens,
        )

    async def search(
        self,
        question: str,
        *,
        filters: RetrievalFilters,
    ) -> RetrievalResult:
        normalized_query = normalize_query(question)
        if not normalized_query:
            raise ValueError("question cannot be empty")

        timings: dict[str, float] = {}
        warnings: list[str] = []

        started = time.perf_counter()
        query_embedding = await self._embedder.embed_query(normalized_query)
        timings["embedding"] = _elapsed_ms(started)
        _validate_query_embedding(query_embedding, self._embedder.dimensions)

        started = time.perf_counter()
        dense_result, lexical_result = await asyncio.gather(
            self._repository.dense_search(
                query_embedding=query_embedding,
                limit=self._config.dense_limit,
                filters=filters,
            ),
            self._repository.lexical_search(
                query=normalized_query,
                limit=self._config.lexical_limit,
                filters=filters,
            ),
            return_exceptions=True,
        )
        timings["repository_search"] = _elapsed_ms(started)

        dense_candidates, dense_error = _unpack_search_result(dense_result)
        lexical_candidates, lexical_error = _unpack_search_result(lexical_result)
        if dense_error:
            warnings.append(f"dense_search_failed:{type(dense_error).__name__}")
        if lexical_error:
            warnings.append(f"lexical_search_failed:{type(lexical_error).__name__}")
        if dense_error and lexical_error:
            raise RetrievalError("both dense and lexical retrieval failed") from dense_error

        _validate_candidates(dense_candidates)
        _validate_candidates(lexical_candidates)

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            dense_candidates,
            lexical_candidates,
            dense_weight=self._config.dense_weight,
            lexical_weight=self._config.lexical_weight,
            rank_constant=self._config.rrf_rank_constant,
            limit=self._config.fusion_limit,
        )
        fused_count = len(fused)
        deduplicated = deduplicate_ranked_candidates(
            fused,
            near_duplicate_threshold=self._config.near_duplicate_threshold,
        )
        deduplicated_count = len(deduplicated)

        reranker_used = False
        if self._reranker is not None and deduplicated:
            try:
                rerank_scores = await self._reranker.rerank(
                    normalized_query,
                    [item.candidate for item in deduplicated],
                )
                if any(not math.isfinite(float(score)) for score in rerank_scores):
                    raise ValueError("reranker produced a non-finite score")
                deduplicated = attach_rerank_scores(
                    deduplicated,
                    rerank_scores,
                    rerank_weight=self._config.rerank_weight,
                )
                reranker_used = True
            except Exception as error:  # ranking remains usable without this optional stage
                warnings.append(f"reranker_failed:{type(error).__name__}")

        selected, diversity_used = select_diverse_candidates(
            deduplicated,
            limit=self._config.source_limit,
            lambda_mult=self._config.mmr_lambda,
            max_per_document=self._config.max_per_document,
            max_per_section=self._config.max_per_section,
        )
        timings["ranking"] = _elapsed_ms(started)

        diagnostics = RetrievalDiagnostics(
            normalized_query=normalized_query,
            embedding_provider=_embedding_identifier(self._embedder),
            dense_candidates=len(dense_candidates),
            lexical_candidates=len(lexical_candidates),
            fused_candidates=fused_count,
            deduplicated_candidates=deduplicated_count,
            selected_candidates=len(selected),
            reranker_used=reranker_used,
            diversity_used=diversity_used,
            warnings=tuple(warnings),
            timings_ms=timings,
        )
        return RetrievalResult(
            question=question,
            candidates=tuple(selected),
            diagnostics=diagnostics,
        )

    async def answer(
        self,
        question: str,
        *,
        filters: RetrievalFilters,
        conversation: Sequence[ChatMessage] = (),
    ) -> AnswerResult:
        retrieval = await self.search(question, filters=filters)
        diagnostics = retrieval.diagnostics

        if not retrieval.candidates:
            return _abstention(
                question,
                diagnostics,
                reason="no_relevant_sources",
            )
        if (
            self._config.minimum_top_score is not None
            and retrieval.candidates[0].final_score < self._config.minimum_top_score
        ):
            return _abstention(
                question,
                diagnostics,
                reason="relevance_below_threshold",
            )

        started = time.perf_counter()
        context = self._context_builder.build(retrieval.candidates)
        timings = dict(diagnostics.timings_ms)
        timings["context_assembly"] = _elapsed_ms(started)
        diagnostics = replace(
            diagnostics,
            selected_candidates=len(context.sources),
            context_estimated_tokens=context.estimated_tokens,
            context_truncated=context.truncated,
            timings_ms=timings,
        )
        if not context.sources:
            return _abstention(
                question,
                diagnostics,
                reason="context_budget_too_small",
            )

        attempts = 2 if self._config.retry_invalid_citations else 1
        feedback: str | None = None
        accumulated_invalid: list[str] = []
        last_validation: CitationValidation | None = None

        for attempt in range(1, attempts + 1):
            request = GenerationRequest(
                question=normalize_query(question),
                context=context.text,
                sources=context.sources,
                conversation=tuple(conversation)[-6:],
                validation_feedback=feedback,
            )
            started = time.perf_counter()
            try:
                result = await self._generator.generate(request)
            except Exception as error:
                timings = dict(diagnostics.timings_ms)
                timings["generation"] = _elapsed_ms(started)
                diagnostics = replace(
                    diagnostics,
                    generation_attempts=attempt,
                    warnings=diagnostics.warnings + (f"generation_failed:{type(error).__name__}",),
                    timings_ms=timings,
                )
                return _abstention(
                    question,
                    diagnostics,
                    reason="generation_error",
                    sources=context.sources,
                )

            timings = dict(diagnostics.timings_ms)
            timings["generation"] = timings.get("generation", 0.0) + _elapsed_ms(started)
            last_validation = validate_citations(
                result,
                request.allowed_source_ids,
                require_citation=self._config.require_citations,
            )
            accumulated_invalid.extend(last_validation.invalid_source_ids)
            output_missing = not result.answer.strip()
            if last_validation.valid and not output_missing:
                diagnostics = replace(
                    diagnostics,
                    generation_attempts=attempt,
                    invalid_citations=tuple(dict.fromkeys(accumulated_invalid)),
                    timings_ms=timings,
                )
                return _answer_from_generation(
                    question=question,
                    result=result,
                    validation=last_validation,
                    sources=context.sources,
                    diagnostics=diagnostics,
                )

            feedback = _validation_feedback(
                last_validation,
                request.allowed_source_ids,
                output_missing=output_missing,
            )
            diagnostics = replace(diagnostics, timings_ms=timings)

        diagnostics = replace(
            diagnostics,
            generation_attempts=attempts,
            invalid_citations=tuple(dict.fromkeys(accumulated_invalid)),
        )
        return _abstention(
            question,
            diagnostics,
            reason="invalid_or_missing_citations",
            sources=context.sources,
            missing_information=(
                "The generated response could not be tied safely to the supplied sources.",
            ),
        )


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", query)).strip()


def _validate_query_embedding(vector: Sequence[float], expected_dimensions: int | None) -> None:
    if not vector:
        raise ValueError("embedding provider returned an empty query vector")
    if expected_dimensions is not None and len(vector) != expected_dimensions:
        raise ValueError(
            f"query vector has {len(vector)} dimensions; expected {expected_dimensions}"
        )
    if any(not math.isfinite(float(value)) for value in vector):
        raise ValueError("embedding provider returned a non-finite query vector")


def _validate_candidates(candidates: Sequence[SearchCandidate]) -> None:
    if any(not isinstance(item, SearchCandidate) for item in candidates):
        raise TypeError("repository search methods must return SearchCandidate objects")
    if any(not math.isfinite(float(item.score)) for item in candidates):
        raise ValueError("repository returned a non-finite candidate score")


def _unpack_search_result(
    result: Sequence[SearchCandidate] | BaseException,
) -> tuple[list[SearchCandidate], BaseException | None]:
    if isinstance(result, BaseException):
        return [], result
    return list(result), None


def _embedding_identifier(embedder: EmbeddingProvider) -> str:
    suffix = f":{embedder.dimensions}" if embedder.dimensions is not None else ""
    return f"{embedder.provider_name}:{embedder.model_name}{suffix}"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _validation_feedback(
    validation: CitationValidation,
    allowed: Sequence[str],
    *,
    output_missing: bool,
) -> str:
    issues: list[str] = []
    if output_missing:
        issues.append("The previous answer was empty.")
    if validation.invalid_source_ids:
        issues.append(
            "It used source labels that were not supplied: "
            + ", ".join(validation.invalid_source_ids)
            + "."
        )
    if not validation.has_required_citation:
        issues.append("It did not include an inline [S#] citation.")
    issues.append("Regenerate using only these labels: " + ", ".join(allowed) + ".")
    return " ".join(issues)


def _answer_from_generation(
    *,
    question: str,
    result: GenerationResult,
    validation: CitationValidation,
    sources: tuple[SourceExcerpt, ...],
    diagnostics: RetrievalDiagnostics,
) -> AnswerResult:
    confidence = result.confidence
    if confidence is not None:
        confidence = min(1.0, max(0.0, float(confidence)))
    reason = "generator_abstained" if result.abstained else None
    return AnswerResult(
        question=question,
        answer=result.answer.strip(),
        sources=sources,
        cited_source_ids=validation.cited_source_ids,
        confidence=confidence,
        missing_information=tuple(result.missing_information),
        abstained=result.abstained,
        abstention_reason=reason,
        diagnostics=diagnostics,
    )


def _abstention(
    question: str,
    diagnostics: RetrievalDiagnostics,
    *,
    reason: str,
    sources: tuple[SourceExcerpt, ...] = (),
    missing_information: tuple[str, ...] = (),
) -> AnswerResult:
    return AnswerResult(
        question=question,
        answer=(
            "I couldn't find enough reliable information in the uploaded "
            "documents to answer that question."
        ),
        sources=sources,
        cited_source_ids=(),
        confidence=0.0,
        missing_information=missing_information,
        abstained=True,
        abstention_reason=reason,
        diagnostics=diagnostics,
    )
