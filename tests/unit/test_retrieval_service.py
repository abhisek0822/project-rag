from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from rag.retrieval import (
    GenerationRequest,
    GenerationResult,
    RetrievalConfig,
    RetrievalError,
    RetrievalFilters,
    RetrievalService,
    SearchCandidate,
)


class FakeEmbedder:
    provider_name = "fake"
    model_name = "fake-v1"
    dimensions = 3

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0, 0.0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeRepository:
    def __init__(
        self,
        dense: Sequence[SearchCandidate] = (),
        lexical: Sequence[SearchCandidate] = (),
        *,
        dense_error: Exception | None = None,
        lexical_error: Exception | None = None,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.dense_error = dense_error
        self.lexical_error = lexical_error
        self.dense_calls: list[dict] = []
        self.lexical_calls: list[dict] = []

    async def dense_search(self, **kwargs):
        self.dense_calls.append(kwargs)
        if self.dense_error:
            raise self.dense_error
        return self.dense

    async def lexical_search(self, **kwargs):
        self.lexical_calls.append(kwargs)
        if self.lexical_error:
            raise self.lexical_error
        return self.lexical


class FakeGenerator:
    provider_name = "fake"
    model_name = "fake-generator"

    def __init__(self, outputs: Sequence[GenerationResult] = ()) -> None:
        self.outputs = list(outputs)
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("unexpected generator call")
        return self.outputs.pop(0)


class FakeReranker:
    name = "fake-reranker"

    def __init__(self, scores: Sequence[float] | Exception) -> None:
        self.scores = scores
        self.calls = 0

    async def rerank(self, query, candidates):
        self.calls += 1
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


def make_candidate(
    chunk_id: str,
    text: str,
    *,
    score: float,
    document_id: str = "doc-1",
    embedding: tuple[float, ...] | None = None,
    content_hash: str | None = None,
) -> SearchCandidate:
    return SearchCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        version_id=f"v-{document_id}",
        filename=f"{document_id}.md",
        text=text,
        score=score,
        page_start=1,
        section_path=("Benefits", chunk_id),
        embedding=embedding,
        content_hash=content_hash,
    )


def service(
    repository: FakeRepository,
    generator: FakeGenerator,
    *,
    reranker: FakeReranker | None = None,
) -> tuple[RetrievalService, FakeEmbedder]:
    embedder = FakeEmbedder()
    instance = RetrievalService(
        repository=repository,
        embedder=embedder,
        generator=generator,
        reranker=reranker,
        config=RetrievalConfig(
            dense_limit=4,
            lexical_limit=5,
            fusion_limit=6,
            source_limit=3,
            max_context_tokens=600,
            min_context_passage_tokens=1,
        ),
    )
    return instance, embedder


FILTERS = RetrievalFilters(tenant_id="tenant-a", collection_id="handbook")


def test_service_runs_owned_hybrid_pipeline_and_returns_grounded_answer() -> None:
    shared = make_candidate(
        "shared",
        "Employees receive twelve casual leave days each calendar year.",
        score=0.80,
        embedding=(1.0, 0.0, 0.0),
    )
    repository = FakeRepository(
        dense=[
            make_candidate(
                "semantic",
                "Unused casual leave does not carry into the following year.",
                score=0.95,
                embedding=(0.99, 0.01, 0.0),
            ),
            shared,
        ],
        lexical=[
            shared,
            make_candidate(
                "exact",
                "The HR code for casual leave is CL-12.",
                score=4.0,
                document_id="doc-2",
                embedding=(0.0, 1.0, 0.0),
            ),
        ],
    )
    generator = FakeGenerator(
        [
            GenerationResult(
                answer="Employees receive twelve casual leave days per year [S1].",
                cited_source_ids=("S1",),
                confidence=0.92,
            )
        ]
    )
    rag, embedder = service(repository, generator)

    answer = asyncio.run(rag.answer("  How many\n casual leave days?  ", filters=FILTERS))

    assert answer.abstained is False
    assert answer.cited_source_ids == ("S1",)
    assert answer.sources
    assert answer.sources[0].source_id == "S1"
    assert embedder.queries == ["How many casual leave days?"]
    assert repository.dense_calls[0]["filters"] is FILTERS
    assert repository.dense_calls[0]["limit"] == 4
    assert repository.lexical_calls[0]["query"] == "How many casual leave days?"
    assert generator.requests[0].context.count("Chunk ID: shared") == 1
    assert "untrusted reference text" in generator.requests[0].context
    assert answer.diagnostics.dense_candidates == 2
    assert answer.diagnostics.lexical_candidates == 2
    assert answer.diagnostics.fused_candidates == 3
    assert answer.diagnostics.embedding_provider == "fake:fake-v1:3"


def test_service_abstains_without_calling_generator_when_nothing_is_retrieved() -> None:
    generator = FakeGenerator()
    rag, _ = service(FakeRepository(), generator)

    result = asyncio.run(rag.answer("What is the holiday policy?", filters=FILTERS))

    assert result.abstained is True
    assert result.abstention_reason == "no_relevant_sources"
    assert result.sources == ()
    assert generator.requests == []


def test_service_retries_once_after_invalid_citation() -> None:
    repository = FakeRepository(
        dense=[make_candidate("a", "The allowance is 500 rupees.", score=0.9)]
    )
    generator = FakeGenerator(
        [
            GenerationResult(answer="The allowance is 500 [S99].", cited_source_ids=("S99",)),
            GenerationResult(answer="The allowance is 500 [S1].", cited_source_ids=("S1",)),
        ]
    )
    rag, _ = service(repository, generator)

    result = asyncio.run(rag.answer("What is the allowance?", filters=FILTERS))

    assert result.abstained is False
    assert result.cited_source_ids == ("S1",)
    assert result.diagnostics.generation_attempts == 2
    assert result.diagnostics.invalid_citations == ("S99",)
    assert "S99" in (generator.requests[1].validation_feedback or "")


def test_service_rejects_output_when_citations_remain_invalid() -> None:
    repository = FakeRepository(
        dense=[make_candidate("a", "The allowance is 500 rupees.", score=0.9)]
    )
    generator = FakeGenerator(
        [
            GenerationResult(answer="It is 500 [S88].", cited_source_ids=("S88",)),
            GenerationResult(answer="It is 500 [S99].", cited_source_ids=("S99",)),
        ]
    )
    rag, _ = service(repository, generator)

    result = asyncio.run(rag.answer("What is the allowance?", filters=FILTERS))

    assert result.abstained is True
    assert result.abstention_reason == "invalid_or_missing_citations"
    assert result.cited_source_ids == ()
    assert result.diagnostics.invalid_citations == ("S88", "S99")


def test_service_retries_when_answer_has_no_inline_citation() -> None:
    repository = FakeRepository(dense=[make_candidate("a", "The office closes at six.", score=0.9)])
    generator = FakeGenerator(
        [
            GenerationResult(answer="The office closes at six.", cited_source_ids=("S1",)),
            GenerationResult(answer="The office closes at six [S1].", cited_source_ids=("S1",)),
        ]
    )
    rag, _ = service(repository, generator)
    result = asyncio.run(rag.answer("When does the office close?", filters=FILTERS))
    assert result.abstained is False
    assert "inline [S#]" in (generator.requests[1].validation_feedback or "")


def test_one_failed_retrieval_channel_degrades_gracefully() -> None:
    lexical = [make_candidate("exact", "Policy identifier AB-123.", score=4)]
    repository = FakeRepository(dense_error=RuntimeError("vector unavailable"), lexical=lexical)
    generator = FakeGenerator(
        [GenerationResult(answer="The identifier is AB-123 [S1].", cited_source_ids=("S1",))]
    )
    rag, _ = service(repository, generator)

    result = asyncio.run(rag.answer("What is the identifier?", filters=FILTERS))

    assert result.abstained is False
    assert result.diagnostics.dense_candidates == 0
    assert "dense_search_failed:RuntimeError" in result.diagnostics.warnings


def test_both_failed_retrieval_channels_raise_clear_error() -> None:
    repository = FakeRepository(
        dense_error=RuntimeError("dense"),
        lexical_error=RuntimeError("lexical"),
    )
    rag, _ = service(repository, FakeGenerator())
    with pytest.raises(RetrievalError, match="both dense and lexical"):
        asyncio.run(rag.search("question", filters=FILTERS))


def test_optional_reranker_failure_falls_back_to_fusion() -> None:
    repository = FakeRepository(dense=[make_candidate("a", "Evidence from policy.", score=0.9)])
    generator = FakeGenerator(
        [GenerationResult(answer="The policy provides evidence [S1].", cited_source_ids=("S1",))]
    )
    rag, _ = service(repository, generator, reranker=FakeReranker(RuntimeError("offline")))
    result = asyncio.run(rag.answer("What does policy say?", filters=FILTERS))
    assert result.abstained is False
    assert result.diagnostics.reranker_used is False
    assert "reranker_failed:RuntimeError" in result.diagnostics.warnings


def test_generation_error_becomes_safe_abstention() -> None:
    repository = FakeRepository(dense=[make_candidate("a", "Evidence from policy.", score=0.9)])

    class BrokenGenerator(FakeGenerator):
        async def generate(self, request):
            self.requests.append(request)
            raise TimeoutError("provider timeout with potentially sensitive detail")

    rag, _ = service(repository, BrokenGenerator())
    result = asyncio.run(rag.answer("What does policy say?", filters=FILTERS))
    assert result.abstained is True
    assert result.abstention_reason == "generation_error"
    assert "generation_failed:TimeoutError" in result.diagnostics.warnings
    assert "sensitive" not in result.answer


def test_search_rejects_empty_or_invalid_query_embeddings() -> None:
    rag, embedder = service(FakeRepository(), FakeGenerator())
    with pytest.raises(ValueError, match="question cannot be empty"):
        asyncio.run(rag.search("  \n ", filters=FILTERS))

    async def wrong_dimension(_):
        return [1.0, 2.0]

    embedder.embed_query = wrong_dimension  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="expected 3"):
        asyncio.run(rag.search("valid question", filters=FILTERS))
