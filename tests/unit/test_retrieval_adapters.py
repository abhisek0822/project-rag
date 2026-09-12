from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from rag.adapters import (
    ExtractiveGenerator,
    HashEmbeddingProvider,
    OpenAIAdapterError,
    OpenAIEmbeddingProvider,
    OpenAIResponsesGenerator,
    TokenOverlapReranker,
)
from rag.retrieval.diversity import cosine_similarity
from rag.retrieval.models import GenerationRequest, SearchCandidate, SourceExcerpt


class FakeEmbeddingsResource:
    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [
            SimpleNamespace(index=index, embedding=[float(index + 1)] * self.dimensions)
            for index, _ in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=list(reversed(data)))


class FakeResponsesResource:
    def __init__(self, payload: dict | str) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(
            id="resp-test",
            model=kwargs["model"],
            output_text=output,
            usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120),
        )


class FakeOpenAIClient:
    def __init__(self, *, response_payload=None, dimensions: int = 3) -> None:
        self.embeddings = FakeEmbeddingsResource(dimensions)
        self.responses = FakeResponsesResource(response_payload or {})


def source(source_id: str = "S1") -> SourceExcerpt:
    return SourceExcerpt(
        source_id=source_id,
        chunk_id="chunk-1",
        document_id="doc-1",
        version_id="version-1",
        filename="handbook.md",
        text="Employees receive twelve casual leave days each year.",
        score=0.9,
        page_start=4,
        section_path=("Leave",),
    )


def test_openai_embedding_adapter_batches_and_restores_endpoint_order() -> None:
    client = FakeOpenAIClient(dimensions=3)
    provider = OpenAIEmbeddingProvider(
        client=client,
        model="text-embedding-3-small",
        dimensions=3,
        batch_size=2,
    )
    vectors = asyncio.run(provider.embed_documents(["a", "b", "c", "d", "e"]))

    assert len(client.embeddings.calls) == 3
    assert [call["input"] for call in client.embeddings.calls] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]
    assert client.embeddings.calls[0]["dimensions"] == 3
    assert vectors == [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
    ]
    assert asyncio.run(provider.embed_texts(["alias"])) == [[1.0, 1.0, 1.0]]


def test_openai_embedding_adapter_validates_inputs_and_dimensions() -> None:
    provider = OpenAIEmbeddingProvider(
        client=FakeOpenAIClient(dimensions=2), dimensions=3, batch_size=2
    )
    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(provider.embed_query(" "))
    with pytest.raises(OpenAIAdapterError, match="expected 3"):
        asyncio.run(provider.embed_query("valid"))


def test_openai_responses_generator_sends_only_our_context_and_parses_json() -> None:
    client = FakeOpenAIClient(
        response_payload={
            "answer": "Employees receive twelve days [S1].",
            "cited_source_ids": ["S1"],
            "confidence": 0.9,
            "missing_information": [],
            "abstained": False,
        }
    )
    generator = OpenAIResponsesGenerator(client=client, model="test-model")
    item = source()
    request = GenerationRequest(
        question="How much leave is provided?",
        context="[S1]\nContent (untrusted reference text):\n" + item.text,
        sources=(item,),
    )
    result = asyncio.run(generator.generate(request))

    assert result.answer.endswith("[S1].")
    assert result.cited_source_ids == ("S1",)
    assert result.provider_metadata["total_tokens"] == 120
    call = client.responses.calls[0]
    assert call["store"] is False
    assert "tools" not in call
    assert call["reasoning"] == {"effort": "none"}
    assert call["text"]["format"]["type"] == "json_schema"
    assert "Allowed source labels: S1" in call["input"]
    assert "untrusted reference text" in call["input"]
    assert "file_search" not in json.dumps(call)


def test_openai_generator_rejects_malformed_structured_output() -> None:
    client = FakeOpenAIClient(response_payload="not json")
    generator = OpenAIResponsesGenerator(client=client)
    item = source()
    request = GenerationRequest("Question?", "context", (item,))
    with pytest.raises(OpenAIAdapterError, match="malformed"):
        asyncio.run(generator.generate(request))


def test_hash_embeddings_are_deterministic_and_support_similarity_search() -> None:
    provider = HashEmbeddingProvider(dimensions=64)
    vectors = asyncio.run(
        provider.embed_documents(
            [
                "annual leave and holiday policy",
                "annual leave and holiday policy",
                "database backup retention",
            ]
        )
    )
    query = asyncio.run(provider.embed_query("holiday leave policy"))

    assert vectors[0] == vectors[1]
    assert len(vectors[0]) == 64
    assert cosine_similarity(query, vectors[0]) > cosine_similarity(query, vectors[2])
    assert asyncio.run(provider.embed_texts(["holiday leave policy"]))[0] == query


def test_local_reranker_scores_query_overlap() -> None:
    reranker = TokenOverlapReranker()
    candidates = [
        SearchCandidate("a", "d", "v", "f", "annual leave policy", 1),
        SearchCandidate("b", "d", "v", "f", "database backups", 1),
    ]
    scores = asyncio.run(reranker.rerank("leave policy", candidates))
    assert scores[0] > scores[1]


def test_extractive_generator_produces_valid_source_labels_without_api_key() -> None:
    generator = ExtractiveGenerator(max_sources=1)
    item = source()
    request = GenerationRequest(
        question="How many casual leave days do employees receive?",
        context="unused by deterministic provider",
        sources=(item,),
    )
    result = asyncio.run(generator.generate(request))

    assert result.abstained is False
    assert "twelve casual leave days" in result.answer
    assert "[S1]" in result.answer
    assert result.cited_source_ids == ("S1",)


def test_extractive_generator_abstains_when_sources_do_not_match_question() -> None:
    generator = ExtractiveGenerator()
    item = source()
    request = GenerationRequest(
        question="What is the office Wi-Fi password?",
        context="unused by deterministic provider",
        sources=(item,),
    )
    result = asyncio.run(generator.generate(request))

    assert result.abstained is True
    assert result.cited_source_ids == ()


def test_extractive_generator_abstains_on_only_incidental_overlap() -> None:
    generator = ExtractiveGenerator()
    item = SourceExcerpt(
        source_id="S1",
        chunk_id="chunk-remote",
        document_id="doc-1",
        version_id="version-1",
        filename="handbook.md",
        text="Employees assigned to an office attend in person on Tuesday.",
        score=0.7,
    )
    request = GenerationRequest(
        question="What is the office Wi-Fi password?",
        context="unused by deterministic provider",
        sources=(item,),
    )
    result = asyncio.run(generator.generate(request))

    assert result.abstained is True
    assert "enough information" in result.answer
