from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from rag.ingestion import (
    Block,
    Chunker,
    EmbeddingValidationError,
    IngestionService,
    IngestionStatus,
    NeedsOCRError,
    ParserRegistry,
)


def word_count(text: str) -> int:
    return len(text.split())


class FakeBlobStore:
    def __init__(self, data: bytes = b"ignored") -> None:
        self.data = data
        self.requests: list[str] = []

    def get_bytes(self, blob_uri: str) -> bytes:
        self.requests.append(blob_uri)
        return self.data


class StaticParser:
    def __init__(self, blocks: Sequence[Block]) -> None:
        self.blocks = list(blocks)

    def parse(self, data: bytes, filename: str) -> list[Block]:
        return list(self.blocks)


class OCRParser:
    def parse(self, data: bytes, filename: str) -> list[Block]:
        raise NeedsOCRError(filename)


class FakeRepository:
    def __init__(self) -> None:
        self.version = {
            "id": "version-1",
            "blob_uri": "blobs/source.txt",
            "filename": "source.txt",
            "title": "Source guide",
        }
        self.statuses: list[dict[str, Any]] = []
        self.chunks = []
        self.embeddings = []
        self.dimensions = None
        self.published = []

    def get_version(self, version_id: str):
        return self.version

    def set_status(self, version_id: str, status: str, **details: Any) -> None:
        self.statuses.append({"version_id": version_id, "status": status, **details})

    def replace_chunks(self, version_id: str, chunks) -> None:
        self.chunks = list(chunks)

    def save_embeddings(self, version_id: str, embeddings, dimensions: int) -> None:
        self.embeddings = list(embeddings)
        self.dimensions = dimensions

    def publish_version(self, version_id: str, expected_chunk_count: int) -> None:
        self.published.append((version_id, expected_chunk_count))


class FakeEmbeddingProvider:
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: Sequence[str]):
        self.calls.append(list(texts))
        return [[float(index), 0.5, 1.0] for index, _ in enumerate(texts)]


def make_service(repository, parser, provider) -> IngestionService:
    return IngestionService(
        FakeBlobStore(),
        repository,
        provider,
        parser_registry=ParserRegistry({".txt": parser}),
        chunker=Chunker(
            target_tokens=4,
            max_tokens=5,
            min_tokens=1,
            overlap_tokens=0,
            token_counter=word_count,
        ),
        embedding_batch_size=2,
        embedding_batch_token_limit=8,
    )


def test_service_batches_validates_persists_and_publishes() -> None:
    repository = FakeRepository()
    parser = StaticParser([Block("paragraph", "one two three four five six seven eight nine", 0)])
    provider = FakeEmbeddingProvider()
    service = make_service(repository, parser, provider)

    result = asyncio.run(service.ingest("version-1"))

    assert result.status == IngestionStatus.READY
    assert result.chunk_count == 3
    assert result.embedding_dimensions == 3
    assert [len(call) for call in provider.calls] == [2, 1]
    assert len(repository.embeddings) == 3
    assert repository.dimensions == 3
    assert repository.published == [("version-1", 3)]
    assert [entry["status"] for entry in repository.statuses] == [
        "PARSING",
        "CHUNKING",
        "EMBEDDING",
        "INDEXING",
    ]

    first_ids = [chunk.id for chunk in repository.chunks]
    asyncio.run(service.ingest("version-1"))
    assert [chunk.id for chunk in repository.chunks] == first_ids


def test_invalid_embedding_dimensions_fail_without_publishing() -> None:
    class WrongDimensions:
        dimensions = 3

        def embed_texts(self, texts):
            return [[0.1, 0.2] for _ in texts]

    repository = FakeRepository()
    service = make_service(
        repository,
        StaticParser([Block("paragraph", "some useful source text", 0)]),
        WrongDimensions(),
    )

    with pytest.raises(EmbeddingValidationError):
        asyncio.run(service.ingest("version-1"))

    assert repository.published == []
    assert repository.chunks == []
    assert repository.statuses[-1]["status"] == "FAILED_EMBED"
    assert repository.statuses[-1]["error_code"] == "invalid_embedding_response"


def test_needs_ocr_has_an_explicit_terminal_status() -> None:
    repository = FakeRepository()
    service = make_service(repository, OCRParser(), FakeEmbeddingProvider())

    with pytest.raises(NeedsOCRError):
        asyncio.run(service.ingest("version-1"))

    assert repository.statuses[-1]["status"] == "NEEDS_OCR"
    assert repository.statuses[-1]["error_code"] == "needs_ocr"
