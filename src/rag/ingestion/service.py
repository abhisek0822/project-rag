"""Provider-neutral orchestration for parsing, chunking, and embedding files."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .chunking import Chunker
from .errors import (
    DocumentParseError,
    EmbeddingError,
    EmbeddingValidationError,
    EmptyDocumentError,
    IndexingError,
    IngestionError,
    IngestionStatus,
)
from .models import Chunk
from .normalization import Normalizer
from .parsers import ParserRegistry

MaybeAwaitable = Any | Awaitable[Any]


class BlobStore(Protocol):
    def get_bytes(self, blob_uri: str) -> MaybeAwaitable:
        """Read an immutable source object in full for a worker."""


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed_texts(self, texts: Sequence[str]) -> MaybeAwaitable:
        """Return one finite, fixed-size vector per input text."""


class IngestionRepository(Protocol):
    """Persistence port implemented by an adapter outside this package.

    ``publish_version`` must atomically make this version current/searchable and
    mark it READY. ``replace_chunks`` rather than append semantics makes a retry
    idempotent when deterministic chunk IDs are used.
    """

    def get_version(self, version_id: str) -> MaybeAwaitable: ...

    def set_status(
        self,
        version_id: str,
        status: str,
        *,
        stage: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> MaybeAwaitable: ...

    def replace_chunks(self, version_id: str, chunks: Sequence[Chunk]) -> MaybeAwaitable: ...

    def save_embeddings(
        self,
        version_id: str,
        embeddings: Sequence[tuple[str, Sequence[float]]],
        dimensions: int,
    ) -> MaybeAwaitable: ...

    def publish_version(self, version_id: str, expected_chunk_count: int) -> MaybeAwaitable: ...


@dataclass(frozen=True)
class IngestionResult:
    version_id: str
    status: IngestionStatus
    chunk_count: int
    embedding_dimensions: int


async def _resolve(value: MaybeAwaitable) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _value(record: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


class IngestionService:
    """Run a complete ingestion attempt for one immutable document version."""

    def __init__(
        self,
        blob_store: BlobStore,
        repository: IngestionRepository,
        embedding_provider: EmbeddingProvider,
        *,
        parser_registry: ParserRegistry | None = None,
        normalizer: Normalizer | None = None,
        chunker: Chunker | None = None,
        embedding_batch_size: int = 64,
        embedding_batch_token_limit: int = 16_000,
    ) -> None:
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        if embedding_batch_token_limit <= 0:
            raise ValueError("embedding_batch_token_limit must be positive")
        self.blob_store = blob_store
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.parser_registry = parser_registry or ParserRegistry()
        self.normalizer = normalizer or Normalizer()
        self.chunker = chunker or Chunker()
        self.embedding_batch_size = embedding_batch_size
        self.embedding_batch_token_limit = embedding_batch_token_limit

    async def ingest(self, version_id: str) -> IngestionResult:
        phase = IngestionStatus.PARSING
        try:
            version = await _resolve(self.repository.get_version(version_id))
            if version is None:
                raise DocumentParseError(f"Unknown document version: {version_id}")

            stored_version_id = str(_value(version, "id", default=version_id))
            blob_uri = _value(version, "blob_uri", "storage_uri", "object_key")
            filename = _value(
                version,
                "filename",
                "original_filename",
                "display_filename",
                default="document",
            )
            title = _value(version, "title", default=None) or Path(str(filename)).stem
            if not blob_uri:
                raise DocumentParseError(f"Document version {stored_version_id} has no blob URI")

            await self._set_status(version_id, phase)
            raw = await _resolve(self.blob_store.get_bytes(str(blob_uri)))
            if not isinstance(raw, bytes):
                raise DocumentParseError("BlobStore.get_bytes() must return bytes")
            blocks = self.parser_registry.parse(raw, str(filename))
            blocks = self.normalizer.normalize(blocks)
            if not blocks:
                raise EmptyDocumentError(str(filename))

            phase = IngestionStatus.CHUNKING
            await self._set_status(version_id, phase)
            chunks = self.chunker.chunk(blocks, stored_version_id, str(title))
            if not chunks:
                raise EmptyDocumentError(str(filename))

            phase = IngestionStatus.EMBEDDING
            await self._set_status(version_id, phase)
            vectors, dimensions = await self._embed(chunks)

            phase = IngestionStatus.INDEXING
            await self._set_status(version_id, phase)
            try:
                await _resolve(self.repository.replace_chunks(version_id, chunks))
                await _resolve(self.repository.save_embeddings(version_id, vectors, dimensions))
                await _resolve(self.repository.publish_version(version_id, len(chunks)))
            except IngestionError:
                raise
            except Exception as exc:
                raise IndexingError(str(exc)) from exc

            return IngestionResult(
                version_id=stored_version_id,
                status=IngestionStatus.READY,
                chunk_count=len(chunks),
                embedding_dimensions=dimensions,
            )
        except Exception as exc:
            status, code = self._failure_for(exc, phase)
            await self._record_failure(version_id, status, code, str(exc))
            raise

    def ingest_sync(self, version_id: str) -> IngestionResult:
        """Convenience entry point for a synchronous worker process."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ingest(version_id))
        raise RuntimeError("ingest_sync() cannot run inside an active event loop; await ingest()")

    async def _embed(
        self, chunks: Sequence[Chunk]
    ) -> tuple[list[tuple[str, Sequence[float]]], int]:
        configured_dimensions = getattr(self.embedding_provider, "dimensions", None)
        if callable(configured_dimensions):
            configured_dimensions = configured_dimensions()
        if configured_dimensions is not None:
            try:
                configured_dimensions = int(configured_dimensions)
            except (TypeError, ValueError) as exc:
                raise EmbeddingValidationError(
                    "Embedding provider dimensions must be an integer"
                ) from exc
            if configured_dimensions <= 0:
                raise EmbeddingValidationError("Embedding provider dimensions must be positive")

        all_vectors: list[tuple[str, Sequence[float]]] = []
        dimensions: int | None = configured_dimensions
        for batch in self._embedding_batches(chunks):
            texts = [chunk.embedding_text for chunk in batch]
            try:
                response = await _resolve(self.embedding_provider.embed_texts(texts))
            except IngestionError:
                raise
            except Exception as exc:
                raise EmbeddingError(str(exc)) from exc

            try:
                response_vectors = list(response)
            except TypeError as exc:
                raise EmbeddingValidationError(
                    "Embedding provider returned a non-iterable response"
                ) from exc
            if len(response_vectors) != len(batch):
                raise EmbeddingValidationError(
                    f"Embedding provider returned {len(response_vectors)} vectors for "
                    f"{len(batch)} texts"
                )

            for chunk, vector in zip(batch, response_vectors, strict=True):
                try:
                    vector_values = list(vector)
                except TypeError as exc:
                    raise EmbeddingValidationError(
                        f"Embedding for chunk {chunk.id} is not a vector"
                    ) from exc
                if not vector_values:
                    raise EmbeddingValidationError(f"Embedding for chunk {chunk.id} is empty")
                if dimensions is None:
                    dimensions = len(vector_values)
                if len(vector_values) != dimensions:
                    raise EmbeddingValidationError(
                        f"Embedding for chunk {chunk.id} has dimension "
                        f"{len(vector_values)}; expected {dimensions}"
                    )
                clean_vector: list[float] = []
                for value in vector_values:
                    if isinstance(value, bool):
                        raise EmbeddingValidationError(
                            f"Embedding for chunk {chunk.id} contains a boolean"
                        )
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError) as exc:
                        raise EmbeddingValidationError(
                            f"Embedding for chunk {chunk.id} contains a non-number"
                        ) from exc
                    if not math.isfinite(numeric):
                        raise EmbeddingValidationError(
                            f"Embedding for chunk {chunk.id} contains a non-finite value"
                        )
                    clean_vector.append(numeric)
                all_vectors.append((chunk.id, clean_vector))

        if dimensions is None:
            raise EmbeddingValidationError("No embeddings were produced")
        return all_vectors, dimensions

    def _embedding_batches(self, chunks: Sequence[Chunk]) -> list[list[Chunk]]:
        batches: list[list[Chunk]] = []
        current: list[Chunk] = []
        current_tokens = 0
        for chunk in chunks:
            would_exceed_count = len(current) >= self.embedding_batch_size
            would_exceed_tokens = (
                bool(current)
                and current_tokens + chunk.token_count > self.embedding_batch_token_limit
            )
            if would_exceed_count or would_exceed_tokens:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(chunk)
            current_tokens += chunk.token_count
        if current:
            batches.append(current)
        return batches

    async def _set_status(self, version_id: str, status: IngestionStatus) -> None:
        await _resolve(
            self.repository.set_status(
                version_id,
                status.value,
                error_code=None,
                error_message=None,
            )
        )

    async def _record_failure(
        self,
        version_id: str,
        status: IngestionStatus,
        code: str,
        message: str,
    ) -> None:
        try:
            await _resolve(
                self.repository.set_status(
                    version_id,
                    status.value,
                    error_code=code,
                    error_message=message[:2000],
                )
            )
        except Exception:
            # Persistence trouble must not hide the actionable root failure.
            return

    @staticmethod
    def _failure_for(error: Exception, phase: IngestionStatus) -> tuple[IngestionStatus, str]:
        if isinstance(error, IngestionError):
            return error.status, error.code
        if phase == IngestionStatus.EMBEDDING:
            return IngestionStatus.FAILED_EMBED, "embedding_failed"
        if phase == IngestionStatus.INDEXING:
            return IngestionStatus.FAILED_INDEX, "indexing_failed"
        return IngestionStatus.FAILED_PARSE, "document_parse_failed"
